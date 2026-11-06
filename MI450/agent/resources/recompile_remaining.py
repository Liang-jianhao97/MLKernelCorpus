#!/usr/bin/env python3
"""Fix remaining failures from first recompile pass."""
import json
import os
import re
import shutil
import subprocess
import tempfile

TRITON_DIR = "/root/KernelExtract/triton"
CORPUS_DIR = "/root/MI450KernelCorpus/MI450/kernels"

ENV = {
    **os.environ,
    "TRITON_OVERRIDE_ARCH": "gfx1250",
    "TRITON_ALWAYS_COMPILE": "1",
}


def fix_run_command(cmd):
    """Apply all known command fixups."""
    if not cmd:
        return cmd
    cmd = re.sub(r'\s*--target\s+\S+', '', cmd)
    cmd = re.sub(r'(?<=\s)--([MNK])\b', r'-\1', cmd)
    cmd = re.sub(r'device=cuda\b', "device='cuda'", cmd)
    return cmd.strip()


def compile_kernel(cmd, cache_dir):
    env = {**ENV, "TRITON_CACHE_DIR": cache_dir}
    cmd = fix_run_command(cmd)
    try:
        result = subprocess.run(
            ["bash", "-c", f"cd {TRITON_DIR} && {cmd}"],
            env=env, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return [], "timeout", cmd

    llir_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".llir"):
                llir_files.append(os.path.join(root, f))

    stderr = result.stderr[-200:] if result.stderr else ""
    return llir_files, stderr, cmd


def fix_fp4_nonbaseline(entry):
    """fp4 non-baseline: needs both --with_a_scale AND --scale_preshuffled, BK>=256."""
    cmd = entry["run_command"]
    if "--with_a_scale" not in cmd:
        cmd += " --with_a_scale"
    if "--scale_preshuffled" not in cmd:
        cmd += " --scale_preshuffled"
    bk_match = re.search(r'-BK\s+(\d+)', cmd)
    if bk_match and int(bk_match.group(1)) < 256:
        cmd = re.sub(r'-BK\s+\d+', '-BK 256', cmd)
    return cmd


def process_subdir(subdir):
    mapping_path = os.path.join(CORPUS_DIR, subdir, "mapping.json")
    if not os.path.exists(mapping_path):
        return 0

    with open(mapping_path) as f:
        mapping = json.load(f)

    fixed = 0
    for entry in mapping["kernels"]:
        if entry["status"] in ("success", "launch_failed_ir_captured"):
            continue

        err = entry.get("error", "")
        label = entry["label"]
        cmd = entry["run_command"]

        # Skip unfixable
        if "OutOfMemoryError" in err or "OOM" in err.upper():
            continue
        if "make_opt_flags" in err:
            continue
        if "block scaled" in label.lower() and "10-block-scaled" in cmd:
            continue

        new_cmd = None

        # fp4 non-baseline (has both --with_a_scale from first pass, needs --scale_preshuffled)
        if "mxfp4" in label.lower() and ("slice" in label.lower() or "slice" in cmd.lower()):
            new_cmd = fix_fp4_nonbaseline(entry)

        # device=cuda quoting issue
        elif "device=cuda" in cmd and "device='cuda'" not in cmd:
            new_cmd = cmd  # fix_run_command will handle it

        # aiter mxfp_gemm with --target still in command
        elif "--target" in cmd and "mxfp_gemm" in cmd:
            new_cmd = cmd
            if "--scale_preshuffled" not in new_cmd:
                new_cmd += " --scale_preshuffled"
            bk_match = re.search(r'-BK\s+(\d+)', new_cmd)
            if bk_match and int(bk_match.group(1)) < 256:
                new_cmd = re.sub(r'-BK\s+\d+', '-BK 256', new_cmd)
            elif not bk_match:
                new_cmd += " -BK 256"

        # Remaining matmul/swiglu/reduce from vanilla that had device=cuda
        elif "Recompile failed" in err and "NameError" in err:
            new_cmd = cmd

        else:
            continue

        if not new_cmd:
            continue

        print(f"  Fixing: {label}")

        cache_dir = tempfile.mkdtemp(prefix=f"triton_fix2_{subdir}_")
        try:
            llir_files, stderr, actual_cmd = compile_kernel(new_cmd, cache_dir)

            if llir_files:
                filename = entry.get("filename")
                if not filename:
                    safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', label.lower())
                    safe_label = re.sub(r'_+', '_', safe_label).strip('_')
                    filename = safe_label + ".llir"

                out_dir = os.path.join(CORPUS_DIR, subdir)
                shutil.copy(llir_files[0], os.path.join(out_dir, filename))

                entry["filename"] = filename
                entry["status"] = "launch_failed_ir_captured"
                entry["run_command"] = actual_cmd
                entry.pop("error", None)
                entry["additional_files"] = []

                for i, extra in enumerate(llir_files[1:], 1):
                    extra_name = filename.replace(".llir", f"_helper{i}.llir")
                    shutil.copy(extra, os.path.join(out_dir, extra_name))
                    entry["additional_files"].append(extra_name)

                fixed += 1
                print(f"    OK: {len(llir_files)} .llir file(s)")
            else:
                print(f"    STILL FAILED: {stderr[:150]}")
                entry["run_command"] = actual_cmd
                entry["error"] = f"Recompile2 failed: {stderr[:100]}"
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    # Update summary
    success = sum(1 for k in mapping["kernels"]
                  if k["status"] in ("success", "launch_failed_ir_captured"))
    failed = sum(1 for k in mapping["kernels"]
                 if k["status"] not in ("success", "launch_failed_ir_captured"))
    mapping["summary"] = {
        "total": len(mapping["kernels"]),
        "success": success,
        "failed": failed
    }

    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    return fixed


def main():
    total = 0
    for sd in ["gluon", "triton", "best", "aiter"]:
        print(f"\n=== {sd} ===")
        n = process_subdir(sd)
        total += n
        print(f"  Fixed {n}")
    print(f"\n=== Total fixed: {total} ===")


if __name__ == "__main__":
    main()
