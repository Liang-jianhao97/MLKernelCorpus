#!/usr/bin/env python3
"""Third pass: fix remaining failures with targeted config changes."""
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


def fix_aiter_mxfp_small_m(entry):
    """Small M values need baseline schedule, not sliceNK."""
    cmd = entry["run_command"]
    m_match = re.search(r'-M\s+(\d+)', cmd)
    if not m_match:
        return None
    m = int(m_match.group(1))

    # sliceNK requires M >= 256
    if "--schedule" in cmd:
        cmd = re.sub(r'--schedule\s+\w+', '--schedule baseline', cmd)
    else:
        cmd += " --schedule baseline"

    # For baseline, --scale_preshuffled is not needed (but doesn't hurt)
    # Remove BK override if baseline doesn't need it
    # But keep it if K is small
    k_match = re.search(r'-K\s+(\d+)', cmd)
    k = int(k_match.group(1)) if k_match else 8192
    if k < 256:
        cmd = re.sub(r'-BK\s+\d+', f'-BK {max(32, k // 2)}', cmd)
    elif '-BK' not in cmd:
        cmd += ' -BK 128'
    else:
        cmd = re.sub(r'-BK\s+256', '-BK 128', cmd)

    return cmd


def fix_best_swiglu(entry):
    """Fix best swiglu with PrecisionConfig."""
    cmd = entry["run_command"]
    cmd = fix_run_command(cmd)
    cmd = cmd.replace(
        "from triton_kernels.swiglu import swiglu",
        "from triton_kernels.swiglu import swiglu, PrecisionConfig"
    )
    cmd = cmd.replace(
        "swiglu(a, alpha=1.0)",
        "swiglu(a, alpha=1.0, precision_config=PrecisionConfig(limit=float('inf')))"
    )
    return cmd


def fix_best_reduce(entry):
    """Fix best reduce with 3D input."""
    cmd = entry["run_command"]
    cmd = fix_run_command(cmd)
    cmd = cmd.replace("torch.randn(", "torch.randn(1, ")
    cmd = cmd.replace("dim=1", "dim=2")
    return cmd


def save_result(entry, llir_files, actual_cmd, subdir):
    filename = entry.get("filename")
    if not filename:
        safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', entry["label"].lower())
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

        # Skip unfixable
        if "OutOfMemory" in err:
            continue
        if "make_opt_flags" in err:
            continue

        new_cmd = None

        if subdir == "aiter" and "mxfp_gemm" in entry.get("run_command", ""):
            new_cmd = fix_aiter_mxfp_small_m(entry)
        elif subdir == "best" and "SwiGLU" in label:
            new_cmd = fix_best_swiglu(entry)
        elif subdir == "best" and "Reduce" in label:
            new_cmd = fix_best_reduce(entry)
        else:
            continue

        if not new_cmd:
            continue

        print(f"  Fixing: {label}")
        cache_dir = tempfile.mkdtemp(prefix=f"triton_fix3_{subdir}_")
        try:
            llir_files, stderr, actual_cmd = compile_kernel(new_cmd, cache_dir)
            if llir_files:
                save_result(entry, llir_files, actual_cmd, subdir)
                fixed += 1
                print(f"    OK: {len(llir_files)} .llir file(s)")
            else:
                print(f"    STILL FAILED: {stderr[:150]}")
                entry["run_command"] = actual_cmd
                entry["error"] = f"Recompile3: {stderr[:100]}"
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    success = sum(1 for k in mapping["kernels"]
                  if k["status"] in ("success", "launch_failed_ir_captured"))
    failed = sum(1 for k in mapping["kernels"]
                 if k["status"] not in ("success", "launch_failed_ir_captured"))
    mapping["summary"] = {"total": len(mapping["kernels"]), "success": success, "failed": failed}
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    return fixed


def main():
    total = 0
    for sd in ["aiter", "best"]:
        print(f"\n=== {sd} ===")
        n = process_subdir(sd)
        total += n
        print(f"  Fixed {n}")
    print(f"\n=== Total: {total} ===")


if __name__ == "__main__":
    main()
