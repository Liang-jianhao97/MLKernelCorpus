#!/usr/bin/env python3
"""Recompile failed kernels with corrected configs.

Reads mapping.json files, identifies fixable failures, applies config fixes,
and recompiles. Updates mapping.json with results.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import re
from pathlib import Path

TRITON_DIR = "/root/KernelExtract/triton"
CORPUS_DIR = "/root/MI450KernelCorpus/MI450/kernels"

ENV = {
    **os.environ,
    "TRITON_OVERRIDE_ARCH": "gfx1250",
    "TRITON_ALWAYS_COMPILE": "1",
}


def fix_gluon_mxfp_gemm(entry):
    """Fix SCALE_PRESHUFFLE and fp4 skips for mxfp_gemm_cdna5.py."""
    cmd = entry["run_command"]
    err = entry.get("error", "")

    if "SCALE_PRESHUFFLE" in err:
        if "--scale_preshuffled" not in cmd:
            cmd += " --scale_preshuffled"
        # Ensure BK >= 256 for non-baseline schedules
        bk_match = re.search(r'-BK\s+(\d+)', cmd)
        if bk_match and int(bk_match.group(1)) < 256:
            cmd = re.sub(r'-BK\s+\d+', '-BK 256', cmd)
        return cmd

    if "fp4 x mxfp" in err:
        if "--with_a_scale" not in cmd:
            cmd += " --with_a_scale"
        return cmd

    return None


def fix_gluon_f16_gemm(entry):
    """Fix warp-specialized and single-warp assertion failures."""
    cmd = entry["run_command"]
    err = entry.get("error", "")

    if err != "AssertionError":
        return None

    if "--warp-specialized" in cmd:
        # warp-specialized requires num_warps 8 or 16 (non-persistent)
        cmd = re.sub(r'--num-warps\s+\d+', '--num-warps 8', cmd)
        # Also needs num_buffers >= 3
        if "--num-buffers" in cmd:
            cmd = re.sub(r'--num-buffers\s+\d+', '--num-buffers 3', cmd)
        else:
            cmd += " --num-buffers 3"
        return cmd

    if "--single-warp-schedule" in cmd:
        # single-warp requires num_warps=4 and --prefetch-lds
        cmd = re.sub(r'--num-warps\s+\d+', '--num-warps 4', cmd)
        if "--prefetch-lds" not in cmd:
            cmd += " --prefetch-lds"
        return cmd

    return None


def fix_gluon_f16_fa(entry):
    """Fix GQA reference check failure by using check=False wrapper."""
    err = entry.get("error", "")
    if "size of tensor a" not in err:
        return None

    cmd = entry["run_command"]
    # Extract args from the CLI command
    match = re.search(r'-b\s+(\d+)', cmd)
    batch = match.group(1) if match else "1"
    match = re.search(r'--seqlen-q\s+(\d+)', cmd)
    sq = match.group(1) if match else "4096"
    match = re.search(r'--seqlen-k\s+(\d+)', cmd)
    sk = match.group(1) if match else "4096"
    match = re.search(r'--num-heads-q\s+(\d+)', cmd)
    hq = match.group(1) if match else "8"
    match = re.search(r'--num-heads-k\s+(\d+)', cmd)
    hk = match.group(1) if match else "8"
    match = re.search(r'--head-size\s+(\d+)', cmd)
    hs = match.group(1) if match else "128"
    match = re.search(r'--block-m\s+(\d+)', cmd)
    bm = match.group(1) if match else "128"
    match = re.search(r'--block-n\s+(\d+)', cmd)
    bn = match.group(1) if match else "128"
    match = re.search(r'--attention-type\s+(\w+)', cmd)
    at = match.group(1) if match else "default"

    wrapper = (
        f'python3 -c "'
        f"import sys; sys.path.insert(0, 'third_party/amd/python/examples/gluon'); "
        f"from f16_fa_cdna5 import run_attention; "
        f"run_attention({{"
        f"'BATCH': {batch}, 'SEQLEN_Q': {sq}, 'SEQLEN_K': {sk}, "
        f"'NUM_Q_HEADS': {hq}, 'NUM_K_HEADS': {hk}, 'HEAD_SZ': {hs}, "
        f"'BLOCK_M': {bm}, 'BLOCK_N': {bn}, 'ATTN_FN': '{at}'"
        f"}}, check=False)"
        f'"'
    )
    return wrapper


def fix_gluon_mxfp_fa(entry):
    """Fix CompiledKernel.module error by removing --profile."""
    err = entry.get("error", "")
    if "CompiledKernel" not in err and "module" not in err:
        return None

    cmd = entry["run_command"]
    cmd = cmd.replace(" --profile", "")
    return cmd


def fix_gluon_f16_gemm_small_k(entry):
    """Fix K/BLOCK_K < NUM_BUFFERS skip."""
    err = entry.get("error", "")
    if "K/BLOCK_K" not in err:
        return None

    cmd = entry["run_command"]
    # strip --target if present
    cmd = re.sub(r'\s*--target\s+\S+', '', cmd)
    # fix --M to -M etc
    cmd = re.sub(r'(?<=\s)--([MNK])\b', r'-\1', cmd)

    # Extract K value
    k_match = re.search(r'-K\s+(\d+)', cmd)
    if not k_match:
        return None
    k = int(k_match.group(1))
    new_bk = max(32, k // 2)
    if "--block_k" in cmd:
        cmd = re.sub(r'--block_k\s+\d+', f'--block_k {new_bk}', cmd)
    else:
        cmd += f" --block_k {new_bk}"
    return cmd


def fix_vanilla_matmul(entry):
    """Fix missing bias arg."""
    cmd = entry["run_command"]
    err = entry.get("error", "")
    if "bias" not in err:
        return None

    # Replace matmul(a, b) with matmul(a, b, bias=None)
    cmd = cmd.replace("matmul(a, b)", "matmul(a, b, bias=None)")
    cmd = cmd.replace("matmul(a,b)", "matmul(a, b, bias=None)")
    return cmd


def fix_vanilla_swiglu(entry):
    """Fix missing precision_config arg."""
    err = entry.get("error", "")
    if "precision_config" not in err:
        return None

    cmd = entry["run_command"]
    # Add PrecisionConfig import and pass it
    cmd = cmd.replace(
        "from triton_kernels.swiglu import swiglu",
        "from triton_kernels.swiglu import swiglu, PrecisionConfig"
    )
    cmd = cmd.replace(
        "swiglu(a, alpha=1.0)",
        "swiglu(a, alpha=1.0, precision_config=PrecisionConfig(limit=float('inf')))"
    )
    return cmd


def fix_vanilla_reduce(entry):
    """Fix 2D input -> 3D."""
    err = entry.get("error", "")
    if "3D" not in err:
        return None

    cmd = entry["run_command"]
    # Change tensor creation to 3D and fix dim
    cmd = cmd.replace("torch.randn(", "torch.randn(1, ")
    cmd = cmd.replace("dim=1", "dim=2")
    return cmd


def fix_vanilla_dropout(entry):
    """Fix missing tabulate."""
    err = entry.get("error", "")
    if "tabulate" not in err:
        return None
    # tabulate should already be installed now; just return original command
    return entry["run_command"]


def get_fixer(source, subdir):
    """Return the appropriate fix function based on source file and subdir."""
    if "mxfp_gemm_cdna5" in source:
        return fix_gluon_mxfp_gemm
    if "f16_gemm_cdna5" in source:
        def combined(entry):
            r = fix_gluon_f16_gemm(entry)
            if r:
                return r
            return fix_gluon_f16_gemm_small_k(entry)
        return combined
    if "f16_fa_cdna5" in source:
        return fix_gluon_f16_fa
    if "mxfp_fa_cdna5" in source:
        return fix_gluon_mxfp_fa
    if subdir == "triton":
        def vanilla_fix(entry):
            for fn in [fix_vanilla_matmul, fix_vanilla_swiglu, fix_vanilla_reduce, fix_vanilla_dropout]:
                r = fn(entry)
                if r:
                    return r
            return None
        return vanilla_fix
    return None


def compile_kernel(cmd, cache_dir):
    """Run a kernel command and return list of .llir files found."""
    env = {**ENV, "TRITON_CACHE_DIR": cache_dir}
    try:
        result = subprocess.run(
            ["bash", "-c", f"cd {TRITON_DIR} && {cmd}"],
            env=env, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return [], "timeout"

    # Find .llir files
    llir_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".llir"):
                llir_files.append(os.path.join(root, f))

    stderr = result.stderr[-200:] if result.stderr else ""
    return llir_files, stderr


def process_subdir(subdir):
    """Process all failures in a subdirectory."""
    mapping_path = os.path.join(CORPUS_DIR, subdir, "mapping.json")
    if not os.path.exists(mapping_path):
        print(f"  No mapping.json in {subdir}")
        return 0

    with open(mapping_path) as f:
        mapping = json.load(f)

    fixed = 0
    for entry in mapping["kernels"]:
        if entry["status"] in ("success", "launch_failed_ir_captured"):
            continue

        source = entry.get("source", "")
        err = entry.get("error", "")

        # Skip unfixable errors
        if "OutOfMemoryError" in err or "OOM" in err.upper():
            continue
        if "make_opt_flags" in err:
            continue
        if "block scaled" in entry.get("label", "").lower():
            continue

        fixer = get_fixer(source, subdir)
        if not fixer:
            continue

        fixed_cmd = fixer(entry)
        if not fixed_cmd:
            continue

        label = entry["label"]
        print(f"  Fixing: {label}")
        print(f"    Old: {entry['run_command'][:100]}...")
        print(f"    New: {fixed_cmd[:100]}...")

        # Create isolated cache dir
        cache_dir = tempfile.mkdtemp(prefix=f"triton_recompile_{subdir}_")

        try:
            llir_files, stderr = compile_kernel(fixed_cmd, cache_dir)

            if llir_files:
                # Generate filename
                filename = entry.get("filename")
                if not filename:
                    safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', label.lower())
                    safe_label = re.sub(r'_+', '_', safe_label).strip('_')
                    filename = safe_label + ".llir"

                out_dir = os.path.join(CORPUS_DIR, subdir)
                primary = llir_files[0]
                out_path = os.path.join(out_dir, filename)
                shutil.copy(primary, out_path)

                entry["filename"] = filename
                entry["status"] = "launch_failed_ir_captured"
                entry["run_command"] = fixed_cmd
                entry.pop("error", None)
                entry["additional_files"] = []

                # Copy additional .llir files
                for i, extra in enumerate(llir_files[1:], 1):
                    extra_name = filename.replace(".llir", f"_helper{i}.llir")
                    shutil.copy(extra, os.path.join(out_dir, extra_name))
                    entry["additional_files"].append(extra_name)

                fixed += 1
                print(f"    OK: {len(llir_files)} .llir file(s)")
            else:
                print(f"    STILL FAILED: {stderr[:100]}")
                entry["run_command"] = fixed_cmd
                entry["error"] = f"Recompile failed: {stderr[:100]}"
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
    subdirs = ["gluon", "best", "triton", "aiter"]
    total_fixed = 0
    for sd in subdirs:
        print(f"\n=== Processing {sd} ===")
        n = process_subdir(sd)
        total_fixed += n
        print(f"  Fixed {n} kernels in {sd}")

    print(f"\n=== Total fixed: {total_fixed} ===")


if __name__ == "__main__":
    main()
