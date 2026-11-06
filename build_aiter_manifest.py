#!/usr/bin/env python3 -I
"""Build aiter model kernel manifest for MI450/gfx1250."""

import csv
import json
import os
import glob
import re
from datetime import datetime, timezone
from collections import defaultdict

AITER_ROOT = "/root/KernelExtract/aiter"
TRITON_ROOT = "/root/KernelExtract/triton"
OUTPUT_PATH = "/root/MI450KernelCorpus/MI450/agent/aiter_model_manifest.json"
TARGET_GFX = "gfx1250"
TRITON_COMMIT = "fd91c147ef635b9e86b87e9cf073448cf87d769d"

# ----- Phase 1: Parse CSVs and extract gfx1250 model kernel data -----

def parse_model_name_from_filename(filename):
    """Extract model name from CSV filename."""
    base = os.path.basename(filename).replace(".csv", "")

    # Patterns for model-specific configs in model_configs/
    # e.g., dsv4_bf16_tuned_gemm -> deepseek-v4
    # kimik3_gfx1250_tuned_grouped_fmoe -> kimi-k3
    model_map = {
        "dsv4": "deepseek-v4",
        "dsv41": "deepseek-v4.1",
        "dsv3": "deepseek-v3",
        "dsr1": "deepseek-r1",
        "glm47": "glm-4-7b",
        "glm53": "glm-5-3",
        "glm5": "glm-5",
        "kimik3tp8": "kimi-k3-tp8",
        "kimik3": "kimi-k3",
        "kimik2": "kimi-k2",
        "kimi": "kimi",
        "qwen3_5_397b": "qwen3.5-397b",
        "qwen3_5_35b": "qwen3.5-35b",
        "qwen3_8_27b": "qwen3.8-27b",
        "qwen3_8_2400b": "qwen3.8-2400b",
        "qwen3_8_2p4t": "qwen3.8-2.4t",
        "qwen3_8_flash": "qwen3.8-flash",
        "qwen3_8b": "qwen3-8b",
        "qwen3_6_35b": "qwen3.6-35b-a3b",
        "qwen3_235b": "qwen3-235b",
        "qwen3next_80b": "qwen3-next-80b",
        "qwen3_vl": "qwen3-vl",
        "qwen38_27b": "qwen3.8-27b",
        "qwen38_2p4t": "qwen3.8-2.4t",
        "qwen38_flash_next": "qwen3.8-flash-next",
        "qwen32B": "qwen-32b",
        "llama405B": "llama-405b",
        "llama70B": "llama-70b",
        "minimax_m3_eagle": "minimax-m3-eagle",
        "minimax_m3_mxfp4": "minimax-m3-mxfp4",
        "minimax_m3_mxfp8": "minimax-m3-mxfp8",
        "minimax_m3": "minimax-m3",
        "minimax_m25": "minimax-m2.5",
        "gptoss": "gptoss",
        "flux2": "flux-2",
        "wan21_vae": "wan2.1-vae",
        "wan22": "wan2.2",
        "qwenimage_vae": "qwenimage-vae",
        "inkling": "inkling",
        "hunyuan3": "hunyuan-3",
        "beam": "beam",
        "mimo_v2_5_pro": "mimo-v2.5-pro",
        "muse_glimmer": "muse-glimmer",
        "muse-spark-open-mxfp4": "muse-spark-open-mxfp4",
        "narrow_n": "narrow-n",
        "q3vl": "qwen3-vl",
        "k2_horizon_375b": "k2-horizon-375b",
        "gemma4_31b": "gemma4-31b",
        "hstu_attention": "hstu",
        "glm53_flash": "glm-5.3-flash",
        "glm53flash": "glm-5.3-flash",
        "a8w8_blockscale_tuned_fmoe_ds_v3": "deepseek-v3",
        "a8w8_blockscale_tuned_fmoe_glm5": "glm-5",
        "a8w8_blockscale_tuned_fmoe_glm5_1": "glm-5.1",
        "a8w8_blockscale_tuned_fmoe_glm5_3": "glm-5.3",
        "a8w8_blockscale_tuned_fmoe_glm5_3_flash": "glm-5.3-flash",
        "a8w8_blockscale_tuned_fmoe_minimax-m2_5": "minimax-m2.5",
        "a8w8_blockscale_tuned_fmoe_qwen3_235b": "qwen3-235b",
        "a8w8_blockscale_tuned_fmoe_qwen3_5_397b": "qwen3.5-397b",
        "a8w8_blockscale_tuned_fmoe_qwen3_8_flash_next": "qwen3.8-flash-next",
        "a8w8_blockscale_tuned_gemm_ds_v3": "deepseek-v3",
        "a8w8_blockscale_tuned_gemm_gemma4_31b": "gemma4-31b",
        "a8w8_blockscale_tuned_gemm_glm5": "glm-5",
        "a8w8_bpreshuffle_tuned_gemm_glm47": "glm-4-7b",
        "a8w8_bpreshuffle_tuned_gemm_kimik3": "kimi-k3",
        "a8w8_bpreshuffle_tuned_gemm_minimax_m3": "minimax-m3",
        "a8w8_bpreshuffle_tuned_gemm_kimik3tp8": "kimi-k3-tp8",
        "paged_mqa_logits_tuned_glm5": "glm-5",
    }

    # Try longest prefix match
    for prefix, model in sorted(model_map.items(), key=lambda x: -len(x[0])):
        if base.startswith(prefix + "_") or base == prefix or base.startswith(prefix + "."):
            return model

    # Try matching inside the name for model_configs subdir patterns
    for prefix, model in sorted(model_map.items(), key=lambda x: -len(x[0])):
        if prefix in base:
            return model

    return base


def determine_op_type(filename, kernel_name=""):
    """Determine operation type from filename and kernel name."""
    base = os.path.basename(filename).lower()
    kn = kernel_name.lower() if kernel_name else ""

    if "conv3d" in base or "conv3d" in kn:
        return "conv3d"
    if "batched_gemm" in base or "batched" in kn:
        return "batched_gemm"
    if "fmoe" in base or "fhmoe" in base or "grouped_fmoe" in base or "moe" in kn or "grouped_gemm" in kn:
        return "moe"
    if "mqa_logits" in base or "mqa" in kn:
        return "mqa_logits"
    if "attention" in base or "attn" in kn:
        return "attention"
    if "chunk_gdn" in base or "gdn" in kn:
        return "gated_delta_net"
    if "mega_moe_tp" in base:
        return "mega_moe_tp"
    if "gemm" in base or "gemm" in kn:
        return "gemm"
    if "hc_gated_residual" in base:
        return "gated_residual"
    if "hstu" in base:
        return "attention"
    if "iq2r" in base:
        return "quantization"

    return "gemm"


def determine_framework(kernel_name):
    """Determine kernel framework from kernel name prefix."""
    if not kernel_name or kernel_name == "auto":
        return "auto"
    kn = kernel_name.lower()
    if kn.startswith("flydsl_"):
        return "flydsl"
    if kn.startswith("grouped_gemm"):
        return "flydsl"  # grouped_gemm kernels in aiter are flydsl-based
    if "gluon" in kn:
        return "gluon"
    if kn.startswith("triton_") or kn.startswith("_triton_"):
        return "triton"
    return "flydsl"  # most aiter kernels are flydsl


def extract_dtype_info(row, headers):
    """Extract dtype information from CSV row."""
    dtype_info = {}
    for key in ["dtype", "q_dtype_a", "q_dtype_w", "outdtype"]:
        if key in headers:
            idx = headers.index(key)
            if idx < len(row):
                val = row[idx].strip()
                if val and val != "":
                    dtype_info[key] = val
    return dtype_info


def parse_gemm_csv(filepath):
    """Parse a GEMM-type CSV and extract gfx1250 rows."""
    results = []
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            headers = [h.strip() for h in headers]

            for row in reader:
                if not row or len(row) < 2:
                    continue
                gfx_idx = headers.index("gfx") if "gfx" in headers else 0
                if gfx_idx < len(row) and TARGET_GFX in row[gfx_idx].strip():
                    entry = {}
                    for i, h in enumerate(headers):
                        if i < len(row):
                            entry[h] = row[i].strip()
                    results.append(entry)
    except Exception as e:
        pass
    return results


def parse_all_model_configs():
    """Parse all model config CSVs and return gfx1250 entries."""
    model_data = defaultdict(lambda: {"config_files": [], "entries": []})

    # Process model_configs/ directory
    model_configs_dir = os.path.join(AITER_ROOT, "aiter/configs/model_configs")
    if os.path.isdir(model_configs_dir):
        for csv_file in sorted(glob.glob(os.path.join(model_configs_dir, "*.csv"))):
            rows = parse_gemm_csv(csv_file)
            if rows:
                model_name = parse_model_name_from_filename(csv_file)
                fname = os.path.basename(csv_file)
                if fname not in model_data[model_name]["config_files"]:
                    model_data[model_name]["config_files"].append(fname)
                for row in rows:
                    row["_source_file"] = fname
                    model_data[model_name]["entries"].append(row)

    # Also check top-level configs/ CSVs for gfx1250 data
    configs_dir = os.path.join(AITER_ROOT, "aiter/configs")
    for csv_file in sorted(glob.glob(os.path.join(configs_dir, "*.csv"))):
        rows = parse_gemm_csv(csv_file)
        if rows:
            fname = os.path.basename(csv_file)
            # These are generic configs, add them under a generic model name
            model_name = "generic"
            if fname not in model_data[model_name]["config_files"]:
                model_data[model_name]["config_files"].append(fname)
            for row in rows:
                row["_source_file"] = fname
                model_data[model_name]["entries"].append(row)

    return model_data


# ----- Phase 2: Locate kernel sources -----

def find_gluon_sources_triton():
    """Find Gluon kernel sources in the Triton repo."""
    sources = {}
    gluon_dir = os.path.join(TRITON_ROOT, "third_party/amd/python/examples/gluon")
    if os.path.isdir(gluon_dir):
        for f in glob.glob(os.path.join(gluon_dir, "*.py")):
            basename = os.path.basename(f)
            sources[basename] = f
    return sources


def find_triton_kernel_sources():
    """Find vanilla Triton kernel sources."""
    sources = {}
    # triton_kernels directory
    tk_dir = os.path.join(TRITON_ROOT, "python/triton_kernels/triton_kernels")
    if os.path.isdir(tk_dir):
        for f in glob.glob(os.path.join(tk_dir, "*.py")):
            sources[os.path.basename(f)] = f

    # tutorials
    tut_dir = os.path.join(TRITON_ROOT, "python/tutorials")
    if os.path.isdir(tut_dir):
        for f in glob.glob(os.path.join(tut_dir, "*.py")):
            sources[os.path.basename(f)] = f

    return sources


def find_aiter_gluon_sources():
    """Find Gluon kernel sources in aiter repo for gfx1250."""
    sources = {}
    gluon_dir = os.path.join(AITER_ROOT, "aiter/ops/triton/_gluon_kernels/gfx1250")
    if os.path.isdir(gluon_dir):
        for root, dirs, files in os.walk(gluon_dir):
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, AITER_ROOT)
                    sources[f] = {"path": full, "rel": rel}
    return sources


def find_aiter_flydsl_sources():
    """Find FlyDSL kernel sources in aiter for gfx1250."""
    sources = {}
    flydsl_dir = os.path.join(AITER_ROOT, "aiter/ops/flydsl")
    if os.path.isdir(flydsl_dir):
        for root, dirs, files in os.walk(flydsl_dir):
            for f in files:
                if f.endswith(".py") and "gfx1250" in f:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, AITER_ROOT)
                    sources[f] = {"path": full, "rel": rel}
    return sources


def find_aiter_triton_sources():
    """Find Triton kernel sources in aiter."""
    sources = {}
    triton_dir = os.path.join(AITER_ROOT, "aiter/ops/triton/_triton_kernels")
    if os.path.isdir(triton_dir):
        for root, dirs, files in os.walk(triton_dir):
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, AITER_ROOT)
                    sources[f] = {"path": full, "rel": rel}
    return sources


def map_kernel_to_source(kernel_name, op_type, dtype_info, gluon_triton_sources,
                         triton_kernel_sources, aiter_gluon_sources,
                         aiter_flydsl_sources, aiter_triton_sources):
    """Map a kernel name to its source file(s) and build run commands."""
    result = {"gluon": None, "triton": None, "flydsl": None}

    kn = kernel_name.lower() if kernel_name else ""

    # Determine the primary dtype for source matching
    dtype = dtype_info.get("dtype", "torch.bfloat16")

    # Map op type + dtype to Gluon source in Triton repo
    if op_type in ("gemm", "batched_gemm"):
        if "mxfp" in kn or "mxfp" in dtype or "float4" in dtype:
            gluon_source = "mxfp_gemm_cdna5.py"
        else:
            gluon_source = "f16_gemm_cdna5.py"
        if gluon_source in gluon_triton_sources:
            result["gluon"] = {
                "source": os.path.relpath(gluon_triton_sources[gluon_source], TRITON_ROOT),
                "source_repo": "triton",
                "entry_point": "persistent_gemm_tdm_pipelined_kernel" if "f16" in gluon_source else "mxfp_gemm_kernel",
                "target": "hip:gfx1250:64",
            }
    elif op_type == "moe":
        if "moe_cdna5.py" in gluon_triton_sources:
            result["gluon"] = {
                "source": os.path.relpath(gluon_triton_sources["moe_cdna5.py"], TRITON_ROOT),
                "source_repo": "triton",
                "entry_point": "moe_kernel",
                "target": "hip:gfx1250:64",
            }
    elif op_type == "attention":
        if "f16_fa_cdna5.py" in gluon_triton_sources:
            result["gluon"] = {
                "source": os.path.relpath(gluon_triton_sources["f16_fa_cdna5.py"], TRITON_ROOT),
                "source_repo": "triton",
                "entry_point": "flash_attention_kernel",
                "target": "hip:gfx1250:64",
            }

    # Check aiter Gluon kernels for gfx1250
    aiter_gluon_match = None
    if op_type in ("gemm", "batched_gemm"):
        for fname, info in aiter_gluon_sources.items():
            if "gemm" in fname.lower():
                if "a8w8" in kn and "a8w8" in fname.lower():
                    aiter_gluon_match = info
                    break
                elif "a16w16" in fname.lower() and ("bf16" in dtype.lower() or "float16" in dtype.lower()):
                    aiter_gluon_match = info
                    break
                elif "mxfp" in kn and "mxfp" in fname.lower():
                    aiter_gluon_match = info
                    break
    elif op_type == "moe":
        for fname, info in aiter_gluon_sources.items():
            if "moe" in fname.lower():
                aiter_gluon_match = info
                break

    if aiter_gluon_match and result["gluon"] is None:
        result["gluon"] = {
            "source": aiter_gluon_match["rel"],
            "source_repo": "aiter",
            "entry_point": "kernel",
            "target": "hip:gfx1250:64",
        }

    # Check for FlyDSL source
    if kn.startswith("flydsl_"):
        # Map flydsl kernel name to source
        for fname, info in aiter_flydsl_sources.items():
            fn_lower = fname.lower()
            if op_type == "gemm" and "gemm" in fn_lower:
                if "a8w8" in kn and "a8w8" in fn_lower:
                    result["flydsl"] = {
                        "source": info["rel"],
                        "source_repo": "aiter",
                        "entry_point": kernel_name,
                    }
                    break
                elif "a4w4" in kn and "a4w4" in fn_lower:
                    result["flydsl"] = {
                        "source": info["rel"],
                        "source_repo": "aiter",
                        "entry_point": kernel_name,
                    }
                    break
                elif "hgemm" in kn and "a16w16" in fn_lower:
                    result["flydsl"] = {
                        "source": info["rel"],
                        "source_repo": "aiter",
                        "entry_point": kernel_name,
                    }
                    break
                elif "mxfp" in kn and "mxfp" in fn_lower:
                    result["flydsl"] = {
                        "source": info["rel"],
                        "source_repo": "aiter",
                        "entry_point": kernel_name,
                    }
                    break
            elif op_type == "conv3d" and "conv3d" in fn_lower:
                result["flydsl"] = {
                    "source": info["rel"],
                    "source_repo": "aiter",
                    "entry_point": kernel_name,
                }
                break
            elif op_type == "moe" and "moe" in fn_lower:
                result["flydsl"] = {
                    "source": info["rel"],
                    "source_repo": "aiter",
                    "entry_point": kernel_name,
                }
                break
            elif op_type == "mqa_logits" and "mqa" in fn_lower:
                result["flydsl"] = {
                    "source": info["rel"],
                    "source_repo": "aiter",
                    "entry_point": kernel_name,
                }
                break

        # fallback: generic match
        if result["flydsl"] is None:
            for fname, info in aiter_flydsl_sources.items():
                fn_lower = fname.lower()
                if op_type == "gemm" and "gemm" in fn_lower:
                    result["flydsl"] = {
                        "source": info["rel"],
                        "source_repo": "aiter",
                        "entry_point": kernel_name,
                    }
                    break

    # Vanilla Triton source matching
    if op_type in ("gemm", "batched_gemm"):
        if "matmul.py" in triton_kernel_sources:
            result["triton"] = {
                "source": os.path.relpath(triton_kernel_sources["matmul.py"], TRITON_ROOT),
                "source_repo": "triton",
                "entry_point": "matmul",
            }
        elif "03-matrix-multiplication.py" in triton_kernel_sources:
            result["triton"] = {
                "source": os.path.relpath(triton_kernel_sources["03-matrix-multiplication.py"], TRITON_ROOT),
                "source_repo": "triton",
                "entry_point": "matmul_kernel",
            }

    return result


# ----- Phase 3: Build run commands -----

def build_run_command(kernel_name, op_type, shapes, source_info, framework):
    """Build a complete run command for a kernel."""
    if not source_info:
        return None

    source = source_info.get("source", "")
    source_repo = source_info.get("source_repo", "triton")
    entry_point = source_info.get("entry_point", "")

    if source_repo == "triton":
        base_dir = TRITON_ROOT
    else:
        base_dir = AITER_ROOT

    full_source = os.path.join(base_dir, source)

    M = shapes.get("M", shapes.get("token", "1"))
    N = shapes.get("N", shapes.get("model_dim", ""))
    K = shapes.get("K", shapes.get("inter_dim", ""))

    if framework == "gluon":
        target = source_info.get("target", "hip:gfx1250:64")
        if "gemm" in op_type or "batched" in op_type:
            cmd = (f"python3 {full_source} --target {target} "
                   f"--M {M} --N {N} --K {K}")
        elif "moe" in op_type:
            cmd = (f"python3 {full_source} --target {target}")
        elif "attention" in op_type:
            cmd = (f"python3 {full_source} --target {target}")
        else:
            cmd = f"python3 {full_source} --target {target}"

        compile_call = (f"triton.compile(fn={entry_point}, "
                        f"target=triton.runtime.driver.active.get_current_target(), "
                        f"signature=...)")
        return cmd, compile_call

    elif framework == "triton":
        if "gemm" in op_type:
            cmd = f"python3 {full_source} --M {M} --N {N} --K {K}"
        else:
            cmd = f"python3 {full_source}"
        compile_call = (f"triton.compile(fn={entry_point}, "
                        f"target='hip:gfx1250', signature=...)")
        return cmd, compile_call

    elif framework == "flydsl":
        cmd = f"python3 {full_source}"
        compile_call = None
        return cmd, compile_call

    return None, None


# ----- Phase 4: Build the manifest -----

def build_kernel_entry(entry, op_type, source_map, all_sources):
    """Build a single kernel entry for the manifest."""
    # Extract shapes (handle both upper and lowercase column names)
    shapes = {}
    for key in ["M", "N", "K", "B", "token", "model_dim", "inter_dim", "expert", "topk",
                 "D", "H", "W", "C", "kT", "kH", "kW",
                 "m", "n", "k", "b",
                 "stride_d", "stride_h", "stride_w",
                 "pad_d", "pad_h", "pad_w",
                 "dil_d", "dil_h", "dil_w", "groups"]:
        if key in entry:
            try:
                val = entry[key]
                if val and val not in ("", "False", "True"):
                    # Normalize: lowercase single-letter keys to uppercase
                    normalized_key = key.upper() if len(key) == 1 else key
                    shapes[normalized_key] = val
            except:
                pass

    # Kernel names (needed early for dtype inference)
    kernel_name_1 = entry.get("kernelName", entry.get("kernelName1", "auto"))
    kernel_name_2 = entry.get("kernelName2", None)

    # Extract dtype
    dtype_info = {}
    for key in ["dtype", "q_dtype_a", "q_dtype_w", "outdtype"]:
        if key in entry:
            dtype_info[key] = entry[key]

    # Primary dtype for display - infer from filename and kernel name if not in CSV
    primary_dtype = dtype_info.get("dtype", "")
    if not primary_dtype or primary_dtype == "":
        source_file = entry.get("_source_file", "").lower()
        kn_lower = (kernel_name_1 or "").lower()
        if "bf16" in source_file or "hgemm" in kn_lower or "abf16" in kn_lower:
            primary_dtype = "bf16"
        elif "mxfp8" in source_file or "mxfp8" in kn_lower:
            primary_dtype = "mxfp8"
        elif "mxfp4" in source_file or "mxfp4" in kn_lower:
            primary_dtype = "mxfp4"
        elif "fp8fp4" in source_file or "fp8fp4" in kn_lower:
            primary_dtype = "fp8fp4"
        elif "fp8" in source_file or "fp8" in kn_lower:
            primary_dtype = "fp8"
        elif "fp4" in source_file or "fp4" in kn_lower:
            primary_dtype = "fp4"
        elif "a8w8" in source_file or "a8w8" in kn_lower:
            primary_dtype = "fp8"
        elif "a4w4" in source_file or "a4w4" in kn_lower:
            primary_dtype = "fp4"
        elif "a4w6" in source_file:
            primary_dtype = "int4xint6"
        elif "a6w6" in source_file:
            primary_dtype = "int6"
        elif "a6w4" in source_file:
            primary_dtype = "int6xint4"
        elif "a16w4" in source_file:
            primary_dtype = "fp16xint4"
        elif "a16w16" in kn_lower:
            primary_dtype = "bf16"
        elif "bmm_mxfp8" in kn_lower:
            primary_dtype = "mxfp8"
        else:
            # Check w_scale_block for mxfp hints
            wsb = entry.get("w_scale_block", "")
            if wsb:
                primary_dtype = "mxfp8"  # blockscale configs are typically fp8
            else:
                primary_dtype = "unknown"
    if "torch." in primary_dtype:
        primary_dtype = primary_dtype.replace("torch.", "")

    framework = determine_framework(kernel_name_1)

    # Build shapes dict for output
    out_shapes = {}

    def safe_int(val):
        """Convert to int if possible, otherwise keep as string."""
        try:
            if isinstance(val, str) and val.lstrip("-").isdigit():
                return int(val)
        except:
            pass
        return val

    if "M" in shapes:
        out_shapes["M"] = safe_int(shapes["M"])
    elif "token" in shapes:
        out_shapes["M"] = safe_int(shapes["token"])
    if "N" in shapes:
        out_shapes["N"] = safe_int(shapes["N"])
    elif "model_dim" in shapes:
        out_shapes["N"] = safe_int(shapes["model_dim"])
    if "K" in shapes:
        out_shapes["K"] = safe_int(shapes["K"])
    elif "inter_dim" in shapes:
        out_shapes["K"] = safe_int(shapes["inter_dim"])
    if "B" in shapes:
        out_shapes["B"] = safe_int(shapes["B"])
    out_shapes["dtype"] = primary_dtype

    # Add extra shape params for special ops
    # For MOE entries, also include act_type and quant info in shapes
    if op_type == "moe":
        for key in ["expert", "topk", "model_dim", "inter_dim"]:
            if key in shapes:
                try:
                    out_shapes[key] = int(shapes[key]) if shapes[key].lstrip("-").isdigit() else shapes[key]
                except:
                    out_shapes[key] = shapes[key]
        # Extract q_dtype_a and q_dtype_w for dtype display
        q_a = dtype_info.get("q_dtype_a", "")
        q_w = dtype_info.get("q_dtype_w", "")
        if q_a and "torch." in q_a:
            out_shapes["q_dtype_a"] = q_a.replace("torch.", "")
        if q_w and "torch." in q_w:
            out_shapes["q_dtype_w"] = q_w.replace("torch.", "")
    if op_type == "conv3d":
        for key in ["D", "H", "W", "C", "kT", "kH", "kW", "B",
                     "stride_d", "stride_h", "stride_w",
                     "pad_d", "pad_h", "pad_w",
                     "dil_d", "dil_h", "dil_w", "groups"]:
            if key in shapes:
                out_shapes[key] = safe_int(shapes[key])

    # Additional params
    extra = {}
    for key in ["bias", "scaleAB", "bpreshuffle", "libtype", "splitK",
                 "w_scale_block", "act_type", "q_type", "use_g1u1", "ep_fused",
                 "a_preshuffle"]:
        if key in entry and entry[key] not in ("", None):
            extra[key] = entry[key]

    # Performance data
    perf = {}
    for key in ["us", "tflops", "bw"]:
        if key in entry and entry[key] not in ("", "0", None):
            try:
                perf[key] = float(entry[key])
            except:
                perf[key] = entry[key]

    # Map to sources
    gluon_triton_srcs, triton_kernel_srcs, aiter_gluon_srcs, aiter_flydsl_srcs, aiter_triton_srcs = all_sources
    source_mapping = map_kernel_to_source(
        kernel_name_1, op_type, dtype_info,
        gluon_triton_srcs, triton_kernel_srcs,
        aiter_gluon_srcs, aiter_flydsl_srcs, aiter_triton_srcs
    )

    # Build gluon entry
    gluon_entry = None
    if source_mapping["gluon"]:
        src = source_mapping["gluon"]
        cmd, compile_call = build_run_command(
            kernel_name_1, op_type, shapes, src, "gluon")
        gluon_entry = {
            "source": src["source"],
            "source_repo": src["source_repo"],
            "entry_point": src["entry_point"],
            "target": src.get("target", "hip:gfx1250:64"),
            "run_command": cmd,
            "compile_call": compile_call,
        }

    # Build triton entry
    triton_entry = None
    if source_mapping["triton"]:
        src = source_mapping["triton"]
        cmd, compile_call = build_run_command(
            kernel_name_1, op_type, shapes, src, "triton")
        triton_entry = {
            "source": src["source"],
            "source_repo": src["source_repo"],
            "entry_point": src["entry_point"],
            "run_command": cmd,
            "compile_call": compile_call,
        }

    # Build flydsl entry
    flydsl_entry = None
    if source_mapping["flydsl"]:
        src = source_mapping["flydsl"]
        cmd, compile_call = build_run_command(
            kernel_name_1, op_type, shapes, src, "flydsl")
        flydsl_entry = {
            "source": src["source"],
            "source_repo": src["source_repo"],
            "entry_point": src["entry_point"],
            "run_command": cmd,
        }

    kernel_entry = {
        "op_type": op_type,
        "shapes": out_shapes,
        "aiter_kernel_name": kernel_name_1,
        "framework": framework,
        "gluon": gluon_entry,
        "triton": triton_entry,
    }

    if flydsl_entry:
        kernel_entry["flydsl"] = flydsl_entry

    if kernel_name_2 and kernel_name_2.strip():
        kernel_entry["aiter_kernel_name_stage2"] = kernel_name_2.strip()

    if extra:
        kernel_entry["extra_params"] = extra

    if perf:
        kernel_entry["performance"] = perf

    return kernel_entry


def deduplicate_kernels(models):
    """Add shared_with field for kernels that appear across models."""
    # Build index of (kernel_name, shapes_key) -> list of models
    kernel_index = defaultdict(list)
    for model_name, model_data in models.items():
        for kernel in model_data.get("kernels", []):
            kname = kernel.get("aiter_kernel_name", "")
            shapes = kernel.get("shapes", {})
            # Create a hashable key
            shapes_key = json.dumps(shapes, sort_keys=True)
            key = (kname, shapes_key)
            kernel_index[key].append(model_name)

    # Add shared_with where applicable
    for model_name, model_data in models.items():
        for kernel in model_data.get("kernels", []):
            kname = kernel.get("aiter_kernel_name", "")
            shapes = kernel.get("shapes", {})
            shapes_key = json.dumps(shapes, sort_keys=True)
            key = (kname, shapes_key)
            shared = sorted(set(m for m in kernel_index[key] if m != model_name))
            if shared:
                kernel["shared_with"] = shared

    return models


def main():
    print("Phase 1: Extracting model kernel requirements from aiter configs...")
    model_data = parse_all_model_configs()

    # Filter out models with no gfx1250 data
    print(f"  Found {len(model_data)} model groups with gfx1250 data")
    for m, d in model_data.items():
        print(f"    {m}: {len(d['entries'])} gfx1250 entries from {len(d['config_files'])} config files")

    print("\nPhase 2: Locating kernel sources...")
    gluon_triton_srcs = find_gluon_sources_triton()
    print(f"  Triton Gluon sources: {len(gluon_triton_srcs)}")
    triton_kernel_srcs = find_triton_kernel_sources()
    print(f"  Triton kernel sources: {len(triton_kernel_srcs)}")
    aiter_gluon_srcs = find_aiter_gluon_sources()
    print(f"  Aiter Gluon gfx1250 sources: {len(aiter_gluon_srcs)}")
    aiter_flydsl_srcs = find_aiter_flydsl_sources()
    print(f"  Aiter FlyDSL gfx1250 sources: {len(aiter_flydsl_srcs)}")
    aiter_triton_srcs = find_aiter_triton_sources()
    print(f"  Aiter Triton sources: {len(aiter_triton_srcs)}")

    all_sources = (gluon_triton_srcs, triton_kernel_srcs, aiter_gluon_srcs,
                   aiter_flydsl_srcs, aiter_triton_srcs)

    print("\nPhase 3: Building kernel entries and run commands...")
    models = {}
    for model_name, data in sorted(model_data.items()):
        if model_name == "generic":
            continue  # Skip generic entries
        kernels = []
        seen_keys = set()
        for entry in data["entries"]:
            source_file = entry.get("_source_file", "")
            kernel_name = entry.get("kernelName", entry.get("kernelName1", "auto"))
            op_type = determine_op_type(source_file, kernel_name)

            # Build dedup key
            shapes_vals = []
            for k in ["M", "N", "K", "token", "model_dim", "inter_dim", "B"]:
                if k in entry:
                    shapes_vals.append(f"{k}={entry[k]}")
            dedup_key = f"{kernel_name}|{'|'.join(shapes_vals)}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            kernel_entry = build_kernel_entry(entry, op_type, {}, all_sources)
            kernels.append(kernel_entry)

        if kernels:
            models[model_name] = {
                "aiter_config_files": data["config_files"],
                "kernels": kernels,
            }

    print(f"  Built entries for {len(models)} models")
    for m, d in sorted(models.items()):
        print(f"    {m}: {len(d['kernels'])} kernel entries")

    print("\nPhase 4: Deduplication and writing manifest...")
    models = deduplicate_kernels(models)

    # Collect gluon tuning config info
    gluon_tuning_configs = []
    gluon_config_dir = os.path.join(AITER_ROOT, "aiter/ops/triton/configs/gfx1250")
    if os.path.isdir(gluon_config_dir):
        for root, dirs, files in os.walk(gluon_config_dir):
            for f in files:
                if f.endswith(".json"):
                    rel = os.path.relpath(os.path.join(root, f), AITER_ROOT)
                    gluon_tuning_configs.append(rel)

    manifest = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "aiter_repo": "https://github.com/ROCm/aiter.git",
            "aiter_branch": "main",
            "triton_repo": "https://github.com/triton-lang/triton.git",
            "triton_branch": "main",
            "triton_commit": TRITON_COMMIT,
            "target_arch": TARGET_GFX,
            "description": "Model kernel manifest mapping AI models to gfx1250 Triton/Gluon/FlyDSL kernel shapes and sources",
        },
        "gluon_tuning_configs": {
            "base_path": "aiter/ops/triton/configs/gfx1250",
            "count": len(gluon_tuning_configs),
            "categories": sorted(set(
                os.path.dirname(c).replace("aiter/ops/triton/configs/gfx1250/", "").split("/")[0]
                for c in gluon_tuning_configs if "/" in c.replace("aiter/ops/triton/configs/gfx1250/", "")
            )),
        },
        "flydsl_kernel_sources": {
            "base_path": "aiter/ops/flydsl",
            "gfx1250_files": sorted([info["rel"] for info in aiter_flydsl_srcs.values()]),
        },
        "gluon_kernel_sources": {
            "triton_repo": sorted([
                os.path.relpath(p, TRITON_ROOT) for p in gluon_triton_srcs.values()
            ]),
            "aiter_repo_gfx1250": sorted([info["rel"] for info in aiter_gluon_srcs.values()]),
        },
        "models": models,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nManifest written to {OUTPUT_PATH}")
    # Print size info
    file_size = os.path.getsize(OUTPUT_PATH)
    total_kernels = sum(len(m["kernels"]) for m in models.values())
    print(f"  File size: {file_size / 1024:.1f} KB")
    print(f"  Total models: {len(models)}")
    print(f"  Total kernel entries: {total_kernels}")


if __name__ == "__main__":
    main()
