# MI450 LLVM IR Corpus — Generation Report

**Generated:** 2026-11-05  
**Target architecture:** gfx1250 (MI450)  
**Triton commit:** `fd91c147ef635b9e86b87e9cf073448cf87d769d` (main)  
**Method:** Cross-compilation via `TRITON_OVERRIDE_ARCH=gfx1250` inside ROCm Docker container (no gfx1250 GPU required)

## Summary

| Manifest | Compiled | Failed | Total | .llir files |
|----------|----------|--------|-------|-------------|
| OAI | 8 | 0 | 8 | 8 |
| Gluon | 167 | 21 | 188 | 216 |
| Best | 12 | 1 | 13 | 12 |
| Triton (vanilla) | 62 | 3 | 65 | 131 |
| Aiter | 30 | 15 | 45 | 30 |
| **Total** | **279** | **40** | **319** | **397** |

Some kernels (notably persistent GEMM) produce multiple `.llir` files per invocation, which is why the .llir file count exceeds the compiled kernel count.

## Manifests

| File | Kernels | Description |
|------|---------|-------------|
| `OAI_kernel_manifest.json` | 8 | Pre-existing. GQA/MQA flash attention with MXFP (e4m3, e2m1) at decode and prefill shapes. |
| `gluon_kernel_manifest.json` | 188 | Gluon DSL kernels: MXFP FA, FP16 FA, MXFP GEMM, FP16 GEMM, StreamK GEMM, Warp-pipeline GEMM, MoE. Multiple sizes, dtypes, and algorithmic variants per kernel family. |
| `best_kernel_manifest.json` | 13 | Best single config per op type. Excludes inline assembly and sched_barriers where possible. |
| `vanilla_kernel_manifest.json` | 65 | Standard Triton kernels from tutorials (vector-add, softmax, GEMM, FA, layer norm, dropout, grouped GEMM, persistent matmul) and production triton_kernels package (matmul, reduce, swiglu, topk, compaction). |
| `aiter_model_manifest.json` | 45 | Model-specific kernels from aiter configs (DeepSeek R1/V3/V4, Kimi K2/K3, MiniMax M3, Qwen 3.5). Trimmed to 15 invocations per unique kernel script with diverse shape sampling. |

## Kernel directories

```
MI450/kernels/
├── OAI/          8 .llir — MXFP flash attention (prefill + decode)
├── gluon/      216 .llir — Gluon CDNA5 kernels (FA, GEMM, MoE)
├── best/        12 .llir — Best config per op (GEMM, FA, softmax, layernorm, topk, swiglu, reduce)
├── triton/     131 .llir — Vanilla Triton tutorials + production kernels
└── aiter/       30 .llir — Model-specific GEMM kernels from aiter configs
```

Each directory contains a `mapping.json` documenting which manifest entry produced which `.llir` file, along with the run command, shapes, and status.

## Failure analysis

### OAI — 0 failures
All 8 MXFP flash attention kernels compiled successfully.

### Gluon — 21 failures (11%)

| Count | Category | Root cause | Fixable? |
|-------|----------|------------|----------|
| 18 | GPU OOM | Medium/large MoE kernels (batch≥2048, ≥128 experts) require >16 GB VRAM for tensor allocation before compilation | No — needs ≥32 GB GPU |
| 3 | make_opt_flags upstream bug | MoE fp8/fp8 dtype: `make_opt_flags()` missing `intermediate_out_dtype` argument | No — upstream Triton fix needed |

> **Note:** 67 of the original 88 failures were config issues (wrong flags, missing args, reference check crashes) that were fixed by correcting the invocation commands. See `MI450/agent/knowledge/gluon_configs.md` for the full configuration reference.

### Best — 1 failure

| Kernel | Reason | Fixable? |
|--------|--------|----------|
| MoE fp8/mx4 large batch=8192 expts=256 | GPU OOM (31.6 GB allocation) | No — needs ≥32 GB GPU |

### Triton (vanilla) — 3 failures

| Count | Category | Root cause | Fixable? |
|-------|----------|------------|----------|
| 3 | Block-Scaled GEMM | Tutorial uses CDNA4-specific kernel (`gfx950`); no CDNA5 path exists. Equivalent gfx1250 functionality is in `mxfp_gemm_cdna5.py` (Gluon). | No — not applicable to gfx1250 |

### Aiter — 15 failures

| Count | Category | Root cause | Fixable? |
|-------|----------|------------|----------|
| 15 | GPU OOM | MoE kernels need >16 GB VRAM for tensor allocation | No — needs ≥32 GB GPU |

## Config fixes applied

The following issues were identified and corrected during recompilation (108 kernels recovered):

| Category | Count | Fix applied |
|----------|-------|-------------|
| MXFP GEMM missing `--scale_preshuffled` | 36 | Added `--scale_preshuffled` flag (required for non-baseline schedules); ensured `BLOCK_K ≥ 256` |
| MXFP GEMM fp4 missing `--with_a_scale` | 12 | Added `--with_a_scale` flag (required for fp4 × fp4 GEMM) |
| F16 FA GQA reference check crash | 10 | Invoked via Python with `check=False` (the reference SDPA doesn't handle GQA, but the kernel compiles fine) |
| F16 GEMM warp-specialized wrong num_warps | 5 | Changed `--num-warps 4` to `--num-warps 8` (warp-specialized requires 8 or 16) |
| F16 GEMM single-warp missing `--prefetch-lds` | 5 | Added `--prefetch-lds` (required for single-warp-schedule mode) |
| Production matmul missing `bias` | 10 | Added `bias=None` argument |
| Production swiglu missing `precision_config` | 4 | Added `PrecisionConfig(limit=float("inf"))` |
| Production reduce 2D→3D | 5 | Reshaped input to 3D with `unsqueeze(0)` |
| MXFP FA `--profile` crash at seqlen=1024 | 7 | Removed `--profile` flag (causes `CompiledKernel.module` attribute error during cross-compilation) |
| Seeded dropout missing `tabulate` | 4 | Installed `tabulate` dependency |
| MXFP GEMM small M with sliceNK | 9 | Switched to `--schedule baseline` (sliceNK requires M ≥ 256) |
| F16 GEMM small K | 1 | Reduced `--block_k` to satisfy `K/BLOCK_K ≥ NUM_BUFFERS` |

See `MI450/agent/knowledge/` for detailed configuration references per kernel family.

## Environment

- **Docker container:** ROCm PyTorch image with HIP 7.16, Python 3.12, GCC 13.3, CMake 3.28
- **Triton version:** 3.9.0 (built from source with AMD codegen revision `08b05a3352ee`)
- **Cross-compilation:** `TRITON_OVERRIDE_ARCH=gfx1250` + `TRITON_ALWAYS_COMPILE=1` — no gfx1250 GPU required
- **IR capture method:** per-kernel Triton cache isolation (`TRITON_CACHE_DIR`) with `TRITON_KERNEL_DUMP` fallback

## Known limitations

- **MoE kernels require large GPU memory.** All MoE kernels at medium/large configurations (batch≥2048, ≥128 experts) fail with OOM on a 16 GB GPU. Tensor allocation happens before the Triton JIT sees the kernel launch. Requires ≥32 GB VRAM.
- **MoE fp8/fp8 dtype combination is broken.** `moe_cdna5.py` calls `make_opt_flags()` without the required `intermediate_out_dtype` argument. This is an upstream bug in Triton's MoE example code. bf16/mx4 and fp8/mx4 work fine.
- **Block-scaled matmul tutorial is CDNA4-only.** The `10-block-scaled-matmul.py` tutorial only supports `gfx950` (MI300) on the AMD path. The equivalent gfx1250 functionality is covered by `mxfp_gemm_cdna5.py` in the Gluon manifest.
- **Some .llir files may be identical** when the kernel's shape parameters (M, N, K) are runtime arguments rather than compile-time specialization constants. The Triton JIT compiles once per unique set of meta-parameters (block sizes, num_warps, pipeline config), not per runtime shape.
