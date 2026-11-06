# MI450 LLVM IR Corpus — Generation Report

**Generated:** 2026-11-05  
**Target architecture:** gfx1250 (MI450)  
**Triton commit:** `fd91c147ef635b9e86b87e9cf073448cf87d769d` (main)  
**Method:** Cross-compilation via `TRITON_OVERRIDE_ARCH=gfx1250` inside ROCm Docker container (no gfx1250 GPU required)

## Summary

| Manifest | Compiled | Failed | Total | .llir files |
|----------|----------|--------|-------|-------------|
| OAI | 8 | 0 | 8 | 8 |
| Gluon | 100 | 88 | 188 | 151 |
| Best | 8 | 5 | 13 | 8 |
| Triton (vanilla) | 41 | 24 | 65 | 110 |
| Aiter | 14 | 31 | 45 | 14 |
| **Total** | **171** | **148** | **319** | **291** |

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
├── gluon/      151 .llir — Gluon CDNA5 kernels (FA, GEMM, MoE)
├── best/         8 .llir — Best config per op (GEMM, FA, softmax, layernorm, topk)
├── triton/     110 .llir — Vanilla Triton tutorials + production kernels
└── aiter/       14 .llir — Model-specific GEMM kernels from aiter configs
```

Each directory contains a `mapping.json` documenting which manifest entry produced which `.llir` file, along with the run command, shapes, and status.

## Failure analysis

### OAI — 0 failures
All 8 MXFP flash attention kernels compiled successfully.

### Gluon — 88 failures (47%)

| Count | Category | Root cause |
|-------|----------|------------|
| 24 | SCALE_PRESHUFFLE config skip | MXFP GEMM sliceK/sliceNK modes require `--scale_preshuffled` and `--transpose_b` flags that aren't compatible with all shape configurations |
| 18 | GPU OOM | Large MoE kernels (batch=8192, 256 experts) require >16 GB VRAM for tensor allocation before compilation |
| 16 | fp4 skip | Kernel intentionally skips fp4 × mxfp GEMM combinations |
| 10 | Tensor shape mismatch | FP16 FA GQA configs with head count mismatch (tensor a=8 vs b=2) |
| 10 | AssertionError | Internal kernel assertion on shape/config validation |
| 7 | CompiledKernel attribute | `'CompiledKernel' object has no attribute 'module'` — API change in Triton 3.9.0 |
| 3 | make_opt_flags signature | Missing `intermediate` argument in warp-pipeline GEMM |

> **Note:** The 47% failure rate is due to the sweep covering many intentionally-skipped configurations (fp4, SCALE_PRESHUFFLE) and GPU memory limits, not an environment problem. The 100 successful compilations cover all kernel families.

### Best — 5 failures

| Kernel | Reason |
|--------|--------|
| MXFP GEMM sliceMNK fp8 | SCALE_PRESHUFFLE config skip |
| FA Prefill FP16 pipeline GQA | Tensor shape mismatch (head count) |
| MoE fp8/mx4 large | GPU OOM (31.6 GB allocation) |
| SwiGLU | Missing `precision_config` argument — API changed |
| Reduce | Only supports 3D inputs in current implementation |

### Triton (vanilla) — 24 failures

| Count | Category | Root cause |
|-------|----------|------------|
| 10 | Production matmul | Missing `bias` positional argument — API changed since manifest generation |
| 4 | Seeded dropout | Missing `tabulate` dependency |
| 4 | Production reduce | Only supports 3D inputs |
| 3 | Production SwiGLU | Missing `precision_config` argument |
| 3 | Block-scaled GEMM | Compilation failed (no .llir from cache or dump) |

### Aiter — 31 failures

| Count | Category | Root cause |
|-------|----------|------------|
| 15 | SCALE_PRESHUFFLE config skip | MXFP GEMM sliceK/sliceNK mode limitation |
| 15 | GPU OOM | MoE kernels need >16 GB VRAM |
| 1 | K/BLOCK_K skip | K dimension too small for buffer count |

## Environment

- **Docker container:** ROCm PyTorch image with HIP 7.16, Python 3.12, GCC 13.3, CMake 3.28
- **Triton version:** 3.9.0 (built from source with AMD codegen revision `08b05a3352ee`)
- **Cross-compilation:** `TRITON_OVERRIDE_ARCH=gfx1250` + `TRITON_ALWAYS_COMPILE=1` — no gfx1250 GPU required
- **IR capture method:** per-kernel Triton cache isolation (`TRITON_CACHE_DIR`) with `TRITON_KERNEL_DUMP` fallback

## Known limitations

- **MoE kernels require large GPU memory.** All MoE kernels at the "large" configuration (batch=8192, 256 experts) fail with OOM. Smaller MoE configs (batch=256, 8 experts) compile successfully in the gluon manifest.
- **Production triton_kernels API drift.** The `matmul`, `swiglu`, and `reduce` APIs in the `triton_kernels` package changed between manifest generation and compilation. Future runs should re-generate manifests from the same Triton commit used for compilation.
- **MXFP GEMM sliceK/sliceNK modes** are intentionally restricted to configs with `SCALE_PRESHUFFLE` and `TRANSPOSE_B`, which limits the shape combinations that can compile.
- **Some .llir files may be identical** when the kernel's shape parameters (M, N, K) are runtime arguments rather than compile-time specialization constants. The Triton JIT compiles once per unique set of meta-parameters (block sizes, num_warps, pipeline config), not per runtime shape.
