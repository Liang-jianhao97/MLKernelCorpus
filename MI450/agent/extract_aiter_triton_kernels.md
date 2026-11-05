# IR Corpus Agent: Build Model Kernel Manifest

## Objective

Build `aiter_model_manifest.json` — a machine-readable manifest that maps every tracked AI model to the Triton kernels it uses, with exact shapes, run commands, and source locations. This manifest will be consumed by a downstream agent that compiles each kernel and collects LLVM IR.

## Setup

Read `forks.txt` in your working directory to find:
- The path and branch of the **aiter** source repository
- The path and branch of the **Triton** source repository

Check out the specified branches before proceeding.

## Phase 1: Extract model kernel requirements from aiter

The aiter repo maintains per-model kernel configuration files. Start by looking in `aiter/configs/model_configs/` — this directory contains CSV files that map AI models to the kernel shapes they use. If this directory has moved or been restructured, search the repo for the model config data (look for CSV files containing columns like `gfx`, `M`, `N`, `K`, `dtype`, `kernelName`).

For each CSV file:
1. **Identify the model name** from the filename (e.g. `bf16_tuned_gemm_dsv4.csv` → model `dsv4` which is DeepSeek-V4).
2. **Filter to gfx1250 rows only** — we are targeting MI450. The `gfx` column will contain `gfx1250`.
3. **Extract the shape parameters** for each row: M, N, K, dtype, and any other shape-defining columns (bias, scaleAB, bpreshuffle, splitK, etc.).
4. **Extract the kernelName** — this encodes which kernel implementation wins for that shape. The kernel name prefix indicates the framework:
   - `flydsl_*` → FlyDSL kernel
   - Names matching entries in `aiter/ops/triton/_gluon_kernels/gfx1250/` → Gluon kernel
   - Other names may map to CK/HIP or HSA implementations
5. **Identify the op type** from the CSV filename and kernel name. Common op types: GEMM (various dtypes), fused MoE, attention (MLA, FMHA, MQA), normalization, quant/dequant, conv.

Also check `aiter/ops/triton/configs/gfx1250/` for Gluon tuning configuration JSONs — these contain the tuning parameters (tile sizes, wave counts, pipeline stages, etc.) needed to run each Gluon kernel at its tuned configuration.

Produce an intermediate data structure mapping:
```
model → [{ op_type, shapes: {M, N, K, dtype, ...}, kernel_name, framework_hint }]
```

## Phase 2: Locate Triton kernel sources and build run commands

For each unique kernel identified in Phase 1, find the corresponding Triton kernel source in the Triton repo. Search in this order:

### Gluon kernels (gfx1250-optimized)
Look in the Triton repo under `third_party/amd/python/examples/gluon/` for gfx1250/cdna5 kernels:
- `f16_gemm_cdna5.py` — FP16 GEMM with WMMA
- `f16_gemm_streamk_cdna5.py` — stream-K persistent GEMM
- `gemm_warp_pipeline_cdna5.py` — warp-pipelined GEMM
- `mxfp_gemm_cdna5.py` — MxFP GEMM with TDM
- `f16_fa_cdna5.py` — FP16 flash attention
- `mxfp_fa_cdna5.py` — MxFP flash attention with TDM
- `moe_cdna5.py` — full MoE kernel

Also check `third_party/amd/python/examples/gluon/` for any new kernels that may have been added since this prompt was written.

### Vanilla Triton kernels
Look in:
- `python/triton_kernels/` — the `triton-kernels` package (matmul, topk, swiglu, reduce, mxfp numerics)
- `python/tutorials/` — tutorial kernels (fused softmax, matrix multiplication, layer norm, fused attention, grouped GEMM, persistent matmul, block-scaled matmul)

### Aiter's own Triton/Gluon kernels
If a kernel type exists in aiter but not in the Triton repo (e.g. MLA decode, index-score attention, delta attention, paged attention), look in the aiter repo under:
- `aiter/ops/triton/_gluon_kernels/gfx1250/` — Gluon kernels
- `aiter/ops/triton/_triton_kernels/` — vanilla Triton kernels

For each kernel found, determine:
1. **Source file path** (relative to the repo root)
2. **Entry point** — the `@triton.jit` or `@triton.autotune` decorated function name
3. **How to compile it** — the `triton.compile()` call or command-line invocation needed. For Gluon kernels, the target is `hip:gfx1250:64`. Check if the file has a `__main__` block, argument parser, or test harness that shows how to invoke it with specific shapes.
4. **Shape parameters** — map the aiter config shapes (M, N, K, dtype) to the kernel's actual parameter names.

## Phase 3: Build run commands

For each (model, kernel, shape) triple, construct the exact command to compile and run the kernel. There are two cases:

### Gluon kernel available
If a Gluon version exists for gfx1250, record **both**:
- The Gluon command (targeting `hip:gfx1250:64`)
- The vanilla Triton equivalent command (if one exists)

### Vanilla Triton only
If only a vanilla Triton kernel exists, record that command.

For each command, include:
- The Python invocation or `triton.compile()` call
- All shape arguments (M, N, K, batch, heads, etc.)
- dtype arguments
- The target architecture string
- Any tuning parameters from the aiter config

## Phase 4: Write the manifest

Write `aiter_model_manifest.json` with this schema:

```json
{
  "metadata": {
    "generated_at": "ISO-8601 timestamp",
    "aiter_repo": "path from forks.txt",
    "aiter_branch": "branch from forks.txt",
    "triton_repo": "path from forks.txt",
    "triton_branch": "branch from forks.txt",
    "target_arch": "gfx1250"
  },
  "models": {
    "deepseek-v4": {
      "aiter_config_files": ["bf16_tuned_gemm_dsv4.csv", "..."],
      "kernels": [
        {
          "op_type": "gemm",
          "shapes": {
            "M": 1,
            "N": 7168,
            "K": 2048,
            "dtype": "bf16"
          },
          "aiter_kernel_name": "flydsl_hgemm_256x128_gfx1250",
          "gluon": {
            "source": "third_party/amd/python/examples/gluon/f16_gemm_cdna5.py",
            "entry_point": "matmul_kernel",
            "target": "hip:gfx1250:64",
            "run_command": "python f16_gemm_cdna5.py --M 1 --N 7168 --K 2048",
            "compile_call": "triton.compile(src, target='hip:gfx1250:64', ...)"
          },
          "triton": {
            "source": "python/triton_kernels/matmul/matmul.py",
            "entry_point": "matmul_kernel",
            "run_command": "python -m triton_kernels.matmul --M 1 --N 7168 --K 2048",
            "compile_call": "triton.compile(src, ...)"
          }
        },
        {
          "op_type": "fused_moe",
          "shapes": { "M": 2048, "N": 7168, "K": 2048, "experts": 256, "topk": 8, "dtype": "fp8" },
          "aiter_kernel_name": "grouped_gemm1_a8w4_gugu",
          "gluon": {
            "source": "third_party/amd/python/examples/gluon/moe_cdna5.py",
            "entry_point": "moe_kernel",
            "target": "hip:gfx1250:64",
            "run_command": "python moe_cdna5.py --M 2048 --N 7168 --K 2048 --experts 256 --topk 8"
          },
          "triton": null
        }
      ]
    },
    "kimi-k3": { "...": "..." }
  }
}
```

### Rules for the manifest:
- If a Gluon kernel exists, `"gluon"` is populated and `"triton"` is populated only if a vanilla equivalent also exists. If no vanilla equivalent, set `"triton": null`.
- If only a vanilla Triton kernel exists, `"gluon": null` and `"triton"` is populated.
- If neither exists in the Triton repo (kernel is aiter-only), still include the entry but use aiter source paths and note `"source_repo": "aiter"` in both the gluon and triton objects.
- Deduplicate: if multiple models use the same kernel at the same shape, each model still gets its own entry (the downstream agent needs per-model organization), but add a `"shared_with": ["model1", "model2"]` field.
- Every `run_command` must be a complete, copy-pasteable command. Don't use placeholders. If the kernel doesn't have a CLI entry point, write the minimal Python one-liner that compiles it.
- Record the `compile_call` as the Python code to compile the kernel with `triton.compile()`, including all necessary imports and arguments. This is what the downstream IR collection agent will actually execute.

## Important notes

- **Target architecture is gfx1250 only.** Ignore rows for other architectures.
- **The aiter config structure may evolve.** Don't hardcode CSV column positions — read the header row to find column names.
- **Some kernels may not have a direct Triton equivalent.** DeepSeek's MLA (Multi-Latent Attention), MiniMax's index-score attention, and Kimi's delta attention are model-specific and may only exist in aiter. Include them in the manifest with aiter source paths.
- **Tuning configs matter.** The tuning parameters (tile size, num_warps, num_stages, etc.) determine the exact IR that gets generated. Include them in the manifest when available (from `aiter/ops/triton/configs/gfx1250/` or from the kernel's own autotuning configs).
- **Both `gfx1250` and `gfx1250-strict` are valid arch variants.** Check aiter's `arch_info.py` for `is_gluon_avail()` / `is_tdm_avail()` to understand which features each variant enables.
