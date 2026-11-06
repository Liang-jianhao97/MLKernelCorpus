# IR Corpus Agent: Build Triton Kernel Manifests for MI450

## Objective

Investigate the Triton source repository to discover every ML operation kernel that targets MI450 / gfx1250. For each kernel, generate multiple run commands covering different problem sizes and configurations. Produce three separate manifests:

- `vanilla_kernel_manifest.json` — kernels written in standard Triton (`@triton.jit` / `@triton.autotune`)
- `gluon_kernel_manifest.json` — kernels written using the Gluon DSL (typically in `third_party/amd/python/examples/gluon/`)
- `best_kernel_manifest.json` — kernels with known best configurations, such as those documented in tutorials or examples with hand-tuned parameters. Only include configs that do **not** use inline assembly (`tl.inline_asm`, `__asm`, `asm_block`) or explicit scheduling barriers (`sched_barrier`, `ttgl.sched_barrier`). If multiple implementations of the same op exist, use the best-performing variant that satisfies this constraint.

## Setup

Read `forks.txt` in your working directory to find the path and branch of the **Triton** source repository. Check out the specified branch before proceeding.

## Phase 1: Discover all MI450/gfx1250 kernels

Search the Triton repository systematically for all kernel implementations that can target gfx1250. Search in this order:

### 1.1 Gluon kernels (CDNA5 / gfx1250-optimized)

Look in `third_party/amd/python/examples/gluon/` for files matching `*_cdna5.py`. As of the time of writing, these include:

- `mxfp_fa_cdna5.py` — MXFP Flash Attention (FP8/MXFP4, block-scaled and global-scaled)
- `f16_fa_cdna5.py` — FP16/BF16 Flash Attention
- `mxfp_gemm_cdna5.py` — MXFP GEMM with TDM pipeline (FP8/FP4)
- `f16_gemm_cdna5.py` — FP16 GEMM (single/multi-CTA, warp-specialized, persistent)
- `f16_gemm_streamk_cdna5.py` — FP16 Stream-K GEMM
- `gemm_warp_pipeline_cdna5.py` — Warp-pipeline GEMM (multi-dtype)
- `moe_cdna5.py` — Mixture of Experts (gating, dispatch, combine, end-to-end)

**Do not assume this list is complete.** Search for any new files that may have been added since this prompt was written. Also check subdirectories.

For each Gluon kernel:
1. Read the source file to find all `@triton.jit`-decorated kernel functions.
2. Read the `argparse` section (in `if __name__ == "__main__":`) to discover all CLI arguments, their types, choices, and defaults.
3. Read the kernel launch logic to understand how arguments map to kernel parameters, especially `num_warps`, `num_ctas`, `waves_per_eu`, block sizes, and pipeline parameters.
4. Identify the entry-point function that creates tensors, launches the kernel, and returns the compiled kernel object (e.g. `run_attention`, `run_gemm`).
5. Check for inline assembly or sched_barrier usage — grep for `tl.inline_asm`, `__asm`, `asm_block`, `sched_barrier`, `ttgl.sched_barrier`. Record whether the kernel uses these.

### 1.2 Vanilla Triton tutorials

Look in `python/tutorials/` for tutorial kernels. These are standard Triton kernels (not Gluon) that demonstrate common ML ops:

- `01-vector-add.py` — vector addition (elementwise)
- `02-fused-softmax.py` — fused softmax
- `03-matrix-multiplication.py` — GEMM with autotuning
- `04-low-memory-dropout.py` — dropout
- `05-layer-norm.py` — layer normalization (forward + backward)
- `06-fused-attention.py` — flash attention
- `07-extern-functions.py` — extern function calls
- `08-grouped-gemm.py` — grouped GEMM
- `09-persistent-matmul.py` — persistent matrix multiplication
- `10-block-scaled-matmul.py` — block-scaled MXFP matmul
- `11-programmatic-dependent-launch.py` — programmatic dependent launch

**Do not assume this list is complete.** Check for additional tutorials.

For each tutorial:
1. Read the source to find the `@triton.jit` kernel function(s).
2. Determine how to invoke the kernel with specific shapes — look for `__main__` blocks, benchmark harnesses, or test functions.
3. If the tutorial uses `@triton.autotune`, record the autotuning configs and identify which config would be selected for the sizes you generate (or just pass the config explicitly to bypass autotuning).
4. Check if the tutorial has AMD/HIP-specific autotuning configs (e.g. `get_hip_autotune_config()`).

### 1.3 `triton_kernels` production package

Look in `python/triton_kernels/triton_kernels/` for production-grade portable kernels:

- `matmul.py` / `matmul_details/` — general and persistent matmul with MXFP support
- `reduce.py` — general reduction
- `swiglu.py` / `swiglu_details/` — SwiGLU activation
- `topk.py` / `topk_details/` — TopK forward + backward
- `compaction.py` — masked compaction
- Any other kernel modules

Also check `python/triton_kernels/bench/` for benchmark scripts that show how to invoke these kernels with specific sizes.

For each kernel:
1. Find the `@triton.jit` function and its shape parameters.
2. Find the benchmark or test harness that shows how to invoke it.
3. Check `matmul_details/opt_flags_details/` for AMD-specific optimization flags and tuning configs.

### 1.4 Additional AMD-specific kernels

Check `third_party/amd/python/examples/` for non-Gluon examples that target gfx1250. Also check `third_party/amd/python/test/` for test kernels that exercise specific MI450 features.

## Phase 2: Generate configurations

For each kernel discovered in Phase 1, generate multiple configurations varying problem size and kernel parameters.

### 2.1 Waves-per-EU targeting

**Critical requirement:** wherever the kernel accepts a `num_warps` parameter or `waves_per_eu` hint, configure the kernel to target **1 wave per EU**. The specific mechanism depends on the kernel:

- For Gluon kernels: most already set `waves_per_eu=1` internally. If the kernel has a `--num-warps` CLI argument, use `num_warps=4` as the default (this gives 4 warps per workgroup; combined with `waves_per_eu=1`, only one workgroup's worth of waves runs per EU). If the kernel's own logic selects `num_warps` internally based on the problem shape, record whatever it selects.
- For vanilla Triton kernels: pass `num_warps=4` and `waves_per_eu=1` in the kernel launch kwargs. If the kernel uses `@triton.autotune`, select the config with `num_warps=4` (or the closest available) and bypass autotuning.
- If a kernel does not support `waves_per_eu`, still set `num_warps=4` to keep workgroup size consistent.

Record the actual `num_warps` and `waves_per_eu` values used in every manifest entry.

### 2.2 Size sweep

For each kernel, generate configs at multiple representative sizes. Use sizes that reflect real ML workloads (LLM inference/training dimensions).

**GEMM kernels** (matmul, persistent matmul, stream-k, block-scaled matmul):
- Small: M=1, N=4096, K=4096 (decode-like)
- Medium: M=4096, N=4096, K=4096
- Large: M=8192, N=8192, K=4096
- XL: M=16384, N=8192, K=8192
- Also vary dtypes where supported: fp16/bf16, fp8 (e4m3, e5m2), mxfp4 (e2m1)

**Flash Attention kernels** (fused attention, MXFP FA):
- Short context: batch=1, seqlen_q=1024, seqlen_k=1024, heads=32, head_dim=128
- Medium context: batch=1, seqlen_q=4096, seqlen_k=4096, heads=32, head_dim=128
- Long context: batch=1, seqlen_q=8192, seqlen_k=8192, heads=64, head_dim=128
- Decode: batch=1, seqlen_q=1, seqlen_k=8192, heads=64, head_dim=128
- Vary head grouping: MHA (q_heads==k_heads), GQA (q_heads=64, k_heads=2), MQA (q_heads=64, k_heads=1)

**Softmax / LayerNorm / elementwise**:
- Small: N=1024
- Medium: N=4096
- Large: N=8192
- XL: N=32768

**MoE kernels**:
- Small: batch_per_expt=256, dim1=4096, dim2=4096, n_expts_tot=8, n_expts_act=2
- Large: batch_per_expt=2048, dim1=7168, dim2=2048, n_expts_tot=256, n_expts_act=8

**TopK / Reduce**:
- Vary input size and K value

### 2.3 Variant sweep

For kernels that support multiple algorithmic variants, generate a config for each:

**MXFP GEMM**: sweep `schedule` ∈ {sliceMNK, sliceNK, sliceK, baseline} and `pingpong` ∈ {true, false}
**F16 GEMM**: sweep `warp-specialized`, `persistent`, `subtiled` flags
**F16 FA**: sweep `attention-type` ∈ {default, pipeline, pingpong, decode}
**MXFP FA**: sweep `pipelined` and `pingpong` flags, `scale_type` ∈ {block, global}
**MoE**: sweep `action` ∈ {gating, dispatch, combine, e2e} and `schedule` variants
**Warp-pipeline GEMM**: sweep `dtype` ∈ {f16, bf16, mxfp8, mxfp4}

### 2.4 SwiGLU fusion

For GEMM kernels that support `--activation swiglu` (or equivalent), generate an additional config with the fused SwiGLU activation enabled.

## Phase 3: Identify best configurations

For the `best_kernel_manifest.json`, identify the single best configuration for each op type. Sources for "best" configs:

1. **Tutorial defaults**: the tutorials often encode a known-good config for their target hardware. Look for AMD/HIP-specific config selection paths (e.g. `get_hip_autotune_config()`, `is_hip()` branches).
2. **Gluon example defaults**: the Gluon examples encode optimal block sizes, warp counts, and pipeline settings internally based on the problem shape. The kernel's own `run_*` function selects these automatically.
3. **Benchmark scripts**: check `python/triton_kernels/bench/` for configs that the benchmarks use as their "fast path."

**Exclusion rule**: if the best version of a kernel uses inline assembly (`tl.inline_asm`, `__asm`, `asm_block`) or explicit scheduling barriers (`sched_barrier`, `ttgl.sched_barrier`), **exclude it** from `best_kernel_manifest.json` and use the next-best variant that does not use these. If no variant is free of inline asm / sched_barriers, include the kernel anyway but set `"uses_inline_asm": true` and/or `"uses_sched_barrier": true` in the entry so the downstream consumer knows.

For each best config entry, record why it was selected in a `"selection_reason"` field (e.g. "default config from tutorial HIP path", "Gluon example auto-selects this for M>=4096").

## Phase 4: Write the manifests

Write three files to your working directory:

- `vanilla_kernel_manifest.json`
- `gluon_kernel_manifest.json`
- `best_kernel_manifest.json`

### Manifest schema

Each manifest follows the same JSON schema. Use the structure from `OAI_kernel_manifest.json` as the reference:

```json
{
  "metadata": {
    "generated_at": "ISO-8601 timestamp",
    "triton_repo": "repo URL from forks.txt",
    "triton_branch": "branch from forks.txt",
    "triton_commit": "full SHA of HEAD at time of generation",
    "target_arch": "gfx1250"
  },
  "models": {
    "<category_name>": {
      "aiter_config_files": [],
      "kernels": [
        {
          "op_type": "gemm | flash_attention | softmax | layernorm | dropout | moe | topk | reduce | swiglu | vector_add | grouped_gemm | ...",
          "label": "Human-readable description, e.g. 'FP16 GEMM M4096 N4096 K4096 sliceMNK'",
          "shapes": {
            "M": 4096,
            "N": 4096,
            "K": 4096,
            "dtype": "fp16"
          },
          "aiter_kernel_name": null,
          "gluon": {
            "source": "relative/path/to/kernel.py",
            "entry_point": "function_name",
            "target": "hip:gfx1250:64",
            "run_command": "python3 relative/path/to/kernel.py --arg1 val1 --arg2 val2",
            "compile_call": "function_name(arg1=val1, arg2=val2, ...)"
          },
          "triton": {
            "source": "relative/path/to/kernel.py",
            "entry_point": "function_name",
            "run_command": "python3 relative/path/to/kernel.py --arg1 val1 --arg2 val2",
            "compile_call": "function_name(arg1=val1, arg2=val2, ...)"
          }
        }
      ]
    }
  }
}
```

### Per-manifest rules

**`vanilla_kernel_manifest.json`:**
- Contains only standard Triton kernels (not Gluon). Set `"gluon": null` for all entries.
- Organize under `"models"` by source location: `"tutorials"` for tutorial kernels, `"triton_kernels"` for the production package.
- Every `run_command` must be a complete, copy-pasteable command runnable from the Triton repo root. If a kernel doesn't have a CLI entry point, write the minimal Python invocation as: `python3 -c "import ...; ..."`.
- Include `num_warps` and `waves_per_eu` in the `shapes` dict.

**`gluon_kernel_manifest.json`:**
- Contains only Gluon kernels. Set `"triton": null` for all entries.
- Organize under `"models"` by kernel family: `"mxfp_fa"`, `"f16_fa"`, `"mxfp_gemm"`, `"f16_gemm"`, `"f16_gemm_streamk"`, `"gemm_warp_pipeline"`, `"moe"`, etc.
- Every `run_command` must be a complete, copy-pasteable command runnable from the Triton repo root.
- Include `num_warps` and `waves_per_eu` in the `shapes` dict.

**`best_kernel_manifest.json`:**
- Contains the single best config per op type and size category (small/medium/large).
- May contain both Gluon and vanilla Triton entries — populate whichever field (`"gluon"` or `"triton"`) applies, set the other to `null`.
- Organize under `"models"` by op type: `"gemm"`, `"flash_attention"`, `"softmax"`, `"layernorm"`, `"moe"`, etc.
- Add these extra fields to each kernel entry:
  - `"selection_reason"`: string explaining why this config was chosen as best
  - `"uses_inline_asm"`: boolean (default false)
  - `"uses_sched_barrier"`: boolean (default false)
- Every `run_command` must be a complete, copy-pasteable command.

### General rules for all manifests

- **Limit to 15 invocations per unique kernel script per manifest.** Each manifest should include at most 15 entries that invoke the same kernel source file. Select a diverse spread of shapes and configurations (small/medium/large sizes, different dtypes, different algorithmic variants) to maximize IR coverage. It is okay if the same kernel appears in multiple manifests (e.g. a Gluon kernel in `gluon_kernel_manifest.json` and also in `best_kernel_manifest.json`).
- Every `run_command` must be a complete command that can be copy-pasted and run from the Triton repo root directory. No placeholders. Use `python3` not `python`.
- The `compile_call` field contains the Python function call (with all arguments) to invoke the kernel programmatically.
- Source paths are relative to the Triton repo root.
- `"target"` for Gluon kernels is `"hip:gfx1250:64"`.
- `"aiter_kernel_name"` is `null` for all entries (these are not aiter-sourced).
- `"aiter_config_files"` is `[]` for all entries.
- The `shapes` dict should contain all shape-defining parameters for that kernel invocation (problem sizes, dtypes, block sizes, num_warps, waves_per_eu, and any variant flags like schedule or attention_type).
- Deduplicate: do not generate identical kernel configs across different size categories if the parameters are the same.

## Important notes

- **Target architecture is gfx1250 only.** Only include kernels that can target gfx1250.
- **Read the actual source code.** Do not guess at argument names or defaults. Read the argparse definitions, the kernel launch code, and the `run_*` functions to get exact parameter names and valid values.
- **Some kernels auto-select configs.** Several Gluon kernels (especially flash attention) auto-select `num_warps`, `block_m`, `block_n`, and pipeline settings based on the problem shape. Record whatever the kernel selects for each shape — do not override its internal logic unless you are explicitly setting `num_warps` for the 1-wave-per-EU requirement.
- **The inline asm / sched_barrier check is critical for `best_kernel_manifest.json`.** Grep every kernel source file for these patterns. If a kernel conditionally uses inline asm (e.g. only in certain code paths), note this in the entry.
- **Completeness matters.** The goal is to capture every kernel the Triton repo offers for gfx1250, not just the obvious ones. Check for kernels in test files, benchmark scripts, and utility modules that might not be in the examples directories.
