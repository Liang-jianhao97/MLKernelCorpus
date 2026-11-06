# Aiter Kernel Configuration Reference (gfx1250 / CDNA5)

Validated against Triton 3.9.0 at commit `fd91c147ef635b9e86b87e9cf073448cf87d769d`.

Aiter manifests invoke kernels from the Triton repo (Gluon kernels). All constraints from `gluon_configs.md` apply.

## mxfp_gemm_cdna5.py (MXFP GEMM)

15 of 45 aiter entries failed with the SCALE_PRESHUFFLE skip. These are mxfp8 and mxfp4 GEMM entries that used non-baseline schedules without `--scale_preshuffled`.

**Fix:** add `--scale_preshuffled` and ensure `BLOCK_K ≥ 256` for non-baseline schedules. See `gluon_configs.md` for full constraints.

Note: The aiter manifest commands also included `--target hip:gfx1250:64` which is not a valid argparse argument for these scripts. The IR collection script strips this with `fix_run_command()`. Additionally, some commands used `--M` instead of `-M` (the scripts use `-M` as a short option); `fix_run_command()` also corrects this.

## moe_cdna5.py (MoE)

15 of 45 entries failed with GPU OOM. All are large MoE configs with many experts (96–896) that try to allocate >15.8 GB before compilation begins.

**Not fixable** with a 16 GB GPU. These configs require ≥32 GB VRAM just for tensor allocation. The kernel never reaches compilation.

Working MoE configs from gluon manifest: batch_per_expt=256, n_expts_tot=8, bf16/mx4 or fp8/mx4 dtypes.

## f16_gemm_cdna5.py (FP16 GEMM)

1 failure: `kimi-k3_gemm_1` with K=128 hit `K/BLOCK_K < NUM_BUFFERS` skip.

**Fix:** use `--block_k 64` when K < 256, or `--block_k` ≤ K/2 generally.

## Command sanitization

Aiter manifest `run_command` fields need these corrections before execution:

| Pattern | Fix | Reason |
|---------|-----|--------|
| `--target hip:gfx1250:64` | Remove | Not a valid argparse argument for the kernel scripts |
| `--M`, `--N`, `--K` | `-M`, `-N`, `-K` | Scripts use single-dash short options |
| `device=cuda` | `device='cuda'` | Needs quoting in Python expressions |

The `collect_ir.py` script handles these via `fix_run_command()`.
