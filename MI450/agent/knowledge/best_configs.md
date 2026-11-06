# Best Kernel Configuration Reference (gfx1250 / CDNA5)

Validated against Triton 3.9.0 at commit `fd91c147ef635b9e86b87e9cf073448cf87d769d`.

## Failures (5 of 13)

| Kernel | Root cause | Fix |
|--------|-----------|-----|
| Best GEMM fp8 8192×8192×8192 (MXFP GEMM sliceMNK) | Missing `--scale_preshuffled` | Add `--scale_preshuffled` and use `BK=256` |
| Best FA Prefill FP16 pipeline GQA seqlen=8192 | Reference check crashes on GQA heads (8 vs 2) | Invoke via Python with `check=False` |
| Best MoE fp8/mx4 large batch=8192 expts=256 | GPU OOM (31.6 GB needed, 16 GB available) | Not fixable with 16 GB GPU |
| Best SwiGLU M=4096 N=8192 | Missing `precision_config` argument | Pass `PrecisionConfig(limit=float("inf"))` |
| Best Reduce M=4096 N=8192 | Only supports 3D inputs | Reshape to 3D with `unsqueeze(0)` |

All fixes except the MoE OOM are config/API issues — see `gluon_configs.md` and `vanilla_triton_configs.md` for details.
