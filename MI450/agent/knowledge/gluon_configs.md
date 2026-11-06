# Gluon Kernel Configuration Reference (gfx1250 / CDNA5)

Validated against Triton 3.9.0 at commit `fd91c147ef635b9e86b87e9cf073448cf87d769d`.

## mxfp_gemm_cdna5.py

### Working configurations

| Schedule | Required flags | Block sizes | Warps | Notes |
|----------|---------------|-------------|-------|-------|
| baseline | (none) | BM=256, BN=256, BK=128 | 4 | Works for fp8×fp8, fp4×fp4 (with `--with_a_scale`) |
| sliceK | `--scale_preshuffled` | BM≥128, BN≥128, **BK≥256** | 4 | `__main__` hardcodes TRANSPOSE_B=True |
| sliceNK | `--scale_preshuffled` | BM≥128, BN≥256, **BK≥256** | 4 | Also requires M≥256; BLOCK_K≥256 and BLOCK_N≥256 |
| sliceMNK | `--scale_preshuffled` | BM≥128, BN≥128, **BK≥256** | 4 | |
| any (8 warps) | `--scale_preshuffled` | BM≥128, BN≥128, BK≥256 | 8 | Requires `--num_buffers 3` or `4`; no `--async_copy_scale` |

### Non-baseline schedules: mandatory flags

For any schedule other than `baseline`, the test function requires both `SCALE_PRESHUFFLE=True` and `TRANSPOSE_B=True`. The `__main__` block always sets `TRANSPOSE_B=True`, so the only CLI flag needed is `--scale_preshuffled`. Without it, the kernel skips with:

> "Only test with SCALE_PRESHUFFLE and TRANSPOSE_B in sliceK and sliceNK schedules"

### fp4 × fp4: requires `--with_a_scale`

When `dtype_a == float4` and `--with_a_scale` is not set, the kernel skips:

> "Skip fp4 x mxfp gemm to reduce test cases."

Fix: always pass `--with_a_scale` for fp4 inputs.

### Large blocks with ≥3 buffers (fp8×fp8)

Skip condition: `NUM_BUFFERS >= 3 AND (BLOCK_M >= 256 OR BLOCK_N >= 256 OR BLOCK_K >= 256) AND is_fp8_a AND is_fp8_b`. This is to avoid exceeding LDS limits. Use `--num_buffers 2` for large fp8 block sizes, or reduce block dimensions.

### Preshuffled block size minimums

When `--scale_preshuffled` is set:
- BLOCK_M must be ≥ 128
- BLOCK_N (effective, ×2 for swiglu) must be ≥ 128
- For non-baseline schedules: BLOCK_K must be ≥ 256

---

## f16_gemm_cdna5.py

### Mode constraints

| Mode | Num warps | Other requirements |
|------|-----------|-------------------|
| default (no flags) | 4, 8 | num_warps ≠ 16 |
| `--warp-specialized` | **8** or **16** | `--num-buffers 3`+; no `--single-warp-schedule`; no activation; NUM_CTAS=1; no L2 prefetch |
| `--warp-specialized --persistent` | **12** | Same restrictions as warp-specialized |
| `--single-warp-schedule` | **4** | Requires `--prefetch-lds`; no `--persistent`; no `--warp-specialized`; no activation |
| `--persistent` | 4, 8 | Cannot combine with `--single-warp-schedule` |

### K too small for buffer count

Skip condition: `K / BLOCK_K < NUM_BUFFERS`. For small K values (e.g., K=128 with default BLOCK_K=128 and NUM_BUFFERS=2), reduce `--block_k` or use `--num-buffers 1`. Example fix: K=128 → `--block_k 64 --num-buffers 2`.

### Fused activation restrictions

`--activation swiglu` is only supported with:
- No `--warp-specialized`
- No `--single-warp-schedule`
- `NUM_CTAS == 1` (default, no `--ctas-per-cga`)

---

## mxfp_fa_cdna5.py

### Working configurations

All head configurations (MHA, GQA, MQA) work. The `--profile` flag causes `CompiledKernel.module` attribute error at seqlen=1024 but **not** at larger seqlens (4096, 8192). Fix: omit `--profile` for cross-compilation, or use seqlen ≥ 4096 with `--profile`.

### Recommended invocation for cross-compilation

```bash
python3 mxfp_fa_cdna5.py --q_type e4m3 --kv_type e4m3 \
  --batch 1 --seqlen_q 4096 --seqlen_k 4096 \
  --num_q_heads 64 --num_k_heads 8 --head_sz 128 \
  --pipelined --scale_type block --disable_p_scaling
```

Omit `--profile` when cross-compiling (no GPU present). The profiling code accesses `CompiledKernel.module` which doesn't exist when the kernel can't load on the device.

---

## f16_fa_cdna5.py

### GQA/MQA head mismatch

The `run_attention()` function with `check=True` (default) calls `torch.nn.functional.scaled_dot_product_attention` as a reference. PyTorch's SDPA on ROCm with this Triton build does not handle GQA (num_q_heads ≠ num_k_heads) — it crashes with:

> "The size of tensor a (8) must match the size of tensor b (2) at non-singleton dimension 1"

The kernel itself compiles fine. Fix: invoke via Python with `check=False`:

```python
from f16_fa_cdna5 import run_attention
config = {"BATCH": 1, "SEQLEN_Q": 4096, "SEQLEN_K": 4096,
          "NUM_Q_HEADS": 8, "NUM_K_HEADS": 2, "HEAD_SZ": 128,
          "BLOCK_M": 128, "BLOCK_N": 128, "ATTN_FN": "default"}
run_attention(config, check=False)
```

MHA configs (q_heads == k_heads) work with the default CLI.

---

## moe_cdna5.py

### Dtype restrictions

| x_dtype | w_dtype | Status | Notes |
|---------|---------|--------|-------|
| bf16 | mx4 | Works | All schedules |
| fp8 | mx4 | Works | All schedules |
| fp8 | fp8 | **Fails** | `make_opt_flags()` missing `intermediate_out_dtype` — upstream bug |

### GPU memory limits

MoE kernels allocate tensors before compilation. With 16 GB VRAM:
- **Works:** small configs (batch_per_expt=256, n_expts_tot=8)
- **Fails (OOM):** medium+ configs (batch_per_expt≥2048 or n_expts_tot≥256)

This is inherent — the tensors must exist on GPU before the Triton JIT sees the kernel launch. No config workaround; requires more VRAM.

---

## gemm_warp_pipeline_cdna5.py

No failures observed in tested configurations. Supports fp16, bf16, mxfp8, mxfp4 dtypes.

---

## f16_gemm_streamk_cdna5.py

No failures observed in tested configurations.
