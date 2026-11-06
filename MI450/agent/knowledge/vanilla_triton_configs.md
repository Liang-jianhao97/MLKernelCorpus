# Vanilla Triton Kernel Configuration Reference (gfx1250 / CDNA5)

Validated against Triton 3.9.0 at commit `fd91c147ef635b9e86b87e9cf073448cf87d769d`.

## triton_kernels (production package)

### matmul

**Correct invocation:**
```python
from triton_kernels.matmul import matmul
matmul(a, b, bias=None)
```

The `bias` parameter is positional and required (no default). Pass `bias=None` for standard matmul without bias. The manifest commands omitted this argument.

### swiglu

**Correct invocation:**
```python
from triton_kernels.swiglu import swiglu, PrecisionConfig
pc = PrecisionConfig(limit=float("inf"))
swiglu(a, alpha=1.0, precision_config=pc)
```

`precision_config` is a required positional argument. `PrecisionConfig` takes a `limit` (float) and optional `flex_ctx`. Use `limit=float("inf")` for default behavior.

### reduce

**Correct invocation:**
```python
from triton_kernels.reduce import reduce
x = torch.randn(1, M, N, dtype=torch.float16, device="cuda")  # must be 3D
reduce(x, dim=2)
```

Only accepts 3D inputs. Reshape 2D tensors with `x.unsqueeze(0)` before calling. Raises `NotImplementedError` on 2D input.

---

## Tutorials

### 04-low-memory-dropout.py

Requires `tabulate` package: `pip install tabulate`. This is not listed in Triton's main dependencies.

### 10-block-scaled-matmul.py

**Not compilable for gfx1250.** The tutorial's `supports_block_scaling()` check requires either:
- CUDA compute capability 10 or 11 (Blackwell/Rubin), or
- HIP with `gfx950` (CDNA4 / MI300)

gfx1250 (CDNA5 / MI450) is not recognized. The AMD path uses a CDNA4-specific kernel (`block_scaled_matmul_kernel_cdna4`). The equivalent CDNA5 functionality is in `mxfp_gemm_cdna5.py` (Gluon), which already covers block-scaled MXFP GEMM for gfx1250.

### All other tutorials (01-09, 11)

Work correctly with `TRITON_OVERRIDE_ARCH=gfx1250`. The kernel launch fails (no GPU) but `.llir` files are produced from the Triton cache.

---

## Summary of API fixes needed for manifest commands

| Kernel | Original command | Fix |
|--------|-----------------|-----|
| matmul | `matmul(a, b)` | `matmul(a, b, bias=None)` |
| swiglu | `swiglu(a, alpha=1.0)` | `swiglu(a, alpha=1.0, precision_config=PrecisionConfig(limit=float("inf")))` |
| reduce | `reduce(x_2d, dim=1)` | `reduce(x_2d.unsqueeze(0), dim=2)` |
| dropout | (import fails) | `pip install tabulate` before running |
| block-scaled | (guard exits) | Not applicable to gfx1250; use `mxfp_gemm_cdna5.py` instead |
