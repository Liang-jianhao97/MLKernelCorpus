# IR Corpus Agent: Install Triton for AMDGPU (gfx1250)

## Objective

Clone the Triton repository and build it from source with full AMDGPU support, so that Triton kernels can be compiled and run targeting MI450 / gfx1250. This includes building the standalone AMD code generation library (`libtriton_amd_codegen.so`), which is a separate build step from the main Triton install.

## Setup

Read `forks.txt` in your working directory to find the URL and branch of the **Triton** source repository. Clone it (or use an existing checkout) and check out the specified branch.

Also read `MI450/agent/resources/buildTriton.txt` for a condensed reference of the build commands. Note: that file assumes a custom LLVM path — **do not use a custom LLVM**. The standard build will download the pinned LLVM automatically. The file is useful for understanding the two-step structure (base Triton + AMD codegen) but the exact commands below supersede it.

## Prerequisites

Ensure the following are installed before building:

- **Python 3.11+** (3.12 or 3.13 recommended)
- **CMake >= 3.20**
- **Ninja** build system
- **gcc/g++** or **clang/clang++** (C++20 support required)
- **ROCm** (for GPU runtime — `rocm-dev` or equivalent package, providing `hipcc`, `rocm-smi`, and the HIP runtime)
- **PyTorch with ROCm support** (needed to run kernels that create torch tensors — install the ROCm variant: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.4` or equivalent for your ROCm version)
- **git** (for submodule initialization)

Create and activate a Python virtual environment before proceeding:

```bash
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
```

## Phase 1: Install build dependencies

From the Triton repo root:

```bash
pip install -r python/requirements.txt
```

This installs: `setuptools`, `wheel`, `cmake`, `ninja`, `nanobind==2.10.2`, and `lit`.

For running tests and benchmarks later, also install:

```bash
pip install -r python/test-requirements.txt
```

## Phase 2: Build and install Triton (editable / development mode)

The main Triton install builds the compiler core, including the MLIR-based Triton IR pipeline, `triton-opt`, and the Python bindings. It also automatically downloads a pinned LLVM for the core compiler and builds the AMD backend.

From the Triton repo root:

```bash
pip install -e . --no-build-isolation
```

Or equivalently using the Makefile:

```bash
make dev-install
```

### What happens during this step

1. **LLVM download**: the build system reads `cmake/llvm-info.json` to find the pinned LLVM commit hash and downloads a pre-built LLVM archive from `oaitriton.blob.core.windows.net` to `~/.triton/llvm/`. This is the core LLVM used by the Triton compiler itself.

2. **AMD codegen library**: the build system reads `cmake/amd-llvm-info.json` to find a **separate** pinned LLVM for the AMD code generator. It first checks if a pre-built `amd-codegen-*` archive is available for download. If one is available (i.e. the `sha256sum` entry exists for the current platform in `amd-llvm-info.json`), it downloads and installs it directly. If not, it:
   - Downloads a bootstrap LLVM from the `bootstrap_llvm` section of `amd-llvm-info.json`
   - Builds `libtriton_amd_codegen.so` from source using `third_party/amd/backend/codegen/`
   - Installs the library to `third_party/amd/backend/lib/`

3. **Backend registration**: the AMD backend (`third_party/amd/backend/`) and NVIDIA backend (`third_party/nvidia/backend/`) are both registered as Triton backends. The AMD backend's `compiler.py` loads `libtriton_amd_codegen.so` at runtime from `third_party/amd/backend/lib/`.

4. **Ninja build**: the CMake build produces `triton-opt`, the Python C extension (`triton/_C/`), and all MLIR passes.

### Controlling the build

Key environment variables:

| Variable | Purpose |
|----------|---------|
| `TRITON_BUILD_DIR` | Override the CMake build directory (default: platform-dependent under `python/build/`) |
| `TRITON_AMD_CODEGEN_PATH` | Point to a pre-built `libtriton_amd_codegen.so` — skips the AMD codegen build/download entirely |
| `LLVM_SYSPATH` | Point to an existing LLVM build — skips the LLVM download for the core compiler |
| `MAX_JOBS` | Parallel build jobs (default: `2 * cpu_count`) |
| `DEBUG=1` | Build in Debug mode (default is `TritonRelBuildWithAsserts`) |
| `TRITON_BUILD_WITH_CCACHE=1` | Use ccache for faster rebuilds |

### Incremental rebuilds

After the initial install, incremental C++ rebuilds (e.g. after changing a compiler pass) use ninja directly:

```bash
make          # equivalent to: ninja -C $(BUILD_DIR)
```

Python-only changes (e.g. changes to `python/triton/` or `third_party/amd/python/`) do **not** require a rebuild — the editable install picks them up immediately.

## Phase 3: Build AMD codegen separately (only if needed)

If the AMD codegen library was not built or downloaded during Phase 2 (e.g. because `TRITON_AMD_CODEGEN_PATH` was set to an older version, or you want to rebuild it against a different LLVM), you can build it independently.

### Using the build system's automatic bootstrap

The simplest approach — let the build system download the pinned LLVM and build codegen:

```bash
pip install -e . --no-build-isolation
```

This re-runs the full install and rebuilds codegen if needed.

### Manual build against the auto-downloaded LLVM

If you need more control, find the LLVM the build system downloaded:

```bash
# The build system caches LLVM under ~/.triton/amd/llvm/
ls ~/.triton/amd/llvm/
# Find the directory named llvm-<hash>-<platform>-<build_number>
```

Then build codegen manually:

```bash
LLVM_DIR=$(find ~/.triton/amd/llvm/ -name "LLVMConfig.cmake" -path "*/lib/cmake/llvm/*" | head -1 | xargs dirname)

cmake -G Ninja \
  -S third_party/amd/backend/codegen \
  -B build-amd-codegen \
  -DLLVM_DIR=$LLVM_DIR \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build-amd-codegen
```

Then either:
- Copy the library: `cp build-amd-codegen/libtriton_amd_codegen.so third_party/amd/backend/lib/`
- Or point to it at runtime: `export TRITON_AMD_CODEGEN_PATH=$(pwd)/build-amd-codegen/libtriton_amd_codegen.so`

## Phase 4: Verify the installation

### 4.1 Check Triton imports

```bash
python3 -c "import triton; print(triton.__version__)"
```

### 4.2 Check AMD backend is available

```bash
python3 -c "from triton.backends.amd.compiler import get_amd_codegen_revision; print('AMD codegen revision:', get_amd_codegen_revision())"
```

### 4.3 Check GPU visibility

```bash
rocm-smi            # should show MI450 / gfx1250 GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.4 Run a simple kernel

```bash
python3 python/tutorials/01-vector-add.py
```

### 4.5 Run a Gluon kernel targeting gfx1250

```bash
python3 third_party/amd/python/examples/gluon/mxfp_fa_cdna5.py \
  --q_type e4m3 --kv_type e4m3 \
  --batch 1 --seqlen_q 1024 --seqlen_k 1024 \
  --num_q_heads 32 --num_k_heads 32 --head_sz 128 \
  --pipelined --scale_type block --disable_p_scaling --profile
```

### 4.6 Verify IR dumping works

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_ir_test TRITON_ALWAYS_COMPILE=1 \
  python3 python/tutorials/01-vector-add.py

# Check that .llir and .amdgcn files were generated
find /tmp/triton_ir_test -name "*.llir" -o -name "*.amdgcn" | head -5
```

## Troubleshooting

### `libtriton_amd_codegen.so` not found

The library should be at `third_party/amd/backend/lib/libtriton_amd_codegen.so`. If it's missing:

1. Re-run `pip install -e . --no-build-isolation` — the build system will download or build it.
2. Or build it manually per Phase 3 and copy it to that path.
3. Or set `TRITON_AMD_CODEGEN_PATH` to point to it.

### Build fails downloading LLVM

If behind a firewall or proxy, the LLVM download from `oaitriton.blob.core.windows.net` may fail. Options:
- Set `https_proxy` / `http_proxy` environment variables.
- Download the archive manually and place it in `~/.triton/` at the expected path.
- Build LLVM from source using `make dev-install-llvm` and set `LLVM_SYSPATH`.

### `ninja: error: loading 'build.ninja'`

The build directory may be stale. Remove it and reinstall:

```bash
rm -rf python/build/
pip install -e . --no-build-isolation
```

### HIP runtime errors

Ensure ROCm is properly installed and `ROCM_PATH` or `HIP_PATH` is set if ROCm is installed in a non-standard location. The GPU must be visible via `rocm-smi`.

### PyTorch not finding GPU

Install the ROCm variant of PyTorch, not the CPU or CUDA variant. Check with:

```bash
python3 -c "import torch; print(torch.version.hip)"
```

This should print a ROCm/HIP version string, not `None`.
