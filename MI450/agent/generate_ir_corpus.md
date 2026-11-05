# IR Corpus Agent: Generate LLVM IR Corpus for MI450

## Objective

Orchestrate the full pipeline to generate a library of LLVM IR files for AMDGPU MI450 (gfx1250) kernels. This involves:

1. Setting up a Docker environment with ROCm + PyTorch
2. Cloning and installing Triton from source
3. Generating kernel manifests (via subagents)
4. Cross-compiling every kernel for gfx1250 and collecting the LLVM IR
5. Organizing the IR files into `MI450/kernels/` with meaningful names and mappings

The base working directory for this agent is the **MI450KernelCorpus** repository root.

## Phase 0: Docker Environment

**This agent requires the user to provide a ROCm PyTorch-compatible Docker container.**

Before doing any work, check whether the user has provided a Docker image or container. If they have not, **stop immediately** and inform the user:

> "I need a ROCm PyTorch-compatible Docker container to proceed. Please provide a Docker image name (e.g. `rocm/pytorch:rocm6.4_ubuntu22.04_py3.12_pytorch_release_2.7.0`) or a running container ID. The container must have:
> - ROCm 6.x+ runtime
> - PyTorch with ROCm/HIP support
> - Python 3.11+
> - CMake >= 3.20, Ninja, and a C++20 compiler (gcc 11+ or clang 14+)
>
> I will cross-compile kernels for gfx1250 inside this container. A gfx1250 GPU is not required — only the compilation toolchain."

Do not attempt to proceed without a Docker environment.

### Launching the Docker

Once the user provides a Docker image/container:

1. Launch the Docker container (or exec into the running one), mounting the MI450KernelCorpus repo:

   ```bash
   docker run -it \
     -v /path/to/MI450KernelCorpus:/root/MI450KernelCorpus \
     <docker_image> \
     /bin/bash
   ```

   Adjust the mount path based on where MI450KernelCorpus lives on the host.

2. Inside the container, navigate to or create the working directory:

   ```bash
   mkdir -p /root/KernelExtract
   cd /root/KernelExtract
   ```

## Phase 1: Clone / Update Triton

Read `MI450/agent/forks.txt` (relative to the MI450KernelCorpus repo root) to find the Triton repository URL and branch.

- If `/root/KernelExtract/triton` already exists, update it:
  ```bash
  cd /root/KernelExtract/triton
  git fetch origin
  git checkout <branch>
  git pull origin <branch>
  git submodule update --init
  ```

- If it does not exist, clone it:
  ```bash
  cd /root/KernelExtract
  git clone --branch <branch> <repo_url> triton
  cd triton
  git submodule update --init
  ```

Record the Triton commit hash for later use in manifests:
```bash
cd /root/KernelExtract/triton
git rev-parse HEAD
```

## Phase 2: Verify Docker Requirements

Before launching any subagents, verify the Docker container meets Triton's build requirements. Check each of the following:

| Requirement | Check Command | Minimum |
|-------------|--------------|---------|
| Python version | `python3 --version` | 3.11+ |
| CMake version | `cmake --version` | 3.20+ |
| Ninja | `ninja --version` | any |
| C++ compiler | `g++ --version` or `clang++ --version` | C++20 support (gcc 11+ or clang 14+) |
| PyTorch ROCm | `python3 -c "import torch; print(torch.version.hip)"` | Must print a version, not `None` |
| pip | `pip --version` | any |

If **any** requirement is not met, **stop immediately** and report to the user:

> "The Docker container does not meet the following requirements for building Triton:
> - [list each unmet requirement with what was found vs. what is needed]
>
> Please provide a container that satisfies all requirements."

Do not attempt to work around missing requirements by installing them — the user chose the Docker image and must provide one that works.

## Phase 3: Launch Subagents (Manifest Generation + Triton Install)

Launch **three subagents in parallel**. Each subagent operates on different tasks and can run concurrently.

### Subagent 1: Aiter Kernel Manifest

Launch a subagent using the prompt from `MI450/agent/extract_aiter_triton_kernels.md`.

When launching this subagent, provide these additional instructions:

- The Triton repository is already cloned at `/root/KernelExtract/triton` on branch `<branch>` — use this checkout instead of cloning a new one. Set the triton path in your working context accordingly.
- The aiter repository must be cloned separately. Read `MI450/agent/forks.txt` for the aiter repo URL and branch. Clone it to `/root/KernelExtract/aiter`.
- Write the output manifest to `/root/MI450KernelCorpus/MI450/agent/aiter_model_manifest.json`. Overwrite if it already exists.
- Your working directory for reading `forks.txt` is `/root/MI450KernelCorpus/MI450/agent/`.

### Subagent 2: Triton Kernel Manifests

Launch a subagent using the prompt from `MI450/agent/extract_triton_kernels.md`.

When launching this subagent, provide these additional instructions:

- The Triton repository is already cloned at `/root/KernelExtract/triton` on branch `<branch>` — use this checkout instead of cloning a new one.
- Write the three output manifests to `/root/MI450KernelCorpus/MI450/agent/`:
  - `vanilla_kernel_manifest.json`
  - `gluon_kernel_manifest.json`
  - `best_kernel_manifest.json`
- Overwrite any existing manifests.
- Your working directory for reading `forks.txt` is `/root/MI450KernelCorpus/MI450/agent/`.

### Subagent 3: Install Triton

Launch a subagent using the prompt from `MI450/agent/install_triton_amdgpu.md`.

When launching this subagent, provide these additional instructions:

- The Triton repository is already cloned at `/root/KernelExtract/triton` — use this checkout. Do not clone again.
- The branch is already checked out. Proceed directly to installing build dependencies and building Triton.
- Install Triton in editable mode (`pip install -e . --no-build-isolation`) from `/root/KernelExtract/triton`.
- After installation, verify that `import triton` works and that the AMD codegen library is loaded (`get_amd_codegen_revision()` returns a value).
- Skip the GPU visibility check — there may not be a gfx1250 GPU in this Docker. We are cross-compiling.
- Skip the "run a Gluon kernel" verification — kernels will not run without the target GPU.
- Do verify that IR dumping works by checking that `TRITON_KERNEL_DUMP=1` produces `.llir` files from the vector-add tutorial (this tutorial may work even without a GPU, or may fail at launch but still produce cached IR — either is acceptable as long as .llir files appear).

### Wait for all subagents

Wait for all three subagents to complete before proceeding to Phase 4. If any subagent fails, report the failure to the user and stop.

After all subagents complete, verify these files exist:

- `MI450/agent/aiter_model_manifest.json`
- `MI450/agent/vanilla_kernel_manifest.json`
- `MI450/agent/gluon_kernel_manifest.json`
- `MI450/agent/best_kernel_manifest.json`
- `MI450/agent/OAI_kernel_manifest.json` (this one was pre-created — do not overwrite it)

Also verify that Triton is importable:
```bash
cd /root/KernelExtract/triton
python3 -c "import triton; print(triton.__version__)"
```

## Phase 4: Generate LLVM IR

This is the core work. For each manifest, compile every kernel and collect the LLVM IR.

### Setup

```bash
cd /root/KernelExtract/triton
```

Set the environment for cross-compilation to gfx1250:

```bash
export TRITON_OVERRIDE_ARCH=gfx1250
export TRITON_ALWAYS_COMPILE=1
```

Create the output directories:

```bash
mkdir -p /root/MI450KernelCorpus/MI450/kernels/{OAI,gluon,triton,best,aiter}
```

### IR Collection Procedure

For **each manifest file**, process **each kernel entry** using this procedure:

1. **Clear the Triton cache** to ensure a clean mapping:
   ```bash
   rm -rf ~/.triton/cache
   ```

2. **Generate a meaningful filename** from the kernel entry. Use the `label` field (or construct one from `op_type` + key shape parameters). Sanitize for filesystem use (replace spaces/slashes with underscores, lowercase). Examples:
   - `oai7_gqa_prefill_e4m3_B1_Q8192_K8192`
   - `mxfp_gemm_sliceMNK_fp8_M8192_N8192_K1024_w4`
   - `softmax_N4096_w4`

3. **Run the kernel command** from the manifest. The command is in the `run_command` field of the `gluon` or `triton` object (use whichever is non-null; prefer `gluon` if both exist). Prepend the cross-compilation env vars:

   ```bash
   cd /root/KernelExtract/triton
   TRITON_OVERRIDE_ARCH=gfx1250 TRITON_ALWAYS_COMPILE=1 \
     <run_command> 2>&1 || true
   ```

   **The command will likely fail at kernel launch** because there is no gfx1250 GPU. This is expected. The Triton JIT compiler still compiles the kernel and produces the LLVM IR in the cache before attempting to launch. The `|| true` ensures the script continues.

   If the command fails before compilation (e.g. import error, argument error), note it in the mapping file and move on.

4. **Find the generated .llir file** in the Triton cache:
   ```bash
   find ~/.triton/cache -name "*.llir" -type f
   ```

   There should be exactly one `.llir` file (since the cache was cleared before this run). If there are multiple (some kernels compile helper kernels), collect all of them — name the primary one with the kernel name and suffix additional ones with `_helper1`, `_helper2`, etc.

   If no `.llir` file is found, the compilation failed before producing LLVM IR. Note this in the mapping file as a failure and move on.

5. **Copy the .llir file** to the appropriate output directory:
   ```bash
   cp <path_to_llir> /root/MI450KernelCorpus/MI450/kernels/<subdirectory>/<meaningful_name>.llir
   ```

   Also copy the `.amdgcn` file if present (same name, `.amdgcn` extension) — it is useful for occupancy analysis.

6. **Also try TRITON_KERNEL_DUMP** as a fallback if the cache approach fails for a particular kernel:
   ```bash
   rm -rf /tmp/triton_ir_dump
   TRITON_OVERRIDE_ARCH=gfx1250 TRITON_ALWAYS_COMPILE=1 \
     TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_ir_dump \
     <run_command> 2>&1 || true
   find /tmp/triton_ir_dump -name "*.llir" -type f
   ```

### Manifest-to-Directory Mapping

| Manifest File | Output Directory |
|---------------|-----------------|
| `OAI_kernel_manifest.json` | `MI450/kernels/OAI/` |
| `gluon_kernel_manifest.json` | `MI450/kernels/gluon/` |
| `vanilla_kernel_manifest.json` | `MI450/kernels/triton/` |
| `best_kernel_manifest.json` | `MI450/kernels/best/` |
| `aiter_model_manifest.json` | `MI450/kernels/aiter/` |

### Processing Order

Process manifests in this order (OAI kernels are highest priority):

1. `OAI_kernel_manifest.json` → `MI450/kernels/OAI/`
2. `gluon_kernel_manifest.json` → `MI450/kernels/gluon/`
3. `best_kernel_manifest.json` → `MI450/kernels/best/`
4. `vanilla_kernel_manifest.json` → `MI450/kernels/triton/`
5. `aiter_model_manifest.json` → `MI450/kernels/aiter/`

## Phase 5: Create Mapping Files

In each output subdirectory under `MI450/kernels/`, create a `mapping.json` file that documents which kernel + command produced which `.llir` file. Schema:

```json
{
  "source_manifest": "OAI_kernel_manifest.json",
  "target_arch": "gfx1250",
  "triton_commit": "<commit_hash>",
  "generated_at": "ISO-8601 timestamp",
  "kernels": [
    {
      "filename": "oai7_gqa_prefill_e4m3_B1_Q8192_K8192.llir",
      "label": "OAI #7 GQA Prefill e4m3/e4m3 B1 Q8192 K8192",
      "op_type": "flash_attention",
      "run_command": "python3 third_party/amd/python/examples/gluon/mxfp_fa_cdna5.py --q_type e4m3 ...",
      "source": "third_party/amd/python/examples/gluon/mxfp_fa_cdna5.py",
      "shapes": { "batch": 1, "seqlen_q": 8192, "..." : "..." },
      "status": "success",
      "has_amdgcn": true,
      "additional_files": []
    },
    {
      "filename": null,
      "label": "Some kernel that failed",
      "op_type": "gemm",
      "run_command": "python3 ...",
      "source": "...",
      "shapes": {},
      "status": "compilation_failed",
      "error": "brief error description",
      "has_amdgcn": false,
      "additional_files": []
    }
  ],
  "summary": {
    "total": 10,
    "success": 8,
    "failed": 2
  }
}
```

### Rules for mapping.json

- **Every kernel entry** from the manifest must appear in the mapping, even if it failed.
- `"status"` is one of: `"success"`, `"compilation_failed"`, `"launch_failed_ir_captured"`, `"import_error"`, `"skipped"`.
- `"launch_failed_ir_captured"` means the kernel launch failed (no GPU) but the `.llir` was captured from the cache — this is the expected outcome for most kernels during cross-compilation.
- `"filename"` is `null` when no `.llir` was produced.
- `"additional_files"` lists any extra `.llir` files from helper kernels.
- `"error"` is included only on failure, with a brief (one-line) description.

## Phase 6: Final Report

After processing all manifests, produce a summary report for the user:

```
IR Corpus Generation Complete
==============================
Target: gfx1250 (MI450)
Triton commit: <hash>

OAI:     X/Y kernels compiled (Z failed)
Gluon:   X/Y kernels compiled (Z failed)
Best:    X/Y kernels compiled (Z failed)
Triton:  X/Y kernels compiled (Z failed)
Aiter:   X/Y kernels compiled (Z failed)

Total .llir files: N
Output: MI450/kernels/

Failed kernels (if any):
- <label>: <brief reason>
- ...
```

If more than 50% of kernels in any manifest failed, flag this prominently — it likely indicates an installation or environment problem rather than individual kernel issues.

## Important Notes

- **Cross-compilation**: `TRITON_OVERRIDE_ARCH=gfx1250` tells the Triton compiler to generate code for gfx1250 regardless of what GPU (if any) is visible. The kernel will compile but not run. This is by design — we only want the LLVM IR.
- **Cache clearing is critical**: always `rm -rf ~/.triton/cache` before each kernel command to ensure the `.llir` file you find corresponds to the kernel you just compiled.
- **Kernel launch failures are expected**: the run commands try to launch the kernel on a GPU that isn't present. The compilation happens before the launch, so the IR is produced. Capture stderr to diagnose whether the failure was at launch time (expected) or at compilation time (unexpected).
- **Do not modify the manifests**: they are the source of truth. If a manifest entry has an invalid command, record it as failed in the mapping — do not fix the manifest.
- **`OAI_kernel_manifest.json` is pre-existing**: it was created before this agent ran and must not be overwritten. The other four manifests are generated by subagents in Phase 3.
- **aiter kernels may require aiter to be installed**: if the aiter manifest references kernels in the aiter repo, you may need to `pip install -e .` from the aiter checkout at `/root/KernelExtract/aiter`. Check the `source` field — if it starts with an aiter path, install aiter first.
