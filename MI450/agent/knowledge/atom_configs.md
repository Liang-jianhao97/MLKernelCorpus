# MiniMax-M3 ATOM compute-side cases

`../atom_model_manifest.json` follows the model/kernel/source/command/performance
layout of the AITER manifest. These three kernels belong to **ATOM**, so their
source ownership and regeneration are recorded separately. All three use Triton
DSL; none uses Gluon. The selection is the compute side of the 18-case roofline
study: FP8 prefill index scoring, local INT64 top-k, and packed INT64 top-k with
sparse page emission.

## Pinned environment

Use a compatible ROCm PyTorch container with a visible gfx1250 GPU and ATOM/AITER
dependencies installed. The validated image is
`rocm/fw-bringup@sha256:98f134cfecb76816750e0af114139a9d867eb077205305c7603829c88139e17f`.
The validation Python has Triton
`3.9.0+amd.rocm7.2.0.gitf1bf9132`; versions for the reference measurements are
also recorded per entry. No dependency installation is performed by the runner.

Prepare source checkouts at the revisions in manifest metadata:

```bash
git clone https://github.com/ROCm/ATOM.git /root/KernelExtract/ATOM
git -C /root/KernelExtract/ATOM checkout f7cc3a6a47f72444fecb0c3a6b1ea79493bc0706
git clone https://github.com/ROCm/aiter.git /root/KernelExtract/aiter
git -C /root/KernelExtract/aiter checkout f4bc2aa3f7f13c9ac983dc2b9f41fd0fcaa71cb1
export PYTHONPATH=/root/KernelExtract/ATOM:/root/KernelExtract/aiter
```

If the image already provides these sources, use them after verifying the hashes.
Set GPU visibility consistently before starting Python; the validation used
`ROCR_VISIBLE_DEVICES=0` with `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`
unset. The runner preserves the caller's device selection.

## Extraction

From the corpus root, run each entry's literal `triton.run_command` under the
node's shared GPU lock, for example:

```bash
/var/lib/gpu-lock/gpu-lock python3 MI450/agent/resources/run_minimax_m3_compute.py \
  --case minimax_m3.prefill_score \
  --output-dir /tmp/minimax-m3-corpus
```

Repeat for `minimax_m3.local_topk` and `minimax_m3.packed_topk`. A supported GPU
is required: this is correctness-checked extraction, not compile-only emulation.
The local `resources/minimax_m3_inputs.py` supplies host launch adapters, original
input shapes, and independent correctness oracles, derived from the public AITER
MiniMax-M3 tests with adapters for the pinned ATOM revision. No private benchmark
repository is required. The runner checks imported native source hashes, builds
inputs, captures exactly one target launch, validates its output,
then exports that compiled kernel's `asm['llir']` and a JSON evidence record.
Preparation kernels and reference kernels are excluded from the exported IR.
For packed top-k, the score tensor is computed before target capture.

For corpus regeneration, treat this manifest as an additional source group:
`atom_model_manifest.json` -> `MI450/kernels/atom/`. Use the runner's named
`.llir` and JSON evidence instead of collecting every file from the Triton cache.
Check both process exit code and `gpu_completion_confirmed`; do not release a
shared GPU lock with unconfirmed work. Add the three results to the generation
mapping/report when regenerating; this contribution contains source manifests
and the extraction command, like the referenced A4W4 contribution, not generated IR.

## Roofline and measured metrics

Classification uses `AI = useful work / estimated logical IO bytes`, with ridge
point `compute reference / HBM reference`. `AI > ridge` selects the compute side.
This model does not establish that the measured kernel saturates compute units.
Bytes are a logical read/write proxy, not hardware HBM-counter measurements.

Prefill scoring has AI 3203.92 FLOP/byte versus an FP8 ridge of 465.65 FLOP/byte
(10710 TFLOP/s / 23 TB/s). Its measured useful rate is 1091.76 TFLOP/s.
Local and packed top-k have AI 2.462 and 2.953 comparisons/byte versus a ridge of
0.691 comparisons/byte. Their reference is **15.8821 Tcmp/s measured sustained
INT64 compare/select**, not an FP matrix peak or a vendor theoretical peak.
The comparison network count uses `C(n,16) = 8*n - 48` for power-of-two tiles;
packing, address arithmetic and emission are excluded from that work count.

The recorded kernel durations are 102.814, 4.753, and 27.535 us, respectively.
Each is the median of three independent process means, with 10,000 target
dispatches per process, using ROCprofiler kernel timestamps and AITER rotating
buffers. They exclude setup, input scoring for packed top-k, correctness and
warmup. The pool sizes, all process means, source hashes and rotating utility
revision are recorded in the manifest. These values are references, not regression
thresholds. The extraction command deliberately reports no replacement timing.

## Extraction validation (2026-11-26)

All three literal manifest commands passed on gfx1250. Each captured one native
kernel launch and produced one target LLVM IR. The compiler metadata confirms
HIP/gfx1250 with wave32: prefill score uses 4 warps, local top-k 8, and packed
top-k 1. These native policies are preserved, not forced to one wave.

Top-k outputs are checked in full. Prefill scoring checks 16 query rows of the
large shape against FP32, plus full Q=31/Q=257 boundary cases. Each entry records
its correctness scope, compiled target and IR SHA256 under `extraction_validation`.
The fingerprint documents this validation compiler; future compiler revisions
are expected to change the IR. All three processes synchronized GPU completion.
