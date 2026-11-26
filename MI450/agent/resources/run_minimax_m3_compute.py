#!/usr/bin/env python3
"""Validate one pinned ATOM workload and export only its target kernel's LLVM IR.

Run from the corpus root, inside the shared GPU lock. See ../knowledge/atom_configs.md.
The native kernel bodies remain upstream; local fixtures supply inputs and oracles.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "atom_model_manifest.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, result):
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["models"]["minimax-m3"]["kernels"]
    entry = next(k for k in entries if k["case_id"] == args.case)
    # Do not silently change a caller's GPU selection.
    for key, value in entry["environment"].items():
        if key in os.environ and os.environ[key] != value:
            raise RuntimeError(f"Set {key}={value} before running this command")
        os.environ[key] = value

    import torch
    import triton
    from minimax_m3_inputs import build_case

    interface = []
    for source in manifest["metadata"]["native_sources"]:
        module = importlib.import_module(source["module"])
        path = Path(inspect.getfile(module))
        digest = sha256(path)
        if digest != source["sha256"]:
            raise RuntimeError(f"Pinned ATOM source mismatch: {path}")
        interface.append({"module_file": str(path), "source_sha256": digest})
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if not torch.version.hip or not properties.gcnArchName.startswith("gfx1250"):
        raise RuntimeError("This manifest requires a visible gfx1250 HIP GPU")
    torch.manual_seed(0)
    invoke, validate, layouts = build_case(entry)
    module_name = entry["triton"]["source"].removesuffix(".py").replace("/", ".")
    kernel = getattr(
        importlib.import_module(module_name), entry["triton"]["entry_point"]
    )
    native_run = kernel.run
    captured = []

    def capture(*positional, **keywords):
        compiled = native_run(*positional, **keywords)
        captured.append(compiled)
        return compiled

    # Scope capture to the target invocation, after all setup/reference kernels.
    kernel.run = capture
    try:
        output = invoke()
    finally:
        kernel.run = native_run
    torch.cuda.synchronize()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one target launch, captured {len(captured)}")
    correctness = validate(output)
    if correctness["status"] != "pass":
        raise RuntimeError(f"Correctness failed: {correctness}")
    compiled = captured[0]
    llir = compiled.asm["llir"]
    target = compiled.metadata.target
    if target.backend != "hip" or target.arch != "gfx1250":
        raise RuntimeError(f"Unexpected compiled target: {target}")
    destination = args.output_dir / (args.case.replace(".", "_") + ".llir")
    destination.write_text(llir)
    result.update(
        status="success",
        filename=destination.name,
        llir_sha256=sha256(destination),
        target_kernel_launches=len(captured),
        compiled_target={
            "backend": target.backend,
            "architecture": target.arch,
            "warp_size": target.warp_size,
            "num_warps": compiled.metadata.num_warps,
        },
        correctness=correctness,
        input_layouts=layouts,
        native_interface=interface,
        versions={
            "torch": torch.__version__,
            "triton": triton.__version__,
            "hip": torch.version.hip,
            "architecture": properties.gcnArchName,
        },
        scope="Correctness and target-only IR extraction; no performance measurement",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=[
            "minimax_m3.prefill_score",
            "minimax_m3.local_topk",
            "minimax_m3.packed_topk",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "case_id": args.case,
        "status": "error",
        "gpu_completion_confirmed": False,
    }
    try:
        run(args, result)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            result["gpu_completion_confirmed"] = True
        finally:
            path = args.output_dir / (args.case.replace(".", "_") + ".json")
            path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
