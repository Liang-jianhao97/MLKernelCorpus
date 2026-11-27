#!/usr/bin/env python3
"""Export LLVM IR for the MiniMax-M3 kernels in the manifest."""

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "atom_model_manifest.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefill_score(
    idx_q,
    index_cache,
    block_table,
    seq_lens,
    context,
    query,
    cu_seqlens_q,
    prefix_lens,
    score,
):
    import triton

    op = importlib.import_module("atom.model_ops.minimax_m3.index_topk")
    heads, _, blocks = score.shape
    batch = cu_seqlens_q.numel() - 1
    tiles = triton.cdiv(query, op.SCORE_BLOCK_SIZE_Q)
    chunk = op._score_chunk_blocks(blocks, tiles, batch, heads, idx_q.device)
    warps, stages = op._prefill_score_launch_config(index_cache.dtype)
    op._index_block_score_kernel[(tiles, batch * heads, triton.cdiv(blocks, chunk))](
        idx_q,
        index_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        heads,
        idx_q.shape[-1],
        128**-0.5,
        chunk,
        *idx_q.stride(),
        *index_cache.stride(),
        *score.stride(),
        block_table.stride(0),
        BLOCK_SIZE_Q=op.SCORE_BLOCK_SIZE_Q,
        BLOCK_SIZE_K=op.SPARSE_BLOCK_SIZE,
        num_warps=warps,
        num_stages=stages,
    )
    return score


def packed_topk(
    score, topk_idx, cu_seqlens_q, prefix_lens, block_table, emit_args, query
):
    op = importlib.import_module("atom.model_ops.minimax_m3.index_topk")
    op._launch_select(
        score,
        topk_idx,
        cu_seqlens_q,
        prefix_lens,
        cu_seqlens_q.numel() - 1,
        16,
        0,
        0,
        block_table,
        emit_args,
        score.shape[0],
        rows_per_req=query,
        decode_max_q=0,
        block_size_k=op._prefill_topk_block_size_k(score.shape[-1]),
        num_warps=op.PREFILL_TOPK_NUM_WARPS,
        emit=True,
        block_page_stride=op.PAGES_PER_SPARSE_BLOCK,
        n_valid_column_per_row=None,
    )
    return topk_idx, emit_args[0], emit_args[1]


def build_case(entry):
    import torch

    shape = entry["shapes"]
    torch.manual_seed(0)
    if entry["case_id"].endswith("local_topk"):
        scores = (
            (
                torch.arange(shape["local_blocks"], device="cuda", dtype=torch.float32)
                * shape["cp_world"]
            )[None, None]
            .expand(shape["heads"], shape["token_rows"], shape["local_blocks"])
            .contiguous()
        )
        lengths = torch.full(
            (shape["batch"],), shape["context"], dtype=torch.int32, device="cuda"
        )
        op = importlib.import_module(
            "atom.model_ops.minimax_m3.indexer_candidate_exchange"
        )
        return lambda: op.local_candidate_keys(
            scores,
            lengths,
            shape["topk"],
            shape["cp_rank"],
            shape["cp_world"],
            shape["query_per_request"],
            shape["global_blocks"],
            init_blocks=shape["init_blocks"],
            local_blocks=shape["local_keep_blocks"],
        )

    query, context, heads = shape["query"], shape["context"], shape["heads"]
    blocks, dim = shape["kv_blocks"], shape.get("head_dim", 128)
    data = {
        "idx_q": torch.randn(query, heads, dim, dtype=torch.bfloat16, device="cuda")
        * 0.1,
        "index_cache": torch.randn(
            blocks, 128, dim, dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn),
        "block_table": torch.randperm(blocks, device="cuda").to(torch.int32)[None],
        "seq_lens": torch.full((1,), context, dtype=torch.int32, device="cuda"),
        "context": context,
        "query": query,
        "cu_seqlens_q": torch.tensor([0, query], dtype=torch.int32, device="cuda"),
        "prefix_lens": torch.tensor(
            [shape["prefix"]], dtype=torch.int32, device="cuda"
        ),
        "score": torch.empty(
            (heads, query, blocks), dtype=torch.float32, device="cuda"
        ),
    }
    if entry["case_id"].endswith("prefill_score"):
        return lambda: prefill_score(**data)
    score = prefill_score(**data)  # Precompute selector input.
    op = importlib.import_module("atom.model_ops.minimax_m3.index_topk")
    _, emit = op._alloc_emit(
        query, heads, shape["topk"], data["block_table"], True, score.device
    )
    indices = torch.empty(
        (heads, query, shape["topk"]), dtype=torch.int32, device="cuda"
    )
    return lambda: packed_topk(
        score,
        indices,
        data["cu_seqlens_q"],
        data["prefix_lens"],
        data["block_table"],
        emit,
        query,
    )


def run(args, result):
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["models"]["minimax-m3"]["kernels"]
    entry = next(k for k in entries if k["case_id"] == args.case)
    import torch
    import triton

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
    invoke = build_case(entry)
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

    kernel.run = capture
    try:
        _output = invoke()
    finally:
        kernel.run = native_run
    torch.cuda.synchronize()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one target launch, captured {len(captured)}")
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
        native_interface=interface,
        versions={
            "torch": torch.__version__,
            "triton": triton.__version__,
            "hip": torch.version.hip,
            "architecture": properties.gcnArchName,
        },
        scope="Target-only IR extraction; accuracy is covered by the native ATOM UT command in the manifest",
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
