# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed MiniMax-M3 inputs, native launch adapters, and independent oracles.

Input generators follow the public MiniMax-M3 UTs in AITER PR #8021:
https://github.com/ROCm/aiter/tree/a4f7afd69fdc1b05bee8c89b499b3a922966c1d1/op_tests/triton_tests
Native launch adapters follow ATOM f7cc3a6a47f72444fecb0c3a6b1ea79493bc0706.
Only host adapters and references are here; kernel bodies remain upstream.
"""

import importlib

import torch
import triton


def index_inputs(batch=2, query=4, context=4096, heads=4):
    blocks = (context + 127) // 128
    return {
        "idx_q": torch.randn(
            batch * query, heads, 128, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1,
        "index_cache": torch.randn(
            batch * blocks, 128, 128, dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn),
        "block_table": torch.randperm(batch * blocks, device="cuda")
        .to(torch.int32)
        .reshape(batch, blocks),
        "seq_lens": torch.full((batch,), context, dtype=torch.int32, device="cuda"),
        "context": context,
        "query": query,
    }


def prefill_inputs(query=31, context=4096):
    inputs = index_inputs(1, query, context, 1)
    # Exact FP8 values give a distinct score to each block of this oracle case.
    if context == 4096:
        inputs["idx_q"].fill_(1)
        block = torch.arange(32, 64, dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
        inputs["index_cache"] = (
            block[:, None, None]
            .expand(32, 128, 128)
            .contiguous()
            .to(torch.float8_e4m3fn)
        )
        inputs["block_table"] = torch.arange(32, dtype=torch.int32, device="cuda")[None]
    inputs["cu_seqlens_q"] = torch.tensor([0, query], dtype=torch.int32, device="cuda")
    inputs["prefix_lens"] = torch.tensor(
        [context - query], dtype=torch.int32, device="cuda"
    )
    return inputs


def prefill_score_inputs(query=31, context=4096):
    """Preallocate the output of the pinned ATOM prefill score launch."""
    data = prefill_inputs(query, context)
    data["score"] = torch.empty(
        (1, query, triton.cdiv(context, 128)), dtype=torch.float32, device="cuda"
    )
    return data


def candidate_inputs(tokens=8, context=4096):
    blocks = (context + 127) // 128
    local = (blocks + 3) // 4
    scores = torch.arange(local, dtype=torch.float32, device="cuda") * 4
    return {
        "scores": scores[None, None].expand(4, tokens, local).contiguous(),
        "seq_lens": torch.full(
            (tokens // 4,), context, dtype=torch.int32, device="cuda"
        ),
        "context": context,
    }


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


def tensor_layouts(torch, data):
    result = {}

    def visit(name, value):
        if isinstance(value, torch.Tensor):
            result[name] = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(f"{name}[{i}]", item)
        elif hasattr(value, "__dict__"):
            for key, item in vars(value).items():
                visit(f"{name}.{key}", item)

    for name, value in data.items():
        visit(name, value)
    return result


def _prefill_score_reference(torch, data, full=False):
    query, context = data["query"], data["context"]
    rows = (
        torch.linspace(0, query - 1, min(query, 16), device="cuda")
        .round()
        .long()
        .unique()
    )
    if full:
        rows = torch.arange(query, device="cuda")
    q = data["idx_q"][rows].to(data["index_cache"].dtype).float()
    k = data["index_cache"][data["block_table"][0].long()].float()
    logits = torch.einsum("qhd,btd->hqbt", q, k)
    positions = torch.arange(k.shape[0] * 128, device="cuda").reshape(-1, 128)
    ends = context - query + rows + 1
    logits.masked_fill_(
        positions[None, None] >= ends[None, :, None, None], -float("inf")
    )
    expected = logits.amax(-1) * (128**-0.5) * 1.4426950408889634
    valid = torch.arange(k.shape[0], device="cuda")[None, :] * 128 < ends[:, None]
    return rows, valid, expected


def _selection_reference(torch, scores, query, context, table):
    """Sort by score and then block id; emit the partial causal block last."""
    blocks = scores.shape[-1]
    descending_ids = torch.arange(blocks - 1, -1, -1, device="cuda")
    ends = context - query + torch.arange(query, device="cuda") + 1
    valid = descending_ids[None, :] * 128 < ends[:, None]
    ordered_scores = scores[0, :, descending_ids].masked_fill(~valid, -float("inf"))
    order = torch.argsort(ordered_scores, descending=True, stable=True)[:, :16]
    indices = descending_ids[order]
    tail = indices == ((ends - 1) // 128)[:, None]
    emission_order = torch.argsort(tail.int(), stable=True)
    emitted = indices.gather(1, emission_order)
    pages = table[0, emitted].long()[:, :, None] * 8 + torch.arange(8, device="cuda")
    lengths = 2048 - tail.any(-1).int() * (128 - ((ends - 1) % 128 + 1))
    return indices[None].int(), pages.reshape(query, 128).int(), lengths.int()


def selector_inputs(base, score):
    query = base["query"]
    op = importlib.import_module("atom.model_ops.minimax_m3.index_topk")
    _, emit_args = op._alloc_emit(query, 1, 16, base["block_table"], True, score.device)
    return {
        "score": score,
        "topk_idx": torch.empty((1, query, 16), dtype=torch.int32, device="cuda"),
        "cu_seqlens_q": base["cu_seqlens_q"],
        "prefix_lens": base["prefix_lens"],
        "block_table": base["block_table"],
        "emit_args": emit_args,
        "query": query,
    }


def check_prefill_boundaries():
    """Full FP32 scoring/sorting oracles across a query tile boundary."""
    for query in (31, 257):
        base = prefill_score_inputs(query, 4096)
        score = prefill_score(**base)
        rows, valid, expected = _prefill_score_reference(torch, base, full=True)
        torch.testing.assert_close(
            score[0, rows][valid], expected[0][valid], atol=0.003, rtol=0.02
        )
        actual = packed_topk(**selector_inputs(base, score))
        expected = _selection_reference(torch, score, query, 4096, base["block_table"])
        for output, oracle in zip(actual, expected):
            torch.testing.assert_close(output, oracle, rtol=0, atol=0)


def build_case(entry):
    operation = entry["case_id"].split(".")[-1]
    shape = entry["shapes"]
    if operation != "local_topk":
        check_prefill_boundaries()
    # Match the original measured workload after any reduced correctness checks.
    torch.manual_seed(0)
    if operation == "local_topk":
        data = candidate_inputs(shape["token_rows"], shape["context"])
        local = data["scores"].shape[-1]
        ids = torch.arange(local - 1, local - 17, -1, device="cuda")
        bits = (
            data["scores"][0, 0, ids].contiguous().view(torch.int32).long() ^ 0x80000000
        )
        expected = (1 << 48) | (bits << 16) | (ids * 4 + 1)
        op = importlib.import_module(
            "atom.model_ops.minimax_m3.indexer_candidate_exchange"
        )

        def invoke():
            return op.local_candidate_keys(
                data["scores"],
                data["seq_lens"],
                16,
                0,
                4,
                4,
                (shape["context"] + 127) // 128,
            )

        def check(output):
            torch.testing.assert_close(
                output, expected.expand_as(output), rtol=0, atol=0
            )

        method = "Full packed-key oracle for ascending rank-local shard scores."
        full = True
    else:
        base = prefill_score_inputs(shape["query"], shape["context"])
        if operation == "prefill_score":
            data = base
            rows, valid, expected = _prefill_score_reference(torch, data)

            def invoke():
                return prefill_score(**data)

            def check(output):
                torch.testing.assert_close(
                    output[0, rows][valid], expected[0][valid], atol=0.003, rtol=0.02
                )

            method = (
                "16 query rows against FP32 scoring; full Q31/Q257 boundary checks."
            )
            full = False
        else:
            score = prefill_score(**base)
            data = selector_inputs(base, score)
            expected = _selection_reference(
                torch, score, shape["query"], shape["context"], base["block_table"]
            )

            def invoke():
                return packed_topk(**data)

            def check(output):
                for actual, reference in zip(output, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

            method = "Full stable sort, causal page emission and sparse length oracle."
            full = True

    def validate(output):
        check(output)
        return {"status": "pass", "method": method, "full_output_checked": full}

    return invoke, validate, tensor_layouts(torch, data)
