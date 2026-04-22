import ast
from pathlib import Path

import pytest


paddle = pytest.importorskip("paddle")
pytest.importorskip("paddleformers")


def test_public_api_imports():
    import rrattn

    assert callable(rrattn.xattn_estimate)
    assert callable(rrattn.xattn_prefill)
    assert callable(rrattn.rrattn_estimate)
    assert callable(rrattn.rrattn_estimate_legacy)
    assert callable(rrattn.rrattn_prefill)
    assert callable(rrattn.flex_prefill)
    assert callable(rrattn.patch_llama_attention)
    assert callable(rrattn.patch_qwen_attention)
    assert callable(rrattn.patch_ernie_attention)


def test_rrattn_version_default():
    source = Path(__file__).resolve().parents[1] / "rrattn" / "llama_patch.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_llama_attention":
            assert isinstance(node.args.defaults[-1], ast.Constant)
            assert node.args.defaults[-1].value == "v1"
            return

    raise AssertionError("patch_llama_attention not found")


def test_patch_utils_supports_all_methods():
    from rrattn.patch_utils import SUPPORTED_METHODS

    assert SUPPORTED_METHODS == ("xattn", "rrattn", "flex", "full")


def test_xattn_keep_sink_marks_key_block_axis(monkeypatch):
    import rrattn.xattention as xattention

    def fake_find_blocks_chunked(attn_sum, *args, **kwargs):
        return paddle.zeros(attn_sum.shape, dtype=paddle.bool)

    monkeypatch.setattr(xattention, "find_blocks_chunked", fake_find_blocks_chunked)

    query_states = paddle.arange(2, dtype=paddle.float32).reshape([1, 1, 2, 1])
    key_states = paddle.arange(2, dtype=paddle.float32).reshape([1, 1, 2, 1])

    _, simple_masks = xattention.xattn_estimate(
        query_states,
        key_states,
        block_size=1,
        stride=1,
        threshold=1.0,
        chunk_size=2,
        use_triton=False,
        causal=False,
        keep_sink=True,
    )

    assert simple_masks.shape == [1, 1, 2, 2]
    assert bool(paddle.all(simple_masks[:, :, :, 0]))
    assert not bool(paddle.any(simple_masks[:, :, :, 1]))


def test_xattn_causal_fallback_handles_aligned_lengths(monkeypatch):
    import rrattn.xattention as xattention

    def fake_find_blocks_chunked(attn_sum, *args, **kwargs):
        return paddle.zeros(attn_sum.shape, dtype=paddle.bool)

    monkeypatch.setattr(xattention, "find_blocks_chunked", fake_find_blocks_chunked)

    query_states = paddle.arange(2, dtype=paddle.float32).reshape([1, 1, 2, 1])
    key_states = paddle.arange(2, dtype=paddle.float32).reshape([1, 1, 2, 1])

    attn_sums, simple_masks = xattention.xattn_estimate(
        query_states,
        key_states,
        block_size=1,
        stride=1,
        threshold=1.0,
        chunk_size=2,
        use_triton=False,
        causal=True,
    )

    assert bool(paddle.all(paddle.isfinite(attn_sums)))
    assert simple_masks.shape == [1, 1, 2, 2]


def test_rrattn_estimate_passes_chunk_size_to_triton(monkeypatch):
    import rrattn.rrattention as rrattention

    captured = {}

    def fake_estimate_triton(q, k, startend_row_indices, **kwargs):
        captured["chunk_size"] = kwargs["chunk_size"]
        shape = [1, 1, 1, 1]
        return (
            paddle.zeros(shape, dtype=paddle.float32),
            paddle.zeros(shape, dtype=paddle.bool),
            paddle.zeros(shape, dtype=paddle.bool),
        )

    monkeypatch.setattr(rrattention, "can_use_triton_kernels", lambda: True)
    monkeypatch.setattr(rrattention, "rr_attn_estimate_triton_func", fake_estimate_triton)

    query_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)

    rrattention.rrattn_estimate(
        query_states,
        key_states,
        chunk_size=512,
        rrattn_version="v2",
    )

    assert captured["chunk_size"] == 512


def test_rrattn_estimate_v2_uses_kernel_selected_blocks(monkeypatch):
    import rrattn.rrattention as rrattention

    def fake_estimate_triton(q, k, startend_row_indices, **kwargs):
        shape = [1, 1, 1, 1]
        return (
            paddle.zeros(shape, dtype=paddle.float32),
            paddle.ones(shape, dtype=paddle.bool),
            paddle.zeros(shape, dtype=paddle.bool),
        )

    monkeypatch.setattr(rrattention, "can_use_triton_kernels", lambda: True)
    monkeypatch.setattr(rrattention, "rr_attn_estimate_triton_func", fake_estimate_triton)

    query_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)

    _, block_mask = rrattention.rrattn_estimate(
        query_states,
        key_states,
        rrattn_version="v2",
    )

    assert not bool(paddle.any(block_mask))


def test_rrattn_estimate_routes_by_version(monkeypatch):
    import rrattn.rrattention as rrattention

    calls = []
    startend = paddle.zeros([1, 1, 1, 1], dtype=paddle.int32)

    def fake_legacy(*args, **kwargs):
        calls.append(("v1", kwargs["rrattn_version"], kwargs["startend_row_indices"]))
        shape = [1, 1, 1, 1]
        return paddle.zeros(shape), paddle.zeros(shape, dtype=paddle.bool)

    def fake_v2(*args, **kwargs):
        calls.append(("v2", kwargs["rrattn_version"], kwargs["startend_row_indices"]))
        shape = [1, 1, 1, 1]
        return paddle.zeros(shape), paddle.zeros(shape, dtype=paddle.bool)

    monkeypatch.setattr(rrattention, "rrattn_estimate_legacy", fake_legacy)
    monkeypatch.setattr(rrattention, "_rrattn_estimate_v2", fake_v2)

    query_states = paddle.zeros([1, 1, 1, 1], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 1, 1], dtype=paddle.float32)

    rrattention.rrattn_estimate(query_states, key_states, rrattn_version="v1", startend_row_indices=startend)
    rrattention.rrattn_estimate(query_states, key_states, rrattn_version="v2", startend_row_indices=startend)

    assert calls[0] == ("v1", "v1", None)
    assert calls[1][0] == "v2"
    assert calls[1][1] == "v2"
    assert calls[1][2] is startend


def test_rrattn_estimate_rejects_unknown_version():
    import rrattn.rrattention as rrattention

    query_states = paddle.zeros([1, 1, 1, 1], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 1, 1], dtype=paddle.float32)

    with pytest.raises(ValueError, match="Unsupported rrattn_version"):
        rrattention.rrattn_estimate(query_states, key_states, rrattn_version="bad")


def test_rrattn_prefill_drops_startend_for_v1(monkeypatch):
    import rrattn.rrattention as rrattention

    captured = {}

    def fake_estimate(*args, startend_row_indices=None, **kwargs):
        captured["estimate_startend"] = startend_row_indices
        return (
            paddle.zeros([1, 1, 1, 1], dtype=paddle.float32),
            paddle.ones([1, 1, 1, 1], dtype=paddle.bool),
        )

    def fake_block_sparse_attention(*args, startend_row_indices=None, **kwargs):
        captured["attention_startend"] = startend_row_indices
        return paddle.zeros([1, 128, 1, 128], dtype=paddle.float32)

    monkeypatch.setattr(rrattention, "rrattn_estimate", fake_estimate)
    monkeypatch.setattr(rrattention, "block_sparse_attention", fake_block_sparse_attention)

    query_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    value_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    startend = paddle.full([1, 1, 128, 1], 128, dtype=paddle.int32)

    rrattention.rrattn_prefill(
        query_states,
        key_states,
        value_states,
        rrattn_version="v1",
        startend_row_indices=startend,
    )

    assert captured["estimate_startend"] is None
    assert captured["attention_startend"] is None
