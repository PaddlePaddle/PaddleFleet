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
