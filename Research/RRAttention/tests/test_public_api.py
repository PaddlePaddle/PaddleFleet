import inspect

import pytest


paddle = pytest.importorskip("paddle")
pytest.importorskip("paddleformers")


def test_public_api_imports():
    import rrattn

    assert callable(rrattn.rrattn_estimate)
    assert callable(rrattn.rrattn_prefill)
    assert callable(rrattn.patch_llama_attention)
    assert callable(rrattn.patch_qwen_attention)
    assert callable(rrattn.patch_ernie_attention)
    assert callable(rrattn.RRAttnConfig)
    assert callable(rrattn.get_rrattn_config)

    assert not hasattr(rrattn, "xattn_estimate")
    assert not hasattr(rrattn, "xattn_prefill")
    assert not hasattr(rrattn, "flex_prefill")
    assert not hasattr(rrattn, "TUNABLE_FIELDS")
    assert rrattn.__all__ == [
        "rrattn_estimate",
        "rrattn_prefill",
        "patch_llama_attention",
        "patch_qwen_attention",
        "patch_ernie_attention",
        "RRAttnConfig",
        "get_rrattn_config",
    ]


def _assert_default(function, arg_name, expected):
    parameter = inspect.signature(function).parameters[arg_name]
    assert parameter.default == expected


def test_rrattn_version_placeholder_signature():
    import rrattn

    _assert_default(rrattn.rrattn_estimate, "rrattn_version", "v1")
    _assert_default(rrattn.rrattn_prefill, "rrattn_version", "v1")
    _assert_default(rrattn.patch_llama_attention, "rrattn_version", "v1")
    _assert_default(rrattn.patch_qwen_attention, "rrattn_version", "v1")
    _assert_default(rrattn.patch_ernie_attention, "rrattn_version", "v1")


def test_patch_utils_supports_all_methods():
    from rrattn.patch_utils import SUPPORTED_METHODS

    assert SUPPORTED_METHODS == ("xattn", "rrattn", "flex", "full")


def test_attention_branch_never_repeats_kv_for_rrattn(monkeypatch):
    import rrattn.patch_utils as patch_utils

    captured = []

    def fake_rrattn_prefill(query_states, key_states, value_states, **kwargs):
        captured.append((key_states.shape, value_states.shape, "rrattn_version" in kwargs))
        return paddle.zeros(
            [query_states.shape[0], query_states.shape[2], query_states.shape[1], query_states.shape[3]],
            dtype=query_states.dtype,
        ), paddle.to_tensor(0.0)

    class Module:
        method = "rrattn"
        num_key_value_groups = 4
        training = False
        threshold = 0.9
        stride = 8

    monkeypatch.setattr(patch_utils, "rrattn_prefill", fake_rrattn_prefill)

    query_states = paddle.zeros([1, 4, 128, 128], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    value_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)

    module = Module()
    module.rrattn_version = "v1"
    patch_utils.attention_branch(module, query_states, key_states, value_states)

    module.rrattn_version = "bad"
    patch_utils.attention_branch(module, query_states, key_states, value_states)

    assert captured[0] == ([1, 1, 128, 128], [1, 1, 128, 128], False)
    assert captured[1] == ([1, 1, 128, 128], [1, 1, 128, 128], False)


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
        rrattn_version="bad",
    )

    assert captured["chunk_size"] == 512


def test_rrattn_estimate_uses_kernel_selected_blocks(monkeypatch):
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
        rrattn_version="bad",
    )

    assert not bool(paddle.any(block_mask))


def test_rrattn_estimate_ignores_version_and_forwards_startend(monkeypatch):
    import rrattn.rrattention as rrattention

    captured = {}
    startend = paddle.full([1, 1, 128, 1], 128, dtype=paddle.int32)

    def fake_estimate_triton(q, k, startend_row_indices, **kwargs):
        captured["q_shape"] = q.shape
        captured["k_shape"] = k.shape
        captured["startend"] = startend_row_indices
        shape = [1, 4, 1, 1]
        return (
            paddle.zeros(shape, dtype=paddle.float32),
            paddle.zeros(shape, dtype=paddle.bool),
            paddle.ones(shape, dtype=paddle.bool),
        )

    monkeypatch.setattr(rrattention, "can_use_triton_kernels", lambda: True)
    monkeypatch.setattr(rrattention, "rr_attn_estimate_triton_func", fake_estimate_triton)

    query_states = paddle.zeros([1, 4, 128, 128], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)

    rrattention.rrattn_estimate(
        query_states,
        key_states,
        rrattn_version="bad",
        startend_row_indices=startend,
    )

    assert captured["q_shape"] == [1, 128, 4, 128]
    assert captured["k_shape"] == [1, 128, 1, 128]
    assert captured["startend"] is startend


def test_rrattn_prefill_reuses_gqa_startend_and_blsd_layout(monkeypatch):
    import rrattn.rrattention as rrattention

    captured = {}

    def fake_estimate_blsd(query_states, key_states, *args, startend_row_indices=None, **kwargs):
        captured["estimate_q_shape"] = query_states.shape
        captured["estimate_k_shape"] = key_states.shape
        captured["estimate_startend"] = startend_row_indices
        return (
            paddle.zeros([1, 4, 1, 1], dtype=paddle.float32),
            paddle.ones([1, 4, 1, 1], dtype=paddle.bool),
        )

    def fake_block_sparse_attention_blsd(
        query_states,
        key_states,
        value_states,
        block_mask,
        *,
        startend_row_indices=None,
        **kwargs,
    ):
        captured["attention_q_shape"] = query_states.shape
        captured["attention_k_shape"] = key_states.shape
        captured["attention_v_shape"] = value_states.shape
        captured["attention_startend"] = startend_row_indices
        captured["block_mask_shape"] = block_mask.shape
        return paddle.zeros([1, 128, 4, 128], dtype=query_states.dtype)

    monkeypatch.setattr(rrattention, "_rrattn_estimate_blsd", fake_estimate_blsd)
    monkeypatch.setattr(rrattention, "_block_sparse_attention_blsd", fake_block_sparse_attention_blsd)

    query_states = paddle.zeros([1, 4, 128, 128], dtype=paddle.float32)
    key_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    value_states = paddle.zeros([1, 1, 128, 128], dtype=paddle.float32)
    startend = paddle.full([1, 1, 128, 1], 128, dtype=paddle.int32)

    rrattention.rrattn_prefill(
        query_states,
        key_states,
        value_states,
        rrattn_version="bad",
        startend_row_indices=startend,
    )

    assert captured["estimate_q_shape"] == [1, 128, 4, 128]
    assert captured["estimate_k_shape"] == [1, 128, 1, 128]
    assert captured["attention_q_shape"] == [1, 128, 4, 128]
    assert captured["attention_k_shape"] == [1, 128, 1, 128]
    assert captured["attention_v_shape"] == [1, 128, 1, 128]
    assert captured["block_mask_shape"] == [1, 4, 1, 1]
    assert captured["estimate_startend"] is startend
    assert captured["attention_startend"] is startend
