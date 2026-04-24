import pytest


paddle = pytest.importorskip("paddle")
pytest.importorskip("triton")


def _cuda_available():
    if not paddle.device.is_compiled_with_cuda():
        return False
    try:
        return paddle.device.cuda.device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="RRAttention estimate tests require CUDA and Triton.",
)


def _set_device():
    paddle.set_device("gpu:0")


def _random_bhsd(batch_size, num_heads, seq_len, seed):
    paddle.seed(seed)
    return paddle.randn([batch_size, num_heads, seq_len, 128], dtype="float16")


def _causal_window_startend(batch_size, k_len, q_len, window):
    lt_start = paddle.arange(k_len, dtype=paddle.int32).reshape([1, 1, k_len, 1])
    lt_start = paddle.minimum(
        lt_start + window,
        paddle.full([1, 1, k_len, 1], q_len, dtype=paddle.int32),
    )
    return lt_start.tile([batch_size, 1, 1, 1])


@pytest.mark.parametrize("seq_len", [1024, 2048])
def test_rrattn_estimate_nomask_smoke(seq_len):
    import rrattn.rrattention as rrattention

    _set_device()
    query_states = _random_bhsd(1, 2, seq_len, seed=101)
    key_states = _random_bhsd(1, 2, seq_len, seed=202)

    attn_sums, block_mask = rrattention.rrattn_estimate(
        query_states,
        key_states,
        stride=8,
        threshold=0.95,
        causal=True,
        keep_sink=False,
        keep_recent=False,
        chunk_size=seq_len,
        rrattn_version="bad",
    )

    expected_shape = [1, 2, (seq_len + 127) // 128, (seq_len + 127) // 128]
    assert attn_sums.shape == expected_shape
    assert block_mask.shape == expected_shape
    assert block_mask.dtype == paddle.bool
    assert bool(paddle.all(paddle.isfinite(attn_sums.astype(paddle.float32))))


def test_rrattn_estimate_flashmask_smoke_uses_startend():
    import rrattn.rrattention as rrattention

    _set_device()
    query_states = _random_bhsd(1, 4, 257, seed=303)
    key_states = _random_bhsd(1, 2, 257, seed=404)
    startend_row_indices = _causal_window_startend(1, 257, 257, window=96)

    attn_sums, block_mask = rrattention.rrattn_estimate(
        query_states,
        key_states,
        stride=8,
        threshold=0.95,
        causal=True,
        rrattn_version="bad",
        startend_row_indices=startend_row_indices,
    )

    assert attn_sums.shape == [1, 4, 3, 3]
    assert block_mask.shape == [1, 4, 3, 3]
    assert block_mask.dtype == paddle.bool
    assert bool(paddle.all(paddle.isfinite(attn_sums.astype(paddle.float32))))


def test_fa3_causal_block_masks_are_bottom_right_aligned():
    import rrattn.rrattention as rrattention

    input_tensor = paddle.zeros([1, 1, 2, 4], dtype=paddle.float32)

    visible_mask = rrattention._build_fa3_causal_block_visible_mask(input_tensor, 256, 512)
    mandatory_mask = rrattention._build_causal_prefill_mandatory_mask(input_tensor, 256, 512)

    expected_visible = paddle.to_tensor(
        [[[[True, True, True, False], [True, True, True, True]]]],
        dtype=paddle.bool,
    )
    expected_mandatory = paddle.to_tensor(
        [[[[True, False, True, False], [True, False, False, True]]]],
        dtype=paddle.bool,
    )

    assert bool(paddle.equal_all(visible_mask, expected_visible))
    assert bool(paddle.equal_all(mandatory_mask, expected_mandatory))


def test_fa3_causal_block_masks_use_token_offset():
    import rrattn.rrattention as rrattention

    input_tensor = paddle.zeros([1, 1, 3, 3], dtype=paddle.float32)

    visible_mask = rrattention._build_fa3_causal_block_visible_mask(input_tensor, 257, 384)
    mandatory_mask = rrattention._build_causal_prefill_mandatory_mask(input_tensor, 257, 384)

    expected_visible = paddle.to_tensor(
        [[[[True, True, False], [True, True, True], [True, True, True]]]],
        dtype=paddle.bool,
    )
    expected_mandatory = paddle.to_tensor(
        [[[[True, True, False], [True, False, True], [True, False, True]]]],
        dtype=paddle.bool,
    )

    assert bool(paddle.equal_all(visible_mask, expected_visible))
    assert bool(paddle.equal_all(mandatory_mask, expected_mandatory))


def test_rrattn_estimate_causal_suffix_uses_token_offset():
    import rrattn.rrattention as rrattention

    _set_device()
    query_states = _random_bhsd(1, 2, 257, seed=707)
    key_states = _random_bhsd(1, 2, 384, seed=808)

    _, block_mask = rrattention.rrattn_estimate(
        query_states,
        key_states,
        stride=8,
        threshold=0.95,
        causal=True,
        rrattn_version="bad",
        keep_sink=False,
        keep_recent=False,
        chunk_size=512,
    )

    assert block_mask.shape == [1, 2, 3, 3]
    assert not bool(paddle.any(block_mask[:, :, 0, 2]).item())


def test_block_sparse_attention_expands_shared_startend_heads(monkeypatch):
    import rrattn.rrattention as rrattention

    captured = {}

    def fake_flashmask_attention(query, key, value, *, startend_row_indices, dropout, causal, block_mask):
        captured["startend_shape"] = startend_row_indices.shape
        captured["block_mask_shape"] = block_mask.shape
        captured["startend"] = startend_row_indices
        return paddle.zeros(query.shape, dtype=query.dtype)

    monkeypatch.setattr(rrattention.F, "flashmask_attention", fake_flashmask_attention)

    query_states = paddle.zeros([1, 4, 128, 128], dtype=paddle.float16)
    key_states = paddle.zeros([1, 2, 128, 128], dtype=paddle.float16)
    value_states = paddle.zeros([1, 2, 128, 128], dtype=paddle.float16)
    block_mask = paddle.ones([1, 4, 1, 1], dtype=paddle.bool)
    startend_row_indices = paddle.full([1, 1, 128, 1], 128, dtype=paddle.int32)

    output = rrattention.block_sparse_attention(
        query_states,
        key_states,
        value_states,
        block_mask,
        startend_row_indices=startend_row_indices,
    )

    assert output.shape == [1, 128, 4, 128]
    assert captured["startend_shape"] == list(captured["block_mask_shape"][:2]) + [128, 1]
    assert bool(paddle.equal_all(captured["startend"][:, 0], captured["startend"][:, 3]))
