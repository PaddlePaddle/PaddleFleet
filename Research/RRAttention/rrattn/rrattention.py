import math

import paddle
import paddle.nn.functional as F

from .kernels import flat_group_gemm_fuse_reshape, softmax_fuse_block_sum
from .kernels_rrattn import rr_attn_estimate_triton_func
from .config_rrattn import RRAttnConfig
from .utils import find_blocks_chunked


enable_profile = False
attn_time_ms = 0.0
estimate_func_time_ms = 0.0


def set_profile(enable=True):
    global enable_profile
    enable_profile = enable


def is_enable_profile():
    global enable_profile
    return enable_profile


def set_attn_time(attn_time=0.0):
    global attn_time_ms
    attn_time_ms = attn_time


def get_attn_time():
    global attn_time_ms
    return attn_time_ms


def add_attn_time(attn_time):
    global attn_time_ms
    attn_time_ms += attn_time


def set_estimate_func_time(time_ms=0.0):
    global estimate_func_time_ms
    estimate_func_time_ms = time_ms


def get_estimate_func_time():
    global estimate_func_time_ms
    return estimate_func_time_ms


def add_estimate_func_time(time_ms):
    global estimate_func_time_ms
    estimate_func_time_ms += time_ms


def can_use_triton_kernels():
    if not paddle.device.is_compiled_with_cuda():
        return False
    try:
        return paddle.device.get_device().startswith("gpu")
    except Exception:
        return False


def _compute_sparse_ratio(
    block_mask: paddle.Tensor,
    *,
    q_block_num: int,
    k_block_num: int,
    num_heads: int,
    causal: bool,
):
    if causal:
        offset = k_block_num - q_block_num
        visible_blocks = 0
        for q_idx in range(q_block_num):
            visible_blocks += min(k_block_num, max(0, q_idx + offset + 1))
        num_to_compute = visible_blocks * num_heads
    else:
        num_to_compute = q_block_num * k_block_num * num_heads
    sparse_ratio = 1.0 - (block_mask.astype(paddle.float32).sum() / max(float(num_to_compute), 1.0))
    return paddle.clip(sparse_ratio, min=0.0, max=1.0)


def _build_nomask_startend(
    *,
    batch_size: int,
    q_len: int,
    k_len: int,
    causal: bool,
    device,
) -> paddle.Tensor:
    if causal:
        return paddle.full(
            (batch_size, 1, k_len, 1),
            q_len,
            dtype=paddle.int32,
            device=device,
        )

    start = paddle.full(
        (batch_size, 1, k_len, 1),
        q_len,
        dtype=paddle.int32,
        device=device,
    )
    end = paddle.zeros((batch_size, 1, k_len, 1), dtype=paddle.int32, device=device)
    return paddle.concat([start, end], axis=-1)


def block_sparse_attention(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    block_mask: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    block_size: int = 128,
    causal: bool = True,
):
    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape

    assert block_size == 128, "F.flashmask_attention block_mask only supports block_size=128"
    assert head_dim == 128, "F.flashmask_attention block_mask only supports head_dim=128"

    block_mask = block_mask.astype(paddle.int32).contiguous()
    if startend_row_indices is None:
        startend_row_indices = _build_nomask_startend(
            batch_size=batch_size,
            q_len=q_len,
            k_len=k_len,
            causal=causal,
            device=query_states.device,
        )
    if startend_row_indices.shape[1] != block_mask.shape[1]:
        num_indices_heads = int(startend_row_indices.shape[1])
        assert block_mask.shape[1] % num_indices_heads == 0, "block_mask heads must be divisible by startend heads"
        startend_row_indices = startend_row_indices.repeat_interleave(
            block_mask.shape[1] // num_indices_heads,
            axis=1,
        )

    attn_output = F.flashmask_attention(
        query_states.transpose(1, 2).contiguous(),
        key_states.transpose(1, 2).contiguous(),
        value_states.transpose(1, 2).contiguous(),
        startend_row_indices=startend_row_indices,
        dropout=0.0,
        causal=causal,
        block_mask=block_mask,
    )
    return attn_output.contiguous()


def rrattn_estimate_legacy(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    block_size=128,
    stride=8,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    rrattn_version="v1",
    layer_idx=None,
    startend_row_indices=None,
    config=None,
    **kwargs,
):
    del layer_idx, startend_row_indices, config, kwargs

    batch_size, num_q_head, q_len, head_dim = query_states.shape
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    assert num_q_head == num_kv_head
    assert rrattn_version == "v1"

    if use_triton and not can_use_triton_kernels():
        raise RuntimeError("rrattn v1 Triton kernels require a CUDA gpu device")

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to(key_states.device)
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(key_states.device)
    else:
        pad_query_states = query_states

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    q_reshaped_seq_len = (q_len + q_num_to_pad) // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    if not use_triton:
        raise NotImplementedError("rrattn v1 currently requires use_triton=True")

    attn_sum_list = []
    for chunk_idx in range(q_chunk_num):
        if kdb != 1:
            raise ValueError("use_triton and kdb cannot be used together")
        attn_weights_slice = flat_group_gemm_fuse_reshape(
            pad_query_states[
                :,
                :,
                (chunk_idx * reshaped_chunk_size) * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                * stride,
                :,
            ],
            pad_key_states,
            stride,
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
            is_causal=causal,
        )
        attn_sum = softmax_fuse_block_sum(
            attn_weights_slice,
            reshaped_block_size,
            min(4096, reshaped_block_size),
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
            q_reshaped_seq_len - q_reshaped_num_to_pad,
            k_reshaped_seq_len - k_reshaped_num_to_pad,
            1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
            is_causal=causal,
        )
        attn_sum_list.append(attn_sum)

    attn_sums = paddle.concat(attn_sum_list, dim=-2)
    simple_masks = find_blocks_chunked(
        attn_sums,
        0,
        threshold,
        None,
        decoding=False,
        mode="prefill",
        causal=causal,
    )

    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = paddle.where(
            paddle.tril(
                paddle.ones(q_block_num, q_block_num, dtype=paddle.bool, device=key_states.device),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )

    if keep_sink:
        simple_masks[:, :, :, 0] = True
    if keep_recent:
        simple_masks[:, :, (q_len + block_size - 1) // block_size - 1, :(k_len + block_size - 1) // block_size] = True

    # simple_masks[:, :, (q_len + block_size - 1) // block_size - 1, :(k_len + block_size - 1) // block_size] = True
    return attn_sums, simple_masks


def _rrattn_estimate_v2(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    block_size=128,
    stride=8,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    rrattn_version="v1",
    layer_idx=None,
    startend_row_indices=None,
    config: RRAttnConfig | None = None,
    **kwargs,
):
    del norm, softmax, select_mode, rrattn_version, layer_idx, kwargs

    if not use_triton:
        raise NotImplementedError("rrattn v2 currently requires use_triton=True")
    if not can_use_triton_kernels():
        raise RuntimeError("rrattn v2 Triton kernels require a CUDA gpu device")
    if kdb != 1:
        raise ValueError("rrattn v2 Triton kernels require kdb=1")
    assert block_size == 128, "RRAttention currently requires block_size=128"
    assert stride > 0, "stride must be positive"
    assert block_size % stride == 0, "stride must divide block_size=128"
    assert chunk_size is not None and chunk_size > 0, "chunk_size must be positive"

    batch_size, num_q_heads, q_len, _ = query_states.shape
    _, num_kv_heads, k_len, _ = key_states.shape
    assert num_q_heads % num_kv_heads == 0, "MHA/GQA requires num_q_heads % num_kv_heads == 0"

    if key_states.device != query_states.device:
        key_states = key_states.to(query_states.device)

    if startend_row_indices is None:
        startend_row_indices = _build_nomask_startend(
            batch_size=batch_size,
            q_len=q_len,
            k_len=k_len,
            causal=causal,
            device=query_states.device,
        )

    attn_sums, _, selected_blocks = rr_attn_estimate_triton_func(
        query_states.transpose(1, 2).contiguous(),
        key_states.transpose(1, 2).contiguous(),
        startend_row_indices,
        stride=stride,
        causal=causal,
        threshold=threshold,
        chunk_size=chunk_size,
        config=config,
    )

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    attn_sums = attn_sums[:, :, :q_block_num, :k_block_num].contiguous()
    block_mask = selected_blocks[:, :, :q_block_num, :k_block_num].contiguous()

    if keep_sink and k_block_num > 0:
        block_mask[:, :, :, 0] = True
    if keep_recent and q_block_num > 0:
        block_mask[:, :, -1, :k_block_num] = True

    return attn_sums, block_mask


def rrattn_estimate(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    block_size=128,
    stride=8,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    rrattn_version="v1",
    layer_idx=None,
    startend_row_indices=None,
    config: RRAttnConfig | None = None,
    **kwargs,
):
    if rrattn_version == "v1":
        return rrattn_estimate_legacy(
            query_states,
            key_states,
            block_size=block_size,
            stride=stride,
            norm=norm,
            softmax=softmax,
            threshold=threshold,
            chunk_size=chunk_size,
            select_mode=select_mode,
            use_triton=use_triton,
            causal=causal,
            kdb=kdb,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            rrattn_version=rrattn_version,
            layer_idx=layer_idx,
            startend_row_indices=None,
            config=config,
            **kwargs,
        )
    if rrattn_version == "v2":
        return _rrattn_estimate_v2(
            query_states,
            key_states,
            block_size=block_size,
            stride=stride,
            norm=norm,
            softmax=softmax,
            threshold=threshold,
            chunk_size=chunk_size,
            select_mode=select_mode,
            use_triton=use_triton,
            causal=causal,
            kdb=kdb,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            rrattn_version=rrattn_version,
            layer_idx=layer_idx,
            startend_row_indices=startend_row_indices,
            config=config,
            **kwargs,
        )
    raise ValueError(f"Unsupported rrattn_version={rrattn_version!r}; expected 'v1' or 'v2'")


def rrattn_prefill(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    stride=8,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
    rrattn_version="v1",
    layer_idx=None,
    startend_row_indices=None,
    config=None,
):
    batch_size, num_q_heads, q_len, _ = query_states.shape
    _, num_kv_heads, k_len, _ = key_states.shape
    assert num_q_heads % num_kv_heads == 0, "MHA/GQA requires num_q_heads % num_kv_heads == 0"
    assert value_states.shape[1] == num_kv_heads, "key/value head count mismatch"
    assert value_states.shape[2] == k_len, "key/value sequence length mismatch"

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )
    chunk_size = min(
        (q_len + (block_size * stride) - 1) // (block_size * stride) * (block_size * stride),
        chunk_size,
    )
    rrattn_startend_row_indices = startend_row_indices if rrattn_version == "v2" else None

    if is_enable_profile():
        paddle.cuda.synchronize()
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()

    attn_sums, approx_simple_mask = rrattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        rrattn_version=rrattn_version,
        layer_idx=layer_idx,
        startend_row_indices=rrattn_startend_row_indices,
        config=config,
    )
    if is_enable_profile():
        end_event.record()
        paddle.cuda.synchronize()
        add_estimate_func_time(start_event.elapsed_time(end_event))

    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    approx_simple_mask = approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous()
    sparse_ratio = _compute_sparse_ratio(
        approx_simple_mask,
        q_block_num=q_block_num,
        k_block_num=k_block_num,
        num_heads=num_q_heads,
        causal=causal,
    )

    if is_enable_profile():
        paddle.cuda.synchronize()
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()
    attn_output = block_sparse_attention(
        query_states,
        key_states,
        value_states,
        approx_simple_mask,
        startend_row_indices=rrattn_startend_row_indices,
        block_size=block_size,
        causal=causal,
    )
    if is_enable_profile():
        end_event.record()
        paddle.cuda.synchronize()
        add_attn_time(start_event.elapsed_time(end_event))

    del query_states
    del approx_simple_mask, attn_sums
    return attn_output, sparse_ratio


__all__ = [
    "rrattn_estimate",
    "rrattn_estimate_legacy",
    "rrattn_prefill",
    "can_use_triton_kernels",
    "set_profile",
    "is_enable_profile",
    "set_attn_time",
    "get_attn_time",
    "add_attn_time",
    "set_estimate_func_time",
    "get_estimate_func_time",
    "add_estimate_func_time",
]
