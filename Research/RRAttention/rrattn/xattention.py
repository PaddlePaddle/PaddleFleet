# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from contextlib import nullcontext

import paddle
import paddle.nn.functional as F
import triton
import triton.language as tl
from paddle.compat import use_torch_proxy_guard

from .utils import find_blocks_chunked


@triton.jit
def softmax_fuse_block_sum_kernel_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size
    num_iters_before_causal = (
        chunk_start + (block_id + 1) * block_size - 1
    ) // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = (
        In
        + batch_id * input_stride_0
        + head_id * input_stride_1
        + block_id * block_size * input_stride_2
    )
    input_ptr = (
        input_ptr
        + tl.arange(0, segment_size)
        + tl.arange(0, block_size)[:, None] * input_stride_2
    )

    output_ptr = (
        Out
        + batch_id * output_stride_0
        + head_id * output_stride_1
        + block_id * output_stride_2
    )
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i
    q_valid = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(q_valid, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(
            output_ptr + iter * segment_size // block_size,
            X.to(Out.type.element_ty),
        )

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(q_valid, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(
            output_ptr + iter * segment_size // block_size,
            X.to(Out.type.element_ty),
        )

    for iter in range(num_iters_before_causal + 1, num_iters):
        X = tl.zeros([segment_size // block_size], dtype=tl.float32)
        tl.store(
            output_ptr + iter * segment_size // block_size,
            X.to(Out.type.element_ty),
        )


@triton.jit
def softmax_fuse_block_sum_kernel_non_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)
    num_iters = k_len // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = (
        In
        + batch_id * input_stride_0
        + head_id * input_stride_1
        + block_id * block_size * input_stride_2
    )
    input_ptr = (
        input_ptr
        + tl.arange(0, segment_size)
        + tl.arange(0, block_size)[:, None] * input_stride_2
    )

    output_ptr = (
        Out
        + batch_id * output_stride_0
        + head_id * output_stride_1
        + block_id * output_stride_2
    )
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i
    q_valid = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(q_valid, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(
            output_ptr + iter * segment_size // block_size,
            X.to(Out.type.element_ty),
        )


@triton.jit
def flat_group_gemm_fuse_reshape_kernel(
    Q,
    K,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_oz,
    stride_oh,
    stride_on,
    chunk_start,
    chunk_end,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    is_caual: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if is_caual:
        if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
            return

    Q_ptrs = (
        Q
        + batch_id * stride_qz
        + head_id * stride_qh
        + block_m * BLOCK_M * STRIDE * stride_qn
    )
    K_ptrs = (
        K
        + batch_id * stride_kz
        + head_id * stride_kh
        + block_n * BLOCK_N * STRIDE * stride_kn
    )

    Q_ptrs = (
        Q_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * (stride_qn * STRIDE)
        + tl.arange(0, HEAD_DIM)[None, :]
        + stride_qn * (STRIDE - 1)
    )
    K_ptrs = (
        K_ptrs
        + tl.arange(0, BLOCK_N)[None, :] * (stride_kn * STRIDE)
        + tl.arange(0, HEAD_DIM)[:, None]
    )

    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for iter in range(STRIDE):
        q = tl.load(Q_ptrs - iter * stride_qn)
        k = tl.load(K_ptrs + iter * stride_kn)
        o += tl.dot(q, k)

    O_ptrs = (
        Out
        + batch_id * stride_oz
        + head_id * stride_oh
        + block_m * BLOCK_M * stride_on
        + block_n * BLOCK_N
    )
    O_ptrs = (
        O_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * stride_on
        + tl.arange(0, BLOCK_N)[None, :]
    )
    tl.store(O_ptrs, o.to(Out.type.element_ty))


def softmax_fuse_block_sum(
    attn_weights_slice,
    reshaped_block_size,
    segment_size,
    chunk_start,
    chunk_end,
    real_q_len,
    scale,
    is_causal=True,
):
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    assert k_len % segment_size == 0
    assert segment_size % reshaped_block_size == 0
    assert attn_weights_slice.stride(-1) == 1

    output = paddle.empty(
        (
            batch_size,
            num_heads,
            q_len // reshaped_block_size,
            k_len // reshaped_block_size,
        ),
        dtype=attn_weights_slice.dtype,
        device=attn_weights_slice.device,
    )

    grid = (q_len // reshaped_block_size, num_heads, batch_size)
    if is_causal:
        softmax_fuse_block_sum_kernel_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )
    else:
        softmax_fuse_block_sum_kernel_non_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )

    return output


def flat_group_gemm_fuse_reshape(
    query_states,
    key_states,
    stride,
    chunk_start,
    chunk_end,
    is_causal=True,
):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert key_states.shape[0] == batch_size
    assert key_states.shape[1] == num_heads
    assert key_states.shape[3] == head_dim

    output = paddle.empty(
        (batch_size, num_heads, q_len // stride, kv_len // stride),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    block_m = 128
    block_n = 128
    assert q_len % (stride * block_m) == 0
    assert kv_len % (stride * block_n) == 0

    grid = (
        q_len // stride // block_m,
        kv_len // stride // block_n,
        batch_size * num_heads,
    )
    flat_group_gemm_fuse_reshape_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        stride,
        head_dim,
        block_m,
        block_n,
        is_causal,
    )

    return output


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


def set_estimate_func_time(estimate_func_time=0.0):
    global estimate_func_time_ms
    estimate_func_time_ms = estimate_func_time


def get_estimate_func_time():
    global estimate_func_time_ms
    return estimate_func_time_ms


def add_estimate_func_time(estimate_func_time):
    global estimate_func_time_ms
    estimate_func_time_ms += estimate_func_time


def can_use_triton_kernels():
    if not paddle.device.is_compiled_with_cuda():
        return False
    try:
        if not paddle.device.get_device().startswith("gpu"):
            return False
        device_name = paddle.device.cuda.get_device_name()
        return "100" in device_name or "H800" in device_name in device_name
    except Exception:
        return False


def block_sparse_attention(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    block_mask: paddle.Tensor,
    block_size: int = 128,
    causal: bool = True,
):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape

    assert block_size == 128, (
        "F.flashmask_attention block_mask only supports block_size=128"
    )
    assert head_dim == 128, (
        "F.flashmask_attention block_mask only supports head_dim=128"
    )
    if not causal:
        raise NotImplementedError(
            "F.flashmask_attention block_mask path currently supports causal=True"
        )

    block_mask = block_mask.astype(paddle.int32).contiguous()
    startend_row_indices = paddle.full(
        (batch_size, num_heads, k_len, 1),
        q_len,
        dtype=paddle.int32,
        device=query_states.device,
    ).contiguous()

    attn_output = F.flashmask_attention(
        query_states.transpose(1, 2).contiguous(),
        key_states.transpose(1, 2).contiguous(),
        value_states.transpose(1, 2).contiguous(),
        startend_row_indices,
        dropout=0.0,
        causal=causal,
        block_mask=block_mask,
    )
    return attn_output.contiguous()


def xattn_estimate(
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
    layer_idx=None,
    **kwargs,
):
    del layer_idx, kwargs

    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    if use_triton and not can_use_triton_kernels():
        use_triton = False
    if use_triton and kdb != 1:
        raise ValueError("use_triton and kdb cannot be used together")

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to(
            key_states.device
        )
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(
            query_states, (0, 0, 0, q_num_to_pad), value=0
        ).to(query_states.device)
    else:
        pad_query_states = query_states

    attn_sum_list = []
    simple_mask_list = []
    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    if not use_triton:
        if select_mode == "random":
            perm_idx = paddle.randperm(stride).tolist()
            reshaped_key = paddle.concat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)],
                dim=-1,
            )
            reshaped_query = paddle.concat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            reshaped_key = paddle.concat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)],
                dim=-1,
            )
            reshaped_query = paddle.concat(
                [
                    (
                        pad_query_states[
                            :, :, (stride - 1 - q) :: (stride * kdb), :
                        ]
                    )
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = paddle.concat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)],
                dim=-1,
            )
            reshaped_query = paddle.concat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)],
                dim=-1,
            )
        elif select_mode == "double":
            reshaped_key = paddle.concat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)],
                dim=-1,
            )
            reshaped_key = reshaped_key + paddle.concat(
                [
                    reshaped_key[:, :, :, head_dim:],
                    reshaped_key[:, :, :, :head_dim],
                ],
                dim=-1,
            )
            reshaped_query = paddle.concat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = paddle.concat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)],
                dim=-1,
            )
            reshaped_key = reshaped_key + paddle.concat(
                [
                    reshaped_key[:, :, :, head_dim:],
                    reshaped_key[:, :, :, :head_dim],
                ],
                dim=-1,
            )
            reshaped_key = reshaped_key + paddle.concat(
                [
                    reshaped_key[:, :, :, -head_dim:],
                    reshaped_key[:, :, :, :-head_dim],
                ],
                dim=-1,
            )
            reshaped_query = paddle.concat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError(
                f"Unsupported select_mode={select_mode!r}"
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    proxy_guard = (
        use_torch_proxy_guard(silent=True) if use_triton else nullcontext()
    )
    with proxy_guard:
        for chunk_idx in range(q_chunk_num):
            if use_triton:
                attn_weights_slice = flat_group_gemm_fuse_reshape(
                    pad_query_states[
                        :,
                        :,
                        (chunk_idx * reshaped_chunk_size) * stride : (
                            chunk_idx * reshaped_chunk_size
                            + reshaped_chunk_size
                        )
                        * stride,
                        :,
                    ],
                    pad_key_states,
                    stride,
                    (k_block_num - q_block_num) * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size,
                    (k_block_num - q_block_num) * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size
                    + reshaped_chunk_size,
                    is_causal=causal,
                )
                attn_sum = softmax_fuse_block_sum(
                    attn_weights_slice,
                    reshaped_block_size,
                    min(4096, reshaped_block_size),
                    (k_block_num - q_block_num) * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size,
                    (k_block_num - q_block_num) * reshaped_block_size
                    + chunk_idx * reshaped_chunk_size
                    + reshaped_chunk_size,
                    k_reshaped_seq_len - k_reshaped_num_to_pad,
                    1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                    is_causal=causal,
                )
            else:
                chunked_query = reshaped_query[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size) // kdb : (
                        chunk_idx * reshaped_chunk_size + reshaped_chunk_size
                    )
                    // kdb,
                    :,
                ]
                attn_weights_slice = paddle.matmul(
                    chunked_query,
                    reshaped_key.transpose(2, 3),
                ).to(chunked_query.device)
                attn_weights_slice = (
                    attn_weights_slice / math.sqrt(head_dim) / stride / norm
                )

                if causal:
                    causal_mask = paddle.zeros(
                        (
                            batch_size,
                            num_q_head,
                            reshaped_chunk_size,
                            reshaped_chunk_size * k_chunk_num,
                        ),
                        device=key_states.device,
                    )
                    if k_reshaped_num_to_pad > 0:
                        causal_mask[:, :, :, -k_reshaped_num_to_pad:] = float(
                            "-inf"
                        )
                    chunk_start = (
                        chunk_idx + offset_token_chunk_num
                    ) * reshaped_chunk_size
                    chunk_end = chunk_start + reshaped_chunk_size
                    causal_mask[:, :, :, chunk_start:chunk_end] = paddle.triu(
                        paddle.ones(
                            1,
                            num_q_head,
                            reshaped_chunk_size,
                            reshaped_chunk_size,
                            device=key_states.device,
                        )
                        * float("-inf"),
                        diagonal=1,
                    )
                    if (
                        chunk_idx == q_chunk_num - 1
                        and q_reshaped_num_to_pad != 0
                    ):
                        causal_mask[
                            :, :, -(q_reshaped_num_to_pad // kdb) :, :
                        ] = float("-inf")
                    causal_mask[:, :, :, chunk_end:] = float("-inf")
                    causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                    attn_weights_slice = attn_weights_slice + causal_mask.to(
                        attn_weights_slice.device
                    )

                if softmax:
                    attn_weights_slice = F.softmax(
                        attn_weights_slice, dim=-1, dtype=paddle.float32
                    ).to(pad_query_states.dtype)
                else:
                    attn_weights_slice = paddle.exp(attn_weights_slice).to(
                        pad_query_states.dtype
                    )
                attn_weights_slice = F.dropout(
                    attn_weights_slice, p=0, training=False
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    attn_weights_slice[
                        :, :, -(q_reshaped_num_to_pad // kdb) :, :
                    ] = 0

                attn_sum = (
                    attn_weights_slice.view(
                        batch_size,
                        num_kv_head,
                        num_blocks_per_chunk,
                        reshaped_block_size // kdb,
                        -1,
                        reshaped_block_size,
                    )
                    .sum(dim=-1)
                    .sum(dim=-2)
                    .to(chunked_query.device)
                )
                del chunked_query

            simple_mask = find_blocks_chunked(
                attn_sum,
                k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
                threshold,
                None,
                decoding=False,
                mode="prefill",
                causal=causal,
            )
            attn_sum_list.append(attn_sum)
            simple_mask_list.append(simple_mask)
            del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key

    attn_sums = paddle.concat(attn_sum_list, dim=-2)
    simple_masks = paddle.concat(simple_mask_list, dim=-2)

    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = paddle.where(
            paddle.tril(
                paddle.ones(
                    q_block_num,
                    q_block_num,
                    dtype=paddle.bool,
                    device=key_states.device,
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    if keep_sink:
        simple_masks[:, :, :, 0] = True
    if keep_recent:
        eye_matrix = paddle.eye(
            q_block_num, device=simple_masks.device, dtype=paddle.int32
        ).astype(paddle.bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = paddle.where(
            eye_matrix_expanded,
            True,
            simple_masks[:, :, -q_block_num:, -q_block_num:],
        )

    return attn_sums, simple_masks


def xattn_prefill(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    stride,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
    layer_idx=None,
):
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

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
        (q_len + (block_size * stride) - 1)
        // (block_size * stride)
        * (block_size * stride),
        chunk_size,
    )

    if is_enable_profile():
        paddle.cuda.synchronize()
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()

    attn_sums, approx_simple_mask = xattn_estimate(
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
        layer_idx=layer_idx,
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

    approx_simple_mask = approx_simple_mask[
        :, :, :q_block_num, :k_block_num
    ].contiguous()
    if causal and q_block_num == k_block_num:
        num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads
    else:
        num_to_compute = q_block_num * k_block_num * num_heads
    sparse_ratio = 1.0 - (
        approx_simple_mask.astype(paddle.float32).sum()
        / max(float(num_to_compute), 1.0)
    )
    sparse_ratio = paddle.clip(sparse_ratio, min=0.0, max=1.0)

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
    "xattn_estimate",
    "xattn_prefill",
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
