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

"""Paddle wrapper around the cuDNN-frontend DSA indexer backward.

This bridges the Python API
``cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_wrapper``
(distributed inside ``nvidia-cudnn-frontend``) into PaddleFleet via dlpack
zero-copy.

It is a drop-in replacement for ``paddlefleet.tilelang_ops.csa_indexer_bwd``,
but exposes the *raw* (target, predict) pair instead of pre-computed
``grad_scores``: cuDNN performs the score-gradient precompute kernel itself,
matching Megatron's ``FusedIndexerSparseAttnFunc`` data flow.

Returns d_index_q / d_weights / d_index_k_comp matching input dtypes/shapes.
"""

from __future__ import annotations

import paddle

_CUDNN_API_IMPORT_ERROR: str | None = None


def _lazy_import_cudnn():
    """Import the cuDNN-frontend DSA indexer-backward API on first use."""
    global _CUDNN_API_IMPORT_ERROR
    try:
        from cudnn.deepseek_sparse_attention.indexer_backward.api import (
            indexer_backward_wrapper,
        )

        return indexer_backward_wrapper
    except Exception as exc:  # pragma: no cover - environment-specific
        _CUDNN_API_IMPORT_ERROR = str(exc)
        raise RuntimeError(
            "csa_indexer_bwd requires the `nvidia-cudnn-frontend` "
            "Python package (provides `cudnn.deepseek_sparse_attention`). "
            f"Import failed: {exc}"
        ) from exc


def _lazy_import_torch():
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "csa_indexer_bwd requires `torch` to be importable so that "
            "tensors can be exchanged with cuDNN frontend via dlpack."
        ) from exc


def _paddle_to_torch(t: paddle.Tensor):
    """Bridge paddle.Tensor -> torch.Tensor via dlpack.

    Zero-copy in the common case (input already contiguous): the returned
    tensor aliases the same GPU memory and mutating it mutates the original.

    Falls back to ``t.contiguous()`` when the input has non-trivial stride.
    In that case a fresh contiguous copy is allocated and the returned
    tensor does **not** alias ``t`` -- callers that rely on aliasing must
    ensure the input is contiguous before invoking.
    """
    torch = _lazy_import_torch()
    if not t.is_contiguous():
        t = t.contiguous()
    # paddle.Tensor implements __dlpack__ (PEP 3118); torch.from_dlpack
    # accepts either a capsule or an object with __dlpack__.
    return torch.from_dlpack(t)


def _torch_to_paddle(t) -> paddle.Tensor:
    """Bridge torch.Tensor -> paddle.Tensor via dlpack.

    Zero-copy when the input is contiguous (cuDNN-allocated outputs always
    are). Otherwise falls back to ``t.contiguous()``, which allocates a
    fresh buffer and breaks aliasing with ``t``.
    """
    if not t.is_contiguous():
        t = t.contiguous()
    return paddle.utils.dlpack.from_dlpack(t)


def _to_bf16(t: paddle.Tensor) -> paddle.Tensor:
    if t.dtype != paddle.bfloat16:
        return t.cast(paddle.bfloat16)
    return t


def csa_indexer_bwd(
    index_q: paddle.Tensor,
    weights: paddle.Tensor,
    index_k_comp: paddle.Tensor,
    target: paddle.Tensor,
    topk_probs: paddle.Tensor,
    topk_indices: paddle.Tensor,
    loss_coeff: float,
    grad_loss: paddle.Tensor | None = None,
    block_I: int = 128,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Run cuDNN-frontend DSA indexer backward on Paddle tensors.

    Args:
        index_q:       [B, S, H, D] bf16, indexer queries.
        weights:       [B, S, H], per-head weights. bf16 expected; fp32 is
            cast to bf16 internally and the gradient cast back.
        index_k_comp:  [B, S_comp, D] bf16, compressed indexer keys.
        target:        [B, S, topk] fp32, multi-head aggregated attention
            target (probability over selected slots).
        topk_probs:    [B, S, topk] fp32, indexer post-softmax probs over the
            same selected slots.
        topk_indices:  [B, S, topk] int32, selected positions (-1 = invalid).
        loss_coeff:    scalar, KL loss coefficient (e.g. 0.01).
        grad_loss:     0-D fp32 paddle tensor, upstream gradient w.r.t. the
            scalar KL loss. ``None`` is treated as 1.0.
        block_I:       cuDNN tile size. 128 matches Megatron production.

    Returns:
        (grad_index_q, grad_weights, grad_index_k_comp) with the same shapes
        as the corresponding inputs and dtypes restored to the original
        ``weights`` dtype.
    """
    indexer_backward_wrapper = _lazy_import_cudnn()
    torch = _lazy_import_torch()

    orig_weights_dtype = weights.dtype
    orig_q_dtype = index_q.dtype
    orig_k_dtype = index_k_comp.dtype

    index_q_bf = _to_bf16(index_q)
    weights_bf = _to_bf16(weights)
    index_k_bf = _to_bf16(index_k_comp)

    # cuDNN overwrites attn_score (target) and index_score (topk_probs)
    # in-place during the score-grad precompute. Clone so the saved
    # forward tensors are untouched.
    target_buf = target.clone()
    predict_buf = topk_probs.clone()
    if target_buf.dtype != paddle.float32:
        target_buf = target_buf.cast(paddle.float32)
    if predict_buf.dtype != paddle.float32:
        predict_buf = predict_buf.cast(paddle.float32)
    if topk_indices.dtype != paddle.int32:
        topk_indices = topk_indices.cast(paddle.int32)

    # grad_loss as a 0-D fp32 tensor.
    if grad_loss is None:
        grad_loss_paddle = paddle.ones([], dtype=paddle.float32)
    else:
        grad_loss_paddle = grad_loss
        if grad_loss_paddle.dtype != paddle.float32:
            grad_loss_paddle = grad_loss_paddle.cast(paddle.float32)

    # Move every tensor over to torch via dlpack (same GPU memory).
    t_index_q = _paddle_to_torch(index_q_bf)
    t_weights = _paddle_to_torch(weights_bf)
    t_index_k = _paddle_to_torch(index_k_bf)
    t_attn = _paddle_to_torch(target_buf)
    t_index_score = _paddle_to_torch(predict_buf)
    t_topk = _paddle_to_torch(topk_indices)
    t_grad_loss = _paddle_to_torch(grad_loss_paddle)

    # Make sure cuDNN executes on Paddle's current CUDA stream so that the
    # output tensors are visible to subsequent paddle ops without an
    # explicit synchronize.
    cur_stream_handle = paddle.device.current_stream().stream_base.cuda_stream
    torch_stream = torch.cuda.ExternalStream(cur_stream_handle)
    with torch.cuda.stream(torch_stream):
        out = indexer_backward_wrapper(
            t_index_q,
            t_weights,
            t_index_k,
            t_attn,
            t_index_score,
            t_topk,
            sm_scale=float(index_q_bf.shape[-1]) ** -0.5,
            loss_coeff=float(loss_coeff),
            grad_loss=t_grad_loss,
            block_I=int(block_I),
        )

    grad_q = _torch_to_paddle(out["d_index_q"])
    grad_weights = _torch_to_paddle(out["d_weights"])
    grad_k = _torch_to_paddle(out["d_index_k"])

    if grad_q.dtype != orig_q_dtype:
        grad_q = grad_q.cast(orig_q_dtype)
    if grad_weights.dtype != orig_weights_dtype:
        grad_weights = grad_weights.cast(orig_weights_dtype)
    if grad_k.dtype != orig_k_dtype:
        grad_k = grad_k.cast(orig_k_dtype)

    return grad_q, grad_weights, grad_k
