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

Calls ``paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api
.indexer_backward_wrapper`` directly on Paddle tensors. The wrapper performs
the score-gradient precompute kernel internally, matching Megatron's
``FusedIndexerSparseAttnFunc`` data flow, so this entry point exposes the
*raw* (target, predict) pair instead of pre-computed ``grad_scores``.

Returns d_index_q / d_weights / d_index_k_comp matching input dtypes/shapes.
"""

from __future__ import annotations

import functools
import inspect
import logging

import paddle
from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available

logger = logging.getLogger(__name__)

# cuDNN 1.28 added a second GEMM stage for this kernel, selected by
# ``backend="sm100_v2"``. It keeps kernel 1 (the score-grad precompute)
# verbatim and replaces the GEMM: ``weights`` are upcast to fp32 exactly and the
# per-slot gradient matrix ``A = grad_signal * weights`` is split into a
# two-term bf16 expansion before the MMA, with a deterministic in-CTA
# ``d_weights`` reduction. Measured on an SM103 part at ``S=8192 H=64 D=128 topk=2048``,
# relative L2 against an fp64 evaluation of the kernel's documented contract:
# ``d_index_q`` 2.32e-03 -> 1.66e-03, ``d_index_k`` 1.52e-03 -> 1.21e-05, and
# ``d_weights`` 1.66e-03 -> 1.70e-07 once it is handed an fp32 buffer (below).
# Nothing regresses, so this is selected automatically rather than exposed as a
# switch.
#
# The cost is speed at the top of the topk range: the two-term expansion doubles
# the GEMM work, so it wins where the GEMM does not dominate and loses where it
# does. Measured on an idle SM103 part at the shape above, min of 3 alternating rounds
# of 20 iterations (spread within a round < 1.5%):
#
#     topk    default    v2       speedup
#      128     0.457     0.293     1.559x
#      256     0.643     0.537     1.197x
#      512     0.909     0.879     1.035x
#     1024     1.550     1.579     0.982x
#     2048     2.910     2.984     0.975x
#
# The crossover sits between 512 and 1024, so production (topk=2048) pays about
# 2.5% on this kernel. That is deliberately *not* gated on topk: two orders of
# magnitude on ``d_index_k`` is worth 2.5% of one backward kernel, and a
# topk-dependent backend would make the numerics a function of the sequence
# budget, which is worse to reason about than a uniform 2.5%.
_V2_BACKEND = "sm100_v2"

# The two refusals cuDNN raises for this backend when the device is not one it
# accepts. Matched by message because cuDNN has no dedicated exception type for
# them: ``check_support`` raises ``RuntimeError("backend='sm100_v2' requires an
# SM100 device ...")`` and the kernel factory raises
# ``RuntimeError("indexer_backward_v2_sm100 requires SM100 ...")``. Both fire
# strictly before ``execute`` -- ``check_support`` at plan construction and the
# factory from ``compile`` (api.py:475, inside ``compile`` at :457, while
# ``execute`` starts at :534) -- so the score buffers are still untouched when
# either is seen and retrying on the generic backend is exact, not approximate.
#
# The match is deliberately narrow. Catching ``RuntimeError`` broadly would also
# swallow a compile or launch failure raised *after* kernel 1 had already
# overwritten ``attn_score`` with the grad signal, and the retry would then
# compute gradients from a consumed buffer -- silently wrong, and with the real
# error permanently downgraded to a fallback.
_V2_REFUSAL_MARKERS = (
    "backend='sm100_v2' requires an SM100",
    "indexer_backward_v2_sm100 requires SM100",
)

# Latched once cuDNN has refused, so the refusal costs one plan construction per
# process rather than one per step.
_V2_REFUSED = False


def _is_v2_device_refusal(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and any(
        marker in str(exc) for marker in _V2_REFUSAL_MARKERS
    )


def _latch_v2_refused() -> None:
    global _V2_REFUSED
    _V2_REFUSED = True


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


@functools.cache
def _select_backend(
    heads: int, head_dim: int, topk: int, block_I: int
) -> str | None:
    """The cuDNN backend to request, or ``None`` for the architecture-generic one.

    Every condition here is one cuDNN enforces itself, and it enforces them by
    raising: the backend is **request-or-fail**, so asking outside the envelope
    aborts the step instead of selecting a slower kernel. The predicate is
    therefore evaluated up front rather than caught, and it has to track the
    *vendored* cuDNN -- which is why this commit moves the submodule and this
    check together. The vendored version gates the device on the SM100 family
    (``capability[0] == 10``), so SM100 and SM103 both qualify; an older
    vendored cuDNN gated on an exact ``(10, 0)``, hence the pairing.

    Cached because it would otherwise re-run on every indexer backward of every
    layer.
    """
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api import (
        indexer_backward_wrapper,
    )

    # An installed ``paddlefleet_ops`` wheel can predate the vendored cuDNN bump
    # that introduced the parameter, in which case passing it would be a
    # TypeError rather than a slower kernel.
    if "backend" not in inspect.signature(indexer_backward_wrapper).parameters:
        return None
    if paddle.device.cuda.get_device_capability()[0] != 10:
        return None
    if heads != 64 or head_dim != 128:
        return None
    if block_I != 128:
        return None
    if topk % 128 != 0 or not (128 <= topk <= 2048):
        return None
    return _V2_BACKEND


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
    orig_weights_dtype = weights.dtype
    orig_q_dtype = index_q.dtype
    orig_k_dtype = index_k_comp.dtype

    index_q_bf = _to_bf16(index_q)
    weights_bf = _to_bf16(weights)
    index_k_bf = _to_bf16(index_k_comp)

    # cuDNN overwrites attn_score (target) and index_score (topk_probs)
    # in-place during the score-grad precompute. Clone so the saved
    # forward tensors are untouched, casting to fp32 in one step.
    target_buf = (
        target.cast(paddle.float32)
        if target.dtype != paddle.float32
        else target.clone()
    )
    predict_buf = (
        topk_probs.cast(paddle.float32)
        if topk_probs.dtype != paddle.float32
        else topk_probs.clone()
    )
    if topk_indices.dtype != paddle.int32:
        topk_indices = topk_indices.cast(paddle.int32)

    # grad_loss as a 0-D fp32 tensor.
    if grad_loss is None:
        grad_loss_paddle = paddle.ones([], dtype=paddle.float32)
    else:
        grad_loss_paddle = grad_loss
        if grad_loss_paddle.dtype != paddle.float32:
            grad_loss_paddle = grad_loss_paddle.cast(paddle.float32)

    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api import (
        indexer_backward_wrapper,
    )

    def _call(backend):
        extra: dict = {}
        if backend is not None:
            extra["backend"] = backend
            # The accumulator is fp32 on both backends, so a bf16 buffer would
            # round the whole accuracy gain back to the bf16 representation
            # floor -- with the buffer the wrapper allocates by default,
            # ``d_weights`` measured 1.66e-03, identical to the generic backend.
            # fp32 is only accepted on this backend; the generic one takes bf16
            # alone. The trailing casts restore the caller's dtypes either way,
            # so the buffer dtype stays an internal detail.
            extra["d_weights"] = paddle.empty(
                weights_bf.shape, dtype=paddle.float32
            )
            # This backend owns a per-plan workspace (a self-resetting ticket
            # counter), so two executions of one plan must never be in flight on
            # the device at once. That holds today because every launch lands on
            # the calculation stream in issue order -- including under
            # ``dsa_indexer_loss_bwd_p2p_overlap``, which defers the launch on
            # the host and owns no side stream. Wrapping this call in a side
            # stream or a CUDA-graph capture would break it *silently*: the
            # ticket counter races and the gradients come out wrong with nothing
            # raised.
        return indexer_backward_wrapper(
            index_q_bf,
            weights_bf,
            index_k_bf,
            target_buf,
            predict_buf,
            topk_indices,
            sm_scale=float(index_q_bf.shape[-1]) ** -0.5,
            loss_coeff=float(loss_coeff),
            grad_loss=grad_loss_paddle,
            block_I=int(block_I),
            # dK is reduced with fp32 atomics whatever the buffer dtype, so the
            # dtype only selects *how* the result lands. Left to allocate its own
            # bf16 buffer, the wrapper takes a branch ending in
            # ``dIndexK.copy_(dIndexK_f32.astype(...))``, and Paddle's blocking
            # ``copy_`` lowers to ``GpuMemcpySync`` on the CUDA *legacy default*
            # stream -- a bidirectional full-device barrier, so every later
            # kernel waits for whatever is in flight. Under
            # ``dsa_indexer_loss_bwd_p2p_overlap`` that pushed the whole indexer
            # projection backward out past the pipeline send/recv it was supposed
            # to hide behind.
            #
            # A pre-zeroed fp32 buffer takes the wrapper's in-place path instead:
            # same kernel, same fp32 atomicAdd, same single fp32 -> bf16
            # rounding, except the rounding is now the asynchronous
            # ``grad_k.cast`` below and there is no barrier. ``zeros`` rather
            # than ``empty`` is load-bearing -- the kernel only accumulates, and
            # on this sparse path nothing else zeroes the buffer. Same reason as
            # the dense KL path (``dense_indexer_kl_cudnn.py``).
            d_index_k=paddle.zeros(index_k_bf.shape, dtype=paddle.float32),
            **extra,
        )

    backend = (
        None
        if _V2_REFUSED
        else _select_backend(
            int(index_q_bf.shape[-2]),
            int(index_q_bf.shape[-1]),
            int(topk_indices.shape[-1]),
            int(block_I),
        )
    )
    if backend is None:
        out = _call(None)
    else:
        try:
            out = _call(backend)
        except RuntimeError as exc:
            if not _is_v2_device_refusal(exc):
                raise
            # The installed ``paddlefleet_ops`` can be older than the vendored
            # cuDNN this checkout points at -- a stale wheel is the normal state
            # between a submodule bump and a rebuild -- and then cuDNN's own gate
            # is narrower than the one above. Fall back rather than abort the
            # step; the gradients stay correct, only less accurate.
            _latch_v2_refused()
            logger.warning(
                "cuDNN refused backend=%r for the sparse indexer backward "
                "(%s); using the architecture-generic backend for the rest of "
                "this process. Rebuild paddlefleet_ops against the vendored "
                "cuDNN to get the higher-precision gradients.",
                backend,
                exc,
            )
            out = _call(None)

    grad_q = out["d_index_q"]
    grad_weights = out["d_weights"]
    grad_k = out["d_index_k"]

    if grad_q.dtype != orig_q_dtype:
        grad_q = grad_q.cast(orig_q_dtype)
    if grad_weights.dtype != orig_weights_dtype:
        grad_weights = grad_weights.cast(orig_weights_dtype)
    if grad_k.dtype != orig_k_dtype:
        grad_k = grad_k.cast(orig_k_dtype)

    return grad_q, grad_weights, grad_k
