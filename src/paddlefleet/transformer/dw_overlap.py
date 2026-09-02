# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Deferred weight-gradient (dW) computation, used to cover pp p2p transfers.

The pp scheduler drains ``WeightGradStore`` inside the window where a stage is
waiting on p2p, so any dW pushed here runs for free as long as the window is
wide enough. Which projections push is chosen per model by
``config.p2p_overlap_dw_calc``; see ``P2P_OVERLAP_DW_CALC_CHOICES``.
"""

from __future__ import annotations

from functools import partial

import paddle
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)

from paddlefleet.transformer.transformer_config import (
    dw_overlap_enabled,
    dw_overlap_scheduler_supported,
)

__all__ = [
    "DeferredWeightGradLinear",
    "deferrable_linear",
    "deferrable_linear_bare",
    "deferred_grouped_dw_accumulator",
    "install_sonic_moe_dw_deferral",
]

SONIC_MOE_DW_POINTS = (
    "moe_sonic_expert_up_gate_proj",
    "moe_sonic_expert_down_proj",
)


def _accumulate_into_main_grad(param, dw):
    """Add `dw` to `param.main_grad`, creating it if the first backward."""
    if not hasattr(param, "main_grad"):
        raise AssertionError(
            "deferred weight grad needs the main_grad attribute; "
            "enable amp_master_grad"
        )
    if param.main_grad is None:
        param.main_grad = paddle.zeros(param.shape, dtype=paddle.float32)
    param.main_grad.add_(dw.reshape(param.shape).cast(paddle.float32))
    if hasattr(param, "_apply_backward_hook"):
        param._apply_backward_hook()


def deferred_grouped_dw_accumulator(config, point, param):
    """Accumulator for ``fused_grouped_matmul``'s dw, deferred to a p2p window.

    Returns None when `point` is not selected, which makes the grouped matmul
    keep its normal inline behaviour. Otherwise returns a callable that takes the
    thunk producing dw and queues it on WeightGradStore.

    The grouped matmul takes a 3-D ``[G, R, D]`` *view* of the weight, which is
    not the Parameter and carries no ``main_grad``; that is why the real 2-D
    `param` has to be passed in separately and written directly here, exactly as
    DeferredWeightGradLinear does.

    Cost: the queued thunk pins the ``x`` and ``dy`` it closed over until the
    scheduler pops it -- for the dsv4 o-group projection that is
    ``[M, G, D]`` + ``[M, G, R]`` bf16 per layer per in-flight microbatch.
    """
    if not dw_overlap_enabled(config, point):
        return None

    def _accumulate(compute_dw):
        WeightGradStore.enabled = True
        WeightGradStore.put(
            lambda: _accumulate_into_main_grad(param, compute_dw())
        )
        WeightGradStore.enabled = False

    return _accumulate


def _run_sonic_moe_wgrad(weight, run_wgrad):
    run_wgrad()
    # SonicMoE skips its own _apply_weight_backward_hook for a deferred GEMM,
    # because main_grad only becomes complete here. Sharding uses that hook to
    # start reducing the grad, so firing it earlier would reduce a partial sum.
    if hasattr(weight, "_apply_backward_hook") and not weight.stop_gradient:
        weight._apply_backward_hook()


def _defer_sonic_moe_wgrad(points, point, weight, run_wgrad):
    if point not in points:
        return False
    WeightGradStore.enabled = True
    WeightGradStore.put(partial(_run_sonic_moe_wgrad, weight, run_wgrad))
    WeightGradStore.enabled = False
    return True


def install_sonic_moe_dw_deferral(config):
    """Route SonicMoE's routed-expert wgrad GEMMs through WeightGradStore.

    SonicMoE already splits each expert backward into a dx GEMM and a dW GEMM
    that read disjoint inputs, and accumulates dW straight into ``main_grad``,
    so deferring only means holding on to the dW call. It offers the hook we
    install here; ``run_wgrad`` closes over the column-major fp8 activations,
    which is what makes these two points expensive in memory.

    The hook is process-global, matching WeightGradStore, so this is idempotent
    and can be called from every MoE layer's constructor.
    """
    points = frozenset(
        p for p in SONIC_MOE_DW_POINTS if dw_overlap_enabled(config, p)
    )
    if not points or not dw_overlap_scheduler_supported(config):
        return

    try:
        from paddlefleet_ops.sonicmoe.functional import (
            set_wgrad_deferral_hook,
        )
    except ImportError as exc:
        raise ImportError(
            f"p2p_overlap_dw_calc selects {sorted(points)}, which needs "
            "sonicmoe.functional.set_wgrad_deferral_hook; this sonicmoe build "
            "does not have it"
        ) from exc

    set_wgrad_deferral_hook(partial(_defer_sonic_moe_wgrad, points))


class DeferredWeightGradLinear(paddle.autograd.PyLayer):
    """``y = x @ weight`` with the weight grad deferred to WeightGradStore.

    Stands in for a bias-free Column/RowParallelLinear at ``mp == 1``: dx is
    computed eagerly so the rest of the backward can proceed, while dW is queued
    for the pp scheduler to run inside a p2p window. Bit-exact with
    ``F.linear(x, weight)`` for arbitrary leading batch dims.
    """

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        # Bit-exact with Column/RowParallelLinear at mp == 1, no bias:
        # F.linear(x, weight) = x @ weight, weight shape: [in, out]
        return paddle.nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, out_grad):
        x, weight = ctx.saved_tensor()

        def _compute_weight_grad(x, out_grad, weight):
            with paddle.amp.auto_cast(False):
                # Flatten leading batch dims so dw = x_2d.T @ og_2d has
                # shape [in, out] == weight.shape
                x_2d = x.reshape([-1, x.shape[-1]])
                og_2d = out_grad.reshape([-1, out_grad.shape[-1]])
                w_grad = paddle.matmul(x_2d, og_2d, transpose_x=True)

            if not hasattr(weight, "main_grad"):
                raise AssertionError(
                    "deferred weight grad needs the main_grad attribute; "
                    "enable amp_master_grad"
                )
            if weight.main_grad is None:
                weight.main_grad = paddle.zeros(
                    weight.shape, dtype=paddle.float32
                )
            weight.main_grad.add_(w_grad)

            if hasattr(weight, "_apply_backward_hook"):
                weight._apply_backward_hook()

        dx = paddle.matmul(out_grad, weight, transpose_y=True)

        if not weight.stop_gradient:
            WeightGradStore.enabled = True
            WeightGradStore.put(
                partial(
                    _compute_weight_grad,
                    x.detach(),
                    out_grad.detach(),
                    weight,
                )
            )
            WeightGradStore.enabled = False

        return dx, None


def _can_defer(config, point, layer):
    """Whether `layer`'s dW may be deferred for `point`.

    `point` may be None, which simply means "this call site has no deferral
    point", so callers can pass a class attribute without branching. The
    WeightGradStore consumer is only installed by the interleaved PP
    scheduler, so ordinary PP (including PP=1) must stay inline.
    """
    return (
        point is not None
        and dw_overlap_enabled(config, point)
        and getattr(config, "tensor_model_parallel_size", 1) == 1
        and dw_overlap_scheduler_supported(config)
        and not getattr(config, "use_bias", False)
        and getattr(layer, "bias", None) is None
    )


def deferrable_linear(config, point, layer, x):
    """Call ``layer`` on ``x``, deferring its dW when ``point`` is selected.

    For projection layers that return an ``(output, bias)`` pair. Falls back to
    calling the layer unchanged whenever deferral cannot apply: the point is not
    selected, tensor parallel is on (the stand-in has no collective), or the
    layer carries a bias.
    """
    if _can_defer(config, point, layer):
        return DeferredWeightGradLinear.apply(x, layer.weight), None
    return layer(x)


def deferrable_linear_bare(config, point, layer, x):
    """Same as ``deferrable_linear`` for layers that return a bare tensor.

    ``paddle.nn.Linear`` does not follow the ``(out, bias)`` convention that the
    Fleet projection layers use.
    """
    if _can_defer(config, point, layer):
        return DeferredWeightGradLinear.apply(x, layer.weight)
    return layer(x)
