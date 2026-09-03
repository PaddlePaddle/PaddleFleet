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

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict

import numpy as np
import paddle
from paddle import Tensor
from paddle.utils import dlpack as paddle_dlpack

try:
    from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
        RecomputeStore,
    )

    HAS_RECOMPUTE_STORE = True
except ImportError:
    HAS_RECOMPUTE_STORE = False

    class RecomputeStore:
        """Disabled fallback for Paddle releases without overlap support."""

        enabled = False

        @classmethod
        def put(cls, span) -> None:
            return None

        @classmethod
        def drop(cls, span) -> None:
            return None


_PADDLE_RUNTIME_PATCHED = False


def _accuracy_compatible_enabled() -> bool:
    return os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"


def _group_indices(parameters, group_size, helper):
    indices = helper.core.eager_assign_group_by_size(
        parameters,
        [False] * len(parameters),
        [group_size, group_size],
    )
    groups = []
    for group in indices:
        pending = []
        for index in group:
            if "gptlm_head" in getattr(parameters[index], "name", ""):
                if pending:
                    groups.append(pending)
                    pending = []
                groups.append([index])
            else:
                pending.append(index)
        if pending:
            groups.append(pending)
    return groups


def _install_fusion_patch():
    from paddle.distributed.fleet.utils import tensor_fusion_helper as helper

    def assign_group_by_size(parameters, group_size=128 * 1024 * 1024):
        groups = _group_indices(parameters, group_size, helper)
        var_groups = OrderedDict()
        group_msg = []
        for group_idx, indices in enumerate(groups):
            numel = 0
            for index in indices:
                var_groups.setdefault(group_idx, []).append(parameters[index])
                numel += np.prod(parameters[index].shape)
            dtype = parameters[indices[0]].dtype
            size_bytes = numel * helper.core.size_of_dtype(dtype)
            group_msg.append(
                f"group_{group_idx}: {size_bytes / 1024**2:.4f} MB, dtype: {dtype!s}"
            )
        helper.logger.info(f"Tensor Fusion Group Info:\n{group_msg}\n")
        return var_groups

    def get_group_size(parameters, group_size=128 * 1024 * 1024):
        sizes = []
        for indices in _group_indices(parameters, group_size, helper):
            numel = sum(np.prod(parameters[index].shape) for index in indices)
            dtype = parameters[indices[0]].dtype
            size_bytes = numel * helper.core.size_of_dtype(dtype)
            sizes.append(
                size_bytes / 1024**3 * 12 / helper.core.size_of_dtype(dtype)
            )
        return sizes

    def comm_grads(self):
        if not self.need_reduce_scale_sync():
            return

        prescale_sum = (
            self._act == helper.HOOK_ACTION.REDUCE_SCATTER
            and self._use_reduce_avg
            and _accuracy_compatible_enabled()
        )
        reduce_op = (
            paddle.distributed.ReduceOp.AVG
            if self._use_reduce_avg and not prescale_sum
            else paddle.distributed.ReduceOp.SUM
        )
        if prescale_sum or (
            not self._scale_after_comm and not self._use_reduce_avg
        ):
            self.grad_storage.scale_(1.0 / self._comm_group.nranks)

        need_check = helper.strtobool(os.getenv("FLAGS_pp_check_naninf", "0"))
        if need_check:
            err_msg = helper.check_naninf(self.grad_storage)
            if err_msg is not None:
                rank = paddle.distributed.get_rank()
                raise ValueError(
                    f"{err_msg}. Tensor contains inf or nan values at rank "
                    f"{rank} before gradient communication"
                )

        if self._act == helper.HOOK_ACTION.ALL_REDUCE:
            task = paddle.distributed.all_reduce(
                self.grad_storage,
                op=reduce_op,
                group=self._comm_group,
                sync_op=False,
            )
        elif self._act == helper.HOOK_ACTION.REDUCE:
            task = paddle.distributed.reduce(
                self.grad_storage,
                dst=self._dst,
                op=reduce_op,
                group=self._comm_group,
                sync_op=False,
            )
        elif self._act == helper.HOOK_ACTION.REDUCE_SCATTER:
            if paddle.distributed.in_auto_parallel_align_mode():
                reduce_op = paddle.distributed.ReduceOp.SUM
            shard_size = self.grad_storage._numel() // self._comm_group.nranks
            begin = shard_size * max(self._comm_group.rank, 0)
            end = begin + shard_size
            reduced = (
                paddle.empty_like(self.grad_storage._slice(begin, end))
                if self._free_grads_in_comm
                else self.grad_storage._slice(begin, end)
            )
            task = paddle.distributed.reduce_scatter(
                reduced,
                self.grad_storage,
                op=reduce_op,
                group=self._comm_group,
                sync_op=False,
            )
            if self._free_grads_in_comm:
                self._reset_grad_storage(reduced)
        else:
            raise ValueError(f"Unsupported communication action: {self._act}")
        self._task = task

    helper.assign_group_by_size = assign_group_by_size
    helper.get_group_size = get_group_size
    helper.FusedCommBuffer._comm_grads = helper.imperative_base.no_grad(
        comm_grads
    )

    # These modules import the helpers by value, so update already-imported
    # references as well as the defining module.
    consumers = (
        "paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer",
        "paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer",
        "paddle.distributed.fleet.meta_parallel.pipeline_parallel",
    )
    for module_name in consumers:
        try:
            module = __import__(module_name, fromlist=["_"])
        except ImportError:
            continue
        if hasattr(module, "assign_group_by_size"):
            module.assign_group_by_size = assign_group_by_size
        if hasattr(module, "get_group_size"):
            module.get_group_size = get_group_size


def _install_sharding_shape_patch():
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
        dygraph_sharding_optimizer as sharding,
    )

    cls = sharding.DygraphShardingOptimizerV2
    original = cls._create_slice_param
    if getattr(original, "_fleet_accuracy_compatible", False):
        return

    def create_slice_param(self, param):
        slice_param = original(self, param)
        slice_param._accuracy_compatible_original_shape = tuple(param.shape)
        return slice_param

    create_slice_param._fleet_accuracy_compatible = True
    cls._create_slice_param = create_slice_param


def _install_adamw_patch():
    from paddle.optimizer import AdamW

    original = AdamW._append_optimize_op
    if getattr(original, "_fleet_accuracy_compatible", False):
        return

    def append_optimize_op(self, block, param_and_grad):
        if not _accuracy_compatible_enabled():
            return original(self, block, param_and_grad)

        param = (
            param_and_grad["params"][0]
            if isinstance(param_and_grad, dict)
            else param_and_grad[0]
        )
        original_shape = getattr(
            param,
            "_accuracy_compatible_original_shape",
            tuple(param.shape),
        )
        apply_decay = self._apply_decay_param_fun
        old_flag = os.environ.get("FLAGS_use_accuracy_compatible_kernel")
        self._apply_decay_param_fun = lambda _name: len(original_shape) != 1
        os.environ["FLAGS_use_accuracy_compatible_kernel"] = "0"
        try:
            return original(self, block, param_and_grad)
        finally:
            self._apply_decay_param_fun = apply_decay
            if old_flag is None:
                os.environ.pop("FLAGS_use_accuracy_compatible_kernel", None)
            else:
                os.environ["FLAGS_use_accuracy_compatible_kernel"] = old_flag

    append_optimize_op._fleet_accuracy_compatible = True
    AdamW._append_optimize_op = append_optimize_op


def install_accuracy_compatible_paddle_patches() -> bool:
    """Install Paddle runtime accuracy overrides once and return enablement."""

    global _PADDLE_RUNTIME_PATCHED
    if not _accuracy_compatible_enabled():
        return False
    if not _PADDLE_RUNTIME_PATCHED:
        _install_fusion_patch()
        _install_sharding_shape_patch()
        _install_adamw_patch()
        _PADDLE_RUNTIME_PATCHED = True
    return True


_MEGATRON_SITE_PACKAGES = os.environ.get("MEGATRON_SITE_PACKAGES", "")
_TORCH = None


def _import_torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    if _MEGATRON_SITE_PACKAGES and _MEGATRON_SITE_PACKAGES not in sys.path:
        sys.path.insert(0, _MEGATRON_SITE_PACKAGES)
    import torch

    _TORCH = torch
    return _TORCH


def _to_torch(tensor):
    torch = _import_torch()
    return torch.utils.dlpack.from_dlpack(
        paddle_dlpack.to_dlpack(tensor.contiguous())
    )


def _to_paddle(tensor):
    torch = _import_torch()
    return paddle_dlpack.from_dlpack(
        torch.utils.dlpack.to_dlpack(tensor.contiguous())
    )


def sum_for_small_rows(value: Tensor) -> Tensor:
    if len(value.shape) != 2 or value.shape[0] >= 16:
        return paddle.sum(value, axis=-1, keepdim=True)
    pad_rows = 16 - value.shape[0]
    if pad_rows <= 0:
        return paddle.sum(value, axis=-1, keepdim=True)
    padding = paddle.zeros([pad_rows, value.shape[1]], dtype=value.dtype)
    padded = paddle.concat([value, padding], axis=0)
    return paddle.sum(padded, axis=-1, keepdim=True)[: value.shape[0]]


class LossScaleBeforeBackward:
    _acc_steps = 1

    @classmethod
    def set_acc_steps(cls, acc_steps: int) -> None:
        cls._acc_steps = max(1, acc_steps)

    @classmethod
    def scale(cls, loss: Tensor) -> Tensor:
        if cls._acc_steps <= 1:
            return loss
        return loss / cls._acc_steps


class CompatibleCSASinkSoftmax(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, scores: Tensor, sink: Tensor):
        ctx.save_for_backward(scores, sink)
        scores_max = scores.max(axis=-1, keepdim=True)
        scores_max = paddle.maximum(scores_max, sink)
        exp_scores = paddle.exp(scores - scores_max)
        exp_sink = paddle.exp(sink - scores_max)
        sum_exp = exp_scores.sum(axis=-1, keepdim=True) + exp_sink
        return exp_scores / sum_exp

    @staticmethod
    def backward(ctx, grad_attn_weights: Tensor):
        torch = _import_torch()
        scores, sink = ctx.saved_tensor()
        scores_t = _to_torch(scores.detach())
        sink_t = _to_torch(sink.detach())
        grad_t = _to_torch(grad_attn_weights.detach())

        with torch.enable_grad():
            scores_t = scores_t.detach().requires_grad_(True)
            sink_t = sink_t.detach().requires_grad_(True)
            scores_max_t = torch.maximum(
                scores_t.max(dim=-1, keepdim=True).values, sink_t
            )
            exp_scores_t = torch.exp(scores_t - scores_max_t)
            exp_sink_t = torch.exp(sink_t - scores_max_t)
            sum_exp_t = exp_scores_t.sum(dim=-1, keepdim=True) + exp_sink_t
            attn_weights_t = exp_scores_t / sum_exp_t
            attn_weights_t.backward(grad_t)

        grad_scores = _to_paddle(scores_t.grad)
        grad_sink = _to_paddle(sink_t.grad)
        return grad_scores, grad_sink


def compatible_einsum(grad_scores: Tensor, q: Tensor) -> Tensor:
    torch = _import_torch()
    grad_scores_t = _to_torch(grad_scores)
    q_t = _to_torch(q)
    grad_k_t = torch.einsum(
        "sbht,sbhd->tbd", grad_scores_t, q_t.to(dtype=torch.float32)
    ).to(dtype=q_t.dtype)
    return _to_paddle(grad_k_t)


class CompatibleHadamard(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x: Tensor, scale: float):
        from fast_hadamard_transform import hadamard_transform

        scale = float(scale)
        ctx.scale = scale
        x_t = _to_torch(x)
        y_t = hadamard_transform(x_t, scale=scale)
        return _to_paddle(y_t)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        from fast_hadamard_transform import hadamard_transform

        grad_t = _to_torch(grad_output)
        grad_x_t = hadamard_transform(grad_t, scale=ctx.scale)
        return _to_paddle(grad_x_t)


class CompatibleOGroupProjection(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        o_local_groups: int,
        o_lora_rank: int,
        layer_number: int,
        call_idx: int,
    ):
        torch = _import_torch()

        ctx.o_local_groups = o_local_groups
        ctx.o_lora_rank = o_lora_rank
        ctx.layer_number = layer_number
        ctx.call_idx = call_idx
        ctx.save_for_backward(x, weight)

        x_seqfirst = x.detach().transpose([1, 0, 2, 3]).contiguous()
        weight_grouped = (
            weight.detach()
            .reshape([o_local_groups, o_lora_rank, -1])
            .contiguous()
        )
        x_t = _to_torch(x_seqfirst).detach()
        weight_t = _to_torch(weight_grouped).detach()
        out_t = torch.einsum("...gd,grd->...gr", x_t, weight_t)
        out = _to_paddle(out_t)
        return out.transpose([1, 0, 2, 3]).contiguous()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        torch = _import_torch()

        x, weight = ctx.saved_tensor()
        x_seqfirst = x.detach().transpose([1, 0, 2, 3]).contiguous()
        grad_seqfirst = (
            grad_output.detach().transpose([1, 0, 2, 3]).contiguous()
        )
        weight_grouped = (
            weight.detach()
            .reshape([ctx.o_local_groups, ctx.o_lora_rank, -1])
            .contiguous()
        )

        x_t = _to_torch(x_seqfirst).detach()
        grad_t = _to_torch(grad_seqfirst).detach()
        weight_t = _to_torch(weight_grouped).detach()

        with torch.enable_grad():
            x_t = x_t.requires_grad_(True)
            weight_t = weight_t.requires_grad_(True)
            out_t = torch.einsum("...gd,grd->...gr", x_t, weight_t)
            out_t.backward(grad_t)
            x_grad_t = x_t.grad.contiguous()
            w_grad_t = weight_t.grad.reshape(tuple(weight.shape)).contiguous()

        x_grad = _to_paddle(x_grad_t)
        x_grad = x_grad.transpose([1, 0, 2, 3]).contiguous()
        w_grad = _to_paddle(w_grad_t)
        w_grad_accum = w_grad.detach().cast(paddle.float32)
        prev = getattr(weight, "_dsv4_attn_o_group_seqfirst_wgrad", None)
        if prev is None:
            weight._dsv4_attn_o_group_seqfirst_wgrad = w_grad_accum
        else:
            weight._dsv4_attn_o_group_seqfirst_wgrad = prev + w_grad_accum
        return x_grad, w_grad


class CompatibleQRMSNorm(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, q: Tensor, eps: float) -> Tensor:
        r = paddle.rsqrt(q.square().mean(axis=-1, keepdim=True) + eps)
        ctx.save_for_backward(q, r)
        return q * r

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        q, r = ctx.saved_tensor()
        hidden_size = q.shape[-1]
        grad_r = (grad_output * q).sum(axis=-1, keepdim=True)
        grad_q = grad_output * r
        grad_add = grad_r * (-0.5) * (r * r * r)
        grad_q = grad_q + (paddle.full_like(q, 2.0) * q) * (
            grad_add / hidden_size
        )
        return grad_q


class CompatibleEmbeddingIndexBackward(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, masked_input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(masked_input, weight)
        return weight[masked_input]

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        torch = _import_torch()

        masked_input, weight = ctx.saved_tensor()
        ids_t = _to_torch(masked_input.detach().cast("int64")).detach()
        grad_t = _to_torch(grad_output.detach()).detach()
        weight_t = _to_torch(weight.detach()).detach()

        with torch.enable_grad():
            weight_t = weight_t.requires_grad_(True)
            output_t = weight_t[ids_t]
            output_t.backward(grad_t)
            grad_weight_t = weight_t.grad

        grad_weight = _to_paddle(grad_weight_t)
        return None, grad_weight


def te_matmul(grad_output: Tensor, weight: Tensor) -> Tensor:
    torch = _import_torch()
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

    grad_output_torch = _to_torch(grad_output)
    weight_oi_torch = _to_torch(weight.t())
    grad_input_torch, *_ = general_gemm(
        weight_oi_torch,
        grad_output_torch,
        out_dtype=torch.bfloat16,
        layout="NN",
        grad=True,
    )

    return _to_paddle(grad_input_torch)


def linear_seqfirst_wgrad(
    input: Tensor, grad_output: Tensor, weight: Tensor
) -> Tensor | None:
    if input.dim() not in (2, 3, 4) or grad_output.dim() != input.dim():
        return None
    if weight.dtype != paddle.bfloat16:
        return None

    torch = _import_torch()

    out_features = grad_output.shape[-1]
    if input.dim() == 2:
        flat_rows, hidden = input.shape
        input_seqfirst = (
            input.detach().reshape([flat_rows, hidden]).contiguous()
        )
        grad_seqfirst = (
            grad_output.detach().reshape([flat_rows, out_features]).contiguous()
        )
    elif input.dim() == 3:
        first, second, hidden = input.shape
        flat_rows = first * second
        if first < second:
            input_seqfirst = (
                input.detach()
                .reshape([first, second, hidden])
                .transpose([1, 0, 2])
                .reshape([flat_rows, hidden])
                .contiguous()
            )
            grad_seqfirst = (
                grad_output.detach()
                .reshape([first, second, out_features])
                .transpose([1, 0, 2])
                .reshape([flat_rows, out_features])
                .contiguous()
            )
        else:
            input_seqfirst = (
                input.detach().reshape([flat_rows, hidden]).contiguous()
            )
            grad_seqfirst = (
                grad_output.detach()
                .reshape([flat_rows, out_features])
                .contiguous()
            )
    else:
        first, second, streams, hidden = input.shape
        flat_rows = first * second * streams
        if first < second:
            input_seqfirst = (
                input.detach()
                .reshape([first, second, streams, hidden])
                .transpose([1, 0, 2, 3])
                .reshape([flat_rows, hidden])
                .contiguous()
            )
            grad_seqfirst = (
                grad_output.detach()
                .reshape([first, second, streams, out_features])
                .transpose([1, 0, 2, 3])
                .reshape([flat_rows, out_features])
                .contiguous()
            )
        else:
            input_seqfirst = (
                input.detach().reshape([flat_rows, hidden]).contiguous()
            )
            grad_seqfirst = (
                grad_output.detach()
                .reshape([flat_rows, out_features])
                .contiguous()
            )

    input_t = _to_torch(input_seqfirst).detach()
    grad_t = _to_torch(grad_seqfirst).detach()
    weight_oi_t = _to_torch(weight.t()).detach()

    with torch.enable_grad():
        input_t = input_t.requires_grad_(True)
        weight_oi_t = weight_oi_t.requires_grad_(True)
        output_t = torch.matmul(input_t, weight_oi_t.t())
        output_t.backward(grad_t)
        grad_weight_t = weight_oi_t.grad.t().contiguous()

    return _to_paddle(grad_weight_t)


class MoEInputBranches(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x: Tensor):
        return x.clone(), x.clone(), x.clone()

    @staticmethod
    def backward(
        ctx, routed_grad: Tensor, router_grad: Tensor, shared_grad: Tensor
    ):
        return (routed_grad + router_grad) + shared_grad


def indices_to_multihot(
    indices: Tensor, probs: Tensor, num_local_experts: int
) -> tuple[Tensor, Tensor]:
    mask = indices != -1
    safe_indices = paddle.where(mask, indices, paddle.zeros_like(indices))
    one_hot = paddle.nn.functional.one_hot(
        safe_indices.astype("int64"),
        num_classes=num_local_experts,
    )
    valid = mask.astype(one_hot.dtype).unsqueeze(-1)
    one_hot = one_hot * valid
    multihot_routing_map = paddle.sum(one_hot, axis=1)
    multihot_probs = paddle.sum(
        one_hot.astype(probs.dtype) * probs.unsqueeze(-1),
        axis=1,
    ).astype("float32")
    return multihot_routing_map.cast(paddle.bool), multihot_probs


def _accumulate_hc_mapping_seqfirst_wgrad(
    weight: Tensor, w_grad: Tensor
) -> None:
    if w_grad is None:
        return
    w_grad = w_grad.detach().cast(paddle.float32)
    prev = getattr(weight, "_dsv4_hc_mapping_seqfirst_wgrad", None)
    if prev is None:
        weight._dsv4_hc_mapping_seqfirst_wgrad = w_grad
    else:
        weight._dsv4_hc_mapping_seqfirst_wgrad = prev + w_grad


def _hc_mapping_wgrad(
    x_seqfirst: Tensor,
    grad_seqfirst: Tensor,
    weight: Tensor,
) -> Tensor:
    torch = _import_torch()

    x_t = _to_torch(x_seqfirst).detach().requires_grad_(True)
    grad_t = _to_torch(grad_seqfirst).detach()
    weight_oi_t = _to_torch(weight.t()).detach().requires_grad_(True)
    with torch.enable_grad():
        proj_t = torch.matmul(x_t, weight_oi_t.t())
        proj_t.backward(grad_t)
    return _to_paddle(weight_oi_t.grad.t().detach())


def _register_hc_mapping_seqfirst_wgrad_hook(
    proj_2d: Tensor, x: Tensor, weight: Tensor
) -> None:
    if len(x.shape) != 3 or x.shape[0] <= 1 or proj_2d.stop_gradient:
        return

    batch_size, seq_len, hidden_width = x.shape

    def _seqfirst_wgrad_hook(grad: Tensor) -> None:
        with paddle.no_grad():
            out_width = grad.shape[-1]
            x_seqfirst = (
                x.detach()
                .reshape([batch_size, seq_len, hidden_width])
                .transpose([1, 0, 2])
                .reshape([-1, hidden_width])
            )
            grad_seqfirst = (
                grad.detach()
                .reshape([batch_size, seq_len, out_width])
                .transpose([1, 0, 2])
                .reshape([-1, out_width])
            )
            w_grad = _hc_mapping_wgrad(
                x_seqfirst,
                grad_seqfirst,
                weight,
            )
            _accumulate_hc_mapping_seqfirst_wgrad(weight, w_grad)

    proj_2d.register_hook(_seqfirst_wgrad_hook)


class TorchOrderRmsScale(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x: Tensor, eps: float):
        nC = x.shape[-1]
        norm = x.norm(axis=-1, keepdim=True)
        r = 1.0 / (norm / math.sqrt(nC) + eps)
        r = r.astype(x.dtype)
        ctx.nC = nC
        ctx.eps = eps
        ctx.save_for_backward(x, norm, r)
        return r

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, _, _ = ctx.saved_tensor()
        grad = grad.astype(x.dtype)
        torch = _import_torch()

        x_t = _to_torch(x).detach().requires_grad_(True)
        grad_t = _to_torch(grad).detach()
        with torch.enable_grad():
            norm_t = x_t.norm(dim=-1, keepdim=True)
            r_t = 1.0 / (norm_t / math.sqrt(ctx.nC) + ctx.eps)
            r_t.backward(grad_t)
        return _to_paddle(x_t.grad.detach()).astype(x.dtype)


def compatible_projection_and_norm(
    x: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    nC = x.shape[-1]
    x_2d = x.reshape([-1, nC])
    r = TorchOrderRmsScale.apply(x_2d, eps)
    # Match Megatron clean path: torch.matmul(x, weight.t()). Paddle
    # nn.Linear uses a different BF16 cuBLAS path for this shape and drifts
    # before the first HC BDA.
    proj_2d = paddle.matmul(x_2d, weight.t(), transpose_y=True)
    _register_hc_mapping_seqfirst_wgrad_hook(proj_2d, x, weight)
    proj = proj_2d.reshape([*x.shape[:-1], weight.shape[-1]])
    r = r.reshape([*x.shape[:-1], 1])
    return proj, r


def compatible_sinkhorn_backward(
    logits: Tensor, grad_output: Tensor, num_iterations: int, eps: float
) -> Tensor:
    torch = _import_torch()

    logits_t = _to_torch(logits).to(torch.bfloat16)
    grad_t = _to_torch(grad_output).to(torch.bfloat16)
    with torch.enable_grad():
        logits_t = logits_t.detach().requires_grad_(True)
        row_max = logits_t.max(dim=-1, keepdim=True).values
        m = torch.exp(logits_t - row_max)
        for _ in range(num_iterations):
            m = m / m.sum(dim=-1, keepdim=True).clamp(min=eps)
            m = m / m.sum(dim=-2, keepdim=True).clamp(min=eps)
        m.backward(grad_t)
    grad_input_t = logits_t.grad.contiguous()
    return _to_paddle(grad_input_t)


class CompatibleLearnedOutputContract(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        head_fn: Tensor,
        base: Tensor,
        scale: Tensor,
        n: int,
        eps: float,
        out_dtype,
    ):
        torch = _import_torch()
        import torch.nn.functional as torch_F

        head_fn_out_in = head_fn.transpose([1, 0]).contiguous()
        hidden_t = _to_torch(hidden_states)
        head_t = _to_torch(head_fn_out_in)
        base_t = _to_torch(base)
        scale_t = _to_torch(scale)
        torch_dtype = (
            torch.bfloat16
            if str(out_dtype) == "paddle.bfloat16"
            else hidden_t.dtype
        )
        with torch.no_grad():
            rsqrt_t = torch.rsqrt(
                hidden_t.square().mean(-1, keepdim=True) + eps
            )
            proj_t = torch_F.linear(hidden_t, head_t)
            mixes_t = proj_t * rsqrt_t
            pre_arg_t = mixes_t * scale_t + base_t
            sig_t = torch.sigmoid(pre_arg_t)
            pre_t = sig_t + eps
            y_t = torch.sum(
                pre_t.unsqueeze(-1)
                * hidden_t.reshape(*hidden_t.shape[:-1], n, -1),
                dim=-2,
            )
            out_t = y_t.to(torch_dtype)
        out = _to_paddle(out_t)

        ctx.save_for_backward(
            hidden_states,
            head_fn,
            base,
            scale,
        )
        ctx.n = n
        ctx.eps = eps
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            hidden_input,
            head_fn_input,
            base_input,
            scale_input,
        ) = ctx.saved_tensor()
        n = ctx.n
        eps = ctx.eps
        torch = _import_torch()
        import torch.nn.functional as torch_F

        hidden_input_t = _to_torch(hidden_input).detach().requires_grad_(True)
        head_input_t = _to_torch(head_fn_input).detach().requires_grad_(True)
        base_input_t = _to_torch(base_input).detach().requires_grad_(True)
        scale_input_t = _to_torch(scale_input).detach().requires_grad_(True)
        grad_t = _to_torch(grad_output).detach()
        torch_dtype = (
            torch.bfloat16
            if str(ctx.out_dtype) == "paddle.bfloat16"
            else hidden_input_t.dtype
        )

        with torch.enable_grad():
            hidden_t = hidden_input_t.to(torch.float32)
            head_io_t = head_input_t.to(torch.float32)
            base_t = base_input_t.to(torch.float32)
            scale_t = scale_input_t.to(torch.float32)
            head_oi_t = head_io_t.transpose(0, 1).contiguous()
            rsqrt_t = torch.rsqrt(
                hidden_t.square().mean(-1, keepdim=True) + eps
            )
            mixes_t = torch_F.linear(hidden_t, head_oi_t) * rsqrt_t
            pre_t = torch.sigmoid(mixes_t * scale_t + base_t) + eps
            y_t = torch.sum(
                pre_t.unsqueeze(-1)
                * hidden_t.reshape(*hidden_t.shape[:-1], n, -1),
                dim=-2,
            )
            out_t = y_t.to(torch_dtype)
            out_t.backward(grad_t)

        head_grad_t = head_input_t.grad
        if hidden_input_t.ndim == 3 and hidden_input_t.shape[0] > 1:
            hidden_seq_input_t = (
                hidden_input_t.detach()
                .transpose(0, 1)
                .contiguous()
                .requires_grad_(True)
            )
            grad_seq_t = grad_t.transpose(0, 1).contiguous()
            head_seq_t = head_input_t.detach().requires_grad_(True)
            base_seq_t = base_input_t.detach()
            scale_seq_t = scale_input_t.detach()
            with torch.enable_grad():
                hidden_seq_t = hidden_seq_input_t.to(torch.float32)
                head_seq_oi_t = (
                    head_seq_t.to(torch.float32).transpose(0, 1).contiguous()
                )
                base_seq_f32_t = base_seq_t.to(torch.float32)
                scale_seq_f32_t = scale_seq_t.to(torch.float32)
                rsqrt_seq_t = torch.rsqrt(
                    hidden_seq_t.square().mean(-1, keepdim=True) + eps
                )
                mixes_seq_t = (
                    torch_F.linear(hidden_seq_t, head_seq_oi_t) * rsqrt_seq_t
                )
                pre_seq_t = (
                    torch.sigmoid(
                        mixes_seq_t * scale_seq_f32_t + base_seq_f32_t
                    )
                    + eps
                )
                y_seq_t = torch.sum(
                    pre_seq_t.unsqueeze(-1)
                    * hidden_seq_t.reshape(*hidden_seq_t.shape[:-1], n, -1),
                    dim=-2,
                )
                out_seq_t = y_seq_t.to(torch_dtype)
                out_seq_t.backward(grad_seq_t)
            head_grad_t = head_seq_t.grad

        return (
            _to_paddle(hidden_input_t.grad.detach()),
            _to_paddle(head_grad_t.detach()),
            _to_paddle(base_input_t.grad.detach()),
            _to_paddle(scale_input_t.grad.detach()),
        )
