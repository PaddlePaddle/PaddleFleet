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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed.communication.reduce_scatter import _reduce_scatter_base
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)

from ..parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# from ..dist_checkpointing.mapping import ShardedStateDict
# from ..transformer.utils import make_sharded_tensors_for_checkpoint
from ..utils import (
    divide,
    get_pg_rank,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
    prepare_input_tensors_for_wgrad_compute,
)
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from .utils import VocabUtility

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_deep_gemm_available = True
try:
    from paddlefleet_ops import deep_gemm
except (ImportError, RuntimeError):
    _deep_gemm_available = False
    deep_gemm = None

# ---- FP8 shadow debug switch --------------------------------------------
# Enable via env: PADDLEFLEET_FP8_SHADOW_DEBUG=1
# For every fp8 GEMM inside general_gemm, also run a bf16 reference and log
# element-wise diff stats. Zero overhead when the env var is unset.
_FP8_SHADOW_DEBUG = os.environ.get("PADDLEFLEET_FP8_SHADOW_DEBUG", "0") == "1"


def _fp8_shadow_log(fp8_out, bf16_ref, pass_name, a_shape, b_shape):
    """Log DeepGEMM-style diff between fp8 GEMM output and a bf16 reference.

    Primary metric follows DeepGEMM's calc_diff:
        diff = ||x - y||^2 / (||x||^2 + ||y||^2)
    which is a normalized MSE (sensitive to both direction and magnitude).
    Empirical thresholds for fp8_gemm_nt (blockwise E4M3):
        diff < 1e-3  -> healthy
        diff < 1e-4  -> high precision
        diff > 5e-3  -> suspicious (check scale/transpose/layout)

    Secondary: Range-Relative RMSE (TE tests/pytorch/utils.py:238):
        rmse       = sqrt(mean((a-b)^2))
        rmse_range = max(a,b) - min(a,b)   (union span)
        rrmse      = rmse / rmse_range

    Tertiary: Standard relative RMSE (comparable with FP8 unit roundoff):
        rrmse_std  = rmse / sqrt(mean(ref^2))
    For FP8 E4M3 GEMM, theoretical bound rrmse_std ~ sqrt(2/3) * 2^-4 ~= 5.1%.

    cos_sim and max_abs are kept as auxiliary signals.
    """
    with paddle.no_grad():
        # Use float64 (as DeepGEMM does) to avoid fp32 accumulation error.
        out_f = fp8_out.detach().astype("float64").reshape([-1])
        ref_f = bf16_ref.detach().astype("float64").reshape([-1])
        denom = (out_f * out_f + ref_f * ref_f).sum()
        if float(denom) == 0.0:
            diff = 0.0
        else:
            sim = 2.0 * (out_f * ref_f).sum() / denom
            diff = float(1.0 - sim)
        abs_diff = (out_f - ref_f).abs()
        max_abs = float(abs_diff.max())
        norm_out = float((out_f * out_f).sum().sqrt())
        norm_ref = float((ref_f * ref_f).sum().sqrt())
        dot = float((out_f * ref_f).sum())
        cos_sim = dot / (norm_out * norm_ref + 1e-12)
        # Range-Relative RMSE
        rmse = float((out_f - ref_f).square().mean().sqrt())
        _hi = float(paddle.maximum(out_f.max(), ref_f.max()))
        _lo = float(paddle.minimum(out_f.min(), ref_f.min()))
        _rng = _hi - _lo
        rrmse = rmse / _rng if _rng > 0.0 else 0.0
        # Standard relative RMSE: rmse / sqrt(mean(ref^2))
        _ref_rms = float(ref_f.square().mean().sqrt())
        rrmse_std = rmse / _ref_rms if _ref_rms > 0.0 else 0.0
    print(
        f"[FP8_SHADOW][{pass_name}] "
        f"A={a_shape} B={b_shape} out={list(fp8_out.shape)} "
        f"diff={diff:.4e} max_abs={max_abs:.4e} cos_sim={cos_sim:.6f} "
        f"rmse={rmse:.4e} rrmse={rrmse:.4e} rrmse_std={rrmse_std:.4e}",
        flush=True,
    )


HAVE_TE = False

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..transformer.transformer_config import TransformerConfig

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def param_is_not_tensor_parallel_duplicate(param):
    """Returns true if the passed-in parameter is not a duplicate parameter
    on another TP rank."""
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    """Sets tp attributes to tensor"""
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    tensor.tensor_model_parallel = is_parallel
    tensor.partition_dim = dim
    tensor.partition_stride = stride


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    """Set default model parallel attributes if not set explicitly already."""

    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    """Copy model parallel attributes from one tensor to another."""

    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(
                destination_tensor, attribute, getattr(source_tensor, attribute)
            )

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, is_expert=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if not is_expert:
        if dist.get_world_size() <= 1:
            init_method(weight)
        else:
            with get_cuda_rng_tracker().fork():
                init_method(weight)
    else:
        if dist.get_world_size() <= 1:
            init_method(weight)
        else:
            with get_cuda_rng_tracker().fork(
                get_expert_parallel_rng_tracker_name()
            ):
                init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    input_size,
    output_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=paddle.float32,
    rank=None,
    world_size=None,
    skip_set_tensor_parallel_attributes=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    if not skip_set_tensor_parallel_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    # Initialize master weight
    master_weight = paddle.empty(
        [input_size, output_size], dtype=paddle.float, requires_grad=False
    )
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)
    # Split and copy

    per_partition_per_stride_size = divide(per_partition_size, stride)
    split_num = divide(
        master_weight.shape[partition_dim], per_partition_per_stride_size
    )
    weight_list = paddle.split(
        master_weight, num_or_sections=split_num, axis=partition_dim
    )
    if rank is None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with paddle.no_grad():
        # all tensors must live on the same device
        cpu_weight = paddle.cat(my_weight_list, dim=partition_dim)
        weight.copy_(cpu_weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(paddle.nn.Layer):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from paddle.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        reduce_scatter_embeddings: Decides whether to perform ReduceScatter after embedding lookup

    Keyword Args:
        config: A fleet.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: callable,
        reduce_scatter_embeddings: bool = False,
        config: TransformerConfig,
        tp_group: paddle.distributed.ProcessGroup | None = None,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.tp_group = tp_group
        self._dtype = config.params_dtype

        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, check_initialized=False
        )

        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_pg_rank(self.tp_group),
                get_pg_size(self.tp_group),
            )
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.deterministic_mode = config.deterministic_mode
        self.config = config
        self.world_size = get_pg_size(self.tp_group)

        # Allocate weights and initialize.
        if config.use_cpu_initialization:
            self.weight = self.create_parameter(
                shape=[self.num_embeddings_per_partition, self.embedding_dim],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                    rank=get_pg_rank(self.tp_group),
                    world_size=get_pg_size(self.tp_group),
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.num_embeddings_per_partition, self.embedding_dim],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=0, stride=1
                )
        self.weight.is_distributed = True if self.world_size > 1 else False

    def forward(self, input_):
        """Forward.

        Args:
            input_ (paddle.Tensor): Input tensor.
        """
        if getattr(self.config, "gpt_model_use_experimental_version", False):
            return vocab_parallel_embedding(
                input_,
                self.weight,
                vocab_start_index=self.vocab_start_index,
                num_embeddings=self.num_embeddings,
                mp_group=self.tp_group,
            )

        if get_pg_size(self.tp_group) > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        if self.deterministic_mode:
            output_parallel = self.weight[masked_input]
        else:
            # F.embedding currently has a non-deterministic backward function
            output_parallel = F.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if get_pg_size(self.tp_group) > 1:
            output_parallel[input_mask, :] = 0.0

        if self.reduce_scatter_embeddings:
            # Data format change to avoid explicit transpose : [b s h] --> [s b h].
            # output_parallel = output_parallel.transpose(0, 1).contiguous()
            output_parallel = output_parallel.transpose([1, 0, 2]).contiguous()
            output = reduce_scatter_to_sequence_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            # Reduce across all the model parallel GPUs.
            output = reduce_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        return output

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )


class LinearWithFrozenWeight(paddle.autograd.Function):
    """Linear operator that does not calculate gradient for weight.
    This op and LinearWithGradAccumulationAndAsyncCommunication performs
    mathematically-identical forward and DGRAD.

    Conceptually this op is the same as linear with weight.requires_grad==False,
    but in experiments they are not identical mathematically."""

    @staticmethod
    def forward(ctx, input, weight, bias, allreduce_dgrad, tp_group):
        """Forward with frozen weight."""
        ctx.save_for_backward(weight, bias)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        output = paddle.matmul(input, weight)

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward with frozen weight."""
        (weight, bias) = ctx.saved_tensor()
        grad_input = grad_output.matmul(weight.t())

        if ctx.allreduce_dgrad:
            # All-reduce. Note: here async and sync are effectively the same.
            dist.all_reduce(grad_input, group=ctx.tp_group)
        if bias is None:
            return grad_input, None
        else:
            return grad_input, None, None


def linear_with_frozen_weight(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    tp_group: paddle.core.ProcessGroup | None,
    grad_output_buffer: list[paddle.Tensor] | None = None,
    wgrad_deferral_limit: None = None,
    async_grad_allreduce: bool | None = None,
    use_accuracy_compatible: bool = False,
    **kwargs,
) -> paddle.Tensor:
    """Linear layer execution with weight.requires_grad == False.

    This function handles linear layers with weight frozen (untrainable).
    In the forward, it only saves weight and does not save input activations.
    In the backward, it does not perform weight gradient calculation, or
    weight gradient allreduce.

    Args:

    input (paddle.Tensor required): input like paddle.nn.functional.linear

    weight (paddle.Tensor required): weight like paddle.nn.functional.linear

    bias (paddle.Tensor optional): bias like paddle.nn.functional.linear

    gradient_accumulation_fusion (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    allreduce_dgrad (bool, required): Do the allreduce of input gradients.
        Here, async and sync allreduce are the same. If sequence_parallel is
        True, this must be False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.

    tp_group (paddle.core.ProcessGroup): The process group to use for tensor
                                                       parallel operations.

    grad_output_buffer (List[paddle.Tensor] optional): dummy argument, used to
    keep the API unified between all forward implementation functions.

    wgrad_deferral_limit (int optional): dummy argument, used to
    keep the API unified between all forward implementation functions.


    async_grad_allreduce (bool optional): Will be removed with 0.11.0.
                                          Please use allreduce_dgrad instead.

    """

    if async_grad_allreduce is not None:
        warnings.warn(
            "async_grad_allreduce is deprecated, not in use anymore and will"
            " be fully removed with 0.11.0. Please use allreduce_dgrad instead."
        )

    assert grad_output_buffer is None, (
        "grad_output_buffer kwarg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )

    assert wgrad_deferral_limit is None, (
        "This arg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )

    tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    if sequence_parallel:
        input = gather_from_sequence_parallel_region(
            input, tensor_parallel_output_grad=True, group=tp_group
        )
    else:
        input = input

    args = [input, weight, bias, allreduce_dgrad, tp_group]

    return LinearWithFrozenWeight.apply(*args)


def general_gemm(
    a,
    b,
    bias=None,
    fp8=False,
    inp_quant_func=None,
    weight_quant_func=None,
    out=None,
    recipe=None,
    use_accuracy_compatible=False,
    pass_name="fwd",
    shadow_ref=None,
):
    """Unified GEMM interface supporting both bf16 and fp8 paths.

    Args:
        a: Left operand. Raw tensor or pre-quantized (fp8_tensor, scale) tuple.
        b: Right operand. Same convention as *a*.
        bias: Optional bias (only used in bf16 path).
        fp8: If True, use fp8 gemm via deep_gemm.fp8_gemm_nt.
        inp_quant_func: Quantization callable for input (when fp8=True).
        weight_quant_func: Quantization callable for weight (when fp8=True).
        out: Pre-allocated output buffer. If not None, accumulates into it;
             if None, creates a new buffer.
        pass_name: Debug label ("fwd" / "dgrad" / "wgrad") used in shadow logs.
        shadow_ref: Optional bf16 reference output. When provided and
            PADDLEFLEET_FP8_SHADOW_DEBUG=1, diff stats vs this ref are logged.
            If not provided, an auto-shadow is computed only when both a and b
            are raw (non-pre-quantized) tensors.

    Returns:
        output tensor, and optionally quant cache tuple when fp8=True.
    """
    if fp8:
        assert _deep_gemm_available, "FP8 GEMM requires paddlefleet_ops.deep_gemm"

        from paddlefleet.fp8.utils import is_fp8_tensor

        # fp8_quant_blockwise requires 2D input, reshape if needed
        a_orig_shape = None
        if not is_fp8_tensor(a) and a.ndim > 2:
            a_orig_shape = a.shape
            a = a.reshape([-1, a.shape[-1]])

        if is_fp8_tensor(a):
            inp_fp8, inp_scale = a
            inp_t_fp8, inp_t_scale = None, None
        else:
            quant_result = inp_quant_func(a)
            if len(quant_result) == 2:
                inp_fp8, inp_scale = quant_result
                inp_t_fp8, inp_t_scale = None, None
            else:
                inp_fp8, inp_scale, inp_t_fp8, inp_t_scale = quant_result

        if is_fp8_tensor(b):
            weight_fp8, weight_scale = b
        else:
            # weight is [K, N] but fp8_gemm_nt expects B as [N, K]
            weight_fp8, weight_scale = weight_quant_func(b.T.contiguous())

        # print(f"[general_gemm] fp8 path: "
        #       f"A shape={inp_fp8.shape} stride={inp_fp8.strides} dtype={inp_fp8.dtype} "
        #       f"B shape={weight_fp8.shape} stride={weight_fp8.strides} dtype={weight_fp8.dtype} "
        #       f"A_scale shape={inp_scale.shape} B_scale shape={weight_scale.shape} "
        #       f"out={'accumulate '+str(out.shape) if out is not None else 'new'}")

        if out is not None:
            # Accumulate into existing buffer
            deep_gemm.fp8_gemm_nt(
                (inp_fp8, inp_scale), (weight_fp8, weight_scale), out,
                c=out, recipe=recipe,
            )
        else:
            out = paddle.empty(
                [inp_fp8.shape[0], weight_fp8.shape[0]], dtype=paddle.bfloat16
            )
            deep_gemm.fp8_gemm_nt(
                (inp_fp8, inp_scale), (weight_fp8, weight_scale), out,
                recipe=recipe,
            )

        # print(f"[general_gemm] fp8 done: out shape={out.shape}")
        if a_orig_shape is not None:
            out = out.reshape(list(a_orig_shape[:-1]) + [out.shape[-1]])

        # -------- FP8 shadow bf16 diff (debug only) -----------------------
        if _FP8_SHADOW_DEBUG:
            a_is_fp8 = is_fp8_tensor(a)
            b_is_fp8 = is_fp8_tensor(b)
            a_raw = (not a_is_fp8) and isinstance(a, paddle.Tensor)
            b_raw = (not b_is_fp8) and isinstance(b, paddle.Tensor)
            print(
                f"[FP8_PATH][{pass_name}] fp8 "
                f"A_raw={a_raw} B_raw={b_raw} out_shape={list(out.shape)}",
                flush=True,
            )
            local_shadow = shadow_ref
            if local_shadow is None and a_raw and b_raw:
                with paddle.no_grad():
                    if use_accuracy_compatible:
                        local_shadow = paddle.nn.functional.linear(a, b)
                    else:
                        local_shadow = paddle.matmul(a, b)
                    if a_orig_shape is not None:
                        local_shadow = local_shadow.reshape(
                            list(a_orig_shape[:-1]) + [local_shadow.shape[-1]]
                        )
            if local_shadow is not None:
                a_shape = list(a[0].shape) if a_is_fp8 else list(a.shape)
                b_shape = list(b[0].shape) if b_is_fp8 else list(b.shape)
                _fp8_shadow_log(out, local_shadow, pass_name, a_shape, b_shape)
            else:
                print(
                    f"[FP8_SHADOW][{pass_name}] skipped "
                    f"(pre-quantized inputs, pass shadow_ref= to enable)",
                    flush=True,
                )

        return out, (inp_fp8, inp_scale, inp_t_fp8, inp_t_scale, weight_fp8, weight_scale)

    else:
        # Standard bf16/fp16 path
        if bias is not None:
            output = paddle.nn.functional.linear(a, b, bias)
        else:
            if use_accuracy_compatible:
                output = paddle.nn.functional.linear(a, b)
            else:
                output = paddle.matmul(a, b)
        if _FP8_SHADOW_DEBUG:
            print(
                f"[FP8_PATH][{pass_name}] bf16 "
                f"A={list(a.shape)} B={list(b.shape)} out={list(output.shape)}",
                flush=True,
            )
        return output, None


class LinearWithGradAccumulationAndAsyncCommunication(paddle.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        use_accuracy_compatible=False,
        fp8=False,
        fp8_wgrad=False,
        inp_quant_func=None,
        weight_quant_func=None,
        use_pow2_scale=False,
        save_original_input=False,
    ):
        """Forward."""
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None
        # When fp8 is enabled we can skip saving the bf16 activation for
        # backward and rely solely on the fp8-quantized activation stored on
        # ctx (populated below from ``fp8_meta``). ``save_original_input``
        # forces us to keep the bf16 activation (useful for debugging or when
        # the caller wants bf16 wgrad).
        skip_bf16_input_save = fp8 and not save_original_input
        if skip_bf16_input_save:
            ctx.save_for_backward(weight)
        else:
            ctx.save_for_backward(input._new_shared_tensor(), weight)
        ctx.bf16_input_saved = not skip_bf16_input_save
        # ``input.shape`` is still needed in backward for sequence-parallel
        # collectives; save it as a plain tuple so we do not retain the tensor.
        ctx.input_shape = tuple(input.shape)
        ctx.input_dtype = input.dtype
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.tp_group = tp_group
        ctx.input_stop_gradient = bool(input.stop_gradient)
        # FP8 attributes
        ctx.fp8 = fp8
        ctx.fp8_wgrad = fp8_wgrad
        ctx.inp_quant_func = inp_quant_func
        ctx.weight_quant_func = weight_quant_func
        ctx.use_pow2_scale = use_pow2_scale
        ctx.save_original_input = save_original_input

        if sequence_parallel:
            dim_size = list(input.shape)
            dim_size[0] = dim_size[0] * tp_group.world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            dist.stream.all_gather(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output, fp8_meta = general_gemm(
            total_input, weight, bias=bias,
            fp8=fp8,
            inp_quant_func=inp_quant_func,
            weight_quant_func=weight_quant_func,
            use_accuracy_compatible=use_accuracy_compatible,
            pass_name="fwd",
        )

        # Save fp8 quantized tensors for backward reuse
        if fp8 and fp8_meta is not None:
            _, _, inp_t_fp8, inp_t_scale, weight_fp8, weight_scale = fp8_meta
            ctx.inp_t_fp8 = inp_t_fp8
            ctx.inp_t_scale = inp_t_scale
            ctx.weight_fp8 = weight_fp8
            ctx.weight_scale = weight_scale
        else:
            ctx.inp_t_fp8 = None
            ctx.inp_t_scale = None
            ctx.weight_fp8 = None
            ctx.weight_scale = None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward."""
        # When fp8=True and save_original_input=False the forward pass stores
        # only the weight tensor via ``save_for_backward``. In that case the
        # bf16 activation is not available and both dgrad/wgrad must run on
        # the fp8-quantized tensors stashed on ctx.
        saved = ctx.saved_tensor()
        if ctx.bf16_input_saved:
            input, weight = saved
        else:
            (weight,) = saved
            input = None
        main_grad = ctx.main_grad
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        handle = None
        tp_group = ctx.tp_group
        fp8 = ctx.fp8
        fp8_wgrad = ctx.fp8_wgrad

        input_needs_grad = not ctx.input_stop_gradient

        if ctx.gradient_accumulation_fusion:
            weight.main_grad = main_grad

        wgrad_compute = True
        if grad_output_buffer is not None:
            if (
                wgrad_deferral_limit == 0
                or len(grad_output_buffer) < wgrad_deferral_limit
            ):
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                dim_size = list(input.shape)
                dim_size[0] = dim_size[0] * tp_group.world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = dist.stream.all_gather(
                    all_gather_buffer, input, group=tp_group, sync_op=False
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input

        # Compute grad_input
        if input_needs_grad:
            if fp8 and ctx.weight_fp8 is not None:
                # FP8 dgrad: grad_input = grad_output @ weight
                # fp8_gemm_nt(A, B) = A @ B.T, so B = weight.T = [K, N] K-major
                # print(f"[backward] dgrad fp8: grad_output={grad_output.shape} weight_fp8={ctx.weight_fp8.shape} weight_scale={ctx.weight_scale.shape}")
                if _FP8_SHADOW_DEBUG:
                    with paddle.no_grad():
                        # dgrad math: grad_input = grad_output @ weight.T
                        # (mirrors the bf16 fallback branch below)
                        dgrad_shadow = paddle.matmul(grad_output, weight.t())
                else:
                    dgrad_shadow = None
                grad_input, _ = general_gemm(
                    grad_output,
                    (ctx.weight_fp8.T.contiguous(), ctx.weight_scale.T.contiguous()),
                    fp8=True,
                    inp_quant_func=lambda x: paddle.incubate.nn.functional.fp8_quant_blockwise(
                        x,
                        output_scale_transpose=False,
                        quant_method="1x128",
                        input_transpose=False,
                        using_pow2_scale=ctx.use_pow2_scale,
                    )[:2],
                    pass_name="dgrad",
                    shadow_ref=dgrad_shadow,
                )
            else:
                # print(f"[backward] dgrad bf16: grad_output={grad_output.shape} weight={weight.shape}")
                grad_input, _ = general_gemm(
                    grad_output, weight.t(), pass_name="dgrad",
                )
        else:
            grad_input = None

        if ctx.sequence_parallel and wgrad_compute:
            # pylint: disable=possibly-used-before-assignment
            handle.wait()

        if wgrad_compute:
            if total_input is not None:
                grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                    grad_output, total_input
                )
            else:
                # fp8 + save_original_input=False: no bf16 input available.
                # Only reshape grad_output to 2D for the fp8 wgrad path.
                grad_output = grad_output.contiguous()
                if grad_output.dim() == 3:
                    grad_output = grad_output.reshape(
                        [
                            grad_output.shape[0] * grad_output.shape[1],
                            grad_output.shape[2],
                        ]
                    )

        if ctx.allreduce_dgrad and input_needs_grad:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=tp_group, sync_op=False)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            if input_needs_grad:
                dim_size = list(input.shape)
                sub_grad_input = paddle.empty(
                    dim_size, dtype=input.dtype, requires_grad=False
                )
                # reduce_scatter
                handle = _reduce_scatter_base(
                    sub_grad_input, grad_input, group=tp_group, sync_op=False
                )
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # reduce scatter is scheduled before the weight gradient computation
            else:
                sub_grad_input = None

        # Compute grad_weight
        if fp8_wgrad and wgrad_compute:
            # FP8 wgrad: grad_output^T @ total_input
            # grad_output and total_input are already 2D from prepare_input_tensors_for_wgrad_compute
            # Quantize grad_output transposed and total_input transposed
            # print(f"[backward] wgrad fp8: grad_output={grad_output.shape} total_input={total_input.shape}")
            grad_out_t_fp8, grad_out_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                grad_output.T.contiguous(),
                output_scale_transpose=False,
                quant_method="1x128",
                input_transpose=False,
                using_pow2_scale=ctx.use_pow2_scale,
            )[:2]
            inp_t_fp8, inp_t_scale = ctx.inp_t_fp8, ctx.inp_t_scale
            if inp_t_fp8 is None:
                assert total_input is not None, (
                    "fp8 wgrad requires either pre-quantized inp_t_fp8 (from"
                    " fp8 forward with input_trans=True) or the bf16 total_input"
                    " (set save_original_input=True to keep bf16 activation)"
                )
                inp_t_fp8, inp_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    total_input.T.contiguous(),
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_pow2_scale=ctx.use_pow2_scale,
                )[:2]

            grad_weight, _ = general_gemm(
                (inp_t_fp8, inp_t_scale),
                (grad_out_t_fp8, grad_out_t_scale),
                fp8=True,
                recipe=(1, 1, 128),
                pass_name="wgrad",
                shadow_ref=(
                    paddle.matmul(total_input.T.contiguous(), grad_output)
                    if _FP8_SHADOW_DEBUG and total_input is not None else None
                ),
            )

        elif ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == paddle.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (
                    paddle.float16,
                    paddle.bfloat16,
                ):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError(
                        "Unsupported gradient type for gradient accumulation fusion"
                    )

            if hasattr(weight, "grad_added_to_main_grad"):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, "zero_out_wgrad", False):
                    grad_weight = paddle.zeros(
                        weight.main_grad.shape,
                        dtype=ctx.input_dtype,
                        requires_grad=False,
                    )
                else:
                    grad_weight = paddle.empty(
                        weight.main_grad.shape,
                        dtype=ctx.input_dtype,
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            if wgrad_compute:
                assert total_input is not None, (
                    "bf16 wgrad requires the saved bf16 activation; set"
                    " save_original_input=True or enable fp8_wgrad when using fp8"
                )
                # print(f"[backward] wgrad bf16: total_input={total_input.shape} grad_output={grad_output.shape}")
            grad_weight, _ = general_gemm(total_input.t(), grad_output)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            if input_needs_grad:
                handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            if use_bias:
                return sub_grad_input, grad_weight, grad_bias
            else:
                return sub_grad_input, grad_weight

        if ctx.allreduce_dgrad and input_needs_grad:
            handle.wait()

        # PyLayer requires the number of output in backward
        # function matches the number of Tensors in forward's
        # input args
        if use_bias:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight


def linear_with_grad_accumulation_and_async_allreduce(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[paddle.Tensor] | None = None,
    wgrad_deferral_limit: int | None = 0,
    async_grad_allreduce: bool | None = None,
    tp_group: paddle.core.ProcessGroup | None = None,
    use_accuracy_compatible: bool = False,
    fp8: bool = False,
    fp8_wgrad: bool = False,
    inp_quant_func=None,
    weight_quant_func=None,
    use_pow2_scale: bool = False,
    save_original_input: bool = False,
) -> paddle.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calculation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (paddle.Tensor required): input like paddle.nn.functional.linear

        weight (paddle.Tensor required): weight like paddle.nn.functional.linear

        bias (paddle.Tensor optional): bias like paddle.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        allreduce_dgrad (bool required): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        tp_group (paddle.core.ProcessGroup required): The process group to use for tensor
                                                   parallel operations.

        grad_output_buffer (List[paddle.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        wgrad_deferral_limit (int optional): Limit on the number of
            micro-batches for which embedding weight gradient GEMM should be
            deferred. Disable by setting this to 0. Defaults to 0.

        async_grad_allreduce (bool optional): Will be removed with 0.11.0.
                                            Please use allreduce_dgrad instead.
    """

    if async_grad_allreduce is not None:
        warnings.warn(
            "async_grad_allreduce is deprecated, not in use anymore and will"
            " be fully removed with 0.11.0. Please use allreduce_dgrad instead."
        )

    tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        use_accuracy_compatible,
        fp8,
        fp8_wgrad,
        inp_quant_func,
        weight_quant_func,
        use_pow2_scale,
        save_original_input,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False


def _log_fp8_linear_once(layer, input_, weight):
    if not getattr(layer, "fp8", False) or getattr(layer, "_fp8_linear_logged", False):
        return
    layer._fp8_linear_logged = True
    print(
        "[fp8_linear] "
        f"class={layer.__class__.__name__} "
        f"name={getattr(layer, 'tp_comm_buffer_name', None)} "
        f"input_shape={list(input_.shape)} "
        f"weight_shape={list(weight.shape)} "
        f"fp8_wgrad={getattr(layer, 'fp8_wgrad', False)} "
        f"recipe={getattr(layer.config, 'fp8_recipe', 'blockwise')}",
        flush=True,
    )


class Linear(paddle.nn.Layer):
    """Linear layer with no tensor parallelism (weight duplicated across TP ranks).

    The linear layer is defined as Y = XA + b. Weight is not split and is
    replicated on all tensor parallel ranks. Interface is identical to
    ColumnParallelLinear for drop-in compatibility.

    Refer to Megatron-LM's TELinear with parallel_mode="duplicated" for the
    equivalent design.

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias.
        gather_output:
            Unused. Kept for interface compatibility with ColumnParallelLinear.
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It
            returns the master weights used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by
            the caller. This enables performance optimizations where bias can be
            fused with other elementwise operations.
        skip_weight_param_allocation:
            If True, weight parameter is not allocated and must be passed as a
            keyword argument `weight` during the forward pass. Defaults to False.
        embedding_activation_buffer:
            This buffer holds the input activations of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute
            is enabled.
        grad_output_buffer:
            This buffer holds the gradient outputs of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute
            is enabled.
        is_expert:
            If True, the layer is treated as an MoE expert layer.
        config:
            ModelParallelConfig object.
        tp_comm_buffer_name:
            Not used. Kept for interface compatibility.
        disable_grad_reduce:
            Not used. Weight is replicated so no TP grad reduction is needed.
        tp_group:
            Not used. Kept for interface compatibility.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: list[paddle.Tensor] | None = None,
        grad_output_buffer: list[paddle.Tensor] | None = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        disable_grad_reduce: bool = False,
        tp_group: paddle.core.ProcessGroup | None = None,
        fp8: bool = False,
        fp8_wgrad: bool = False,
        save_original_input: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self._dtype = config.params_dtype
        self.tp_comm_buffer_name = tp_comm_buffer_name
        self._fp8_linear_logged = False

        # No TP: output_size_per_partition equals the full output_size.
        self.output_size_per_partition = output_size

        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if config.perform_initialization:
                    _initialize_affine_weight_cpu(
                        self.weight,
                        self.input_size,
                        self.output_size,
                        self.output_size,  # full output, no partition
                        1,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=0,
                        world_size=1,
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )

            # Weight is duplicated across TP ranks; reduce gradient on DP group.
            self.weight.allreduce = True
            self.weight.is_distributed = False
        else:
            self.weight = None

        if bias:
            self.bias = self.create_parameter(
                shape=[self.output_size],
                dtype=config.params_dtype,
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                with paddle.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = True
            self.bias.is_distributed = False
        else:
            self.bias = None

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # FP8 config: opt-in via explicit kwargs (no longer auto-detected from
        # config). ``save_original_input`` controls whether to keep bf16
        # activations for backward when fp8 is on; when False (default), only
        # fp8-quantized activations are stashed on ctx.
        self.fp8 = bool(fp8)
        self.fp8_wgrad = bool(fp8_wgrad) and self.fp8
        self.save_original_input = bool(save_original_input)
        self.use_pow2_scale = False
        self.inp_quant_func = None
        self.weight_quant_func = None
        if self.fp8:
            # Linear has no TP; the assertion is trivially satisfied but kept
            # for symmetry with ColumnParallelLinear / RowParallelLinear.
            from paddlefleet.fp8.quantization import get_quant_func

            self.use_pow2_scale = (
                paddle.device.cuda.get_device_capability()[0] == 10
            )
            self.inp_quant_func, self.weight_quant_func = get_quant_func(
                getattr(config, "fp8_recipe", "blockwise"),
                input_trans=True,
                out_scale_trans=False,
                pow2_scale=self.use_pow2_scale,
            )

    def forward(
        self,
        input_: paddle.Tensor,
        weight: paddle.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ):
        """Forward of Linear (no tensor parallelism).

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden].
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Unused. Kept for interface compatibility.

        Returns:
            - output
            - bias
        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to Linear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            expected_shape = [self.input_size, self.output_size]
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer)
                < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_)

        if not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = (
                linear_with_grad_accumulation_and_async_allreduce
            )

        _log_fp8_linear_once(self, input_, weight)

        output = self._forward_impl(
            input=input_,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
            grad_output_buffer=(
                self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            wgrad_deferral_limit=(
                self.config.wgrad_deferral_limit
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            tp_group=None,
            use_accuracy_compatible=getattr(
                self.config, "use_accuracy_compatible", False
            ),
            fp8=self.fp8,
            fp8_wgrad=self.fp8_wgrad,
            inp_quant_func=self.inp_quant_func,
            weight_quant_func=self.weight_quant_func,
            use_pow2_scale=self.use_pow2_scale,
            save_original_input=self.save_original_input,
        )

        output_bias = (
            self.bias.clone()
            if (self.skip_bias_add and self.bias is not None)
            else None
        )

        return output, output_bias

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Weight is replicated, no sharding rules needed."""
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(
            state_dict, None, structured_name_prefix
        )

    def set_extra_state(self, state):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def __repr__(self):
        use_bias = self.bias is not None
        return (
            f"{type(self).__name__}(in_features={self.input_size}, "
            f"out_features={self.output_size}, bias={use_bias}, TP=1)"
        )


def column_sequence_parallel_linear(
    x,
    weight,
    bias=None,
    mp_group=None,
):
    """Functional version of ColumnSequenceParallelLinear using fused_linear.

    Forward: all-gather input along seq dim -> fused_linear.
    Input shape: [seq/mp, batch, hidden], output shape: [seq, batch, hidden/mp].

    Args:
        x: Input tensor (sequence-parallel partitioned).
        weight: Weight tensor, shape [in_features, out_features_per_partition].
        bias: Bias tensor, shape [out_features_per_partition], or None.
        mp_group: Tensor parallel process group.

    Returns:
        Output tensor.
    """
    from paddle.incubate.nn.functional import fused_linear

    from paddlefleet.tensor_parallel.sequence_parallel_utils_legacy import (
        AllGatherOpLegacy,
    )

    is_mp = mp_group is not None and mp_group.nranks > 1

    if is_mp:
        input_parallel = AllGatherOpLegacy.apply(x, 0, mp_group)
    else:
        input_parallel = x

    output = fused_linear(input_parallel, weight, bias)
    return output


def row_sequence_parallel_linear(
    x,
    weight,
    bias=None,
    mp_group=None,
):
    """Functional version of RowSequenceParallelLinear using fused_linear.

    Forward: fused_linear -> reduce-scatter along seq dim.
    Input shape: [seq, batch, hidden/mp], output shape: [seq/mp, batch, hidden].

    Args:
        x: Input tensor (already column-parallel partitioned).
        weight: Weight tensor, shape [in_features_per_partition, out_features].
        bias: Bias tensor, shape [out_features], or None.
        mp_group: Tensor parallel process group.

    Returns:
        Output tensor.
    """
    from paddle.incubate.nn.functional import fused_linear

    from paddlefleet.tensor_parallel.sequence_parallel_utils_legacy import (
        ReduceScatterOpLegacy,
    )

    is_mp = mp_group is not None and mp_group.nranks > 1

    if is_mp:
        output_parallel = fused_linear(x, weight, None)
        output_ = ReduceScatterOpLegacy.apply(output_parallel, mp_group)
        if bias is not None:
            output = output_ + bias
        else:
            output = output_
    else:
        output = fused_linear(x, weight, bias)

    return output


def vocab_parallel_embedding(
    input_,
    weight,
    vocab_start_index,
    num_embeddings,
    mp_group=None,
):
    """Functional version of VocabParallelEmbedding.

    Uses _c_lookup_table + _mp_allreduce (same as paddle fleet mp_layers).

    Args:
        input_: Input token ids tensor.
        weight: Embedding weight, shape [num_embeddings_per_partition, embedding_dim].
        vocab_start_index: Start index of the vocab partition on this rank.
        num_embeddings: Total vocabulary size.
        mp_group: Tensor parallel process group.

    Returns:
        Output embedding tensor.
    """
    from paddle.distributed.fleet.layers.mpu import mp_ops

    is_mp = mp_group is not None and mp_group.nranks > 1

    if is_mp:
        output_parallel = mp_ops._c_lookup_table(
            weight,
            input_,
            start_index=vocab_start_index,
            vocab_size=num_embeddings,
        )
        output = mp_ops._mp_allreduce(
            output_parallel,
            group=mp_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )
    else:
        output = F.embedding(input_, weight=weight)

    return output


class ColumnParallelLinear(paddle.nn.Layer):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias
        gather_output:
            If true, call all-gather on output and make Y available to all GPUs,
            otherwise, every GPU will have its output which is Y_i = XA_i
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It
            returns the master weights used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimations where bias can be fused with other
            elementwise operations.
        skip_weight_param_allocation:
            If True, weight parameter is not allocated and must be passed
            as a keyword argument `weight` during the forward pass. Note that this does not
            affect bias, which will be allocated if bias is True. Defaults to False.
        embedding_activation_buffer:
            This buffer holds the input activations of the final embedding
            linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        grad_output_buffer:
            This buffer holds the gradient outputs of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        is_expert:
            If True, the layer is treated as an MoE expert layer.
        config:
            ModelParallelConfig object
        tp_comm_buffer_name:
            Communication buffer name is not used in non-Transformer-Engine modules.
        disable_grad_reduce:
            If True, reduction of output gradients across tensor-parallel ranks
            will be disabled. Defaults to False. This feature is used by Lora Adapter in Nemo to
            delay and fuse reduction along with other gradients for performance optimization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: list[paddle.Tensor] | None = None,
        grad_output_buffer: list[paddle.Tensor] | None = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,  # Not used
        disable_grad_reduce: bool = False,
        tp_group: paddle.core.ProcessGroup | None = None,
        fp8: bool = False,
        fp8_wgrad: bool = False,
        save_original_input: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self.disable_grad_reduce = disable_grad_reduce
        self.tp_group = tp_group
        self._dtype = config.params_dtype
        self.tp_comm_buffer_name = tp_comm_buffer_name
        self._fp8_linear_logged = False

        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert, check_initialized=False
        )
        self.world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)
        self.rank = rank
        self.explicit_expert_comm = self.is_expert and (
            self.world_size > 1 or self.expert_parallel
        )
        self.output_size_per_partition = divide(output_size, self.world_size)

        # Parameters.
        # Initialize weight.
        # Note: create the transpose weight, in linear function, the weight
        # should be transposed.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size_per_partition],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )

                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.input_size,
                        self.output_size,
                        self.output_size_per_partition,
                        1,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=self.world_size,
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[self.input_size, self.output_size_per_partition],
                    dtype=config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )

            self.weight.allreduce = not (
                self.is_expert and self.expert_parallel
            )
            self.weight.is_distributed = True if self.world_size > 1 else False
        else:
            self.weight = None

        if bias:
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                dtype=config.params_dtype,
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )

            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with paddle.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = not (self.is_expert and self.expert_parallel)
            self.bias.is_distributed = True if self.world_size > 1 else False
        else:
            self.bias = None
            # self.register_parameter("bias", None)

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and self.world_size <= 1:
            warnings.warn(
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {self.world_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = (
            self.world_size > 1
            and not self.sequence_parallel
            and not self.disable_grad_reduce
        )

        self.gradient_accumulation_fusion = False

        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # FP8 config (opt-in)
        self.fp8 = bool(fp8)
        self.fp8_wgrad = bool(fp8_wgrad) and self.fp8
        self.save_original_input = bool(save_original_input)
        self.use_pow2_scale = False
        self.inp_quant_func = None
        self.weight_quant_func = None
        if self.fp8:
            assert self.world_size == 1, (
                "ColumnParallelLinear FP8 currently requires TP=1, "
                f"got world_size={self.world_size}"
            )
            from paddlefleet.fp8.quantization import get_quant_func

            self.use_pow2_scale = (
                paddle.device.cuda.get_device_capability()[0] == 10
            )
            self.inp_quant_func, self.weight_quant_func = get_quant_func(
                getattr(config, "fp8_recipe", "blockwise"),
                input_trans=True,
                out_scale_trans=False,
                pow2_scale=self.use_pow2_scale,
            )

    def forward(
        self,
        input_: paddle.Tensor,
        weight: paddle.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ):
        """Forward of ColumnParallelLinear

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `gather_output` arg in the constructor will be used.

        Returns:
            - output
            - bias

        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = [self.input_size, self.output_size_per_partition]
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if getattr(self.config, "gpt_model_use_experimental_version", False):
            output = column_sequence_parallel_linear(
                input_, weight, self.bias, mp_group=self.tp_group
            )
            return output, None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
            or (self.tp_group is not None and self.tp_group.world_size == -1)
            or self.tp_group is None
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(
                input_,
                group=self.tp_group,
                is_expert=self.is_expert,
            )

        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer)
                < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = (
                linear_with_grad_accumulation_and_async_allreduce
            )

        allreduce_dgrad = (
            False if self.explicit_expert_comm else self.allreduce_dgrad
        )

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                if not HAVE_TE:
                    assert self.config.cpu_offloading is False, (
                        "CPU Offloading cannot be enabled while TE is not present"
                    )
                else:
                    input_parallel.activation_offloading = (
                        self.config.cpu_offloading_activations
                    )

        _log_fp8_linear_once(self, input_parallel, weight)

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=False
            if self.explicit_expert_comm
            else self.sequence_parallel,
            grad_output_buffer=(
                self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            wgrad_deferral_limit=(
                self.config.wgrad_deferral_limit
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            tp_group=self.tp_group,
            use_accuracy_compatible=getattr(
                self.config, "use_accuracy_compatible", False
            ),
            fp8=self.fp8,
            fp8_wgrad=self.fp8_wgrad,
            inp_quant_func=self.inp_quant_func,
            weight_quant_func=self.weight_quant_func,
            use_pow2_scale=self.use_pow2_scale,
            save_original_input=self.save_original_input,
        )

        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output = output_parallel
        output_bias = (
            self.bias.clone()
            if (self.skip_bias_add and self.bias is not None)
            else None
        )

        return output, output_bias

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Sharding along axis 1, bias sharded"""
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 1, "bias": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )

    def set_extra_state(self, state):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def __repr__(self):
        tp = self.output_size // self.output_size_per_partition
        use_bias = self.bias is not None
        return (
            f"{type(self).__name__}(in_features={self.input_size}, "
            f"out_features={self.output_size}, bias={use_bias}, TP={tp})"
        )


class RowParallelLinear(paddle.nn.Layer):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias. Note that bias is not parallelized.
        input_is_parallel:
            If true, we assume that the input is already split across the GPUs
            and we do not split again.
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It returns the master weights
            used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimations where bias can be fused with other
            elementwise operations.
        is_expert:
            If True, the layer is treated as an MoE expert layer
        tp_comm_buffer_name:
            Communication buffer name. Not used in non-Transformer-Engine modules.
        config:
            FleetConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,  # Not used
        tp_group: paddle.core.ProcessGroup | None = None,
        fp8: bool = False,
        fp8_wgrad: bool = False,
        save_original_input: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        # self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.gradient_accumulation_fusion = False
        self.sequence_parallel = config.sequence_parallel
        self.tp_group = tp_group
        self._dtype = config.params_dtype
        self.tp_comm_buffer_name = tp_comm_buffer_name
        self._fp8_linear_logged = False

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_parallel`, `input_is_parallel` must be `True`"
            )

        # Divide the weight matrix along the last dimension.
        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert, check_initialized=False
        )

        self.world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)
        self.explicit_expert_comm = self.is_expert and (
            self.world_size > 1 or self.expert_parallel
        )

        self.input_size_per_partition = divide(input_size, self.world_size)

        # Parameters.
        # Note: create the transposed weight here, and the weight should
        # be transposed back in the forward function of linear.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.output_size],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.input_size,
                    self.output_size,
                    self.input_size_per_partition,
                    0,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=self.world_size,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.output_size],
                dtype=config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    is_expert=self.is_expert,
                )
        self.weight.allreduce = not (self.is_expert and self.expert_parallel)
        self.weight.is_distributed = True if self.world_size > 1 else False

        if bias:
            self.bias = self.create_parameter(
                shape=[self.output_size],
                dtype=config.params_dtype,
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )

            if config.perform_initialization:
                # Always initialize bias to zero.
                with paddle.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = not (self.is_expert and self.expert_parallel)
            self.bias.sequence_parallel = self.sequence_parallel
        else:
            self.bias = None
            # self.register_parameter("bias", None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # FP8 config (opt-in)
        self.fp8 = bool(fp8)
        self.fp8_wgrad = bool(fp8_wgrad) and self.fp8
        self.save_original_input = bool(save_original_input)
        self.use_pow2_scale = False
        self.inp_quant_func = None
        self.weight_quant_func = None
        if self.fp8:
            assert self.world_size == 1, (
                "RowParallelLinear FP8 currently requires TP=1, "
                f"got world_size={self.world_size}"
            )
            from paddlefleet.fp8.quantization import get_quant_func

            self.use_pow2_scale = (
                paddle.device.cuda.get_device_capability()[0] == 10
            )
            self.inp_quant_func, self.weight_quant_func = get_quant_func(
                getattr(config, "fp8_recipe", "blockwise"),
                input_trans=True,
                out_scale_trans=False,
                pow2_scale=self.use_pow2_scale,
            )

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        if getattr(self.config, "gpt_model_use_experimental_version", False):
            output = row_sequence_parallel_linear(
                input_, self.weight, self.bias, mp_group=self.tp_group
            )
            return output, None

        # Set up backprop all-reduce.
        if (
            self.input_is_parallel
            or self.tp_group is None
            or (self.tp_group is not None and self.tp_group.nranks == 1)
        ):
            # NOTE: if tp_group only contains one rank, directly set input_parallel to input_
            # otherwise it will fail in scatter_to_tensor_model_parallel_region.
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(
                input_, group=self.tp_group
            )

        # Matrix multiply.
        if not self.weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = (
                linear_with_grad_accumulation_and_async_allreduce
            )

        allreduce_dgrad = False

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                if not HAVE_TE:
                    assert self.config.cpu_offloading is False, (
                        "CPU Offloading cannot be enabled while TE is not present"
                    )
                else:
                    input_parallel.activation_offloading = (
                        self.config.cpu_offloading_activations
                    )

        _log_fp8_linear_once(self, input_parallel, self.weight)

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=False,
            tp_group=None,
            grad_output_buffer=None,
            use_accuracy_compatible=getattr(
                self.config, "use_accuracy_compatible", False
            ),
            fp8=self.fp8,
            fp8_wgrad=self.fp8_wgrad,
            inp_quant_func=self.inp_quant_func,
            weight_quant_func=self.weight_quant_func,
            use_pow2_scale=self.use_pow2_scale,
            save_original_input=self.save_original_input,
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output_ = reduce_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group, is_expert=self.is_expert
            )
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias.clone() if self.bias is not None else None
        return output, output_bias

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(structured_name_prefix="")
        shard_rules = None if self.world_size == 1 else {"weight": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )

    def set_extra_state(self, state):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def __repr__(self):
        tp = self.input_size // self.input_size_per_partition
        use_bias = self.bias is not None
        return (
            f"{type(self).__name__}(in_features={self.input_size}, "
            f"out_features={self.output_size}, bias={use_bias}, TP={tp})"
        )
