# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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
"""FP8 Utils"""

import numpy
import paddle
import paddle.nn.functional as F

from paddlefleet.fusions.fused_swiglu_scale import (
    fused_swiglu_scale_backward,
    fused_swiglu_scale_forward,
)

try:
    from paddlefleet.ops import (
        deep_gemm as paddlefleet_deep_gemm,
        fuse_stack_fp8_quant,
        fuse_stack_transpose_fp8_quant,
        fuse_weighted_swiglu_fp8_quant,
    )
except (ImportError, RuntimeError):
    pass

# 优先使用 FusedQuantOps.fused_swiglu_probs_bwd（inplace，行为对齐）。
# 若环境中没有 FusedQuantOps，则回退到 paddle.incubate 的 out-of-place 实现。
# TODO: 迁移fused_swiglu_probs_bwd至paddlefleet.ops
try:
    import FusedQuantOps as _FQO

    _fused_swiglu_probs_bwd = _FQO.fused_swiglu_probs_bwd
    USE_INPLACE_SWIGLU_BWD = True
except (ImportError, AttributeError):
    _fused_swiglu_probs_bwd = None
    USE_INPLACE_SWIGLU_BWD = False

try:
    from paddle.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        """
            使用swiglu函数对输入的张量进行Sigmoid-weighted Linear Unit操作，并返回结果。
        如果没有提供y参数，则将输入的张量分割成两个部分，一个是Sigmoid函数的输入，另一个是Linear Unit的输入。
        否则，将x视为Sigmoid函数的输入，y视为Linear Unit的输入。

        Args:
            x (Tensor): 要进行Sigmoid-weighted Linear Unit操作的输入张量，其形状可以是任意维度。（默认值：None）
            y (Tensor, optional): 要与x相乘的常数项，其形状应该和x相同。（默认值：None）

        Returns:
            Tensor: Sigmoid-weighted Linear Unit后的输出张量，其形状与x相同。

        Raises:
            TypeError: 当x不是Tensor类型时会抛出此类型错误。
            ValueError: 当x和y的形状不匹配时会抛出此值错误。
        """
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


try:
    from paddlefleet.ops import deep_gemm
except:
    pass

try:
    from paddle.incubate.nn.functional import fused_transpose_wlch_split_quant
except ImportError:
    fused_transpose_wlch_split_quant = None

__all__ = [
    "ExpertsGroupGemmContiguousNode",
]


FP8_ALIGN = 128


def _get_fp8_weight_and_scale(weight, transpose=False):
    """_get_fp8_weight_and_scale"""
    fp8_weight, fp8_scale = (
        weight.fp8_weight_stacked,
        weight.fp8_scale_stacked,
    )

    if transpose:
        if (
            hasattr(weight, "fp8_weight_stacked_transpose")
            and weight.fp8_weight_stacked_transpose is not None
        ):
            fp8_weight = weight.fp8_weight_stacked_transpose
            fp8_scale = weight.fp8_scale_stacked_transpose
        else:
            # 只有非转置版，on-the-fly reshape+transpose
            assert fp8_weight.shape[0] % weight.shape[0] == 0
            assert fp8_weight.ndim == 2
            expert_num = fp8_weight.shape[0] // weight.shape[0]

            def transpose_tensor(tensor):
                assert tensor.ndim == 2
                h0 = tensor.shape[0] // expert_num
                h1 = tensor.shape[1]
                tensor = tensor.reshape([expert_num, h0, h1])
                return (
                    tensor.contiguous()
                    .transpose([0, 2, 1])
                    .reshape([-1, h0])
                    .contiguous()
                )

            fp8_weight, fp8_scale = (
                transpose_tensor(fp8_weight),
                transpose_tensor(fp8_scale),
            )

    return fp8_weight, fp8_scale


def fused_stack_quant_without_cache(
    expert_weight_list, transpose=False, use_ue8m0=False
):
    use_pow2_scale = False
    if paddle.device.cuda.get_device_capability()[0] == 10:
        # Blackwell GPUs require the use of pow2_scales quantization.
        use_pow2_scale = True
    if transpose:
        w, scale = fuse_stack_transpose_fp8_quant(
            expert_weight_list,
            use_pow2_scale,
            use_ue8m0,
            use_ue8m0,
        )
    else:
        w, scale = fuse_stack_fp8_quant(
            expert_weight_list,
            use_pow2_scale,
            use_ue8m0,
            use_ue8m0,
        )

    if use_ue8m0:
        scale = scale.T
    return w, scale


def fused_stack_quant(expert_weight_list, transpose=False, use_ue8m0=False):
    if hasattr(expert_weight_list[0], "fp8_weight_stacked"):
        w, scale = _get_fp8_weight_and_scale(
            expert_weight_list[0], transpose=transpose
        )
    else:
        w, scale = fused_stack_quant_without_cache(
            expert_weight_list, transpose, use_ue8m0
        )
    return w, scale


def tilewise_quant(x):
    """
    Tile-wise FP8 quantization: quantize input tensor to FP8 with per-tile (1x128) scaling.
    """
    pow_2_scales = False
    if paddle.device.cuda.get_device_capability()[0] == 10:
        # Blackwell GPUs require the use of pow2_scales quantization.
        pow_2_scales = True
    if x.shape[0] > 0:
        return paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
        )
    else:
        shape = list(x.shape)
        x_fp8 = paddle.empty(x.shape, dtype=paddle.float8_e4m3fn)
        assert shape[-1] % FP8_ALIGN == 0, shape
        shape[-1] //= FP8_ALIGN
        x_scale = paddle.empty(shape, dtype=paddle.float32)
        return x_fp8, x_scale


def split_group_gemm(
    x_fp8, x_scale, w_fp8, w_scale, tokens_per_expert, gemm_out
):
    """
    将输入的张量分割成多个小的矩阵乘

    Args:
        x_fp8 (paddle.Tensor, shape=(N, T)): 需要进行矩阵乘法的FP8格式的张量。
        x_scale (paddle.Tensor, shape=(N, T)): 与x_fp8对应的缩放因子。
        w_fp8 (List[paddle.Tensor], length=6): 包含6个FP8格式的张量，每个张量代表一个专家的权重。
        w_scale (List[paddle.Tensor], length=6): 与w_fp8对应的缩放因子。
        tokens_per_expert (List[int], length=6): 每个专家处理的token数量。
        gemm_out (paddle.Tensor, shape=(N, T)): 存储结果的张量。

    Returns:
        paddle.Tensor, shape=(N, T): 返回计算结果存储在gemm_out中的张量。
    """
    start_idx = 0
    for i, token_num in enumerate(tokens_per_expert):
        if token_num == 0:
            continue
        end_idx = start_idx + token_num

        x_i = x_fp8[start_idx:end_idx]
        x_scale_tma_align = x_scale[start_idx:end_idx].T.contiguous().T

        deep_gemm.fp8_gemm_nt(
            (x_i, x_scale_tma_align),
            (w_fp8[i].contiguous(), w_scale[i].contiguous()),
            gemm_out[start_idx:end_idx],
        )

        start_idx = end_idx

    return gemm_out


def has_config(config_map, key):
    """
    判断给定的配置字典中是否存在指定键，并且该键对应的值不为空。

    Args:
        config_map (Optional[Dict[str, Any]]): 配置字典，可以为None。
        key (str): 需要查找的键名。

    Returns:
        bool: 如果配置字典不为None，且包含指定键，且该键对应的值不为空，则返回True；否则返回False。
    """
    return bool(
        config_map is not None and key in config_map and config_map[key]
    )


def kitchen_gemm(
    x_fp8,
    x_scale,
    w_fp8,
    w_scale,
    is_a_1d_scaled,
    is_b_1d_scaled,
    out=None,
    rtn_dtype=paddle.bfloat16,
):
    # if USE_DS_GEMM:
    #     if out is None:
    #         out = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], rtn_dtype)
    #     if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
    #         deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt((x_fp8, x_scale), (w_fp8, w_scale), out, num_sms=get_sm_num())
    #     return out

    if out is not None:
        accumulate = True
        out_dtype = out.dtype
    else:
        accumulate = False
        out_dtype = rtn_dtype
    if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
        y = paddle.incubate.nn.functional.fp8_gemm_blockwise(
            a=x_fp8,
            a_decode_scale=x_scale,
            b=w_fp8,
            b_decode_scale=w_scale,
            out_dtype=out_dtype,
            out=out,
            accumulate=accumulate,
            use_split_accumulator=True,
            is_a_1d_scaled=is_a_1d_scaled,
            is_b_1d_scaled=is_b_1d_scaled,
        )
    else:
        y = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], out_dtype)
        if out is not None:
            out = out + y
            return out

    return y


class ExpertsGroupGemmContiguousNode:
    """ExpertsGroupGemmContiguousNode"""

    def __init__(
        self,
        custom_map,
        recompute_moe_gate_up=False,
        dequant_input=False,
        group=None,
        name="experts_group_gemm_contiguous_node",
        expert_id=None,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        use_fp8_mlp=True,
        moe_deep_gemm=True,
        moe_grouped_gemm=False,
    ):
        """
            Initializes the experts group gemm contiguous node.

        Args:
            custom_map (CustomMapping): Custom mapping for the model.
            recompute_moe_gate_up (bool, optional): Whether to recompute forward gate up. Defaults to False.
            dequant_input (bool, optional): Whether to dequantize input. Defaults to False.
            name (str, optional): Name of the node. Defaults to "experts_group_gemm_contiguous_node".
        """
        if not moe_grouped_gemm or use_fp8_mlp:
            if expert_id is None:
                self.experts = custom_map.experts
            else:
                self.experts = [custom_map.experts[expert_id]]
        else:
            self.grouped_gemm_experts = custom_map.grouped_gemm_experts
        self.expert_id = expert_id
        self.recompute_moe_gate_up = recompute_moe_gate_up
        self.dequant_input = dequant_input
        self.tokens_per_expert = None
        self.m_indices = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None
        self.fp8_fused_ops_configs = {}
        self.group = group
        self.moe_subbatch_token_num_after_dispatch = (
            moe_subbatch_token_num_after_dispatch
        )
        if self.moe_subbatch_token_num_after_dispatch is not None:
            assert (
                self.moe_subbatch_token_num_after_dispatch > 0
                and self.moe_subbatch_token_num_after_dispatch % FP8_ALIGN == 0
            ), self.moe_subbatch_token_num_after_dispatch
        self.use_bf16_gemm_weight_grad = use_bf16_gemm_weight_grad
        self.use_fp8_mlp = use_fp8_mlp
        self.moe_deep_gemm = moe_deep_gemm
        self.moe_grouped_gemm = moe_grouped_gemm
        self.is_split_group_gemm = not moe_grouped_gemm

    def cached_tensors(self):
        """
        cached_tensors
        """
        return [
            self.tokens_per_expert,
            self.m_indices,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ]

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        (
            self.tokens_per_expert,
            self.m_indices,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ) = tensors

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    def reset_state(self):
        """
        reset_state
        """
        self.tokens_per_expert = None
        self.m_indices = None
        self.clear_activation_tensors()

    def clear_activation_tensors(self):
        """
        clear_activation_tensors
        """
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def gen_m_indices(self, tokens_per_expert):
        """
        generate m indices
        """
        tokens = []
        for i in range(len(tokens_per_expert)):
            tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
        out = paddle.concat(tokens, axis=0)
        return out

    def fwd_gate_up_bf16(self, x, expert_w1):
        """
        fwd_gate_up bf16
        """

        if x is None:
            assert self.input is not None
            x = self.input
        if numpy.prod(x.shape) != 0:
            if self.moe_grouped_gemm:
                if self.moe_deep_gemm:
                    o1 = paddle.zeros(
                        [x.shape[0], expert_w1.shape[2]], dtype="bfloat16"
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                        x,
                        expert_w1,
                        o1,
                        self.tokens_per_expert_indices,
                    )
                else:
                    o1 = paddle.incubate.nn.functional.batched_gemm(
                        x,
                        expert_w1,
                        self.tokens_per_expert,
                    )
            else:
                expert_output_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    x_i = x[start_idx:end_idx].contiguous()
                    expert_w1_i = expert_w1[i]
                    expert_output_list.append(
                        F.linear(x=x_i, weight=expert_w1_i)
                    )
                    start_idx = end_idx
                o1 = paddle.concat(expert_output_list, axis=0)
        else:
            if self.moe_grouped_gemm:
                o1 = paddle.empty(
                    [x.shape[0], expert_w1.shape[2]], dtype=expert_w1[0].dtype
                )
            else:
                o1 = paddle.empty(
                    [x.shape[0], expert_w1[0].shape[1]],
                    dtype=expert_w1[0].dtype,
                )
        self.input = x
        return o1

    def fwd_gate_up(
        self, x, expert_w1, num_expert, tokens_per_expert, scale=None
    ):
        self.tokens_per_expert = tokens_per_expert
        if self.moe_deep_gemm:
            self.tokens_per_expert_indices = paddle.repeat_interleave(
                paddle.arange(len(self.tokens_per_expert)),
                paddle.to_tensor(self.tokens_per_expert),
            ).cast("int32")
        if not self.use_fp8_mlp:
            return self.fwd_gate_up_bf16(x, expert_w1)
        else:
            return self.fwd_gate_up_fp8(
                x, expert_w1, num_expert, tokens_per_expert, scale
            )

    def fwd_gate_up_fp8(
        self, x, expert_w1, num_expert, tokens_per_expert, scale=None
    ):
        """
        o1 = x * w1
        [m_sum, n] = [m_sum, k] * [num_groups, k, n] (m_sum = sum(tokens_per_expert))
        """

        if self.moe_grouped_gemm:
            self.m_indices = self.gen_m_indices(tokens_per_expert)
        # concat w1, shape is [num_groups, n, k]
        w1_t_quant, w1_t_scale = fused_stack_quant(expert_w1, transpose=True)
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        if x is None:
            x_fp8, x_scale = self.input_fp8, self.input_scale
            assert x_fp8 is not None and x_scale is not None
            x_scale = paddle.transpose(
                paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0]
            )
        elif scale is not None:
            x_fp8, x_scale = x, scale
            assert self.dequant_input, (
                "如果传入了scale, 说明a2a使用了fp8,。必须开启dequant_input"
            )
            x_scale = paddle.transpose(
                paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0]
            )
        else:
            # quant x_bf16
            x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=False,
            )
            x_scale = x_scale.T

        # compute gemm
        o1 = paddle.empty(
            [x_fp8.shape[0], w1_t_quant.shape[1]], dtype=expert_w1[0].dtype
        )
        if numpy.prod(x_fp8.shape) != 0:
            if not self.moe_grouped_gemm:
                split_group_gemm(
                    x_fp8,
                    x_scale,
                    w1_t_quant,
                    w1_t_scale,
                    tokens_per_expert,
                    o1,
                )
            else:
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (x_fp8, x_scale),
                    (w1_t_quant, w1_t_scale),
                    o1,
                    m_indices=self.m_indices,
                )

        if self.dequant_input:
            self.input_fp8 = x_fp8
            self.input_scale = x_scale
        else:
            self.input = x
        return o1

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down_bf16(self, o1, unzipped_probs, expert_w2, clear_o1=False):
        """
        fwd_down_bf16
        """

        o2 = fused_swiglu_scale_forward(o1, unzipped_probs)

        if clear_o1:
            o1._clear_to_zero_allocation()

        # down proj
        if numpy.prod(o2.shape) != 0:
            if self.moe_grouped_gemm:
                if self.moe_deep_gemm:
                    o3 = paddle.zeros(
                        [o2.shape[0], expert_w2.shape[2]], dtype="bfloat16"
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                        o2,
                        expert_w2,
                        o3,
                        self.tokens_per_expert_indices,
                    )
                else:
                    o3 = paddle.incubate.nn.functional.batched_gemm(
                        o2,
                        expert_w2,
                        self.tokens_per_expert,
                    )
            else:
                expert_output_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    o1_i = o2[start_idx:end_idx].contiguous()
                    expert_w2_i = expert_w2[i]
                    expert_output_list.append(
                        F.linear(x=o1_i, weight=expert_w2_i)
                    )
                    start_idx = end_idx
                o3 = paddle.concat(expert_output_list, axis=0)
        else:
            if self.moe_grouped_gemm:
                o3_shape = [o2.shape[0], expert_w2.shape[2]]
            else:
                o3_shape = [o2.shape[0], expert_w2[0].shape[1]]
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        return o3

    def fwd_down(
        self, o1, unzipped_probs, expert_w2, num_expert, o3=None, clear_o1=False
    ):
        if not self.use_fp8_mlp:
            return self.fwd_down_bf16(o1, unzipped_probs, expert_w2, clear_o1)
        else:
            return self.fwd_down_fp8(
                o1, unzipped_probs, expert_w2, num_expert, o3, clear_o1
            )

    def fwd_down_fp8(
        self, o1, unzipped_probs, expert_w2, num_expert, o3=None, clear_o1=False
    ):
        """
        o3 = o2 * w2
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # concat and transpose w2
        w2_quant, w2_scale = fused_stack_quant(expert_w2, transpose=True)
        w2_quant = w2_quant.reshape([num_expert, -1, w2_quant.shape[-1]])
        w2_scale = w2_scale.reshape([num_expert, -1, w2_scale.shape[-1]])

        # TODO:support ue8m0 on SM100
        o2_fp8, o2_scale = fuse_weighted_swiglu_fp8_quant(
            o1, unzipped_probs, using_pow2_scaling=True, use_ue8m0=False
        )
        o2_scale = paddle.transpose(
            paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0]
        )

        if clear_o1:
            o1._clear_to_zero_allocation()
        # fused_weighted_swiglu_act_quant 已消费完 o1 产出 o2_fp8，此时 o1 可以安全释放。
        o3_shape = [o2_fp8.shape[0], w2_quant.shape[1]]
        if o3 is not None:
            assert o3.shape == o3_shape, f"{o3.shape} vs {o3_shape}"
            o3.zero_()
        else:
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            if not self.moe_grouped_gemm:
                split_group_gemm(
                    o2_fp8,
                    o2_scale,
                    w2_quant,
                    w2_scale,
                    self.tokens_per_expert,
                    o3,
                )
            else:
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (o2_fp8, o2_scale),
                    (w2_quant, w2_scale),
                    o3,
                    m_indices=self.m_indices,
                )
        return o3

    def bwd_down_input_bf16(self, expert_w2, unzipped_grad, o1, unzipped_probs):
        """
        bwd_down_input_bf16
        """
        if numpy.prod(unzipped_grad.shape) != 0:
            if self.moe_grouped_gemm and not self.use_fp8_mlp:
                if self.moe_deep_gemm:
                    do2_s = paddle.zeros(
                        [unzipped_grad.shape[0], expert_w2.shape[1]],
                        dtype=paddle.bfloat16,
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                        unzipped_grad,
                        expert_w2,
                        do2_s,
                        self.tokens_per_expert_indices,
                    )
                else:
                    do2_s = paddle.incubate.nn.functional.batched_gemm(
                        unzipped_grad,
                        expert_w2,
                        self.tokens_per_expert,
                        trans_rhs=True,
                    )
            else:
                do2_s_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    unzipped_grad_i = unzipped_grad[
                        start_idx:end_idx
                    ].contiguous()
                    expert_w2_i = expert_w2[i].T.contiguous()
                    do2_s_list.append(
                        F.linear(x=unzipped_grad_i, weight=expert_w2_i)
                    )
                    start_idx = end_idx
                do2_s = paddle.concat(do2_s_list, axis=0)
        else:
            if self.moe_grouped_gemm and not self.use_fp8_mlp:
                do2_s_shape = [unzipped_grad.shape[0], expert_w2.shape[1]]
            else:
                do2_s_shape = [unzipped_grad.shape[0], expert_w2[0].shape[1]]
            do2_s = paddle.empty(do2_s_shape, dtype=unzipped_grad.dtype)

        o2_s = fused_swiglu_scale_forward(o1, unzipped_probs)
        do1, probs_grad = fused_swiglu_scale_backward(o1, unzipped_probs, do2_s)

        return do1, o2_s, probs_grad

    def bwd_down_input_fp8(
        self,
        expert_w2,
        unzipped_grad,
        o1,
        unzipped_probs,
        inplace_swiglu_prob=False,
    ):
        """
        do2 = do3 * w2_t
        [m_sum, n] = [m_sum, k] * [num_groups, k, n]
        """
        # recompute concated_w2_2d
        # fp8_gemm_nt(do3[m,k], w2[n,k]) = do3 @ w2^T = do3 @ [k,n]
        bw_w2_quant, bw_w2_scale = fused_stack_quant(expert_w2, transpose=False)
        bw_w2_quant = bw_w2_quant.reshape(
            [len(expert_w2), -1, bw_w2_quant.shape[-1]]
        )
        bw_w2_scale = bw_w2_scale.reshape(
            [len(expert_w2), -1, bw_w2_scale.shape[-1]]
        )
        if hasattr(
            expert_w2[0], "fp8_weight_stacked_transpose"
        ) and not hasattr(expert_w2[0], "fp8_weight_stacked"):
            bw_w2_quant = (
                bw_w2_quant.contiguous().transpose([0, 2, 1]).contiguous()
            )
            bw_w2_scale = (
                bw_w2_scale.contiguous().transpose([0, 2, 1]).contiguous()
            )

        # compute gemm
        unzipped_grad_fp8, unzipped_grad_scale = (
            paddle.incubate.nn.functional.fp8_quant_blockwise(
                unzipped_grad,
                output_scale_transpose=False,
                quant_method="1x128",
                input_transpose=False,
            )
        )

        do2_s = paddle.empty(
            [unzipped_grad_fp8.shape[0], bw_w2_quant.shape[1]],
            dtype=unzipped_grad.dtype,
        )
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            if not self.moe_grouped_gemm:
                split_group_gemm(
                    unzipped_grad_fp8,
                    unzipped_grad_scale,
                    bw_w2_quant,
                    bw_w2_scale,
                    self.tokens_per_expert,
                    do2_s,
                )
            else:
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (unzipped_grad_fp8, unzipped_grad_scale),
                    (bw_w2_quant, bw_w2_scale),
                    do2_s,
                    m_indices=self.m_indices,
                )

        with paddle.amp.auto_cast(False):
            if USE_INPLACE_SWIGLU_BWD:
                # inplace，do1 复用 o1 的 GPU buffer（data_ptr 相同）。
                # del o1 后 do1 仍持有引用，refcount 不归零，物理页不会被 VMM 提前回收。
                # 显存峰值：o1/do1(2H) + do2_s(H) + o2_s(H) = 4H（C 点），
                #           do1(2H) + o2_s(H) + n2_s(2H) = 5H（D 点峰值）
                do1, probs_grad, o2_s = _fused_swiglu_probs_bwd(
                    o1, do2_s, unzipped_probs, True
                )
            else:
                # out-of-place，do1 是全新分配的 buffer。
                # del o1 必须推迟到 bwd_gate_up_input_fp8 的 synchronize 之后，
                # 否则 GPU 异步读 o1 时物理页已被 VMM 回收（Bug 2）。
                # 显存峰值：o1(2H) + do2_s(H) + do1(2H) + o2_s(H) = 6H（C 点），
                #           o1(2H) + do1(2H) + o2_s(H) + n2_s(2H) = 7H（D 点峰值）
                do1, probs_grad, o2_s = (
                    paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                        o1, do2_s, unzipped_probs
                    )
                )

        return do1, o2_s, probs_grad

    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input_bf16(self, do1, expert_w1):
        """
        bwd_gate_up_input_bf16
        """
        if numpy.prod(do1.shape) != 0:
            if self.moe_grouped_gemm and not self.use_fp8_mlp:
                if self.moe_deep_gemm:
                    dx = paddle.zeros(
                        [do1.shape[0], expert_w1.shape[1]],
                        dtype=paddle.bfloat16,
                    )
                    paddlefleet_deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                        do1,
                        expert_w1,
                        dx,
                        self.tokens_per_expert_indices,
                    )
                else:
                    dx = paddle.incubate.nn.functional.batched_gemm(
                        do1,
                        expert_w1,
                        self.tokens_per_expert,
                        trans_rhs=True,
                    )
            else:
                dx_list = []
                start_idx = 0
                for i, token_num in enumerate(self.tokens_per_expert):
                    if token_num == 0:
                        continue
                    end_idx = start_idx + token_num
                    do1_i = do1[start_idx:end_idx].contiguous()
                    expert_w1_i = expert_w1[i].T.contiguous()
                    dx_list.append(F.linear(x=do1_i, weight=expert_w1_i))
                    start_idx = end_idx
                dx = paddle.concat(dx_list, axis=0)
        else:
            if self.moe_grouped_gemm and not self.use_fp8_mlp:
                dx_shape = [do1.shape[0], expert_w1.shape[1]]
            else:
                dx_shape = [do1.shape[0], expert_w1[0].shape[0]]
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        return dx

    def bwd_gate_up_input_fp8(self, do1, expert_w1, dx=None):
        """
        dx = do1 * w1_t
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # recompute concated_w1_t
        bw_w1_quant, bw_w1_scale = fused_stack_quant(expert_w1, transpose=False)
        bw_w1_quant = bw_w1_quant.reshape(
            [len(expert_w1), -1, bw_w1_quant.shape[-1]]
        )
        bw_w1_scale = bw_w1_scale.reshape(
            [len(expert_w1), -1, bw_w1_scale.shape[-1]]
        )

        # quant do1
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
        )

        # compute gemm
        dx_shape = [do1_fp8.shape[0], bw_w1_quant.shape[1]]
        if dx is None:
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        else:
            assert dx.shape == dx_shape, f"{dx.shape} vs {dx_shape}"
            dx.zero_()
        if numpy.prod(do1_fp8.shape) != 0:
            if not self.moe_grouped_gemm:
                split_group_gemm(
                    do1_fp8,
                    do1_scale,
                    bw_w1_quant,
                    bw_w1_scale,
                    self.tokens_per_expert,
                    dx,
                )
            else:
                paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (do1_fp8, do1_scale),
                    (bw_w1_quant, bw_w1_scale),
                    dx,
                    m_indices=self.m_indices,
                )

        return dx

    def fused_transpose_split_quant(
        self, x, scale, tokens_per_expert, pow_2_scales
    ):
        out, scale = paddle.incubate.nn.functional.fused_transpose_split_quant(
            x, scale, tokens_per_expert, pow_2_scales
        )
        return out, scale

    def bwd_down_weight(self, do3, o2, expert_w2):
        """
        dw2 = do2_t * do3
        [n, k] = [n, m_sum] * [m_sum, k] (m_sum = sum(tokens_per_expert))
        """
        o2_t_fp8, o2_t_scale = self.fused_transpose_split_quant(
            o2, None, self.tokens_per_expert, True
        )
        do3_t_fp8, do3_t_scale = self.fused_transpose_split_quant(
            do3, None, self.tokens_per_expert, True
        )

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                if expert_w2[i].main_grad is None:
                    expert_w2[i].main_grad = paddle.zeros(
                        shape=expert_w2[i].shape, dtype=paddle.float32
                    )
                kitchen_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    do3_t_fp8[i],
                    do3_t_scale[i],
                    True,
                    True,
                    expert_w2[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w2[i].grad is None:
                    expert_w2[i].grad = paddle.zeros(
                        shape=expert_w2[i].shape, dtype=paddle.float32
                    )
                kitchen_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    do3_t_fp8[i],
                    do3_t_scale[i],
                    True,
                    True,
                    expert_w2[i].grad,
                    paddle.float32,
                )
            if (
                hasattr(expert_w2[i], "_apply_backward_hook")
                and not expert_w2[i].stop_gradient
            ):
                expert_w2[i]._apply_backward_hook()

    def bwd_gate_up_weight(self, do1, input_x, expert_w1, clear_input=False):
        """
        dw1 = dx_t * do1
        [k, n] = [k, m_sum] * [m_sum, n] (m_sum = sum(tokens_per_expert))
        """

        if input_x is None:
            if self.dequant_input:
                input_x_t_fp8, input_x_t_scale = (
                    self.fused_transpose_split_quant(
                        self.input_fp8,
                        self.input_scale,
                        self.tokens_per_expert,
                        True,
                    )
                )
            else:
                input_x_t_fp8, input_x_t_scale = (
                    self.fused_transpose_split_quant(
                        self.input, None, self.tokens_per_expert, True
                    )
                )
        else:
            input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(
                input_x, None, self.tokens_per_expert, True
            )

        if clear_input:
            self.input = None
            self.input_fp8 = None
            self.input_scale = None

        do1_t_fp8, do1_t_scale = self.fused_transpose_split_quant(
            do1, None, self.tokens_per_expert, True
        )
        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                if expert_w1[i].main_grad is None:
                    expert_w1[i].main_grad = paddle.zeros(
                        shape=expert_w1[i].shape, dtype=paddle.float32
                    )
                kitchen_gemm(
                    input_x_t_fp8[i],
                    input_x_t_scale[i],
                    do1_t_fp8[i],
                    do1_t_scale[i],
                    True,
                    True,
                    expert_w1[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w1[i].grad is None:
                    expert_w1[i].grad = paddle.zeros(
                        shape=expert_w1[i].shape, dtype=paddle.float32
                    )
                kitchen_gemm(
                    input_x_t_fp8[i],
                    input_x_t_scale[i],
                    do1_t_fp8[i],
                    do1_t_scale[i],
                    True,
                    True,
                    expert_w1[i].grad,
                    paddle.float32,
                )
            if (
                hasattr(expert_w1[i], "_apply_backward_hook")
                and not expert_w1[i].stop_gradient
            ):
                expert_w1[i]._apply_backward_hook()

    @paddle.no_grad()
    def forward(
        self,
        hs_out,
        unzipped_probs,
        tokens_per_expert,
        origin_token_per_experts,
        output=None,
        scale=None,
    ):
        """如果传入了scale, 说明在a2a之前就做了quant, 这里的hs_out就是fp8。否则, hs_out是bf16"""
        self.origin_token_per_experts = origin_token_per_experts
        if hs_out is None:
            assert self.input_fp8 is not None
            assert self.input_scale is not None
            shape = self.input_fp8.shape
            dtype = paddle.bfloat16
        elif scale is not None:
            shape = hs_out.shape
            dtype = paddle.bfloat16
        else:
            shape = hs_out.shape
            dtype = hs_out.dtype

        if shape[0] == 0:
            o3 = paddle.zeros(shape, dtype=dtype)
            return o3
        # get w1/w2
        if self.moe_grouped_gemm and not self.use_fp8_mlp:
            expert_w1 = self.grouped_gemm_experts.weight1
            expert_w2 = self.grouped_gemm_experts.weight2
        else:
            expert_w1 = [
                x.up_gate_proj.weight for x in self.experts if x is not None
            ]
            expert_w2 = [
                x.down_proj.weight for x in self.experts if x is not None
            ]

        num_expert = len(expert_w1)

        # o1
        o1 = self.fwd_gate_up(
            hs_out, expert_w1, num_expert, tokens_per_expert, scale=scale
        )
        if not self.recompute_moe_gate_up:
            self.o1 = o1
            clear_o1 = False
        else:
            clear_o1 = True

        # o3
        # 只有 output 是 bf16/float32 时才传给 fwd_down（auto_subbatch 场景）
        # FP8 的 output 是复用给 gate_up 的，不应作为 down proj 输出 buffer
        fwd_down_output = (
            output
            if output is not None
            and output.dtype in (paddle.bfloat16, paddle.float32)
            else None
        )
        o3 = self.fwd_down(
            o1,
            unzipped_probs,
            expert_w2,
            num_expert,
            o3=fwd_down_output,
            clear_o1=clear_o1,
        )
        return o3

    @paddle.no_grad()
    def backward(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        反向传播函数，用于计算输入的梯度和参数的梯度。
            该函数会根据输出梯度更新模型的参数，并返回输入的梯度和隐藏状态的梯度。

            Args:
                out_grad (Tensor, optional): 输出梯度张量，默认为None，表示没有输出梯度。
                    shape为（batch_size, ...），dtype为float32。如果不为None，则需要保证batch_size大于等于1。

            Returns:
                tuple (dx, probs_grad) (Tensor, Tensor):
                    - dx (Tensor) - 输入的梯度张量，shape为（batch_size, ...），dtype为float32。
                    - probs_grad (Tensor) - 隐藏状态的梯度张量，shape为（batch_size, hidden_size），dtype为float32。
        """
        unzipped_probs = unzipped_probs.unsqueeze(-1)
        if out_grad.shape[0] == 0:
            # for cornet case, Get 0 teken in full train step
            dx = paddle.zeros_like(out_grad)
            probs_grad = paddle.zeros_like(unzipped_probs)

            if not self.moe_grouped_gemm or self.use_fp8_mlp:
                for expert in self.experts:
                    if expert is None:
                        continue

                    if hasattr(expert.down_proj.weight, "main_grad"):
                        if expert.down_proj.weight.main_grad is None:
                            expert.down_proj.weight.main_grad = paddle.zeros(
                                shape=expert.down_proj.weight.shape,
                                dtype=paddle.float32,
                            )
                    else:
                        if expert.down_proj.weight.grad is None:
                            expert.down_proj.weight.grad = paddle.zeros(
                                shape=expert.down_proj.weight.shape,
                                dtype=paddle.float32,
                            )

                    if hasattr(expert.up_gate_proj.weight, "main_grad"):
                        if expert.up_gate_proj.weight.main_grad is None:
                            expert.up_gate_proj.weight.main_grad = paddle.zeros(
                                shape=expert.up_gate_proj.weight.shape,
                                dtype=paddle.float32,
                            )
                    else:
                        if expert.up_gate_proj.weight.grad is None:
                            expert.up_gate_proj.weight.grad = paddle.zeros(
                                shape=expert.up_gate_proj.weight.shape,
                                dtype=paddle.float32,
                            )
            else:
                if hasattr(self.grouped_gemm_experts.weight1, "main_grad"):
                    if self.grouped_gemm_experts.weight1.main_grad is None:
                        self.grouped_gemm_experts.weight1.main_grad = (
                            paddle.zeros(
                                shape=self.grouped_gemm_experts.weight1.shape,
                                dtype=paddle.float32,
                            )
                        )
                else:
                    if self.grouped_gemm_experts.weight1.grad is None:
                        self.grouped_gemm_experts.weight1.grad = paddle.zeros(
                            shape=self.grouped_gemm_experts.weight1.shape,
                            dtype=paddle.float32,
                        )

                if hasattr(self.grouped_gemm_experts.weight2, "main_grad"):
                    if self.grouped_gemm_experts.weight2.main_grad is None:
                        self.grouped_gemm_experts.weight2.main_grad = (
                            paddle.zeros(
                                shape=self.grouped_gemm_experts.weight2.shape,
                                dtype=paddle.float32,
                            )
                        )
                else:
                    if self.grouped_gemm_experts.weight2.grad is None:
                        self.grouped_gemm_experts.weight2.grad = paddle.zeros(
                            shape=self.grouped_gemm_experts.weight2.shape,
                            dtype=paddle.float32,
                        )

            if a2a_async_fn:
                dx, task = a2a_async_fn(dx)
                task.wait()
            return dx, probs_grad

        subbatch_rows = self.moe_subbatch_token_num_after_dispatch
        if subbatch_rows is None:
            return self.backward_impl(
                out_grad, unzipped_probs, a2a_async_fn=a2a_async_fn
            )

        assert a2a_async_fn is None, (
            "a2a_async_fn should be None when moe_subbatch_token_num_after_dispatch is not None"
        )
        assert self.expert_id is not None, self.expert_id

        rows, _ = out_grad.shape
        nparts = (rows + subbatch_rows - 1) // subbatch_rows
        if nparts <= 1:
            return self.backward_impl(
                out_grad, unzipped_probs, a2a_async_fn=a2a_async_fn
            )

        input = self.input
        input_fp8 = self.input_fp8
        input_scale = self.input_scale.contiguous()
        o1 = self.o1
        tokens_per_expert = self.tokens_per_expert

        probs_grad = []
        for i in range(nparts):
            s_idx = subbatch_rows * i
            e_idx = min(rows, subbatch_rows * (i + 1))
            if input is not None:
                self.input = input._slice(s_idx, e_idx)

            if input_fp8 is not None:
                self.input_fp8 = input_fp8._slice(s_idx, e_idx)
                self.input_scale = input_scale._slice(s_idx, e_idx)

            if o1 is not None:
                self.o1 = o1._slice(s_idx, e_idx)
            self.tokens_per_expert = [e_idx - s_idx]
            if self.moe_deep_gemm:
                self.tokens_per_expert_indices = paddle.repeat_interleave(
                    paddle.arange(len(self.tokens_per_expert)),
                    paddle.to_tensor(self.tokens_per_expert),
                ).cast("int32")

            tmp_out_grad = out_grad._slice(s_idx, e_idx)
            tmp_unzipped_probs = unzipped_probs._slice(s_idx, e_idx)

            tmp_dx, tmp_probs_grad = self.backward_impl(
                tmp_out_grad, tmp_unzipped_probs
            )
            assert tmp_dx is tmp_out_grad
            probs_grad.append(tmp_probs_grad)

        if self.input is not None:
            self.input = input

        if self.input_fp8 is not None:
            self.input_fp8 = input_fp8
            self.input_scale = input_scale

        if self.o1 is not None:
            self.o1 = o1

        self.tokens_per_expert = tokens_per_expert
        if self.moe_deep_gemm:
            self.tokens_per_expert_indices = paddle.repeat_interleave(
                paddle.arange(len(self.tokens_per_expert)),
                paddle.to_tensor(self.tokens_per_expert),
            ).cast("int32")
        probs_grad = paddle.concat(probs_grad, axis=0)
        return out_grad, probs_grad

    def _lora_weight_grad(
        self, dw, lora_A, lora_B, scaling, grad_attr="main_grad"
    ):
        """
        Given dw (gradient w.r.t. effective weight = w + lora_A @ lora_B * scaling),
        compute and accumulate gradients for lora_A and lora_B.
        dw shape: [E, in_features, out_features]
        lora_A:   [E, in_features, r]
        lora_B:   [E, r, out_features]
        d_lora_B = lora_A.transpose(1,2) @ dw * scaling  -> [E, r, out_features]
        d_lora_A = dw @ lora_B.transpose(1,2) * scaling  -> [E, in_features, r]
        """
        dw_f32 = dw.cast("float32")
        # d_lora_B: [E, r, out] = [E, r, in] @ [E, in, out]
        d_lora_B = (
            paddle.bmm(lora_A.cast("float32").transpose([0, 2, 1]), dw_f32)
            * scaling
        )
        # d_lora_A: [E, in, r] = [E, in, out] @ [E, out, r]
        d_lora_A = (
            paddle.bmm(dw_f32, lora_B.cast("float32").transpose([0, 2, 1]))
            * scaling
        )

        if not hasattr(self, "_lora_grad_log_count"):
            self._lora_grad_log_count = 0
        if self._lora_grad_log_count < 3:
            self._lora_grad_log_count += 1
            import logging as _logging

            _log = _logging.getLogger(__name__)
            _log.info(
                f"[LORA GRAD EP] step={self._lora_grad_log_count}: "
                f"dw norm={float(dw_f32.norm()):.6f} "
                f"d_lora_A norm={float(d_lora_A.norm()):.6f} amax={float(d_lora_A.abs().max()):.6f} "
                f"d_lora_B norm={float(d_lora_B.norm()):.6f} amax={float(d_lora_B.abs().max()):.6f}"
            )

        def _accumulate(param, dgrad):
            dgrad = dgrad.cast(param.dtype)
            if hasattr(param, "main_grad"):
                if param.main_grad is None:
                    param.main_grad = paddle.zeros(
                        param.shape, dtype=paddle.float32
                    )
                param.main_grad.add_(dgrad.cast(paddle.float32))
            else:
                if param.grad is None:
                    param.grad = paddle.zeros(param.shape, dtype=paddle.float32)
                param.grad.add_(dgrad.cast(paddle.float32))

        _accumulate(lora_A, d_lora_A)
        _accumulate(lora_B, d_lora_B)

    def backward_impl_bf16(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        backward_impl_bf16
        """
        if a2a_async_fn is not None:
            raise NotImplementedError(
                "bf16 fuse node do not support a2a_async_fn currently"
            )
        # Detect LoRA on grouped_gemm_experts
        _ge = (
            getattr(self, "grouped_gemm_experts", None)
            if self.moe_grouped_gemm
            else None
        )
        _has_lora = (
            _ge is not None
            and hasattr(_ge, "get_delta_weight")
            and not getattr(_ge, "disable_lora", False)
            and not getattr(_ge, "merged", False)
        )

        if self.moe_grouped_gemm and not self.use_fp8_mlp:
            if _has_lora:
                expert_w1 = _ge.weight1 + _ge.get_delta_weight(
                    _ge.weight1_lora_A, _ge.weight1_lora_B
                )
                expert_w2 = _ge.weight2 + _ge.get_delta_weight(
                    _ge.weight2_lora_A, _ge.weight2_lora_B
                )
            else:
                expert_w1 = self.grouped_gemm_experts.weight1
                expert_w2 = self.grouped_gemm_experts.weight2
        else:
            expert_w2 = [
                x.down_proj.weight for x in self.experts if x is not None
            ]
            expert_w1 = [
                x.up_gate_proj.weight for x in self.experts if x is not None
            ]
        if self.recompute_moe_gate_up:
            o1 = self.fwd_gate_up(
                None, expert_w1, len(expert_w1), self.tokens_per_expert
            )
        else:
            o1 = self.o1

        do1, o2_s, probs_grad = self.bwd_down_input_bf16(
            expert_w2, out_grad, o1, unzipped_probs
        )
        del o1
        self.o1 = None

        # dw1 / lora grads for w1
        if _has_lora and self.moe_grouped_gemm:
            # compute dw_eff into a temporary tensor instead of accumulating to frozen weight
            if self.input is not None:
                _input = self.input
            elif self.dequant_input and self.input_fp8 is not None:
                _input = paddle.incubate.nn.functional.fused_act_dequant(
                    self.input_fp8, self.input_scale
                )
            else:
                _input = None
            if _input is not None and _input.shape[0] > 0:
                dw1 = paddle.incubate.nn.functional.batched_gemm(
                    _input, do1, self.tokens_per_expert, trans_lhs=True
                )
                self._lora_weight_grad(
                    dw1, _ge.weight1_lora_A, _ge.weight1_lora_B, _ge.scaling
                )
            self.input = None
        else:
            self.bf16_weight_grad(do1, self.input, expert_w1)
            self.input = None

        # dw2 / lora grads for w2
        if _has_lora and self.moe_grouped_gemm:
            if o2_s is not None and o2_s.shape[0] > 0:
                dw2 = paddle.incubate.nn.functional.batched_gemm(
                    o2_s, out_grad, self.tokens_per_expert, trans_lhs=True
                )
                self._lora_weight_grad(
                    dw2, _ge.weight2_lora_A, _ge.weight2_lora_B, _ge.scaling
                )
        else:
            self.bf16_weight_grad(out_grad, o2_s, expert_w2)

        # dx
        dx = self.bwd_gate_up_input_bf16(do1, expert_w1)
        del do1
        self.reset_state()
        return dx, probs_grad

    def backward_impl(self, out_grad, unzipped_probs, a2a_async_fn=None):
        if not self.use_fp8_mlp:
            return self.backward_impl_bf16(
                out_grad, unzipped_probs, a2a_async_fn
            )
        else:
            return self.backward_impl_fp8(
                out_grad, unzipped_probs, a2a_async_fn
            )

    def backward_impl_fp8(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        backward_impl
        """
        # recompute expert_w2 and expert_w1
        expert_w2 = [x.down_proj.weight for x in self.experts if x is not None]
        expert_w1 = [
            x.up_gate_proj.weight for x in self.experts if x is not None
        ]

        if self.recompute_moe_gate_up:
            o1 = self.fwd_gate_up(
                None, expert_w1, len(expert_w1), self.tokens_per_expert
            )
        else:
            o1 = self.o1

        # do2
        do1, o2_s, probs_grad = self.bwd_down_input_fp8(
            expert_w2, out_grad, o1, unzipped_probs, inplace_swiglu_prob=True
        )
        # del o1 时机：
        #   inplace（USE_INPLACE_SWIGLU_BWD=True）：do1 与 o1 共用 buffer，refcount
        #     不归零，立即 del 安全。
        #   out-of-place（USE_INPLACE_SWIGLU_BWD=False）：do1 是独立 buffer，GPU 异步
        #     kernel 仍在读 o1，必须等 bwd_gate_up_input_fp8 的 synchronize 后再 del。
        if USE_INPLACE_SWIGLU_BWD:
            del o1
        self.o1 = None

        if a2a_async_fn is None:
            # dw1
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(do1, None, expert_w1)
            else:
                self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
            # 不调用 _record_stream，直接 None。
            # _record_stream 会触发 VMM 积极回收物理页，在 nparts loop 中
            # slice 被释放后原始 input_fp8 的物理页可能被提前回收，
            # 导致后续 npart 访问时 CUDA_ERROR_ILLEGAL_ADDRESS。
            self.input_fp8 = None
            self.input_scale = None
            self.input = None

            # dw2
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(out_grad, o2_s, expert_w2)
            else:
                self.bwd_down_weight(out_grad, o2_s, expert_w2)

            # dx
            dx = self.bwd_gate_up_input_fp8(do1, expert_w1, dx=out_grad)
            # out-of-place 路径下 fused_swiglu_weighted_bwd 异步读 o1，但此时
            # 中间已经执行了 dw1、dw2 等多个 GEMM kernel（同一 stream 顺序入队），
            # 到达此处时 o1 的读取早已完成，del 安全。
            if not USE_INPLACE_SWIGLU_BWD:
                del o1
            del do1
        else:
            # 为了更充分地overlap, 将dx提前。不过这样可能会增加峰值显存。

            # dx
            dx = self.bwd_gate_up_input_fp8(do1, expert_w1, dx=out_grad)

            dx, task = a2a_async_fn(dx)
            # dw1
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(do1, None, expert_w1)
            else:
                self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
            self.input_fp8 = None
            self.input_scale = None
            self.input = None
            del do1

            # dw2
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(out_grad, o2_s, expert_w2)
            else:
                self.bwd_down_weight(out_grad, o2_s, expert_w2)

            task.wait()
            # task.wait() 后所有异步 kernel（含 out-of-place 路径下
            # fused_swiglu_weighted_bwd 对 o1 的读取）已完成，安全释放。
            if not USE_INPLACE_SWIGLU_BWD:
                del o1

        self.reset_state()
        return dx, probs_grad

    def bf16_weight_grad(self, dy, x, weights):
        """
        BF16 GEMM for weight grad
        """
        if x is None:
            if self.dequant_input:
                x = paddle.incubate.nn.functional.fused_act_dequant(
                    self.input_fp8, self.input_scale
                )
            else:
                x = self.input

        if self.moe_grouped_gemm and not self.use_fp8_mlp:
            if hasattr(weights, "main_grad"):
                if weights.main_grad is None:
                    weights.main_grad = paddle.zeros(
                        weights.shape, dtype=paddle.float32
                    )
                if self.moe_deep_gemm:
                    paddlefleet_deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                        a=x,
                        b=dy,
                        d=weights.main_grad,
                        ks=self.tokens_per_expert,
                        ks_tensor=paddle.to_tensor(
                            self.tokens_per_expert, dtype="int32"
                        ),
                        c=weights.main_grad,
                    )
                else:
                    weights_res = paddle.incubate.nn.functional.batched_gemm(
                        x,
                        dy,
                        self.tokens_per_expert,
                        trans_lhs=True,
                    )
                    weights.main_grad.add_(
                        weights_res.cast(weights.main_grad.dtype)
                    )
            else:
                if weights.grad is None:
                    weights.grad = paddle.zeros(
                        weights.shape, dtype=paddle.float32
                    )
                if self.moe_deep_gemm:
                    paddlefleet_deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                        a=x,
                        b=dy,
                        d=weights.grad,
                        ks=self.tokens_per_expert,
                        ks_tensor=paddle.to_tensor(
                            self.tokens_per_expert, dtype="int32"
                        ),
                        c=weights.grad,
                    )
                else:
                    weights_res = paddle.incubate.nn.functional.batched_gemm(
                        x,
                        dy,
                        self.tokens_per_expert,
                        trans_lhs=True,
                    )
                    weights.grad.add_(weights_res.cast(weights.grad.dtype))
            if (
                hasattr(weights, "_apply_backward_hook")
                and not weights.stop_gradient
            ):
                weights._apply_backward_hook()
        else:
            start_idx = 0
            for i, n in enumerate(self.tokens_per_expert):
                if hasattr(weights[i], "main_grad"):
                    if weights[i].main_grad is None:
                        weights[i].main_grad = paddle.zeros(
                            weights[i].shape, dtype=paddle.float32
                        )
                    grad_attr = weights[i].main_grad
                else:
                    if weights[i].grad is None:
                        weights[i].grad = paddle.zeros(
                            weights[i].shape, dtype=paddle.float32
                        )
                    grad_attr = weights[i].grad

                if n > 0:
                    n = (n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN
                    end_idx = start_idx + n
                    paddle._C_ops.fused_linear_param_grad_add(
                        x._slice(start_idx, end_idx),
                        dy._slice(start_idx, end_idx),
                        grad_attr,
                        None,
                        True,
                        False,
                    )
                    start_idx = end_idx

                if (
                    hasattr(weights[i], "_apply_backward_hook")
                    and not weights[i].stop_gradient
                ):
                    weights[i]._apply_backward_hook()
