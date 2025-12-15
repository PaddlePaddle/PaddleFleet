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

try:
    from paddle.incubate.nn.functional import swiglu
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
    from paddle.incubate.fp8 import deep_gemm
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
    if transpose:
        fp8_weight = weight.fp8_weight_stacked_transpose
        fp8_scale = weight.fp8_scale_stacked_transpose
    else:
        fp8_weight, fp8_scale = (
            weight.fp8_weight_stacked,
            weight.fp8_scale_stacked,
        )

    return fp8_weight, fp8_scale


def fused_stack_quant(expert_weight_list, transpose=False):
    if transpose is False and hasattr(
        expert_weight_list[0], "fp8_weight_stacked"
    ):
        w, scale = _get_fp8_weight_and_scale(
            expert_weight_list[0], transpose=False
        )
    elif transpose is True and hasattr(
        expert_weight_list[0], "fp8_weight_stacked_transpose"
    ):
        w, scale = _get_fp8_weight_and_scale(
            expert_weight_list[0], transpose=True
        )
    elif transpose is True and hasattr(
        expert_weight_list[0], "fp8_weight_stacked"
    ):
        w, scale = _get_fp8_weight_and_scale(
            expert_weight_list[0], transpose=False
        )
    elif transpose is False and hasattr(
        expert_weight_list[0], "fp8_weight_stacked_transpose"
    ):
        w, scale = _get_fp8_weight_and_scale(
            expert_weight_list[0], transpose=True
        )
    else:
        w, scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            expert_weight_list, transpose=transpose
        )
    return w, scale


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

        x_scale_tma_align = x_scale[start_idx:end_idx].T.contiguous().T

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x_fp8[start_idx:end_idx], x_scale_tma_align),
            (w_fp8[i], w_scale[i]),
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
        recompute_fwd_gate_up=False,
        dequant_input=False,
        group=None,
        name="experts_group_gemm_contiguous_node",
        expert_id=None,
        backward_subbatch_rows=None,
        use_bf16_gemm_weight_grad=False,
        use_fp8_mlp=True,
    ):
        """
            Initializes the experts group gemm contiguous node.

        Args:
            custom_map (CustomMapping): Custom mapping for the model.
            recompute_fwd_gate_up (bool, optional): Whether to recompute forward gate up. Defaults to False.
            dequant_input (bool, optional): Whether to dequantize input. Defaults to False.
            name (str, optional): Name of the node. Defaults to "experts_group_gemm_contiguous_node".
        """
        if expert_id is None:
            self.experts = custom_map.experts
        else:
            self.experts = [custom_map.experts[expert_id]]
        self.expert_id = expert_id
        self.recompute_fwd_gate_up = recompute_fwd_gate_up
        self.dequant_input = dequant_input
        self.tokens_per_expert = None
        self.m_indices = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None
        self.fp8_fused_ops_configs = {}
        self.is_split_group_gemm = True
        # self.is_split_group_gemm = has_config(self.fp8_fused_ops_configs, "split_group_gemm")
        self.group = group
        self.backward_subbatch_rows = backward_subbatch_rows
        if self.backward_subbatch_rows is not None:
            assert (
                self.backward_subbatch_rows > 0
                and self.backward_subbatch_rows % FP8_ALIGN == 0
            ), self.backward_subbatch_rows
        self.use_bf16_gemm_weight_grad = use_bf16_gemm_weight_grad
        self.use_fp8_mlp = use_fp8_mlp

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
        self.set_cached_tensors([None] * len(self.cached_tensors))

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
        if self.is_split_group_gemm is False:
            raise NotImplementedError(
                "fuse node do not support group gemm currently"
            )

        if x is None:
            assert self.input is not None
            x = self.input
        if numpy.prod(x.shape) != 0:
            expert_w1 = paddle.stack(expert_w1, axis=0)
            o1 = paddle.incubate.nn.functional.batched_gemm(
                x,
                expert_w1,
                self.tokens_per_expert,
            )
        else:
            o1 = paddle.empty(
                [x.shape[0], expert_w1[0].shape[1]], dtype=expert_w1[0].dtype
            )
        self.input = x
        return o1

    def fwd_gate_up(
        self, x, expert_w1, num_expert, tokens_per_expert, scale=None
    ):
        self.tokens_per_expert = tokens_per_expert
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

        if not self.is_split_group_gemm:
            self.m_indices = self.gen_m_indices(tokens_per_expert)
        # concat w1, shape is [num_groups, n, k]
        w1_t_quant, w1_t_scale = fused_stack_quant(expert_w1, transpose=True)
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        if hasattr(expert_w1[0], "fp8_weight_stacked") and not hasattr(
            expert_w1[0], "fp8_weight_stacked_transpose"
        ):
            w1_t_quant = (
                w1_t_quant.contiguous().transpose([0, 2, 1]).contiguous()
            )
            w1_t_scale = (
                w1_t_scale.contiguous().transpose([0, 2, 1]).contiguous()
            )

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
            if self.is_split_group_gemm:
                split_group_gemm(
                    x_fp8,
                    x_scale,
                    w1_t_quant,
                    w1_t_scale,
                    tokens_per_expert,
                    o1,
                )
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
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
        if self.is_split_group_gemm is False:
            raise NotImplementedError(
                "fuse node do not support group gemm currently"
            )

        # swiglu
        o2 = self.fwd_swiglu(o1)

        unzipped_probs = unzipped_probs.unsqueeze(-1)
        o2 = (o2 * unzipped_probs).cast(paddle.bfloat16)

        if clear_o1:
            o1._clear_to_zero_allocation()

        # down proj
        if numpy.prod(o2.shape) != 0:
            expert_w2 = paddle.stack(expert_w2, axis=0)
            o3 = paddle.incubate.nn.functional.batched_gemm(
                o2,
                expert_w2,
                self.tokens_per_expert,
            )

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

        # quant o2
        with paddle.amp.auto_cast(False):
            unzipped_probs = unzipped_probs.squeeze(-1)
            o2_fp8, o2_scale = (
                paddle.incubate.nn.functional.fused_weighted_swiglu_act_quant(
                    o1, unzipped_probs, using_pow2_scaling=True
                )
            )
        o2_scale = paddle.transpose(
            paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0]
        )

        if clear_o1:
            o1._clear_to_zero_allocation()

        # compute gemm
        o3_shape = [o2_fp8.shape[0], w2_quant.shape[1]]
        if o3 is not None:
            assert o3.shape == o3_shape, f"{o3.shape} vs {o3_shape}"
            o3.zero_()
        else:
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    o2_fp8,
                    o2_scale,
                    w2_quant,
                    w2_scale,
                    self.tokens_per_expert,
                    o3,
                )
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
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
            expert_w2 = paddle.stack([t.T for t in expert_w2], axis=0)
            do2_s = paddle.incubate.nn.functional.batched_gemm(
                unzipped_grad,
                expert_w2,
                self.tokens_per_expert,
            )
        else:
            do2_s_shape = [unzipped_grad.shape[0], expert_w2[0].shape[1]]
            do2_s = paddle.empty(do2_s_shape, dtype=unzipped_grad.dtype)

        # recompute o2
        o2 = self.fwd_swiglu(o1)
        o2_s = (o2 * unzipped_probs).cast(paddle.bfloat16)
        # do2: 前向从bfloat16-->float32，反向从float32-->bfloat16,do2 需要保持 bfloat16（因为 o2 是 bfloat16)
        do2 = (do2_s.cast(paddle.float32) * unzipped_probs).cast(
            paddle.bfloat16
        )

        # probs_grad: probs_grad 需要保持 float32（因为 unzipped_probs 是 float32）
        probs_grad = (
            do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))
        ).sum(axis=-1)
        # do1
        do1 = self.bwd_swiglu(o1, do2)

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
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=False,
            )
        )
        unzipped_grad_scale = unzipped_grad_scale.T

        do2_s = paddle.empty(
            [unzipped_grad_fp8.shape[0], bw_w2_quant.shape[1]],
            dtype=unzipped_grad.dtype,
        )
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    unzipped_grad_fp8,
                    unzipped_grad_scale,
                    bw_w2_quant,
                    bw_w2_scale,
                    self.tokens_per_expert,
                    do2_s,
                )
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (unzipped_grad_fp8, unzipped_grad_scale),
                    (bw_w2_quant, bw_w2_scale),
                    do2_s,
                    m_indices=self.m_indices,
                )

        with paddle.amp.auto_cast(False):
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
            expert_w1 = paddle.stack([t.T for t in expert_w1], axis=0)
            dx = paddle.incubate.nn.functional.batched_gemm(
                do1,
                expert_w1,
                self.tokens_per_expert,
            )
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

        if hasattr(
            expert_w1[0], "fp8_weight_stacked_transpose"
        ) and not hasattr(expert_w1[0], "fp8_weight_stacked"):
            bw_w1_quant = (
                bw_w1_quant.contiguous().transpose([0, 2, 1]).contiguous()
            )
            bw_w1_scale = (
                bw_w1_scale.contiguous().transpose([0, 2, 1]).contiguous()
            )

        # quant do1
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
        )
        do1_scale = do1_scale.T

        # compute gemm
        dx_shape = [do1_fp8.shape[0], bw_w1_quant.shape[1]]
        if dx is None:
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        else:
            assert dx.shape == dx_shape, f"{dx.shape} vs {dx_shape}"
            dx.zero_()
        if numpy.prod(do1_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    do1_fp8,
                    do1_scale,
                    bw_w1_quant,
                    bw_w1_scale,
                    self.tokens_per_expert,
                    dx,
                )
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
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
        expert_w1 = [
            x.up_gate_proj.weight for x in self.experts if x is not None
        ]
        expert_w2 = [x.down_proj.weight for x in self.experts if x is not None]

        num_expert = len(expert_w1)

        # o1
        o1 = self.fwd_gate_up(
            hs_out, expert_w1, num_expert, tokens_per_expert, scale=scale
        )
        if not self.recompute_fwd_gate_up:
            self.o1 = o1
            clear_o1 = False
        else:
            clear_o1 = True

        # o3
        o3 = self.fwd_down(
            o1, unzipped_probs, expert_w2, num_expert, clear_o1=clear_o1
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

            if a2a_async_fn:
                dx, task = a2a_async_fn(dx)
                task.wait()
            return dx, probs_grad

        subbatch_rows = self.backward_subbatch_rows
        if subbatch_rows is None:
            return self.backward_impl(
                out_grad, unzipped_probs, a2a_async_fn=a2a_async_fn
            )

        assert a2a_async_fn is None, (
            "a2a_async_fn should be None when backward_subbatch_rows is not None"
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
        probs_grad = paddle.concat(probs_grad, axis=0)
        return out_grad, probs_grad

    def backward_impl_bf16(self, out_grad, unzipped_probs, a2a_async_fn=None):
        """
        backward_impl_bf16
        """
        if a2a_async_fn is not None:
            raise NotImplementedError(
                "bf16 fuse node do not support a2a_async_fn currently"
            )
        expert_w2 = [x.down_proj.weight for x in self.experts if x is not None]
        expert_w1 = [
            x.up_gate_proj.weight for x in self.experts if x is not None
        ]
        if self.recompute_fwd_gate_up:
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

        # dw1
        self.bf16_weight_grad(do1, self.input, expert_w1)
        self.input = None

        # dw2
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

        if self.recompute_fwd_gate_up:
            o1 = self.fwd_gate_up(
                None, expert_w1, len(expert_w1), self.tokens_per_expert
            )
        else:
            o1 = self.o1

        # do2
        do1, o2_s, probs_grad = self.bwd_down_input_fp8(
            expert_w2, out_grad, o1, unzipped_probs, inplace_swiglu_prob=True
        )
        del o1
        self.o1 = None

        if a2a_async_fn is None:
            # dw1
            if self.use_bf16_gemm_weight_grad:
                self.bf16_weight_grad(do1, None, expert_w1)
            else:
                self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
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
