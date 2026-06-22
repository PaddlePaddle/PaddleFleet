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


from copy import deepcopy

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.moe.fp8_utils import (
    fused_stack_quant,
    fused_stack_quant_without_cache,
    kitchen_gemm,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

try:
    from paddlefleet_ops import (
        deep_gemm as paddlefleet_deep_gemm,
        fuse_weighted_swiglu_fp8_quant,
        fuse_weighted_swiglu_fp8_quant_clamp,
        fused_swiglu_weighted_clamp_bwd,
    )
except (ImportError, RuntimeError):
    paddlefleet_deep_gemm = None
    fuse_weighted_swiglu_fp8_quant = None
    fuse_weighted_swiglu_fp8_quant_clamp = None
    fused_swiglu_weighted_clamp_bwd = None


class SharedExpertFP8PyLayer(paddle.autograd.PyLayer):
    """PyLayer for Shared Expert FP8 forward/backward with custom gradient computation.

    Reuses pre-quantized FP8 caches passed from the Layer:
      - ``w1_fp8_t`` / ``w1_scale_t``: transpose=True quant of W1 (used in
        forward W1 GEMM and backward dx GEMM).
      - ``w2_fp8_t`` / ``w2_scale_t``: transpose=True quant of W2 (used in
        forward W2 GEMM).
      - ``w2_fp8`` / ``w2_scale``: transpose=False quant of W2 (used in
        backward do2_s GEMM).
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        w1,
        w2,
        w1_fp8_t,
        w1_scale_t,
        w1_fp8,
        w1_scale,
        w2_fp8_t,
        w2_scale_t,
        w2_fp8,
        w2_scale,
        clamp_value,
        fp8_wgrad,
    ):
        # 1. Quantize input to FP8
        x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            hidden_states,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
            using_ue8m0_scale=False,
        )
        x_scale = x_scale.T

        # 2. W1 GEMM: intermediate = x @ W1
        # fp8_gemm_nt: out[S, n] = A[S, k] @ B[n, k]^T
        # w1_fp8_t after transpose=True of [hidden, 2*inter]: shape = [2*inter, hidden]
        # So n=2*inter, k=hidden -> out = [S, 2*inter]
        intermediate = paddle.empty(
            [x_fp8.shape[0], w1_fp8_t.shape[0]], dtype=hidden_states.dtype
        )
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (x_fp8, x_scale),
            (w1_fp8_t.contiguous(), w1_scale_t.contiguous()),
            intermediate,
        )

        # 3. Fused SwiGLU + FP8 quant (prob=ones)
        S = intermediate.shape[0]
        ones = paddle.ones([S, 1], dtype="float32")
        if clamp_value is not None and clamp_value > 0:
            i_fp8, i_scale = fuse_weighted_swiglu_fp8_quant_clamp(
                intermediate,
                ones,
                using_pow2_scaling=True,
                use_ue8m0=False,
                clamp_value=float(clamp_value),
            )
        else:
            i_fp8, i_scale = fuse_weighted_swiglu_fp8_quant(
                intermediate,
                ones,
                using_pow2_scaling=True,
                use_ue8m0=False,
            )
        i_scale = paddle.transpose(
            paddle.transpose(i_scale, [1, 0]).contiguous(), [1, 0]
        )

        # 4. W2 GEMM: output = activated @ W2
        # fp8_gemm_nt: out[S, n] = A[S, k] @ B[n, k]^T
        # i_fp8: [S, inter], w2_fp8_t: [hidden, inter] -> out: [S, hidden]
        output = paddle.empty(
            [i_fp8.shape[0], w2_fp8_t.shape[0]], dtype=hidden_states.dtype
        )
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (i_fp8, i_scale),
            (w2_fp8_t.contiguous(), w2_scale_t.contiguous()),
            output,
        )

        # Save for backward (tensors via save_for_backward, others via attrs)
        ctx.save_for_backward(hidden_states, intermediate)
        ctx.w1 = w1
        ctx.w2 = w2
        ctx.w1_fp8 = w1_fp8
        ctx.w1_scale = w1_scale
        ctx.w2_fp8 = w2_fp8
        ctx.w2_scale = w2_scale
        ctx.clamp_value = clamp_value
        ctx.fp8_wgrad = fp8_wgrad
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, o1 = ctx.saved_tensor()
        w1, w2 = ctx.w1, ctx.w2
        clamp_value = ctx.clamp_value
        S = grad_output.shape[0]

        # Step A: do2_s = grad_output @ W2 (FP8 GEMM for down_proj input grad)
        # Reuse cached non-transposed W2: shape [intermediate_size, hidden_size]
        w2_bwd_fp8, w2_bwd_scale = ctx.w2_fp8, ctx.w2_scale
        grad_out_fp8, grad_out_scale = (
            paddle.incubate.nn.functional.fp8_quant_blockwise(
                grad_output,
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=False,
                using_ue8m0_scale=False,
            )
        )
        grad_out_scale = grad_out_scale.T
        # fp8_gemm_nt: input[S, N] @ weight[K, N]^T -> output[S, K]
        # grad_output[S, hidden_size] @ w2_bwd[intermediate_size, hidden_size]^T -> [S, intermediate_size]
        do2_s = paddle.empty([S, w2_bwd_fp8.shape[0]], dtype=grad_output.dtype)
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (grad_out_fp8, grad_out_scale),
            (w2_bwd_fp8.contiguous(), w2_bwd_scale.contiguous()),
            do2_s,
        )

        # Step B: SwiGLU backward -> do1, o2_s
        ones = paddle.ones([S, 1], dtype="float32")
        if clamp_value is not None and clamp_value > 0:
            do1, _, o2_s = fused_swiglu_weighted_clamp_bwd(
                o1, ones, do2_s, float(clamp_value)
            )
        else:
            # No clamp: use fused_swiglu_weighted_bwd
            do1, _, o2_s = (
                paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                    o1, do2_s, ones
                )
            )

        # Step C: dW2 = o2_s^T @ grad_output
        # Select gradient buffer: prefer main_grad, fallback to .grad
        if hasattr(w2, "main_grad"):
            if w2.main_grad is None:
                w2.main_grad = paddle.zeros(w2.shape, dtype=paddle.float32)
            w2_grad_buf = w2.main_grad
        else:
            if w2.grad is None:
                w2.grad = paddle.zeros(w2.shape, dtype=paddle.float32)
            w2_grad_buf = w2.grad

        if ctx.fp8_wgrad:
            o2_t_fp8, o2_t_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    o2_s,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=True,
                    using_ue8m0_scale=False,
                )
            )
            go_t_fp8, go_t_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    grad_output,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=True,
                    using_ue8m0_scale=False,
                )
            )
            kitchen_gemm(
                o2_t_fp8,
                o2_t_scale,
                go_t_fp8,
                go_t_scale,
                True,
                True,
                w2_grad_buf,
                paddle.float32,
            )
        else:
            paddle._C_ops.fused_linear_param_grad_add(
                o2_s, grad_output, w2_grad_buf, None, True, False
            )
        if hasattr(w2, "_apply_backward_hook") and not w2.stop_gradient:
            w2._apply_backward_hook()

        # Step D: dW1 = hidden_states^T @ do1
        # Select gradient buffer: prefer main_grad, fallback to .grad
        if hasattr(w1, "main_grad"):
            if w1.main_grad is None:
                w1.main_grad = paddle.zeros(w1.shape, dtype=paddle.float32)
            w1_grad_buf = w1.main_grad
        else:
            if w1.grad is None:
                w1.grad = paddle.zeros(w1.shape, dtype=paddle.float32)
            w1_grad_buf = w1.grad

        if ctx.fp8_wgrad:
            hs_t_fp8, hs_t_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    hidden_states,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=True,
                    using_ue8m0_scale=False,
                )
            )
            do1_t_fp8, do1_t_scale = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    do1,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=True,
                    using_ue8m0_scale=False,
                )
            )
            kitchen_gemm(
                hs_t_fp8,
                hs_t_scale,
                do1_t_fp8,
                do1_t_scale,
                True,
                True,
                w1_grad_buf,
                paddle.float32,
            )
        else:
            paddle._C_ops.fused_linear_param_grad_add(
                hidden_states, do1, w1_grad_buf, None, True, False
            )
        if hasattr(w1, "_apply_backward_hook") and not w1.stop_gradient:
            w1._apply_backward_hook()

        # Step E: dx = do1 @ W1^T (FP8 GEMM for input grad)
        # fp8_gemm_nt: do1[S, 2*inter] @ W1[hidden, 2*inter]^T -> [S, hidden]
        # Need non-transposed W1: shape [hidden, 2*inter]
        w1_bwd_fp8, w1_bwd_scale = ctx.w1_fp8, ctx.w1_scale
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
            using_ue8m0_scale=False,
        )
        do1_scale = do1_scale.T
        # fp8_gemm_nt: do1[S, 2*inter] @ w1_bwd[hidden, 2*inter]^T = do1 @ [2*inter, hidden] -> [S, hidden]
        dx = paddle.empty([S, w1_bwd_fp8.shape[0]], dtype=grad_output.dtype)
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (do1_fp8, do1_scale),
            (w1_bwd_fp8.contiguous(), w1_bwd_scale.contiguous()),
            dx,
        )

        # Backward precision comparison (debug only)

        # Grads for tensor inputs:
        # hidden_states, w1, w2, w1_fp8_t, w1_scale_t, w1_fp8, w1_scale, w2_fp8_t, w2_scale_t, w2_fp8, w2_scale
        return dx, None, None, None, None, None, None, None, None, None, None


class StandardMLPSharedExpert(MLP):
    def __init__(
        self,
        config: TransformerConfig,
        moe_intermediate_size: int,
        is_expert: bool,
        mlp_spec: MLPSublayersSpec,
    ):
        if moe_intermediate_size == config.intermediate_size:
            super().__init__(
                config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        else:
            # Local SequentialMLP can still be used here by overriding the intermediate_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.intermediate_size = moe_intermediate_size
            super().__init__(
                sequential_mlp_config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        self.use_shared_expert_gate = config.moe_shared_expert_gate
        if self.use_shared_expert_gate:
            self.gate_weight = paddle.create_parameter(
                shape=[config.hidden_size, 1],
                dtype=config.params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            # Initialize with Normal distribution aligned with Megatron.
            config.init_method(self.gate_weight)
        else:
            self.gate_weight = None

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        use_fp8_forward = getattr(self, "_shared_expert_fp8", False)
        if use_fp8_forward:
            return self._fp8_forward(hidden_states)

        output, output_bias = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = F.linear(hidden_states, self.gate_weight)
            gate_score = F.sigmoid(logits)
            output = output * gate_score
        return output, output_bias

    def _quantize_weights(self):
        """Quantize BF16 weights to FP8 and cache on the parameter.

        Aligned with routed experts: caches both non-transposed and transposed
        versions so that ``fused_stack_quant`` can detect the cache via
        ``hasattr(weight, 'fp8_weight_stacked')``.
        """
        w1 = self.up_gate_proj.weight
        w1_fp8, w1_scale = fused_stack_quant_without_cache(
            [w1], transpose=False, use_ue8m0=False
        )
        w1.fp8_weight_stacked = w1_fp8
        w1.fp8_scale_stacked = w1_scale
        w1_fp8_t, w1_scale_t = fused_stack_quant_without_cache(
            [w1], transpose=True, use_ue8m0=False
        )
        w1.fp8_weight_stacked_transpose = w1_fp8_t
        w1.fp8_scale_stacked_transpose = w1_scale_t

        w2 = self.down_proj.weight
        w2_fp8, w2_scale = fused_stack_quant_without_cache(
            [w2], transpose=False, use_ue8m0=False
        )
        w2.fp8_weight_stacked = w2_fp8
        w2.fp8_scale_stacked = w2_scale
        w2_fp8_t, w2_scale_t = fused_stack_quant_without_cache(
            [w2], transpose=True, use_ue8m0=False
        )
        w2.fp8_weight_stacked_transpose = w2_fp8_t
        w2.fp8_scale_stacked_transpose = w2_scale_t

    def _fp8_forward(self, hidden_states: paddle.Tensor):
        """FP8 forward using fused_stack_quant (aligned with routed experts).

        - Offline (cache exists): reads pre-quantized cache, zero overhead.
        - Online (no cache): fused_stack_quant falls back to on-the-fly quantization.
        """
        if paddlefleet_deep_gemm is None:
            raise RuntimeError(
                "deep_gemm is not available for FP8 shared expert forward"
            )
        if getattr(self.config, "use_bias", False):
            raise ValueError(
                "Bias is not supported in FP8 shared expert yet, "
                "please set 'use_bias' to False."
            )

        orig_shape = hidden_states.shape
        if len(orig_shape) == 3:
            hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])

        w1 = self.up_gate_proj.weight
        w2 = self.down_proj.weight

        w1_fp8_t, w1_scale_t = fused_stack_quant(
            [w1], transpose=True, use_ue8m0=False
        )
        w1_fp8, w1_scale = fused_stack_quant(
            [w1], transpose=False, use_ue8m0=False
        )
        w2_fp8_t, w2_scale_t = fused_stack_quant(
            [w2], transpose=True, use_ue8m0=False
        )
        w2_fp8, w2_scale = fused_stack_quant(
            [w2], transpose=False, use_ue8m0=False
        )

        clamp_value = getattr(self.config, "activation_func_clamp_value", None)
        fp8_wgrad = getattr(self.config, "fp8_wgrad", False)

        output = SharedExpertFP8PyLayer.apply(
            hidden_states,
            w1,
            w2,
            w1_fp8_t,
            w1_scale_t,
            w1_fp8,
            w1_scale,
            w2_fp8_t,
            w2_scale_t,
            w2_fp8,
            w2_scale,
            clamp_value,
            fp8_wgrad,
        )

        if self.use_shared_expert_gate:
            logits = F.linear(hidden_states, self.gate_weight)
            gate_score = F.sigmoid(logits)
            output = output * gate_score

        if len(orig_shape) == 3:
            output = output.reshape(orig_shape)

        return output, None
