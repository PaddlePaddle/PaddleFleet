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
from paddlefleet.transformer.transformer_config import TransformerConfig

# === HACK: Shared Expert FP8 imports ===
from paddlefleet.transformer.moe.fp8_utils import fused_stack_quant_without_cache
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
# === HACK END ===


class SharedExpertFP8PyLayer(paddle.autograd.PyLayer):
    """PyLayer for Shared Expert FP8 forward/backward with custom gradient computation."""

    @staticmethod
    def forward(ctx, hidden_states, w1, w2, w1_fp8, w1_scale, w2_fp8, w2_scale,
                clamp_value):
        # 1. Quantize input to FP8
        x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            hidden_states,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
            using_ue8m0_scale=False,
        )
        x_scale = x_scale.T

        # 2. W1 GEMM: intermediate = x @ W1^T
        # fp8_gemm_nt: out[M, n] = A[M, k] @ B[n, k]^T
        # w1_fp8 after transpose=True of [hidden, 2*inter]: shape = [2*inter, hidden]
        # So n=2*inter, k=hidden -> out = [M, 2*inter]
        intermediate = paddle.empty(
            [x_fp8.shape[0], w1_fp8.shape[0]], dtype=hidden_states.dtype
        )
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (x_fp8, x_scale),
            (w1_fp8.contiguous(), w1_scale.contiguous()),
            intermediate,
        )

        # 3. Fused SwiGLU + FP8 quant (prob=ones)
        M = intermediate.shape[0]
        ones = paddle.ones([M, 1], dtype="float32")
        if clamp_value is not None and clamp_value > 0:
            i_fp8, i_scale = fuse_weighted_swiglu_fp8_quant_clamp(
                intermediate, ones, using_pow2_scaling=True,
                use_ue8m0=False, clamp_value=float(clamp_value),
            )
        else:
            i_fp8, i_scale = fuse_weighted_swiglu_fp8_quant(
                intermediate, ones, using_pow2_scaling=True, use_ue8m0=False,
            )
        i_scale = paddle.transpose(
            paddle.transpose(i_scale, [1, 0]).contiguous(), [1, 0]
        )

        # 4. W2 GEMM: output = activated @ W2^T
        # fp8_gemm_nt: out[M, n] = A[M, k] @ B[n, k]^T
        # i_fp8: [M, inter], w2_fp8: [hidden, inter] -> out: [M, hidden]
        output = paddle.empty(
            [i_fp8.shape[0], w2_fp8.shape[0]], dtype=hidden_states.dtype
        )
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (i_fp8, i_scale),
            (w2_fp8.contiguous(), w2_scale.contiguous()),
            output,
        )

        # Save for backward
        ctx.save_for_backward(hidden_states, intermediate)
        ctx.w1 = w1
        ctx.w2 = w2
        ctx.clamp_value = clamp_value
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, o1 = ctx.saved_tensor()
        w1, w2 = ctx.w1, ctx.w2
        clamp_value = ctx.clamp_value
        M = grad_output.shape[0]

        # Step A: do2_s = grad_output @ W2^T (FP8 GEMM for down_proj input grad)
        # Need W2 in non-transposed FP8 form
        w2_bwd_fp8, w2_bwd_scale = fused_stack_quant_without_cache(
            [w2], transpose=False, use_ue8m0=False
        )
        grad_out_fp8, grad_out_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            grad_output,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
            using_ue8m0_scale=False,
        )
        grad_out_scale = grad_out_scale.T
        # W2 original: [intermediate_size, hidden_size], non-transposed fp8: [intermediate_size, hidden_size]
        # fp8_gemm_nt: input[M, N] @ weight[K, N]^T -> output[M, K]
        # grad_output[M, hidden_size] @ w2_bwd[intermediate_size, hidden_size]^T -> [M, intermediate_size]
        do2_s = paddle.empty(
            [M, w2_bwd_fp8.shape[0]], dtype=grad_output.dtype
        )
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (grad_out_fp8, grad_out_scale),
            (w2_bwd_fp8.contiguous(), w2_bwd_scale.contiguous()),
            do2_s,
        )

        # Step B: SwiGLU backward -> do1, o2_s
        ones = paddle.ones([M, 1], dtype="float32")
        if clamp_value is not None and clamp_value > 0:
            do1, _, o2_s = fused_swiglu_weighted_clamp_bwd(
                o1, ones, do2_s, float(clamp_value)
            )
        else:
            # No clamp: use fused_swiglu_weighted_bwd
            do1, _, o2_s = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                o1, do2_s, ones
            )

        # Step C: dW2 = o2_s^T @ grad_output (BF16 weight grad)
        if hasattr(w2, "main_grad"):
            if w2.main_grad is None:
                w2.main_grad = paddle.zeros(w2.shape, dtype=paddle.float32)
            paddle._C_ops.fused_linear_param_grad_add(
                o2_s, grad_output, w2.main_grad, None, True, False
            )
        if hasattr(w2, "_apply_backward_hook") and not w2.stop_gradient:
            w2._apply_backward_hook()

        # Step D: dW1 = hidden_states^T @ do1 (BF16 weight grad)
        if hasattr(w1, "main_grad"):
            if w1.main_grad is None:
                w1.main_grad = paddle.zeros(w1.shape, dtype=paddle.float32)
            paddle._C_ops.fused_linear_param_grad_add(
                hidden_states, do1, w1.main_grad, None, True, False
            )
        if hasattr(w1, "_apply_backward_hook") and not w1.stop_gradient:
            w1._apply_backward_hook()

        # Step E: dx = do1 @ W1 (FP8 GEMM for input grad)
        # do1: [M, 2*inter], W1 original: [2*inter, hidden]
        # Want: do1 @ W1_orig = [M, 2*inter] @ [2*inter, hidden] -> [M, hidden]
        # fp8_gemm_nt(A, B) = A @ B^T, need B^T = W1 so B = W1^T (which is transpose=True)
        # But: fused_stack_quant(transpose=False) gives shape [2*inter, hidden] (original)
        # fp8_gemm_nt(do1, W1_orig) = do1 @ W1_orig^T = [M, 2*inter] @ [hidden, 2*inter] -> wrong!
        # Actually: fp8_gemm_nt(do1, B) = do1 @ B^T -> [M, B.shape[1]]
        # We need output [M, hidden], so B^T must be [2*inter, hidden], thus B = [hidden, 2*inter]
        # That's transpose=True of original [2*inter, hidden] -> [hidden, 2*inter]
        w1_bwd_fp8, w1_bwd_scale = fused_stack_quant_without_cache(
            [w1], transpose=True, use_ue8m0=False
        )
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
            using_ue8m0_scale=False,
        )
        do1_scale = do1_scale.T
        # fp8_gemm_nt: do1[M, 2*inter] @ w1_bwd[hidden, 2*inter]^T = do1 @ [2*inter, hidden] -> [M, hidden]
        dx = paddle.empty([M, w1_bwd_fp8.shape[0]], dtype=grad_output.dtype)
        paddlefleet_deep_gemm.fp8_gemm_nt(
            (do1_fp8, do1_scale),
            (w1_bwd_fp8.contiguous(), w1_bwd_scale.contiguous()),
            dx,
        )

        # Return grads for: hidden_states, w1, w2, w1_fp8, w1_scale, w2_fp8, w2_scale
        return dx, None, None, None, None, None, None


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
            )
        else:
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.intermediate_size = moe_intermediate_size
            super().__init__(
                sequential_mlp_config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
            )
        self.use_shared_expert_gate = config.moe_shared_expert_gate
        if self.use_shared_expert_gate:
            self.gate_weight = paddle.create_parameter(
                shape=[config.hidden_size, 1],
                dtype=config.params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            config.init_method(self.gate_weight)
        else:
            self.gate_weight = None

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        use_fp8_forward = getattr(self, '_shared_expert_fp8', False)
        if use_fp8_forward:
            fp8_result = self._fp8_forward(hidden_states)
            # Precision comparison mode
            if getattr(self, '_shared_expert_fp8_compare', False):
                import logging as _log
                with paddle.no_grad():
                    h = hidden_states.reshape([-1, hidden_states.shape[-1]]) if hidden_states.ndim == 3 else hidden_states
                    w1 = self.up_gate_proj.weight
                    w2 = self.down_proj.weight
                    clamp_value = getattr(self.config, 'activation_func_clamp_value', None)

                    # BF16 baseline: W1 -> SwiGLU -> W2
                    bf16_intermediate = paddle.matmul(h, w1)
                    hidden_dim = bf16_intermediate.shape[-1] // 2
                    if clamp_value is not None and clamp_value > 0:
                        gate = bf16_intermediate[..., :hidden_dim].cast("float32")
                        val = bf16_intermediate[..., hidden_dim:].cast("float32")
                        gate = paddle.clip(gate, max=clamp_value)
                        val = paddle.clip(val, min=-clamp_value, max=clamp_value)
                        bf16_activated = (F.silu(gate) * val).cast(h.dtype)
                    else:
                        gate = bf16_intermediate[..., :hidden_dim]
                        val = bf16_intermediate[..., hidden_dim:]
                        bf16_activated = F.silu(gate) * val
                    bf16_output = paddle.matmul(bf16_activated, w2)

                    # Compare final output
                    fp8_output = fp8_result[0]
                    fp8_f = fp8_output.cast("float32")
                    bf16_f = bf16_output.reshape(fp8_f.shape).cast("float32")
                    abs_diff = paddle.abs(fp8_f - bf16_f)
                    cos = F.cosine_similarity(fp8_f.reshape([1, -1]), bf16_f.reshape([1, -1])).item()
                    _log.info(f"[SharedExpert FP8 vs BF16] cosine={cos:.6f}, "
                              f"max_abs_diff={abs_diff.max().item():.4e}, "
                              f"mean_abs_diff={abs_diff.mean().item():.4e}, "
                              f"shape={fp8_output.shape}")
            return fp8_result

        output, output_bias = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = F.linear(hidden_states, self.gate_weight)
            gate_score = F.sigmoid(logits)
            output = output * gate_score
        return output, output_bias

    def _quantize_weights(self):
        """Quantize weights to FP8 and attach to weight parameters."""
        w1 = self.up_gate_proj.weight
        w1_fp8, w1_scale = fused_stack_quant_without_cache(
            [w1], transpose=True, use_ue8m0=False
        )
        w1.fp8_weight_stacked = w1_fp8
        w1.fp8_scale_stacked = w1_scale

        w2 = self.down_proj.weight
        w2_fp8, w2_scale = fused_stack_quant_without_cache(
            [w2], transpose=True, use_ue8m0=False
        )
        w2.fp8_weight_stacked = w2_fp8
        w2.fp8_scale_stacked = w2_scale

    def _fp8_forward(self, hidden_states: paddle.Tensor):
        """FP8 forward with custom backward via PyLayer."""
        if paddlefleet_deep_gemm is None:
            raise RuntimeError("deep_gemm is not available for FP8 shared expert forward")

        orig_shape = hidden_states.shape
        if len(orig_shape) == 3:
            hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])

        # Always re-quantize to ensure FP8 weights match current BF16 weights
        self._quantize_weights()

        w1 = self.up_gate_proj.weight
        w2 = self.down_proj.weight
        clamp_value = getattr(self.config, 'activation_func_clamp_value', None)

        output = SharedExpertFP8PyLayer.apply(
            hidden_states,
            w1, w2,
            w1.fp8_weight_stacked, w1.fp8_scale_stacked,
            w2.fp8_weight_stacked, w2.fp8_scale_stacked,
            clamp_value,
        )

        if self.use_shared_expert_gate:
            logits = F.linear(hidden_states, self.gate_weight)
            gate_score = F.sigmoid(logits)
            output = output * gate_score

        if len(orig_shape) == 3:
            output = output.reshape(orig_shape)

        return output, None
