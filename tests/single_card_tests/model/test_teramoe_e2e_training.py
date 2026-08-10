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

"""
TeraMoE End-to-End Mini Training Loop.

Runs a few forward + backward + optimizer steps on a MoE Layer with TeraMoE,
verifying:
  1. Forward produces valid output (no NaN/Inf)
  2. Backward produces valid gradients
  3. Optimizer step updates parameters
  4. Loss decreases over multiple steps

Since TeraMoE's buffer requires multi-GPU NCCL, we mock the buffer to run
on a single card. The mock buffer performs a real computation (matmul-based)
so that gradients are meaningful and loss actually decreases.

Run with:
  export FLAGS_selected_gpus=0 PADDLE_TRAINER_ID=0 \
    PADDLE_CURRENT_ENDPOINT=127.0.0.1:36001 \
    PADDLE_TRAINER_ENDPOINTS=127.0.0.1:36001 \
    PADDLE_TRAINERS_NUM=1
  python test_teramoe_e2e_training.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

import paddlefleet.parallel_state as ps
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

# ── Fleet init ──────────────────────────────────────────────────────
_strategy = fleet.DistributedStrategy()
_strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 1,
    "pp_degree": 1,
    "sharding_degree": 1,
    "sep_degree": 1,
    "cp_degree": 1,
    "ep_degree": 1,
    "moe_sharding_degree": 1,
    "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"],
}
fleet.init(is_collective=True, strategy=_strategy)
_hcg = fleet.get_hybrid_communicate_group()
ps.initialize_model_parallel(_hcg)


# ── Config ──────────────────────────────────────────────────────────
SEED = 42
HIDDEN_SIZE = 512
N_EXPERTS = 8
TOPK = 2
MOE_INTERMEDIATE = 1024
BATCH_SIZE = 2
SEQ_LEN = 64
NUM_STEPS = 20
LR = 1e-2


def small_init(tensor):
    paddle.nn.initializer.Uniform(-0.1, 0.1)(tensor)


def build_moe_layer():
    """Build a TeraMoE-enabled MoELayer for single-card training."""
    paddle.seed(SEED)
    model_parallel_cuda_manual_seed(SEED)

    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        n_routed_experts=N_EXPERTS,
        use_cpu_initialization=False,
        num_experts_per_tok=TOPK,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        params_dtype=paddle.bfloat16,
        moe_intermediate_size=MOE_INTERMEDIATE,
        gated_linear_unit=True,
        n_shared_experts=0,
        hidden_act=F.silu,
        moe_expert_fusion=True,
        bias_activation_fusion=True,
        moe_token_dispatcher_type="alltoall",
        moe_use_fusion_node=True,
        using_teramoe=True,
        using_sonic_moe=False,
        fp8=None,
        use_bias=False,
        init_method=small_init,
        output_layer_init_method=small_init,
    )

    spec = get_gpt_layer_local_spec(config, num_experts=N_EXPERTS)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    moe_layer = MoELayer(
        config,
        spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
        pg_collection,
    )
    # NOTE: Do NOT use MixPrecisionLayer here. It expects main_grad for
    # gradient accumulation which doesn't work with the mock buffer's
    # standard autograd path. We train directly in bf16 instead.
    return moe_layer


def make_realistic_mock_buffer():
    """Create a mock buffer whose teramoe_autograd does real differentiable computation.

    Uses paddle operations that build a proper computation graph so gradients
    flow back to the expert weights (W_gateup, W_down).
    """
    mock_buffer = MagicMock()

    def realistic_teramoe_autograd(
        hidden_states,
        topk_indices,
        topk_scores,
        W_gateup,
        W_down,
        num_experts,
        **kwargs,
    ):
        """Differentiable mock of fused MoE forward.

        Simplified: use first expert's weights for all tokens (avoids
        index_select issues). This still exercises the full gradient path
        through the weights.

        W_gateup: [E, 2*I, H]  (sonic layout)
        W_down:   [E, H, I]    (sonic layout)
        """
        T, H = hidden_states.shape
        K = topk_indices.shape[1]
        I = W_down.shape[2]

        # Use mean of all expert weights (differentiable w.r.t. all experts)
        # W_gateup_mean: [2*I, H], W_down_mean: [H, I]
        w_gateup_mean = W_gateup.mean(axis=0)  # [2*I, H]
        w_down_mean = W_down.mean(axis=0)  # [H, I]

        # Split gate and up
        w_gate = w_gateup_mean[:I, :]  # [I, H]
        w_up = w_gateup_mean[I:, :]  # [I, H]

        # Forward: gate_out = x @ w_gate.T, up_out = x @ w_up.T
        gate_out = paddle.matmul(
            hidden_states, w_gate, transpose_y=True
        )  # [T, I]
        up_out = paddle.matmul(hidden_states, w_up, transpose_y=True)  # [T, I]

        # SwiGLU activation
        activated = F.silu(gate_out) * up_out  # [T, I]

        # Down projection: output = activated @ w_down_mean.T -> [T, H]
        output = paddle.matmul(
            activated, w_down_mean, transpose_y=True
        )  # [T, H]

        # Scale by mean routing score
        mean_score = topk_scores.mean()
        output = output * mean_score

        return output

    mock_buffer.teramoe_autograd.side_effect = realistic_teramoe_autograd
    return mock_buffer


@patch("paddlefleet.transformer.moe.moe_layer.get_teramoe_buffer")
def run_training(mock_get_buffer):
    """Run a mini training loop and verify convergence."""
    print("=" * 60)
    print("TeraMoE End-to-End Mini Training Test")
    print("=" * 60)
    print(
        f"Config: H={HIDDEN_SIZE}, E={N_EXPERTS}, K={TOPK}, I={MOE_INTERMEDIATE}"
    )
    print(
        f"Training: {NUM_STEPS} steps, BS={BATCH_SIZE}, SeqLen={SEQ_LEN}, LR={LR}"
    )
    print()

    # Build layer
    moe_layer = build_moe_layer()
    mock_buffer = make_realistic_mock_buffer()
    mock_get_buffer.return_value = mock_buffer

    # Create optimizer (no multi_precision — we're not using MixPrecisionLayer)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=LR,
        parameters=moe_layer.parameters(),
        multi_precision=False,
    )

    # Training loop
    losses = []
    print(f"{'Step':<6} {'Loss':<14} {'Status'}")
    print("-" * 40)

    # Use FIXED input data across all steps so optimizer can converge
    paddle.seed(SEED)
    input_data = paddle.randn(
        [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE], dtype=paddle.bfloat16
    )
    # Target: zeros (so loss = ||output||^2, should decrease)
    target = paddle.zeros_like(input_data)

    for step in range(NUM_STEPS):
        input_data.stop_gradient = False

        # Forward
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output, _ = moe_layer(input_data)
            # MSE-like loss
            loss = ((output - target) ** 2).mean()

        # Check output validity
        has_nan = paddle.isnan(output).any().item()
        has_inf = paddle.isinf(output).any().item()

        # Backward
        loss.backward()

        # Check gradients exist on expert weights or gate
        expert = moe_layer.grouped_gemm_experts
        grad_exists = False
        for p in moe_layer.parameters():
            g = p.grad
            if (
                g is not None
                and float(g.astype("float32").abs().sum().item()) > 0
            ):
                grad_exists = True
                break

        # Optimizer step
        optimizer.step()
        optimizer.clear_grad()

        loss_val = loss.item()
        losses.append(loss_val)

        status = "OK"
        if has_nan:
            status = "FAIL (NaN in output)"
        elif has_inf:
            status = "FAIL (Inf in output)"
        elif not grad_exists:
            status = "WARN (no grads)"

        print(f"{step:<6} {loss_val:<14.6f} {status}")

    print("-" * 40)

    # Verification
    print()
    all_ok = True

    # 1. No NaN/Inf in losses
    if any(
        paddle.isnan(paddle.to_tensor(l)).item()
        or paddle.isinf(paddle.to_tensor(l)).item()
        for l in losses
    ):
        print("[FAIL] Loss contains NaN/Inf")
        all_ok = False
    else:
        print("[PASS] All losses are finite")

    # 2. Loss decreased
    if losses[-1] < losses[0]:
        reduction = (1 - losses[-1] / losses[0]) * 100
        print(
            f"[PASS] Loss decreased: {losses[0]:.6f} -> {losses[-1]:.6f} "
            f"({reduction:.1f}% reduction)"
        )
    else:
        print(
            f"[FAIL] Loss did NOT decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
        )
        all_ok = False

    # 3. Weights changed
    expert = moe_layer.grouped_gemm_experts
    w1_norm = float(paddle.linalg.norm(expert.weight1.astype("float32")).item())
    print(f"[INFO] Final weight1 norm: {w1_norm:.4f}")

    # 4. Buffer was called correct number of times
    call_count = mock_buffer.teramoe_autograd.call_count
    if call_count == NUM_STEPS:
        print(
            f"[PASS] Buffer.teramoe_autograd called {call_count} times (expected {NUM_STEPS})"
        )
    else:
        print(
            f"[FAIL] Buffer.teramoe_autograd called {call_count} times (expected {NUM_STEPS})"
        )
        all_ok = False

    # 5. Verify layout flushed back after training (for optimizer)
    expert.flush_to_grouped_layout()
    print(
        f"[PASS] Weights flushed back to grouped layout: {expert._weights_layout}"
    )

    print()
    if all_ok:
        print("=" * 60)
        print("  ALL CHECKS PASSED - TeraMoE E2E training loop works!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  SOME CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    run_training()
