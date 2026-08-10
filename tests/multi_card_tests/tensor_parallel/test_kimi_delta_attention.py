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

"""Distributed precision tests for KimiDeltaAttention (TP / SP correctness).

Verifies that KDA with tensor (and sequence) parallelism reproduces the
single-device TP=1 baseline for the output, the input gradient and every
parameter gradient, and that ``sharded_state_dict`` declares global shapes
matching that baseline. Follows tests/multi_card_tests/tensor_parallel/
test_gated_delta_net.py.

The interesting cases beyond GDN are the low-rank gate pairs
(f_a_proj / f_b_proj and, when use_full_rank_gate=False, g_a_proj / g_b_proj):
a_proj is replicated and only sees the local sequence shard under SP, so its
weight gradient is a partial sum that must be all-reduced across TP.

Launch:
    python -m paddle.distributed.launch --gpus="0,1,2,3" \
        tensor_parallel/test_kimi_delta_attention.py
"""

from __future__ import annotations

import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)

import paddlefleet.parallel_state as ps
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from paddlefleet.tensor_parallel.mappings import (
    _gather_along_first_dim,
    _gather_along_last_dim,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.kimi_delta_attention import (
    HAVE_FLA,
    KimiDeltaAttention,
    KimiDeltaAttentionSublayersSpec,
)
from paddlefleet.transformer.paddle_norm import RMSNorm
from paddlefleet.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Test dimensions (kept small for fast multi-GPU CI)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
CONV_KERNEL_DIM = 4
KEY_HEAD_DIM = 32
VALUE_HEAD_DIM = 32
NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 8
GATE_LORA_RANK = 32
GATE_LOWER_BOUND = -5.0
MICRO_BATCH_SIZE = 2
SEQ_LENGTH = 64
SEED = 123
INPUT_SEED = 42
TENSOR_PARALLEL = 4

QK_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
V_DIM = NUM_VALUE_HEADS * VALUE_HEAD_DIM

# conv1d channel layout: [q, k, v]
_CONV_SECTIONS = [QK_DIM, QK_DIM, V_DIM]


def _in_proj_sections(use_full_rank_gate: bool) -> list[int]:
    """in_proj output layout: [q, k, v, beta] (+ [gate] when full rank)."""
    sections = [QK_DIM, QK_DIM, V_DIM, NUM_VALUE_HEADS]
    if use_full_rank_gate:
        sections.append(V_DIM)
    return sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)


def _shard_by_sections(full_tensor, sections, tp_rank, tp_size, dim):
    """Split into *sections* along *dim*, keep this rank's slice of each, concat.

    This is the layout a fused column-parallel projection implies: rank r owns
    [sec0_r, sec1_r, ...] contiguously.
    """
    local_parts = []
    for p in paddle.split(full_tensor, sections, axis=dim):
        chunk = p.shape[dim] // tp_size
        slices = [slice(None)] * p.ndim
        slices[dim] = slice(tp_rank * chunk, (tp_rank + 1) * chunk)
        local_parts.append(p[tuple(slices)])
    return paddle.concat(local_parts, axis=dim)


def _gather_by_sections(local_tensor, sections, tp_group, tp_size, dim):
    """Inverse of _shard_by_sections: all-gather, then de-interleave sections."""
    last = dim == -1 or dim == local_tensor.ndim - 1
    gathered = (
        _gather_along_last_dim(local_tensor, tp_group)
        if last
        else _gather_along_first_dim(local_tensor, tp_group)
    )
    rank_chunk = gathered.shape[dim] // tp_size
    rank_tensors = paddle.split(gathered, [rank_chunk] * tp_size, axis=dim)
    local_section_sizes = [s // tp_size for s in sections]
    per_rank = [
        paddle.split(rt, local_section_sizes, axis=dim) for rt in rank_tensors
    ]
    return paddle.concat(
        [
            paddle.concat([pr[i] for pr in per_rank], axis=dim)
            for i in range(len(sections))
        ],
        axis=dim,
    )


def _build_config(
    tp_size: int = 1, sp: bool = False, fused: bool = False
) -> TransformerConfig:
    return TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_VALUE_HEADS,
        num_hidden_layers=1,
        hidden_act=F.silu,
        rms_norm_eps=1e-6,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sp,
        # deterministic_mode keeps the paddle native fallback; turning it off
        # selects the fused triton kernels.
        deterministic_mode=not fused,
    )


def _build_kda(
    config: TransformerConfig,
    use_full_rank_gate: bool,
    pg_collection: ProcessGroupCollection | None = None,
    tp_group=None,
) -> KimiDeltaAttention:
    # f_a_proj / g_a_proj are replicated (Linear); the b_proj of each pair is
    # column parallel and consumes the full rank.
    sublayers_spec = KimiDeltaAttentionSublayersSpec(
        in_proj=ColumnParallelLinear,
        f_a_proj=Linear,
        f_b_proj=ColumnParallelLinear,
        g_a_proj=Linear,
        g_b_proj=ColumnParallelLinear,
        out_norm=RMSNorm,
        out_proj=RowParallelLinear,
    )
    kwargs = {}
    if pg_collection is not None:
        kwargs["pg_collection"] = pg_collection
    if tp_group is not None:
        kwargs["pg_collection"] = ProcessGroupCollection(tp=tp_group)

    return KimiDeltaAttention(
        config=config,
        sublayers_spec=sublayers_spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        conv_kernel_dim=CONV_KERNEL_DIM,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        gate_lora_rank=GATE_LORA_RANK,
        use_full_rank_gate=use_full_rank_gate,
        gate_lower_bound=GATE_LOWER_BOUND,
        **kwargs,
    )


def _shard_param(name, full_param, tp_rank, tp_size, in_proj_sections):
    """Slice a full (TP=1) parameter down to this rank's shard."""
    if name == "in_proj.weight":
        return _shard_by_sections(
            full_param, in_proj_sections, tp_rank, tp_size, dim=-1
        )
    if name.endswith("_b_proj.weight"):
        # ColumnParallelLinear weight [rank, v_dim] -> shard the output dim
        chunk = full_param.shape[-1] // tp_size
        return full_param[:, tp_rank * chunk : (tp_rank + 1) * chunk]
    if name == "out_proj.weight":
        # RowParallelLinear weight [v_dim, hidden] -> shard the input dim
        chunk = full_param.shape[0] // tp_size
        return full_param[tp_rank * chunk : (tp_rank + 1) * chunk, :]
    if name.startswith("conv1d."):
        return _shard_by_sections(
            full_param, _CONV_SECTIONS, tp_rank, tp_size, dim=0
        )
    if name in ("dt_bias", "A_log"):
        chunk = full_param.shape[0] // tp_size
        return full_param[tp_rank * chunk : (tp_rank + 1) * chunk]
    # Replicated: f_a_proj.weight, g_a_proj.weight, out_norm.weight
    return full_param.clone()


def _gather_grads(kda, tp_group, tp_size, in_proj_sections):
    """Gather every parameter gradient back to its full (TP=1) shape."""
    grads = {}
    for name, param in kda.named_parameters():
        if param.grad is None:
            grads[name] = None
            continue
        grad = param.grad
        if name == "in_proj.weight":
            grad = _gather_by_sections(
                grad, in_proj_sections, tp_group, tp_size, dim=-1
            )
        elif name.endswith("_b_proj.weight"):
            grad = _gather_along_last_dim(grad, tp_group)
        elif name == "out_proj.weight":
            grad = _gather_along_first_dim(grad, tp_group)
        elif name.startswith("conv1d."):
            grad = _gather_by_sections(
                grad, _CONV_SECTIONS, tp_group, tp_size, dim=0
            )
        elif name in ("dt_bias", "A_log"):
            grad = _gather_along_first_dim(grad, tp_group)
        # Replicated params (f_a_proj / g_a_proj / out_norm weights) are expected
        # to already hold the full gradient thanks to the sequence-parallel
        # all-reduce hook, so they are compared as-is.
        grads[name] = grad
    return grads


def _make_input():
    """Full [b, s, h] input, identical on every rank."""
    _set_random_seed(INPUT_SEED)
    return paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])


def _run_baseline(seed: int, use_full_rank_gate: bool, fused: bool = False):
    """TP=1 reference: returns output, input grad and the module (for weights)."""
    _set_random_seed(seed)
    config = _build_config(tp_size=1, sp=False, fused=fused)
    tp1_group = dist.new_group([dist.get_rank()])
    kda = _build_kda(config, use_full_rank_gate, tp_group=tp1_group)
    assert kda.use_fused_kernels == fused, (
        f"expected use_fused_kernels={fused}, got {kda.use_fused_kernels}"
    )

    hidden_states = _make_input()
    hidden_states.stop_gradient = False
    output, _ = kda(hidden_states, attention_mask=None)
    output.sum().backward()

    return output.detach(), hidden_states.grad.detach(), kda


def _run_distributed(
    seed: int,
    tp_size: int,
    sp: bool,
    use_full_rank_gate: bool,
    output_baseline,
    input_grad_baseline,
    kda_baseline,
    fused: bool = False,
):
    """Run KDA under TP(/SP) with the baseline weights and compare everything."""
    _set_random_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    config = _build_config(tp_size=tp_size, sp=sp, fused=fused)
    tp_group = ps.get_tensor_model_parallel_group()
    tp_rank = ps.get_tensor_model_parallel_rank()
    sp_size = tp_size if sp else 1
    in_proj_sections = _in_proj_sections(use_full_rank_gate)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp"]
    )
    kda_dist = _build_kda(
        config, use_full_rank_gate, pg_collection=pg_collection
    )
    # Provides the TP all-reduce for replicated-but-partial gradients
    # (out_norm.weight and, under SP, the gate a_proj weights).
    register_sequence_parallel_allreduce_hooks(kda_dist, 1, False)

    baseline_sd = {
        name: param.detach() for name, param in kda_baseline.named_parameters()
    }
    with paddle.no_grad():
        for name, param in kda_dist.named_parameters():
            param.set_value(
                _shard_param(
                    name,
                    baseline_sd[name],
                    tp_rank,
                    tp_size,
                    in_proj_sections,
                )
            )

    hidden_states = _make_input()
    if sp:
        # [b, s, h] -> [s, b, h], then keep this rank's sequence chunk
        hidden_states = hidden_states.transpose([1, 0, 2])
        sp_seg = SEQ_LENGTH // sp_size
        hidden_states = hidden_states[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]
    hidden_states.stop_gradient = False
    hidden_states = hidden_states.contiguous()

    output_dist, _ = kda_dist(hidden_states, attention_mask=None)
    output_dist.sum().backward()

    # Sharding changes the per-rank tile shapes fed to the triton kernels, so the
    # fused path only agrees to the TF32 matmul floor rather than to round-off.
    atol = rtol = 5e-3 if fused else 5e-4
    tag = (
        f"TP={tp_size}, SP={sp}, full_rank_gate={use_full_rank_gate}, "
        f"fused={fused}"
    )

    # --- sharded_state_dict declares the right global shapes ---
    sharded_sd = kda_dist.sharded_state_dict("self_attn.")
    for name, param in kda_dist.named_parameters():
        key = f"self_attn.{name}"
        assert key in sharded_sd, (
            f"{tag}: {key} missing from sharded_state_dict"
        )
        weight = sharded_sd[key]
        assert list(weight.local_tensor.shape) == list(param.shape), (
            f"{tag}: {key} local shape {list(weight.local_tensor.shape)} != "
            f"{list(param.shape)}"
        )
        assert list(weight.global_shape) == list(baseline_sd[name].shape), (
            f"{tag}: {key} global shape {list(weight.global_shape)} != "
            f"TP=1 shape {list(baseline_sd[name].shape)}"
        )
    assert len(sharded_sd) == len(list(kda_dist.named_parameters())), (
        f"{tag}: sharded_state_dict has {len(sharded_sd)} entries for "
        f"{len(list(kda_dist.named_parameters()))} parameters"
    )

    def _check(name, got, expected):
        assert paddle.all(paddle.isfinite(got)).item(), (
            f"{tag}: {name} contains NaN/Inf"
        )
        assert paddle.allclose(got, expected, atol=atol, rtol=rtol).item(), (
            f"{tag}: {name} mismatch, max_diff="
            f"{(got - expected).abs().max().item():.6e}"
        )

    # --- output ---
    if sp:
        output_gathered = _gather_along_first_dim(
            output_dist, tp_group
        ).transpose([1, 0, 2])
    else:
        output_gathered = output_dist
    _check("output", output_gathered, output_baseline)

    # --- input gradient ---
    if sp:
        input_grad = _gather_along_first_dim(
            hidden_states.grad, tp_group
        ).transpose([1, 0, 2])
    else:
        input_grad = hidden_states.grad
    _check("input grad", input_grad, input_grad_baseline)

    # --- parameter gradients ---
    dist_grads = _gather_grads(kda_dist, tp_group, tp_size, in_proj_sections)
    checked = 0
    for name, param in kda_baseline.named_parameters():
        if param.grad is None or dist_grads.get(name) is None:
            continue
        b_grad, d_grad = param.grad.detach(), dist_grads[name]
        assert list(b_grad.shape) == list(d_grad.shape), (
            f"{tag}: grad shape mismatch for {name}: "
            f"{list(d_grad.shape)} vs {list(b_grad.shape)}"
        )
        _check(f"grad[{name}]", d_grad, b_grad)
        checked += 1
    assert checked == len(list(kda_baseline.named_parameters())), (
        f"{tag}: only {checked} parameter gradients were compared"
    )

    if dist.get_rank() == 0:
        print(f"  [PASS] {tag} ({checked} param grads)")


class TestKimiDeltaAttentionDistributed(unittest.TestCase):
    """TP / SP correctness for KimiDeltaAttention."""

    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": TENSOR_PARALLEL,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        initialize_fleet(strategy)

    def setUp(self):
        self.tp_size = TENSOR_PARALLEL
        self.seed = SEED

    def _check_forward_shape(self, sp: bool, use_full_rank_gate: bool):
        _set_random_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)

        config = _build_config(tp_size=self.tp_size, sp=sp)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp"]
        )
        kda = _build_kda(
            config, use_full_rank_gate, pg_collection=pg_collection
        )

        if sp:
            local_seq = SEQ_LENGTH // self.tp_size
            hidden_states = paddle.randn(
                [local_seq, MICRO_BATCH_SIZE, HIDDEN_SIZE]
            )
            expected = [local_seq, MICRO_BATCH_SIZE, HIDDEN_SIZE]
        else:
            hidden_states = paddle.randn(
                [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
            )
            expected = [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]

        output, _ = kda(hidden_states, attention_mask=None)
        self.assertEqual(output.shape, expected)
        self.assertEqual(output.dtype, hidden_states.dtype)
        self.assertTrue(paddle.all(paddle.isfinite(output)).item())

    def _check_precision(
        self, sp: bool, use_full_rank_gate: bool, fused: bool = False
    ):
        out_ref, grad_ref, kda_ref = _run_baseline(
            self.seed, use_full_rank_gate, fused=fused
        )
        _run_distributed(
            self.seed,
            self.tp_size,
            sp=sp,
            use_full_rank_gate=use_full_rank_gate,
            output_baseline=out_ref,
            input_grad_baseline=grad_ref,
            kda_baseline=kda_ref,
            fused=fused,
        )

    def test_all_cases(self):
        # Kimi's own setting (gate folded into in_proj) and the low-rank gate
        # variant, which additionally exercises g_a_proj / g_b_proj.
        for use_full_rank_gate in (True, False):
            for sp in (False, True):
                self._check_forward_shape(sp, use_full_rank_gate)
                self._check_precision(sp, use_full_rank_gate)

    @unittest.skipUnless(HAVE_FLA, "paddlefleet_ops fla kernels not available")
    def test_fused_kernels(self):
        """Same TP/SP wiring, but with the fused triton kernels."""
        for sp in (False, True):
            self._check_precision(sp, use_full_rank_gate=True, fused=True)


if __name__ == "__main__":
    unittest.main()
