#  Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Context-parallel correctness for KimiDeltaAttention.

Each rank runs the same layer twice: once over the whole sequence with a CP=1
group (reference) and once over its contiguous shard with the real CP group.
The sharded output must match the corresponding slice of the reference, which
exercises the cross-rank recurrent state and the conv halo exchange.

Launch:
    python -m paddle.distributed.launch --gpus="0,1,2,3" \
        transformer/test_kimi_delta_attention_cp.py
"""

from __future__ import annotations

import unittest

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet

import paddlefleet.parallel_state as ps
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
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

CONTEXT_PARALLEL = 4
HIDDEN_SIZE = 128
KEY_HEAD_DIM = VALUE_HEAD_DIM = 32
NUM_KEY_HEADS = NUM_VALUE_HEADS = 4
CONV_KERNEL_DIM = 4
SEQ_LENGTH = 256
SEED = 1234
# Documents of the full packed sequence; deliberately not aligned to SEQ/CP
DOCS = [100, 60, 96]


def _config(cp_size):
    return TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_VALUE_HEADS,
        num_hidden_layers=1,
        hidden_act=F.silu,
        rms_norm_eps=1e-6,
        normalization="RMSNorm",
        context_parallel_size=cp_size,
        cp_balance_mode="contiguous_allgather",
        deterministic_mode=False,
    )


def _build(config, pg_collection):
    spec = KimiDeltaAttentionSublayersSpec(
        in_proj=ColumnParallelLinear,
        f_a_proj=Linear,
        f_b_proj=ColumnParallelLinear,
        out_norm=RMSNorm,
        out_proj=RowParallelLinear,
    )
    return KimiDeltaAttention(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        pg_collection=pg_collection,
        conv_kernel_dim=CONV_KERNEL_DIM,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        gate_lora_rank=VALUE_HEAD_DIM,
        use_full_rank_gate=True,
        gate_lower_bound=-5.0,
    )


def _indices():
    """[1, 1, S, 1] exclusive document ends for the full sequence."""
    row, end = [], 0
    for length in DOCS:
        end += length
        row += [end] * length
    assert len(row) == SEQ_LENGTH, (len(row), SEQ_LENGTH)
    return paddle.to_tensor([row], dtype="int32").reshape([1, 1, SEQ_LENGTH, 1])


@unittest.skipUnless(HAVE_FLA, "paddlefleet_ops fla kernels not available")
class TestKimiDeltaAttentionCP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": CONTEXT_PARALLEL,
            "sep_degree": 1,
            "cp_degree": CONTEXT_PARALLEL,
            "ep_degree": CONTEXT_PARALLEL,
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

    def test_matches_full_sequence(self):
        paddle.seed(SEED)
        model_parallel_cuda_manual_seed(SEED)
        cp_group = ps.get_context_parallel_group()
        cp_rank = dist.get_rank(cp_group)
        indices = _indices()

        # --- reference: whole sequence, CP disabled ---
        ref_layer = _build(
            _config(1),
            ProcessGroupCollection(
                tp=None, cp=dist.new_group([dist.get_rank()])
            ),
        )
        self.assertEqual(ref_layer.cp_size, 1)
        paddle.seed(SEED + 1)
        x = paddle.randn([1, SEQ_LENGTH, HIDDEN_SIZE])
        xf = x.clone()
        xf.stop_gradient = False
        ref, _ = ref_layer(
            hidden_states=xf, attn_mask_startend_row_indices=indices
        )
        ref.sum().backward()

        # --- context parallel: this rank's contiguous shard ---
        cp_layer = _build(
            _config(CONTEXT_PARALLEL),
            ProcessGroupCollection(tp=None, cp=cp_group),
        )
        self.assertEqual(cp_layer.cp_size, CONTEXT_PARALLEL)
        with paddle.no_grad():
            ref_params = dict(ref_layer.named_parameters())
            for name, param in cp_layer.named_parameters():
                param.set_value(ref_params[name])

        local = SEQ_LENGTH // CONTEXT_PARALLEL
        shard = x[:, cp_rank * local : (cp_rank + 1) * local].clone()
        shard.stop_gradient = False
        out, _ = cp_layer(
            hidden_states=shard, attn_mask_startend_row_indices=indices
        )
        out.sum().backward()

        want = ref[:, cp_rank * local : (cp_rank + 1) * local]
        err = float((out - want).norm() / want.norm())
        assert err < 6e-4, f"rank {cp_rank}: output rel_err={err:.3e}"

        want_g = xf.grad[:, cp_rank * local : (cp_rank + 1) * local]
        gerr = float((shard.grad - want_g).norm() / want_g.norm())
        assert gerr < 6e-4, f"rank {cp_rank}: grad_x rel_err={gerr:.3e}"
        if cp_rank == 0:
            print(f"  [PASS] CP={CONTEXT_PARALLEL} out={err:.3e} gx={gerr:.3e}")


if __name__ == "__main__":
    unittest.main()
