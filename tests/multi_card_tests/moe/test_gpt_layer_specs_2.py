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

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

EP_DEGREE = 4
NUM_EXPERTS = 4
HIDDEN_SIZE = 64
TOP_K = 2
SEED = 1234

_MOE_LAYER = None


def _init_fleet_ep4():
    """Initialize a 4-GPU expert-parallel topology (ep=4, everything else 1)."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": EP_DEGREE,
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
    initialize_fleet(strategy=strategy)
    model_parallel_cuda_manual_seed(SEED)


def _moe_config():
    return TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        n_routed_experts=NUM_EXPERTS,
        use_cpu_initialization=False,
        num_experts_per_tok=TOP_K,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        bf16=False,
        params_dtype=paddle.float32,
        moe_intermediate_size=HIDDEN_SIZE,
        gated_linear_unit=True,
        n_shared_experts=0,
        hidden_act=F.silu,
        moe_expert_fusion=False,
        moe_token_dispatcher_type="alltoall",
    )


def _build_moe_layer():
    """Build the MoELayer that get_gpt_layer_local_spec assembles for a MoE layer.

    The MoE sublayers spec is taken directly from the spec produced by the
    production entry get_gpt_layer_local_spec, so this entry stays on the
    validation chain (it is what get_gpt_decoder_layers_spec consumes).
    """
    global _MOE_LAYER
    if _MOE_LAYER is not None:
        return _MOE_LAYER
    config = _moe_config()
    layer_spec = get_gpt_layer_local_spec(
        config, num_experts=NUM_EXPERTS, moe_expert_fusion=False
    )
    moe_layer = MoELayer(
        config,
        layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
        ProcessGroupCollection.use_mpu_process_groups(),
    )
    # eval() disables any training-only stochasticity so the routing and the
    # forward are deterministic across ranks.
    moe_layer.eval()
    _MOE_LAYER = moe_layer
    return moe_layer


def test_moe_expert_parallel_expert_ownership():
    """Each EP rank owns exactly one distinct expert; the union is the full set.

    With num_experts=4 and ep=4 the non-fusion MoE builds an nn.LayerList of 4
    entries where entry i is a real module only on the rank whose ep-rank equals
    i (num_experts_per_device == 1), and None elsewhere. An all_gather of the
    locally-owned expert index over the real EP process group must therefore
    return [0, 1, 2, 3] in rank order -- a wrong shard assignment (a rank owning
    the wrong expert, two ranks owning the same expert, or a rank owning none)
    changes this exact list. Expected values are derived by hand from the
    partition rule, not from the layer.
    """
    moe_layer = _build_moe_layer()
    ep_group = moe_layer.moe_group
    ep_rank = moe_layer.moe_rank

    assert moe_layer.expert_model_parallel_size == EP_DEGREE
    assert moe_layer.num_local_experts == 1
    assert moe_layer.num_experts_per_device == 1

    owned = [i for i, e in enumerate(moe_layer.experts) if e is not None]
    # Exactly one real expert locally, and it is the expert whose global index
    # equals this rank's ep-rank.
    assert owned == [ep_rank], (owned, ep_rank)

    held_index = paddle.to_tensor([ep_rank], dtype="int64").cuda()
    gathered = []
    dist.all_gather(gathered, held_index, group=ep_group)
    gathered_indices = [int(t.item()) for t in gathered]
    assert gathered_indices == list(range(EP_DEGREE)), gathered_indices


def test_moe_forward_identical_tokens_produce_identical_rows():
    """Identical input tokens must yield identical output rows after dispatch.

    Every token is set to the same (broadcast) hidden vector, so the router
    assigns all tokens to the same top-k experts with the same gate weights and
    the combined output for every token must be identical. This exercises the
    real all-to-all dispatch (tokens leave for the ranks owning their experts)
    and combine (results return and are summed): a dispatch/combine bug that
    mislabels or misplaces tokens would make some rows differ. The invariant
    (all rows equal row 0) is derived by hand from the identical-token setup.
    """
    moe_layer = _build_moe_layer()
    batch, seq = 2, 8

    base = paddle.randn([HIDDEN_SIZE], dtype="float32").cuda()
    dist.broadcast(base, src=0)
    hidden_states = base.reshape([1, 1, HIDDEN_SIZE]).expand(
        [batch, seq, HIDDEN_SIZE]
    )
    hidden_states = paddle.assign(hidden_states)

    output = moe_layer(hidden_states)[0]
    assert list(output.shape) == [batch, seq, HIDDEN_SIZE]

    out_np = output.astype("float32").numpy()
    assert np.isfinite(out_np).all()

    rows = out_np.reshape([batch * seq, HIDDEN_SIZE])
    for r in range(1, rows.shape[0]):
        np.testing.assert_allclose(rows[r], rows[0], rtol=1e-5, atol=1e-6)

    # The experts must actually transform the input, otherwise "all rows equal"
    # would be trivially satisfied by an identity/zero combine.
    in_np = (
        hidden_states.astype("float32")
        .numpy()
        .reshape([batch * seq, HIDDEN_SIZE])
    )
    assert not np.allclose(rows[0], in_np[0], rtol=1e-4, atol=1e-4)


def test_moe_forward_replicated_input_consistent_across_ranks():
    """Replicated input + replicated router => identical MoE output on every rank.

    The full input is broadcast so all ranks share it; the router is replicated
    (no tensor parallelism here). Correct expert parallelism reconstructs each
    token from experts spread across ranks and the all-to-all combine delivers
    the same per-token result to every rank. Gathering each rank's output over
    the real EP process group and comparing to rank 0 therefore locks the
    cross-rank combine: a wrong peer, split size, or dropped contribution would
    make the gathered outputs diverge. This needs a real multi-rank process
    group -- a single-process run cannot exercise the all-to-all path.
    """
    moe_layer = _build_moe_layer()
    batch, seq = 2, 8

    hidden_states = paddle.randn(
        [batch, seq, HIDDEN_SIZE], dtype="float32"
    ).cuda()
    dist.broadcast(hidden_states, src=0)

    output = moe_layer(hidden_states)[0]
    assert list(output.shape) == [batch, seq, HIDDEN_SIZE]

    out_contig = paddle.assign(output.astype("float32"))
    gathered = []
    dist.all_gather(gathered, out_contig, group=moe_layer.moe_group)
    assert len(gathered) == EP_DEGREE

    ref = gathered[0].numpy()
    assert np.isfinite(ref).all()
    for r in range(1, EP_DEGREE):
        np.testing.assert_allclose(
            gathered[r].numpy(), ref, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    _init_fleet_ep4()
    test_moe_expert_parallel_expert_ownership()
    test_moe_forward_identical_tokens_produce_identical_rows()
    test_moe_forward_replicated_input_consistent_across_ranks()
