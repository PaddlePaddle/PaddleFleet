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

import functools

import numpy as np
import paddle
import paddle.distributed as dist

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.moe.moe_router import TopKRouter
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

SEED = 2024
HIDDEN_SIZE = 8
NUM_EXPERTS = 4
TOP_K = 2


def _build_router_config(**overrides):
    # A minimal-but-real MoE routing config. Defaults exercise the standard
    # path: scoring_func="softmax", topk_method="greedy", norm_topk_prob=True,
    # routed_scaling_factor=1.0, moe_router_load_balancing_type="aux_loss".
    # tensor_model_parallel_size=4 matches the launched EP/TP=4 topology; the
    # router gate is a replicated (non-sharded) parameter under TP.
    defaults = dict(  # noqa: C408
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=NUM_EXPERTS,
        use_cpu_initialization=True,
        num_experts_per_tok=TOP_K,
        tensor_model_parallel_size=4,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        bf16=False,
        params_dtype=paddle.float32,
        moe_intermediate_size=16,
        moe_deep_gemm=False,
        gated_linear_unit=True,
        n_shared_experts=0,
        rms_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


# --- Independent (numpy) reference for greedy softmax top-k routing ---------
# These helpers deliberately re-derive the routing contract WITHOUT calling the
# router under test, so a wrong top-k order, a wrong normalisation, a swapped
# probs/mask layout or a wrong aux-loss reduction is actually rejected.


def _softmax_rows(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _reference_routing(logits, top_idx):
    """Given exact logits and the hand-picked top-k indices, derive the full
    routing contract: normalised combine weights, sparse probs, one-hot mask
    and the Switch-style aux loss ``num_experts * sum_e mean_t(prob) * mean_t(mask)``.
    """
    probs_full = _softmax_rows(logits)
    top_idx = np.asarray(top_idx)
    top_raw = np.take_along_axis(probs_full, top_idx, axis=1)
    top_norm = top_raw / top_raw.sum(axis=1, keepdims=True)

    probs_sparse = np.zeros_like(probs_full)
    np.put_along_axis(probs_sparse, top_idx, top_norm, axis=1)

    mask = np.zeros_like(probs_full)
    np.put_along_axis(mask, top_idx, 1.0, axis=1)

    me = probs_full.mean(axis=0)
    ce = mask.mean(axis=0)
    aux_loss = float((me * ce).sum() * probs_full.shape[1])
    return top_norm, probs_sparse, mask, aux_loss


def test_greedy_softmax_routing_matches_independent_reference():
    """Real GPU router forward vs. an independent hand derivation.

    The gate weight is pinned to a known [num_experts, hidden] matrix and the
    hidden states to known values, so the logits, softmax scores, top-2
    selection, renormalised combine weights, sparse probs, one-hot mask and
    aux loss are all fixed and re-derived in numpy (never from the function
    under test). Each token deliberately has a distinct expert ordering, so a
    reversed top-k, a wrong renormalisation or a probs/mask transpose changes
    the exact numbers checked here. Runs the real ``TopKRouter.forward`` on
    ``.cuda()`` under the launched MP=4 process group.
    """
    config = _build_router_config()
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    router = TopKRouter(config, pg_collection=pg_collection).cuda()

    # Gate reads the first NUM_EXPERTS hidden channels one-to-one, so
    # logits == first NUM_EXPERTS columns of the hidden states.
    weight_np = np.zeros((NUM_EXPERTS, HIDDEN_SIZE), dtype=np.float32)
    for e in range(NUM_EXPERTS):
        weight_np[e, e] = 1.0
    with paddle.no_grad():
        router.weight.set_value(
            paddle.to_tensor(weight_np, dtype=router.weight.dtype).cuda()
        )

    # 4 tokens, each with a unique descending expert order.
    x_np = np.zeros((4, HIDDEN_SIZE), dtype=np.float32)
    x_np[0, :4] = [3.0, 1.0, 0.0, -1.0]  # order 0,1
    x_np[1, :4] = [0.0, 2.0, 3.0, -1.0]  # order 2,1
    x_np[2, :4] = [-1.0, 0.0, 1.0, 2.0]  # order 3,2
    x_np[3, :4] = [2.0, -1.0, 0.0, 3.0]  # order 3,0
    expected_idx = [[0, 1], [2, 1], [3, 2], [3, 0]]

    hidden = paddle.to_tensor(x_np, dtype="float32").reshape(
        [4, 1, HIDDEN_SIZE]
    )
    hidden = hidden.cuda()

    with paddle.no_grad():
        (
            capacity,
            top_gate,
            top_idx,
            probs,
            mask,
            priorities,
            aux_loss,
            z_loss,
        ) = router(hidden)

    logits_np = x_np @ weight_np.T
    ref_weight, ref_probs, ref_mask, ref_aux = _reference_routing(
        logits_np, expected_idx
    )

    # Contract fields that must be None on this path.
    assert capacity is None
    assert priorities is None
    assert z_loss is None  # router_z_loss_coef == 0.0

    # Exact expert selection (order matters: topk is sorted descending).
    assert top_idx.numpy().tolist() == expected_idx
    # Renormalised combine weights sum to 1 per token and match softmax.
    np.testing.assert_allclose(
        top_gate.numpy(), ref_weight, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        top_gate.numpy().sum(axis=-1),
        np.ones(4, dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    # Sparse combine-weight layout: exactly the selected positions are filled.
    np.testing.assert_allclose(probs.numpy(), ref_probs, rtol=1e-5, atol=1e-6)
    # One-hot routing mask, exact positions.
    np.testing.assert_array_equal(mask.numpy(), ref_mask)
    # Load-balancing aux loss against the independent Switch-style formula.
    np.testing.assert_allclose(
        float(aux_loss.numpy()), ref_aux, rtol=1e-5, atol=1e-6
    )


def test_router_gate_replicated_across_tp_ranks():
    """The router gate is a replicated (non-sharded) parameter under TP=4, so
    every rank must hold a bit-identical gate and produce identical routing for
    the same input. This is a genuine cross-rank contract: it is verified with
    a real ``all_gather`` over the tensor-parallel group, not simulated on a
    single card. A gate that was wrongly initialised per-rank (or sharded)
    would make the gathered weights differ and fail here.
    """
    config = _build_router_config()
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    # Real init path: gate comes from config.init_method under the
    # model-parallel RNG seeded identically for replicated params.
    router = TopKRouter(config, pg_collection=pg_collection).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    assert tp_group.world_size == 4, (
        f"expected a real TP=4 group, got world_size={tp_group.world_size}"
    )

    # Gather every rank's gate weight and require exact equality.
    local_weight = router.weight.detach()
    gathered_weights = []
    dist.all_gather(gathered_weights, local_weight, group=tp_group)
    for other in gathered_weights[1:]:
        assert paddle.equal_all(gathered_weights[0], other), (
            "router gate weight is not replicated across TP ranks"
        )

    # Same fixed input on every rank must yield the same routing decision.
    x_np = (
        np.arange(4 * HIDDEN_SIZE, dtype=np.float32).reshape([4, HIDDEN_SIZE])
        - 15.0
    ) * 0.1
    hidden = paddle.to_tensor(x_np, dtype="float32").reshape(
        [4, 1, HIDDEN_SIZE]
    )
    hidden = hidden.cuda()
    with paddle.no_grad():
        out = router(hidden)
    top_idx = out[2].cast("int64")
    top_gate = out[1].cast("float32")

    gathered_idx = []
    dist.all_gather(gathered_idx, top_idx, group=tp_group)
    for other in gathered_idx[1:]:
        assert paddle.equal_all(gathered_idx[0], other), (
            "top-k expert selection differs across TP ranks"
        )

    gathered_gate = []
    dist.all_gather(gathered_gate, top_gate, group=tp_group)
    for other in gathered_gate[1:]:
        assert paddle.equal_all(gathered_gate[0], other), (
            "top-k combine weights differ across TP ranks"
        )


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    model_parallel_cuda_manual_seed(SEED)
    test_greedy_softmax_routing_matches_independent_reference()
    test_router_gate_replicated_across_tp_ranks()
