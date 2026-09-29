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

# Real 4-GPU expert-parallel (EP=4, TP=1, "Pattern D") behaviour test for the
# MoE AllToAllTokenDispatcher. Token dispatch and combine run over the REAL
# all-to-all / all-gather collectives of the 4-rank EP group -- no mocked
# collective, no faked world size, no CPU emulation of the exchange. Every
# expected value is derived independently by hand / numpy from the routing
# map and probs, never from the dispatcher under test.
#
# Topology: n_routed_experts=8 over EP=4 => 2 local experts per rank; rank r
# owns experts {2r, 2r+1}. Each of the 8 local tokens picks top_k=2 experts
# via a fixed hand-built routing map, so every expert is selected and the
# all-to-all genuinely crosses ranks.

import numpy as np
import paddle
from paddle.distributed import fleet

from paddlefleet import parallel_state
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
)

EP_DEGREE = 4
N_ROUTED_EXPERTS = 8
NUM_LOCAL_EXPERTS = N_ROUTED_EXPERTS // EP_DEGREE  # 2
TOPK = 2
D_MODEL = 8
NUM_TOKENS = 8
# Second expert offset: token t -> experts {t, (t + 4) % 8}. Offset 4 puts the
# two experts on different ranks (rank = expert // 2), forcing cross-rank
# traffic, and makes every expert selected by exactly two tokens per rank.
EXPERT_OFFSET = 4


def _second_expert(t):
    return (t + EXPERT_OFFSET) % N_ROUTED_EXPERTS


def _init_expert_parallel():
    """Initialise a real EP=4 Fleet topology across the 4-GPU world.

    ``EPHybridCommunicateGroup`` requires the *dense* (non-expert) degrees to
    cover the whole world, otherwise it cannot assign each rank a data-parallel
    id and ``fleet.init`` aborts. Following the proven pattern of the sibling
    ``test_allgather_dispatcher_ep`` / ``test_ring_dispatcher_ep`` tests, the
    dense world is filled with ``sharding_degree = EP_DEGREE`` while the expert
    group overlays it (``ep_degree = EP_DEGREE``). The whole 4-GPU world thus
    forms a single 4-rank expert-parallel group -- the topology the
    AllToAllTokenDispatcher's cross-rank all-to-all is built for.
    """
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": EP_DEGREE,
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


def _local_expert_indices(rank):
    start = rank * NUM_LOCAL_EXPERTS
    return list(range(start, start + NUM_LOCAL_EXPERTS))


def _make_dispatcher(ep_group):
    return AllToAllTokenDispatcher(
        moe_group=ep_group,
        expert_model_parallel_size=ep_group.nranks,
        num_experts_per_device=NUM_LOCAL_EXPERTS,
        local_expert_indices=_local_expert_indices(ep_group.rank),
    )


def _routing_map_np():
    """One hand-built [num_tokens, num_experts] routing map, top_k=2.

    token t selects experts t and (t + 4) % 8. Independent of production.
    """
    routing = np.zeros([NUM_TOKENS, N_ROUTED_EXPERTS], dtype=np.int32)
    for t in range(NUM_TOKENS):
        routing[t, t % N_ROUTED_EXPERTS] = 1
        routing[t, _second_expert(t)] = 1
    return routing


def _hidden_np(rank):
    """Rank/token/channel-distinguishable hidden states [num_tokens, d].

    hidden[t, c] = rank*1000 + t*100 + c, so a wrong all-to-all peer, a
    reversed direction or a dropped rank changes the exact delivered values.
    """
    cols = np.arange(D_MODEL, dtype=np.float32)
    return np.stack(
        [rank * 1000 + t * 100 + cols for t in range(NUM_TOKENS)], axis=0
    )


def _probs_np(weights_first, weights_second):
    """Probs [num_tokens, num_experts]; weight on the two selected experts.

    weights_first[t] lands on expert t, weights_second[t] on expert
    (t + 4) % 8. All other entries stay 0.
    """
    probs = np.zeros([NUM_TOKENS, N_ROUTED_EXPERTS], dtype=np.float32)
    for t in range(NUM_TOKENS):
        probs[t, t % N_ROUTED_EXPERTS] = weights_first[t]
        probs[t, _second_expert(t)] = weights_second[t]
    return probs


def _to_cuda(array, dtype):
    return paddle.to_tensor(array, dtype=dtype).cuda()


def _expected_dispatched_tokens(rank, ws):
    """Hand-derive this rank's post all-to-all tokens, before local sorting.

    token_dispatch's all-to-all output concatenates, in source-rank order,
    the block each source rank sends to ``rank``. That block is the source's
    permuted tokens for the experts this rank owns -- experts {2r, 2r+1} --
    grouped by expert and, within an expert, in ascending token order.

    Expert e is selected exactly by tokens {e, (e + 4) % 8}. So the segment
    every source rank i sends here is, for experts (2r, 2r+1):
        [sorted tokens of expert 2r] + [sorted tokens of expert 2r+1]
    and row content for source i, token t is i*1000 + t*100 + channel.
    """
    cols = np.arange(D_MODEL, dtype=np.float32)
    segment_tokens = []
    for expert in _local_expert_indices(rank):
        pair = sorted([expert, _second_expert(expert)])
        segment_tokens.extend(pair)
    rows = [
        i * 1000 + t * 100 + cols for i in range(ws) for t in segment_tokens
    ]
    return np.stack(rows, axis=0)


def test_alltoall_dispatch_delivers_cross_rank_tokens():
    """Real all-to-all dispatch delivers each rank exactly the tokens routed
    to its local experts, gathered across ranks, byte-for-byte.

    Expected is hand-derived (see _expected_dispatched_tokens); it is never
    read back from the dispatcher. tokens_per_expert is verified against the
    independent count: every expert is picked by 2 local tokens on each of
    the ws ranks, so each of this rank's local experts must receive 2*ws.
    """
    ep_group = parallel_state.get_expert_model_parallel_group()
    ws = ep_group.nranks
    rank = ep_group.rank

    hidden = _to_cuda(_hidden_np(rank), "float32")
    routing_map = _to_cuda(_routing_map_np(), "int32")
    probs = _to_cuda(
        _probs_np([0.5] * NUM_TOKENS, [0.5] * NUM_TOKENS), "float32"
    )

    dispatcher = _make_dispatcher(ep_group)
    permuted = dispatcher.dispatch_preprocess(hidden, probs, routing_map)
    global_tokens, _ = dispatcher.token_dispatch(permuted)
    _, tokens_per_expert = dispatcher.dispatch_postprocess(global_tokens)

    expected = _expected_dispatched_tokens(rank, ws)
    assert list(global_tokens.shape) == [
        ws * TOPK * NUM_LOCAL_EXPERTS,
        D_MODEL,
    ]
    np.testing.assert_array_equal(global_tokens.numpy(), expected)
    assert tokens_per_expert.tolist() == [2 * ws] * NUM_LOCAL_EXPERTS


def test_alltoall_round_trip_is_identity_when_probs_normalized():
    """Dispatch -> identity expert -> combine reconstructs the local tokens
    exactly when each token's two routing probabilities sum to 1.

    By hand, an identity expert makes the round trip yield
        out[t] = hidden[t] * (probs[t].sum over selected experts).
    With normalised, per-token-distinct weights that sum is 1, so out must
    equal hidden. A reversed all-to-all, a lost token or a scrambled
    unpermute mapping would corrupt the rank-distinguishable content.
    """
    ep_group = parallel_state.get_expert_model_parallel_group()
    rank = ep_group.rank

    hidden_np = _hidden_np(rank)
    hidden = _to_cuda(hidden_np, "float32")
    routing_map = _to_cuda(_routing_map_np(), "int32")
    w_first = [0.60 + 0.02 * t for t in range(NUM_TOKENS)]
    w_second = [1.0 - w for w in w_first]
    probs = _to_cuda(_probs_np(w_first, w_second), "float32")

    dispatcher = _make_dispatcher(ep_group)
    permuted = dispatcher.dispatch_preprocess(hidden, probs, routing_map)
    global_tokens, _ = dispatcher.token_dispatch(permuted)
    sorted_tokens, _ = dispatcher.dispatch_postprocess(global_tokens)

    # Identity expert: feed dispatched tokens straight back into combine.
    pre = dispatcher.combine_preprocess(sorted_tokens)
    combined = dispatcher.token_combine(pre)
    output = dispatcher.combine_postprocess(combined)

    assert list(output.shape) == [NUM_TOKENS, D_MODEL]
    np.testing.assert_allclose(output.numpy(), hidden_np, rtol=1e-5, atol=1e-3)


def test_alltoall_combine_applies_router_probs():
    """Combine actually applies the per-token routing probabilities.

    With an identity expert and probabilities that do NOT sum to 1, the
    hand-derived round trip is out[t] = hidden[t] * (w_first[t] + w_second[t]).
    The per-token scale is distinct and != 1, so this rejects a combine that
    drops the probability weighting, hard-codes 1/top_k, or mixes tokens
    across positions.
    """
    ep_group = parallel_state.get_expert_model_parallel_group()
    rank = ep_group.rank

    hidden_np = _hidden_np(rank)
    hidden = _to_cuda(hidden_np, "float32")
    routing_map = _to_cuda(_routing_map_np(), "int32")
    w_first = [0.30 + 0.05 * t for t in range(NUM_TOKENS)]
    w_second = [0.20] * NUM_TOKENS
    probs = _to_cuda(_probs_np(w_first, w_second), "float32")

    dispatcher = _make_dispatcher(ep_group)
    permuted = dispatcher.dispatch_preprocess(hidden, probs, routing_map)
    global_tokens, _ = dispatcher.token_dispatch(permuted)
    sorted_tokens, _ = dispatcher.dispatch_postprocess(global_tokens)

    pre = dispatcher.combine_preprocess(sorted_tokens)
    combined = dispatcher.token_combine(pre)
    output = dispatcher.combine_postprocess(combined)

    scale = np.asarray(
        [w_first[t] + w_second[t] for t in range(NUM_TOKENS)],
        dtype=np.float32,
    ).reshape([NUM_TOKENS, 1])
    expected = hidden_np * scale
    assert list(output.shape) == [NUM_TOKENS, D_MODEL]
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-3)
    # The weighting is non-trivial: it must change the values (sum != 1).
    assert not np.allclose(output.numpy(), hidden_np)


if __name__ == "__main__":
    _init_expert_parallel()
    ep_group = parallel_state.get_expert_model_parallel_group()
    assert ep_group is not None, "expert-model-parallel group not initialized"
    assert ep_group.nranks == EP_DEGREE, (
        f"expected EP world size {EP_DEGREE}, got {ep_group.nranks}"
    )
    test_alltoall_dispatch_delivers_cross_rank_tokens()
    test_alltoall_round_trip_is_identity_when_probs_normalized()
    test_alltoall_combine_applies_router_probs()
