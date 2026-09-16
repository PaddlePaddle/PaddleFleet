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
from paddle.distributed import fleet

from paddlefleet import parallel_state
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
)

# Expert-parallel degree under test: 4 GPUs, EP=4, one expert per rank.
EP_DEGREE = 4
D_MODEL = 8
# Per-token routing weight for the single expert each token selects (top_k=1).
# Deliberately not summing to 1 and all distinct, so combine's probability
# application is a scale-sensitive, per-token-distinguishable contract rather
# than a degenerate identity.
TOKEN_PROBS = [0.5, 0.6, 0.7, 0.8]


def _init_expert_parallel():
    """Initialize a real EP=4 Fleet topology (DP=MP=PP=sharding=1)."""
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


def _make_dispatcher(ep_group):
    """Build an AllToAllTokenDispatcher owning exactly local expert == rank."""
    rank = ep_group.rank
    return AllToAllTokenDispatcher(
        moe_group=ep_group,
        expert_model_parallel_size=ep_group.nranks,
        num_experts_per_device=1,
        local_expert_indices=[rank],
    )


def _rank_distinguishable_inputs(ep_group):
    """Build hidden states / routing_map / probs unique per (rank, token).

    Layout (ws == num_experts == num_tokens == EP_DEGREE):
      * token j on rank r selects ONLY expert j (top_k=1, one-hot routing_map),
        so expert e is owned by rank e and receives token e from every rank.
      * hidden[j, c] = rank*1000 + j*100 + c, so a wrong all-to-all peer, a
        reversed direction or a dropped rank changes exact delivered values.
      * probs[j, j] = TOKEN_PROBS[j]; every other entry is 0.
    """
    ws = ep_group.nranks
    rank = ep_group.rank
    cols = np.arange(D_MODEL, dtype=np.float32)
    hidden_np = np.stack(
        [rank * 1000 + j * 100 + cols for j in range(ws)], axis=0
    )
    hidden = paddle.to_tensor(hidden_np, dtype="float32").cuda()

    # One-hot integer routing map [num_tokens, num_experts]; token j -> expert j.
    routing_map = paddle.eye(ws, dtype="int32").cuda()

    p = paddle.to_tensor(TOKEN_PROBS[:ws], dtype="float32")
    probs = (paddle.eye(ws, dtype="float32") * p.reshape([ws, 1])).cuda()
    return hidden, routing_map, probs


def test_alltoall_dispatch_delivers_expert_tokens():
    """Real all-to-all dispatch must deliver each rank exactly the tokens
    routed to its local expert, in source-rank order, byte-for-byte.

    Expected derived by hand (not from production or the coverage source):
    rank r owns expert r and thus receives token r from every source rank i,
    so delivered row i column c == i*1000 + r*100 + c. tokens_per_expert for
    the single local expert must be ws (one token per rank).
    """
    ep_group = parallel_state.get_expert_model_parallel_group()
    ws = ep_group.nranks
    rank = ep_group.rank
    hidden, routing_map, probs = _rank_distinguishable_inputs(ep_group)

    dispatcher = _make_dispatcher(ep_group)
    permuted = dispatcher.dispatch_preprocess(hidden, probs, routing_map)
    global_tokens, _ = dispatcher.token_dispatch(permuted)
    sorted_tokens, tokens_per_expert = dispatcher.dispatch_postprocess(
        global_tokens
    )

    cols = np.arange(D_MODEL, dtype=np.float32)
    expected = np.stack(
        [i * 1000 + rank * 100 + cols for i in range(ws)], axis=0
    )
    assert list(sorted_tokens.shape) == [ws, D_MODEL]
    np.testing.assert_array_equal(sorted_tokens.numpy(), expected)
    # Every rank contributed exactly one token to this rank's expert.
    assert tokens_per_expert.tolist() == [ws]


def test_alltoall_dispatch_combine_round_trip():
    """Dispatch -> identity expert -> combine must reconstruct each local
    token weighted by its own routing probability.

    With an identity expert (dispatched tokens fed straight back into
    combine), the permutation-preserving round trip yields, by hand:
    out[j] = hidden[j] * TOKEN_PROBS[j]. Because the probs are distinct and
    != 1, this rejects a reversed all-to-all, a lost token, a scrambled
    unpermute mapping and a dropped probability application alike.
    """
    ep_group = parallel_state.get_expert_model_parallel_group()
    ws = ep_group.nranks
    rank = ep_group.rank
    hidden, routing_map, probs = _rank_distinguishable_inputs(ep_group)

    dispatcher = _make_dispatcher(ep_group)
    permuted = dispatcher.dispatch_preprocess(hidden, probs, routing_map)
    global_tokens, _ = dispatcher.token_dispatch(permuted)
    sorted_tokens, _ = dispatcher.dispatch_postprocess(global_tokens)

    # Identity expert: pass the dispatched tokens straight into combine.
    pre = dispatcher.combine_preprocess(sorted_tokens)
    combined = dispatcher.token_combine(pre)
    output = dispatcher.combine_postprocess(combined)

    cols = np.arange(D_MODEL, dtype=np.float32)
    hidden_np = np.stack(
        [rank * 1000 + j * 100 + cols for j in range(ws)], axis=0
    )
    scale = np.asarray(TOKEN_PROBS[:ws], dtype=np.float32).reshape([ws, 1])
    expected = hidden_np * scale
    assert list(output.shape) == [ws, D_MODEL]
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-3)
    # Guard against a degenerate identity round trip: the applied per-token
    # scale must actually change the values (probs != 1).
    assert not np.allclose(output.numpy(), hidden_np)


if __name__ == "__main__":
    _init_expert_parallel()
    ep_group = parallel_state.get_expert_model_parallel_group()
    assert ep_group is not None, "expert-model-parallel group not initialized"
    assert ep_group.nranks == EP_DEGREE, (
        f"expected EP world size {EP_DEGREE}, got {ep_group.nranks}"
    )
    test_alltoall_dispatch_delivers_expert_tokens()
    test_alltoall_dispatch_combine_round_trip()
