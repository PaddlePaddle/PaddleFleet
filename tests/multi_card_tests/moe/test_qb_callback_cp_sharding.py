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
"""Multi-card tests for the QB callback's histogram all-reduce.

Covers the two claims the single-card tests can only assert against a mocked
``dist.all_reduce``:

1. ``hcg.get_sharding_parallel_group()`` really does cover the context-parallel
   dimension, so the removed CP reduce is redundant rather than missing.
2. The int64 all-reduce works on the real NCCL backend and each rank's
   histogram is counted exactly once (no CP double-counting).

Topology (8 GPUs): mp=2, sharding=4, cp=2 (nested in sharding), ep=8.
`EPHybridCommunicateGroup` requires dp=sep=1 and ep*moe_sharding == mp*sharding,
so this is the smallest topology that still gives a non-trivial TP group, a
sharding group strictly larger than the CP group, and
``tp_size * sharding_size == world_size`` — the last property means the reduced
histogram must equal the sum over every rank.

Run with:
  python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
      tests/multi_card_tests/moe/test_qb_callback_cp_sharding.py
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.qb_callback import (
    MoEQuantileBalancingCallback,
    _try_get_comm_groups,
)

E, B, K, N_EXPERTS = 4, 64, 2, 4

_fleet_state = {"done": False, "error": None}


def _ensure_fleet():
    """Initialise fleet once per process; re-raise the original failure after."""
    if _fleet_state["error"] is not None:
        raise _fleet_state["error"]
    if _fleet_state["done"]:
        return
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 2,
        "pp_degree": 1,
        "sharding_degree": 4,
        "sep_degree": 1,
        "cp_degree": 2,
        "ep_degree": 8,
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
    try:
        initialize_fleet(strategy=strategy)
    except Exception as exc:
        _fleet_state["error"] = exc
        raise
    _fleet_state["done"] = True


def _local_histogram(rank):
    """Deterministic per-rank histogram, reproducible from the rank id alone.

    Every rank can therefore reconstruct any other rank's contribution and
    build the expected global sum without an extra collective.
    """
    rng = np.random.RandomState(1000 + rank)
    return rng.randint(0, 8, (E, B)).astype(np.int32)


def _make_layer(
    histogram_np, sequence_parallel, ep_size, b_min=-1.0, b_max=1.0
):
    """Mock router layer exposing exactly what `_update_single_layer` touches."""
    layer = MagicMock()
    layer.qb_histogram = paddle.to_tensor(histogram_np, dtype=paddle.int32)
    layer.num_experts_per_tok = K
    layer.num_experts = N_EXPERTS
    layer.qb_bin_min = paddle.to_tensor(b_min, dtype=paddle.float32)
    layer.qb_bin_max = paddle.to_tensor(b_max, dtype=paddle.float32)
    layer.config = MagicMock()
    layer.config.sequence_parallel = sequence_parallel
    layer.config.expert_model_parallel_size = ep_size

    class BiasMock:
        ndim = 1

        def set_value(self, val):
            layer._bias_value = val

    layer.e_score_correction_bias = BiasMock()
    return layer


class _QBCommTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_fleet()
        cls.hcg = fleet.get_hybrid_communicate_group()
        cls.world_size = dist.get_world_size()
        cls.rank = dist.get_rank()

    def setUp(self):
        self.tp_group, self.dp_group, self.sd_group = _try_get_comm_groups()
        self.cp_group = self.__class__.hcg.get_context_parallel_group()

    def _reference_bias(self, global_histogram):
        """Single-rank recovery on an already-merged histogram (no collectives)."""
        layer = _make_layer(
            global_histogram.astype(np.int32),
            sequence_parallel=False,
            ep_size=1,
        )
        MoEQuantileBalancingCallback()._update_single_layer(
            layer, tp_group=None, dp_group=None, sd_group=None
        )
        return layer._bias_value.numpy()


class TestCommGroupTopology(_QBCommTestBase):
    """The CP group must be a sub-slice of the sharding group."""

    def test_topology_is_representative(self):
        self.assertEqual(self.world_size, 8)
        self.assertIsNotNone(
            self.tp_group, "TP group missing: mp_degree>1 expected"
        )
        self.assertIsNotNone(self.sd_group, "sharding group missing")
        self.assertGreater(self.cp_group.nranks, 1, "cp_degree>1 expected")
        self.assertGreater(
            self.sd_group.nranks,
            self.cp_group.nranks,
            "sharding must be strictly larger than CP for this test to be meaningful",
        )

    def test_sharding_group_covers_context_parallel(self):
        """`get_sharding_parallel_group()` subsumes the CP ranks."""
        sd_ranks = list(self.sd_group.ranks)
        cp_ranks = list(self.cp_group.ranks)

        self.assertTrue(
            set(cp_ranks).issubset(set(sd_ranks)),
            f"rank {self.rank}: cp_ranks={cp_ranks} not a subset of "
            f"sd_ranks={sd_ranks}; the CP reduce would NOT be redundant",
        )

        # Stronger: CP is a contiguous slice of the sharding list, which is what
        # `split_context_comm_list` builds it from.
        start = sd_ranks.index(cp_ranks[0])
        self.assertEqual(
            sd_ranks[start : start + len(cp_ranks)],
            cp_ranks,
            f"rank {self.rank}: cp_ranks={cp_ranks} is not a contiguous slice "
            f"of sd_ranks={sd_ranks}",
        )

    def test_tp_times_sharding_covers_world(self):
        """Precondition used by the "counted exactly once" test below."""
        self.assertEqual(
            self.tp_group.nranks * self.sd_group.nranks,
            self.world_size,
            "TP x sharding must span every rank in this topology",
        )


class TestInt64AllReduce(_QBCommTestBase):
    """Real NCCL int64 all_reduce, including counts beyond fp32's exact range."""

    def test_int64_all_reduce_is_exact(self):
        base = 2**24 + 1  # first integer fp32 cannot represent exactly
        local = paddle.full([E, B], base + self.rank, dtype=paddle.int64)
        dist.all_reduce(local, group=self.sd_group)

        sd_ranks = list(self.sd_group.ranks)
        expected = sum(base + r for r in sd_ranks)
        got = local.numpy()

        self.assertEqual(got.dtype, np.int64)
        np.testing.assert_array_equal(
            got, np.full((E, B), expected, dtype=np.int64)
        )

    def test_int64_all_reduce_overflows_int32(self):
        """Counts whose sum exceeds int32 range survive the int64 reduction."""
        base = 2**30  # 4 ranks x 2**30 > 2**31 - 1
        local = paddle.full([E, B], base + self.rank, dtype=paddle.int64)
        dist.all_reduce(local, group=self.sd_group)

        expected = sum(base + r for r in self.sd_group.ranks)
        self.assertGreater(expected, 2**31 - 1)
        np.testing.assert_array_equal(
            local.numpy(), np.full((E, B), expected, dtype=np.int64)
        )


class TestHistogramReduceCorrectness(_QBCommTestBase):
    """End-to-end `_update_single_layer` against real collectives."""

    def test_each_rank_counted_exactly_once(self):
        """With SP+EP the tp/sharding reduces sum every rank exactly once."""
        local = _local_histogram(self.rank)
        layer = _make_layer(local, sequence_parallel=True, ep_size=2)
        MoEQuantileBalancingCallback()._update_single_layer(
            layer, self.tp_group, self.dp_group, self.sd_group
        )
        got = layer._bias_value.numpy()

        global_once = sum(
            _local_histogram(r).astype(np.int64) for r in range(self.world_size)
        )
        np.testing.assert_array_equal(got, self._reference_bias(global_once))

    def test_extra_cp_reduce_inflates_the_merged_histogram(self):
        """Negative control, asserted on the histogram via real collectives.

        The merged histogram -- not the recovered bias -- is where CP
        double-counting is observable. Because CP is a contiguous slice that
        partitions the sharding group, an extra CP all-reduce inflates the
        merged counts by exactly ``cp_nranks`` whatever the per-rank data is
        (before the sharding reduce each ``h_j`` is counted once per rank of
        its CP block; after it, every sharding rank already holds the same
        value). And the QB recovery is invariant under a uniform scaling of
        the histogram: ``q_target``, ``c`` and ``h`` all scale together, so
        ``beta`` and ``fraction`` are unchanged. The only leak is the floor in
        ``q_target = floor(total * k / n)``, which bites only when some expert
        total is odd -- with every total even the bias is bit-identical. So an
        assertion on the bias would be luck, and both of the reduces below are
        checked against exact expected counts instead.
        """
        local = _local_histogram(self.rank)

        # The production reduce sequence under SP + EP: tp, then sharding.
        merged = paddle.to_tensor(local, dtype=paddle.int64)
        dist.all_reduce(merged, group=self.tp_group)
        dist.all_reduce(merged, group=self.sd_group)

        # tp x sharding spans the world (asserted in TestCommGroupTopology),
        # so every rank must be counted exactly once.
        expected = sum(
            _local_histogram(r).astype(np.int64) for r in range(self.world_size)
        )
        np.testing.assert_array_equal(
            merged.numpy(),
            expected,
            err_msg="tp+sharding reduce did not count every rank exactly once",
        )

        # The removed CP reduce would have run on top of that.
        cp_nranks = self.cp_group.nranks
        self.assertGreater(cp_nranks, 1, "cp_degree>1 expected")
        doubled = merged.clone()
        dist.all_reduce(doubled, group=self.cp_group)
        np.testing.assert_array_equal(
            doubled.numpy(),
            expected * cp_nranks,
            err_msg=(
                "an extra CP all_reduce must inflate the merged counts by "
                "exactly cp_nranks; if it does not, CP is not nested in "
                "sharding the way this test assumes"
            ),
        )

    def test_sharding_only_reduce_matches_its_group(self):
        """Without SP the TP reduce is skipped, so the orbit is the sharding group."""
        local = _local_histogram(self.rank)
        layer = _make_layer(local, sequence_parallel=False, ep_size=2)
        MoEQuantileBalancingCallback()._update_single_layer(
            layer, self.tp_group, self.dp_group, self.sd_group
        )
        got = layer._bias_value.numpy()

        expected_hist = sum(
            _local_histogram(r).astype(np.int64) for r in self.sd_group.ranks
        )
        np.testing.assert_array_equal(got, self._reference_bias(expected_hist))

    def test_bias_is_identical_on_every_rank(self):
        """All ranks must agree bit-for-bit, otherwise routing diverges."""
        local = _local_histogram(self.rank)
        layer = _make_layer(local, sequence_parallel=True, ep_size=2)
        MoEQuantileBalancingCallback()._update_single_layer(
            layer, self.tp_group, self.dp_group, self.sd_group
        )
        bias = paddle.to_tensor(layer._bias_value, dtype=paddle.float32)

        gathered = []
        dist.all_gather(gathered, bias)
        ref = gathered[0].numpy()
        for r, other in enumerate(gathered):
            np.testing.assert_array_equal(
                other.numpy(),
                ref,
                err_msg=f"rank {r} recovered a different QB bias",
            )

    def test_histogram_and_usage_reset_after_update(self):
        local = _local_histogram(self.rank)
        layer = _make_layer(local, sequence_parallel=True, ep_size=2)
        MoEQuantileBalancingCallback()._update_single_layer(
            layer, self.tp_group, self.dp_group, self.sd_group
        )
        self.assertEqual(int(layer.qb_histogram.sum().item()), 0)
        layer.expert_usage.zero_.assert_called_once()


if __name__ == "__main__":
    unittest.main()
