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
"""Multi-card (EP>1) test for ``muon_utils.ortho_ep_full_intermediate``.

Under the 'allgather' MoE dispatcher a rank holds every expert but only
``moe_intermediate_size // EP`` of each one, so Muon has to redistribute before
orthogonalising. Both directions of that redistribution are exercised here: the
helper returns a tensor in the rank-local layout, so a wrong expert/rank/gate-up
permutation on either leg shows up as a mismatch against the reference.

Every rank builds the *same* full-width weights from a fixed seed and keeps its
own column range, so each rank's shard is distinguishable and a mis-permutation
cannot pass by accident.

Run with:
  python -m paddle.distributed.launch --gpus=0,1 \
      tests/multi_card_tests/moe/test_muon_ep_slice.py
"""

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.muon_utils import ortho_ep_full_intermediate

DTYPE = "float64"

_pg_collection = None


def _ensure_fleet():
    global _pg_collection
    if _pg_collection is not None:
        return _pg_collection
    ep_degree = dist.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": ep_degree,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": ep_degree,
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
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return _pg_collection


def _ortho(matrix):
    """Deterministic stand-in for Newton-Schulz.

    Not separable across a column or row block, so orthogonalising a shard
    gives a different answer than orthogonalising the full matrix -- which is
    exactly the regression under test. The leading axis is a batch, matching
    how Muon feeds stacked expert weights to ``ortho_fn``.
    """
    norm = paddle.linalg.norm(matrix, axis=[-2, -1], keepdim=True)
    return matrix / (norm + 1e-9) + 0.5 * matrix.mean(axis=-1, keepdim=True)


class TestMuonEPFullIntermediate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg_collection = _ensure_fleet()

    def setUp(self):
        self.ep_group = self.__class__.pg_collection.ep
        self.ep_size = self.ep_group.nranks
        self.rank = dist.get_rank(self.ep_group)

        # experts and intermediate width both scale with EP so the test runs
        # unchanged on 2 or 4 cards.
        self.num_experts = 2 * self.ep_size
        self.hidden = 8
        self.intermediate = 4 * self.ep_size
        self.per_rank = self.intermediate // self.ep_size

        rng = np.random.RandomState(0)
        shape_up = (self.num_experts, self.hidden, self.intermediate)
        shape_down = (self.num_experts, self.intermediate, self.hidden)
        self.gate = paddle.to_tensor(rng.randn(*shape_up), dtype=DTYPE)
        self.up = paddle.to_tensor(rng.randn(*shape_up), dtype=DTYPE)
        self.down = paddle.to_tensor(rng.randn(*shape_down), dtype=DTYPE)

        self.cols = slice(
            self.rank * self.per_rank, (self.rank + 1) * self.per_rank
        )

    def _local_weight1(self):
        """fc1 shard: ``[E, H, 2 * I/EP]`` laid out as [gate_shard | up_shard]."""
        return paddle.concat(
            [self.gate[:, :, self.cols], self.up[:, :, self.cols]], axis=-1
        )

    def _local_weight2(self):
        """fc2 shard: ``[E, I/EP, H]``."""
        return self.down[:, self.cols, :]

    def _assert_close(self, got, want):
        self.assertEqual(list(got.shape), list(want.shape))
        np.testing.assert_allclose(
            got.astype("float64").numpy(),
            want.astype("float64").numpy(),
            rtol=0,
            atol=1e-10,
        )

    def test_weight1_ffn_split_orthogonalises_gate_and_up_separately(self):
        got = ortho_ep_full_intermediate(
            self._local_weight1(),
            _ortho,
            ep_group=self.ep_group,
            shard_axis=-1,
            gate_up=True,
            split_gate_up=True,
        )
        want = paddle.concat(
            [
                _ortho(self.gate)[:, :, self.cols],
                _ortho(self.up)[:, :, self.cols],
            ],
            axis=-1,
        )
        self._assert_close(got, want)

    def test_weight1_without_ffn_split_orthogonalises_fused_matrix(self):
        got = ortho_ep_full_intermediate(
            self._local_weight1(),
            _ortho,
            ep_group=self.ep_group,
            shard_axis=-1,
            gate_up=True,
            split_gate_up=False,
        )
        fused = _ortho(paddle.concat([self.gate, self.up], axis=-1))
        want = paddle.concat(
            [
                fused[:, :, self.cols],
                fused[:, :, self.intermediate :][:, :, self.cols],
            ],
            axis=-1,
        )
        self._assert_close(got, want)

    def test_weight2_uses_full_intermediate(self):
        got = ortho_ep_full_intermediate(
            self._local_weight2(),
            _ortho,
            ep_group=self.ep_group,
            shard_axis=-2,
        )
        self._assert_close(got, _ortho(self.down)[:, self.cols, :])

    def test_shard_local_orthogonalisation_would_differ(self):
        """Guard: the pre-fix behaviour must not match, else nothing is proven."""
        local = self._local_weight1()
        naive = paddle.concat(
            [
                _ortho(local[:, :, : self.per_rank]),
                _ortho(local[:, :, self.per_rank :]),
            ],
            axis=-1,
        )
        want = paddle.concat(
            [
                _ortho(self.gate)[:, :, self.cols],
                _ortho(self.up)[:, :, self.cols],
            ],
            axis=-1,
        )
        self.assertGreater(float((naive - want).abs().max()), 1e-6)

    def test_rejects_missing_ep_group(self):
        with self.assertRaises(ValueError):
            ortho_ep_full_intermediate(
                self._local_weight2(), _ortho, ep_group=None, shard_axis=-2
            )

    def test_rejects_non_3d_weight(self):
        with self.assertRaises(ValueError):
            ortho_ep_full_intermediate(
                paddle.randn([self.hidden, self.intermediate]).astype(DTYPE),
                _ortho,
                ep_group=self.ep_group,
                shard_axis=-1,
            )


if __name__ == "__main__":
    unittest.main()
