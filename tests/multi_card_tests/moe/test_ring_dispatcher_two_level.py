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
"""Two-level (G>1 AND N>1) coverage for RingMoETokenDispatcher.

``test_ring_dispatcher_ep.py`` runs on two cards, so it can only reach G=2/N=1
(intra only) or G=1/N=2 (inter only) -- never both levels at once. That leaves
the interesting interactions uncovered: the per-round intra AllGather/
ReduceScatter running *inside* an inter-node rotation, and the deferred wait on
the in-flight output reduce (``_RingReduceScatterAsync``), which degenerates to
a no-op whenever ``intra_group`` is None.

Four cards with ``_RING_GPUS_PER_NODE`` patched to 2 give G=2, N=2:
  intra groups [0,1] [2,3]   inter groups [0,2] [1,3]

With an expert fn that is linear in the tokens, the ring's output is analytic:
every rank's own rows come back scaled by EP, because the intra ReduceScatter
sums G identical partials and the inter ReduceScatter sums N of those.

Run with:
  python -m paddle.distributed.launch --gpus=0,1,2,3 \
      tests/multi_card_tests/moe/test_ring_dispatcher_two_level.py
"""

import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.training.initialize import initialize_fleet

_pg_collection = None


def _ensure_fleet():
    """Bring up a 4-rank EP-only topology (or reuse one already standing)."""
    global _pg_collection
    if _pg_collection is not None:
        return _pg_collection
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 4,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 4,
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
    if getattr(fleet.fleet, "_hcg", None) is None:
        initialize_fleet(strategy=strategy)
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ep = _pg_collection.ep
    if ep is None or ep.nranks != 4:
        # The G/N assertions below assume EP==4; reusing a foreign topology
        # would silently test a different ring, so fail loudly.
        raise RuntimeError(
            f"expected a 4-rank EP group, got {None if ep is None else ep.nranks}"
        )
    return _pg_collection


def _dispatcher(ep_group, num_experts, gpus_per_node=2):
    from paddlefleet.transformer.moe import token_dispatcher as td

    with mock.patch.object(td, "_RING_GPUS_PER_NODE", gpus_per_node):
        return td.RingMoETokenDispatcher(
            ep_group, ep_group.nranks, num_experts=num_experts
        )


def _scale_expert_fn(scale):
    """Linear in tokens, ignores routing -- output is analytically predictable."""

    def expert_fn(g_tok, g_idx, g_w, use_fp8, **kwargs):
        return g_tok * scale

    return expert_fn


def _weighted_expert_fn():
    """Uses ``g_w`` so the padded-lane mask and router gradient are observable."""

    def expert_fn(g_tok, g_idx, g_w, use_fp8, **kwargs):
        return g_tok * g_w.sum(axis=-1, keepdim=True)

    return expert_fn


class TestTwoLevelRing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg_collection = _ensure_fleet()

    def setUp(self):
        paddle.seed(1234)
        np.random.seed(1234)
        self.ep_group = self.__class__.pg_collection.ep
        self.ep_size = self.ep_group.nranks
        self.rank = dist.get_rank(self.ep_group)
        self.num_experts = 4
        self.d_latent = 8
        self.T_local = 4
        self.K = 2

    def _tokens(self, requires_grad=True):
        # Distinct per rank so a mis-routed rotation cannot pass by symmetry.
        x = paddle.full(
            [self.T_local, self.d_latent], float(self.rank + 1), dtype="float32"
        )
        x = x + paddle.arange(self.T_local, dtype="float32").reshape([-1, 1])
        x.stop_gradient = not requires_grad
        return x

    def _routing(self, pad_last=False):
        idx = paddle.to_tensor(
            [
                [i % self.num_experts, (i + 1) % self.num_experts]
                for i in range(self.T_local)
            ],
            dtype="int32",
        )
        if pad_last:
            # Mark the final token's lanes as padding; its weights must be
            # zeroed by the mask that ring_forward hoists out of the loop.
            idx[-1, :] = -1
        # Rank- AND row-dependent on purpose. With a uniform weight a wrong
        # home-node slice in ring_forward would be invisible, because every
        # node's routing would be interchangeable -- verified by mutation test.
        w = paddle.full(
            [self.T_local, self.K], 0.25 * (self.rank + 1), dtype="float32"
        ) + 0.01 * paddle.arange(self.T_local, dtype="float32").reshape([-1, 1])
        w.stop_gradient = False
        return idx, w

    def test_topology_is_two_level(self):
        disp = _dispatcher(self.ep_group, self.num_experts)
        self.assertEqual((disp.G, disp.N), (2, 2))
        self.assertIsNotNone(disp.intra_group)
        self.assertIsNotNone(disp.inter_group)
        self.assertEqual(disp.intra_group.nranks, 2)
        self.assertEqual(disp.inter_group.nranks, 2)

    def test_forward_returns_own_rows_scaled_by_ep(self):
        """intra RS sums G partials, inter RS sums N of them -> EP*scale*x."""
        disp = _dispatcher(self.ep_group, self.num_experts)
        x = self._tokens()
        idx, w = self._routing()
        out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
        self.assertEqual(out.shape, [self.T_local, self.d_latent])
        np.testing.assert_allclose(
            out.numpy(),
            (x.detach() * 2.0 * self.ep_size).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_backward_scales_token_grad_by_ep(self):
        disp = _dispatcher(self.ep_group, self.num_experts)
        x = self._tokens()
        idx, w = self._routing()
        out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.numpy(),
            np.full(
                [self.T_local, self.d_latent],
                2.0 * self.ep_size,
                dtype="float32",
            ),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_router_weights_are_applied_and_masked(self):
        """Padded lanes (idx<0) must contribute nothing.

        Guards the mask hoist: ring_forward zeroes cur_w once on the local
        [T, K] instead of once per round on the gathered [G*T, K].
        """
        disp = _dispatcher(self.ep_group, self.num_experts)
        x = self._tokens(requires_grad=False)
        idx, w = self._routing(pad_last=True)
        out = disp.ring_forward(x, w, idx, _weighted_expert_fn(), w.dtype)
        s = paddle.where(idx < 0, paddle.zeros_like(w), w).sum(
            axis=-1, keepdim=True
        )
        np.testing.assert_allclose(
            out.numpy(),
            (x * s * self.ep_size).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        # The padded row must be exactly zero, not merely small.
        np.testing.assert_array_equal(
            out.numpy()[-1], np.zeros([self.d_latent], dtype="float32")
        )

    def test_router_grad_is_zero_on_padded_lanes(self):
        disp = _dispatcher(self.ep_group, self.num_experts)
        x = self._tokens(requires_grad=False)
        idx, w = self._routing(pad_last=True)
        out = disp.ring_forward(x, w, idx, _weighted_expert_fn(), w.dtype)
        out.sum().backward()
        self.assertIsNotNone(w.grad)
        np.testing.assert_array_equal(
            w.grad.numpy()[-1], np.zeros([self.K], dtype="float32")
        )
        # Non-padded lanes must actually receive gradient.
        self.assertGreater(float(np.abs(w.grad.numpy()[:-1]).sum()), 0.0)

    def test_repeated_forwards_are_stable(self):
        """A deferred reduce must not leak state across calls.

        ``_RingReduceScatterAsync`` hands its task to the caller; if a handle
        were ever left undrained the next call would read a half-written buffer,
        which shows up as a second iteration disagreeing with the first.
        """
        disp = _dispatcher(self.ep_group, self.num_experts)
        idx, w = self._routing()
        first = None
        for _ in range(3):
            x = self._tokens(requires_grad=False)
            out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
            if first is None:
                first = out.numpy()
            else:
                np.testing.assert_allclose(
                    out.numpy(), first, rtol=1e-6, atol=1e-6
                )

    def test_async_reduce_handle_is_drained(self):
        """_rs_async must leave a task behind, and _drain must consume it."""
        from paddlefleet.transformer.moe import token_dispatcher as td

        disp = _dispatcher(self.ep_group, self.num_experts)
        part = paddle.randn([self.T_local * disp.G, self.d_latent])
        handle = {}
        out = disp._rs_async(part, disp.intra_group, handle)
        self.assertIn("task", handle)
        td.RingMoETokenDispatcher._drain([handle])
        self.assertEqual(out.shape, [self.T_local, self.d_latent])
        # Draining a taskless handle (degenerate group) must be a no-op.
        self.assertIs(disp._rs_async(part, None, {}), part)
        td.RingMoETokenDispatcher._drain([{}])


if __name__ == "__main__":
    unittest.main()
