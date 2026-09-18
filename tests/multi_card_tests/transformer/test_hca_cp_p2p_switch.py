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

"""Multi-card (CP=2) coverage for the configurable HCA P2P compression.

The collective halves of the [P2P optim] diff that the single-card companion
cannot reach:

  * the rewritten ``NeighbourWindow`` PyLayer on both sides
    (``append_next_window`` / ``prepend_prev_window``): the forward equals the
    global slice and the backward routes the borrowed rows' gradient back to the
    rank that lent them
  * ``Compressor.forward`` with ``cp_compress_p2p=True`` -- the owner-sharded
    P2P path (two one-hop windows + one reassembly all-gather) reproduces the
    dense pooling, and with the switch off it falls back to the all-gather
    baseline (two projected all-gathers). Both are pinned to the same non-CP
    reference, so neither branch is merely executed.

Run with:
    python -m paddle.distributed.launch --gpus 0,1 \
        tests/multi_card_tests/transformer/test_hca_cp_p2p_switch.py
"""

import contextlib
import types
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet

import paddlefleet.transformer.csa_attention as csa_mod
from paddlefleet.transformer.cp_utils import (
    all_gather_cp,
    append_next_window,
    prepend_prev_window,
)
from paddlefleet.transformer.csa_attention import (
    Compressor,
    CompressorSublayersSpec,
    CSADocMaskMetadata,
)

CP_SIZE = CP_RANK = CP_GROUP = None
DTYPE = "float32"
FWD_RTOL = 1e-6
BWD_RTOL = 1e-4


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP
    world = dist.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": world,
        "sep_degree": 1,
        "cp_degree": world,
        "ep_degree": world,
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
    fleet.init(is_collective=True, strategy=strategy)
    CP_GROUP = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    CP_RANK = CP_GROUP.rank
    CP_SIZE = CP_GROUP.nranks


class _Linear(nn.Layer):
    def __init__(self, input_size, output_size, dtype=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[output_size, input_size],
            dtype=dtype or DTYPE,
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight.T), None


class _RMSNorm(nn.Layer):
    def __init__(self, hidden_size=None, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x, **kwargs):
        normed = x * paddle.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return normed * self.weight.cast(x.dtype)


_SPEC = CompressorSublayersSpec(
    linear_wkv=_Linear, linear_wgate=_Linear, norm=_RMSNorm
)


def _local(x_global, sq):
    start = CP_RANK * sq
    return x_global[:, start : start + sq]


def _rel_err(actual, expected):
    a, b = actual.cast("float32"), expected.cast("float32")
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def _doc_ends(doc_lens):
    rows, cum = [], 0
    for length in doc_lens:
        cum += length
        rows += [cum] * length
    return paddle.to_tensor(rows, dtype="int32").reshape([1, 1, len(rows), 1])


def _meta(doc_lens, ratio):
    ends = _doc_ends(doc_lens)
    return CSADocMaskMetadata.build(
        ratio, 1, ends.shape[2], ends, dense_mode=False
    )


def _build(hidden_size, ratio, head_dim, cp_compress_p2p):
    config = types.SimpleNamespace(
        hidden_size=hidden_size,
        qk_pos_emb_head_dim=0,
        init_method=None,
        init_method_std=0.02,
        rms_norm_eps=1e-5,
        cp_compress_p2p=cp_compress_p2p,
    )
    return Compressor(
        config=config,
        sublayers_spec=_SPEC,
        compress_ratio=ratio,
        head_dim=head_dim,
        rotate=False,
        rotary_pos_emb=None,
    )


@contextlib.contextmanager
def _trace():
    """Count the CP collectives the compressor issues in a forward."""
    seen = {"append": 0, "gather": 0}
    orig_append, orig_gather = (
        csa_mod.append_next_window,
        csa_mod.all_gather_cp,
    )

    def append(x, window, group):
        seen["append"] += 1
        return orig_append(x, window, group)

    def gather(x, dim, group):
        seen["gather"] += 1
        return orig_gather(x, dim, group)

    csa_mod.append_next_window, csa_mod.all_gather_cp = append, gather
    try:
        yield seen
    finally:
        csa_mod.append_next_window = orig_append
        csa_mod.all_gather_cp = orig_gather


class TestNeighbourWindowCP(unittest.TestCase):
    """The one-hop window PyLayer, both sides, vs the global slice it replaces.

    ``append_next_window`` (side="next") and ``prepend_prev_window`` (side="prev")
    are the two faces of the same rewritten ``NeighbourWindow``; testing both
    here keeps the diff covered without leaning on another file.
    """

    def test_forward_matches_global_slice(self):
        window, sq, d = 4, 16, 8
        paddle.seed(7)
        x_global = paddle.randn([1, sq * CP_SIZE, d], dtype=DTYPE)
        x = _local(x_global, sq).clone()
        start = CP_RANK * sq

        nxt = append_next_window(x, window, CP_GROUP)
        self.assertEqual(nxt.shape, [1, sq + window, d])
        want_next = (
            paddle.concat(
                [x, paddle.zeros([1, window, d], dtype=DTYPE)], axis=1
            )
            if CP_RANK == CP_SIZE - 1
            else x_global[:, start : start + sq + window]
        )
        np.testing.assert_array_equal(nxt.numpy(), want_next.numpy())

        prev = prepend_prev_window(x, window, CP_GROUP)
        self.assertEqual(prev.shape, [1, window + sq, d])
        want_prev = (
            paddle.concat(
                [paddle.zeros([1, window, d], dtype=DTYPE), x], axis=1
            )
            if CP_RANK == 0
            else x_global[:, start - window : start + sq]
        )
        np.testing.assert_array_equal(prev.numpy(), want_prev.numpy())

    def _grad(self, fn):
        window, sq, d = 4, 16, 8
        x = paddle.randn([1, sq, d], dtype=DTYPE)
        x.stop_gradient = False
        out = fn(x, window, CP_GROUP)
        # rank-dependent upstream so each contribution is identifiable
        upstream = paddle.full(out.shape, float(CP_RANK + 1), dtype=DTYPE)
        (out * upstream).sum().backward()
        return x.grad, window, sq, d

    def test_backward_next_routes_head_grad_to_prev_rank(self):
        grad, window, sq, d = self._grad(append_next_window)
        # rank-1 borrowed our first ``window`` rows and returns its value (CP_RANK)
        expected = paddle.full([1, sq, d], float(CP_RANK + 1), dtype=DTYPE)
        if CP_RANK > 0:
            expected[:, :window] += float(CP_RANK)
        np.testing.assert_array_equal(grad.numpy(), expected.numpy())

    def test_backward_prev_routes_tail_grad_to_next_rank(self):
        grad, window, sq, d = self._grad(prepend_prev_window)
        # rank+1 borrowed our last ``window`` rows and returns its value (CP_RANK+2)
        expected = paddle.full([1, sq, d], float(CP_RANK + 1), dtype=DTYPE)
        if CP_RANK < CP_SIZE - 1:
            expected[:, -window:] += float(CP_RANK + 2)
        np.testing.assert_array_equal(grad.numpy(), expected.numpy())


_RATIO = 128
_SQ_LOCAL = 2 * _RATIO


def _docs(cp_size):
    sq_global = _SQ_LOCAL * cp_size
    return [sq_global - 212, 212]


class TestCompressorSwitch(unittest.TestCase):
    """Both switch positions reproduce the dense pooling; the collectives differ.

    The reference is the pre-CP path: hand the compressor the already-gathered
    sequence with ``cp_group=None``, which disables every collective and every
    shard, so it is an independent oracle for both branches. ``all_gather_cp``
    is differentiable, so the reference's input gradient comes back on the local
    shard and is directly comparable.
    """

    def _run_switch(self, cp_compress_p2p, expect):
        docs = _docs(CP_SIZE)
        sq_global = sum(docs)
        sq = sq_global // CP_SIZE
        hidden_size, head_dim = 64, 32
        meta = _meta(docs, _RATIO)

        paddle.seed(2026)
        comp = _build(hidden_size, _RATIO, head_dim, cp_compress_p2p)
        paddle.seed(2026)
        ref = _build(hidden_size, _RATIO, head_dim, cp_compress_p2p=False)

        paddle.seed(11)
        x_global = paddle.randn([1, sq_global, hidden_size], dtype=DTYPE)
        xa = _local(x_global, sq).clone()
        xa.stop_gradient = False
        xb = _local(x_global, sq).clone()
        xb.stop_gradient = False

        with _trace() as seen:
            out = comp(xa, cp_group=CP_GROUP, docmask_meta=meta)
        self.assertEqual(seen, expect)

        out_ref = ref(
            all_gather_cp(xb, 1, CP_GROUP), cp_group=None, docmask_meta=meta
        )
        self.assertLess(_rel_err(out, out_ref), FWD_RTOL)

        paddle.seed(5)
        upstream = paddle.randn(out.shape, dtype=DTYPE)
        (out * upstream).sum().backward()
        (out_ref * upstream).sum().backward()
        self.assertLess(_rel_err(xa.grad, xb.grad), BWD_RTOL)

    def test_p2p_path(self):
        # two projected windows on the wire, one reassembly all-gather
        self._run_switch(True, {"append": 2, "gather": 1})

    def test_baseline_path(self):
        # switch off: two projected all-gathers + one reassembly, no windows
        self._run_switch(False, {"append": 0, "gather": 3})


if __name__ == "__main__":
    unittest.main()
