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

"""Multi-card (CP=2) tests for the HCA context-parallel optimizations.

The optimizations are HCA-only accelerations; CSA must keep running on the
pre-existing global-all-gather path. This file checks both halves:

  * ``all_gather_contiguous`` fuses any ``axis`` whose leading dims are 1 into a
    single collective, and still matches the buffers-plus-concat result
  * ``prepend_prev_window`` forward equals the global all-gather slice and its
    backward routes the prefix gradient to the owning rank
  * the HCA compressor's group-index sharding reproduces the global pooling,
    while the CSA (overlap) compressor never enters the sharded path
  * ``_forward_cp`` takes the one-hop window on HCA layers, and the global
    all-gather branch it replaced still matches the same non-CP reference

Run with:
    python -m paddle.distributed.launch --gpus 0,1 \
        tests/multi_card_tests/transformer/test_hca_cp_optim.py
"""

import contextlib
import types
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec

import paddlefleet.transformer.csa_attention as csa_mod
from paddlefleet.context_parallel_utils import all_gather_contiguous
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.transformer.cp_utils import all_gather_cp, prepend_prev_window
from paddlefleet.transformer.csa_attention import (
    CSA_MQA_RATIO,
    CompressedSparseAttention,
    CompressedSparseAttentionSublayersSpec,
    Compressor,
    CompressorSublayersSpec,
    CSADocMaskMetadata,
    CSAIndexer,
    CSAIndexerSublayersSpec,
)

CP_SIZE = None
CP_RANK = None
CP_GROUP = None

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


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------
class _TestLinear(nn.Layer):
    def __init__(self, input_size, output_size, dtype=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[output_size, input_size],
            dtype=dtype or DTYPE,
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight.T), None


class _TestRMSNorm(nn.Layer):
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


_COMPRESSOR_SPEC = CompressorSublayersSpec(
    linear_wkv=_TestLinear, linear_wgate=_TestLinear, norm=_TestRMSNorm
)


def _startend(doc_lens):
    """``[1, 1, seqlen, 1]`` int32 exclusive document ends."""
    rows, cum = [], 0
    for length in doc_lens:
        cum += length
        rows += [cum] * length
    return paddle.to_tensor(rows, dtype="int32").reshape([1, 1, len(rows), 1])


def _meta(doc_lens, ratio):
    startend = _startend(doc_lens)
    return CSADocMaskMetadata.build(
        ratio, 1, startend.shape[2], startend, dense_mode=False
    )


def _rel_err(actual, expected):
    a = actual.cast("float32")
    b = expected.cast("float32")
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def _local(x_global, sq):
    """This rank's contiguous CP shard along the sequence axis."""
    start = CP_RANK * sq
    return x_global[:, start : start + sq]


@contextlib.contextmanager
def _trace_cp_comm():
    """Record the CP collectives ``csa_attention`` issues during a forward."""
    seen = {"gather": [], "prepend": []}
    orig_gather, orig_prepend = (
        csa_mod.all_gather_cp,
        csa_mod.prepend_prev_window,
    )

    def gather(x, dim, group):
        seen["gather"].append(list(x.shape))
        return orig_gather(x, dim, group)

    def prepend(x, window, group):
        seen["prepend"].append((list(x.shape), window))
        return orig_prepend(x, window, group)

    csa_mod.all_gather_cp, csa_mod.prepend_prev_window = gather, prepend
    try:
        yield seen
    finally:
        csa_mod.all_gather_cp = orig_gather
        csa_mod.prepend_prev_window = orig_prepend


class TestAllGatherContiguousAxis(unittest.TestCase):
    """Fused single-buffer all-gather for axes with all-ones leading dims."""

    def _reference(self, x, axis):
        bufs = [paddle.empty(x.shape, dtype=x.dtype) for _ in range(CP_SIZE)]
        dist.stream.all_gather(
            bufs, x.contiguous(), group=CP_GROUP, use_calc_stream=True
        )
        return paddle.concat(bufs, axis=axis)

    def test_matches_buffers_plus_concat(self):
        cases = (([8, 4], 0), ([1, 8, 4], 1), ([1, 1, 8, 4], 2), ([2, 8, 4], 1))
        for shape, axis in cases:
            paddle.seed(3 + CP_RANK)
            x = paddle.randn(shape, dtype=DTYPE)
            got = all_gather_contiguous(x, group=CP_GROUP, axis=axis)
            expected_shape = list(shape)
            expected_shape[axis] *= CP_SIZE
            self.assertEqual(got.shape, expected_shape)
            np.testing.assert_array_equal(
                got.numpy(),
                self._reference(x, axis).numpy(),
                err_msg=f"shape={shape} axis={axis}",
            )

    def test_fused_branch_is_taken(self):
        """One collective into one buffer when the leading dims are all 1."""
        into_list = []
        orig = dist.stream.all_gather

        def spy(out, tensor, **kwargs):
            into_list.append(isinstance(out, (list, tuple)))
            return orig(out, tensor, **kwargs)

        dist.stream.all_gather = spy
        try:
            all_gather_contiguous(
                paddle.randn([1, 8, 4], dtype=DTYPE), group=CP_GROUP, axis=1
            )
            all_gather_contiguous(
                paddle.randn([2, 8, 4], dtype=DTYPE), group=CP_GROUP, axis=1
            )
        finally:
            dist.stream.all_gather = orig
        self.assertEqual(into_list, [False, True])


class TestPrependPrevWindowCP(unittest.TestCase):
    """One-hop window exchange versus the global all-gather it replaces."""

    def test_forward_matches_global_slice(self):
        window, sq, d = 4, 16, 8
        paddle.seed(7)
        x_global = paddle.randn([1, sq * CP_SIZE, d], dtype=DTYPE)
        x = _local(x_global, sq).clone()
        out = prepend_prev_window(x, window, CP_GROUP)

        self.assertEqual(out.shape, [1, window + sq, d])
        start = CP_RANK * sq
        if CP_RANK == 0:
            expected = paddle.concat(
                [paddle.zeros([1, window, d], dtype=DTYPE), x], axis=1
            )
        else:
            expected = x_global[:, start - window : start + sq]
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_backward_routes_prefix_grad_to_owner(self):
        window, sq, d = 4, 16, 8
        x = paddle.randn([1, sq, d], dtype=DTYPE)
        x.stop_gradient = False
        out = prepend_prev_window(x, window, CP_GROUP)
        # rank-dependent upstream grad, so each contribution is identifiable
        upstream = paddle.full(
            [1, window + sq, d], float(CP_RANK + 1), dtype=DTYPE
        )
        (out * upstream).sum().backward()

        expected = paddle.full([1, sq, d], float(CP_RANK + 1), dtype=DTYPE)
        if CP_RANK < CP_SIZE - 1:
            # the next rank sends back the gradient of the tail it borrowed
            expected[:, -window:] += float(CP_RANK + 2)
        np.testing.assert_array_equal(x.grad.numpy(), expected.numpy())


class TestCompressorGroupSharding(unittest.TestCase):
    """Pooling ``ceil(G / cp_size)`` groups per rank versus pooling all of them.

    The reference is the pre-optimization path, reachable without touching the
    network code: hand the compressor an already-gathered sequence and
    ``cp_group=None``, which disables both its internal all-gathers and the
    group sharding.
    """

    def _build(self, hidden_size, ratio, head_dim):
        config = types.SimpleNamespace(
            hidden_size=hidden_size,
            qk_pos_emb_head_dim=0,
            init_method=None,
            init_method_std=0.02,
            rms_norm_eps=1e-5,
        )
        return Compressor(
            config=config,
            sublayers_spec=_COMPRESSOR_SPEC,
            compress_ratio=ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=None,
        )

    def _compare(self, ratio, doc_lens, expect_gathers):
        sq_global = sum(doc_lens)
        sq = sq_global // CP_SIZE
        hidden_size, head_dim = 64, 32
        meta = _meta(doc_lens, ratio)

        paddle.seed(2026)
        sharded = self._build(hidden_size, ratio, head_dim)
        paddle.seed(2026)
        reference = self._build(hidden_size, ratio, head_dim)

        paddle.seed(11)
        x_global = paddle.randn([1, sq_global, hidden_size], dtype=DTYPE)
        x_a = _local(x_global, sq).clone()
        x_a.stop_gradient = False
        x_b = _local(x_global, sq).clone()
        x_b.stop_gradient = False

        with _trace_cp_comm() as seen:
            out_a = sharded(x_a, cp_group=CP_GROUP, docmask_meta=meta)
        self.assertEqual(len(seen["gather"]), expect_gathers)
        if expect_gathers == 3:
            n_shard = (meta.actual_n_compressed + CP_SIZE - 1) // CP_SIZE
            self.assertEqual(seen["gather"][2][1], n_shard)

        out_b = reference(
            all_gather_cp(x_b, dim=1, group=CP_GROUP),
            cp_group=None,
            docmask_meta=meta,
        )
        # not bitwise: the reference projects sq_global rows in one matmul
        self.assertLess(_rel_err(out_a, out_b), FWD_RTOL)

        paddle.seed(5)
        upstream = paddle.randn(out_a.shape, dtype=DTYPE)
        (out_a * upstream).sum().backward()
        (out_b * upstream).sum().backward()
        self.assertLess(_rel_err(x_a.grad, x_b.grad), BWD_RTOL)

    def test_hca_shards_divide_evenly(self):
        self._compare(ratio=128, doc_lens=[512], expect_gathers=3)

    def test_hca_last_shard_is_padded(self):
        # 300 -> 2 groups, 212 -> 1 group: 3 groups over 2 ranks
        self._compare(ratio=128, doc_lens=[300, 212], expect_gathers=3)

    def test_csa_overlap_keeps_the_global_path(self):
        # overlap pulls the previous group's projection, so groups cannot be
        # split: the sharding must stay disabled and only kv/score are gathered
        self._compare(ratio=4, doc_lens=[100, 156], expect_gathers=2)


def _csa_config(ratio, window_size, hidden_size=256, head_dim=64):
    return types.SimpleNamespace(
        num_attention_heads=8,
        v_head_dim=head_dim,
        hidden_size=hidden_size,
        q_lora_rank=64,
        qk_pos_emb_head_dim=32,
        csa_window_size=window_size,
        csa_compress_ratios=[ratio],
        csa_dense_mode=False,
        dsa_index_n_heads=16,
        dsa_index_head_dim=32,
        dsa_index_topk=16,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=False,
        csa_indexer_backend="unfused",
        csa_sparse_attn_backend="unfused",
        init_method=None,
        init_method_std=0.02,
        layernorm_epsilon=1e-5,
        num_hidden_layers=1,
    )


def _build_csa(config, ratio, cp_enabled):
    rope = RotaryEmbedding(32, rotary_percent=1.0, rotary_base=160000)
    spec = CompressedSparseAttentionSublayersSpec(
        compressor=LayerSpec(layer=Compressor, sublayers_spec=_COMPRESSOR_SPEC),
        indexer=LayerSpec(
            layer=CSAIndexer,
            sublayers_spec=CSAIndexerSublayersSpec(
                linear_wq_b=_TestLinear,
                linear_weights_proj=_TestLinear,
                compressor=LayerSpec(
                    layer=Compressor, sublayers_spec=_COMPRESSOR_SPEC
                ),
            ),
        ),
    )
    layer = CompressedSparseAttention(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        attn_mask_type=None,
        attention_type="self",
        k_channels=config.v_head_dim,
        v_channels=config.v_head_dim,
        compress_ratio=ratio,
        rotary_pos_emb=rope,
    )
    layer.cp_group = CP_GROUP if cp_enabled else None
    layer.cp_size = CP_SIZE if cp_enabled else 1
    layer.cp_rank = CP_RANK if cp_enabled else 0
    layer.cp_enabled = cp_enabled
    return layer


class TestHCAGate(unittest.TestCase):
    """``is_hca_layer`` is the only switch, and it follows the compressor."""

    def test_flag_follows_compressor_overlap(self):
        for ratio, is_hca in ((128, True), (4, False)):
            layer = _build_csa(_csa_config(ratio, 64), ratio, cp_enabled=True)
            self.assertEqual(layer.compressor.overlap, not is_hca)
            self.assertIs(layer.is_hca_layer, is_hca)

    def test_mqa_layer_has_no_compressor_and_is_not_hca(self):
        ratio = CSA_MQA_RATIO
        layer = _build_csa(_csa_config(ratio, 64), ratio, cp_enabled=True)
        self.assertIsNone(layer.compressor)
        self.assertIs(layer.is_hca_layer, False)


# 512 tokens over 2 ranks: 3 real compressed groups out of 4 slots, so the
# compressor's group sharding also hits its padded tail here.
_HCA_CASE = {"ratio": 128, "doc_lens": [300, 212], "window_size": 128}


class TestForwardCPPathSelection(unittest.TestCase):
    """``_forward_cp``: the fast window path and the path it replaced.

    Both are compared against the same non-CP reference, so the ``else`` branch
    is pinned to the pre-optimization behaviour rather than merely unused.
    """

    def _run(self, ratio, doc_lens, window_size, legacy=False):
        sq_global = sum(doc_lens)
        sq = sq_global // CP_SIZE
        config = _csa_config(ratio, window_size)
        head_dim = config.v_head_dim
        meta = _meta(doc_lens, ratio)

        paddle.seed(2026)
        ref = _build_csa(config, ratio, cp_enabled=False)
        paddle.seed(2026)
        cp = _build_csa(config, ratio, cp_enabled=True)
        if legacy:
            # what a non-HCA layer runs: global KV all-gather, window ids left
            # in global coordinates
            cp.is_hca_layer = False

        paddle.seed(1000)
        shape = [1, sq_global, config.num_attention_heads, head_dim]
        query = paddle.randn(shape, dtype=DTYPE)
        key = paddle.randn([1, sq_global, 1, head_dim], dtype=DTYPE)
        x = paddle.randn([1, sq_global, config.hidden_size], dtype=DTYPE)
        qr = paddle.randn([1, sq_global, config.q_lora_rank], dtype=DTYPE)

        def sides(tensors):
            out = []
            for t in tensors:
                t = t.clone()
                t.stop_gradient = False
                out.append(t)
            return out

        q_r, k_r, x_r, qr_r = sides([query, key, x, qr])
        out_ref = ref.forward(
            q_r, k_r, k_r, None, x=x_r, qr=qr_r, docmask_meta=meta
        )
        out_ref.sum().backward()

        q_c, k_c, x_c, qr_c = sides(
            [_local(t, sq) for t in (query, key, x, qr)]
        )
        with _trace_cp_comm() as seen:
            out_cp = cp.forward(
                q_c, k_c, k_c, None, x=x_c, qr=qr_c, docmask_meta=meta
            )
        out_cp.sum().backward()

        for p in cp.parameters():
            if p.grad is not None:
                grad = p.grad.contiguous()
                dist.all_reduce(grad, group=CP_GROUP)
                paddle.assign(grad, p.grad)

        return {
            "cp": cp,
            "seen": seen,
            "sq": sq,
            "hn": head_dim,
            "out": (out_cp, _local(out_ref, sq)),
            "dq": (q_c.grad, _local(q_r.grad, sq)),
            "dx": (x_c.grad, _local(x_r.grad, sq)),
            "dwkv": (
                cp.compressor.linear_wkv.weight.grad,
                ref.compressor.linear_wkv.weight.grad,
            ),
        }

    def _assert_close(self, got, tol=2e-2):
        for name in ("out", "dq", "dx", "dwkv"):
            actual, expected = got[name]
            err = _rel_err(actual, expected)
            self.assertLess(err, tol, f"{name} relative error {err:.3e}")

    def test_hca_takes_the_one_hop_window(self):
        got = self._run(**_HCA_CASE)
        self.assertTrue(got["cp"].is_hca_layer)
        # the raw KV crosses the wire once, window_size rows instead of a shard
        self.assertEqual(
            got["seen"]["prepend"], [([1, got["sq"], got["hn"]], 128)]
        )
        # only the compressor's kv/score/reassembly all-gathers are left
        self.assertEqual(len(got["seen"]["gather"]), 3)
        self._assert_close(got)

    def test_legacy_path_still_matches_the_reference(self):
        got = self._run(**_HCA_CASE, legacy=True)
        self.assertEqual(got["seen"]["prepend"], [])
        # the KV all-gather is back, on top of the compressor's three
        self.assertEqual(len(got["seen"]["gather"]), 4)
        self._assert_close(got)


if __name__ == "__main__":
    unittest.main()
