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

"""``FLAGS_use_dsv4_accuracy`` must gate every DSV4 attention replay call site.

The flag defaults to 0. With it off the attention numeric paths have to stay
exactly where they were before the DSV4 replay landed - the DSA indexer, the
DSv4 hybrid attention output projection / qkv ordering, and the CSA sparse
attention kernels all have a "historical" path and a "replay" path, and the
flag is the only thing allowed to switch between them. Other alignment targets
run with ``use_accuracy_compatible=True`` but *without* this flag, so a DSV4
branch that keys off the older switches would silently move their loss curve.

Each test below pins one attention call site on both sides of the flag with an
observable difference: either a numeric agreement between the two spellings, or
a code-path assertion (``assert_called`` / ``assert_not_called``) proving the
replay helper only runs when the flag is on. Every test targets the smallest
isolatable helper that contains the branch so it runs on a single card with no
distributed process group.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
from paddle import nn

import paddlefleet.utils as paddlefleet_utils
from paddlefleet import accuracy_compatible_patch, triton_ops
from paddlefleet.cudnn_ops.indexer import csa_indexer_fwd_cudnn
from paddlefleet.fusions import csa_sparse_attn
from paddlefleet.transformer import (
    csa_attention,
    dsa_attention,
    dsv4_hybrid_attention,
)
from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    CompressedSparseAttentionSublayersSpec,
    Compressor,
    CompressorSublayersSpec,
)
from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig


def _dsv4_flag(module, enabled):
    """Pin ``use_dsv4_accuracy_compatible`` on the object that reads it.

    Most call sites import the flag into their own module namespace, so the
    patch target is that module. ``csa_sparse_attn`` re-imports it *inside* the
    function from ``paddlefleet.utils``, so for that one the module passed in is
    ``paddlefleet.utils`` itself.
    """
    return patch.object(
        module, "use_dsv4_accuracy_compatible", return_value=enabled
    )


class TestDsaHadamardGating(unittest.TestCase):
    """``dsa_attention.hadamard_transform`` (line 99).

    Flag off runs the pure-Paddle butterfly transform; flag on hands the work to
    the ``CompatibleHadamard`` PyLayer (the fast_hadamard_transform replay). The
    branch is the only chooser, so the replay class must be untouched when the
    flag is off and must be the sole producer when the flag is on.
    """

    def setUp(self):
        paddle.seed(0)
        # dim must be a power of two for the butterfly transform.
        self.x = paddle.to_tensor(
            np.random.RandomState(0).randn(2, 8).astype("float32")
        )

    def test_flag_off_uses_pure_paddle_butterfly_and_skips_compatible(self):
        with (
            _dsv4_flag(dsa_attention, False),
            patch.object(
                accuracy_compatible_patch, "CompatibleHadamard"
            ) as compat,
        ):
            out = dsa_attention.hadamard_transform(self.x, 1.0)
            # Applying the transform twice must scale by the dimension: the
            # Hadamard matrix H satisfies H @ H = dim * I, which pins the real
            # butterfly maths without needing the (unavailable) fast kernel.
            round_trip = dsa_attention.hadamard_transform(out, 1.0)

        compat.apply.assert_not_called()
        np.testing.assert_allclose(
            round_trip.numpy(), self.x.numpy() * 8, rtol=1e-5, atol=1e-5
        )

    def test_flag_on_routes_to_compatible_hadamard(self):
        sentinel = paddle.zeros_like(self.x)
        with (
            _dsv4_flag(dsa_attention, True),
            patch.object(
                accuracy_compatible_patch, "CompatibleHadamard"
            ) as compat,
        ):
            compat.apply.return_value = sentinel
            out = dsa_attention.hadamard_transform(self.x, 2.0)

        compat.apply.assert_called_once()
        called_x, called_scale = compat.apply.call_args.args
        np.testing.assert_array_equal(called_x.numpy(), self.x.numpy())
        self.assertEqual(called_scale, 2.0)
        self.assertIs(out, sentinel)


class TestDsaIndexerBackwardGradKEinsum(unittest.TestCase):
    """``dsa_attention._bwd_fused_indexer_loss`` grad_k (line 1079).

    ``grad_k`` is computed with ``paddle.einsum('bsht,bshd->btd', ...)`` by
    default, and through ``accuracy_compatible_patch.compatible_einsum`` (the
    Torch ``'sbht,sbhd->tbd'`` spelling with the surrounding transposes) when
    the flag is on. The two spellings are the same contraction, so the flag must
    (a) leave grad_k numerically unchanged and (b) only route through the replay
    helper when it is on.
    """

    def _inputs(self):
        rs = np.random.RandomState(0)
        b, sq, sk, h, d, np_heads, hn = 1, 3, 3, 2, 4, 1, 4
        return {
            "q": paddle.to_tensor(rs.randn(b, sq, h, d).astype("float32")),
            "weights": paddle.to_tensor(rs.randn(b, sq, h).astype("float32")),
            "k": paddle.to_tensor(rs.randn(b, sk, d).astype("float32")),
            "query": paddle.to_tensor(
                rs.randn(b, sq, np_heads, hn).astype("float32")
            ),
            "key": paddle.to_tensor(
                rs.randn(b, sk, np_heads, hn).astype("float32")
            ),
            "topk_indices": paddle.to_tensor(
                np.array([[[0, 1], [1, 2], [0, 2]]]).astype("int64")
            ),
            "grad_loss": paddle.to_tensor(np.array(1.0, dtype="float32")),
        }

    def _run(self, args):
        return dsa_attention._bwd_fused_indexer_loss(
            args["q"],
            args["weights"],
            args["k"],
            args["query"],
            args["key"],
            args["topk_indices"],
            0.5,
            1.0,
            True,
            args["grad_loss"],
            None,
        )

    def test_flag_off_uses_paddle_einsum_for_grad_k(self):
        args = self._inputs()
        with (
            _dsv4_flag(dsa_attention, False),
            patch.object(
                accuracy_compatible_patch, "compatible_einsum"
            ) as compat,
        ):
            _, _, grad_k = self._run(args)

        compat.assert_not_called()
        self.assertEqual(grad_k.shape, args["k"].shape)

    def test_flag_on_routes_grad_k_through_compatible_einsum(self):
        args = self._inputs()
        with _dsv4_flag(dsa_attention, False):
            _, _, grad_k_off = self._run(args)

        def _torch_spelling(grad_scores_sbht, q_sbhd):
            return paddle.einsum(
                "sbht,sbhd->tbd",
                grad_scores_sbht.cast("float32"),
                q_sbhd.cast("float32"),
            )

        wrapped = MagicMock(side_effect=_torch_spelling)
        with (
            _dsv4_flag(dsa_attention, True),
            patch.object(
                accuracy_compatible_patch, "compatible_einsum", wrapped
            ),
        ):
            _, _, grad_k_on = self._run(args)

        wrapped.assert_called_once()
        np.testing.assert_allclose(
            grad_k_on.numpy(), grad_k_off.numpy(), rtol=1e-6, atol=1e-6
        )


class TestOGroupProjectionGating(unittest.TestCase):
    """``DSv4HybridSelfAttention._o_group_proj_forward`` (line 1421).

    Flag on runs the grouped output projection through ``CompatibleOGroupProjection``
    (the seq-first Torch replay that also stashes an fp32 weight-grad); flag off
    (with fp8 disabled) falls to the historical ``fused_grouped_matmul`` path.
    Only the flag may pick between them.
    """

    OG = 2
    R = 4

    def _fake_self(self):
        s = types.SimpleNamespace()
        s.o_local_groups = self.OG
        s.config = types.SimpleNamespace(
            o_lora_rank=self.R, fp8=None, full_fp8_computation=False
        )
        s.linear_o_group_proj = paddle.to_tensor(
            np.random.RandomState(0).randn(self.OG, self.R, 4).astype("float32")
        )
        s.layer_number = 1
        return s

    def _run(self, enabled):
        core_in = paddle.to_tensor(
            np.random.RandomState(1).randn(1, 2, self.OG * 4).astype("float32")
        )
        with (
            _dsv4_flag(dsv4_hybrid_attention, enabled),
            patch.object(
                dsv4_hybrid_attention,
                "inspect_tensor",
                side_effect=lambda name, layer, tensor: tensor,
            ),
            patch.object(
                dsv4_hybrid_attention,
                "deferred_grouped_dw_accumulator",
                return_value=None,
            ),
            patch.object(
                accuracy_compatible_patch, "CompatibleOGroupProjection"
            ) as compat,
            patch.object(triton_ops, "fused_grouped_matmul") as fused,
        ):
            compat.apply.return_value = paddle.zeros([1, 2, self.OG, 4])
            fused.return_value = paddle.zeros([1, 2, self.OG * 4])
            DSv4HybridSelfAttention._o_group_proj_forward(
                self._fake_self(), core_in
            )
        return compat, fused

    def test_flag_off_uses_fused_grouped_matmul(self):
        compat, fused = self._run(False)
        compat.apply.assert_not_called()
        fused.assert_called_once()

    def test_flag_on_routes_through_compatible_ogroup_projection(self):
        compat, fused = self._run(True)
        compat.apply.assert_called_once()
        fused.assert_not_called()


class TestQkvForwardGating(unittest.TestCase):
    """``DSv4HybridSelfAttention._qkv_forward`` (line 1481).

    Flag off (and grad-enabled, non-detached input) rebuilds q/k/v through the
    ``_FixedOrderQKV`` PyLayer so the recompute segment replays projections in a
    fixed order; flag on takes the plain ``get_query_key_value_tensors`` call.
    The ``value is key`` alias contract must hold either way.
    """

    def _fake_self(self):
        s = types.SimpleNamespace()
        self._query = paddle.zeros([1, 2, 3])
        self._key = paddle.zeros([1, 2, 3])
        self._q_compressed = paddle.zeros([1, 1])
        s.get_query_key_value_tensors = MagicMock(
            return_value=(
                self._query,
                self._key,
                self._key,  # value aliases key
                self._q_compressed,
                None,
            )
        )
        return s

    def _run(self, enabled):
        s = self._fake_self()
        hidden = paddle.zeros([1, 2, 3])
        hidden.stop_gradient = False
        with (
            _dsv4_flag(dsv4_hybrid_attention, enabled),
            patch.object(dsv4_hybrid_attention, "_FixedOrderQKV") as fixed,
        ):
            fixed.apply.return_value = (
                self._query,
                self._key,
                self._q_compressed,
            )
            DSv4HybridSelfAttention._qkv_forward(s, hidden, 0, None)
        return s.get_query_key_value_tensors, fixed

    def test_flag_off_uses_fixed_order_qkv(self):
        gqkv, fixed = self._run(False)
        fixed.apply.assert_called_once()
        gqkv.assert_not_called()

    def test_flag_on_calls_get_query_key_value_tensors_directly(self):
        gqkv, fixed = self._run(True)
        gqkv.assert_called_once()
        fixed.apply.assert_not_called()


class _StubLinear(nn.Layer):
    """Weight-shape ``[in, out]`` linear matching ``linear_bf16_fp32`` (x @ w)."""

    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[in_size, out_size],
            dtype="float32",
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x.cast("float32"), self.weight).cast(x.dtype), None


class _StubNorm(nn.Layer):
    def __init__(self, hidden_size=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size or 1],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = 1e-5

    def forward(self, x, **kwargs):
        return x


def _make_compressor_config():
    return types.SimpleNamespace(
        hidden_size=256,
        qk_pos_emb_head_dim=64,
        init_method=None,
        init_method_std=0.02,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        use_fp8_qat=False,
        swa_high_precision_norm=True,
        use_fast_hadamard=False,
        high_precision_rope=False,
    )


def _make_min_csa_config():
    """Smallest ``TransformerConfig`` that builds an MQA CSA layer.

    ``compress_ratio = -1`` (``CSA_MQA_RATIO``) skips the Compressor/Indexer
    sub-builds, so ``__init__`` reaches the ``_cast_to_low_precision`` branch
    without any tensor-parallel projection machinery.
    """
    return TransformerConfig(
        num_hidden_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        v_head_dim=32,
        qk_pos_emb_head_dim=16,
        params_dtype=paddle.bfloat16,
        bf16=True,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        csa_compress_ratios=[-1],
        csa_sparse_attn_backend="unfused",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
    )


class _FakeGroup:
    nranks = 1
    world_size = 1
    ranks = [0]
    rank = 0


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeGroup()
        self.cp = _FakeGroup()


class TestCompressorCastToLowPrecisionGating(unittest.TestCase):
    """``Compressor.__init__`` (line 1610).

    Flag off pins ``_cast_to_low_precision = False`` so the compressor keeps its
    fp32 params out of AMP low-precision casting (the historical contract); flag
    on leaves the nn.Layer default (``True``) untouched.
    """

    def _build(self, enabled):
        spec = CompressorSublayersSpec(
            linear_wkv=_StubLinear,
            linear_wgate=_StubLinear,
            norm=_StubNorm,
        )
        with _dsv4_flag(csa_attention, enabled):
            comp = Compressor(
                config=_make_compressor_config(),
                sublayers_spec=spec,
                compress_ratio=2,
                head_dim=128,
                rotate=False,
                rotary_pos_emb=None,
            )
        return comp._cast_to_low_precision

    def test_flag_off_disables_low_precision_cast(self):
        self.assertFalse(self._build(False))

    def test_flag_on_keeps_default_low_precision_cast(self):
        self.assertTrue(self._build(True))


class TestCsaAttentionCastToLowPrecisionGating(unittest.TestCase):
    """``CompressedSparseAttention.__init__`` (line 2411).

    Same ``_cast_to_low_precision`` contract as the Compressor: flag off forces
    it ``False``, flag on keeps the nn.Layer default ``True``.
    """

    def _build(self, enabled):
        with _dsv4_flag(csa_attention, enabled):
            core = CompressedSparseAttention(
                config=_make_min_csa_config(),
                sublayers_spec=CompressedSparseAttentionSublayersSpec(),
                layer_number=0,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                pg_collection=_FakePGCollection(),
                compress_ratio=-1,
            )
        return core._cast_to_low_precision

    def test_flag_off_disables_low_precision_cast(self):
        self.assertFalse(self._build(False))

    def test_flag_on_keeps_default_low_precision_cast(self):
        self.assertTrue(self._build(True))


class TestIndexerCompressedTopkBackendGating(unittest.TestCase):
    """``CompressedSparseAttention._compute_indexer_compressed_topk_idxs`` (line 2510).

    The flag overrides the configured indexer backend to ``"unfused"``. With a
    configured ``"cudnn"`` backend, flag off keeps the cuDNN branch (which calls
    ``indexer.forward_before_topk`` + the cuDNN kernel), while flag on forces the
    unfused branch (which calls the indexer module directly). ``training=False``
    keeps the loss path out so the branch choice is the only observable.
    """

    def _fake_self(self, backend):
        s = types.SimpleNamespace()
        s.config = types.SimpleNamespace(
            csa_indexer_backend=backend,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
        )
        s.training = False
        s.compress_ratio = 2
        s.softmax_scale = 0.5
        s.tp_group = None
        s.layer_number = 0
        topk = paddle.zeros([1, 3, 2], dtype="int64")
        s.indexer = MagicMock()
        s.indexer.forward_before_topk = MagicMock(
            return_value=(MagicMock(), MagicMock(), MagicMock())
        )
        s.indexer.return_value = (None, topk)
        s._resolve_topk_effective = MagicMock(return_value=2)
        self._topk = topk
        return s

    def _run(self, enabled):
        s = self._fake_self("cudnn")
        query = paddle.zeros([1, 3, 2, 4])
        x = paddle.zeros([1, 3, 8])
        qr = paddle.zeros([1, 3, 8])
        compressed_kv = paddle.zeros([1, 4, 4])
        with (
            _dsv4_flag(csa_attention, enabled),
            patch.object(
                csa_attention,
                "_build_compressed_causal_mask",
                return_value=paddle.zeros([1, 3, 4]),
            ),
            patch.object(csa_attention, "get_valid_range", return_value=None),
            patch.object(
                csa_attention,
                "_map_compressed_topk_to_kv_full",
                return_value="SENTINEL",
            ),
            patch.object(
                csa_indexer_fwd_cudnn,
                "cudnn_indexer_topk_fwd",
                return_value=(self._topk, None),
            ) as cudnn,
        ):
            csa_attention.CompressedSparseAttention._compute_indexer_compressed_topk_idxs(
                s, query, x, qr, compressed_kv, 4, 0
            )
        return s, cudnn

    def test_flag_off_keeps_configured_cudnn_backend(self):
        s, cudnn = self._run(False)
        s.indexer.forward_before_topk.assert_called_once()
        cudnn.assert_called_once()
        s.indexer.assert_not_called()

    def test_flag_on_forces_unfused_backend(self):
        s, cudnn = self._run(True)
        s.indexer.assert_called_once()
        s.indexer.forward_before_topk.assert_not_called()
        cudnn.assert_not_called()


class TestCompressedSparseAttnBackendGating(unittest.TestCase):
    """``CompressedSparseAttention.compressed_sparse_attn`` (line 3700).

    The flag overrides the configured sparse-attention backend to ``"unfused"``.
    The chosen backend is passed straight to ``csa_sparse_attn`` as its
    ``backend`` kwarg, so patching that call reveals which path won.
    """

    def _run(self, enabled):
        s = types.SimpleNamespace()
        s.config = types.SimpleNamespace(csa_sparse_attn_backend="tilelang")
        s.indexer = None
        s.global_kv_idx_remap_fusion = False
        query = paddle.zeros([1, 2, 2, 4])
        kv_full = paddle.zeros([1, 5, 4])
        attn_sink = paddle.zeros([2])
        topk_idxs = paddle.zeros([1, 2, 3], dtype="int64")
        with (
            _dsv4_flag(csa_attention, enabled),
            patch.object(csa_sparse_attn, "csa_sparse_attn") as sparse,
            patch.object(
                csa_sparse_attn,
                "_csa_bwd_honours_topk_length_holes",
                return_value=True,
            ),
        ):
            sparse.return_value = "OUT"
            CompressedSparseAttention.compressed_sparse_attn(
                s, query, kv_full, attn_sink, topk_idxs, 0.5
            )
        return sparse.call_args.kwargs["backend"]

    def test_flag_off_keeps_configured_backend(self):
        self.assertEqual(self._run(False), "tilelang")

    def test_flag_on_forces_unfused_backend(self):
        self.assertEqual(self._run(True), "unfused")


class TestUnfusedSparseAttnSinkSoftmaxGating(unittest.TestCase):
    """``csa_sparse_attn.unfused_compressed_sparse_attn`` (line 383).

    The flag is re-imported from ``paddlefleet.utils`` *inside* the function, so
    it is pinned on ``paddlefleet.utils``. Flag on routes the sink-softmax through
    ``CompatibleCSASinkSoftmax``; flag off runs the inline stable softmax. The two
    forward maths are identical, so the outputs must agree and the replay PyLayer
    must only run when the flag is on.
    """

    def setUp(self):
        b, sq, np_heads, hn, n_kv = 1, 3, 2, 4, 5
        self.query = paddle.to_tensor(
            np.random.RandomState(1)
            .randn(b, sq, np_heads, hn)
            .astype("float32")
        )
        self.kv_full = paddle.to_tensor(
            np.random.RandomState(2).randn(b, n_kv, hn).astype("float32")
        )
        self.attn_sink = paddle.to_tensor(
            np.random.RandomState(3).randn(np_heads).astype("float32")
        )
        self.topk_indices = paddle.to_tensor(
            np.array([[[0, 1, 2], [1, 2, 3], [0, 2, 4]]]).astype("int64")
        )

    def _call(self):
        return csa_sparse_attn.unfused_compressed_sparse_attn(
            self.query, self.kv_full, self.attn_sink, self.topk_indices, 0.5
        )

    def test_both_sides_agree_numerically(self):
        with _dsv4_flag(paddlefleet_utils, False):
            off = self._call()
        with _dsv4_flag(paddlefleet_utils, True):
            on = self._call()

        np.testing.assert_allclose(
            on.numpy(), off.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_flag_on_routes_through_compatible_sink_softmax(self):
        real = accuracy_compatible_patch.CompatibleCSASinkSoftmax
        wrapped = MagicMock(wraps=real)
        wrapped.apply = MagicMock(side_effect=real.apply)
        with (
            _dsv4_flag(paddlefleet_utils, True),
            patch.object(
                accuracy_compatible_patch,
                "CompatibleCSASinkSoftmax",
                wrapped,
            ),
        ):
            self._call()
        wrapped.apply.assert_called_once()

    def test_flag_off_skips_compatible_sink_softmax(self):
        with (
            _dsv4_flag(paddlefleet_utils, False),
            patch.object(
                accuracy_compatible_patch, "CompatibleCSASinkSoftmax"
            ) as compat,
        ):
            self._call()
        compat.apply.assert_not_called()


class TestQRmsNormGating(unittest.TestCase):
    """``dsv4_hybrid_attention._q_rms_norm`` (lines 99, 101).

    ``_q_rms_norm`` normalizes the query by its per-row RMS with no learnable
    weight. Flag off runs the pure-Paddle spelling (``q * rsqrt(mean(q^2)+eps)``,
    optionally in fp32 for ``high_precision_norm``); flag on hands the work to
    the ``CompatibleQRMSNorm`` PyLayer (the Torch replay, which additionally
    supplies the fp32 backward). The two forward maths are identical, so the
    flag must (a) leave the output numerically unchanged and (b) only route
    through the replay PyLayer when it is on. The flag is imported into the
    module namespace, and ``CompatibleQRMSNorm`` is imported *inside* the
    function from ``accuracy_compatible_patch``, so the spy is pinned there.
    """

    def setUp(self):
        self.q = paddle.to_tensor(
            np.random.RandomState(0).randn(2, 8).astype("float32")
        )
        self.eps = 1e-6

    def _reference(self):
        return (
            self.q
            * paddle.rsqrt(self.q.square().mean(-1, keepdim=True) + self.eps)
        ).numpy()

    def test_flag_on_routes_through_compatible_qrms_norm(self):
        real = accuracy_compatible_patch.CompatibleQRMSNorm
        wrapped = MagicMock(wraps=real)
        wrapped.apply = MagicMock(side_effect=real.apply)
        with (
            _dsv4_flag(dsv4_hybrid_attention, True),
            patch.object(
                accuracy_compatible_patch, "CompatibleQRMSNorm", wrapped
            ),
        ):
            out = dsv4_hybrid_attention._q_rms_norm(
                self.q, self.eps, high_precision_norm=True, use_fusion=False
            )

        wrapped.apply.assert_called_once()
        called_q, called_eps = wrapped.apply.call_args.args
        np.testing.assert_array_equal(called_q.numpy(), self.q.numpy())
        self.assertEqual(called_eps, self.eps)
        np.testing.assert_allclose(
            out.numpy(), self._reference(), rtol=1e-6, atol=1e-6
        )

    def test_flag_off_uses_the_pure_paddle_path(self):
        with (
            _dsv4_flag(dsv4_hybrid_attention, False),
            patch.object(
                accuracy_compatible_patch, "CompatibleQRMSNorm"
            ) as compat,
        ):
            out = dsv4_hybrid_attention._q_rms_norm(
                self.q, self.eps, high_precision_norm=True, use_fusion=False
            )

        compat.apply.assert_not_called()
        np.testing.assert_allclose(
            out.numpy(), self._reference(), rtol=1e-6, atol=1e-6
        )

    def test_both_sides_agree_numerically(self):
        with _dsv4_flag(dsv4_hybrid_attention, False):
            off = dsv4_hybrid_attention._q_rms_norm(
                self.q, self.eps, high_precision_norm=False, use_fusion=False
            )
        with _dsv4_flag(dsv4_hybrid_attention, True):
            on = dsv4_hybrid_attention._q_rms_norm(
                self.q, self.eps, high_precision_norm=False, use_fusion=False
            )

        np.testing.assert_allclose(
            on.numpy(), off.numpy(), rtol=1e-6, atol=1e-6
        )


class TestCsaForwardCpTilelangIndexerGating(unittest.TestCase):
    """``CompressedSparseAttention._forward_cp`` tilelang-indexer branch (line 3468).

    SKIPPED: line 3468 is the ``else`` (tilelang) arm of the indexer-backend
    dispatch *inside* ``_forward_cp`` -- the context-parallel forward path. It is
    only reachable after ``_forward_cp`` has already issued CP collectives on a
    real process group: ``all_gather_cp(kv_local, group=self.cp_group)`` /
    ``prepend_prev_window(..., self.cp_group)`` (lines 3319/3326), the compressor
    all-gather (line 3359), and ``self.indexer.forward_before_topk(..., cp_group=
    self.cp_group)`` (line 3419). Even with ``cp_size == 1`` the method is written
    around ``self.cp_group`` and the compressor/indexer submodules, and the branch
    itself dispatches to the ``paddlefleet.tilelang_ops.csa_indexer_topk_fwd`` GPU
    kernel. There is no single-card, no-process-group way to reach it without
    standing up a CP group and the full CSA compressor+indexer stack, which
    violates the single-card rules.
    """

    def test_forward_cp_tilelang_indexer_requires_cp_group(self):
        self.skipTest(
            "csa_attention.py:3468 is inside _forward_cp; reaching the tilelang "
            "indexer branch needs a context-parallel process group (all_gather_cp "
            "/ prepend_prev_window on self.cp_group) plus the compressor+indexer "
            "stack and a tilelang GPU kernel -- not single-card isolatable."
        )


if __name__ == "__main__":
    unittest.main()
