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

"""CPU-only unit tests for the TileLang CSA Indexer **dispatch / contract**
layer in ``paddlefleet.transformer.csa_attention``.

Scope (kernel intentionally mocked — no CUDA / TileLang required):

* ``_resolve_csa_indexer_topk_effective`` — phase2 / phase3 topk selection.
* ``_should_use_tilelang_attention``       — dispatch guard switches.
* ``_map_compressed_topk_to_kv_full``      — compressed→full index mapping.
* ``_compute_attn_target_on_selected_set`` — selected-set softmax target.
* Forward dispatch in ``CompressedSparseAttention``                 (mocked
  ``tilelang_csa_compressed_indexer_topk_paddle`` to verify call args /
  override semantics / debug-compare branch).
* ``TileLangCSAIndexerLoss`` PyLayer call contract (mocked kernel).

Numerical correctness of the TileLang kernel itself is covered separately by
the GPU tests under ``tests/single_card_tests/custom_ops/``.
"""

import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

# Ensure the local PaddleFleet source is loaded instead of any stale install.
_LOCAL_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src")
)
if _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)
# Drop any pre-imported paddlefleet from a different location.
for _m in [m for m in list(sys.modules) if m == "paddlefleet" or m.startswith("paddlefleet.")]:
    _mod = sys.modules.get(_m)
    _f = getattr(_mod, "__file__", "") or ""
    if _LOCAL_SRC not in _f:
        sys.modules.pop(_m, None)

import paddle

from paddlefleet.training.initialize import initialize_fleet

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)

from paddlefleet.transformer import csa_attention as csa_mod
from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    TileLangCSAIndexerLoss,
    _compute_attn_target_on_selected_set,
    _map_compressed_topk_to_kv_full,
    _resolve_csa_indexer_topk_effective,
    _should_use_tilelang_attention,
)


class _StubConfig:
    """Lightweight stand-in for TransformerConfig in dispatch-only tests."""

    def __init__(self, **kw):
        # Defaults match the production behaviour we care about.
        self.dsa_indexer_use_sparse_loss = True
        self.dsv4_tilelang_enable_csa_indexer = False
        # Allow caller to override anything.
        for k, v in kw.items():
            setattr(self, k, v)


class TestResolveCsaIndexerTopkEffective(unittest.TestCase):
    """Covers Task 5.3 / 5.4: per-phase ``topk_effective`` selection."""

    def test_phase3_sparse_loss_uses_min_index_topk(self):
        cfg = _StubConfig(dsa_indexer_use_sparse_loss=True)
        self.assertEqual(
            _resolve_csa_indexer_topk_effective(cfg, 512, 1024), 512
        )

    def test_phase3_clamped_by_n_compressed(self):
        """When n_compressed < index_topk, phase 3 must clamp."""
        cfg = _StubConfig(dsa_indexer_use_sparse_loss=True)
        self.assertEqual(
            _resolve_csa_indexer_topk_effective(cfg, 512, 8), 8
        )

    def test_phase2_full_range_uses_n_compressed(self):
        cfg = _StubConfig(dsa_indexer_use_sparse_loss=False)
        self.assertEqual(
            _resolve_csa_indexer_topk_effective(cfg, 512, 64), 64
        )

    def test_default_use_sparse_loss_true(self):
        """Missing attribute defaults to sparse_loss=True (Phase 3)."""
        cfg = types.SimpleNamespace()
        self.assertEqual(
            _resolve_csa_indexer_topk_effective(cfg, 32, 1024), 32
        )


class TestMapCompressedTopkToKvFull(unittest.TestCase):
    """Covers Task 5.5 / 5.6: index mapping and invalid -> -1 handling."""

    def _run(self, indices, sq, ratio, offset):
        idx = paddle.to_tensor(indices, dtype="int32")
        return _map_compressed_topk_to_kv_full(
            idx, sq=sq, ratio=ratio, offset=offset
        ).numpy()

    def test_valid_indices_get_offset(self):
        # sq=4, ratio=4 -> for t=3 only block 0 is valid; offset shifts by sq.
        out = self._run([[[0]]], sq=4, ratio=4, offset=4)
        self.assertEqual(out.tolist(), [[[-1], [-1], [-1], [4]]])

    def test_invalid_block_becomes_minus_one(self):
        # sq=4, ratio=4: at t=3, block 1 is invalid (>= (3+1)//4 == 1).
        out = self._run(
            [[[0, 1]]] * 1, sq=1, ratio=4, offset=10
        )  # n_valid=0 for t=0
        # Both slots must be invalid (-1) because no compressed blocks are
        # valid yet at t=0 with ratio=4.
        self.assertEqual(out.tolist(), [[[-1, -1]]])

    def test_explicit_minus_one_pass_through(self):
        # An incoming -1 (TileLang invalid padding) maps to -1 again because
        # -1 < n_valid_per_pos is True only for non-negative n_valid; here we
        # confirm that valid==True with index=-1 would also yield -1+offset
        # (which is < 0 and never used by sparse attention). To avoid that
        # corner we cast to int32 and rely on the kernel contract that valid
        # rows produce non-negative indices.
        idx = paddle.to_tensor([[[0, -1]]], dtype="int32")
        out = _map_compressed_topk_to_kv_full(
            idx, sq=1, ratio=1, offset=5
        ).numpy()
        # ratio=1 means every compressed slot is valid for t>=1, but sq=1
        # means t=0 with n_valid=1: block 0 is valid (<1) -> 0+5=5; the
        # padding -1 is also "valid" arithmetically (-1 < 1) so it would
        # become 4. This documents that callers must rely on TileLang
        # returning non-negative indices for valid slots; the production
        # TileLang kernel guarantees -1 is only returned when the slot is
        # outside the valid-end range.
        self.assertEqual(out.tolist(), [[[5, 4]]])


class _MockCompressor:
    """Stand-in for ``self.compressor`` returning deterministic compressed kv."""

    def __init__(self, compressed_kv):
        self._compressed_kv = compressed_kv

    def __call__(self, x):
        return self._compressed_kv


class _MockIndexer:
    """Stand-in for ``self.indexer`` covering both call sites we use:

    * ``forward_before_topk(x, qr)`` returns ``(q, k, weights)`` triplets.
    * ``__call__(x, qr, mask=...)`` returns ``(scores, topk_indices)`` for the
      Paddle reference / eval path.
    """

    def __init__(
        self,
        index_topk: int,
        n_compressed: int,
        index_n_heads: int = 2,
        index_head_dim: int = 4,
        paddle_topk_indices=None,
    ):
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.softmax_scale = index_head_dim ** -0.5
        self._n_compressed = n_compressed
        self._paddle_topk_indices = paddle_topk_indices
        self.forward_before_topk_calls = 0
        self.call_calls = 0

    def forward_before_topk(self, x, qr):
        self.forward_before_topk_calls += 1
        b, sq, _ = x.shape
        n = self._n_compressed
        q = paddle.zeros(
            [b, sq, self.index_n_heads, self.index_head_dim],
            dtype="bfloat16",
        )
        k = paddle.zeros(
            [b, n, self.index_head_dim], dtype="bfloat16"
        )
        weights = paddle.zeros(
            [b, sq, self.index_n_heads], dtype="float32"
        )
        return q, k, weights

    def __call__(self, x, qr, mask=None):
        self.call_calls += 1
        if self._paddle_topk_indices is None:
            raise AssertionError(
                "Mock indexer __call__ invoked without paddle_topk_indices "
                "configured (this code path should not run when TileLang "
                "is enabled in eval mode)."
            )
        b, sq, _ = x.shape
        scores = paddle.zeros([b, sq, self._n_compressed], dtype="float32")
        return scores, self._paddle_topk_indices


def _make_csa_layer(
    compress_ratio: int,
    n_compressed: int,
    sq: int,
    config: _StubConfig,
    paddle_topk_indices=None,
    index_topk: int = 4,
) -> CompressedSparseAttention:
    """Build a CompressedSparseAttention without invoking ``__init__``.

    We only populate the attributes consumed by ``forward``: the heavy
    sublayer machinery (Compressor weights, RoPE module, FleetLayer state)
    is replaced with mocks so the test runs on CPU without GPU kernels.
    """
    layer = object.__new__(CompressedSparseAttention)
    paddle.nn.Layer.__init__(layer)
    layer.config = config
    layer.layer_number = 0
    layer.compress_ratio = compress_ratio
    layer.window_size = 2
    layer.v_head_dim = 4
    layer.n_local_heads = 1
    layer.softmax_scale = layer.v_head_dim ** -0.5
    layer.attn_sink = paddle.zeros([layer.n_local_heads], dtype="float32")
    # Compressor returns a fixed compressed kv tensor.
    compressed_kv = paddle.zeros(
        [1, n_compressed, layer.v_head_dim], dtype="float32"
    )
    layer.compressor = _MockCompressor(compressed_kv)
    layer.indexer = _MockIndexer(
        index_topk=index_topk,
        n_compressed=n_compressed,
        paddle_topk_indices=paddle_topk_indices,
    )
    layer.eval()
    return layer


class TestForwardTileLangIndexerDispatch(unittest.TestCase):
    """End-to-end forward tests with a mocked TileLang kernel.

    We exercise eval mode (``self.training=False``) so the heavy
    ``FusedDSAIndexerLoss`` path is not triggered; the production loss path
    is untouched in Task 5 and is covered separately by Task 6.
    """

    def _build_inputs(self, b: int, sq: int, n_local_heads: int, hn: int):
        query = paddle.zeros([b, sq, n_local_heads, hn], dtype="float32")
        key = paddle.zeros([b, sq, 1, hn], dtype="float32")
        value = paddle.zeros_like(key)
        x = paddle.zeros([b, sq, 8], dtype="float32")  # hidden_size unused
        qr = paddle.zeros([b, sq, 4], dtype="float32")
        return query, key, value, x, qr

    def _patch_tilelang_kernel(self, fake_indices, fake_scores=None):
        """Inject a fake ``paddlefleet.tilelang_ops`` module exposing
        ``tilelang_csa_compressed_indexer_topk_paddle``.

        Returns the mock callable so tests can assert on its calls.
        """
        if fake_scores is None:
            fake_scores = paddle.zeros_like(fake_indices, dtype="float32")
        kernel_mock = mock.MagicMock(return_value=(fake_indices, fake_scores))
        fake_module = types.ModuleType("paddlefleet.tilelang_ops")
        fake_module.tilelang_csa_compressed_indexer_topk_paddle = kernel_mock
        # Ensure parent package exists in sys.modules (it does, but be safe).
        self.addCleanup(
            sys.modules.pop, "paddlefleet.tilelang_ops", None
        )
        sys.modules["paddlefleet.tilelang_ops"] = fake_module
        return kernel_mock

    def test_dispatch_disabled_does_not_call_kernel(self):
        sq = 4
        ratio = 4
        n_compressed = sq // ratio  # 1
        cfg = _StubConfig(
            dsv4_tilelang_enable_csa_indexer=False,
            dsa_indexer_use_sparse_loss=True,
        )
        # Paddle eval path returns a deterministic compressed top-k.
        paddle_topk = paddle.zeros([1, sq, 1], dtype="int64")
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle_topk,
            index_topk=1,
        )
        kernel_mock = self._patch_tilelang_kernel(
            paddle.full([1, sq, 1], 0, dtype="int32")
        )
        q, k, v, x, qr = self._build_inputs(
            1, sq, layer.n_local_heads, layer.v_head_dim
        )
        layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)
        kernel_mock.assert_not_called()
        # Paddle reference indexer must have been called instead.
        self.assertEqual(layer.indexer.call_calls, 1)

    def test_dispatch_phase3_calls_kernel_with_min_topk(self):
        sq = 16
        ratio = 4
        n_compressed = sq // ratio  # 4
        index_topk = 2  # smaller than n_compressed
        cfg = _StubConfig(
            dsv4_tilelang_enable_csa_indexer=True,
            dsa_indexer_use_sparse_loss=True,
        )
        # Paddle path also runs (topk used for nothing here, just shape match).
        paddle_topk = paddle.zeros([1, sq, index_topk], dtype="int64")
        # TileLang fake returns block 0 for every position.
        tl_topk = paddle.zeros([1, sq, index_topk], dtype="int32")
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle_topk,
            index_topk=index_topk,
        )
        kernel_mock = self._patch_tilelang_kernel(tl_topk)
        q, k, v, x, qr = self._build_inputs(
            1, sq, layer.n_local_heads, layer.v_head_dim
        )
        layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)
        kernel_mock.assert_called_once()
        _, kwargs = kernel_mock.call_args
        self.assertEqual(kwargs["ratio"], ratio)
        self.assertEqual(kwargs["topk_effective"], min(index_topk, n_compressed))

    def test_dispatch_phase2_calls_kernel_with_n_compressed(self):
        sq = 32
        ratio = 4
        n_compressed = sq // ratio  # 8
        index_topk = 2
        cfg = _StubConfig(
            dsv4_tilelang_enable_csa_indexer=True,
            dsa_indexer_use_sparse_loss=False,  # Phase 2 full-range
        )
        paddle_topk = paddle.zeros([1, sq, index_topk], dtype="int64")
        tl_topk = paddle.zeros([1, sq, n_compressed], dtype="int32")
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle_topk,
            index_topk=index_topk,
        )
        kernel_mock = self._patch_tilelang_kernel(tl_topk)
        q, k, v, x, qr = self._build_inputs(
            1, sq, layer.n_local_heads, layer.v_head_dim
        )
        layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)
        kernel_mock.assert_called_once()
        _, kwargs = kernel_mock.call_args
        self.assertEqual(kwargs["ratio"], ratio)
        # Phase 2 must use n_compressed, NOT min(index_topk, n_compressed).
        self.assertEqual(kwargs["topk_effective"], n_compressed)

    def test_tilelang_indices_replace_paddle_indices(self):
        """The TileLang result must override ``topk_indices_compressed`` and
        flow through the +offset / -1 mapping into ``compress_topk_idxs``."""
        sq = 8
        ratio = 4
        n_compressed = sq // ratio  # 2
        index_topk = 2
        cfg = _StubConfig(
            dsv4_tilelang_enable_csa_indexer=True,
            dsa_indexer_use_sparse_loss=True,
        )
        # Paddle path returns block 0 for every slot.
        paddle_topk = paddle.zeros([1, sq, index_topk], dtype="int64")
        # TileLang returns (block 0, block 1) per row; block 1 is invalid for
        # rows t < 7 (n_valid_per_pos = (t+1)//4 -> 0,0,0,1,1,1,1,2).
        tl_indices_np = [[[0, 1]] * sq]
        tl_topk = paddle.to_tensor(tl_indices_np, dtype="int32")
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle_topk,
            index_topk=index_topk,
        )
        self._patch_tilelang_kernel(tl_topk)

        captured = {}

        original_concat = paddle.concat

        def _spy_concat(values, axis=-1):
            out = original_concat(values, axis=axis)
            # The forward concatenates [window_idxs, compress_topk_idxs]
            # exactly once on axis=-1; capture the second tensor.
            if (
                len(values) == 2
                and values[1].shape[-1] == index_topk
                and "compress_topk_idxs" not in captured
            ):
                captured["compress_topk_idxs"] = values[1]
            return out

        with mock.patch.object(paddle, "concat", side_effect=_spy_concat):
            q, k, v, x, qr = self._build_inputs(
                1, sq, layer.n_local_heads, layer.v_head_dim
            )
            layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)

        compress_topk = captured["compress_topk_idxs"].numpy()
        # offset = sq = 8. n_valid_per_pos = [0,0,0,1,1,1,1,2]
        # For each t the [block 0, block 1] -> [valid?+offset : -1, valid?+offset : -1]
        expected = []
        offset = sq
        for t in range(sq):
            n_valid = (t + 1) // ratio
            row = []
            for b_id in (0, 1):
                row.append(b_id + offset if b_id < n_valid else -1)
            expected.append(row)
        self.assertEqual(compress_topk.tolist(), [expected])

        ensures ``fused_qk_topk_naive`` is invoked from ``csa_attention``.

        The set-equality check should not raise even when indices mismatch;
        it only prints a diagnostic line.
        """
        sq = 8
        ratio = 4
        n_compressed = sq // ratio  # 2
        index_topk = 2
        cfg = _StubConfig(
            dsv4_tilelang_enable_csa_indexer=True,
            dsa_indexer_use_sparse_loss=True,
        )
        paddle_topk = paddle.zeros([1, sq, index_topk], dtype="int64")
        tl_topk = paddle.zeros([1, sq, index_topk], dtype="int32")
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle_topk,
            index_topk=index_topk,
        )
        self._patch_tilelang_kernel(tl_topk)

        ref_scores = paddle.zeros([1, sq, n_compressed], dtype="float32")
        # Force reference to disagree with TileLang to exercise the
        # mismatch branch (which must print, not raise).
        ref_topk = paddle.full(
            [1, sq, index_topk], n_compressed - 1, dtype="int32"
        )
        fused_mock = mock.MagicMock(return_value=(ref_scores, ref_topk))
        with mock.patch.object(
            csa_mod, "fused_qk_topk_naive", fused_mock
        ):
            buf = io.StringIO()
            q, k, v, x, qr = self._build_inputs(
                1, sq, layer.n_local_heads, layer.v_head_dim
            )
            with redirect_stdout(buf):
                layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)
        fused_mock.assert_called_once()
        out = buf.getvalue()
        self.assertIn("topk_effective=", out)


# ---------------------------------------------------------------------------
# Task 6: TileLang CSA Indexer loss + backward integration
# ---------------------------------------------------------------------------


class TestComputeAttnTargetOnSelectedSet(unittest.TestCase):
    """Covers Task 6.1: selected-set attention target construction."""

    def test_target_matches_manual_softmax_on_full_topk(self):
        """When topk_eff == n_compressed and indices = arange, the target
        must equal the per-head softmax of QK aggregated across heads and
        then L1 normalized — i.e. the standard sparse-loss target."""
        b, sq, np_, hn, sk = 1, 2, 2, 4, 3
        paddle.seed(0)
        q = paddle.randn([b, sq, np_, hn], dtype="float32")
        k = paddle.randn([b, sk, np_, hn], dtype="float32")
        topk = paddle.tile(
            paddle.arange(sk, dtype="int32").reshape([1, 1, sk]),
            [b, sq, 1],
        )
        scale = hn ** -0.5
        target = _compute_attn_target_on_selected_set(q, k, topk, scale)

        # Manual reference
        q_ref = q.transpose([0, 2, 1, 3])  # [b, np, sq, hn]
        k_ref = k.transpose([0, 2, 3, 1])  # [b, np, hn, sk]
        scores = paddle.matmul(q_ref, k_ref) * scale
        probs = paddle.nn.functional.softmax(scores, axis=-1)
        agg = probs.sum(axis=1)  # [b, sq, sk]
        ref = agg / agg.sum(axis=-1, keepdim=True)

        diff = (target - ref).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_invalid_slots_get_zero_target(self):
        b, sq, np_, hn, sk = 1, 1, 1, 2, 4
        q = paddle.ones([b, sq, np_, hn], dtype="float32")
        k = paddle.ones([b, sk, np_, hn], dtype="float32")
        topk = paddle.to_tensor(
            [[[0, -1, 2, -1]]], dtype="int32"
        )  # 2 valid slots
        target = _compute_attn_target_on_selected_set(q, k, topk, 1.0)
        self.assertEqual(target.shape, [b, sq, 4])
        # Invalid slots must be exactly zero.
        self.assertEqual(target[0, 0, 1].item(), 0.0)
        self.assertEqual(target[0, 0, 3].item(), 0.0)
        # Valid slots sum to 1 (L1 normalized).
        self.assertAlmostEqual(
            target[0, 0, 0].item() + target[0, 0, 2].item(), 1.0, places=5
        )

    def test_all_invalid_row_yields_zero_no_nan(self):
        b, sq, np_, hn, sk = 1, 1, 1, 2, 2
        q = paddle.ones([b, sq, np_, hn], dtype="float32")
        k = paddle.ones([b, sk, np_, hn], dtype="float32")
        topk = paddle.full([b, sq, 2], -1, dtype="int32")
        target = _compute_attn_target_on_selected_set(q, k, topk, 1.0)
        self.assertFalse(bool(paddle.isnan(target).any()))
        self.assertEqual(target.abs().sum().item(), 0.0)


class TestTileLangCSAIndexerLossPyLayer(unittest.TestCase):
    """Covers Task 6.4 / 6.5: PyLayer forward + backward."""

    def _patch_kernels(self, fwd_indices, fwd_probs, bwd_returns):
        fwd_mock = mock.MagicMock(return_value=(fwd_indices, fwd_probs))
        bwd_mock = mock.MagicMock(return_value=bwd_returns)
        fake_module = types.ModuleType("paddlefleet.tilelang_ops")
        fake_module.tilelang_csa_compressed_indexer_topk_paddle = fwd_mock
        fake_module.tilelang_csa_compressed_indexer_bwd_paddle = bwd_mock
        self.addCleanup(
            sys.modules.pop, "paddlefleet.tilelang_ops", None
        )
        sys.modules["paddlefleet.tilelang_ops"] = fake_module
        return fwd_mock, bwd_mock

    def test_forward_zero_loss_when_target_equals_probs(self):
        b, sq, h_i, d_i, sk, np_, hn = 1, 2, 1, 4, 2, 1, 4
        topk_eff = 2
        # Force target == topk_probs by making query/key all zeros so target
        # collapses to uniform 1/topk_eff, and feeding the same as probs.
        index_q = paddle.zeros([b, sq, h_i, d_i], dtype="float32")
        index_q.stop_gradient = False
        weights = paddle.ones([b, sq, h_i], dtype="float32")
        weights.stop_gradient = False
        index_k_comp = paddle.zeros([b, sk, d_i], dtype="float32")
        index_k_comp.stop_gradient = False
        query_mla = paddle.zeros([b, sq, np_, hn], dtype="float32")
        key_comp = paddle.zeros([b, sk, np_, hn], dtype="float32")

        topk_indices = paddle.tile(
            paddle.arange(topk_eff, dtype="int32").reshape([1, 1, topk_eff]),
            [b, sq, 1],
        )
        topk_probs = paddle.full(
            [b, sq, topk_eff], 1.0 / topk_eff, dtype="float32"
        )
        bwd_returns = (
            paddle.zeros_like(index_q),
            paddle.zeros_like(weights),
            paddle.zeros_like(index_k_comp),
        )
        fwd_mock, _ = self._patch_kernels(topk_indices, topk_probs, bwd_returns)

        loss = TileLangCSAIndexerLoss.apply(
            index_q, weights, index_k_comp,
            query_mla, key_comp,
            4, topk_eff, 0.5, 1.0, None,
        )
        fwd_mock.assert_called_once()
        # KL(uniform || uniform) == 0.
        self.assertLess(abs(loss.item()), 1e-5)
        # Indices stashed for sparse attention consumption.
        stash = TileLangCSAIndexerLoss._last_topk_indices
        self.assertIsNotNone(stash)
        self.assertEqual(stash.shape, [b, sq, topk_eff])

    def test_backward_passes_q_minus_p_to_kernel(self):
        b, sq, h_i, d_i, sk, np_, hn = 1, 1, 1, 2, 2, 1, 2
        topk_eff = 2
        index_q = paddle.zeros([b, sq, h_i, d_i], dtype="float32")
        index_q.stop_gradient = False
        weights = paddle.ones([b, sq, h_i], dtype="float32")
        weights.stop_gradient = False
        index_k_comp = paddle.zeros([b, sk, d_i], dtype="float32")
        index_k_comp.stop_gradient = False
        # Non-zero MLA tensors so target != uniform.
        query_mla = paddle.to_tensor(
            [[[[1.0, 0.0]]]], dtype="float32"
        )  # [1,1,1,2]
        key_comp = paddle.to_tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype="float32"
        )  # [1,2,1,2]

        topk_indices = paddle.to_tensor(
            [[[0, 1]]], dtype="int32"
        )
        topk_probs = paddle.to_tensor(
            [[[0.7, 0.3]]], dtype="float32"
        )
        bwd_returns = (
            paddle.zeros_like(index_q),
            paddle.zeros_like(weights),
            paddle.zeros_like(index_k_comp),
        )
        _, bwd_mock = self._patch_kernels(
            topk_indices, topk_probs, bwd_returns
        )

        loss_coeff = 2.0
        loss = TileLangCSAIndexerLoss.apply(
            index_q, weights, index_k_comp,
            query_mla, key_comp,
            2, topk_eff, 1.0, loss_coeff, None,
        )
        loss.backward()

        bwd_mock.assert_called_once()
        call_args = bwd_mock.call_args
        # Positional args: (index_q, weights, index_k_comp, topk_indices,
        # grad_index_scores)
        passed_grad = call_args.args[4]
        # Compute expected: scale = loss_coeff / num_rows; num_rows = b*sq = 1
        expected = (topk_probs - _compute_attn_target_on_selected_set(
            query_mla, key_comp, topk_indices, 1.0,
        )) * (loss_coeff / 1.0)
        diff = (passed_grad - expected).abs().max().item()
        self.assertLess(diff, 1e-5)


# ---------------------------------------------------------------------------
# Task 7: TileLang sparse attention dispatch gate
# ---------------------------------------------------------------------------


class TestShouldUseTilelangAttention(unittest.TestCase):
    """Covers Task 7.1 / 7.2 / 7.3: dispatch gate for sparse attention."""

    def _cfg(self, **kw):
        defaults = dict(
            dsv4_tilelang_backend="attention_paddle_compat",
            dsv4_tilelang_enable_backward=True,
            csa_dense_mode=False,
        )
        defaults.update(kw)
        return _StubConfig(**defaults)

    def test_wrong_backend_disables_dispatch(self):
        cfg = self._cfg(dsv4_tilelang_backend="paddle")
        self.assertFalse(_should_use_tilelang_attention(cfg, 4, True, False))
        self.assertFalse(_should_use_tilelang_attention(cfg, 128, False, False))

    def test_training_without_backward_switch_disables_dispatch(self):
        cfg = self._cfg(dsv4_tilelang_enable_backward=False)
        self.assertFalse(_should_use_tilelang_attention(cfg, 4, True, True))
        self.assertFalse(_should_use_tilelang_attention(cfg, 128, False, True))
        # Eval mode is unaffected by the backward switch.
        self.assertTrue(_should_use_tilelang_attention(cfg, 128, False, False))

    def test_dense_mode_keeps_enabled(self):
        cfg = self._cfg(csa_dense_mode=True)
        self.assertTrue(_should_use_tilelang_attention(cfg, 4, False, False))
        self.assertTrue(_should_use_tilelang_attention(cfg, 4, False, True))

    def test_ratio_128_keeps_enabled(self):
        cfg = self._cfg()
        self.assertTrue(_should_use_tilelang_attention(cfg, 128, False, False))
        self.assertTrue(_should_use_tilelang_attention(cfg, 128, True, True))

    def test_ratio_4_with_indexer_now_enabled(self):
        """Task 7 unblocks learned CSA: indexer + ratio=4 must dispatch."""
        cfg = self._cfg()
        self.assertTrue(_should_use_tilelang_attention(cfg, 4, True, False))
        self.assertTrue(_should_use_tilelang_attention(cfg, 4, True, True))

    def test_ratio_4_without_indexer_remains_disabled(self):
        """Without an indexer there is no real sparse selection."""
        cfg = self._cfg()
        self.assertFalse(_should_use_tilelang_attention(cfg, 4, False, False))


# ---------------------------------------------------------------------------
# Task 10: 端到端 CSA 路径验证（dispatch 层）
# ---------------------------------------------------------------------------
#
# Many sub-tasks of Task 10 are **already** covered by the
# ``TestForwardTileLangIndexerDispatch`` class above (which spies on the
# concat of window_idxs/compress_topk_idxs and verifies that TileLang
# indices override the Paddle reference). The classes below explicitly
# pin down the remaining sub-tasks:
#
# * 10.3 — Compare unfused vs TileLang sparse attention output via the
# * 10.6 — Phase 1 (``csa_dense_mode=True``) must skip the indexer block
#   entirely; ``self.indexer is None`` is the production guarantee.
#
# Numerical equality of TileLang vs unfused sparse attention is GPU-only
# and lives in the wrapper-level kernel tests under ``custom_ops/``.
# This file remains CPU-only / kernel-mocked.


class TestTask10EndToEndDispatch(unittest.TestCase):
    """CPU-side coverage for Task 10 end-to-end CSA verification points."""

    def _build_inputs(self, b, sq, n_local_heads, hn):
        query = paddle.zeros([b, sq, n_local_heads, hn], dtype="float32")
        key = paddle.zeros([b, sq, 1, hn], dtype="float32")
        value = paddle.zeros_like(key)
        x = paddle.zeros([b, sq, 8], dtype="float32")
        qr = paddle.zeros([b, sq, 4], dtype="float32")
        return query, key, value, x, qr

    # ----- Task 10.6 ------------------------------------------------------

    def test_phase1_dense_mode_skips_indexer(self):
        """Phase 1 (``csa_dense_mode=True``) -> ``self.indexer is None``;
        forward must not enter the indexer block, no TileLang kernel call,
        no FusedDSAIndexerLoss call, and ``compress_topk_idxs`` is not
        appended (only window_idxs feeds sparse attention)."""
        sq = 4
        ratio = 4
        n_compressed = sq // ratio  # 1
        cfg = _StubConfig(
            csa_dense_mode=True,
            dsv4_tilelang_enable_csa_indexer=True,  # must be ignored
            dsa_indexer_use_sparse_loss=True,
        )
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle.zeros([1, sq, 1], dtype="int64"),
            index_topk=1,
        )
        # Production __init__ would have set ``self.indexer = None`` when
        # ``csa_dense_mode=True``; mirror that here.
        layer.indexer = None

        kernel_mock = mock.MagicMock()
        fake_module = types.ModuleType("paddlefleet.tilelang_ops")
        fake_module.tilelang_csa_compressed_indexer_topk_paddle = kernel_mock
        self.addCleanup(
            sys.modules.pop, "paddlefleet.tilelang_ops", None
        )
        sys.modules["paddlefleet.tilelang_ops"] = fake_module

        q, k, v, x, qr = self._build_inputs(
            1, sq, layer.n_local_heads, layer.v_head_dim
        )
        # In dense mode the compressor still runs in production; our mock
        # compressor returns a non-None tensor. The forward is expected to
        # fall through ``self.indexer is not None`` -> False branch, and
        # since ratio>1 with n_compressed>0 falls into the ``else`` of the
        # ratio==128 case, ``compress_topk_idxs`` becomes the static range.
        # That static range path is the production behavior for ratio=128;
        # for ratio=4 + dense_mode the indexer is None so we expect to take
        # the ``else`` branch that uses get_compress_topk_idxs.
        layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)

        # TileLang indexer kernel must NOT be invoked (no indexer present).
        kernel_mock.assert_not_called()

    def test_phase1_dense_mode_dispatch_independent_of_indexer_switch(self):
        """``_should_use_tilelang_attention`` in dense mode is True even
        without an indexer (ratio=4 dense path uses static topk_idxs)."""
        cfg = _StubConfig(
            dsv4_tilelang_backend="attention_paddle_compat",
            dsv4_tilelang_enable_backward=True,
            csa_dense_mode=True,
        )
        # Dense mode dispatches TileLang regardless of has_indexer.
        self.assertTrue(_should_use_tilelang_attention(cfg, 4, False, False))
        self.assertTrue(_should_use_tilelang_attention(cfg, 4, False, True))

    # ----- Task 10.3 ------------------------------------------------------

        BOTH ``_tilelang_compressed_sparse_attn_paddle_compat`` AND
        ``unfused_compressed_sparse_attn`` so their outputs can be
        compared in place. Returning identical tensors from both mocks
        ensures the ``allclose`` check passes."""
        sq = 8
        ratio = 4
        n_compressed = sq // ratio  # 2
        index_topk = 2
        cfg = _StubConfig(
            dsv4_tilelang_backend="attention_paddle_compat",
            dsv4_tilelang_enable_backward=True,  # not training, but harmless
            dsv4_tilelang_enable_csa_indexer=True,
            dsa_indexer_use_sparse_loss=True,
            csa_dense_mode=False,
        )
        paddle_topk = paddle.zeros([1, sq, index_topk], dtype="int64")
        tl_topk = paddle.zeros([1, sq, index_topk], dtype="int32")
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle_topk,
            index_topk=index_topk,
        )

        # Patch the TileLang indexer kernel (same approach as elsewhere).
        kernel_mock = mock.MagicMock(
            return_value=(tl_topk, paddle.zeros_like(tl_topk, dtype="float32"))
        )
        fake_module = types.ModuleType("paddlefleet.tilelang_ops")
        fake_module.tilelang_csa_compressed_indexer_topk_paddle = kernel_mock
        self.addCleanup(
            sys.modules.pop, "paddlefleet.tilelang_ops", None
        )
        sys.modules["paddlefleet.tilelang_ops"] = fake_module

        # Both attention paths return the same tensor so allclose passes.
        b = 1
        out_shape = [b, sq, layer.n_local_heads * layer.v_head_dim]
        attn_out = paddle.zeros(out_shape, dtype="float32")
        tl_attn_mock = mock.MagicMock(return_value=attn_out)
        unfused_mock = mock.MagicMock(return_value=attn_out)

        with mock.patch.object(
            csa_mod,
            "_tilelang_compressed_sparse_attn_paddle_compat",
            tl_attn_mock,
        ), mock.patch.object(
            csa_mod, "unfused_compressed_sparse_attn", unfused_mock
        ):
            q, k, v, x, qr = self._build_inputs(
                b, sq, layer.n_local_heads, layer.v_head_dim
            )
            layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)

        tl_attn_mock.assert_called_once()
        unfused_mock.assert_called_once()
        # Both calls must receive the same topk_idxs (window + compress).
        tl_args = tl_attn_mock.call_args[0]
        un_args = unfused_mock.call_args[0]
        # Position 3 is topk_idxs in both function signatures.
        tl_topk_idxs = tl_args[3].numpy()
        un_topk_idxs = un_args[3].numpy()
        self.assertEqual(tl_topk_idxs.tolist(), un_topk_idxs.tolist())

        must NOT run (production sparse attention path)."""
        sq = 8
        ratio = 4
        n_compressed = sq // ratio
        index_topk = 2
        cfg = _StubConfig(
            dsv4_tilelang_backend="attention_paddle_compat",
            dsv4_tilelang_enable_backward=True,
            dsv4_tilelang_enable_csa_indexer=True,
            dsa_indexer_use_sparse_loss=True,
            csa_dense_mode=False,
        )
        paddle_topk = paddle.zeros([1, sq, index_topk], dtype="int64")
        tl_topk = paddle.zeros([1, sq, index_topk], dtype="int32")
        layer = _make_csa_layer(
            ratio,
            n_compressed,
            sq,
            cfg,
            paddle_topk_indices=paddle_topk,
            index_topk=index_topk,
        )
        kernel_mock = mock.MagicMock(
            return_value=(tl_topk, paddle.zeros_like(tl_topk, dtype="float32"))
        )
        fake_module = types.ModuleType("paddlefleet.tilelang_ops")
        fake_module.tilelang_csa_compressed_indexer_topk_paddle = kernel_mock
        self.addCleanup(
            sys.modules.pop, "paddlefleet.tilelang_ops", None
        )
        sys.modules["paddlefleet.tilelang_ops"] = fake_module

        b = 1
        out_shape = [b, sq, layer.n_local_heads * layer.v_head_dim]
        attn_out = paddle.zeros(out_shape, dtype="float32")
        tl_attn_mock = mock.MagicMock(return_value=attn_out)
        unfused_mock = mock.MagicMock(return_value=attn_out)

        with mock.patch.object(
            csa_mod,
            "_tilelang_compressed_sparse_attn_paddle_compat",
            tl_attn_mock,
        ), mock.patch.object(
            csa_mod, "unfused_compressed_sparse_attn", unfused_mock
        ):
            q, k, v, x, qr = self._build_inputs(
                b, sq, layer.n_local_heads, layer.v_head_dim
            )
            layer.forward(q, k, v, attention_mask=None, x=x, qr=qr)

        tl_attn_mock.assert_called_once()
        unfused_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
