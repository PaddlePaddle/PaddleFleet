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
"""Behavior tests for ``softmax_scale`` in ``DotProductAttention``.

Two independent contracts are exercised:

1. Constructor derivation of ``softmax_scale`` / ``_has_custom_softmax_scale``
   from the config (``apply_query_key_layer_scaling``), an explicit override,
   and the per-head dim. These are pure Python and run on CPU.

2. Propagation of that derived scale into each attention kernel the forward can
   dispatch to (SDPA, the packed-sequence flashmask path, the main flashmask
   path) and the refined-recompute guard that rejects a custom scale. The flash
   kernels are genuine *not-under-test* collaborators, so they are mocked with a
   distinguishable marker return; the tests assert both the exact ``scale`` /
   ``softmax_scale`` argument the forward hands them AND that the kernel output
   is actually consumed (reshaped and returned), never that they were merely
   "called". The default-scale variants are kept as negative controls so a
   forward that always (or never) forwards the scale is rejected.

Paddle is not guaranteed to be importable here; the whole file is skipped with
an honest reason when the real imports fail, and only
``ImportError``/``ModuleNotFoundError`` count as "dependency missing" so a
compile break or API rename surfaces instead of being silently skipped.
The CP (context_parallel) kernels need a real >1-rank process group and are NOT
certified from this single-process file.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}"

# Module path used for patching collaborators exactly where forward looks them
# up.
_MOD = "paddlefleet.transformer.dot_product_attention"

# --- fixture dimensions (small, distinguishable, non-degenerate) ---
_HEAD_DIM = 32
_NUM_HEADS = 4
_HIDDEN = _HEAD_DIM * _NUM_HEADS
_SEQ = 4
_BATCH = 1


def _make_config(**overrides):
    """Minimal but real ``TransformerConfig`` for a dense attention layer.

    Only the plumbing needed to instantiate ``DotProductAttention`` is set;
    every numeric expectation in the tests is derived independently and never
    read back from this object.
    """
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": _HIDDEN,
        "num_attention_heads": _NUM_HEADS,
        "num_key_value_heads": _NUM_HEADS,
        "head_dim": _HEAD_DIM,
        "softmax_scale": None,
        "use_bias": True,
        "recompute_granularity": None,
        "recompute_modules": None,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "fa_version": None,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _build_attn(config, layer_number=1, **kwargs):
    return DotProductAttention(
        config=config,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        **kwargs,
    )


def _bf16_qkv():
    """Distinguishable bf16 q/k/v of shape ``[B, S, H, D]`` (flash paths only
    accept fp16/bf16)."""
    rng = np.random.RandomState(0)
    shape = (_BATCH, _SEQ, _NUM_HEADS, _HEAD_DIM)
    q = paddle.to_tensor(
        rng.standard_normal(shape).astype(np.float32) * 0.5, dtype="bfloat16"
    )
    k = paddle.to_tensor(
        rng.standard_normal(shape).astype(np.float32) * 0.5, dtype="bfloat16"
    )
    v = paddle.to_tensor(
        rng.standard_normal(shape).astype(np.float32) * 0.5, dtype="bfloat16"
    )
    return q, k, v


def _marker():
    """A distinguishable kernel-output stand-in: reshaping it wrong (or
    dropping it) changes the observed forward output."""
    flat = paddle.arange(
        _BATCH * _SEQ * _NUM_HEADS * _HEAD_DIM, dtype="float32"
    )
    return flat.reshape([_BATCH, _SEQ, _NUM_HEADS, _HEAD_DIM]).astype(
        "bfloat16"
    )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSoftmaxScaleDerivation(unittest.TestCase):
    """Constructor derivation of ``softmax_scale`` and the load-bearing flag.

    The flag ``_has_custom_softmax_scale`` is what later decides whether an
    explicit ``scale`` is forwarded to the kernels, so each numeric case also
    pins the flag.
    """

    def test_default_scale_is_inv_sqrt_per_head_dim_not_config_head_dim(self):
        # per-head dim = projection_size / num_heads = (k_channels * H) / H =
        # k_channels. Passing k_channels=64 while head_dim=32 proves the scale
        # follows the *per-head* dim (64), not config.head_dim (32).
        attn = _build_attn(_make_config(), k_channels=64, v_channels=64)
        self.assertFalse(attn._has_custom_softmax_scale)
        self.assertAlmostEqual(
            attn.softmax_scale, 1.0 / math.sqrt(64), places=6
        )
        self.assertNotAlmostEqual(
            attn.softmax_scale, 1.0 / math.sqrt(_HEAD_DIM), places=6
        )

    def test_explicit_scale_is_stored_verbatim_and_sets_flag(self):
        attn = _build_attn(_make_config(), softmax_scale=0.123)
        self.assertTrue(attn._has_custom_softmax_scale)
        self.assertEqual(attn.softmax_scale, 0.123)

    def test_layer_scaling_clamps_layer_number_to_at_least_one(self):
        # coeff = max(1, layer_number); layer_number=0 must clamp to 1 (no
        # divide-by-zero, no divide-by-0-index), leaving the base scale intact
        # but still arming the custom-scale flag.
        base = 1.0 / math.sqrt(_HEAD_DIM)
        attn = _build_attn(
            _make_config(apply_query_key_layer_scaling=True), layer_number=0
        )
        self.assertTrue(attn._has_custom_softmax_scale)
        self.assertAlmostEqual(attn.softmax_scale, base, places=6)

    def test_layer_scaling_divides_base_scale_by_coeff(self):
        base = 1.0 / math.sqrt(_HEAD_DIM)
        attn = _build_attn(
            _make_config(apply_query_key_layer_scaling=True), layer_number=4
        )
        self.assertTrue(attn._has_custom_softmax_scale)
        self.assertAlmostEqual(attn.softmax_scale, base / 4, places=6)

    def test_explicit_scale_is_also_divided_by_layer_coeff(self):
        # A user-supplied scale is still divided by coeff when QK-layer-scaling
        # is on: 0.8 / max(1, 4) = 0.2.
        attn = _build_attn(
            _make_config(apply_query_key_layer_scaling=True),
            layer_number=4,
            softmax_scale=0.8,
        )
        self.assertTrue(attn._has_custom_softmax_scale)
        self.assertAlmostEqual(attn.softmax_scale, 0.2, places=6)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSdpaSoftmaxScalePropagation(unittest.TestCase):
    """SDPA path (no startend indices, no packed seq): the derived scale is
    forwarded via ``scale=`` iff it is custom, and the kernel output is
    consumed."""

    def _run(self, attn, mock_sdpa):
        q, k, v = _bf16_qkv()
        marker = _marker()
        mock_sdpa.return_value = marker
        attn.eval()
        out = attn(q, k, v, None, attn_mask_startend_row_indices=None)
        return out, mock_sdpa.call_args

    @patch(f"{_MOD}.paddle.nn.functional.scaled_dot_product_attention")
    def test_custom_scale_forwarded_and_output_consumed(self, mock_sdpa):
        attn = _build_attn(_make_config(), softmax_scale=0.25)
        out, call = self._run(attn, mock_sdpa)
        self.assertEqual(call.kwargs["scale"], 0.25)
        # Output is the kernel marker reshaped to [B, S, H*Dv] -- proves the
        # return value flows through instead of being discarded.
        np.testing.assert_array_equal(
            out.numpy(), _marker().reshape([_BATCH, _SEQ, _HIDDEN]).numpy()
        )

    @patch(f"{_MOD}.paddle.nn.functional.scaled_dot_product_attention")
    def test_default_scale_not_forwarded(self, mock_sdpa):
        attn = _build_attn(_make_config())
        _, call = self._run(attn, mock_sdpa)
        self.assertNotIn("scale", call.kwargs)

    @patch(f"{_MOD}.paddle.nn.functional.scaled_dot_product_attention")
    def test_layer_scaled_scale_forwarded_with_exact_value(self, mock_sdpa):
        attn = _build_attn(
            _make_config(apply_query_key_layer_scaling=True), layer_number=4
        )
        _, call = self._run(attn, mock_sdpa)
        self.assertIn("scale", call.kwargs)
        self.assertAlmostEqual(
            call.kwargs["scale"], (1.0 / math.sqrt(_HEAD_DIM)) / 4, places=6
        )


def _packed_seq_params():
    class _PSP:
        # Two segments of length 2 over the seq of length 4.
        cu_seqlens_kv = paddle.to_tensor([0, 2, 4])

    return _PSP()


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestFlashmaskPackedSoftmaxScalePropagation(unittest.TestCase):
    """Packed-sequence flashmask path: the derived scale is forwarded via the
    ``softmax_scale=`` kwarg iff custom, and the kernel output is consumed."""

    @patch(f"{_MOD}.flashmask_attention")
    def test_custom_scale_forwarded_and_output_consumed(self, mock_fm):
        attn = _build_attn(_make_config(), softmax_scale=0.3)
        attn.eval()
        mock_fm.return_value = _marker()
        q, k, v = _bf16_qkv()
        out = attn(
            q,
            k,
            v,
            None,
            attn_mask_startend_row_indices=None,
            packed_seq_params=_packed_seq_params(),
        )
        call = mock_fm.call_args
        self.assertEqual(call.kwargs["softmax_scale"], 0.3)
        # Packed path attends within-segment only: causal must be False.
        self.assertFalse(call.kwargs["causal"])
        np.testing.assert_array_equal(
            out.numpy(), _marker().reshape([_BATCH, _SEQ, _HIDDEN]).numpy()
        )

    @patch(f"{_MOD}.flashmask_attention")
    def test_default_scale_not_forwarded(self, mock_fm):
        attn = _build_attn(_make_config())
        attn.eval()
        mock_fm.return_value = _marker()
        q, k, v = _bf16_qkv()
        attn(
            q,
            k,
            v,
            None,
            attn_mask_startend_row_indices=None,
            packed_seq_params=_packed_seq_params(),
        )
        self.assertNotIn("softmax_scale", mock_fm.call_args.kwargs)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestFlashmaskMainSoftmaxScalePropagation(unittest.TestCase):
    """Main flashmask path (startend indices given, non-CP, non-packed): the
    derived scale is forwarded via ``softmax_scale=`` iff custom, and the
    kernel output is consumed."""

    def _startend(self):
        return paddle.zeros([_BATCH, 1, _SEQ, _SEQ], dtype="int32")

    @patch(f"{_MOD}.flashmask_attention")
    def test_custom_scale_forwarded_and_output_consumed(self, mock_fm):
        attn = _build_attn(_make_config(), softmax_scale=0.7)
        attn.eval()
        mock_fm.return_value = _marker()
        q, k, v = _bf16_qkv()
        out = attn(
            q,
            k,
            v,
            None,
            attn_mask_startend_row_indices=self._startend(),
        )
        self.assertEqual(mock_fm.call_args.kwargs["softmax_scale"], 0.7)
        np.testing.assert_array_equal(
            out.numpy(), _marker().reshape([_BATCH, _SEQ, _HIDDEN]).numpy()
        )

    @patch(f"{_MOD}.flashmask_attention")
    def test_default_scale_not_forwarded(self, mock_fm):
        attn = _build_attn(_make_config())
        attn.eval()
        mock_fm.return_value = _marker()
        q, k, v = _bf16_qkv()
        attn(
            q,
            k,
            v,
            None,
            attn_mask_startend_row_indices=self._startend(),
        )
        self.assertNotIn("softmax_scale", mock_fm.call_args.kwargs)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestRefinedRecomputeRejectsCustomScale(unittest.TestCase):
    """Refined-recompute flash paths do not support a custom softmax_scale and
    must raise ``NotImplementedError`` before dispatching. Only the
    single-process (non-CP) branches are certified here; the CP branch needs a
    real >1-rank group."""

    @patch(f"{_MOD}.flashmask_attention")
    def test_packed_rr_custom_scale_raises_and_does_not_dispatch(self, mock_fm):
        attn = _build_attn(_make_config(), softmax_scale=0.5)
        attn.eval()
        attn.rr_flashmask_attention_func = MagicMock()
        q, k, v = _bf16_qkv()
        with self.assertRaises(NotImplementedError) as ctx:
            attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=None,
                packed_seq_params=_packed_seq_params(),
                use_rr_flash_attention=True,
            )
        self.assertIn("RefinedRcomputeFlashMaskAttention", str(ctx.exception))
        attn.rr_flashmask_attention_func.assert_not_called()

    @patch(f"{_MOD}.flashmask_attention")
    def test_main_rr_custom_scale_raises_and_does_not_dispatch(self, mock_fm):
        attn = _build_attn(_make_config(), softmax_scale=0.5)
        attn.eval()
        attn.rr_flashmask_attention_func = MagicMock()
        q, k, v = _bf16_qkv()
        with self.assertRaises(NotImplementedError) as ctx:
            attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=paddle.zeros(
                    [_BATCH, 1, _SEQ, _SEQ], dtype="int32"
                ),
                use_rr_flash_attention=True,
            )
        self.assertIn("RefinedRcomputeFlashMaskAttention", str(ctx.exception))
        attn.rr_flashmask_attention_func.assert_not_called()

    @patch(f"{_MOD}.flashmask_attention")
    def test_packed_rr_default_scale_dispatches_without_scale(self, mock_fm):
        # Negative control: with the default scale the RR path is taken, the RR
        # kernel output is consumed, and no softmax_scale is forwarded.
        attn = _build_attn(_make_config())
        attn.eval()
        rr = MagicMock(return_value=_marker())
        attn.rr_flashmask_attention_func = rr
        q, k, v = _bf16_qkv()
        out = attn(
            q,
            k,
            v,
            None,
            attn_mask_startend_row_indices=None,
            packed_seq_params=_packed_seq_params(),
            use_rr_flash_attention=True,
        )
        rr.assert_called_once()
        self.assertNotIn("softmax_scale", rr.call_args.kwargs)
        np.testing.assert_array_equal(
            out.numpy(), _marker().reshape([_BATCH, _SEQ, _HIDDEN]).numpy()
        )


if __name__ == "__main__":
    unittest.main()
