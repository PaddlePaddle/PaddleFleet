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

"""Behavior unit tests for paddlefleet.fusions.fused_layer_norm (part _2).

Scope of this file (the sibling test_fused_layer_norm.py concurrently covers
construction / config plumbing). Here we pin, on CPU only, the control flow
that ``FusedLayerNorm`` decides without touching the CUDA kernel:

* the persist-layer-norm eligibility resolution (``config.persist_layer_norm``
  gated by hidden-size membership and ``HAVE_PERSIST_LAYER_NORM``), including
  the fact that the ``persist_layer_norm`` *constructor argument* is shadowed
  and never consulted;
* the "Apex must be installed" guard that fires when neither a persistent nor a
  fused kernel is available;
* ``reset_parameters`` numerics (init.ones_/zeros_ on weight/bias), which run on
  CPU and are hand-derived below;
* ``forward`` dispatch: shape validation (exact error message), the
  zero-centered-gamma ``weight + 1`` adjustment, the begin_norm_axis
  computation, hidden_size list-normalisation, and the exact arguments handed to
  the fused kernel plus the ``output[0]`` unpacking.

The fused CUDA kernel is a genuine external collaborator, not the code under
test; where forward reaches it we replace it with an input-distinguishable
marker and assert the exact arguments it receives and that its result is
returned. GPU layer-norm numerics are therefore intentionally NOT asserted here
-- they require a device and cannot be reproduced as CPU arithmetic offline.
"""

import os
import re
import sys
import unittest

import numpy as np

# Reach src/ so that ``paddlefleet`` is importable when running the file directly.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

from unittest.mock import patch

try:
    import paddle

    from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm

    HAS_PADDLE = True
    _IMPORT_ERROR = None
except ImportError as exc:  # honest: only a genuinely missing dep skips.
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = f"paddle / paddlefleet not importable: {_IMPORT_ERROR}"

_MODULE = "paddlefleet.fusions.fused_layer_norm"

# A hidden size that IS in FusedLayerNorm.persist_ln_hidden_sizes, and one that
# is NOT, so membership in that whitelist is genuinely exercised.
_HS_IN_WHITELIST = 1024
_HS_NOT_IN_WHITELIST = 1000  # deliberately absent from the persist list


class _Config:
    """Lightweight stand-in for TransformerConfig.

    FusedLayerNorm reads exactly these four attributes; using a real object with
    only those fields (rather than a permissive mock) means an unexpected
    attribute access would fail loudly instead of silently passing.
    """

    def __init__(
        self,
        normalization="LayerNorm",
        layernorm_zero_centered_gamma=False,
        persist_layer_norm=False,
        sequence_parallel=False,
    ):
        self.normalization = normalization
        self.layernorm_zero_centered_gamma = layernorm_zero_centered_gamma
        self.persist_layer_norm = persist_layer_norm
        self.sequence_parallel = sequence_parallel


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestPersistLayerNormEligibility(unittest.TestCase):
    """self.persist_layer_norm resolves from config, whitelist and HAVE flag.

    Production logic (fused_layer_norm.py):
        persist_layer_norm = self.config.persist_layer_norm
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:
            persist_layer_norm = False
    """

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", True)
    def test_stays_true_when_all_conditions_hold(self):
        # config True, hidden in whitelist, HAVE_PERSIST True -> no reset.
        cfg = _Config(persist_layer_norm=True)
        layer = FusedLayerNorm(cfg, hidden_size=_HS_IN_WHITELIST)
        self.assertTrue(layer.persist_layer_norm)

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", True)
    def test_reset_to_false_when_hidden_not_in_whitelist(self):
        # Only the whitelist membership differs from the previous case; a hidden
        # size outside the persist list must force persist off.
        cfg = _Config(persist_layer_norm=True)
        layer = FusedLayerNorm(cfg, hidden_size=_HS_NOT_IN_WHITELIST)
        self.assertFalse(layer.persist_layer_norm)

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_reset_to_false_when_persist_kernel_absent(self):
        # config True and hidden in whitelist, but HAVE_PERSIST False -> off.
        # This isolates the HAVE_PERSIST_LAYER_NORM guard specifically.
        cfg = _Config(persist_layer_norm=True)
        layer = FusedLayerNorm(cfg, hidden_size=_HS_IN_WHITELIST)
        self.assertFalse(layer.persist_layer_norm)

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", True)
    def test_config_false_keeps_persist_off_even_when_eligible(self):
        # Whitelist + HAVE_PERSIST both favourable, but config says False, so the
        # config value is genuinely consumed (not ignored).
        cfg = _Config(persist_layer_norm=False)
        layer = FusedLayerNorm(cfg, hidden_size=_HS_IN_WHITELIST)
        self.assertFalse(layer.persist_layer_norm)

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", True)
    def test_constructor_persist_argument_is_shadowed(self):
        # Observed contract (documented, not a modification target): the
        # ``persist_layer_norm`` constructor argument is overwritten on the first
        # line of the resolution by ``self.config.persist_layer_norm``. So a True
        # constructor argument cannot turn persist on when config says False.
        cfg = _Config(persist_layer_norm=False)
        layer = FusedLayerNorm(
            cfg, hidden_size=_HS_IN_WHITELIST, persist_layer_norm=True
        )
        self.assertFalse(layer.persist_layer_norm)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestApexRequiredGuard(unittest.TestCase):
    """No persistent and no fused kernel -> explicit ValueError."""

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", False)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_raises_when_neither_kernel_available(self):
        cfg = _Config(persist_layer_norm=True)
        with self.assertRaises(ValueError) as ctx:
            FusedLayerNorm(cfg, hidden_size=_HS_IN_WHITELIST)
        self.assertEqual(
            str(ctx.exception),
            "Apex must be installed to use FusedLayerNorm.",
        )

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_does_not_raise_when_fused_kernel_available(self):
        # Same persist resolution (ends up False) but the fused kernel exists, so
        # construction must succeed -- proves the guard needs BOTH to be absent.
        cfg = _Config(persist_layer_norm=True)
        layer = FusedLayerNorm(cfg, hidden_size=_HS_IN_WHITELIST)
        self.assertFalse(layer.persist_layer_norm)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestResetParameters(unittest.TestCase):
    """reset_parameters writes init.ones_/zeros_ per zero_centered_gamma."""

    _HS = 16  # not in the persist whitelist; small, exact CPU init values.

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_default_gamma_resets_weight_ones_bias_zeros(self):
        cfg = _Config(layernorm_zero_centered_gamma=False)
        layer = FusedLayerNorm(cfg, hidden_size=self._HS)
        # Overwrite with distinguishable junk, then reset, to prove the reset
        # genuinely writes the values (not leftover empty-init coincidence).
        layer.weight.set_value(paddle.arange(self._HS, dtype="float32") + 5.0)
        layer.bias.set_value(paddle.arange(self._HS, dtype="float32") + 7.0)
        layer.reset_parameters()
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones(self._HS, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros(self._HS, dtype=np.float32)
        )

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_zero_centered_gamma_resets_weight_and_bias_zeros(self):
        cfg = _Config(layernorm_zero_centered_gamma=True)
        layer = FusedLayerNorm(cfg, hidden_size=self._HS)
        layer.weight.set_value(paddle.arange(self._HS, dtype="float32") + 5.0)
        layer.bias.set_value(paddle.arange(self._HS, dtype="float32") + 7.0)
        layer.reset_parameters()
        # The distinguishing effect of zero_centered_gamma: weight is zeros, not
        # ones. If the if/else were swapped this branch would give ones.
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.zeros(self._HS, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros(self._HS, dtype=np.float32)
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestForwardShapeValidation(unittest.TestCase):
    """forward raises a fully-specified ValueError on a shape mismatch."""

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_mismatch_message_is_exact(self):
        cfg = _Config()
        layer = FusedLayerNorm(cfg, hidden_size=512)
        x = paddle.zeros([2, 4, 1024], dtype="float32")
        # Hand-derived from forward(): normalized_shape=[512], begin_norm_axis=2,
        # input_shape[2:]=[1024] != [512] -> raise. str([512])="[512]",
        # "[512]"[1:]="512]"; input shape str is "[2, 4, 1024]".
        expected = (
            "Given normalized_shape is [512], expected input with shape "
            "[*, 512], but got input shape [2, 4, 1024]"
        )
        with self.assertRaisesRegex(ValueError, re.escape(expected)):
            layer(x)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestForwardKernelDispatch(unittest.TestCase):
    """forward marshals exact arguments to the fused kernel and returns [0].

    The fused kernel is an external collaborator (not the code under test); we
    replace it with an input-distinguishable marker to verify dispatch and
    argument marshalling, and that ``output[0]`` is the return value. The
    kernel's layer-norm numerics are NOT asserted here (they need a GPU).
    """

    def _make_layer(self, hidden_size, zero_centered_gamma=False):
        cfg = _Config(layernorm_zero_centered_gamma=zero_centered_gamma)
        return FusedLayerNorm(cfg, hidden_size=hidden_size)

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_default_gamma_passes_weight_bias_eps_axis_and_returns_first(self):
        layer = self._make_layer(4, zero_centered_gamma=False)
        layer.weight.set_value(
            paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        )
        layer.bias.set_value(
            paddle.to_tensor([10.0, 20.0, 30.0, 40.0], dtype="float32")
        )
        x = paddle.arange(2 * 3 * 4, dtype="float32").reshape([2, 3, 4])
        marker = paddle.full([2, 3, 4], 7.0, dtype="float32")
        captured = {}

        def fake_kernel(inp, weight, bias, eps, begin_norm_axis=None):
            captured.update(
                inp=inp,
                weight=weight.numpy().copy(),
                bias=bias.numpy().copy(),
                eps=eps,
                begin_norm_axis=begin_norm_axis,
            )
            return (marker,)

        with patch(
            _MODULE + ".fused_layer_norm", create=True, side_effect=fake_kernel
        ):
            out = layer(x)

        self.assertIs(captured["inp"], x)  # input handed through unchanged
        # zero_centered_gamma is False -> weight passed as-is (no +1).
        np.testing.assert_array_equal(
            captured["weight"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            captured["bias"],
            np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        )
        self.assertEqual(captured["eps"], 1e-5)
        self.assertEqual(captured["begin_norm_axis"], 2)  # input_ndim(3) - 1
        self.assertIs(out, marker)  # forward returns output[0]

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_zero_centered_gamma_adds_one_to_kernel_weight(self):
        layer = self._make_layer(4, zero_centered_gamma=True)
        layer.weight.set_value(
            paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        )
        x = paddle.arange(2 * 3 * 4, dtype="float32").reshape([2, 3, 4])
        captured = {}

        def fake_kernel(inp, weight, bias, eps, begin_norm_axis=None):
            captured["weight"] = weight.numpy().copy()
            return (paddle.zeros([1], dtype="float32"),)

        with patch(
            _MODULE + ".fused_layer_norm", create=True, side_effect=fake_kernel
        ):
            layer(x)

        # zero_centered_gamma -> weight = self.weight + 1 handed to the kernel.
        np.testing.assert_array_equal(
            captured["weight"], np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        )

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_begin_norm_axis_tracks_input_rank(self):
        layer = self._make_layer(4)
        captured = {}

        def fake_kernel(inp, weight, bias, eps, begin_norm_axis=None):
            captured["axis"] = begin_norm_axis
            return (paddle.zeros([1], dtype="float32"),)

        with patch(
            _MODULE + ".fused_layer_norm", create=True, side_effect=fake_kernel
        ):
            layer(paddle.zeros([2, 3, 4], dtype="float32"))
            self.assertEqual(captured["axis"], 2)  # rank 3 - 1
            layer(paddle.zeros([5, 4], dtype="float32"))
            self.assertEqual(captured["axis"], 1)  # rank 2 - 1

    @patch(_MODULE + ".HAVE_FUSED_LAYER_NORM", True)
    @patch(_MODULE + ".HAVE_PERSIST_LAYER_NORM", False)
    def test_forward_normalises_hidden_size_tuple_to_list(self):
        layer = self._make_layer(4)
        # __init__ stores an integral hidden_size as a tuple.
        self.assertEqual(layer.hidden_size, (4,))
        with patch(
            _MODULE + ".fused_layer_norm",
            create=True,
            return_value=(paddle.zeros([1], dtype="float32"),),
        ):
            layer(paddle.zeros([2, 3, 4], dtype="float32"))
        # forward converts the tuple to a list in place.
        self.assertIsInstance(layer.hidden_size, list)
        self.assertEqual(layer.hidden_size, [4])


if __name__ == "__main__":
    unittest.main()
