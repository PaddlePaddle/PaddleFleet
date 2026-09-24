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

"""Behavior unit tests for paddlefleet.fusions.fused_rms_norm (part _2).

FusedRmsNorm is a thin wrapper around the GPU-only kernel
``paddle.incubate.nn.functional.fused_rms_norm``. This file pins the
CPU-observable control flow of the wrapper (a sibling file covers plain
construction/config); the GPU kernel numerics are intentionally NOT asserted
here because they cannot be reproduced offline.

Slice covered here:

* ``reset_parameters``: the exact weight/bias values it writes for both the
  zero-centered and non-zero-centered config, verified as a re-runnable method
  (not just as post-construction state).
* ``forward`` slow path: the rank/shape validation that raises ``ValueError``
  *before* the kernel is ever reached -- pure CPU, no kernel needed.
* ``forward`` fast-path dispatch: the ``weight + 1`` adjustment (weight only,
  never bias), the ``begin_norm_axis`` computation, ``eps`` forwarding and the
  tuple-output unwrap. The GPU kernel is a genuine *non-tested* collaborator; it
  is replaced by a marker so the wrapper's real control flow around it is
  observed. The kernel's own numerics are declared unverified.
* persist eligibility: ``persist_layer_norm`` is stored but (in this
  implementation) not consulted by ``forward``; ``HAVE_PERSIST_LAYER_NORM`` is a
  ``False`` module constant. Documented as observed behavior, not asserted as a
  defect.
"""

import os
import sys
import unittest
from unittest.mock import patch

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

try:
    import paddle

    from paddlefleet.fusions import fused_rms_norm as frn_mod
    from paddlefleet.fusions.fused_rms_norm import (
        HAVE_PERSIST_LAYER_NORM,
        FusedRmsNorm,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = None
except ImportError as exc:  # honest: only a genuinely missing dep skips.
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = f"paddle / paddlefleet.fusions.fused_rms_norm not importable: {_IMPORT_ERROR}"


class _Config:
    """Lightweight stand-in for TransformerConfig.

    FusedRmsNorm reads exactly three attributes off ``config``:
    ``layernorm_zero_centered_gamma``, ``normalization`` and
    ``sequence_parallel``. Constructing a real TransformerConfig runs
    ModelParallelConfig.__post_init__ (parallel-state wiring) which is out of
    scope for this offline control-flow test. ``config`` is a collaborator, not
    the code under test, so an object supplying just those reads is honest and
    sufficient.
    """

    def __init__(
        self,
        zero_centered=False,
        normalization="RMSNorm",
        sequence_parallel=False,
    ):
        self.layernorm_zero_centered_gamma = zero_centered
        self.normalization = normalization
        self.sequence_parallel = sequence_parallel


def _f32(tensor):
    """bf16/other tensor -> float32 numpy (our fixtures are bf16-exact ints)."""
    return tensor.cast("float32").numpy()


def _make_layer(zero_centered=False, hidden_size=4, eps=1e-6, persist=False):
    return FusedRmsNorm(
        _Config(zero_centered=zero_centered),
        hidden_size=hidden_size,
        eps=eps,
        persist_layer_norm=persist,
    )


def _set_weight_bias(layer, weight_vals, bias_vals):
    """Inject distinguishable, bf16-exact small-integer weight/bias."""
    layer.weight.set_value(paddle.to_tensor(weight_vals, dtype="float32"))
    layer.bias.set_value(paddle.to_tensor(bias_vals, dtype="float32"))


# Distinguishable fixtures. All values are < 256 and integral, hence exactly
# representable in bfloat16, so the wrapper's cast("bfloat16") is lossless and
# the arguments handed to the (mocked) kernel can be compared exactly.
_WEIGHT = [1.0, 2.0, 3.0, 4.0]
_BIAS = [10.0, 20.0, 30.0, 40.0]
# input [2, 3, 4] with unique per-element values 1..24.
_INPUT = np.arange(1.0, 25.0, dtype="float32").reshape([2, 3, 4]).tolist()


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestResetParameters(unittest.TestCase):
    """reset_parameters writes exact, hand-derived weight/bias values."""

    def test_non_zero_centered_reset_values(self):
        # config zero_centered=False => init.ones_(weight), init.zeros_(bias).
        layer = _make_layer(zero_centered=False, hidden_size=4)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones(4, "float32")
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros(4, "float32")
        )

    def test_zero_centered_reset_values(self):
        # config zero_centered=True => init.zeros_(weight), init.zeros_(bias).
        layer = _make_layer(zero_centered=True, hidden_size=4)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.zeros(4, "float32")
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros(4, "float32")
        )

    def test_reset_parameters_reinitializes_after_mutation(self):
        # Overwrite with garbage, then re-run the method: it must restore the
        # documented values, proving reset_parameters itself does the writing.
        layer = _make_layer(zero_centered=False, hidden_size=4)
        _set_weight_bias(layer, [7.0, 8.0, 9.0, 11.0], [1.0, 2.0, 3.0, 4.0])
        layer.reset_parameters()
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones(4, "float32")
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros(4, "float32")
        )

    def test_constructor_zero_centered_param_is_ignored_config_wins(self):
        # Observed behavior: the constructor's zero_centered_gamma argument is
        # dead; self.zero_centered_gamma is taken from config only. Passing True
        # here while config says False must still produce the non-centered reset.
        layer = FusedRmsNorm(
            _Config(zero_centered=False),
            hidden_size=4,
            zero_centered_gamma=True,
        )
        self.assertFalse(layer.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones(4, "float32")
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestForwardSlowPath(unittest.TestCase):
    """Shape/rank validation raises ValueError before the kernel is reached.

    These cases need no kernel: the exception is raised prior to the
    fused_rms_norm call, so they run entirely on CPU.
    """

    def test_shape_mismatch_raises_valueerror_with_message(self):
        layer = _make_layer(zero_centered=False, hidden_size=4)
        # input last dim 8 != normalized shape [4] -> raise. begin_norm_axis=2.
        x = paddle.zeros([2, 3, 8], dtype="float32")
        with self.assertRaises(ValueError) as cm:
            layer(x)
        # Hand-derived from the production string concatenation:
        #   str([4]) == "[4]"; "[4]"[1:] == "4]"; str([2,3,8]) == "[2, 3, 8]".
        expected = (
            "Given normalized_shape is [4], expected input with shape "
            "[*, 4], but got input shape [2, 3, 8]"
        )
        self.assertEqual(str(cm.exception), expected)

    def test_input_rank_less_than_normalized_rank_raises(self):
        # Multi-dim hidden_size stays a tuple in __init__; a 1-D input has fewer
        # dims than the 2-D normalized shape -> the input_ndim < normalized_ndim
        # branch fires (short-circuits before the same_shape check).
        layer = FusedRmsNorm(_Config(zero_centered=False), hidden_size=(4, 5))
        x = paddle.zeros([4], dtype="float32")
        with self.assertRaises(ValueError) as cm:
            layer(x)
        expected = (
            "Given normalized_shape is [4, 5], expected input with shape "
            "[*, 4, 5], but got input shape [4]"
        )
        self.assertEqual(str(cm.exception), expected)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestForwardFastPathDispatch(unittest.TestCase):
    """Fast path: the wrapper's real control flow around the (mocked) kernel.

    The GPU kernel ``fused_rms_norm`` is a genuine collaborator that is NOT the
    code under test; it is replaced by a distinguishable marker so the wrapper's
    argument construction is observable. The kernel's own numerics are declared
    unverified (GPU-only).
    """

    def test_non_zero_centered_forwards_weight_bias_axis_eps(self):
        layer = _make_layer(zero_centered=False, hidden_size=4, eps=0.125)
        _set_weight_bias(layer, _WEIGHT, _BIAS)
        x = paddle.to_tensor(_INPUT, dtype="float32")
        marker = paddle.to_tensor(_INPUT, dtype="float32") + 1000.0

        with patch.object(frn_mod, "fused_rms_norm", return_value=marker) as k:
            out = layer(x)

        k.assert_called_once()
        args, kwargs = k.call_args
        # weight not centered => passed through unchanged (== _WEIGHT).
        np.testing.assert_array_equal(_f32(args[1]), np.array(_WEIGHT))
        # bias is never adjusted.
        np.testing.assert_array_equal(_f32(args[2]), np.array(_BIAS))
        # input forwarded (cast to bf16, lossless here).
        np.testing.assert_array_equal(_f32(args[0]), np.array(_INPUT))
        # eps forwarded positionally; begin_norm_axis = 3 - 1 = 2.
        self.assertEqual(args[3], 0.125)
        self.assertEqual(kwargs["begin_norm_axis"], 2)
        # non-tuple kernel output returned verbatim.
        self.assertIs(out, marker)

    def test_zero_centered_adds_one_to_weight_only(self):
        layer = _make_layer(zero_centered=True, hidden_size=4, eps=1e-6)
        # Inject the SAME weight as the non-centered case; the only difference in
        # the argument must be the +1 on weight (bias untouched).
        _set_weight_bias(layer, _WEIGHT, _BIAS)
        x = paddle.to_tensor(_INPUT, dtype="float32")
        marker = paddle.to_tensor(_INPUT, dtype="float32")

        with patch.object(frn_mod, "fused_rms_norm", return_value=marker) as k:
            layer(x)

        args, _ = k.call_args
        np.testing.assert_array_equal(
            _f32(args[1]), np.array(_WEIGHT) + 1.0
        )  # [2, 3, 4, 5]
        np.testing.assert_array_equal(
            _f32(args[2]), np.array(_BIAS)
        )  # unchanged

    def test_begin_norm_axis_2d_input(self):
        layer = _make_layer(zero_centered=False, hidden_size=4)
        _set_weight_bias(layer, _WEIGHT, _BIAS)
        x = paddle.zeros([5, 4], dtype="float32")
        marker = paddle.zeros([5, 4], dtype="float32")

        with patch.object(frn_mod, "fused_rms_norm", return_value=marker) as k:
            layer(x)

        _, kwargs = k.call_args
        self.assertEqual(kwargs["begin_norm_axis"], 1)  # 2 - 1

    def test_tuple_output_returns_first_element(self):
        layer = _make_layer(zero_centered=False, hidden_size=4)
        _set_weight_bias(layer, _WEIGHT, _BIAS)
        x = paddle.to_tensor(_INPUT, dtype="float32")
        first = paddle.to_tensor(_INPUT, dtype="float32")
        second = first + 1.0
        third = first + 2.0

        with patch.object(
            frn_mod, "fused_rms_norm", return_value=(first, second, third)
        ):
            out = layer(x)

        self.assertIs(out, first)  # exactly output[0], not output[1] or a merge

    def test_non_tuple_output_returned_as_is(self):
        layer = _make_layer(zero_centered=False, hidden_size=4)
        _set_weight_bias(layer, _WEIGHT, _BIAS)
        x = paddle.to_tensor(_INPUT, dtype="float32")
        marker = paddle.to_tensor(_INPUT, dtype="float32") - 5.0

        with patch.object(frn_mod, "fused_rms_norm", return_value=marker):
            out = layer(x)

        self.assertIs(out, marker)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestPersistEligibility(unittest.TestCase):
    """persist_layer_norm is stored but currently inert in this wrapper."""

    def test_persist_flag_stored_verbatim(self):
        self.assertTrue(_make_layer(persist=True).persist_layer_norm)
        self.assertFalse(_make_layer(persist=False).persist_layer_norm)

    def test_have_persist_layer_norm_constant_is_false(self):
        # The persistent-kernel path is not wired up in this implementation.
        self.assertFalse(HAVE_PERSIST_LAYER_NORM)

    def test_persist_flag_does_not_change_dispatch(self):
        # Documented observed behavior: forward never reads persist_layer_norm,
        # so True vs False produce identical kernel arguments.
        x = paddle.to_tensor(_INPUT, dtype="float32")

        def run(persist):
            layer = _make_layer(
                zero_centered=False, hidden_size=4, persist=persist
            )
            _set_weight_bias(layer, _WEIGHT, _BIAS)
            marker = paddle.to_tensor(_INPUT, dtype="float32")
            with patch.object(
                frn_mod, "fused_rms_norm", return_value=marker
            ) as k:
                layer(x)
            args, kwargs = k.call_args
            return (
                _f32(args[1]),
                _f32(args[2]),
                args[3],
                kwargs["begin_norm_axis"],
            )

        w_t, b_t, eps_t, axis_t = run(True)
        w_f, b_f, eps_f, axis_f = run(False)
        np.testing.assert_array_equal(w_t, w_f)
        np.testing.assert_array_equal(b_t, b_f)
        self.assertEqual(eps_t, eps_f)
        self.assertEqual(axis_t, axis_f)


if __name__ == "__main__":
    unittest.main()
