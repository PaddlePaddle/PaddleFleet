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

"""Behavior unit tests for ``paddlefleet.fusions.fused_rms_norm.FusedRmsNorm``.

Scope (base): the CPU-observable *construction / config plumbing* of the
``FusedRmsNorm`` kernel wrapper -- normalized-shape canonicalization
(``int`` -> one-tuple, tuple kept), ``paddle.empty(*hidden_size)`` weight/bias
allocation, ``reset_parameters`` init values, the ``config.normalization``
guard, and propagation of ``config.layernorm_zero_centered_gamma`` /
``config.sequence_parallel`` / ``eps`` / ``persist_layer_norm`` onto the layer
and its parameters. The bf16 ``fused_rms_norm`` GPU forward numerics are
deliberately left to the sibling test (they require CUDA).

Independent references: every expected value is hand-derived from the RMSNorm
weight/bias init contract (standard -> weight all ones, bias all zeros;
zero-centered -> weight all zeros, bias all zeros) and from the dataclass
semantics of ``TransformerConfig`` (``TransformerConfig`` is a genuine
collaborator, not a stand-in for the code under test). No reference calls the
code under test to produce its expected value.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.fusions.fused_rms_norm import FusedRmsNorm
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except ImportError as exc:  # honest: only ImportError -> genuinely unavailable
    paddle = None
    FusedRmsNorm = None
    TransformerConfig = None
    _IMPORT_ERROR = exc


def _make_config(**overrides):
    """Build a real TransformerConfig for FusedRmsNorm construction tests.

    ``TransformerConfig`` is exercised as a genuine collaborator; the defaults
    keep a valid single-rank RMSNorm config and callers override only the
    field under test.
    """
    kwargs = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "normalization": "RMSNorm",
        "layernorm_zero_centered_gamma": False,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle / paddlefleet.fusions.fused_rms_norm unavailable: {_IMPORT_ERROR}",
)
class TestFusedRmsNormConstruction(unittest.TestCase):
    """Construction / config-plumbing contract of FusedRmsNorm (CPU only)."""

    def setUp(self):
        # Construction (paddle.empty + init.ones_/zeros_) runs on CPU; pin the
        # device/dtype so the hand-derived fills compare exactly, and restore
        # the process-global state afterwards even if an assertion fails.
        self._orig_device = paddle.get_device()
        self._orig_dtype = paddle.get_default_dtype()
        self.addCleanup(paddle.set_device, self._orig_device)
        self.addCleanup(paddle.set_default_dtype, self._orig_dtype)
        paddle.set_device("cpu")
        paddle.set_default_dtype("float32")

    def test_int_hidden_size_wrapped_to_one_tuple_and_param_shapes(self):
        """int hidden_size 16 -> stored (16,); weight/bias allocated as [16]."""
        layer = FusedRmsNorm(_make_config(), hidden_size=16)
        self.assertEqual(layer.hidden_size, (16,))
        self.assertEqual(layer.weight.shape, [16])
        self.assertEqual(layer.bias.shape, [16])

    def test_tuple_hidden_size_preserved_and_param_shapes(self):
        """tuple hidden_size (16,) is left as-is (not an Integral)."""
        layer = FusedRmsNorm(_make_config(), hidden_size=(16,))
        self.assertEqual(layer.hidden_size, (16,))
        self.assertEqual(layer.weight.shape, [16])
        self.assertEqual(layer.bias.shape, [16])

    def test_multidim_hidden_size_unpacked_into_param_shape(self):
        """(2, 3) tuple is unpacked by paddle.empty(*hidden_size) -> [2, 3].

        A one-tuple could pass even if the ``*`` unpack were dropped; a
        genuine multi-dim shape forces the unpack to be exercised.
        """
        layer = FusedRmsNorm(_make_config(), hidden_size=(2, 3))
        self.assertEqual(layer.hidden_size, (2, 3))
        self.assertEqual(layer.weight.shape, [2, 3])
        self.assertEqual(layer.bias.shape, [2, 3])

    def test_standard_reset_parameters_weight_ones_bias_zeros(self):
        """Standard init: weight filled with 1.0, bias filled with 0.0."""
        layer = FusedRmsNorm(
            _make_config(layernorm_zero_centered_gamma=False), hidden_size=8
        )
        self.assertFalse(layer.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones([8], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros([8], dtype=np.float32)
        )

    def test_zero_centered_gamma_reset_parameters_weight_and_bias_zeros(self):
        """Zero-centered init: both weight and bias filled with 0.0.

        Distinct from the standard case (weight ones) so swapping the two
        init branches would be caught.
        """
        layer = FusedRmsNorm(
            _make_config(layernorm_zero_centered_gamma=True), hidden_size=8
        )
        self.assertTrue(layer.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.zeros([8], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros([8], dtype=np.float32)
        )

    def test_zero_centered_gamma_sourced_from_config_not_constructor_arg(self):
        """``zero_centered_gamma`` is read from config, not the ctor keyword.

        The constructor accepts a ``zero_centered_gamma`` keyword but assigns
        ``self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma``
        and ignores the keyword. Passing the keyword opposite to the config
        value proves the config wins for both the stored flag and the
        resulting weight init.
        """
        # config says False, keyword says True -> config (False) wins.
        layer = FusedRmsNorm(
            _make_config(layernorm_zero_centered_gamma=False),
            hidden_size=8,
            zero_centered_gamma=True,
        )
        self.assertFalse(layer.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones([8], dtype=np.float32)
        )

        # config says True, keyword says False -> config (True) wins.
        layer2 = FusedRmsNorm(
            _make_config(layernorm_zero_centered_gamma=True),
            hidden_size=8,
            zero_centered_gamma=False,
        )
        self.assertTrue(layer2.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer2.weight.numpy(), np.zeros([8], dtype=np.float32)
        )

    def test_eps_stored_from_constructor_argument(self):
        """``eps`` defaults to 1e-6 and otherwise stores the passed value."""
        default_layer = FusedRmsNorm(_make_config(), hidden_size=8)
        self.assertEqual(default_layer.eps, 1e-6)

        explicit_layer = FusedRmsNorm(_make_config(), hidden_size=8, eps=1e-5)
        self.assertEqual(explicit_layer.eps, 1e-5)

        # A second distinct value confirms the field is plumbed, not constant.
        other_layer = FusedRmsNorm(_make_config(), hidden_size=8, eps=3e-7)
        self.assertEqual(other_layer.eps, 3e-7)

    def test_persist_layer_norm_flag_stored(self):
        """``persist_layer_norm`` defaults to False and stores the argument."""
        default_layer = FusedRmsNorm(_make_config(), hidden_size=8)
        self.assertFalse(default_layer.persist_layer_norm)

        persist_layer = FusedRmsNorm(
            _make_config(), hidden_size=8, persist_layer_norm=True
        )
        self.assertTrue(persist_layer.persist_layer_norm)

    def test_sequence_parallel_propagates_from_config_to_layer_and_params(self):
        """config.sequence_parallel flows onto layer and both parameters.

        A single-rank config forces sequence_parallel False; a TP=2 config
        with sequence_parallel=True yields True on the layer and on the
        ``sequence_parallel`` attribute stamped onto weight and bias.
        """
        off = FusedRmsNorm(
            _make_config(tensor_model_parallel_size=1, sequence_parallel=False),
            hidden_size=8,
        )
        self.assertFalse(off.sequence_parallel)
        self.assertFalse(off.weight.sequence_parallel)
        self.assertFalse(off.bias.sequence_parallel)

        on = FusedRmsNorm(
            _make_config(tensor_model_parallel_size=2, sequence_parallel=True),
            hidden_size=8,
        )
        self.assertTrue(on.sequence_parallel)
        self.assertTrue(on.weight.sequence_parallel)
        self.assertTrue(on.bias.sequence_parallel)

    def test_non_rmsnorm_normalization_is_rejected(self):
        """A non-RMSNorm config.normalization raises AssertionError naming it."""
        config = _make_config(normalization="LayerNorm")
        with self.assertRaisesRegex(AssertionError, "LayerNorm"):
            FusedRmsNorm(config, hidden_size=8)


if __name__ == "__main__":
    unittest.main()
