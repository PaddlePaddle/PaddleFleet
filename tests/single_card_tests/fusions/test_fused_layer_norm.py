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

"""CPU construction / dispatch tests for ``paddlefleet.fusions.fused_layer_norm``.

``FusedLayerNorm`` is a kernel wrapper. This file owns the ``__init__`` config
plumbing that is observable on CPU without an accelerator:

* which persistent-vs-fused kernel flag ``self.persist_layer_norm`` is selected
  for a given ``(config.persist_layer_norm, hidden_size, HAVE_PERSIST_LAYER_NORM,
  HAVE_FUSED_LAYER_NORM)`` combination;
* that ``config`` (not the constructor arguments of the same name) is the
  authoritative source for ``normalization`` and ``zero_centered_gamma``;
* the weight/bias parameter setup contract (shape, ``reset_parameters`` values,
  ``sequence_parallel`` propagation) and ``eps`` forwarding.

Scope split (sibling ``test_fused_layer_norm_2.py`` owns the forward path): the
GPU ``fused_layer_norm`` kernel numerics in ``forward`` are NOT exercised here;
they need an accelerator and are left to the sibling. Construction paths that
reach ``fused_layer_norm`` fallback require the kernel to be importable, so the
relevant tests skip honestly when it is absent rather than faking a pass.
"""

import unittest
from unittest.mock import patch

try:
    import numpy as np
    import paddle

    from paddlefleet.fusions import fused_layer_norm as mod
    from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm
    from paddlefleet.transformer.transformer_config import TransformerConfig

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

# Whether the fused kernel is importable in this build. When it is absent AND
# the persistent kernel is not selected, ``__init__`` raises ValueError before
# any parameter is created, so construction tests gate on this flag.
HAVE_FUSED = bool(HAS_PADDLE and mod.HAVE_FUSED_LAYER_NORM)


def _make_config(
    hidden_size,
    *,
    normalization="LayerNorm",
    zero_centered_gamma=False,
    persist=False,
    sequence_parallel=None,
):
    """Build a TransformerConfig wired the way FusedLayerNorm reads it.

    ``persist_layer_norm`` is NOT a TransformerConfig dataclass field, yet
    ``FusedLayerNorm.__init__`` reads ``self.config.persist_layer_norm``
    directly, so it must be attached explicitly here. ``sequence_parallel`` is
    a real field but ``ModelParallelConfig.__post_init__`` forces it to False
    whenever ``tensor_model_parallel_size <= 1``; to exercise propagation of a
    True value we set the attribute directly after construction.
    """
    cfg = TransformerConfig(
        hidden_size=hidden_size,
        normalization=normalization,
        layernorm_zero_centered_gamma=zero_centered_gamma,
    )
    cfg.persist_layer_norm = persist
    if sequence_parallel is not None:
        cfg.sequence_parallel = sequence_parallel
    return cfg


@unittest.skipUnless(
    HAVE_FUSED,
    "fused_layer_norm kernel unavailable; FusedLayerNorm.__init__ raises "
    "ValueError before building parameters when neither persist nor fused "
    "kernel is present",
)
class TestFusedLayerNormConstruction(unittest.TestCase):
    """__init__ paths that build weight/bias via the fused fallback."""

    def setUp(self):
        # These are CPU-observable construction assertions; pin to CPU and
        # restore the prior device so we neither depend on nor leak a device.
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def test_int_hidden_size_normalized_to_tuple(self):
        # An integral hidden_size is wrapped into a 1-tuple, and weight/bias are
        # created with exactly that shape.
        layer = FusedLayerNorm(_make_config(16), hidden_size=16)
        self.assertEqual(layer.hidden_size, (16,))
        self.assertEqual(layer.weight.shape, [16])
        self.assertEqual(layer.bias.shape, [16])

    def test_weight_bias_init_standard(self):
        # zero_centered_gamma False -> reset_parameters sets weight to all ones
        # and bias to all zeros (hand-derived, exact).
        layer = FusedLayerNorm(
            _make_config(16, zero_centered_gamma=False), hidden_size=16
        )
        self.assertFalse(layer.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones([16], "float32")
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros([16], "float32")
        )

    def test_weight_bias_init_zero_centered(self):
        # zero_centered_gamma True -> weight AND bias are all zeros.
        layer = FusedLayerNorm(
            _make_config(16, zero_centered_gamma=True), hidden_size=16
        )
        self.assertTrue(layer.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.zeros([16], "float32")
        )
        np.testing.assert_array_equal(
            layer.bias.numpy(), np.zeros([16], "float32")
        )

    def test_zero_centered_gamma_param_is_ignored_config_authoritative(self):
        # The ``zero_centered_gamma`` constructor argument is shadowed: __init__
        # reads config.layernorm_zero_centered_gamma. Passing the opposite value
        # as the argument must NOT change behaviour.
        layer = FusedLayerNorm(
            _make_config(16, zero_centered_gamma=False),
            hidden_size=16,
            zero_centered_gamma=True,  # argument says True, config says False
        )
        self.assertFalse(layer.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer.weight.numpy(), np.ones([16], "float32")
        )

        layer2 = FusedLayerNorm(
            _make_config(16, zero_centered_gamma=True),
            hidden_size=16,
            zero_centered_gamma=False,  # argument says False, config says True
        )
        self.assertTrue(layer2.zero_centered_gamma)
        np.testing.assert_array_equal(
            layer2.weight.numpy(), np.zeros([16], "float32")
        )

    def test_normalization_param_does_not_override_config(self):
        # The guard checks config.normalization, not the ``normalization``
        # argument. A "RMSNorm" argument with a "LayerNorm" config constructs
        # successfully and keeps the config value.
        layer = FusedLayerNorm(
            _make_config(16, normalization="LayerNorm"),
            hidden_size=16,
            normalization="RMSNorm",
        )
        self.assertEqual(layer.config.normalization, "LayerNorm")

    def test_eps_forwarded_from_argument(self):
        # eps is one of the few values taken from the argument (not config).
        layer = FusedLayerNorm(_make_config(16), hidden_size=16, eps=1e-3)
        self.assertEqual(layer.eps, 1e-3)

    def test_eps_defaults_to_1e_minus_5(self):
        layer = FusedLayerNorm(_make_config(16), hidden_size=16)
        self.assertEqual(layer.eps, 1e-5)

    def test_persist_disabled_when_capability_flag_false(self):
        # Real build state: HAVE_PERSIST_LAYER_NORM is hard-coded False in the
        # module, so even a supported size (1024) with config.persist_layer_norm
        # True still falls back to the non-persistent path.
        self.assertFalse(mod.HAVE_PERSIST_LAYER_NORM)
        layer = FusedLayerNorm(
            _make_config(1024, persist=True), hidden_size=1024
        )
        self.assertFalse(layer.persist_layer_norm)

    def test_persist_disabled_for_unsupported_size(self):
        # With the capability flag forced on and config requesting persist, an
        # unsupported hidden size (48 is not a persist size) still disables it.
        with patch.object(mod, "HAVE_PERSIST_LAYER_NORM", True):
            layer = FusedLayerNorm(
                _make_config(48, persist=True), hidden_size=48
            )
        self.assertFalse(layer.persist_layer_norm)

    def test_persist_disabled_when_config_requests_false(self):
        # Capability flag on and supported size (1024), but config.persist_layer_norm
        # False -> disabled. Isolates the config read from the size/flag branches.
        with patch.object(mod, "HAVE_PERSIST_LAYER_NORM", True):
            layer = FusedLayerNorm(
                _make_config(1024, persist=False), hidden_size=1024
            )
        self.assertFalse(layer.persist_layer_norm)

    def test_sequence_parallel_true_propagates_to_params(self):
        # self.sequence_parallel is copied from config and stamped onto both the
        # weight and bias parameters.
        layer = FusedLayerNorm(
            _make_config(16, sequence_parallel=True), hidden_size=16
        )
        self.assertTrue(layer.sequence_parallel)
        self.assertTrue(layer.weight.sequence_parallel)
        self.assertTrue(layer.bias.sequence_parallel)

    def test_sequence_parallel_false_propagates_to_params(self):
        layer = FusedLayerNorm(
            _make_config(16, sequence_parallel=False), hidden_size=16
        )
        self.assertFalse(layer.sequence_parallel)
        self.assertFalse(layer.weight.sequence_parallel)
        self.assertFalse(layer.bias.sequence_parallel)


@unittest.skipUnless(
    HAS_PADDLE, "paddle is not installed; CPU dispatch test needs paddle"
)
class TestFusedLayerNormDispatch(unittest.TestCase):
    """Dispatch branches that do not depend on the fused kernel being present."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def test_wrong_normalization_from_config_raises(self):
        # The guard reads config.normalization; a non-LayerNorm config raises
        # regardless of the ``normalization`` argument value.
        cfg = _make_config(16, normalization="RMSNorm")
        with self.assertRaises(AssertionError) as ctx:
            FusedLayerNorm(cfg, hidden_size=16, normalization="LayerNorm")
        self.assertIn("(RMSNorm)", str(ctx.exception))

    def test_persist_enabled_when_all_conditions_met(self):
        # Capability flag on + supported size (1024) + config requests persist
        # -> persist kernel selected. Because persist is True the fused-kernel
        # ValueError guard is bypassed, so this holds without the fused kernel.
        with patch.object(mod, "HAVE_PERSIST_LAYER_NORM", True):
            layer = FusedLayerNorm(
                _make_config(1024, persist=True), hidden_size=1024
            )
        self.assertTrue(layer.persist_layer_norm)

    def test_no_kernel_available_raises_value_error(self):
        # persist disabled (default) AND fused kernel absent -> ValueError with
        # the documented message, raised before any parameter is built.
        with patch.object(mod, "HAVE_FUSED_LAYER_NORM", False):  # noqa: SIM117
            with self.assertRaises(ValueError) as ctx:
                FusedLayerNorm(_make_config(64, persist=False), hidden_size=64)
        self.assertIn("Apex must be installed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
