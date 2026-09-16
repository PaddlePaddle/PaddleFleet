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

import os
import sys
import unittest

# Bootstrap: make the repo `src/` importable when the test is run standalone
# (CI normally puts it on PYTHONPATH; this keeps the file runnable directly).
_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle
    from paddle import nn
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
    from paddlefleet.transformer.transformer_block import (
        TransformerBlock,
        TransformerBlockSublayersSpec,
        _get_block_sublayers_spec,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.transformer.transformer_layer import TransformerLayer
    from paddlefleet.utils import WrappedTensor

    PADDLE_AVAILABLE = True
    _IMPORT_ERROR = ""

    class _RecordingLayer(nn.Layer):
        """Minimal real layer used to observe TransformerBlock orchestration.

        It records the ``layer_number`` / ``pg_collection`` the block feeds it
        (so we can check the block wires them correctly) and applies a
        distinguishable, order-sensitive affine transform to ``hidden_states``.
        Because ``layer_number`` shifts the value differently at each position,
        a mis-threaded loop or a swapped layer order changes the output.
        """

        def __init__(self, config, layer_number, pg_collection, scale):
            super().__init__()
            self.config = config
            self.layer_number = layer_number
            self.pg_collection = pg_collection
            self.scale = float(scale)

        def forward(self, dict_args):
            hidden = dict_args["hidden_states"]
            dict_args["hidden_states"] = hidden * self.scale + float(
                self.layer_number
            )
            return dict_args

    class _RecordingNorm(nn.Layer):
        """Real final-norm stand-in that records the build kwargs it received
        and applies a large, distinguishable shift so the test can tell whether
        the block actually ran the final norm."""

        def __init__(self, config, hidden_size, eps):
            super().__init__()
            self.config = config
            self.hidden_size = hidden_size
            self.eps = eps

        def forward(self, x):
            return x + 100.0

    class _DummyTransformerLayer(TransformerLayer):
        # Subclass only; never instantiated. Drives the
        # ``issubclass(spec.layer, TransformerLayer)`` branch of the dispatcher.
        pass

except (
    ImportError,
    ModuleNotFoundError,
) as exc:  # CPU-only env may lack paddle
    PADDLE_AVAILABLE = False
    _IMPORT_ERROR = repr(exc)


_SKIP_REASON = (
    "paddle / paddlefleet not importable in this CPU-only environment: "
    + _IMPORT_ERROR
)


def _make_config(**overrides):
    kwargs = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGetBlockSublayersSpec(unittest.TestCase):
    """Dispatch logic of ``_get_block_sublayers_spec`` (real function, no mocks)."""

    def test_existing_sublayers_spec_returned_unchanged(self):
        cfg = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(layer=_RecordingLayer)],
            layer_norm=None,
        )
        out = _get_block_sublayers_spec(cfg, spec)
        self.assertIs(out, spec)

    def test_transformer_layer_spec_expands_to_num_hidden_layers(self):
        cfg = _make_config(num_hidden_layers=3)
        layer_spec = LayerSpec(layer=_DummyTransformerLayer)
        out = _get_block_sublayers_spec(cfg, layer_spec)
        self.assertIsInstance(out, TransformerBlockSublayersSpec)
        # One entry per hidden layer, each the *same* spec object, in order.
        self.assertEqual(len(out.layer_specs), 3)
        for entry in out.layer_specs:
            self.assertIs(entry, layer_spec)
        # The expansion pins the final norm to the module's LayerNormImpl.
        self.assertIs(out.layer_norm, WrappedPaddleNorm)

    def test_unrelated_layer_spec_raises(self):
        cfg = _make_config()
        # nn.Linear is neither a TransformerBlock nor a TransformerLayer.
        spec = LayerSpec(layer=nn.Linear, extra_kwargs={})
        with self.assertRaisesRegex(Exception, "specialize"):
            _get_block_sublayers_spec(cfg, spec)

    def test_non_spec_input_raises(self):
        cfg = _make_config()
        with self.assertRaisesRegex(Exception, "specialize"):
            _get_block_sublayers_spec(cfg, object())


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestTransformerBlockConstructionGuards(unittest.TestCase):
    """The two construction-time assertions must reject unsupported configs."""

    def test_vp_stage_not_none_raises(self):
        cfg = _make_config()
        spec = TransformerBlockSublayersSpec()
        with self.assertRaisesRegex(AssertionError, "pipeline parallel"):
            TransformerBlock(config=cfg, spec=spec, vp_stage=0)

    def test_cpu_offloading_true_raises(self):
        cfg = _make_config(cpu_offloading=True)
        spec = TransformerBlockSublayersSpec()
        # Pass an explicit pg_collection so the guard is reached without
        # requiring a real distributed process-group collection.
        with self.assertRaises(AssertionError):
            TransformerBlock(config=cfg, spec=spec, pg_collection=object())


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestTransformerBlockBuildAndForward(unittest.TestCase):
    """Build a real block from lightweight real layers (via the real
    ``build_spec_layer`` collaborator) and observe construction wiring and the
    forward threading / final-norm behavior. Nothing under test is mocked."""

    def setUp(self):
        self.cfg = _make_config()
        self.pg = object()
        self.scales = (2.0, 3.0)
        self.layer_specs = [
            LayerSpec(layer=_RecordingLayer, extra_kwargs={"scale": s})
            for s in self.scales
        ]

    def _build(self, layer_norm, **block_kwargs):
        spec = TransformerBlockSublayersSpec(
            layer_specs=list(self.layer_specs),
            layer_norm=layer_norm,
        )
        return TransformerBlock(
            config=self.cfg,
            spec=spec,
            pg_collection=self.pg,
            **block_kwargs,
        )

    def _input(self):
        return paddle.arange(2 * 3 * 8, dtype="float32").reshape([2, 3, 8])

    def _ref_layers(self, hidden_np):
        # Independent re-derivation of the two-layer affine sweep. Layer i
        # (1-based) computes ``h * scale_i + i``; layers run in order.
        out = hidden_np * self.scales[0] + 1.0
        out = out * self.scales[1] + 2.0
        return out

    def test_num_layers_matches_spec(self):
        block = self._build(layer_norm=LayerSpec(layer=_RecordingNorm))
        self.assertEqual(len(block.layers), 2)
        self.assertEqual(block.num_layers_per_pipeline_rank, 2)

    def test_layers_receive_sequential_numbers_pg_and_scale(self):
        block = self._build(layer_norm=None)
        self.assertEqual([layer.layer_number for layer in block.layers], [1, 2])
        for layer in block.layers:
            self.assertIs(layer.pg_collection, self.pg)
            self.assertIs(layer.config, self.cfg)
        self.assertEqual([layer.scale for layer in block.layers], [2.0, 3.0])

    def test_get_layer_returns_indexed_layer(self):
        block = self._build(layer_norm=None)
        self.assertIs(block._get_layer(0), block.layers[0])
        self.assertIs(block._get_layer(1), block.layers[1])

    def test_norm_built_with_config_hidden_size_and_eps(self):
        block = self._build(layer_norm=LayerSpec(layer=_RecordingNorm))
        self.assertIsInstance(block.norm, _RecordingNorm)
        self.assertEqual(block.norm.hidden_size, self.cfg.hidden_size)
        self.assertEqual(block.norm.eps, self.cfg.rms_norm_eps)
        self.assertIs(block.norm.config, self.cfg)

    def test_forward_threads_layers_in_order_and_applies_norm(self):
        block = self._build(layer_norm=LayerSpec(layer=_RecordingNorm))
        h0 = self._input()
        out = block(h0, None)
        expected = self._ref_layers(h0.numpy().astype(np.float64)) + 100.0
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-6, atol=1e-6
        )
        # A block that skipped the final norm would land on ``expected - 100``;
        # confirm the two are distinguishable so the norm assertion has teeth.
        self.assertFalse(np.allclose(expected, expected - 100.0))

    def test_forward_without_norm_returns_layer_sweep_only(self):
        block = self._build(layer_norm=None)
        self.assertIsNone(block.norm)
        h0 = self._input()
        out = block(h0, None)
        expected = self._ref_layers(h0.numpy().astype(np.float64))
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-6, atol=1e-6
        )

    def test_no_norm_when_post_layer_norm_false(self):
        block = self._build(
            layer_norm=LayerSpec(layer=_RecordingNorm), post_layer_norm=False
        )
        self.assertIsNone(block.norm)

    def test_no_norm_when_post_process_false(self):
        block = self._build(
            layer_norm=LayerSpec(layer=_RecordingNorm), post_process=False
        )
        self.assertIsNone(block.norm)

    def test_set_input_tensor_stores_tensor(self):
        block = self._build(layer_norm=None, pre_process=False)
        t = self._input()
        block.set_input_tensor(t)
        self.assertIs(block.input_tensor, t)

    def test_pre_process_false_uses_stored_input_tensor(self):
        block = self._build(layer_norm=None, pre_process=False)
        real = self._input()
        decoy = paddle.full([2, 3, 8], -999.0, dtype="float32")
        block.set_input_tensor(real)
        # With pre_process=False the forward argument is ignored in favor of
        # the stored input_tensor.
        out = block(decoy, None)
        expected = self._ref_layers(real.numpy().astype(np.float64))
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-6, atol=1e-6
        )
        decoy_expected = self._ref_layers(decoy.numpy().astype(np.float64))
        self.assertFalse(
            np.allclose(out.numpy().astype(np.float64), decoy_expected)
        )

    def test_wrapped_tensor_input_is_unwrapped(self):
        block = self._build(layer_norm=None)
        h0 = self._input()
        wrapped = WrappedTensor(h0)
        out = block(wrapped, None)
        expected = self._ref_layers(h0.numpy().astype(np.float64))
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
