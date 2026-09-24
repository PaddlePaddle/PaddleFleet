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

from __future__ import annotations

import unittest
from unittest import mock

try:
    import paddle
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
    from paddlefleet.transformer.transformer_block import (
        TransformerBlock,
        TransformerBlockSublayersSpec,
        _get_block_sublayers_spec,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    _IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_DEPS_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle/paddlefleet transformer stack not importable in this "
    f"environment: {_IMPORT_ERROR!r}"
)

_BUILD_TARGET = "paddlefleet.transformer.transformer_block.build_spec_layer"


def _make_config(**overrides):
    defaults = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestGetBlockSublayersSpec(unittest.TestCase):
    """Behavior of the spec-resolution helper _get_block_sublayers_spec."""

    def test_passthrough_returns_same_sublayers_spec(self):
        # An already-resolved TransformerBlockSublayersSpec must be returned
        # unchanged (identity), not rebuilt.
        inner = TransformerBlockSublayersSpec(
            layer_specs=["a", "b"], layer_norm="norm"
        )
        result = _get_block_sublayers_spec(_make_config(), inner)
        self.assertIs(result, inner)

    def test_transformer_layer_spec_is_replicated_per_layer(self):
        # A LayerSpec pointing at TransformerLayer must expand to exactly
        # num_hidden_layers copies of that SAME spec, with layer_norm defaulted
        # to the real WrappedPaddleNorm implementation.
        config = _make_config(num_hidden_layers=3)
        layer_spec = LayerSpec(layer=TransformerLayer)

        result = _get_block_sublayers_spec(config, layer_spec)

        self.assertIsInstance(result, TransformerBlockSublayersSpec)
        self.assertEqual(len(result.layer_specs), config.num_hidden_layers)
        for entry in result.layer_specs:
            self.assertIs(entry, layer_spec)
        self.assertIs(result.layer_norm, WrappedPaddleNorm)

    def test_transformer_block_spec_returns_its_inner_sublayers_spec(self):
        # A LayerSpec pointing at TransformerBlock must hand back the exact
        # sublayers_spec attached to it.
        inner = TransformerBlockSublayersSpec(layer_specs=["x"], layer_norm="n")
        block_spec = LayerSpec(layer=TransformerBlock, sublayers_spec=inner)

        result = _get_block_sublayers_spec(_make_config(), block_spec)

        self.assertIs(result, inner)

    def test_layer_spec_with_unknown_layer_raises_named_error(self):
        # LayerSpec whose layer is neither a TransformerBlock nor a
        # TransformerLayer must raise, naming the offending layer class.
        class _Foreign(paddle.nn.Layer):
            pass

        foreign_spec = LayerSpec(layer=_Foreign)
        with self.assertRaises(Exception) as cm:
            _get_block_sublayers_spec(_make_config(), foreign_spec)
        self.assertIn("_Foreign", str(cm.exception))

    def test_non_spec_argument_raises_typed_error(self):
        # A completely unsupported spec type reports its own type name.
        with self.assertRaises(Exception) as cm:
            _get_block_sublayers_spec(_make_config(), "not-a-spec")
        self.assertIn("str", str(cm.exception))


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestTransformerBlockConstruction(unittest.TestCase):
    """Layer wiring and the post-norm gating of TransformerBlock.__init__."""

    def _recording_build(self):
        # Fake for the (not-under-test) build_spec_layer collaborator. It
        # returns real nn.Layer instances (LayerList requires them) tagged with
        # their build order, and records every call so we can assert the exact
        # arguments the block forwarded.
        records = []

        def fake_build(spec, **kwargs):
            records.append({"spec": spec, "kwargs": kwargs})
            layer = paddle.nn.Layer()
            layer._probe_index = len(records) - 1
            return layer

        return records, fake_build

    def test_layers_built_in_order_with_forwarded_arguments(self):
        records, fake_build = self._recording_build()
        config = _make_config()
        pg = object()
        layer_specs = ["spec-0", "spec-1", "spec-2"]
        norm_marker = "layer-norm-marker"
        spec = TransformerBlockSublayersSpec(
            layer_specs=layer_specs, layer_norm=norm_marker
        )

        with mock.patch(_BUILD_TARGET, fake_build):
            block = TransformerBlock(
                config=config,
                spec=spec,
                pg_collection=pg,
                post_layer_norm=True,
                post_process=True,
            )

        # 3 transformer layers + 1 final norm were built.
        self.assertEqual(len(records), 4)
        for i in range(3):
            rec = records[i]
            self.assertIs(rec["spec"], layer_specs[i])
            self.assertEqual(rec["kwargs"]["layer_number"], i + 1)
            self.assertIs(rec["kwargs"]["config"], config)
            self.assertIs(rec["kwargs"]["pg_collection"], pg)

        norm_rec = records[3]
        self.assertIs(norm_rec["spec"], norm_marker)
        self.assertNotIn("layer_number", norm_rec["kwargs"])
        self.assertEqual(norm_rec["kwargs"]["hidden_size"], config.hidden_size)
        self.assertEqual(norm_rec["kwargs"]["eps"], config.rms_norm_eps)

        # LayerList holds exactly the built layers, in build order.
        self.assertEqual(len(block.layers), 3)
        for i in range(3):
            self.assertEqual(block.layers[i]._probe_index, i)
        self.assertEqual(block.num_layers_per_pipeline_rank, 3)
        self.assertIsNotNone(block.norm)
        self.assertEqual(block.norm._probe_index, 3)

    def _build_two_layer_block(self, layer_norm, **block_kwargs):
        records, fake_build = self._recording_build()
        spec = TransformerBlockSublayersSpec(
            layer_specs=["s0", "s1"], layer_norm=layer_norm
        )
        with mock.patch(_BUILD_TARGET, fake_build):
            block = TransformerBlock(
                config=_make_config(),
                spec=spec,
                pg_collection=object(),
                **block_kwargs,
            )
        return block, records

    def test_no_norm_built_when_post_layer_norm_false(self):
        block, records = self._build_two_layer_block(
            layer_norm="norm-marker", post_layer_norm=False, post_process=True
        )
        self.assertIsNone(block.norm)
        self.assertEqual(len(records), 2)  # only the two layers, no norm build

    def test_no_norm_built_when_post_process_false(self):
        block, records = self._build_two_layer_block(
            layer_norm="norm-marker", post_layer_norm=True, post_process=False
        )
        self.assertIsNone(block.norm)
        self.assertEqual(len(records), 2)

    def test_no_norm_built_when_layer_norm_absent(self):
        block, records = self._build_two_layer_block(
            layer_norm=None, post_layer_norm=True, post_process=True
        )
        self.assertIsNone(block.norm)
        self.assertEqual(len(records), 2)

    def test_vp_stage_is_rejected(self):
        spec = TransformerBlockSublayersSpec(
            layer_specs=["s0"], layer_norm=None
        )
        with self.assertRaises(AssertionError):
            TransformerBlock(
                config=_make_config(),
                spec=spec,
                pg_collection=object(),
                vp_stage=1,
            )

    def test_get_layer_maps_index_to_stored_layer(self):
        block, _ = self._build_two_layer_block(layer_norm=None)
        self.assertIs(block._get_layer(0), block.layers[0])
        self.assertIs(block._get_layer(1), block.layers[1])

    def test_set_input_tensor_stores_reference(self):
        block, _ = self._build_two_layer_block(layer_norm=None)
        tensor = paddle.full([1, 1, 16], 5.0, dtype="float32")
        block.set_input_tensor(tensor)
        self.assertIs(block.input_tensor, tensor)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestTransformerBlockForward(unittest.TestCase):
    """Forward-pass orchestration: layer chaining and final norm."""

    def test_forward_chains_layers_then_applies_norm(self):
        # Layers and norm are (not-under-test) collaborators, replaced with
        # input-dependent fakes so we can prove the block feeds each layer the
        # running hidden_states and applies the final norm to the last output.
        log = []

        class _AddLayer(paddle.nn.Layer):
            def __init__(self, delta):
                super().__init__()
                self.delta = delta

            def forward(self, dict_args):
                seen = float(dict_args["hidden_states"].reshape([-1])[0])
                log.append(("layer", self.delta, seen))
                out = dict(dict_args)
                out["hidden_states"] = dict_args["hidden_states"] + self.delta
                return out

        class _ScaleNorm(paddle.nn.Layer):
            def forward(self, x):
                log.append(("norm", float(x.reshape([-1])[0])))
                return x * 10.0

        deltas = iter([1.0, 2.0])

        def fake_build(spec, **kwargs):
            if "hidden_size" in kwargs:  # the final-norm build call
                return _ScaleNorm()
            return _AddLayer(next(deltas))

        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=["s0", "s1"], layer_norm="norm-marker"
        )
        with mock.patch(_BUILD_TARGET, fake_build):
            block = TransformerBlock(
                config=config,
                spec=spec,
                pg_collection=object(),
                post_layer_norm=True,
                post_process=True,
            )

        hidden = paddle.zeros([1, 1, config.hidden_size], dtype="float32")
        attention_mask = paddle.zeros([1, 1, 1, 1], dtype="float32")
        out = block(hidden, attention_mask)

        # layer0 sees 0.0 -> outputs 1.0; layer1 sees 1.0 -> outputs 3.0;
        # norm sees 3.0 and scales by 10 -> 30.0.
        self.assertEqual(log[0], ("layer", 1.0, 0.0))
        self.assertEqual(log[1], ("layer", 2.0, 1.0))
        self.assertEqual(log[2], ("norm", 3.0))
        self.assertEqual(out.shape, [1, 1, config.hidden_size])
        self.assertAlmostEqual(float(out.reshape([-1])[0]), 30.0, places=5)

    def test_forward_uses_input_tensor_when_pre_process_false(self):
        # With pre_process=False the block must ignore the hidden_states passed
        # to forward and consume the tensor set via set_input_tensor instead.
        seen = []

        class _CaptureLayer(paddle.nn.Layer):
            def forward(self, dict_args):
                seen.append(float(dict_args["hidden_states"].reshape([-1])[0]))
                return dict_args

        def fake_build(spec, **kwargs):
            return _CaptureLayer()

        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=["s0"], layer_norm=None
        )
        with mock.patch(_BUILD_TARGET, fake_build):
            block = TransformerBlock(
                config=config,
                spec=spec,
                pg_collection=object(),
                pre_process=False,
            )

        injected = paddle.full([1, 1, config.hidden_size], 7.0, dtype="float32")
        ignored = paddle.full([1, 1, config.hidden_size], 99.0, dtype="float32")
        block.set_input_tensor(injected)
        block(ignored, paddle.zeros([1, 1, 1, 1], dtype="float32"))

        self.assertEqual(seen, [7.0])


if __name__ == "__main__":
    unittest.main()
