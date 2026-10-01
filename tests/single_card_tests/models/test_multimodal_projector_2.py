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

"""Model-layer tests for ``paddlefleet.models.vision.multimodal_projector``.

Target under test: ``MultimodalProjector.__init__`` branch selection and
``MultimodalProjector.forward`` bias-add logic. Every expectation is
hand-derived from the production control flow, not read back from the object
under test.

Coverage slice notes:
 - The projector picks an encoder by ``projector_type``: ``"mlp"`` builds an
   ``MLP``, ``"affine"`` builds a ``build_spec_layer`` linear, anything else
   raises. ``MLP`` / ``build_spec_layer`` are independently-tested collaborators,
   so they are isolated with distinguishable stand-ins and the *orchestration*
   (exact forwarded arguments, stored ``encoder``, stored ``projector_type``) is
   observed -- see the arg-capture assertions.
 - ``forward`` returns ``encoder_output`` and, only when the encoder returns a
   non-None bias, ``encoder_output + bias``. Both branches are exercised with
   hand-derived tensors, and the None branch asserts object identity so that a
   spurious add would be caught.

The module imports ``paddle`` transitively; when Paddle (and hence the package)
is unavailable the whole suite is skipped with an honest reason rather than
faking a pass.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
# Also make the ``src/`` layout importable when the package is not installed.
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
    import numpy as np
    import paddle

    from paddlefleet.models.vision.multimodal_projector import (
        MultimodalProjector,
    )
    from paddlefleet.transformer.mlp import MLPSublayersSpec
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # pragma: no cover - depends on runtime deps
    _IMPORT_OK = False
    _IMPORT_ERR = repr(exc)
    np = None
    paddle = None
    MultimodalProjector = None
    MLPSublayersSpec = None
    TransformerConfig = None

_SKIP_REASON = (
    "paddlefleet.models.vision.multimodal_projector not importable "
    "(missing paddle or package): " + _IMPORT_ERR
)

_PROJ_MODULE = "paddlefleet.models.vision.multimodal_projector"


def _small_config():
    """A minimal CPU-only TransformerConfig sufficient for projector init."""
    return TransformerConfig(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )


class _RecordingEncoder:
    """Stand-in encoder that records its call and returns fixed outputs.

    Used only to isolate ``forward`` from the encoder collaborator while the
    projector's own bias-add logic runs for real.
    """

    def __init__(self, output, bias):
        self._output = output
        self._bias = bias
        self.received = []

    def __call__(self, hidden_states):
        self.received.append(hidden_states)
        return self._output, self._bias


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestMultimodalProjectorConstruction(unittest.TestCase):
    """Branch selection and argument forwarding in ``__init__``."""

    def test_missing_sublayers_spec_raises_assertion(self):
        # __init__ asserts ``sublayers_spec is not None`` after storing
        # projector_type; a None spec must be rejected with the exact message.
        config = _small_config()
        with self.assertRaisesRegex(
            AssertionError, "MLPSublayersSpec must be provided"
        ):
            MultimodalProjector(
                config=config,
                sublayers_spec=None,
                projector_type="mlp",
                input_size=64,
            )

    def test_unsupported_projector_type_raises(self):
        # A valid spec passes the assert; an unknown type hits the else branch
        # and raises with the type echoed into the message.
        config = _small_config()
        spec = MLPSublayersSpec()
        with self.assertRaisesRegex(
            Exception, "Unsupported multimodal projection type conv"
        ):
            MultimodalProjector(
                config=config,
                sublayers_spec=spec,
                projector_type="conv",
                input_size=64,
            )

    def test_mlp_path_forwards_arguments_and_stores_encoder(self):
        # The "mlp" branch must build MLP with the *same* config/spec objects,
        # the given input_size and tp_group, and store the result as ``encoder``.
        config = _small_config()
        spec = MLPSublayersSpec()
        tp_group = object()
        sentinel_encoder = object()

        with patch(_PROJ_MODULE + ".MLP") as mock_mlp:
            mock_mlp.return_value = sentinel_encoder
            model = MultimodalProjector(
                config=config,
                sublayers_spec=spec,
                projector_type="mlp",
                input_size=77,
                tp_group=tp_group,
            )

        self.assertEqual(model.projector_type, "mlp")
        self.assertIs(model.encoder, sentinel_encoder)
        mock_mlp.assert_called_once()
        _, kwargs = mock_mlp.call_args
        self.assertIs(kwargs["config"], config)
        self.assertIs(kwargs["sublayers_spec"], spec)
        self.assertEqual(kwargs["input_size"], 77)
        self.assertIs(kwargs["tp_group"], tp_group)

    def test_mlp_path_does_not_call_build_spec_layer(self):
        # Selecting "mlp" must not go through the affine build path.
        config = _small_config()
        spec = MLPSublayersSpec()
        with (
            patch(_PROJ_MODULE + ".MLP") as mock_mlp,
            patch(_PROJ_MODULE + ".build_spec_layer") as mock_build,
        ):
            mock_mlp.return_value = object()
            MultimodalProjector(
                config=config,
                sublayers_spec=spec,
                projector_type="mlp",
                input_size=64,
            )
        mock_build.assert_not_called()

    @unittest.expectedFailure
    def test_affine_path_builds_encoder(self):
        # CORRECT behavior: the "affine" branch should build the encoder via
        # build_spec_layer with gather_output=True / skip_bias_add=True and the
        # given tp_group, then store it as ``encoder``.
        #
        # REAL BUG (currently fails): the affine branch reads attributes that do
        # not exist on the installed classes, so it raises AttributeError before
        # build_spec_layer is ever invoked:
        #   * multimodal_projector.py:61 -> ``sublayers_spec.linear_fc1``
        #     (MLPSublayersSpec fields are up_gate_proj / hidden_act / down_proj;
        #      there is no linear_fc1).
        #   * multimodal_projector.py:67 -> ``config.add_bias_linear``
        #     (TransformerConfig exposes ``use_bias``, not ``add_bias_linear``).
        # This test asserts the intended contract and is marked expectedFailure
        # so production is left untouched; it will flag as unexpected success
        # once the attribute references are corrected.
        config = _small_config()
        spec = MLPSublayersSpec()
        tp_group = object()
        sentinel_encoder = object()

        with patch(_PROJ_MODULE + ".build_spec_layer") as mock_build:
            mock_build.return_value = sentinel_encoder
            model = MultimodalProjector(
                config=config,
                sublayers_spec=spec,
                projector_type="affine",
                input_size=55,
                tp_group=tp_group,
            )

        self.assertEqual(model.projector_type, "affine")
        self.assertIs(model.encoder, sentinel_encoder)
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        self.assertTrue(kwargs["gather_output"])
        self.assertTrue(kwargs["skip_bias_add"])
        self.assertIs(kwargs["tp_group"], tp_group)
        self.assertEqual(kwargs["config"], config)


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestMultimodalProjectorForward(unittest.TestCase):
    """``forward`` bias handling with hand-derived expectations."""

    def _build_mlp_projector(self, encoder):
        # Build a real projector through __init__ (mlp branch) with MLP isolated,
        # then let the real forward run against a distinguishable encoder.
        config = _small_config()
        spec = MLPSublayersSpec()
        with patch(_PROJ_MODULE + ".MLP") as mock_mlp:
            mock_mlp.return_value = encoder
            model = MultimodalProjector(
                config=config,
                sublayers_spec=spec,
                projector_type="mlp",
                input_size=3,
            )
        self.assertIs(model.encoder, encoder)
        return model

    def test_forward_adds_nonzero_bias(self):
        hidden_states = paddle.to_tensor(
            [[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]], dtype="float32"
        )
        output = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        bias = paddle.to_tensor([10.0, 20.0, 30.0], dtype="float32")
        encoder = _RecordingEncoder(output, bias)
        model = self._build_mlp_projector(encoder)

        result = model.forward(hidden_states)

        # Encoder must be called exactly once with the *same* hidden_states.
        self.assertEqual(len(encoder.received), 1)
        self.assertIs(encoder.received[0], hidden_states)
        # Hand-derived: output + broadcast bias.
        expected = np.array(
            [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(result.numpy(), expected)
        # A bias add produces a new tensor, not the encoder output object.
        self.assertIsNot(result, output)

    def test_forward_returns_encoder_output_when_bias_none(self):
        hidden_states = paddle.to_tensor([[7.0, 8.0, 9.0]], dtype="float32")
        output = paddle.to_tensor([[1.5, -2.5, 3.5]], dtype="float32")
        encoder = _RecordingEncoder(output, None)
        model = self._build_mlp_projector(encoder)

        result = model.forward(hidden_states)

        self.assertEqual(len(encoder.received), 1)
        self.assertIs(encoder.received[0], hidden_states)
        # None bias: the exact encoder output object is returned unchanged.
        self.assertIs(result, output)
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([[1.5, -2.5, 3.5]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
