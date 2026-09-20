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

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.transformer_layer import TransformerLayerNode

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:
    paddle = None
    TransformerLayerNode = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet.transformer.transformer_layer not importable on this "
    f"CPU-only host (real dependency error, not faked): {_IMPORT_ERROR!r}"
)


class _DenseLayer:
    """Minimal but genuine collaborator implementing the compute_attention /
    compute_mlp contract that ``TransformerLayerNode`` wires into schedule
    nodes.

    ``mlp`` is a plain object (NOT a ``MoELayer``), so the node takes the dense
    schedule path. The stub math is position-distinguishable and uses a
    different op in attention (multiply) versus mlp (add) so that a swapped
    stage order or a dropped stage changes the observed output.
    """

    full_recompute = False

    def __init__(self, with_context=False):
        self.with_context = with_context
        self.mlp = object()  # not a MoELayer -> dense schedule path

    def compute_attention(self, inputs, is_first_fwd=False):
        del is_first_fwd
        source = inputs["hidden_states"]
        hidden_states = source * 2.0
        context = None
        if self.with_context:
            context = source * 2.0 + 100.0
        return hidden_states, context

    def compute_mlp(self, hidden_states, is_first_fwd=False):
        del is_first_fwd
        return hidden_states + 10.0


def _config(num_nextn_predict_layers=0, mtp_load_weight_only=False):
    # forward()/backward() read exactly these two attributes off self.config.
    return SimpleNamespace(
        num_nextn_predict_layers=num_nextn_predict_layers,
        mtp_load_weight_only=mtp_load_weight_only,
    )


def _first_tensor(value):
    if isinstance(value, (list, tuple)):
        return _first_tensor(value[0])
    return value


@unittest.skipUnless(
    paddle is not None and TransformerLayerNode is not None, _SKIP_REASON
)
class TestTransformerLayerNodeDenseSchedule(unittest.TestCase):
    def test_dense_forward_composes_attention_then_mlp(self):
        node = TransformerLayerNode(_DenseLayer(), _config(), name="dense")
        hidden_states = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float32"
        )
        mask = paddle.to_tensor([[9.0, 8.0]], dtype="float32")

        result = node.forward(
            {
                "hidden_states": hidden_states,
                "attention_mask": mask,
                "dynamic_inference_decode_only": True,
            }
        )

        # output = mlp(attn(x)) = (x * 2) + 10, elementwise and order sensitive.
        # A swapped order would give (x + 10) * 2 = 2x + 20 -> different values.
        np.testing.assert_allclose(
            result["hidden_states"].numpy(),
            np.array([[12.0, 14.0], [16.0, 18.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        # Non-hidden inputs pass through untouched via {**inputs, **rst}.
        np.testing.assert_array_equal(
            result["attention_mask"].numpy(), mask.numpy()
        )
        # Dense layer produced no context -> the key must be absent.
        self.assertNotIn("context", result)
        # forward() pops dynamic_inference_decode_only; it must not be echoed.
        self.assertNotIn("dynamic_inference_decode_only", result)

    def test_dense_forward_propagates_attention_context(self):
        node = TransformerLayerNode(_DenseLayer(with_context=True), _config())
        source = paddle.to_tensor([[1.0, 2.0]], dtype="float32")

        result = node.forward({"hidden_states": source})

        # context is the second attention output threaded straight into rst.
        np.testing.assert_allclose(
            result["context"].numpy(),
            np.array([[102.0, 104.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        # hidden_states is still (x * 2) + 10 and independent of the context.
        np.testing.assert_allclose(
            result["hidden_states"].numpy(),
            np.array([[12.0, 14.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_dense_backward_propagates_scaled_gradient(self):
        node = TransformerLayerNode(_DenseLayer(), _config())
        hidden_states = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float32"
        )
        hidden_states.stop_gradient = False

        node.forward({"hidden_states": hidden_states})
        upstream = paddle.to_tensor([[1.0, 5.0], [2.0, 3.0]], dtype="float32")
        grads = node.backward(upstream)
        grad = _first_tensor(grads)

        # output = 2x + 10 -> d(output)/dx = 2, so input grad = 2 * upstream.
        # Non-uniform upstream + exact (scale-sensitive) compare rejects a
        # halved / doubled / zeroed gradient and any transpose.
        self.assertIsNotNone(grad)
        np.testing.assert_allclose(
            grad.numpy(),
            2.0 * upstream.numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertGreater(float(np.abs(grad.numpy()).max()), 0.0)

    def test_forward_asserts_mtp_layers_unsupported(self):
        # transformer_layer.py:2633-2638 hard-asserts num_nextn_predict_layers
        # in (None, 0) at the very top of forward(). Consequently the
        # decoder_input_i handling block right below it (lines 2639-2648) is
        # unreachable in forward(): the dense/sparse forward path can never
        # consume MTP inputs even though backward()/forward_backward() carry
        # MTP grad-splitting code. This asserts the guard genuinely fires.
        node = TransformerLayerNode(
            _DenseLayer(), _config(num_nextn_predict_layers=1)
        )
        with self.assertRaises(AssertionError):
            node.forward(
                {
                    "hidden_states": paddle.to_tensor(
                        [[1.0, 2.0]], dtype="float32"
                    ),
                    "decoder_input_0": paddle.to_tensor(
                        [[3.0, 3.0]], dtype="float32"
                    ),
                }
            )


if __name__ == "__main__":
    unittest.main()
