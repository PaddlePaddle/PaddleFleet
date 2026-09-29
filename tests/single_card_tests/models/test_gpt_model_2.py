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
"""No-card unit tests for a disjoint slice of ``paddlefleet.models.gpt.gpt_model``.

Slice under test (keyed to the imports exercised by the source coverage file):
``build_overlapped_nodes`` scheduling partition, the ``is_vision_merge_key``
checkpoint guard, and the ``GPTModel.fp8_quant_weight`` / ``GPTModel.use_fp8``
dispatch. These are CPU-executable control-flow / string helpers; they are
driven through the real production entry points (no reimplementation of the
logic inside the test, no faking of ``super()``). Paddle is required only
because the module imports ``paddle.distributed.fleet.meta_parallel`` at import
time, so the whole suite is guarded and honestly skipped when it is absent.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

try:
    import paddle  # noqa: F401
    from paddle.distributed.fleet.meta_parallel import (
        ScheduleChunk,
        ScheduleNode,
    )

    from paddlefleet.models.gpt import gpt_model
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
    )

    HAS_PADDLE = True
    IMPORT_ERROR = ""
except ImportError as exc:  # narrow: only a genuine missing dependency
    HAS_PADDLE = False
    IMPORT_ERROR = repr(exc)
    gpt_model = None
    ScheduleChunk = ScheduleNode = None
    TransformerLayerNode = TransformerLayerOverlappedScheduleNode = None

SKIP_REASON = (
    "paddle/paddlefleet not importable in this environment; the gpt_model "
    "module cannot be loaded, so this no-card slice is UNRUN (not "
    "inapplicable): " + IMPORT_ERROR
)


class _DenseLayer:
    """Minimal decoder-layer stand-in accepted by ``TransformerLayerNode``.

    Only used so a real ``TransformerLayerNode`` can be constructed; its
    ``compute_*`` methods are never executed by ``build_overlapped_nodes``,
    which only inspects node *types* and partitions the node lists.
    """

    full_recompute = False
    mlp = object()

    def compute_attention(self, inputs, is_first_fwd=False):
        del is_first_fwd
        return inputs["hidden_states"] + 1.0, None

    def compute_mlp(self, hidden_states, is_first_fwd=False):
        del is_first_fwd
        return hidden_states + 2.0


class _TinyConfig:
    num_nextn_predict_layers = None
    mtp_load_weight_only = False


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """``build_overlapped_nodes`` partitions fwd/bwd schedule chunks."""

    def _plain(self, name):
        # A ScheduleNode is NOT a TransformerLayerNode, so it is treated as a
        # pre/post (non-overlappable) node by the partition logic.
        return ScheduleNode(lambda inputs: inputs, name=name)

    def _decoder(self):
        return TransformerLayerNode(_DenseLayer(), _TinyConfig())

    def test_asymmetric_counts_place_nodes_by_identity(self):
        # Forward holds 2 decoder layers, backward holds 1 -> overlap = 1.
        # The extra forward decoder layer must be demoted to the forward post
        # chunk, and surrounding plain nodes must keep their pre/post side.
        fpre = self._plain("fpre")
        f_t1 = self._decoder()
        f_t2 = self._decoder()
        fpost = self._plain("fpost")
        bpre = self._plain("bpre")
        b_t1 = self._decoder()
        bpost = self._plain("bpost")

        (
            forward_pre,
            backward_pre,
            overlap,
            forward_post,
            backward_post,
        ) = gpt_model.build_overlapped_nodes(
            ScheduleChunk([fpre, f_t1, f_t2, fpost]),
            ScheduleChunk([bpre, b_t1, bpost]),
        )

        # Hand-derived membership (identity, not just counts):
        self.assertEqual(forward_pre.nodes, [fpre])
        self.assertEqual(forward_post.nodes, [f_t2, fpost])
        # backward chunk is walked in reverse then reversed back, so the
        # single trailing plain node lands in pre and the leading one in post.
        self.assertEqual(backward_pre.nodes, [bpost])
        self.assertEqual(backward_post.nodes, [bpre])

        # Exactly one overlapped node, built from the first fwd/bwd decoder.
        self.assertEqual(len(overlap.nodes), 1)
        self.assertIsInstance(
            overlap.nodes[0], TransformerLayerOverlappedScheduleNode
        )
        # The overlapped decoder layers must be consumed, not leaked into
        # pre/post (a swap or off-by-one here would otherwise be invisible).
        for node in (
            forward_pre.nodes
            + forward_post.nodes
            + backward_pre.nodes
            + backward_post.nodes
        ):
            self.assertNotIn(node, (f_t1, b_t1))

    def test_zero_overlap_keeps_decoder_layers_in_post(self):
        # Backward has no decoder layer -> overlap = min(2, 0) = 0. All forward
        # decoder layers must survive in the forward post chunk (not dropped).
        fpre = self._plain("fpre")
        f_t1 = self._decoder()
        f_t2 = self._decoder()
        fpost = self._plain("fpost")
        bpre = self._plain("bpre")
        bpost = self._plain("bpost")

        (
            forward_pre,
            backward_pre,
            overlap,
            forward_post,
            backward_post,
        ) = gpt_model.build_overlapped_nodes(
            ScheduleChunk([fpre, f_t1, f_t2, fpost]),
            ScheduleChunk([bpre, bpost]),
        )

        self.assertEqual(len(overlap.nodes), 0)
        self.assertEqual(forward_pre.nodes, [fpre])
        self.assertEqual(forward_post.nodes, [f_t1, f_t2, fpost])
        # No decoder layers on the backward side -> both plain nodes stay in
        # pre. The backward chunk is walked in reverse then re-reversed, so the
        # original order [bpre, bpost] is preserved.
        self.assertEqual(backward_pre.nodes, [bpre, bpost])
        self.assertEqual(backward_post.nodes, [])


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestIsVisionMergeKey(unittest.TestCase):
    """``is_vision_merge_key`` classifies keys and rejects stray ones."""

    def test_non_vision_merge_key_is_false(self):
        self.assertFalse(
            gpt_model.is_vision_merge_key(
                "model.layers.0.self_attn.q_proj.weight"
            )
        )

    def test_vision_model_key_is_true(self):
        self.assertTrue(
            gpt_model.is_vision_merge_key(
                "vision_merge.vision_model.encoder.blocks.0.attn.qkv.weight"
            )
        )

    def test_stray_vision_merge_key_raises(self):
        # Starts with vision_merge. but not vision_merge.vision_model. -> the
        # parameter would be silently dropped from the checkpoint, so the
        # helper must raise (a bare ``assert`` would vanish under python -O).
        with self.assertRaises(ValueError):
            gpt_model.is_vision_merge_key("vision_merge.extra_head.weight")


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestFp8Dispatch(unittest.TestCase):
    """``GPTModel.fp8_quant_weight`` / ``use_fp8`` dispatch over layers."""

    def _bare_model(self):
        # Build a GPTModel instance without running __init__: the methods
        # under test only read _num_virtual_pipeline_stages, _model_chunks and
        # run_function, so no distributed/topology setup is needed.
        return gpt_model.GPTModel.__new__(gpt_model.GPTModel)

    def _quant_layer_cls(self):
        class _QuantLayer(gpt_model.TransformerLayer):
            def __init__(self, enabled):
                # Deliberately skip TransformerLayer.__init__; isinstance still
                # holds and only the two leaf hooks below are exercised.
                self.enabled = enabled
                self.quant_calls = []

            def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
                self.quant_calls.append((batch_mode, quant_transpose))

            def use_fp8(self):
                return self.enabled

        return _QuantLayer

    def _mtp_wrapper(self, inner):
        class _MTPWrapper(gpt_model.MultiTokenPredictionLayer):
            def __init__(self, layer):
                object.__setattr__(self, "transformer_layer", layer)

        return _MTPWrapper(inner)

    def test_fp8_quant_weight_non_vpp_forwards_flags_and_routes_mtp(self):
        quant_cls = self._quant_layer_cls()
        direct = quant_cls(enabled=True)
        inner = quant_cls(enabled=True)
        mtp = self._mtp_wrapper(inner)
        model = self._bare_model()
        model._num_virtual_pipeline_stages = 1
        # A non-layer object must be silently skipped (not raise).
        model.run_function = [object(), direct, mtp]

        model.fp8_quant_weight(batch_mode=True, quant_transpose=False)

        # Exact flags are forwarded, and the MTP layer is routed through its
        # wrapped .transformer_layer rather than quantized directly.
        self.assertEqual(direct.quant_calls, [(True, False)])
        self.assertEqual(inner.quant_calls, [(True, False)])

    def test_fp8_quant_weight_vpp_dispatches_over_chunks(self):
        quant_cls = self._quant_layer_cls()
        layer = quant_cls(enabled=True)
        model = self._bare_model()
        model._num_virtual_pipeline_stages = 2
        # First chunk holds only a non-layer object; second holds the real one.
        model._model_chunks = [[object()], [layer]]

        model.fp8_quant_weight(batch_mode=False, quant_transpose=True)

        self.assertEqual(layer.quant_calls, [(False, True)])

    def test_use_fp8_non_vpp_reports_true_and_false(self):
        quant_cls = self._quant_layer_cls()
        model = self._bare_model()
        model._num_virtual_pipeline_stages = 1

        model.run_function = [object(), quant_cls(enabled=True)]
        self.assertIs(model.use_fp8(), True)

        # No fp8 layer -> the non-VPP branch explicitly returns False.
        model.run_function = [quant_cls(enabled=False), object()]
        self.assertIs(model.use_fp8(), False)

    @unittest.expectedFailure
    def test_use_fp8_vpp_returns_false_when_no_fp8_layer(self):
        # REAL BUG (gpt_model.py:986-996): the VPP branch has no terminal
        # ``return False``; when no chunk layer uses fp8 the function falls off
        # the end and returns None instead of False. Assert the correct
        # contract (a bool False); this is expected to fail until production is
        # fixed. Production is intentionally NOT modified.
        quant_cls = self._quant_layer_cls()
        model = self._bare_model()
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [[quant_cls(enabled=False)], [object()]]

        self.assertIs(model.use_fp8(), False)


if __name__ == "__main__":
    unittest.main()
