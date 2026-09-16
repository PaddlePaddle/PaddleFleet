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

"""Behaviour tests for the pure-Python orchestration of
``paddlefleet.transformer.transformer_encoder.TransformerEncoder``.

Two independent pieces of the encoder are exercised here, both of which run on
CPU without an accelerator and without a Fleet hybrid-communicate group:

* ``get_layer_desc_list`` / ``get_encoder_layer_desc_list`` -- the descriptor
  assembly that turns a ``sublayers_spec`` into an ordered list of
  ``{"layer": ..., "name_prefix": ...}`` entries.  The exact prefix strings,
  the continuous ``.layers.<i>`` index shared across the head / transformer /
  tail loops, and the *asymmetry* whereby the embedding is wrapped in a
  ``LayerDesc`` but the trailing ``layer_norm`` is passed through raw are all
  derived by hand from the production source and asserted in full.

* ``overlapped_forward_backward`` -- the forward/backward interleaving driver.
  It references no instance state, so it is invoked on a bare ``__new__``
  instance (``PipelineLayer.__init__`` is neither run nor faked).  Its genuine
  collaborator ``build_overlapped_nodes`` (covered on its own elsewhere) is
  replaced by a recording double that returns distinguishable pre / overlap /
  post chunks, so the *real* orchestration is observed: the threading of the
  forward activations through pre -> overlap -> post, the scaler branch of the
  backward-loss node, the ``forward_loss`` selection, and the guard that no
  ``TransformerLayerNode`` survives into the overlap chunk.

Expected values are hand-derived from the production implementation; no coverage
fixture is used as an oracle.
"""

import unittest
from unittest import mock

_IMPORT_ERROR = None
try:
    from paddle.distributed.fleet.meta_parallel import LayerDesc

    from paddlefleet.transformer import transformer_encoder
    from paddlefleet.transformer.transformer_encoder import TransformerEncoder
    from paddlefleet.transformer.transformer_layer import TransformerLayerNode
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    LayerDesc = None
    transformer_encoder = None
    TransformerEncoder = None
    TransformerLayerNode = None

_SKIP_REASON = (
    "requires an importable paddle + paddlefleet.transformer stack "
    "(paddle.distributed.fleet.meta_parallel.LayerDesc, TransformerEncoder, "
    f"TransformerLayerNode); import failed with: {_IMPORT_ERROR!r}"
)


# --- Distinguishable marker classes for the descriptor spec -----------------
# The descriptor helpers only *store* / *wrap* these classes; they are never
# instantiated, so plain (non-paddle) classes are sufficient and keep each slot
# individually identifiable to catch swaps between emb / head / tf / tail.
class _Emb:
    pass


class _Head:
    pass


class _T0:
    pass


class _T1:
    pass


class _Tail:
    pass


class _Norm:
    pass


class _Spec:
    """Minimal ``sublayers_spec`` shape consumed by the descriptor helpers."""

    def __init__(
        self,
        embedding,
        head_empty_layers,
        transformer_layers,
        tail_empty_layers,
        layer_norm,
    ):
        self.embedding = embedding
        self.head_empty_layers = head_empty_layers
        self.transformer_layers = transformer_layers
        self.tail_empty_layers = tail_empty_layers
        self.layer_norm = layer_norm


if _IMPORT_ERROR is None:

    class _HelperEncoder(TransformerEncoder):
        """Keep the real descriptor helpers but skip ``PipelineLayer.__init__``.

        The helpers read only ``self.modal`` and call the real
        ``add_sequential_layer`` / ``get_encoder_layer_desc_list`` methods, so
        populating ``modal`` is the only setup required.
        """

        def __init__(self, modal=None):
            self.modal = modal

else:  # pragma: no cover
    _HelperEncoder = None


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestTransformerEncoderDescriptorAssembly(unittest.TestCase):
    """``get_layer_desc_list`` / ``get_encoder_layer_desc_list`` contract."""

    def _spec(self):
        # Two transformer layers so ordering is discriminated beyond a single
        # degenerate element; a head and a tail empty layer so the shared index
        # counter is observed spanning all three loops.
        return _Spec(
            embedding=_Emb,
            head_empty_layers=[_Head],
            transformer_layers=[_T0, _T1],
            tail_empty_layers=[_Tail],
            layer_norm=_Norm,
        )

    def test_encoder_layer_index_is_continuous_across_loops(self):
        enc = _HelperEncoder(modal=None)
        layers = []

        enc.get_encoder_layer_desc_list(layers, self._spec(), "model")

        # Hand-derived: head -> .0, transformer -> .1 & .2, tail -> .3. A bug
        # that reset the counter per loop would put the tail at .0/.1.
        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            [
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
            ],
        )
        expected_funcs = [_Head, _T0, _T1, _Tail]
        for entry, expected_cls in zip(layers, expected_funcs):
            self.assertIsInstance(entry["layer"], LayerDesc)
            self.assertIs(entry["layer"].layer_func, expected_cls)

    def test_get_layer_desc_list_model_prefix_and_wrapping(self):
        enc = _HelperEncoder(modal=None)

        layers = enc.get_layer_desc_list(self._spec())

        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            [
                "model",
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.layers.3",
                "model",
            ],
        )
        # Embedding + the four encoder layers are wrapped in LayerDesc and keep
        # the exact class in the exact slot.
        expected_funcs = [_Emb, _Head, _T0, _T1, _Tail]
        for entry, expected_cls in zip(layers[:5], expected_funcs):
            self.assertIsInstance(entry["layer"], LayerDesc)
            self.assertIs(entry["layer"].layer_func, expected_cls)
        # Asymmetry: the trailing layer_norm is passed through RAW (not wrapped
        # in a LayerDesc), unlike the embedding.
        self.assertIs(layers[5]["layer"], _Norm)
        self.assertNotIsInstance(layers[5]["layer"], LayerDesc)

    def test_get_layer_desc_list_modal_prefix(self):
        enc = _HelperEncoder(modal="vision")

        layers = enc.get_layer_desc_list(self._spec())

        # A set modal switches every prefix from "model" to "model.vision".
        self.assertEqual(
            [entry["name_prefix"] for entry in layers],
            [
                "model.vision",
                "model.vision.layers.0",
                "model.vision.layers.1",
                "model.vision.layers.2",
                "model.vision.layers.3",
                "model.vision",
            ],
        )


# --- Recording doubles for the build_overlapped_nodes result ----------------
class _StubChunk:
    """Stand-in for a ScheduleChunk returned by ``build_overlapped_nodes``.

    Records the exact argument each ``forward`` / ``backward`` receives and
    returns a value tagged with its identity, so the driver's threading of
    activations and gradients can be followed precisely.
    """

    def __init__(self, tag, nodes=None):
        self.tag = tag
        self.nodes = list(nodes or [])
        self.forward_seen = []
        self.backward_seen = []

    def forward(self, x):
        self.forward_seen.append(x)
        return (self.tag, "fwd", x)

    def backward(self, g):
        self.backward_seen.append(g)
        return (self.tag, "bwd", g)


class _OverlapItem:
    """A non-decoder overlap node exposing ``forward_backward``."""

    def __init__(self, tag):
        self.tag = tag
        self.seen = None

    def forward_backward(self, fwd, bwd):
        self.seen = (fwd, bwd)
        return ((self.tag, "ov_fwd", fwd), (self.tag, "ov_bwd", bwd))


class _RecordingBackwardLoss:
    """Backward-loss node recording whether ``scaler`` was passed."""

    _UNSET = object()

    def __init__(self, ret):
        self._ret = ret
        self.scaler_seen = []

    def backward(self, scaler=_UNSET):
        self.scaler_seen.append(scaler)
        return self._ret


class _RecordingForwardLoss:
    def __init__(self, ret):
        self._ret = ret
        self.forward_seen = []

    def forward(self, x):
        self.forward_seen.append(x)
        return self._ret


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestOverlappedForwardBackward(unittest.TestCase):
    """``overlapped_forward_backward`` orchestration contract.

    ``build_overlapped_nodes`` is a genuine collaborator (verified on its own
    elsewhere); it is replaced by a recording double so this test isolates the
    driver's own logic.  The double is given distinguishable responses and the
    real orchestration around it is observed.
    """

    def _encoder(self):
        # The method references no instance state; a bare instance is enough
        # and avoids the Fleet-dependent PipelineLayer.__init__.
        return TransformerEncoder.__new__(TransformerEncoder)

    def _patch_build(self, f_pre, b_pre, overlap, f_post, b_post):
        captured = {}

        def fake_build(forward_chunk, backward_chunk):
            captured["args"] = (forward_chunk, backward_chunk)
            return (f_pre, b_pre, overlap, f_post, b_post)

        patcher = mock.patch.object(
            transformer_encoder,
            "build_overlapped_nodes",
            side_effect=fake_build,
        )
        return patcher, captured

    def test_forward_threading_and_no_loss(self):
        enc = self._encoder()
        f_pre = _StubChunk("f_pre")
        b_pre = _StubChunk("b_pre")
        overlap = _StubChunk("overlap", nodes=[])
        f_post = _StubChunk("f_post")
        b_post = _StubChunk("b_post")
        patcher, captured = self._patch_build(
            f_pre, b_pre, overlap, f_post, b_post
        )

        fwd_chunk = object()
        bwd_chunk = object()
        with patcher:
            forward_inputs, forward_loss, backward_grads = (
                enc.overlapped_forward_backward(
                    forward_chunk=fwd_chunk,
                    forward_inputs="FWD_IN",
                    forward_loss_fn_node=None,
                    backward_chunk=bwd_chunk,
                    backward_loss_fn_node=None,
                    backward_input_grads="BWD_IN",
                    scaler=None,
                    p2p_async_handle=None,
                )
            )

        # build_overlapped_nodes is driven with the real forward/backward chunks.
        self.assertIs(captured["args"][0], fwd_chunk)
        self.assertIs(captured["args"][1], bwd_chunk)

        # Forward activations flow pre -> post (overlap empty): pre sees the
        # original input, post sees pre's output, the return is post's output.
        self.assertEqual(f_pre.forward_seen, ["FWD_IN"])
        self.assertEqual(f_post.forward_seen, [("f_pre", "fwd", "FWD_IN")])
        self.assertEqual(
            forward_inputs, ("f_post", "fwd", ("f_pre", "fwd", "FWD_IN"))
        )

        # Backward gradients flow pre -> post symmetrically.
        self.assertEqual(b_pre.backward_seen, ["BWD_IN"])
        self.assertEqual(b_post.backward_seen, [("b_pre", "bwd", "BWD_IN")])
        self.assertEqual(
            backward_grads, ("b_post", "bwd", ("b_pre", "bwd", "BWD_IN"))
        )

        # No forward-loss node -> forward_loss is None (not some stray value).
        self.assertIsNone(forward_loss)

    def test_forward_loss_from_node_consumes_post_activations(self):
        enc = self._encoder()
        f_pre = _StubChunk("f_pre")
        b_pre = _StubChunk("b_pre")
        overlap = _StubChunk("overlap", nodes=[])
        f_post = _StubChunk("f_post")
        b_post = _StubChunk("b_post")
        loss_node = _RecordingForwardLoss(ret="LOSS_VALUE")
        patcher, _ = self._patch_build(f_pre, b_pre, overlap, f_post, b_post)

        with patcher:
            _, forward_loss, _ = enc.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs="FWD_IN",
                forward_loss_fn_node=loss_node,
                backward_chunk=object(),
                backward_loss_fn_node=None,
                backward_input_grads="BWD_IN",
                scaler=None,
                p2p_async_handle=None,
            )

        # forward_loss is the node's output, and the node consumed the fully
        # post-processed activations (proving loss runs after pre+post).
        self.assertEqual(forward_loss, "LOSS_VALUE")
        self.assertEqual(
            loss_node.forward_seen,
            [("f_post", "fwd", ("f_pre", "fwd", "FWD_IN"))],
        )

    def test_overlap_nodes_transform_and_thread_between_pre_and_post(self):
        enc = self._encoder()
        f_pre = _StubChunk("f_pre")
        b_pre = _StubChunk("b_pre")
        item = _OverlapItem("ov0")
        overlap = _StubChunk("overlap", nodes=[item])
        f_post = _StubChunk("f_post")
        b_post = _StubChunk("b_post")
        patcher, _ = self._patch_build(f_pre, b_pre, overlap, f_post, b_post)

        with patcher:
            forward_inputs, _, backward_grads = enc.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs="FWD_IN",
                forward_loss_fn_node=None,
                backward_chunk=object(),
                backward_loss_fn_node=None,
                backward_input_grads="BWD_IN",
                scaler=None,
                p2p_async_handle=None,
            )

        # The overlap item received pre outputs for both directions...
        self.assertEqual(
            item.seen,
            (("f_pre", "fwd", "FWD_IN"), ("b_pre", "bwd", "BWD_IN")),
        )
        # ...and post consumed the overlap item's transformed outputs.
        self.assertEqual(
            f_post.forward_seen, [("ov0", "ov_fwd", ("f_pre", "fwd", "FWD_IN"))]
        )
        self.assertEqual(
            b_post.backward_seen,
            [("ov0", "ov_bwd", ("b_pre", "bwd", "BWD_IN"))],
        )
        self.assertEqual(
            forward_inputs,
            ("f_post", "fwd", ("ov0", "ov_fwd", ("f_pre", "fwd", "FWD_IN"))),
        )
        self.assertEqual(
            backward_grads,
            ("b_post", "bwd", ("ov0", "ov_bwd", ("b_pre", "bwd", "BWD_IN"))),
        )

    def test_backward_loss_node_passes_scaler_when_present(self):
        enc = self._encoder()
        f_pre = _StubChunk("f_pre")
        b_pre = _StubChunk("b_pre")
        overlap = _StubChunk("overlap", nodes=[])
        f_post = _StubChunk("f_post")
        b_post = _StubChunk("b_post")
        bwd_loss = _RecordingBackwardLoss(ret="GRAD_FROM_LOSS")
        scaler = object()  # truthy
        patcher, _ = self._patch_build(f_pre, b_pre, overlap, f_post, b_post)

        with patcher:
            enc.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs="FWD_IN",
                forward_loss_fn_node=None,
                backward_chunk=object(),
                backward_loss_fn_node=bwd_loss,
                backward_input_grads="IGNORED_INITIAL",
                scaler=scaler,
                p2p_async_handle=None,
            )

        # scaler is truthy -> backward(scaler=scaler); the initial
        # backward_input_grads is discarded in favour of the loss node's output,
        # which is what the backward pre-chunk then consumes.
        self.assertEqual(bwd_loss.scaler_seen, [scaler])
        self.assertEqual(b_pre.backward_seen, ["GRAD_FROM_LOSS"])

    def test_backward_loss_node_omits_scaler_when_falsy(self):
        enc = self._encoder()
        f_pre = _StubChunk("f_pre")
        b_pre = _StubChunk("b_pre")
        overlap = _StubChunk("overlap", nodes=[])
        f_post = _StubChunk("f_post")
        b_post = _StubChunk("b_post")
        bwd_loss = _RecordingBackwardLoss(ret="GRAD_NO_SCALER")
        patcher, _ = self._patch_build(f_pre, b_pre, overlap, f_post, b_post)

        with patcher:
            enc.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs="FWD_IN",
                forward_loss_fn_node=None,
                backward_chunk=object(),
                backward_loss_fn_node=bwd_loss,
                backward_input_grads="IGNORED_INITIAL",
                scaler=None,
                p2p_async_handle=None,
            )

        # scaler falsy -> backward() called with no scaler argument (sentinel),
        # and its return still feeds the backward pre-chunk.
        self.assertEqual(bwd_loss.scaler_seen, [_RecordingBackwardLoss._UNSET])
        self.assertEqual(b_pre.backward_seen, ["GRAD_NO_SCALER"])

    def test_transformer_layer_node_in_overlap_is_rejected(self):
        enc = self._encoder()

        # A genuine-typed decoder marker: subclassing the real
        # TransformerLayerNode makes the production ``isinstance`` guard fire
        # without running its heavy __init__.
        class _TLNMarker(TransformerLayerNode):
            def __init__(self):
                pass

        f_pre = _StubChunk("f_pre")
        b_pre = _StubChunk("b_pre")
        overlap = _StubChunk("overlap", nodes=[_TLNMarker()])
        f_post = _StubChunk("f_post")
        b_post = _StubChunk("b_post")
        patcher, _ = self._patch_build(f_pre, b_pre, overlap, f_post, b_post)

        # The driver asserts no TransformerLayerNode survives into the overlap
        # chunk; a decoder layer there must raise AssertionError.
        with patcher, self.assertRaises(AssertionError):
            enc.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs="FWD_IN",
                forward_loss_fn_node=None,
                backward_chunk=object(),
                backward_loss_fn_node=None,
                backward_input_grads="BWD_IN",
                scaler=None,
                p2p_async_handle=None,
            )


if __name__ == "__main__":
    unittest.main()
