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
"""Unit tests for paddlefleet.models.gpt.gpt_model helpers.

Covered production definitions (disjoint slice keyed to this file's imports):
  * ``build_overlapped_nodes``            (gpt_model.py:81)
  * ``GPTModel.overlapped_forward_backward`` (gpt_model.py:486)
  * ``GPTModel.fp8_quant_weight``         (gpt_model.py:946)

Environment: these are CPU-executable control-flow / orchestration paths (no
collective, no device kernels), so they belong to the no-card tier. Importing
``paddlefleet.models.gpt.gpt_model`` nonetheless requires Paddle; when Paddle is
not installed every test is honestly skipped (never reported as passed).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import paddle  # noqa: F401
    from paddle.distributed.fleet.meta_parallel import (
        ScheduleChunk,
        ScheduleNode,
    )

    from paddlefleet.models.gpt import gpt_model as gpt_model_mod
    from paddlefleet.models.gpt.gpt_model import (
        GPTModel,
        build_overlapped_nodes,
    )
    from paddlefleet.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer,
    )
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerNode,
        TransformerLayerOverlappedScheduleNode,
    )

    _IMPORT_ERROR = None
except (
    ImportError
) as exc:  # precise probe: only missing-module skips, not API breaks
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "Paddle / paddlefleet not importable in this environment: "
    f"{_IMPORT_ERROR!r}"
)


def _make_layer_node():
    """Construct a real ``TransformerLayerNode`` on the dense (non-MoE) path.

    ``TransformerLayerNode.__init__`` only touches ``node.compute_attention``,
    ``node.full_recompute``, ``node.mlp`` (isinstance MoELayer check) and
    ``node.compute_mlp``; a lightweight fake satisfies it so the node is a
    genuine instance for the ``isinstance`` partitioning in
    ``build_overlapped_nodes``.
    """
    fake = SimpleNamespace(
        compute_attention=lambda *a, **k: None,
        full_recompute=False,
        mlp=object(),  # not a MoELayer -> dense branch, no dispatcher needed
        compute_mlp=lambda *a, **k: None,
    )
    cfg = SimpleNamespace(
        num_nextn_predict_layers=0, mtp_load_weight_only=False
    )
    return TransformerLayerNode(fake, cfg, name="tln")


def _make_marker(name):
    """A real ScheduleNode that is NOT a TransformerLayerNode (pre/post marker)."""
    return ScheduleNode(lambda x: x, name=name)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """build_overlapped_nodes: partition order, overlap capping, pairing."""

    def test_partitions_reverses_backward_and_pairs_at_min(self):
        # forward chunk holds 3 transformer layers, backward holds 2, so the
        # overlap width is min(3, 2) == 2 and one forward layer spills to post.
        e_f = _make_marker("emb_f")
        t_f0, t_f1, t_f2 = (
            _make_layer_node(),
            _make_layer_node(),
            _make_layer_node(),
        )
        h_f = _make_marker("head_f")
        e_b = _make_marker("emb_b")
        t_b0, t_b1 = _make_layer_node(), _make_layer_node()
        h_b = _make_marker("head_b")

        fwd = ScheduleChunk([e_f, t_f0, t_f1, t_f2, h_f])
        bwd = ScheduleChunk([e_b, t_b0, t_b1, h_b])

        fpre, bpre, overlap, fpost, bpost = build_overlapped_nodes(fwd, bwd)

        # Forward keeps encounter order: leading marker -> pre, extra layer +
        # trailing marker -> post.
        self.assertEqual([id(n) for n in fpre.nodes], [id(e_f)])
        self.assertEqual([id(n) for n in fpost.nodes], [id(t_f2), id(h_f)])

        # Backward is walked in REVERSED order: the trailing marker becomes the
        # (re-reversed) pre chunk, the leading marker becomes the post chunk.
        # A non-reversed implementation would put e_b in pre and h_b in post.
        self.assertEqual([id(n) for n in bpre.nodes], [id(h_b)])
        self.assertEqual([id(n) for n in bpost.nodes], [id(e_b)])

        # Overlap pairs forward layers (in order) with backward layers taken in
        # reversed order: (t_f0, t_b1), (t_f1, t_b0).
        self.assertEqual(len(overlap.nodes), 2)
        for node in overlap.nodes:
            self.assertIsInstance(node, TransformerLayerOverlappedScheduleNode)
        self.assertIs(overlap.nodes[0].forward_node, t_f0)
        self.assertIs(overlap.nodes[0].backward_node, t_b1)
        self.assertIs(overlap.nodes[1].forward_node, t_f1)
        self.assertIs(overlap.nodes[1].backward_node, t_b0)

    def test_no_transformer_layers_yields_empty_overlap(self):
        e_f = _make_marker("emb_f")
        h_f = _make_marker("head_f")
        e_b = _make_marker("emb_b")
        h_b = _make_marker("head_b")

        fwd = ScheduleChunk([e_f, h_f])
        bwd = ScheduleChunk([e_b, h_b])

        fpre, bpre, overlap, fpost, bpost = build_overlapped_nodes(fwd, bwd)

        # With no overlap element, is_pre never flips: everything lands in pre.
        self.assertEqual([id(n) for n in fpre.nodes], [id(e_f), id(h_f)])
        self.assertEqual(fpost.nodes, [])
        self.assertEqual(overlap.nodes, [])
        # Backward pre is reversed twice -> original order.
        self.assertEqual([id(n) for n in bpre.nodes], [id(e_b), id(h_b)])
        self.assertEqual(bpost.nodes, [])


# --- overlapped_forward_backward orchestration ------------------------------


class _Rec:
    """Collaborator node returning input-dependent, distinguishable markers.

    Appending a ``(kind, tag)`` pair per step lets each assertion detect a
    dropped stage, a swapped pre/post, or reordered overlap nodes.
    """

    def __init__(self, tag):
        self.tag = tag

    def forward(self, x):
        return [*x, ("fwd", self.tag)]

    def backward(self, g):
        return [*g, ("bwd", self.tag)]

    def forward_backward(self, x, g):
        return [*x, ("ovf", self.tag)], [*g, ("ovb", self.tag)]


class _Overlap:
    def __init__(self, nodes):
        self.nodes = nodes


class _LossFwd:
    def __init__(self, ret):
        self.ret = ret
        self.seen = "not-called"

    def forward(self, x):
        self.seen = x
        return self.ret


class _LossBwd:
    def __init__(self, ret):
        self.ret = ret
        self.scalers = []

    def backward(self, scaler=None):
        self.scalers.append(scaler)
        return self.ret


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestOverlappedForwardBackward(unittest.TestCase):
    """GPTModel.overlapped_forward_backward: data threading + branch logic.

    Only the node-builder collaborator ``build_overlapped_nodes`` is stubbed
    (it is verified independently above). The real orchestration body -- call
    order, forward/backward threading through pre/overlap/post, the scaler
    branch and the loss-node branches -- runs unchanged and is observed.
    The p2p_async_handle path is intentionally not exercised here (it depends
    on real ``dict_to_tuple_helper`` tensor plumbing).
    """

    def _stub_build(self):
        fpre, bpre = _Rec("fpre"), _Rec("bpre")
        fpost, bpost = _Rec("fpost"), _Rec("bpost")
        overlap = _Overlap([_Rec("ov1"), _Rec("ov2")])
        return fpre, bpre, overlap, fpost, bpost

    def test_threads_inputs_and_grads_with_scaler(self):
        model = GPTModel.__new__(GPTModel)
        built = self._stub_build()
        loss_fwd = _LossFwd(ret="LOSSVAL")
        loss_bwd = _LossBwd(ret=["BIN"])
        scaler = object()

        with mock.patch.object(
            gpt_model_mod, "build_overlapped_nodes", return_value=built
        ):
            fwd_out, loss_out, grad_out = model.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs=["FIN"],
                forward_loss_fn_node=loss_fwd,
                backward_chunk=object(),
                backward_loss_fn_node=loss_bwd,
                backward_input_grads=None,
                scaler=scaler,
                p2p_async_handle=None,
            )

        # scaler present -> backward called exactly once with scaler=scaler.
        self.assertEqual(loss_bwd.scalers, [scaler])

        # forward: pre -> ov1 -> ov2 -> post (hand-derived).
        expected_fwd = [
            "FIN",
            ("fwd", "fpre"),
            ("ovf", "ov1"),
            ("ovf", "ov2"),
            ("fwd", "fpost"),
        ]
        # backward starts from loss_bwd's return ["BIN"]: pre -> ov1 -> ov2 -> post.
        expected_grad = [
            "BIN",
            ("bwd", "bpre"),
            ("ovb", "ov1"),
            ("ovb", "ov2"),
            ("bwd", "bpost"),
        ]
        self.assertEqual(fwd_out, expected_fwd)
        self.assertEqual(grad_out, expected_grad)
        # forward_loss is computed from the fully-threaded forward output.
        self.assertEqual(loss_out, "LOSSVAL")
        self.assertEqual(loss_fwd.seen, expected_fwd)

    def test_no_scaler_calls_backward_without_scaler_and_no_forward_loss(self):
        model = GPTModel.__new__(GPTModel)
        built = self._stub_build()
        loss_bwd = _LossBwd(ret=["BIN"])

        with mock.patch.object(
            gpt_model_mod, "build_overlapped_nodes", return_value=built
        ):
            _, loss_out, _ = model.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs=["FIN"],
                forward_loss_fn_node=None,
                backward_chunk=object(),
                backward_loss_fn_node=loss_bwd,
                backward_input_grads=None,
                scaler=None,
                p2p_async_handle=None,
            )

        # falsy scaler -> backward() called with default scaler=None.
        self.assertEqual(loss_bwd.scalers, [None])
        # no forward loss node -> forward_loss is None.
        self.assertIsNone(loss_out)

    def test_passthrough_grads_when_no_backward_loss_node(self):
        model = GPTModel.__new__(GPTModel)
        built = self._stub_build()
        loss_fwd = _LossFwd(ret="L")

        with mock.patch.object(
            gpt_model_mod, "build_overlapped_nodes", return_value=built
        ):
            _, _, grad_out = model.overlapped_forward_backward(
                forward_chunk=object(),
                forward_inputs=["FIN"],
                forward_loss_fn_node=loss_fwd,
                backward_chunk=object(),
                backward_loss_fn_node=None,
                backward_input_grads=["SEED"],
                scaler=None,
                p2p_async_handle=None,
            )

        # No backward loss node: the supplied backward_input_grads flow directly
        # into the backward chain instead of being recomputed.
        self.assertEqual(
            grad_out,
            [
                "SEED",
                ("bwd", "bpre"),
                ("ovb", "ov1"),
                ("ovb", "ov2"),
                ("bwd", "bpost"),
            ],
        )


# --- fp8_quant_weight dispatch ----------------------------------------------


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestFp8QuantWeight(unittest.TestCase):
    """GPTModel.fp8_quant_weight: per-layer dispatch on both stage layouts."""

    @staticmethod
    def _transformer_layer(sink):
        tl = TransformerLayer.__new__(TransformerLayer)
        tl.fp8_quant_weight = lambda batch_mode, quant_transpose: sink.append(
            ("tl", batch_mode, quant_transpose)
        )
        return tl

    @staticmethod
    def _mtp_layer(sink):
        mtp = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
        mtp.transformer_layer = SimpleNamespace(
            fp8_quant_weight=lambda batch_mode, quant_transpose: sink.append(
                ("mtp", batch_mode, quant_transpose)
            )
        )
        return mtp

    def test_single_stage_dispatches_and_forwards_flags(self):
        sink = []
        model = GPTModel.__new__(GPTModel)
        model._num_virtual_pipeline_stages = 1
        # run_function order: TransformerLayer, unrelated layer (skipped), MTP.
        model.run_function = [
            self._transformer_layer(sink),
            object(),
            self._mtp_layer(sink),
        ]

        model.fp8_quant_weight(batch_mode=True, quant_transpose=False)

        # Both matching layers quantized, flags forwarded verbatim, in order;
        # the unrelated layer neither errors nor is quantized.
        self.assertEqual(sink, [("tl", True, False), ("mtp", True, False)])

    def test_virtual_pipeline_dispatches_across_chunks(self):
        sink = []
        model = GPTModel.__new__(GPTModel)
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [
            [self._transformer_layer(sink)],
            [self._mtp_layer(sink), object()],
        ]

        model.fp8_quant_weight(batch_mode=False, quant_transpose=True)

        self.assertEqual(sink, [("tl", False, True), ("mtp", False, True)])


if __name__ == "__main__":
    unittest.main()
