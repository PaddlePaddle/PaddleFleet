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
"""Unit tests for the plumbing helpers in ``qwen3_5_model.py``.

This file deliberately covers a slice that is *disjoint* from the two existing
Qwen3.5 tests:

* ``tests/single_card_tests/models/test_qwen3_5_hf_rmsnorm.py`` already pins the
  ``_HFBroadcastScale`` PyLayer and the ``"hf"``-target weight-gradient
  reduction, so nothing here touches that autograd path.
* ``tests/multi_card_tests/tensor_parallel/test_qwen3_5_model.py`` exercises the
  full TP+SP model loss and nothing at the helper level.

What is verified here instead:

* ``Qwen3_5RMSNorm.__init__`` -- the two calling conventions collapse into a
  single ``normalized_shape``/``variance_epsilon`` resolution, plus the
  1-centered (zero) weight initialization.
* ``Qwen3_5RMSNorm.forward`` -- the *default* (non-``hf``) branch is exactly
  ``rms_norm(x) * (1 + weight)`` in the input dtype, against a hand-derived
  NumPy reference with a non-zero weight.
* ``Qwen3_5RMSNorm.enable_sequence_parallel`` -- the real ``weight`` parameter is
  handed to ``mark_as_sequence_parallel_parameter`` (via the direct call and via
  the ``input_is_parallel`` constructor path).
* ``Qwen3_5RMSNormPipe`` -- dict passthrough, the MTP split/normalize-first/
  concat contract with distinguishable per-chunk content, and the schedule node.
* ``Qwen3_5VisionSublayersSpec`` dataclass defaults.
* ``Qwen3_5VisionModel.get_layer_desc_list`` -- the embedding -> encoder ->
  merger assembly order and the ``modal`` name-prefix branch.

There is no accelerator on the authoring host and Paddle is not importable, so
the whole module skips honestly when the import fails. On the single-card CI the
imports succeed and every assertion runs on CPU.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import paddle
except ModuleNotFoundError as exc:  # pragma: no cover - depends on host
    paddle = None
    _IMPORT_ERROR = repr(exc)
else:
    _IMPORT_ERROR = None

if paddle is not None:
    from paddle.distributed.fleet.meta_parallel import ScheduleNode

    from paddlefleet.models.qwen3_5.qwen3_5_model import (
        Qwen3_5RMSNorm,
        Qwen3_5RMSNormPipe,
        Qwen3_5VisionModel,
        Qwen3_5VisionSublayersSpec,
    )

_SKIP_REASON = f"paddle not importable on this host: {_IMPORT_ERROR}"
requires_paddle = unittest.skipUnless(paddle is not None, _SKIP_REASON)


class _Cfg:
    """Lightweight stand-in for the ``TransformerConfig`` fields these layers read.

    ``Qwen3_5RMSNorm`` only reads ``hidden_size``, ``rms_norm_eps`` and
    ``use_accuracy_compatible`` (the last via ``getattr(..., False)``);
    ``Qwen3_5RMSNormPipe`` additionally reads ``num_nextn_predict_layers`` and
    ``mtp_load_weight_only``. A real config would work too but drags in Fleet
    initialization that is irrelevant to the plumbing under test.
    """

    def __init__(
        self,
        hidden_size=8,
        rms_norm_eps=1e-6,
        use_accuracy_compatible=False,
        num_nextn_predict_layers=None,
        mtp_load_weight_only=False,
    ):
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.use_accuracy_compatible = use_accuracy_compatible
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_load_weight_only = mtp_load_weight_only


def _rms_norm_ref(x_np, weight_np, eps):
    """Independent NumPy RMSNorm with 1-centered scale (float32 accumulation)."""
    xf = x_np.astype(np.float64)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    normed = xf / np.sqrt(var + eps)
    return normed * (1.0 + weight_np.astype(np.float64))


@requires_paddle
class TestQwen3_5RMSNormInit(unittest.TestCase):
    """The constructor merges two calling conventions into one shape/eps."""

    def test_hidden_size_wins_over_normalized_shape(self):
        # Both provided: ``hidden_size`` takes precedence (it is checked first).
        norm = Qwen3_5RMSNorm(
            _Cfg(hidden_size=999),
            hidden_size=32,
            normalized_shape=64,
        )
        self.assertEqual(norm.normalized_shape, 32)

    def test_normalized_shape_used_when_no_hidden_size(self):
        norm = Qwen3_5RMSNorm(_Cfg(hidden_size=999), normalized_shape=17)
        self.assertEqual(norm.normalized_shape, 17)

    def test_falls_back_to_config_hidden_size(self):
        norm = Qwen3_5RMSNorm(_Cfg(hidden_size=48))
        self.assertEqual(norm.normalized_shape, 48)

    def test_eps_precedence_over_norm_eps_and_config(self):
        # ``eps`` beats ``norm_eps`` beats ``config.rms_norm_eps``.
        norm = Qwen3_5RMSNorm(
            _Cfg(rms_norm_eps=1e-3),
            hidden_size=8,
            eps=1e-7,
            norm_eps=1e-5,
        )
        self.assertEqual(norm.variance_epsilon, 1e-7)

    def test_norm_eps_used_when_eps_none(self):
        norm = Qwen3_5RMSNorm(
            _Cfg(rms_norm_eps=1e-3), hidden_size=8, norm_eps=1e-5
        )
        self.assertEqual(norm.variance_epsilon, 1e-5)

    def test_config_rms_norm_eps_is_the_last_resort(self):
        norm = Qwen3_5RMSNorm(_Cfg(rms_norm_eps=1e-6), hidden_size=8)
        self.assertEqual(norm.variance_epsilon, 1e-6)

    def test_weight_shape_and_one_centered_zero_init(self):
        # 1-centered parameterization: the learnable weight starts at all zeros
        # with exactly ``[normalized_shape]`` elements.
        norm = Qwen3_5RMSNorm(_Cfg(), hidden_size=6)
        self.assertEqual(list(norm.weight.shape), [6])
        np.testing.assert_array_equal(
            norm.weight.numpy(), np.zeros([6], dtype=norm.weight.numpy().dtype)
        )


@requires_paddle
class TestQwen3_5RMSNormDefaultForward(unittest.TestCase):
    """The default (non-``hf``) branch: ``rms_norm(x) * (1 + weight)``."""

    def test_default_branch_matches_hand_derived_reference(self):
        # use_accuracy_compatible=False -> targets_hf(...) is False, so the
        # plain expression branch runs regardless of weight.stop_gradient.
        norm = Qwen3_5RMSNorm(
            _Cfg(use_accuracy_compatible=False), hidden_size=4
        )
        # A non-zero, sign-varied weight so the ``(1 + weight)`` scale actually
        # bites (a zero weight would hide a dropped scale term).
        weight_np = np.array([0.5, -0.25, 2.0, -1.0], dtype=np.float32)
        norm.weight.set_value(
            paddle.to_tensor(weight_np, dtype=norm.weight.dtype)
        )
        x_np = np.array(
            [[[1.0, -2.0, 3.0, -4.0], [0.5, 0.5, -0.5, -0.5]]],
            dtype=np.float32,
        )
        out = norm(paddle.to_tensor(x_np))
        expected = _rms_norm_ref(x_np, weight_np, norm.variance_epsilon)
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-5, atol=1e-6
        )

    def test_output_dtype_follows_the_input(self):
        norm = Qwen3_5RMSNorm(_Cfg(), hidden_size=8)
        x = paddle.randn([2, 3, 8], dtype="bfloat16")
        self.assertEqual(norm(x).dtype, paddle.bfloat16)


@requires_paddle
class TestQwen3_5RMSNormSequenceParallel(unittest.TestCase):
    """``enable_sequence_parallel`` marks the real ``weight`` parameter."""

    def test_direct_call_marks_the_weight_parameter(self):
        # Build without the parallel flag so no marking happens yet, then patch
        # the collaborator and drive the real method.
        norm = Qwen3_5RMSNorm(_Cfg(), hidden_size=8, input_is_parallel=False)
        captured = {}

        def fake_mark(param):
            captured["param"] = param

        with mock.patch(
            "paddlefleet.models.qwen3_5.qwen3_5_model."
            "mark_as_sequence_parallel_parameter",
            side_effect=fake_mark,
        ) as marker:
            norm.enable_sequence_parallel()

        marker.assert_called_once()
        # The exact ``weight`` object must be handed over, not a copy.
        self.assertIs(captured["param"], norm.weight)

    def test_input_is_parallel_marks_at_construction(self):
        captured = {}

        def fake_mark(param):
            captured["param"] = param

        with mock.patch(
            "paddlefleet.models.qwen3_5.qwen3_5_model."
            "mark_as_sequence_parallel_parameter",
            side_effect=fake_mark,
        ) as marker:
            norm = Qwen3_5RMSNorm(_Cfg(), hidden_size=8, input_is_parallel=True)

        marker.assert_called_once()
        self.assertIs(captured["param"], norm.weight)


@requires_paddle
class TestQwen3_5RMSNormPipe(unittest.TestCase):
    """The pipeline wrapper: dict I/O and MTP split/normalize-first/concat."""

    def test_init_builds_inner_norm_with_dim_and_eps(self):
        pipe = Qwen3_5RMSNormPipe(
            _Cfg(rms_norm_eps=1e-6), hidden_size=8, eps=2e-5
        )
        self.assertIsInstance(pipe.norm, Qwen3_5RMSNorm)
        self.assertEqual(pipe.norm.normalized_shape, 8)
        # The explicit ``eps`` argument, not ``config.rms_norm_eps``, is used.
        self.assertEqual(pipe.norm.variance_epsilon, 2e-5)

    def test_forward_passthrough_and_norm_without_mtp(self):
        # num_nextn_predict_layers=None -> neither split nor concat runs.
        cfg = _Cfg(num_nextn_predict_layers=None)
        pipe = Qwen3_5RMSNormPipe(cfg, hidden_size=4, eps=1e-6)
        x = paddle.to_tensor(
            np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        )
        pos = paddle.to_tensor([[0, 1, 2]])
        out = pipe({"hidden_states": x, "position_ids": pos})
        # Non-target fields must survive untouched.
        self.assertIs(out["position_ids"], pos)
        # hidden_states must be exactly the inner norm applied to the whole x.
        np.testing.assert_allclose(
            out["hidden_states"].numpy(), pipe.norm(x).numpy(), rtol=0, atol=0
        )

    def test_forward_mtp_normalizes_first_chunk_then_concats(self):
        # num_nextn_predict_layers=2 -> split into 3 chunks along axis 0,
        # normalize ONLY chunk 0, then concat back in original order.
        cfg = _Cfg(num_nextn_predict_layers=2, mtp_load_weight_only=False)
        pipe = Qwen3_5RMSNormPipe(cfg, hidden_size=4, eps=1e-6)
        # Distinguishable, non-degenerate per-chunk content along dim 0.
        chunk0 = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
        chunk1 = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        chunk2 = np.array([[-5.0, -6.0, -7.0, -8.0]], dtype=np.float32)
        x_np = np.stack([chunk0, chunk1, chunk2], axis=0)  # [3, 1, 4]
        out = pipe({"hidden_states": paddle.to_tensor(x_np)})["hidden_states"]
        got = out.numpy()
        self.assertEqual(list(got.shape), [3, 1, 4])

        weight_np = pipe.norm.weight.numpy()  # zeros at init
        expected_c0 = _rms_norm_ref(
            chunk0, weight_np, pipe.norm.variance_epsilon
        )
        # Chunk 0 is normalized...
        np.testing.assert_allclose(
            got[0].astype(np.float64), expected_c0, rtol=1e-5, atol=1e-6
        )
        # ...while the trailing MTP chunks pass through byte-for-byte, in order.
        np.testing.assert_array_equal(got[1], chunk1)
        np.testing.assert_array_equal(got[2], chunk2)

    def test_mtp_disabled_by_load_weight_only_flag(self):
        # Same shape, but mtp_load_weight_only=True must skip split/concat and
        # normalize the WHOLE tensor instead of only the first chunk.
        cfg = _Cfg(num_nextn_predict_layers=2, mtp_load_weight_only=True)
        pipe = Qwen3_5RMSNormPipe(cfg, hidden_size=4, eps=1e-6)
        x_np = np.arange(12, dtype=np.float32).reshape([3, 1, 4]) + 1.0
        x = paddle.to_tensor(x_np)
        out = pipe({"hidden_states": x})["hidden_states"]
        np.testing.assert_allclose(
            out.numpy(), pipe.norm(x).numpy(), rtol=0, atol=0
        )

    def test_build_schedule_node_wraps_forward_with_name(self):
        pipe = Qwen3_5RMSNormPipe(_Cfg(), hidden_size=8, eps=1e-6)
        node = pipe.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)
        self.assertEqual(node.name, "Qwen3_5RMSNormPipe")


@requires_paddle
class TestQwen3_5VisionSublayersSpec(unittest.TestCase):
    """The spec dataclass defaults every LayerSpec slot to ``None``."""

    def test_defaults_all_none(self):
        spec = Qwen3_5VisionSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.merger)

    def test_fields_are_assignable_by_keyword(self):
        emb, mid, mrg = object(), [object()], object()
        spec = Qwen3_5VisionSublayersSpec(
            embedding=emb, transformer_layers=mid, merger=mrg
        )
        self.assertIs(spec.embedding, emb)
        self.assertIs(spec.transformer_layers, mid)
        self.assertIs(spec.merger, mrg)
        # Unset slots stay ``None``.
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.tail_empty_layers)


@requires_paddle
class TestQwen3_5VisionModelLayerDescList(unittest.TestCase):
    """``get_layer_desc_list`` assembles embedding -> encoder -> merger.

    ``LayerDesc``/``add_sequential_layer``/``get_encoder_layer_desc_list`` are
    genuine collaborators here (their own correctness is covered elsewhere); we
    replace them with recorders so the test observes the ORDER and the
    name-prefix argument the method under test actually produces, without
    needing real ``LayerSpec`` instances.
    """

    def _run(self, modal):
        emb, mrg = object(), object()
        spec = Qwen3_5VisionSublayersSpec(
            embedding=emb, transformer_layers=[object()], merger=mrg
        )
        model = Qwen3_5VisionModel.__new__(Qwen3_5VisionModel)
        model.modal = modal

        events = []

        def fake_layer_desc(wrapped):
            return ("LD", wrapped)

        def fake_add(layers, layer_desc, name):
            events.append(("add", name, layer_desc))

        def fake_encoder(layers, spec_arg, name):
            events.append(("encoder", name, spec_arg))

        model.add_sequential_layer = fake_add
        model.get_encoder_layer_desc_list = fake_encoder

        with mock.patch(
            "paddlefleet.models.qwen3_5.qwen3_5_model.LayerDesc",
            side_effect=fake_layer_desc,
        ):
            result = model.get_layer_desc_list(spec)

        return events, result, spec, emb, mrg

    def test_order_and_names_without_modal(self):
        events, result, spec, emb, mrg = self._run(modal=None)
        # Exactly three orchestration steps, in this order.
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0], ("add", "model", ("LD", emb)))
        self.assertEqual(events[1], ("encoder", "model", spec))
        self.assertEqual(events[2], ("add", "model.merger", ("LD", mrg)))
        # The collaborators were stubbed, so the accumulator is returned as-is.
        self.assertIsInstance(result, list)

    def test_name_prefix_uses_modal(self):
        events, _result, spec, emb, mrg = self._run(modal="vision")
        self.assertEqual(events[0], ("add", "model.vision", ("LD", emb)))
        self.assertEqual(events[1], ("encoder", "model.vision", spec))
        self.assertEqual(events[2], ("add", "model.vision.merger", ("LD", mrg)))


if __name__ == "__main__":
    unittest.main()
