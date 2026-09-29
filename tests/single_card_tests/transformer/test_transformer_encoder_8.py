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

"""Behaviour tests for the pipeline-name/state-dict remapping and the fp8
helper methods of
``paddlefleet.transformer.transformer_encoder.TransformerEncoder``.

The methods exercised here -- ``_set_pipeline_name_mapping``,
``set_state_dict``, ``state_dict``, ``_check_shared_model_state``,
``fp8_quant_weight`` and ``use_fp8`` -- are pure Python orchestration over
dictionaries and ``run_function`` / ``_model_chunks`` layer lists. They are
safe to run on CPU without an accelerator and do not need the expensive
``PipelineLayer.__init__`` (which requires a Fleet hybrid-communicate group).

A ``_Encoder`` subclass therefore skips that base initialiser but keeps every
method under test as the *real* production implementation. The only isolated
collaborator is the base-class persistence layer (``PipelineLayer.state_dict``
/ ``PipelineLayer.set_state_dict``, i.e. the ``super()`` calls), which is the
raw data source / sink for the remapping under test; it is replaced by a
controlled stand-in so the real remapping algorithm can be observed end to end
(what keys/values it forwards, in which direction). No production logic is
re-implemented in the test.

Expected values are derived by hand from the production source. This is a
single-process, CPU-only orchestration test; it makes no claim about
pipeline-parallel communication or numerics.
"""

import unittest
from unittest.mock import patch

try:
    from paddle.distributed.fleet.meta_parallel import PipelineLayer

    from paddlefleet.transformer.transformer_encoder import TransformerEncoder
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    class _Config:
        # ``state_dict`` / ``sharded_state_dict`` read ``config.model_type``;
        # a non-"qwen3_vl" value selects the empty name_prefix branch.
        model_type = "gpt"

    class _Encoder(TransformerEncoder):
        """Skip ``PipelineLayer.__init__`` while keeping the real methods.

        Only the attributes the methods under test actually read are
        populated here; individual tests further set ``run_function``,
        ``_model_chunks``, ``_sequential_layers`` and the two mapping dicts.
        """

        def __init__(self):
            self.config = _Config()
            self._pipeline_name_mapping = None
            self._pp_to_single_mapping = None
            self._num_virtual_pipeline_stages = 1

    class _RecordingLayer(TransformerLayer):
        """A genuine ``TransformerLayer`` (real class in the MRO) whose heavy
        ``__init__`` is intentionally bypassed. ``fp8_quant_weight`` records
        the exact arguments it is forwarded and ``use_fp8`` returns a fixed,
        controllable answer, so both the ``isinstance`` gate and the argument
        forwarding can be observed rather than merely counted."""

        def __init__(self, fp8=False):
            self._fp8 = fp8
            self.fp8_calls = []

        def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
            self.fp8_calls.append((batch_mode, quant_transpose))

        def use_fp8(self):
            return self._fp8

    class _NotALayer:
        """Not a ``TransformerLayer``. Its fp8 hooks MUST never be consulted
        because of the ``isinstance`` gate; ``use_fp8`` deliberately claims
        True to prove a non-layer cannot flip the encoder's answer."""

        def __init__(self):
            self.fp8_calls = []

        def fp8_quant_weight(self, *args, **kwargs):
            self.fp8_calls.append((args, kwargs))

        def use_fp8(self):
            return True

    class _Param:
        """Opaque state-dict value carrier. The remapping logic treats values
        as opaque and may set a ``.key`` attribute on them; a plain holder is
        sufficient and avoids depending on tensor internals."""

        def __init__(self, tag):
            self.tag = tag

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    PipelineLayer = None
    TransformerEncoder = None
    TransformerLayer = None
    _Config = None
    _Encoder = None
    _RecordingLayer = None
    _NotALayer = None
    _Param = None
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSetPipelineNameMapping(unittest.TestCase):
    def test_explicit_mapping_is_stored_and_returned(self):
        enc = _Encoder()
        explicit = {"model.embed.weight": "0.weight"}

        result = enc._set_pipeline_name_mapping(explicit)

        # Explicit-arg branch: the mapping is adopted verbatim and returned;
        # the pp->single mapping is left untouched (None) on this path.
        self.assertIs(result, explicit)
        self.assertIs(enc._pipeline_name_mapping, explicit)
        self.assertIsNone(enc._pp_to_single_mapping)

    def test_computed_mapping_substitutes_prefixes_per_index(self):
        enc = _Encoder()
        # Prefix table consumed via get_sequential_name_prefixes():
        #   index "0" -> "model", "1" -> "model.layers.0", "2" -> "model".
        enc._sequential_layers = [
            {"name_prefix": "model"},
            {"name_prefix": "model.layers.0"},
            {"name_prefix": "model"},
        ]
        # Raw pipeline-stage parameter names produced by the base layer.
        raw = {
            "0.embed.weight": _Param("a"),
            "1.attn.weight": _Param("b"),
            "2.weight": _Param("c"),
            "extra.weight": _Param("d"),  # non-digit head -> pass-through
        }

        def fake_state_dict(self, *args, **kwargs):
            return dict(raw)

        with patch.object(PipelineLayer, "state_dict", fake_state_dict):
            returned = enc._set_pipeline_name_mapping()

        # first_key "0.embed.weight" -> head "0" digit, second "embed" not
        # digit => the non-virtual-pp branch. Each digit index is replaced by
        # its prefix; a non-digit head is mapped to itself.
        expected_single_to_pp = {
            "model.embed.weight": "0.embed.weight",
            "model.layers.0.attn.weight": "1.attn.weight",
            "model.weight": "2.weight",
            "extra.weight": "extra.weight",
        }
        expected_pp_to_single = {
            "0.embed.weight": "model.embed.weight",
            "1.attn.weight": "model.layers.0.attn.weight",
            "2.weight": "model.weight",
            "extra.weight": "extra.weight",
        }
        self.assertEqual(returned, expected_single_to_pp)
        self.assertEqual(enc._pipeline_name_mapping, expected_single_to_pp)
        self.assertEqual(enc._pp_to_single_mapping, expected_pp_to_single)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSetStateDict(unittest.TestCase):
    def test_remaps_known_keys_and_drops_unknown(self):
        enc = _Encoder()
        enc._pipeline_name_mapping = {
            "model.embed.weight": "0.weight",
            "model.norm.weight": "5.weight",
        }
        a, b, c = object(), object(), object()
        incoming = {
            "model.embed.weight": a,
            "model.norm.weight": b,
            "unknown.weight": c,  # not in mapping -> silently dropped
        }
        captured = {}
        sentinel = object()

        def fake_super(self, state_dict, *args, **kwargs):
            captured["sd"] = dict(state_dict)
            return sentinel

        with patch.object(PipelineLayer, "set_state_dict", fake_super):
            ret = enc.set_state_dict(incoming)

        # Return value of the base loader is propagated unchanged.
        self.assertIs(ret, sentinel)
        forwarded = captured["sd"]
        # Keys are remapped single->pipeline; the unmapped key is removed
        # entirely (not passed through), and values keep their identity.
        self.assertEqual(set(forwarded), {"0.weight", "5.weight"})
        self.assertIs(forwarded["0.weight"], a)
        self.assertIs(forwarded["5.weight"], b)
        self.assertNotIn(c, forwarded.values())

    def test_empty_mapping_raises_assertion(self):
        enc = _Encoder()
        enc._pipeline_name_mapping = {}  # non-None but empty -> guard fires

        with self.assertRaises(AssertionError):
            enc.set_state_dict({"model.embed.weight": object()})


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestStateDict(unittest.TestCase):
    def test_remaps_pp_to_single_and_passes_through_unmapped(self):
        enc = _Encoder()
        # Non-None so state_dict() does not recompute the mapping itself.
        enc._pipeline_name_mapping = {"model.embed.weight": "0.weight"}
        enc._pp_to_single_mapping = {
            "0.weight": "model.embed.weight",
            "1.weight": "model.attn.weight",
        }
        pa, pb, pc = _Param("a"), _Param("b"), _Param("c")

        def fake_super(self, *args, **kwargs):
            return {"0.weight": pa, "1.weight": pb, "extra": pc}

        with patch.object(PipelineLayer, "state_dict", fake_super):
            out = enc.state_dict()

        # pipeline->single remap for known keys; unmapped "extra" survives
        # under its own name. Values keep identity and mapped ones get .key.
        self.assertEqual(
            set(out), {"model.embed.weight", "model.attn.weight", "extra"}
        )
        self.assertIs(out["model.embed.weight"], pa)
        self.assertIs(out["model.attn.weight"], pb)
        self.assertIs(out["extra"], pc)
        self.assertEqual(pa.key, "model.embed.weight")
        self.assertEqual(pb.key, "model.attn.weight")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCheckSharedModelState(unittest.TestCase):
    def test_no_missing_when_mappings_are_bijective(self):
        enc = _Encoder()
        enc._pp_to_single_mapping = {
            "0.w": "model.embed",
            "1.w": "model.attn",
        }
        enc._pipeline_name_mapping = {
            "model.embed": "0.w",
            "model.attn": "1.w",
        }
        va, vb = object(), object()

        def fake_super(self, *args, **kwargs):
            return {"0.w": va, "1.w": vb}

        with patch.object(PipelineLayer, "state_dict", fake_super):
            missing = enc._check_shared_model_state()

        self.assertEqual(missing, {})

    def test_tied_weight_reports_noncanonical_pipeline_key(self):
        enc = _Encoder()
        # Two pipeline keys share one single-name (a tied weight across
        # stages); the single-name can only point back to one canonical
        # pipeline key.
        enc._pp_to_single_mapping = {
            "0.weight": "shared",
            "5.weight": "shared",
            "2.w": "model.mid",
        }
        enc._pipeline_name_mapping = {"shared": "0.weight", "model.mid": "2.w"}
        shared_v, mid_v = object(), object()

        def fake_super(self, *args, **kwargs):
            # Both tied keys must reference the SAME tensor object to pass
            # the identity guard.
            return {"0.weight": shared_v, "5.weight": shared_v, "2.w": mid_v}

        with patch.object(PipelineLayer, "state_dict", fake_super):
            missing = enc._check_shared_model_state()

        # The non-canonical pipeline key of the shared tensor is reported,
        # pointing at the canonical one; consistent keys are absent.
        self.assertEqual(missing, {"5.weight": "0.weight"})

    def test_shared_tensor_identity_guard_raises(self):
        enc = _Encoder()
        enc._pp_to_single_mapping = {"0.weight": "shared", "5.weight": "shared"}
        enc._pipeline_name_mapping = {"shared": "0.weight"}
        va, vb = object(), object()  # DIFFERENT objects for the same name

        def fake_super(self, *args, **kwargs):
            return {"0.weight": va, "5.weight": vb}

        with (
            patch.object(PipelineLayer, "state_dict", fake_super),
            self.assertRaises(AssertionError),
        ):
            enc._check_shared_model_state()


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestFp8QuantWeight(unittest.TestCase):
    def test_single_stage_forwards_args_only_to_transformer_layers(self):
        enc = _Encoder()
        enc._num_virtual_pipeline_stages = 1
        tl0 = _RecordingLayer()
        plain = _NotALayer()
        tl1 = _RecordingLayer()
        enc.run_function = [tl0, plain, tl1]

        enc.fp8_quant_weight(batch_mode=True, quant_transpose=False)

        # Every TransformerLayer receives the exact kwargs, in order; the
        # non-layer is skipped by the isinstance gate (never invoked).
        self.assertEqual(tl0.fp8_calls, [(True, False)])
        self.assertEqual(tl1.fp8_calls, [(True, False)])
        self.assertEqual(plain.fp8_calls, [])

    def test_single_stage_forwards_default_args(self):
        enc = _Encoder()
        enc._num_virtual_pipeline_stages = 1
        tl0 = _RecordingLayer()
        enc.run_function = [tl0]

        enc.fp8_quant_weight()

        self.assertEqual(tl0.fp8_calls, [(False, True)])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestUseFp8(unittest.TestCase):
    def test_single_stage_false_when_no_layer_uses_fp8(self):
        enc = _Encoder()
        enc._num_virtual_pipeline_stages = 1
        enc.run_function = [_RecordingLayer(fp8=False)]

        self.assertIs(enc.use_fp8(), False)

    def test_single_stage_true_when_any_layer_uses_fp8(self):
        enc = _Encoder()
        enc._num_virtual_pipeline_stages = 1
        enc.run_function = [
            _RecordingLayer(fp8=False),
            _RecordingLayer(fp8=True),
        ]

        self.assertIs(enc.use_fp8(), True)

    def test_single_stage_ignores_non_layer_claiming_fp8(self):
        enc = _Encoder()
        enc._num_virtual_pipeline_stages = 1
        # _NotALayer.use_fp8() returns True, but the isinstance gate must
        # keep it from flipping the answer.
        enc.run_function = [_NotALayer()]

        self.assertIs(enc.use_fp8(), False)

    def test_vpp_true_when_a_chunk_layer_uses_fp8(self):
        enc = _Encoder()
        enc._num_virtual_pipeline_stages = 2
        enc._model_chunks = [
            [_RecordingLayer(fp8=False)],
            [_RecordingLayer(fp8=True)],
        ]

        self.assertIs(enc.use_fp8(), True)

    @unittest.expectedFailure
    def test_vpp_false_when_no_layer_uses_fp8(self):
        # PRODUCTION BUG (transformer_encoder.py:569-579): in the virtual-pp
        # branch use_fp8() lacks a trailing ``return False``, so when no layer
        # uses fp8 it falls through and returns None instead of False -- unlike
        # the single-stage branch which correctly returns False. The correct
        # contract is asserted here; this expected-failure documents the bug
        # without editing production.
        enc = _Encoder()
        enc._num_virtual_pipeline_stages = 2
        enc._model_chunks = [[_RecordingLayer(fp8=False)]]

        self.assertIs(enc.use_fp8(), False)


if __name__ == "__main__":
    unittest.main()
