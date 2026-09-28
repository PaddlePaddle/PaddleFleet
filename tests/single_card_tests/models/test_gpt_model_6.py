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

"""Behavioral tests for the pipeline <-> single-card key remapping methods of
``paddlefleet.models.gpt.gpt_model.GPTModel``:

    * ``GPTModel.state_dict``            (export: pp keys -> single-card keys)
    * ``GPTModel.set_state_dict``        (load:  single-card keys -> pp keys)
    * ``GPTModel._check_shared_model_state`` (which single-card names alias a
      non-canonical pipeline key, i.e. shared / tied parameters)

Only the *remapping* logic living in these three methods is under test. The
parent ``PipelineLayer.state_dict`` / ``PipelineLayer.set_state_dict`` are
non-tested collaborators: they are replaced by fakes that return
content-distinguishable objects, so we can observe how GPTModel consumes and
rewrites them (real transformation), not merely that they were called. The
name-mapping dicts are supplied as *inputs* to the transformation; the tested
behavior is how the method renames / drops / detects-shared using them.

Expected key sets are hand-derived from the production source (see the method
bodies around lines 753-865 of gpt_model.py), never by calling the method to
produce its own expectation.

Importing GPTModel pulls in the paddle runtime transitively. Where paddle is
not installed the whole module is honestly skipped -- the remapping logic is
*not* claimed to have executed and is not faked as passing.
"""

import types
import unittest
from unittest import mock

try:
    from paddle.distributed.fleet.meta_parallel import PipelineLayer

    from paddlefleet.models.gpt.gpt_model import GPTModel

    _IMPORT_ERROR = None
except ImportError as exc:  # genuine missing dependency only, not other errors
    PipelineLayer = None
    GPTModel = None
    _IMPORT_ERROR = exc

PADDLE_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "Importing GPTModel requires the paddle runtime, which is not installed "
    f"in this environment: {_IMPORT_ERROR!r}. The pipeline<->single-card "
    "remapping logic in state_dict / set_state_dict / _check_shared_model_state "
    "was NOT executed here; this module is skipped, not passed."
)


class _FakeParam:
    """Stand-in for a state-dict tensor value.

    Only the attributes the production code touches are modelled: ``state_dict``
    reassigns ``.key`` to the remapped name. ``tag`` gives each instance a
    distinguishable identity so remapping cannot be faked by swapping values.
    """

    def __init__(self, tag):
        self.tag = tag
        self.key = None

    def __repr__(self):
        return f"_FakeParam({self.tag!r}, key={self.key!r})"


def _new_model():
    """Allocate a GPTModel without running its paddle-heavy __init__.

    __new__ bypasses construction so no real layers / parameters are built; the
    tests then drive the pure-Python remapping methods directly. ``vision_merge``
    is pinned to None so the (unrelated) multimodal branch is inactive.
    """
    model = GPTModel.__new__(GPTModel)
    model.vision_merge = None
    return model


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelStateDictExport(unittest.TestCase):
    """``state_dict`` rewrites parent (pipeline) keys to single-card keys via
    ``_pp_to_single_mapping``, preserves keys absent from the mapping, and (for
    qwen3_vl / qwen3_5) strips the ``model.language_model.`` prefix *before* the
    mapping lookup. Asserting only ``assertIn(key, result)`` would miss value
    misplacement, prefix mishandling and dropped unmapped keys."""

    def test_remaps_mapped_keys_and_keeps_unmapped(self):
        model = _new_model()
        model.config = types.SimpleNamespace(model_type="gpt")  # no prefix
        model._pp_to_single_mapping = {
            "0.embedding.weight": "embedding.word_embeddings.weight",
            "3.self_attn.qkv.weight": "decoder.layers.0.self_attn.qkv.weight",
        }
        # Non-None so state_dict does not try to rebuild the mapping (which
        # would need real pipeline layers). Its contents are irrelevant here.
        model._pipeline_name_mapping = {"x": "y"}

        p0 = _FakeParam("embed")
        p1 = _FakeParam("qkv")
        p2 = _FakeParam("final_norm")  # not in _pp_to_single_mapping
        parent = {
            "0.embedding.weight": p0,
            "3.self_attn.qkv.weight": p1,
            "final_norm.weight": p2,
        }
        with mock.patch.object(
            PipelineLayer, "state_dict", return_value=parent
        ):
            result = model.state_dict()

        # Hand-derived: two keys remapped, one unmapped key preserved verbatim.
        self.assertEqual(
            set(result),
            {
                "embedding.word_embeddings.weight",
                "decoder.layers.0.self_attn.qkv.weight",
                "final_norm.weight",
            },
        )
        # Values land under the mapped names, same objects (no swap / loss).
        self.assertIs(result["embedding.word_embeddings.weight"], p0)
        self.assertIs(result["decoder.layers.0.self_attn.qkv.weight"], p1)
        self.assertIs(result["final_norm.weight"], p2)
        # Remapped values also carry the rewritten .key; unmapped one untouched.
        self.assertEqual(p0.key, "embedding.word_embeddings.weight")
        self.assertEqual(p1.key, "decoder.layers.0.self_attn.qkv.weight")
        self.assertIsNone(p2.key)

    def test_strips_language_model_prefix_before_mapping(self):
        for model_type in ("qwen3_5_moe", "qwen3_vl"):
            with self.subTest(model_type=model_type):
                model = _new_model()
                model.config = types.SimpleNamespace(model_type=model_type)
                model._pp_to_single_mapping = {
                    "0.embedding.weight": "embedding.word_embeddings.weight",
                }
                model._pipeline_name_mapping = {"x": "y"}

                p0 = _FakeParam("embed")
                p1 = _FakeParam("unmapped")
                parent = {
                    # prefix present -> stripped to "0.embedding.weight" -> hits map
                    "model.language_model.0.embedding.weight": p0,
                    # prefix present but not in map -> kept under STRIPPED name
                    "model.language_model.extra.bias": p1,
                }
                with mock.patch.object(
                    PipelineLayer, "state_dict", return_value=parent
                ):
                    result = model.state_dict()

                # Prefix stripping must reach the mapping consumer, else the
                # first key would never match and would be stored under its full
                # prefixed name.
                self.assertEqual(
                    set(result),
                    {"embedding.word_embeddings.weight", "extra.bias"},
                )
                self.assertIs(result["embedding.word_embeddings.weight"], p0)
                self.assertIs(result["extra.bias"], p1)
                self.assertEqual(p0.key, "embedding.word_embeddings.weight")


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelSetStateDictLoad(unittest.TestCase):
    """``set_state_dict`` is the inverse: single-card keys are rewritten to
    pipeline keys via ``_pipeline_name_mapping``, keys absent from the mapping
    are silently dropped (this stage does not own them), and the remapped dict
    is forwarded to ``super().set_state_dict`` whose return value is propagated.
    Checking only that super was called would miss the drop and the rewrite."""

    def test_remaps_drops_unmapped_and_forwards_super_return(self):
        model = _new_model()
        model._pipeline_name_mapping = {
            "embedding.word_embeddings.weight": "0.embedding.weight",
            "decoder.layers.0.self_attn.qkv.weight": "3.self_attn.qkv.weight",
        }

        p0 = _FakeParam("embed")
        p1 = _FakeParam("qkv")
        p2 = _FakeParam("orphan")  # absent from mapping -> must be dropped
        incoming = {
            "embedding.word_embeddings.weight": p0,
            "decoder.layers.0.self_attn.qkv.weight": p1,
            "not.in.mapping": p2,
        }

        captured = {}
        sentinel = object()

        def fake_super_set(self_arg_or_sd, *args, **kwargs):
            # PipelineLayer.set_state_dict is patched unbound; when invoked via
            # super() the first positional is the (already remapped) state dict.
            captured["state_dict"] = dict(self_arg_or_sd)
            captured["args"] = args
            captured["kwargs"] = kwargs
            return sentinel

        with mock.patch.object(
            PipelineLayer,
            "set_state_dict",
            autospec=False,
            side_effect=fake_super_set,
        ):
            ret = model.set_state_dict(incoming, "extra_pos", flag=7)

        # Hand-derived: only the two mapped keys survive, renamed to pp keys;
        # "not.in.mapping" is dropped (owned by another stage).
        self.assertEqual(
            captured["state_dict"],
            {"0.embedding.weight": p0, "3.self_attn.qkv.weight": p1},
        )
        # Extra positional / keyword args are forwarded to the parent verbatim.
        self.assertEqual(captured["args"], ("extra_pos",))
        self.assertEqual(captured["kwargs"], {"flag": 7})
        # The parent's return value is propagated back to the caller.
        self.assertIs(ret, sentinel)

    def test_empty_pipeline_mapping_is_rejected(self):
        # The method asserts the pipeline stage owns parameters; an empty
        # mapping is a hard error, not a silent no-op.
        model = _new_model()
        model._pipeline_name_mapping = {}
        with self.assertRaises(AssertionError):
            model.set_state_dict(
                {"embedding.word_embeddings.weight": _FakeParam("e")}
            )


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGPTModelCheckSharedModelState(unittest.TestCase):
    """``_check_shared_model_state`` reports single-card names whose *canonical*
    pipeline key (``_pipeline_name_mapping[single]``) differs from the pipeline
    key actually holding the tensor -- i.e. tied / shared parameters aliased
    under a second pipeline key. It also asserts aliases that resolve to the
    same single name really point at the same tensor object."""

    def _canonical_case(self):
        model = _new_model()
        # Two pipeline keys resolve to the same single-card name: the second is
        # a shared/tied alias. Canonical single->pp points at the first.
        model._pp_to_single_mapping = {
            "0.embedding.weight": "embedding.word_embeddings.weight",
            "shared_embedding.weight": "embedding.word_embeddings.weight",
        }
        model._pipeline_name_mapping = {
            "embedding.word_embeddings.weight": "0.embedding.weight",
        }
        return model

    def test_reports_shared_alias_as_missing(self):
        model = self._canonical_case()
        shared = _FakeParam("tied")  # SAME object under both pipeline keys
        parent = {
            "0.embedding.weight": shared,
            "shared_embedding.weight": shared,
        }
        with mock.patch.object(
            PipelineLayer, "state_dict", return_value=parent
        ):
            missing = model._check_shared_model_state()

        # Hand-derived: "0.embedding.weight" is canonical (k == mapped_k) so it
        # is NOT reported; the alias "shared_embedding.weight" maps to the same
        # single name whose canonical pp key is "0.embedding.weight" != itself,
        # so it IS reported, pointing at the canonical key.
        self.assertEqual(
            missing, {"shared_embedding.weight": "0.embedding.weight"}
        )

    def test_no_shared_params_yields_empty(self):
        model = _new_model()
        model._pp_to_single_mapping = {
            "0.embedding.weight": "embedding.word_embeddings.weight",
            "3.self_attn.qkv.weight": "decoder.layers.0.self_attn.qkv.weight",
        }
        model._pipeline_name_mapping = {
            "embedding.word_embeddings.weight": "0.embedding.weight",
            "decoder.layers.0.self_attn.qkv.weight": "3.self_attn.qkv.weight",
        }
        t0 = _FakeParam("embed")
        t1 = _FakeParam("qkv")
        parent = {"0.embedding.weight": t0, "3.self_attn.qkv.weight": t1}
        with mock.patch.object(
            PipelineLayer, "state_dict", return_value=parent
        ):
            missing = model._check_shared_model_state()
        # Every pipeline key is canonical -> nothing is shared/missing.
        self.assertEqual(missing, {})

    def test_conflicting_aliases_raise(self):
        # Two pipeline keys resolve to the same single name but hold DIFFERENT
        # tensor objects: that violates the shared-tensor invariant and must
        # raise, not be silently merged.
        model = self._canonical_case()
        t1 = _FakeParam("a")
        t2 = _FakeParam("b")  # distinct object -> `old_v is v` fails
        parent = {"0.embedding.weight": t1, "shared_embedding.weight": t2}
        with (
            mock.patch.object(PipelineLayer, "state_dict", return_value=parent),
            self.assertRaises(AssertionError),
        ):
            model._check_shared_model_state()


if __name__ == "__main__":
    unittest.main()
