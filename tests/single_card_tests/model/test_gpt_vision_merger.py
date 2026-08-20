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
"""Tests for ``vision_merge`` handling in ``GPTModel`` state-dict plumbing.

In Qwen3.5 PP mode the vision encoder is attached to the pipeline model as a
plain ``vision_merge`` sublayer, so its parameters show up as ``vision_merge.*``
in ``super().state_dict()``. Those keys must never go through the pipeline
name mapping (they have no pipeline stage index); instead they are dropped and
re-added from the vision model's own state dict, which already produces
``model.vision_model.*`` names.

These tests drive the five affected methods with a lightweight ``GPTModel``
subclass that bypasses ``PipelineLayer.__init__`` and stubs ``super()``.
"""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import paddle
from paddle import nn

from paddlefleet.models.gpt import gpt_model
from paddlefleet.models.gpt.gpt_model import GPTModel

VISION_KEY = "vision_merge.vision_model.blocks.0.weight"
VISION_SINGLE_KEY = "model.vision_model.blocks.0.weight"
VISION_PREFIX = "model.vision_model."


class Value:
    """Stand-in for a parameter: only ``key`` and the sharding markers matter."""

    def __init__(self, name="", global_expert_id_offset=None, layer_cnt=None):
        self.key = name
        if global_expert_id_offset is not None:
            self.global_expert_id_offset = global_expert_id_offset
        if layer_cnt is not None:
            self.layer_cnt = layer_cnt


class Config:
    def __init__(self, model_type=""):
        self.model_type = model_type


class VisionModel:
    """Vision encoder stub exposing the three state-dict entry points."""

    def __init__(self):
        self.loaded_state = None
        self.sharded_prefix = None

    def state_dict(self):
        return {VISION_SINGLE_KEY: Value(VISION_SINGLE_KEY)}

    def set_state_dict(self, state_dict):
        self.loaded_state = dict(state_dict)

    def sharded_state_dict(self, structured_name_prefix=""):
        self.sharded_prefix = structured_name_prefix
        return {VISION_SINGLE_KEY: Value(VISION_SINGLE_KEY)}


class OpaqueVisionModel:
    """Vision encoder stub without any state-dict entry point."""


class RealVisionModel(nn.Layer):
    """A real nested vision model with real parameters.

    Mirrors ``Qwen3_5VisionModel``: it owns the parameters and does its own
    pipeline-to-structured name mapping, publishing ``model.vision_model.*``
    keys from both ``state_dict`` and ``sharded_state_dict`` and accepting the
    same names in ``set_state_dict``.
    """

    def __init__(self, fill=None, hidden=4):
        super().__init__()
        self.blocks = nn.LayerList(
            [nn.Linear(hidden, hidden, bias_attr=False) for _ in range(2)]
        )
        self.merger = nn.Linear(hidden, hidden, bias_attr=False)
        if fill is not None:
            for index, param in enumerate(self.parameters()):
                param.set_value(paddle.full(param.shape, fill + index))

    def raw_state_dict(self):
        """Unprefixed names, i.e. what the parent sees as ``vision_merge.*``."""
        return dict(super().state_dict())

    def state_dict(self, *args, **kwargs):
        del args, kwargs
        return {VISION_PREFIX + k: v for k, v in super().state_dict().items()}

    def set_state_dict(self, state_dict, *args, **kwargs):
        del args, kwargs
        stripped = {
            k.removeprefix(VISION_PREFIX): v for k, v in state_dict.items()
        }
        return super().set_state_dict(stripped)

    def sharded_state_dict(self, structured_name_prefix=""):
        return {
            structured_name_prefix + k: v for k, v in self.state_dict().items()
        }


class VisionMerge:
    def __init__(self, vision_model):
        self.vision_model = vision_model


class ParentMethods:
    def __init__(self, model):
        self._model = model

    def _values(self):
        if self._model.share_values:
            shared = Value("shared")
            values = dict.fromkeys(self._model.keys, shared)
        else:
            values = {key: Value(key) for key in self._model.keys}
        # a real PipelineLayer also reports the vision encoder's parameters
        values.update(self._model.parent_extra)
        return values

    def state_dict(self, *args, **kwargs):
        del args, kwargs
        return self._values()

    def set_state_dict(self, state_dict, *args, **kwargs):
        del args, kwargs
        self._model.loaded_state = dict(state_dict)
        return "loaded"

    def sharded_state_dict(self, *args, **kwargs):
        del args, kwargs
        result = {}
        for key in self._model.keys:
            if "experts." in key:
                result[key] = Value(key, global_expert_id_offset=3)
            else:
                result[key] = Value(key)
        result.update(self._model.parent_extra)
        return result


class LightweightGPT(GPTModel):
    """``GPTModel`` without ``PipelineLayer.__init__``."""

    def __init__(
        self,
        keys,
        name_prefixes,
        model_type="",
        vision_merge=None,
        share_values=False,
        num_virtual_pipeline_stages=1,
    ):
        self.config = Config(model_type)
        self.keys = list(keys)
        self.share_values = share_values
        # ``_set_pipeline_name_mapping`` asks the pipeline layer whether its
        # layers are chunked instead of guessing from the key shapes, so these
        # two fields -- normally set by the skipped ``PipelineLayer.__init__``
        # -- decide whether keys are read as `{stage}.rest` or
        # `{chunk_start}.{local_idx}.rest`. Callers must set them to match the
        # keys they pass in.
        self._num_virtual_pipeline_stages = num_virtual_pipeline_stages
        self._use_dualpipev = False
        # extra entries the stubbed parent reports verbatim, used to inject the
        # real ``vision_merge.vision_model.*`` parameters
        object.__setattr__(self, "parent_extra", {})
        self._sequential_layers = [
            {"layer": object(), "name_prefix": prefix}
            for prefix in name_prefixes
        ]
        self._pipeline_name_mapping = None
        self.layers = []
        self._stage_id = 0
        self.loaded_state = None
        if vision_merge is not None:
            self.vision_merge = vision_merge

    def set_parent_extra(self, extra):
        """Install extra parent keys.

        Uses ``object.__setattr__`` because ``Layer.__setattr__`` would try to
        register the parameters held by the dict on this half-built layer.
        """
        object.__setattr__(self, "parent_extra", dict(extra))


class VisionMergeStateDictTestBase(unittest.TestCase):
    def setUp(self):
        self._original_super = gpt_model.__dict__.get("super", None)

    def tearDown(self):
        if self._original_super is None:
            gpt_model.__dict__.pop("super", None)
        else:
            gpt_model.super = self._original_super

    def _install_parent(self, model):
        gpt_model.super = lambda: ParentMethods(model)

    def _make_model(self, **kwargs):
        model = LightweightGPT(**kwargs)
        self._install_parent(model)
        return model


class TestPipelineNameMapping(VisionMergeStateDictTestBase):
    """``vision_merge.*`` keys are excluded from the pipeline name mapping."""

    def test_vision_merge_keys_skipped(self):
        model = self._make_model(
            # vision_merge key comes first, so a mapping built by iterating the
            # state dict in order would trip over it before any language key
            keys=[VISION_KEY, "0.weight", "1.experts.1.weight"],
            name_prefixes=["model.embed", "model.layers.1"],
            vision_merge=VisionMerge(VisionModel()),
        )

        mapping = model._set_pipeline_name_mapping()

        self.assertEqual(mapping["model.embed.weight"], "0.weight")
        self.assertEqual(
            mapping["model.layers.1.experts.1.weight"], "1.experts.1.weight"
        )
        self.assertNotIn(VISION_KEY, model._pp_to_single_mapping)
        self.assertNotIn(VISION_KEY, mapping.values())

    def test_vision_merge_keys_skipped_under_virtual_pipeline(self):
        model = self._make_model(
            keys=[VISION_KEY, "0.0.weight", "extra.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(VisionModel()),
            num_virtual_pipeline_stages=2,
        )

        mapping = model._set_pipeline_name_mapping()

        # "0.0.weight" -> virtual pp naming; chunk 0 + offset 0 => prefix "0"
        self.assertEqual(mapping["model.embed.weight"], "0.0.weight")
        # a key without a stage index maps to itself
        self.assertEqual(mapping["extra.weight"], "extra.weight")
        self.assertNotIn(VISION_KEY, model._pp_to_single_mapping)

    def test_parameter_directly_on_vision_merge_is_rejected(self):
        """Only vision_merge.vision_model.* is re-exported; guard the rest."""
        model = self._make_model(
            keys=["vision_merge.projector.weight", "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(VisionModel()),
        )

        with self.assertRaises(ValueError) as ctx:
            model._set_pipeline_name_mapping()
        self.assertIn("vision_merge.projector.weight", str(ctx.exception))

    def test_guard_survives_optimized_mode(self):
        """``python -O`` strips asserts, so this guard must be a real raise."""
        script = (
            "from paddlefleet.models.gpt.gpt_model import is_vision_merge_key\n"
            "if is_vision_merge_key('0.weight') is not False:\n"
            "    raise SystemExit('plain key misclassified')\n"
            "if is_vision_merge_key('vision_merge.vision_model.w') is not True:\n"
            "    raise SystemExit('vision_model key misclassified')\n"
            "try:\n"
            "    is_vision_merge_key('vision_merge.projector.weight')\n"
            "except ValueError:\n"
            "    print('GUARD_FIRED')\n"
            "else:\n"
            "    raise SystemExit('guard was stripped by -O')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH", "")]
        )
        proc = subprocess.run(
            [sys.executable, "-O", "-c", script],
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("GUARD_FIRED", proc.stdout)


class TestStateDict(VisionMergeStateDictTestBase):
    """``state_dict`` drops ``vision_merge.*`` and re-adds vision model keys."""

    def test_rejects_parameter_directly_on_vision_merge(self):
        """The guard also holds when the mapping was cached earlier."""
        model = self._make_model(
            keys=[VISION_KEY, "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(VisionModel()),
        )
        model.state_dict()  # caches the mapping

        model.keys.append("vision_merge.projector.weight")
        with self.assertRaises(ValueError) as ctx:
            model.state_dict()
        self.assertIn("vision_merge.projector.weight", str(ctx.exception))

    def test_vision_keys_replaced_by_vision_model_state(self):
        model = self._make_model(
            keys=[VISION_KEY, "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(VisionModel()),
        )

        state = model.state_dict()

        self.assertNotIn(VISION_KEY, state)
        self.assertIn(VISION_SINGLE_KEY, state)
        self.assertIn("model.embed.weight", state)
        self.assertEqual(state["model.embed.weight"].key, "model.embed.weight")

    def test_unmapped_keys_pass_through(self):
        model = self._make_model(
            keys=["0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(OpaqueVisionModel()),
        )
        model.state_dict()  # builds and caches the mapping

        # a key that appeared after the mapping was built stays untouched
        model.keys.append("extra.weight")
        state = model.state_dict()

        self.assertIn("extra.weight", state)
        self.assertIn("model.embed.weight", state)
        # vision model without state_dict() contributes nothing
        self.assertNotIn(VISION_SINGLE_KEY, state)

    def test_without_vision_merge(self):
        model = self._make_model(
            keys=["0.weight"], name_prefixes=["model.embed"]
        )
        state = model.state_dict()
        self.assertEqual(list(state.keys()), ["model.embed.weight"])


class TestSetStateDict(VisionMergeStateDictTestBase):
    """``set_state_dict`` routes ``model.vision_model.*`` to the vision model."""

    def test_vision_state_forwarded_and_rest_remapped(self):
        vision_model = VisionModel()
        model = self._make_model(
            keys=[VISION_KEY, "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(vision_model),
        )

        ret = model.set_state_dict(
            {
                VISION_SINGLE_KEY: Value(VISION_SINGLE_KEY),
                "model.embed.weight": Value("model.embed.weight"),
                "not.a.known.key": Value("not.a.known.key"),
            }
        )

        self.assertEqual(ret, "loaded")
        self.assertEqual(
            list(vision_model.loaded_state.keys()), [VISION_SINGLE_KEY]
        )
        # only pipeline-mapped keys reach the parent loader
        self.assertEqual(list(model.loaded_state.keys()), ["0.weight"])

    def test_vision_model_without_setter_is_skipped(self):
        model = self._make_model(
            keys=[VISION_KEY, "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(OpaqueVisionModel()),
        )

        ret = model.set_state_dict(
            {
                VISION_SINGLE_KEY: Value(VISION_SINGLE_KEY),
                "model.embed.weight": Value("model.embed.weight"),
            }
        )

        self.assertEqual(ret, "loaded")
        self.assertEqual(list(model.loaded_state.keys()), ["0.weight"])

    def test_missing_vision_state_warns(self):
        """A text-only checkpoint must not silently leave the tower random."""
        vision_model = VisionModel()
        model = self._make_model(
            keys=[VISION_KEY, "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(vision_model),
        )

        with self.assertLogs(
            "paddlefleet.models.gpt.gpt_model", level="WARNING"
        ) as logs:
            ret = model.set_state_dict(
                {"model.embed.weight": Value("model.embed.weight")}
            )

        self.assertEqual(ret, "loaded")
        self.assertIsNone(vision_model.loaded_state)
        self.assertIn("model.vision_model.*", "\n".join(logs.output))

    def test_present_vision_state_does_not_warn(self):
        vision_model = VisionModel()
        model = self._make_model(
            keys=[VISION_KEY, "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(vision_model),
        )

        with self.assertNoLogs(
            "paddlefleet.models.gpt.gpt_model", level="WARNING"
        ):
            model.set_state_dict(
                {
                    VISION_SINGLE_KEY: Value(VISION_SINGLE_KEY),
                    "model.embed.weight": Value("model.embed.weight"),
                }
            )

        self.assertEqual(
            list(vision_model.loaded_state.keys()), [VISION_SINGLE_KEY]
        )

    def test_without_vision_merge(self):
        model = self._make_model(
            keys=["0.weight"], name_prefixes=["model.embed"]
        )

        ret = model.set_state_dict(
            {"model.embed.weight": Value("model.embed.weight")}
        )

        self.assertEqual(ret, "loaded")
        self.assertEqual(list(model.loaded_state.keys()), ["0.weight"])


class TestVisionMergeRoundTrip(VisionMergeStateDictTestBase):
    """Export/import a real nested vision model through the PP plumbing.

    The stub above checks key bookkeeping; this class uses a real
    ``nn.Layer`` vision model with real parameters and verifies the full
    round trip, so a prefix mistake that drops, duplicates or fails to load
    the visual weights cannot pass unnoticed.
    """

    LANG_KEYS = ["0.weight", "1.weight"]
    LANG_PREFIXES = ["model.embed", "model.layers.1"]

    def _make_pair(
        self, keys=None, name_prefixes=None, num_virtual_pipeline_stages=1
    ):
        """An exporter (filled weights) and an importer (zeroed weights)."""
        keys = self.LANG_KEYS if keys is None else keys
        name_prefixes = (
            self.LANG_PREFIXES if name_prefixes is None else name_prefixes
        )
        pair = []
        for fill in (1.0, 0.0):
            vision = RealVisionModel(fill=fill)
            model = LightweightGPT(
                keys=keys,
                name_prefixes=name_prefixes,
                vision_merge=VisionMerge(vision),
                num_virtual_pipeline_stages=num_virtual_pipeline_stages,
            )
            model.set_parent_extra(
                {
                    f"vision_merge.vision_model.{k}": v
                    for k, v in vision.raw_state_dict().items()
                }
            )
            pair.append((model, vision))
        return pair

    def _assert_vision_keys(self, exported, vision):
        expected = set(vision.state_dict().keys())
        self.assertTrue(expected, "the vision model must own parameters")
        # every visual weight is exported exactly once, under its single name
        self.assertEqual(
            {k for k in exported if k.startswith(VISION_PREFIX)}, expected
        )
        # ... and no raw pipeline-side name leaks through
        self.assertEqual(
            [k for k in exported if k.startswith("vision_merge.")], []
        )

    def test_state_dict_round_trip(self):
        (exporter, src_vision), (importer, dst_vision) = self._make_pair()

        self._install_parent(exporter)
        exported = exporter.state_dict()

        self._assert_vision_keys(exported, src_vision)
        # language keys are remapped to their structured names
        self.assertIn("model.embed.weight", exported)
        self.assertIn("model.layers.1.weight", exported)

        self._install_parent(importer)
        importer.set_state_dict(dict(exported))

        # visual weights land in the fresh instance, value for value
        src_state, dst_state = src_vision.state_dict(), dst_vision.state_dict()
        self.assertEqual(set(src_state), set(dst_state))
        for key, src_value in src_state.items():
            np.testing.assert_allclose(
                dst_state[key].numpy(),
                src_value.numpy(),
                err_msg=f"vision weight {key} did not round-trip",
            )
        # language weights reached the parent loader under pipeline names
        self.assertEqual(
            sorted(importer.loaded_state.keys()), sorted(self.LANG_KEYS)
        )

    def test_sharded_state_dict_round_trip(self):
        (exporter, src_vision), (importer, dst_vision) = self._make_pair()

        self._install_parent(exporter)
        sharded = exporter.sharded_state_dict()
        plain = exporter.state_dict()

        self._assert_vision_keys(sharded, src_vision)
        self.assertIn("model.embed.weight", sharded)
        # both entry points agree on the visual key set
        self.assertEqual(
            {k for k in sharded if k.startswith(VISION_PREFIX)},
            {k for k in plain if k.startswith(VISION_PREFIX)},
        )

        self._install_parent(importer)
        importer.set_state_dict(dict(sharded))
        for key, src_value in src_vision.state_dict().items():
            np.testing.assert_allclose(
                dst_vision.state_dict()[key].numpy(), src_value.numpy()
            )

    def test_round_trip_with_virtual_pipeline_prefixes(self):
        """VPP names carry two numeric components (``chunk.layer.param``)."""
        (exporter, src_vision), (importer, dst_vision) = self._make_pair(
            keys=["0.0.weight", "0.1.weight"],
            name_prefixes=["model.embed", "model.layers.1"],
            num_virtual_pipeline_stages=2,
        )

        self._install_parent(exporter)
        exported = exporter.state_dict()

        # the vision keys must not disturb the virtual-pp layout detection
        self.assertEqual(
            exporter._pipeline_name_mapping["model.embed.weight"], "0.0.weight"
        )
        self.assertEqual(
            exporter._pipeline_name_mapping["model.layers.1.weight"],
            "0.1.weight",
        )
        self._assert_vision_keys(exported, src_vision)

        self._install_parent(importer)
        importer.set_state_dict(dict(exported))
        for key, src_value in src_vision.state_dict().items():
            np.testing.assert_allclose(
                dst_vision.state_dict()[key].numpy(), src_value.numpy()
            )
        self.assertEqual(
            sorted(importer.loaded_state.keys()),
            ["0.0.weight", "0.1.weight"],
        )


class TestCheckSharedModelState(VisionMergeStateDictTestBase):
    """``_check_shared_model_state`` ignores ``vision_merge.*`` keys."""

    def test_vision_keys_do_not_raise_key_error(self):
        model = self._make_model(
            keys=[VISION_KEY, "0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(VisionModel()),
        )

        missing = model._check_shared_model_state()

        self.assertEqual(missing, {})

    def test_tied_weight_shared_across_stages(self):
        """Two pipeline keys pointing at one tensor share a structure name."""
        model = self._make_model(
            keys=[VISION_KEY, "0.weight", "1.weight"],
            # both layers share the same structure prefix -> same single name
            name_prefixes=["model.embed", "model.embed"],
            vision_merge=VisionMerge(VisionModel()),
            share_values=True,
        )

        missing = model._check_shared_model_state()

        # the shared tensor is reported: "0.weight" lost the reverse mapping
        self.assertEqual(missing, {"0.weight": "1.weight"})


class TestShardedStateDict(VisionMergeStateDictTestBase):
    """``sharded_state_dict`` re-adds remapped vision keys and expert offsets."""

    def test_vision_keys_replaced_and_expert_ids_globalized(self):
        vision_model = VisionModel()
        model = self._make_model(
            keys=[VISION_KEY, "0.weight", "1.experts.1.weight"],
            name_prefixes=["model.embed", "model.layers.1"],
            vision_merge=VisionMerge(vision_model),
        )

        sharded = model.sharded_state_dict()

        self.assertNotIn(VISION_KEY, sharded)
        self.assertIn(VISION_SINGLE_KEY, sharded)
        self.assertEqual(vision_model.sharded_prefix, "")
        # local expert 1 + offset 3 -> global expert 4
        self.assertIn("model.layers.1.experts.4.weight", sharded)
        self.assertIn("model.embed.weight", sharded)

    def test_unmapped_and_opaque_vision_model(self):
        model = self._make_model(
            keys=["0.weight"],
            name_prefixes=["model.embed"],
            vision_merge=VisionMerge(OpaqueVisionModel()),
        )
        model.sharded_state_dict()  # caches the mapping

        model.keys.append("extra.weight")
        sharded = model.sharded_state_dict()

        self.assertIn("extra.weight", sharded)
        self.assertNotIn(VISION_SINGLE_KEY, sharded)

    def test_without_vision_merge(self):
        model = self._make_model(
            keys=["0.weight"], name_prefixes=["model.embed"]
        )

        sharded = model.sharded_state_dict()

        self.assertEqual(list(sharded.keys()), ["model.embed.weight"])


if __name__ == "__main__":
    unittest.main()
