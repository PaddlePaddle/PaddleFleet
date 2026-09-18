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

"""Single card tests for `PipelinePretrainedModel._set_pipeline_name_mapping`.

The mapping between pipeline parameter names (`{layer_idx}.{sublayer}.weight`)
and single card names (`model.layers.0.sublayer.weight`) decides which
checkpoint tensor lands in which layer. Getting it wrong loads the wrong
weights *silently*, so it needs tests that do not require a real pipeline group.

`_set_pipeline_name_mapping` only reads a handful of attributes off the pipeline
layer, so `FakePipelineModel` below supplies exactly those and nothing else: the
sequential layer descriptors, the chunking flags, and real parameters whose
names reproduce the key shapes a `PipelineLayer` would produce. No distributed
init, no GPU.

`legacy_pp_to_single_mapping` is a copy of the pre-fix implementation. Each test
asserts both what the current code produces and what the old code produced, so
the tests double as executable evidence of the behaviour change.
"""

import unittest
from unittest import mock

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import LayerDesc, SharedLayerDesc

from paddlefleet.transformers import model_utils
from paddlefleet.transformers.model_utils import PipelinePretrainedModel


class _Dummy(nn.Layer):
    pass


def _build_param_tree(root, keys):
    """Register real parameters so `state_dict()` yields exactly `keys`."""
    for key in keys:
        *path, leaf = key.split(".")
        cur = root
        for seg in path:
            child = cur._sub_layers.get(seg)
            if child is None:
                child = _Dummy()
                cur.add_sublayer(seg, child)
            cur = child
        cur.add_parameter(leaf, cur.create_parameter([1], dtype="float32"))


class FakePipelineModel(PipelinePretrainedModel):
    """Key-only stand-in for a `PipelineLayer`, built without a pp group."""

    def __init__(
        self,
        sequential_layers,
        state_dict_keys,
        num_virtual_pipeline_stages=1,
        use_dualpipev=False,
    ):
        nn.Layer.__init__(self)
        self.__init_hook__()
        for desc, prefix in sequential_layers:
            self.add_sequential_layer(desc, prefix)
        self._num_virtual_pipeline_stages = num_virtual_pipeline_stages
        self._use_dualpipev = use_dualpipev
        self._stage_id = 0
        _build_param_tree(self, state_dict_keys)

    @property
    def _layers_desc(self):
        return self.get_sequential_layers()

    def get_stage_from_index(self, index):
        # Single fake stage: every descriptor is considered local.
        return 0

    def pp_to_single_mapping(self):
        self._set_pipeline_name_mapping()
        return dict(self._pp_to_single_mapping)


def legacy_pp_to_single_mapping(model):
    """The pre-fix implementation, kept only so tests can show the contrast."""
    pp_to_single_mapping = {}
    state_dict_keys = list(nn.Layer.state_dict(model).keys())

    first_key = ""
    for k in state_dict_keys:
        if "shared_layers" not in k:
            first_key = k
            break
    first_key = first_key.split(".")
    use_vpp = first_key[0].isdigit() and first_key[1].isdigit()

    prefixes = model.get_sequential_name_prefixes()
    for k in state_dict_keys:
        name_splited = k.split(".")
        if use_vpp:
            if name_splited[0].isdigit():
                if name_splited[1].isdigit():
                    idx = str(int(name_splited[0]) + int(name_splited[1]))
                    single_name = [prefixes[idx]]
                    single_name.extend(name_splited[2:])
                else:
                    # "treat this key as last layer"
                    single_name = [prefixes[str(len(prefixes) - 1)]]
                    single_name.extend(name_splited[2:])
            elif name_splited[0] == "shared_layers":
                single_name = [model.get_shardlayer_prefix(name_splited)]
                single_name.extend(name_splited[2:])
            else:
                raise ValueError(f"Unexpected key: {k} for pp layer.")
        else:
            idx = name_splited[0]
            if idx.isdigit():
                single_name = [] if prefixes[idx] == "" else [prefixes[idx]]
                single_name.extend(name_splited[1:])
            elif idx == "shared_layers":
                single_name = [model.get_shardlayer_prefix(name_splited)]
                single_name.extend(name_splited[2:])
            else:
                raise ValueError(f"Unexpected key: {k} for pp layer.")
        pp_to_single_mapping[k] = ".".join(single_name)
    return pp_to_single_mapping


class TestPipelineNameMapping(unittest.TestCase):
    def test_non_vpp_with_nested_sequential_is_not_mistaken_for_vpp(self):
        """A1: a plain PP model holding a `LayerDesc(nn.Sequential, ...)`.

        Its first key is `0.0.weight` -- two digits -- which the old probe read
        as "this model is chunked". Nothing raises; the weights just go to the
        wrong layers.
        """
        model = FakePipelineModel(
            sequential_layers=[
                (LayerDesc(_Dummy), "model.embed_tokens"),
                (LayerDesc(_Dummy), "model.layers.0"),
                (LayerDesc(_Dummy), "lm_head"),
            ],
            state_dict_keys=[
                "0.0.weight",
                "1.self_attn.q_proj.weight",
                "2.weight",
            ],
            num_virtual_pipeline_stages=1,
        )

        self.assertEqual(
            model.pp_to_single_mapping(),
            {
                "0.0.weight": "model.embed_tokens.0.weight",
                "1.self_attn.q_proj.weight": (
                    "model.layers.0.self_attn.q_proj.weight"
                ),
                "2.weight": "lm_head.weight",
            },
        )

        legacy = legacy_pp_to_single_mapping(model)
        # Every single key was wrong, and none of it raised.
        # lost the `.0`
        self.assertEqual(legacy["0.0.weight"], "model.embed_tokens.weight")
        # wrong layer
        self.assertEqual(
            legacy["1.self_attn.q_proj.weight"], "lm_head.q_proj.weight"
        )
        # lost the parameter name
        self.assertEqual(legacy["2.weight"], "lm_head")

    def test_vpp_detected_even_when_first_key_is_a_directly_added_layer(self):
        """A1: under VPP the first key may be `lm_head`, i.e. `{idx}.weight`.

        The old probe then read "not chunked" and prepended the chunk offset as
        if it were a submodule name, so every transformer layer key was wrong.
        """
        model = FakePipelineModel(
            sequential_layers=[
                (LayerDesc(_Dummy), "model.layers.0"),
                (LayerDesc(_Dummy), "model.layers.1"),
                (LayerDesc(_Dummy), "model.layers.2"),
                (LayerDesc(_Dummy), "model.layers.3"),
                (LayerDesc(_Dummy), "lm_head"),
            ],
            state_dict_keys=[
                "4.weight",
                "0.0.self_attn.weight",
                "0.1.self_attn.weight",
            ],
            num_virtual_pipeline_stages=2,
        )

        self.assertEqual(
            model.pp_to_single_mapping(),
            {
                "4.weight": "lm_head.weight",
                "0.0.self_attn.weight": "model.layers.0.self_attn.weight",
                "0.1.self_attn.weight": "model.layers.1.self_attn.weight",
            },
        )

        legacy = legacy_pp_to_single_mapping(model)
        self.assertEqual(
            legacy["0.0.self_attn.weight"],
            "model.layers.0.0.self_attn.weight",
        )
        self.assertEqual(
            legacy["0.1.self_attn.weight"],
            "model.layers.0.1.self_attn.weight",
        )

    def test_vpp_directly_added_layers_do_not_collapse_onto_last_prefix(self):
        """A1 fallback: `{global_idx}.rest` was forced onto the last prefix.

        Two such layers then collided, so one of the two tensors was lost
        outright.
        """
        model = FakePipelineModel(
            sequential_layers=[
                (LayerDesc(_Dummy), "model.layers.0"),
                (LayerDesc(_Dummy), "model.layers.1"),
                (LayerDesc(_Dummy), "model.layers.2"),
                (LayerDesc(_Dummy), "model.layers.3"),
                (LayerDesc(_Dummy), "lm_head"),
                (LayerDesc(_Dummy), "mtp_head"),
            ],
            state_dict_keys=[
                "0.0.self_attn.weight",
                "4.linear.weight",
                "5.linear.weight",
            ],
            num_virtual_pipeline_stages=2,
        )

        mapping = model.pp_to_single_mapping()
        self.assertEqual(mapping["4.linear.weight"], "lm_head.linear.weight")
        self.assertEqual(mapping["5.linear.weight"], "mtp_head.linear.weight")
        # No two pipeline keys share a single card name.
        self.assertEqual(len(set(mapping.values())), len(mapping))

        legacy = legacy_pp_to_single_mapping(model)
        self.assertEqual(legacy["4.linear.weight"], "mtp_head.weight")
        self.assertEqual(legacy["5.linear.weight"], "mtp_head.weight")
        # The collision is what silently drops a tensor on save/load.
        self.assertLess(len(set(legacy.values())), len(legacy))

    def test_vpp_shared_layer_on_chunk_resolves_to_the_shared_prefix(self):
        """A5-2 / A2: a `SharedLayerDesc` with `forward_func` is named on the
        chunk.

        Its key is `{chunk_start}.{shared_name}.weight`, which the old code sent
        to the last prefix -- the embedding weight ended up named after
        `lm_head`.
        """
        shared = SharedLayerDesc(key="embed_share", layer_func=_Dummy)
        model = FakePipelineModel(
            sequential_layers=[
                (shared, "model.embed_tokens"),
                (LayerDesc(_Dummy), "model.layers.0"),
                (LayerDesc(_Dummy), "model.layers.1"),
                (LayerDesc(_Dummy), "lm_head"),
            ],
            state_dict_keys=[
                "0.1.self_attn.weight",
                "0.embed_share.weight",
            ],
            num_virtual_pipeline_stages=2,
        )

        mapping = model.pp_to_single_mapping()
        self.assertEqual(
            mapping["0.embed_share.weight"], "model.embed_tokens.weight"
        )
        self.assertEqual(
            mapping["0.1.self_attn.weight"],
            "model.layers.0.self_attn.weight",
        )

        legacy = legacy_pp_to_single_mapping(model)
        self.assertEqual(legacy["0.embed_share.weight"], "lm_head.weight")

    def test_shared_layer_aliases_agree(self):
        """`shared_layers.{name}.w` and `{chunk}.{name}.w` alias one parameter.

        They must resolve to the same single card name, otherwise the tied
        weight is written twice under two different names.
        """
        shared = SharedLayerDesc(key="embed_share", layer_func=_Dummy)
        sequential_layers = [
            (shared, "model.embed_tokens"),
            (LayerDesc(_Dummy), "model.layers.0"),
            (LayerDesc(_Dummy), "lm_head"),
        ]
        on_chunk = FakePipelineModel(
            sequential_layers=sequential_layers,
            state_dict_keys=["0.embed_share.weight"],
            num_virtual_pipeline_stages=2,
        )
        as_alias = FakePipelineModel(
            sequential_layers=sequential_layers,
            state_dict_keys=["shared_layers.embed_share.weight"],
            num_virtual_pipeline_stages=2,
        )

        self.assertEqual(
            on_chunk.pp_to_single_mapping()["0.embed_share.weight"],
            as_alias.pp_to_single_mapping()["shared_layers.embed_share.weight"],
        )

    def test_dualpipev_counts_as_chunked(self):
        """`_use_dualpipev` chunks the layers too, even when
        `_num_virtual_pipeline_stages == 1`."""
        model = FakePipelineModel(
            sequential_layers=[
                (LayerDesc(_Dummy), "model.layers.0"),
                (LayerDesc(_Dummy), "model.layers.1"),
                (LayerDesc(_Dummy), "lm_head"),
            ],
            state_dict_keys=["0.1.self_attn.weight"],
            num_virtual_pipeline_stages=1,
            use_dualpipev=True,
        )
        self.assertEqual(
            model.pp_to_single_mapping(),
            {"0.1.self_attn.weight": "model.layers.1.self_attn.weight"},
        )


class TestSetStateDictReporting(unittest.TestCase):
    """`set_state_dict` must say something when a parameter gets no value.

    Unmapped incoming keys are dropped on purpose -- loading a single card
    checkpoint hands every stage the whole key set -- so their absence from the
    mapping is not by itself an error. The error is the other direction: a
    parameter this stage *owns* that no checkpoint entry reached. That is what a
    broken name mapping looks like, and it is what `missing_keys` reports.
    """

    def _model(self):
        return FakePipelineModel(
            sequential_layers=[
                (LayerDesc(_Dummy), "model.embed_tokens"),
                (LayerDesc(_Dummy), "lm_head"),
            ],
            state_dict_keys=["0.weight", "1.weight"],
        )

    def _full_state_dict(self, model):
        model._set_pipeline_name_mapping()
        return {
            v: paddle.zeros([1], dtype="float32")
            for v in model._pp_to_single_mapping.values()
        }

    def test_missing_parameter_is_reported(self):
        model = self._model()
        full = self._full_state_dict(model)

        with mock.patch.object(model_utils, "logger") as log:
            missing_keys, _ = model.set_state_dict(
                {"model.embed_tokens.weight": full["model.embed_tokens.weight"]}
            )
        self.assertEqual(missing_keys, ["1.weight"])
        warned = " ".join(str(call) for call in log.warning.call_args_list)
        self.assertIn("got no value from the checkpoint", warned)

    def test_complete_state_dict_is_silent(self):
        model = self._model()
        full = self._full_state_dict(model)

        with mock.patch.object(model_utils, "logger") as log:
            missing_keys, unexpected_keys = model.set_state_dict(full)
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])
        log.warning.assert_not_called()

    def test_unmapped_incoming_key_is_dropped_and_logged(self):
        model = self._model()
        full = self._full_state_dict(model)
        full["some.other.stage.weight"] = paddle.zeros([1], dtype="float32")

        with mock.patch.object(model_utils, "logger") as log:
            missing_keys, unexpected_keys = model.set_state_dict(full)
        # Dropping it is correct, and it must not be mistaken for a missing
        # parameter.
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])
        warned = " ".join(str(call) for call in log.warning.call_args_list)
        self.assertIn("were dropped", warned)
        self.assertNotIn("got no value from the checkpoint", warned)

    def _aliased_model(self):
        """A stage holding both physical paths of one shared parameter.

        Under VPP the same `SharedLayerDesc` is reachable as
        `shared_layers.{name}.rest` and as `{chunk_start}.{name}.rest`
        (`pp_layers.py:1167-1171`), and both resolve to one single card name.
        Only one of them can be the value of `_single_to_pp_mapping`, so a
        checkpoint entry reaches exactly one path.
        """
        shared = SharedLayerDesc(key="embed_share", layer_func=_Dummy)
        return FakePipelineModel(
            sequential_layers=[
                (shared, "model.embed_tokens"),
                (LayerDesc(_Dummy), "model.layers.0"),
                (LayerDesc(_Dummy), "lm_head"),
            ],
            state_dict_keys=[
                "shared_layers.embed_share.weight",
                "0.embed_share.weight",
                "2.weight",
            ],
            num_virtual_pipeline_stages=2,
        )

    def test_shared_layer_alias_is_not_reported_as_missing(self):
        model = self._aliased_model()
        model._set_pipeline_name_mapping()
        single_names = set(model._pp_to_single_mapping.values())
        # Both physical paths collapse onto one single card name.
        self.assertEqual(
            single_names, {"model.embed_tokens.weight", "lm_head.weight"}
        )

        with mock.patch.object(model_utils, "logger") as log:
            missing_keys, _ = model.set_state_dict(
                {
                    name: paddle.zeros([1], dtype="float32")
                    for name in single_names
                }
            )
        # The path that lost the collision keeps its initial value here only
        # because the fake tree gives each key its own Parameter; in a real
        # PipelineLayer both paths are the same Parameter. Either way it is not
        # a broken mapping, so stay quiet.
        self.assertEqual(missing_keys, [])
        warned = " ".join(str(call) for call in log.warning.call_args_list)
        self.assertNotIn("got no value from the checkpoint", warned)

    def test_alias_filter_does_not_hide_genuinely_missing_shared_param(self):
        model = self._aliased_model()
        model._set_pipeline_name_mapping()

        with mock.patch.object(model_utils, "logger") as log:
            missing_keys, _ = model.set_state_dict(
                {"lm_head.weight": paddle.zeros([1], dtype="float32")}
            )
        # Nothing fed the shared parameter, so *both* physical paths are
        # missing and the filter must not swallow them -- that is a real
        # broken mapping.
        self.assertEqual(
            sorted(missing_keys),
            ["0.embed_share.weight", "shared_layers.embed_share.weight"],
        )
        warned = " ".join(str(call) for call in log.warning.call_args_list)
        self.assertIn("got no value from the checkpoint", warned)


if __name__ == "__main__":
    unittest.main()
