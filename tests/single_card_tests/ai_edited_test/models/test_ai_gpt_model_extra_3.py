# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.models.gpt.gpt_model import (
    GPTModel,
)


class TestGPTModelSetPipelineNameMapping(unittest.TestCase):
    """Tests for GPTModel._set_pipeline_name_mapping."""

    def test_with_explicit_mappings(self):
        """_set_pipeline_name_mapping should set mapping when provided."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            mappings = {"model.embed.weight": "0.weight"}
            model._pipeline_name_mapping = None
            result = model._set_pipeline_name_mapping(mappings)
            self.assertEqual(result, mappings)
            self.assertEqual(model._pipeline_name_mapping, mappings)


class TestGPTModelSetStateDict(unittest.TestCase):
    """Tests for GPTModel.set_state_dict."""

    def test_set_state_dict_remaps_keys(self):
        """set_state_dict should remap keys using pipeline_name_mapping."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._pipeline_name_mapping = {
                "model.embed.weight": "0.weight",
                "model.lm_head.weight": "1.weight",
            }
            input_sd = {
                "model.embed.weight": paddle.randn([4, 4]),
                "model.lm_head.weight": paddle.randn([4, 4]),
            }
            with patch.object(
                type(model).__mro__[1], "set_state_dict", return_value=True
            ) as mock_super:
                result = model.set_state_dict(input_sd)
                self.assertTrue(result)
                call_args = mock_super.call_args[0][0]
                self.assertIn("0.weight", call_args)
                self.assertIn("1.weight", call_args)


class TestGPTModelCheckSharedModelState(unittest.TestCase):
    """Tests for GPTModel._check_shared_model_state."""

    def test_returns_empty_dict_when_mappings_consistent(self):
        """_check_shared_model_state should return empty dict when mappings are consistent."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._pipeline_name_mapping = {"a": "0.a", "b": "1.b"}
            model._pp_to_single_mapping = {"0.a": "a", "1.b": "b"}
            with patch.object(
                type(model).__mro__[1], "state_dict", return_value={}
            ):
                result = model._check_shared_model_state()
                self.assertEqual(len(result), 0)


class TestGPTModelShardedStateDict(unittest.TestCase):
    """Tests for GPTModel.sharded_state_dict."""

    def test_handles_expert_offset_attribute(self):
        """sharded_state_dict should increment expert number for entries with global_expert_id_offset."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._pipeline_name_mapping = {}
            model._pp_to_single_mapping = {}
            model.config = MagicMock()
            model.config.model_type = "gpt"

            mock_val = MagicMock()
            mock_val.global_expert_id_offset = 4
            mock_val.key = "model.layers.0.experts.0.weight"

            mock_super = MagicMock()
            mock_super.sharded_state_dict.return_value = {
                "model.layers.0.experts.0.weight": mock_val
            }
            with patch.object(
                type(model).__mro__[1],
                "sharded_state_dict",
                mock_super.sharded_state_dict,
            ):
                result = model.sharded_state_dict()
                # Should have incremented expert number
                self.assertTrue(any("experts.4" in k for k in result.keys()))


class TestGPTModelGetLayerDescListQwenVL(unittest.TestCase):
    """Tests for GPTModel.get_layer_desc_list with qwen3_vl model type."""

    def test_uses_language_model_prefix_for_qwen3_vl(self):
        """get_layer_desc_list should use 'model.language_model' prefix for qwen3_vl."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model.config = MagicMock()
            model.config.model_type = "qwen3_vl"
            model.config.gpt_model_use_experimental_version = False
            model.config.num_nextn_predict_layers = 0

            layers = []
            model.add_sequential_layer = lambda ls, d, p: ls.append(
                {"layer": d, "name_prefix": p}
            )
            mock_spec = MagicMock()
            mock_spec.embedding = MagicMock()
            mock_spec.head_empty_layers = []
            mock_spec.transformer_layers = []
            mock_spec.tail_empty_layers = []
            mock_spec.mtp = None
            mock_spec.mtp_lm_head = None
            mock_spec.mtp_loss = None
            mock_spec.layer_norm = MagicMock()
            mock_spec.lm_head = MagicMock()

            result = model.get_layer_desc_list(
                mock_spec, tie_word_embeddings=False
            )
            self.assertTrue(
                layers[0]["name_prefix"].startswith("model.language_model")
            )


if __name__ == "__main__":
    unittest.main()
