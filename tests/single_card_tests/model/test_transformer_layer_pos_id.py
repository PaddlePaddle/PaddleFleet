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
"""``TransformerLayer.forward`` slices MTP ``position_ids`` on the last axis.

With MTP enabled the layer splits ``position_ids`` into a decoder part and an
MTP part, and concatenates them back before returning. Slicing/concatenating on
axis 1 works for plain ``[B, S]`` ids but silently corrupts mRoPE's
``[3, B, S]`` ids (it would slice the batch axis), which is what Qwen3.5 VL
feeds in. Both layouts must round-trip unchanged.
"""

import os
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import paddle

from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import TransformerLayer

SEQ = 5
HIDDEN = 8
NUM_MTP = 1


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": HIDDEN,
        "num_attention_heads": 4,
        "use_cpu_initialization": True,
        "num_nextn_predict_layers": NUM_MTP,
        "mtp_load_weight_only": False,
        "experimental_dataflow": True,
        "gpt_model_use_experimental_version": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_layer(config):
    """A ``TransformerLayer`` whose core computation is a stub."""
    layer = TransformerLayer.__new__(TransformerLayer)
    layer.__dict__.setdefault("_parameters", {})
    layer.__dict__.setdefault("_buffers", {})
    layer.__dict__.setdefault("_sub_layers", {})
    layer.__dict__.setdefault("_loaddict_holder", {})
    layer.__dict__.setdefault("_non_persistable_buffers", set())
    object.__setattr__(layer, "config", config)
    object.__setattr__(layer, "full_recompute", False)
    object.__setattr__(
        layer, "_forward_impl", lambda hidden_states, **kwargs: hidden_states
    )
    return layer


class TestTransformerLayerMTPPositionIds(unittest.TestCase):
    def _run(self, position_ids):
        layer = _make_layer(_make_config())
        # main chunk + NUM_MTP chunks stacked on axis 0
        hidden_states = paddle.randn([NUM_MTP + 1, SEQ, HIDDEN])
        with patch(
            "paddlefleet.transformer.transformer_layer.has_recovered",
            return_value=True,
        ):
            return layer.forward(
                {
                    "hidden_states": hidden_states,
                    "position_ids": position_ids,
                }
            ), hidden_states

    def test_mrope_position_ids_round_trip(self):
        position_ids = paddle.arange(3 * SEQ, dtype="int64").reshape(
            [3, 1, SEQ]
        )

        rst, hidden_states = self._run(position_ids)

        self.assertEqual(list(rst["position_ids"].shape), [3, 1, SEQ])
        np.testing.assert_array_equal(
            rst["position_ids"].numpy(), position_ids.numpy()
        )
        self.assertEqual(
            list(rst["hidden_states"].shape), [NUM_MTP + 1, SEQ, HIDDEN]
        )
        np.testing.assert_allclose(
            rst["hidden_states"].numpy(), hidden_states.numpy()
        )

    def test_plain_position_ids_round_trip(self):
        position_ids = paddle.arange(2 * SEQ, dtype="int64").reshape([2, SEQ])

        rst, _ = self._run(position_ids)

        self.assertEqual(list(rst["position_ids"].shape), [2, SEQ])
        np.testing.assert_array_equal(
            rst["position_ids"].numpy(), position_ids.numpy()
        )

    def test_without_position_ids(self):
        layer = _make_layer(_make_config())
        hidden_states = paddle.randn([NUM_MTP + 1, SEQ, HIDDEN])
        with patch(
            "paddlefleet.transformer.transformer_layer.has_recovered",
            return_value=True,
        ):
            rst = layer.forward({"hidden_states": hidden_states})

        self.assertNotIn("position_ids", rst)
        self.assertEqual(
            list(rst["hidden_states"].shape), [NUM_MTP + 1, SEQ, HIDDEN]
        )

    def test_experimental_dataflow_version_skips_the_split(self):
        """The experimental version handles position_ids itself."""
        layer = _make_layer(
            _make_config(gpt_model_use_experimental_version=True)
        )
        position_ids = paddle.arange(3 * SEQ, dtype="int64").reshape(
            [3, 1, SEQ]
        )
        hidden_states = paddle.randn([NUM_MTP + 1, SEQ, HIDDEN])

        with patch(
            "paddlefleet.transformer.transformer_layer.has_recovered",
            return_value=True,
        ):
            rst = layer.forward(
                {
                    "hidden_states": hidden_states,
                    "position_ids": position_ids,
                }
            )

        # untouched: neither sliced nor re-concatenated
        self.assertIs(rst["position_ids"], position_ids)


if __name__ == "__main__":
    unittest.main()
