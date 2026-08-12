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
"""Tests for the ``mtp_load_weight_only`` guard in ``Qwen3_5RMSNormPipe``.

The pipe splits ``hidden_states`` into ``num_nextn_predict_layers + 1`` chunks,
normalizes the main chunk and concatenates the MTP chunks back. When MTP layers
are only loaded (``mtp_load_weight_only=True``) the incoming tensor holds the
main branch only, so splitting it would normalize a fraction of the sequence and
return a wrong-length tensor.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import paddle

from paddlefleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNormPipe
from paddlefleet.transformer.transformer_config import TransformerConfig

HIDDEN = 8


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": HIDDEN,
        "num_attention_heads": 4,
        "use_cpu_initialization": True,
        "num_nextn_predict_layers": 0,
        "mtp_load_weight_only": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestQwen3_5RMSNormPipeMTP(unittest.TestCase):
    def _reference_norm(self, pipe, tensor):
        return pipe.norm(tensor)

    def test_active_mtp_normalizes_main_chunk_only(self):
        config = _make_config(
            num_nextn_predict_layers=2, mtp_load_weight_only=False
        )
        pipe = Qwen3_5RMSNormPipe(config, hidden_size=HIDDEN)
        hidden_states = paddle.randn([3, 4, HIDDEN])

        out = pipe.forward({"hidden_states": hidden_states})["hidden_states"]

        self.assertEqual(list(out.shape), [3, 4, HIDDEN])
        # main chunk normalized, MTP chunks passed through untouched
        np.testing.assert_allclose(
            out[:1].numpy(),
            self._reference_norm(pipe, hidden_states[:1]).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out[1:].numpy(), hidden_states[1:].numpy(), rtol=1e-6, atol=1e-6
        )

    def test_mtp_load_weight_only_normalizes_whole_tensor(self):
        config = _make_config(
            num_nextn_predict_layers=2, mtp_load_weight_only=True
        )
        pipe = Qwen3_5RMSNormPipe(config, hidden_size=HIDDEN)
        hidden_states = paddle.randn([3, 4, HIDDEN])

        out = pipe.forward({"hidden_states": hidden_states})["hidden_states"]

        self.assertEqual(list(out.shape), [3, 4, HIDDEN])
        np.testing.assert_allclose(
            out.numpy(),
            self._reference_norm(pipe, hidden_states).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_no_mtp_normalizes_whole_tensor(self):
        config = _make_config(num_nextn_predict_layers=0)
        pipe = Qwen3_5RMSNormPipe(config, hidden_size=HIDDEN)
        hidden_states = paddle.randn([2, 4, HIDDEN])

        rst = pipe.forward(
            {"hidden_states": hidden_states, "position_ids": None}
        )

        self.assertEqual(list(rst["hidden_states"].shape), [2, 4, HIDDEN])
        # other dict entries survive
        self.assertIn("position_ids", rst)


if __name__ == "__main__":
    unittest.main()
