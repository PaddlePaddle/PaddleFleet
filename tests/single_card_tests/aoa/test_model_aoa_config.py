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
"""Hand-written per-model AOA generators (``_gen_aoa_config`` /
``_gen_inv_aoa_config``) for the MoE models touched by the empty-layer naming
decoupling. Each entry point is a pure classmethod that assembles AOA
statements from a config, so we drive it with a real ``PretrainedConfig`` and
assert (a) it returns a non-empty list of string statements and (b) the real
transformer/MTP layers it references are numbered contiguously from
``layers.0`` -- i.e. decoupled from any empty-layer offset.
"""

import re
import unittest

from paddlefleet.transformers.deepseek_v4.configuration import DeepseekV4Config
from paddlefleet.transformers.deepseek_v4.modeling import (
    DeepseekV4PreTrainedModel,
)
from paddlefleet.transformers.gemma4_moe.configuration import Gemma4MoeConfig
from paddlefleet.transformers.gemma4_moe.modeling import Gemma4MoeForCausalLM
from paddlefleet.transformers.glm4_moe.configuration import Glm4MoeConfig
from paddlefleet.transformers.glm4_moe.modeling import Glm4MoePreTrainedModel
from paddlefleet.transformers.kimi_k3.configuration import KimiK3TextConfig
from paddlefleet.transformers.kimi_k3.modeling import KimiK3PretrainedModel
from paddlefleet.transformers.minimax_m2.configuration import MiniMaxM2Config
from paddlefleet.transformers.minimax_m2.modeling import (
    MiniMaxM2PreTrainedModel,
)

# ``\blayers.N`` matches ``model.layers.0`` etc. but not ``empty_layers.0``
# (no word boundary between ``_`` and ``layers``), so we only pick up the real
# transformer/MTP layer indices.
_LAYER_RE = re.compile(r"\blayers\.(\d+)")


def _layer_indices(stmts):
    idx = set()
    for s in stmts:
        for m in _LAYER_RE.finditer(s):
            idx.add(int(m.group(1)))
    return idx


class _AoaConfigMixin:
    """Runs both AOA directions and pins the decoupled 0-based numbering."""

    def _check(self, cls, config):
        for meth in ("_gen_aoa_config", "_gen_inv_aoa_config"):
            out = getattr(cls, meth)(config)
            self.assertIn("aoa_statements", out, f"{meth} missing key")
            stmts = out["aoa_statements"]
            self.assertIsInstance(stmts, list)
            self.assertTrue(stmts, f"{meth} produced no statements")
            self.assertTrue(
                all(isinstance(s, str) and s for s in stmts),
                f"{meth} produced a non-string / empty statement",
            )
            idx = _layer_indices(stmts)
            self.assertTrue(idx, f"{meth} referenced no numbered layers")
            # Decoupled naming: real layers start at 0 and are contiguous, with
            # no ``+num_empty_layers_add_in_head`` gap.
            self.assertIn(0, idx, f"{meth} does not start at layers.0")
            self.assertEqual(
                idx,
                set(range(max(idx) + 1)),
                f"{meth} layer indices are not contiguous from 0: "
                f"{sorted(idx)}",
            )


class TestGemma4MoeAoaConfig(_AoaConfigMixin, unittest.TestCase):
    def test_gen(self):
        self._check(
            Gemma4MoeForCausalLM,
            Gemma4MoeConfig(num_hidden_layers=6),
        )


class TestDeepseekV4AoaConfig(_AoaConfigMixin, unittest.TestCase):
    def test_gen(self):
        self._check(
            DeepseekV4PreTrainedModel,
            DeepseekV4Config(
                num_hidden_layers=2,
                n_routed_experts=2,
                mtp_num_layers=1,
                enable_mtp_magic_send=True,
                csa_compress_ratios=[1, 1, 1, 1],
            ),
        )


class TestGlm4MoeAoaConfig(_AoaConfigMixin, unittest.TestCase):
    def test_gen(self):
        self._check(
            Glm4MoePreTrainedModel,
            Glm4MoeConfig(
                num_hidden_layers=2,
                n_routed_experts=2,
                first_k_dense_replace=1,
                mtp_num_layers=1,
                num_nextn_predict_layers=1,
            ),
        )


class TestKimiK3AoaConfig(_AoaConfigMixin, unittest.TestCase):
    def test_gen(self):
        self._check(
            KimiK3PretrainedModel,
            KimiK3TextConfig(
                num_hidden_layers=1,
                n_routed_experts=1,
                layer_types=["multi_latent_attention"],
                num_nextn_predict_layers=1,
            ),
        )


class TestMiniMaxM2AoaConfig(_AoaConfigMixin, unittest.TestCase):
    def test_gen_dense_mtp(self):
        self._check(
            MiniMaxM2PreTrainedModel,
            MiniMaxM2Config(
                num_hidden_layers=2,
                n_routed_experts=2,
                mtp_num_layers=1,
                num_nextn_predict_layers=1,
                use_dense_mtp=True,
            ),
        )

    def test_gen_sparse_mtp(self):
        self._check(
            MiniMaxM2PreTrainedModel,
            MiniMaxM2Config(
                num_hidden_layers=2,
                n_routed_experts=2,
                mtp_num_layers=1,
                num_nextn_predict_layers=1,
                use_dense_mtp=False,
            ),
        )


if __name__ == "__main__":
    unittest.main()
