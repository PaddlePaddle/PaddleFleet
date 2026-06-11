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

"""Single-card tests for the new TransformerConfig fields and the new
``"allgather"`` value of ``moe_token_dispatcher_type`` (commit d231bb9
"add allgather dispatcher").

Pure config-level checks — no MoELayer instantiation, no distributed
init.  These cover the additions to ``transformer_config.py``.
"""

import unittest

from paddlefleet.transformer.transformer_config import TransformerConfig


class TestAllGatherDispatcherConfig(unittest.TestCase):
    """New fields/options added by the allgather dispatcher commit."""

    def test_allgather_dispatcher_type_is_preserved(self):
        cfg = TransformerConfig(
            num_hidden_layers=4,
            n_routed_experts=8,
            moe_token_dispatcher_type="allgather",
        )
        self.assertEqual(cfg.moe_token_dispatcher_type, "allgather")
        # Default for moe_use_fusion_node remains True.
        self.assertTrue(cfg.moe_use_fusion_node)

    def test_moe_allgather_gate_overlap_default_false(self):
        cfg = TransformerConfig(num_hidden_layers=4)
        self.assertFalse(cfg.moe_allgather_gate_overlap)

    def test_moe_allgather_gate_overlap_can_be_enabled(self):
        cfg = TransformerConfig(
            num_hidden_layers=4,
            moe_allgather_gate_overlap=True,
        )
        self.assertTrue(cfg.moe_allgather_gate_overlap)

    def test_default_dispatcher_type_unchanged(self):
        # The commit only *added* "allgather" to the option list; the
        # default is still "deepep".
        cfg = TransformerConfig(num_hidden_layers=4)
        self.assertEqual(cfg.moe_token_dispatcher_type, "deepep")

    def test_all_dispatcher_types_accepted(self):
        for dt in ("allgather", "alltoall", "deepep", "hybridep"):
            cfg = TransformerConfig(
                num_hidden_layers=4,
                n_routed_experts=8,
                moe_token_dispatcher_type=dt,
            )
            self.assertEqual(cfg.moe_token_dispatcher_type, dt)


if __name__ == "__main__":
    unittest.main()
