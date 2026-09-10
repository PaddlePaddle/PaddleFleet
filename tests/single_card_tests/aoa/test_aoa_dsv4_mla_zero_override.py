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
#
# Scope: DeepSeek-V4 MLA is a ZERO-OVERRIDE case for its ordinary layout, NOT
# an adapter subclass. An earlier design pass assumed a ``NonTransposeLinear``
# adapter would be needed for ``linear_o_group_proj``, but that projection is a
# bare ``create_parameter`` -> the generic Layer recursion emits an identity
# statement (no ``^T``), which is exactly correct. The sibling ``o_proj`` is a
# plain Linear covered by the generic Linear family. So on the ordinary layout
# the whole MLA stack (``MultiLatentAttention`` / ``MLASelfAttention`` /
# ``MQASelfAttention``) adds NO component AOA behaviour: the generic
# ``paddle.nn.Layer.gen_(inv_)aoa_statements`` recursion is what runs, and the
# checkpoint<->model name / dtype divergences are model-level concerns handled
# during per-model migration.
#
# The one exception is ``mqa_split_kv_b_proj``, where ``MLASelfAttention`` does
# not build ``kv_b_proj`` at all and holds two standalone absorption parameters
# instead. That needs a real chain over one checkpoint key, so
# ``MLASelfAttention`` defines both generators -- but they must delegate to the
# generic recursion whenever the mode is off (see
# ``test_ai_aoa_mla_split_kv_b_component.py`` for the chain itself).
#
# This test locks in the narrowed contract: no MLA class other than
# ``MLASelfAttention`` may define AOA generators, and ``MLASelfAttention``'s
# ``super()`` must still land on the generic recursion, so a future edit that
# resurrects the abandoned adapter path fails loudly.
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


class TestMLAZeroOverride(unittest.TestCase):
    """Only ``MLASelfAttention`` may define AOA generators (for the split
    ``kv_b_proj`` mode); the ordinary layout rides the generic Layer recursion
    (identity for the bare ``linear_o_group_proj`` param, ``^T`` for the
    ``o_proj`` Linear child)."""

    def _zero_override_classes(self):
        from paddlefleet.transformer.multi_latent_attention import (
            MQASelfAttention,
            MultiLatentAttention,
        )

        # MQASelfAttention subclasses MLASelfAttention and inherits its split
        # handling unchanged, so it must not add generators of its own.
        return (MultiLatentAttention, MQASelfAttention)

    def _mla_self_attention(self):
        from paddlefleet.transformer.multi_latent_attention import (
            MLASelfAttention,
        )

        return MLASelfAttention

    def test_no_forward_override(self):
        for cls in self._zero_override_classes():
            self.assertNotIn(
                "gen_aoa_statements",
                cls.__dict__,
                f"{cls.__name__} must stay zero-override "
                f"(the generic Layer recursion is already correct)",
            )

    def test_no_inverse_override(self):
        for cls in self._zero_override_classes():
            self.assertNotIn(
                "gen_inv_aoa_statements",
                cls.__dict__,
                f"{cls.__name__} must stay zero-override "
                f"(the generic Layer recursion is already correct)",
            )

    def test_mqa_inherits_mla_generators(self):
        from paddlefleet.transformer.multi_latent_attention import (
            MQASelfAttention,
        )

        mla = self._mla_self_attention()
        self.assertIs(
            MQASelfAttention.gen_aoa_statements, mla.gen_aoa_statements
        )
        self.assertIs(
            MQASelfAttention.gen_inv_aoa_statements, mla.gen_inv_aoa_statements
        )

    def test_generic_recursion_reachable(self):
        # ``MultiLatentAttention`` sits ABOVE ``MLASelfAttention``, so its
        # inherited generators resolve straight to the generic paddle.nn.Layer
        # recursion, confirming the zero-override path is what actually runs.
        # ``MQASelfAttention`` sits below and therefore inherits
        # ``MLASelfAttention``'s split-mode generators instead (pinned by
        # ``test_mqa_inherits_mla_generators``); those reach the same generic
        # recursion through ``super()`` (pinned by
        # ``test_mla_super_lands_on_generic_recursion``).
        import paddle

        from paddlefleet.transformer.multi_latent_attention import (
            MultiLatentAttention,
        )

        self.assertIs(
            MultiLatentAttention.gen_aoa_statements,
            paddle.nn.Layer.gen_aoa_statements,
        )
        self.assertIs(
            MultiLatentAttention.gen_inv_aoa_statements,
            paddle.nn.Layer.gen_inv_aoa_statements,
        )

    def test_mla_super_lands_on_generic_recursion(self):
        # MLASelfAttention's non-split branch is a bare ``super()`` call, so the
        # first ancestor defining each generator has to be paddle.nn.Layer for
        # the ordinary layout to keep its current (correct) behaviour.
        import paddle

        mla = self._mla_self_attention()
        for method_name in ("gen_aoa_statements", "gen_inv_aoa_statements"):
            self.assertIn(method_name, mla.__dict__)
            owner = next(
                cls for cls in mla.__mro__[1:] if method_name in cls.__dict__
            )
            self.assertIs(owner, paddle.nn.Layer)


if __name__ == "__main__":
    unittest.main()
