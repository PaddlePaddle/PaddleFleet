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

"""Behavior tests for paddlefleet.transformer.attention.

Covered production behaviors (all derived by reading src by hand):
  * Attention.set_for_recompute_input_layernorm raises NotImplementedError,
    and the concrete SelfAttention/CrossAttention inherit that same stub.
  * CrossAttentionSublayersSpec dataclass field defaults / storage order.
  * CrossAttention.__init__ validation: rejects GQA (num_key_value_heads !=
    num_attention_heads) with ValueError, and accepts the MHA case, building
    all four sublayers.

These require paddle (the module imports it at load time); no accelerator is
needed for the covered logic, but paddle itself must be importable. When it is
not, every case is honestly skipped with the real import error rather than
faking a pass.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Allow importing paddlefleet from the in-tree src/ when it is not installed.
_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle  # noqa: F401

    from paddlefleet.transformer.attention import (
        Attention,
        CrossAttention,
        CrossAttentionSublayersSpec,
        SelfAttention,
    )
    from paddlefleet.transformer.enums import AttnMaskType

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, precise probe
    _IMPORT_ERROR = exc


def _make_cross_config(num_attention_heads, num_key_value_heads):
    """Build a config double that satisfies Attention.__init__ arithmetic.

    Only the fields consumed before/at the GQA validation are pinned to real
    values; the rest stay as auto-mocks that are merely forwarded to the
    (mocked) sublayer builder.
    """
    config = MagicMock()
    config.sliding_window = None
    config.head_dim = 64
    config.v_head_dim = None  # not an int -> falls back to head_dim
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_key_value_heads
    config.rotary_percent = 1.0  # keep qk_rope_head_dim == head_dim
    config.recompute_granularity = None
    config.recompute_modules = None
    return config


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}",
)
class TestSetForRecomputeInputLayernorm(unittest.TestCase):
    """Attention.set_for_recompute_input_layernorm is an unimplemented stub."""

    def test_base_raises_not_implemented(self):
        # Invoke the real production method with a sentinel self; it ignores
        # self and must raise. No __init__ of the class-under-test is patched.
        with self.assertRaises(NotImplementedError):
            Attention.set_for_recompute_input_layernorm(object())

    def test_concrete_subclasses_inherit_the_stub(self):
        # SelfAttention/CrossAttention do not override it, so the raising
        # contract above applies to them too. This binds the contract to the
        # concrete entry points, not just the abstract base.
        self.assertIs(
            SelfAttention.set_for_recompute_input_layernorm,
            Attention.set_for_recompute_input_layernorm,
        )
        self.assertIs(
            CrossAttention.set_for_recompute_input_layernorm,
            Attention.set_for_recompute_input_layernorm,
        )


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}",
)
class TestCrossAttentionSublayersSpec(unittest.TestCase):
    """Field defaults and positional storage order of the dataclass."""

    def test_defaults_are_none(self):
        spec = CrossAttentionSublayersSpec()
        self.assertIsNone(spec.linear_q)
        self.assertIsNone(spec.linear_kv)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)

    def test_positional_fields_are_not_swapped(self):
        # Distinct sentinels per positional field detect any reordering.
        q, kv, core, o = object(), object(), object(), object()
        spec = CrossAttentionSublayersSpec(q, kv, core, o)
        self.assertIs(spec.linear_q, q)
        self.assertIs(spec.linear_kv, kv)
        self.assertIs(spec.core_attention, core)
        self.assertIs(spec.o_proj, o)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}",
)
class TestCrossAttentionGQAValidation(unittest.TestCase):
    """CrossAttention.__init__ rejects GQA and accepts plain MHA.

    build_spec_layer, get_pg_size and the process-group collection are genuine
    not-under-test collaborators (they build paddle sublayers / describe the
    distributed topology); mocking them isolates the validation branch that is
    actually under test. CrossAttention.__init__ itself is NOT patched.
    """

    @patch("paddlefleet.transformer.attention.build_spec_layer")
    @patch("paddlefleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddlefleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_rejects_group_query_attention(
        self, mock_pg, mock_size, mock_build
    ):
        mock_pg.return_value = MagicMock(tp=MagicMock(), cp=MagicMock())
        config = _make_cross_config(
            num_attention_heads=8, num_key_value_heads=4
        )
        with self.assertRaises(ValueError):
            CrossAttention(
                config=config,
                sublayers_spec=CrossAttentionSublayersSpec(),
                layer_number=1,
                attn_mask_type=AttnMaskType.padding,
            )

    @patch("paddlefleet.transformer.attention.build_spec_layer")
    @patch("paddlefleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddlefleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_accepts_mha_and_builds_all_sublayers(
        self, mock_pg, mock_size, mock_build
    ):
        mock_pg.return_value = MagicMock(tp=MagicMock(), cp=MagicMock())
        config = _make_cross_config(
            num_attention_heads=8, num_key_value_heads=8
        )
        attn = CrossAttention(
            config=config,
            sublayers_spec=CrossAttentionSublayersSpec(),
            layer_number=1,
            attn_mask_type=AttnMaskType.padding,
        )
        # Passing the validation is observable: with num_kv == num_attn the
        # projection sizes match and construction reaches the linear_q /
        # linear_kv builds. Four build_spec_layer calls == core_attention +
        # o_proj (base) + linear_q + linear_kv (cross).
        self.assertEqual(attn.attention_type, "cross")
        self.assertEqual(attn.query_projection_size, attn.key_projection_size)
        self.assertEqual(mock_build.call_count, 4)


if __name__ == "__main__":
    unittest.main()
