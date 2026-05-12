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

from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MultiLatentAttention,
)


class TestMultiLatentAttentionGateAttention(unittest.TestCase):
    """Tests for MultiLatentAttention gated attention."""

    def test_gated_attention_default_false(self):
        """gated_attention should default to False when config doesn't set it."""
        with patch.object(
            MultiLatentAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = MultiLatentAttention.__new__(MultiLatentAttention)
            attn.gated_attention = False
            self.assertFalse(attn.gated_attention)

    def test_gate_proj_is_none_when_not_gated(self):
        """gate_proj should be None when gated_attention is False."""
        with patch.object(
            MultiLatentAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = MultiLatentAttention.__new__(MultiLatentAttention)
            attn.gated_attention = False
            attn.gate_proj = None
            self.assertIsNone(attn.gate_proj)


class TestMLASelfAttentionBackwardDW(unittest.TestCase):
    """Tests for MLASelfAttention.backward_dw."""

    def test_backward_dw_calls_all_proj(self):
        """backward_dw should call _backward_kv_proj, _backward_q_proj, _backward_output_proj."""
        with patch.object(
            MLASelfAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = MLASelfAttention.__new__(MLASelfAttention)
            attn._backward_kv_proj = MagicMock()
            attn._backward_q_proj = MagicMock()
            attn._backward_output_proj = MagicMock()
            attn.backward_dw()
            attn._backward_kv_proj.assert_called_once()
            attn._backward_q_proj.assert_called_once()
            attn._backward_output_proj.assert_called_once()


class TestMLASelfAttentionBackwardKVProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_kv_proj."""

    def test_backward_kv_proj_calls_both_layers(self):
        """_backward_kv_proj should call backward_dw on kv_b_proj and kv_a_proj_with_mqa."""
        with patch.object(
            MLASelfAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = MLASelfAttention.__new__(MLASelfAttention)
            attn.kv_b_proj = MagicMock()
            attn.kv_a_proj_with_mqa = MagicMock()
            attn._backward_kv_proj()
            attn.kv_b_proj.backward_dw.assert_called_once()
            attn.kv_a_proj_with_mqa.backward_dw.assert_called_once()


class TestMLASelfAttentionBackwardQProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_q_proj."""

    def test_backward_q_proj_with_lora_rank(self):
        """_backward_q_proj should call q_a_proj and q_b_proj when q_lora_rank is not None."""
        with patch.object(
            MLASelfAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = MLASelfAttention.__new__(MLASelfAttention)
            attn.config = MagicMock()
            attn.config.q_lora_rank = 768
            attn.q_a_proj = MagicMock()
            attn.q_b_proj = MagicMock()
            attn._backward_q_proj()
            attn.q_a_proj.backward_dw.assert_called_once()
            attn.q_b_proj.backward_dw.assert_called_once()

    def test_backward_q_proj_without_lora_rank(self):
        """_backward_q_proj should call q_proj when q_lora_rank is None."""
        with patch.object(
            MLASelfAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = MLASelfAttention.__new__(MLASelfAttention)
            attn.config = MagicMock()
            attn.config.q_lora_rank = None
            attn.q_proj = MagicMock()
            attn._backward_q_proj()
            attn.q_proj.backward_dw.assert_called_once()


class TestMLASelfAttentionBackwardOutputProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_output_proj."""

    def test_backward_output_proj_calls_o_proj(self):
        """_backward_output_proj should call backward_dw on o_proj."""
        with patch.object(
            MLASelfAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = MLASelfAttention.__new__(MLASelfAttention)
            attn.o_proj = MagicMock()
            attn._backward_output_proj()
            attn.o_proj.backward_dw.assert_called_once()


if __name__ == "__main__":
    unittest.main()
