# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
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
    MLASelfAttentionSublayersSpec,
    _accuracy_compatible_mla_rope_apply,
    _accuracy_compatible_projection,
    _accuracy_compatible_q_down_projection,
    _accuracy_compatible_q_up_projection,
)


def _make_mla_self_attn(**attrs):
    """Create MLASelfAttention with mocked __init__."""
    with patch.object(
        MLASelfAttention, "__init__", lambda self, *a, **kw: None
    ):
        attn = MLASelfAttention.__new__(MLASelfAttention)
        object.__setattr__(attn, "_sub_layers", {})
        object.__setattr__(attn, "_parameters", {})
        object.__setattr__(attn, "_buffers", {})
        object.__setattr__(attn, "_non_persistable_buffers", set())
        for k, v in attrs.items():
            object.__setattr__(attn, k, v)
        config = attrs.get("config", None)
        if config is not None and not hasattr(attn, "q_lora_rank"):
            object.__setattr__(attn, "q_lora_rank", config.q_lora_rank)
        return attn


class TestAccuracyCompatibleMLARope(unittest.TestCase):
    def test_roll_position_contract_matches_expected_rotation(self):
        import paddle

        q_pe = paddle.to_tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        k_pe = q_pe.clone()
        position_ids = paddle.to_tensor([1], dtype="int64")
        q_out, k_out = _accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, rope_base=100.0, position_ids=position_ids
        )

        inv_freq = paddle.to_tensor([1.0, 0.1], dtype="float32")
        freqs = paddle.concat((inv_freq, inv_freq)).reshape([1, 1, 1, 4])
        ordered = paddle.to_tensor([[[[1.0, 3.0, 2.0, 4.0]]]])
        rotated = paddle.to_tensor([[[[-2.0, -4.0, 1.0, 3.0]]]])
        expected = ordered * paddle.cos(freqs) + rotated * paddle.sin(freqs)
        paddle.testing.assert_close(q_out, expected)
        paddle.testing.assert_close(k_out, expected)

    def test_sequence_parallel_trims_mtp_lookahead_position(self):
        import paddle

        seq_len, heads, dim = 4, 2, 4
        q_pe = paddle.randn([seq_len, 1, heads, dim])
        k_pe = paddle.randn([seq_len, 1, 1, dim])
        position_ids = paddle.arange(seq_len + 1, dtype="int64")

        q_out, k_out = _accuracy_compatible_mla_rope_apply(
            q_pe,
            k_pe,
            rope_base=100.0,
            position_ids=position_ids,
            sequence_parallel=True,
        )
        expected_q, expected_k = _accuracy_compatible_mla_rope_apply(
            q_pe,
            k_pe,
            rope_base=100.0,
            position_ids=position_ids[:seq_len],
            sequence_parallel=True,
        )

        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)
        paddle.testing.assert_close(q_out, expected_q)
        paddle.testing.assert_close(k_out, expected_k)

    def test_sequence_parallel_applies_local_key_window(self):
        import paddle

        q_pe = paddle.randn([4, 1, 2, 4])
        k_pe = paddle.randn([2, 1, 1, 4])
        position_ids = paddle.arange(4, dtype="int64")
        with patch(
            "paddlefleet.transformer.multi_latent_attention.paddle.distributed.get_rank",
            return_value=0,
        ):
            q_out, k_out = _accuracy_compatible_mla_rope_apply(
                q_pe,
                k_pe,
                rope_base=100.0,
                position_ids=position_ids,
                sequence_parallel=True,
            )
        expected_q, _ = _accuracy_compatible_mla_rope_apply(
            q_pe,
            q_pe,
            rope_base=100.0,
            position_ids=position_ids,
            sequence_parallel=True,
        )
        _, expected_k = _accuracy_compatible_mla_rope_apply(
            k_pe.expand([2, 1, 2, 4]),
            k_pe,
            rope_base=100.0,
            position_ids=position_ids[:2],
            sequence_parallel=True,
        )
        self.assertEqual(tuple(q_out.shape), (4, 1, 2, 4))
        self.assertEqual(tuple(k_out.shape), (2, 1, 1, 4))
        paddle.testing.assert_close(q_out, expected_q)
        paddle.testing.assert_close(k_out, expected_k)

    def test_rejects_short_position_ids(self):
        import paddle

        q_pe = paddle.randn([4, 1, 2, 4])
        k_pe = paddle.randn([4, 1, 1, 4])
        with self.assertRaisesRegex(
            ValueError, "shorter than the query sequence"
        ):
            _accuracy_compatible_mla_rope_apply(
                q_pe,
                k_pe,
                rope_base=100.0,
                position_ids=paddle.arange(3, dtype="int64"),
                sequence_parallel=True,
            )


class TestAccuracyCompatibleProjection(unittest.TestCase):
    def test_matches_functional_linear_and_preserves_skip_bias_contract(self):
        import paddle

        hidden = paddle.randn([1, 3, 4])
        weight = paddle.randn([4, 6])
        bias = paddle.randn([6])
        projection = MagicMock(weight=weight, bias=bias, skip_bias_add=True)

        output, output_bias = _accuracy_compatible_projection(
            projection, hidden
        )

        expected = paddle.nn.functional.linear(hidden, weight)
        self.assertTrue(paddle.equal_all(output, expected).item())
        self.assertIs(output_bias, bias)

    def test_adds_bias_when_skip_bias_add_is_false(self):
        import paddle

        hidden = paddle.randn([1, 3, 4])
        weight = paddle.randn([4, 6])
        bias = paddle.randn([6])
        projection = MagicMock(weight=weight, bias=bias, skip_bias_add=False)

        output, output_bias = _accuracy_compatible_projection(
            projection, hidden
        )

        expected = paddle.nn.functional.linear(hidden, weight, bias)
        self.assertTrue(paddle.equal_all(output, expected).item())
        self.assertIsNone(output_bias)


class TestAccuracyCompatibleQUpProjection(unittest.TestCase):
    def test_preserves_forward_weight_grad_and_materialized_input_dgrad(self):
        import paddle

        hidden = paddle.randn([1, 3, 4])
        weight = paddle.randn([4, 6])
        hidden.stop_gradient = False
        weight.stop_gradient = False
        projection = MagicMock(weight=weight, bias=None, skip_bias_add=False)
        projection.side_effect = lambda value: (
            paddle.nn.functional.linear(value, weight),
            None,
        )
        grad_output = paddle.randn([1, 3, 6])

        output, output_bias = _accuracy_compatible_q_up_projection(
            projection, hidden
        )
        output.backward(grad_output)
        expected_input_grad = paddle.matmul(
            grad_output, weight.detach().transpose([1, 0]).contiguous()
        )
        expected_weight_grad = paddle.matmul(
            hidden.detach().reshape([-1, 4]).transpose([1, 0]),
            grad_output.reshape([-1, 6]),
        )

        self.assertTrue(
            paddle.equal_all(
                output.detach(), paddle.nn.functional.linear(hidden, weight)
            )
        )
        self.assertTrue(paddle.equal_all(hidden.grad, expected_input_grad))
        self.assertTrue(paddle.equal_all(weight.grad, expected_weight_grad))
        self.assertIsNone(output_bias)


class TestAccuracyCompatibleQDownProjection(unittest.TestCase):
    def test_matches_functional_linear_and_preserves_skip_bias_contract(self):
        import paddle

        hidden = paddle.randn([1, 3, 4])
        weight = paddle.randn([4, 6])
        bias = paddle.randn([6])
        projection = MagicMock(weight=weight, bias=bias, skip_bias_add=True)

        output, output_bias = _accuracy_compatible_q_down_projection(
            projection, hidden
        )

        expected = paddle.nn.functional.linear(hidden, weight)
        self.assertTrue(paddle.equal_all(output, expected).item())
        self.assertIs(output_bias, bias)

    def test_adds_bias_when_skip_bias_add_is_false(self):
        import paddle

        hidden = paddle.randn([1, 3, 4])
        weight = paddle.randn([4, 6])
        bias = paddle.randn([6])
        projection = MagicMock(weight=weight, bias=bias, skip_bias_add=False)

        output, output_bias = _accuracy_compatible_q_down_projection(
            projection, hidden
        )

        expected = paddle.nn.functional.linear(hidden, weight, bias)
        self.assertTrue(paddle.equal_all(output, expected).item())
        self.assertIsNone(output_bias)

    def test_replicated_linear_skips_sp_gather(self):
        import paddle

        hidden = paddle.randn([1, 3, 4])
        weight = paddle.randn([4, 6])
        projection = MagicMock(
            weight=weight,
            bias=None,
            skip_bias_add=True,
            sequence_parallel=False,
            tp_group=None,
        )
        with patch(
            "paddlefleet.tensor_parallel.mappings.gather_from_sequence_parallel_region"
        ) as mock_gather:
            output, _ = _accuracy_compatible_q_down_projection(
                projection, hidden
            )
        mock_gather.assert_not_called()
        expected = paddle.nn.functional.linear(hidden, weight)
        self.assertTrue(paddle.equal_all(output, expected).item())


class TestMLASelfAttentionBackwardDW(unittest.TestCase):
    """Tests for MLASelfAttention.backward_dw."""

    def test_backward_dw_calls_all_proj(self):
        """backward_dw should call _backward_kv_proj, _backward_q_proj, _backward_output_proj."""
        attn = _make_mla_self_attn()
        object.__setattr__(attn, "_backward_kv_proj", MagicMock())
        object.__setattr__(attn, "_backward_q_proj", MagicMock())
        object.__setattr__(attn, "_backward_output_proj", MagicMock())
        attn.backward_dw()
        attn._backward_kv_proj.assert_called_once()
        attn._backward_q_proj.assert_called_once()
        attn._backward_output_proj.assert_called_once()


class TestMLASelfAttentionBackwardKVProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_kv_proj."""

    def test_backward_kv_proj_calls_both_layers(self):
        """_backward_kv_proj should call backward_dw on kv_b_proj and kv_a_proj_with_mqa."""
        attn = _make_mla_self_attn()
        object.__setattr__(attn, "kv_b_proj", MagicMock())
        object.__setattr__(attn, "kv_a_proj_with_mqa", MagicMock())
        attn._backward_kv_proj()
        attn.kv_b_proj.backward_dw.assert_called_once()
        attn.kv_a_proj_with_mqa.backward_dw.assert_called_once()


class TestMLASelfAttentionBackwardQProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_q_proj."""

    def test_backward_q_proj_with_lora_rank(self):
        """_backward_q_proj should call q_a_proj and q_b_proj when q_lora_rank is not None."""
        config = MagicMock()
        config.q_lora_rank = 768
        attn = _make_mla_self_attn(config=config)
        object.__setattr__(attn, "q_a_proj", MagicMock())
        object.__setattr__(attn, "q_b_proj", MagicMock())
        attn._backward_q_proj()
        attn.q_a_proj.backward_dw.assert_called_once()
        attn.q_b_proj.backward_dw.assert_called_once()

    def test_backward_q_proj_without_lora_rank(self):
        """_backward_q_proj should call q_proj when q_lora_rank is None."""
        config = MagicMock()
        config.q_lora_rank = None
        attn = _make_mla_self_attn(config=config)
        object.__setattr__(attn, "q_proj", MagicMock())
        attn._backward_q_proj()
        attn.q_proj.backward_dw.assert_called_once()


class TestMLASelfAttentionBackwardOutputProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_output_proj."""

    def test_backward_output_proj_calls_o_proj(self):
        """_backward_output_proj should call backward_dw on o_proj."""
        attn = _make_mla_self_attn()
        object.__setattr__(attn, "o_proj", MagicMock())
        attn._backward_output_proj()
        attn.o_proj.backward_dw.assert_called_once()


class TestMLASelfAttentionSublayersSpecDefaults(unittest.TestCase):
    """Tests for MLASelfAttentionSublayersSpec defaults."""

    def test_default_q_a_layernorm_is_none(self):
        """q_a_layernorm should default to None."""
        spec = MLASelfAttentionSublayersSpec()
        self.assertIsNone(spec.q_a_layernorm)

    def test_default_kv_a_layernorm_is_none(self):
        """kv_a_layernorm should default to None."""
        spec = MLASelfAttentionSublayersSpec()
        self.assertIsNone(spec.kv_a_layernorm)


if __name__ == "__main__":
    unittest.main()
