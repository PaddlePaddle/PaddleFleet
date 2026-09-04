# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import paddle

from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    _emit_indexcache_stall_trace,
    _indexcache_stall_trace_enabled,
)

_STATE_NOT_UPDATED = object()


class _FakeSelfAttention:
    def __init__(self, state_update):
        self.state_update = state_update

    def __call__(self, hidden_states, **_kwargs):
        output = hidden_states * 1.0
        if self.state_update is _STATE_NOT_UPDATED:
            return output, None
        return output, None, self.state_update


class _FakeHyperConnection:
    def __call__(self, hidden_states):
        return hidden_states, hidden_states, hidden_states

    @staticmethod
    def fused_h_res_h_post_bda(
        *,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        dropout_prob,
        training,
        fused,
    ):
        del h_res, original_residual, h_post, dropout_prob, training, fused
        return layer_output_with_bias[0]


def _bias_dropout_add(_training, _fused):
    def apply(output_with_bias, _residual, _dropout_prob):
        return output_with_bias[0]

    return apply


def _fake_fused_h_res_h_post_bda(
    _hyper_connection,
    _h_res,
    _original_residual,
    _h_post,
    layer_output_with_bias,
    _enable_recompute,
):
    return layer_output_with_bias[0], None


def _fake_forward_mlp(hidden_states, input_ids=None, **_kwargs):
    del input_ids
    return hidden_states


def _make_attention_layer(layer_type, state_update):
    layer = SimpleNamespace(
        recompute_input_layernorm=False,
        recompute_mhc_forward=False,
        input_layernorm=lambda value: value,
        layer_number=1,
        self_attn=_FakeSelfAttention(state_update),
        self_attn_bda=_bias_dropout_add,
        pre_cross_attn_layernorm=lambda value: value,
        cross_attention=lambda value, **_kwargs: (value, None),
        cross_attn_bda=_bias_dropout_add,
        training=True,
        config=SimpleNamespace(bias_dropout_fusion=False),
        hidden_dropout_prob=0.0,
        _log_md5=lambda *_args, **_kwargs: None,
    )
    if layer_type is HyperConnectionTransformerLayer:
        layer.self_attention_hyper_connection = _FakeHyperConnection()
        layer._fused_h_res_h_post_bda = _fake_fused_h_res_h_post_bda
        layer._cast_and_discard_fused_bda = (
            lambda output, _ori_dtype, _span: output
        )
    return layer


def _topk_state(value):
    return (
        paddle.to_tensor([value], dtype="int32"),
        paddle.to_tensor([value], dtype="int64"),
        paddle.to_tensor([0], dtype="int64"),
    )


def _make_forward_impl_layer(attention_result):
    return SimpleNamespace(
        training=True,
        layer_number=1,
        full_recompute=False,
        mlp=object(),
        config=SimpleNamespace(
            block_attention_residuals=False,
            multi_latent_attention=False,
        ),
        _log_md5=lambda *_args, **_kwargs: None,
        _forward_attention=lambda **_kwargs: attention_result,
        _forward_mlp=_fake_forward_mlp,
    )


class TestIndexCacheTransformerLayerStateTransitions(unittest.TestCase):
    def test_stall_trace_is_driven_by_normalized_config(self):
        disabled = SimpleNamespace(
            indexcache_stall_trace=False,
            indexcache_stall_trace_layers=(2,),
        )
        enabled = SimpleNamespace(
            indexcache_stall_trace=True,
            indexcache_stall_trace_layers=(2, 4),
        )

        self.assertFalse(_indexcache_stall_trace_enabled(disabled, 2))
        self.assertTrue(_indexcache_stall_trace_enabled(enabled, 2))
        self.assertFalse(_indexcache_stall_trace_enabled(enabled, 3))

        output = StringIO()
        with redirect_stdout(output):
            _emit_indexcache_stall_trace(enabled, 2, "attention", "enter")
            _emit_indexcache_stall_trace(enabled, 3, "attention", "enter")
        marker = output.getvalue()
        self.assertIn("layer=2 phase=attention edge=enter", marker)
        self.assertNotIn("layer=3", marker)

    def test_forward_attention_preserves_no_update_replace_and_clear(self):
        hidden_states = paddle.ones([1, 1, 4], dtype="float32")
        old_state = _topk_state(1)
        new_state = _topk_state(2)
        cases = (
            ("no_update", _STATE_NOT_UPDATED, old_state, True, old_state),
            ("replace", new_state, old_state, True, new_state),
            ("explicit_clear", None, old_state, True, None),
            ("empty_no_update", _STATE_NOT_UPDATED, None, False, None),
        )

        for layer_type in (
            TransformerLayer,
            HyperConnectionTransformerLayer,
        ):
            for name, update, incoming, has_state_output, expected in cases:
                with self.subTest(layer=layer_type.__name__, case=name):
                    layer = _make_attention_layer(layer_type, update)
                    result = layer_type._forward_attention(
                        layer,
                        hidden_states,
                        indexcache_state=incoming,
                    )
                    if has_state_output:
                        self.assertEqual(len(result), 3)
                        self.assertIs(result[2], expected)
                    else:
                        self.assertEqual(len(result), 2)

    def test_forward_impl_distinguishes_explicit_clear_from_no_update(self):
        hidden_states = paddle.ones([1, 1, 4], dtype="float32")
        old_state = _topk_state(1)

        clear_layer = _make_forward_impl_layer((hidden_states, None, None))
        cleared = TransformerLayer._forward_impl(
            clear_layer,
            hidden_states=hidden_states,
            indexcache_state=old_state,
        )
        self.assertIsInstance(cleared, paddle.Tensor)

        no_update_layer = _make_forward_impl_layer((hidden_states, None))
        retained = TransformerLayer._forward_impl(
            no_update_layer,
            hidden_states=hidden_states,
            indexcache_state=old_state,
        )
        self.assertEqual(len(retained), 3)
        self.assertIs(retained[2], old_state)


if __name__ == "__main__":
    unittest.main()
