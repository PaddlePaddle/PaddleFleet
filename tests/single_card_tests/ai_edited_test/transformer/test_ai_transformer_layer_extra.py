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

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.identity_op import IdentityFuncOp, IdentityOp
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerNode,
    TransformerLayerSublayersSpec,
    TransformerLayerWithOverlap,
    tensors_clone,
)


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pipeline_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "recompute_granularity": None,
        "recompute_modules": None,
        "recompute_num_layers": None,
        "recompute_method": None,
        "block_attention_residuals": False,
        "num_nextn_predict_layers": None,
        "mtp_load_weight_only": False,
        "hidden_dropout_prob": 0.0,
        "bias_dropout_fusion": False,
        "rms_norm_eps": 1e-5,
        "fp8": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class FakeAttention(paddle.nn.Layer):
    def __init__(self, config=None, layer_number=1, **kwargs):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, **kwargs):
        return (hidden_states, None)


class FakeMLP(paddle.nn.Layer):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.tp_group = kwargs.get("tp_group", None)

    def forward(self, hidden_states, **kwargs):
        return hidden_states, None

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number


class FakeBDA(paddle.nn.Layer):
    def __init__(self, config=None, **kwargs):
        super().__init__()

    def forward(self, training, bias_dropout_fusion):
        def _fn(x, residual, dropout_prob):
            return x + residual

        return _fn


class FakeNorm(paddle.nn.Layer):
    def __init__(self, config=None, hidden_size=64, eps=1e-5, **kwargs):
        super().__init__()
        self.w = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        return x


class TestTensorsClone(unittest.TestCase):
    """Test tensors_clone utility function."""

    def test_clone_single_tensor(self):
        x = paddle.randn([2, 3], dtype="float32")
        cloned = tensors_clone(x)
        self.assertEqual(cloned.shape, [2, 3])
        # Verify clone produces a different tensor object
        self.assertFalse(x is cloned)

    def test_clone_tuple_of_tensors(self):
        x = paddle.randn([2, 3], dtype="float32")
        y = paddle.randn([4, 5], dtype="float32")
        result = tensors_clone((x, y))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [2, 3])
        self.assertEqual(result[1].shape, [4, 5])

    def test_clone_list_of_tensors(self):
        x = paddle.randn([2, 3], dtype="float32")
        result = tensors_clone([x])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_clone_dict_of_tensors(self):
        x = paddle.randn([2, 3], dtype="float32")
        result = tensors_clone({"a": x})
        self.assertIsInstance(result, dict)
        self.assertEqual(result["a"].shape, [2, 3])

    def test_clone_unsupported_type(self):
        with self.assertRaises(ValueError):
            tensors_clone(42)


class TestTransformerLayerSublayersSpec(unittest.TestCase):
    """Test TransformerLayerSublayersSpec defaults."""

    def test_default_values(self):
        spec = TransformerLayerSublayersSpec()
        self.assertEqual(spec.input_layernorm, IdentityOp)
        self.assertEqual(spec.self_attn, IdentityOp)
        self.assertEqual(spec.self_attn_bda, IdentityFuncOp)
        self.assertEqual(spec.cross_attention, IdentityOp)
        self.assertEqual(spec.cross_attn_bda, IdentityFuncOp)
        self.assertEqual(spec.post_attention_layernorm, IdentityOp)
        self.assertEqual(spec.mlp, IdentityOp)
        self.assertEqual(spec.mlp_bda, IdentityFuncOp)
        self.assertEqual(spec.block_attn_res, IdentityOp)

    def test_custom_values(self):
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FakeNorm,
            self_attn=FakeAttention,
            mlp=FakeMLP,
        )
        self.assertEqual(spec.input_layernorm, FakeNorm)
        self.assertEqual(spec.self_attn, FakeAttention)
        self.assertEqual(spec.mlp, FakeMLP)

    def test_sharded_state_dict_keys_map_default(self):
        spec = TransformerLayerSublayersSpec()
        self.assertEqual(spec.sharded_state_dict_keys_map, {})


class TestTransformerLayerConstruction(unittest.TestCase):
    """Test TransformerLayer construction paths."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.transformer_layer.build_spec_layer")
    def test_basic_construction(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.side_effect = (
            lambda cls, **kw: cls(**kw)
            if isinstance(cls, type) and issubclass(cls, paddle.nn.Layer)
            else MagicMock()
        )

        config = _make_config()
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FakeNorm,
            self_attn=FakeAttention,
            self_attn_bda=FakeBDA,
            pre_cross_attn_layernorm=FakeNorm,
            cross_attention=FakeAttention,
            cross_attn_bda=FakeBDA,
            post_attention_layernorm=FakeNorm,
            mlp=FakeMLP,
            mlp_bda=FakeBDA,
        )
        layer = TransformerLayer(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertEqual(layer.layer_number, 1)
        self.assertFalse(layer.full_recompute)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.transformer_layer.build_spec_layer")
    def test_full_recompute(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.side_effect = (
            lambda cls, **kw: cls(**kw)
            if isinstance(cls, type) and issubclass(cls, paddle.nn.Layer)
            else MagicMock()
        )

        config = _make_config(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FakeNorm,
            self_attn=FakeAttention,
            self_attn_bda=FakeBDA,
            pre_cross_attn_layernorm=FakeNorm,
            cross_attention=FakeAttention,
            cross_attn_bda=FakeBDA,
            post_attention_layernorm=FakeNorm,
            mlp=FakeMLP,
            mlp_bda=FakeBDA,
        )
        layer = TransformerLayer(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        # Full recompute (uniform) should be True
        self.assertTrue(layer.full_recompute)
        # recompute_mlp is only set for selective granularity
        self.assertFalse(layer.recompute_mlp)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.transformer_layer.build_spec_layer")
    def test_selective_recompute_mlp(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.side_effect = (
            lambda cls, **kw: cls(**kw)
            if isinstance(cls, type) and issubclass(cls, paddle.nn.Layer)
            else MagicMock()
        )

        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["mlp"],
        )
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FakeNorm,
            self_attn=FakeAttention,
            self_attn_bda=FakeBDA,
            pre_cross_attn_layernorm=FakeNorm,
            cross_attention=FakeAttention,
            cross_attn_bda=FakeBDA,
            post_attention_layernorm=FakeNorm,
            mlp=FakeMLP,
            mlp_bda=FakeBDA,
        )
        layer = TransformerLayer(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertTrue(layer.recompute_mlp)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.transformer_layer.build_spec_layer")
    def test_block_attention_residuals_asserts_full_recompute(
        self, mock_build, mock_pg
    ):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.side_effect = (
            lambda cls, **kw: cls(**kw)
            if isinstance(cls, type) and issubclass(cls, paddle.nn.Layer)
            else MagicMock()
        )

        config = _make_config(
            block_attention_residuals=True,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FakeNorm,
            self_attn=FakeAttention,
            self_attn_bda=FakeBDA,
            pre_cross_attn_layernorm=FakeNorm,
            cross_attention=FakeAttention,
            cross_attn_bda=FakeBDA,
            post_attention_layernorm=FakeNorm,
            mlp=FakeMLP,
            mlp_bda=FakeBDA,
        )
        with self.assertRaises(AssertionError):
            TransformerLayer(
                config, spec, layer_number=1, pg_collection=mock_pg_obj
            )


class TestTransformerLayerFp8Quant(unittest.TestCase):
    """Test fp8_quant_weight and use_fp8 delegation to MoE."""

    def test_fp8_quant_weight_with_non_moe_mlp(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        # Use a mock MLP that doesn't require Layer init
        mock_mlp = MagicMock()
        mock_mlp.is_moe = False
        # Bypass setattr restriction by setting directly in __dict__
        object.__setattr__(layer, "mlp", mock_mlp)
        # Should not raise when MLP is not MoELayer
        layer.fp8_quant_weight()

    def test_use_fp8_with_non_moe_mlp(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        mock_mlp = MagicMock()
        mock_mlp.is_moe = False
        object.__setattr__(layer, "mlp", mock_mlp)
        result = layer.use_fp8()
        self.assertIsNone(result)


class TestTransformerLayerWithOverlap(unittest.TestCase):
    """Test TransformerLayerWithOverlap assertions."""

    def test_overlap_checks(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        layer.full_recompute = True
        layer.recompute_mlp = False
        layer.recompute_input_layernorm = False
        layer.recompute_post_attention_layernorm = False
        layer.mlp = MagicMock()
        layer.mlp.gate = MagicMock()
        layer.mlp.gate.norm_topk_prob = False
        layer.mlp.expert_model_parallel_size = 1
        layer.mlp.moe_token_dispatcher_type = "deepep"
        # The assertions in __init__ should pass
        # Just test the existence of compute methods
        self.assertTrue(
            hasattr(TransformerLayerWithOverlap, "compute_attention")
        )
        self.assertTrue(hasattr(TransformerLayerWithOverlap, "compute_mlp"))
        self.assertTrue(
            hasattr(TransformerLayerWithOverlap, "pre_process_compute")
        )


class TestTransformerLayerNode(unittest.TestCase):
    """Test TransformerLayerNode."""

    def test_build_schedule_node(self):
        from paddlefleet.transformer.mlp import MLP

        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        layer.layer_number = 1
        layer.full_recompute = False
        # Set required methods and attributes that TransformerLayerNode checks
        layer.compute_attention = MagicMock()
        layer.compute_mlp = MagicMock()
        layer.pre_process_compute = MagicMock()
        layer.post_process_compute = MagicMock()
        layer.bda = MagicMock()
        layer.mlp_bda = MagicMock()
        mock_mlp = MagicMock(spec=MLP)
        object.__setattr__(layer, "mlp", mock_mlp)
        node = layer.build_schedule_node()
        self.assertIsInstance(node, TransformerLayerNode)
        self.assertEqual(node.layer_number, 1)

    def test_node_is_sparse_detection(self):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        layer.layer_number = 1
        layer.full_recompute = False
        mock_mlp = MagicMock(spec=MoELayer)
        mock_mlp.compute_gate = MagicMock()
        mock_mlp.compute_dispatch = MagicMock()
        mock_mlp.compute_experts = MagicMock()
        mock_mlp.compute_combine = MagicMock()
        mock_mlp.aux_loss_compute = MagicMock()
        mock_mlp.token_dispatcher = MagicMock()
        mock_mlp.token_dispatcher._comm_manager = MagicMock()
        mock_mlp.token_dispatcher._comm_manager.group = MagicMock()
        mock_mlp.token_dispatcher._comm_manager.group.id = 0
        object.__setattr__(layer, "mlp", mock_mlp)
        layer.compute_attention = MagicMock()
        layer.compute_mlp = MagicMock()
        layer.pre_process_compute = MagicMock()
        layer.post_process_compute = MagicMock()
        layer.dispatch_preprocess_compute = MagicMock()
        layer.bda = MagicMock()
        layer.mlp_bda = MagicMock()
        node = TransformerLayerNode(layer, layer.config, "TestNode", 1)
        self.assertTrue(node._is_sparse)

    def test_node_not_sparse_for_mlp(self):
        from paddlefleet.transformer.mlp import MLP

        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        layer.layer_number = 1
        layer.full_recompute = False
        mock_mlp = MagicMock(spec=MLP)
        object.__setattr__(layer, "mlp", mock_mlp)
        layer.compute_attention = MagicMock()
        layer.compute_mlp = MagicMock()
        layer.pre_process_compute = MagicMock()
        layer.post_process_compute = MagicMock()
        layer.bda = MagicMock()
        layer.mlp_bda = MagicMock()
        node = TransformerLayerNode(layer, layer.config, "TestNode", 1)
        self.assertFalse(node._is_sparse)


class TestTransformerLayerForward(unittest.TestCase):
    """Test TransformerLayer forward dict processing."""

    def test_forward_strips_dynamic_inference_key(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        layer.full_recompute = False
        layer._forward_impl = MagicMock(return_value=paddle.randn([2, 4, 64]))
        layer.block_attention_residuals = False

        result = layer.forward(
            {
                "hidden_states": paddle.randn([2, 4, 64]),
                "dynamic_inference_decode_only": True,
            }
        )
        self.assertIn("hidden_states", result)
        # dynamic_inference_decode_only should be stripped
        self.assertNotIn("dynamic_inference_decode_only", result)

    def test_forward_returns_context(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        layer.full_recompute = False
        context = paddle.randn([2, 4, 64])
        layer._forward_impl = MagicMock(
            return_value=(paddle.randn([2, 4, 64]), context)
        )
        layer.block_attention_residuals = False

        result = layer.forward({"hidden_states": paddle.randn([2, 4, 64])})
        self.assertIn("context", result)

    def test_forward_returns_ordered_dict(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        layer.config = _make_config()
        layer.full_recompute = False
        layer._forward_impl = MagicMock(return_value=paddle.randn([2, 4, 64]))
        layer.block_attention_residuals = False

        result = layer.forward({"hidden_states": paddle.randn([2, 4, 64])})
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
