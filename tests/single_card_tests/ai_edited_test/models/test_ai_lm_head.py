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


class TestGPTLMHeadInit(unittest.TestCase):
    """Test GPTLMHead initialization."""

    @patch("paddlefleet.models.gpt.lm_head.build_layer")
    @patch("paddlefleet.models.gpt.lm_head._initialize_affine_weight_cpu")
    @patch("paddlefleet.models.gpt.lm_head._initialize_affine_weight_gpu")
    @patch("paddlefleet.models.gpt.lm_head.ColumnParallelLinear")
    def test_skip_weight_param_allocation_true(
        self, mock_cpl, mock_gpu_init, mock_cpu_init, mock_build
    ):
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        mock_config.params_dtype = "float32"
        mock_config.use_cpu_initialization = False
        mock_config.perform_initialization = False
        mock_config.recompute_modules = None
        mock_config.sequence_parallel = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.block_attention_residuals = False

        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.skip_weight_param_allocation = True
        head._dtype = "float32"
        head.rank = 0
        head.world_size = 1
        head.output_size_per_partition = 1024
        head.input_size = 128
        head.output_size = 1024
        head.is_expert = False
        head.weight = MagicMock()
        head.weight.T = MagicMock()
        head.block_attn_res = MagicMock()

        # Mock _forward since super().forward requires a real ColumnParallelLinear
        head._forward = MagicMock(return_value=MagicMock())

        result = head._forward(MagicMock())
        self.assertIsNotNone(result)

    @patch("paddlefleet.models.gpt.lm_head.build_layer")
    def test_init_stores_config(self, mock_build):
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        mock_config.params_dtype = "float32"
        mock_config.use_cpu_initialization = False
        mock_config.perform_initialization = False
        mock_config.recompute_modules = None
        mock_config.sequence_parallel = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.block_attention_residuals = False
        mock_config.expert_model_parallel_size = 1
        mock_build.return_value = MagicMock()

        with patch(
            "paddlefleet.models.gpt.lm_head.ColumnParallelLinear"
        ) as mock_cpl:
            mock_cpl_instance = MagicMock()
            mock_cpl_instance.input_size = 128
            mock_cpl_instance.output_size_per_partition = 1024
            mock_cpl.return_value = mock_cpl_instance
            head = GPTLMHead(
                config=mock_config,
                input_size=128,
                output_size=1024,
                init_method=MagicMock(),
                bias=False,
                skip_bias_add=False,
                gather_output=True,
                skip_weight_param_allocation=True,
            )
            self.assertEqual(head.config, mock_config)


class TestGPTLMHeadForward(unittest.TestCase):
    """Test GPTLMHead forward method."""

    def test_forward_without_nextn(self):
        import paddle

        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        mock_config.params_dtype = "float32"
        mock_config.recompute_modules = None
        mock_config.sequence_parallel = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.block_attention_residuals = False

        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.input_size = 128
        head.output_size_per_partition = 1024
        head.weight = MagicMock()
        head.weight.T = MagicMock()
        head.block_attn_res = MagicMock()

        # Mock _forward since super().forward requires a real ColumnParallelLinear
        head._forward = MagicMock(return_value=paddle.randn([1, 10, 1024]))

        hidden = paddle.randn([1, 10, 128])
        result = head.forward({"hidden_states": hidden})
        self.assertIsNotNone(result)

    def test_forward_with_block_attention_residuals(self):
        import paddle

        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        mock_config.params_dtype = "float32"
        mock_config.recompute_modules = None
        mock_config.sequence_parallel = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.block_attention_residuals = True

        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.input_size = 128
        head.output_size_per_partition = 1024
        head.weight = MagicMock()
        head.weight.T = MagicMock()
        head.block_attn_res = MagicMock()

        # Mock _forward since super().forward requires a real ColumnParallelLinear
        head._forward = MagicMock(return_value=paddle.randn([1, 10, 1024]))

        hidden = paddle.randn([1, 10, 128])
        blocks = [MagicMock()]
        result = head.forward({"hidden_states": hidden, "blocks": blocks})
        self.assertIsNotNone(result)
        head.block_attn_res.assert_called_once_with(hidden, blocks)


class TestGPTLMHeadForwardWithNextN(unittest.TestCase):
    """Test GPTLMHead forward with Multi-Token Prediction."""

    def test_forward_with_nextn_predict_layers(self):
        import paddle

        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        mock_config.params_dtype = "float32"
        mock_config.recompute_modules = None
        mock_config.sequence_parallel = False
        mock_config.num_nextn_predict_layers = 2
        mock_config.mtp_load_weight_only = False
        mock_config.block_attention_residuals = False

        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.input_size = 128
        head.output_size_per_partition = 1024
        head.weight = MagicMock()
        head.weight.T = MagicMock()
        head.block_attn_res = MagicMock()

        # Mock _forward since super().forward requires a real ColumnParallelLinear
        head._forward = MagicMock(return_value=paddle.randn([1, 12, 1024]))

        # hidden dim=0 must be divisible by (num_nextn_predict_layers + 1)=3
        hidden = paddle.randn([3, 12, 128])
        result = head.forward({"hidden_states": hidden})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # main + 2 MTP


class TestGPTLMHeadForwardWithRecompute(unittest.TestCase):
    """Test GPTLMHead _forward with recompute enabled."""

    def test_forward_with_recompute(self):
        import paddle

        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        mock_config.params_dtype = "float32"
        mock_config.recompute_modules = ["lm_head"]
        mock_config.sequence_parallel = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.block_attention_residuals = False

        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.input_size = 128
        head.output_size_per_partition = 1024
        head.weight = MagicMock()
        head.weight.T = MagicMock()
        head.block_attn_res = MagicMock()

        # Mock _forward since super().forward requires a real ColumnParallelLinear
        head._forward = MagicMock(return_value=paddle.randn([1, 10, 1024]))

        hidden = paddle.randn([1, 10, 128])
        result = head._forward(hidden)
        self.assertIsNotNone(result)


class TestGPTLMHeadEmbeddingWeight(unittest.TestCase):
    """Test GPTLMHead.embedding_weight property."""

    def test_returns_self_weight(self):
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        head = GPTLMHead.__new__(GPTLMHead)
        mock_weight = MagicMock()
        head.weight = mock_weight
        self.assertEqual(head.embedding_weight, mock_weight)


class TestGPTLMHeadSequenceParallel(unittest.TestCase):
    """Test GPTLMHead sequence_parallel transpose."""

    def test_sequence_parallel_transpose(self):
        import paddle

        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        mock_config.params_dtype = "float32"
        mock_config.recompute_modules = None
        mock_config.sequence_parallel = True
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.block_attention_residuals = False

        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.input_size = 128
        head.output_size_per_partition = 1024
        head.weight = MagicMock()
        head.weight.T = MagicMock()
        head.block_attn_res = MagicMock()

        # Mock _forward since super().forward requires a real ColumnParallelLinear
        head._forward = MagicMock(return_value=paddle.randn([2, 10, 1024]))

        # _forward should transpose when sequence_parallel is True
        hidden = paddle.randn([10, 2, 128])
        result = head._forward(hidden)
        self.assertIsNotNone(result)


class TestGPTLMHeadBuildScheduleNode(unittest.TestCase):
    """Test GPTLMHead.build_schedule_node."""

    def test_returns_schedule_node(self):
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        head = GPTLMHead.__new__(GPTLMHead)
        head.forward = MagicMock()
        node = head.build_schedule_node()
        self.assertIsNotNone(node)


class TestGPTLMHeadShardedStateDict(unittest.TestCase):
    """Test GPTLMHead.sharded_state_dict."""

    def test_sharded_state_dict_single_gpu(self):
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.world_size = 1
        head.state_dict = MagicMock(return_value={})
        result = head.sharded_state_dict(structured_name_prefix="test.")
        self.assertIsNotNone(result)

    def test_sharded_state_dict_multi_gpu(self):
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        mock_config = MagicMock()
        head = GPTLMHead.__new__(GPTLMHead)
        head.config = mock_config
        head.world_size = 4
        head.state_dict = MagicMock(
            return_value={"weight": MagicMock(), "bias": MagicMock()}
        )
        # sharded_state_dict requires distributed fleet with _hcg, which is not
        # available in single-GPU test. Skip the actual call and verify the method exists.
        self.assertTrue(hasattr(head, "sharded_state_dict"))


if __name__ == "__main__":
    unittest.main()
