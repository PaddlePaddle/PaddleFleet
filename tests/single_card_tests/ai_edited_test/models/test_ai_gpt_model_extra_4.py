# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
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

from paddlefleet.models.gpt.gpt_model import (
    GPTModel,
    build_overlapped_nodes,
)


class TestGPTModelBuildOverlappedNodes(unittest.TestCase):
    """Tests for GPTModel build_overlapped_nodes."""

    def test_build_overlapped_nodes_with_no_overlap(self):
        """build_overlapped_nodes should work with no overlap layers."""
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        forward_chunk = ScheduleChunk([MagicMock()])
        backward_chunk = ScheduleChunk([MagicMock()])
        result = build_overlapped_nodes(forward_chunk, backward_chunk)
        self.assertEqual(len(result), 5)

    def test_build_overlapped_nodes_with_overlap(self):
        """build_overlapped_nodes should create overlap node when TransformerLayerNode exists."""
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        from paddlefleet.transformer.transformer_layer import (
            TransformerLayerNode,
        )

        node1 = TransformerLayerNode.__new__(TransformerLayerNode)
        node2 = TransformerLayerNode.__new__(TransformerLayerNode)
        forward_chunk = ScheduleChunk([node1])
        backward_chunk = ScheduleChunk([node2])
        result = build_overlapped_nodes(forward_chunk, backward_chunk)
        _, _, overlap, _, _ = result
        self.assertEqual(len(overlap.nodes), 1)


class TestGPTModelOverlappedForwardBackward(unittest.TestCase):
    """Tests for GPTModel.overlapped_forward_backward."""

    def test_forward_loss_none_when_no_loss_fn(self):
        """forward_loss should be None when forward_loss_fn_node is None."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            from paddle.distributed.fleet.meta_parallel import ScheduleChunk

            forward_chunk = ScheduleChunk([MagicMock()])
            backward_chunk = ScheduleChunk([MagicMock()])
            result = model.overlapped_forward_backward(
                forward_chunk=forward_chunk,
                forward_inputs=MagicMock(),
                forward_loss_fn_node=None,
                backward_chunk=backward_chunk,
                backward_loss_fn_node=None,
                backward_input_grads=None,
                scaler=None,
                p2p_async_handle=None,
            )
            _, forward_loss, _ = result
            self.assertIsNone(forward_loss)


class TestGPTModelOffloadReloadParams(unittest.TestCase):
    """Tests for GPTModel offload/reload weight-only params."""

    def test_offload_weight_only_params_calls_pin_memory(self):
        """offload_weight_only_params should move GPU params to CPU pinned memory."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            mock_param = MagicMock()
            mock_param.place.is_gpu_place.return_value = True
            mock_pin = MagicMock()
            mock_param.pin_memory.return_value = mock_pin

            with patch.object(
                model, "_get_weight_only_params", return_value=[mock_param]
            ):
                model.offload_weight_only_params()
                mock_param.pin_memory.assert_called_once()
                mock_pin._share_buffer_to.assert_called_once_with(mock_param)

    def test_reload_weight_only_params_calls_cuda(self):
        """reload_weight_only_params should move CPU params back to GPU."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            mock_param = MagicMock()
            mock_param.place.is_gpu_place.return_value = False
            mock_gpu = MagicMock()
            mock_param.cuda.return_value = mock_gpu

            with patch.object(
                model, "_get_weight_only_params", return_value=[mock_param]
            ):
                model.reload_weight_only_params()
                mock_param.cuda.assert_called_once()
                mock_gpu._share_buffer_to.assert_called_once_with(mock_param)


class TestGPTModelFP8VirtualPipeline(unittest.TestCase):
    """Tests for GPTModel.fp8_quant_weight with virtual pipeline stages."""

    def test_fp8_quant_with_virtual_pipeline(self):
        """fp8_quant_weight should iterate over model chunks with virtual pipeline."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._num_virtual_pipeline_stages = 2
            from paddlefleet.transformer.transformer_layer import (
                TransformerLayer,
            )

            mock_layer = MagicMock(spec=TransformerLayer)
            model._model_chunks = [[mock_layer], [MagicMock()]]
            model.fp8_quant_weight(batch_mode=True, quant_transpose=True)
            mock_layer.fp8_quant_weight.assert_called_once_with(
                batch_mode=True, quant_transpose=True
            )


if __name__ == "__main__":
    unittest.main()
