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


class TestLayerDescInit(unittest.TestCase):
    """Tests for LayerDesc initialization and build."""

    @patch("paddlefleet.pipeline_parallel.pp_layers.build_layer")
    def test_layer_desc_build_layer(self, mock_build):
        from paddlefleet.pipeline_parallel.pp_layers import LayerDesc

        mock_spec = MagicMock()
        mock_spec.extra_kwargs = {"a": 1}
        # Need isinstance check to pass
        from paddlefleet import spec_utils

        mock_spec.__class__ = spec_utils.LayerSpec
        mock_layer = MagicMock()
        mock_build.return_value = mock_layer
        desc = LayerDesc(mock_spec, b=2)

        result = desc.build_layer(c=3)
        mock_build.assert_called_once()
        self.assertEqual(mock_spec.extra_kwargs, {"a": 1, "b": 2, "c": 3})


class TestSharedLayerDescInit(unittest.TestCase):
    """Tests for SharedLayerDesc initialization."""

    def test_shared_layer_desc_str_attr(self):
        from paddlefleet.pipeline_parallel.pp_layers import SharedLayerDesc

        mock_spec = MagicMock()
        mock_spec.extra_kwargs = {}
        from paddlefleet import spec_utils

        mock_spec.__class__ = spec_utils.LayerSpec
        desc = SharedLayerDesc(
            key="shared_emb",
            layer_spec=mock_spec,
            shared_weight_attr="weight",
        )
        self.assertEqual(desc.layer_name, "shared_emb")
        self.assertEqual(desc.shared_weight_attr, ["weight"])

    def test_shared_layer_desc_list_attr(self):
        from paddlefleet.pipeline_parallel.pp_layers import SharedLayerDesc

        mock_spec = MagicMock()
        mock_spec.extra_kwargs = {}
        from paddlefleet import spec_utils

        mock_spec.__class__ = spec_utils.LayerSpec
        desc = SharedLayerDesc(
            key="shared_layer",
            layer_spec=mock_spec,
            shared_weight_attr=["weight", "bias"],
        )
        self.assertEqual(desc.shared_weight_attr, ["weight", "bias"])

    def test_shared_layer_desc_with_forward_func(self):
        from paddlefleet.pipeline_parallel.pp_layers import SharedLayerDesc

        mock_spec = MagicMock()
        mock_spec.extra_kwargs = {}
        from paddlefleet import spec_utils

        mock_spec.__class__ = spec_utils.LayerSpec
        fn = MagicMock()
        desc = SharedLayerDesc(
            key="test",
            layer_spec=mock_spec,
            forward_func=fn,
            shared_weight_attr="weight",
        )
        self.assertEqual(desc.forward_func, fn)

    def test_shared_layer_desc_invalid_attr_type(self):
        from paddlefleet.pipeline_parallel.pp_layers import SharedLayerDesc

        mock_spec = MagicMock()
        mock_spec.extra_kwargs = {}
        from paddlefleet import spec_utils

        mock_spec.__class__ = spec_utils.LayerSpec
        with self.assertRaises(AssertionError):
            SharedLayerDesc(
                key="test",
                layer_spec=mock_spec,
                shared_weight_attr=123,
            )

    def test_shared_layer_desc_list_with_non_str(self):
        from paddlefleet.pipeline_parallel.pp_layers import SharedLayerDesc

        mock_spec = MagicMock()
        mock_spec.extra_kwargs = {}
        from paddlefleet import spec_utils

        mock_spec.__class__ = spec_utils.LayerSpec
        with self.assertRaises(AssertionError):
            SharedLayerDesc(
                key="test",
                layer_spec=mock_spec,
                shared_weight_attr=[123],
            )

    def test_shared_layer_desc_is_subclass(self):
        from paddlefleet.pipeline_parallel.pp_layers import (
            LayerDesc,
            SharedLayerDesc,
        )

        self.assertTrue(issubclass(SharedLayerDesc, LayerDesc))


class TestSegmentLayersInit(unittest.TestCase):
    """Tests for SegmentLayers initialization."""

    def test_segment_layers_uniform(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(12)]
        seg = SegmentLayers(mock_descs, num_parts=4, method="uniform")
        self.assertEqual(seg.num_parts, 4)
        self.assertEqual(seg.num_items, 12)

    def test_segment_layers_with_vpp(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(12)]
        seg = SegmentLayers(
            mock_descs,
            num_parts=4,
            method="uniform",
            num_virtual_pipeline_stage=2,
        )
        self.assertEqual(seg.total_parts, 8)

    def test_segment_layers_too_few_items(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(2)]
        with self.assertRaises(AssertionError):
            SegmentLayers(mock_descs, num_parts=4, method="uniform")

    def test_segment_layers_no_vpp(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(8)]
        seg = SegmentLayers(mock_descs, num_parts=2, method="uniform")
        self.assertIsNone(seg.num_virtual_pipeline_stage)


class TestSegmentLayersDoSegment(unittest.TestCase):
    """Tests for SegmentLayers.do_segment."""

    def test_uniform_segmentation(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(10)]
        seg = SegmentLayers(mock_descs, num_parts=2, method="uniform")
        result = seg.do_segment()
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 10)

    def test_uniform_segmentation_exact_division(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(8)]
        seg = SegmentLayers(mock_descs, num_parts=4, method="uniform")
        result = seg.do_segment()
        self.assertEqual(result, [0, 2, 4, 6, 8])

    def test_list_segmentation(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(10)]
        seg = SegmentLayers(mock_descs, num_parts=3, method=[0, 3, 7])
        result = seg.do_segment()
        self.assertEqual(result, [0, 3, 7, 10])

    def test_unsupported_method(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(10)]
        seg = SegmentLayers(mock_descs, num_parts=2, method="unknown")
        with self.assertRaises(ValueError):
            seg.do_segment()

    def test_list_segmentation_invalid_start(self):
        from paddlefleet.pipeline_parallel.pp_layers import SegmentLayers

        mock_descs = [MagicMock() for _ in range(10)]
        seg = SegmentLayers(mock_descs, num_parts=2, method=[1, 5])
        with self.assertRaises(AssertionError):
            seg.do_segment()


class TestPipelineLayerChunk(unittest.TestCase):
    """Tests for PipelineLayerChunk."""

    def test_init_run_function(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineLayerChunk

        chunk = PipelineLayerChunk()
        self.assertEqual(chunk.run_function, [])

    def test_extend(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineLayerChunk

        chunk = PipelineLayerChunk()
        layers = [MagicMock() for _ in range(3)]
        chunk.extend(layers)
        self.assertEqual(len(chunk.run_function), 3)

    def test_get_run_function(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineLayerChunk

        chunk = PipelineLayerChunk()
        fn = MagicMock()
        chunk.append(fn)
        result = chunk.get_run_function()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], fn)

    def test_iter(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineLayerChunk

        chunk = PipelineLayerChunk()
        items = [MagicMock() for _ in range(3)]
        for item in items:
            chunk.append(item)
        collected = list(chunk)
        self.assertEqual(len(collected), 3)

    def test_forward_raises(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineLayerChunk

        chunk = PipelineLayerChunk()
        with self.assertRaises(PermissionError):
            chunk.forward()


class TestPipelineSublayers(unittest.TestCase):
    """Tests for PipelineSublayers."""

    def test_run_function_stored(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineSublayers

        run_fn = [MagicMock() for _ in range(2)]
        sub = PipelineSublayers.__new__(PipelineSublayers)
        sub.run_function = run_fn
        self.assertEqual(len(sub.run_function), 2)

    def test_iter(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineSublayers

        run_fn = [MagicMock() for _ in range(3)]
        sub = PipelineSublayers.__new__(PipelineSublayers)
        sub.run_function = run_fn
        collected = list(sub)
        self.assertEqual(len(collected), 3)

    def test_iter_empty(self):
        from paddlefleet.pipeline_parallel.pp_layers import PipelineSublayers

        sub = PipelineSublayers.__new__(PipelineSublayers)
        sub.run_function = []
        collected = list(sub)
        self.assertEqual(len(collected), 0)


if __name__ == "__main__":
    unittest.main()
