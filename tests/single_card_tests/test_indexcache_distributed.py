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

"""Native PP delegation and IndexCache tensor/state contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle
from paddle.distributed.fleet.meta_parallel import (
    PipelineLayer,
    PipelineParallel,
)
from paddle.distributed.fleet.meta_parallel.pp_utils import (
    utils as paddle_utils,
)

from paddlefleet.distributed.model import distributed_model
from paddlefleet.models.gpt.gpt_model import GPTModel
from paddlefleet.transformer.csa_attention import CompressedSparseAttention
from paddlefleet.transformer.indexcache_state import apply_stop_gradient_mask


class TestIndexCacheNativePipeline(unittest.TestCase):
    def test_gpt_uses_native_pipeline_forward(self):
        self.assertIn(PipelineLayer, GPTModel.__bases__)
        self.assertIs(GPTModel.forward, PipelineLayer.forward)

    def test_native_codec_preserves_state_tensors_and_gradient_mask(self):
        for size in (3, 8):
            with self.subTest(size=size):
                state = apply_stop_gradient_mask(
                    tuple(paddle.ones([1]) for _ in range(size))
                )
                encoded = paddle_utils.dict_to_tuple_helper(
                    {"hidden": paddle.ones([1]), "indexcache_state": state}
                )
                decoded, is_dict = paddle_utils.tuple_to_dict_helper(encoded)
                self.assertTrue(is_dict)
                for i, tensor in enumerate(decoded["indexcache_state"]):
                    self.assertIs(tensor, state[i])
                    self.assertEqual(
                        tensor.stop_gradient, not (size == 8 and i == 5)
                    )

    def test_state_lifetime_stops_before_next_producer(self):
        # Output liveness is independent of the PP partition.
        for pattern, expected in (
            ("F", [False]),
            ("FF", [False, False]),
            ("FSF", [True, False, False]),
            ("FSSF", [True, True, False, False]),
            ("FSFS", [True, False, True, False]),
        ):
            with self.subTest(pattern=pattern):
                actual = [
                    CompressedSparseAttention._indexcache_has_future_served_layer(
                        pattern, i
                    )
                    for i in range(len(pattern))
                ]
                self.assertEqual(actual, expected)

    def test_disabled_factory_delegates_without_patching(self):
        original = PipelineParallel._backward_step
        model = SimpleNamespace(
            config=SimpleNamespace(indexcache_topk_pattern=None)
        )
        with patch(
            "paddlefleet.distributed.model.fleet.distributed_model",
            return_value=model,
        ) as native:
            self.assertIs(distributed_model(model), model)
            native.assert_called_once_with(model)
        self.assertIs(PipelineParallel._backward_step, original)

    def test_enabled_factory_delegates_and_rejects_unsupported_modes(self):
        model = object.__new__(PipelineLayer)
        paddle.nn.Layer.__init__(model)
        model.config = SimpleNamespace(indexcache_topk_pattern="FS")
        model._num_virtual_pipeline_stages = 1
        pp_config = SimpleNamespace(
            use_dualpipev=False, forward_backward_overlap_scheduler=False
        )
        strategy = SimpleNamespace(
            amp=False, hybrid_configs={"pp_configs": pp_config}
        )
        hcg = SimpleNamespace(get_pipe_parallel_world_size=lambda: 2)
        marker = object()
        with (
            patch(
                "paddlefleet.distributed.model.fleet.fleet._user_defined_strategy",
                strategy,
                create=True,
            ),
            patch(
                "paddlefleet.distributed.model.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            patch(
                "paddlefleet.distributed.model.fleet.distributed_model",
                return_value=marker,
            ) as native,
        ):
            self.assertIs(distributed_model(model), marker)
            native.assert_called_once_with(model)
            native.reset_mock()
            for attribute in (
                "use_dualpipev",
                "forward_backward_overlap_scheduler",
            ):
                setattr(pp_config, attribute, True)
                with self.assertRaises(NotImplementedError):
                    distributed_model(model)
                setattr(pp_config, attribute, False)
            model._num_virtual_pipeline_stages = 2
            with self.assertRaisesRegex(NotImplementedError, "VPP=1"):
                distributed_model(model)
            model._num_virtual_pipeline_stages = 1
            strategy.amp = True
            with self.assertRaisesRegex(
                NotImplementedError, "Trainer-managed AMP"
            ):
                distributed_model(model)
            native.assert_not_called()
            hcg.get_pipe_parallel_world_size = lambda: 1
            self.assertIs(distributed_model(model), marker)
            native.assert_called_once_with(model)
