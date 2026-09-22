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

"""Instance-scoped IndexCache pipeline boundaries and gradient contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle
from paddle.distributed.fleet.meta_parallel import pipeline_parallel
from paddle.distributed.fleet.meta_parallel.pp_utils import (
    forward_backward_overlap_utils as fbo,
    utils as paddle_utils,
)

from paddlefleet.distributed.model import distributed_model
from paddlefleet.pipeline_parallel.indexcache_adapter import (
    IndexCachePipelineLayer,
    IndexCachePipelineParallel,
    _normalize_pipeline_input_gradients,
    prepare_indexcache_pipeline_boundary,
)


def make_state(distill=True):
    if not distill:
        return (
            paddle.zeros([1, 2], dtype="int32"),
            paddle.ones([1], dtype="int64"),
            paddle.ones([1], dtype="int64"),
        )
    return tuple(
        paddle.ones([1, 2], dtype="float32")
        if i == 5
        else paddle.ones([1], dtype="int64")
        for i in range(8)
    )


class TestIndexCachePipelineBoundary(unittest.TestCase):
    def test_native_codec_roundtrip_preserves_leaves_and_masks(self):
        for distill in (False, True):
            with self.subTest(distill=distill):
                state = make_state(distill)
                source = {
                    "hidden": paddle.ones([1]),
                    "indexcache_state": state,
                    "unused": None,
                }
                encoded = paddle_utils.dict_to_tuple_helper(
                    prepare_indexcache_pipeline_boundary(source)
                )
                decoded, is_dict = paddle_utils.tuple_to_dict_helper(encoded)
                self.assertTrue(is_dict)
                result = prepare_indexcache_pipeline_boundary(decoded)
                self.assertNotIn("unused", result)
                self.assertIsInstance(result["indexcache_state"], tuple)
                for index, tensor in enumerate(result["indexcache_state"]):
                    self.assertIs(tensor, state[index])
                    self.assertEqual(
                        tensor.stop_gradient, not (distill and index == 5)
                    )
                self.assertIn("unused", source)

    def test_missing_state_gradient_uses_preserved_metadata(self):
        state = make_state()
        encoded = paddle_utils.dict_to_tuple_helper(
            prepare_indexcache_pipeline_boundary({"indexcache_state": state})
        )
        decoded, _ = paddle_utils.tuple_to_dict_helper(encoded)
        prepare_indexcache_pipeline_boundary(decoded)
        state[5]._clear_dataptr()
        grads = _normalize_pipeline_input_gradients(encoded, (None,))
        self.assertEqual(list(grads[0].shape), [1, 2])
        self.assertEqual(grads[0].dtype, paddle.float32)
        self.assertEqual(float(grads[0].sum()), 0.0)

    def test_missing_regular_gradient_still_raises(self):
        hidden = paddle.ones([1])
        hidden.stop_gradient = False
        encoded = paddle_utils.dict_to_tuple_helper(
            prepare_indexcache_pipeline_boundary(
                {"hidden": hidden, "indexcache_state": make_state()}
            )
        )
        with self.assertRaisesRegex(RuntimeError, "outside IndexCache"):
            _normalize_pipeline_input_gradients(encoded, (None, None))

    def test_bad_gradient_arity_is_rejected(self):
        encoded = paddle_utils.dict_to_tuple_helper(
            prepare_indexcache_pipeline_boundary(
                {"indexcache_state": make_state()}
            )
        )
        with self.assertRaisesRegex(RuntimeError, "arity"):
            _normalize_pipeline_input_gradients(encoded, ())

    def test_standard_model_factory_delegates_without_patching(self):
        original = (
            paddle_utils.dict_to_tuple_helper,
            paddle_utils.tuple_to_dict_helper,
            pipeline_parallel.PipelineParallel._backward_step,
            fbo.ScheduleNode.backward,
            fbo.detach_and_requires_grad,
            fbo.clone_and_clear_dataptr,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(indexcache_topk_pattern=None)
        )
        with patch(
            "paddlefleet.distributed.model.fleet.distributed_model",
            return_value=model,
        ) as delegate:
            self.assertIs(distributed_model(model), model)
            delegate.assert_called_once_with(model)
        self.assertEqual(
            original,
            (
                paddle_utils.dict_to_tuple_helper,
                paddle_utils.tuple_to_dict_helper,
                pipeline_parallel.PipelineParallel._backward_step,
                fbo.ScheduleNode.backward,
                fbo.detach_and_requires_grad,
                fbo.clone_and_clear_dataptr,
            ),
        )

    def test_enabled_factory_selects_a_local_wrapper_and_rejects_unsupported_modes(
        self,
    ):
        model = object.__new__(IndexCachePipelineLayer)
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
                "paddlefleet.distributed.model.IndexCachePipelineParallel",
                return_value=marker,
            ) as build,
        ):
            self.assertIs(distributed_model(model), marker)
            build.assert_called_once_with(model, hcg, strategy=strategy)
            self.assertTrue(model._indexcache_instance_wrapper)
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

    def test_backward_override_calls_parent_and_preserves_regular_result(self):
        value = paddle.ones([1])
        with patch.object(
            pipeline_parallel.PipelineParallel,
            "_backward_step",
            return_value=value,
        ):
            self.assertIs(
                IndexCachePipelineParallel._backward_step(
                    object.__new__(IndexCachePipelineParallel), None
                ),
                value,
            )
