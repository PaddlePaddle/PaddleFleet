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

"""Two-rank PP regression for IndexCache F -> S state transport.

Run with:
    PYTHONPATH=.:./src python -m paddle.distributed.launch --devices 0,1 \
        tests/multi_card_tests/pipeline_parallel/test_indexcache_pipeline.py
"""

import unittest
from types import SimpleNamespace

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    PipelineParallel,
)
from paddle.distributed.fleet.meta_parallel.pp_utils import (
    forward_backward_overlap_utils as fbo,
    utils as paddle_utils,
)

from paddlefleet.distributed.model import distributed_model
from paddlefleet.pipeline_parallel.indexcache_adapter import (
    IndexCachePipelineLayer,
)
from paddlefleet.transformer.indexcache_state import apply_stop_gradient_mask


class _IndexCacheProducer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.producer_scale = self.create_parameter(
            shape=[1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, inputs):
        return self._outputs(self.producer_scale, inputs["hidden_states"])

    @staticmethod
    def _outputs(scale, hidden_states):
        batch, seq = hidden_states.shape
        logits = paddle.stack([scale, -scale], axis=-1).reshape([1, 1, 2])
        topk_probs = F.softmax(logits, axis=-1).expand([batch, seq, 2])
        topk_indices = paddle.arange(batch * seq * 2, dtype="int32").reshape(
            [batch, seq, 2]
        )
        state = apply_stop_gradient_mask(
            (
                topk_indices,
                paddle.zeros([1], dtype="float32"),
                paddle.zeros([1], dtype="float32"),
                paddle.zeros([1], dtype="float32"),
                paddle.zeros([1], dtype="int32"),
                topk_probs,
                paddle.full([1], 1, dtype="int64"),
                paddle.full([1], 1, dtype="int64"),
            )
        )
        return {
            "hidden_states": hidden_states,
            "indexcache_state": state,
        }


class _IndexCacheServed(nn.Layer):
    def forward(self, inputs):
        state = inputs["indexcache_state"]
        hidden_states = inputs["hidden_states"]

        assert isinstance(state, tuple)
        assert len(state) == 8
        expected_shapes = (
            [1, 2, 2],
            [1],
            [1],
            [1],
            [1],
            [1, 2, 2],
            [1],
            [1],
        )
        expected_dtypes = (
            paddle.int32,
            paddle.float32,
            paddle.float32,
            paddle.float32,
            paddle.int32,
            paddle.float32,
            paddle.int64,
            paddle.int64,
        )
        assert tuple(list(tensor.shape) for tensor in state) == expected_shapes
        assert tuple(tensor.dtype for tensor in state) == expected_dtypes
        assert tuple(tensor.stop_gradient for tensor in state) == (
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
        )
        expected_topk = paddle.arange(4, dtype="int32").reshape([1, 2, 2])
        assert bool(paddle.equal_all(state[0], expected_topk))

        # Keep a zero-valued hidden-state edge so the test also exercises a
        # regular pipeline tensor alongside the differentiable state-5 edge.
        return hidden_states * 0.0 + state[5][..., 0]


class _RecomputedProducer(_IndexCacheProducer):
    def forward(self, inputs):
        from paddle.distributed.fleet.recompute import recompute

        def body(scale, hidden):
            result = self._outputs(scale, hidden)
            return tuple(
                t.clone()
                for t in (result["hidden_states"], *result["indexcache_state"])
            )

        result = recompute(
            body,
            self.producer_scale,
            inputs["hidden_states"],
            use_reentrant=True,
        )
        return {
            "hidden_states": result[0],
            "indexcache_state": apply_stop_gradient_mask(result[1:]),
        }


class _ReplacingProducer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.local_scale = self.create_parameter(
            [1], default_initializer=nn.initializer.Constant(1.0)
        )

    def forward(self, inputs):
        # A new producer ends the upstream producer's gradient lifetime.
        return inputs["hidden_states"] * 0 + self.local_scale


class _MeanLoss(nn.Layer):
    def forward(self, output, _labels):
        return output.mean()


class TestIndexCachePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if dist.get_world_size() != 2:
            raise unittest.SkipTest("IndexCache PP regression requires 2 ranks")

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": 2,
            "micro_batch_size": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_f_to_s_state_and_gradient_cross_stage(self):
        self._run_case(_IndexCacheProducer, _IndexCacheServed, False)

    def test_recomputed_producer_gradient_cross_stage(self):
        self._run_case(_RecomputedProducer, _IndexCacheServed, False)

    def test_replaced_producer_sends_zero_gradient(self):
        self._run_case(_IndexCacheProducer, _ReplacingProducer, True)

    def _run_case(self, producer_type, served_type, zero_gradient):
        originals = (
            paddle_utils.dict_to_tuple_helper,
            paddle_utils.tuple_to_dict_helper,
            PipelineParallel._backward_step,
            fbo.ScheduleNode.backward,
            fbo.detach_and_requires_grad,
            fbo.clone_and_clear_dataptr,
        )
        model = IndexCachePipelineLayer(
            layers=[
                LayerDesc(producer_type),
                LayerDesc(served_type),
            ],
            num_stages=2,
            loss_fn=_MeanLoss(),
        )
        model.config = SimpleNamespace(indexcache_topk_pattern="FS")
        pipeline = distributed_model(model)
        inputs = {"hidden_states": paddle.ones([2, 2], dtype="float32")}
        labels = paddle.zeros([2, 1], dtype="float32")

        loss = pipeline.forward_backward_pipeline((inputs, labels))
        self.assertIsInstance(loss, paddle.Tensor)

        stage_id = fleet.get_hybrid_communicate_group().get_stage_id()
        if stage_id == 0:
            producer_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if name.endswith("producer_scale")
            ]
            self.assertEqual(len(producer_parameters), 1)
            producer_grad = producer_parameters[0].grad
            self.assertIsNotNone(producer_grad)
            reference = paddle.to_tensor([0.0], stop_gradient=False)
            # forward_backward_pipeline accumulates the two micro-batch losses.
            reference_loss = (
                2
                * F.softmax(
                    paddle.stack([reference, -reference], axis=-1), axis=-1
                )[..., 0].mean()
            )
            reference_loss.backward()
            self.assertAlmostEqual(
                float(producer_grad.item()),
                0.0 if zero_gradient else float(reference.grad.item()),
                places=5,
            )

        evaluated = pipeline.eval_batch((inputs, labels), compute_loss=True)
        self.assertTrue(bool(paddle.isfinite(evaluated).all()))
        baseline = PipelineLayer(
            layers=[LayerDesc(nn.Linear, 2, 2), LayerDesc(nn.Linear, 2, 2)],
            num_stages=2,
            loss_fn=_MeanLoss(),
        )
        ordinary = distributed_model(baseline)
        self.assertIs(type(ordinary), PipelineParallel)
        baseline_loss = ordinary.forward_backward_pipeline(
            (paddle.ones([2, 2]), labels)
        )
        self.assertTrue(bool(paddle.isfinite(baseline_loss).all()))
        self.assertEqual(
            originals,
            (
                paddle_utils.dict_to_tuple_helper,
                paddle_utils.tuple_to_dict_helper,
                PipelineParallel._backward_step,
                fbo.ScheduleNode.backward,
                fbo.detach_and_requires_grad,
                fbo.clone_and_clear_dataptr,
            ),
        )
        dist.barrier()


if __name__ == "__main__":
    unittest.main(failfast=True)
