# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

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
from paddle.distributed.fleet import distributed_model
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

from paddlefleet.pipeline_parallel.indexcache_adapter import (
    register_indexcache_pipeline_adapter,
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
        hidden_states = inputs["hidden_states"]
        batch, seq = hidden_states.shape
        logits = paddle.stack(
            [self.producer_scale, -self.producer_scale], axis=-1
        ).reshape([1, 1, 2])
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
            "accumulate_steps": 1,
            "micro_batch_size": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        register_indexcache_pipeline_adapter(
            SimpleNamespace(
                index_topk_pattern="FS",
                indexcache_train_debug=False,
            )
        )

    def test_f_to_s_state_and_gradient_cross_stage(self):
        model = PipelineLayer(
            layers=[
                LayerDesc(_IndexCacheProducer),
                LayerDesc(_IndexCacheServed),
            ],
            num_stages=2,
            loss_fn=_MeanLoss(),
        )
        pipeline = distributed_model(model)
        inputs = {"hidden_states": paddle.ones([1, 2], dtype="float32")}
        labels = paddle.zeros([1, 1], dtype="float32")

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
            self.assertGreater(abs(float(producer_grad.item())), 0.0)

        dist.barrier()


if __name__ == "__main__":
    unittest.main()
