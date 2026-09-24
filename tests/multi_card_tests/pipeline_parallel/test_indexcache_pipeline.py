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

"""Native PP state lifetimes, gradients and recomputation on two or three ranks."""

import json
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
from paddle.distributed.fleet.recompute import recompute

from paddlefleet.distributed.model import distributed_model
from paddlefleet.transformer import indexcache_state as sm
from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    IndexCacheServedDistillLossAutoScaler as Served,
)

has_future = CompressedSparseAttention._indexcache_has_future_served_layer


class Step(nn.Layer):
    def __init__(
        self,
        idx,
        actions,
        distill=True,
        checkpoint=False,
        frozen=False,
        coeff=0.4,
        masked=False,
    ):
        super().__init__()
        self.idx = idx
        self.actions = actions
        self.action = actions[idx]
        self.pattern = actions.replace("H", "")
        self.ordinal = sum(x != "H" for x in actions[: idx + 1]) - 1
        self.distill = distill
        self.checkpoint = checkpoint
        self.frozen = frozen
        self.coeff = coeff
        self.masked = masked
        self.gain = self.create_parameter(
            [1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.2 + idx * 0.05),
        )
        self.gain.stop_gradient = frozen and self.action != "F"

    def body(self, gain, h, *slots):
        incoming = sm.apply_stop_gradient_mask(slots) if slots else None
        if self.action == "F":
            assert incoming is None, (
                "Dead state was transported to a replacing F"
            )
        if self.action == "S":
            assert incoming is not None, "Live state was not transported to S"
        # Freeze the backbone while preserving a trainable producer in the frozen case.
        hidden = h * 1.0 if self.frozen else h * (gain + 1.0)
        outgoing = incoming
        if self.action == "F":
            probs = (
                F.softmax(paddle.stack([gain, -gain], axis=-1), axis=-1)
                .reshape([1, 1, 2])
                .expand([h.shape[0], h.shape[1], 2])
            )
            idxs = paddle.zeros([h.shape[0], h.shape[1], 2], dtype="int32")
            if self.distill:
                outgoing = sm.apply_stop_gradient_mask(
                    (
                        idxs,
                        paddle.zeros([1]),
                        paddle.zeros([1]),
                        paddle.zeros([1]),
                        paddle.zeros([1], dtype="int32"),
                        probs,
                        paddle.full([1], self.idx, dtype="int64"),
                        paddle.ones([1], dtype="int64"),
                    )
                )
            else:
                outgoing = sm.apply_stop_gradient_mask(
                    (
                        idxs,
                        paddle.full([1], self.idx, dtype="int64"),
                        paddle.ones([1], dtype="int64"),
                    )
                )
        elif self.action == "S" and self.distill:
            probs = incoming[5]
            target = paddle.full_like(probs, 0.5)
            mask = paddle.zeros(probs.shape[:2]) if self.masked else None
            hidden = Served.apply(
                hidden,
                probs,
                target,
                self.coeff,
                float(probs.shape[0] * probs.shape[1]),
                mask,
            )
        if self.action != "H" and not has_future(self.pattern, self.ordinal):
            outgoing = None
        if self.idx == len(self.actions) - 1:
            assert outgoing is None
            return hidden.clone()
        if outgoing is None:
            return (hidden.clone(),)
        # Ordinary intermediate layers can forward the same tensor object.
        # Only the reentrant checkpoint path needs output clones for PyLayer.
        if self.action == "H" and not self.checkpoint:
            return (hidden.clone(), *outgoing)
        return (hidden.clone(), *sm.clone_state_outputs(outgoing))

    def forward(self, inputs):
        slots = inputs.get("indexcache_state") or ()
        args = (self.gain, inputs["hidden_states"], *slots)
        out = (
            recompute(self.body, *args, use_reentrant=True)
            if self.checkpoint and self.training
            else self.body(*args)
        )
        if self.idx == len(self.actions) - 1:
            return out
        result = {"hidden_states": out[0]}
        if len(out) > 1:
            result["indexcache_state"] = sm.apply_stop_gradient_mask(out[1:])
        return result


class MeanLoss(nn.Layer):
    def forward(self, out, labels):
        return out.mean()


def _run_native_pipeline_cases():
    world = dist.get_world_size()
    assert world in (2, 3)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": world,
    }
    strategy.pipeline_configs = {"accumulate_steps": 2, "micro_batch_size": 1}
    fleet.init(is_collective=True, strategy=strategy)
    rank = dist.get_rank()
    if world == 2:
        cases = [
            ("FS", [0, 1, 2], {}),
            ("FS_recompute", [0, 1, 2], {"checkpoint": True}),
            ("FS_frozen", [0, 1, 2], {"frozen": True}),
            (
                "FS_frozen_recompute",
                [0, 1, 2],
                {"frozen": True, "checkpoint": True},
            ),
            ("FS_topk_only", [0, 1, 2], {"distill": False}),
            ("FS_zero_coeff", [0, 1, 2], {"coeff": 0.0}),
            ("FS_all_masked", [0, 1, 2], {"masked": True}),
            ("FSF_cut_after_last_S", [0, 2, 3], {}),
            ("FFS_cut_before_new_F", [0, 1, 3], {}),
            ("FSSF", [0, 2, 4], {"checkpoint": True}),
            ("FF", [0, 1, 2], {}),
        ]
    else:
        cases = [
            ("FHS_passthrough", [0, 1, 2, 3], {}),
            ("FHS_passthrough_recompute", [0, 1, 2, 3], {"checkpoint": True}),
            ("FHS_passthrough_frozen", [0, 1, 2, 3], {"frozen": True}),
            (
                "FHS_passthrough_frozen_recompute",
                [0, 1, 2, 3],
                {"frozen": True, "checkpoint": True},
            ),
            ("FHS_topk_only", [0, 1, 2, 3], {"distill": False}),
            ("FSHFS_cut_after_last_S", [0, 2, 3, 5], {"checkpoint": True}),
        ]
    results = []
    for name, cuts, kw in cases:
        actions = name.split("_")[0]
        model = PipelineLayer(
            layers=[
                LayerDesc(Step, i, actions, **kw) for i in range(len(actions))
            ],
            num_stages=world,
            seg_method=cuts,
            loss_fn=MeanLoss(),
        )
        model.config = SimpleNamespace(
            indexcache_topk_pattern=actions.replace("H", "")
        )
        pipeline = distributed_model(model)
        assert (
            type(model) is PipelineLayer and type(pipeline) is PipelineParallel
        )
        assert (
            pipeline._backward_step.__func__ is PipelineParallel._backward_step
        )
        reference = [
            Step(i, actions, **{**kw, "checkpoint": False})
            for i in range(len(actions))
        ]
        # The auxiliary PyLayer injects its own scaled gradient. Match the
        # actual two microbatch backward calls instead of scaling one loss.
        for _ in range(2):
            ref_value = {"hidden_states": paddle.ones([1, 2])}
            for layer in reference:
                ref_value = layer(ref_value)
            ref_value.mean().backward()
        inp = {"hidden_states": paddle.ones([2, 2])}
        labels = paddle.zeros([2, 1])
        loss = pipeline.forward_backward_pipeline((inp, labels))
        assert paddle.isfinite(loss).all().item(), name
        errors = []
        for layer in model.sublayers():
            if isinstance(layer, Step):
                actual = layer.gain.grad
                expected = reference[layer.idx].gain.grad
                if expected is None:
                    assert actual is None or paddle.all(actual == 0).item(), (
                        name,
                        rank,
                        layer.idx,
                        "unexpected grad",
                    )
                else:
                    assert actual is not None, (
                        name,
                        rank,
                        layer.idx,
                        "missing grad",
                    )
                    err = float(
                        paddle.max(paddle.abs(actual - expected)).item()
                    )
                    errors.append(err)
                    assert paddle.allclose(
                        actual, expected, rtol=2e-5, atol=2e-6
                    ).item(), (
                        name,
                        rank,
                        layer.idx,
                        actual.numpy(),
                        expected.numpy(),
                    )
        ev = pipeline.eval_batch((inp, labels), compute_loss=True)
        assert paddle.isfinite(ev).all().item()
        result = {
            "case": name,
            "rank": rank,
            "max_grad_error": max(errors, default=0.0),
            "loss": float(loss.item()),
            "native_pp": True,
        }
        print("CASE_PASS", json.dumps(result), flush=True)
        results.append(result)
        dist.barrier()
        del pipeline, model, reference, ref_value, inp, labels
    print(
        "NATIVE_PP_GPU_ASSESSMENT_PASS", world, rank, len(results), flush=True
    )


class TestIndexCacheNativePipeline(unittest.TestCase):
    def test_native_pipeline_matrix(self):
        if dist.get_world_size() not in (2, 3):
            self.skipTest(
                "IndexCache PP regression requires two or three ranks"
            )
        _run_native_pipeline_cases()


if __name__ == "__main__":
    unittest.main(failfast=True)
