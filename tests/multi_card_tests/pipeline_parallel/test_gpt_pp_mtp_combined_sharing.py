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

"""Multi-card (pp=2) test for mtp_shared_weights + mtp_shared_last_layer.

In the combined mode the MTP transformer body is shared with the last backbone
TransformerLayer through the ``mtp_reuse_transformer`` SharedLayerDesc key, and
the per-depth fusion parameters are aliased across MTP depths within one stage.
The single-card test only sees the all-on-one-rank layout; this file covers the
three pipeline layouts that behave differently:

  * backbone last layer and every MTP depth on the same stage
    (``layer:TransformerLayer|EmptyLayer``, the usual production split);
  * backbone last layer on stage 0, every MTP depth on stage 1 -- the body then
    crosses the stage boundary through ``PipelineLayer.shared_comm``, and on the
    MTP stage depth 0 becomes the stored shared layer, so the later depths are
    aliased MTP-to-MTP. Pure ``mtp_shared_last_layer`` cannot build this layout
    at num_nextn_predict_layers > 1 (see TransformerConfig), the combined mode
    must;
  * MTP depths split across stages -- rejected, because the fusion aliases are
    rank-local and nothing would keep them in sync across stages.

The two accepted layouts also run optimizer steps and check that the shared
parameters stay tied: identical Parameter objects within a stage, identical
values across stages.

Per-rank checks are gathered over the pipe group before asserting, so a
violation seen by one stage fails every rank instead of leaving the other stage
blocked in the next p2p.
"""

import functools
import os
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from paddlefleet.transformer.transformer_layer import TransformerLayer

PP_DEGREE = 2
MTP_DEGREE = 3
NUM_HIDDEN_LAYERS = 4
NUM_STEPS = 2
REPO_FLAG = os.getenv("repo_flag")
SKIP_TESTS = REPO_FLAG != "paddlefleet"

# Layer descs for NUM_HIDDEN_LAYERS=4, MTP_DEGREE=3: embedding (0), backbone
# layers (1..4, the last one is the mtp_reuse_transformer pivot), one more
# backbone-side desc (5), MTP depths (6..8), LM head (9). The tests assert the
# resulting placement, so a change in this order fails loudly.
SEG_COLOCATED = "layer:TransformerLayer|EmptyLayer"
SEG_BACKBONE_LAST_OFF_STAGE = [0, 6, 10]
SEG_SPLIT_MTP_DEPTHS = [0, 7, 10]


def _config(**extra_config):
    return GPTConfig(
        moe_expert_fusion=False,
        vocab_size=128,
        max_sequence_length=64,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=512,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        use_qk_norm=True,
        pipeline_model_parallel_size=PP_DEGREE,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        n_shared_experts=1,
        n_routed_experts=8,
        moe_intermediate_size=512,
        gated_linear_unit=True,
        num_nextn_predict_layers=MTP_DEGREE,
        mtp_shared_weights=True,
        mtp_shared_last_layer=True,
        **extra_config,
    )


def _build(seg_method, seed=46, **extra_config):
    np.random.seed(seed)
    paddle.seed(seed)
    return gpt_builder(
        _config(**extra_config), num_stages=PP_DEGREE, seg_method=seg_method
    )


def _inputs(seed, num_acc=2):
    paddle.seed(seed)
    data = paddle.randint(low=0, high=128, shape=(1, 64 + MTP_DEGREE + 1))
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.arange(input_ids.shape[1]).reshape([1, -1])
    return (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
        },
        [labels] * num_acc,
    )


def _mtp_layers(model):
    return [
        layer
        for layer in model.run_function
        if isinstance(layer, MultiTokenPredictionLayer)
    ]


def _backbone_layers(model):
    return [
        layer
        for layer in model.run_function
        if isinstance(layer, TransformerLayer)
    ]


def _pipe_group():
    return fleet.get_hybrid_communicate_group().get_pipe_parallel_group()


def _gather_object(obj):
    gathered = []
    dist.all_gather_object(gathered, obj, group=_pipe_group())
    return gathered


def _assert_on_all_ranks(errors):
    """Fail every pipe rank if any rank reported an error."""
    gathered = _gather_object(errors)
    failures = [
        f"stage {stage}: {err}"
        for stage, errs in enumerate(gathered)
        for err in errs
    ]
    assert not failures, "\n".join(failures)


def _mtp_depth_tie_errors(mtp_layers):
    """Every MTP depth on this rank must share depth 0's Parameters (body + fusion)."""
    errors = []
    if len(mtp_layers) < 2:
        return errors
    d0 = dict(mtp_layers[0].all_weights)
    for layer in mtp_layers[1:]:
        not_shared = [n for n, p in layer.all_weights if d0.get(n) is not p]
        if not_shared:
            errors.append(
                f"depth {layer.layer_number} does not share depth 0's "
                f"parameters: {not_shared[:5]}"
            )
    return errors


def _train(model, steps=NUM_STEPS):
    pipe = distributed_model(model)
    optimizer = fleet.distributed_optimizer(
        paddle.optimizer.SGD(learning_rate=1e-2, parameters=model.parameters())
    )
    losses = []
    for step in range(steps):
        loss = pipe.train_batch(_inputs(seed=100 + step), optimizer)
        losses.append(loss)
    return losses


def _shared_body(model):
    """(names, flat values) of this stage's copy of the mtp_reuse_transformer body."""
    layer = model.shared_layers["mtp_reuse_transformer"]
    named = list(layer.transformer_layer_weights)
    names = [n for n, _ in named]
    flat = paddle.concat(
        [p.detach().astype("float32").flatten() for _, p in named]
    )
    return names, flat


@unittest.skipIf(SKIP_TESTS, "requires repo_flag=paddlefleet multi-card env")
class TestMTPCombinedSharingPP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": PP_DEGREE,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
            "pp_configs": {
                "overlap_p2p_comm": True,
                "enable_dynamic_shape": True,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": 2,
            "micro_batch_size": 1,
        }
        initialize_fleet(strategy)

    def _assert_finite(self, losses):
        for loss in losses:
            assert loss is not None, "loss is None"
            assert bool(paddle.isfinite(loss).all()), f"loss not finite: {loss}"

    def test_colocated_with_backbone_last(self):
        model = _build(SEG_COLOCATED)
        mtp_layers = _mtp_layers(model)
        layout = _gather_object(
            (
                len(_backbone_layers(model)),
                sorted(la.layer_number for la in mtp_layers),
            )
        )
        self.assertEqual(
            layout,
            [(2, []), (2, list(range(MTP_DEGREE)))],
            "expected backbone layers split 2/2 and every MTP depth on the last "
            "stage together with the backbone last layer",
        )

        def _check_ties():
            errors = []
            if mtp_layers:
                backbone_last = dict(
                    _backbone_layers(model)[-1].transformer_layer_weights
                )
                for layer in mtp_layers:
                    untied = [
                        name
                        for name, param in layer.transformer_layer_weights
                        if param is not backbone_last.get(name)
                    ]
                    if untied:
                        errors.append(
                            f"MTP depth {layer.layer_number} body params are not "
                            f"the backbone last layer's: {untied[:5]}"
                        )
                errors += _mtp_depth_tie_errors(mtp_layers)
            _assert_on_all_ranks(errors)

        _check_ties()
        self._assert_finite(_train(model))
        _check_ties()

    def test_colocated_with_depth_sampling(self):
        """Combined sharing + sampling: K=1 skips depths >= 1, ties still hold."""
        model = _build(SEG_COLOCATED, mtp_depth_sampling=[1.0, 0.0, 0.0])
        mtp_layers = _mtp_layers(model)
        body_calls = {la.layer_number: 0 for la in mtp_layers}
        for layer in mtp_layers:

            def _count(_mod, _inp, _depth=layer.layer_number):
                body_calls[_depth] += 1

            layer.transformer_layer.register_forward_pre_hook(_count)

        self._assert_finite(_train(model))
        errors = []
        if mtp_layers:
            if body_calls[0] == 0:
                errors.append(f"depth 0 must run, body_calls={body_calls}")
            if any(n for d, n in body_calls.items() if d >= 1):
                errors.append(
                    f"K=1 must skip every depth >= 1, body_calls={body_calls}"
                )
            errors += _mtp_depth_tie_errors(mtp_layers)
        _assert_on_all_ranks(errors)

    def test_backbone_last_off_stage(self):
        model = _build(SEG_BACKBONE_LAST_OFF_STAGE)
        mtp_layers = _mtp_layers(model)
        layout = _gather_object(
            (
                len(_backbone_layers(model)),
                sorted(la.layer_number for la in mtp_layers),
            )
        )
        self.assertEqual(
            layout,
            [(NUM_HIDDEN_LAYERS, []), (0, list(range(MTP_DEGREE)))],
            "expected every backbone layer on stage 0 and every MTP depth on "
            "stage 1",
        )
        _assert_on_all_ranks(_mtp_depth_tie_errors(mtp_layers))

        names, before = _shared_body(model)
        gathered_names = _gather_object(names)
        self.assertEqual(
            gathered_names[0],
            gathered_names[1],
            "backbone last layer and MTP body expose different parameter lists",
        )

        def _gather_values(flat):
            out = []
            dist.all_gather(out, flat, group=_pipe_group())
            return [t.numpy() for t in out]

        initial = _gather_values(before)
        np.testing.assert_array_equal(
            initial[0],
            initial[1],
            err_msg="shared body not broadcast across stages at construction",
        )

        self._assert_finite(_train(model))
        _assert_on_all_ranks(_mtp_depth_tie_errors(mtp_layers))

        _, after = _shared_body(model)
        trained = _gather_values(after)
        # Both stages apply the same SGD update to the allreduced gradient, so
        # the copies must stay bit-identical.
        np.testing.assert_array_equal(
            trained[0],
            trained[1],
            err_msg="shared body diverged across stages after optimizer steps",
        )
        self.assertGreater(
            float(np.abs(trained[0] - initial[0]).max()),
            0.0,
            "shared body did not change, so the sync check proves nothing",
        )

    def test_split_mtp_depths_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"mtp_shared_weights \+ mtp_shared_last_layer requires all MTP "
            r"depths on one pipeline stage",
        ):
            _build(SEG_SPLIT_MTP_DEPTHS)


if __name__ == "__main__":
    unittest.main()
