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
"""``Linear`` marks its replicated parameters for the sequence-parallel reduction.

``Linear`` holds a weight duplicated across TP ranks, but under sequence
parallelism each rank only feeds ``s / TP`` tokens through it, so the local
weight gradient is a PARTIAL sum over the sequence and the full gradient is the
sum over the TP group. ``Linear.forward`` passes ``sequence_parallel=False`` into
the autograd function on purpose (the layer never gathers or scatters), so
nothing else contributes that term.

Paddle transports the requirement on the ``sequence_parallel`` parameter
attribute: ``mark_as_sequence_parallel_parameter`` sets it and PaddleFormers'
``SPGradSyncCallback`` all-reduces exactly the marked parameters over the
model-parallel group. Megatron-Core does the same for
``parallel_mode="duplicated"`` in
``megatron/core/extensions/transformer_engine.py``, and PaddleFormers' own
``deepseek_v3`` marks precisely its replicated ``q_a_proj`` /
``kv_a_proj_with_mqa`` this way.

These tests pin the marking, its use_accuracy_compatible gate, and the remaining
guards. They are pure attribute assertions on a single card: no TP group is
created, so nothing here depends on a distributed launch.
"""

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

import paddle

from paddlefleet.tensor_parallel.layers import Linear


def _make_config(**kwargs):
    """A real ``ModelParallelConfig``, not a mock.

    ``__post_init__`` forces ``sequence_parallel = False`` when
    ``tensor_model_parallel_size <= 1``, which is itself part of the contract
    under test, so the tests must not bypass it with a mock.
    """
    from paddlefleet.model_parallel_config import ModelParallelConfig

    defaults = {
        "params_dtype": paddle.float32,
        "perform_initialization": True,
        "use_cpu_initialization": True,
        "sequence_parallel": False,
        "deterministic_mode": False,
        "gradient_accumulation_fusion": False,
        "defer_embedding_wgrad_compute": False,
        "wgrad_deferral_limit": 0,
        "expert_model_parallel_size": 1,
    }
    use_accuracy_compatible = kwargs.pop("use_accuracy_compatible", False)
    defaults.update(kwargs)
    config = ModelParallelConfig(**defaults)
    # ModelParallelConfig does not declare the alignment switch; Linear reads it
    # with getattr, matching TransformerConfig.use_accuracy_compatible.
    config.use_accuracy_compatible = use_accuracy_compatible
    return config


def _build(bias=False, **config_kwargs):
    return Linear(
        8,
        16,
        config=_make_config(**config_kwargs),
        init_method=paddle.nn.initializer.Constant(1.0),
        bias=bias,
    )


def _is_marked(parameter):
    return bool(getattr(parameter, "sequence_parallel", False))


class TestLinearMarksReplicatedGradForTPReduction(unittest.TestCase):
    """The mark is applied exactly when the partial-sum situation exists."""

    def test_marked_when_accuracy_compatible_and_sequence_parallel(self):
        layer = _build(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            use_accuracy_compatible=True,
        )
        self.assertTrue(
            _is_marked(layer.weight),
            "a replicated weight under SP+TP holds a partial sequence sum and "
            "must be all-reduced over the TP group",
        )

    def test_bias_is_marked_too(self):
        layer = _build(
            bias=True,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            use_accuracy_compatible=True,
        )
        self.assertTrue(_is_marked(layer.bias))

    def test_not_marked_without_accuracy_compatible(self):
        layer = _build(
            bias=True,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            use_accuracy_compatible=False,
        )
        self.assertFalse(_is_marked(layer.weight))
        self.assertFalse(_is_marked(layer.bias))

    def test_not_marked_without_sequence_parallel(self):
        # Without SP every rank sees the whole sequence, so the local gradient is
        # already complete and an extra all-reduce would multiply it by TP.
        layer = _build(
            bias=True,
            sequence_parallel=False,
            tensor_model_parallel_size=2,
            use_accuracy_compatible=True,
        )
        self.assertFalse(_is_marked(layer.weight))
        self.assertFalse(_is_marked(layer.bias))

    def test_not_marked_at_tp_one(self):
        # ModelParallelConfig.__post_init__ also forces sequence_parallel off
        # here; assert the outcome rather than the intermediate.
        layer = _build(
            bias=True,
            sequence_parallel=True,
            tensor_model_parallel_size=1,
            use_accuracy_compatible=True,
        )
        self.assertFalse(_is_marked(layer.weight))
        self.assertFalse(_is_marked(layer.bias))

    def test_expert_parameters_are_not_marked(self):
        # Expert gradients are reduced in their own (expert-)data-parallel domain,
        # which is also why mcore only takes the duplicated branch for
        # non-expert parameters.
        layer = Linear(
            8,
            16,
            config=_make_config(
                sequence_parallel=True,
                tensor_model_parallel_size=2,
                use_accuracy_compatible=True,
            ),
            init_method=paddle.nn.initializer.Constant(1.0),
            bias=True,
            is_expert=True,
        )
        self.assertFalse(_is_marked(layer.weight))
        self.assertFalse(_is_marked(layer.bias))

    def test_marking_does_not_disturb_the_replication_attributes(self):
        layer = _build(
            bias=True,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            use_accuracy_compatible=True,
        )
        # The weight is still replicated, still DP-reduced, and the layer still
        # reports TP=1 -- the mark adds the TP reduction, it does not turn the
        # layer into a sharded one.
        self.assertTrue(layer.weight.allreduce)
        self.assertFalse(layer.weight.is_distributed)
        self.assertEqual(layer.output_size_per_partition, 16)
        self.assertIn("TP=1", repr(layer))

    def test_forward_still_works_when_marked(self):
        layer = _build(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            use_accuracy_compatible=True,
        )
        output, output_bias = layer(paddle.randn([2, 4, 8]))
        self.assertEqual(output.shape, [2, 4, 16])
        self.assertIsNone(output_bias)

    def test_skip_weight_param_allocation_marks_nothing(self):
        layer = Linear(
            8,
            16,
            config=_make_config(
                sequence_parallel=True,
                tensor_model_parallel_size=2,
                use_accuracy_compatible=True,
            ),
            init_method=paddle.nn.initializer.Constant(1.0),
            bias=False,
            skip_weight_param_allocation=True,
        )
        self.assertIsNone(layer.weight)


class TestMLADownProjectionsAreReplicated(unittest.TestCase):
    """MLA down-projections replicate only under use_accuracy_compatible.

    Sharding them produces a gradient that is validly but DIFFERENTLY reduced
    from the reference implementations. The default path keeps the historical
    column-parallel spec.
    """

    def _mla_spec(self, *, use_accuracy_compatible):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
        from paddlefleet.transformer.enums import AttnMaskType
        from paddlefleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            multi_latent_attention=True,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            qk_pos_emb_head_dim=4,
            v_head_dim=8,
            use_qk_norm=False,
            use_accuracy_compatible=use_accuracy_compatible,
        )
        return get_attention_spec(
            config=config,
            attention_layer_type="multi_latent_attention",
            attn_mask_type=AttnMaskType.causal,
        )

    def test_replicated_linear_and_column_parallel_are_distinct_classes(self):
        from paddlefleet.models.backends import LocalSpecProvider

        backend = LocalSpecProvider()
        self.assertIsNot(backend.linear(), backend.column_parallel_linear())

    def test_spec_uses_replicated_linear_under_accuracy_compatible(self):
        from paddlefleet.models.backends import LocalSpecProvider

        spec = self._mla_spec(use_accuracy_compatible=True)
        backend = LocalSpecProvider()
        self.assertIs(spec.sublayers_spec.q_a_proj, backend.linear())
        self.assertIs(spec.sublayers_spec.kv_a_proj_with_mqa, backend.linear())

    def test_spec_stays_column_parallel_by_default(self):
        from paddlefleet.models.backends import LocalSpecProvider

        spec = self._mla_spec(use_accuracy_compatible=False)
        backend = LocalSpecProvider()
        self.assertIs(
            spec.sublayers_spec.q_a_proj, backend.column_parallel_linear()
        )
        self.assertIs(
            spec.sublayers_spec.kv_a_proj_with_mqa,
            backend.column_parallel_linear(),
        )

    def test_the_neighbouring_projections_stay_tensor_parallel(self):
        from paddlefleet.models.backends import LocalSpecProvider

        spec = self._mla_spec(use_accuracy_compatible=True)
        backend = LocalSpecProvider()
        self.assertIs(
            spec.sublayers_spec.q_b_proj, backend.column_parallel_linear()
        )
        self.assertIs(
            spec.sublayers_spec.kv_b_proj, backend.column_parallel_linear()
        )
        self.assertIs(spec.sublayers_spec.o_proj, backend.row_parallel_linear())


if __name__ == "__main__":
    unittest.main()
