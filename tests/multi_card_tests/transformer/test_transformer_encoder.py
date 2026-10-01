# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Real multi-card behavior tests for ``TransformerEncoder`` (pipeline=2).

Topology: 2 GPUs, pipeline-model-parallel size 2 (``pp_degree=2``). Launch
with::

    python -m paddle.distributed.launch --gpus 0,1 \
        tests/multi_card_tests/transformer/test_transformer_encoder.py

What these tests genuinely exercise (and what the CPU orchestration siblings
in ``tests/single_card_tests/transformer`` deliberately cannot): a REAL
``PipelineLayer`` partition of one encoder across two pipeline stages. Because
each rank runs ``PipelineLayer.__init__`` against the live 2-rank hybrid
communicate group, every rank ends up holding a *different* slice of the model.
The production ``state_dict`` pipeline-name remapping (stage-local ``"0.xxx"`` /
``"1.xxx"`` keys -> canonical ``"model.layers.i.xxx"`` names) is then observed
end to end and cross-checked with a REAL collective (``all_gather_object`` over
the pipeline group). The reference is the pipeline-partition contract derived
by hand: every transformer layer's parameters land on exactly one stage, the
two stages are disjoint, and their union covers all layers plus the non-layer
(embedding / final-norm) parameters.

The ``use_fp8`` scan is also driven on the real per-stage ``run_function`` list.
The virtual-pipeline branch of ``use_fp8`` carries a production bug
(``transformer_encoder.py`` lines 569-579: no trailing ``return False``), which
is documented against the CORRECT contract via ``expectedFailure`` without
editing production.

This is a metadata / control-logic multi-card test: it proves stage assignment
and key remapping under a real 2-rank topology; it makes no claim about
pipeline forward/backward numerics.
"""

import functools
import re
import unittest

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.models.common.empty_layer import EmptyLayer
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_spec,
)
from paddlefleet.parallel_state import get_pipeline_model_parallel_group
from paddlefleet.transformer.transformer_encoder import TransformerEncoder
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

PP_DEGREE = 2
NUM_HIDDEN_LAYERS = 4
HIDDEN_SIZE = 64
NUM_HEADS = 4
INTERMEDIATE = 128
VOCAB = 128
SEQ_LEN = 32
SEG_METHOD = "layer:TransformerLayer|EmptyLayer"


def _build_config(virtual_pipeline_model_parallel_size=1):
    """Small dense (no MoE) GPT config wired for pipeline-parallel size 2.

    Kept intentionally tiny (4 real transformer layers, no head/tail empty
    layers) so the hand-derived pipeline partition contract is unambiguous:
    the canonical layer indices are exactly ``{0, 1, 2, 3}`` and every one of
    them owns real parameters (empty layers would contribute none).
    """
    return GPTConfig(
        moe_expert_fusion=False,
        vocab_size=VOCAB,
        max_sequence_length=SEQ_LEN,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        parallel_output=True,
        tie_word_embeddings=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        use_qk_norm=True,
        num_empty_layers_add_in_head=0,
        num_empty_layers_add_in_tail=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=PP_DEGREE,
        virtual_pipeline_model_parallel_size=(
            virtual_pipeline_model_parallel_size
        ),
        bf16=False,
    )


def _build_sublayers_spec(config):
    """Replicate ``gpt_builder``'s dense spec assembly, then hand the encoder
    its ``GPTSublayersSpec``.

    This mirrors the production builder (``gpt_builders.gpt_builder``) exactly
    for the dense / no-MoE / no-MTP path, so the encoder is fed the same
    ``sublayers_spec`` production would build -- rather than the single-layer
    spec the deleted coverage stub incorrectly passed.
    """
    spec_func = functools.partial(
        get_gpt_layer_local_spec,
        config=config,
        use_qk_norm=config.use_qk_norm,
        num_experts=config.n_routed_experts,
        multi_latent_attention=config.multi_latent_attention,
        normalization=config.normalization,
    )
    transformer_layers_spec = []
    for layer_number in range(config.num_hidden_layers):
        real_layer_number = layer_number + config.num_empty_layers_add_in_head
        transformer_layers_spec.append(
            spec_func(layer_number=real_layer_number)
        )

    head_empty_layers_spec = [
        LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        for _ in range(config.num_empty_layers_add_in_head)
    ]
    tail_empty_layers_spec = [
        LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        for _ in range(config.num_empty_layers_add_in_tail)
    ]

    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=head_empty_layers_spec,
        transformer_layers_spec=transformer_layers_spec,
        tail_empty_layers_spec=tail_empty_layers_spec,
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        swa_rotary_base=config.swa_rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )
    return gpt_spec.sublayers_spec


def _build_encoder(virtual_pipeline_model_parallel_size=1):
    """Construct a REAL pipeline-partitioned ``TransformerEncoder``.

    ``num_stages`` / ``seg_method`` flow through ``**kwargs`` into the
    ``PipelineLayer`` base, exactly as ``build_spec_layer`` forwards them for
    the production GPT model, so the layer list is genuinely split across the
    two live pipeline ranks.
    """
    config = _build_config(virtual_pipeline_model_parallel_size)
    sublayers_spec = _build_sublayers_spec(config)
    return TransformerEncoder(
        sublayers_spec,
        config=config,
        num_stages=PP_DEGREE,
        seg_method=SEG_METHOD,
    )


def _layer_indices(keys):
    """Canonical ``model.layers.<i>.*`` -> the set of layer indices <i>."""
    out = set()
    for key in keys:
        match = re.match(r"^model\.layers\.(\d+)\.", key)
        if match is not None:
            out.add(int(match.group(1)))
    return out


class TestTransformerEncoderPipelinePartition(unittest.TestCase):
    """Real pp=2 partition + state_dict name remapping across both stages."""

    @classmethod
    def setUpClass(cls):
        # Built once against the live 2-rank pipeline group. A construction
        # failure surfaces here as a hard error for every test in the class
        # (never silently skipped), so it cannot be mistaken for a pass.
        cls.encoder = _build_encoder(virtual_pipeline_model_parallel_size=1)

    def test_state_dict_partition_is_disjoint_and_complete(self):
        """Each transformer layer lands on exactly one stage; union is whole.

        Hand-derived reference (independent of where PipelineLayer chooses to
        cut): with 4 real transformer layers and pp=2, the canonical layer
        indices seen across the two ranks must be exactly ``{0, 1, 2, 3}``,
        split into two non-empty disjoint groups (a whole layer is never torn
        across stages). Non-layer parameters (embedding + final norm) must also
        appear in the union, proving the ``state_dict`` remapping handled the
        ``"model"``-prefixed keys too, not just the ``"model.layers.*"`` ones.
        A broken remapping (e.g. leaking raw ``"0.xxx"`` stage-local keys, or
        dropping a stage's parameters) would violate disjointness or
        completeness and fail here.
        """
        local_keys = sorted(self.encoder.state_dict().keys())
        # Sanity: this rank actually owns part of the model.
        self.assertTrue(local_keys, "this pipeline stage holds no parameters")

        pp_group = get_pipeline_model_parallel_group()
        gathered = []
        dist.all_gather_object(gathered, local_keys, group=pp_group)

        self.assertEqual(
            len(gathered), PP_DEGREE, "expected one key set per pipeline stage"
        )
        stage_sets = [set(keys) for keys in gathered]

        # Every stage is non-empty and holds canonical (remapped) names, never
        # raw stage-local integer-prefixed keys like "0.xxx".
        for idx, stage in enumerate(stage_sets):
            self.assertTrue(stage, f"pipeline stage {idx} holds no parameters")
            for key in stage:
                self.assertFalse(
                    re.match(r"^\d+\.", key),
                    f"stage {idx} leaked a raw stage-local key: {key}",
                )

        # Stages are pairwise disjoint: a parameter belongs to one stage only.
        for i in range(len(stage_sets)):
            for j in range(i + 1, len(stage_sets)):
                self.assertEqual(
                    stage_sets[i] & stage_sets[j],
                    set(),
                    f"stages {i} and {j} share parameters",
                )

        idx_per_stage = [_layer_indices(stage) for stage in stage_sets]
        # No transformer layer is split across two stages.
        for i in range(len(idx_per_stage)):
            for j in range(i + 1, len(idx_per_stage)):
                self.assertEqual(
                    idx_per_stage[i] & idx_per_stage[j],
                    set(),
                    "a transformer layer was torn across two pipeline stages",
                )
        # All layers are present exactly once across the union.
        union_indices = set().union(*idx_per_stage)
        self.assertEqual(
            union_indices,
            set(range(NUM_HIDDEN_LAYERS)),
            "pipeline stages do not cover every transformer layer exactly once",
        )

        # Non-layer parameters (embedding / final norm) were remapped too.
        union_keys = set().union(*stage_sets)
        non_layer_keys = {
            key for key in union_keys if not key.startswith("model.layers.")
        }
        self.assertTrue(
            non_layer_keys,
            "embedding / final-norm parameters missing from remapped keys",
        )

    def test_use_fp8_non_vpp_returns_false_on_every_stage(self):
        """Single-stage ``use_fp8`` scans the real per-rank ``run_function``.

        No layer is fp8-configured, so the non-virtual-pipeline branch must
        return ``False`` (identity, not just falsy) on every pipeline rank. A
        real collective confirms all stages agree.
        """
        result = self.encoder.use_fp8()
        self.assertIs(result, False)

        pp_group = get_pipeline_model_parallel_group()
        gathered = []
        dist.all_gather_object(gathered, result, group=pp_group)
        self.assertEqual(len(gathered), PP_DEGREE)
        for stage_result in gathered:
            self.assertIs(stage_result, False)


class TestUseFp8VirtualPipelineBug(unittest.TestCase):
    """Document the ``use_fp8`` virtual-pipeline fall-through bug on real vpp."""

    @classmethod
    def setUpClass(cls):
        # A real virtual-pipeline (vpp=2) encoder over the 2-rank group. Errors
        # here fail every test in the class visibly, so the expectedFailure
        # below can never mask a construction problem as a "known bug".
        cls.encoder = _build_encoder(virtual_pipeline_model_parallel_size=2)

    def test_virtual_pipeline_stages_are_really_two(self):
        """Guard: prove the vpp>1 encoder actually built with two v-stages."""
        self.assertEqual(self.encoder._num_virtual_pipeline_stages, 2)

    @unittest.expectedFailure
    def test_vpp_use_fp8_should_return_false_without_fp8_layers(self):
        """CORRECT contract: with no fp8 layer, ``use_fp8`` must return False.

        PRODUCTION BUG (transformer_encoder.py:569-579): the
        ``_num_virtual_pipeline_stages > 1`` branch has no trailing
        ``return False``; when no chunk layer uses fp8 it falls through and
        returns ``None`` -- unlike the single-stage branch which correctly
        returns ``False``. Asserting the correct behavior here; marked
        ``expectedFailure`` until production is fixed (a fix turns this into an
        unexpected success and flags the update). Production is NOT edited.
        """
        self.assertIs(self.encoder.use_fp8(), False)


if __name__ == "__main__":
    Utils.initialize_model_parallel(pipeline_parallel_size=PP_DEGREE)
    unittest.main()
