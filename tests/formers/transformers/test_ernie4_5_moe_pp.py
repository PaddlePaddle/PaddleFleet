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

"""No-card (CPU) behavior tests for the Ernie4_5-MoE pipeline-parallel model.

The non-VL ``Ernie4_5_MoeForCausalLMPipe`` (src/paddlefleet/transformers/
ernie4_5_moe/modeling.py) is a thin subclass of
``GeneralModelForCausalLMPipe`` (src/paddlefleet/nn/pp_model.py). All of the
pipeline plumbing it relies on -- per-stage input un-packing (``parse_args``),
the PP/VP layer-partitioning that decides which layer indices skip recompute
(``get_pp_vp_split_layers``), and the tail-padding pass-through
(``EmptyLayer``) -- lives in ``pp_model``. These are exercised here with real
imports and independently derived expectations.

Scope note: building the full pipeline (``GeneralModelForCausalLMPipe.__init__``
-> ``PipelineLayer.__init__``) requires a real Fleet hybrid-parallel process
group with pp_size > 1 and per-rank stage assignment; that is a multi-card
concern and is NOT executed here (see ``TestErnie45MoePipeWiring`` for the
CPU-verifiable static wiring, and the skipped construction test for the reason).
``get_hcg().get_pipe_parallel_world_size()`` is a topology-degree query (not a
collective), so it is stubbed to drive the CPU-computable partition arithmetic;
no real cross-rank communication is claimed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.nn.pp_model import (
    EmptyLayer,
    GeneralModelForCausalLMPipe,
    get_pp_vp_split_layers,
    parse_args,
)
from paddlefleet.transformers.ernie4_5_moe.configuration import (
    Ernie4_5_MoeConfig,
)
from paddlefleet.transformers.ernie4_5_moe.modeling import (
    Ernie4_5_MoeDecoderLayer,
    Ernie4_5_MoeForCausalLMPipe,
    Ernie4_5_MoeModel,
)


class TestParseArgsRouting(unittest.TestCase):
    """``parse_args`` routes each per-stage tuple layout into the correct slot.

    Contract (5-tuple return):
        (hidden_states, attention_mask, position_ids,
         position_embeddings, nbatch_pack_offset)

    The len-3 case is genuinely ambiguous: ``mtp_enable`` and ``is_embed``
    select three different routings from the *same* arity, so distinguishable
    content is used to catch a mis-route (a shape/None-only check would not).
    """

    def setUp(self):
        # Distinguishable content so any slot swap is caught.
        self.h = paddle.to_tensor([[1.0, 1.1, 1.2]], dtype="float32")
        self.attn = paddle.to_tensor([[2.0, 2.1, 2.2]], dtype="float32")
        self.pos_emb = paddle.to_tensor([[3.0, 3.1, 3.2]], dtype="float32")
        self.pid = paddle.to_tensor([[10, 11, 12]], dtype="int64")
        self.npk = paddle.to_tensor([20, 21], dtype="int64")

    def test_single_tensor_only_hidden(self):
        h, am, pid, pe, npk = parse_args(self.h)
        np.testing.assert_array_equal(h.numpy(), self.h.numpy())
        self.assertIsNone(am)
        self.assertIsNone(pid)
        self.assertIsNone(pe)
        self.assertIsNone(npk)

    def test_len1_tuple(self):
        h, am, pid, pe, npk = parse_args((self.h,))
        np.testing.assert_array_equal(h.numpy(), self.h.numpy())
        self.assertEqual((am, pid, pe, npk), (None, None, None, None))

    def test_len2_default_hidden_attn(self):
        h, am, pid, pe, npk = parse_args((self.h, self.attn))
        np.testing.assert_array_equal(am.numpy(), self.attn.numpy())
        self.assertIsNone(pid)
        self.assertIsNone(pe)
        self.assertIsNone(npk)

    def test_len2_mtp_routes_second_to_nbatch_offset(self):
        # mtp path: 2nd element is nbatch_pack_offset, NOT attention_mask.
        h, am, pid, pe, npk = parse_args((self.h, self.npk), mtp_enable=True)
        self.assertIsNone(am)
        self.assertIsNone(pid)
        np.testing.assert_array_equal(npk.numpy(), self.npk.numpy())

    def test_len3_default_routes_pid_and_posemb(self):
        # non-mtp, non-embed: (hidden, position_ids, position_embeddings)
        h, am, pid, pe, npk = parse_args((self.h, self.pid, self.pos_emb))
        self.assertIsNone(am)
        np.testing.assert_array_equal(pid.numpy(), self.pid.numpy())
        np.testing.assert_array_equal(pe.numpy(), self.pos_emb.numpy())
        self.assertIsNone(npk)

    def test_len3_is_embed_routes_attn_and_pid(self):
        h, am, pid, pe, npk = parse_args(
            (self.h, self.attn, self.pid), is_embed=True
        )
        np.testing.assert_array_equal(am.numpy(), self.attn.numpy())
        np.testing.assert_array_equal(pid.numpy(), self.pid.numpy())
        self.assertIsNone(pe)
        self.assertIsNone(npk)

    def test_len3_mtp_routes_attn_and_nbatch_offset(self):
        h, am, pid, pe, npk = parse_args(
            (self.h, self.attn, self.npk), mtp_enable=True
        )
        np.testing.assert_array_equal(am.numpy(), self.attn.numpy())
        self.assertIsNone(pid)
        np.testing.assert_array_equal(npk.numpy(), self.npk.numpy())

    def test_len4_full_without_offset(self):
        h, am, pid, pe, npk = parse_args(
            (self.h, self.attn, self.pid, self.pos_emb)
        )
        np.testing.assert_array_equal(am.numpy(), self.attn.numpy())
        np.testing.assert_array_equal(pid.numpy(), self.pid.numpy())
        np.testing.assert_array_equal(pe.numpy(), self.pos_emb.numpy())
        self.assertIsNone(npk)

    def test_len5_full(self):
        h, am, pid, pe, npk = parse_args(
            (self.h, self.attn, self.pid, self.pos_emb, self.npk)
        )
        np.testing.assert_array_equal(h.numpy(), self.h.numpy())
        np.testing.assert_array_equal(am.numpy(), self.attn.numpy())
        np.testing.assert_array_equal(pid.numpy(), self.pid.numpy())
        np.testing.assert_array_equal(pe.numpy(), self.pos_emb.numpy())
        np.testing.assert_array_equal(npk.numpy(), self.npk.numpy())

    def test_stop_gradient_side_effects(self):
        # Float inputs let us prove parse_args *flips* stop_gradient to True
        # for the boundary tensors while leaving hidden_states untouched.
        h = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        attn = paddle.to_tensor([[0.0, 1.0, 0.0]], dtype="float32")
        pos_emb = paddle.to_tensor([[4.0, 5.0, 6.0]], dtype="float32")
        h.stop_gradient = False
        attn.stop_gradient = False
        pos_emb.stop_gradient = False
        out_h, out_am, _, out_pe, _ = parse_args((h, attn, None, pos_emb))
        # boundary tensors are forced to stop gradient across the PP hand-off
        self.assertTrue(out_am.stop_gradient)
        self.assertTrue(out_pe.stop_gradient)
        # hidden_states must NOT be forced; gradient still flows through it
        self.assertFalse(out_h.stop_gradient)


def _pp_cfg(num_hidden, vp, empty=0):
    """Minimal config carrier: get_pp_vp_split_layers only reads these ints."""
    return SimpleNamespace(
        num_hidden_layers=num_hidden,
        virtual_pipeline_model_parallel_size=vp,
        num_empty_layers_add_in_tail=empty,
    )


def _hcg_with_pp(pp_size):
    """Patch the topology-degree query used by the partitioner (no collective)."""
    hcg = SimpleNamespace(
        get_pipe_parallel_world_size=lambda: pp_size,
    )
    return patch(
        "paddlefleet.nn.pp_model.get_hcg",
        return_value=hcg,
    )


class TestGetPPVPSplitLayers(unittest.TestCase):
    """PP/VP partition: which layer indices are marked no-recompute.

    Expectations are derived independently of the implementation. The real
    contract is: split ``layer_num`` into ``pp*vp`` contiguous chunks, hand
    chunk ``i`` to stage ``i % pp`` (round-robin, VP interleave), then keep the
    LAST ``skip_recompute_num`` chunks of every stage. A stage-major split or a
    first-N slice would yield different index sets and be rejected here.
    """

    def test_round_robin_last_chunk_per_stage(self):
        # pp=2, vp=2, layer_num=8, skip=1 -> chunk_size=2
        # chunks: s0=[[0,1],[4,5]] s1=[[2,3],[6,7]]; last-1 each -> {4,5,6,7}
        with _hcg_with_pp(2):
            got = get_pp_vp_split_layers(
                _pp_cfg(num_hidden=8, vp=2), skip_recompute_num=1
            )
        self.assertEqual(got, {4, 5, 6, 7})

    def test_round_robin_vp3(self):
        # pp=2, vp=3, layer_num=12, skip=1 -> chunk_size=2
        # s0=[[0,1],[4,5],[8,9]] s1=[[2,3],[6,7],[10,11]]; last-1 -> {8,9,10,11}
        with _hcg_with_pp(2):
            got = get_pp_vp_split_layers(
                _pp_cfg(num_hidden=12, vp=3), skip_recompute_num=1
            )
        self.assertEqual(got, {8, 9, 10, 11})

    def test_tail_empty_layers_count_toward_partition(self):
        # num_hidden=6 + empty=2 -> layer_num=8, same shape as the pp2/vp2 case
        with _hcg_with_pp(2):
            got = get_pp_vp_split_layers(
                _pp_cfg(num_hidden=6, vp=2, empty=2), skip_recompute_num=1
            )
        self.assertEqual(got, {4, 5, 6, 7})

    def test_default_skip_equals_vp_takes_all_chunks(self):
        # skip=-1 -> vp(=2); last-2 chunks per stage == every chunk -> all 8
        with _hcg_with_pp(2):
            got = get_pp_vp_split_layers(_pp_cfg(num_hidden=8, vp=2))
        self.assertEqual(got, set(range(8)))

    def test_skip_zero_returns_empty(self):
        with _hcg_with_pp(2):
            got = get_pp_vp_split_layers(
                _pp_cfg(num_hidden=8, vp=2), skip_recompute_num=0
            )
        self.assertEqual(got, set())

    def test_vp1_default_skips_all_layers(self):
        # vp==1: no model-chunk selection, so skip>0 marks every layer
        with _hcg_with_pp(2):
            got = get_pp_vp_split_layers(_pp_cfg(num_hidden=5, vp=1))
        self.assertEqual(got, set(range(5)))

    def test_pp_size_one_rejected(self):
        with _hcg_with_pp(1):
            with self.assertRaises(AssertionError):
                get_pp_vp_split_layers(_pp_cfg(num_hidden=8, vp=2))

    def test_non_divisible_layer_num_rejected(self):
        # layer_num=6 not divisible by pp(2)*vp(2)=4
        with _hcg_with_pp(2):
            with self.assertRaises(AssertionError):
                get_pp_vp_split_layers(
                    _pp_cfg(num_hidden=6, vp=2), skip_recompute_num=1
                )


class TestEmptyLayerPassthrough(unittest.TestCase):
    """EmptyLayer is the tail padding used to make layer_num divisible; it must
    return its input unchanged so padding never perturbs hidden states."""

    def test_returns_same_object_and_content(self):
        layer = EmptyLayer()
        x = paddle.to_tensor([[7.0, 8.0], [9.0, 10.0]], dtype="float32")
        out = layer(x)
        self.assertIs(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())


class TestErnie45MoePipeWiring(unittest.TestCase):
    """Static (CPU) wiring that decides *what* each pipeline stage builds.

    ``GeneralModelForCausalLMPipe.__init__`` reads ``self._decoder_layer_cls``
    to construct every ``model.layers.{i}`` stage and pulls the TP mapping /
    weight-init from the bound model class. Exact-identity assertions catch a
    mis-wired decoder or a wrong config class -- either of which would silently
    build the wrong pipeline.
    """

    def test_subclasses_general_pp(self):
        self.assertTrue(
            issubclass(Ernie4_5_MoeForCausalLMPipe, GeneralModelForCausalLMPipe)
        )

    def test_decoder_layer_class_is_moe_decoder(self):
        self.assertIs(
            Ernie4_5_MoeForCausalLMPipe._decoder_layer_cls,
            Ernie4_5_MoeDecoderLayer,
        )

    def test_config_class(self):
        self.assertIs(
            Ernie4_5_MoeForCausalLMPipe.config_class, Ernie4_5_MoeConfig
        )

    def test_tied_weight_keys(self):
        self.assertEqual(
            Ernie4_5_MoeForCausalLMPipe._tied_weights_keys, ["lm_head.weight"]
        )

    def test_tp_mapping_and_init_bound_from_model(self):
        self.assertIs(
            Ernie4_5_MoeForCausalLMPipe._get_tensor_parallel_mappings,
            Ernie4_5_MoeModel._get_tensor_parallel_mappings,
        )
        self.assertIs(
            Ernie4_5_MoeForCausalLMPipe._init_weights,
            Ernie4_5_MoeModel._init_weights,
        )

    @unittest.skip(
        "Full pipeline construction (GeneralModelForCausalLMPipe.__init__ -> "
        "PipelineLayer.__init__) needs a real Fleet hybrid-parallel process "
        "group with pp_size>1 and per-rank stage assignment. That is a "
        "multi-card concern; stage partitioning across ranks cannot be proven "
        "in a single CPU process. See tests/multi_card_tests for PP runs."
    )
    def test_full_pp_construction_multi_card_only(self):
        pass


if __name__ == "__main__":
    unittest.main()
