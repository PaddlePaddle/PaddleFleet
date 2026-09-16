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

"""Behavior tests for paddlefleet.nn.pp_model.

Scope (no-card / CPU-verifiable control logic of pipeline-parallel model
assembly):
  * parse_args           -- argument-slot routing across tuple arities + flags,
                            and the stop_gradient contract.
  * get_pp_vp_split_layers-- PP/VP layer partitioning: chunking, round-robin
                            stage assignment and the no-recompute layer set.
                            The topology query (get_hcg) only supplies pp_size;
                            no collective communication is involved, so it is
                            stubbed and the resulting layer set is compared
                            against an independently hand-derived expectation.
  * get_attr             -- recursive attribute resolution through inner _layer.
  * RotaryEmbedding      -- cos/sin numerics vs an independent numpy reference,
                            and head_dim fallback resolution.
  * EmptyLayer           -- identity pass-through.
  * make_decoder_layer_pipe -- the generated DecoderLayerPipe.forward
                            orchestration: attention-mask type dispatch
                            (int32 startend-row-indices vs 4D float mask),
                            position-embeddings tuple conversion, and the
                            reconstructed output tuple. A stub decoder captures
                            exactly what the pipe forwards.
  * _prepare_pipeline_inputs_func -- first/last pipeline-stage key routing for
                            dict and list inputs (identity + order of the
                            selected tensors, not just element count).
  * GeneralModelForCausalLMPipe -- the _decoder_layer_cls guard, the
                            _tied_weights_keys contract and register_cls_attr
                            (validated on an isolated subclass so the production
                            class is never mutated).

Deliberately NOT covered here, with reason (would require GPU / real Fleet
initialization / real multi-stage communication, i.e. not no-card testable):
  * EmbeddingPipe.forward, RMSNormPipe, LayerNormPipe, LMHeadPipe.forward,
    CriterionLayerPipe.forward -- require the real Embedding / LMHead /
    CriterionLayer / norm layers and sequence-parallel Fleet groups.
  * GeneralModelForCausalLMPipe full __init__ (PipelineLayer segmentation,
    add_sequential_layer wiring, get_loss_fn) and any real pipeline forward
    numerics -- these depend on a live hybrid-parallel process group and
    multi-stage send/recv, which cannot be verified without a real multi-card
    run. They are intentionally left to multi-card tests; this file does not
    fake world_size + mock collectives to claim PP numerics.
"""

import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.nn import pp_model
from paddlefleet.nn.pp_model import (
    EmptyLayer,
    GeneralModelForCausalLMPipe,
    RotaryEmbedding,
    get_attr,
    get_pp_vp_split_layers,
    make_decoder_layer_pipe,
    parse_args,
)


class _SimpleConfig:
    """Lightweight config double exposing attribute + dict-style .get access.

    Only used to feed plain configuration values into the control logic under
    test; it performs no behaviour of its own.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


class TestParseArgs(unittest.TestCase):
    """parse_args maps positional inputs to fixed output slots and enforces the
    stop_gradient contract. Uses distinguishable tensors so a slot swap fails."""

    @staticmethod
    def _t(fill):
        # distinguishable, grad-enabled leaf so stop_gradient changes are visible
        t = paddle.full([2, 3], float(fill))
        t.stop_gradient = False
        return t

    def test_single_tensor_only_hidden(self):
        x = self._t(1)
        hidden, mask, pos, pe, nbatch = parse_args(x)
        self.assertIs(hidden, x)
        self.assertEqual([mask, pos, pe, nbatch], [None, None, None, None])
        # hidden_states must NOT be forced to stop_gradient by parse_args
        self.assertFalse(hidden.stop_gradient)

    def test_len1_tuple(self):
        x = self._t(1)
        hidden, mask, pos, pe, nbatch = parse_args((x,))
        self.assertIs(hidden, x)
        self.assertEqual([mask, pos, pe, nbatch], [None, None, None, None])

    def test_len2_default_is_hidden_and_mask(self):
        x, m = self._t(1), self._t(2)
        hidden, mask, pos, pe, nbatch = parse_args((x, m))
        self.assertIs(hidden, x)
        self.assertIs(mask, m)
        self.assertEqual([pos, pe, nbatch], [None, None, None])
        # mask is the only non-hidden tensor -> only it gets stop_gradient
        self.assertTrue(mask.stop_gradient)
        self.assertFalse(hidden.stop_gradient)

    def test_len2_mtp_second_arg_is_nbatch_offset(self):
        x, off = self._t(1), self._t(9)
        hidden, mask, pos, pe, nbatch = parse_args((x, off), mtp_enable=True)
        self.assertIs(hidden, x)
        # under mtp the 2nd element is the nbatch_pack_offset, not the mask
        self.assertIsNone(mask)
        self.assertIsNone(pos)
        self.assertIs(nbatch, off)
        self.assertTrue(nbatch.stop_gradient)

    def test_len3_default_is_hidden_pos_pe(self):
        # NOTE: the default (non-embed, non-mtp) 3-tuple contract is
        # (hidden_states, position_ids, position_embeddings) with mask=None.
        x, pos_in, pe_in = self._t(1), self._t(5), self._t(7)
        hidden, mask, pos, pe, nbatch = parse_args((x, pos_in, pe_in))
        self.assertIs(hidden, x)
        self.assertIsNone(mask)
        self.assertIs(pos, pos_in)
        self.assertIs(pe, pe_in)
        self.assertIsNone(nbatch)
        self.assertTrue(pos.stop_gradient)
        self.assertTrue(pe.stop_gradient)

    def test_len3_is_embed_is_hidden_mask_pos(self):
        x, m, pos_in = self._t(1), self._t(2), self._t(5)
        hidden, mask, pos, pe, nbatch = parse_args(
            (x, m, pos_in), is_embed=True
        )
        self.assertIs(hidden, x)
        self.assertIs(mask, m)
        self.assertIs(pos, pos_in)
        self.assertIsNone(pe)
        self.assertIsNone(nbatch)

    def test_len3_mtp_is_hidden_mask_nbatch(self):
        x, m, off = self._t(1), self._t(2), self._t(9)
        hidden, mask, pos, pe, nbatch = parse_args((x, m, off), mtp_enable=True)
        self.assertIs(hidden, x)
        self.assertIs(mask, m)
        self.assertIsNone(pos)
        self.assertIsNone(pe)
        self.assertIs(nbatch, off)

    def test_len4_and_len5_full_mapping(self):
        x, m, pos_in, pe_in = self._t(1), self._t(2), self._t(5), self._t(7)
        hidden, mask, pos, pe, nbatch = parse_args((x, m, pos_in, pe_in))
        self.assertIs(hidden, x)
        self.assertIs(mask, m)
        self.assertIs(pos, pos_in)
        self.assertIs(pe, pe_in)
        self.assertIsNone(nbatch)

        off = self._t(9)
        hidden, mask, pos, pe, nbatch = parse_args((x, m, pos_in, pe_in, off))
        self.assertIs(nbatch, off)
        # every non-hidden tensor must be marked stop_gradient
        for name, tensor in (
            ("mask", mask),
            ("pos", pos),
            ("pe", pe),
            ("nbatch", nbatch),
        ):
            self.assertTrue(tensor.stop_gradient, name)
        self.assertFalse(hidden.stop_gradient)


class TestGetPPVPSplitLayers(unittest.TestCase):
    """get_pp_vp_split_layers computes the set of layer indices that skip
    recompute from PP size, VP size and layer count. Expected sets below are
    derived by hand, independent of the implementation."""

    def _run(self, pp_size, cfg, **kw):
        # get_hcg only supplies pp_size here (a topology query, not a
        # collective); the partition arithmetic under test is pure CPU logic.
        with patch.object(pp_model, "get_hcg") as mock_hcg:
            mock_hcg.return_value.get_pipe_parallel_world_size.return_value = (
                pp_size
            )
            return get_pp_vp_split_layers(cfg, **kw)

    def test_pp_size_must_exceed_one(self):
        cfg = _SimpleConfig(
            num_hidden_layers=4,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=1,
        )
        with self.assertRaises(AssertionError):
            self._run(1, cfg)

    def test_skip_zero_returns_empty(self):
        cfg = _SimpleConfig(
            num_hidden_layers=8,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=2,
        )
        self.assertEqual(self._run(2, cfg, skip_recompute_num=0), set())

    def test_vp1_positive_skip_returns_all_layers(self):
        cfg = _SimpleConfig(
            num_hidden_layers=8,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=1,
        )
        # vp==1 cannot select a chunk, so a positive skip marks every layer.
        self.assertEqual(self._run(4, cfg, skip_recompute_num=1), set(range(8)))

    def test_vp1_default_skip_is_vp_then_all(self):
        cfg = _SimpleConfig(
            num_hidden_layers=8,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=1,
        )
        # default skip == -1 -> vp_size(1) -> positive -> all layers
        self.assertEqual(self._run(4, cfg), set(range(8)))

    def test_vp1_explicit_negative_skip_returns_empty(self):
        cfg = _SimpleConfig(
            num_hidden_layers=8,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=1,
        )
        # skip == -5 stays negative (only -1 is remapped to vp_size); vp==1 and
        # skip<0 -> empty set.
        self.assertEqual(self._run(4, cfg, skip_recompute_num=-5), set())

    def test_pp2_vp2_skip1_partition(self):
        # layer_num=8, pp*vp=4, chunk_size=2 -> chunks
        #   c0=[0,1] c1=[2,3] c2=[4,5] c3=[6,7]
        # round-robin over pp stages (i % pp):
        #   stage0 = [c0, c2] = [[0,1],[4,5]]
        #   stage1 = [c1, c3] = [[2,3],[6,7]]
        # skip=1 -> last chunk of each stage -> [4,5] and [6,7]
        cfg = _SimpleConfig(
            num_hidden_layers=8,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=2,
        )
        self.assertEqual(self._run(2, cfg, skip_recompute_num=1), {4, 5, 6, 7})

    def test_pp2_vp2_skip2_covers_all(self):
        cfg = _SimpleConfig(
            num_hidden_layers=8,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=2,
        )
        # skip=2 -> both chunks of each stage -> all 8 layers
        self.assertEqual(self._run(2, cfg, skip_recompute_num=2), set(range(8)))

    def test_pp2_vp3_skip1_partition(self):
        # layer_num=12, pp*vp=6, chunk_size=2 -> chunks c0..c5 of width 2
        # stage0 = [c0,c2,c4] = [[0,1],[4,5],[8,9]]
        # stage1 = [c1,c3,c5] = [[2,3],[6,7],[10,11]]
        # skip=1 -> last chunk each -> [8,9] and [10,11]
        cfg = _SimpleConfig(
            num_hidden_layers=12,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=3,
        )
        self.assertEqual(
            self._run(2, cfg, skip_recompute_num=1), {8, 9, 10, 11}
        )

    def test_empty_tail_layers_count_toward_partition(self):
        # num_hidden=6 alone is not divisible by pp*vp=4, but +2 empty tail
        # layers makes layer_num=8. This proves the tail layers are included.
        cfg = _SimpleConfig(
            num_hidden_layers=6,
            num_empty_layers_add_in_tail=2,
            virtual_pipeline_model_parallel_size=2,
        )
        self.assertEqual(self._run(2, cfg, skip_recompute_num=1), {4, 5, 6, 7})

    def test_non_divisible_layer_count_raises(self):
        # layer_num=5, pp*vp=4 -> not divisible -> assertion (reached only when
        # vp>1 and skip!=0)
        cfg = _SimpleConfig(
            num_hidden_layers=5,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=2,
        )
        with self.assertRaises(AssertionError):
            self._run(2, cfg, skip_recompute_num=1)


class TestGetAttr(unittest.TestCase):
    """get_attr returns a present non-None attribute, otherwise recurses into
    layer._layer until found."""

    def test_direct_attribute(self):
        class Holder:
            def __init__(self):
                self.weight = 123

        self.assertEqual(get_attr(Holder(), "weight"), 123)

    def test_recurses_into_inner_layer(self):
        class Inner:
            def __init__(self):
                self.weight = 456

        class Outer:
            def __init__(self):
                # attribute missing on Outer -> getattr(..., None) is None
                self._layer = Inner()

        self.assertEqual(get_attr(Outer(), "weight"), 456)

    def test_none_value_triggers_recursion(self):
        # attribute present but None on the outer object -> must recurse, not
        # return None.
        class Inner:
            def __init__(self):
                self.weight = 789

        class Outer:
            def __init__(self):
                self.weight = None
                self._layer = Inner()

        self.assertEqual(get_attr(Outer(), "weight"), 789)


class TestRotaryEmbedding(unittest.TestCase):
    """RotaryEmbedding produces cos/sin from position_ids and inv-freq. Verified
    against an independent numpy computation."""

    def _reference(self, position_ids, head_dim, base):
        pos = np.asarray(position_ids, dtype=np.float64)
        idx = np.arange(0, head_dim, 2, dtype=np.float64)
        inv = 1.0 / base ** (idx / head_dim)
        sinusoid = pos[..., None] * inv[None, :]  # [b, s, head_dim/2]
        emb = np.concatenate([sinusoid, sinusoid], axis=-1)  # [b, s, head_dim]
        return np.cos(emb), np.sin(emb)

    def test_head_dim_from_explicit_config(self):
        cfg = _SimpleConfig(
            hidden_size=64,
            num_attention_heads=4,
            rope_theta=10000.0,
            head_dim=16,
        )
        emb = RotaryEmbedding(cfg)
        self.assertEqual(emb.head_dim, 16)
        self.assertEqual(emb.base, 10000.0)

    def test_head_dim_falls_back_to_hidden_over_heads(self):
        # no head_dim attribute -> hidden_size // num_attention_heads
        cfg = _SimpleConfig(
            hidden_size=64, num_attention_heads=8, rope_theta=10000.0
        )
        emb = RotaryEmbedding(cfg)
        self.assertEqual(emb.head_dim, 8)

    def test_forward_matches_independent_reference(self):
        head_dim, base = 8, 10000.0
        cfg = _SimpleConfig(
            hidden_size=32,
            num_attention_heads=4,
            rope_theta=base,
            head_dim=head_dim,
        )
        emb = RotaryEmbedding(cfg)

        position_ids = paddle.to_tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype="int64"
        )
        x = paddle.zeros([2, 4, head_dim])
        cos, sin = emb(x, position_ids)

        ref_cos, ref_sin = self._reference(position_ids.numpy(), head_dim, base)
        # output shape is [b, s, head_dim] (two concatenated halves)
        self.assertEqual(list(cos.shape), [2, 4, head_dim])
        self.assertEqual(list(sin.shape), [2, 4, head_dim])
        np.testing.assert_allclose(cos.numpy(), ref_cos, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(sin.numpy(), ref_sin, rtol=1e-5, atol=1e-6)
        # the two halves must be identical (cat of the same sinusoid)
        np.testing.assert_allclose(
            cos.numpy()[..., : head_dim // 2],
            cos.numpy()[..., head_dim // 2 :],
            rtol=1e-6,
            atol=1e-7,
        )


class TestEmptyLayer(unittest.TestCase):
    """EmptyLayer is an identity pass-through."""

    def test_returns_input_unchanged(self):
        layer = EmptyLayer()
        x = paddle.arange(12, dtype="float32").reshape([3, 4])
        out = layer(x)
        self.assertIs(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())


class _RecordingDecoder(paddle.nn.Layer):
    """Stand-in decoder layer that records exactly what the generated
    DecoderLayerPipe.forward hands it, and returns a distinguishable marker so
    the reconstructed output tuple can be checked. It does no real attention
    math -- the behaviour under test is the pipe wrapper's dispatch, not the
    decoder numerics."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.captured = {}

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        position_embeddings=None,
    ):
        self.captured = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "position_embeddings": position_embeddings,
        }
        # marker output, unrelated to the input values on purpose
        return paddle.full_like(hidden_states, 42.0)


def _decoder_config():
    # values chosen so the direct (non-recompute) branch is taken
    return _SimpleConfig(
        num_nextn_predict_layers=0,
        enable_mtp_magic_send=False,
        sequence_parallel=False,
        recompute_granularity="none",
        recompute_method="none",
        recompute_num_layers=0,
        recompute_use_reentrant=False,
        tensor_model_parallel_size=1,
    )


class TestMakeDecoderLayerPipe(unittest.TestCase):
    """The generated DecoderLayerPipe wraps a decoder layer: it parses the
    packed args, dispatches the correct mask kind to the inner layer, converts
    position embeddings to a tuple, and rebuilds the pipeline output tuple."""

    def _build(self):
        Cls = make_decoder_layer_pipe(_RecordingDecoder)
        return Cls, Cls(config=_decoder_config(), layer_idx=0)

    def test_class_name_and_subclass(self):
        Cls = make_decoder_layer_pipe(_RecordingDecoder)
        self.assertEqual(Cls.__name__, "DecoderLayerPipe")
        self.assertTrue(issubclass(Cls, _RecordingDecoder))

    def test_single_tensor_passthrough_no_mask(self):
        _, layer = self._build()
        x = paddle.ones([2, 4, 3])
        out = layer(x)
        # no mask/pos/pe -> output is a bare tensor (tuple of length 1 unpacked)
        self.assertIsInstance(out, paddle.Tensor)
        np.testing.assert_array_equal(
            out.numpy(), np.full([2, 4, 3], 42.0, dtype=np.float32)
        )
        cap = layer.captured
        self.assertIs(cap["hidden_states"], x)
        self.assertIsNone(cap["attention_mask"])
        self.assertIsNone(cap["attn_mask_startend_row_indices"])
        self.assertIsNone(cap["position_embeddings"])

    def test_float_mask_is_sliced_and_dispatched_as_tgt_mask(self):
        _, layer = self._build()
        seq = 4
        x = paddle.ones([2, seq, 3])
        # 4D float mask, larger than seq on the last two axes to prove slicing
        mask = paddle.arange(2 * 1 * 6 * 6, dtype="float32").reshape(
            [2, 1, 6, 6]
        )
        out = layer((x, mask))

        cap = layer.captured
        # float mask -> goes to attention_mask, startend indices stay None
        self.assertIsNone(cap["attn_mask_startend_row_indices"])
        self.assertIsNotNone(cap["attention_mask"])
        self.assertEqual(list(cap["attention_mask"].shape), [2, 1, seq, seq])
        np.testing.assert_array_equal(
            cap["attention_mask"].numpy(),
            mask.numpy()[:, :, :seq, :seq],
        )
        # output is a tuple carrying the ORIGINAL (unsliced) mask onward
        self.assertIsInstance(out, tuple)
        self.assertEqual(list(out[1].shape), [2, 1, 6, 6])
        np.testing.assert_array_equal(out[1].numpy(), mask.numpy())

    def test_int32_mask_routed_to_startend_row_indices(self):
        _, layer = self._build()
        seq = 4
        x = paddle.ones([2, seq, 3])
        # int32 mask -> startend-row-indices path, sliced on the last axis only
        mask = paddle.arange(2 * 2 * 6, dtype="int32").reshape([2, 2, 6])
        out = layer((x, mask))

        cap = layer.captured
        self.assertIsNone(cap["attention_mask"])
        self.assertIsNotNone(cap["attn_mask_startend_row_indices"])
        self.assertEqual(
            list(cap["attn_mask_startend_row_indices"].shape), [2, 2, seq]
        )
        np.testing.assert_array_equal(
            cap["attn_mask_startend_row_indices"].numpy(),
            mask.numpy()[:, :, :seq],
        )
        self.assertIsInstance(out, tuple)

    def test_position_embeddings_converted_to_tuple(self):
        _, layer = self._build()
        seq = 4
        x = paddle.ones([2, seq, 3])
        # position_embeddings is a stacked (cos, sin); shape [2, ..., seq, d]
        pe = paddle.arange(2 * 2 * 6 * 3, dtype="float32").reshape([2, 2, 6, 3])
        # default 4-tuple ordering: (hidden, mask, pos, pe)
        pos = paddle.arange(2 * seq, dtype="int64").reshape([2, seq])
        mask = paddle.ones([2, 1, 6, 6])
        out = layer((x, mask, pos, pe))

        cap = layer.captured
        tpe = cap["position_embeddings"]
        self.assertIsInstance(tpe, tuple)
        self.assertEqual(len(tpe), 2)
        # each half sliced to seq on the -2 axis
        self.assertEqual(list(tpe[0].shape), [*[2, 6, 3][:1], seq, 3])
        np.testing.assert_array_equal(tpe[0].numpy(), pe.numpy()[0, :, :seq, :])
        np.testing.assert_array_equal(tpe[1].numpy(), pe.numpy()[1, :, :seq, :])
        # output tuple carries mask, pos and pe forward
        self.assertIsInstance(out, tuple)
        self.assertGreaterEqual(len(out), 4)


class TestPreparePipelineInputs(unittest.TestCase):
    """_prepare_pipeline_inputs_func selects which keys travel to the first vs
    last pipeline stage, popping the actual tensors in a fixed order."""

    def test_dict_with_attention_mask(self):
        input_ids = paddle.arange(16, dtype="int64").reshape([2, 8])
        attention_mask = paddle.ones([2, 8])
        position_ids = paddle.arange(16, dtype="int64").reshape([2, 8]) + 100
        labels = paddle.arange(16, dtype="int64").reshape([2, 8]) + 200
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }
        first, last = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(
            inputs
        )
        # first stage: (input_ids, attention_mask, position_ids) in that order
        self.assertIsInstance(first, tuple)
        self.assertEqual(len(first), 3)
        self.assertIs(first[0], input_ids)
        self.assertIs(first[1], attention_mask)
        self.assertIs(first[2], position_ids)
        # last stage: only labels present -> unwrapped to the single tensor
        self.assertIs(last, labels)

    def test_dict_with_startend_row_indices(self):
        input_ids = paddle.arange(16, dtype="int64").reshape([2, 8])
        startend = paddle.ones([2, 2, 8], dtype="int32")
        labels = paddle.arange(16, dtype="int64").reshape([2, 8]) + 200
        inputs = {
            "input_ids": input_ids,
            "attn_mask_startend_row_indices": startend,
            "labels": labels,
        }
        first, last = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(
            inputs
        )
        # no attention_mask -> default first keys pick up the startend indices
        self.assertEqual(len(first), 2)
        self.assertIs(first[0], input_ids)
        self.assertIs(first[1], startend)
        self.assertIs(last, labels)

    def test_list_of_dicts_batches_per_key(self):
        a = {
            "input_ids": paddle.to_tensor([[1, 2]], dtype="int64"),
            "attention_mask": paddle.to_tensor([[1, 1]], dtype="int64"),
            "labels": paddle.to_tensor([[7, 8]], dtype="int64"),
        }
        b = {
            "input_ids": paddle.to_tensor([[3, 4]], dtype="int64"),
            "attention_mask": paddle.to_tensor([[1, 0]], dtype="int64"),
            "labels": paddle.to_tensor([[9, 10]], dtype="int64"),
        }
        first, last = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(
            [a, b]
        )
        # each first-stage entry is the per-key list gathered across the batch
        self.assertEqual(len(first), 2)  # input_ids, attention_mask
        self.assertEqual(len(first[0]), 2)
        np.testing.assert_array_equal(first[0][0].numpy(), [[1, 2]])
        np.testing.assert_array_equal(first[0][1].numpy(), [[3, 4]])
        np.testing.assert_array_equal(last[0].numpy(), [[7, 8]])
        np.testing.assert_array_equal(last[1].numpy(), [[9, 10]])


class TestGeneralModelForCausalLMPipe(unittest.TestCase):
    """Class-level contracts that are reachable without a live pipeline group."""

    def test_missing_decoder_layer_cls_raises(self):
        # _decoder_layer_cls defaults to None; __init__ guards this before any
        # topology / PipelineLayer work, so no Fleet init is needed.
        cfg = _SimpleConfig(layer_types=[])
        with self.assertRaises(ValueError) as ctx:
            GeneralModelForCausalLMPipe(cfg)
        self.assertIn("_decoder_layer_cls", str(ctx.exception))

    def test_tied_weights_keys_contract(self):
        self.assertEqual(
            GeneralModelForCausalLMPipe._tied_weights_keys, ["lm_head.weight"]
        )

    def test_register_cls_attr_on_isolated_subclass(self):
        # Use a dedicated subclass so the production class is never mutated
        # (avoids cross-test pollution of GeneralModelForCausalLMPipe).
        class _Probe(GeneralModelForCausalLMPipe):
            pass

        class _Cfg:
            pass

        class _Model:
            _get_tensor_parallel_mappings = "tp_map"
            _get_fuse_or_split_param_mappings = "fuse_split"
            _init_weights = "init_w"
            _keep_in_fp32_modules = ["ln"]
            transpose_weight_keys = ["w"]

        returned = _Probe.register_cls_attr(
            config_class=_Cfg, pretrained_model_class=_Model
        )
        self.assertIs(returned, _Probe)
        self.assertIs(_Probe.config_class, _Cfg)
        self.assertEqual(_Probe._get_tensor_parallel_mappings, "tp_map")
        self.assertEqual(_Probe._get_fuse_or_split_param_mappings, "fuse_split")
        self.assertEqual(_Probe._init_weights, "init_w")
        self.assertEqual(_Probe._keep_in_fp32_modules, ["ln"])
        self.assertEqual(_Probe.transpose_weight_keys, ["w"])
        # the production base class must remain untouched
        self.assertIs(
            GeneralModelForCausalLMPipe.config_class, pp_model.PretrainedConfig
        )

    def test_register_cls_attr_config_only(self):
        class _Probe2(GeneralModelForCausalLMPipe):
            pass

        class _Cfg2:
            pass

        _Probe2.register_cls_attr(config_class=_Cfg2)
        self.assertIs(_Probe2.config_class, _Cfg2)


if __name__ == "__main__":
    unittest.main()
