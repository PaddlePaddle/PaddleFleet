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

"""Accuracy-compatible branches inside the MLA attention classes.

``MultiLatentAttention.forward`` and
``MLASelfAttention.get_query_key_value_tensors`` are driven as unbound
functions against stubs, because the branches under test are choices
between collectives and projections rather than kernels:

* which projection helper the q-down / q-up / o-proj steps go through,
* the absorbed K/V de-absorption views the torch-aligned core attention
  is handed,
* the MTP depth-wise wrap of the RoPE table,
* which side of the KV branch keeps the sequence gathered under TP.

Every collective is patched (single card, no process group) and the
sequence-parallel drives stop at a named probe once the branch under test
has run, so the parts that need a real attention kernel stay out of scope.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.transformer import multi_latent_attention as mla
from paddlefleet.transformer.transformer_layer import TransformerLayer

TP_GROUP = "tp-group-sentinel"


def _tensor(array, stop_gradient=False):
    out = paddle.to_tensor(np.asarray(array, dtype="float32"))
    out.stop_gradient = stop_gradient
    return out


def _ramp(*shape, lo=-1.0, hi=1.5):
    total = int(np.prod(shape))
    values = np.linspace(lo, hi, total, dtype="float32")
    return _tensor(values.reshape(shape))


def _projection(weight):
    return SimpleNamespace(
        weight=weight,
        bias=None,
        skip_bias_add=True,
        sequence_parallel=False,
        tp_group=TP_GROUP,
    )


class _Stop(Exception):
    """Ends a drive once the branch under test has run."""


def _stop_at_k_pos_emb(tensor, name, layer_idx):
    if name == "mla_k_pos_emb_raw":
        raise _Stop(name)


class _RotaryStub:
    """Stands in for the rotary embedding module of the layer."""

    def __init__(self, table):
        self.table = table
        self.seq_len_requests = []
        self.table_requests = []

    def get_rotary_seq_len(self, hidden_states, config, packed_seq_params):
        self.seq_len_requests.append(packed_seq_params)
        return int(self.table.shape[1])

    def __call__(self, seq_len, packed_seq=False, position_ids=None):
        self.table_requests.append((seq_len, packed_seq, position_ids))
        return self.table


q_lora_rank_default = 4


class QkvTensorsBranchTest(unittest.TestCase):
    """``MLASelfAttention.get_query_key_value_tensors`` branch selection."""

    batch = 1
    seq_len = 3
    hidden_size = 4
    q_lora_rank = q_lora_rank_default
    kv_lora_rank = 4
    qk_rope_head_dim = 2

    def setUp(self):
        self.table = _ramp(1, self.seq_len, 1, 4, lo=1.0, hi=12.0)
        self.rotary = _RotaryStub(self.table)
        self.hidden_states = _ramp(self.batch, self.seq_len, self.hidden_size)
        # A column-sharded q-down: the local output is half of q_lora_rank.
        self.q_a_proj = _projection(_ramp(self.hidden_size, 2))
        self.q_b_proj = _projection(_ramp(self.q_lora_rank, 6))
        self.q_proj = _projection(_ramp(self.hidden_size, 6))
        self.kv_width = self.kv_lora_rank + self.qk_rope_head_dim
        self.deferrable_calls = []
        self.deferrable_layers = []
        self.stop_on_q_proj = False
        self.rolls = []

    def _attention(
        self,
        *,
        use_accuracy_compatible=True,
        sequence_parallel=True,
        is_mtp_layer=True,
        kv_sharded=True,
        layer_number=1,
        q_lora_rank=q_lora_rank_default,
    ):
        self.kv_sharded = kv_sharded
        return SimpleNamespace(
            config=SimpleNamespace(
                mla_use_nope=False,
                rope_type="rope",
                apply_rope_fusion=False,
                use_accuracy_compatible=use_accuracy_compatible,
                sequence_parallel=sequence_parallel,
                tensor_model_parallel_size=2,
            ),
            rotary_pos_emb=self.rotary,
            training=True,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            mqa_latent=False,
            pg_collection=SimpleNamespace(tp=TP_GROUP),
            q_a_proj=self.q_a_proj,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            kv_a_proj_with_mqa=_projection(
                _ramp(self.hidden_size, self.kv_width)
            ),
            q_a_layernorm=lambda value: value,
            kv_a_layernorm=lambda value: value,
            recompute_qkv_up_porj_and_rope=False,
        )

    def _deferrable_linear(self, config, point, layer, value):
        self.deferrable_calls.append(point)
        self.deferrable_layers.append(layer)
        if self.stop_on_q_proj and point == "attn_q_proj":
            raise _Stop(point)
        if point == "attn_kv_proj":
            width = self.kv_width if not self.kv_sharded else self.kv_width // 2
            return _ramp(self.batch, self.seq_len, width), None
        return F.linear(value, layer.weight), None

    def _drive(self, attention, *, stop_at_probe=True, q_up=None, pg_size=2):
        """Run the QKV chain with every collective patched out."""
        real_roll = paddle.roll

        def _record_roll(*args, **kwargs):
            out = real_roll(*args, **kwargs)
            self.rolls.append((kwargs, out))
            return out

        mocks = {}
        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            enter(
                patch.object(mla, "get_context_parallel_world_size", lambda: 1)
            )
            enter(patch.object(mla, "get_pg_size", return_value=pg_size))
            enter(
                patch.object(
                    mla,
                    "inspect_tensor",
                    side_effect=lambda name, layer, value: value,
                )
            )
            enter(patch.object(mla, "get_current_layer", return_value=0))
            mocks["deferrable_linear"] = enter(
                patch.object(
                    mla,
                    "deferrable_linear",
                    side_effect=self._deferrable_linear,
                )
            )
            mocks["tp_gather"] = enter(
                patch.object(
                    mla,
                    "gather_from_tensor_model_parallel_region",
                    side_effect=lambda value, **kwargs: paddle.concat(
                        [value, value], axis=-1
                    ),
                )
            )
            mocks["sp_scatter"] = enter(
                patch.object(
                    mla,
                    "scatter_to_sequence_parallel_region",
                    side_effect=lambda value: value,
                )
            )
            mocks["sp_gather"] = enter(
                patch.object(
                    mla,
                    "gather_from_sequence_parallel_region",
                    side_effect=lambda value, group=None: value,
                )
            )
            enter(patch.object(paddle, "roll", side_effect=_record_roll))
            if stop_at_probe:
                enter(
                    patch.object(
                        TransformerLayer, "_log_md5", _stop_at_k_pos_emb
                    )
                )
            if q_up is not None:
                mocks["q_up"] = enter(
                    patch.object(
                        mla,
                        "_accuracy_compatible_q_up_projection",
                        side_effect=q_up,
                    )
                )
            with self.assertRaises(_Stop):
                mla.MLASelfAttention.get_query_key_value_tensors(
                    attention, self.hidden_states
                )
        return mocks

    def test_accuracy_compatible_q_down_replaces_the_deferred_linear(self):
        captured = []

        def _q_up(projection, hidden_states):
            captured.append((projection, hidden_states))
            raise _Stop("q_up")

        mocks = self._drive(self._attention(), stop_at_probe=False, q_up=_q_up)

        # Only the KV down projection goes through deferrable_linear; both
        # q-down and q-up take the accuracy-compatible helpers.
        self.assertEqual(self.deferrable_calls, ["attn_kv_proj"])
        mocks["q_up"].assert_called_once()
        projection, q_compressed = captured[0]
        self.assertIs(projection, self.q_b_proj)
        # q-down output (local shard 2) gathered to q_lora_rank, then
        # sequence-scattered back, and finally layer-normed.
        self.assertEqual(
            list(q_compressed.shape),
            [self.batch, self.seq_len, self.q_lora_rank],
        )
        local = F.linear(self.hidden_states, self.q_a_proj.weight)
        np.testing.assert_array_equal(
            q_compressed.numpy(),
            paddle.concat([local, local], axis=-1).numpy(),
        )

    def test_accuracy_compatible_keeps_the_kv_sequence_gathered(self):
        mocks = self._drive(self._attention())

        # q and kv are both TP-gathered ...
        self.assertEqual(mocks["tp_gather"].call_count, 2)
        for call in mocks["tp_gather"].call_args_list:
            self.assertTrue(call.kwargs["use_accuracy_compatible"])
        # ... but only q is scattered back: under the accuracy-compatible
        # path the constructor clears kv_b_proj.sequence_parallel instead.
        self.assertEqual(mocks["sp_scatter"].call_count, 1)

    def test_default_path_scatters_both_branches(self):
        mocks = self._drive(
            self._attention(use_accuracy_compatible=False, is_mtp_layer=False)
        )

        self.assertEqual(self.deferrable_calls, ["attn_q_proj", "attn_kv_proj"])
        self.assertEqual(mocks["tp_gather"].call_count, 2)
        for call in mocks["tp_gather"].call_args_list:
            self.assertFalse(call.kwargs["use_accuracy_compatible"])
        # q and kv must end up on the same sequence length.
        self.assertEqual(mocks["sp_scatter"].call_count, 2)

    def test_replicated_kv_down_gathers_the_positional_branch(self):
        mocks = self._drive(self._attention(kv_sharded=False))

        # A replicated kv_a_proj_with_mqa needs no TP gather for kv, so the
        # positional half is gathered on its own before RoPE.
        self.assertEqual(mocks["tp_gather"].call_count, 1)
        mocks["sp_gather"].assert_called_once()
        self.assertEqual(mocks["sp_gather"].call_args.kwargs["group"], TP_GROUP)

    def test_replicated_kv_down_without_tp_keeps_the_local_branch(self):
        mocks = self._drive(self._attention(kv_sharded=False), pg_size=1)

        mocks["sp_gather"].assert_not_called()

    def test_without_q_lora_the_up_projection_stays_deferred(self):
        # No q-down at all: the hidden states go straight into q_proj, and
        # that projection is not part of the accuracy-compatible rewrite.
        self.stop_on_q_proj = True

        self._drive(self._attention(q_lora_rank=None), stop_at_probe=False)

        self.assertEqual(self.deferrable_calls, ["attn_kv_proj", "attn_q_proj"])
        self.assertIs(self.deferrable_layers[-1], self.q_proj)

    def test_mtp_layer_wraps_the_rope_table_by_depth(self):
        self._drive(self._attention(layer_number=1))

        # Paddle MTP layers are zero-indexed, the reference starts at one,
        # so layer 1 shifts by two.
        self.assertEqual(len(self.rolls), 1)
        kwargs, rolled = self.rolls[0]
        self.assertEqual(kwargs["shifts"], -2)
        self.assertEqual(kwargs["axis"], 1)
        np.testing.assert_array_equal(
            rolled.numpy(), np.roll(self.table.numpy(), -2, axis=1)
        )

    def test_backbone_layer_leaves_the_rope_table_alone(self):
        self._drive(self._attention(is_mtp_layer=False))

        self.assertEqual(self.rolls, [])

    def test_default_path_leaves_the_rope_table_alone(self):
        self._drive(self._attention(use_accuracy_compatible=False))

        self.assertEqual(self.rolls, [])


class _CoreAttentionStub:
    """Callable core attention with the ``config`` the decode probe reads."""

    def __init__(self, output):
        self.config = SimpleNamespace()
        self.output = output
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.output


class ForwardOutputPathTest(unittest.TestCase):
    """``MultiLatentAttention.forward``: absorbed core views and o-proj."""

    batch = 1
    seq_len = 2
    heads = 2
    head_dim = 4
    hidden_size = 4
    kv_lora_rank = 3
    qk_nope_head_dim = 2
    v_head_dim = 2

    def setUp(self):
        self.kv_b_weight = _ramp(
            self.kv_lora_rank,
            self.heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.core_attn_out = _ramp(self.batch, self.seq_len, self.hidden_size)
        self.o_proj = SimpleNamespace(
            weight=_ramp(self.hidden_size, self.hidden_size),
            bias=_ramp(self.hidden_size),
            skip_bias_add=True,
            sequence_parallel=False,
            tp_group=TP_GROUP,
        )
        self.core_attention = _CoreAttentionStub(self.core_attn_out)

    def _attention(self, *, use_accuracy_compatible=True):
        shape = (self.batch, self.seq_len, self.heads, self.head_dim)
        tensors = (
            _ramp(*shape),
            _ramp(*shape),
            _ramp(*shape),
            _ramp(self.batch, self.seq_len, self.kv_lora_rank),
            _ramp(self.batch, self.seq_len, self.kv_lora_rank),
            _ramp(self.batch, self.seq_len, 1, 2),
        )
        return SimpleNamespace(
            layer_number=0,
            attn_mask_type="attn-mask-type-sentinel",
            config=SimpleNamespace(
                sequence_parallel=False,
                use_accuracy_compatible=use_accuracy_compatible,
                enable_hy_sparse_attention=False,
            ),
            mqa_latent=False,
            kv_b_proj=SimpleNamespace(weight=self.kv_b_weight),
            kv_lora_rank=self.kv_lora_rank,
            num_attention_heads_per_partition=self.heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
            recompute_core_attention=False,
            training=False,
            core_attention=self.core_attention,
            use_rr_flash_attention=False,
            recompute_qkv_up_porj_and_rope=False,
            use_vha_postmix=False,
            gated_attention=False,
            o_proj=self.o_proj,
            get_query_key_value_tensors=lambda *args, **kwargs: tensors,
        )

    def _forward(self, attention):
        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            enter(
                patch.object(
                    mla,
                    "inspect_tensor",
                    side_effect=lambda name, layer, value: value,
                )
            )
            enter(patch.object(mla, "get_current_layer", return_value=0))
            enter(patch.object(mla, "get_pg_size", return_value=1))
            enter(patch.object(mla, "inspect_tensor_set_current_layer"))
            enter(patch.object(TransformerLayer, "_log_md5", lambda *a: None))
            deferred = enter(
                patch.object(
                    mla,
                    "deferrable_linear",
                    side_effect=lambda config, point, layer, value: (
                        value,
                        None,
                    ),
                )
            )
            output, bias = mla.MultiLatentAttention.forward(
                attention,
                _ramp(self.batch, self.seq_len, self.hidden_size),
                None,
            )
        return output, bias, deferred

    def _absorbed_views(self):
        weight = self.kv_b_weight.numpy()
        folded = weight.reshape(self.kv_lora_rank, self.heads, -1).transpose(
            1, 2, 0
        )
        return folded[:, : self.qk_nope_head_dim, :], folded[
            :, -self.v_head_dim :, :
        ]

    def test_absorbed_core_receives_both_de_absorption_halves(self):
        self._forward(self._attention())

        self.assertEqual(len(self.core_attention.calls), 1)
        kwargs = self.core_attention.calls[0]
        expected_k, expected_v = self._absorbed_views()
        # Query is not pre-absorbed on this path: the core attention builds
        # it from its own q with these K / V weights.
        self.assertIsNone(kwargs["q_absorbed"])
        np.testing.assert_array_equal(
            kwargs["k_abs_weight"].numpy(), expected_k
        )
        np.testing.assert_array_equal(
            kwargs["v_b_proj_weight"].numpy(), expected_v
        )

    def test_output_projection_uses_the_accuracy_compatible_helper(self):
        output, bias, deferred = self._forward(self._attention())

        deferred.assert_not_called()
        self.assertIs(bias, self.o_proj.bias)
        np.testing.assert_array_equal(
            output.numpy(),
            F.linear(self.core_attn_out, self.o_proj.weight).numpy(),
        )

    def test_default_path_defers_the_output_projection(self):
        output, bias, deferred = self._forward(
            self._attention(use_accuracy_compatible=False)
        )

        deferred.assert_called_once()
        self.assertEqual(deferred.call_args.args[1], "attn_out_proj")
        self.assertIs(deferred.call_args.args[2], self.o_proj)
        self.assertIsNone(bias)
        np.testing.assert_array_equal(
            output.numpy(), self.core_attn_out.numpy()
        )
        # No absorbed weights are built outside the accuracy-compatible arm.
        self.assertIsNone(self.core_attention.calls[0]["k_abs_weight"])
        self.assertIsNone(self.core_attention.calls[0]["v_b_proj_weight"])


if __name__ == "__main__":
    unittest.main()
