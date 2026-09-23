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

"""``TransformerConfig.use_dsv4_accuracy`` must gate the model-side DSV4 replay call sites.

Model-layer companion to ``test_dsv4_accuracy_flag_gating.py``. The same switch
also toggles numeric / structural branches inside the loss (``LanguageLoss``),
the LM head (``GPTLMHead``), the GPT embedding (``GPTEmbedding``) and the
DeepSeek-V4 HF->PF weight-conversion generator
(``DeepseekV4PreTrainedModel._gen_aoa_config``). With the flag off each branch
must keep the pre-DSV4 numeric / layout behaviour so the other alignment
targets are not silently perturbed. Every test below drives the smallest
isolatable unit that contains one call site and pins it on both sides of the
flag.

Every module imports the flag into its own namespace
(``from paddlefleet.utils import use_dsv4_accuracy_compatible``), so each test
patches the name on *that* module object.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddlefleet.models.common.language_loss import language_loss
from paddlefleet.models.gpt import gpt_embedding, lm_head
from paddlefleet.transformers.deepseek_v4 import modeling


def _dsv4_flag(module, enabled):
    return patch.object(
        module, "use_dsv4_accuracy_compatible", return_value=enabled
    )


# --------------------------------------------------------------------------- #
# LanguageLoss helpers
# --------------------------------------------------------------------------- #
def _ones_loss_func(logits, labels):
    """Deterministic per-token loss of 1.0, shaped like the label grid."""
    return paddle.ones([logits.shape[0], logits.shape[1]], dtype="float32")


def _make_loss_stub(
    *,
    use_accuracy_compatible=False,
    variant="dsv4_hybrid",
    num_nextn=0,
    add_mtp_loss=True,
    mtp_scaling=1.0,
):
    config = types.SimpleNamespace(
        gpt_model_use_experimental_version=False,
        sequence_parallel=False,
        experimental_attention_variant=variant,
        num_nextn_predict_layers=num_nextn,
        mtp_load_weight_only=False,
        use_erndata=False,
        mtp_distillation_loss=False,
        train_mtp_only=False,
        add_mtp_loss=add_mtp_loss,
        mtp_loss_scaling_factor=mtp_scaling,
        cp_balance_mode="contiguous_allgather",
    )
    stub = types.SimpleNamespace(
        config=config,
        use_subbatch=False,
        ignored_index=-100,
        use_accuracy_compatible=use_accuracy_compatible,
        pg_collection=None,
        enable_parallel_cross_entropy=False,
        loss_func=_ones_loss_func,
    )
    stub.forward_impl = types.MethodType(
        language_loss.LanguageLoss.forward_impl, stub
    )
    stub._forward = types.MethodType(language_loss.LanguageLoss._forward, stub)
    stub.forward = types.MethodType(language_loss.LanguageLoss.forward, stub)
    return stub


class TestLanguageLossFinalNormalizationGating(unittest.TestCase):
    """language_loss.py:466 - the DSV4 hybrid loss skips the EP-replicated
    float64 accumulation and takes the plain float32 mean instead.

    When ``use_accuracy_compatible`` is on and the attention variant is
    ``dsv4_hybrid``, the flag decides whether ``forward_impl`` walks the
    EP-replicated float64 accumulation (which queries
    ``get_expert_model_parallel_group``) or the plain float32 sum/mean. The two
    code paths are distinguished by whether the EP group is fetched at all.
    """

    def _run(self, enabled, ep_spy):
        stub = _make_loss_stub(
            use_accuracy_compatible=True, variant="dsv4_hybrid"
        )
        logits = paddle.zeros([1, 4, 6], dtype="float32")
        labels = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        with (
            _dsv4_flag(language_loss, enabled),
            patch.object(
                language_loss, "get_expert_model_parallel_group", ep_spy
            ),
        ):
            return stub.forward_impl(logits, labels)

    def test_flag_off_takes_the_ep_replicated_accumulation(self):
        ep_spy = MagicMock(return_value=None)
        loss = self._run(False, ep_spy)
        ep_spy.assert_called_once()
        np.testing.assert_allclose(float(loss), 1.0, rtol=1e-6)

    def test_flag_on_takes_the_plain_float32_mean(self):
        ep_spy = MagicMock(return_value=None)
        loss = self._run(True, ep_spy)
        ep_spy.assert_not_called()
        np.testing.assert_allclose(float(loss), 1.0, rtol=1e-6)


class TestLanguageLossScaleGatingNonMtp(unittest.TestCase):
    """language_loss.py:1010 - the non-MTP loss is only run through
    ``LossScaleBeforeBackward.scale`` when the flag is on."""

    def _run(self, enabled, scale_mock):
        stub = _make_loss_stub(use_accuracy_compatible=False, num_nextn=0)
        logits = paddle.zeros([1, 2, 6], dtype="float32")
        labels = paddle.to_tensor([[1, 2]], dtype="int64")
        with (
            _dsv4_flag(language_loss, enabled),
            patch.object(
                language_loss, "module_needs_recompute", return_value=False
            ),
            patch.object(
                language_loss.LossScaleBeforeBackward, "scale", scale_mock
            ),
        ):
            return stub.forward(logits, labels)

    def test_flag_on_scales_the_loss(self):
        scale_mock = MagicMock(side_effect=lambda loss: loss * 5.0)
        loss = self._run(True, scale_mock)
        scale_mock.assert_called_once()
        np.testing.assert_allclose(float(loss), 5.0, rtol=1e-6)

    def test_flag_off_leaves_the_loss_unscaled(self):
        scale_mock = MagicMock(side_effect=lambda loss: loss * 5.0)
        loss = self._run(False, scale_mock)
        scale_mock.assert_not_called()
        np.testing.assert_allclose(float(loss), 1.0, rtol=1e-6)


class TestLanguageLossScaleGatingMtp(unittest.TestCase):
    """language_loss.py:1004 - the MTP-branch loss is only run through
    ``LossScaleBeforeBackward.scale`` when the flag is on."""

    def _mtp_logits_labels(self):
        # main + one MTP depth; both length seq_len=2, labels length 3.
        logits = [
            paddle.zeros([1, 2, 6], dtype="float32"),
            paddle.zeros([1, 2, 6], dtype="float32"),
        ]
        labels = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        return logits, labels

    def _run(self, enabled, scale_mock):
        stub = _make_loss_stub(
            use_accuracy_compatible=False, num_nextn=1, add_mtp_loss=True
        )
        logits, labels = self._mtp_logits_labels()
        with (
            _dsv4_flag(language_loss, enabled),
            patch.object(
                language_loss, "module_needs_recompute", return_value=False
            ),
            patch.object(
                language_loss.LossScaleBeforeBackward, "scale", scale_mock
            ),
        ):
            return stub.forward(logits, labels)

    def test_flag_on_scales_the_mtp_loss(self):
        # add_mtp_loss=True, kernel switch off -> loss = lm(1.0) + mtp(1.0).
        scale_mock = MagicMock(side_effect=lambda loss: loss * 5.0)
        loss = self._run(True, scale_mock)
        scale_mock.assert_called_once()
        np.testing.assert_allclose(float(loss), 10.0, rtol=1e-6)

    def test_flag_off_leaves_the_mtp_loss_unscaled(self):
        scale_mock = MagicMock(side_effect=lambda loss: loss * 5.0)
        loss = self._run(False, scale_mock)
        scale_mock.assert_not_called()
        np.testing.assert_allclose(float(loss), 2.0, rtol=1e-6)


class TestLanguageLossMtpAddLossOrderGating(unittest.TestCase):
    """language_loss.py:958 - the Megatron-aligned MTP fold-in order.

    Under ``_use_accuracy_compatible_kernel()`` and ``add_mtp_loss`` the flag
    picks between ``mtp - mtp.detach() + main`` (on) and
    ``main + mtp - mtp.detach()`` (off). Both are algebraically ``main``, but
    the association order is observable in finite precision: with a huge MTP
    contribution the flag-off ``(main + mtp) - mtp`` rounds ``main`` away in
    float32, while the flag-on ``(mtp - mtp) + main`` keeps it exactly.
    """

    def _run(self, enabled):
        # main loss ~ 1.0, MTP contribution ~ 2**30 so float32 addition drops
        # the main term in the flag-off association order.
        stub = _make_loss_stub(
            use_accuracy_compatible=False,
            num_nextn=1,
            add_mtp_loss=True,
            mtp_scaling=float(2**30),
        )
        logits = [
            paddle.zeros([1, 2, 6], dtype="float32"),
            paddle.zeros([1, 2, 6], dtype="float32"),
        ]
        labels = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        with (
            _dsv4_flag(language_loss, enabled),
            patch.object(
                language_loss, "module_needs_recompute", return_value=False
            ),
            patch.object(
                language_loss,
                "_use_accuracy_compatible_kernel",
                return_value=True,
            ),
        ):
            return stub.forward(logits, labels)

    def test_flag_on_keeps_the_main_loss_after_the_fold_in(self):
        np.testing.assert_allclose(float(self._run(True)), 1.0, rtol=1e-6)

    def test_flag_off_rounds_the_main_loss_away(self):
        np.testing.assert_allclose(float(self._run(False)), 0.0, atol=1e-3)

    def test_the_two_fold_in_orders_are_actually_distinguishable(self):
        self.assertNotEqual(float(self._run(True)), float(self._run(False)))


class TestLmHeadSequenceFirstLinear(unittest.TestCase):
    """lm_head.py:233 - the DSV4 replay transposes hidden states to
    sequence-first ``[S, B, H]`` before the LM-head projection.

    Exercised through the fused-CE path (``fused_linear_ce_loss_chunk`` > 0),
    which returns the (possibly transposed) hidden states without needing the
    tensor-parallel ``ColumnParallelLinear.forward``.
    """

    def _run(self, enabled, hidden):
        config = types.SimpleNamespace(
            sequence_parallel=False, fused_linear_ce_loss_chunk=1
        )
        stub = types.SimpleNamespace(
            config=config, weight=paddle.zeros([1]), bias=None
        )
        with _dsv4_flag(lm_head, enabled):
            return lm_head.GPTLMHead._forward(stub, hidden)

    def test_flag_off_keeps_batch_first_layout(self):
        hidden = paddle.randn([2, 3, 4], dtype="float32")
        out = self._run(False, hidden)
        self.assertEqual(out[0].shape, [2, 3, 4])
        np.testing.assert_allclose(out[0].numpy(), hidden.numpy())

    def test_flag_on_switches_to_sequence_first_layout(self):
        hidden = paddle.randn([2, 3, 4], dtype="float32")
        out = self._run(True, hidden)
        self.assertEqual(out[0].shape, [3, 2, 4])
        np.testing.assert_allclose(
            out[0].numpy(), hidden.transpose([1, 0, 2]).numpy()
        )


# --------------------------------------------------------------------------- #
# GPTEmbedding forward: a single stubbed forward exercises both MTP call sites.
# --------------------------------------------------------------------------- #
def _make_embedding_stub(embed_table):
    config = types.SimpleNamespace(
        gpt_model_use_experimental_version=False,
        expert_model_parallel_size=2,  # + tp < 2 -> input_ids_for_moe_mask set
        tensor_model_parallel_size=1,
        pad_token_id=0,
        num_nextn_predict_layers=1,
        mtp_load_weight_only=False,
        use_erndata=False,
        enable_mtp_magic_send=False,
        experimental_dataflow=False,
        separate_mtp_input=True,
        apply_rope_fusion=False,
        sequence_parallel=False,
        cp_balance_mode="contiguous_allgather",
    )

    def embedding(input_ids=None, position_ids=None):
        return paddle.nn.functional.embedding(input_ids, embed_table)

    stub = types.SimpleNamespace(
        config=config,
        embedding=embedding,
        multimodal_embedding=False,
        sequence_parallel=False,
        position_embedding_type="learned_absolute",
        rotary_pos_emb=None,
        swa_rotary_pos_emb=None,
        mrope_section=None,
        has_kda_layer=False,
    )
    stub._embed_shifted_mtp = types.MethodType(
        gpt_embedding.GPTEmbedding._embed_shifted_mtp, stub
    )
    stub.forward = types.MethodType(gpt_embedding.GPTEmbedding.forward, stub)
    return stub


def _run_embedding_forward(enabled, embed_spy=None):
    embed_table = paddle.to_tensor(
        np.arange(10 * 4, dtype="float32").reshape([10, 4])
    )
    stub = _make_embedding_stub(embed_table)
    if embed_spy is not None:
        embed_spy.side_effect = stub._embed_shifted_mtp
        stub._embed_shifted_mtp = embed_spy
    input_ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
    with _dsv4_flag(gpt_embedding, enabled), paddle.no_grad():
        return stub.forward({"input_ids": input_ids})


class TestGptEmbeddingMtpMoeMaskGating(unittest.TestCase):
    """gpt_embedding.py:617 (inside ``GPTEmbedding.forward`` - NOT inside
    ``_embed_shifted_mtp``, so it is a distinct call site to cover).

    The per-depth MTP ``input_ids`` for the MoE router are built by shifting and
    zero-padding when the flag is on, versus reading the real tail tokens (a
    sliding window into the L+K ids) when the flag is off.
    """

    def test_flag_on_shifts_and_zero_pads_the_moe_mask_ids(self):
        out = _run_embedding_forward(True)
        np.testing.assert_array_equal(
            out["mtp_input_ids_for_moe_mask"].numpy(), [[[2, 3, 4, 0]]]
        )

    def test_flag_off_uses_the_sliding_window_of_real_ids(self):
        out = _run_embedding_forward(False)
        np.testing.assert_array_equal(
            out["mtp_input_ids_for_moe_mask"].numpy(), [[[2, 3, 4, 5]]]
        )


class TestGptEmbeddingMtpShiftedEmbeddingGating(unittest.TestCase):
    """gpt_embedding.py:725 (inside ``GPTEmbedding.forward`` - NOT inside
    ``_embed_shifted_mtp``, so it is a distinct call site to cover).

    The flag decides whether the shifted MTP embedding is re-derived from the
    ids via ``_embed_shifted_mtp`` (zero-padded tail) or spliced from the
    already-computed backbone embeddings (real tail token). The two differ in
    the last position: a padded id-0 embedding vs the real last-token
    embedding.
    """

    def test_flag_on_calls_embed_shifted_mtp_and_pads_the_tail(self):
        spy = MagicMock()
        out = _run_embedding_forward(True, embed_spy=spy)
        spy.assert_called_once()
        # last row is the padded id-0 embedding (embed_table row 0).
        last_row = out["mtp_decoder_inputs"].numpy()[0, 0, -1, :]
        np.testing.assert_array_equal(last_row, [0.0, 1.0, 2.0, 3.0])

    def test_flag_off_splices_backbone_embeddings_with_the_real_tail(self):
        spy = MagicMock()
        out = _run_embedding_forward(False, embed_spy=spy)
        spy.assert_not_called()
        # last row is the real token-5 embedding (embed_table row 5).
        last_row = out["mtp_decoder_inputs"].numpy()[0, 0, -1, :]
        np.testing.assert_array_equal(last_row, [20.0, 21.0, 22.0, 23.0])

    def test_the_two_tail_embeddings_differ(self):
        on = _run_embedding_forward(True)["mtp_decoder_inputs"].numpy()
        off = _run_embedding_forward(False)["mtp_decoder_inputs"].numpy()
        self.assertFalse(np.array_equal(on, off))


class TestDeepseekV4AoaDtypeGating(unittest.TestCase):
    """deepseek_v4/modeling.py:544 - the HF->PF weight-conversion generator
    keeps the mHC / head-contraction parameters in float32 unless the flag
    switches them to the DSV4 bfloat16 replay dtype (and drops the explicit
    float32 cast on the MoE gate weight)."""

    def _config(self):
        return types.SimpleNamespace(
            num_hidden_layers=1,
            n_routed_experts=1,
            n_shared_experts=1,
            moe_n_hash_layers=3,
            csa_dense_mode=False,
            csa_compress_ratios=[0],
            mtp_num_layers=0,
            num_nextn_predict_layers=0,
            tie_word_embeddings=True,
            enable_mtp_magic_send=False,
            moe_expert_fusion=False,
            fp8=False,
            moe_deep_gemm=False,
        )

    def _gen(self, enabled):
        with _dsv4_flag(modeling, enabled):
            return modeling.DeepseekV4PreTrainedModel._gen_aoa_config(
                self._config()
            )["aoa_statements"]

    @staticmethod
    def _find(stmts, needle):
        return next(s for s in stmts if needle in s)

    def test_flag_off_keeps_float32_conversion_dtypes(self):
        stmts = self._gen(False)
        self.assertIn(
            "dtype='float32'",
            self._find(stmts, "model.mhc_contract.hc_head_base"),
        )
        self.assertIn(
            "dtype='float32'",
            self._find(stmts, "self_attention_hyper_connection.alpha_pre,"),
        )
        # the MoE gate weight carries an explicit float32 cast when off.
        self.assertIn(
            "dtype='float32'",
            self._find(stmts, "-> model.layers.0.mlp.gate.weight"),
        )

    def test_flag_on_switches_to_bfloat16_and_drops_the_gate_cast(self):
        stmts = self._gen(True)
        self.assertIn(
            "dtype='bfloat16'",
            self._find(stmts, "model.mhc_contract.hc_head_base"),
        )
        self.assertIn(
            "dtype='bfloat16'",
            self._find(stmts, "self_attention_hyper_connection.alpha_pre,"),
        )
        gate = self._find(stmts, "-> model.layers.0.mlp.gate.weight")
        self.assertNotIn("dtype=", gate)


class TestLmHeadUnfusedSequenceFirstTranspose(unittest.TestCase):
    """lm_head.py:281-282 - the unfused LM-head projection transposes the logits
    back to batch-first after the sequence-first linear.

    The already-covered ``TestLmHeadSequenceFirstLinear`` runs the fused path
    (``fused_linear_ce_loss_chunk`` > 0), which returns the hidden states before
    ever reaching ``super().forward`` and so never touches line 282. Here the
    fused chunk is 0, so ``_forward`` runs the real
    ``ColumnParallelLinear.forward`` (stubbed to skip tensor-parallel comms).
    With the flag on the ``[S, B, V]`` logits are transposed back to
    ``[B, S, V]`` at line 282; with the flag off no transpose happens.
    """

    def _run(self, enabled):
        head = lm_head.GPTLMHead.__new__(lm_head.GPTLMHead)
        # Populate __dict__ directly: ``paddle.nn.Layer.__setattr__`` rejects
        # tensor assignment before ``__init__`` runs, and we deliberately skip
        # the real (TP-requiring) constructor here.
        head.__dict__["config"] = types.SimpleNamespace(
            sequence_parallel=False,
            fused_linear_ce_loss_chunk=0,
            gpt_model_use_experimental_version=False,
        )
        head.__dict__["weight"] = paddle.zeros([5, 4], dtype="float32")
        head.__dict__["bias"] = None
        # The (stubbed) column-parallel linear returns fixed [S, B, V] logits.
        seq_first_logits = paddle.to_tensor(
            np.arange(3 * 2 * 5, dtype="float32").reshape([3, 2, 5])
        )

        def fake_linear_forward(_self, _hidden, _weight):
            return seq_first_logits, None

        hidden = paddle.randn([2, 3, 4], dtype="float32")  # [B, S, H]
        with (
            _dsv4_flag(lm_head, enabled),
            patch.object(lm_head, "module_needs_recompute", return_value=False),
            patch.object(
                lm_head.ColumnParallelLinear, "forward", fake_linear_forward
            ),
        ):
            out = lm_head.GPTLMHead._forward(head, hidden)
        return out, seq_first_logits

    def test_flag_on_transposes_logits_back_to_batch_first(self):
        out, seq_first_logits = self._run(True)
        self.assertEqual(out.shape, [2, 3, 5])
        np.testing.assert_array_equal(
            out.numpy(), seq_first_logits.transpose([1, 0, 2]).numpy()
        )

    def test_flag_off_keeps_the_linear_output_untransposed(self):
        out, seq_first_logits = self._run(False)
        self.assertEqual(out.shape, [3, 2, 5])
        np.testing.assert_array_equal(out.numpy(), seq_first_logits.numpy())


class TestDeepseekV4MtpGateDtypeGating(unittest.TestCase):
    """deepseek_v4/modeling.py:821,824-825 - the MTP-layer MoE gate weight cast.

    The already-covered ``TestDeepseekV4AoaDtypeGating`` runs ``_gen_aoa_config``
    with ``mtp_num_layers=0``, so the MTP-layer loop (and its gate-weight
    statement at 821-829) never executes. With at least one MTP layer the same
    float32-cast gating applies: flag off keeps the explicit ``dtype='float32'``
    cast on the MTP gate weight (line 825), flag on drops it (bfloat16 replay).
    """

    def _config(self):
        return types.SimpleNamespace(
            num_hidden_layers=1,
            n_routed_experts=1,
            n_shared_experts=1,
            moe_n_hash_layers=3,
            csa_dense_mode=False,
            csa_compress_ratios=[0],
            mtp_num_layers=1,
            num_nextn_predict_layers=0,
            tie_word_embeddings=True,
            enable_mtp_magic_send=False,
            moe_expert_fusion=False,
            fp8=False,
            moe_deep_gemm=False,
        )

    def _gen(self, enabled):
        with _dsv4_flag(modeling, enabled):
            return modeling.DeepseekV4PreTrainedModel._gen_aoa_config(
                self._config()
            )["aoa_statements"]

    @staticmethod
    def _find(stmts, needle):
        return next(s for s in stmts if needle in s)

    def test_flag_off_keeps_the_float32_gate_cast(self):
        stmts = self._gen(False)
        gate = self._find(stmts, "mtp.0.ffn.gate.weight ->")
        self.assertIn("dtype='float32'", gate)

    def test_flag_on_drops_the_gate_cast(self):
        stmts = self._gen(True)
        gate = self._find(stmts, "mtp.0.ffn.gate.weight ->")
        self.assertNotIn("dtype=", gate)


if __name__ == "__main__":
    unittest.main()
