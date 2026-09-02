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

"""HySparseTransformerLayer._mtp_split must honor use_erndata=True.

Under megatron style position_ids / attn_mask_startend_row_indices arrive at
main-decoder length L (per-doc shifting happens inside the MTP layer via
roll_tensor), so the ernie5 L+K -> L seq-dim trims must be skipped -- exactly
like the ``_mtp_is_megatron`` guards in ``TransformerLayer.forward``. The
hidden-states batch-dim (K+1)-way split is style-independent and must keep
working in both styles.

Pure CPU, no fleet init: ``_mtp_split`` is exercised unbound with a fake self.
"""

from types import SimpleNamespace

import paddle
import pytest

from paddlefleet.transformer.transformer_layer import HySparseTransformerLayer

K = 2
B, L, H = 1, 12, 4


def _make_config(use_erndata):
    return SimpleNamespace(
        num_nextn_predict_layers=K,
        mtp_load_weight_only=False,
        enable_mtp_magic_send=False,
        gpt_model_use_experimental_version=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        experimental_dataflow=False,
        separate_mtp_input=False,
        use_erndata=use_erndata,
    )


def _make_fake_layer(use_erndata):
    fake = SimpleNamespace(config=_make_config(use_erndata))
    fake._mtp_enabled = lambda is_mtp: HySparseTransformerLayer._mtp_enabled(
        fake, is_mtp
    )
    return fake


def _make_dict_args(seq_len):
    """dict_args as seen by the main decoder layer.

    hidden_states is the MTP (K+1)-way batch-dim stack in both styles;
    position_ids / mask carry seq_len entries (L for megatron, L+K for ernie5).
    input_ids stays None to keep the test free of fleet/CP process groups.
    """
    return {
        "hidden_states": paddle.randn([(K + 1) * B, L, H]),
        "position_ids": paddle.arange(seq_len)
        .reshape([1, seq_len])
        .tile([B, 1]),
        "attn_mask_startend_row_indices": paddle.full(
            [B, 1, seq_len, 1], seq_len, dtype="int32"
        ),
        "input_ids": None,
    }


class TestMegatronSkipsSeqTrims:
    def test_position_ids_kept_full_length(self):
        fake = _make_fake_layer(True)
        dict_args = _make_dict_args(L)
        ctx = HySparseTransformerLayer._mtp_split(fake, dict_args, is_mtp=False)
        assert ctx is not None
        assert dict_args["position_ids"].shape == [B, L]
        assert ctx["mtp_ids"] is None

    def test_attn_mask_kept_full_length(self):
        fake = _make_fake_layer(True)
        dict_args = _make_dict_args(L)
        ctx = HySparseTransformerLayer._mtp_split(fake, dict_args, is_mtp=False)
        assert dict_args["attn_mask_startend_row_indices"].shape == [B, 1, L, 1]
        assert ctx["attn_mask_mtp"] is None

    def test_hidden_states_split_still_applies(self):
        fake = _make_fake_layer(True)
        dict_args = _make_dict_args(L)
        stacked = dict_args["hidden_states"].clone()
        ctx = HySparseTransformerLayer._mtp_split(fake, dict_args, is_mtp=False)
        assert dict_args["hidden_states"].shape == [B, L, H]
        assert len(ctx["mtp_input"]) == K
        import numpy as np

        np.testing.assert_array_equal(
            dict_args["hidden_states"].numpy(), stacked[:B].numpy()
        )

    def test_restore_roundtrips_without_touching_aux(self):
        fake = _make_fake_layer(True)
        dict_args = _make_dict_args(L)
        ctx = HySparseTransformerLayer._mtp_split(fake, dict_args, is_mtp=False)
        output = dict_args["hidden_states"]
        restored = HySparseTransformerLayer._mtp_restore(
            fake, dict_args, output, ctx
        )
        assert restored.shape == [(K + 1) * B, L, H]
        assert dict_args["position_ids"].shape == [B, L]
        assert dict_args["attn_mask_startend_row_indices"].shape == [B, 1, L, 1]


class TestErnie5TrimsStillApply:
    """Regression guard: adding the megatron guard must not disturb ernie5."""

    def test_position_ids_and_mask_trimmed(self):
        fake = _make_fake_layer(False)
        dict_args = _make_dict_args(L + K)
        ctx = HySparseTransformerLayer._mtp_split(fake, dict_args, is_mtp=False)
        assert dict_args["position_ids"].shape == [B, L]
        assert ctx["mtp_ids"].shape == [B, K]
        assert dict_args["attn_mask_startend_row_indices"].shape == [B, 1, L, 1]
        assert ctx["attn_mask_mtp"].shape == [B, 1, K, 1]

    def test_restore_reassembles_full_length(self):
        fake = _make_fake_layer(False)
        dict_args = _make_dict_args(L + K)
        ctx = HySparseTransformerLayer._mtp_split(fake, dict_args, is_mtp=False)
        output = dict_args["hidden_states"]
        HySparseTransformerLayer._mtp_restore(fake, dict_args, output, ctx)
        assert dict_args["position_ids"].shape == [B, L + K]
        assert dict_args["attn_mask_startend_row_indices"].shape == [
            B,
            1,
            L + K,
            1,
        ]


class TestMtpDisabledIsNoop:
    @pytest.mark.parametrize("use_erndata", [False, True])
    def test_is_mtp_layer_returns_none(self, use_erndata):
        fake = _make_fake_layer(use_erndata)
        dict_args = _make_dict_args(L)
        assert (
            HySparseTransformerLayer._mtp_split(fake, dict_args, is_mtp=True)
            is None
        )
