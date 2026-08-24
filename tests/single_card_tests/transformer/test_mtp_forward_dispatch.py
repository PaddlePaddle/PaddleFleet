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

"""Single-card coverage for MultiTokenPredictionLayer._forward_megatron_style.

Drives the REAL method (not an inline replica) via ``__new__`` +
MagicMock config + a stubbed ``_proj_and_transformer_layer`` so the whole
megatron-style prologue is exercised:

- ``forward`` dispatch to ``_forward_megatron_style`` under
  ``use_erndata=True`` (multi_token_prediction.py:1062).
- (K+1) split, per-depth hidden_states/decoder_input dispatch, field pops
  (lines 1636-1656).
- per-depth attn_mask_startend_row_indices derivation from cu_seqlens_q,
  both the 1-col (fleet) and 2-col (experimental / include_pos) layouts
  (lines 1667-1717), plus batch_size>1 expand (1713-1716).
- concat write-back (lines 1723-1726).
- guard raises: cross-attention (1621-1625), magic-send incompatibility
  (1626-1634), and 3-D input_ids (1656-1660).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import paddle

from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)


def _make_layer(
    K: int,
    *,
    layer_number: int = 0,
    include_pos: bool = False,
    sequence_parallel: bool = False,
    magic_send: bool = False,
):
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    cfg = MagicMock()
    cfg.use_erndata = True
    cfg.enable_mtp_magic_send = magic_send
    cfg.num_nextn_predict_layers = K
    cfg.gpt_model_use_experimental_version = include_pos
    cfg.sequence_parallel = sequence_parallel
    layer.config = cfg
    layer.layer_number = layer_number

    recorded = {}

    def _stub_proj(hidden_states, decoder_input, **kwargs):
        recorded["hidden_states_shape"] = list(hidden_states.shape)
        recorded["decoder_input_shape"] = list(decoder_input.shape)
        am = kwargs.get("attn_mask_startend_row_indices")
        recorded["attn_mask_shape"] = None if am is None else list(am.shape)
        recorded["attn_mask"] = None if am is None else am
        recorded["kwargs"] = kwargs
        return decoder_input

    layer._proj_and_transformer_layer = _stub_proj
    return layer, recorded


class TestMtpForwardMegatron(unittest.TestCase):
    def test_forward_dispatches_to_megatron(self) -> None:
        # forward() must route to _forward_megatron_style (line 1062).
        K, S, H = 1, 8, 4
        layer, recorded = _make_layer(K)
        hs = paddle.arange((K + 1) * S * H, dtype="float32").reshape(
            [K + 1, S, H]
        )
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        out = layer.forward(
            {"hidden_states": hs, "cu_seqlens_q": cu, "context": None}
        )
        self.assertEqual(recorded["hidden_states_shape"], [1, S, H])
        self.assertEqual(list(out["hidden_states"].shape), [K + 1, S, H])
        self.assertNotIn("decoder_input", out)

    def test_fleet_layout_attn_mask_values(self) -> None:
        # 1-col [1,1,S,1] layout (include_pos=False) with derived ends.
        K, S, H = 1, 8, 4
        layer, recorded = _make_layer(K, include_pos=False)
        hs = paddle.arange((K + 1) * S * H, dtype="float32").reshape(
            [K + 1, S, H]
        )
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        layer._forward_megatron_style(
            {"hidden_states": hs, "cu_seqlens_q": cu, "context": None}
        )
        self.assertEqual(recorded["attn_mask_shape"], [1, 1, S, 1])
        ends = [row[0] for row in recorded["attn_mask"].numpy().tolist()[0][0]]
        self.assertEqual(ends, [3, 3, 3, 8, 8, 8, 8, 8])

    def test_experimental_include_pos_layout(self) -> None:
        # include_pos=True -> 2-col [1,1,S,2] layout (lines 1697-1700).
        K, S, H = 1, 8, 4
        layer, recorded = _make_layer(K, include_pos=True)
        hs = paddle.arange((K + 1) * S * H, dtype="float32").reshape(
            [K + 1, S, H]
        )
        cu = paddle.to_tensor([0, 4, 8], dtype="int32")
        layer._forward_megatron_style(
            {"hidden_states": hs, "cu_seqlens_q": cu, "context": None}
        )
        self.assertEqual(recorded["attn_mask_shape"], [1, 1, S, 2])
        vals = recorded["attn_mask"].numpy().tolist()[0][0]
        ends = [row[0] for row in vals]
        pos = [row[1] for row in vals]
        self.assertEqual(ends, [4, 4, 4, 4, 8, 8, 8, 8])
        self.assertEqual(pos, list(range(S)))

    def test_batch_size_gt_one_expands(self) -> None:
        # batch_size>1 -> expand branch (lines 1713-1716).
        K, S, H, B = 1, 8, 4, 2
        layer, recorded = _make_layer(K)
        total = (K + 1) * B
        hs = paddle.arange(total * S * H, dtype="float32").reshape(
            [total, S, H]
        )
        cu = paddle.to_tensor([0, 5, 8], dtype="int32")
        layer._forward_megatron_style(
            {"hidden_states": hs, "cu_seqlens_q": cu, "context": None}
        )
        self.assertEqual(recorded["attn_mask_shape"], [B, 1, S, 1])

    def test_no_cu_seqlens_skips_mask(self) -> None:
        # cu_seqlens_q=None -> derivation skipped (line 1668 False branch);
        # no attn_mask_startend_row_indices set.
        K, S, H = 2, 6, 4
        layer, recorded = _make_layer(K)
        hs = paddle.arange((K + 1) * S * H, dtype="float32").reshape(
            [K + 1, S, H]
        )
        out = layer._forward_megatron_style(
            {"hidden_states": hs, "context": None}
        )
        self.assertIsNone(recorded["attn_mask_shape"])
        self.assertEqual(list(out["hidden_states"].shape), [K + 1, S, H])

    def test_cross_attention_raises(self) -> None:
        # context is not None -> NotImplementedError (lines 1621-1625).
        K, S, H = 1, 8, 4
        layer, _ = _make_layer(K)
        hs = paddle.zeros([(K + 1), S, H], dtype="float32")
        with self.assertRaises(NotImplementedError):
            layer._forward_megatron_style(
                {"hidden_states": hs, "context": object()}
            )

    def test_magic_send_incompatible_raises(self) -> None:
        # enable_mtp_magic_send=True -> ValueError (lines 1626-1634).
        K, S, H = 1, 8, 4
        layer, _ = _make_layer(K, magic_send=True)
        hs = paddle.zeros([(K + 1), S, H], dtype="float32")
        with self.assertRaises(ValueError):
            layer._forward_megatron_style(
                {"hidden_states": hs, "context": None}
            )

    def test_mtp_input_embeds_incompatible_raises(self) -> None:
        # mtp_input_embeds present -> ValueError (lines 1626-1634).
        K, S, H = 1, 8, 4
        layer, _ = _make_layer(K)
        hs = paddle.zeros([(K + 1), S, H], dtype="float32")
        with self.assertRaises(ValueError):
            layer._forward_megatron_style(
                {
                    "hidden_states": hs,
                    "context": None,
                    "mtp_input_embeds": paddle.zeros([1], dtype="float32"),
                }
            )

    def test_3d_input_ids_raises(self) -> None:
        # input_ids with ndim>2 -> RuntimeError (lines 1656-1660).
        K, S, H = 1, 8, 4
        layer, _ = _make_layer(K)
        hs = paddle.zeros([(K + 1), S, H], dtype="float32")
        bad_ids = paddle.zeros([1, K, S], dtype="int64")
        with self.assertRaises(RuntimeError):
            layer._forward_megatron_style(
                {
                    "hidden_states": hs,
                    "context": None,
                    "input_ids": bad_ids,
                }
            )


if __name__ == "__main__":
    unittest.main()
