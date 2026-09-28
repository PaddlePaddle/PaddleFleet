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
  ``use_erndata=True``.
- (K+1) split, per-depth hidden_states/decoder_input dispatch, field pops.
- backbone ``attn_mask_startend_row_indices`` is reused as-is (the adapter
  ships it; this path does not derive it from cu_seqlens_q).
- concat write-back.
- guard raises: cross-attention, missing packed magic metadata, and 3-D input_ids.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
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
        # forward() must route to _forward_megatron_style.
        K, S, H = 1, 8, 4
        layer, recorded = _make_layer(K)
        hs = paddle.arange((K + 1) * S * H, dtype="float32").reshape(
            [K + 1, S, H]
        )
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        mask = paddle.full([1, 1, S, 1], S, dtype="int32")
        out = layer.forward(
            {
                "hidden_states": hs,
                "cu_seqlens_q": cu,
                "attn_mask_startend_row_indices": mask,
                "context": None,
            }
        )
        self.assertEqual(recorded["hidden_states_shape"], [1, S, H])
        self.assertEqual(list(out["hidden_states"].shape), [K + 1, S, H])
        self.assertNotIn("decoder_input", out)

    def test_supplied_mask_is_reused(self) -> None:
        # Adapter-owned 1-col mask is forwarded unchanged; not rebuilt from
        # cu_seqlens_q.
        K, S, H = 1, 8, 4
        layer, recorded = _make_layer(K, include_pos=False)
        hs = paddle.arange((K + 1) * S * H, dtype="float32").reshape(
            [K + 1, S, H]
        )
        mask = paddle.full([1, 1, S, 1], 8, dtype="int32")
        layer._forward_megatron_style(
            {
                "hidden_states": hs,
                "cu_seqlens_q": paddle.to_tensor([0, 3, 8], dtype="int32"),
                "attn_mask_startend_row_indices": mask,
                "context": None,
            }
        )
        self.assertEqual(recorded["attn_mask_shape"], [1, 1, S, 1])
        np.testing.assert_array_equal(
            recorded["attn_mask"].numpy(), mask.numpy()
        )

    def test_absent_mask_raises(self) -> None:
        K, S, H = 2, 6, 4
        layer, _ = _make_layer(K)
        hs = paddle.arange((K + 1) * S * H, dtype="float32").reshape(
            [K + 1, S, H]
        )
        with self.assertRaisesRegex(
            RuntimeError, r"attn_mask_startend_row_indices"
        ):
            layer._forward_megatron_style(
                {
                    "hidden_states": hs,
                    "cu_seqlens_q": paddle.to_tensor([0, 3, 6], dtype="int32"),
                    "context": None,
                }
            )

    def test_cross_attention_raises(self) -> None:
        # context is not None -> NotImplementedError.
        K, S, H = 1, 8, 4
        layer, _ = _make_layer(K)
        hs = paddle.zeros([(K + 1), S, H], dtype="float32")
        with self.assertRaises(NotImplementedError):
            layer._forward_megatron_style(
                {"hidden_states": hs, "context": object()}
            )

    def test_magic_send_missing_metadata_raises(self) -> None:
        K, S, H = 1, 8, 4
        layer, _ = _make_layer(K, magic_send=True)
        layer.mhc_enabled = False
        layer.sequence_parallel = False
        hs = paddle.zeros([1, S, H], dtype="float32")
        with self.assertRaisesRegex(RuntimeError, r"mtp_full_input_ids"):
            layer._forward_megatron_style(
                {"hidden_states": hs, "context": None}
            )

    def test_mtp_input_embeds_incompatible_raises(self) -> None:
        # mtp_input_embeds present -> ValueError.
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
        # input_ids with ndim>2 -> RuntimeError.
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
