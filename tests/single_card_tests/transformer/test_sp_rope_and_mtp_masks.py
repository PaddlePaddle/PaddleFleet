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

"""Sequence-parallel rotary rows and per-depth MTP mask delivery contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.transformer import multi_token_prediction as mtp


class TestSequenceParallelRotary(unittest.TestCase):
    def test_global_and_local_tables_use_the_same_rank_rows_and_gradient(self):
        values = np.arange(1, 9, dtype="float32").reshape(1, 2, 1, 4)
        angles = np.arange(16, dtype="float32").reshape(1, 4, 1, 4) / 16
        upstream = values / 8
        for time_major in [False, True]:
            for rank in [0, 1]:
                for local_table in [False, True]:
                    with self.subTest(
                        time_major=time_major,
                        rank=rank,
                        local_table=local_table,
                    ):
                        selected = angles[:, rank * 2 : rank * 2 + 2]
                        frequency = selected if local_table else angles
                        inputs = paddle.to_tensor(
                            values.transpose(1, 0, 2, 3)
                            if time_major
                            else values
                        )
                        inputs.stop_gradient = False
                        freqs = paddle.to_tensor(
                            frequency.transpose(1, 0, 2, 3)
                            if time_major
                            else frequency
                        )
                        output = _apply_rotary_pos_emb_bshd(
                            inputs,
                            freqs,
                            time_major=time_major,
                            sp_group=SimpleNamespace(nranks=2, rank=rank),
                        )
                        weights = paddle.to_tensor(
                            upstream.transpose(1, 0, 2, 3)
                            if time_major
                            else upstream
                        )
                        (output * weights).sum().backward()
                        rotated = np.concatenate(
                            [-values[..., 2:], values[..., :2]], axis=-1
                        )
                        expected = values * np.cos(selected) + rotated * np.sin(
                            selected
                        )
                        weighted_sine = upstream * np.sin(selected)
                        transpose_rotation = np.concatenate(
                            [weighted_sine[..., 2:], -weighted_sine[..., :2]],
                            axis=-1,
                        )
                        gradient = (
                            upstream * np.cos(selected) + transpose_rotation
                        )
                        actual = (
                            output.numpy().transpose(1, 0, 2, 3)
                            if time_major
                            else output.numpy()
                        )
                        actual_gradient = (
                            inputs.grad.numpy().transpose(1, 0, 2, 3)
                            if time_major
                            else inputs.grad.numpy()
                        )
                        np.testing.assert_allclose(
                            actual, expected, rtol=2e-6, atol=2e-6
                        )
                        np.testing.assert_allclose(
                            actual_gradient, gradient, rtol=2e-6, atol=2e-6
                        )


class TestMTPDepthMaskDelivery(unittest.TestCase):
    def _run(
        self,
        *,
        depth=1,
        train_only=False,
        separate=False,
        original_mask=True,
        invalid=None,
    ):
        chunks = np.arange(36, dtype="float32").reshape(3, 2, 3, 2) / 8
        masks = np.arange(36, dtype="float32").reshape(2, 2, 3, 3) / 8
        hidden_masks = np.array(
            [[[1, 0, 1], [0, 1, 1]], [[0, 1, 1], [1, 0, 1]]], dtype="float32"
        )
        dense = paddle.to_tensor(masks)
        hidden = paddle.to_tensor(hidden_masks)
        backbone_mask = paddle.zeros([2, 1, 3, 3])
        recorded = []

        def project(
            hidden_states,
            decoder_input,
            attention_mask,
            mtp_hidden_inputs_mask=None,
            **kwargs,
        ):
            if invalid is not None:
                self.fail(
                    "invalid MTP masks reached the transformer projection"
                )
            recorded.append(
                (attention_mask.numpy(), mtp_hidden_inputs_mask.numpy())
            )
            offset = attention_mask.sum(axis=-1).squeeze(1).unsqueeze(-1)
            gate = mtp_hidden_inputs_mask.squeeze(1).unsqueeze(-1)
            return hidden_states + decoder_input * gate + offset

        config = SimpleNamespace(
            use_erndata=False,
            enable_mtp_magic_send=False,
            separate_mtp_input=separate,
            sequence_parallel=False,
            num_nextn_predict_layers=2,
            train_mtp_only=train_only,
            gpt_model_use_experimental_version=False,
            recompute_granularity=None,
        )
        layer = SimpleNamespace(
            config=config,
            layer_number=depth,
            training=False,
            _proj_and_transformer_layer=project,
        )
        payload = {
            "hidden_states": paddle.to_tensor(
                chunks[: depth + 1].reshape(-1, 3, 2)
                if separate
                else chunks.reshape(-1, 3, 2)
            ),
            "mtp_attn_mask": dense,
            "mtp_hidden_inputs_mask_all": hidden,
        }
        if original_mask:
            payload["attention_mask"] = backbone_mask
        if separate:
            payload["mtp_decoder_inputs"] = [
                paddle.to_tensor(chunk) for chunk in chunks[1:]
            ]
        if invalid == "both":
            payload["mtp_startend_row_indices_all"] = paddle.zeros(
                [2, 2, 3, 1], dtype="int32"
            )
        if invalid == "missing_hidden":
            payload.pop("mtp_hidden_inputs_mask_all")
        with patch.object(
            mtp, "get_context_parallel_world_size", return_value=1
        ):
            output = mtp.MultiTokenPredictionLayer.forward(layer, payload)
        depths = list(range(2)) if train_only else [depth]
        expected = chunks.copy()
        for i in depths:
            expected[i + 1] = (
                expected[i]
                + expected[i + 1] * hidden_masks[:, i, :, None]
                + masks[:, i].sum(axis=-1)[..., None]
            )
        np.testing.assert_allclose(
            output["hidden_states"].numpy(),
            expected.reshape(-1, 3, 2),
            rtol=0,
            atol=0,
        )
        for (actual_mask, actual_hidden), i in zip(
            recorded, depths, strict=True
        ):
            np.testing.assert_array_equal(actual_mask, masks[:, i : i + 1])
            np.testing.assert_array_equal(
                actual_hidden, hidden_masks[:, i : i + 1]
            )
        np.testing.assert_array_equal(output["mtp_attn_mask"].numpy(), masks)
        np.testing.assert_array_equal(
            output["mtp_hidden_inputs_mask_all"].numpy(), hidden_masks
        )
        if not separate:
            if original_mask:
                self.assertIs(output["attention_mask"], backbone_mask)
            else:
                self.assertNotIn("attention_mask", output)
            self.assertNotIn("mtp_hidden_inputs_mask", output)
        np.testing.assert_array_equal(dense.numpy(), masks)
        np.testing.assert_array_equal(hidden.numpy(), hidden_masks)

    def test_concat_masks_follow_depth_and_restore_the_backbone_mask(self):
        for train_only in [False, True]:
            for original_mask in [False, True]:
                with self.subTest(
                    train_only=train_only, original_mask=original_mask
                ):
                    self._run(
                        train_only=train_only, original_mask=original_mask
                    )

    def test_separate_embeddings_receive_the_matching_dense_and_hidden_masks(
        self,
    ):
        self._run(separate=True)

    def test_conflicting_or_unpaired_masks_are_rejected_before_projection(self):
        for separate in [False, True]:
            for invalid in ["both", "missing_hidden"]:
                with (
                    self.subTest(separate=separate, invalid=invalid),
                    self.assertRaisesRegex(
                        ValueError,
                        "mutually exclusive|requires mtp_hidden_inputs_mask_all",
                    ),
                ):
                    self._run(separate=separate, invalid=invalid)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
