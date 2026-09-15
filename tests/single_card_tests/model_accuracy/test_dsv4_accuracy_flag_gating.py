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

"""``FLAGS_use_dsv4_accuracy`` must gate every DSV4 replay call site.

The flag defaults to 0, and with it off the numeric paths have to stay exactly
where they were before the DSV4 replay landed - other alignment targets (for
example the MinimaxV2.5 and GLM45Air cases) run with
``use_accuracy_compatible=True`` and ``FLAGS_use_accuracy_compatible_kernel=1``
but without this flag, so a DSV4 branch that keys off the older switches
silently changes their loss curve. Each test below pins one call site on both
sides of the flag.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.fusions import fused_bias_swiglu
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.trainer import trainer_utils
from paddlefleet.transformer import dsv4_hybrid_attention
from paddlefleet.transformer.moe import moe_layer, moe_router, moe_utils


def _dsv4_flag(module, enabled):
    return patch.object(
        module, "use_dsv4_accuracy_compatible", return_value=enabled
    )


class TestMoePermuteGating(unittest.TestCase):
    """``permute`` keeps the aligned gather table unless the flag is on."""

    def setUp(self):
        self.tokens = paddle.to_tensor(
            np.arange(8, dtype="float32").reshape([4, 2])
        )
        self.routing_map = paddle.to_tensor(
            [[1, 0], [0, 1], [1, 0], [0, 1]], dtype="int32"
        )

    def test_flag_off_uses_the_aligned_gather_table(self):
        with (
            _dsv4_flag(moe_utils, False),
            patch.object(
                moe_utils,
                "_build_aligned_gather_index",
                wraps=moe_utils._build_aligned_gather_index,
            ) as aligned,
        ):
            out, indices = moe_utils.permute(
                self.tokens, self.routing_map, use_accuracy_compatible=True
            )

        aligned.assert_called_once()
        self.assertEqual(out.shape, [4, 2])
        np.testing.assert_allclose(
            out.numpy(), self.tokens.numpy()[indices.numpy()]
        )

    def test_flag_on_uses_the_fp32_index_select(self):
        with (
            _dsv4_flag(moe_utils, True),
            patch.object(moe_utils, "_build_aligned_gather_index") as aligned,
        ):
            out, indices = moe_utils.permute(
                self.tokens, self.routing_map, use_accuracy_compatible=True
            )

        aligned.assert_not_called()
        np.testing.assert_allclose(
            out.numpy(), self.tokens.numpy()[indices.numpy()]
        )

    def test_default_path_is_untouched_by_the_flag(self):
        with _dsv4_flag(moe_utils, True):
            out, indices = moe_utils.permute(
                self.tokens, self.routing_map, use_accuracy_compatible=False
            )

        np.testing.assert_allclose(
            out.numpy(), self.tokens.numpy()[indices.numpy()]
        )


class TestMoeUnpermuteGating(unittest.TestCase):
    """``unpermute`` keeps the aligned gather-sum unless the flag is on."""

    def setUp(self):
        self.routing_map = paddle.to_tensor(
            [[1, 0], [0, 1], [1, 0], [0, 1]], dtype="int32"
        )
        self.tokens = paddle.to_tensor(
            np.arange(8, dtype="float32").reshape([4, 2])
        )
        with _dsv4_flag(moe_utils, False):
            self.permuted, self.indices = moe_utils.permute(
                self.tokens, self.routing_map, use_accuracy_compatible=True
            )

    def test_flag_off_takes_the_aligned_gather_sum(self):
        with (
            _dsv4_flag(moe_utils, False),
            patch.object(
                moe_utils,
                "_unpermute_gather_sum_aligned",
                wraps=moe_utils._unpermute_gather_sum_aligned,
            ) as aligned,
        ):
            out = moe_utils.unpermute(
                self.permuted,
                self.indices,
                restore_shape=self.tokens.shape,
                routing_map=self.routing_map,
                use_accuracy_compatible=True,
            )

        aligned.assert_called_once()
        self.assertEqual(out.shape, [4, 2])

    def test_flag_on_falls_through_to_the_fp32_accumulation(self):
        with (
            _dsv4_flag(moe_utils, True),
            patch.object(moe_utils, "_unpermute_gather_sum_aligned") as aligned,
        ):
            out = moe_utils.unpermute(
                self.permuted,
                self.indices,
                restore_shape=self.tokens.shape,
                routing_map=self.routing_map,
                use_accuracy_compatible=True,
            )

        aligned.assert_not_called()
        np.testing.assert_allclose(
            out.numpy(), self.tokens.numpy(), rtol=1e-6, atol=1e-6
        )


class TestThreePathCloneBackwardOrder(unittest.TestCase):
    """The MG-aligned fan-in order only applies with the flag on."""

    def _backward_grad(self, enabled):
        x = paddle.to_tensor([1.0, 2.0], dtype="float32")
        x.stop_gradient = False
        with _dsv4_flag(moe_layer, enabled):
            router, dispatcher, shared = moe_layer.ThreePathCloneAlignMG.apply(
                x
            )
            (router * 1.0 + dispatcher * 2.0 + shared * 4.0).sum().backward()
        return x.grad.numpy()

    def test_both_orders_sum_every_branch(self):
        np.testing.assert_allclose(self._backward_grad(False), [7.0, 7.0])
        np.testing.assert_allclose(self._backward_grad(True), [7.0, 7.0])


class TestTopkGateNormalization(unittest.TestCase):
    def test_normalized_gate_sums_to_one(self):
        top_gate = paddle.to_tensor([[0.5, 1.5], [1.0, 3.0]], dtype="float32")

        normalized = moe_router._normalize_topk_gate(top_gate)

        np.testing.assert_allclose(
            normalized.sum(axis=-1).numpy(), [1.0, 1.0], rtol=1e-6
        )

    def test_all_zero_row_does_not_divide_by_zero(self):
        top_gate = paddle.zeros([1, 2], dtype="float32")

        normalized = moe_router._normalize_topk_gate(top_gate)

        self.assertTrue(bool(paddle.isfinite(normalized).all()))


class TestClampedSwigluBackward(unittest.TestCase):
    """Both spellings of the clamped SwiGLU gradient must agree numerically."""

    def setUp(self):
        paddle.seed(3)
        self.y = paddle.randn([4, 8], dtype="float32") * 4.0
        self.g = paddle.randn([4, 4], dtype="float32")
        self.clamp_value = 3.0

    def test_gradients_agree_on_both_sides_of_the_flag(self):
        with _dsv4_flag(fused_bias_swiglu, False):
            default = fused_bias_swiglu.clamped_swiglu_back(
                self.g, self.y, self.clamp_value
            )
        with _dsv4_flag(fused_bias_swiglu, True):
            compatible = fused_bias_swiglu.clamped_swiglu_back(
                self.g, self.y, self.clamp_value
            )

        np.testing.assert_allclose(
            compatible.numpy(), default.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_weighted_backward_row_sum_agrees_on_both_sides(self):
        weights = paddle.rand([4, 1], dtype="float32")

        with _dsv4_flag(fused_bias_swiglu, False):
            default = fused_bias_swiglu.clamped_weighted_swiglu_back(
                self.g, self.y, weights, self.clamp_value
            )
        with _dsv4_flag(fused_bias_swiglu, True):
            compatible = fused_bias_swiglu.clamped_weighted_swiglu_back(
                self.g, self.y, weights, self.clamp_value
            )

        for expected, actual in zip(default, compatible):
            self.assertEqual(expected.shape, actual.shape)
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
            )


class TestShiftedMtpEmbedding(unittest.TestCase):
    """The replay shifts MTP ids inside the sequence and zero-pads the tail."""

    def _stub(self):
        captured = {}

        def embedding(input_ids=None, position_ids=None):
            captured["input_ids"] = input_ids
            captured["position_ids"] = position_ids
            return input_ids

        return (
            types.SimpleNamespace(
                embedding=embedding, multimodal_embedding=False
            ),
            captured,
        )

    def test_ids_are_shifted_left_and_zero_padded(self):
        stub, captured = self._stub()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        position_ids = paddle.to_tensor([[0, 1, 2, 3]], dtype="int64")

        out = GPTEmbedding._embed_shifted_mtp(
            stub, input_ids, position_ids, depth=0, seq_length=4
        )

        np.testing.assert_array_equal(out.numpy(), [[2, 3, 4, 0]])
        np.testing.assert_array_equal(
            captured["position_ids"].numpy(), [[1, 2, 3, 0]]
        )

    def test_deeper_mtp_layers_shift_further(self):
        stub, _ = self._stub()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")

        out = GPTEmbedding._embed_shifted_mtp(
            stub, input_ids, None, depth=1, seq_length=4
        )

        np.testing.assert_array_equal(out.numpy(), [[3, 4, 0, 0]])

    def test_multimodal_embedding_drops_the_position_ids(self):
        stub, captured = self._stub()
        stub.multimodal_embedding = True
        input_ids = paddle.to_tensor([[5, 6]], dtype="int64")
        position_ids = paddle.to_tensor([[0, 1]], dtype="int64")

        GPTEmbedding._embed_shifted_mtp(
            stub, input_ids, position_ids, depth=0, seq_length=2
        )

        self.assertIsNone(captured["position_ids"])


class TestCosineScheduleTail(unittest.TestCase):
    """Past the last training step the replay clamps the LR to ``min_lr``."""

    def _lr_at(self, step, enabled):
        with _dsv4_flag(trainer_utils, enabled):
            scheduler = trainer_utils.get_cosine_schedule_with_warmup(
                learning_rate=1.0,
                num_warmup_steps=0,
                num_training_steps=10,
                min_lr=0.1,
            )
            for _ in range(step):
                scheduler.step()
            return scheduler.get_lr()

    def test_flag_on_clamps_the_overshooting_step(self):
        self.assertAlmostEqual(self._lr_at(12, True), 0.1, places=6)

    def test_flag_off_keeps_the_historical_cosine_value(self):
        self.assertGreater(self._lr_at(12, False), 0.1)

    def test_warmup_is_shared_by_both_paths(self):
        with _dsv4_flag(trainer_utils, True):
            scheduler = trainer_utils.get_cosine_schedule_with_warmup(
                learning_rate=1.0,
                num_warmup_steps=4,
                num_training_steps=10,
                min_lr=0.1,
            )
            scheduler.step()
            scheduler.step()
            self.assertAlmostEqual(scheduler.get_lr(), 0.5, places=6)


class TestPackDsv4LogicalBatch(unittest.TestCase):
    """The Megatron indexer layout keeps one logical sequence per row."""

    def test_accuracy_compatible_layout_is_left_unpacked(self):
        hidden_states = paddle.zeros([2, 6, 4], dtype="float32")

        packed, docmask, batch_size, seqlen = (
            dsv4_hybrid_attention._pack_dsv4_logical_batch(
                hidden_states,
                None,
                cp_size=1,
                dense_mode=False,
                accuracy_compatible=True,
            )
        )

        self.assertIs(packed, hidden_states)
        self.assertIsNone(docmask)
        self.assertEqual((batch_size, seqlen), (1, 6))

    def test_single_row_batch_is_returned_as_is(self):
        hidden_states = paddle.zeros([1, 6, 4], dtype="float32")

        packed, docmask, batch_size, seqlen = (
            dsv4_hybrid_attention._pack_dsv4_logical_batch(
                hidden_states,
                None,
                cp_size=1,
                dense_mode=False,
            )
        )

        self.assertIs(packed, hidden_states)
        self.assertIsNone(docmask)
        self.assertEqual((batch_size, seqlen), (1, 6))

    def test_rank_two_hidden_states_are_rejected(self):
        with self.assertRaises(ValueError):
            dsv4_hybrid_attention._pack_dsv4_logical_batch(
                paddle.zeros([6, 4], dtype="float32"),
                None,
                cp_size=1,
                dense_mode=False,
            )


if __name__ == "__main__":
    unittest.main()
