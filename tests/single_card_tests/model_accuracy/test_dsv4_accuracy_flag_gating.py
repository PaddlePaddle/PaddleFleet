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

import os
import types
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet import accuracy_compatible_patch
from paddlefleet.accuracy_compatible_patch import FixedTrainingData
from paddlefleet.datasets import collate
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
    """Only the flag may switch the fan-in order, and the order is visible.

    The two orders differ solely in BF16 rounding, so the coefficients below are
    chosen to make that difference observable: ``512 + 3`` rounds up to ``516``
    at BF16 precision (the ULP at 512 is 4), so adding ``-512`` afterwards keeps
    ``4``, while cancelling ``512 + (-512)`` first keeps the exact ``3``. An
    FP32 or same-magnitude probe would return the same value either way and
    would not pin the order down at all.
    """

    # (g_router, g_dispatcher, g_shared) as seen by backward().
    COEFFS = (-512.0, 512.0, 3.0)

    def _backward_grad(self, enabled):
        x = paddle.ones([2], dtype="bfloat16")
        x.stop_gradient = False
        g_router, g_dispatcher, g_shared = self.COEFFS
        with _dsv4_flag(moe_layer, enabled):
            router, dispatcher, shared = moe_layer.ThreePathCloneAlignMG.apply(
                x
            )
            (
                router * g_router
                + dispatcher * g_dispatcher
                + shared * g_shared
            ).sum().backward()
        return x.grad.astype("float32").numpy()

    def _reference(self, order):
        g_router, g_dispatcher, g_shared = (
            paddle.to_tensor([value] * 2, dtype="bfloat16")
            for value in self.COEFFS
        )
        if order == "megatron":
            out = (g_dispatcher + g_router) + g_shared
        else:
            out = (g_dispatcher + g_shared) + g_router
        return out.astype("float32").numpy()

    def test_flag_off_keeps_the_historical_fan_in_order(self):
        np.testing.assert_array_equal(
            self._backward_grad(False), self._reference("historical")
        )

    def test_flag_on_uses_the_megatron_fan_in_order(self):
        np.testing.assert_array_equal(
            self._backward_grad(True), self._reference("megatron")
        )

    def test_the_two_orders_are_actually_distinguishable(self):
        self.assertFalse(
            np.array_equal(
                self._backward_grad(False), self._backward_grad(True)
            )
        )


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


def _fixed_data_collate_args():
    """Minimal args that drive ``collate_fn`` down the fixed-token branches."""
    training_args = types.SimpleNamespace(
        num_nextn_predict_layers=0,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        fp8=False,
        max_seq_len=4,
    )
    model_args = types.SimpleNamespace(
        mtp_attention_flexible=False,
        use_attn_mask_startend_row_indices=True,
        use_global_causal_attn=False,
    )
    tokenizer = types.SimpleNamespace(pad_token_id=0)
    return training_args, model_args, tokenizer


class TestCollateFixedDataGating(unittest.TestCase):
    """``LOAD_FIXED_DATA_PATH`` only reaches the DSV4 replay loader when the flag is on.

    ``collate_fn`` has two fixed-token sources: the DSV4 ``load_fixed_training_data``
    replay and the historical ``np.load`` of ``tokens_*.npy`` / ``labels_*.npy``.
    ``FLAGS_use_dsv4_accuracy`` picks between them, so with the flag off the replay
    loader must never run and the historical ``.npy`` path must produce the batch.
    """

    def _run_with_fixed_path(self, enabled, fixed_data):
        training_args, model_args, tokenizer = _fixed_data_collate_args()
        with (
            _dsv4_flag(collate, enabled),
            patch.dict(os.environ, {"LOAD_FIXED_DATA_PATH": "/tmp/fixed"}),
            patch.object(
                accuracy_compatible_patch,
                "load_fixed_training_data",
                return_value=fixed_data,
            ) as load_fixed,
            patch.object(
                collate.np, "load", return_value=np.array([20, 21, 22, 23])
            ) as np_load,
        ):
            result = collate.collate_fn(
                batch=[None],
                tokenizer=tokenizer,
                training_args=training_args,
                model_args=model_args,
                max_seq_len=0,
                padding_free=False,
            )
        return result, load_fixed, np_load

    def test_flag_on_loads_via_the_dsv4_replay_loader(self):
        fixed_data = FixedTrainingData(
            input_ids=[10, 11, 12, 13],
            labels=[10, 11, 12, 13],
            position_ids=[0, 1, 2, 3],
            max_seq_len=4,
        )

        result, load_fixed, np_load = self._run_with_fixed_path(
            True, fixed_data
        )

        load_fixed.assert_called_once()
        np_load.assert_not_called()
        np.testing.assert_array_equal(result["input_ids"][0], [10, 11, 12, 13])

    def test_flag_off_falls_back_to_the_historical_npy_load(self):
        # The replay loader would happily return data, but the flag being off
        # must keep it from ever running.
        fixed_data = FixedTrainingData(
            input_ids=[10, 11, 12, 13],
            labels=[10, 11, 12, 13],
            position_ids=[0, 1, 2, 3],
            max_seq_len=4,
        )

        result, load_fixed, np_load = self._run_with_fixed_path(
            False, fixed_data
        )

        load_fixed.assert_not_called()
        self.assertEqual(np_load.call_count, 2)  # tokens_*.npy + labels_*.npy
        np.testing.assert_array_equal(result["input_ids"][0], [20, 21, 22, 23])

    def test_flag_on_without_the_path_never_calls_the_loader(self):
        # The replay call site is ``fixed_tokens_path and use_dsv4_...``; the
        # flag alone must not fire it when the env var is absent.
        training_args, model_args, tokenizer = _fixed_data_collate_args()
        sequence = types.SimpleNamespace(
            token_ids=[1, 2, 3],
            labels=[1, 2, 3],
            position_ids=[0, 1, 2],
        )

        with (
            _dsv4_flag(collate, True),
            patch.dict(os.environ),
            patch.object(
                accuracy_compatible_patch, "load_fixed_training_data"
            ) as load_fixed,
            patch.object(collate.np, "load") as np_load,
        ):
            os.environ.pop("LOAD_FIXED_DATA_PATH", None)
            result = collate.collate_fn(
                batch=[[sequence]],
                tokenizer=tokenizer,
                training_args=training_args,
                model_args=model_args,
                max_seq_len=0,
                padding_free=False,
            )

        load_fixed.assert_not_called()
        np_load.assert_not_called()
        np.testing.assert_array_equal(result["input_ids"][0], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
