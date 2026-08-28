# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import paddle

from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    IndexCacheServedDistillLossAutoScaler,
    TileLangCSAIndexerDistillBridge,
    TileLangCSAIndexerLossAutoScaler,
    _indexcache_offload_saved_delta,
    _indexcache_restore_saved_delta,
    _unpack_indexcache_pipeline_topk,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossAutoScaler
from paddlefleet.transformer.indexcache_state import (
    INDEXCACHE_DISTILL_GRAD_INDICES,
    INDEXCACHE_DISTILL_STATE_TOPK_INDICES_PLACEHOLDER,
    INDEXCACHE_STATE_KIND_DISTILL,
    INDEXCACHE_STATE_KIND_TOPK_ONLY,
    apply_stop_gradient_mask,
    format_indexcache_gradient_summary,
    state_from_slots,
    state_kind,
    state_to_slots,
    summarize_indexcache_gradients,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


def _config_for_pattern(pattern, **overrides):
    ratios = [item for _ in pattern for item in (4, 128)]
    kwargs = {
        "num_hidden_layers": len(ratios),
        "experimental_attention_variant": "dsv4_hybrid",
        "csa_compress_ratios": ratios,
        "index_topk_pattern": pattern,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _layer_config(pattern, **overrides):
    config = SimpleNamespace(
        index_topk_pattern=pattern,
        indexcache_multi_layer_distill=False,
        recompute_granularity=None,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        csa_compress_ratios=[4] * len(pattern),
        num_empty_layers_add_in_head=0,
        dsa_indexer_loss_coeff=0.03,
        csa_indexer_backend="tilelang",
        _indexcache_last_topk_idxs=None,
        _indexcache_last_layer_number=None,
        _indexcache_last_distill_state=None,
        _indexcache_last_served_count=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_layer(config, layer_number):
    layer = object.__new__(CompressedSparseAttention)
    object.__setattr__(layer, "config", config)
    object.__setattr__(layer, "compress_ratio", 4)
    object.__setattr__(layer, "indexer", object())
    object.__setattr__(layer, "layer_number", layer_number)
    object.__setattr__(layer, "cp_enabled", False)
    object.__setattr__(layer, "cp_rank", 0)
    object.__setattr__(layer, "cp_size", 1)
    object.__setattr__(layer, "training", True)
    return layer


class TestIndexCacheGradientNumerics(unittest.TestCase):
    def test_fused_summary_distinguishes_finite_nonzero_zero_nan_and_inf(self):
        for dtype in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                summaries = summarize_indexcache_gradients(
                    [
                        (
                            "normal",
                            paddle.to_tensor([0.0, 2.0, -3.0], dtype=dtype),
                        ),
                        ("zero", paddle.zeros([3], dtype=dtype)),
                        (
                            "nan",
                            paddle.to_tensor([0.0, float("nan")], dtype=dtype),
                        ),
                        (
                            "inf",
                            paddle.to_tensor([0.0, float("inf")], dtype=dtype),
                        ),
                        ("missing", None),
                    ]
                )

                self.assertTrue(summaries["normal"]["finite"])
                self.assertTrue(summaries["normal"]["nonzero"])
                self.assertEqual(summaries["normal"]["zero"], 1)
                self.assertTrue(summaries["zero"]["finite"])
                self.assertFalse(summaries["zero"]["nonzero"])
                self.assertEqual(summaries["zero"]["zero"], 3)
                self.assertFalse(summaries["nan"]["finite"])
                self.assertEqual(summaries["nan"]["nan"], 1)
                self.assertFalse(summaries["inf"]["finite"])
                self.assertEqual(summaries["inf"]["inf"], 1)
                self.assertFalse(summaries["missing"]["present"])
                self.assertIn(
                    "normal_nonzero=True",
                    format_indexcache_gradient_summary(
                        "normal", summaries["normal"]
                    ),
                )


class TestIndexCacheConfig(unittest.TestCase):
    def test_common_patterns_are_normalized_and_accepted(self):
        for pattern in ("F", "FSF", "FSSF"):
            config = _config_for_pattern(pattern.lower())
            self.assertEqual(config.index_topk_pattern, pattern)

    def test_distill_and_supported_recompute_are_accepted(self):
        distill = _config_for_pattern(
            "FSS",
            indexcache_multi_layer_distill=True,
            num_nextn_predict_layers=0,
        )
        self.assertTrue(distill.indexcache_multi_layer_distill)

        recompute = _config_for_pattern(
            "FSS",
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        self.assertEqual(recompute.recompute_method, "uniform")

    def test_invalid_pattern_contracts_fail_fast(self):
        cases = (
            ({"pattern": "FX"}, ValueError),
            ({"pattern": "SF"}, ValueError),
            (
                {
                    "pattern": "F",
                    "csa_compress_ratios": [4, 128, 4, 128],
                    "num_hidden_layers": 4,
                },
                ValueError,
            ),
            (
                {
                    "pattern": "FS",
                    "index_topk_pattern": None,
                    "indexcache_multi_layer_distill": True,
                },
                ValueError,
            ),
            (
                {
                    "pattern": "FS",
                    "recompute_granularity": "selective",
                    "recompute_modules": ["core_attn"],
                },
                NotImplementedError,
            ),
        )
        for values, error in cases:
            pattern = values.pop("pattern")
            with self.subTest(values=values), self.assertRaises(error):
                _config_for_pattern(pattern, **values)


class TestIndexCacheCoreState(unittest.TestCase):
    def test_saved_delta_offload_roundtrip_preserves_device_id(self):
        class FakePlace:
            @staticmethod
            def is_gpu_place():
                return True

            @staticmethod
            def gpu_device_id():
                return 3

        class FakeGpuValue:
            place = FakePlace()

            @staticmethod
            def cpu():
                return FakeCpuValue()

        class FakeCpuValue:
            @staticmethod
            def cuda(device_id):
                return ("restored", device_id)

        saved, device_id = _indexcache_offload_saved_delta(FakeGpuValue())

        self.assertIsInstance(saved, FakeCpuValue)
        self.assertEqual(device_id, 3)
        self.assertEqual(
            _indexcache_restore_saved_delta(saved, device_id),
            ("restored", 3),
        )

    def test_fused_producer_loss_does_not_fall_through_to_scalar_loss(self):
        producer = _make_layer(_layer_config("FS"), 0)
        output = paddle.ones([1], dtype="float32")
        indexer_loss = paddle.ones([], dtype="float32")
        tilelang_state = (paddle.ones([1], dtype="float32"),)

        with (
            patch.object(
                TileLangCSAIndexerLossAutoScaler, "apply"
            ) as tilelang_apply,
            patch.object(DSAIndexerLossAutoScaler, "apply") as scalar_apply,
        ):
            result = producer._indexcache_attach_indexer_loss(
                output,
                indexer_loss,
                tilelang_state,
                producer_loss_fused=True,
            )

        self.assertIs(result, output)
        tilelang_apply.assert_not_called()
        scalar_apply.assert_not_called()

    def test_action_mapping_for_f_s_fsf_and_fssf(self):
        for pattern in ("F", "FS", "FSF", "FSSF"):
            config = _layer_config(pattern)
            actions = []
            for ordinal, layer_number in enumerate(range(1, len(pattern) + 1)):
                layer = _make_layer(config, layer_number)
                actions.append(layer._indexcache_next_c4_action())
            self.assertEqual(
                actions,
                [(idx, action, pattern) for idx, action in enumerate(pattern)],
            )

    def test_full_model_c4_mapping_uses_normalized_one_based_layers(self):
        pattern = "FSSSFSSSFSSSFSSSFSSSF"
        ratios = [128] * 43
        for ratio_index in range(2, 43, 2):
            ratios[ratio_index] = 4

        for empty_head in (0, 7):
            with self.subTest(empty_head=empty_head):
                config = _layer_config(
                    pattern,
                    csa_compress_ratios=ratios,
                    num_empty_layers_add_in_head=empty_head,
                )
                first_c4 = _make_layer(config, 3)
                second_c4 = _make_layer(config, 5)
                self.assertEqual(
                    first_c4._indexcache_c4_layers(), list(range(3, 44, 2))
                )
                self.assertEqual(
                    first_c4._indexcache_next_c4_action(), (0, "F", pattern)
                )
                self.assertEqual(
                    second_c4._indexcache_next_c4_action(), (1, "S", pattern)
                )
                self.assertEqual(
                    second_c4._indexcache_infer_producer_layer(1, pattern), 3
                )

    def test_topk_only_state_is_reused_without_running_an_s_indexer(self):
        pattern = "FSS"
        config = _layer_config(pattern)
        producer = _make_layer(config, 0)
        topk = paddle.arange(6, dtype="int32").reshape([1, 2, 3])
        state = producer._indexcache_cache_topk(topk, 0, pattern)

        self.assertEqual(state_kind(state), INDEXCACHE_STATE_KIND_TOPK_ONLY)
        served = _make_layer(config, 1)
        reused = served._indexcache_reuse_topk(1, 2, 1, pattern, state)
        self.assertTrue(paddle.equal_all(reused, topk).item())

    def test_distill_state_and_recompute_slots_preserve_gradient_contract(self):
        pattern = "FSS"
        config = _layer_config(
            pattern,
            indexcache_multi_layer_distill=True,
            pipeline_model_parallel_size=2,
        )
        producer = _make_layer(config, 0)
        topk = paddle.arange(8, dtype="int32").reshape([1, 2, 4])
        raw_topk = paddle.to_tensor(
            [[[-1, 0, 1, 8191], [2, 3, 4, 5]]], dtype="int32"
        )
        loss_state = (
            paddle.ones([1], dtype="float32"),
            paddle.ones([1], dtype="float32") * 2,
            paddle.ones([1, 2, 1], dtype="float32") * 3,
            raw_topk,
            paddle.ones(topk.shape, dtype="float32"),
        )
        state = producer._indexcache_cache_topk(
            topk,
            0,
            pattern,
            tilelang_indexer_loss_state=loss_state,
            served_count=2,
            loss_scale=0.015,
        )

        self.assertEqual(state_kind(state), INDEXCACHE_STATE_KIND_DISTILL)
        self.assertEqual(INDEXCACHE_DISTILL_GRAD_INDICES, (5,))
        self.assertEqual(
            [state[idx].numel() for idx in (1, 2, 3, 4)], [0, 0, 0, 0]
        )
        self.assertEqual(state[0].dtype, paddle.int32)
        self.assertEqual(list(state[0].shape), [1, 2, 2])
        self.assertEqual(state[5].dtype, paddle.bfloat16)
        self.assertTrue(
            paddle.equal_all(
                _unpack_indexcache_pipeline_topk(state[0], 4), raw_topk
            ).item()
        )
        for idx, tensor in enumerate(state):
            self.assertEqual(
                tensor.stop_gradient,
                idx not in INDEXCACHE_DISTILL_GRAD_INDICES,
            )

        slots = state_to_slots(state)
        self.assertEqual(len(slots), 8)
        restored = state_from_slots(slots)
        self.assertEqual(state_kind(restored), INDEXCACHE_STATE_KIND_DISTILL)
        self.assertEqual(producer._indexcache_served_count(pattern, 0), 3)
        self.assertAlmostEqual(producer._indexcache_scaled_loss_coeff(2), 0.015)

    def test_distill_state_reconstructs_attention_indices_for_reuse(self):
        pattern = "FS"
        config = _layer_config(
            pattern,
            indexcache_multi_layer_distill=True,
            pipeline_model_parallel_size=2,
        )
        producer = _make_layer(config, 0)
        raw_topk = paddle.zeros([1, 4, 2], dtype="int32")
        mapped_topk = paddle.to_tensor(
            [[[-1, -1], [-1, -1], [-1, -1], [4, 4]]], dtype="int32"
        )
        loss_state = (
            paddle.ones([1], dtype="float32"),
            paddle.ones([1], dtype="float32"),
            paddle.ones([1, 1, 1], dtype="float32"),
            raw_topk,
            paddle.ones(raw_topk.shape, dtype="float32"),
        )
        state = producer._indexcache_cache_topk(
            mapped_topk,
            0,
            pattern,
            tilelang_indexer_loss_state=loss_state,
            served_count=1,
            loss_scale=0.03,
        )

        self.assertEqual(INDEXCACHE_DISTILL_STATE_TOPK_INDICES_PLACEHOLDER, 4)
        self.assertEqual(state[0].dtype, paddle.int32)
        self.assertEqual(list(state[0].shape), [1, 4, 1])
        self.assertEqual(state[5].dtype, paddle.bfloat16)
        self.assertTrue(
            paddle.equal_all(
                _unpack_indexcache_pipeline_topk(state[0], 2), raw_topk
            ).item()
        )
        self.assertEqual(state[4].numel(), 0)
        served = _make_layer(config, 1)
        reused = served._indexcache_reuse_topk(1, 4, 1, pattern, state)
        self.assertEqual(reused.dtype, paddle.int32)
        self.assertTrue(paddle.equal_all(reused, mapped_topk).item())

    def test_cp_distill_state_reconstructs_global_attention_indices(self):
        pattern = "FS"
        config = _layer_config(
            pattern,
            indexcache_multi_layer_distill=True,
            pipeline_model_parallel_size=2,
            context_parallel_size=8,
        )
        producer = _make_layer(config, 0)
        raw_topk = paddle.to_tensor(
            [[[0, 7], [0, 7], [0, 7], [0, 7]]], dtype="int32"
        )
        loss_state = (
            paddle.ones([1], dtype="float32"),
            paddle.ones([1], dtype="float32"),
            paddle.ones([1, 8, 1], dtype="float32"),
            raw_topk,
            paddle.ones(raw_topk.shape, dtype="float32"),
        )
        state = producer._indexcache_cache_topk(
            paddle.full(raw_topk.shape, -1, dtype="int32"),
            0,
            pattern,
            tilelang_indexer_loss_state=loss_state,
            served_count=1,
            loss_scale=0.03,
        )

        expected_by_rank = {
            0: paddle.to_tensor(
                [[[-1, -1], [-1, -1], [-1, -1], [32, -1]]],
                dtype="int32",
            ),
            7: paddle.to_tensor(
                [[[32, -1], [32, -1], [32, -1], [32, 39]]],
                dtype="int32",
            ),
        }
        for cp_rank, expected in expected_by_rank.items():
            with self.subTest(cp_rank=cp_rank):
                served = _make_layer(config, 1)
                served.cp_enabled = True
                served.cp_rank = cp_rank
                served.cp_size = 8
                reused = served._indexcache_reuse_topk(
                    1, 4, 1, pattern, state
                )
                self.assertTrue(paddle.equal_all(reused, expected).item())

    @patch("paddlefleet.tilelang_ops.csa_indexer_bwd")
    def test_pipeline_probability_cast_preserves_bridge_gradient(
        self, mock_bwd
    ):
        pattern = "FS"
        config = _layer_config(
            pattern,
            indexcache_multi_layer_distill=True,
            pipeline_model_parallel_size=2,
        )
        producer = _make_layer(config, 0)
        q = paddle.ones([1, 2], dtype="float32")
        weights = paddle.ones([1, 1], dtype="float32")
        k = paddle.ones([1, 1, 1], dtype="float32")
        for tensor in (q, weights, k):
            tensor.stop_gradient = False
        raw_topk = paddle.to_tensor([[[-1, 0]]], dtype="int32")
        topk_probs = paddle.to_tensor([[[0.25, 0.75]]], dtype="float32")
        producer_target = paddle.to_tensor([[[0.5, 0.5]]], dtype="float32")
        mock_bwd.return_value = (
            paddle.full_like(q, 2.0),
            paddle.full_like(weights, 3.0),
            paddle.full_like(k, 4.0),
        )
        state = producer._indexcache_pack_state(
            raw_topk,
            (
                q,
                weights,
                k,
                raw_topk,
                topk_probs,
                producer_target,
                0.2,
                "tilelang",
                1.0,
                None,
            ),
            served_count=1,
            fuse_producer_loss=True,
        )

        self.assertEqual(state[5].dtype, paddle.bfloat16)
        state[5].cast("float32").sum().backward()
        expected_score_grad = paddle.ones_like(topk_probs) + (
            (topk_probs - producer_target).cast("float16").cast("float32") * 0.2
        )
        self.assertTrue(
            paddle.allclose(
                mock_bwd.call_args.args[4], expected_score_grad
            ).item()
        )
        mock_bwd.assert_called_once()
        self.assertTrue(paddle.equal_all(q.grad, paddle.full_like(q, 2.0)))
        self.assertTrue(
            paddle.equal_all(weights.grad, paddle.full_like(weights, 3.0))
        )
        self.assertTrue(paddle.equal_all(k.grad, paddle.full_like(k, 4.0)))

    def test_explicit_state_does_not_retain_config_fallback_tensors(self):
        pattern = "FS"
        config = _layer_config(
            pattern,
            indexcache_multi_layer_distill=True,
            pipeline_model_parallel_size=2,
        )
        producer = _make_layer(config, 0)
        raw_topk = paddle.zeros([1, 1, 2], dtype="int32")
        loss_state = (
            paddle.ones([1], dtype="float32"),
            paddle.ones([1], dtype="float32"),
            paddle.ones([1, 1, 1], dtype="float32"),
            raw_topk,
            paddle.ones(raw_topk.shape, dtype="float32"),
        )
        state = producer._indexcache_cache_topk(
            raw_topk,
            0,
            pattern,
            tilelang_indexer_loss_state=loss_state,
            served_count=1,
            loss_scale=0.03,
        )

        self.assertEqual(state_kind(state), INDEXCACHE_STATE_KIND_DISTILL)
        self.assertIsNone(config._indexcache_last_topk_idxs)
        self.assertIsNone(config._indexcache_last_layer_number)
        self.assertIsNone(config._indexcache_last_distill_state)
        self.assertIsNone(config._indexcache_last_served_count)

    def test_distill_state_keeps_full_int32_above_pair_pack_domain(self):
        pattern = "FS"
        config = _layer_config(pattern, indexcache_multi_layer_distill=True)
        producer = _make_layer(config, 0)
        raw_topk = paddle.zeros([1, 1, 2], dtype="int32")
        loss_state = (
            paddle.ones([1], dtype="float32"),
            paddle.ones([1], dtype="float32"),
            paddle.ones([1, (1 << 15) + 1, 1], dtype="float32"),
            raw_topk,
            paddle.ones(raw_topk.shape, dtype="float32"),
        )
        state = producer._indexcache_cache_topk(
            raw_topk,
            0,
            pattern,
            tilelang_indexer_loss_state=loss_state,
            served_count=1,
            loss_scale=0.03,
        )

        self.assertEqual(state[0].dtype, paddle.int32)
        self.assertEqual(list(state[0].shape), [1, 1, 2])
        self.assertTrue(paddle.equal_all(state[0], raw_topk).item())

    def test_distill_state_keeps_full_int32_for_odd_topk_width(self):
        pattern = "FS"
        config = _layer_config(pattern, indexcache_multi_layer_distill=True)
        producer = _make_layer(config, 0)
        raw_topk = paddle.zeros([1, 1, 3], dtype="int32")
        loss_state = (
            paddle.ones([1], dtype="float32"),
            paddle.ones([1], dtype="float32"),
            paddle.ones([1, 8, 1], dtype="float32"),
            raw_topk,
            paddle.ones(raw_topk.shape, dtype="float32"),
        )
        state = producer._indexcache_cache_topk(
            raw_topk,
            0,
            pattern,
            tilelang_indexer_loss_state=loss_state,
            served_count=1,
            loss_scale=0.03,
        )

        self.assertEqual(state[0].dtype, paddle.int32)
        self.assertEqual(list(state[0].shape), [1, 1, 3])
        self.assertTrue(paddle.equal_all(state[0], raw_topk).item())

    def test_missing_explicit_state_fails_for_pipeline_or_recompute(self):
        pattern = "FS"
        config = _layer_config(pattern, pipeline_model_parallel_size=2)
        served = _make_layer(config, 1)
        with self.assertRaisesRegex(
            RuntimeError, "explicit producer top-k state"
        ):
            served._indexcache_reuse_topk(1, 2, 1, pattern, None)

    def test_invalid_state_length_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_stop_gradient_mask((paddle.ones([1]), paddle.ones([1])))

    @patch("paddlefleet.tilelang_ops.csa_indexer_bwd")
    def test_compact_distill_bridge_preserves_selected_score_gradient(
        self, mock_bwd
    ):
        q = paddle.ones([1, 2], dtype="float32")
        weights = paddle.ones([1, 1], dtype="float32")
        k = paddle.ones([1, 2], dtype="float32")
        for tensor in (q, weights, k):
            tensor.stop_gradient = False
        topk_indices = paddle.to_tensor([[[0, 1]]], dtype="int32")
        topk_probs = paddle.to_tensor([[[0.25, 0.75]]], dtype="float32")
        target = paddle.to_tensor([[[0.5, 0.5]]], dtype="float32")
        mock_bwd.return_value = (
            paddle.full_like(q, 2.0),
            paddle.full_like(weights, 3.0),
            paddle.full_like(k, 4.0),
        )

        previous_scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        DSAIndexerLossAutoScaler._main_loss_backward_scale = None
        marker_output = StringIO()
        with (
            patch.dict("os.environ", {"INDEXCACHE_TRAIN_DEBUG": "1"}),
            redirect_stdout(marker_output),
        ):
            try:
                bridged_probs = TileLangCSAIndexerDistillBridge.apply(
                    q,
                    weights,
                    k,
                    topk_indices,
                    topk_probs,
                    None,
                    0.0,
                    None,
                    2,
                )
                output_leaf = paddle.ones([1], dtype="float32")
                output_leaf.stop_gradient = False
                output = output_leaf * 1.0
                output = IndexCacheServedDistillLossAutoScaler.apply(
                    output,
                    bridged_probs,
                    target,
                    0.4,
                    None,
                    None,
                    3,
                    2,
                )
                output.sum().backward()
            finally:
                DSAIndexerLossAutoScaler._main_loss_backward_scale = (
                    previous_scale
                )

        expected_score_grad = (topk_probs - target) * 0.4
        actual_score_grad = mock_bwd.call_args.args[4]
        self.assertTrue(
            paddle.allclose(actual_score_grad, expected_score_grad).item()
        )
        self.assertTrue(
            paddle.allclose(q.grad, paddle.full_like(q, 2.0)).item()
        )
        self.assertTrue(
            paddle.allclose(weights.grad, paddle.full_like(weights, 3.0)).item()
        )
        self.assertTrue(
            paddle.allclose(k.grad, paddle.full_like(k, 4.0)).item()
        )
        markers = marker_output.getvalue()
        self.assertIn(
            "boundary=served_score_backward served_layer=3 producer_layer=2",
            markers,
        )
        self.assertIn("score_grad_finite=True", markers)
        self.assertIn("score_grad_nonzero=True", markers)
        self.assertIn(
            "boundary=producer_bridge_grad_ready producer_layer=2", markers
        )
        for prefix in ("q_grad", "weights_grad", "k_grad"):
            self.assertIn(f"{prefix}_finite=True", markers)
            self.assertIn(f"{prefix}_nonzero=True", markers)

    @patch("paddlefleet.tilelang_ops.csa_indexer_bwd")
    def test_compact_distill_bridge_combines_producer_and_served_gradients(
        self, mock_bwd
    ):
        q = paddle.ones([1, 2], dtype="float32")
        weights = paddle.ones([1, 1], dtype="float32")
        k = paddle.ones([1, 2], dtype="float32")
        for tensor in (q, weights, k):
            tensor.stop_gradient = False
        topk_indices = paddle.to_tensor([[[0, 1]]], dtype="int32")
        topk_probs = paddle.to_tensor([[[0.25, 0.75]]], dtype="float32")
        producer_target = paddle.to_tensor([[[0.75, 0.25]]], dtype="float32")
        served_target = paddle.to_tensor([[[0.5, 0.5]]], dtype="float32")
        producer_delta = (topk_probs - producer_target).cast("float16")
        mock_bwd.return_value = (
            paddle.full_like(q, 2.0),
            paddle.full_like(weights, 3.0),
            paddle.full_like(k, 4.0),
        )

        previous_scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        DSAIndexerLossAutoScaler._main_loss_backward_scale = paddle.to_tensor(
            2.0, dtype="float32"
        )
        try:
            bridged_probs = TileLangCSAIndexerDistillBridge.apply(
                q,
                weights,
                k,
                topk_indices,
                topk_probs,
                producer_delta,
                0.2,
                1.0,
            )
            output_leaf = paddle.ones([1], dtype="float32")
            output_leaf.stop_gradient = False
            output = IndexCacheServedDistillLossAutoScaler.apply(
                output_leaf * 1.0,
                bridged_probs,
                served_target,
                0.4,
                1.0,
                None,
            )
            output.sum().backward()
        finally:
            DSAIndexerLossAutoScaler._main_loss_backward_scale = previous_scale

        expected_served = (
            (topk_probs - served_target).cast("float16").cast("float32")
            * 0.4
            * 2.0
        )
        expected_producer = producer_delta.cast("float32") * 0.2 * 2.0
        actual_score_grad = mock_bwd.call_args.args[4]
        mock_bwd.assert_called_once()
        self.assertTrue(
            paddle.allclose(
                actual_score_grad, expected_served + expected_producer
            ).item()
        )
        self.assertTrue(
            paddle.allclose(q.grad, paddle.full_like(q, 2.0)).item()
        )
        self.assertTrue(
            paddle.allclose(weights.grad, paddle.full_like(weights, 3.0)).item()
        )
        self.assertTrue(
            paddle.allclose(k.grad, paddle.full_like(k, 4.0)).item()
        )

    def test_served_distill_gradient_preserves_mask_and_main_loss_scale(self):
        topk_probs = paddle.to_tensor(
            [[[0.25, 0.75], [0.6, 0.4]]], dtype="float32"
        )
        topk_probs.stop_gradient = False
        target = paddle.to_tensor([[[0.5, 0.5], [0.2, 0.8]]], dtype="float32")
        loss_mask = paddle.to_tensor([[1.0, 0.0]], dtype="float32")
        output_leaf = paddle.ones([1], dtype="float32")
        output_leaf.stop_gradient = False
        output = output_leaf * 1.0

        previous_scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        DSAIndexerLossAutoScaler._main_loss_backward_scale = paddle.to_tensor(
            2.0, dtype="float32"
        )
        try:
            output = IndexCacheServedDistillLossAutoScaler.apply(
                output,
                topk_probs,
                target,
                0.4,
                1.0,
                loss_mask,
            )
            output.sum().backward()
        finally:
            DSAIndexerLossAutoScaler._main_loss_backward_scale = previous_scale

        expected = (
            (topk_probs - target).cast("float16").cast("float32") * 0.4 * 2.0
        )
        expected[:, 1, :] = 0.0
        self.assertTrue(paddle.allclose(topk_probs.grad, expected).item())


if __name__ == "__main__":
    unittest.main()
