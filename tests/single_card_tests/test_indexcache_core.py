# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import paddle

from paddlefleet.transformer.csa_attention import (
    CompressOrSkip,
    CompressedSparseAttention,
    IndexCacheServedDistillLossAutoScaler,
    TilelangIndexerLossState,
    TileLangCSAIndexerDistillBridge,
    TileLangCSAIndexerLossAutoScaler,
    _indexcache_offload_saved_delta,
    _indexcache_restore_saved_delta,
    _unpack_indexcache_pipeline_topk,
)
from paddlefleet.transformer.dsa_attention import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
)
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
        indexcache_train_debug=False,
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
    def test_debug_config_defaults_and_normalizes_trace_layers(self):
        default = _config_for_pattern("F")
        self.assertFalse(default.indexcache_train_debug)
        self.assertFalse(default.indexcache_stall_trace)
        self.assertEqual(default.indexcache_stall_trace_layers, (2,))

        configured = _config_for_pattern(
            "F",
            indexcache_train_debug=True,
            indexcache_stall_trace=True,
            indexcache_stall_trace_layers=[2, 1, 2],
        )
        self.assertTrue(configured.indexcache_train_debug)
        self.assertTrue(configured.indexcache_stall_trace)
        self.assertEqual(configured.indexcache_stall_trace_layers, (1, 2))

    def test_debug_config_defaults_survive_from_config(self):
        config = TransformerConfig.from_config(
            SimpleNamespace(
                num_hidden_layers=2,
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[4, 128],
                index_topk_pattern="F",
            )
        )
        self.assertFalse(config.indexcache_train_debug)
        self.assertFalse(config.indexcache_stall_trace)
        self.assertEqual(config.indexcache_stall_trace_layers, (2,))

    def test_debug_config_rejects_invalid_values(self):
        cases = (
            ({"indexcache_multi_layer_distill": 1}, TypeError),
            ({"indexcache_train_debug": 1}, TypeError),
            ({"indexcache_stall_trace": "true"}, TypeError),
            ({"indexcache_stall_trace_layers": "2"}, TypeError),
            ({"indexcache_stall_trace_layers": [0]}, ValueError),
            ({"indexcache_stall_trace_layers": [True]}, ValueError),
            (
                {
                    "indexcache_stall_trace": True,
                    "indexcache_stall_trace_layers": [],
                },
                ValueError,
            ),
        )
        for overrides, error in cases:
            with self.subTest(overrides=overrides), self.assertRaises(error):
                _config_for_pattern("F", **overrides)

    def test_indexcache_fields_reject_non_dsv4_variants(self):
        for overrides in (
            {"index_topk_pattern": "F"},
            {"indexcache_multi_layer_distill": True},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(
                    ValueError,
                    "experimental_attention_variant='dsv4_hybrid'",
                ),
            ):
                TransformerConfig(num_hidden_layers=1, **overrides)

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

    def test_baseline_cp_distill_recompute_contract_is_accepted(self):
        config = _config_for_pattern(
            "FS",
            csa_compress_ratios=[4, 128, 4, 128, 0],
            indexcache_multi_layer_distill=True,
            context_parallel_size=8,
            num_nextn_predict_layers=1,
            mtp_load_weight_only=True,
            csa_indexer_backend="tilelang",
            csa_sparse_attn_backend="cudnn",
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )

        self.assertTrue(config.mtp_load_weight_only)
        self.assertEqual(config.csa_sparse_attn_backend, "cudnn")
        self.assertEqual(config.recompute_granularity, "full")

    def test_cp_distill_rejects_active_mtp(self):
        with self.assertRaisesRegex(NotImplementedError, "MTP forward"):
            _config_for_pattern(
                "FS",
                csa_compress_ratios=[4, 128, 4, 128, 0],
                indexcache_multi_layer_distill=True,
                context_parallel_size=8,
                num_nextn_predict_layers=1,
                mtp_load_weight_only=False,
            )

    def test_distill_rejects_non_tilelang_indexer(self):
        with self.assertRaisesRegex(
            NotImplementedError, "csa_indexer_backend='tilelang'"
        ):
            _config_for_pattern(
                "FS",
                indexcache_multi_layer_distill=True,
                csa_indexer_backend="unfused",
                csa_sparse_attn_backend="cudnn",
            )

    def test_recompute_rejects_non_tilelang_indexer(self):
        with self.assertRaisesRegex(
            NotImplementedError, "csa_indexer_backend='tilelang'"
        ):
            _config_for_pattern(
                "FS",
                csa_indexer_backend="unfused",
                csa_sparse_attn_backend="cudnn",
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )

    def test_recompute_rejects_block_attention_residuals(self):
        with self.assertRaisesRegex(
            NotImplementedError, "block_attention_residuals"
        ):
            _config_for_pattern(
                "FS",
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
                block_attention_residuals=True,
            )

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
    def _assert_replay_only_changes_attention(self, action, cp_enabled):
        pattern = "FS"
        config = _layer_config(
            pattern,
            indexcache_multi_layer_distill=True,
            context_parallel_size=2 if cp_enabled else 1,
        )
        layer = _make_layer(config, 1 if action == "F" else 2)
        object.__setattr__(layer, "is_mqa_layer", False)
        object.__setattr__(layer, "is_hca_layer", False)
        object.__setattr__(layer, "compressor", object())
        object.__setattr__(layer, "window_size", 2)
        object.__setattr__(layer, "indexer_backend", "tilelang")
        object.__setattr__(layer, "sparse_attn_backend", "unfused")
        object.__setattr__(
            layer, "attn_sink", paddle.zeros([1], dtype="float32")
        )
        object.__setattr__(layer, "softmax_scale", 1.0)
        object.__setattr__(layer, "tp_group", None)

        if cp_enabled:
            object.__setattr__(layer, "cp_enabled", True)
            object.__setattr__(layer, "cp_size", 2)
            object.__setattr__(layer, "cp_rank", 0)
            object.__setattr__(layer, "cp_group", None)

        batch, seq = 1, 8
        query = paddle.zeros([batch, seq, 1, 2], dtype="float32")
        key = paddle.zeros([batch, seq, 1, 2], dtype="float32")
        x = paddle.zeros([batch, seq, 4], dtype="float32")
        qr = paddle.zeros([batch, seq, 2], dtype="float32")
        window = paddle.full([batch, seq, 1], -1, dtype="int32")
        original = paddle.full([batch, seq, 2], 8, dtype="int32")
        replay = paddle.full([batch, seq, 2], 9, dtype="int32")
        topk_probs = paddle.full(
            [batch, seq, 2], 0.5, dtype="float32"
        )
        loss_state = TilelangIndexerLossState(
            paddle.ones([1], dtype="float32"),
            paddle.ones([1], dtype="float32"),
            paddle.ones([1, 2, 1], dtype="float32"),
            paddle.zeros([batch, seq, 2], dtype="int32"),
            topk_probs,
            None,
            0.1,
            "tilelang",
            None,
            None,
        )
        packed_state = (paddle.ones([1], dtype="float32"),)
        attention_output = paddle.ones(
            [batch, seq, 1, 2], dtype="float32"
        )

        class _Compressor:
            def __call__(self, *_args, **_kwargs):
                return paddle.zeros([batch, 4, 2], dtype="float32")

        if cp_enabled:
            object.__setattr__(layer, "compressor", _Compressor())

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_next_c4_action",
                    return_value=(0 if action == "F" else 1, action, pattern),
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_has_future_served_layer",
                    return_value=action == "F",
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_multi_layer_distill_enabled",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_served_count",
                    return_value=1,
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_scaled_loss_coeff",
                    return_value=0.1,
                )
            )
            cache_topk = stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_cache_topk",
                    return_value=packed_state,
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_reuse_topk",
                    return_value=original,
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_served_distill_state",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_clear_cached_state",
                )
            )
            replay_hook = stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_postprocess_indexer_replay",
                    return_value=replay,
                )
            )
            sparse_attn = stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "compressed_sparse_attn",
                    return_value=attention_output,
                )
            )
            fused_target = stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_compute_fused_indexer_target",
                    return_value=paddle.full_like(topk_probs, 0.5),
                )
            )
            stack.enter_context(
                patch.object(
                    CompressedSparseAttention,
                    "_indexcache_attach_indexer_loss",
                    side_effect=lambda output, *_args, **_kwargs: output,
                )
            )

            if cp_enabled:
                stack.enter_context(
                    patch(
                        "paddlefleet.transformer.csa_attention."
                        "get_window_topk_idxs_cp",
                        return_value=window,
                    )
                )
                stack.enter_context(
                    patch(
                        "paddlefleet.transformer.csa_attention.all_gather_cp",
                        return_value=paddle.zeros(
                            [batch, seq * 2, 2], dtype="float32"
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        CompressedSparseAttention,
                        "_compute_indexer_compressed_topk_idxs_cp",
                        return_value=(
                            original,
                            None,
                            loss_state if action == "F" else None,
                            0.1,
                            action == "F",
                            1,
                        ),
                    )
                )
                result = CompressedSparseAttention._forward_cp(
                    layer,
                    query,
                    key,
                    x,
                    qr,
                    indexcache_state=(
                        packed_state if action == "S" else None
                    ),
                )
            else:
                stack.enter_context(
                    patch(
                        "paddlefleet.transformer.csa_attention."
                        "get_window_topk_idxs",
                        return_value=window,
                    )
                )
                stack.enter_context(
                    patch.object(
                        CompressOrSkip,
                        "apply",
                        return_value=paddle.zeros(
                            [batch, seq + 2, 2], dtype="float32"
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        CompressedSparseAttention,
                        "_compute_indexer_compressed_topk_idxs",
                        return_value=(
                            original,
                            None,
                            loss_state if action == "F" else None,
                        ),
                    )
                )
                result = CompressedSparseAttention.forward(
                    layer,
                    query,
                    key,
                    key,
                    x=x,
                    qr=qr,
                    indexcache_state=(
                        packed_state if action == "S" else None
                    ),
                )

        replay_hook.assert_called_once()
        self.assertIs(replay_hook.call_args.args[0], original)
        compressed_attention_topk = sparse_attn.call_args.args[3][..., -2:]
        self.assertTrue(
            paddle.equal_all(compressed_attention_topk, replay).item()
        )
        if action == "F":
            self.assertIs(cache_topk.call_args.args[0], original)
            self.assertIs(fused_target.call_args.args[2], original)
            self.assertIs(result[1], packed_state)
        else:
            cache_topk.assert_not_called()
            fused_target.assert_not_called()

    def test_replay_preserves_native_f_s_state_and_loss_indices(self):
        for cp_enabled in (False, True):
            for action in ("F", "S"):
                with self.subTest(cp_enabled=cp_enabled, action=action):
                    self._assert_replay_only_changes_attention(
                        action, cp_enabled
                    )

    def test_train_debug_is_driven_by_config(self):
        config = _layer_config("F", indexcache_train_debug=True)
        layer = _make_layer(config, 1)
        output = StringIO()
        with redirect_stdout(output):
            layer._indexcache_debug("action=test")
        self.assertIn("action=test", output.getvalue())

        config.indexcache_train_debug = False
        output = StringIO()
        with redirect_stdout(output):
            layer._indexcache_debug("action=disabled")
        self.assertEqual(output.getvalue(), "")

    def test_cp_indexer_helper_preserves_legacy_and_extended_contracts(self):
        config = _layer_config(
            "F",
            csa_indexer_backend="unfused",
            dsa_indexer_use_sparse_loss=True,
            num_hidden_layers=1,
        )
        layer = _make_layer(config, 1)
        layer.cp_size = 2
        object.__setattr__(layer, "cp_group", None)
        object.__setattr__(layer, "tp_group", None)
        object.__setattr__(layer, "softmax_scale", 1.0)

        q_indexer = paddle.zeros([1, 4, 1, 2], dtype="float32")
        k_indexer = paddle.zeros([1, 2, 2], dtype="float32")
        weights = paddle.zeros([1, 4, 1], dtype="float32")
        indexer = SimpleNamespace(
            index_topk=2,
            softmax_scale=1.0,
            forward_before_topk=lambda *_args, **_kwargs: (
                q_indexer,
                k_indexer,
                weights,
            ),
        )
        object.__setattr__(layer, "indexer", indexer)

        query = paddle.zeros([1, 4, 2, 3], dtype="float32")
        x = paddle.zeros([1, 4, 8], dtype="float32")
        qr = paddle.zeros([1, 4, 6], dtype="float32")
        compressed_kv = paddle.zeros([1, 2, 3], dtype="float32")
        q_positions = paddle.arange(4, dtype="int64")
        topk_indices = paddle.zeros([1, 4, 2], dtype="int32")
        mapped_topk = paddle.full([1, 4, 2], 4, dtype="int32")
        indexer_loss = paddle.to_tensor(2.0, dtype="float32")

        with (
            patch.object(
                FusedDSAIndexerLoss,
                "apply",
                return_value=indexer_loss,
            ),
            patch.object(
                FusedDSAIndexerLoss,
                "_last_topk_indices",
                topk_indices,
                create=True,
            ),
            patch.object(
                DSAIndexerLossLoggingHelper,
                "save_loss_to_tracker",
            ) as save_loss,
            patch(
                "paddlefleet.transformer.csa_attention."
                "map_compressed_topk_to_kv_full_cp",
                return_value=mapped_topk,
            ),
        ):
            legacy = layer._compute_indexer_compressed_topk_idxs_cp(
                query,
                x,
                qr,
                compressed_kv,
                2,
                4,
                q_positions,
                0,
            )
            extended = layer._compute_indexer_compressed_topk_idxs_cp(
                query,
                x,
                qr,
                compressed_kv,
                2,
                4,
                q_positions,
                0,
                indexcache_action=(0, "F", "F"),
            )

        self.assertEqual(len(legacy), 3)
        self.assertIs(legacy[0], mapped_topk)
        self.assertAlmostEqual(float(legacy[1]), 1.0)
        self.assertEqual(len(extended), 6)
        self.assertEqual(extended[3:], (None, False, None))
        self.assertEqual(save_loss.call_count, 2)

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
            [state[idx].numel() for idx in (1, 2, 3, 4)], [1, 1, 1, 1]
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
        self.assertEqual(state[4].numel(), 1)
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
        with redirect_stdout(marker_output):
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
                    True,
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
                    True,
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
