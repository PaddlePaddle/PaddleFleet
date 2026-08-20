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

"""Coverage and unfused-regression tests for the CuTe DSL indexer top-k."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle

import paddlefleet.cutedsl_ops.compiler as compiler_mod
import paddlefleet.cutedsl_ops.dlpack as dlpack_mod
import paddlefleet.cutedsl_ops.indexer_topk_prefill as topk_mod
from paddlefleet import cutedsl_ops
from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
    _indexer_top_k_unfused,
)
from paddlefleet.cutedsl_ops.dlpack import _PaddleDLPackAdapter


def _has_cutedsl_sm100():
    if not paddle.device.is_compiled_with_cuda():
        return False
    if paddle.device.cuda.device_count() == 0:
        return False
    try:
        if paddle.device.cuda.get_device_capability()[0] != 10:
            return False
        topk_mod._require_cutedsl()
    except (ImportError, RuntimeError, AttributeError):
        return False
    return True


_HAS_CUTEDSL_SM100 = _has_cutedsl_sm100()
_CUTEDSL_SKIP_REASON = "CuTe DSL CLC top-k requires CUTLASS DSL and SM10x"


def setUpModule():
    if _HAS_CUTEDSL_SM100:
        paddle.set_device("gpu")


class TestIndexerTopKCoverage(unittest.TestCase):
    """Exercise host-side helpers without requiring a CUDA device."""

    def test_cutedsl_backend_dispatch(self):
        input_values = MagicMock()
        input_values.shape = (1, 128)
        contiguous_values = input_values.contiguous.return_value
        seq_lens = MagicMock()
        int32_lengths = seq_lens.cast.return_value
        contiguous_lengths = int32_lengths.contiguous.return_value
        indices = object()
        values = object()

        with patch.object(
            topk_mod,
            "indexer_topk_prefill",
            return_value=(indices, values),
        ) as topk:
            result = _indexer_top_k_unfused(
                input_values,
                seq_lens,
                64,
                return_val=False,
                topk_backend="cutedsl",
                cutedsl_min_cols=64,
            )

        self.assertEqual(result, {"indices": indices, "values": values})
        seq_lens.cast.assert_called_once_with("int32")
        topk.assert_called_once_with(
            contiguous_values,
            contiguous_lengths,
            64,
            return_val=False,
            sort_mode="score",
        )

    def test_cutedsl_index_backend_selects_index_kernel(self):
        input_values = MagicMock()
        input_values.shape = (1, 4096)
        seq_lens = MagicMock()
        with patch.object(
            topk_mod,
            "indexer_topk_prefill",
            return_value=(object(), None),
        ) as topk:
            _indexer_top_k_unfused(
                input_values,
                seq_lens,
                2048,
                return_val=False,
                topk_backend="cutedsl_index",
            )
        self.assertEqual(topk.call_args.kwargs["sort_mode"], "index")

    def test_cutedsl_backend_falls_back_for_short_scores(self):
        input_values = paddle.to_tensor([[1.0, 3.0, 2.0, 4.0]])
        seq_lens = paddle.to_tensor([4], dtype="int32")
        with patch.object(topk_mod, "indexer_topk_prefill") as topk:
            result = _indexer_top_k_unfused(
                input_values,
                seq_lens,
                2,
                return_val=True,
                topk_backend="cutedsl",
                cutedsl_min_cols=5,
            )

        topk.assert_not_called()
        self.assertEqual(result["indices"].tolist(), [[3, 1]])
        self.assertEqual(result["values"].tolist(), [[4.0, 3.0]])

    def test_cutedsl_backend_uses_inclusive_column_threshold(self):
        input_values = MagicMock()
        input_values.shape = (1, 5)
        seq_lens = MagicMock()
        indices = object()
        values = object()
        with patch.object(
            topk_mod,
            "indexer_topk_prefill",
            return_value=(indices, values),
        ) as topk:
            result = _indexer_top_k_unfused(
                input_values,
                seq_lens,
                64,
                topk_backend="cutedsl",
                cutedsl_min_cols=5,
            )

        self.assertEqual(result, {"indices": indices, "values": values})
        topk.assert_called_once()

    def test_invalid_topk_backend_raises(self):
        with self.assertRaisesRegex(
            ValueError, "topk_backend='invalid_backend' is invalid"
        ):
            _indexer_top_k_unfused(
                MagicMock(),
                MagicMock(),
                64,
                topk_backend="invalid_backend",
            )

    def test_cutedsl_index_backend_rejects_nonpositive_topk(self):
        with self.assertRaisesRegex(
            ValueError,
            r"topk_backend='cutedsl_index' requires top_k >= 1",
        ):
            _indexer_top_k_unfused(
                MagicMock(),
                MagicMock(),
                0,
                topk_backend="cutedsl_index",
            )

    def test_lazy_public_wrappers(self):
        package = importlib.reload(cutedsl_ops)
        index_result = object()
        precompile_result = object()
        with (
            patch.object(
                topk_mod,
                "indexer_topk_prefill",
                return_value=index_result,
            ) as index_impl,
            patch.object(
                topk_mod,
                "precompile_indexer_topk_clc",
                return_value=precompile_result,
            ) as precompile_impl,
        ):
            self.assertIs(
                package.indexer_topk_prefill("scores", return_val=False),
                index_result,
            )
            self.assertIs(
                package.precompile_indexer_topk_clc(16384, 2048),
                precompile_result,
            )
        index_impl.assert_called_once_with("scores", return_val=False)
        precompile_impl.assert_called_once_with(16384, 2048)

    def test_compile_bucket_saturates(self):
        cases = {
            1: 1,
            3: 4,
            2048: 2048,
            8193: 16384,
            16384: 16384,
            256 * 1024: 16384,
        }
        for num_cols, expected in cases.items():
            with self.subTest(num_cols=num_cols):
                self.assertEqual(
                    topk_mod._compile_num_cols_bucket(num_cols), expected
                )

    def test_launch_and_workspace_helpers(self):
        self.assertEqual(
            topk_mod._persistent_launch_config(
                7, persistent=False, schedule_mode="static"
            ),
            (7, 1),
        )
        with self.assertRaisesRegex(ValueError, "non-empty row"):
            topk_mod._persistent_launch_config(0)
        with self.assertRaisesRegex(ValueError, "unsupported schedule_mode"):
            topk_mod._persistent_launch_config(1, schedule_mode="invalid")
        with self.assertRaisesRegex(ValueError, "requires persistent=True"):
            topk_mod._persistent_launch_config(
                1, persistent=False, schedule_mode="clc"
            )

        self.assertEqual(
            topk_mod._workspace_slot_count(9, 4, 256, False, "clc"), 9
        )
        self.assertEqual(
            topk_mod._workspace_slot_count(9, 4, 256, True, "static"), 4
        )
        self.assertEqual(
            topk_mod._workspace_slot_count(9, 4, 512, True, "clc"), 16
        )

    def test_clc_capability_guard(self):
        capability = "paddle.device.cuda.get_device_capability"
        with (
            patch(capability, return_value=(9, 0)),
            self.assertRaisesRegex(RuntimeError, "SM100-family"),
        ):
            # The mocked capability fails before CUTLASS compilation;
            # calling the public precompile entry verifies its guard.
            topk_mod.precompile_indexer_topk_clc(max_num_cols=16, top_k=1)

    def test_compiler_arch_flags_and_options(self):
        capability = "paddle.device.cuda.get_device_capability"
        for compute_capability, expected in compiler_mod._ARCH_MAP.items():
            with self.subTest(compute_capability=compute_capability):
                compiler_mod.gpu_arch_flag.cache_clear()
                with (
                    patch("paddle.is_compiled_with_cuda", return_value=True),
                    patch(capability, return_value=compute_capability),
                ):
                    self.assertEqual(compiler_mod.gpu_arch_flag(), expected)

        compiler_mod.gpu_arch_flag.cache_clear()
        with (
            patch("paddle.is_compiled_with_cuda", return_value=False),
            self.assertRaisesRegex(
                RuntimeError, "requires a CUDA Paddle build"
            ),
        ):
            compiler_mod.gpu_arch_flag()

        compiler_mod.gpu_arch_flag.cache_clear()
        with (
            patch("paddle.is_compiled_with_cuda", return_value=True),
            patch(capability, return_value=(8, 0)),
            self.assertRaisesRegex(RuntimeError, "unsupported CUDA"),
        ):
            compiler_mod.gpu_arch_flag()

        with patch.object(
            compiler_mod, "gpu_arch_flag", return_value="sm_100a"
        ):
            self.assertEqual(
                compiler_mod.compile_options(),
                "--enable-tvm-ffi --gpu-arch sm_100a",
            )
            self.assertEqual(
                compiler_mod.compile_options("--debug"),
                "--enable-tvm-ffi --gpu-arch sm_100a --debug",
            )
        compiler_mod.gpu_arch_flag.cache_clear()

    def test_precompile_enumeration_and_process_cache(self):
        fake_cutlass = SimpleNamespace(
            Float16=object(),
            BFloat16=object(),
            Float32=object(),
        )
        topk_mod._CLC_PRECOMPILE_CACHE.clear()
        with (
            patch(
                "paddle.device.cuda.get_device_capability",
                return_value=(10, 3),
            ),
            patch.object(
                topk_mod,
                "_require_cutedsl",
                return_value=(fake_cutlass, None, None, None),
            ),
            patch.object(
                topk_mod,
                "_persistent_launch_config",
                return_value=(8, 1),
            ),
            patch.object(topk_mod, "_compile_kernel") as compile_kernel,
        ):
            buckets = topk_mod.precompile_indexer_topk_clc(
                max_num_cols=256 * 1024,
                top_k=2048,
                dtype=paddle.float16,
            )
            self.assertEqual(buckets, (2048, 4096, 8192, 16384))
            self.assertEqual(compile_kernel.call_count, 8)
            for call in compile_kernel.call_args_list:
                self.assertEqual(call.args[7], "clc")

            repeated = topk_mod.precompile_indexer_topk_clc(
                max_num_cols=256 * 1024,
                top_k=2048,
                dtype=paddle.float16,
            )
            self.assertEqual(repeated, buckets)
            self.assertEqual(compile_kernel.call_count, 8)
        topk_mod._CLC_PRECOMPILE_CACHE.clear()

    def test_precompile_rejects_invalid_arguments(self):
        fake_cutlass = SimpleNamespace(
            Float16=object(),
            BFloat16=object(),
            Float32=object(),
        )
        cases = (
            (
                {"max_num_cols": 0, "top_k": 1},
                "max_num_cols must be positive",
            ),
            (
                {"max_num_cols": 16, "top_k": 0},
                r"top_k must be in \[1, 32768\]",
            ),
            (
                {"max_num_cols": 16, "top_k": 32},
                "max_num_cols must be >= top_k",
            ),
            (
                {"max_num_cols": 32768, "top_k": 16385},
                r"top_k must be in \[1, 16384\] for float32",
            ),
            (
                {
                    "max_num_cols": 16,
                    "top_k": 1,
                    "return_values": (),
                },
                "return_values must contain",
            ),
        )
        with (
            patch(
                "paddle.device.cuda.get_device_capability",
                return_value=(10, 3),
            ),
            patch.object(
                topk_mod,
                "_require_cutedsl",
                return_value=(fake_cutlass, None, None, None),
            ),
        ):
            for kwargs, message in cases:
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    topk_mod.precompile_indexer_topk_clc(**kwargs)

    def test_index_sort_mode_has_explicit_topk_boundary(self):
        self.assertEqual(
            topk_mod._validate_sort_mode("score", 64),
            "score",
        )
        self.assertEqual(
            topk_mod._validate_sort_mode("index", 1),
            "index",
        )
        self.assertEqual(
            topk_mod._validate_sort_mode("index", 64),
            "index",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"sort_mode='index' requires top_k >= 1",
        ):
            topk_mod._validate_sort_mode("index", 0)

    def test_precompile_index_sort_mode_accepts_small_topk(self):
        with (
            patch(
                "paddle.device.cuda.get_device_capability",
                return_value=(10, 3),
            ),
            patch.object(topk_mod, "_compile_kernel"),
        ):
            buckets = topk_mod.precompile_indexer_topk_clc(
                max_num_cols=4096,
                top_k=64,
                sort_mode="index",
            )
            self.assertEqual(buckets, (64, 128, 256, 512, 1024, 2048, 4096))
        topk_mod._CLC_PRECOMPILE_CACHE.clear()

    def test_dlpack_adapter_forwards_consumer_stream(self):
        sentinel = object()

        class Tensor:
            def __init__(self):
                self.stream = None

            def __dlpack__(self, *, stream=None):
                self.stream = stream
                return sentinel

            def __dlpack_device__(self):
                return (2, 0)

        tensor = Tensor()
        adapter = _PaddleDLPackAdapter(tensor)
        self.assertIs(adapter.__dlpack__(), sentinel)
        self.assertIsNone(tensor.stream)
        self.assertIs(adapter.__dlpack__(stream=12345), sentinel)
        self.assertEqual(tensor.stream, 12345)
        self.assertEqual(adapter.__dlpack_device__(), (2, 0))

    def test_dlpack_adapter_falls_back_to_paddle_capsule(self):
        tensor = object()
        capsule = object()
        with patch(
            "paddle.utils.dlpack.to_dlpack", return_value=capsule
        ) as to_dlpack:
            self.assertIs(
                _PaddleDLPackAdapter(tensor).__dlpack__(stream=12345),
                capsule,
            )
        to_dlpack.assert_called_once_with(tensor)

    def test_paddle_stream_pointer_variants(self):
        cuda_stream = SimpleNamespace(
            stream_base=SimpleNamespace(cuda_stream=12345)
        )
        raw_stream = SimpleNamespace(raw_stream=67890)
        with patch("paddle.device.current_stream", return_value=cuda_stream):
            self.assertEqual(dlpack_mod.current_paddle_stream_ptr(), 12345)
        with patch("paddle.device.current_stream", return_value=raw_stream):
            self.assertEqual(dlpack_mod.current_paddle_stream_ptr(), 67890)
        with (
            patch("paddle.device.current_stream", return_value=object()),
            self.assertRaisesRegex(RuntimeError, "cannot obtain"),
        ):
            dlpack_mod.current_paddle_stream_ptr()

    def test_current_cu_stream_wraps_paddle_pointer(self):
        import cuda.bindings.driver as cuda_driver

        cu_stream = object()
        with (
            patch.object(
                dlpack_mod,
                "current_paddle_stream_ptr",
                return_value=12345,
            ),
            patch.object(
                cuda_driver,
                "CUstream",
                return_value=cu_stream,
            ) as constructor,
        ):
            self.assertIs(dlpack_mod.current_cu_stream(), cu_stream)
        constructor.assert_called_once_with(12345)

    def test_paddle_to_cute_tensor_dynamic_layouts(self):
        result = MagicMock()
        dynamic = object()
        result.mark_layout_dynamic.return_value = dynamic
        with patch(
            "cutlass.cute.runtime.from_dlpack",
            return_value=result,
        ) as from_dlpack:
            tensor = object()
            self.assertIs(
                dlpack_mod.paddle_to_cute_tensor(
                    tensor,
                    assumed_align=16,
                    stream_ptr=12345,
                ),
                dynamic,
            )
            result.mark_layout_dynamic.assert_called_once_with()
            adapter = from_dlpack.call_args.args[0]
            self.assertIs(adapter._tensor, tensor)
            self.assertEqual(
                from_dlpack.call_args.kwargs,
                {"assumed_align": 16, "enable_tvm_ffi": True},
            )

            result.mark_layout_dynamic.reset_mock()
            result.mark_layout_dynamic.return_value = dynamic
            self.assertIs(
                dlpack_mod.paddle_to_cute_tensor(
                    tensor,
                    assumed_align=4,
                    leading_dim=1,
                ),
                dynamic,
            )
            result.mark_layout_dynamic.assert_called_once_with(leading_dim=1)


@unittest.skipUnless(_HAS_CUTEDSL_SM100, _CUTEDSL_SKIP_REASON)
class TestIndexerTopKUnfusedRegression(unittest.TestCase):
    """Compare the production CUTEDSL dispatch against unfused Paddle top-k."""

    @staticmethod
    def _scores(rows, cols, dtype):
        # The largest selected value is 0 and the tested top-k range is exactly
        # representable in fp16/bf16, avoiding tie-breaking differences.
        base = -paddle.arange(cols, dtype=paddle.float32).reshape([1, cols])
        return paddle.tile(base, [rows, 1]).cast(dtype).contiguous()

    def _compare_backends(
        self,
        scores,
        lengths,
        top_k,
        return_val,
        topk_backend="cutedsl",
    ):
        reference = _indexer_top_k_unfused(
            scores,
            lengths,
            top_k,
            return_val=return_val,
            topk_backend="unfused",
        )
        actual = _indexer_top_k_unfused(
            scores,
            lengths,
            top_k,
            return_val=return_val,
            topk_backend=topk_backend,
        )
        paddle.device.synchronize()
        actual_indices = actual["indices"]
        reference_indices = reference["indices"]
        valid_counts = paddle.minimum(
            lengths.reshape([-1, 1]),
            paddle.full(
                [lengths.shape[0], 1],
                top_k,
                dtype=lengths.dtype,
            ),
        )
        valid_slots = (
            paddle.arange(top_k, dtype=lengths.dtype).reshape([1, -1])
            < valid_counts
        )
        adjacent_valid = valid_slots[:, 1:] & valid_slots[:, :-1]
        self.assertTrue(
            bool(
                paddle.all(
                    (~adjacent_valid)
                    | (actual_indices[:, 1:] > actual_indices[:, :-1])
                ).item()
            )
        )
        self.assertTrue(
            bool(
                paddle.all(
                    paddle.where(
                        valid_slots,
                        (actual_indices >= 0)
                        & (actual_indices < lengths.reshape([-1, 1])),
                        actual_indices == -1,
                    )
                ).item()
            )
        )
        normalized_indices = paddle.where(
            valid_slots,
            actual_indices,
            paddle.full_like(actual_indices, scores.shape[1]),
        )
        sorted_indices = paddle.sort(normalized_indices, axis=1)
        duplicate_indices = (
            sorted_indices[:, 1:] == sorted_indices[:, :-1]
        ) & (sorted_indices[:, 1:] != scores.shape[1])
        self.assertTrue(bool(paddle.all(~duplicate_indices).item()))

        safe_actual = paddle.clip(
            actual_indices, min=0, max=scores.shape[1] - 1
        )
        safe_reference = paddle.clip(
            reference_indices, min=0, max=scores.shape[1] - 1
        )
        actual_scores = paddle.take_along_axis(
            scores, safe_actual, axis=1
        ).cast("float32")
        reference_scores = paddle.take_along_axis(
            scores, safe_reference, axis=1
        ).cast("float32")
        neg_inf = paddle.full_like(actual_scores, float("-inf"))
        actual_scores = paddle.where(valid_slots, actual_scores, neg_inf)
        reference_scores = paddle.where(valid_slots, reference_scores, neg_inf)
        self.assertTrue(
            bool(
                paddle.all(
                    paddle.sort(actual_scores, axis=1)
                    == paddle.sort(reference_scores, axis=1)
                ).item()
            )
        )
        if return_val:
            self.assertTrue(
                bool(
                    paddle.all(
                        paddle.where(
                            valid_slots,
                            actual["values"].cast("float32") == actual_scores,
                            actual["values"] == actual["values"],
                        )
                    ).item()
                )
            )
        else:
            self.assertIsNone(actual["values"])
            self.assertIsNone(reference["values"])

    def test_input_validation_errors(self):
        gpu_input = paddle.zeros([2, 8], dtype=paddle.float32)
        gpu_lengths = paddle.full([2], 8, dtype=paddle.int32)
        cpu_input = paddle.to_tensor(
            [[0.0] * 8] * 2,
            dtype=paddle.float32,
            place=paddle.CPUPlace(),
        )
        cpu_lengths = paddle.to_tensor(
            [8, 8],
            dtype=paddle.int32,
            place=paddle.CPUPlace(),
        )
        cases = (
            (cpu_input, cpu_lengths, 1, ValueError, "CUDA tensors"),
            (
                paddle.zeros([8], dtype=paddle.float32),
                paddle.full([8], 8, dtype=paddle.int32),
                1,
                ValueError,
                "must be 2-D",
            ),
            (
                gpu_input,
                paddle.full([2, 1], 8, dtype=paddle.int32),
                1,
                ValueError,
                "must be 1-D",
            ),
            (
                gpu_input,
                paddle.full([3], 8, dtype=paddle.int32),
                1,
                ValueError,
                "one entry per input row",
            ),
            (
                gpu_input,
                paddle.full([2], 8, dtype=paddle.int64),
                1,
                TypeError,
                "dtype paddle.int32",
            ),
            (gpu_input, gpu_lengths, 0, ValueError, r"\[1, 32768\]"),
            (
                gpu_input,
                gpu_lengths,
                16385,
                ValueError,
                r"\[1, 16384\] for float32",
            ),
            (gpu_input, gpu_lengths, 32769, ValueError, r"\[1, 32768\]"),
            (
                paddle.empty([2, 0], dtype=paddle.float32),
                paddle.zeros([2], dtype=paddle.int32),
                1,
                ValueError,
                "non-empty column",
            ),
        )
        for input_values, row_lengths, top_k, error, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(error, message),
            ):
                topk_mod.indexer_topk_prefill(input_values, row_lengths, top_k)

    def test_output_and_workspace_validation_errors(self):
        rows, cols, top_k = 2, 64, 4
        scores = self._scores(rows, cols, paddle.float32)
        lengths = paddle.full([rows], cols, dtype=paddle.int32)
        cases = (
            (
                {
                    "out_indices": paddle.empty(
                        [rows, top_k + 1], dtype=paddle.int32
                    )
                },
                "out_indices must have shape",
            ),
            (
                {
                    "out_indices": paddle.empty(
                        [rows, top_k], dtype=paddle.float32
                    )
                },
                "out_indices must be contiguous",
            ),
            (
                {
                    "out_values": paddle.empty(
                        [rows, top_k + 1], dtype=paddle.float32
                    )
                },
                "out_values must have shape",
            ),
            (
                {
                    "out_values": paddle.empty(
                        [rows, top_k], dtype=paddle.float16
                    )
                },
                "out_values must be contiguous",
            ),
            (
                {"workspace": paddle.empty([1, 1, 1], dtype=paddle.int32)},
                "workspace must be contiguous",
            ),
            (
                {"scheduler_state": paddle.zeros([2], dtype=paddle.int32)},
                "scheduler_state must be contiguous",
            ),
        )
        with patch.object(
            topk_mod, "_compile_kernel", return_value=MagicMock()
        ):
            for kwargs, message in cases:
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    topk_mod.indexer_topk_prefill(
                        scores,
                        lengths,
                        top_k,
                        **kwargs,
                    )

    def test_preallocated_outputs_workspace_and_scheduler_state(self):
        rows, cols, top_k = 2, 64, 4
        scores = self._scores(rows, cols, paddle.float32)
        lengths = paddle.full([rows], cols, dtype=paddle.int32)
        indices = paddle.empty([rows, top_k], dtype=paddle.int32)
        values = paddle.empty([rows, top_k], dtype=paddle.float32)
        scheduler_state = paddle.ones([1], dtype=paddle.int32)
        num_ctas, _ = topk_mod._persistent_launch_config(rows)
        smem_bytes = topk_mod._estimate_smem_bytes_per_cta(
            256,
            top_k,
            2,
            min(16384, cols),
            "clc",
        )
        slots = topk_mod._workspace_slot_count(
            rows,
            num_ctas,
            256,
            True,
            "clc",
            smem_bytes_per_cta=smem_bytes,
        )
        workspace = paddle.empty([slots, 2, 1], dtype=paddle.int32)
        compiled = MagicMock()
        cute_tensor = object()
        with (
            patch.object(
                topk_mod,
                "_compile_kernel",
                return_value=compiled,
            ),
            patch.object(
                dlpack_mod,
                "paddle_to_cute_tensor",
                return_value=cute_tensor,
            ),
        ):
            actual_indices, actual_values = topk_mod.indexer_topk_prefill(
                scores,
                lengths,
                top_k,
                out_indices=indices,
                out_values=values,
                workspace=workspace,
                scheduler_state=scheduler_state,
            )
        self.assertIs(actual_indices, indices)
        self.assertIs(actual_values, values)
        self.assertEqual(compiled.call_count, 1)
        self.assertEqual(len(compiled.call_args.args), 6)

    def test_matches_unfused_for_supported_dtypes_and_prefixes(self):
        rows, cols, top_k = 4, 4096, 64
        lengths = paddle.to_tensor([cols, 3073, top_k, 0], dtype=paddle.int32)
        for dtype in (paddle.float16, paddle.bfloat16, paddle.float32):
            with self.subTest(dtype=str(dtype)):
                self._compare_backends(
                    self._scores(rows, cols, dtype),
                    lengths,
                    top_k,
                    return_val=True,
                )

    def test_matches_unfused_without_return_values(self):
        rows, cols, top_k = 3, 4096, 64
        lengths = paddle.to_tensor([cols, 127, 0], dtype=paddle.int32)
        self._compare_backends(
            self._scores(rows, cols, paddle.float16),
            lengths,
            top_k,
            return_val=False,
        )

    def test_matches_unfused_for_long_overflow_columns(self):
        rows, cols, top_k = 2, 32768, 64
        lengths = paddle.to_tensor([cols, 20001], dtype=paddle.int32)
        self._compare_backends(
            self._scores(rows, cols, paddle.float16),
            lengths,
            top_k,
            return_val=True,
        )

    def test_clc_compiled_kernel_reuses_runtime_row_shapes(self):
        cols, top_k = 2048, 32
        topk_mod._COMPILE_CACHE.clear()
        for rows in (2, 5):
            lengths = paddle.arange(cols, cols - rows, -1, dtype=paddle.int32)
            self._compare_backends(
                self._scores(rows, cols, paddle.float32),
                lengths,
                top_k,
                return_val=True,
            )
            self.assertEqual(len(topk_mod._COMPILE_CACHE), 1)

    def test_index_kernel_matches_candidate_set_at_boundary(self):
        rows, cols, top_k = 2, 4096, 2048
        lengths = paddle.to_tensor([cols, 3073], dtype=paddle.int32)
        self._compare_backends(
            self._scores(rows, cols, paddle.float16),
            lengths,
            top_k,
            return_val=True,
            topk_backend="cutedsl_index",
        )

    def test_short_rows_write_prefix_directly_for_static_and_clc(self):
        rows, cols, top_k = 3, 4096, 64
        scores = paddle.tile(
            -paddle.arange(cols, dtype=paddle.float32).reshape([1, -1]),
            [rows, 1],
        ).cast(paddle.float16)
        lengths = paddle.to_tensor([0, 31, cols], dtype=paddle.int32)

        for persistent, schedule_mode in ((False, "static"), (True, "clc")):
            with self.subTest(schedule_mode=schedule_mode):
                indices, values = topk_mod.indexer_topk_prefill(
                    scores,
                    lengths,
                    top_k,
                    return_val=True,
                    persistent=persistent,
                    schedule_mode=schedule_mode,
                    sort_mode="index",
                )
                paddle.device.synchronize()

                self.assertTrue(bool(paddle.all(indices[0] == -1).item()))
                self.assertTrue(
                    bool(
                        paddle.all(
                            paddle.isinf(values[0]) & (values[0] < 0)
                        ).item()
                    )
                )
                self.assertEqual(indices[1, :31].tolist(), list(range(31)))
                self.assertEqual(
                    values[1, :31].cast("float32").tolist(),
                    [-float(i) for i in range(31)],
                )
                self.assertTrue(bool(paddle.all(indices[1, 31:] == -1).item()))
                self.assertTrue(
                    bool(
                        paddle.all(
                            paddle.isinf(values[1, 31:]) & (values[1, 31:] < 0)
                        ).item()
                    )
                )
                self.assertEqual(
                    indices[2].tolist(),
                    list(range(top_k)),
                )


if __name__ == "__main__":
    unittest.main()
