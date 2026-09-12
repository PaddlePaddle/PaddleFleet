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

"""Paddle-facing DSA indexer top-k prefill operator.

This is a standalone CuTe DSL operator adapted from the cuDNN Frontend DSA
indexer top-k implementation.  It intentionally has a smaller first-stage
scope:

* prefill only;
* one persistent CTA per SM, with each CTA processing multiple rows;
* ``next_n == 1``;
* no multi-CTA or merge-block path;
* no autograd integration.

The public entry point accepts Paddle CUDA tensors.  Tensor conversion is
zero-copy through :mod:`.dlpack`.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

try:
    import cutlass
    from cutlass import cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.pipeline import (
        Agent,
        CooperativeGroup,
        PipelineClcFetchAsync,
        PipelineUserType,
        make_pipeline_state,
        pipeline_init_arrive,
        pipeline_init_wait,
    )
    from cutlass.utils import (
        ClcDynamicPersistentTileScheduler,
        ClcDynamicPersistentTileSchedulerParams,
    )
    from cutlass.utils.distributed import atomicAdd
    from cutlass.utils.smem_allocator import SmemAllocator

    from .block_scan import (
        block_prefix_sum_kernel,
        fence_acq_rel_cta,
    )
except ImportError:
    cutlass = None

    class _MissingCute:
        @staticmethod
        def jit(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1:
                return args[0]
            return lambda fn: fn

        kernel = jit

    cute = _MissingCute()
    llvm = None
    make_fake_stream = None
    atomicAdd = None
    SmemAllocator = None
    ClcDynamicPersistentTileScheduler = None
    ClcDynamicPersistentTileSchedulerParams = None
    Agent = None
    CooperativeGroup = None
    PipelineClcFetchAsync = None
    PipelineUserType = None
    make_pipeline_state = None
    pipeline_init_arrive = None
    pipeline_init_wait = None
    block_prefix_sum_kernel = None
    fence_acq_rel_cta = None

_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}
_CLC_PRECOMPILE_CACHE: set[tuple[Any, ...]] = set()
_OUTPUT_ALLOC_WARNING_EMITTED = False
_CUTEDSL_AVAILABLE = cutlass is not None


_SMEM_CANDIDATE_CAPACITY = 16384
_MAX_THREADS_PER_SM = 2048
_SMEM_BYTES_PER_SM100_SM = 228 * 1024
_MAX_TOP_K = 32768
_MAX_TOP_K_FLOAT32 = 16384
# The integer index-order network is valid for every positive top_k. Its
# compile-time sorting extent is the next power of two, with invalid slots
# padded by -1.
_INDEX_SORT_MIN_TOP_K = 1
_SORT_MODES = {"score", "index"}


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _validate_sort_mode(sort_mode: str, top_k: int) -> str:
    sort_mode = str(sort_mode)
    if sort_mode not in _SORT_MODES:
        raise ValueError(
            f"sort_mode={sort_mode!r} is invalid. "
            "Must be one of {'score', 'index'}."
        )
    if sort_mode == "index" and top_k < _INDEX_SORT_MIN_TOP_K:
        raise ValueError(
            "sort_mode='index' requires "
            f"top_k >= {_INDEX_SORT_MIN_TOP_K}, got {top_k}"
        )
    return sort_mode


def _validate_top_k_for_dtype(top_k: int, dtype, cutlass) -> None:
    max_top_k = _MAX_TOP_K_FLOAT32 if dtype == cutlass.Float32 else _MAX_TOP_K
    if top_k > max_top_k:
        dtype_name = (
            "float32" if dtype == cutlass.Float32 else "float16/bfloat16"
        )
        raise ValueError(
            f"top_k must be in [1, {max_top_k}] for {dtype_name}, got {top_k}"
        )


def _compile_num_cols_bucket(num_cols: int) -> int:
    """Return the largest compile bucket needed by the CLC implementation.

    The kernel reads the actual column extent at runtime.  Its only
    compile-time column-dependent resource is the candidate capacity, which
    saturates at ``_SMEM_CANDIDATE_CAPACITY``.  Consequently all larger
    sequence lengths can share the saturated binary (including 256k and
    beyond).
    """
    return min(_next_power_of_two(int(num_cols)), _SMEM_CANDIDATE_CAPACITY)


def _persistent_launch_config(
    num_rows: int, persistent: bool = True, schedule_mode: str = "clc"
) -> tuple[int, int]:
    import paddle

    if num_rows <= 0:
        raise ValueError("input_values must have a non-empty row dimension")
    if schedule_mode not in {"static", "clc"}:
        raise ValueError(
            f"unsupported schedule_mode={schedule_mode!r}; "
            "expected 'static' or 'clc'"
        )
    if schedule_mode == "clc" and not persistent:
        raise ValueError("schedule_mode='clc' requires persistent=True")
    if not persistent:
        return num_rows, 1
    sm_count = int(
        paddle.device.cuda.get_device_properties().multi_processor_count
    )
    if schedule_mode == "static":
        # Static persistent kernels use a runtime-bounded strided row loop.
        # Keep the resident CTA count independent of the current batch shape
        # so one compiled kernel can serve different num_rows values.
        return max(1, sm_count), 1
    # CLC also receives the row count at runtime.  Keep the resident CTA
    # count independent of the current shape so its compiled kernel can be
    # reused across different num_rows values.
    return max(1, sm_count), 1


def _workspace_slot_count(
    num_rows: int,
    num_persistent_ctas: int,
    num_threads: int,
    persistent: bool,
    schedule_mode: str,
    smem_bytes_per_cta: int = 0,
) -> int:
    """Return the maximum number of concurrently allocated CLC slots.

    CLC launches one CTA per logical row, but only resident CTAs execute the
    slot allocation. Bound the resident count by both threads and the
    allocator-rounded shared-memory footprint. Registers can only reduce the
    actual resident CTA count further, so this remains a safe upper bound.
    """
    if not persistent:
        return num_rows
    if schedule_mode != "clc":
        return num_persistent_ctas
    max_ctas_per_sm = max(1, _MAX_THREADS_PER_SM // num_threads)
    if smem_bytes_per_cta > 0:
        max_ctas_per_sm = min(
            max_ctas_per_sm,
            max(1, _SMEM_BYTES_PER_SM100_SM // smem_bytes_per_cta),
        )
    # The workspace is indexed by resident CTA, not by logical row.  Allocate
    # the resident upper bound even for a small input so the CLC binary can
    # share one runtime-bounded layout across all row counts.
    return num_persistent_ctas * max_ctas_per_sm


def _estimate_smem_bytes_per_cta(
    num_threads: int,
    top_k: int,
    num_candidate_pages: int,
    candidate_capacity: int,
    schedule_mode: str,
) -> int:
    """Estimate allocator-rounded shared memory used by one top-k CTA."""
    alignment = 128

    def align(size):
        return ((size + alignment - 1) // alignment) * alignment

    num_warps = num_threads // 32
    sizes = (
        (num_warps * 256 + 1) * 4,  # replicated histogram + threshold
        4 * 4,  # counters
        _next_power_of_two(top_k) * 4,  # selected indices
        num_warps * 4,  # scan warp sums
        2 * 4,  # shared candidate counts
        2 * 4,  # global candidate counts
        num_candidate_pages * candidate_capacity * 4,
    )
    if schedule_mode == "clc":
        sizes += (2 * 8, 4 * 4, 4 * 4)  # mbarriers, response, work metadata
    return sum(align(size) for size in sizes)


def _require_cutedsl():
    global _CUTEDSL_AVAILABLE
    if not _CUTEDSL_AVAILABLE:
        try:
            import cutlass as _cutlass
            import cutlass.cute as _cute
            from cutlass._mlir.dialects import llvm as _llvm
            from cutlass.cute.runtime import (
                make_fake_stream as _make_fake_stream,
            )
            from cutlass.pipeline import (
                Agent as _agent,
                CooperativeGroup as _cooperative_group,
                PipelineClcFetchAsync as _pipeline_clc_fetch_async,
                PipelineUserType as _pipeline_user_type,
                make_pipeline_state as _make_pipeline_state,
                pipeline_init_arrive as _pipeline_init_arrive,
                pipeline_init_wait as _pipeline_init_wait,
            )
            from cutlass.utils import (
                ClcDynamicPersistentTileScheduler as _clc_scheduler,
                ClcDynamicPersistentTileSchedulerParams as _clc_params,
            )
            from cutlass.utils.distributed import atomicAdd as _atomic_add
            from cutlass.utils.smem_allocator import (
                SmemAllocator as _smem_allocator,
            )

            from .block_scan import (
                block_prefix_sum_kernel as _block_prefix_sum_kernel,
                fence_acq_rel_cta as _fence_acq_rel_cta,
            )
        except ImportError as exc:
            raise ImportError(
                "paddlefleet.cutedsl_ops requires the CuTe DSL Python runtime"
            ) from exc
        globals().update(
            cutlass=_cutlass,
            cute=_cute,
            llvm=_llvm,
            make_fake_stream=_make_fake_stream,
            atomicAdd=_atomic_add,
            SmemAllocator=_smem_allocator,
            ClcDynamicPersistentTileScheduler=_clc_scheduler,
            ClcDynamicPersistentTileSchedulerParams=_clc_params,
            Agent=_agent,
            CooperativeGroup=_cooperative_group,
            PipelineClcFetchAsync=_pipeline_clc_fetch_async,
            PipelineUserType=_pipeline_user_type,
            make_pipeline_state=_make_pipeline_state,
            pipeline_init_arrive=_pipeline_init_arrive,
            pipeline_init_wait=_pipeline_init_wait,
            block_prefix_sum_kernel=_block_prefix_sum_kernel,
            fence_acq_rel_cta=_fence_acq_rel_cta,
        )
        _CUTEDSL_AVAILABLE = True
    return cutlass, cute, make_fake_stream, SmemAllocator


def _cutlass_dtype(paddle_dtype, cutlass):
    import paddle

    mapping = {
        paddle.float16: cutlass.Float16,
        paddle.bfloat16: cutlass.BFloat16,
        paddle.float32: cutlass.Float32,
    }
    try:
        return mapping[paddle_dtype]
    except KeyError as exc:
        raise TypeError(
            "indexer_topk_prefill supports paddle.float16, "
            "paddle.bfloat16, and paddle.float32"
        ) from exc


def _is_cuda_tensor(tensor) -> bool:
    place = getattr(tensor, "place", None)
    checker = getattr(place, "is_gpu_place", None)
    if checker is not None:
        return bool(checker())
    return "gpu" in str(place).lower() or "cuda" in str(place).lower()


class IndexerTopKPrefillKernel:
    """Single-CTA-per-row CuTe DSL radix-select top-k kernel.

    The implementation is adapted from the cuDNN Frontend DSA indexer
    ``IndexerTopKKernelVarlen``.  Its ordinary one-CTA-per-row algorithm is
    scheduled through a persistent CTA grid; decode scheduling, multi-CTA
    collection, and merge-block payloads are intentionally absent.
    """

    def __init__(
        self,
        dtype,
        max_num_cols: int,
        top_k: int,
        return_val: bool,
        num_persistent_ctas: int,
        rows_per_cta: int,
        schedule_mode: str = "clc",
        num_rows: int = 0,
        num_workspace_slots: int = 0,
        sort_mode: str = "score",
    ):
        cutlass, cute, _, _ = _require_cutedsl()
        self.cutlass = cutlass
        self.cute = cute
        self.dtype = dtype
        self.max_num_cols = int(max_num_cols)
        self.top_k = int(top_k)
        self.sort_mode = _validate_sort_mode(sort_mode, self.top_k)
        self.return_val = bool(return_val)
        self.num_persistent_ctas = int(num_persistent_ctas)
        self.rows_per_cta = int(rows_per_cta)
        self.schedule_mode = str(schedule_mode)
        self.use_clc = self.schedule_mode == "clc"
        # Both schedulers use a shape-independent compiled kernel.  Static
        # launches a fixed resident grid; CLC launches the runtime row count
        # from the input tensor in its host wrapper.
        self.num_launch_ctas = self.num_persistent_ctas
        self.num_candidate_pages = 2 if dtype == cutlass.Float32 else 1
        self.smem_candidate_capacity = min(
            _SMEM_CANDIDATE_CAPACITY, self.max_num_cols
        )
        self.num_threads = 256 if self.max_num_cols < 8192 else 512
        self.num_warps = self.num_threads // 32
        self.histogram_warp_base = 0
        self.histogram_threshold_idx = (
            self.histogram_warp_base + self.num_warps * 256
        )
        self.num_workspace_slots = int(
            num_workspace_slots or self.num_persistent_ctas
        )
        self.sort_size = _next_power_of_two(self.top_k)
        self.first_refine_shift = 24 if dtype == cutlass.Float32 else 0
        self.num_refine_rounds = 4 if dtype == cutlass.Float32 else 1
        # Production DSA sequence lengths are bounded by 262144 columns, so
        # three index-refinement bytes fully order every valid column index.
        self.num_index_refine_rounds = 3

    @cute.jit
    def _topk_per_row(
        self,
        input_values,
        row_lengths,
        output_indices,
        output_values,
        overflow,
        s_histogram,
        s_counter,
        s_indices,
        s_warp_sums,
        s_num_input,
        g_num_input,
        s_input_idx,
        row_idx,
        workspace_slot,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        neg_inf = self.dtype(self.dtype.inf * self.dtype(-1.0))
        length = row_lengths[row_idx]
        if length > input_values.shape[1]:
            length = input_values.shape[1]
        if length < 0:
            length = 0
        short_row = length < self.top_k

        # This is a CTA-wide branch: every thread takes the same path, and
        # the barriers remain balanced for both short and ordinary rows.
        if short_row:
            for i in cutlass.range(tidx, self.top_k, self.num_threads):
                if i < length:
                    output_indices[row_idx, i] = i
                    if cutlass.const_expr(self.return_val):
                        output_values[row_idx, i] = input_values[row_idx, i]
                else:
                    output_indices[row_idx, i] = -1
                    if cutlass.const_expr(self.return_val):
                        output_values[row_idx, i] = neg_inf
        cute.arch.barrier()
        if not short_row:
            self._topk_per_row_full(
                input_values,
                row_lengths,
                output_indices,
                output_values,
                overflow,
                s_histogram,
                s_counter,
                s_indices,
                s_warp_sums,
                s_num_input,
                g_num_input,
                s_input_idx,
                row_idx,
                workspace_slot,
            )
        cute.arch.barrier()

    @cute.jit
    def _topk_per_row_full(
        self,
        input_values,
        row_lengths,
        output_indices,
        output_values,
        overflow,
        s_histogram,
        s_counter,
        s_indices,
        s_warp_sums,
        s_num_input,
        g_num_input,
        s_input_idx,
        row_idx,
        workspace_slot,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx = row_idx

        neg_inf = self.dtype(self.dtype.inf * self.dtype(-1.0))
        val_one = cutlass.Int32(1)
        if tidx < self.num_candidate_pages:
            s_num_input[tidx] = 0
            g_num_input[tidx] = 0

        # Initialize the selected buffer before coarse collection.  Entries
        # from bins strictly better than the threshold are written into this
        # buffer and must survive into the refinement/output phases.
        for i in cutlass.range(tidx, self.sort_size, self.num_threads):
            s_indices[i] = -1
        cute.arch.barrier()

        # Stage 1: build a coarse radix histogram.  This is the same ordered
        # floating-point key used by the extracted cuDNN Frontend kernel.
        self._clear_histogram(tidx, s_histogram)

        length = row_lengths[row_idx]
        if length > input_values.shape[1]:
            length = input_values.shape[1]
        if length < 0:
            length = 0
        for col in cutlass.range(tidx, length, self.num_threads):
            key = self._to_coarse_key(input_values[row_idx, col])
            bin_offset = self.histogram_warp_base + (tidx // 32) * 256
            atomicAdd(
                s_histogram.iterator
                + cutlass.Int32(bin_offset)
                + cutlass.Int32(key),
                val_one,
            )
        fence_acq_rel_cta()
        self._reduce_histogram(tidx, s_histogram)
        if tidx == 0:
            s_histogram[self.histogram_threshold_idx] = 255

        # Convert the histogram to an inclusive prefix sum.  This follows the
        # reference block-scan path: one thread handles one radix bin and
        # warp shuffles handle the intra-warp scan.
        val = cutlass.Int32(0)
        if tidx < 256:
            val = s_histogram[tidx]
            val, _ = block_prefix_sum_kernel(
                val,
                s_warp_sums,
                tidx,
                256,
                8,
                barrier_id=1,
            )
            s_histogram[tidx] = val
        cute.arch.barrier()
        if tidx < 256:
            previous = 0
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous < self.top_k and val >= self.top_k:
                s_histogram[self.histogram_threshold_idx] = tidx
        if tidx == 0:
            s_counter[0] = 0  # selected count
            s_counter[1] = 0  # final boundary-score candidate count
            s_counter[2] = 0  # final threshold remaining count
            s_counter[3] = -1  # no index threshold unless a tie overflows
        cute.arch.barrier()

        threshold = s_histogram[self.histogram_threshold_idx]
        # Store only candidates in the selected coarse prefix.  The exact
        # ordering inside this final bin is resolved below.
        for col in cutlass.range(tidx, length, self.num_threads):
            key = self._to_coarse_key(input_values[row_idx, col])
            if key < threshold:
                pos = atomicAdd(s_counter.iterator, val_one)
                if pos < self.top_k:
                    s_indices[pos] = col
            elif key == threshold:
                pos = atomicAdd(s_num_input.iterator, val_one)
                if pos < self.smem_candidate_capacity:
                    s_input_idx[0, pos] = col
                else:
                    overflow_pos = atomicAdd(
                        g_num_input.iterator,
                        val_one,
                    )
                    overflow[workspace_slot, 0, overflow_pos] = col
        fence_acq_rel_cta()
        cute.arch.barrier()

        # Refine the selected coarse bin one byte at a time.  Candidates
        # below the threshold are committed to s_indices; candidates equal
        # to it are carried to the shared/global ping-pong candidate buffers.
        for refine_round in range(self.num_refine_rounds):
            current = refine_round & 1
            next_buffer = current ^ 1
            candidate_count = s_num_input[current]
            run_refinement = s_counter[0] < self.top_k
            self._clear_histogram(tidx, s_histogram)
            if tidx == 0:
                s_num_input[next_buffer] = 0
                g_num_input[next_buffer] = 0
                s_histogram[self.histogram_threshold_idx] = 255
            cute.arch.barrier()

            if run_refinement:
                smem_count = candidate_count
                if smem_count > self.smem_candidate_capacity:
                    smem_count = self.smem_candidate_capacity
                for candidate in cutlass.range(
                    tidx, smem_count, self.num_threads
                ):
                    col = s_input_idx[current, candidate]
                    key = (
                        self._to_ordered(input_values[bidx, col])
                        >> (self.first_refine_shift - refine_round * 8)
                    ) & 0xFF
                    bin_offset = self.histogram_warp_base + (tidx // 32) * 256
                    atomicAdd(
                        s_histogram.iterator
                        + cutlass.Int32(bin_offset)
                        + cutlass.Int32(key),
                        val_one,
                    )
                for candidate in cutlass.range(
                    tidx, g_num_input[current], self.num_threads
                ):
                    col = overflow[workspace_slot, current, candidate]
                    key = (
                        self._to_ordered(input_values[bidx, col])
                        >> (self.first_refine_shift - refine_round * 8)
                    ) & 0xFF
                    bin_offset = self.histogram_warp_base + (tidx // 32) * 256
                    atomicAdd(
                        s_histogram.iterator
                        + cutlass.Int32(bin_offset)
                        + cutlass.Int32(key),
                        val_one,
                    )
            fence_acq_rel_cta()
            self._reduce_histogram(tidx, s_histogram)

            remaining = cutlass.Int32(0)
            val = cutlass.Int32(0)
            if run_refinement and tidx < 256:
                remaining = self.top_k - s_counter[0]
                val = s_histogram[tidx]
                val, _ = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    256,
                    8,
                    barrier_id=1,
                )
                s_histogram[tidx] = val
            cute.arch.barrier()
            if run_refinement and tidx < 256:
                previous = 0
                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous < remaining and val >= remaining:
                    s_histogram[self.histogram_threshold_idx] = tidx
                    s_counter[2] = remaining - previous
                    if refine_round == self.num_refine_rounds - 1:
                        s_counter[1] = val - previous
                if tidx == 255 and val < remaining:
                    # No radix bin reaches the remaining K.  All candidates
                    # are selected and the rest are sentinel padding.
                    s_histogram[self.histogram_threshold_idx] = 255
                    s_counter[2] = val - previous
                    if refine_round == self.num_refine_rounds - 1:
                        s_counter[1] = val - previous
            cute.arch.barrier()

            if run_refinement:
                threshold = s_histogram[self.histogram_threshold_idx]
                if (
                    refine_round == self.num_refine_rounds - 1
                    and s_counter[1] > s_counter[2]
                ):
                    # The final score byte identifies the exact boundary
                    # score. Select the smallest required column indices from
                    # that tie group before committing candidates. Membership
                    # is now data-defined rather than atomic-arrival-defined.
                    self._select_boundary_index(
                        tidx,
                        input_values,
                        overflow,
                        s_histogram,
                        s_counter,
                        s_warp_sums,
                        s_num_input,
                        g_num_input,
                        s_input_idx,
                        bidx,
                        workspace_slot,
                        current,
                        candidate_count,
                        threshold,
                        self.first_refine_shift - refine_round * 8,
                    )
                smem_count = candidate_count
                if smem_count > self.smem_candidate_capacity:
                    smem_count = self.smem_candidate_capacity
                for candidate in cutlass.range(
                    tidx, smem_count, self.num_threads
                ):
                    col = s_input_idx[current, candidate]
                    key = (
                        self._to_ordered(input_values[bidx, col])
                        >> (self.first_refine_shift - refine_round * 8)
                    ) & 0xFF
                    if key < threshold:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        if pos < self.top_k:
                            s_indices[pos] = col
                    elif key == threshold:
                        if refine_round == self.num_refine_rounds - 1:
                            if s_counter[3] < 0 or col <= s_counter[3]:
                                pos = atomicAdd(s_counter.iterator, val_one)
                                if pos < self.top_k:
                                    s_indices[pos] = col
                        else:
                            pos = atomicAdd(
                                s_num_input.iterator + next_buffer,
                                val_one,
                            )
                            if pos < self.smem_candidate_capacity:
                                s_input_idx[next_buffer, pos] = col
                            else:
                                overflow_pos = atomicAdd(
                                    g_num_input.iterator + next_buffer,
                                    val_one,
                                )
                                overflow[
                                    workspace_slot, next_buffer, overflow_pos
                                ] = col
                for candidate in cutlass.range(
                    tidx, g_num_input[current], self.num_threads
                ):
                    col = overflow[workspace_slot, current, candidate]
                    key = (
                        self._to_ordered(input_values[bidx, col])
                        >> (self.first_refine_shift - refine_round * 8)
                    ) & 0xFF
                    if key < threshold:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        if pos < self.top_k:
                            s_indices[pos] = col
                    elif key == threshold:
                        if refine_round == self.num_refine_rounds - 1:
                            if s_counter[3] < 0 or col <= s_counter[3]:
                                pos = atomicAdd(s_counter.iterator, val_one)
                                if pos < self.top_k:
                                    s_indices[pos] = col
                        else:
                            pos = atomicAdd(
                                s_num_input.iterator + next_buffer,
                                val_one,
                            )
                            if pos < self.smem_candidate_capacity:
                                s_input_idx[next_buffer, pos] = col
                            else:
                                overflow_pos = atomicAdd(
                                    g_num_input.iterator + next_buffer,
                                    val_one,
                                )
                                overflow[
                                    workspace_slot, next_buffer, overflow_pos
                                ] = col
                cute.arch.barrier()

            fence_acq_rel_cta()
            cute.arch.barrier()

        if cutlass.const_expr(self.sort_mode == "index"):
            self._sort_selected_by_index(tidx, s_indices)
        else:
            # The score implementation accepts a value scratch tensor, but the
            # bitonic network reads score keys directly from input_values.
            # s_indices is passed as the unused compatibility argument so the
            # score kernel keeps its original interface without extra smem.
            self._sort_selected_by_score(
                tidx,
                s_indices,
                s_indices,
                input_values,
                bidx,
            )
        cute.arch.barrier()

        for i in cutlass.range(tidx, self.top_k, self.num_threads):
            # ``row_lengths`` remains a hard validity boundary even on the
            # full selection path. Never expose a corrupted candidate.
            index = s_indices[i]
            if i < length:
                if index >= 0:
                    if index < length:
                        output_indices[bidx, i] = index
                        if cutlass.const_expr(self.return_val):
                            output_values[bidx, i] = input_values[bidx, index]
                    else:
                        output_indices[bidx, i] = -1
                        if cutlass.const_expr(self.return_val):
                            output_values[bidx, i] = neg_inf
                else:
                    output_indices[bidx, i] = -1
                    if cutlass.const_expr(self.return_val):
                        output_values[bidx, i] = neg_inf
            else:
                output_indices[bidx, i] = -1
                if cutlass.const_expr(self.return_val):
                    output_values[bidx, i] = neg_inf
        cute.arch.barrier()

    @cute.kernel
    def _topk_static_kernel(
        self,
        input_values,
        row_lengths,
        output_indices,
        output_values,
        overflow,
        scheduler_state,
    ):
        smem = SmemAllocator()
        s_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (self.histogram_threshold_idx + 1,),
                order=(0,),
            ),
            byte_alignment=128,
        )
        s_counter = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=128,
        )
        s_indices = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.sort_size,), order=(0,)),
            byte_alignment=128,
        )
        s_warp_sums = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (self.num_threads // 32,),
                order=(0,),
            ),
            byte_alignment=128,
        )
        s_num_input = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=128,
        )
        g_num_input = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=128,
        )
        s_input_idx = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (self.num_candidate_pages, self.smem_candidate_capacity),
                order=(1, 0),
            ),
            byte_alignment=128,
        )
        slot_idx, _, _ = cute.arch.block_idx()
        for row_idx in cutlass.range(
            slot_idx,
            input_values.shape[0],
            self.num_persistent_ctas,
        ):
            if row_idx < input_values.shape[0]:
                self._topk_per_row(
                    input_values,
                    row_lengths,
                    output_indices,
                    output_values,
                    overflow,
                    s_histogram,
                    s_counter,
                    s_indices,
                    s_warp_sums,
                    s_num_input,
                    g_num_input,
                    s_input_idx,
                    row_idx,
                    slot_idx,
                )

    @cute.jit
    def _clear_histogram(self, tidx, s_histogram):
        for i in cutlass.range(
            self.histogram_warp_base + tidx,
            self.histogram_threshold_idx,
            self.num_threads,
        ):
            s_histogram[i] = 0
        cute.arch.barrier()

    @cute.jit
    def _reduce_histogram(self, tidx, s_histogram):
        cute.arch.barrier()
        if tidx < 256:
            total = cutlass.Int32(0)
            for warp_idx in range(self.num_warps):
                total = (
                    total
                    + s_histogram[
                        self.histogram_warp_base + warp_idx * 256 + tidx
                    ]
                )
            s_histogram[tidx] = total
        cute.arch.barrier()

    @cute.jit
    def _to_coarse_key(self, value):
        """Return the descending-order coarse radix key for ``value``."""
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            # Match the cuDNN Frontend source: FP32 uses an FP16 coarse
            # histogram, then refines with the full FP32 ordered key.
            coarse = value.to(cutlass.Float16)
            bits = llvm.bitcast(cutlass.Uint16.mlir_type, coarse.ir_value())
            ordered = cutlass.Uint16(0)
            if bits & 0x8000:
                ordered = cutlass.Uint16(bits)
            else:
                ordered = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(
                    0x7FFF
                )
            coarse_key = cutlass.Uint8((ordered >> 8) & 0xFF)
            return coarse_key

        bits = llvm.bitcast(cutlass.Uint16.mlir_type, value.ir_value())
        ordered = cutlass.Uint16(0)
        if bits & 0x8000:
            ordered = cutlass.Uint16(bits)
        else:
            ordered = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
        coarse_key = cutlass.Uint8((ordered >> 8) & 0xFF)
        return coarse_key

    @cute.jit
    def _to_ordered(self, value):
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            bits = llvm.bitcast(cutlass.Uint32.mlir_type, value.ir_value())
            ordered = cutlass.Uint32(0)
            if bits & 0x80000000:
                ordered = cutlass.Uint32(bits)
            else:
                ordered = cutlass.Uint32(
                    bits ^ cutlass.Uint32(0xFFFFFFFF)
                ) & cutlass.Uint32(0x7FFFFFFF)
        else:
            bits = llvm.bitcast(cutlass.Uint16.mlir_type, value.ir_value())
            ordered = cutlass.Uint16(0)
            if bits & 0x8000:
                ordered = cutlass.Uint16(bits)
            else:
                ordered = cutlass.Uint16(
                    bits ^ cutlass.Uint16(0xFFFF)
                ) & cutlass.Uint16(0x7FFF)
        return ordered

    @cute.jit
    def _sort_selected_by_score(
        self, tidx, s_indices, s_values, input_values, bidx
    ):
        """Original score-descending/index-ascending bitonic output path."""
        sentinel = -1
        for p in cutlass.range(tidx, self.sort_size, self.num_threads):
            if p >= self.top_k:
                s_indices[p] = sentinel
        cute.arch.barrier()
        k = 2
        while k <= self.sort_size:
            j = k >> 1
            while j > 0:
                for p in cutlass.range(tidx, self.sort_size, self.num_threads):
                    partner = p ^ j
                    if partner > p:
                        lhs = s_indices[p]
                        rhs = s_indices[partner]
                        lhs_invalid = lhs < 0
                        rhs_invalid = rhs < 0
                        ascending = (p & k) == 0
                        swap = False
                        if lhs_invalid != rhs_invalid:
                            swap = lhs_invalid if ascending else not lhs_invalid
                        elif not lhs_invalid:
                            lhs_key = self._to_ordered(input_values[bidx, lhs])
                            rhs_key = self._to_ordered(input_values[bidx, rhs])
                            if ascending:
                                swap = lhs_key > rhs_key or (
                                    lhs_key == rhs_key and lhs > rhs
                                )
                            else:
                                swap = lhs_key < rhs_key or (
                                    lhs_key == rhs_key and lhs < rhs
                                )
                        if swap:
                            s_indices[p] = rhs
                            s_indices[partner] = lhs
                cute.arch.barrier()
                j >>= 1
            k <<= 1

    @cute.jit
    def _select_boundary_index(
        self,
        tidx,
        input_values,
        overflow,
        s_histogram,
        s_counter,
        s_warp_sums,
        s_num_input,
        g_num_input,
        s_input_idx,
        bidx,
        workspace_slot,
        current,
        candidate_count,
        score_threshold,
        score_shift,
    ):
        """Find the smallest index prefix containing the boundary tie slots."""
        val_one = cutlass.Int32(1)
        if tidx == 0:
            s_counter[1] = 0  # selected high-order index prefix
            s_counter[3] = -1  # final inclusive index threshold
        cute.arch.barrier()

        for index_round in range(self.num_index_refine_rounds):
            self._clear_histogram(tidx, s_histogram)

            index_prefix = s_counter[1]
            index_shift = (self.num_index_refine_rounds - index_round - 1) * 8
            smem_count = candidate_count
            if smem_count > self.smem_candidate_capacity:
                smem_count = self.smem_candidate_capacity
            for candidate in cutlass.range(tidx, smem_count, self.num_threads):
                col = s_input_idx[current, candidate]
                score_key = (
                    self._to_ordered(input_values[bidx, col]) >> score_shift
                ) & 0xFF
                candidate_prefix = col >> (index_shift + 8)
                if score_key == score_threshold:
                    if candidate_prefix == index_prefix:
                        index_key = (col >> index_shift) & 0xFF
                        bin_offset = (
                            self.histogram_warp_base + (tidx // 32) * 256
                        )
                        atomicAdd(
                            s_histogram.iterator
                            + cutlass.Int32(bin_offset)
                            + cutlass.Int32(index_key),
                            val_one,
                        )
            for candidate in cutlass.range(
                tidx, g_num_input[current], self.num_threads
            ):
                col = overflow[workspace_slot, current, candidate]
                score_key = (
                    self._to_ordered(input_values[bidx, col]) >> score_shift
                ) & 0xFF
                candidate_prefix = col >> (index_shift + 8)
                if score_key == score_threshold:
                    if candidate_prefix == index_prefix:
                        index_key = (col >> index_shift) & 0xFF
                        bin_offset = (
                            self.histogram_warp_base + (tidx // 32) * 256
                        )
                        atomicAdd(
                            s_histogram.iterator
                            + cutlass.Int32(bin_offset)
                            + cutlass.Int32(index_key),
                            val_one,
                        )
            fence_acq_rel_cta()
            self._reduce_histogram(tidx, s_histogram)

            remaining = s_counter[2]
            val = cutlass.Int32(0)
            if tidx < 256:
                val = s_histogram[tidx]
                val, _ = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    256,
                    8,
                    barrier_id=1,
                )
                s_histogram[tidx] = val
            cute.arch.barrier()
            if tidx < 256:
                previous = 0
                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous < remaining and val >= remaining:
                    s_counter[1] = index_prefix * 256 + tidx
                    s_counter[2] = remaining - previous
            cute.arch.barrier()

        if tidx == 0:
            s_counter[3] = s_counter[1]
        cute.arch.barrier()

    @cute.jit
    def _sort_selected_by_index(
        self,
        tidx,
        s_indices,
    ):
        """Sort selected indices ascending with an integer-only network."""
        for p in cutlass.range(tidx, self.sort_size, self.num_threads):
            if p >= self.top_k:
                s_indices[p] = -1
        cute.arch.barrier()
        k = 2
        while k <= self.sort_size:
            j = k >> 1
            while j > 0:
                for p in cutlass.range(tidx, self.sort_size, self.num_threads):
                    partner = p ^ j
                    if partner > p:
                        lhs = s_indices[p]
                        rhs = s_indices[partner]
                        lhs_invalid = lhs < 0
                        rhs_invalid = rhs < 0
                        ascending = (p & k) == 0
                        swap = False
                        if lhs_invalid != rhs_invalid:
                            swap = lhs_invalid if ascending else not lhs_invalid
                        elif not lhs_invalid:
                            if ascending:
                                swap = lhs > rhs
                            else:
                                swap = lhs < rhs
                        if swap:
                            s_indices[p] = rhs
                            s_indices[partner] = lhs
                cute.arch.barrier()
                j >>= 1
            k <<= 1

        # Keep the output phase's selected-buffer contract explicit.
        for p in cutlass.range(tidx, self.sort_size, self.num_threads):
            if p >= self.top_k:
                s_indices[p] = -1

    @cute.jit
    def __call__(
        self,
        input_values,
        row_lengths,
        output_indices,
        output_values,
        overflow,
        scheduler_state,
        stream,
    ):
        self._topk_static_kernel(
            input_values,
            row_lengths,
            output_indices,
            output_values,
            overflow,
            scheduler_state,
        ).launch(
            grid=(self.num_launch_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


class IndexerTopKPrefillClcKernel(IndexerTopKPrefillKernel):
    """Experimental CLC-only kernel with a dedicated scheduler warp.

    The static kernel deliberately has no CLC code in its lowering graph.
    This class keeps the dynamic scheduler in a separate kernel and uses the
    CUTLASS ``PipelineClcFetchAsync`` protocol:

    * warp 0 issues CLC queries;
    * the CTA consumes the 16-byte response through a full/empty mbarrier
      pipeline;
    * the next row is published in shared memory before the CTA-wide top-k
      barriers are entered.

    The row being computed is held in a local SSA value while the scheduler
    publishes the next row.  This keeps ``row_idx`` (global input/output
    coordinate) separate from ``workspace_slot`` (resident CTA storage).
    """

    @cute.kernel
    def _topk_clc_kernel(
        self,
        input_values,
        row_lengths,
        output_indices,
        output_values,
        overflow,
        scheduler_state,
    ):
        smem = SmemAllocator()
        s_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (self.histogram_threshold_idx + 1,),
                order=(0,),
            ),
            byte_alignment=128,
        )
        s_counter = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=128,
        )
        s_indices = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.sort_size,), order=(0,)),
            byte_alignment=128,
        )
        s_warp_sums = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (self.num_threads // 32,),
                order=(0,),
            ),
            byte_alignment=128,
        )
        s_num_input = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=128,
        )
        g_num_input = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=128,
        )
        s_input_idx = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout(
                (self.num_candidate_pages, self.smem_candidate_capacity),
                order=(1, 0),
            ),
            byte_alignment=128,
        )
        # PipelineClcFetchAsync uses one full and one empty mbarrier per
        # stage.  Keep both barriers in the CLC-only kernel; the old branch
        # allocated only one barrier and could not represent the protocol.
        s_clc_mbar = smem.allocate_tensor(
            element_type=cutlass.Int64,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=128,
        )
        s_clc_response = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=128,
        )
        # [0] = resident CTA workspace slot, [1] = current/next row,
        # [2] = next-row validity.  [3] is reserved for future double
        # buffering of the row metadata.
        s_clc_work = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=128,
        )

        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            cute.arch.mbarrier_init(s_clc_mbar.iterator, 1)
            cute.arch.mbarrier_init(
                s_clc_mbar.iterator + 1,
                self.num_threads,
            )
            # CLC launches one logical CTA per row.  Only CTAs that actually
            # become resident execute this allocation.
            s_clc_work[0] = atomicAdd(
                scheduler_state.iterator,
                cutlass.Int32(1),
            )
            block_idx, _, _ = cute.arch.block_idx()
            s_clc_work[1] = block_idx
            s_clc_work[2] = 1

        # The pipeline was created with defer_sync=True.  Publish the
        # mbarrier initialization before any producer/consumer operation.
        pipeline_init_arrive()
        pipeline_init_wait()

        clc_pipeline = PipelineClcFetchAsync.create(
            barrier_storage=s_clc_mbar.iterator,
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(
                Agent.Thread,
                self.num_threads,
            ),
            tx_count=16,
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        clc_producer_state = make_pipeline_state(
            PipelineUserType.ProducerConsumer,
            1,
        )
        clc_consumer_state = make_pipeline_state(
            PipelineUserType.Consumer,
            1,
        )
        tile_sched = ClcDynamicPersistentTileScheduler.create(
            ClcDynamicPersistentTileSchedulerParams(
                (input_values.shape[0], 1, 1),
                (1, 1, 1),
            ),
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            s_clc_response.iterator,
        )

        while s_clc_work[2] != 0:
            # Capture the current row before the scheduler publishes the next
            # response into the same shared metadata slot.  The CTA barrier is
            # required here: without it, warp 0 can overwrite s_clc_work[1:3]
            # while later warps are still loading the current row, causing one
            # CTA to mix row_idx/row_length values from two different rows.
            row_idx = s_clc_work[1]
            work_valid = s_clc_work[2] != 0
            cute.arch.barrier()

            # Only warp 0 is the scheduler producer.  The remainder of the
            # CTA stays out of the CLC query path and participates as the
            # consumer of the response pipeline.
            if tidx < 32:
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(
                    clc_producer_state
                )
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

            clc_pipeline.consumer_wait(clc_consumer_state)
            if tidx < 32:
                next_tile = tile_sched.get_current_work()
                if tidx == 0:
                    s_clc_work[1] = next_tile.tile_idx[0]
                    s_clc_work[2] = cutlass.Int32(next_tile.is_valid_tile)
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()

            # Row metadata is now stable for all top-k participants.  The
            # second barrier keeps the scheduler warp in lockstep with the
            # CTA-wide body and prevents a later response from changing the
            # shared row before the current body has consumed it.
            cute.arch.barrier()
            if work_valid:
                self._topk_per_row(
                    input_values,
                    row_lengths,
                    output_indices,
                    output_values,
                    overflow,
                    s_histogram,
                    s_counter,
                    s_indices,
                    s_warp_sums,
                    s_num_input,
                    g_num_input,
                    s_input_idx,
                    row_idx,
                    s_clc_work[0],
                )
            cute.arch.barrier()

    @cute.jit
    def __call__(
        self,
        input_values,
        row_lengths,
        output_indices,
        output_values,
        overflow,
        scheduler_state,
        stream,
    ):
        self._topk_clc_kernel(
            input_values,
            row_lengths,
            output_indices,
            output_values,
            overflow,
            scheduler_state,
        ).launch(
            # CLC needs one logical CTA for every possible row so it can
            # cancel not-yet-started CTAs and return their block indices.
            grid=(input_values.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


def _compile_kernel(
    dtype,
    bucketed_num_cols: int,
    top_k: int,
    return_val: bool,
    num_candidate_pages: int,
    num_persistent_ctas: int,
    rows_per_cta: int,
    schedule_mode: str,
    num_rows: int,
    num_workspace_slots: int,
    sort_mode: str = "score",
):
    sort_mode = _validate_sort_mode(sort_mode, int(top_k))
    if schedule_mode == "clc":
        import paddle

        capability = tuple(paddle.device.cuda.get_device_capability())
        if len(capability) != 2 or capability[0] != 10:
            raise RuntimeError(
                "paddlefleet CUTEDSL CLC top-k requires an SM100-family GPU "
                "(compute capability 10.x); "
                f"got compute capability {capability!r}"
            )
    cutlass, cute, make_fake_stream, _ = _require_cutedsl()
    from cutlass.base_dsl.dsl import BaseDSL

    from .compiler import compile_options

    key = (
        dtype,
        bucketed_num_cols,
        top_k,
        return_val,
        num_candidate_pages,
        num_persistent_ctas,
        None,
        schedule_mode,
        None,
        num_workspace_slots,
        sort_mode,
    )
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    n_rows = cute.sym_int()
    n_cols = cute.sym_int()
    overflow_capacity = cute.sym_int()
    input_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (n_rows, n_cols),
        stride_order=(1, 0),
        assumed_align=16,
    )
    lengths_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_rows,),
        stride_order=(0,),
        assumed_align=4,
    )
    indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_rows, top_k),
        stride_order=(1, 0),
        assumed_align=4,
    )
    values_fake = (
        cute.runtime.make_fake_compact_tensor(
            dtype,
            (n_rows, top_k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if return_val
        else None
    )
    overflow_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_workspace_slots, num_candidate_pages, overflow_capacity),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    scheduler_state_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        stride_order=(0,),
        assumed_align=4,
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_cls = (
        IndexerTopKPrefillClcKernel
        if schedule_mode == "clc"
        else IndexerTopKPrefillKernel
    )
    kernel = kernel_cls(
        dtype,
        bucketed_num_cols,
        top_k,
        return_val,
        num_persistent_ctas,
        rows_per_cta,
        schedule_mode,
        num_rows,
        num_workspace_slots,
        sort_mode,
    )
    # CuTe's kernel compiler does not always materialize nested ``@cute.jit``
    # methods before executing the outer host wrapper.  Preprocess the
    # per-row method explicitly so dynamic ``cutlass.range`` loops are lowered
    # before ``cute.compile`` enters the kernel body.
    for method in (
        IndexerTopKPrefillKernel._topk_per_row,
        IndexerTopKPrefillKernel._topk_per_row_full,
        IndexerTopKPrefillKernel._clear_histogram,
        IndexerTopKPrefillKernel._reduce_histogram,
        IndexerTopKPrefillKernel._to_coarse_key,
        IndexerTopKPrefillKernel._to_ordered,
        IndexerTopKPrefillKernel._sort_selected_by_score,
        IndexerTopKPrefillKernel._select_boundary_index,
        IndexerTopKPrefillKernel._sort_selected_by_index,
        block_prefix_sum_kernel,
        fence_acq_rel_cta,
    ):
        BaseDSL._preprocess_and_replace_code(
            getattr(method, "__wrapped__", method)
        )
    if schedule_mode == "clc":
        BaseDSL._preprocess_and_replace_code(
            getattr(
                IndexerTopKPrefillClcKernel._topk_clc_kernel,
                "__wrapped__",
                IndexerTopKPrefillClcKernel._topk_clc_kernel,
            )
        )
    compiled = cute.compile(
        kernel,
        input_fake,
        lengths_fake,
        indices_fake,
        values_fake,
        overflow_fake,
        scheduler_state_fake,
        fake_stream,
        options=compile_options(),
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def precompile_indexer_topk_clc(
    max_num_cols: int,
    top_k: int,
    *,
    dtype=None,
    return_values: tuple[bool, ...] = (False, True),
    sort_mode: str = "score",
) -> tuple[int, ...]:
    """Precompile every CLC variant needed up to ``max_num_cols``.

    CuTe DSL executors are process-local, so this must run in each training
    process after its CUDA device is selected.  Only CLC binaries are emitted;
    the static scheduler is intentionally excluded.

    Column buckets start at ``next_power_of_two(top_k)`` because production
    clamps ``top_k`` when fewer candidates exist.  Buckets saturate at the
    shared-memory candidate capacity: larger runtime column extents use the
    same generated code and therefore cover 256k sequences without additional
    compilation.

    ``sort_mode="score"`` compiles the original score-order bitonic path.
    ``sort_mode="index"`` compiles the selected integer index algorithm.
    Returns:
        The compile buckets covered by this call.
    """
    import paddle

    capability = tuple(paddle.device.cuda.get_device_capability())
    if len(capability) != 2 or capability[0] != 10:
        raise RuntimeError(
            "paddlefleet CUTEDSL CLC top-k requires an SM100-family GPU "
            "(compute capability 10.x); "
            f"got compute capability {capability!r}"
        )
    _require_cutedsl()

    max_num_cols = int(max_num_cols)
    top_k = int(top_k)
    sort_mode = _validate_sort_mode(sort_mode, top_k)
    if max_num_cols <= 0:
        raise ValueError(f"max_num_cols must be positive, got {max_num_cols}")
    if top_k <= 0 or top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be in [1, {_MAX_TOP_K}], got {top_k}")
    if max_num_cols < top_k:
        raise ValueError(
            f"max_num_cols must be >= top_k, got {max_num_cols} < {top_k}"
        )
    if not return_values:
        raise ValueError("return_values must contain at least one variant")

    if dtype is None:
        dtype = paddle.float32
    cutlass, _, _, _ = _require_cutedsl()
    cutlass_dtype = _cutlass_dtype(dtype, cutlass)
    _validate_top_k_for_dtype(top_k, cutlass_dtype, cutlass)
    max_bucket = _compile_num_cols_bucket(max_num_cols)
    bucket = _compile_num_cols_bucket(top_k)
    buckets = []
    while True:
        buckets.append(bucket)
        if bucket >= max_bucket:
            break
        bucket = min(bucket * 2, max_bucket)

    precompile_key = (
        cutlass_dtype,
        tuple(buckets),
        top_k,
        tuple(bool(value) for value in return_values),
        sort_mode,
    )
    if precompile_key in _CLC_PRECOMPILE_CACHE:
        return tuple(buckets)

    print(
        "[CUTEDSL PRECOMPILE]"
        f" pid={os.getpid()}"
        f" schedule_mode=clc"
        f" max_num_cols={max_num_cols}"
        f" buckets={buckets}"
        f" top_k={top_k}"
        f" sort_mode={sort_mode}"
        f" return_values={tuple(bool(v) for v in return_values)}",
        flush=True,
    )
    for compile_bucket in buckets:
        num_threads = 256 if compile_bucket < 8192 else 512
        num_candidate_pages = 2 if cutlass_dtype == cutlass.Float32 else 1
        smem_bytes_per_cta = _estimate_smem_bytes_per_cta(
            num_threads,
            top_k,
            num_candidate_pages,
            min(_SMEM_CANDIDATE_CAPACITY, compile_bucket),
            "clc",
        )
        num_persistent_ctas, rows_per_cta = _persistent_launch_config(
            1, persistent=True, schedule_mode="clc"
        )
        num_workspace_slots = _workspace_slot_count(
            1,
            num_persistent_ctas,
            num_threads,
            persistent=True,
            schedule_mode="clc",
            smem_bytes_per_cta=smem_bytes_per_cta,
        )
        for return_val in return_values:
            _compile_kernel(
                cutlass_dtype,
                compile_bucket,
                top_k,
                bool(return_val),
                num_candidate_pages,
                num_persistent_ctas,
                rows_per_cta,
                "clc",
                1,
                num_workspace_slots,
                sort_mode,
            )

    _CLC_PRECOMPILE_CACHE.add(precompile_key)
    return tuple(buckets)


def indexer_topk_prefill(
    input_values,
    row_lengths,
    top_k: int,
    *,
    return_val: bool = True,
    out_indices=None,
    out_values=None,
    workspace=None,
    persistent: bool = True,
    schedule_mode: str = "clc",
    scheduler_state=None,
    sort_mode: str = "score",
):
    """Run standalone DSA indexer top-k on Paddle CUDA tensors.

    Args:
        input_values: Contiguous Paddle tensor ``[num_rows, num_cols]``.
        row_lengths: Contiguous int32 Paddle tensor ``[num_rows]``.  Each
            element is the valid prefix length of the corresponding row. When
            the clamped length is smaller than ``top_k``, the kernel returns
            that complete prefix in index order and pads the remaining slots.
        top_k: Number of entries to return. The maximum is 32768 for
            float16/bfloat16 and 16384 for float32.
        sort_mode: ``"score"`` preserves the original score-descending,
            index-ascending tie order. ``"index"`` uses the deterministic
            index-ascending kernel for any positive ``top_k``. Short rows
            return their complete valid prefix in index order in either mode.
        return_val: Whether to return values alongside indices.
        out_indices: Optional reusable contiguous int32 output tensor with
            shape ``[num_rows, top_k]``.
        out_values: Optional reusable contiguous value output tensor with
            shape ``[num_rows, top_k]`` and the same dtype as
            ``input_values``. Ignored when ``return_val`` is false.
        workspace: Optional reusable contiguous int32 overflow tensor with
            shape ``[num_workspace_slots, num_candidate_pages,
            overflow_capacity]``. Static scheduling uses one slot per
            persistent CTA. CLC uses a conservative resident-CTA upper bound
            based on the device SM count plus thread and shared-memory limits,
            rather than allocating one slot per logical row.
            ``num_candidate_pages`` is
            two for float32 and one otherwise; ``overflow_capacity`` is
            ``max(1, num_cols - min(16384, num_cols))``.
        persistent: Use a fixed SM-sized CTA grid and reuse workspace slots
            across rows. ``False`` restores the one-CTA-per-row baseline for
            controlled performance comparisons.
        schedule_mode: ``"static"`` uses the current strided persistent
            scheduler. ``"clc"`` uses Blackwell CLC to dynamically assign
            rows; it requires ``persistent=True``.
        scheduler_state: Optional contiguous int32 ``[1]`` state buffer used
            by the CLC active-CTA workspace-slot allocator.

    Returns:
        ``(indices, values)`` where indices has dtype int32 and invalid
        entries are ``-1``.  Values are ``-inf`` for invalid entries.
    """
    import paddle

    if not _is_cuda_tensor(input_values) or not _is_cuda_tensor(row_lengths):
        raise ValueError(
            "input_values and row_lengths must be Paddle CUDA tensors"
        )
    if input_values.ndim != 2:
        raise ValueError(f"input_values must be 2-D, got {input_values.shape}")
    if row_lengths.ndim != 1:
        raise ValueError(f"row_lengths must be 1-D, got {row_lengths.shape}")
    if input_values.shape[0] != row_lengths.shape[0]:
        raise ValueError("row_lengths must have one entry per input row")
    if not input_values.is_contiguous() or not row_lengths.is_contiguous():
        raise ValueError("input_values and row_lengths must be contiguous")
    if row_lengths.dtype != paddle.int32:
        raise TypeError("row_lengths must have dtype paddle.int32")
    if top_k <= 0 or top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be in [1, {_MAX_TOP_K}], got {top_k}")
    sort_mode = _validate_sort_mode(sort_mode, int(top_k))
    if input_values.shape[1] <= 0:
        raise ValueError("input_values must have a non-empty column dimension")

    cutlass, _, _, _ = _require_cutedsl()
    dtype = _cutlass_dtype(input_values.dtype, cutlass)
    _validate_top_k_for_dtype(int(top_k), dtype, cutlass)
    num_candidate_pages = 2 if dtype == cutlass.Float32 else 1
    num_persistent_ctas, rows_per_cta = _persistent_launch_config(
        int(input_values.shape[0]),
        persistent=persistent,
        schedule_mode=schedule_mode,
    )
    num_rows = int(input_values.shape[0])
    bucketed_num_cols = _compile_num_cols_bucket(int(input_values.shape[1]))
    num_threads = 256 if bucketed_num_cols < 8192 else 512
    smem_bytes_per_cta = _estimate_smem_bytes_per_cta(
        num_threads,
        int(top_k),
        num_candidate_pages,
        min(_SMEM_CANDIDATE_CAPACITY, bucketed_num_cols),
        schedule_mode,
    )
    num_workspace_slots = _workspace_slot_count(
        num_rows,
        num_persistent_ctas,
        num_threads,
        persistent,
        schedule_mode,
        smem_bytes_per_cta=smem_bytes_per_cta,
    )
    compiled = _compile_kernel(
        dtype,
        bucketed_num_cols,
        int(top_k),
        bool(return_val),
        num_candidate_pages,
        num_persistent_ctas,
        rows_per_cta,
        schedule_mode,
        num_rows,
        num_workspace_slots,
        sort_mode,
    )

    overflow_capacity = max(
        1,
        int(input_values.shape[1])
        - min(_SMEM_CANDIDATE_CAPACITY, int(input_values.shape[1])),
    )
    if out_indices is None or (return_val and out_values is None):
        global _OUTPUT_ALLOC_WARNING_EMITTED
        if not _OUTPUT_ALLOC_WARNING_EMITTED:
            warnings.warn(
                "out_indices/out_values is None; "
                "CUTEDSL indexer_topk_prefill allocates temporary output "
                "tensors on every call. Pass reusable output buffers to "
                "avoid repeated allocations.",
                UserWarning,
                stacklevel=2,
            )
            _OUTPUT_ALLOC_WARNING_EMITTED = True
    indices = out_indices
    if indices is None:
        indices = paddle.full(
            [input_values.shape[0], top_k], -1, dtype=paddle.int32
        )
    values = out_values
    if return_val and values is None:
        values = paddle.full(
            [input_values.shape[0], top_k],
            float("-inf"),
            dtype=input_values.dtype,
        )
    if not return_val:
        values = None
    if workspace is None:
        # TODO: Let the owning layer reuse a bounded workspace once its
        # lifetime and multi-stream synchronization are explicit. A global
        # cache is intentionally avoided because long-sequence buffers can be
        # very large and would otherwise remain resident for the process.
        workspace = paddle.empty(
            [num_workspace_slots, num_candidate_pages, overflow_capacity],
            dtype=paddle.int32,
        )
    if schedule_mode == "clc":
        if scheduler_state is None:
            scheduler_state = paddle.zeros([1], dtype=paddle.int32)
        elif (
            list(scheduler_state.shape) != [1]
            or scheduler_state.dtype != paddle.int32
            or not scheduler_state.is_contiguous()
        ):
            raise ValueError(
                "scheduler_state must be contiguous paddle.int32 [1]"
            )
        else:
            paddle.assign(
                paddle.zeros_like(scheduler_state),
                output=scheduler_state,
            )
    else:
        scheduler_state = paddle.zeros([1], dtype=paddle.int32)
    if list(indices.shape) != [input_values.shape[0], top_k]:
        raise ValueError("out_indices must have shape [num_rows, top_k]")
    if indices.dtype != paddle.int32 or not indices.is_contiguous():
        raise ValueError("out_indices must be contiguous paddle.int32")
    if return_val:
        if list(values.shape) != [input_values.shape[0], top_k]:
            raise ValueError("out_values must have shape [num_rows, top_k]")
        if values.dtype != input_values.dtype or not values.is_contiguous():
            raise ValueError(
                "out_values must be contiguous and match input dtype"
            )
    if (
        list(workspace.shape)
        != [num_workspace_slots, num_candidate_pages, overflow_capacity]
        or workspace.dtype != paddle.int32
        or not workspace.is_contiguous()
    ):
        raise ValueError(
            "workspace must be contiguous int32 "
            "[num_workspace_slots, num_candidate_pages, overflow_capacity]"
        )

    # The adapters keep Paddle allocations zero-copy. The compiled TVM-FFI
    # callable obtains the current stream from its environment because the
    # fake stream was created with ``use_tvm_ffi_env_stream=True``. Keep all
    # Paddle tensors alive until this call returns because the launch is
    # asynchronous.
    from .dlpack import paddle_to_cute_tensor

    input_cute = paddle_to_cute_tensor(
        input_values, assumed_align=16, leading_dim=1
    )
    lengths_cute = paddle_to_cute_tensor(
        row_lengths, assumed_align=4, leading_dim=0
    )
    indices_cute = paddle_to_cute_tensor(
        indices, assumed_align=4, leading_dim=1
    )
    values_cute = (
        paddle_to_cute_tensor(values, assumed_align=16, leading_dim=1)
        if return_val
        else None
    )
    overflow_cute = paddle_to_cute_tensor(
        workspace, assumed_align=4, leading_dim=2
    )
    scheduler_state_cute = paddle_to_cute_tensor(
        scheduler_state, assumed_align=4, leading_dim=0
    )

    # The tensors alias Paddle storage through DLPack. The stream is an
    # implicit TVM-FFI environment parameter, not a callable argument.
    compiled(
        input_cute,
        lengths_cute,
        indices_cute,
        values_cute,
        overflow_cute,
        scheduler_state_cute,
    )
    return indices, values


__all__ = [
    "IndexerTopKPrefillKernel",
    "IndexerTopKPrefillClcKernel",
    "indexer_topk_prefill",
]
