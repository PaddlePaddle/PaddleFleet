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

"""Behavior tests for ``paddlefleet.triton_ops.moe_topk_fusion``.

Module under test lives in the repository's "计算优化 / Fused Ops" boundary.
The only CPU-observable pure logic in this module is the host-side dispatch
wrapper ``routing_map_fusion_forward``: it derives ``seq_len``/``moe_k`` from
``topk_indices.shape`` and ``n_experts`` from ``gate_probs.shape[1]``, computes
the launch grid and block sizes, decides the ``has_input_ids`` /
``has_pure_text_mask`` compile-time flags, substitutes the ``topk_indices``
tensor as a dummy pointer when an optional input is ``None``, and zero-fills the
output tensors before the kernel accumulates into them.

The GPU Triton kernel ``_routing_map_fwd_bitmap_kernel`` is a genuine
not-under-test collaborator (it compiles to PTX and requires CUDA); it is
replaced by a spy that records the launch grid and keyword arguments. These
tests therefore assert only the wrapper's pure-Python launch geometry and
argument forwarding against hand-derived expectations. They do NOT drive or
validate the kernel's numerical output -- that requires a real GPU launch.
The whole module is skipped honestly when it (or its ``paddle`` dependency)
cannot be imported.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.triton_ops import moe_topk_fusion as _mtf
    from paddlefleet.triton_ops.moe_topk_fusion import (
        routing_map_fusion_forward,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuine missing dependency -> honest skip
    _mtf = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.triton_ops.moe_topk_fusion (or its paddle dependency) is not "
    f"importable in this environment: {_IMPORT_ERROR!r}"
)


class _KernelLaunchSpy:
    """Stand-in for the GPU kernel that records how it was launched.

    ``routing_map_fusion_forward`` invokes the kernel as
    ``kernel[grid](**kwargs)``; ``__getitem__`` captures the grid and returns a
    callable that captures the keyword arguments. It performs no computation,
    so the wrapper's own zero-fill of ``routing_map`` / ``dispatch_mask``
    survives and can be checked independently.
    """

    def __init__(self):
        self.grid = None
        self.kwargs = None
        self.launch_count = 0

    def __getitem__(self, grid):
        self.grid = grid

        def _launch(**kwargs):
            self.kwargs = kwargs
            self.launch_count += 1

        return _launch


@unittest.skipUnless(_mtf is not None, _SKIP_REASON)
class TestRoutingMapFusionForwardDispatch(unittest.TestCase):
    """Launch-geometry and argument-forwarding contract of the host wrapper."""

    def setUp(self):
        self.spy = _KernelLaunchSpy()
        original = _mtf._routing_map_fwd_bitmap_kernel
        _mtf._routing_map_fwd_bitmap_kernel = self.spy
        self.addCleanup(
            setattr, _mtf, "_routing_map_fwd_bitmap_kernel", original
        )

    def test_geometry_and_placeholder_pointers_without_masks(self):
        # Distinct dims so an axis/source swap cannot pass unnoticed.
        seq_len, n_experts, moe_k = 130, 40, 6
        gate_probs = paddle.zeros((seq_len, n_experts), dtype="float32")
        topk_indices = paddle.arange(seq_len * moe_k, dtype="int32").reshape(
            [seq_len, moe_k]
        )

        routing_map, topk_indices_out, dispatch_mask = (
            routing_map_fusion_forward(gate_probs, topk_indices)
        )

        self.assertEqual(self.spy.launch_count, 1)
        # grid = (cdiv(seq_len, 64), cdiv(n_experts, 32)); hand-derived
        # cdiv(130, 64) = 3, cdiv(40, 32) = 2 -> ordering catches an axis swap.
        self.assertEqual(self.spy.grid, (3, 2))

        kw = self.spy.kwargs
        self.assertEqual(kw["BLOCK_M"], 64)
        self.assertEqual(kw["BLOCK_N"], 32)
        # BLOCK_K = next_power_of_2(6) = 8 (moe_k is deliberately not a power of 2)
        self.assertEqual(kw["BLOCK_K"], 8)
        # n_experts must come from gate_probs (40), not topk_indices (6).
        self.assertEqual(kw["n_experts"], n_experts)
        self.assertEqual(kw["seq_len"], seq_len)
        self.assertEqual(kw["moe_k"], moe_k)
        self.assertEqual(kw["pad_token_id"], 0)

        # Contiguous row-major strides, hand-derived from the two layouts.
        self.assertEqual(kw["stride_topk_s"], moe_k)
        self.assertEqual(kw["stride_topk_k"], 1)
        self.assertEqual(kw["stride_routing_s"], n_experts)
        self.assertEqual(kw["stride_routing_e"], 1)

        # Optional inputs absent -> flags False and the topk_indices tensor is
        # reused as the dummy pointer for both optional operands.
        self.assertFalse(kw["has_input_ids"])
        self.assertFalse(kw["has_pure_text_mask"])
        self.assertIs(kw["topk_indices_ptr"], topk_indices)
        self.assertIs(kw["input_ids_ptr"], topk_indices)
        self.assertIs(kw["is_pure_text_line_ptr"], topk_indices)

        # Wrapper-owned output initialization (kernel is a no-op spy here).
        self.assertEqual(routing_map.shape, [seq_len, n_experts])
        self.assertEqual(routing_map.dtype, paddle.float32)
        np.testing.assert_array_equal(
            routing_map.numpy(),
            np.zeros((seq_len, n_experts), dtype=np.float32),
        )
        self.assertEqual(dispatch_mask.shape, [n_experts])
        self.assertEqual(dispatch_mask.dtype, paddle.int64)
        np.testing.assert_array_equal(
            dispatch_mask.numpy(), np.zeros((n_experts,), dtype=np.int64)
        )
        self.assertEqual(topk_indices_out.shape, [seq_len, moe_k])
        self.assertEqual(topk_indices_out.dtype, topk_indices.dtype)

    def test_flags_and_pointers_with_masks_and_custom_pad(self):
        seq_len, n_experts, moe_k = 65, 96, 4
        gate_probs = paddle.zeros((seq_len, n_experts), dtype="float32")
        topk_indices = paddle.zeros((seq_len, moe_k), dtype="int32")
        input_ids = paddle.zeros((seq_len,), dtype="int64")
        is_pure_text_line = paddle.zeros((seq_len,), dtype="int32")

        routing_map_fusion_forward(
            gate_probs,
            topk_indices,
            input_ids=input_ids,
            is_pure_text_line=is_pure_text_line,
            pad_token_id=7,
        )

        kw = self.spy.kwargs
        # cdiv(65, 64) = 2, cdiv(96, 32) = 3 -> again distinct across axes.
        self.assertEqual(self.spy.grid, (2, 3))
        # BLOCK_K = next_power_of_2(4) = 4 (already a power of 2).
        self.assertEqual(kw["BLOCK_K"], 4)
        self.assertEqual(kw["pad_token_id"], 7)

        # Optional inputs present -> flags True and the real tensors forwarded,
        # NOT the topk_indices placeholder.
        self.assertTrue(kw["has_input_ids"])
        self.assertTrue(kw["has_pure_text_mask"])
        self.assertIs(kw["input_ids_ptr"], input_ids)
        self.assertIs(kw["is_pure_text_line_ptr"], is_pure_text_line)
        self.assertIsNot(kw["input_ids_ptr"], topk_indices)
        self.assertIsNot(kw["is_pure_text_line_ptr"], topk_indices)


if __name__ == "__main__":
    unittest.main()
