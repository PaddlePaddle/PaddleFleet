# Copyright (c) 2026 PaddleFaddle Authors. All Rights Reserved.
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

"""FA3/FA4 dispatch coverage for the flashmask backward/forward paths.

Since FA3 was enabled on the cutedsl backend, ``fa_version`` 3 and 4 share a
single branch that calls ``_flash_attn_fwd`` / ``_flash_attn_bwd`` instead of
the old ``paddle._C_ops.flashmask_attention_v2*`` operators. These tests pin
that dispatch down for both versions. The ``fa_version == 2`` branch is
naturally covered by CI.
"""

import unittest
from unittest.mock import MagicMock, patch

import paddle

_CUTEDSL_VERSIONS = (3, 4)


# ---------------------------------------------------------------------------
# 1) cp_flashmask_allgatherkv_balance_backward  (context_parallel_utils)
# ---------------------------------------------------------------------------


class TestCpFlashmaskBackwardDispatch(unittest.TestCase):
    """fa_version 3 and 4 both reach ``_flash_attn_bwd`` in the CP backward."""

    def _call_backward(self, fa_version):
        from paddlefleet import context_parallel_utils as cpu

        B, S, H, D = 1, 8, 2, 16
        q = paddle.randn([B, S, H, D])
        k = paddle.randn([B, S, H, D])
        v = paddle.randn([B, S, H, D])
        indices = paddle.zeros([B, 2, S], dtype="int64")
        out = paddle.randn([B, S, H, D])
        lse = paddle.randn([B, H, S])
        out_grad = paddle.randn([B, S, H, D])
        dummy = paddle.randn([B, S, H, D])

        mock_bwd = MagicMock(return_value=(dummy, dummy, dummy, None))
        mock_info = MagicMock(name="FlashMaskInfoPaddle")

        with (
            patch.object(cpu, "_flash_attn_bwd", mock_bwd, create=True),
            patch.object(cpu, "FlashMaskInfoPaddle", mock_info, create=True),
            patch.object(
                cpu, "all_gather_balance", side_effect=lambda x, **kw: x
            ),
            patch.object(
                cpu,
                "reduce_scatter_any_axis_balance",
                side_effect=lambda x, **kw: x,
            ),
        ):
            cpu.cp_flashmask_allgatherkv_balance_backward(
                q,
                k,
                v,
                indices,
                out,
                lse,
                out_grad,
                None,  # learnable_sink
                MagicMock(),  # group
                False,  # causal
                fa_version,
                None,  # softmax_scale
            )

        return mock_bwd, mock_info

    def test_cutedsl_backward_dispatch(self):
        for fa_version in _CUTEDSL_VERSIONS:
            with self.subTest(fa_version=fa_version):
                mock_bwd, mock_info = self._call_backward(fa_version)

                mock_bwd.assert_called_once()
                args, kwargs = mock_bwd.call_args
                # q, k, v, out, out_grad, lse, flashmask_info
                self.assertEqual(len(args), 7)
                self.assertIs(args[6], mock_info.return_value)
                self.assertIsNone(kwargs["learnable_sink"])
                self.assertFalse(kwargs["causal"])
                self.assertIsNone(kwargs["softmax_scale"])
                self.assertIn("deterministic", kwargs)

    def test_no_mask_skips_flashmask_info(self):
        """``startend_row_indices is None`` passes ``flashmask_info=None``."""
        from paddlefleet import context_parallel_utils as cpu

        B, S, H, D = 1, 8, 2, 16
        dummy = paddle.randn([B, S, H, D])
        mock_bwd = MagicMock(return_value=(dummy, dummy, dummy, None))

        with (
            patch.object(cpu, "_flash_attn_bwd", mock_bwd, create=True),
            patch.object(
                cpu, "all_gather_balance", side_effect=lambda x, **kw: x
            ),
            patch.object(
                cpu,
                "reduce_scatter_any_axis_balance",
                side_effect=lambda x, **kw: x,
            ),
        ):
            cpu.cp_flashmask_allgatherkv_balance_backward(
                dummy,
                dummy,
                dummy,
                None,  # startend_row_indices
                dummy,
                paddle.randn([B, H, S]),
                dummy,
                None,
                MagicMock(),
                False,
                4,
                None,
            )

        self.assertIsNone(mock_bwd.call_args[0][6])


# ---------------------------------------------------------------------------
# 2) FlashMaskAttnFunctor.backward  (refined_recompute.flash_attn)
# ---------------------------------------------------------------------------


class TestFlashMaskAttnFunctorBackwardDispatch(unittest.TestCase):
    """fa_version 3 and 4 both reach ``_flash_attn_bwd`` in the functor."""

    def _call_backward(self, fa_version):
        from paddlefleet.refined_recompute import flash_attn as fa

        B, S, H, D = 1, 8, 2, 16
        q = paddle.randn([B, S, H, D])
        k = paddle.randn([B, S, H, D])
        v = paddle.randn([B, S, H, D])
        indices = paddle.zeros([B, 2, S], dtype="int64")
        out = paddle.randn([B, S, H, D])
        lse = paddle.randn([B, H, S])
        grad = paddle.randn([B, S, H, D])
        dummy = paddle.randn([B, S, H, D])

        out._clear_dataptr = MagicMock()
        lse._clear_dataptr = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.fa_version = fa_version
        mock_ctx.sink_requires_grad = False
        mock_ctx.softmax_scale = None
        # The cutedsl branch unpacks 8 items -- learnable_sink is now saved too.
        mock_ctx.saved_tensor.return_value = (
            q,
            k,
            v,
            indices,
            out,
            lse,
            False,
            None,
        )

        mock_bwd = MagicMock(return_value=(dummy, dummy, dummy, None))
        mock_info = MagicMock(name="FlashMaskInfoPaddle")

        with (
            patch.object(fa, "_flash_attn_bwd", mock_bwd, create=True),
            patch.object(fa, "FlashMaskInfoPaddle", mock_info, create=True),
        ):
            grads = fa.FlashMaskAttnFunctor.backward(mock_ctx, grad)

        return mock_bwd, mock_info, grads

    def test_cutedsl_backward_dispatch(self):
        for fa_version in _CUTEDSL_VERSIONS:
            with self.subTest(fa_version=fa_version):
                mock_bwd, mock_info, grads = self._call_backward(fa_version)

                mock_bwd.assert_called_once()
                args, kwargs = mock_bwd.call_args
                # q, k, v, result_attention, grad, softmax_lse, flashmask_info
                self.assertEqual(len(args), 7)
                self.assertIs(args[6], mock_info.return_value)
                self.assertIsNone(kwargs["learnable_sink"])
                self.assertIsNone(kwargs["softmax_scale"])
                self.assertIn("deterministic", kwargs)
                # sink_requires_grad is False -> only q/k/v grads returned.
                self.assertEqual(len(grads), 3)


# ---------------------------------------------------------------------------
# 3) RefinedRcomputeFlashMaskAttention._first_fwd  (refined_recompute.flash_attn)
# ---------------------------------------------------------------------------


class TestRefinedRecomputeFirstFwdDispatch(unittest.TestCase):
    """fa_version 3 and 4 both reach ``_flash_attn_fwd`` in ``_first_fwd``."""

    def _call_forward(self, fa_version):
        from paddlefleet.refined_recompute import flash_attn as fa

        B, S, H, D = 1, 8, 2, 16
        q = paddle.randn([B, S, H, D], dtype=paddle.bfloat16)
        k = paddle.randn([B, S, H, D], dtype=paddle.bfloat16)
        v = paddle.randn([B, S, H, D], dtype=paddle.bfloat16)
        startend = paddle.zeros([B, 2, S], dtype="int64")

        mock_fwd = MagicMock(
            return_value=(
                paddle.randn([B, S, H, D], dtype=paddle.bfloat16),
                paddle.randn([B, H, S], dtype=paddle.float32),
            )
        )

        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False

        with (
            patch.object(
                fa.framework, "_dygraph_tracer", return_value=mock_tracer_obj
            ),
            patch.object(fa, "get_fa_version", return_value=fa_version),
            patch.object(fa, "_flash_attn_fwd", mock_fwd, create=True),
        ):
            obj = fa.RefinedRcomputeFlashMaskAttention()
            obj.forward(q, k, v, startend, causal=False)
            hold = obj._hold_tensors_queue.get()

        return mock_fwd, hold

    def test_cutedsl_forward_dispatch(self):
        for fa_version in _CUTEDSL_VERSIONS:
            with self.subTest(fa_version=fa_version):
                mock_fwd, hold = self._call_forward(fa_version)

                mock_fwd.assert_called_once()
                args, kwargs = mock_fwd.call_args
                # q, k, v positionally, everything else by keyword.
                self.assertEqual(len(args), 3)
                self.assertFalse(kwargs["causal"])
                self.assertTrue(kwargs["return_lse"])
                self.assertFalse(kwargs["pack_gqa"])
                self.assertIsNone(kwargs["learnable_sink"])
                # The cutedsl branch stores sink/scale instead of seed_offset.
                self.assertIn("learnable_sink", hold)
                self.assertIn("softmax_scale", hold)
                self.assertNotIn("seed_offset", hold)


if __name__ == "__main__":
    unittest.main()
