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
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import paddle

from paddlefleet.refined_recompute import flash_attn as fa


class FakeCtx:
    def save_for_backward(self, *tensors):
        self._saved = tensors

    def saved_tensor(self):
        return self._saved


class CpGroup:
    rank = 1
    nranks = 2
    world_size = 2


class TestFlashMaskAttnCpFunctor(unittest.TestCase):
    def test_backward_passes_mode_and_scale(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        hold = {
            "mode": "contiguous_allgather",
            "result_attention": out,
            "softmax_lse": lse,
            "startend_row_indices": startend,
            "fa_version": 4,
            "group": CpGroup(),
            "causal": False,
            "softmax_scale": 0.25,
        }
        ctx = FakeCtx()

        self.assertIs(
            fa.FlashMaskAttnCpFunctor.forward(ctx, q, k, v, None, hold), out
        )
        with patch.object(
            fa,
            "cp_flashmask_allgatherkv_balance_backward",
            return_value=(q, k, v, None),
        ) as mock_backward:
            grads = fa.FlashMaskAttnCpFunctor.backward(
                ctx, paddle.ones_like(out)
            )

        self.assertIs(grads[0], q)
        self.assertIs(grads[1], k)
        self.assertIs(grads[2], v)
        self.assertEqual(mock_backward.call_args.args[-2], 0.25)
        self.assertEqual(
            mock_backward.call_args.args[-1], "contiguous_allgather"
        )


class TestFlashMaskAttnFunctor(unittest.TestCase):
    def test_v3_backward_passes_scalar_softmax_scale(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        hold = {
            "result_attention": out,
            "softmax_lse": lse,
            "causal": False,
            "softmax_scale": None,
        }
        ctx = FakeCtx()

        def fake_flashmask_attention(query, key, value, block_mask=None):
            return None

        with patch.object(fa, "_get_fa_version", return_value=3):
            self.assertIs(
                fa.FlashMaskAttnFunctor.forward(
                    ctx, q, k, v, startend, None, hold
                ),
                out,
            )
        with (
            patch.object(fa, "flashmask_attention", fake_flashmask_attention),
            patch.object(fa, "_C_ops") as mock_c_ops,
        ):
            mock_c_ops.flashmask_attention_v2_grad.return_value = (q, k, v)
            grads = fa.FlashMaskAttnFunctor.backward(ctx, paddle.ones_like(out))

        self.assertIs(grads[0], q)
        scale_arg = mock_c_ops.flashmask_attention_v2_grad.call_args.args[-2]
        self.assertIsInstance(scale_arg, float)
        self.assertNotIsInstance(scale_arg, tuple)


class TestFlashMaskSwaP2PFunctor(unittest.TestCase):
    def test_forward_returns_saved_output_and_backward_uses_saved_tensors(self):
        q = paddle.randn([1, 4, 2, 4])
        k = paddle.randn([1, 4, 2, 4])
        v = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        hold = {
            "result_attention": out,
            "softmax_lse": lse,
            "recv_key": paddle.randn([1, 2, 2, 4]),
            "recv_value": paddle.randn([1, 2, 2, 4]),
            "startend_row_indices": paddle.zeros([1, 1, 4, 2], dtype="int32"),
            "group": CpGroup(),
            "causal": False,
            "softmax_scale": 0.5,
            "window_size": 2,
        }
        ctx = FakeCtx()

        self.assertIs(
            fa.FlashMaskSwaP2PFunctor.forward(ctx, q, k, v, None, hold), out
        )
        with patch.object(
            fa,
            "cp_flashmask_swa_p2p_backward",
            return_value=(q, k, v, None),
        ) as mock_backward:
            grads = fa.FlashMaskSwaP2PFunctor.backward(
                ctx, paddle.ones_like(out)
            )

        self.assertIs(grads[0], q)
        self.assertIs(grads[1], k)
        self.assertIs(grads[2], v)
        mock_backward.assert_called_once()
        self.assertIs(mock_backward.call_args.args[0], q)
        self.assertIs(mock_backward.call_args.args[3], hold["recv_key"])
        self.assertEqual(mock_backward.call_args.args[-1], 2)


class TestUlyssesHelpers(unittest.TestCase):
    def test_slice_ulysses_mask_heads_broadcast_and_per_head(self):
        broadcast = paddle.zeros([1, 1, 4, 2], dtype="int32")
        self.assertIs(
            fa.slice_ulysses_mask_heads(broadcast, 4, CpGroup()), broadcast
        )

        per_head = paddle.arange(1 * 4 * 4 * 2, dtype="int32").reshape(
            [1, 4, 4, 2]
        )
        sliced = fa.slice_ulysses_mask_heads(per_head, 4, CpGroup())
        self.assertEqual(list(sliced.shape), [1, 2, 4, 2])
        self.assertTrue(
            bool(paddle.all(sliced == per_head[:, 2:4, :, :]).item())
        )

    def test_ulysses_local_first_forward_version_3_saves_rr_tensors(self):
        q = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")

        def fake_flashmask_attention(query, key, value, block_mask=None):
            return None

        with (
            patch.object(fa, "_get_fa_version", return_value=3),
            patch.object(fa, "flashmask_attention", fake_flashmask_attention),
            patch.object(fa, "_C_ops") as mock_c_ops,
        ):
            mock_c_ops.flashmask_attention_v2.return_value = (out, lse)
            result, hold = fa.ulysses_local_flashmask_first_fwd(
                q, q, q, startend, False, None
            )

        self.assertIs(result, out)
        self.assertIs(hold["softmax_lse"], lse)
        self.assertFalse(hold["causal"])
        scale_arg = mock_c_ops.flashmask_attention_v2.call_args.args[-2]
        self.assertIsInstance(scale_arg, float)

    def test_ulysses_local_first_forward_version_4_saves_rr_tensors(self):
        q = paddle.randn([1, 4, 2, 4])
        out = paddle.randn([1, 4, 2, 4])
        lse = paddle.randn([1, 2, 4])
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")

        with (
            patch.object(fa, "_get_fa_version", return_value=4),
            patch.object(
                fa, "_flash_attn_fwd", return_value=(out, lse)
            ) as mock_fwd,
        ):
            result, hold = fa.ulysses_local_flashmask_first_fwd(
                q, q, q, startend, False, None
            )

        self.assertIs(result, out)
        self.assertIs(hold["result_attention"], out)
        self.assertIs(hold["softmax_lse"], lse)
        self.assertFalse(hold["causal"])
        mock_fwd.assert_called_once()
        self.assertEqual(mock_fwd.call_args.args[0].dtype, q.dtype)
        self.assertIs(
            mock_fwd.call_args.kwargs["startend_row_indices"], startend
        )
        self.assertFalse(mock_fwd.call_args.kwargs["causal"])


class TestRefinedRcomputeFlashMaskCpAttentionModes(unittest.TestCase):
    def setUp(self):
        self.q = paddle.randn([1, 4, 2, 4])
        self.k = paddle.randn([1, 4, 2, 4])
        self.v = paddle.randn([1, 4, 2, 4])
        self.startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        self.out = paddle.randn([1, 4, 2, 4])
        self.lse = paddle.randn([1, 2, 4])
        self.group = CpGroup()

    def _patch_group(self):
        hcg = MagicMock()
        hcg.get_context_parallel_group.return_value = self.group
        return patch.object(
            fa.fleet, "get_hybrid_communicate_group", return_value=hcg
        )

    def test_allgather_first_forward_stores_mode_for_backward(self):
        with (
            self._patch_group(),
            patch.object(
                fa,
                "cp_flashmask_allgatherkv_balance_forward",
                return_value=(self.out, self.lse, self.startend, 4),
            ) as mock_forward,
        ):
            attn = fa.RefinedRcomputeFlashMaskCpAttention()
            result = attn._first_fwd(
                self.q,
                self.k,
                self.v,
                self.startend,
                mode="contiguous_allgather",
            )

        self.assertIs(result, self.out)
        self.assertEqual(
            mock_forward.call_args.args[-1], "contiguous_allgather"
        )
        hold = attn._hold_tensors_queue.get_nowait()
        self.assertEqual(hold["mode"], "contiguous_allgather")
        self.assertEqual(hold["fa_version"], 4)

    def test_p2p_first_forward_defaults_window_and_saves_recv_tensors(self):
        recv_k = paddle.randn([1, 2, 2, 4])
        recv_v = paddle.randn([1, 2, 2, 4])
        with (
            self._patch_group(),
            patch.object(fa, "_flash_mask_available", True),
            patch.object(
                fa,
                "cp_flashmask_swa_p2p_forward",
                return_value=(
                    self.out,
                    self.lse,
                    recv_k,
                    recv_v,
                    self.startend,
                ),
            ) as mock_forward,
        ):
            attn = fa.RefinedRcomputeFlashMaskCpAttention()
            result = attn._first_fwd(
                self.q, self.k, self.v, self.startend, mode="contiguous_swap2p"
            )

        self.assertIs(result, self.out)
        self.assertEqual(mock_forward.call_args.args[-1], 128)
        hold = attn._hold_tensors_queue.get_nowait()
        self.assertEqual(hold["mode"], "contiguous_swap2p")
        self.assertIs(hold["recv_key"], recv_k)
        self.assertEqual(hold["window_size"], 128)

    def test_p2p_rejects_invalid_window(self):
        with (
            self._patch_group(),
            patch.object(fa, "_flash_mask_available", True),
        ):
            attn = fa.RefinedRcomputeFlashMaskCpAttention()
            with self.assertRaises(ValueError):
                attn._first_fwd(
                    self.q,
                    self.k,
                    self.v,
                    self.startend,
                    mode="contiguous_swap2p",
                    window_size=0,
                )

    def test_ulysses_first_forward_allows_causal_and_odd_local_seq(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        odd_q = self.q[:, :3]
        with (
            self._patch_group(),
            patch.object(
                attn, "_ulysses_first_fwd", return_value=self.out
            ) as mock_ulysses,
        ):
            result = attn._first_fwd(
                odd_q,
                self.k,
                self.v,
                self.startend,
                mode="contiguous_a2a",
                causal=True,
            )

        self.assertIs(result, self.out)
        mock_ulysses.assert_called_once()
        self.assertIs(mock_ulysses.call_args.args[0], odd_q)
        self.assertTrue(mock_ulysses.call_args.args[5])

    def test_ulysses_first_and_second_forward_use_surrogate(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        with (
            patch.object(
                attn,
                "_ulysses_alltoall_qkv",
                side_effect=lambda q, k, v, g: (q, k, v),
            ),
            patch.object(
                attn, "_ulysses_alltoall_output", side_effect=lambda x, g: x
            ),
            patch.object(
                fa,
                "ulysses_local_flashmask_first_fwd",
                return_value=(self.out, {"result_attention": self.out}),
            ),
        ):
            result = attn._ulysses_first_fwd(
                self.q,
                self.k,
                self.v,
                self.startend,
                self.group,
                False,
                None,
                None,
            )
        self.assertIs(result, self.out)
        hold = attn._hold_tensors_queue.get_nowait()
        self.assertEqual(hold["mode"], "contiguous_a2a")

        with (
            patch.object(
                attn,
                "_ulysses_alltoall_qkv",
                side_effect=lambda q, k, v, g: (q, k, v),
            ),
            patch.object(
                attn, "_ulysses_alltoall_output", side_effect=lambda x, g: x
            ),
            patch.object(
                fa.FlashMaskAttnFunctor, "apply", return_value=self.out
            ) as mock_apply,
        ):
            second = attn._ulysses_second_fwd(self.q, self.k, self.v, hold)
        self.assertIs(second, self.out)
        mock_apply.assert_called_once()

    def test_second_forward_dispatches_each_mode(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        for mode, target in (
            ("dualchunk_allgather", fa.FlashMaskAttnCpFunctor),
            ("contiguous_allgather", fa.FlashMaskAttnCpFunctor),
            ("contiguous_swap2p", fa.FlashMaskSwaP2PFunctor),
        ):
            attn._hold_tensors_queue.put({"mode": mode})
            with patch.object(
                target, "apply", return_value=self.out
            ) as mock_apply:
                self.assertIs(
                    attn._second_fwd(self.q, self.k, self.v), self.out
                )
                mock_apply.assert_called_once()

    def test_validation_errors_are_preserved(self):
        attn = fa.RefinedRcomputeFlashMaskCpAttention()
        with self.assertRaises(NotImplementedError):
            attn._first_fwd(self.q, self.k, self.v, self.startend, causal=True)
        with self.assertRaises(NotImplementedError):
            attn._first_fwd(self.q, self.k, self.v, self.startend, dropout=0.1)
        with self._patch_group(), self.assertRaises(AssertionError):
            attn._first_fwd(self.q[:, :3], self.k, self.v, self.startend)


if __name__ == "__main__":
    unittest.main()
