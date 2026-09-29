# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for the dense DSA indexer query-head padding helpers.

Module under test: ``paddlefleet.cudnn_ops.indexer.dense_indexer_kl_cudnn``
(repository module map: "计算优化" / cuDNN Fused Ops). The dense score kernels
tile their MMA ``M`` on the query-head count, so the wrapper first widens (or
rejects) that count to a value the running device can express. This file
exercises only that *device-independent* control logic:

  * ``_dense_score_qheads`` -- narrowest tileable width per arch. On SM100+ the
    ``tcgen05.copy.Repetition`` enum admits only powers of two, floored at 16;
    on SM90 the kernel tiles by 64 and only needs a floor of 2.
  * ``_require_dense_score_qheads`` -- turns an untileable width into a named
    ``ValueError`` at the call site rather than a CuTe-trace failure.
  * ``_pad_attn_kl_heads`` -- widens ``query`` with zero heads and ``lse`` with
    ``+inf`` heads, leaving the real heads byte-for-byte intact.

The device capability is the only real hardware query these helpers make; it is
a genuine not-under-test collaborator, so it is patched to a fixed ``(major, 0)``
to drive both arch tables regardless of the card the suite runs on. The size
math and padding fill are then real code under test, evaluated on CPU tensors.
Every expected value is hand-derived from the documented tiling rules, never
read back from the production helper.

The cuDNN score/backward kernels themselves need a GPU and a cuDNN-frontend
build; they are NOT driven here. When ``paddle`` (and ``paddlefleet``) cannot be
imported the whole suite skips honestly rather than fake-passing on CPU.
"""

import unittest

try:
    import numpy as np
    import paddle

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # missing dependency -> honest skip, not swallowed
    np = None
    paddle = None
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)

if _IMPORT_OK:
    # A genuine API change / compile error must surface; only ImportError is
    # treated as a "dependency absent" skip reason.
    try:
        from paddlefleet.cudnn_ops.indexer import dense_indexer_kl_cudnn as mod
    except ImportError as exc:
        _IMPORT_OK = False
        _IMPORT_ERR = str(exc)


def _as_arch(major):
    """Patch the device-capability query to a fixed ``(major, 0)``.

    The capability read is a hardware collaborator, not the logic under test;
    fixing it lets both arch branches be exercised on any card.
    """
    from unittest.mock import patch

    return patch.object(
        paddle.device.cuda, "get_device_capability", lambda: (major, 0)
    )


@unittest.skipUnless(
    _IMPORT_OK, f"paddle/paddlefleet not importable: {_IMPORT_ERR}"
)
class TestDenseScoreQheads(unittest.TestCase):
    """Hand-derived widths for ``_dense_score_qheads`` on each arch."""

    def test_sm100_rounds_up_to_power_of_two_floored_at_16(self):
        # exp2(ceil(log2(h))) with a floor of 16, hand-computed per entry.
        cases = {
            1: 16,
            8: 16,
            16: 16,
            24: 32,
            32: 32,
            40: 64,
            64: 64,
            128: 128,
            192: 256,
            256: 256,
        }
        with _as_arch(10):
            for heads, want in cases.items():
                self.assertEqual(mod._dense_score_qheads(heads), want)

    def test_sm90_keeps_up_to_64_then_rounds_to_a_multiple_of_64(self):
        # <=64: unchanged but floored at 2; >64: next multiple of 64.
        cases = {
            1: 2,
            2: 2,
            8: 8,
            16: 16,
            24: 24,
            64: 64,
            65: 128,
            96: 128,
            128: 128,
            192: 192,
        }
        with _as_arch(9):
            for heads, want in cases.items():
                self.assertEqual(mod._dense_score_qheads(heads), want)

    def test_the_two_arches_disagree_where_it_matters(self):
        # 192 is native on SM90 (three 64-tiles) but must widen to 256 on
        # SM100+; the branch is real, not a shared constant.
        with _as_arch(9):
            self.assertEqual(mod._dense_score_qheads(192), 192)
        with _as_arch(10):
            self.assertEqual(mod._dense_score_qheads(192), 256)


@unittest.skipUnless(
    _IMPORT_OK, f"paddle/paddlefleet not importable: {_IMPORT_ERR}"
)
class TestRequireDenseScoreQheads(unittest.TestCase):
    """``_require_dense_score_qheads`` accepts tileable widths, rejects others.

    The accept/reject sets are hand-listed from the tiling rules rather than
    derived from ``_dense_score_qheads`` (which would make the reference
    self-referential).
    """

    def test_sm100_accepts_only_powers_of_two_at_least_16(self):
        with _as_arch(10):
            for heads in (16, 32, 64, 128, 256):
                mod._require_dense_score_qheads(heads, "index_n_heads")
            for heads in (1, 8, 24, 40, 96, 192):
                with self.assertRaises(ValueError):
                    mod._require_dense_score_qheads(heads, "index_n_heads")

    def test_sm90_accepts_native_widths_including_192(self):
        with _as_arch(9):
            for heads in (2, 8, 24, 40, 64, 128, 192):
                mod._require_dense_score_qheads(heads, "index_n_heads")
            # 1 must widen to 2; 96 must widen to 128 -> both rejected as-is.
            for heads in (1, 96):
                with self.assertRaises(ValueError):
                    mod._require_dense_score_qheads(heads, "index_n_heads")

    def test_error_names_the_argument_and_both_widths(self):
        # h=24 on SM100+ needs 32: the message must carry the arg name and the
        # observed vs required counts so a misconfig is actionable.
        with _as_arch(10), self.assertRaises(ValueError) as ctx:
            mod._require_dense_score_qheads(24, "index_n_heads")
        msg = str(ctx.exception)
        self.assertIn("index_n_heads", msg)
        self.assertIn("24", msg)
        self.assertIn("32", msg)


@unittest.skipUnless(
    _IMPORT_OK, f"paddle/paddlefleet not importable: {_IMPORT_ERR}"
)
class TestPadAttnKlHeads(unittest.TestCase):
    """``_pad_attn_kl_heads`` fill and preservation, on CPU tensors."""

    def _query_lse(self, s_local, heads, head_dim):
        # Distinguishable, non-degenerate content so a misplaced slice or a
        # dropped head is observable.
        q = paddle.arange(s_local * heads * head_dim, dtype="float32").reshape(
            [s_local, heads, head_dim]
        )
        lse = (
            paddle.arange(s_local * heads, dtype="float32").reshape(
                [s_local, heads]
            )
            + 1.0
        )
        return q, lse

    def test_sm100_pads_24_heads_to_32_with_zeros_and_inf(self):
        s_local, heads, head_dim = 3, 24, 4
        q, lse = self._query_lse(s_local, heads, head_dim)
        with _as_arch(10):
            q_out, lse_out = mod._pad_attn_kl_heads(q, lse)

        self.assertEqual(list(q_out.shape), [s_local, 32, head_dim])
        self.assertEqual(list(lse_out.shape), [s_local, 32])
        # Real heads survive byte-for-byte.
        np.testing.assert_array_equal(q_out[:, :heads].numpy(), q.numpy())
        np.testing.assert_array_equal(lse_out[:, :heads].numpy(), lse.numpy())
        # Pad query heads are exactly zero.
        np.testing.assert_array_equal(
            q_out[:, heads:].numpy(),
            np.zeros([s_local, 32 - heads, head_dim], dtype=np.float32),
        )
        # Pad LSE heads are +inf (positive, infinite) so exp(0 - inf) = 0.
        pad_lse = lse_out[:, heads:].numpy()
        self.assertTrue(np.isinf(pad_lse).all())
        self.assertTrue((pad_lse > 0).all())

    def test_supported_width_is_returned_untouched(self):
        # h=64 is a supported SM100+ width: no copy, same objects back.
        q, lse = self._query_lse(2, 64, 4)
        with _as_arch(10):
            q_out, lse_out = mod._pad_attn_kl_heads(q, lse)
        self.assertIs(q_out, q)
        self.assertIs(lse_out, lse)

    def test_sm90_leaves_24_heads_unpadded(self):
        # 24 <= 64 tiles natively on SM90, so nothing widens.
        q, lse = self._query_lse(2, 24, 4)
        with _as_arch(9):
            q_out, lse_out = mod._pad_attn_kl_heads(q, lse)
        self.assertIs(q_out, q)
        self.assertIs(lse_out, lse)

    def test_sm90_pads_single_head_up_to_two(self):
        # The one SM90 width that must widen: h == 1 -> 2 (kernel asserts
        # qhead_per_kvhead > 1).
        s_local, head_dim = 2, 4
        q, lse = self._query_lse(s_local, 1, head_dim)
        with _as_arch(9):
            q_out, lse_out = mod._pad_attn_kl_heads(q, lse)
        self.assertEqual(list(q_out.shape), [s_local, 2, head_dim])
        self.assertEqual(list(lse_out.shape), [s_local, 2])
        np.testing.assert_array_equal(q_out[:, :1].numpy(), q.numpy())
        np.testing.assert_array_equal(lse_out[:, :1].numpy(), lse.numpy())
        np.testing.assert_array_equal(
            q_out[:, 1:].numpy(),
            np.zeros([s_local, 1, head_dim], dtype=np.float32),
        )
        pad_lse = lse_out[:, 1:].numpy()
        self.assertTrue(np.isinf(pad_lse).all())
        self.assertTrue((pad_lse > 0).all())


if __name__ == "__main__":
    unittest.main()
