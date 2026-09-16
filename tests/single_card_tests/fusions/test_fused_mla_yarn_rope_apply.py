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

"""CPU-observable behavior tests for ``fused_mla_yarn_rope_apply``.

Only pure-Python / control-flow logic that can be exercised without a GPU is
asserted here:

  * ``_get_block_h`` launch-geometry math (per-``nheads`` BLOCK_H selection),
  * the head-block grid dimension derived from BLOCK_H (``nheads // BLOCK_H``),
  * argument-validation guards in the two ``PyLayer.forward`` entry points that
    raise ``AssertionError`` *before* any Triton kernel is launched.

The four Triton kernels (``rotary_fwd_q_kernel`` / ``rotary_bwd_q_kernel`` /
``rotary_fwd_kv_kernel`` / ``rotary_bwd_kv_kernel``) implement the actual YaRN
RoPE numerics and require a real GPU to execute. Their numeric correctness is
NOT covered here and must be verified by a single-card GPU test against an
independent reference. ``cos`` / ``sin`` are pre-computed inputs to this module,
so there is no in-module YaRN scaling-factor / mscale computation to assert.

Every expected value below is hand-derived. ``_get_block_h`` picks the largest
power-of-two that divides ``nheads``, capped at 128, starting the search from
``min(128, next_power_of_2(nheads))`` and halving until it divides ``nheads``:

    nheads=8  -> npo2=8,   start 8,  8%8==0            -> 8
    nheads=7  -> npo2=8,   8->4->2->1 (odd)            -> 1
    nheads=12 -> npo2=16,  start 16, 16->8->4 (12%4=0) -> 4
    nheads=96 -> npo2=128, start 128,128->64->32       -> 32
    nheads=192-> npo2=256, start 128,128->64 (192%64=0)-> 64
    nheads=256-> npo2=256, start 128, 256%128==0       -> 128
"""

import unittest

try:
    import paddle

    from paddlefleet.triton_ops.fused_mla_yarn_rope_apply import (
        _get_block_h,
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest: paddle and/or triton not installed
    _IMPORT_ERROR = str(exc)


# Hand-derived (nheads -> expected BLOCK_H) pairs, covering powers of two,
# odd counts, and non-power-of-two even counts on both sides of the 128 cap.
_EXPECTED_BLOCK_H = {
    1: 1,
    2: 2,
    3: 1,
    4: 4,
    5: 1,
    6: 2,
    7: 1,
    8: 8,
    9: 1,
    10: 2,
    12: 4,
    16: 16,
    24: 8,
    32: 32,
    40: 8,
    48: 16,
    64: 64,
    96: 32,
    128: 128,
    160: 32,
    192: 64,
    256: 128,
    384: 128,
}

# Hand-derived (nheads -> expected head-block count) = nheads // BLOCK_H.
# Because BLOCK_H always divides nheads, the grid's second axis
# triton.cdiv(nheads, BLOCK_H) reduces to exact integer division.
_EXPECTED_HEAD_BLOCKS = {
    8: 1,
    7: 7,
    6: 3,
    12: 3,
    96: 3,
    160: 5,
    192: 3,
    256: 2,
    384: 3,
}


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/triton import failed: {_IMPORT_ERROR}",
)
class TestGetBlockH(unittest.TestCase):
    """Pure-Python launch-geometry math; no GPU required."""

    def test_exact_block_h_values(self):
        for nheads, expected in _EXPECTED_BLOCK_H.items():
            self.assertEqual(
                _get_block_h(nheads),
                expected,
                msg=f"_get_block_h({nheads}) should be {expected}",
            )

    def test_head_block_grid_dimension(self):
        # Mirrors grid = (total_seqlen, triton.cdiv(nheads, BLOCK_H)) in
        # ApplyMLARotaryEmbQ/KV.forward. Since BLOCK_H divides nheads exactly,
        # the number of head-blocks is nheads // BLOCK_H with no partial block.
        for nheads, expected_blocks in _EXPECTED_HEAD_BLOCKS.items():
            block_h = _get_block_h(nheads)
            self.assertEqual(nheads % block_h, 0)
            self.assertEqual(
                nheads // block_h,
                expected_blocks,
                msg=f"head-block count for nheads={nheads}",
            )

    def test_invariants_hold_across_range(self):
        # The forward pass asserts `nheads % BLOCK_H == 0`; BLOCK_H must also be
        # a power of two and never exceed the 128 cap. Verify across a dense
        # range so a regression in the halving loop is caught.
        for nheads in range(1, 257):
            block_h = _get_block_h(nheads)
            self.assertGreaterEqual(block_h, 1)
            self.assertLessEqual(block_h, 128)
            self.assertEqual(
                block_h & (block_h - 1), 0, msg=f"{block_h} not power of two"
            )
            self.assertEqual(
                nheads % block_h, 0, msg=f"BLOCK_H must divide nheads={nheads}"
            )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/triton import failed: {_IMPORT_ERROR}",
)
class TestForwardArgGuards(unittest.TestCase):
    """Argument-validation guards that fire before any kernel launch.

    These asserts run on the host in ApplyMLARotaryEmbQ/KV.forward before the
    Triton grid is launched, so they are observable on CPU-only tensors.
    """

    def setUp(self):
        # Force CPU so no GPU is required; restore the caller's device even on
        # assertion failure.
        self._orig_device = paddle.device.get_device()
        self.addCleanup(paddle.set_device, self._orig_device)
        paddle.set_device("cpu")

    def _q_inputs(self):
        qk_head_dim, emb_dim = 128, 64
        q = paddle.zeros([1, 2, 4, qk_head_dim + emb_dim], dtype="float32")
        cos = paddle.zeros([2, 1, 1, emb_dim], dtype="float32")
        sin = paddle.zeros([2, 1, 1, emb_dim], dtype="float32")
        return q, cos, sin, qk_head_dim, emb_dim

    def _kv_inputs(self):
        k_dim, v_dim, emb_dim = 128, 128, 64
        kv = paddle.zeros([1, 2, 4, k_dim + v_dim], dtype="float32")
        k_pos_emb = paddle.zeros([1, 2, 1, emb_dim], dtype="float32")
        cos = paddle.zeros([2, 1, 1, emb_dim], dtype="float32")
        sin = paddle.zeros([2, 1, 1, emb_dim], dtype="float32")
        return kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim

    def test_q_rejects_rotary_interleaved(self):
        # `assert not rotary_interleaved` is the first statement of forward;
        # it must reject the unsupported interleaved layout.
        q, cos, sin, qk_head_dim, emb_dim = self._q_inputs()
        with self.assertRaises(AssertionError):
            fused_apply_mla_rope_for_q(
                q,
                cos,
                sin,
                qk_head_dim,
                emb_dim,
                None,  # cu_seqlens_q
                0,  # cp_rank
                1,  # cp_size
                True,  # rotary_interleaved -> rejected
            )

    def test_q_rejects_thd_cu_seqlens(self):
        # THD is explicitly unsupported: passing a non-None cu_seqlens_q must
        # raise with the documented message, before any kernel launch.
        q, cos, sin, qk_head_dim, emb_dim = self._q_inputs()
        cu_seqlens_q = paddle.to_tensor([0, 2], dtype="int32")
        with self.assertRaises(AssertionError) as cm:
            fused_apply_mla_rope_for_q(
                q,
                cos,
                sin,
                qk_head_dim,
                emb_dim,
                cu_seqlens_q,
                0,
                1,
                False,
            )
        self.assertIn("THD is not supported", str(cm.exception))

    def test_kv_rejects_rotary_interleaved(self):
        # Same first-line guard in ApplyMLARotaryEmbKV.forward.
        kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim = self._kv_inputs()
        with self.assertRaises(AssertionError):
            fused_apply_mla_rope_for_kv(
                kv,
                k_pos_emb,
                cos,
                sin,
                emb_dim,
                k_dim,
                v_dim,
                None,  # cu_seqlens_kv
                0,
                1,
                True,  # rotary_interleaved -> rejected
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
