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

"""Host-side shape guard for the cuDNN CSA indexer forward.

The cuDNN CSA indexer forward kernel does not reliably support short
compressed-KV shapes:

* ``S_k == 1`` (n_compressed == 1) crashes inside the CUDA kernel with
  ``cudaErrorIllegalInstruction`` (715) regardless of ``S_q`` — verified by a
  per-process sweep: ``S_k == 1`` always crashes, ``S_k >= 2`` is stable.
* the ratio-causal kernel requires ``S_q <= S_k * ratio``.

``_check_cudnn_indexer_shape_support`` converts both into a readable
``ValueError`` so an unsupported short-sequence case fails clearly instead of
poisoning the CUDA context. These checks are pure host-side and need no GPU.
"""

import unittest

import paddle

from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
    _check_cudnn_indexer_shape_support,
)


def _qkw(sq, sk, h=64, d=128):
    index_q = paddle.zeros([1, sq, h, d], dtype="bfloat16")
    index_k = paddle.zeros([1, sk, d], dtype="bfloat16")
    weights = paddle.zeros([1, sq, h], dtype="bfloat16")
    return index_q, index_k, weights


class TestCudnnIndexerShapeGuard(unittest.TestCase):
    """_check_cudnn_indexer_shape_support: short-Sk and S_q>S_k*ratio guards."""

    def test_sk1_raises_value_error(self):
        # S_k == 1 (n_compressed == 1) crashes the CUDA kernel; guard rejects it.
        for sq in (1, 4, 5):
            iq, ik, w = _qkw(sq, 1)
            with self.assertRaises(ValueError) as cm:
                _check_cudnn_indexer_shape_support(iq, ik, ratio=4)
            self.assertIn("compressed KV length >= 2", str(cm.exception))

    def test_sq_gt_sk_ratio_raises_value_error(self):
        # S_q > S_k * ratio violates the ratio-causal contract.
        iq, ik, w = _qkw(9, 2)  # 9 > 2 * 4
        with self.assertRaises(ValueError) as cm:
            _check_cudnn_indexer_shape_support(iq, ik, ratio=4)
        self.assertIn("S_q <= S_k * ratio", str(cm.exception))

    def test_seq_offset_counts_toward_ratio_bound(self):
        # CP causal-only mode pads virtual query rows before the local chunk,
        # so S_q + seq_offset must still fit the global compressed length.
        iq, ik, w = _qkw(4, 2)  # 4 + 5 > 2 * 4
        with self.assertRaises(ValueError) as cm:
            _check_cudnn_indexer_shape_support(iq, ik, ratio=4, seq_offset=5)
        self.assertIn("seq_offset=5", str(cm.exception))

    def test_seq_offset_boundary_passes(self):
        iq, ik, w = _qkw(4, 2)  # 4 + 4 == 2 * 4
        _check_cudnn_indexer_shape_support(iq, ik, ratio=4, seq_offset=4)

    def test_sk1_checked_before_ratio(self):
        # When both would trip, the S_k>=2 message wins (checked first).
        iq, ik, w = _qkw(5, 1)  # S_k==1 and 5 > 1*4
        with self.assertRaises(ValueError) as cm:
            _check_cudnn_indexer_shape_support(iq, ik, ratio=4)
        self.assertIn("compressed KV length >= 2", str(cm.exception))

    def test_valid_shapes_pass(self):
        # S_k >= 2 and S_q <= S_k * ratio: no error.
        for sq, sk in [(8, 2), (4, 2), (16, 4), (1, 2), (32, 8)]:
            iq, ik, w = _qkw(sq, sk)
            _check_cudnn_indexer_shape_support(iq, ik, ratio=4)  # no raise

    def test_boundary_sq_equals_sk_ratio_passes(self):
        # S_q == S_k * ratio is the inclusive upper bound: allowed.
        iq, ik, w = _qkw(8, 2)  # 8 == 2 * 4
        _check_cudnn_indexer_shape_support(iq, ik, ratio=4)  # no raise

    def test_ratio_respected(self):
        # Same shapes, different ratio flips validity at the boundary.
        iq, ik, w = _qkw(8, 2)
        _check_cudnn_indexer_shape_support(iq, ik, ratio=4)  # 8 <= 8 ok
        with self.assertRaises(ValueError):
            _check_cudnn_indexer_shape_support(iq, ik, ratio=2)  # 8 > 4


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer forward requires Blackwell GPU (SM100)",
)
class TestCudnnIndexerForwardGuardIntegration(unittest.TestCase):
    """The guard fires through the public forward / topk_fwd entry points,
    turning the CUDA-715 crash into a clean ValueError without poisoning the
    CUDA context (a subsequent valid call still succeeds)."""

    def test_forward_rejects_sk1_then_valid_call_works(self):
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_forward,
        )

        iq, ik, w = _qkw(5, 1)
        with self.assertRaises(ValueError):
            cudnn_indexer_forward(iq, ik, w, ratio=4)

        # CUDA context must still be usable (guard ran before any kernel).
        iq2, ik2, w2 = _qkw(8, 2)
        scores = cudnn_indexer_forward(iq2, ik2, w2, ratio=4)
        paddle.device.synchronize()
        self.assertEqual(list(scores.shape), [1, 8, 2])

    def test_topk_fwd_rejects_sk1(self):
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        iq, ik, w = _qkw(5, 1)
        with self.assertRaises(ValueError):
            cudnn_indexer_topk_fwd(iq, ik, w, ratio=4, topk_effective=4)


if __name__ == "__main__":
    unittest.main()
