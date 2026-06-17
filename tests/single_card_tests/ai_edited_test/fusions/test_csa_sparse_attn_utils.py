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

"""Unit tests for CSA sparse-attn shared helpers and unfused/dispatch paths.

These cover the pure-Python / pure-Paddle code paths that do not depend on the
FlashMLA or cuDNN custom ops: input validation, local->global index
conversion, the unfused einsum reference attention, backend dispatch, the
FlashMLA wrapper's arch-alignment / unavailable-fallback helpers, and the
``tilelang_ops`` lazy re-export of ``csa_sparse_attn``.
"""

import unittest

import paddle

try:
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu:0")
    from paddlefleet.fusions import csa_sparse_attn_utils

    _IMPORT_OK = paddle.is_compiled_with_cuda()
except Exception:  # pragma: no cover - import guard for non-GPU collection
    _IMPORT_OK = False


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestPrepareInputs(unittest.TestCase):
    def _inputs(self):
        q = paddle.randn([2, 3, 4, 8], dtype="float32")
        kv = paddle.randn([2, 6, 8], dtype="float32")
        sink = paddle.randn([4], dtype="float32")
        idx = paddle.randint(0, 6, [2, 3, 5]).cast("int32")
        return q, kv, sink, idx

    def test_rejects_bad_query_rank(self):
        q, kv, sink, idx = self._inputs()
        with self.assertRaisesRegex(ValueError, "q must have shape"):
            csa_sparse_attn_utils.prepare_inputs(
                q.reshape([2, 3, 4 * 8]), kv, sink, idx
            )

    def test_rejects_bad_kv_rank(self):
        q, kv, sink, idx = self._inputs()
        with self.assertRaisesRegex(ValueError, "kv must have shape"):
            csa_sparse_attn_utils.prepare_inputs(
                q, kv.reshape([2, 6 * 8]), sink, idx
            )

    def test_rejects_bad_topk_rank(self):
        q, kv, sink, idx = self._inputs()
        with self.assertRaisesRegex(ValueError, "topk_idxs must have shape"):
            csa_sparse_attn_utils.prepare_inputs(
                q, kv, sink, idx.reshape([2 * 3, 5])
            )

    def test_casts_dtypes(self):
        q, kv, sink, idx = self._inputs()
        _, _, sink_out, idx_out = csa_sparse_attn_utils.prepare_inputs(
            q, kv, sink.cast("bfloat16"), idx.cast("int64")
        )
        self.assertEqual(idx_out.dtype, paddle.int32)
        self.assertEqual(sink_out.dtype, paddle.float32)

    def test_passthrough_when_dtypes_already_match(self):
        q, kv, sink, idx = self._inputs()
        q_out, kv_out, sink_out, idx_out = csa_sparse_attn_utils.prepare_inputs(
            q, kv, sink, idx
        )
        self.assertIs(q_out, q)
        self.assertIs(kv_out, kv)
        self.assertIs(sink_out, sink)
        self.assertIs(idx_out, idx)
