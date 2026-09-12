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

"""Coverage for the HCA index compaction added to ``csa_attention``:

* ``CSADocMaskMetadata.compact_attn_topk_idxs`` -- the once-per-batch,
  width-keyed densify cache shared across same-ratio HCA layers.
* ``CompressedSparseAttention.compressed_sparse_attn`` -- the arch/indexer
  gate that routes the no-indexer cuDNN path through that cache.
"""

import types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    CSADocMaskMetadata,
)


class TestCompactAttnTopkIdxsCache(unittest.TestCase):
    def _meta(self):
        # Bypass ``build`` (Triton/document-bound); only the cache dict matters.
        return CSADocMaskMetadata.__new__(CSADocMaskMetadata)

    def test_densifies_and_counts(self):
        meta = self._meta()
        # Interior + trailing -1 holes; order-preserving densify moves valid
        # entries to a contiguous prefix and reports the exact per-row count.
        topk = paddle.to_tensor(
            [[0, -1, 2, -1, 5, -1], [1, 3, -1, -1, -1, -1]], dtype="int32"
        )
        compact, lengths = meta.compact_attn_topk_idxs(topk)
        self.assertEqual(
            compact.tolist(),
            [[0, 2, 5, -1, -1, -1], [1, 3, -1, -1, -1, -1]],
        )
        self.assertEqual(lengths.tolist(), [3, 2])
        self.assertEqual(lengths.dtype, paddle.int32)

    def test_cache_hit_returns_same_object_per_width(self):
        meta = self._meta()
        topk = paddle.to_tensor([[0, -1, 2, -1, 5, -1]], dtype="int32")
        first = meta.compact_attn_topk_idxs(topk)
        # Same row width -> served from the cache (no recompute), identical tuple.
        second = meta.compact_attn_topk_idxs(topk)
        self.assertIs(first, second)


class TestCompressedSparseAttnGate(unittest.TestCase):
    """The gate must compact via the shared cache exactly on the no-indexer,
    ``topk_length is None``, cuDNN, hole-honouring-arch path."""

    def _run(self, *, indexer, backend, honours, docmask_meta, topk_length):
        b, sq, np_heads, hn = 1, 2, 2, 8
        query = paddle.randn([b, sq, np_heads, hn])
        kv_full = paddle.randn([b, 16, hn])
        attn_sink = paddle.randn([np_heads])
        topk_idxs = paddle.to_tensor([[[0, -1, 2, -1]] * sq], dtype="int32")

        seen = {}

        def fake_csa_sparse_attn(
            q, kv, sink, idxs, scale, *, backend, topk_length, **kw
        ):
            seen["topk_idxs"] = idxs
            seen["topk_length"] = topk_length
            return paddle.zeros_like(q).reshape([b, sq, np_heads * hn])

        compacted = (
            paddle.to_tensor([[[0, 2, -1, -1]] * sq], dtype="int32"),
            paddle.to_tensor([[2, 2]], dtype="int32"),
        )
        meta = (
            types.SimpleNamespace(compact_attn_topk_idxs=lambda t: compacted)
            if docmask_meta
            else None
        )
        self_ns = types.SimpleNamespace(
            config=types.SimpleNamespace(csa_sparse_attn_backend=backend),
            indexer=indexer,
            global_kv_idx_remap_fusion=False,
        )
        with (
            patch(
                "paddlefleet.fusions.csa_sparse_attn.csa_sparse_attn",
                fake_csa_sparse_attn,
            ),
            patch(
                "paddlefleet.fusions.csa_sparse_attn."
                "_csa_bwd_honours_topk_length_holes",
                lambda: honours,
            ),
        ):
            CompressedSparseAttention.compressed_sparse_attn(
                self_ns,
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                0.125,
                topk_length=topk_length,
                indexer_topk=0,
                docmask_meta=meta,
            )
        return seen, compacted

    def test_gate_compacts_on_hca_cudnn_sm100(self):
        seen, compacted = self._run(
            indexer=None,
            backend="cudnn",
            honours=True,
            docmask_meta=True,
            topk_length=None,
        )
        # Gate fired: the cache's compacted idxs + count reached the kernel.
        self.assertIs(seen["topk_length"], compacted[1])
        self.assertIs(seen["topk_idxs"], compacted[0])

    def test_gate_skipped_when_indexer_present(self):
        seen, _ = self._run(
            indexer=object(),
            backend="cudnn",
            honours=True,
            docmask_meta=True,
            topk_length=None,
        )
        # Indexer layer must not reuse the width-keyed cache.
        self.assertIsNone(seen["topk_length"])

    def test_gate_skipped_on_sm90(self):
        seen, _ = self._run(
            indexer=None,
            backend="cudnn",
            honours=False,
            docmask_meta=True,
            topk_length=None,
        )
        self.assertIsNone(seen["topk_length"])


if __name__ == "__main__":
    unittest.main()
