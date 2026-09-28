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

"""Behavior tests for the HCA index-compaction path of ``csa_attention``.

Two units under test, both derived from document bounds and reused across
same-ratio HCA layers:

* ``CSADocMaskMetadata.compact_attn_topk_idxs`` -- the once-per-batch,
  width-keyed densify cache. Contract: valid (``>= 0``) entries are moved to a
  contiguous prefix in their *original left-to-right order* (NOT value-sorted),
  ``-1`` trails, ``topk_length`` is the exact per-row valid count (``int32``),
  and repeated calls of the same row width are served from the cache without
  recomputation.
* ``CompressedSparseAttention.compressed_sparse_attn`` -- the gate that routes
  the no-indexer / ``topk_length is None`` / cuDNN / SM100 path through that
  cache, compacting before dispatch and flagging ``topk_idxs_compacted``; every
  other branch must forward the original (still holey) indices unchanged with
  ``topk_length`` untouched.

No GPU is required: the densify is pure CPU tensor algebra, and the only
arch-gated collaborator (``_csa_bwd_honours_topk_length_holes``, which reads the
CUDA compute capability) plus the kernel dispatch (``csa_sparse_attn``) are
genuine not-under-test collaborators that are mocked here. Expected values are
hand-derived, independent of the source coverage test.
"""

import types
import unittest
from unittest.mock import patch

try:
    import paddle

    from paddlefleet.fusions import csa_sparse_attn as csa_fusion
    from paddlefleet.transformer.csa_attention import (
        CompressedSparseAttention,
        CSADocMaskMetadata,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # no paddle/paddlefleet here
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}",
)
class TestCompactAttnTopkIdxs(unittest.TestCase):
    """Densify contract + per-width caching of ``compact_attn_topk_idxs``."""

    @staticmethod
    def _fresh_meta():
        # ``build`` is Triton/document bound; the compaction cache only touches
        # ``self._compacted_attn_topk`` (dataclass default ``None``), so a raw
        # instance exercises the real method without a full build.
        return CSADocMaskMetadata.__new__(CSADocMaskMetadata)

    def test_densify_preserves_order_counts_and_empty_rows(self):
        meta = self._fresh_meta()
        # Row 0 valids are DESCENDING (5, 2, 0): order-preserving densify keeps
        # that order, so this row rejects any value-sort regression. Row 1 has
        # leading + interior holes (a later doc's query). Row 2 is all holes.
        topk = paddle.to_tensor(
            [
                [5, -1, 2, -1, 0, -1],
                [-1, -1, 3, -1, 7, 9],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype="int32",
        )
        compact, lengths = meta.compact_attn_topk_idxs(topk)
        self.assertEqual(
            compact.tolist(),
            [
                [5, 2, 0, -1, -1, -1],
                [3, 7, 9, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
        )
        self.assertEqual(lengths.tolist(), [3, 3, 0])
        self.assertEqual(lengths.dtype, paddle.int32)

    def test_cache_computes_once_per_width_and_reuses_result(self):
        meta = self._fresh_meta()
        topk = paddle.to_tensor([[5, -1, 2, -1, 0, -1]], dtype="int32")
        real = csa_fusion._csa_compact_topk_idxs
        widths_seen = []

        def counting(t):  # wrap the real collaborator, just tally invocations
            widths_seen.append(int(t.shape[-1]))
            return real(t)

        with patch.object(
            csa_fusion, "_csa_compact_topk_idxs", side_effect=counting
        ):
            first = meta.compact_attn_topk_idxs(topk)
            second = meta.compact_attn_topk_idxs(topk)  # same width -> cache

        self.assertEqual(widths_seen, [6])  # computed exactly once, not twice
        self.assertIs(first, second)  # identical cached tuple handed back
        # The cached value is the REAL densify, not a stub echo.
        self.assertEqual(first[0].tolist(), [[5, 2, 0, -1, -1, -1]])
        self.assertEqual(first[1].tolist(), [3])

    def test_cache_is_keyed_on_row_width(self):
        meta = self._fresh_meta()
        narrow = paddle.to_tensor([[0, -1, 2, -1]], dtype="int32")  # width 4
        wide = paddle.to_tensor([[0, -1, 2, -1, 5, -1]], dtype="int32")  # 6
        a = meta.compact_attn_topk_idxs(narrow)
        b = meta.compact_attn_topk_idxs(wide)
        self.assertIsNot(a, b)  # distinct widths -> distinct cache entries
        self.assertEqual(a[0].tolist(), [[0, 2, -1, -1]])
        self.assertEqual(a[1].tolist(), [2])
        self.assertEqual(b[0].tolist(), [[0, 2, 5, -1, -1, -1]])
        self.assertEqual(b[1].tolist(), [3])
        # Re-querying width 4 returns the first width-4 object.
        self.assertIs(meta.compact_attn_topk_idxs(narrow), a)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}",
)
class TestCompressedSparseAttnGate(unittest.TestCase):
    """The gate compacts via the shared cache exactly on the no-indexer,
    ``topk_length is None``, cuDNN, SM100 path; all other branches forward the
    original holey indices with ``topk_length`` untouched."""

    # Original (still holey) indices handed to the gate. Hand-derived compaction
    # (order-preserving prefix + per-row valid count):
    #   [0, -1, 2, -1] -> [0, 2, -1, -1], length 2
    #   [3, -1, -1, -1] -> [3, -1, -1, -1], length 1
    _ORIG = [[0, -1, 2, -1], [3, -1, -1, -1]]
    _COMPACT = [[0, 2, -1, -1], [3, -1, -1, -1]]
    _LENGTHS = [2, 1]

    @staticmethod
    def _self_ns(*, indexer, backend):
        return types.SimpleNamespace(
            config=types.SimpleNamespace(csa_sparse_attn_backend=backend),
            indexer=indexer,
            global_kv_idx_remap_fusion=False,
        )

    def _run_gate(
        self, *, indexer, backend, honours, docmask_meta, topk_length
    ):
        query = paddle.zeros([1, 2, 2, 8], dtype="float32")
        kv_full = paddle.zeros([1, 16, 8], dtype="float32")
        attn_sink = paddle.zeros([2], dtype="bfloat16")
        topk_idxs = paddle.to_tensor(self._ORIG, dtype="int32")
        seen = {}

        def fake_kernel(
            q,
            kv,
            sink,
            idxs,
            scale,
            *,
            backend,
            topk_length,
            indexer_topk,
            global_kv_idx_remap_fusion,
            topk_idxs_compacted,
        ):
            seen.update(
                q=q,
                kv=kv,
                sink=sink,
                idxs=idxs,
                scale=scale,
                backend=backend,
                topk_length=topk_length,
                indexer_topk=indexer_topk,
                global_kv_idx_remap_fusion=global_kv_idx_remap_fusion,
                topk_idxs_compacted=topk_idxs_compacted,
            )
            return paddle.zeros([1])

        with (
            patch.object(csa_fusion, "csa_sparse_attn", fake_kernel),
            patch.object(
                csa_fusion,
                "_csa_bwd_honours_topk_length_holes",
                lambda: honours,
            ),
        ):
            CompressedSparseAttention.compressed_sparse_attn(
                self._self_ns(indexer=indexer, backend=backend),
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                0.125,
                topk_length=topk_length,
                indexer_topk=0,
                docmask_meta=docmask_meta,
            )
        return seen, query, kv_full

    def _assert_forwarded_unchanged(self, seen, *, expected_length):
        # Gate did NOT fire: original holey indices reach the kernel verbatim,
        # nothing marked compacted, and topk_length is whatever the caller gave.
        self.assertFalse(seen["topk_idxs_compacted"])
        self.assertEqual(seen["idxs"].tolist(), self._ORIG)
        if expected_length is None:
            self.assertIsNone(seen["topk_length"])
        else:
            self.assertIs(seen["topk_length"], expected_length)

    def test_gate_compacts_on_hca_cudnn_sm100(self):
        meta = CSADocMaskMetadata.__new__(CSADocMaskMetadata)
        seen, query, kv_full = self._run_gate(
            indexer=None,
            backend="cudnn",
            honours=True,
            docmask_meta=meta,
            topk_length=None,
        )
        # Gate fired: the real shared-cache compaction reached the kernel.
        self.assertTrue(seen["topk_idxs_compacted"])
        self.assertEqual(seen["idxs"].tolist(), self._COMPACT)
        self.assertEqual(seen["topk_length"].tolist(), self._LENGTHS)
        # Untouched arguments are forwarded intact.
        self.assertEqual(seen["backend"], "cudnn")
        self.assertEqual(seen["indexer_topk"], 0)
        self.assertEqual(seen["scale"], 0.125)
        self.assertFalse(seen["global_kv_idx_remap_fusion"])
        self.assertIs(seen["q"], query)
        self.assertIs(seen["kv"], kv_full)
        self.assertEqual(seen["sink"].dtype, paddle.float32)  # cast to fp32
        # Result was memoized on the metadata, keyed by row width (4).
        self.assertIn(4, meta._compacted_attn_topk)

    def test_gate_skipped_when_indexer_present(self):
        # A layer WITH an indexer must not reuse the width-keyed cache.
        seen, _, _ = self._run_gate(
            indexer=object(),
            backend="cudnn",
            honours=True,
            docmask_meta=CSADocMaskMetadata.__new__(CSADocMaskMetadata),
            topk_length=None,
        )
        self._assert_forwarded_unchanged(seen, expected_length=None)

    def test_gate_skipped_on_non_cudnn_backend(self):
        # Only cuDNN consumes topk_length; tilelang keeps the holey layout.
        seen, _, _ = self._run_gate(
            indexer=None,
            backend="tilelang",
            honours=True,
            docmask_meta=CSADocMaskMetadata.__new__(CSADocMaskMetadata),
            topk_length=None,
        )
        self.assertEqual(seen["backend"], "tilelang")
        self._assert_forwarded_unchanged(seen, expected_length=None)

    def test_gate_skipped_on_sm90(self):
        # SM90 backward has no empty-row guard -> compacted length is unsafe.
        seen, _, _ = self._run_gate(
            indexer=None,
            backend="cudnn",
            honours=False,
            docmask_meta=CSADocMaskMetadata.__new__(CSADocMaskMetadata),
            topk_length=None,
        )
        self._assert_forwarded_unchanged(seen, expected_length=None)

    def test_gate_skipped_when_topk_length_supplied(self):
        # The MQA path pre-supplies topk_length; it must pass through unchanged.
        provided = paddle.to_tensor([4, 4], dtype="int32")
        seen, _, _ = self._run_gate(
            indexer=None,
            backend="cudnn",
            honours=True,
            docmask_meta=CSADocMaskMetadata.__new__(CSADocMaskMetadata),
            topk_length=provided,
        )
        self._assert_forwarded_unchanged(seen, expected_length=provided)

    def test_gate_skipped_without_docmask_meta(self):
        seen, _, _ = self._run_gate(
            indexer=None,
            backend="cudnn",
            honours=True,
            docmask_meta=None,
            topk_length=None,
        )
        self._assert_forwarded_unchanged(seen, expected_length=None)


if __name__ == "__main__":
    unittest.main()
