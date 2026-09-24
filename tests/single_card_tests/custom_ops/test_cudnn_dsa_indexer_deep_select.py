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

"""``index_topk_backend="deep_select"`` in ``csa_indexer_fwd_cudnn``.

The ``*Contract`` classes replace ``deep_select`` with ``_FakeDeepSelect`` and
run on any card: they pin what the wrapper passes to the kernel (row limits,
alignment, dtypes, fill values) and how its output is padded, remapped and
routed back. They say nothing about kernel numerics; the ``*Kernel`` classes
compare the real kernel against ``paddle.topk`` and need SM100+ plus a
``paddlefleet_ops`` built with DeepSelect.
"""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import paddle
from paddlefleet_ops import is_cudnn_frontend_available

import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as indexer_mod

NEG_INF = float("-inf")
_FWD_API = "paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api"


def _deep_select_kernel_available():
    if (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() == 0
        or paddle.device.cuda.get_device_capability()[0] < 10
    ):
        return False
    try:
        indexer_mod._require_deep_select()
    except ImportError:
        return False
    return True


_HAS_DEEP_SELECT = _deep_select_kernel_available()
_HAS_CUDNN_INDEXER = (
    _HAS_DEEP_SELECT
    and paddle.device.cuda.get_device_capability()[0] == 10
    and is_cudnn_frontend_available()
)


class _FakeDeepSelect:
    """Prefix top-k over ``[0, end)``; ids come back ascending, i.e. unsorted."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def get_stride_requirement():
        return (1024, 32)

    def topk(self, input, k, **kw):
        self.calls.append((input, k, kw))
        indices, values = [], []
        for row, end in zip(input.tolist(), kw["end"].tolist()):
            ids = sorted(sorted(range(end), key=lambda i: -row[i])[:k])
            ids += [-1] * (k - len(ids))
            indices.append(ids)
            values.append([row[i] if i >= 0 else NEG_INF for i in ids])
        return (
            paddle.to_tensor(values, "float32") if kw["return_value"] else None,
            paddle.to_tensor(indices, "int32"),
        )


def _as_scores(ids):
    """Expected scores when column ``j`` scores ``j``."""
    return [[float(i) if i >= 0 else NEG_INF for i in row] for row in ids]


class _ContractCase(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeDeepSelect()
        loader = patch.object(
            indexer_mod, "_require_deep_select", return_value=self.fake
        )
        loader.start()
        self.addCleanup(loader.stop)

    def assert_kernel_calls(self, return_val):
        self.assertGreater(len(self.fake.calls), 0)
        for input, _, kw in self.fake.calls:
            self.assertEqual(input.strides[-1], 1)
            self.assertEqual(input.strides[0] % 256, 0, "row not 1024B aligned")
            self.assertEqual(kw["end"].dtype, paddle.int32)
            self.assertEqual(list(kw["end"].shape), [input.shape[0]])
            kw = {k: v for k, v in kw.items() if k != "end"}
            self.assertEqual(
                kw,
                {
                    "sorted": False,
                    "indices_type": paddle.int32,
                    "idx_oob_fill_value": -1,
                    "value_oob_fill_value": NEG_INF,
                    "return_value": return_val,
                    "abort_when_nan_found": True,
                },
            )


class TestDeepSelectTopKContract(_ContractCase):
    def test_aligned_rows_are_passed_without_copy(self):
        values = paddle.arange(3 * 256, dtype="float32").reshape([3, 256])
        seq_lens = paddle.to_tensor([0, 2, 256], dtype="int64")
        for return_val in (False, True):
            with self.subTest(return_val=return_val):
                self.fake.calls.clear()
                out = indexer_mod._select_indexer_top_k(
                    values, seq_lens, 4, return_val, "deep_select"
                )
                self.assert_kernel_calls(return_val)
                self.assertIs(self.fake.calls[0][0], values)
                self.assertEqual(out["indices"].dtype, paddle.int32)
                self.assertEqual(
                    out["indices"].tolist(),
                    [[-1] * 4, [0, 1, -1, -1], [252, 253, 254, 255]],
                )
                if return_val:
                    self.assertEqual(
                        out["values"].tolist(),
                        [
                            [NEG_INF] * 4,
                            [256, 257, NEG_INF, NEG_INF],
                            [764, 765, 766, 767],
                        ],
                    )
                else:
                    self.assertIsNone(out["values"])

    def test_unaligned_rows_are_repacked_and_short_rows_padded(self):
        base = paddle.arange(3 * 8, dtype="float32").reshape([3, 8])
        values = base[:, :5]  # row stride 8: not 1024B aligned
        seq_lens = paddle.to_tensor([1, 3, 5], dtype="int32")
        for return_val in (False, True):
            with self.subTest(return_val=return_val):
                self.fake.calls.clear()
                out = indexer_mod._indexer_top_k_deep_select(
                    values, seq_lens, 7, return_val
                )
                self.assert_kernel_calls(return_val)
                passed, k, _ = self.fake.calls[0]
                self.assertEqual(k, 5)
                self.assertEqual(passed.tolist(), values.tolist())
                self.assertEqual(
                    out["indices"].tolist(),
                    [
                        [0, -1, -1, -1, -1, -1, -1],
                        [0, 1, 2, -1, -1, -1, -1],
                        [0, 1, 2, 3, 4, -1, -1],
                    ],
                )
                if return_val:
                    self.assertEqual(
                        out["values"].tolist(),
                        [
                            [0] + [NEG_INF] * 6,
                            [8, 9, 10] + [NEG_INF] * 4,
                            [16, 17, 18, 19, 20, NEG_INF, NEG_INF],
                        ],
                    )
                else:
                    self.assertIsNone(out["values"])

    def test_top_k_above_kernel_limit_is_rejected_before_loading(self):
        values = paddle.zeros([1, 4097], dtype="float32")
        with patch.object(
            indexer_mod, "_require_deep_select", side_effect=AssertionError
        ):
            with self.assertRaisesRegex(ValueError, "top_k <= 4096"):
                indexer_mod._indexer_top_k_deep_select(
                    values, paddle.to_tensor([4097]), 4097, False
                )
            # k is clamped to the row width first, so a short row is legal.
            with self.assertRaises(AssertionError):
                indexer_mod._indexer_top_k_deep_select(
                    values[:, :8], paddle.to_tensor([8]), 5000, False
                )


class TestDeepSelectLoading(unittest.TestCase):
    def test_missing_library_raises_import_error_without_fallback(self):
        values = paddle.zeros([1, 256], dtype="float32")
        seq_lens = paddle.to_tensor([256], dtype="int32")
        for error in (ImportError("not built"), RuntimeError("SM90 blocked")):
            ops = types.ModuleType("paddlefleet_ops")

            def _getattr(name, error=error):
                raise error

            ops.__getattr__ = _getattr
            with (
                self.subTest(error=type(error).__name__),
                patch.dict(sys.modules, {"paddlefleet_ops": ops}),
                patch.object(indexer_mod, "_indexer_top_k_unfused") as unfused,
            ):
                with self.assertRaisesRegex(
                    ImportError,
                    "requires paddlefleet_ops built with DeepSelect",
                ) as ctx:
                    indexer_mod._select_indexer_top_k(
                        values, seq_lens, 2, False, "deep_select"
                    )
                self.assertIs(ctx.exception.__cause__, error)
                unfused.assert_not_called()

    def test_paddle_backend_never_loads_deep_select(self):
        values = paddle.to_tensor([[1.0, 3.0, 2.0, 100.0]])
        seq_lens = paddle.to_tensor([3], dtype="int32")
        with patch.object(
            indexer_mod, "_require_deep_select", side_effect=AssertionError
        ):
            out = indexer_mod._select_indexer_top_k(
                values, seq_lens, 2, True, "paddle"
            )
        self.assertEqual(out["indices"].tolist(), [[1, 2]])
        self.assertEqual(out["values"].tolist(), [[3.0, 2.0]])

    def test_unknown_backend_is_rejected(self):
        values = paddle.zeros([1, 4], dtype="float32")
        with self.assertRaisesRegex(ValueError, "index_topk_backend='radix'"):
            indexer_mod._select_indexer_top_k(
                values, paddle.to_tensor([4]), 2, False, "radix"
            )
        with patch.object(indexer_mod, "_validate_indexer_inputs") as validate:
            for doc_lens in (None, [4]):
                with self.assertRaisesRegex(
                    ValueError, "index_topk_backend='radix'"
                ):
                    indexer_mod.cudnn_indexer_topk_fwd(
                        None,
                        None,
                        None,
                        doc_lens=doc_lens,
                        index_topk_backend="radix",
                    )
            validate.assert_not_called()

    def test_scores_require_deep_select_backend(self):
        with self.assertRaisesRegex(ValueError, "return_topk_scores"):
            indexer_mod.cudnn_indexer_topk(
                paddle.zeros([1, 2, 4]), 2, 1, 2, return_topk_scores=True
            )


def _indexer_inputs(sq, sk, heads=32):
    return (
        paddle.zeros([1, sq, heads, 128], dtype="bfloat16"),
        paddle.zeros([1, sk, 128], dtype="bfloat16"),
        paddle.zeros([1, sq, heads], dtype="bfloat16"),
    )


class TestDeepSelectIndexerFwdContract(_ContractCase):
    """``cudnn_indexer_topk_fwd`` routing with the cuDNN score kernel faked."""

    def test_docmask_remap_keeps_value_pairs(self):
        scores = paddle.to_tensor(
            [[[100.0, 2.0, 9.0, 4.0, 200.0], [7.0, 8.0, 9.0, 10.0, 11.0]]]
        )
        valid_range = paddle.to_tensor([[[1, 4], [3, 3]]], dtype="int32")
        ids, lengths, values = indexer_mod.cudnn_indexer_topk(
            scores,
            2,
            1,
            2,
            valid_range=valid_range,
            index_topk_backend="deep_select",
            return_topk_scores=True,
        )
        self.assert_kernel_calls(True)
        self.assertEqual(ids.tolist(), [[[2, 3], [-1, -1]]])
        self.assertEqual(lengths.tolist(), [[2, 0]])
        self.assertEqual(values.tolist(), [[[9.0, 4.0], [NEG_INF] * 2]])

    def test_dense_causal_path_with_query_tiling(self):
        sq, sk = 5, 4
        offsets = []

        def fake_forward(index_q, index_k_comp, weights, **kw):
            offsets.append(kw["seq_offset"])
            rows = int(index_q.shape[1])
            return paddle.arange(sk, dtype="float32").expand([1, rows, sk])

        expected = {
            2: [[0, -1], [0, 1], [1, 2], [2, 3], [2, 3]],
            6: [
                [0, -1, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1],
                [0, 1, 2, -1, -1, -1],
                [0, 1, 2, 3, -1, -1],
                [0, 1, 2, 3, -1, -1],
            ],
        }
        for tile_elems, want_offsets in ((1 << 26, [0]), (2 * sk, [0, 2, 4])):
            for topk, ids in expected.items():
                for scores in (False, True):
                    with (
                        self.subTest(tile=tile_elems, topk=topk, scores=scores),
                        patch.object(
                            indexer_mod, "cudnn_indexer_forward", fake_forward
                        ),
                        patch.object(
                            indexer_mod, "_DEFAULT_QUERY_TILE_ELEMS", tile_elems
                        ),
                    ):
                        offsets.clear()
                        self.fake.calls.clear()
                        out = indexer_mod.cudnn_indexer_topk_fwd(
                            *_indexer_inputs(sq, sk),
                            ratio=1,
                            topk_effective=topk,
                            return_topk_scores=scores,
                            index_topk_backend="deep_select",
                        )
                        self.assertEqual(offsets, want_offsets)
                        self.assert_kernel_calls(scores)
                        self.assertEqual(len(out), 3 if scores else 2)
                        self.assertEqual(out[0].tolist(), [ids])
                        self.assertEqual(
                            out[1].tolist(),
                            [[sum(i >= 0 for i in r) for r in ids]],
                        )
                        if scores:
                            self.assertEqual(out[2].tolist(), [_as_scores(ids)])

    def test_thd_docmask_rows_are_kernel_ready(self):
        # doc_lens [3, 2] at ratio 1: document-local causal windows.
        valid_range = paddle.to_tensor(
            [[[0, 1], [0, 2], [0, 3], [3, 4], [3, 5]]], dtype="int32"
        )
        produced = {}

        def fake_wrapper(q, k, w, **kw):
            max_k = kw["max_seqlen_k"]
            produced["max_k"] = max_k
            produced["scores"] = paddle.tile(
                paddle.arange(max_k, dtype="float32").unsqueeze(0), [5, 1]
            )
            return {"scores": produced["scores"]}

        api = types.ModuleType(_FWD_API)
        api.indexer_forward_wrapper = fake_wrapper
        sorted_ids = [[-1, 0], [0, 1], [1, 2], [-1, 3], [3, 4]]
        for backend, max_k in (("paddle", 4), ("deep_select", 256)):
            with (
                self.subTest(backend=backend),
                patch.dict(sys.modules, {_FWD_API: api}),
                patch.object(indexer_mod, "_require_cudnn_frontend"),
                patch.object(indexer_mod, "cudnn_indexer_forward") as dense,
            ):
                self.fake.calls.clear()
                ids, lengths, values = indexer_mod.cudnn_indexer_topk_fwd(
                    *_indexer_inputs(5, 5),
                    ratio=1,
                    topk_effective=2,
                    valid_range=valid_range,
                    doc_lens=[3, 2],
                    return_topk_scores=True,
                    index_topk_backend=backend,
                )
                dense.assert_not_called()
                self.assertEqual(produced["max_k"], max_k)
                self.assertEqual(
                    np.sort(ids.numpy(), -1).tolist(), [sorted_ids]
                )
                self.assertEqual(lengths.tolist(), [[1, 2, 2, 1, 2]])
                if backend == "deep_select":
                    self.assert_kernel_calls(True)
                    self.assertIs(self.fake.calls[0][0], produced["scores"])
                    self.assertEqual(
                        values.tolist(),
                        [
                            _as_scores(
                                [[0, -1], [0, 1], [1, 2], [0, -1], [0, 1]]
                            )
                        ],
                    )
                else:
                    self.assertEqual(self.fake.calls, [])


def _prefix_scores(rows, width, seq_lens, seed):
    """Random scores with large finite decoys past each row's limit."""
    paddle.seed(seed)
    x = paddle.randn([rows, width], dtype="float32")
    cols = paddle.arange(width, dtype="int64").unsqueeze(0)
    return paddle.where(
        cols < seq_lens.unsqueeze(1), x, paddle.full_like(x, 1e4)
    )


@unittest.skipIf(not _HAS_DEEP_SELECT, "DeepSelect kernel requires SM100+")
class TestDeepSelectKernel(unittest.TestCase):
    """Real ``deep_select.topk`` behind the wrapper vs ``paddle.topk``."""

    def setUp(self):
        paddle.set_device("gpu")

    def test_matches_paddle_topk_as_sets_with_paired_values(self):
        for width in (1000, 2048):  # unaligned / aligned rows
            ends = [0, 1, 63, 64, 65, width // 2, width]
            seq_lens = paddle.to_tensor(ends, dtype="int64")
            values = _prefix_scores(len(ends), width, seq_lens, seed=width)
            for top_k in (64, 2048):
                for return_val in (False, True):
                    with self.subTest(width=width, top_k=top_k, rv=return_val):
                        out = indexer_mod._indexer_top_k_deep_select(
                            values, seq_lens, top_k, return_val
                        )
                        ref = indexer_mod._indexer_top_k_unfused(
                            values, seq_lens, top_k, False
                        )
                        ids = out["indices"]
                        self.assertEqual(ids.dtype, paddle.int32)
                        self.assertEqual(list(ids.shape), [len(ends), top_k])
                        np.testing.assert_array_equal(
                            np.sort(ids.numpy(), -1),
                            np.sort(ref["indices"].numpy(), -1),
                        )
                        if not return_val:
                            self.assertIsNone(out["values"])
                            continue
                        valid = ids >= 0
                        safe = paddle.where(valid, ids, paddle.zeros_like(ids))
                        picked = paddle.take_along_axis(
                            values, safe.cast("int64"), axis=1
                        )
                        np.testing.assert_array_equal(
                            out["values"].numpy(),
                            paddle.where(
                                valid, picked, paddle.full_like(picked, NEG_INF)
                            ).numpy(),
                        )


def _docmask_valid_range(doc_lens, ratio):
    rows, col = [], 0
    for n in doc_lens:
        rows += [[col, col + (t + 1) // ratio] for t in range(n)]
        col += n // ratio
    return paddle.to_tensor([rows], dtype="int32")


@unittest.skipIf(
    not _HAS_CUDNN_INDEXER, "cuDNN indexer + DeepSelect require SM10x"
)
class TestDeepSelectIndexerFwdKernel(unittest.TestCase):
    """End-to-end: both backends select the same scores on real cuDNN scores."""

    S, RATIO = 512, 4

    def setUp(self):
        paddle.set_device("gpu")

    def test_backends_agree_on_selected_scores(self):
        paddle.seed(2026)
        sk = self.S // self.RATIO
        index_q = paddle.randn([1, self.S, 64, 128]).astype("bfloat16")
        index_k = paddle.randn([1, sk, 128]).astype("bfloat16")
        weights = paddle.randn([1, self.S, 64]).astype("bfloat16")
        doc_lens = [160, self.S - 160]
        docmask = _docmask_valid_range(doc_lens, self.RATIO)
        causal_end = paddle.arange(1, self.S + 1, dtype="int32") // self.RATIO
        modes = {
            "causal": ({}, paddle.zeros_like(causal_end), causal_end),
            "dense_docmask": (
                {"valid_range": docmask},
                docmask[0, :, 0],
                docmask[0, :, 1],
            ),
            "thd_docmask": (
                {"valid_range": docmask, "doc_lens": doc_lens},
                docmask[0, :, 0],
                docmask[0, :, 1],
            ),
        }
        for mode, (kwargs, start, end) in modes.items():
            for topk in (64, 256):
                with self.subTest(mode=mode, topk=topk):
                    ref, out = (
                        indexer_mod.cudnn_indexer_topk_fwd(
                            index_q,
                            index_k,
                            weights,
                            ratio=self.RATIO,
                            topk_effective=topk,
                            return_topk_scores=True,
                            index_topk_backend=backend,
                            **kwargs,
                        )
                        for backend in ("paddle", "deep_select")
                    )
                    np.testing.assert_array_equal(
                        out[1].numpy(), ref[1].numpy()
                    )
                    np.testing.assert_array_equal(
                        np.sort(out[2].numpy(), -1), np.sort(ref[2].numpy(), -1)
                    )
                    ids = out[0][0].numpy()
                    valid = ids >= 0
                    self.assertEqual(valid.sum(-1).tolist(), out[1][0].tolist())
                    lo = start.numpy()[:, None]
                    hi = end.numpy()[:, None]
                    self.assertTrue(((ids >= lo) & (ids < hi))[valid].all())
                    for row in ids:
                        picked = row[row >= 0]
                        self.assertEqual(len(set(picked)), len(picked))


if __name__ == "__main__":
    unittest.main()
