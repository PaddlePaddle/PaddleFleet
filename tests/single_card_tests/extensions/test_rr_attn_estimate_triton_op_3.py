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

"""CPU-observable behavior tests for the round-robin attention estimate op.

Target production code:
    packages/paddlefleet_ops/src/paddlefleet_ops/_extensions/flashmask/
        rr_attn_estimate_triton_op.py

Facet owned by this file (distinct from sibling test files that target the
same module): the pure-Python / paddle index math of ``_extract_raw_ptrs`` --
i.e. which column of the ``startend_row_indices`` tensor is routed into which
semantic pointer field for every (mode, causal) combination, plus the object
aliasing contract for the unused fields. This is orchestration / index math
that runs entirely on CPU.

The triton kernels (flashmask_apply, gemm_fuse_softmax_*) require a real GPU to
produce numeric output, so this file deliberately does NOT assert any kernel
output; those numerics must be validated on a single-card (GPU) run instead.

paddlefleet imports paddle at import time and the module also imports triton;
if either is unavailable the whole suite is honestly skipped (never faked pass).
"""

import os
import sys
import unittest

# Allow importing the package from the in-repo source layout when it is not
# pip-installed. On the single-card CI the package is installed and this is a
# harmless no-op.
_PKG_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "packages",
    "paddlefleet_ops",
    "src",
)
if _PKG_SRC not in sys.path:
    sys.path.insert(0, _PKG_SRC)

# Guard the import honestly: paddle (and triton) are hard runtime deps of the
# module under test. Catch only ImportError so that genuine API/compile errors
# still surface instead of being swallowed as a "missing dependency" skip.
_IMPORT_ERROR = None
try:
    import paddle
    from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
        RawPtrs,
        _extract_raw_ptrs,
    )

    _AVAILABLE = True
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = (
    "paddle/triton or paddlefleet_ops not importable in this environment: "
    f"{_IMPORT_ERROR}"
)

# Distinct, hand-derived fixture: value at [0, 0, s, c] == (c + 1) * 1000 + s.
# Every channel and every sequence position carries a unique integer, so a
# wrong column->field mapping, a transposed/positional error, or a dropped
# ``.contiguous()`` copy would all change the observed content.
_SEQ = 5


def _build_indices(mode):
    data = [[[[(c + 1) * 1000 + s for c in range(mode)] for s in range(_SEQ)]]]
    return paddle.to_tensor(data, dtype="int32")


def _expected_channel(c):
    return [(c + 1) * 1000 + s for s in range(_SEQ)]


def _flat(tensor):
    return tensor.reshape([-1]).tolist()


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestExtractRawPtrsColumnMapping(unittest.TestCase):
    """Each (mode, causal) combo routes the correct source column."""

    def test_mode4_maps_all_four_columns_in_order(self):
        mode, raw = _extract_raw_ptrs(_build_indices(4), causal=True)
        self.assertEqual(mode, 4)
        self.assertEqual(_flat(raw.lt_start), _expected_channel(0))
        self.assertEqual(_flat(raw.lt_end), _expected_channel(1))
        self.assertEqual(_flat(raw.ut_start), _expected_channel(2))
        self.assertEqual(_flat(raw.ut_end), _expected_channel(3))

    def test_mode2_causal_uses_lt_end_from_column1(self):
        mode, raw = _extract_raw_ptrs(_build_indices(2), causal=True)
        self.assertEqual(mode, 2)
        # causal=True binds column 1 to lt_end; ut_* fall back to lt_start.
        self.assertEqual(_flat(raw.lt_start), _expected_channel(0))
        self.assertEqual(_flat(raw.lt_end), _expected_channel(1))
        self.assertEqual(_flat(raw.ut_start), _expected_channel(0))
        self.assertEqual(_flat(raw.ut_end), _expected_channel(0))

    def test_mode2_noncausal_uses_ut_end_from_column1(self):
        mode, raw = _extract_raw_ptrs(_build_indices(2), causal=False)
        self.assertEqual(mode, 2)
        # causal=False binds column 1 to ut_end instead; lt_end/ut_start fall
        # back to lt_start. This is the branch that distinguishes the two
        # mode-2 code paths, so column 1 must NOT land in lt_end here.
        self.assertEqual(_flat(raw.lt_start), _expected_channel(0))
        self.assertEqual(_flat(raw.ut_end), _expected_channel(1))
        self.assertEqual(_flat(raw.lt_end), _expected_channel(0))
        self.assertEqual(_flat(raw.ut_start), _expected_channel(0))

    def test_mode1_only_lt_start_is_populated(self):
        mode, raw = _extract_raw_ptrs(_build_indices(1), causal=True)
        self.assertEqual(mode, 1)
        for field in (raw.lt_start, raw.lt_end, raw.ut_start, raw.ut_end):
            self.assertEqual(_flat(field), _expected_channel(0))

    def test_token_level_shape_drops_mode_axis(self):
        # [B, HIDS, seqlen, mode] -> token-level pointer [B, HIDS, seqlen].
        _, raw = _extract_raw_ptrs(_build_indices(4), causal=True)
        self.assertEqual(list(raw.lt_start.shape), [1, 1, _SEQ])


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestExtractRawPtrsAliasing(unittest.TestCase):
    """Unused fields must alias lt_start; used fields must be distinct."""

    def test_mode1_all_fields_alias_lt_start(self):
        _, raw = _extract_raw_ptrs(_build_indices(1), causal=True)
        self.assertIs(raw.lt_end, raw.lt_start)
        self.assertIs(raw.ut_start, raw.lt_start)
        self.assertIs(raw.ut_end, raw.lt_start)

    def test_mode2_causal_only_lt_end_is_distinct(self):
        _, raw = _extract_raw_ptrs(_build_indices(2), causal=True)
        self.assertIsNot(raw.lt_end, raw.lt_start)
        self.assertIs(raw.ut_start, raw.lt_start)
        self.assertIs(raw.ut_end, raw.lt_start)

    def test_mode2_noncausal_only_ut_end_is_distinct(self):
        _, raw = _extract_raw_ptrs(_build_indices(2), causal=False)
        self.assertIsNot(raw.ut_end, raw.lt_start)
        self.assertIs(raw.lt_end, raw.lt_start)
        self.assertIs(raw.ut_start, raw.lt_start)

    def test_mode4_all_four_fields_are_distinct_objects(self):
        _, raw = _extract_raw_ptrs(_build_indices(4), causal=True)
        self.assertIsNot(raw.lt_end, raw.lt_start)
        self.assertIsNot(raw.ut_start, raw.lt_start)
        self.assertIsNot(raw.ut_end, raw.lt_start)
        self.assertIsNot(raw.ut_start, raw.lt_end)
        self.assertIsNot(raw.ut_end, raw.ut_start)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestExtractRawPtrsModeValidation(unittest.TestCase):
    """Only modes 1/2/4 are accepted; others raise a clear ValueError."""

    def test_returns_raw_ptrs_instance(self):
        _, raw = _extract_raw_ptrs(_build_indices(4), causal=True)
        self.assertIsInstance(raw, RawPtrs)

    def test_mode3_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _extract_raw_ptrs(_build_indices(3), causal=True)
        self.assertIn("Unsupported mode", str(ctx.exception))
        # The message should echo the offending mode value.
        self.assertIn("3", str(ctx.exception))

    def test_returned_mode_matches_last_dim(self):
        for m in (1, 2, 4):
            mode, _ = _extract_raw_ptrs(_build_indices(m), causal=True)
            self.assertEqual(mode, m)


if __name__ == "__main__":
    unittest.main()
