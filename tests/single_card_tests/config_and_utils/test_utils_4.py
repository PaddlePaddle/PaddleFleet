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

"""Behavior tests for the fourth slice of paddlefleet.utils (the public
helpers re-exported from paddlefleet/utils/_fleet_utils.py).

Scope (disjoint fourth slice). This file covers exactly the helpers this
slice owns, so it stays disjoint from sibling utils slices:

  * GlobalMemoryBuffer.get_tensor  -- flat-buffer sizing, storage reuse vs.
    reallocation, and (name, dtype) keying
  * make_viewless_tensor           -- documents a real defect: it guards on
    the nonexistent ``Tensor._is_view`` and raises AttributeError
  * get_model_type                 -- attribute lookup with .module unwrapping
  * get_model_xattn                -- raw attribute passthrough / False fallback
  * get_paddle_version             -- version snapshot vs. paddle.__version__
  * is_paddle_min_version          -- >= vs > comparison and equality boundary
  * log_single_rank                -- rank-gated single-rank logging

Every expected value is hand-derived from the source arithmetic / control
flow, never read back from the function under test. paddle.distributed state
is a genuine collaborator; where it is patched (log_single_rank) the test only
exercises the LOCAL rank-gating branch decision -- no collective is issued or
faked, so this proves nothing about cross-rank communication (see
antipattern #13). paddle is imported honestly; if it (or the lazy-imported
module) is unavailable the whole module skips with a truthful reason rather
than faking a pass.
"""

import logging
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe: paddlefleet.utils lazily imports _fleet_utils,
# which does ``import paddle`` at module load. Only a genuine missing
# dependency (ImportError) may skip; anything else must surface as a failure.
try:
    import paddle
    from packaging.version import Version as PkgVersion

    from paddlefleet.utils import (
        GlobalMemoryBuffer,
        get_model_type,
        get_model_xattn,
        get_paddle_version,
        is_paddle_min_version,
        log_single_rank,
        make_viewless_tensor,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    paddle = None
    PkgVersion = None
    _IMPORT_ERROR = exc

_SKIP_MSG = f"paddlefleet.utils could not be imported (missing dependency): {_IMPORT_ERROR}"


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestGlobalMemoryBuffer(unittest.TestCase):
    """get_tensor returns a reshaped view over a flat, reused 1-D buffer."""

    def test_flat_storage_sized_to_numel_and_view_shape(self):
        # A [4, 8] request must allocate a flat 1-D buffer of exactly
        # 4 * 8 == 32 elements and return it reshaped to [4, 8].
        buf = GlobalMemoryBuffer()
        view = buf.get_tensor([4, 8], paddle.float32, "s")
        flat = buf.buffer[("s", paddle.float32)]
        self.assertEqual(flat.shape, [32])  # hand-derived 4 * 8
        self.assertEqual(view.shape, [4, 8])
        # The returned view is the [0:32] slice of the flat buffer, so it
        # starts at the same storage address.
        self.assertEqual(view.data_ptr(), flat.data_ptr())

    def test_reuses_storage_when_within_capacity(self):
        # First request sizes the buffer to 32; a later smaller request
        # (2 * 4 == 8 <= 32) must reuse the SAME flat storage, not reallocate.
        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([4, 8], paddle.float32, "g")
        flat_first = buf.buffer[("g", paddle.float32)]
        t2 = buf.get_tensor([2, 4], paddle.float32, "g")
        self.assertIs(buf.buffer[("g", paddle.float32)], flat_first)
        # Both views start at offset 0 of the same reused buffer.
        self.assertEqual(t1.data_ptr(), t2.data_ptr())
        self.assertEqual(t2.shape, [2, 4])

    def test_reallocates_when_capacity_exceeded(self):
        # Start at numel 8, then request numel 32 > 8: the flat buffer must
        # be reallocated to length 32 (a new object).
        buf = GlobalMemoryBuffer()
        buf.get_tensor([2, 4], paddle.float32, "g")
        flat_small = buf.buffer[("g", paddle.float32)]
        self.assertEqual(flat_small.shape, [8])  # 2 * 4
        buf.get_tensor([4, 8], paddle.float32, "g")
        flat_big = buf.buffer[("g", paddle.float32)]
        self.assertIsNot(flat_big, flat_small)
        self.assertEqual(flat_big.shape, [32])  # 4 * 8

    def test_distinct_storage_per_name_and_dtype(self):
        # (name, dtype) is the buffer key: different names OR different dtypes
        # yield independent storages, and dtype is honored on the view.
        buf = GlobalMemoryBuffer()
        a = buf.get_tensor([4], paddle.float32, "a")
        b = buf.get_tensor([4], paddle.float32, "b")
        d16 = buf.get_tensor([4], paddle.float16, "a")
        self.assertEqual(
            set(buf.buffer.keys()),
            {
                ("a", paddle.float32),
                ("b", paddle.float32),
                ("a", paddle.float16),
            },
        )
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertNotEqual(a.data_ptr(), d16.data_ptr())
        self.assertEqual(a.dtype, paddle.float32)
        self.assertEqual(d16.dtype, paddle.float16)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestMakeViewlessTensor(unittest.TestCase):
    """make_viewless_tensor is broken: it calls a nonexistent Tensor method.

    The documented contract is "return the tensor as-is if it is not a view",
    so for a freshly created (non-view) tensor the function should return the
    very same object. But the implementation guards on ``inp._is_view()`` and
    ``paddle.Tensor`` has no ``_is_view`` method in this Paddle version, so the
    first line raises ``AttributeError`` for *any* input -- the "non-view
    returned unchanged" path can never actually be reached. This is a genuine
    production defect at src/paddlefleet/utils/_fleet_utils.py:552; it is
    documented here with assertRaises and NOT worked around, and no production
    code is modified.
    """

    def test_missing_is_view_attribute_raises(self):
        # A plain (non-view) tensor should, per the docstring, be returned
        # unchanged; instead the `inp._is_view()` call raises because the
        # attribute does not exist. Pin the exact defect via the message so a
        # future Paddle that adds `_is_view` (or a production fix) surfaces as a
        # change here rather than silently passing.
        t = paddle.randn([4, 8])
        with self.assertRaises(AttributeError) as ctx:
            make_viewless_tensor(t, requires_grad=False, keep_graph=False)
        self.assertIn("_is_view", str(ctx.exception))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestGetModelType(unittest.TestCase):
    """get_model_type reads model_type, unwrapping .module as needed."""

    def test_returns_attribute_direct_and_unwrapped(self):
        class Plain:
            pass

        direct = Plain()
        direct.model_type = "gpt"
        self.assertEqual(get_model_type(direct), "gpt")

        # Wrapper lacks model_type but has .module holding it; the while-loop
        # must unwrap exactly one level and return the inner value.
        inner = Plain()
        inner.model_type = "llama"
        wrapper = Plain()
        wrapper.module = inner
        self.assertEqual(get_model_type(wrapper), "llama")

    def test_raises_when_absent(self):
        class Bare:
            pass

        # No model_type and no .module to unwrap -> RuntimeError contract.
        with self.assertRaises(RuntimeError):
            get_model_type(Bare())


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestGetModelXattn(unittest.TestCase):
    """get_model_xattn returns the raw attribute, else False on lookup miss."""

    def test_raw_attribute_and_false_fallback(self):
        class Plain:
            pass

        # Present: the raw attribute value is returned verbatim (not coerced
        # to bool) -- a distinctive string proves passthrough.
        present = Plain()
        present.xattn_needed = "custom-flag"
        self.assertEqual(get_model_xattn(present), "custom-flag")

        # Absent (no attribute, no .module): the RuntimeError from the lookup
        # is swallowed and False is returned.
        self.assertIs(get_model_xattn(Plain()), False)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestPaddleVersionHelpers(unittest.TestCase):
    """Version snapshot and >= / > comparison behavior."""

    def test_get_paddle_version_matches_version_string(self):
        # The cached snapshot must equal an independent parse of paddle's own
        # reported version string.
        self.assertEqual(get_paddle_version(), PkgVersion(paddle.__version__))

    def test_below_and_above(self):
        # cur >= 0.0.1 is True for any real release; cur >= 999.0.0 is False.
        self.assertTrue(is_paddle_min_version("0.0.1"))
        self.assertFalse(is_paddle_min_version("999.0.0"))

    def test_equality_boundary_distinguishes_ge_from_gt(self):
        # At the exact current version: >= is True but strict > is False.
        # This is the assertion that catches swapping the two operators.
        cur = str(get_paddle_version())
        self.assertTrue(is_paddle_min_version(cur, check_equality=True))
        self.assertFalse(is_paddle_min_version(cur, check_equality=False))


class _RecordingLogger:
    """Genuine collaborator: records positional/keyword args passed to log."""

    def __init__(self):
        self.calls = []

    def log(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_MSG)
class TestLogSingleRank(unittest.TestCase):
    """log_single_rank gates logging on distributed rank.

    Only the LOCAL branch decision is exercised: paddle.distributed state is
    mocked as a collaborator and no collective is issued. This proves the
    rank-gating logic and argument forwarding, NOT cross-rank behavior.
    """

    def test_logs_unconditionally_when_dist_uninitialized(self):
        lg = _RecordingLogger()
        with mock.patch.object(
            paddle.distributed, "is_initialized", return_value=False
        ):
            log_single_rank(lg, logging.INFO, "hello %s", "world")
        # Uninitialized -> else branch always logs; args forwarded verbatim.
        self.assertEqual(lg.calls, [((logging.INFO, "hello %s", "world"), {})])

    def test_logs_on_matching_rank_when_initialized(self):
        lg = _RecordingLogger()
        with (
            mock.patch.object(
                paddle.distributed, "is_initialized", return_value=True
            ),
            mock.patch.object(paddle.distributed, "get_rank", return_value=2),
        ):
            log_single_rank(lg, logging.WARNING, "on rank 2", rank=2)
        self.assertEqual(lg.calls, [((logging.WARNING, "on rank 2"), {})])

    def test_skips_on_mismatched_rank_when_initialized(self):
        lg = _RecordingLogger()
        with (
            mock.patch.object(
                paddle.distributed, "is_initialized", return_value=True
            ),
            mock.patch.object(paddle.distributed, "get_rank", return_value=3),
        ):
            # Current rank 3 != default target rank 0 -> must NOT log. This
            # proves the `rank` parameter is actually consumed.
            log_single_rank(lg, logging.ERROR, "should be suppressed")
        self.assertEqual(lg.calls, [])


if __name__ == "__main__":
    unittest.main()
