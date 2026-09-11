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
"""Tests for the pinned host memory pool used by activation offloading.

Three properties here are load-bearing rather than nice to have:

- Buckets are keyed by rounded-up byte size, so activations whose row count
  changes from step to step reuse one buffer. An exact-shape key would miss on
  every allocation, and since ``free`` returns buffers to a bucket instead of
  releasing them, every miss would add permanently to the host footprint.
- A handed-out tensor keeps the pinned place and an exact shape. Reaching that
  through ``_share_buffer_to`` plus ``_set_dims`` is not an optimisation: any
  ordinary view operation moves a pinned tensor to the device instead.
- A buffer is never handed to two consumers, so a double free must be refused.
"""

from __future__ import annotations

import unittest

import paddle

from paddlefleet.activation_offload import PinnedPool
from paddlefleet.activation_offload.pinned_pool import _norm_dtype, _round_pow2

MB = 1 << 20
_REQUIRE_GPU = unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "pinned memory requires a CUDA device",
)


class TestRoundPow2(unittest.TestCase):
    def test_boundaries(self):
        got = [_round_pow2(n) for n in (0, 1, 2, 3, 5, 1 << 20, (1 << 20) + 1)]
        self.assertEqual(got, [1, 1, 2, 4, 8, 1 << 20, 1 << 21])


class TestNormDtype(unittest.TestCase):
    def test_known_string_becomes_a_dtype_object(self):
        self.assertIs(_norm_dtype("float32"), paddle.float32)

    def test_dtype_object_passes_through(self):
        self.assertIs(_norm_dtype(paddle.bfloat16), paddle.bfloat16)

    def test_unknown_string_is_returned_as_is(self):
        # An unrecognised dtype must not raise out of key computation; the pool
        # just keys on the string it was given.
        self.assertEqual(_norm_dtype("nosuchdtype"), "nosuchdtype")


@_REQUIRE_GPU
class TestBucketing(unittest.TestCase):
    def test_jittering_row_counts_share_one_buffer(self):
        """Row counts that vary per step must not each allocate a buffer.

        This is the regression line for unbounded host growth: with an
        exact-shape key every one of these rows is a miss, and misses are never
        released.
        """
        pool = PinnedPool()
        rows = [4001, 3777, 4096, 3300, 4095, 3999]
        first_ptr = None
        for r in rows:
            t = pool.alloc(
                [r, 512], paddle.float32
            )  # 6.4-8.4MB -> one 16MB bucket
            self.assertIsNotNone(t)
            self.assertEqual(list(t.shape), [r, 512])
            if first_ptr is None:
                first_ptr = t.data_ptr()
            else:
                self.assertEqual(t.data_ptr(), first_ptr, "should reuse")
            pool.free(t)
        self.assertEqual(pool.n_alloc, 1)
        self.assertEqual(pool.n_reuse, len(rows) - 1)
        self.assertEqual(pool.total_bytes, _round_pow2(4096 * 512 * 4))

    def test_capacity_counts_the_bucket_not_the_request(self):
        pool = PinnedPool()
        pool.alloc([1000, 257], paddle.float32)  # ~1.0MB, not a power of two
        self.assertEqual(pool.total_bytes, _round_pow2(1000 * 257 * 4))
        self.assertEqual(pool.waste_bytes, pool.total_bytes - 1000 * 257 * 4)

    def test_string_and_object_dtype_share_a_bucket(self):
        # The key stringifies the dtype, so alloc('float32') and free(tensor),
        # which sees a dtype object, must normalise to the same key or the pool
        # silently never hits.
        pool = PinnedPool()
        t1 = pool.alloc([1024, 1024], "float32")
        pool.free(t1)
        pool.alloc([1024, 1024], paddle.float32)
        self.assertEqual((pool.n_alloc, pool.n_reuse), (1, 1))

    def test_distinct_dtypes_do_not_share_a_bucket(self):
        pool = PinnedPool()
        pool.alloc([1024, 1024], paddle.float32)
        pool.alloc([1024, 1024], paddle.float16)
        self.assertEqual(pool.n_alloc, 2)
        self.assertEqual(pool.n_reuse, 0)


@_REQUIRE_GPU
class TestHandedOutView(unittest.TestCase):
    def test_view_keeps_the_pinned_place_and_the_exact_shape(self):
        # Reaching this through reshape or slicing would move the tensor to the
        # device and silently turn every later copy into a device-to-device one.
        pool = PinnedPool()
        t = pool.alloc([1000, 257], paddle.float32)
        self.assertEqual(list(t.shape), [1000, 257])
        self.assertIn("pinned", str(t.place).lower())

    def test_view_round_trips_values(self):
        # _set_dims does not bounds-check, so a wrong bucket size would show up
        # here as corrupted data rather than as an error.
        pool = PinnedPool()
        t = pool.alloc([1000, 257], paddle.float32)
        src = paddle.rand([1000, 257], dtype=paddle.float32)
        t.copy_(src, False)
        back = paddle.empty([1000, 257], dtype=paddle.float32)
        back.copy_(t, False)
        paddle.device.synchronize()
        self.assertTrue(bool((back == src).all()))

    def test_view_works_for_low_precision_dtypes(self):
        pool = PinnedPool()
        for dtype in (paddle.bfloat16, paddle.float16, paddle.uint8):
            t = pool.alloc([333, 129], dtype)
            self.assertIsNotNone(t, f"{dtype}")
            self.assertEqual(list(t.shape), [333, 129])
            self.assertIn("pinned", str(t.place).lower())
            pool.free(t)


@_REQUIRE_GPU
class TestCapacity(unittest.TestCase):
    def test_refuses_instead_of_exhausting_the_host(self):
        pool = PinnedPool(capacity_bytes=8 * MB)
        self.assertIsNotNone(pool.alloc([1024, 1024], paddle.float32))  # 4MB
        self.assertIsNotNone(pool.alloc([1024, 1024], paddle.float32))  # 8MB
        # Exactly at the limit is allowed; past it the caller degrades.
        self.assertIsNone(pool.alloc([2048, 1024], paddle.float32))
        self.assertEqual(pool.n_oob, 1)

    def test_a_returned_buffer_is_reused_rather_than_refused(self):
        pool = PinnedPool(capacity_bytes=8 * MB)
        t = pool.alloc([1024, 1024], paddle.float32)
        pool.alloc([1024, 1024], paddle.float32)
        pool.free(t)
        self.assertIsNotNone(pool.alloc([1024, 1024], paddle.float32))
        self.assertEqual(pool.n_alloc, 2)
        self.assertEqual(pool.n_reuse, 1)


@_REQUIRE_GPU
class TestFreeDiscipline(unittest.TestCase):
    def test_double_free_is_refused(self):
        # Accepting it would hand one buffer to two consumers, which is a silent
        # data error rather than a crash.
        pool = PinnedPool()
        t = pool.alloc([256, 256], paddle.float32)
        pool.free(t)
        pool.free(t)
        self.assertEqual(pool.n_free_dup, 1)

    def test_two_live_allocations_never_share_an_address(self):
        pool = PinnedPool()
        a = pool.alloc([256, 256], paddle.float32)
        b = pool.alloc([256, 256], paddle.float32)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_foreign_tensor_is_ignored(self):
        pool = PinnedPool()
        foreign = paddle.empty([16], dtype=paddle.float32).pin_memory()
        pool.free(foreign)
        self.assertEqual(pool.n_free_unknown, 1)

    def test_free_with_a_completed_event_can_be_reused(self):
        pool = PinnedPool()
        t = pool.alloc([512, 512], paddle.float32)
        stream = paddle.device.Stream()
        with paddle.device.stream_guard(stream):
            event = stream.record_event()
        paddle.device.synchronize()
        pool.free(t, event)
        self.assertIsNotNone(pool.alloc([512, 512], paddle.float32))
        self.assertEqual(pool.n_reuse, 1)


@_REQUIRE_GPU
class TestAccounting(unittest.TestCase):
    def test_in_use_tracks_outstanding_buffers(self):
        pool = PinnedPool()
        ts = [pool.alloc([1024, 1024], paddle.float32) for _ in range(3)]
        self.assertEqual(pool.in_use_bytes, 3 * 4 * MB)
        for t in ts:
            pool.free(t)
        self.assertEqual(pool.in_use_bytes, 0)
        # The pool keeps owning what it allocated; free returns to a bucket.
        self.assertEqual(pool.total_bytes, 3 * 4 * MB)

    def test_release_all_drops_only_idle_buffers(self):
        pool = PinnedPool()
        held = pool.alloc([1024, 1024], paddle.float32)
        idle = pool.alloc([1024, 1024], paddle.float32)
        pool.free(idle)
        pool.release_all()
        self.assertEqual(pool.total_bytes, pool.in_use_bytes)
        self.assertEqual(pool.in_use_bytes, 4 * MB)
        # An outstanding buffer stays registered, so returning it still works.
        pool.free(held)
        self.assertEqual(pool.n_free_unknown, 0)
        self.assertEqual(pool.in_use_bytes, 0)
        self.assertEqual(pool.total_bytes, 4 * MB)

    def test_clear_resets_everything(self):
        pool = PinnedPool()
        t = pool.alloc([256, 256], paddle.float32)
        pool.free(t)
        pool.clear()
        self.assertEqual(pool.total_bytes, 0)
        self.assertEqual(pool.in_use_bytes, 0)

    def test_stats_line_reports_hit_rate(self):
        pool = PinnedPool()
        t = pool.alloc([256, 256], paddle.float32)
        pool.free(t)
        pool.alloc([256, 256], paddle.float32)
        line = pool.stats_line()
        self.assertIn("hit=0.500", line)
        self.assertIn("alloc=1", line)
        self.assertIn("reuse=1", line)


@_REQUIRE_GPU
class TestPrewarm(unittest.TestCase):
    def test_prewarm_allocates_every_requested_buffer(self):
        # Written as alloc/free/alloc it would hit the buffer just returned and
        # only ever allocate one, whatever the count.
        pool = PinnedPool()
        made = pool.prewarm([([1024, 1024], paddle.float32, 4)])
        self.assertEqual(made, 4)
        self.assertEqual(pool.n_alloc, 4)
        self.assertEqual(pool.in_use_bytes, 0, "prewarm returns what it took")

    def test_prewarm_stops_at_capacity(self):
        pool = PinnedPool(capacity_bytes=8 * MB)
        made = pool.prewarm([([1024, 1024], paddle.float32, 4)])
        self.assertEqual(made, 2)
        t = pool.alloc([1024, 1024], paddle.float32)
        self.assertIsNotNone(t)
        self.assertEqual(pool.n_reuse, 1)


if __name__ == "__main__":
    unittest.main()
