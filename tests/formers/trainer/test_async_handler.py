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
"""Behavior tests for AsyncCheckpointHandler (unified_checkpoint/async_handler.py).

Module under test:
    src/paddlefleet/trainer/unified_checkpoint/async_handler.py

Independent oracle strategy (no value is produced by calling the code under
test to build the expected result):

* ``_reset_and_update`` writes a UTF-8 path into a ``multiprocessing.Array("c")``
  after zeroing every byte. The oracle is hand-derived at the byte level:
  ``new_value.encode("utf-8")`` followed by ``b"\\x00"`` padding out to the
  fixed array length, matching exactly how the async worker later reads it back
  with ``arr[:].decode("utf-8").rstrip("\\x00")``. The tests deliberately write a
  long path first and a shorter path second so that a missing/broken zero-clear
  step would leave stale trailing bytes and fail -- a shape/existence check
  could not catch that. A multibyte path pins that the slice length is the
  UTF-8 *byte* length, not the character count.
* ``__init__`` derives ``global_rank`` as ``get_rank() if world_size > 1 else
  -1``. The oracle drives ``get_rank`` to a distinctive value while
  ``world_size == 1`` to prove the rank value is *not* consulted on the
  single-process branch, and separately proves it *is* consulted when
  ``world_size > 1``. The shared buffers allocated under ``async_save`` are
  exercised through the real ``_reset_and_update`` and cross-checked for
  non-aliasing (writing the model-path buffer must leave the signal-path buffer
  empty).
* ``unlink_shared_memory`` must short-circuit before any collective when
  ``async_save`` is absent; the oracle observes that ``dist.barrier`` is never
  reached.

``paddle`` is not installed in the no-card CPU environment, and the production
module imports it at load time, so the real API cannot be imported here. Those
tests skip with an explicit reason (only ``ImportError`` is treated as
"dependency unavailable"); they run for real wherever paddle is installed. The
``multiprocessing`` buffers themselves are pure-stdlib and are exercised for
real, not mocked.
"""

import os
import sys
import unittest
from unittest import mock

# Allow running from a source checkout that has not been pip-installed.
_REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
)
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_IMPORT_ERROR = None
try:
    from paddlefleet.trainer.unified_checkpoint.async_handler import (
        AsyncCheckpointHandler,
    )
except ImportError as exc:  # dependency/availability only, not API errors
    AsyncCheckpointHandler = None
    _IMPORT_ERROR = exc

_MODULE = "paddlefleet.trainer.unified_checkpoint.async_handler"


def _oracle_buffer(value, size):
    """Independent byte-level expectation for a ``multiprocessing.Array('c')``.

    Mirrors the documented contract (utf-8 payload, zero padding) without
    calling the code under test.
    """
    encoded = value.encode("utf-8")
    assert len(encoded) <= size
    return encoded + b"\x00" * (size - len(encoded))


class _HandlerImportMixin(unittest.TestCase):
    def setUp(self):
        if AsyncCheckpointHandler is None:
            self.skipTest(
                f"paddlefleet async_handler not importable "
                f"(dependency unavailable): {_IMPORT_ERROR}"
            )

    def _bare_handler(self):
        # _reset_and_update ignores instance state, so bypass __init__ (which
        # queries paddle.distributed for the rank) to exercise the pure buffer
        # logic on a real multiprocessing.Array. No attribute is set-then-read;
        # assertions observe the method's real effect on real buffers.
        return AsyncCheckpointHandler.__new__(AsyncCheckpointHandler)


class TestResetAndUpdate(_HandlerImportMixin):
    def test_writes_payload_with_zero_padding(self):
        import multiprocessing

        handler = self._bare_handler()
        arr = multiprocessing.Array("c", 32)
        handler._reset_and_update(arr, "/ckpt/model.safetensors")

        expected = _oracle_buffer("/ckpt/model.safetensors", 32)
        self.assertEqual(arr[:], expected)
        # Round-trips exactly the way the async worker decodes it.
        self.assertEqual(
            arr[:].decode("utf-8").rstrip("\x00"), "/ckpt/model.safetensors"
        )

    def test_second_shorter_write_clears_stale_tail(self):
        import multiprocessing

        handler = self._bare_handler()
        arr = multiprocessing.Array("c", 40)
        handler._reset_and_update(arr, "/very/long/output/dir/model.bin")
        handler._reset_and_update(arr, "/short")

        # If the zero-clear step were missing/broken, trailing bytes from the
        # long path would survive and this exact-byte oracle would fail.
        self.assertEqual(arr[:], _oracle_buffer("/short", 40))
        self.assertEqual(arr[:].decode("utf-8").rstrip("\x00"), "/short")

    def test_slice_length_is_utf8_byte_length(self):
        import multiprocessing

        handler = self._bare_handler()
        arr = multiprocessing.Array("c", 48)
        value = "输出/模型.bin"  # multibyte: byte length != char length
        handler._reset_and_update(arr, value)

        encoded = value.encode("utf-8")
        self.assertGreater(len(encoded), len(value))  # guard the premise
        self.assertEqual(arr[:], _oracle_buffer(value, 48))
        self.assertEqual(arr[:].decode("utf-8").rstrip("\x00"), value)

    def test_empty_value_zeroes_entire_buffer(self):
        import multiprocessing

        handler = self._bare_handler()
        arr = multiprocessing.Array("c", 16)
        handler._reset_and_update(arr, "/prefilled/path")
        handler._reset_and_update(arr, "")
        self.assertEqual(arr[:], b"\x00" * 16)


class TestInitGlobalRank(_HandlerImportMixin):
    def _make(self, config, rank, world_size):
        args = mock.MagicMock()
        args.unified_checkpoint_config = config
        with (
            mock.patch(
                f"{_MODULE}.paddle.distributed.get_rank", return_value=rank
            ),
            mock.patch(
                f"{_MODULE}.paddle.distributed.get_world_size",
                return_value=world_size,
            ),
        ):
            return AsyncCheckpointHandler(args)

    def test_single_process_rank_is_minus_one_ignoring_get_rank(self):
        # world_size == 1 -> the get_rank() value (7) must NOT be used.
        handler = self._make({}, rank=7, world_size=1)
        self.assertEqual(handler.global_rank, -1)

    def test_multi_process_rank_uses_get_rank(self):
        handler = self._make({}, rank=3, world_size=4)
        self.assertEqual(handler.global_rank, 3)

    def test_no_async_leaves_buffers_unallocated(self):
        handler = self._make({}, rank=0, world_size=1)
        self.assertIsNone(handler._lock)
        self.assertIsNone(handler._shm_model_weight)
        self.assertIsNone(handler._shm_master_weight)
        self.assertIsNone(handler._shm_optimizer_weight)
        self.assertIsNone(handler._shared_save_model_flag)
        self.assertIsNone(handler._shared_save_master_weight_flag)
        self.assertIsNone(handler._shared_save_optimizer_flag)
        # Path buffers are only created inside the async branch.
        self.assertFalse(hasattr(handler, "_shared_save_model_path"))


class TestInitAsyncBuffers(_HandlerImportMixin):
    def _make_async(self):
        args = mock.MagicMock()
        args.unified_checkpoint_config = {"async_save": True}
        with (
            mock.patch(
                f"{_MODULE}.paddle.distributed.get_rank", return_value=0
            ),
            mock.patch(
                f"{_MODULE}.paddle.distributed.get_world_size", return_value=1
            ),
        ):
            return AsyncCheckpointHandler(args)

    def test_flags_allocated_and_initialised_to_zero(self):
        handler = self._make_async()
        self.assertIsNotNone(handler._lock)
        for flag in (
            handler._shared_save_model_flag,
            handler._shared_save_master_weight_flag,
            handler._shared_save_optimizer_flag,
        ):
            self.assertEqual(len(flag), 1)
            self.assertEqual(flag[0], 0)  # nothing pending on construction

    def test_path_buffers_are_distinct_and_usable(self):
        handler = self._make_async()
        model_path = handler._shared_save_model_path
        signal_path = handler._shared_save_model_signal_path

        # Real production-allocated buffers, correct fixed size, not aliased.
        self.assertEqual(len(model_path), 100000)
        self.assertEqual(len(signal_path), 100000)
        self.assertIsNot(model_path, signal_path)

        # Writing the model path via the real method must not bleed into the
        # signal-path buffer -> catches accidental aliasing of the two Arrays.
        handler._reset_and_update(model_path, "/ckpt/step-100/model")
        self.assertEqual(
            model_path[:].decode("utf-8").rstrip("\x00"),
            "/ckpt/step-100/model",
        )
        self.assertEqual(signal_path[:], b"\x00" * 100000)


class TestUnlinkSharedMemory(_HandlerImportMixin):
    def test_no_async_short_circuits_before_barrier(self):
        args = mock.MagicMock()
        args.unified_checkpoint_config = {}
        with (
            mock.patch(
                f"{_MODULE}.paddle.distributed.get_rank", return_value=0
            ),
            mock.patch(
                f"{_MODULE}.paddle.distributed.get_world_size", return_value=1
            ),
        ):
            handler = AsyncCheckpointHandler(args)

        # The guard must return before touching any collective/state.
        with (
            mock.patch(f"{_MODULE}.dist.barrier") as barrier,
            mock.patch(
                f"{_MODULE}.paddle.distributed.get_world_size", return_value=8
            ),
        ):
            result = handler.unlink_shared_memory()

        self.assertIsNone(result)
        barrier.assert_not_called()
        self.assertIsNone(handler._shm_model_weight)


if __name__ == "__main__":
    unittest.main()
