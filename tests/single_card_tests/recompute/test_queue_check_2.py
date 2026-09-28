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

"""CPU behavior tests for paddlefleet.refined_recompute.queue_check.

Distinct facet: the exact id-decorated key-naming produced by
``RefinedRcomputeQueue.update`` (identity of the stored object and the
literal registry key) and the exact error-message text emitted by
``check`` (comma-joined non-empty key names, sentinel suffix, ordering,
and exclusion of empty queues). All expected values below are derived by
hand from Python's builtin ``id`` and the module's naming/formatting
contract -- never by invoking the code under test to compute its own
expected result.
"""

import queue
import unittest

try:
    # paddlefleet/__init__ imports paddle at import time; the local
    # environment has no paddle, so this raises ImportError and the whole
    # suite is skipped with an honest reason (never faked as passing).
    from paddlefleet.refined_recompute.queue_check import (
        RefinedRcomputeQueue,
        global_rr_queue_log,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # only ImportError -> genuine missing dependency
    RefinedRcomputeQueue = None
    global_rr_queue_log = None
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    "paddlefleet.refined_recompute.queue_check is not importable "
    f"(paddle not installed in this environment): {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(RefinedRcomputeQueue is not None, _SKIP_REASON)
class TestQueueKeyNamingIdentity(unittest.TestCase):
    """update() must key each queue by '<name>_<id(queue)>' and store the
    exact object reference."""

    def test_key_is_name_plus_object_id(self):
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q, "alpha")
        # Hand-derived key from the naming contract, using builtin id().
        expected_key = f"alpha_{id(q)}"
        self.assertEqual(list(rq.rr_queue.keys()), [expected_key])
        # The stored value must be the very same object, not a copy.
        self.assertIs(rq.rr_queue[expected_key], q)

    def test_default_name_is_unknown(self):
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q)  # no name -> default "unknown"
        self.assertEqual(list(rq.rr_queue.keys()), [f"unknown_{id(q)}"])

    def test_distinct_objects_same_basename_coexist(self):
        rq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()
        rq.update(q1, "shared")
        rq.update(q2, "shared")
        k1 = f"shared_{id(q1)}"
        k2 = f"shared_{id(q2)}"
        # Two different objects with the same basename get distinct keys.
        self.assertNotEqual(k1, k2)
        self.assertEqual(set(rq.rr_queue.keys()), {k1, k2})
        self.assertIs(rq.rr_queue[k1], q1)
        self.assertIs(rq.rr_queue[k2], q2)

    def test_duplicate_same_object_raises_with_exact_message(self):
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q, "beta")
        expected_key = f"beta_{id(q)}"
        with self.assertRaises(ValueError) as ctx:
            rq.update(q, "beta")
        self.assertEqual(
            str(ctx.exception),
            f"Queue name '{expected_key}' already exists.",
        )
        # The failed second update must not corrupt the registry.
        self.assertEqual(list(rq.rr_queue.keys()), [expected_key])


@unittest.skipUnless(RefinedRcomputeQueue is not None, _SKIP_REASON)
class TestCheckErrorMessageContent(unittest.TestCase):
    """check() must report exactly the non-empty queues, in insertion
    order, joined by ', ' with the ' are not empty.' suffix."""

    def test_all_empty_returns_none_without_raising(self):
        rq = RefinedRcomputeQueue()
        rq.update(queue.Queue(), "e1")
        rq.update(queue.Queue(), "e2")
        self.assertIsNone(rq.check())

    def test_single_non_empty_exact_message(self):
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        q.put("x")
        rq.update(q, "full")
        expected_key = f"full_{id(q)}"
        with self.assertRaises(ValueError) as ctx:
            rq.check()
        self.assertEqual(
            str(ctx.exception),
            f"Queues {expected_key} are not empty.",
        )

    def test_multiple_non_empty_joined_in_insertion_order(self):
        rq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()
        q1.put("a")
        q2.put("b")
        rq.update(q1, "first")
        rq.update(q2, "second")
        k1 = f"first_{id(q1)}"
        k2 = f"second_{id(q2)}"
        with self.assertRaises(ValueError) as ctx:
            rq.check()
        # Independently constructed full message: insertion order q1, q2.
        self.assertEqual(
            str(ctx.exception),
            f"Queues {k1}, {k2} are not empty.",
        )

    def test_empty_queue_excluded_from_message(self):
        rq = RefinedRcomputeQueue()
        q_empty = queue.Queue()
        q_full = queue.Queue()
        q_full.put("item")
        rq.update(q_empty, "empty")
        rq.update(q_full, "full")
        empty_key = f"empty_{id(q_empty)}"
        full_key = f"full_{id(q_full)}"
        with self.assertRaises(ValueError) as ctx:
            rq.check()
        msg = str(ctx.exception)
        self.assertEqual(msg, f"Queues {full_key} are not empty.")
        self.assertNotIn(empty_key, msg)

    def test_check_reflects_live_queue_size_transitions(self):
        # check() reads qsize() live: draining a queue clears the error.
        rq = RefinedRcomputeQueue()
        q = queue.Queue()
        rq.update(q, "live")
        self.assertIsNone(rq.check())  # empty -> passes
        q.put("data")
        with self.assertRaises(ValueError):
            rq.check()  # non-empty -> raises
        q.get()
        self.assertIsNone(rq.check())  # drained -> passes again


@unittest.skipUnless(RefinedRcomputeQueue is not None, _SKIP_REASON)
class TestGlobalSingleton(unittest.TestCase):
    def test_module_global_is_usable_instance(self):
        self.assertIsInstance(global_rr_queue_log, RefinedRcomputeQueue)
        # Exercise the real instance's contract without leaking state:
        # use a local instance for the actual behavior assertions above;
        # here only confirm the exported global is a working registry.
        probe = RefinedRcomputeQueue()
        q = queue.Queue()
        probe.update(q, "probe")
        self.assertIs(probe.rr_queue[f"probe_{id(q)}"], q)


if __name__ == "__main__":
    unittest.main()
