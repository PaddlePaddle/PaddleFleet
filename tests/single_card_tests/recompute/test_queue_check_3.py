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

"""Behavior tests for ``RefinedRcomputeQueue`` in the refined-recompute
``queue_check`` module.

Facet under test (distinct from the base / _2 siblings, which exercise the
plain init / register / empty-check contract): the *identity-based key
composition* performed by ``update`` and the exact *content* of the failure
reported by ``check``.

The genuine, CPU-observable contract derived from the production source is:

  * ``update`` does NOT store a queue under the human name it is given; it
    stores it under the composed key ``f"{queue_name}_{id(queue)}"`` and the
    stored value is the *same* queue object (by identity).
  * Consequently the duplicate guard keys on object identity, not on the
    human name: registering two *different* queue objects under the *same*
    human name does NOT collide (their ``id`` differs), while registering the
    *same* object under the same name a second time does raise ``ValueError``.
  * ``check`` reports *every* non-empty queue, by its composed key, in
    registration (insertion) order, and silently excludes the empty ones; it
    reads the live ``qsize`` at call time, so draining a queue flips the
    result.

Expected values below are hand-derived from that contract (``id`` and the
``"{name}_{id}"`` format are computed in the test itself, never by calling the
function under test to produce its own expected value).

``paddlefleet`` imports ``paddle`` at import time (the ``refined_recompute``
package pulls in ``flash_attn`` which does ``import paddle``); ``paddle`` is
not installed in the no-card environment, so the whole suite is honestly
skipped there rather than faked green.
"""

import queue
import unittest

try:
    from paddlefleet.refined_recompute.queue_check import (
        RefinedRcomputeQueue,
        global_rr_queue_log,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet unavailable in this env
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestRefinedRcomputeQueueKeying(unittest.TestCase):
    """Identity-based keying of ``update`` and content of ``check``."""

    def test_update_stores_under_id_composed_key_by_identity(self):
        """Key is ``"{name}_{id(queue)}"`` and the value is the same object.

        A naive implementation that stored the queue under the bare human
        name, or that copied the queue, would fail both assertions below.
        """
        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        rrq.update(q, "myqueue")

        expected_key = f"myqueue_{id(q)}"
        self.assertEqual(list(rrq.rr_queue.keys()), [expected_key])
        # Stored value must be the very object passed in, not a copy.
        self.assertIs(rrq.rr_queue[expected_key], q)

    def test_update_default_name_is_unknown(self):
        """Omitting the name uses the literal default prefix ``unknown``."""
        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        rrq.update(q)
        self.assertEqual(list(rrq.rr_queue.keys()), [f"unknown_{id(q)}"])

    def test_same_name_different_objects_do_not_collide(self):
        """Dedup keys on object identity, not on the human name.

        Two distinct queue objects registered under the *same* human name get
        two distinct composed keys (their ``id`` differs), so both survive.
        This is the load-bearing consequence of embedding ``id(queue)`` in the
        key and is what distinguishes this facet from a bare name-based store.
        """
        rrq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()
        rrq.update(q1, "dup")
        rrq.update(q2, "dup")  # same human name, different object -> no clash

        key1 = f"dup_{id(q1)}"
        key2 = f"dup_{id(q2)}"
        self.assertNotEqual(key1, key2)
        self.assertEqual(set(rrq.rr_queue.keys()), {key1, key2})
        self.assertIs(rrq.rr_queue[key1], q1)
        self.assertIs(rrq.rr_queue[key2], q2)

    def test_same_object_same_name_twice_raises_and_leaves_store_intact(self):
        """Re-registering the identical object under the same name collides.

        The composed key is identical on the second call, so ``update`` raises
        ``ValueError`` with a message naming that exact composed key, and the
        original single entry is left untouched.
        """
        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        rrq.update(q, "same")
        expected_key = f"same_{id(q)}"

        with self.assertRaises(ValueError) as ctx:
            rrq.update(q, "same")
        self.assertEqual(
            str(ctx.exception),
            f"Queue name '{expected_key}' already exists.",
        )
        # The failed second update must not have mutated the store.
        self.assertEqual(list(rrq.rr_queue.keys()), [expected_key])
        self.assertIs(rrq.rr_queue[expected_key], q)


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestRefinedRcomputeQueueCheck(unittest.TestCase):
    """Content, ordering and live-``qsize`` semantics of ``check``."""

    def test_check_lists_only_nonempty_queues_by_composed_key_in_order(self):
        """Message enumerates every non-empty queue, in registration order.

        Empty queues are excluded, non-empty ones appear by their composed
        key in the order they were registered. Hand-derive the full expected
        message string so a wrong join order, a missing name, or an empty
        queue leaking into the message is all observable.
        """
        rrq = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()  # stays empty -> must be excluded
        q3 = queue.Queue()
        rrq.update(q1, "first")
        rrq.update(q2, "second")
        rrq.update(q3, "third")

        q1.put("a")
        q3.put("b")

        key1 = f"first_{id(q1)}"
        key2 = f"second_{id(q2)}"
        key3 = f"third_{id(q3)}"

        with self.assertRaises(ValueError) as ctx:
            rrq.check()
        message = str(ctx.exception)
        self.assertEqual(message, f"Queues {key1}, {key3} are not empty.")
        # The empty queue's key must not appear at all.
        self.assertNotIn(key2, message)

    def test_check_passes_silently_when_all_empty(self):
        """With multiple registered-but-empty queues ``check`` returns None."""
        rrq = RefinedRcomputeQueue()
        rrq.update(queue.Queue(), "e1")
        rrq.update(queue.Queue(), "e2")
        self.assertIsNone(rrq.check())

    def test_check_reads_live_qsize_so_draining_flips_result(self):
        """``check`` reads live ``qsize``: draining the queue makes it pass.

        Two items are enqueued (guarding against a ``qsize == 1`` short-cut),
        ``check`` must raise; after both items are removed ``check`` must pass.
        """
        rrq = RefinedRcomputeQueue()
        q = queue.Queue()
        rrq.update(q, "drain")
        q.put(10)
        q.put(20)

        with self.assertRaises(ValueError):
            rrq.check()

        self.assertEqual(q.get(), 10)
        self.assertEqual(q.get(), 20)
        self.assertEqual(q.qsize(), 0)
        self.assertIsNone(rrq.check())


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestGlobalRrQueueLog(unittest.TestCase):
    """The module-level singleton is a usable, independent instance."""

    def test_global_singleton_is_usable_and_isolated(self):
        """``global_rr_queue_log`` is a real ``RefinedRcomputeQueue``.

        Register one queue on it and confirm the composed key lands in its own
        store, without disturbing a freshly constructed instance (the two must
        not share the ``rr_queue`` mapping).
        """
        self.assertIsInstance(global_rr_queue_log, RefinedRcomputeQueue)

        fresh = RefinedRcomputeQueue()
        q = queue.Queue()
        global_rr_queue_log.update(q, "probe")

        expected_key = f"probe_{id(q)}"
        self.assertIn(expected_key, global_rr_queue_log.rr_queue)
        self.assertNotIn(expected_key, fresh.rr_queue)
        # Clean up so we do not leak state into any co-running suite.
        del global_rr_queue_log.rr_queue[expected_key]


if __name__ == "__main__":
    unittest.main()
