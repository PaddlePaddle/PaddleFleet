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

The class is a small registry used to assert, at the end of a refined
recompute step, that every queue that fed intermediate activations has been
fully drained. Its genuine, CPU-observable contract is pure Python (the
module itself imports no paddle; only the ``paddlefleet`` package __init__
chain pulls paddle in, hence the honest import guard below):

  * ``update(q, name)`` registers ``q`` under the exact key
    ``f"{name}_{id(q)}"`` -- the object identity is baked into the key so
    that logically-named queues are still disambiguated by object;
  * the default ``name`` is ``"unknown"``;
  * registering the *same object under the same name* twice raises
    ``ValueError`` whose message contains ``"already exists"``;
  * because the key includes ``id(q)``, two *distinct* objects sharing a
    name do NOT collide -- both are retained (a real, easy-to-miss quirk);
  * ``check()`` returns ``None`` silently when every registered queue is
    empty, and raises ``ValueError`` naming exactly the non-empty queues
    (by their full ``name_id`` key) when any queue still holds items.

Expected keys/messages are hand-derived from Python builtins (``id`` and
string formatting) -- never by calling ``update``/``check`` to produce the
value they are then compared against.
"""

import queue
import unittest

try:
    from paddlefleet.refined_recompute.queue_check import (
        RefinedRcomputeQueue,
        __all__ as QUEUE_CHECK_ALL,
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
class TestRefinedRcomputeQueue(unittest.TestCase):
    """Registration-keying and drain-check contract of RefinedRcomputeQueue."""

    def test_fresh_registry_is_empty(self):
        """A new instance registers nothing until ``update`` is called."""
        rr = RefinedRcomputeQueue()
        self.assertEqual(len(rr.rr_queue), 0)
        self.assertEqual(dict(rr.rr_queue), {})

    def test_update_uses_name_and_object_id_as_key(self):
        """Key must be exactly ``f"{name}_{id(q)}"`` and map to the same object.

        The independent expected key is built here from the ``id`` builtin,
        not from anything ``update`` returns, so a regression that dropped
        the ``id`` suffix, reordered it, or stored a copy would be caught.
        """
        rr = RefinedRcomputeQueue()
        q = queue.Queue()
        expected_key = f"named_{id(q)}"

        rr.update(q, "named")

        self.assertEqual(list(rr.rr_queue.keys()), [expected_key])
        # Identity, not just presence: the exact object must be retained so
        # a later ``check`` inspects the real queue's contents.
        self.assertIs(rr.rr_queue[expected_key], q)

    def test_update_default_name_is_unknown(self):
        """Omitting the name yields the ``unknown_{id}`` key."""
        rr = RefinedRcomputeQueue()
        q = queue.Queue()
        expected_key = f"unknown_{id(q)}"

        rr.update(q)

        self.assertEqual(list(rr.rr_queue.keys()), [expected_key])
        self.assertIs(rr.rr_queue[expected_key], q)

    def test_same_object_same_name_twice_raises(self):
        """Re-registering the identical object+name is rejected."""
        rr = RefinedRcomputeQueue()
        q = queue.Queue()
        expected_key = f"dup_{id(q)}"

        rr.update(q, "dup")
        with self.assertRaises(ValueError) as ctx:
            rr.update(q, "dup")

        message = str(ctx.exception)
        self.assertEqual(
            message, f"Queue name '{expected_key}' already exists."
        )
        # The failed second call must not have mutated the registry.
        self.assertEqual(list(rr.rr_queue.keys()), [expected_key])

    def test_distinct_objects_same_name_do_not_collide(self):
        """Two different queues sharing a name are both kept (id in key).

        This documents a real, non-obvious behavior: the duplicate guard is
        keyed on object identity, so a logical-name clash between distinct
        Queue objects is silently allowed rather than raising.
        """
        rr = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q2 = queue.Queue()

        rr.update(q1, "shared")
        rr.update(q2, "shared")  # different id -> different key, no raise

        self.assertEqual(len(rr.rr_queue), 2)
        self.assertIs(rr.rr_queue[f"shared_{id(q1)}"], q1)
        self.assertIs(rr.rr_queue[f"shared_{id(q2)}"], q2)

    def test_check_returns_none_when_all_empty(self):
        """All-empty registry passes the drain check silently."""
        rr = RefinedRcomputeQueue()
        rr.update(queue.Queue(), "a")
        rr.update(queue.Queue(), "b")

        self.assertIsNone(rr.check())

    def test_check_reports_only_the_nonempty_queue(self):
        """A single non-empty queue is named exactly; empty ones are excluded."""
        rr = RefinedRcomputeQueue()
        empty_q = queue.Queue()
        full_q = queue.Queue()
        full_q.put("activation")

        rr.update(empty_q, "empty")
        rr.update(full_q, "full")
        full_key = f"full_{id(full_q)}"

        with self.assertRaises(ValueError) as ctx:
            rr.check()

        message = str(ctx.exception)
        self.assertEqual(message, f"Queues {full_key} are not empty.")
        # The drained queue must not be blamed.
        self.assertNotIn(f"empty_{id(empty_q)}", message)

    def test_check_lists_multiple_nonempty_in_registration_order(self):
        """Every non-empty queue is listed, joined in insertion order."""
        rr = RefinedRcomputeQueue()
        q1 = queue.Queue()
        q1.put(1)
        q2 = queue.Queue()
        q2.put(2)

        rr.update(q1, "first")
        rr.update(q2, "second")
        key1 = f"first_{id(q1)}"
        key2 = f"second_{id(q2)}"

        with self.assertRaises(ValueError) as ctx:
            rr.check()

        message = str(ctx.exception)
        self.assertEqual(message, f"Queues {key1}, {key2} are not empty.")

    def test_check_reflects_live_queue_state(self):
        """``check`` inspects the retained object, so draining flips the result.

        Confirms the registry holds the real queue (not a snapshot): a queue
        that was non-empty passes once its item is consumed.
        """
        rr = RefinedRcomputeQueue()
        q = queue.Queue()
        q.put("pending")
        rr.update(q, "live")

        with self.assertRaises(ValueError):
            rr.check()

        got = q.get()
        self.assertEqual(got, "pending")
        self.assertIsNone(rr.check())


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestGlobalQueueLog(unittest.TestCase):
    """The module-level singleton and its public export."""

    def test_global_instance_is_refined_queue(self):
        self.assertIsInstance(global_rr_queue_log, RefinedRcomputeQueue)

    def test_public_export_is_exactly_the_singleton_name(self):
        # Contract: only the singleton is exported, as a tuple.
        self.assertEqual(QUEUE_CHECK_ALL, ("global_rr_queue_log",))


if __name__ == "__main__":
    unittest.main()
