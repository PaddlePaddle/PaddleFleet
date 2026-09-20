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

"""Behavior tests for paddlefleet.nn.general.GeneralInterface.

GeneralInterface is a pure-Python MutableMapping (no Paddle / no accelerator
involved) that keeps a class-wide `_global_mapping` plus a per-instance
`_local_mapping`. The local mapping shadows the global one on read, `register`
publishes into the shared class mapping, and `__setitem__`/`__delitem__` only
ever touch the local mapping. These tests pin those exact behaviors with
independent, hand-derived expectations. All of it is CPU/Python-only, so there
is no device numeric path to exercise or to skip.
"""

import unittest

from paddlefleet.nn.general import GeneralInterface


def make_interface_cls():
    """Return a fresh GeneralInterface subclass with its own empty global map.

    Each call produces a brand-new class object with a brand-new
    `_global_mapping` dict. Using a dedicated subclass keeps the production
    `GeneralInterface._global_mapping` untouched (mutating shared production
    class state would leak across tests) while exercising the exact inherited
    method implementations under test.
    """

    class _ProbeInterface(GeneralInterface):
        _global_mapping = {}

    return _ProbeInterface


class TestGeneralInterfaceLookup(unittest.TestCase):
    """__getitem__ precedence, fallback and missing-key contract."""

    def test_global_fallback_when_no_local_override(self):
        cls = make_interface_cls()
        cls.register("g", "gval")
        obj = cls()
        self.assertEqual(obj["g"], "gval")

    def test_local_override_takes_precedence_over_global(self):
        cls = make_interface_cls()
        cls.register("shared", "GLOBAL")
        obj = cls()
        # Before any local override, the instance sees the global value.
        self.assertEqual(obj["shared"], "GLOBAL")
        obj["shared"] = "LOCAL"
        # After a local write, the local value wins on read.
        self.assertEqual(obj["shared"], "LOCAL")
        # The shared global mapping itself must not be mutated by the override.
        self.assertEqual(cls._global_mapping["shared"], "GLOBAL")

    def test_missing_key_raises_key_error(self):
        obj = make_interface_cls()()
        with self.assertRaises(KeyError):
            _ = obj["absent"]

    def test_registered_callable_is_retrievable_and_callable(self):
        cls = make_interface_cls()
        cls.register("double", lambda x: x * 2)
        obj = cls()
        fn = obj["double"]
        # The stored object is the real callable and computes the real result.
        self.assertEqual(fn(21), 42)


class TestGeneralInterfaceRegisterSharing(unittest.TestCase):
    """register publishes into the class-wide mapping shared by all instances."""

    def test_register_visible_to_instances_created_before_and_after(self):
        cls = make_interface_cls()
        before = cls()
        cls.register("k", "shared_value")
        after = cls()
        # A single register call is reflected in every instance of the class,
        # regardless of when the instance was created.
        self.assertEqual(before["k"], "shared_value")
        self.assertEqual(after["k"], "shared_value")

    def test_local_override_does_not_affect_siblings_or_global(self):
        cls = make_interface_cls()
        cls.register("shared", "G")
        a = cls()
        b = cls()
        a["shared"] = "A"
        # Only the instance that set the local override sees it; the sibling and
        # the class-wide mapping keep the original global value.
        self.assertEqual(a["shared"], "A")
        self.assertEqual(b["shared"], "G")
        self.assertEqual(cls._global_mapping["shared"], "G")

    def test_two_subclasses_have_independent_global_mappings(self):
        cls_a = make_interface_cls()
        cls_b = make_interface_cls()
        cls_a.register("only_a", 1)
        # Each subclass declares its own `_global_mapping`, so a register on one
        # does not bleed into the other.
        self.assertEqual(cls_a()["only_a"], 1)
        with self.assertRaises(KeyError):
            _ = cls_b()["only_a"]


class TestGeneralInterfaceSetDelete(unittest.TestCase):
    """__setitem__ and __delitem__ operate only on the local mapping."""

    def test_setitem_is_instance_local(self):
        cls = make_interface_cls()
        a = cls()
        b = cls()
        a["only_a"] = 7
        self.assertEqual(a["only_a"], 7)
        # A local write must not leak into the class-wide mapping...
        self.assertNotIn("only_a", cls._global_mapping)
        # ...nor into a sibling instance's view.
        with self.assertRaises(KeyError):
            _ = b["only_a"]

    def test_delete_local_override_reexposes_global(self):
        cls = make_interface_cls()
        cls.register("k", "GLOBAL")
        obj = cls()
        obj["k"] = "LOCAL"
        self.assertEqual(obj["k"], "LOCAL")
        del obj["k"]
        # Deleting only removes the local shadow; the global value reappears.
        self.assertEqual(obj["k"], "GLOBAL")
        self.assertEqual(cls._global_mapping["k"], "GLOBAL")

    def test_delete_global_only_key_raises(self):
        cls = make_interface_cls()
        cls.register("g", "gval")
        obj = cls()
        # __delitem__ targets the local mapping exclusively, so a key that lives
        # only in the global mapping cannot be deleted through an instance.
        with self.assertRaises(KeyError):
            del obj["g"]
        # The failed delete leaves the global entry intact and still readable.
        self.assertEqual(obj["g"], "gval")

    def test_delete_missing_key_raises(self):
        obj = make_interface_cls()()
        with self.assertRaises(KeyError):
            del obj["absent"]


class TestGeneralInterfaceIterAndLen(unittest.TestCase):
    """__iter__ and __len__ combine the two mappings without double counting."""

    def test_iter_yields_global_then_local_only_in_order(self):
        cls = make_interface_cls()
        cls.register("g1", 1)
        cls.register("g2", 2)
        obj = cls()
        obj["g1"] = 10  # overrides an existing global key
        obj["l1"] = 11  # local-only key
        # Iteration is over {**global, **local}: an overridden key keeps its
        # original global position, and local-only keys are appended afterward.
        self.assertEqual(list(iter(obj)), ["g1", "g2", "l1"])

    def test_len_counts_all_unique_keys(self):
        cls = make_interface_cls()
        cls.register("g1", None)
        cls.register("g2", None)
        obj = cls()
        obj["l1"] = None
        self.assertEqual(len(obj), 3)  # g1, g2, l1

    def test_len_dedups_overlapping_keys(self):
        cls = make_interface_cls()
        cls.register("shared", "G")
        obj = cls()
        obj["shared"] = "L"  # same key present in both mappings
        obj["extra"] = "E"
        # {shared, extra} -> a key present in both mappings is counted once.
        self.assertEqual(len(obj), 2)

    def test_valid_keys_matches_iteration(self):
        cls = make_interface_cls()
        cls.register("g1", None)
        cls.register("g2", None)
        obj = cls()
        obj["l1"] = None
        keys = obj.valid_keys()
        self.assertIsInstance(keys, list)
        self.assertEqual(keys, ["g1", "g2", "l1"])

    def test_empty_interface(self):
        obj = make_interface_cls()()
        self.assertEqual(len(obj), 0)
        self.assertEqual(list(iter(obj)), [])
        self.assertEqual(obj.valid_keys(), [])


class TestGeneralInterfaceMutableMappingContract(unittest.TestCase):
    """Behaviors the class delivers via MutableMapping on top of its dunders."""

    def test_contains_checks_both_mappings(self):
        cls = make_interface_cls()
        cls.register("g", "gval")
        obj = cls()
        obj["l"] = "lval"
        self.assertIn("g", obj)
        self.assertIn("l", obj)
        self.assertNotIn("absent", obj)

    def test_get_returns_default_for_missing(self):
        cls = make_interface_cls()
        cls.register("g", "gval")
        obj = cls()
        self.assertEqual(obj.get("g"), "gval")
        self.assertIsNone(obj.get("absent"))
        self.assertEqual(obj.get("absent", "dflt"), "dflt")

    def test_update_writes_to_local_only(self):
        cls = make_interface_cls()
        obj = cls()
        obj.update({"a": 1, "b": 2})
        self.assertEqual(obj["a"], 1)
        self.assertEqual(obj["b"], 2)
        # MutableMapping.update routes through __setitem__, so nothing lands in
        # the class-wide mapping.
        self.assertNotIn("a", cls._global_mapping)
        self.assertNotIn("b", cls._global_mapping)

    def test_pop_local_key_returns_value_and_removes_it(self):
        cls = make_interface_cls()
        obj = cls()
        obj["k"] = "v"
        self.assertEqual(obj.pop("k"), "v")
        with self.assertRaises(KeyError):
            _ = obj["k"]

    def test_pop_global_only_key_raises_due_to_local_only_delete(self):
        cls = make_interface_cls()
        cls.register("g", "gval")
        obj = cls()
        # MutableMapping.pop reads via __getitem__ (finds the global value) and
        # then calls __delitem__, which only removes from the local mapping.
        # The key is absent locally, so the delete raises KeyError and the pop
        # fails even though the value existed globally.
        with self.assertRaises(KeyError):
            obj.pop("g")
        # The global entry survives the failed pop.
        self.assertEqual(obj["g"], "gval")


if __name__ == "__main__":
    unittest.main()
