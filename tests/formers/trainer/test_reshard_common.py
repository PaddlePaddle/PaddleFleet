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

"""Behavioral unit tests for ``trainer/utils/reshard/common.py``.

Scope: the CPU-runnable, communication-free logic of the reshard layer -- the
optimizer-name -> tensor-name mapping, the ``NodeModelState`` key/rank
bookkeeping, and the module-level split/merge routing helpers. The bucketed
broadcast collectives (``all_gather_state_dict`` / ``all_gather_on_device``) need
a real process group and are out of scope here.

Independent oracle. No expectation is produced by calling the function under
test. Each expected value is hand-derived:

  * ``convert_opt_name_to_tname``: the tensor name is the opt name with its
    longest matching known suffix (``_moment1_0``, ``_beta2_pow_acc_0``,
    ``_fp32_master_0_moment1_0`` ...) removed. Suffix priority is explicit in the
    production ``suffix`` list, so ``x_fp32_master_0_moment1_0`` must resolve to
    ``x`` -- never to ``x_fp32_master_0``.
  * ``NodeModelState.map_names``: an opt key ``(sn, tn, opt_name)`` renames to
    ``(sn, new_tn, new_tn + opt_name[len(tn):])`` -- the algorithmic suffix that
    trails the tensor name is preserved verbatim.
  * split/merge helpers: routing is decided solely by ``group_getter.get_group``,
    and every payload must land in the group of its own key.

Content vs. shape. Parameters deliberately share the same shape but carry
distinct values, so an ownership/routing swap (w1 <-> w2) is caught by content
comparison and would not slip through a shape-only check.

Production bug pinned (no production edit): ``NodeModelState.split_state`` calls
``NodeModelState()`` with no argument, but ``__init__`` requires ``group``; the
test asserting correct per-rank splitting is marked ``expectedFailure``.
"""

import unittest
from collections import OrderedDict

import numpy as np

try:
    import paddle

    from paddlefleet.trainer.utils.reshard.common import (
        SHARDING_STRATEGY_V1,
        SHARDING_STRATEGY_V2,
        NodeModelState,
        convert_opt_name_to_tname,
        is_sharding_opt,
        merge_model_state,
        merge_opt_state,
        split_model_state,
        split_opt_state,
        split_structure_name_mapping,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc


_skip_no_paddle = unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle / paddlefleet reshard.common unavailable: {_IMPORT_ERROR}",
)


def _t(values):
    """A float32 CPU tensor with fully specified, distinguishable content."""
    return paddle.to_tensor(np.array(values, dtype="float32"))


class _Group:
    """Minimal stand-in for a communication group: only ``id`` is consulted."""

    def __init__(self, gid):
        self.id = gid


class _Getter:
    """Real routing collaborator: maps each key to a fixed ``_Group``.

    Distinct group ids per key make the routing observable; a MagicMock that
    returns the same ``id`` for every key would collapse all keys into one
    bucket and hide misrouting.
    """

    def __init__(self, mapping):
        self._mapping = mapping

    def get_group(self, key):
        return self._mapping[key]


@_skip_no_paddle
class TestShardingStrategyConstants(unittest.TestCase):
    def test_constant_values(self):
        # Oracle: the two persisted strategy tags are fixed protocol strings.
        self.assertEqual(SHARDING_STRATEGY_V1, "ShardingV1")
        self.assertEqual(SHARDING_STRATEGY_V2, "ShardingV2")
        self.assertNotEqual(SHARDING_STRATEGY_V1, SHARDING_STRATEGY_V2)


@_skip_no_paddle
class TestConvertOptNameToTname(unittest.TestCase):
    def test_moment_suffixes_strip_to_tensor_name(self):
        result = convert_opt_name_to_tname(
            ["linear.weight"],
            ["linear.weight_moment1_0", "linear.weight_moment2_0"],
        )
        self.assertEqual(result["linear.weight_moment1_0"], "linear.weight")
        self.assertEqual(result["linear.weight_moment2_0"], "linear.weight")

    def test_beta_pow_suffixes_strip_to_tensor_name(self):
        result = convert_opt_name_to_tname(
            ["linear.weight"],
            [
                "linear.weight_beta1_pow_acc_0",
                "linear.weight_beta2_pow_acc_0",
            ],
        )
        self.assertEqual(
            result["linear.weight_beta1_pow_acc_0"], "linear.weight"
        )
        self.assertEqual(
            result["linear.weight_beta2_pow_acc_0"], "linear.weight"
        )

    def test_fp32_master_suffix_takes_priority_over_short_suffix(self):
        # The full "_fp32_master_0_moment1_0" precedes "_moment1_0" in the
        # production suffix list, so it must win. Stripping only "_moment1_0"
        # would wrongly leave "linear.weight_fp32_master_0".
        result = convert_opt_name_to_tname(
            ["linear.weight"],
            ["linear.weight_fp32_master_0_moment1_0"],
        )
        self.assertEqual(
            result["linear.weight_fp32_master_0_moment1_0"], "linear.weight"
        )
        self.assertNotEqual(
            result["linear.weight_fp32_master_0_moment1_0"],
            "linear.weight_fp32_master_0",
        )

    def test_each_opt_state_maps_to_its_own_param_no_swap(self):
        # Two params; every opt slot must resolve to its own owner. A swapped
        # mapping (A's moment -> B) would flip these equalities.
        result = convert_opt_name_to_tname(
            ["layerA.weight", "layerB.weight"],
            [
                "layerA.weight_moment1_0",
                "layerA.weight_beta1_pow_acc_0",
                "layerB.weight_moment2_0",
                "layerB.weight_beta2_pow_acc_0",
            ],
        )
        self.assertEqual(result["layerA.weight_moment1_0"], "layerA.weight")
        self.assertEqual(
            result["layerA.weight_beta1_pow_acc_0"], "layerA.weight"
        )
        self.assertEqual(result["layerB.weight_moment2_0"], "layerB.weight")
        self.assertEqual(
            result["layerB.weight_beta2_pow_acc_0"], "layerB.weight"
        )

    def test_unknown_suffix_raises(self):
        # No known suffix matches -> the production `assert _find, t` fires.
        with self.assertRaises(AssertionError):
            convert_opt_name_to_tname(
                ["linear.weight"], ["linear.weight_unknown_slot"]
            )


@_skip_no_paddle
class TestIsShardingOpt(unittest.TestCase):
    def test_plain_optimizer_is_not_sharding(self):
        # Real dispatch through unwrap_optimizer: a bare object has no
        # `_inner_opt` and is not a sharding-optimizer instance -> False.
        class _Plain:
            pass

        self.assertFalse(is_sharding_opt(_Plain()))

    def test_inner_opt_chain_without_sharding_class_is_false(self):
        # The unwrap loop walks `_inner_opt` but the terminal object is still
        # not a recognized sharding optimizer, so the answer stays False.
        class _Plain:
            pass

        class _Wrapper:
            def __init__(self, inner):
                self._inner_opt = inner

        self.assertFalse(is_sharding_opt(_Wrapper(_Wrapper(_Plain()))))


@_skip_no_paddle
class TestNodeModelStateBookkeeping(unittest.TestCase):
    def test_init_is_empty(self):
        state = NodeModelState(group=None)
        self.assertEqual(len(state.model_weights), 0)
        self.assertEqual(len(state.opt_state), 0)
        self.assertEqual(len(state.master_weights), 0)
        self.assertIsNone(state.lr_scheduler)
        self.assertIsNone(state.group)

    def test_add_weight_rejects_duplicate_key(self):
        state = NodeModelState(group=None)
        state.add_weight("w", _t([[1.0, 2.0], [3.0, 4.0]]))
        with self.assertRaises(AssertionError):
            state.add_weight("w", _t([[9.0, 9.0], [9.0, 9.0]]))

    def test_add_weights_with_rank_keys_and_preserves_content(self):
        # Same shape, distinct content: a key/rank swap would be caught by the
        # value comparison, not just the key presence.
        state = NodeModelState(group=None)
        w1 = _t([[1.0, 2.0], [3.0, 4.0]])
        w2 = _t([[5.0, 6.0], [7.0, 8.0]])
        state.add_weights(OrderedDict([("w1", w1), ("w2", w2)]), rank=3)
        self.assertIn(("w1", 3), state.model_weights)
        self.assertIn(("w2", 3), state.model_weights)
        np.testing.assert_array_equal(
            state.model_weights[("w1", 3)].numpy(), w1.numpy()
        )
        np.testing.assert_array_equal(
            state.model_weights[("w2", 3)].numpy(), w2.numpy()
        )

    def test_add_opts_splits_streams_and_consumes_input(self):
        state = NodeModelState(group=None)
        moment = _t([[1.0, 1.0], [1.0, 1.0]])
        master = _t([[2.0, 2.0], [2.0, 2.0]])
        sched = object()
        opts = OrderedDict(
            [
                ("moment1", moment),
                ("master_weights", OrderedDict([("mw", master)])),
                ("LR_Scheduler", sched),
            ]
        )
        state.add_opts(opts)
        # moment routed to opt_state; master to master_weights; lr captured.
        np.testing.assert_array_equal(
            state.opt_state["moment1"].numpy(), moment.numpy()
        )
        np.testing.assert_array_equal(
            state.master_weights["mw"].numpy(), master.numpy()
        )
        self.assertIs(state.lr_scheduler, sched)
        # add_opts pops the two special keys out of the caller's dict.
        self.assertNotIn("master_weights", opts)
        self.assertNotIn("LR_Scheduler", opts)

    def test_set_lr_scheduler_none_does_not_clear(self):
        state = NodeModelState(group=None)
        sched = object()
        state.set_lr_scheduler(sched)
        state.set_lr_scheduler(None)
        self.assertIs(state.lr_scheduler, sched)

    def test_get_opt_state_dict_composition(self):
        state = NodeModelState(group=None)
        moment = _t([1.0, 2.0, 3.0])
        master = _t([4.0, 5.0, 6.0])
        sched = object()
        state.add_opt("moment1", moment)
        state.add_master_weight("mw", master)
        state.set_lr_scheduler(sched)
        out = state.get_opt_state_dict()
        np.testing.assert_array_equal(out["moment1"].numpy(), moment.numpy())
        self.assertIs(out["LR_Scheduler"], sched)
        np.testing.assert_array_equal(
            out["master_weights"]["mw"].numpy(), master.numpy()
        )

    def test_get_opt_state_dict_omits_lr_when_unset(self):
        state = NodeModelState(group=None)
        state.add_opt("moment1", _t([1.0]))
        out = state.get_opt_state_dict()
        self.assertNotIn("LR_Scheduler", out)
        self.assertIn("master_weights", out)


@_skip_no_paddle
class TestNodeModelStateKeyTransforms(unittest.TestCase):
    def test_map_names_preserves_opt_and_master_suffix(self):
        # Rename the tensor name; the algorithmic suffix trailing it in the opt
        # / master keys must survive verbatim.
        state = NodeModelState(group=None)
        state._model_weights = OrderedDict(
            [(("struct.w", "param0"), _t([[1.0, 2.0], [3.0, 4.0]]))]
        )
        state._opt_state = OrderedDict(
            [(("struct.w", "param0", "param0_moment1_0"), _t([1.0, 2.0]))]
        )
        state._master_weights = OrderedDict(
            [(("struct.w", "param0", "param0"), _t([3.0, 4.0]))]
        )

        state.map_names(lambda structure_name, t_name: t_name + "_X")

        self.assertIn(("struct.w", "param0_X"), state.model_weights)
        self.assertIn(
            ("struct.w", "param0_X", "param0_X_moment1_0"),
            state.opt_state,
        )
        self.assertIn(
            ("struct.w", "param0_X", "param0_X"),
            state.master_weights,
        )

    def test_drop_rank_removes_rank_from_keys(self):
        state = NodeModelState(group=None)
        wa = _t([[1.0, 2.0], [3.0, 4.0]])
        state._model_weights = OrderedDict([((("s", "w"), 0), wa)])
        state._opt_state = OrderedDict(
            [((("s", "w", "w_moment1_0"), 0), _t([1.0]))]
        )
        state._master_weights = OrderedDict([((("s", "w", "w"), 0), _t([2.0]))])
        state.drop_rank()
        self.assertIn(("s", "w"), state.model_weights)
        np.testing.assert_array_equal(
            state.model_weights[("s", "w")].numpy(), wa.numpy()
        )
        self.assertIn(("s", "w", "w_moment1_0"), state.opt_state)
        self.assertIn(("s", "w", "w"), state.master_weights)

    def test_collapse_then_flatten_roundtrips_content_and_rank(self):
        state = NodeModelState(group=None)
        r0 = _t([[1.0, 2.0], [3.0, 4.0]])
        r1 = _t([[5.0, 6.0], [7.0, 8.0]])
        # Insert out of rank order to check collapse sorts by (key, rank).
        state._model_weights = OrderedDict(
            [((("s", "w"), 1), r1), ((("s", "w"), 0), r0)]
        )
        state.collapse_key()
        collapsed = state.model_weights[("s", "w")]
        self.assertEqual([rank for rank, _ in collapsed], [0, 1])
        np.testing.assert_array_equal(collapsed[0][1].numpy(), r0.numpy())
        np.testing.assert_array_equal(collapsed[1][1].numpy(), r1.numpy())

        state.flatten_key()
        np.testing.assert_array_equal(
            state.model_weights[(("s", "w"), 0)].numpy(), r0.numpy()
        )
        np.testing.assert_array_equal(
            state.model_weights[(("s", "w"), 1)].numpy(), r1.numpy()
        )

    def test_merge_items_sorts_by_rank_before_merge(self):
        state = NodeModelState(group=None)
        ta = _t([1.0])
        tb = _t([2.0])
        # Provide list in descending rank; merge_func must see ascending rank.
        state._model_weights = OrderedDict([(("s", "w"), [(1, tb), (0, ta)])])
        seen = {}

        def merge_func(key, items):
            seen[key] = [rank for rank, _ in items]
            return items[0][1]

        state.merge_items(merge_func)
        self.assertEqual(seen[("s", "w")], [0, 1])

    def test_split_items_applies_func_per_key(self):
        state = NodeModelState(group=None)
        w = _t([[1.0, 2.0], [3.0, 4.0]])
        state._model_weights = OrderedDict([(("s", "w"), w)])
        captured = {}

        def split_func(key, value):
            captured["key"] = key
            return [(0, value)]

        state.split_items(split_func)
        self.assertEqual(captured["key"], ("s", "w"))
        self.assertEqual(len(state.model_weights[("s", "w")]), 1)

    def test_merge_from_combines_states_under_rank(self):
        base = NodeModelState(group=None)
        other = NodeModelState(group=None)
        w = _t([[1.0, 2.0], [3.0, 4.0]])
        m = _t([1.0, 2.0])
        mw = _t([3.0, 4.0])
        other.add_weight("w", w)
        other.add_opt("moment1", m)
        other.add_master_weight("mw", mw)
        base.merge_from(other, rank=7)
        self.assertIn(("w", 7), base.model_weights)
        self.assertIn(("moment1", 7), base.opt_state)
        self.assertIn(("mw", 7), base.master_weights)
        np.testing.assert_array_equal(
            base.model_weights[("w", 7)].numpy(), w.numpy()
        )

    def test_merge_from_requires_matching_group(self):
        g = object()
        base = NodeModelState(group=g)
        other = NodeModelState(group=object())
        with self.assertRaises(AssertionError):
            base.merge_from(other)

    @unittest.expectedFailure
    def test_split_state_distributes_by_rank(self):
        # INTENDED behavior: split_state should fan model/opt/master entries out
        # into one NodeModelState per rank returned by split_func.
        # PRODUCTION BUG: split_state constructs ``NodeModelState()`` with no
        # argument, but __init__ requires ``group`` -> TypeError. Marked
        # expectedFailure; a fix (giving the child a group) makes it pass.
        state = NodeModelState(group=None)
        state.add_weight("w0", _t([1.0]))
        state.add_weight("w1", _t([2.0]))
        result = state.split_state(lambda key: 0 if key == "w0" else 1)
        self.assertIn("w0", result[0].model_weights)
        self.assertIn("w1", result[1].model_weights)


@_skip_no_paddle
class TestModuleLevelSplitMerge(unittest.TestCase):
    def test_split_model_state_routes_each_key_to_its_group(self):
        w1 = _t([[1.0, 2.0], [3.0, 4.0]])
        w2 = _t([[5.0, 6.0], [7.0, 8.0]])  # same shape, distinct content
        getter = _Getter({"w1": _Group(10), "w2": _Group(20)})
        result = split_model_state(
            OrderedDict([("w1", w1), ("w2", w2)]), getter
        )
        self.assertEqual(set(result), {10, 20})
        # Ownership: each param lands in its own group with its own content.
        self.assertIn("w1", result[10])
        self.assertNotIn("w1", result[20])
        self.assertIn("w2", result[20])
        np.testing.assert_array_equal(result[10]["w1"].numpy(), w1.numpy())
        np.testing.assert_array_equal(result[20]["w2"].numpy(), w2.numpy())

    def test_merge_model_state_reunites_preserving_content(self):
        w1 = _t([[1.0, 2.0], [3.0, 4.0]])
        w2 = _t([[5.0, 6.0], [7.0, 8.0]])
        result = merge_model_state(
            {
                10: OrderedDict([("w1", w1)]),
                20: OrderedDict([("w2", w2)]),
            }
        )
        self.assertEqual(set(result), {"w1", "w2"})
        np.testing.assert_array_equal(result["w1"].numpy(), w1.numpy())
        np.testing.assert_array_equal(result["w2"].numpy(), w2.numpy())

    def test_split_opt_state_routes_tensors_master_and_copies_lr(self):
        ma = _t([[1.0, 1.0], [1.0, 1.0]])
        mb = _t([[2.0, 2.0], [2.0, 2.0]])  # same shape, distinct content
        mwa = _t([[3.0, 3.0], [3.0, 3.0]])
        mwb = _t([[4.0, 4.0], [4.0, 4.0]])
        sched = object()
        opt_state = OrderedDict(
            [
                ("moment_a", ma),
                ("moment_b", mb),
                (
                    "master_weights",
                    OrderedDict([("mw_a", mwa), ("mw_b", mwb)]),
                ),
                ("LR_Scheduler", sched),
            ]
        )
        getter = _Getter(
            {
                "moment_a": _Group(10),
                "moment_b": _Group(20),
                "mw_a": _Group(10),
                "mw_b": _Group(20),
            }
        )
        result = split_opt_state(opt_state, getter)
        self.assertEqual(set(result), {10, 20})

        np.testing.assert_array_equal(
            result[10]["moment_a"].numpy(), ma.numpy()
        )
        np.testing.assert_array_equal(
            result[20]["moment_b"].numpy(), mb.numpy()
        )
        self.assertNotIn("moment_b", result[10])
        self.assertNotIn("moment_a", result[20])

        # master weights split by their own key's group.
        self.assertIn("mw_a", result[10]["master_weights"])
        self.assertNotIn("mw_b", result[10]["master_weights"])
        np.testing.assert_array_equal(
            result[10]["master_weights"]["mw_a"].numpy(), mwa.numpy()
        )
        np.testing.assert_array_equal(
            result[20]["master_weights"]["mw_b"].numpy(), mwb.numpy()
        )

        # LR_Scheduler copied (identity) into every group's dict.
        self.assertIs(result[10]["LR_Scheduler"], sched)
        self.assertIs(result[20]["LR_Scheduler"], sched)

    def test_split_opt_state_rejects_non_tensor_top_level_value(self):
        # A top-level entry that is neither master_weights/LR_Scheduler nor a
        # Tensor violates the production `assert isinstance(v, paddle.Tensor)`.
        opt_state = OrderedDict([("bad", 123)])
        getter = _Getter({})
        with self.assertRaises(AssertionError):
            split_opt_state(opt_state, getter)

    def test_merge_opt_state_reunites_and_picks_live_scheduler(self):
        ma = _t([[1.0, 1.0], [1.0, 1.0]])
        mb = _t([[2.0, 2.0], [2.0, 2.0]])
        mwa = _t([[3.0, 3.0], [3.0, 3.0]])
        mwb = _t([[4.0, 4.0], [4.0, 4.0]])
        sched = object()
        opt_state_map = {
            10: {
                "moment_a": ma,
                "master_weights": OrderedDict([("mw_a", mwa)]),
                "LR_Scheduler": None,
            },
            20: {
                "moment_b": mb,
                "master_weights": OrderedDict([("mw_b", mwb)]),
                "LR_Scheduler": sched,
            },
        }
        result = merge_opt_state(opt_state_map)
        np.testing.assert_array_equal(result["moment_a"].numpy(), ma.numpy())
        np.testing.assert_array_equal(result["moment_b"].numpy(), mb.numpy())
        self.assertEqual(set(result["master_weights"]), {"mw_a", "mw_b"})
        np.testing.assert_array_equal(
            result["master_weights"]["mw_a"].numpy(), mwa.numpy()
        )
        np.testing.assert_array_equal(
            result["master_weights"]["mw_b"].numpy(), mwb.numpy()
        )
        # The non-None scheduler wins over the None contributed by group 10.
        self.assertIs(result["LR_Scheduler"], sched)

    def test_split_structure_name_mapping_routes_by_group(self):
        getter = _Getter({"w1": _Group(10), "w2": _Group(20)})
        mapping = OrderedDict([("w1", "param_0"), ("w2", "param_1")])
        result = split_structure_name_mapping(mapping, getter)
        self.assertEqual(set(result), {10, 20})
        self.assertEqual(result[10]["w1"], "param_0")
        self.assertNotIn("w1", result[20])
        self.assertEqual(result[20]["w2"], "param_1")


if __name__ == "__main__":
    unittest.main()
