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

"""Behavioral unit tests for ``paddlefleet.trainer.utils.sharding_io``.

Module: Checkpoint 与权重管理层 (无卡 / CPU only).

These tests drive the genuine, CPU-runnable logic of ``sharding_io.py`` with
independent hand-derived oracles and real on-disk reads via ``tempfile``:

* ``to_device`` returns the *same* tensor object and preserves content.
* ``exclude_parameters_in_state_dict`` drops exactly the entries whose tensor
  ``.name`` is a master weight, keyed by tensor name (not dict key), and copies.
* ``filter_sharded_params`` passes the state dict through unchanged for a
  non-sharding optimizer (real ``is_sharding_opt`` branch).
* ``ParameterNameRemapper`` builds the tensor-name map, rejects conflicting /
  missing structure names, and remaps names while preserving tensor identity.
* ``ShardingIO._sharding_meta_suffix`` zero-pads tp/pp (and ep) ranks.
* ``ShardingIO._load_model_meta_impl`` / ``check_same_strategy`` read a real
  JSON meta file and enforce the parallel-degree contract.

NOT covered here: real multi-rank reshard math (cross-rank all_gather / restore /
pp_reshard). Those need a distributed launcher; single-process only reaches the
``nranks < 2`` short-circuit branches.
"""

import json
import os
import tempfile
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

try:
    import paddle

    from paddlefleet.trainer.utils.sharding_io import (
        ParameterNameRemapper,
        ShardingIO,
        exclude_parameters_in_state_dict,
        filter_sharded_params,
        to_device,
    )
    from paddlefleet.utils.env import MODEL_META_NAME

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def _named(array, name):
    """Build a CPU tensor with distinct content and a fixed tensor name."""
    tensor = paddle.to_tensor(np.asarray(array, dtype="float32"))
    tensor.name = name
    return tensor


def _make_args(**overrides):
    """Minimal TrainingArguments-like holder for ShardingIO (a data collaborator)."""
    defaults = dict(
        use_hybrid_parallel=False,
        reshard_bucketed_broadcast_max_chunk_gb=1.0,
        tensor_parallel_rank=0,
        pipeline_parallel_rank=0,
        expert_model_parallel_size=1,
        expert_parallel_rank=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_io(**arg_overrides):
    # world_size == 1 in-process, so __init__ never touches fleet/hcg.
    return ShardingIO(_make_args(**arg_overrides), model=None)


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestToDevice(unittest.TestCase):
    """to_device returns the original tensor object and preserves content."""

    def test_same_place_returns_identity_and_preserves_value(self):
        t = _named(np.arange(6).reshape(2, 3), "w0")
        before = t.numpy().copy()
        result = to_device(t, t.place)
        self.assertIs(result, t)  # contract: original object is returned
        self.assertEqual(result.name, "w0")
        np.testing.assert_array_equal(result.numpy(), before)

    def test_string_cpu_place_preserves_value(self):
        t = _named([[1.0, 2.0], [3.0, 4.0]], "b0")
        before = t.numpy().copy()
        result = to_device(t, "cpu")
        self.assertIs(result, t)
        np.testing.assert_array_equal(result.numpy(), before)


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestExcludeParametersInStateDict(unittest.TestCase):
    """Master-weight params are dropped by tensor name; others survive intact."""

    def _state_dict(self):
        # dict key differs from tensor .name so we can prove filtering is by name.
        sd = OrderedDict()
        sd["k_a"] = _named([1.0, 2.0], "p_a")
        sd["k_b"] = _named([3.0, 4.0], "p_b")
        sd["k_c"] = _named([5.0, 6.0], "p_c")
        sd["k_d"] = _named([7.0, 8.0], "p_d")
        return sd

    def test_drops_only_master_weight_names(self):
        sd = self._state_dict()
        group = SimpleNamespace(rank=0, nranks=1)
        out = exclude_parameters_in_state_dict(sd, ["p_b", "p_d"], group)
        # Independent oracle: keep exactly the complement, keyed by tensor name.
        self.assertEqual(list(out.keys()), ["k_a", "k_c"])
        np.testing.assert_array_equal(out["k_a"].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(out["k_c"].numpy(), [5.0, 6.0])
        # Input is copied, not mutated.
        self.assertEqual(list(sd.keys()), ["k_a", "k_b", "k_c", "k_d"])

    def test_accepts_set_of_names(self):
        sd = self._state_dict()
        group = SimpleNamespace(rank=0, nranks=1)
        out = exclude_parameters_in_state_dict(sd, {"p_a"}, group)
        self.assertEqual(list(out.keys()), ["k_b", "k_c", "k_d"])

    def test_rejects_non_list_master_weights(self):
        sd = self._state_dict()
        group = SimpleNamespace(rank=0, nranks=1)
        with self.assertRaises(AssertionError):
            exclude_parameters_in_state_dict(sd, 123, group)


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestFilterShardedParams(unittest.TestCase):
    """A non-sharding optimizer means the state dict is returned unchanged."""

    def test_non_sharding_optimizer_passthrough(self):
        linear = paddle.nn.Linear(2, 2)
        opt = paddle.optimizer.AdamW(
            learning_rate=0.01, parameters=list(linear.parameters())
        )
        sd = OrderedDict()
        sd["w"] = _named(np.arange(4).reshape(2, 2), "w0")
        sd["b"] = _named([0.0, 1.0], "b0")
        group = SimpleNamespace(rank=0, nranks=1)
        out = filter_sharded_params(sd, opt, group)
        # is_sharding_opt is False for plain AdamW -> exact same object back.
        self.assertIs(out, sd)
        self.assertEqual(list(out.keys()), ["w", "b"])


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestParameterNameRemapper(unittest.TestCase):
    """Old->new tensor name mapping, its guards, and name remapping."""

    def test_init_builds_tensor_name_map(self):
        r = ParameterNameRemapper(
            {"s1": "p0", "s2": "p1"}, {"s1": "q0", "s2": "q1"}, "/ckpt"
        )
        self.assertEqual(r.p_name_map, {"p0": "q0", "p1": "q1"})
        self.assertEqual(r.new_mapping, {"s1": "q0", "s2": "q1"})
        self.assertEqual(set(r.old_p_names), {"p0", "p1"})
        self.assertEqual(r.checkpoint, "/ckpt")

    def test_init_conflicting_new_name_raises(self):
        # s1 and s2 share old param p0 but map to different new names.
        with self.assertRaises(AssertionError):
            ParameterNameRemapper(
                {"s1": "p0", "s2": "p0"}, {"s1": "q0", "s2": "q9"}, "/ckpt"
            )

    def test_init_missing_structure_key_raises(self):
        with self.assertRaises(AssertionError):
            ParameterNameRemapper({"s1": "p0"}, {}, "/ckpt")

    def test_map_tensor_remaps_prefix_preserving_identity(self):
        r = ParameterNameRemapper({"s1": "p0"}, {"s1": "q0"}, "/ckpt")
        t = _named(np.arange(4), "p0_moment1_0")
        before = t.numpy().copy()
        new_name, out = r._map_tensor(t, "p0")
        self.assertEqual(new_name, "q0_moment1_0")  # only the prefix changes
        self.assertIs(out, t)
        self.assertEqual(t.name, "q0_moment1_0")
        np.testing.assert_array_equal(out.numpy(), before)

    def test_map_tensor_unknown_old_pname_raises(self):
        r = ParameterNameRemapper({"s1": "p0"}, {"s1": "q0"}, "/ckpt")
        t = _named(np.arange(4), "pX_moment1_0")
        with self.assertRaises(AssertionError):
            r._map_tensor(t, "pX")

    def test_map_tensor_prefix_mismatch_raises(self):
        r = ParameterNameRemapper({"s1": "p0"}, {"s1": "q0"}, "/ckpt")
        t = _named(np.arange(4), "zzz")  # does not start with mapped p0
        with self.assertRaises(AssertionError):
            r._map_tensor(t, "p0")

    def test_remap_model_state_renames_tensors_skips_numpy(self):
        r = ParameterNameRemapper(
            {"s1": "p0", "s2": "p1"}, {"s1": "q0", "s2": "q1"}, "/ckpt"
        )
        t = _named(np.arange(4), "p0")
        before = t.numpy().copy()
        raw = np.arange(3, dtype="float32")  # numpy value must be left as-is
        ms = OrderedDict([("s1", t), ("s2", raw)])
        out = r.remap_model_state(ms)
        self.assertIs(out["s1"], t)
        self.assertEqual(t.name, "q0")
        np.testing.assert_array_equal(out["s1"].numpy(), before)
        self.assertIs(out["s2"], raw)


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestParameterNameRemapperTupleBug(unittest.TestCase):
    """_map_tensor on a (name, value) pair is broken in production."""

    @unittest.expectedFailure
    def test_map_tensor_accepts_name_value_tuple(self):
        # Intended: the (old_name, value) branch remaps the name to "q0".
        # Production calls ``self._map_name`` (a nested local, not a method),
        # raising AttributeError. Documented, no production edit.
        r = ParameterNameRemapper({"s1": "p0"}, {"s1": "q0"}, "/ckpt")
        new_name, out = r._map_tensor(("p0", 123), None)
        self.assertEqual(new_name, "q0")
        self.assertEqual(out, ("q0", 123))


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestShardingMetaSuffix(unittest.TestCase):
    """Suffix zero-pads tp/pp ranks to width 2 and appends ep when EP>1."""

    def test_explicit_ranks_without_ep(self):
        io = _make_io(expert_model_parallel_size=1)
        self.assertEqual(
            io._sharding_meta_suffix(tp_rank=1, pp_rank=2), "tp01_pp02"
        )

    def test_defaults_read_from_args(self):
        io = _make_io(
            tensor_parallel_rank=3,
            pipeline_parallel_rank=4,
            expert_model_parallel_size=1,
        )
        self.assertEqual(io._sharding_meta_suffix(), "tp03_pp04")

    def test_appends_ep_rank_when_expert_parallel(self):
        io = _make_io(expert_model_parallel_size=2, expert_parallel_rank=5)
        self.assertEqual(
            io._sharding_meta_suffix(tp_rank=0, pp_rank=0), "tp00_pp00_ep05"
        )


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestLoadModelMetaImpl(unittest.TestCase):
    """Real JSON meta read from a temp dir and its parallel-degree checks."""

    def _write_meta(self, dirpath, meta):
        with open(os.path.join(dirpath, MODEL_META_NAME), "w") as handle:
            json.dump(meta, handle)

    def test_roundtrip_returns_parallel_config(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            pc = {"pp_degree": 1, "mp_degree": 1, "sharding_degree": 1}
            self._write_meta(d, {"parallel_config": pc})
            got = io._load_model_meta_impl(d)
            self.assertEqual(got["parallel_config"], pc)

    def test_missing_file_asserts(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                io._load_model_meta_impl(os.path.join(d, "nope"))

    def test_missing_parallel_config_asserts(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            self._write_meta(d, {"something_else": 1})
            with self.assertRaises(AssertionError):
                io._load_model_meta_impl(d)

    def test_ep_degree_inconsistent_asserts(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            # ep_degree=2 but mp*sharding (1*1) != ep*moe_sharding (2*1).
            pc = {
                "pp_degree": 1,
                "mp_degree": 1,
                "sharding_degree": 1,
                "ep_degree": 2,
                "moe_sharding_degree": 1,
            }
            self._write_meta(d, {"parallel_config": pc})
            with self.assertRaises(AssertionError):
                io._load_model_meta_impl(d)

    def test_ep_degree_consistent_loads(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            # 2 * 1 == 2 * 1 -> consistent.
            pc = {
                "pp_degree": 1,
                "mp_degree": 2,
                "sharding_degree": 1,
                "ep_degree": 2,
                "moe_sharding_degree": 1,
            }
            self._write_meta(d, {"parallel_config": pc})
            got = io._load_model_meta_impl(d)
            self.assertEqual(got["parallel_config"]["ep_degree"], 2)


@unittest.skipUnless(HAS_DEPS, "paddle / paddlefleet not importable")
class TestCheckSameStrategy(unittest.TestCase):
    """check_same_strategy compares the saved config against the current one.

    Single-process, non-hybrid: the current strategy is the all-ones default
    ``{pp,mp,sharding,ep,moe_sharding}_degree == 1``.
    """

    def _write_meta(self, dirpath, parallel_config):
        with open(os.path.join(dirpath, MODEL_META_NAME), "w") as handle:
            json.dump({"parallel_config": parallel_config}, handle)

    _CURRENT = {
        "pp_degree": 1,
        "mp_degree": 1,
        "sharding_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
    }

    def test_none_checkpoint_is_same(self):
        io = _make_io()
        self.assertEqual(io.check_same_strategy(None), (True, None))

    def test_matching_config_is_same(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            self._write_meta(d, dict(self._CURRENT))
            ok, reason = io.check_same_strategy(d)
            self.assertTrue(ok)
            self.assertIsNone(reason)

    def test_value_mismatch_reports_key(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(self._CURRENT)
            cfg["sharding_degree"] = 2
            self._write_meta(d, cfg)
            ok, reason = io.check_same_strategy(d)
            self.assertFalse(ok)
            self.assertIn("sharding_degree", reason)
            self.assertIn("2 vs 1", reason)

    def test_extra_key_reports_missing(self):
        io = _make_io()
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(self._CURRENT)
            cfg["custom_key"] = 7
            self._write_meta(d, cfg)
            ok, reason = io.check_same_strategy(d)
            self.assertFalse(ok)
            self.assertEqual(reason, "missing custom_key")


if __name__ == "__main__":
    unittest.main()
