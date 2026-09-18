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

"""Behavior tests for trainer/unified_checkpoint/utils.py.

Scope: the checkpoint/weight-management helpers in ``utils.py`` that run purely
on CPU without a real Fleet process group. The distributed-only helpers
(``reduce_master_weights_status``, ``merge_tensor_parallel_*``, ``filter_params``,
``gather_sharded_object``, ``rename_shard_file``, ``filter_sync_parameters``,
``get_expected_keys``) require ``fleet.get_hybrid_communicate_group()`` and are
therefore out of scope for a no-card test; they belong in a multi-card job.

Independent oracle: every expected value below is derived BY HAND from the
documented semantics, NOT by calling the production routine a second time.

  * ``generate_base_static_name``: the (base, typename) split is read out by a
    human from the string, exercising both the ``fp32_master_0`` path and the
    per-typename scan (moment/velocity/beta) with distinguishable base names.
  * ``mapping_optimizer_tp_actions``: only ``moment*_0``/``velocity_0`` suffixes
    whose base is in ``tp_actions`` survive, and the surviving VALUE must be the
    exact same action object (identity), so a scalar (beta) key or an unknown
    base is dropped -- a mapping that swapped actions or kept beta would fail.
  * ``unwrap_optimizer`` / ``is_need_master_weight``: real (non-mock) wrapper
    objects; the terminal object identity and the ``_inner_opt``-before-``_optim``
    precedence are pinned, plus the ``_multi_precision AND fp16/bf16`` truth table.
  * ``update_master_weight_status``: the branch -> index-file mapping is checked
    against the SPECIFIC env constant each branch is documented to pick (model
    vs. master index, safe vs. paddle serialization, skip/compatible/remove
    options), and the ValueError path when no compatibility option is present.
  * ``select_model_weight_index`` / ``get_optimizer_shard_files`` /
    ``get_sharded_index``: real temp-dir files and real JSON index; the returned
    filenames, the weight->file ownership in ``file_map`` and the merged
    ``weight_map`` content are compared to hand-built dicts (not shape-only), and
    the missing-file / wrong-rank error and gating paths are asserted.
  * ``get_sharded_file_name``: the ``-{idx+1:05d}-of-{size:05d}`` derivation is
    compared to the exact string a human computes from world/dataset/sharding
    sizes, for both ``.safetensors`` and ``.pdparams``.
  * ``is_sharding_split_param_mode``: the three-way ``and`` (sharding>1,
    stage-1 in sharding, split_param) checked with the real ``ShardingOption``.

Nothing patches a routine under test; no expected value is produced by the
production code. SimpleNamespace stand-ins only carry the plain attributes the
routines read, and the real routine bodies execute against them.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

try:
    from paddlefleet.trainer.trainer_utils import ShardingOption
    from paddlefleet.trainer.unified_checkpoint.utils import (
        FP32_MASTER,
        PADDLE_MASTER_WEIGHTS_INDEX_NAME,
        PADDLE_WEIGHTS_INDEX_NAME,
        SAFE_MASTER_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        UnifiedCheckpointOption,
        generate_base_static_name,
        get_optimizer_shard_files,
        get_sharded_file_name,
        get_sharded_index,
        is_need_master_weight,
        is_sharding_split_param_mode,
        mapping_optimizer_tp_actions,
        select_model_weight_index,
        unwrap_optimizer,
        update_master_weight_status,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this runner
    _IMPORT_ERROR = exc


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestGenerateBaseStaticName(unittest.TestCase):
    """generate_base_static_name splits a variable name into (base, typename)."""

    def test_fp32_master_path_splits_on_marker(self):
        # Human reads: the "_fp32_master_0_" marker separates the base weight
        # name from the trailing optimizer typename.
        base, typename = generate_base_static_name(
            "embedding_0.w_0_fp32_master_0_moment1_0"
        )
        self.assertEqual(base, "embedding_0.w_0")
        self.assertEqual(typename, "moment1_0")

    def test_moment1_non_master(self):
        base, typename = generate_base_static_name("moe_gate_1_moment1_0")
        # split on "moment1_0" -> "moe_gate_1_"; the [:-1] drops the trailing "_".
        self.assertEqual(base, "moe_gate_1")
        self.assertEqual(typename, "moment1_0")

    def test_velocity_non_master(self):
        base, typename = generate_base_static_name("layer.linear.w_velocity_0")
        self.assertEqual(base, "layer.linear.w")
        self.assertEqual(typename, "velocity_0")

    def test_beta2_scalar_non_master(self):
        base, typename = generate_base_static_name("blk.attn.w_beta2_pow_acc_0")
        self.assertEqual(base, "blk.attn.w")
        self.assertEqual(typename, "beta2_pow_acc_0")

    def test_fp32_master_constant_is_the_marker(self):
        # The marker used above is exactly the module constant.
        self.assertEqual(FP32_MASTER, "fp32_master_0")


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestMappingOptimizerTpActions(unittest.TestCase):
    """mapping_optimizer_tp_actions keeps only non-scalar suffixes with a known base."""

    def test_ownership_and_value_identity(self):
        action_w1 = object()
        action_w2 = object()
        tp_actions = {"w1": action_w1, "w2": action_w2}
        keys = [
            "w1/moment1_0",  # non-scalar, base known    -> kept, maps to action_w1
            "w1/beta1_pow_acc_0",  # scalar suffix        -> dropped
            "w2/velocity_0",  # non-scalar, base known    -> kept, maps to action_w2
            "w3/moment2_0",  # base not in tp_actions      -> dropped
        ]
        result = mapping_optimizer_tp_actions(tp_actions, keys)

        self.assertEqual(set(result), {"w1/moment1_0", "w2/velocity_0"})
        # Identity, not merely equality: a swapped mapping would be caught.
        self.assertIs(result["w1/moment1_0"], action_w1)
        self.assertIs(result["w2/velocity_0"], action_w2)

    def test_empty_when_no_non_scalar_matches(self):
        tp_actions = {"w1": object()}
        # Only a scalar suffix present -> nothing survives.
        result = mapping_optimizer_tp_actions(
            tp_actions, ["w1/beta2_pow_acc_0"]
        )
        self.assertEqual(result, {})


class _Plain:
    """Terminal optimizer stand-in with neither _inner_opt nor _optim."""


class _InnerWrap:
    def __init__(self, inner):
        self._inner_opt = inner


class _OptimWrap:
    def __init__(self, inner):
        self._optim = inner


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestUnwrapOptimizer(unittest.TestCase):
    """unwrap_optimizer strips _inner_opt/_optim layers down to the core object."""

    def test_terminal_returned_unchanged(self):
        core = _Plain()
        self.assertIs(unwrap_optimizer(core), core)

    def test_single_inner_opt(self):
        core = _Plain()
        self.assertIs(unwrap_optimizer(_InnerWrap(core)), core)

    def test_single_optim(self):
        core = _Plain()
        self.assertIs(unwrap_optimizer(_OptimWrap(core)), core)

    def test_nested_inner_then_optim(self):
        core = _Plain()
        self.assertIs(unwrap_optimizer(_InnerWrap(_OptimWrap(core))), core)

    def test_inner_opt_takes_precedence_when_both_present(self):
        # A single object exposing BOTH attributes: the loop follows _inner_opt
        # first, then re-checks _optim on the *new* object. Here _inner_opt leads
        # straight to a terminal, so _optim on the original is never followed.
        via_inner = _Plain()
        via_optim = _Plain()
        both = SimpleNamespace(_inner_opt=via_inner, _optim=via_optim)
        self.assertIs(unwrap_optimizer(both), via_inner)


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestIsNeedMasterWeight(unittest.TestCase):
    """is_need_master_weight = optimizer._multi_precision AND fp16/bf16."""

    def test_multi_precision_and_low_precision(self):
        opt = SimpleNamespace(_multi_precision=True)
        self.assertTrue(is_need_master_weight(opt, is_fp16_or_bp16=True))

    def test_multi_precision_but_full_precision(self):
        opt = SimpleNamespace(_multi_precision=True)
        self.assertFalse(is_need_master_weight(opt, is_fp16_or_bp16=False))

    def test_multi_precision_false(self):
        opt = SimpleNamespace(_multi_precision=False)
        self.assertFalse(is_need_master_weight(opt, is_fp16_or_bp16=True))

    def test_no_multi_precision_attribute(self):
        opt = _Plain()
        self.assertFalse(is_need_master_weight(opt, is_fp16_or_bp16=True))

    def test_unwraps_before_checking(self):
        # The flag lives on the innermost optimizer; wrapping must not hide it.
        wrapped = _InnerWrap(SimpleNamespace(_multi_precision=True))
        self.assertTrue(is_need_master_weight(wrapped, is_fp16_or_bp16=True))


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestUpdateMasterWeightStatus(unittest.TestCase):
    """update_master_weight_status maps each branch to a specific index file."""

    @staticmethod
    def _args(config, fp16=True, bf16=False):
        return SimpleNamespace(
            fp16=fp16, bf16=bf16, unified_checkpoint_config=list(config)
        )

    @staticmethod
    def _mp_optimizer():
        # multi_precision optimizer so is_need_master_weight is True under fp16.
        return SimpleNamespace(_multi_precision=True)

    def test_missing_master_with_compatible_uses_model_index_paddle(self):
        has, name = update_master_weight_status(
            self._args(
                [UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value]
            ),
            self._mp_optimizer(),
            has_master_weight=False,
            safe_serialization=False,
        )
        self.assertTrue(has)
        self.assertEqual(name, PADDLE_WEIGHTS_INDEX_NAME)

    def test_missing_master_with_remove_uses_model_index_safe(self):
        has, name = update_master_weight_status(
            self._args([UnifiedCheckpointOption.REMOVE_MASTER_WEIGHT.value]),
            self._mp_optimizer(),
            has_master_weight=False,
            safe_serialization=True,
        )
        self.assertTrue(has)
        self.assertEqual(name, SAFE_WEIGHTS_INDEX_NAME)

    def test_missing_master_without_option_raises(self):
        with self.assertRaises(ValueError):
            update_master_weight_status(
                self._args([]),
                self._mp_optimizer(),
                has_master_weight=False,
                safe_serialization=False,
            )

    def test_present_master_paddle_index(self):
        has, name = update_master_weight_status(
            self._args([]),
            self._mp_optimizer(),
            has_master_weight=True,
            safe_serialization=False,
        )
        self.assertTrue(has)
        self.assertEqual(name, PADDLE_MASTER_WEIGHTS_INDEX_NAME)

    def test_present_master_safe_index(self):
        has, name = update_master_weight_status(
            self._args([]),
            self._mp_optimizer(),
            has_master_weight=True,
            safe_serialization=True,
        )
        self.assertTrue(has)
        self.assertEqual(name, SAFE_MASTER_WEIGHTS_INDEX_NAME)

    def test_present_master_with_skip_falls_back_to_model_index(self):
        has, name = update_master_weight_status(
            self._args([UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value]),
            self._mp_optimizer(),
            has_master_weight=True,
            safe_serialization=True,
        )
        self.assertTrue(has)
        self.assertEqual(name, SAFE_WEIGHTS_INDEX_NAME)

    def test_no_master_weight_needed_returns_none(self):
        # Not multi_precision -> master weight not needed regardless of inputs.
        has, name = update_master_weight_status(
            self._args([UnifiedCheckpointOption.REMOVE_MASTER_WEIGHT.value]),
            SimpleNamespace(_multi_precision=False),
            has_master_weight=True,
            safe_serialization=False,
        )
        self.assertFalse(has)
        self.assertIsNone(name)


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestIsShardingSplitParamMode(unittest.TestCase):
    """is_sharding_split_param_mode = size>1 AND stage-1 in sharding AND split_param."""

    def test_all_conditions_true(self):
        args = SimpleNamespace(
            sharding_parallel_size=2,
            sharding=[ShardingOption.SHARD_OP],
            split_param=True,
        )
        self.assertTrue(is_sharding_split_param_mode(args))

    def test_single_shard_size_false(self):
        args = SimpleNamespace(
            sharding_parallel_size=1,
            sharding=[ShardingOption.SHARD_OP],
            split_param=True,
        )
        self.assertFalse(is_sharding_split_param_mode(args))

    def test_wrong_stage_false(self):
        args = SimpleNamespace(
            sharding_parallel_size=2,
            sharding=[ShardingOption.FULL_SHARD],
            split_param=True,
        )
        self.assertFalse(is_sharding_split_param_mode(args))

    def test_split_param_off_false(self):
        args = SimpleNamespace(
            sharding_parallel_size=2,
            sharding=[ShardingOption.SHARD_OP],
            split_param=False,
        )
        self.assertFalse(is_sharding_split_param_mode(args))


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestSelectModelWeightIndex(unittest.TestCase):
    """select_model_weight_index picks the model index, else falls back to master."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.model = object()  # not a LoRAModel -> non-PEFT index names

    def _touch(self, name):
        with open(os.path.join(self.tmp, name), "w") as f:
            f.write("{}")

    def test_returns_paddle_model_index_when_present(self):
        self._touch(PADDLE_WEIGHTS_INDEX_NAME)
        got = select_model_weight_index(
            self.model, self.tmp, safe_serialization=False, local=True
        )
        self.assertEqual(got, PADDLE_WEIGHTS_INDEX_NAME)

    def test_returns_safe_model_index_when_present(self):
        self._touch(SAFE_WEIGHTS_INDEX_NAME)
        got = select_model_weight_index(
            self.model, self.tmp, safe_serialization=True, local=True
        )
        self.assertEqual(got, SAFE_WEIGHTS_INDEX_NAME)

    def test_falls_back_to_master_index(self):
        # No model index on disk, only the master index -> master name returned.
        self._touch(PADDLE_MASTER_WEIGHTS_INDEX_NAME)
        got = select_model_weight_index(
            self.model, self.tmp, safe_serialization=False, local=True
        )
        self.assertEqual(got, PADDLE_MASTER_WEIGHTS_INDEX_NAME)

    def test_raises_when_nothing_present(self):
        with self.assertRaises(ValueError):
            select_model_weight_index(
                self.model, self.tmp, safe_serialization=False, local=True
            )


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestGetOptimizerShardFiles(unittest.TestCase):
    """get_optimizer_shard_files parses a real JSON index into shard files + metadata."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_index(self, payload):
        path = os.path.join(self.tmp, "opt.index.json")
        with open(path, "w") as f:
            f.write(json.dumps(payload))
        return path

    def test_shard_files_metadata_and_file_ownership(self):
        weight_map = {
            "w1/moment1_0": "opt-00001-of-00002.pdopt",
            "w1/beta1_pow_acc_0": "opt-00001-of-00002.pdopt",
            "w2/moment1_0": "opt-00002-of-00002.pdopt",
        }
        index_path = self._write_index(
            {
                "weight_map": weight_map,
                "metadata": {"total_size": 999},
                "master_weights": True,
            }
        )

        shard_files, meta = get_optimizer_shard_files(self.tmp, index_path)

        # Shard files are the sorted unique values, each prefixed by the dir.
        self.assertEqual(
            shard_files,
            [
                os.path.join(self.tmp, "opt-00001-of-00002.pdopt"),
                os.path.join(self.tmp, "opt-00002-of-00002.pdopt"),
            ],
        )
        self.assertEqual(meta["total_size"], 999)
        self.assertTrue(meta["master_weights"])
        self.assertCountEqual(
            meta["all_optimizer_keys"], list(weight_map.keys())
        )
        self.assertEqual(meta["weight_map"], weight_map)
        # Ownership: which weights land in which file. A swapped weight->file
        # assignment would be caught here (shape-only checks would not).
        self.assertEqual(
            meta["file_map"],
            {
                "opt-00001-of-00002.pdopt": {
                    "w1/moment1_0",
                    "w1/beta1_pow_acc_0",
                },
                "opt-00002-of-00002.pdopt": {"w2/moment1_0"},
            },
        )

    def test_master_weights_defaults_false(self):
        index_path = self._write_index(
            {"weight_map": {"w/moment1_0": "s.pdopt"}, "metadata": {}}
        )
        _, meta = get_optimizer_shard_files(self.tmp, index_path)
        self.assertFalse(meta["master_weights"])

    def test_missing_index_raises(self):
        with self.assertRaises(ValueError):
            get_optimizer_shard_files(
                self.tmp, os.path.join(self.tmp, "does_not_exist.json")
            )


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestGetShardedIndex(unittest.TestCase):
    """get_sharded_index merges per-file weight maps on local rank 0 only."""

    def setUp(self):
        self._orig = os.environ.get("PADDLE_RANK_IN_NODE")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._orig is None:
            os.environ.pop("PADDLE_RANK_IN_NODE", None)
        else:
            os.environ["PADDLE_RANK_IN_NODE"] = self._orig

    def test_rank0_merges_maps_and_sums_sizes(self):
        os.environ["PADDLE_RANK_IN_NODE"] = "0"
        index_file_list = [{"a": "f1"}, {"b": "f2", "c": "f1"}]
        total_size_list = [10, 20]
        result = get_sharded_index(index_file_list, total_size_list)
        self.assertEqual(
            result,
            {
                "metadata": {"total_size": 30},
                "weight_map": {"a": "f1", "b": "f2", "c": "f1"},
            },
        )

    def test_non_zero_rank_returns_none(self):
        os.environ["PADDLE_RANK_IN_NODE"] = "1"
        self.assertIsNone(get_sharded_index([{"a": "f1"}], [10]))


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestGetShardedFileName(unittest.TestCase):
    """get_sharded_file_name derives the -{idx+1}-of-{size} shard suffix."""

    def test_model_weight_non_expert_safetensors(self):
        # size = world_size // dataset_world_size = 8 // 2 = 4;
        # idx part = logical_process_index + 1 = 3.
        args = SimpleNamespace(
            sharding_parallel_size=1,
            use_expert_parallel=False,
            world_size=8,
            dataset_world_size=2,
            logical_process_index=2,
        )
        got = get_sharded_file_name(
            args, "model.safetensors", is_optimizer=False
        )
        self.assertEqual(got, "model-00003-of-00004.safetensors")

    def test_model_weight_non_expert_pdparams(self):
        args = SimpleNamespace(
            sharding_parallel_size=1,
            use_expert_parallel=False,
            world_size=8,
            dataset_world_size=2,
            logical_process_index=0,
        )
        got = get_sharded_file_name(
            args, "model_state.pdparams", is_optimizer=False
        )
        self.assertEqual(got, "model_state-00001-of-00004.pdparams")

    def test_model_weight_expert_single_ep(self):
        # use_expert_parallel with expert_model_parallel_size <= 1:
        # size = world_size // sd_degree = 8 // 2 = 4.
        args = SimpleNamespace(
            sharding_parallel_size=2,
            use_expert_parallel=True,
            expert_model_parallel_size=1,
            world_size=8,
            logical_process_index=3,
        )
        got = get_sharded_file_name(
            args, "model.safetensors", is_optimizer=False
        )
        self.assertEqual(got, "model-00004-of-00004.safetensors")


if __name__ == "__main__":
    unittest.main()
