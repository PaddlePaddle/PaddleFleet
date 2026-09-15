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

"""CPU-observable behavior tests for
``trainer/unified_checkpoint/unified_checkpoint.py``.

Scope note (无卡 / CPU-only). The tests below exercise the key-mapping and
dispatch/validation logic that ``UnifiedCheckpointHandler`` and its helpers
compute on the host, without a real process group:

* ``generate_base_static_name`` -- optimizer-state key parsing (master weight
  vs. moment/beta/velocity slots) used by ``save_non_merge_optimizer`` and
  ``unified_optimizer_into_shards`` to rename keys.
* ``is_need_master_weight`` -- the ``skip_save_model_weight`` gate decision.
* ``is_sharding_split_param_mode`` -- the split-param save/load dispatch guard.
* ``get_sharded_file_name`` -- shard file-name construction (index/of/total).
* ``get_sharded_index`` -- index json assembly (total_size + merged weight_map)
  and its local-rank gating.
* ``UnifiedCheckpointHandler.save_unified_checkpoint`` -- the model-type
  validation that rejects unsupported models before any I/O.

Expected values are hand-derived independently; no production function is used
to produce its own oracle. Real save/load *numerics* (tensor bytes, TP merge,
gather across ranks) are deliberately NOT claimed here: they require Paddle
tensors plus a real distributed process group and belong to single-/multi-card
suites. All Paddle-dependent imports are guarded so the file skips cleanly when
Paddle (or an optional dependency in the import chain) is unavailable, rather
than masking a real regression.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

try:
    import paddle  # noqa: F401

    from paddlefleet.trainer.trainer_utils import ShardingOption
    from paddlefleet.trainer.unified_checkpoint.unified_checkpoint import (
        UnifiedCheckpointHandler,
    )
    from paddlefleet.trainer.unified_checkpoint.utils import (
        FP32_MASTER,
        generate_base_static_name,
        get_sharded_file_name,
        get_sharded_index,
        is_need_master_weight,
        is_sharding_split_param_mode,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = exc


class _RequiresPaddle(unittest.TestCase):
    """Skip the whole case if the real production imports are unavailable."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet unified_checkpoint import failed "
                f"(needs paddle + deps): {_IMPORT_ERROR!r}"
            )


class TestGenerateBaseStaticName(_RequiresPaddle):
    """``generate_base_static_name`` splits an optimizer variable name into
    ``(base_static_name, type_name)``. The oracle is derived by hand from the
    documented separators, not from the function itself.
    """

    def test_master_weight_branch_takes_precedence(self):
        # FP32_MASTER == "fp32_master_0"; when present the name is split on
        # "_fp32_master_0_", so the trailing moment slot must survive intact.
        self.assertEqual(FP32_MASTER, "fp32_master_0")
        base, type_name = generate_base_static_name(
            "llama.embed.weight_fp32_master_0_moment1_0"
        )
        self.assertEqual(base, "llama.embed.weight")
        self.assertEqual(type_name, "moment1_0")

    def test_moment_slot_without_master(self):
        base, type_name = generate_base_static_name(
            "llama.embed.weight_moment2_0"
        )
        self.assertEqual(base, "llama.embed.weight")
        self.assertEqual(type_name, "moment2_0")

    def test_beta_scalar_slot(self):
        base, type_name = generate_base_static_name(
            "linear.w_0_beta1_pow_acc_0"
        )
        self.assertEqual(base, "linear.w_0")
        self.assertEqual(type_name, "beta1_pow_acc_0")

    def test_velocity_slot(self):
        base, type_name = generate_base_static_name("conv.w_velocity_0")
        self.assertEqual(base, "conv.w")
        self.assertEqual(type_name, "velocity_0")

    def test_base_and_type_recombine_to_original_stem(self):
        # The rename logic elsewhere does base + "/" + type_name; confirm the
        # split is loss-free for a representative non-master key.
        original = "decoder.layers.0.mlp.w_moment1_0"
        base, type_name = generate_base_static_name(original)
        self.assertEqual(base + "_" + type_name, original)


class TestIsNeedMasterWeight(_RequiresPaddle):
    """``is_need_master_weight`` gates ``skip_save_model_weight``. It unwraps
    nested optimizers and returns ``_multi_precision and is_fp16_or_bp16``.
    """

    def test_multi_precision_and_low_precision(self):
        opt = SimpleNamespace(_multi_precision=True)
        self.assertTrue(is_need_master_weight(opt, is_fp16_or_bp16=True))

    def test_multi_precision_but_full_precision(self):
        opt = SimpleNamespace(_multi_precision=True)
        self.assertFalse(is_need_master_weight(opt, is_fp16_or_bp16=False))

    def test_not_multi_precision(self):
        opt = SimpleNamespace(_multi_precision=False)
        self.assertFalse(is_need_master_weight(opt, is_fp16_or_bp16=True))

    def test_missing_attribute_returns_false(self):
        opt = SimpleNamespace()  # no _multi_precision at all
        self.assertFalse(is_need_master_weight(opt, is_fp16_or_bp16=True))

    def test_unwraps_inner_opt(self):
        # Outer wrapper lacks _multi_precision; only the unwrapped inner opt
        # carries it. A correct unwrap must consult the inner attribute.
        inner = SimpleNamespace(_multi_precision=True)
        outer = SimpleNamespace(_inner_opt=inner)
        self.assertTrue(is_need_master_weight(outer, is_fp16_or_bp16=True))
        self.assertFalse(is_need_master_weight(outer, is_fp16_or_bp16=False))


class TestIsShardingSplitParamMode(_RequiresPaddle):
    """``is_sharding_split_param_mode`` requires all three of: sharding degree
    > 1, SHARD_OP present, and split_param truthy.
    """

    def _args(self, size, sharding, split_param):
        return SimpleNamespace(
            sharding_parallel_size=size,
            sharding=sharding,
            split_param=split_param,
        )

    def test_all_conditions_met(self):
        args = self._args(2, [ShardingOption.SHARD_OP], True)
        self.assertTrue(is_sharding_split_param_mode(args))

    def test_degree_one_disables(self):
        args = self._args(1, [ShardingOption.SHARD_OP], True)
        self.assertFalse(is_sharding_split_param_mode(args))

    def test_missing_shard_op_disables(self):
        args = self._args(2, [ShardingOption.SHARD_GRAD_OP], True)
        self.assertFalse(is_sharding_split_param_mode(args))

    def test_split_param_off_disables(self):
        args = self._args(2, [ShardingOption.SHARD_OP], False)
        self.assertFalse(is_sharding_split_param_mode(args))


class TestGetShardedFileName(_RequiresPaddle):
    """``get_sharded_file_name`` builds ``<stem>-<idx>-of-<total>.<ext>`` for
    the model-weight (non-optimizer) path. Both index and total are derived
    independently here.
    """

    def _args(self, **kw):
        base = dict(
            world_size=8,
            dataset_world_size=2,
            sharding_parallel_size=1,
            use_expert_parallel=False,
            expert_model_parallel_size=1,
            logical_process_index=0,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_safetensors_first_shard(self):
        # size = world_size // dataset_world_size = 8 // 2 = 4; index = 0 + 1.
        name = get_sharded_file_name(self._args(), "model.safetensors")
        self.assertEqual(name, "model-00001-of-00004.safetensors")

    def test_safetensors_higher_process_index(self):
        args = self._args(logical_process_index=3)
        name = get_sharded_file_name(args, "model.safetensors")
        self.assertEqual(name, "model-00004-of-00004.safetensors")

    def test_pdparams_extension(self):
        name = get_sharded_file_name(self._args(), "model_state.pdparams")
        self.assertEqual(name, "model_state-00001-of-00004.pdparams")

    def test_unknown_extension_passthrough(self):
        # No .pdparams/.safetensors token -> returned unchanged.
        name = get_sharded_file_name(self._args(), "weights.bin")
        self.assertEqual(name, "weights.bin")

    def test_expert_parallel_degree_one_uses_sharding_degree(self):
        # use_expert_parallel with expert_model_parallel_size == 1 avoids the
        # dist.get_world_size() branch: size = world_size // sd_degree.
        # sd_degree = sharding_parallel_size(=2) -> 8 // 2 = 4.
        args = self._args(
            use_expert_parallel=True,
            expert_model_parallel_size=1,
            sharding_parallel_size=2,
        )
        name = get_sharded_file_name(args, "model.safetensors")
        self.assertEqual(name, "model-00001-of-00004.safetensors")


class TestGetShardedIndex(_RequiresPaddle):
    """``get_sharded_index`` merges per-rank index fragments into a single
    ``{metadata, weight_map}`` json, but only on local rank 0.
    """

    def _set_rank(self, value):
        orig = os.environ.get("PADDLE_RANK_IN_NODE")

        def _restore():
            if orig is None:
                os.environ.pop("PADDLE_RANK_IN_NODE", None)
            else:
                os.environ["PADDLE_RANK_IN_NODE"] = orig

        self.addCleanup(_restore)
        if value is None:
            os.environ.pop("PADDLE_RANK_IN_NODE", None)
        else:
            os.environ["PADDLE_RANK_IN_NODE"] = value

    def test_builds_index_on_local_rank_zero(self):
        self._set_rank("0")
        index_file_list = [{"a.w": "shard-1"}, {"b.w": "shard-2"}]
        total_size_list = [100, 250]
        out = get_sharded_index(index_file_list, total_size_list)
        self.assertEqual(out["metadata"], {"total_size": 350})
        self.assertEqual(
            out["weight_map"], {"a.w": "shard-1", "b.w": "shard-2"}
        )

    def test_returns_none_on_non_zero_local_rank(self):
        self._set_rank("1")
        out = get_sharded_index([{"a.w": "shard-1"}], [100])
        self.assertIsNone(out)

    def test_weight_map_merges_all_fragments(self):
        self._set_rank("0")
        fragments = [
            {"layer0.w": "s-1", "layer0.b": "s-1"},
            {"layer1.w": "s-2"},
            {"layer2.w": "s-3"},
        ]
        out = get_sharded_index(fragments, [1, 2, 3])
        self.assertEqual(out["metadata"]["total_size"], 6)
        self.assertEqual(
            out["weight_map"],
            {
                "layer0.w": "s-1",
                "layer0.b": "s-1",
                "layer1.w": "s-2",
                "layer2.w": "s-3",
            },
        )


class TestSaveUnifiedCheckpointDispatch(_RequiresPaddle):
    """The public save entry validates the model type before any I/O and
    rejects anything that is not a PretrainedModel / LoRAModel.
    """

    def _handler(self):
        # Non-async config keeps AsyncCheckpointHandler construction light and
        # side-effect free on CPU (global_rank = -1, no shared memory arrays).
        args = SimpleNamespace(unified_checkpoint_config=[])
        return UnifiedCheckpointHandler(args)

    def test_rejects_unsupported_model_type(self):
        handler = self._handler()
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.isdir(tmp) and os.rmdir(tmp))

        class _NotAModel:
            pass

        with self.assertRaises(ValueError):
            handler.save_unified_checkpoint(
                _NotAModel(), optimizer=None, output_dir=tmp
            )


if __name__ == "__main__":
    unittest.main()
