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
"""Single card tests for the accuracy-compatible helpers on ``Trainer``.

Deferred token normalization, native MAIN-loss reporting and the
fused-expert checkpoint views normally only run inside a distributed
training step: they want a hybrid-parallel process group, a sharding
optimizer and fused gradient buffers. Each helper is therefore called as
an unbound function with a ``SimpleNamespace`` in place of ``self``, and
the collectives are replaced by recorders that leave their payload
alone. That keeps the arithmetic, the branch selection and the
collective arguments observable on one card without a process group.
"""

import contextlib
import os
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import paddle
from paddle.distributed import ShardedWeight
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)

from paddlefleet import parallel_state
from paddlefleet.models.common.language_loss import language_loss as loss_mod
from paddlefleet.trainer import trainer as trainer_mod
from paddlefleet.trainer.trainer import (
    Trainer,
    _fused_expert_optimizer_save_views,
    restore_fused_expert_3d_layout,
)

LOSS_MODULE = "paddlefleet.models.common.language_loss.language_loss"
EXPERT_KEY = "layers.0.mlp.grouped_gemm_experts.weight"


def _fp32(values):
    return paddle.to_tensor(values, dtype="float32")


def _scalar(value):
    return paddle.full([], float(value), dtype="float32")


class _Group:
    """Stands in for a constructed communication group."""

    def __init__(self, nranks, ranks=None):
        self.nranks = nranks
        self.ranks = list(range(nranks)) if ranks is None else list(ranks)


class _Collectives:
    """Records collective calls and leaves the payload untouched."""

    def __init__(self):
        self.all_reduce_calls = []
        self.broadcast_calls = []

    def all_reduce(self, tensor, op=None, group=None, sync_op=True):
        self.all_reduce_calls.append((tensor, op, group))

    def broadcast(self, tensor, src=None, group=None, sync_op=True):
        self.broadcast_calls.append((tensor, src, group))

    def ops(self):
        return [call[1] for call in self.all_reduce_calls]

    def groups(self):
        return [call[2] for call in self.all_reduce_calls]


@contextlib.contextmanager
def _patched_collectives():
    """Run a helper without a process group behind the collectives."""
    calls = _Collectives()
    with (
        mock.patch.object(paddle.distributed, "all_reduce", calls.all_reduce),
        mock.patch.object(paddle.distributed, "broadcast", calls.broadcast),
    ):
        yield calls


class _Param:
    """Parameter-shaped holder for a gradient buffer."""

    def __init__(self, name, main_grad=None, grad=None):
        self.name = name
        self.main_grad = main_grad
        self.grad = grad


class _ExplodingOptimizer:
    """Any attribute read proves the optimizer was walked."""

    def __getattr__(self, name):
        raise AssertionError(f"optimizer was inspected: {name}")


def _model(params):
    """Model stand-in exposing only the parameter iteration protocol."""
    return SimpleNamespace(named_parameters=lambda: list(params))


class RestoreFusedExpert3dLayoutTests(unittest.TestCase):
    """``restore_fused_expert_3d_layout``: flat shard -> 3-D parameter."""

    def test_flat_expert_shard_is_restored_to_the_parameter_layout(self):
        param = paddle.zeros([2, 3, 4], dtype="float32")
        flat = paddle.arange(24, dtype="float32").reshape([6, 4])
        shard = ShardedWeight(EXPERT_KEY, flat, (6, 4), (6, 4), (0, 0))
        state = {EXPERT_KEY: shard}

        result = restore_fused_expert_3d_layout(
            _model([(EXPERT_KEY, param)]), state
        )

        self.assertIs(result, state)
        self.assertEqual(list(shard.local_tensor.shape), [2, 3, 4])
        self.assertEqual(shard.local_shape, (2, 3, 4))
        self.assertEqual(shard.global_shape, (2, 3, 4))
        self.assertEqual(shard.global_offset, (0, 0, 0))
        self.assertEqual(shard.local_tensor.name, flat.name)
        self.assertEqual(
            shard.local_tensor.reshape([24]).tolist(), list(range(24))
        )

    def test_shards_outside_the_fused_expert_layout_are_left_alone(self):
        param3d = paddle.zeros([2, 3, 4], dtype="float32")
        param2d = paddle.zeros([6, 4], dtype="float32")
        plain = _fp32([1.0, 2.0])
        wrong_key = "layers.0.mlp.up_proj.weight"
        missing_key = "layers.9.mlp.grouped_gemm_experts.weight"
        numel_key = "layers.1.mlp.grouped_gemm_experts.weight"
        shards = {
            "plain": plain,
            wrong_key: ShardedWeight(
                wrong_key, paddle.zeros([6, 4]), (6, 4), (6, 4), (0, 0)
            ),
            missing_key: ShardedWeight(
                missing_key, paddle.zeros([6, 4]), (6, 4), (6, 4), (0, 0)
            ),
            EXPERT_KEY: ShardedWeight(
                EXPERT_KEY, param2d, (6, 4), (6, 4), (0, 0)
            ),
            numel_key: ShardedWeight(
                numel_key,
                paddle.zeros([2, 3, 4]),
                (2, 3, 4),
                (2, 3, 4),
                (0, 0, 0),
            ),
        }
        second_key = "layers.2.mlp.grouped_gemm_experts.weight"
        shards[second_key] = ShardedWeight(
            second_key, paddle.zeros([5, 4]), (5, 4), (5, 4), (0, 0)
        )
        named = [
            (wrong_key, param2d),
            (EXPERT_KEY, param2d),
            (numel_key, param3d),
            (second_key, param3d),
        ]
        before = {
            key: (shard, getattr(shard, "local_shape", None))
            for key, shard in shards.items()
        }

        restore_fused_expert_3d_layout(_model(named), shards)

        for key, shard in shards.items():
            self.assertIs(shards[key], before[key][0])
            self.assertEqual(
                getattr(shard, "local_shape", None), before[key][1]
            )
        self.assertIs(shards["plain"], plain)


class FusedExpertOptimizerSaveViewsTests(unittest.TestCase):
    """``_fused_expert_optimizer_save_views``: flat optimizer views."""

    def _expert_setup(self):
        param = paddle.zeros([2, 3, 4], dtype="float32")
        shard = ShardedWeight(
            EXPERT_KEY, paddle.zeros([6, 4]), (6, 4), (6, 4), (0, 0)
        )
        model = _model([(EXPERT_KEY, param)])
        return param, model, {EXPERT_KEY: shard}

    def test_without_fused_expert_shards_the_optimizer_is_not_walked(self):
        param = paddle.zeros([6, 4], dtype="float32")
        key = "layers.0.mlp.up_proj.weight"
        shard = ShardedWeight(key, param, (6, 4), (6, 4), (0, 0))

        with _fused_expert_optimizer_save_views(
            _model([(key, param)]), {key: shard}, _ExplodingOptimizer()
        ):
            self.assertEqual(shard.local_shape, (6, 4))

    def test_cyclic_optimizer_wrappers_are_rejected(self):
        _, model, state = self._expert_setup()
        outer = SimpleNamespace()
        inner = SimpleNamespace(_inner_opt=outer)
        outer._inner_opt = inner

        with (
            self.assertRaises(ValueError) as caught,
            _fused_expert_optimizer_save_views(model, state, outer),
        ):
            pass

        self.assertIn("Cyclic optimizer wrapper", str(caught.exception))

    def test_three_dimensional_states_are_viewed_flat_then_restored(self):
        param, model, state = self._expert_setup()
        master = paddle.zeros([2, 3, 4], dtype="float32")
        moment = paddle.ones([2, 3, 4], dtype="float32")
        master_moment = paddle.full([2, 3, 4], 2.0, dtype="float32")
        already_flat = paddle.zeros([6, 4], dtype="float32")
        masters = {param.name: master}
        moments = {param.name: moment, master.name: master_moment}
        flat_states = {param.name: already_flat}
        not_tensor = {param.name: "not-a-tensor"}
        inner = SimpleNamespace(
            _master_weights=masters,
            _accumulators={
                "moment1": moments,
                "moment2": flat_states,
                "moment3": not_tensor,
                "beta1_pow": [1, 2, 3],
            },
        )
        optimizer = SimpleNamespace(_optimizer=inner)

        with _fused_expert_optimizer_save_views(model, state, optimizer):
            self.assertEqual(list(moments[param.name].shape), [6, 4])
            self.assertEqual(list(moments[master.name].shape), [6, 4])
            self.assertEqual(list(masters[param.name].shape), [6, 4])
            self.assertEqual(moments[param.name].name, moment.name)
            self.assertEqual(masters[param.name].name, master.name)
            self.assertIs(flat_states[param.name], already_flat)
            self.assertEqual(not_tensor[param.name], "not-a-tensor")
            self.assertEqual(inner._accumulators["beta1_pow"], [1, 2, 3])

        self.assertIs(moments[param.name], moment)
        self.assertIs(moments[master.name], master_moment)
        self.assertIs(masters[param.name], master)
        self.assertEqual(list(moment.shape), [2, 3, 4])
        self.assertEqual(list(master.shape), [2, 3, 4])

    def test_views_are_restored_when_the_body_raises(self):
        param, model, state = self._expert_setup()
        moment = paddle.ones([2, 3, 4], dtype="float32")
        moments = {param.name: moment}
        inner = SimpleNamespace(
            _master_weights={}, _accumulators={"moment1": moments}
        )

        with (
            self.assertRaises(RuntimeError),
            _fused_expert_optimizer_save_views(
                model, state, SimpleNamespace(inner_opt=inner)
            ),
        ):
            raise RuntimeError("serialization failed")

        self.assertIs(moments[param.name], moment)


class SaveFlexOptimizerStateTests(unittest.TestCase):
    """``Trainer._save_flex_optimizer_state`` around the flat views."""

    def test_master_weights_and_states_are_split_and_saved(self):
        param = paddle.zeros([4, 2], dtype="float32")
        key = "layers.0.linear.weight"
        model_shard = ShardedWeight(key, param, (4, 2), (4, 2), (0, 0))
        model = SimpleNamespace(
            named_parameters=lambda: [(key, param)],
            sharded_state_dict=lambda: {key: model_shard},
        )
        seen = {}

        def optimizer_sharded_state_dict(model_state):
            seen["model_state"] = model_state
            return {"a.w_0": "master", "a.moment1": "state"}

        stub = SimpleNamespace(
            model=model,
            optimizer=SimpleNamespace(
                sharded_state_dict=optimizer_sharded_state_dict
            ),
            args=SimpleNamespace(replicate_saved_into_local=True),
        )
        saves = []

        def save_state_dict(state, path, save_replicas=False):
            saves.append((dict(state), path, save_replicas))

        with (
            tempfile.TemporaryDirectory() as output_dir,
            mock.patch.object(
                paddle.distributed, "save_state_dict", save_state_dict
            ),
        ):
            Trainer._save_flex_optimizer_state(stub, output_dir)
            signal = os.path.join(
                output_dir, f"saved_signal_{paddle.distributed.get_rank()}"
            )
            self.assertTrue(os.path.isfile(signal))
            with open(signal) as handle:
                self.assertEqual(handle.read(), "1")
            self.assertEqual(
                [entry[1] for entry in saves],
                [
                    os.path.join(output_dir, trainer_mod.OPTIMIZER_STATE_DIC),
                    os.path.join(output_dir, trainer_mod.MASTER_WEIGHT_DIC),
                ],
            )

        self.assertIs(seen["model_state"][key], model_shard)
        self.assertEqual(saves[0][0], {"a.moment1": "state"})
        self.assertEqual(saves[1][0], {"a.w_0": "master"})
        self.assertEqual([entry[2] for entry in saves], [True, True])


class DeferredTokenReplicaGroupTests(unittest.TestCase):
    """``Trainer._deferred_token_replica_group``: leftover data replicas."""

    def test_without_an_hcg_the_configured_dp_group_is_used(self):
        dp_group = _Group(2)
        stub = SimpleNamespace(hcg=None, dp_group=dp_group)

        self.assertIs(Trainer._deferred_token_replica_group(stub), dp_group)

    def test_a_multi_rank_sharding_group_wins(self):
        sharding = _Group(4)
        stub = SimpleNamespace(
            hcg=SimpleNamespace(get_sharding_parallel_group=lambda: sharding),
            dp_group=_Group(1),
        )

        self.assertIs(Trainer._deferred_token_replica_group(stub), sharding)

    def test_single_rank_sharding_falls_through_to_data_parallel(self):
        replicas = _Group(8)
        stub = SimpleNamespace(
            hcg=SimpleNamespace(get_sharding_parallel_group=lambda: _Group(1)),
            dp_group=None,
        )

        with mock.patch.object(
            parallel_state, "_DATA_PARALLEL_GROUP", replicas
        ):
            self.assertIs(Trainer._deferred_token_replica_group(stub), replicas)

    def test_failing_lookups_fall_back_to_the_configured_dp_group(self):
        def explode(*args, **kwargs):
            raise RuntimeError("parallel state is not initialized")

        dp_group = _Group(3)
        stub = SimpleNamespace(
            hcg=SimpleNamespace(get_sharding_parallel_group=explode),
            dp_group=dp_group,
        )

        with mock.patch.object(
            parallel_state, "get_data_parallel_group", explode
        ):
            self.assertIs(Trainer._deferred_token_replica_group(stub), dp_group)


class RequiresNativeTokenWeightedLoggingTests(unittest.TestCase):
    """``Trainer._requires_native_token_weighted_logging`` gating."""

    def _stub(self, compatible=True, defer=True, accum=1, group=None):
        return SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    use_accuracy_compatible=compatible,
                    defer_token_normalization=defer,
                )
            ),
            args=SimpleNamespace(gradient_accumulation_steps=accum),
            _deferred_token_replica_group=lambda: group,
        )

    def test_ordinary_runs_do_not_use_native_reporting(self):
        require = Trainer._requires_native_token_weighted_logging
        self.assertFalse(require(self._stub(compatible=False)))
        self.assertFalse(require(self._stub(defer=False)))

    def test_accumulated_microbatches_do_not_use_native_reporting(self):
        require = Trainer._requires_native_token_weighted_logging
        self.assertFalse(require(self._stub(accum=2, group=_Group(4))))

    def test_a_single_replica_does_not_need_native_reporting(self):
        require = Trainer._requires_native_token_weighted_logging
        self.assertFalse(require(self._stub(group=None)))
        self.assertFalse(require(self._stub(group=_Group(1))))

    def test_deferred_normalization_over_several_replicas_is_native(self):
        require = Trainer._requires_native_token_weighted_logging
        self.assertTrue(require(self._stub(group=_Group(4))))


class NoteNativeMicrobatchLossTests(unittest.TestCase):
    """``Trainer._note_native_microbatch_loss``: MAIN numerator carry."""

    def _stub(self, numerator=None, count=None):
        return SimpleNamespace(
            state=SimpleNamespace(global_step=2),
            _reporting_microbatch=1,
            _native_log_numerator=numerator,
            _native_log_count=count,
        )

    def _receipt(self, total, count, step=3, microbatch=1):
        return {
            "sum": _scalar(total),
            "count": float(count),
            "step": step,
            "microbatch": microbatch,
        }

    @contextlib.contextmanager
    def _language_loss(self, receipt, local_tokens, rank, pp_group):
        consumed = []

        def consume(step, microbatch):
            consumed.append((step, microbatch))
            return receipt

        with (
            mock.patch.object(
                loss_mod, "consume_main_reporting_microbatch", consume
            ),
            mock.patch.object(
                loss_mod,
                "get_local_main_valid_tokens",
                lambda: local_tokens,
            ),
            mock.patch.object(
                parallel_state, "_PIPELINE_MODEL_PARALLEL_GROUP", pp_group
            ),
            mock.patch.object(paddle.distributed, "get_rank", lambda: rank),
            _patched_collectives() as calls,
        ):
            yield consumed, calls

    def test_the_owner_publishes_its_numerator_over_the_pp_group(self):
        pp_group = _Group(2)
        stub = self._stub()

        with self._language_loss(
            self._receipt(12.0, 8.0), 8.0, 1, pp_group
        ) as (consumed, calls):
            Trainer._note_native_microbatch_loss(stub)

        self.assertEqual(consumed, [(3, 1)])
        self.assertEqual(len(calls.all_reduce_calls), 1)
        self.assertEqual(
            calls.all_reduce_calls[0][1], paddle.distributed.ReduceOp.MAX
        )
        self.assertIs(calls.all_reduce_calls[0][2], pp_group)
        self.assertEqual(calls.broadcast_calls[0][1], 1)
        self.assertIs(calls.broadcast_calls[0][2], pp_group)
        self.assertEqual(float(stub._native_log_numerator), 12.0)
        self.assertEqual(float(stub._native_log_count), 8.0)

    def test_following_microbatches_accumulate(self):
        pp_group = _Group(2)
        stub = self._stub(numerator=_scalar(12.0), count=_scalar(8.0))

        with self._language_loss(self._receipt(3.0, 8.0), 8.0, 1, pp_group) as (
            _,
            calls,
        ):
            Trainer._note_native_microbatch_loss(stub)

        self.assertEqual(float(stub._native_log_numerator), 15.0)
        self.assertEqual(float(stub._native_log_count), 16.0)
        self.assertEqual(len(calls.broadcast_calls), 1)

    def test_non_owner_stages_contribute_a_zero_numerator(self):
        pp_group = _Group(2)
        stub = self._stub()

        with self._language_loss(None, 8.0, 0, pp_group) as (_, calls):
            Trainer._note_native_microbatch_loss(stub)

        self.assertEqual(float(stub._native_log_numerator), 0.0)
        self.assertEqual(float(stub._native_log_count), 8.0)
        self.assertEqual(calls.broadcast_calls[0][1], 1)

    def test_a_receipt_off_the_loss_owner_is_rejected(self):
        with (
            self._language_loss(self._receipt(12.0, 8.0), 8.0, 0, _Group(2)),
            self.assertRaises(RuntimeError) as caught,
        ):
            Trainer._note_native_microbatch_loss(self._stub())

        self.assertIn("actual PP loss owner", str(caught.exception))

    def test_reporting_without_valid_tokens_is_rejected(self):
        with (
            self._language_loss(self._receipt(12.0, 0.0), None, 0, None),
            self.assertRaises(RuntimeError) as caught,
        ):
            Trainer._note_native_microbatch_loss(self._stub())

        self.assertIn("positive valid-token count", str(caught.exception))

    def test_a_stale_receipt_is_rejected(self):
        with (
            self._language_loss(self._receipt(12.0, 9.0), 8.0, 0, None),
            self.assertRaises(RuntimeError) as caught,
        ):
            Trainer._note_native_microbatch_loss(self._stub())

        self.assertIn("step/microbatch/count mismatch", str(caught.exception))


class NativeTokenWeightedLogLossTests(unittest.TestCase):
    """``Trainer._native_token_weighted_log_loss``: replica reduction."""

    def test_the_numerator_and_count_are_summed_then_divided(self):
        group = _Group(2)
        stub = SimpleNamespace(
            _deferred_token_replica_group=lambda: group,
            _native_log_numerator=_scalar(12.0),
            _native_log_count=_scalar(8.0),
        )

        with _patched_collectives() as calls:
            loss = Trainer._native_token_weighted_log_loss(stub)

        self.assertEqual(float(loss), 1.5)
        self.assertEqual(calls.ops(), [paddle.distributed.ReduceOp.SUM] * 2)
        self.assertEqual(calls.groups(), [group, group])
        self.assertIsNone(stub._native_log_numerator)
        self.assertIsNone(stub._native_log_count)

    def test_reporting_without_recorded_tokens_is_rejected(self):
        stub = SimpleNamespace(
            _deferred_token_replica_group=lambda: _Group(2),
            _native_log_numerator=None,
            _native_log_count=None,
        )

        with (
            _patched_collectives() as calls,
            self.assertRaises(RuntimeError) as caught,
        ):
            Trainer._native_token_weighted_log_loss(stub)

        self.assertIn("no valid MAIN tokens", str(caught.exception))
        self.assertEqual(len(calls.all_reduce_calls), 2)


class ResolveDeferredTokenNormalizationTests(unittest.TestCase):
    """``Trainer._resolve_deferred_token_normalization``: PP MAX + SUM."""

    def tearDown(self):
        loss_mod.clear_pending_gradient_divisor()

    def _stub(self, compatible=True, group=None):
        return SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(use_accuracy_compatible=compatible)
            ),
            _deferred_token_replica_group=lambda: group,
        )

    def test_without_the_language_loss_module_the_helper_is_a_noop(self):
        with mock.patch.dict(sys.modules, {LOSS_MODULE: None}):
            self.assertIsNone(
                Trainer._resolve_deferred_token_normalization(self._stub())
            )

    def test_ordinary_runs_publish_no_divisor(self):
        with _patched_collectives() as calls:
            Trainer._resolve_deferred_token_normalization(
                self._stub(compatible=False)
            )

        self.assertEqual(calls.all_reduce_calls, [])
        self.assertIsNone(loss_mod.get_pending_gradient_divisor())

    def test_a_single_process_run_keeps_its_local_divisor(self):
        loss_mod.set_pending_gradient_divisor(44.0)

        with _patched_collectives() as calls:
            Trainer._resolve_deferred_token_normalization(self._stub())

        self.assertEqual(calls.all_reduce_calls, [])
        self.assertEqual(loss_mod.get_pending_gradient_divisor(), 44.0)

    def test_single_rank_groups_skip_the_collectives(self):
        loss_mod.set_pending_gradient_divisor(44.0)

        with (
            mock.patch.object(
                paddle.distributed, "is_initialized", lambda: True
            ),
            mock.patch.object(
                parallel_state, "_PIPELINE_MODEL_PARALLEL_GROUP", None
            ),
            _patched_collectives() as calls,
        ):
            Trainer._resolve_deferred_token_normalization(
                self._stub(group=_Group(1))
            )

        self.assertEqual(calls.all_reduce_calls, [])
        self.assertEqual(loss_mod.get_pending_gradient_divisor(), 44.0)

    def test_the_divisor_is_maxed_over_pp_then_summed_over_replicas(self):
        pp_group = _Group(2)
        replicas = _Group(4)
        loss_mod.set_pending_gradient_divisor(44.0)
        seen = []

        def all_reduce(tensor, op=None, group=None, sync_op=True):
            seen.append((op, group, tensor.dtype, float(tensor[0])))
            if op == paddle.distributed.ReduceOp.SUM:
                # Two data replicas reported the same MAIN token count.
                paddle.assign(tensor * 2, tensor)

        with (
            mock.patch.object(
                paddle.distributed, "is_initialized", lambda: True
            ),
            mock.patch.object(
                parallel_state, "_PIPELINE_MODEL_PARALLEL_GROUP", pp_group
            ),
            mock.patch.object(paddle.distributed, "all_reduce", all_reduce),
        ):
            Trainer._resolve_deferred_token_normalization(
                self._stub(group=replicas)
            )

        self.assertEqual(
            [entry[0] for entry in seen],
            [
                paddle.distributed.ReduceOp.MAX,
                paddle.distributed.ReduceOp.SUM,
            ],
        )
        self.assertEqual([entry[1] for entry in seen], [pp_group, replicas])
        self.assertEqual([entry[2] for entry in seen], [paddle.float64] * 2)
        self.assertEqual([entry[3] for entry in seen], [44.0, 44.0])
        self.assertEqual(loss_mod.get_pending_gradient_divisor(), 88.0)

    def test_a_missing_replica_group_falls_back_to_data_parallel(self):
        replicas = _Group(2)
        loss_mod.set_pending_gradient_divisor(12.0)

        with (
            mock.patch.object(
                paddle.distributed, "is_initialized", lambda: True
            ),
            mock.patch.object(
                parallel_state, "_PIPELINE_MODEL_PARALLEL_GROUP", None
            ),
            mock.patch.object(parallel_state, "_DATA_PARALLEL_GROUP", replicas),
            _patched_collectives() as calls,
        ):
            Trainer._resolve_deferred_token_normalization(self._stub())

        self.assertEqual(calls.ops(), [paddle.distributed.ReduceOp.SUM])
        self.assertEqual(calls.groups(), [replicas])
        self.assertEqual(loss_mod.get_pending_gradient_divisor(), 12.0)


def _native_sharding_optimizer(param2bucket):
    """A real ``DygraphShardingOptimizerV2`` without its heavy setup."""
    optimizer = object.__new__(DygraphShardingOptimizerV2)
    optimizer.param2bucket = param2bucket
    return optimizer


def _bucket(nranks):
    group = None if nranks is None else _Group(nranks)
    return SimpleNamespace(_comm_group=group)


class ApplyDeferredTokenNormalizationTests(unittest.TestCase):
    """``Trainer._apply_deferred_token_normalization``: fp32 grad scaling."""

    def tearDown(self):
        loss_mod.clear_pending_gradient_divisor()

    def _stub(self, model, optimizer):
        return SimpleNamespace(model=model, optimizer=optimizer)

    def test_without_the_language_loss_module_the_helper_is_a_noop(self):
        grad = _fp32([8.0])
        model = SimpleNamespace(
            parameters=lambda: [_Param("w", main_grad=grad)]
        )

        with mock.patch.dict(sys.modules, {LOSS_MODULE: None}):
            Trainer._apply_deferred_token_normalization(
                self._stub(model, SimpleNamespace()), model
            )

        self.assertEqual(grad.tolist(), [8.0])

    def test_no_pending_divisor_leaves_the_gradients_alone(self):
        grad = _fp32([8.0])
        model = SimpleNamespace(
            parameters=lambda: [_Param("w", main_grad=grad)]
        )

        Trainer._apply_deferred_token_normalization(
            self._stub(model, SimpleNamespace()), model
        )

        self.assertEqual(grad.tolist(), [8.0])
        self.assertIsNone(loss_mod.get_pending_gradient_divisor())

    def test_unfused_optimizers_scale_every_buffer_by_one_over_n(self):
        main_grad = _fp32([8.0, 4.0])
        plain_grad = _fp32([2.0])
        params = [
            _Param("fused", main_grad=main_grad),
            _Param("plain", grad=plain_grad),
            _Param("frozen"),
        ]
        model = SimpleNamespace(
            _layers=SimpleNamespace(parameters=lambda: params)
        )
        # ``_inner_opt`` without a ``__dict__`` must not stop the walk.
        optimizer = SimpleNamespace(_inner_opt=7)
        loss_mod.set_pending_gradient_divisor(4.0)

        Trainer._apply_deferred_token_normalization(
            self._stub(model, optimizer), model
        )

        self.assertEqual(main_grad.tolist(), [2.0, 1.0])
        self.assertEqual(plain_grad.tolist(), [0.5])
        self.assertIsNone(loss_mod.get_pending_gradient_divisor())

    def test_each_optimizer_wrapper_is_visited_once(self):
        shared = SimpleNamespace()
        for optimizer in (
            SimpleNamespace(_inner_opt=shared, _optimizer=shared),
            None,
        ):
            grad = _fp32([8.0])
            params = [_Param("w", main_grad=grad)]
            model = SimpleNamespace(parameters=lambda: params)
            loss_mod.set_pending_gradient_divisor(4.0)

            Trainer._apply_deferred_token_normalization(
                self._stub(model, optimizer), model
            )

            self.assertEqual(grad.tolist(), [2.0])

    def test_native_fused_buffers_undo_the_bucket_average_first(self):
        main_grad = _fp32([16.0])
        plain_grad = _fp32([8.0])
        params = [
            _Param("fused", main_grad=main_grad),
            _Param("plain", grad=plain_grad),
            _Param("frozen"),
        ]
        model = SimpleNamespace(parameters=lambda: params)
        inner = _native_sharding_optimizer(
            {"fused": [_bucket(4), _bucket(2)], "plain": [_bucket(2)]}
        )
        optimizer = SimpleNamespace(_inner_opt=inner)
        loss_mod.set_pending_gradient_divisor(8.0)

        Trainer._apply_deferred_token_normalization(
            self._stub(model, optimizer), model
        )

        # max(nranks) / divisor, so 4/8 for the first and 2/8 for the second.
        self.assertEqual(main_grad.tolist(), [8.0])
        self.assertEqual(plain_grad.tolist(), [2.0])

    def test_several_native_sharding_optimizers_are_rejected(self):
        params = [_Param("w", main_grad=_fp32([8.0]))]
        model = SimpleNamespace(parameters=lambda: params)
        inner = _native_sharding_optimizer({"w": [_bucket(2)]})
        inner._inner_opt = _native_sharding_optimizer({"w": [_bucket(2)]})
        loss_mod.set_pending_gradient_divisor(8.0)

        with self.assertRaises(RuntimeError) as caught:
            Trainer._apply_deferred_token_normalization(
                self._stub(model, SimpleNamespace(_opt=inner)), model
            )

        self.assertIn("multiple native sharding", str(caught.exception))

    def test_a_native_optimizer_without_bucket_mapping_is_rejected(self):
        params = [_Param("w", main_grad=_fp32([8.0]))]
        model = SimpleNamespace(parameters=lambda: params)
        optimizer = SimpleNamespace(inner_opt=_native_sharding_optimizer({}))
        loss_mod.set_pending_gradient_divisor(8.0)

        with self.assertRaises(RuntimeError) as caught:
            Trainer._apply_deferred_token_normalization(
                self._stub(model, optimizer), model
            )

        self.assertIn("missing param2bucket", str(caught.exception))

    def test_an_unmapped_gradient_bearing_parameter_is_rejected(self):
        grad = _fp32([8.0])
        params = [_Param("w", main_grad=grad)]
        model = SimpleNamespace(parameters=lambda: params)
        optimizer = SimpleNamespace(
            _optimizer=_native_sharding_optimizer({"other": [_bucket(2)]})
        )
        loss_mod.set_pending_gradient_divisor(8.0)

        with self.assertRaises(RuntimeError) as caught:
            Trainer._apply_deferred_token_normalization(
                self._stub(model, optimizer), model
            )

        self.assertIn("FusedCommBuffer mapping", str(caught.exception))
        self.assertEqual(grad.tolist(), [8.0])

    def test_a_bucket_without_a_comm_group_is_rejected(self):
        params = [_Param("w", main_grad=_fp32([8.0]))]
        model = SimpleNamespace(parameters=lambda: params)
        optimizer = SimpleNamespace(
            _inner_opt=_native_sharding_optimizer({"w": [_bucket(None)]})
        )
        loss_mod.set_pending_gradient_divisor(8.0)

        with self.assertRaises(RuntimeError) as caught:
            Trainer._apply_deferred_token_normalization(
                self._stub(model, optimizer), model
            )

        self.assertIn(
            "invalid comm group on fused buffer", str(caught.exception)
        )

    def test_a_bucket_group_without_ranks_is_rejected(self):
        params = [_Param("w", main_grad=_fp32([8.0]))]
        model = SimpleNamespace(parameters=lambda: params)
        optimizer = SimpleNamespace(
            _inner_opt=_native_sharding_optimizer({"w": [_bucket(0)]})
        )
        loss_mod.set_pending_gradient_divisor(8.0)

        with self.assertRaises(RuntimeError) as caught:
            Trainer._apply_deferred_token_normalization(
                self._stub(model, optimizer), model
            )

        self.assertIn("invalid comm group nranks", str(caught.exception))


class MaybeLogSaveEvaluateTests(unittest.TestCase):
    """``Trainer._maybe_log_save_evaluate``: which average reaches the log."""

    def _stub(self, native, avg_loss, logged, num_steps=5):
        calls = {"native": 0, "gathered": []}

        def native_log_loss():
            calls["native"] += 1
            return avg_loss

        def nested_gather(tensor):
            calls["gathered"].append(tensor)
            return tensor

        return (
            SimpleNamespace(
                args=SimpleNamespace(
                    train_batch_size=2,
                    gradient_accumulation_steps=1,
                    dataset_world_size=1,
                    skip_memory_metrics=True,
                ),
                control=SimpleNamespace(
                    should_log=True,
                    should_evaluate=False,
                    should_save=False,
                    should_save_hf=False,
                ),
                state=SimpleNamespace(global_step=10),
                model=SimpleNamespace(
                    config=SimpleNamespace(seq_length=128),
                    get_hardware_flops=lambda: 1.0,
                ),
                global_training_logs={},
                _globalstep_last_logged=10 - num_steps,
                _skip_steps_since_last_logged=0,
                _globalstep_last_start_time=time.time() - 1.0,
                _total_loss_scalar=0.0,
                _get_item_from_loss=types.MethodType(
                    Trainer._get_item_from_loss, SimpleNamespace()
                ),
                _requires_native_token_weighted_logging=lambda: native,
                _native_token_weighted_log_loss=native_log_loss,
                _nested_gather=nested_gather,
                _get_learning_rate=lambda: 1e-4,
                log=lambda logs, **kwargs: logged.append((logs, kwargs)),
            ),
            calls,
        )

    def test_native_reporting_logs_the_token_weighted_average(self):
        logged = []
        stub, calls = self._stub(True, _scalar(2.5), logged)
        tr_loss = _scalar(99.0)

        Trainer._maybe_log_save_evaluate(stub, tr_loss, None, 0, None)

        self.assertEqual(calls["native"], 1)
        self.assertEqual(calls["gathered"], [])
        self.assertEqual(logged[0][0]["loss"], 2.5)
        self.assertEqual(logged[0][1]["raw_loss"], 2.5)
        self.assertEqual(logged[0][0]["global_step"], 10)
        self.assertEqual(float(tr_loss), 0.0)
        # The accumulated sum contract keeps ``raw_loss * num_steps``.
        self.assertEqual(stub._total_loss_scalar, 12.5)

    def test_ordinary_reporting_divides_the_gathered_sum(self):
        logged = []
        stub, calls = self._stub(False, None, logged)
        tr_loss = _scalar(10.0)

        Trainer._maybe_log_save_evaluate(stub, tr_loss, None, 0, None)

        self.assertEqual(calls["native"], 0)
        self.assertEqual(len(calls["gathered"]), 1)
        self.assertEqual(logged[0][0]["loss"], 2.0)
        self.assertEqual(stub._total_loss_scalar, 10.0)
        self.assertEqual(float(tr_loss), 0.0)

    def test_native_reporting_without_new_steps_logs_zero(self):
        logged = []
        stub, _calls = self._stub(True, _scalar(2.5), logged, num_steps=0)
        tr_loss = _scalar(7.0)

        Trainer._maybe_log_save_evaluate(stub, tr_loss, None, 0, None)

        self.assertEqual(logged[0][0]["loss"], 0.0)
        self.assertEqual(stub._total_loss_scalar, 0.0)


class BuildGradClipTests(unittest.TestCase):
    """``Trainer._build_grad_clip``: which clip recipe is constructed."""

    def _stub(self, max_grad_norm, accuracy_target):
        return SimpleNamespace(
            args=SimpleNamespace(max_grad_norm=max_grad_norm),
            model=SimpleNamespace(
                config=SimpleNamespace(use_accuracy_compatible=accuracy_target)
            ),
        )

    def test_disabled_clipping_builds_nothing(self):
        self.assertIsNone(Trainer._build_grad_clip(self._stub(0.0, False)))

    def test_ordinary_runs_keep_paddles_stock_clip(self):
        clip = Trainer._build_grad_clip(self._stub(1.0, False))

        self.assertEqual(clip.clip_norm, 1.0)


class WrapDistributedOptimizerTests(unittest.TestCase):
    """``Trainer._wrap_distributed_optimizer``: the global-norm hook."""

    def _setup(self, train_mtp_only=False):
        recorded = []

        def global_norm(*args):
            recorded.append(args)

        clip = SimpleNamespace(_global_norm=global_norm)
        dist_optimizer = object.__new__(HybridParallelOptimizer)
        dist_optimizer._inner_opt = SimpleNamespace(_grad_clip=clip)
        stub = SimpleNamespace(
            args=SimpleNamespace(
                use_expert_parallel=False,
                max_grad_norm=1.0,
                train_mtp_only=train_mtp_only,
            ),
            global_training_logs={},
            optimizer=dist_optimizer,
        )
        return stub, dist_optimizer, clip, recorded

    def _wrap(self, stub, dist_optimizer):
        with mock.patch.object(
            trainer_mod.fleet,
            "distributed_optimizer",
            lambda optimizer: dist_optimizer,
        ):
            return Trainer._wrap_distributed_optimizer(stub, SimpleNamespace())

    def test_the_moe_partition_is_added_before_the_square_root(self):
        stub, dist_optimizer, clip, recorded = self._setup()

        self._wrap(stub, dist_optimizer)
        clip._global_norm(
            _scalar(9.0), _scalar(7.0), _scalar(5.0), _scalar(4.0)
        )

        self.assertEqual(stub.global_training_logs["global_norm"], 5.0)
        self.assertEqual(len(recorded[0]), 4)

    def test_mtp_only_training_reports_no_global_norm(self):
        stub, dist_optimizer, clip, recorded = self._setup(train_mtp_only=True)

        self._wrap(stub, dist_optimizer)
        clip._global_norm(_scalar(9.0), _scalar(16.0))

        self.assertEqual(stub.global_training_logs, {})
        self.assertEqual(recorded, [])


class HfCadencePathsTests(unittest.TestCase):
    """``Trainer._hf_cadence_paths``: mid-training HF snapshot layout."""

    def _stub(self, output_dir, save_hf_output_dir=None, step=7):
        return SimpleNamespace(
            args=SimpleNamespace(
                output_dir=output_dir,
                save_hf_output_dir=save_hf_output_dir,
            ),
            state=SimpleNamespace(global_step=step),
        )

    def test_the_default_layout_nests_under_the_output_dir(self):
        run_dir, ckpt_path = Trainer._hf_cadence_paths(self._stub("outputs"))

        self.assertEqual(run_dir, "outputs")
        self.assertEqual(ckpt_path, os.path.join("outputs", "hf_checkpoint-7"))

    def test_an_explicit_step_overrides_the_global_step(self):
        _run_dir, ckpt_path = Trainer._hf_cadence_paths(
            self._stub("outputs"), step=11
        )

        self.assertEqual(ckpt_path, os.path.join("outputs", "hf_checkpoint-11"))

    def test_the_opt_in_root_moves_both_paths(self):
        run_dir, ckpt_path = Trainer._hf_cadence_paths(
            self._stub("outputs", save_hf_output_dir="hf_root")
        )

        self.assertEqual(run_dir, "hf_root")
        self.assertEqual(ckpt_path, os.path.join("hf_root", "hf_checkpoint-7"))


if __name__ == "__main__":
    unittest.main()
