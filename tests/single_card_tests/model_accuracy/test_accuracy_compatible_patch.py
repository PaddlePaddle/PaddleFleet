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

"""Paddle-only helpers of the DSV4 accuracy-compatible patch module.

``accuracy_compatible_patch`` holds the Megatron/Torch-aligned replay paths
that ``FLAGS_use_dsv4_accuracy`` switches on. The Torch-backed kernels need a
Torch build in the environment, but the trainer/data plumbing, the pure-Paddle
PyLayers and the flag/install gating are all exercisable on a single card, so
they are pinned here.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet import accuracy_compatible_patch as acp


class TestFlagGating(unittest.TestCase):
    def test_enabled_only_for_explicit_one(self):
        for value, expected in (("1", True), ("0", False), ("true", False)):
            with patch.dict(os.environ, {"FLAGS_use_dsv4_accuracy": value}):
                self.assertIs(acp._accuracy_compatible_enabled(), expected)

    def test_install_is_a_no_op_when_flag_is_off(self):
        with (
            patch.dict(os.environ, {"FLAGS_use_dsv4_accuracy": "0"}),
            patch.object(acp, "_install_fusion_patch") as fusion,
        ):
            self.assertFalse(acp.install_accuracy_compatible_paddle_patches())
        fusion.assert_not_called()

    def test_install_runs_every_patch_once(self):
        with (
            patch.dict(os.environ, {"FLAGS_use_dsv4_accuracy": "1"}),
            patch.object(acp, "_PADDLE_RUNTIME_PATCHED", False),
            patch.object(acp, "_install_fusion_patch") as fusion,
            patch.object(acp, "_install_sharding_shape_patch") as sharding,
            patch.object(acp, "_install_adamw_patch") as adamw,
        ):
            self.assertTrue(acp.install_accuracy_compatible_paddle_patches())
            self.assertTrue(acp.install_accuracy_compatible_paddle_patches())
        fusion.assert_called_once()
        sharding.assert_called_once()
        adamw.assert_called_once()


class _FakeCore:
    def __init__(self, groups):
        self._groups = groups
        self.calls = []

    def eager_assign_group_by_size(self, parameters, is_sparse, group_sizes):
        self.calls.append((len(parameters), is_sparse, group_sizes))
        return self._groups


class _FakeHelper:
    def __init__(self, groups):
        self.core = _FakeCore(groups)


class _NamedParam:
    def __init__(self, name):
        self.name = name


class TestGroupIndices(unittest.TestCase):
    """``gptlm_head`` parameters must end up alone in their fusion group."""

    def test_lm_head_is_split_into_its_own_group(self):
        params = [
            _NamedParam("embedding"),
            _NamedParam("layer0.gptlm_head.weight"),
            _NamedParam("layer1.linear"),
            _NamedParam("layer2.linear"),
        ]
        helper = _FakeHelper([[0, 1, 2, 3]])

        groups = acp._group_indices(params, 1024, helper)

        self.assertEqual(groups, [[0], [1], [2, 3]])
        self.assertEqual(helper.core.calls[0][2], [1024, 1024])

    def test_groups_without_lm_head_are_preserved(self):
        params = [_NamedParam(f"linear{i}") for i in range(3)]
        helper = _FakeHelper([[0, 1], [2]])

        self.assertEqual(acp._group_indices(params, 8, helper), [[0, 1], [2]])


class TestSumForSmallRows(unittest.TestCase):
    """Row padding must not change the reduction result, only its kernel."""

    def test_short_matrix_is_padded_but_keeps_its_row_count(self):
        value = paddle.to_tensor(np.arange(12, dtype="float32").reshape([3, 4]))

        out = acp.sum_for_small_rows(value)

        self.assertEqual(out.shape, [3, 1])
        np.testing.assert_allclose(
            out.numpy(), value.numpy().sum(-1, keepdims=True)
        )

    def test_tall_matrix_uses_the_plain_reduction(self):
        value = paddle.ones([16, 3], dtype="float32")

        out = acp.sum_for_small_rows(value)

        self.assertEqual(out.shape, [16, 1])
        np.testing.assert_allclose(out.numpy(), np.full([16, 1], 3.0))

    def test_non_matrix_input_falls_back_to_plain_reduction(self):
        value = paddle.ones([2, 3, 4], dtype="float32")

        self.assertEqual(acp.sum_for_small_rows(value).shape, [2, 3, 1])


class TestLossScaleBeforeBackward(unittest.TestCase):
    def setUp(self):
        self.addCleanup(acp.LossScaleBeforeBackward.set_acc_steps, 1)

    def test_single_step_leaves_the_loss_untouched(self):
        acp.set_loss_acc_steps(1)
        loss = paddle.to_tensor(2.0)

        self.assertIs(acp.LossScaleBeforeBackward.scale(loss), loss)

    def test_accumulation_divides_the_loss(self):
        acp.set_loss_acc_steps(4)
        loss = paddle.to_tensor(2.0)

        np.testing.assert_allclose(
            acp.LossScaleBeforeBackward.scale(loss).numpy(), 0.5
        )

    def test_non_positive_steps_are_clamped_to_one(self):
        acp.set_loss_acc_steps(0)
        loss = paddle.to_tensor(3.0)

        self.assertIs(acp.LossScaleBeforeBackward.scale(loss), loss)


class TestCompatibleCSASinkSoftmax(unittest.TestCase):
    """Forward is a numerically stable softmax over ``[scores, sink]``."""

    def test_forward_matches_softmax_with_the_sink_logit(self):
        paddle.seed(2026)
        scores = paddle.randn([2, 3, 4, 5], dtype="float32")
        sink = paddle.randn([2, 3, 4, 1], dtype="float32")

        with paddle.no_grad():
            out = acp.CompatibleCSASinkSoftmax.apply(scores, sink)

        joint = paddle.concat([scores, sink], axis=-1)
        expected = paddle.nn.functional.softmax(joint, axis=-1)[..., :-1]
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_weights_leave_room_for_the_sink(self):
        scores = paddle.zeros([1, 1, 1, 3], dtype="float32")
        sink = paddle.zeros([1, 1, 1, 1], dtype="float32")

        with paddle.no_grad():
            out = acp.CompatibleCSASinkSoftmax.apply(scores, sink)

        np.testing.assert_allclose(
            out.numpy(), np.full([1, 1, 1, 3], 0.25), rtol=1e-6
        )

    """The hand-written RMSNorm gradient must match Paddle autograd."""

    def test_forward_and_backward_match_the_eager_formula(self):
        paddle.seed(7)
        eps = 1e-6
        data = paddle.randn([2, 4, 8], dtype="float32")

        q_patched = data.clone().detach()
        q_patched.stop_gradient = False
        out_patched = acp.CompatibleQRMSNorm.apply(q_patched, eps)
        out_patched.sum().backward()

        q_ref = data.clone().detach()
        q_ref.stop_gradient = False
        r_ref = paddle.rsqrt(q_ref.square().mean(axis=-1, keepdim=True) + eps)
        out_ref = q_ref * r_ref
        out_ref.sum().backward()

        np.testing.assert_allclose(
            out_patched.numpy(), out_ref.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            q_patched.grad.numpy(), q_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )


class TestCompatibleEmbeddingIndexBackward(unittest.TestCase):
    def test_forward_is_a_plain_row_gather(self):
        weight = paddle.to_tensor(
            np.arange(12, dtype="float32").reshape([4, 3])
        )
        ids = paddle.to_tensor([[0, 3], [2, 1]], dtype="int64")

        with paddle.no_grad():
            out = acp.CompatibleEmbeddingIndexBackward.apply(ids, weight)

        np.testing.assert_allclose(out.numpy(), weight.numpy()[ids.numpy()])


class TestMoEInputBranches(unittest.TestCase):
    """The three branches fan in as ``(routed + router) + shared``."""

    def test_forward_clones_and_backward_sums_every_branch(self):
        x = paddle.to_tensor([1.0, 2.0], dtype="float32")
        x.stop_gradient = False

        routed, router, shared = acp.MoEInputBranches.apply(x)
        (routed * 1.0 + router * 2.0 + shared * 3.0).sum().backward()

        np.testing.assert_allclose(routed.numpy(), x.numpy())
        np.testing.assert_allclose(x.grad.numpy(), [6.0, 6.0])


class TestIndicesToMultihot(unittest.TestCase):
    """``-1`` is the DeepEP padding marker and drops out of both outputs."""

    def test_padding_slots_are_ignored(self):
        indices = paddle.to_tensor([[0, 2], [1, -1]], dtype="int64")
        probs = paddle.to_tensor([[0.25, 0.75], [0.5, 0.125]], dtype="float32")

        routing_map, multihot_probs = acp.indices_to_multihot(indices, probs, 3)

        self.assertEqual(routing_map.dtype, paddle.bool)
        np.testing.assert_array_equal(
            routing_map.numpy(),
            np.array([[True, False, True], [False, True, False]]),
        )
        np.testing.assert_allclose(
            multihot_probs.numpy(),
            np.array([[0.25, 0.0, 0.75], [0.0, 0.5, 0.0]], dtype="float32"),
        )

    def test_repeated_experts_accumulate_their_probability(self):
        indices = paddle.to_tensor([[1, 1]], dtype="int64")
        probs = paddle.to_tensor([[0.25, 0.5]], dtype="float32")

        routing_map, multihot_probs = acp.indices_to_multihot(indices, probs, 2)

        np.testing.assert_array_equal(
            routing_map.numpy(), np.array([[False, True]])
        )
        np.testing.assert_allclose(
            multihot_probs.numpy(), np.array([[0.0, 0.75]], dtype="float32")
        )


class TestProjectionAndNorm(unittest.TestCase):
    """mHC projection keeps the Megatron matmul plus an RMS scale factor."""

    def test_shapes_and_values_of_the_projection_and_scale(self):
        paddle.seed(11)
        eps = 1e-6
        x = paddle.randn([2, 3, 4], dtype="float32")
        weight = paddle.randn([4, 5], dtype="float32")

        with paddle.no_grad():
            proj, r = acp.compatible_projection_and_norm(x, weight, eps)

        self.assertEqual(proj.shape, [2, 3, 5])
        self.assertEqual(r.shape, [2, 3, 1])
        expected_proj = paddle.matmul(x.reshape([-1, 4]), weight)
        np.testing.assert_allclose(
            proj.reshape([-1, 5]).numpy(),
            expected_proj.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        norm = np.linalg.norm(
            x.numpy().reshape([-1, 4]), axis=-1, keepdims=True
        )
        expected_r = 1.0 / (norm / np.sqrt(4.0) + eps)
        np.testing.assert_allclose(
            r.reshape([-1, 1]).numpy(), expected_r, rtol=1e-5, atol=1e-5
        )

    def test_scale_is_registered_as_a_stop_gradient_free_pylayer(self):
        x = paddle.ones([1, 2, 4], dtype="float32")
        weight = paddle.ones([4, 3], dtype="float32")

        with paddle.no_grad():
            _, r = acp.compatible_projection_and_norm(x, weight, 0.0)

        np.testing.assert_allclose(
            r.numpy(), np.full([1, 2, 1], 1.0), rtol=1e-6
        )


class TestSeqfirstWgradAccumulation(unittest.TestCase):
    def test_stash_starts_fresh_and_then_accumulates(self):
        weight = paddle.ones([2, 2], dtype="float32")

        acp._accumulate_hc_mapping_seqfirst_wgrad(weight, None)
        self.assertIsNone(
            getattr(weight, "_dsv4_hc_mapping_seqfirst_wgrad", None)
        )

        acp._accumulate_hc_mapping_seqfirst_wgrad(
            weight, paddle.full([2, 2], 2.0, dtype="float32")
        )
        acp._accumulate_hc_mapping_seqfirst_wgrad(
            weight, paddle.full([2, 2], 3.0, dtype="float32")
        )

        stash = weight._dsv4_hc_mapping_seqfirst_wgrad
        self.assertEqual(stash.dtype, paddle.float32)
        np.testing.assert_allclose(stash.numpy(), np.full([2, 2], 5.0))


class _TrainingArgs:
    def __init__(self, max_seq_len, gradient_accumulation_steps=1):
        self.max_seq_len = max_seq_len
        self.gradient_accumulation_steps = gradient_accumulation_steps


def _identity_padding(length, training_args):
    return length


class TestFixedTrainingData(unittest.TestCase):
    """``LOAD_FIXED_DATA_PATH`` replay: file resolution and sample layout."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        acp._LOAD_FIXED_DATA_CALL_COUNT = 0
        self.addCleanup(setattr, acp, "_LOAD_FIXED_DATA_CALL_COUNT", 0)

    def _dump(self, name, tokens, labels):
        np.save(os.path.join(self.root, f"tokens_{name}.npy"), tokens)
        np.save(os.path.join(self.root, f"labels_{name}.npy"), labels)

    def test_returns_none_without_the_env_var(self):
        env = dict(os.environ)
        env.pop("LOAD_FIXED_DATA_PATH", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(
                acp.load_fixed_training_data(
                    _TrainingArgs(8), 0, _identity_padding
                )
            )

    def test_one_dimensional_tokens_produce_a_flat_sample(self):
        self._dump(
            "step0_micro0_rank0_seq8",
            np.arange(8, dtype="int64"),
            np.arange(1, 9, dtype="int64"),
        )
        with patch.dict(os.environ, {"LOAD_FIXED_DATA_PATH": self.root}):
            fixed = acp.load_fixed_training_data(
                _TrainingArgs(8), 0, _identity_padding
            )

        self.assertFalse(fixed.batched)
        self.assertEqual(fixed.max_seq_len, 8)
        self.assertEqual(fixed.position_ids, list(range(8)))
        self.assertEqual(acp.fixed_data_iter(fixed, ["a", "b"]), ["a", "b"])
        position_ids, tokens, labels, again = acp.fixed_data_sample(fixed, 0)
        self.assertEqual(tokens, [fixed.input_ids])
        self.assertEqual(labels, [fixed.labels])
        self.assertEqual(position_ids, again)

    def test_two_dimensional_tokens_produce_a_batched_sample(self):
        tokens = np.arange(6, dtype="int64").reshape([2, 3])
        self._dump("step0_micro0_rank0_seq3", tokens, tokens + 1)
        with patch.dict(os.environ, {"LOAD_FIXED_DATA_PATH": self.root}):
            fixed = acp.load_fixed_training_data(
                _TrainingArgs(3), 0, _identity_padding
            )

        self.assertTrue(fixed.batched)
        self.assertEqual(fixed.max_seq_len, 3)
        self.assertEqual(acp.fixed_data_iter(fixed, ["only-one"]), [None, None])
        position_ids, sample_tokens, sample_labels, _ = acp.fixed_data_sample(
            fixed, 1
        )
        self.assertEqual(sample_tokens, [[3, 4, 5]])
        self.assertEqual(sample_labels, [[4, 5, 6]])
        self.assertEqual(position_ids, [[0, 1, 2]])

    def test_micro_step_advances_with_the_accumulation_count(self):
        for micro in range(2):
            self._dump(
                f"step0_micro{micro}_rank0_seq4",
                np.full([4], micro, dtype="int64"),
                np.full([4], micro, dtype="int64"),
            )
        args = _TrainingArgs(4, gradient_accumulation_steps=2)
        with patch.dict(os.environ, {"LOAD_FIXED_DATA_PATH": self.root}):
            first = acp.load_fixed_training_data(args, 0, _identity_padding)
            second = acp.load_fixed_training_data(args, 0, _identity_padding)

        self.assertEqual(first.input_ids, [0, 0, 0, 0])
        self.assertEqual(second.input_ids, [1, 1, 1, 1])

    def test_step_only_file_name_is_accepted(self):
        self._dump(
            "step0_rank0_seq4",
            np.zeros([4], dtype="int64"),
            np.zeros([4], dtype="int64"),
        )
        tokens_file, labels_file = acp._resolve_fixed_training_files(
            self.root, 0, 0, 0, 4
        )
        self.assertTrue(tokens_file.endswith("tokens_step0_rank0_seq4.npy"))
        self.assertTrue(labels_file.endswith("labels_step0_rank0_seq4.npy"))

    def test_manifest_entry_is_used_when_the_name_does_not_match(self):
        np.save(os.path.join(self.root, "t.npy"), np.zeros([4], dtype="int64"))
        np.save(os.path.join(self.root, "l.npy"), np.zeros([4], dtype="int64"))
        manifest = os.path.join(self.root, "manifest_rank0.jsonl")
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(
                json.dumps(
                    {
                        "step": 0,
                        "micro": 0,
                        "rank": 0,
                        "tokens_file": "t.npy",
                        "labels_file": "l.npy",
                    }
                )
                + "\n"
            )

        tokens_file, labels_file = acp._resolve_fixed_training_files(
            self.root, 0, 0, 0, 4
        )

        self.assertTrue(tokens_file.endswith("t.npy"))
        self.assertTrue(labels_file.endswith("l.npy"))

    def test_sequence_length_is_globbed_when_it_is_unknown(self):
        self._dump(
            "step1_micro0_rank0_seq16",
            np.zeros([16], dtype="int64"),
            np.zeros([16], dtype="int64"),
        )

        tokens_file, labels_file = acp._resolve_fixed_training_files(
            self.root, 1, 0, 0, 99
        )

        self.assertTrue(tokens_file.endswith("seq16.npy"))
        self.assertTrue(
            labels_file.endswith("labels_step1_micro0_rank0_seq16.npy")
        )

    def test_missing_files_raise_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            acp._resolve_fixed_training_files(self.root, 0, 0, 0, 4)

    def test_label_shape_mismatch_is_rejected(self):
        self._dump(
            "step0_micro0_rank0_seq4",
            np.zeros([2, 4], dtype="int64"),
            np.zeros([2, 3], dtype="int64"),
        )
        with (
            patch.dict(os.environ, {"LOAD_FIXED_DATA_PATH": self.root}),
            self.assertRaises(ValueError),
        ):
            acp.load_fixed_training_data(_TrainingArgs(4), 0, _identity_padding)

    def test_three_dimensional_tokens_are_rejected(self):
        self._dump(
            "step0_micro0_rank0_seq4",
            np.zeros([1, 2, 4], dtype="int64"),
            np.zeros([1, 2, 4], dtype="int64"),
        )
        with (
            patch.dict(os.environ, {"LOAD_FIXED_DATA_PATH": self.root}),
            self.assertRaises(ValueError),
        ):
            acp.load_fixed_training_data(_TrainingArgs(4), 0, _identity_padding)


class _StashModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())


class TestFlushSequenceFirstWgrad(unittest.TestCase):
    """Stashed seq-first wgrads replace the real grad exactly once."""

    def test_main_grad_is_overwritten_and_the_stash_is_cleared(self):
        param = paddle.ones([2, 2], dtype="float32")
        param.main_grad = paddle.zeros([2, 2], dtype="float32")
        param._dsv4_hc_mapping_seqfirst_wgrad = paddle.full(
            [2, 2], 4.0, dtype="float32"
        )

        acp.flush_sequence_first_wgrad(_StashModel({"w": param}))

        np.testing.assert_allclose(
            param.main_grad.numpy(), np.full([2, 2], 4.0)
        )
        self.assertIsNone(param._dsv4_hc_mapping_seqfirst_wgrad)

    def test_attention_stash_is_flushed_as_well(self):
        param = paddle.ones([2], dtype="float32")
        param.main_grad = paddle.zeros([2], dtype="float32")
        param._dsv4_attn_o_group_seqfirst_wgrad = paddle.to_tensor(
            [1.0, 2.0], dtype="float32"
        )

        acp.flush_sequence_first_wgrad(_StashModel({"o": param}))

        np.testing.assert_allclose(param.main_grad.numpy(), [1.0, 2.0])

    def test_parameters_without_a_stash_are_left_alone(self):
        param = paddle.ones([2], dtype="float32")
        param.main_grad = paddle.zeros([2], dtype="float32")

        acp.flush_sequence_first_wgrad(_StashModel({"w": param}))

        np.testing.assert_allclose(param.main_grad.numpy(), [0.0, 0.0])

    def test_a_parameter_without_any_grad_is_skipped(self):
        param = paddle.ones([2], dtype="float32")
        param._dsv4_hc_mapping_seqfirst_wgrad = paddle.to_tensor(
            [1.0, 2.0], dtype="float32"
        )

        acp.flush_sequence_first_wgrad(_StashModel({"w": param}))

        self.assertIsNotNone(param._dsv4_hc_mapping_seqfirst_wgrad)


class TestHasOptimizerState(unittest.TestCase):
    """Pure predicate shared by the replay and HACK_CONVERT_CKPT branches."""

    def test_true_when_any_state_name_is_present(self):
        metadata = {"linear.w_0_moment1_0": object()}

        self.assertTrue(
            acp.has_optimizer_state(
                "linear.w_0", metadata, ["_moment1_0", "_moment2_0"]
            )
        )

    def test_false_when_no_state_name_matches(self):
        metadata = {"other.w_0_moment1_0": object()}

        self.assertFalse(
            acp.has_optimizer_state("linear.w_0", metadata, ["_moment1_0"])
        )

    def test_false_for_an_empty_state_name_list(self):
        self.assertFalse(acp.has_optimizer_state("linear.w_0", {}, []))


class TestPipelineLossScale(unittest.TestCase):
    def test_single_step_does_not_touch_the_auto_scalers(self):
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
        )

        before = MTPLossAutoScaler.main_loss_backward_scale
        acp.set_pipeline_loss_scale(1)

        self.assertIs(MTPLossAutoScaler.main_loss_backward_scale, before)

    def test_accumulation_installs_the_reciprocal_scale(self):
        from paddlefleet.transformer.dsa_attention import (
            DSAIndexerLossAutoScaler,
        )
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
        )

        mtp_before = MTPLossAutoScaler.main_loss_backward_scale
        dsa_before = DSAIndexerLossAutoScaler._main_loss_backward_scale
        self.addCleanup(MTPLossAutoScaler.set_loss_scale, mtp_before)
        self.addCleanup(DSAIndexerLossAutoScaler.set_loss_scale, dsa_before)

        acp.set_pipeline_loss_scale(4)

        np.testing.assert_allclose(
            MTPLossAutoScaler.main_loss_backward_scale.numpy(), 0.25
        )
        np.testing.assert_allclose(
            DSAIndexerLossAutoScaler._main_loss_backward_scale.numpy(), 0.25
        )


if __name__ == "__main__":
    unittest.main()
