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

"""``TransformerConfig.use_dsv4_accuracy`` must gate the MoE dispatch/expert call sites.

The flag defaults to 0, and with it off the numeric paths have to stay exactly
where they were before the DSV4 replay landed - other alignment targets run
with ``use_accuracy_compatible=True`` (and possibly
``FLAGS_use_accuracy_compatible_kernel=1``) but *without* this flag, so a DSV4
branch that keys off the older switches would silently change their loss curve.

This module pins the MoE-side call sites that the sibling
``test_dsv4_accuracy_flag_gating.py`` does not cover: the DeepEP token
dispatcher's multihot build, its ``global_input_probs`` capture and its restore
probs, plus the accuracy-compatible-kernel probs requirement in
``MoELayer.expert_forward``. Every test drives the gated unit directly on a
single card (no process group / collectives) and pins both sides of the flag
with an observable difference.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddlefleet import accuracy_compatible_patch
from paddlefleet.transformer.moe import moe_layer, moe_router, token_dispatcher


def _dsv4_flag(module, enabled):
    return patch.object(
        module, "use_dsv4_accuracy_compatible", return_value=enabled
    )


class TestIndicesToMultihotGating(unittest.TestCase):
    """``_DeepEPManager._indices_to_multihot`` swaps to the DSV4 replay helper.

    token_dispatcher.py:944 - with the flag off the manual paddle scatter builds
    the [tokens, experts] multihot map; with it on the DSV4
    ``accuracy_compatible_patch.indices_to_multihot`` replay runs instead. Both
    must yield the same routing map / probs.
    """

    def setUp(self):
        self.owner = types.SimpleNamespace(num_local_experts=3)
        self.indices = paddle.to_tensor([[0, 2], [1, -1]], dtype="int64")
        self.probs = paddle.to_tensor([[0.5, 0.5], [1.0, 0.0]], dtype="float32")

    def _run(self, enabled):
        with _dsv4_flag(token_dispatcher, enabled):
            return token_dispatcher._DeepEPManager._indices_to_multihot(
                self.owner, self.indices, self.probs
            )

    def test_flag_off_builds_the_multihot_map_manually(self):
        with (
            _dsv4_flag(token_dispatcher, False),
            patch.object(
                accuracy_compatible_patch,
                "indices_to_multihot",
                wraps=accuracy_compatible_patch.indices_to_multihot,
            ) as replay,
        ):
            routing_map, probs = (
                token_dispatcher._DeepEPManager._indices_to_multihot(
                    self.owner, self.indices, self.probs
                )
            )

        replay.assert_not_called()
        np.testing.assert_array_equal(
            routing_map.cast("int64").numpy(), [[1, 0, 1], [0, 1, 0]]
        )
        np.testing.assert_allclose(
            probs.numpy(), [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]
        )

    def test_flag_on_delegates_to_the_dsv4_replay_helper(self):
        with (
            _dsv4_flag(token_dispatcher, True),
            patch.object(
                accuracy_compatible_patch,
                "indices_to_multihot",
                wraps=accuracy_compatible_patch.indices_to_multihot,
            ) as replay,
        ):
            routing_map, probs = (
                token_dispatcher._DeepEPManager._indices_to_multihot(
                    self.owner, self.indices, self.probs
                )
            )

        replay.assert_called_once()
        np.testing.assert_array_equal(
            routing_map.cast("int64").numpy(), [[1, 0, 1], [0, 1, 0]]
        )
        np.testing.assert_allclose(
            probs.numpy(), [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]
        )

    def test_both_paths_agree_numerically(self):
        rm_off, p_off = self._run(False)
        rm_on, p_on = self._run(True)

        np.testing.assert_array_equal(
            rm_off.cast("int64").numpy(), rm_on.cast("int64").numpy()
        )
        np.testing.assert_allclose(p_off.numpy(), p_on.numpy())


class TestDeepEPGlobalInputProbsGating(unittest.TestCase):
    """Only ``flag AND use_accuracy_compatible`` captures ``global_input_probs``.

    token_dispatcher.py:1029 - ``get_permuted_hidden_states_by_experts`` records
    the dispatched router probs (masked-selected in expert-major order) into
    ``global_input_probs`` so the expert path can rescale exactly once. That
    capture must happen only when the DSV4 flag *and* ``use_accuracy_compatible``
    are both on; otherwise the attribute stays ``None``. The multihot build and
    ``permute`` are pinned so the test isolates the flag-gated capture line.
    """

    def setUp(self):
        self.routing_map = paddle.to_tensor([[1, 0], [0, 1]], dtype="bool")
        self.dispatched_probs = paddle.to_tensor(
            [[0.25, 0.0], [0.0, 0.75]], dtype="float32"
        )

    def _owner(self, use_accuracy_compatible):
        owner = types.SimpleNamespace(
            dispatched_indices=None,
            dispatched_probs=None,
            tokens_per_expert=[1, 1],
            use_accuracy_compatible=use_accuracy_compatible,
            global_input_probs=None,
        )
        owner._indices_to_multihot = lambda indices, probs: (
            self.routing_map,
            self.dispatched_probs,
        )
        return owner

    def _run(self, enabled, use_accuracy_compatible):
        owner = self._owner(use_accuracy_compatible)
        with (
            _dsv4_flag(token_dispatcher, enabled),
            patch.object(
                token_dispatcher,
                "permute",
                return_value=(paddle.zeros([2, 4]), None),
            ),
        ):
            token_dispatcher._DeepEPManager.get_permuted_hidden_states_by_experts(
                owner, paddle.zeros([2, 4])
            )
        return owner.global_input_probs

    def test_flag_on_captures_expert_major_probs(self):
        captured = self._run(True, True)
        self.assertIsNotNone(captured)
        np.testing.assert_allclose(captured.numpy(), [0.25, 0.75])

    def test_flag_off_leaves_global_input_probs_unset(self):
        self.assertIsNone(self._run(False, True))

    def test_flag_on_without_accuracy_compatible_stays_unset(self):
        self.assertIsNone(self._run(True, False))


class TestDeepEPRestoreProbsGating(unittest.TestCase):
    """The flag drops the per-token probs from the aligned unpermute call.

    token_dispatcher.py:1060 - ``get_restored_hidden_states_by_experts`` hands
    ``probs=None`` to ``unpermute`` only when the flag and
    ``use_accuracy_compatible`` are both on (the DSV4 path applies router probs
    earlier, so re-applying them here would double-scale). Otherwise it forwards
    ``dispatched_probs`` unchanged.
    """

    def setUp(self):
        self.dispatched_probs = paddle.to_tensor(
            [[0.3, 0.0], [0.0, 0.7]], dtype="float32"
        )
        self.routing_map = paddle.to_tensor([[1, 0], [0, 1]], dtype="bool")

    def _owner(self, use_accuracy_compatible):
        return types.SimpleNamespace(
            dispatched_probs=self.dispatched_probs,
            reversed_mapping_for_combine=paddle.to_tensor(
                [0, 1], dtype="int64"
            ),
            hidden_shape_before_permute=[2, 4],
            dispatched_routing_map=self.routing_map,
            use_accuracy_compatible=use_accuracy_compatible,
        )

    def _probs_arg(self, enabled, use_accuracy_compatible):
        owner = self._owner(use_accuracy_compatible)
        unpermute = MagicMock(
            return_value=paddle.zeros([2, 4], dtype="float32")
        )
        with (
            _dsv4_flag(token_dispatcher, enabled),
            patch.object(token_dispatcher, "unpermute", unpermute),
        ):
            token_dispatcher._DeepEPManager.get_restored_hidden_states_by_experts(
                owner, paddle.zeros([2, 4], dtype="float32")
            )
        return unpermute.call_args.kwargs["probs"]

    def test_flag_on_drops_the_probs(self):
        self.assertIsNone(self._probs_arg(True, True))

    def test_flag_off_forwards_the_dispatched_probs(self):
        self.assertIs(self._probs_arg(False, True), self.dispatched_probs)

    def test_flag_on_without_accuracy_compatible_keeps_probs(self):
        self.assertIs(self._probs_arg(True, False), self.dispatched_probs)


class TestExpertForwardKernelProbsGating(unittest.TestCase):
    """The flag relaxes the ``global_input_probs`` requirement in expert_forward.

    moe_layer.py:1012 - with ``FLAGS_use_accuracy_compatible_kernel`` on, the
    non-grouped ``expert_forward`` demands router probs from the dispatcher and
    raises ``RuntimeError`` when they are missing -- unless the DSV4 flag is on,
    which is allowed to run without them. An empty (zero-token) dispatch keeps
    the expert loop from touching any real expert module, so the branch is
    exercised on a single card.
    """

    def _owner(self):
        return types.SimpleNamespace(
            _use_grouped_mlp_expert=False,
            token_dispatcher=types.SimpleNamespace(global_input_probs=None),
        )

    def test_flag_off_requires_dispatched_probs(self):
        with (
            _dsv4_flag(moe_layer, False),
            patch.object(
                moe_layer, "use_accuracy_compatible_kernel", return_value=True
            ),
            self.assertRaises(RuntimeError),
        ):
            moe_layer.MoELayer.expert_forward(
                self._owner(),
                paddle.zeros([0, 4], dtype="float32"),
                [0],
            )

    def test_flag_on_runs_without_dispatched_probs(self):
        with (
            _dsv4_flag(moe_layer, True),
            patch.object(
                moe_layer, "use_accuracy_compatible_kernel", return_value=True
            ),
        ):
            out = moe_layer.MoELayer.expert_forward(
                self._owner(),
                paddle.zeros([0, 4], dtype="float32"),
                [0],
            )

        self.assertEqual(list(out.shape), [0, 4])


def _allgather_moe_init_config():
    """Minimal config that drives ``MoELayer.__init__`` to the gate check.

    Only the attributes read before line 342 are supplied; every field is a
    plain value so ``deepcopy`` works and no process group is touched.
    """
    return types.SimpleNamespace(
        use_accuracy_compatible=True,
        use_bias=False,
        moe_routed_expert_use_bias=None,
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        hidden_act="swiglu",
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        moe_token_dispatcher_type="allgather",
        moe_allgather_gate_overlap=False,
    )


class TestMoELayerInitAllgatherRejectionGating(unittest.TestCase):
    """``MoELayer.__init__`` intermediate-EP rejection (line 342).

    With ``use_accuracy_compatible=True`` the layer silently rewrites the
    dispatcher to ``alltoall`` -- but only for the plain per-device layouts.
    ``allgather`` / ``ringmoe`` shard experts along the intermediate dim, so
    pairing them with the forced all-to-all path would build experts for a
    layout the dispatcher never produces; the historical path raises instead.
    The DSV4 flag disables that whole rewrite (the replay keeps the configured
    dispatcher), so with the flag *on* the guard must not fire. The raise sits
    before any expert-parallel setup (``pg_collection.ep`` at line 482), so the
    flag-off rejection is reachable on a single card with no process group.
    """

    def test_flag_off_rejects_intermediate_ep_dispatcher(self):
        with (
            _dsv4_flag(moe_layer, False),
            self.assertRaises(ValueError) as ctx,
        ):
            moe_layer.MoELayer(
                config=_allgather_moe_init_config(), pg_collection=None
            )
        self.assertIn("moe_token_dispatcher_type", str(ctx.exception))
        self.assertIn("all-to-all", str(ctx.exception))

    def test_flag_on_skips_the_rejection(self):
        # Flag on disables the rewrite/guard entirely, so __init__ runs past
        # line 342 and only fails later when it dereferences the (absent) EP
        # group -- never with the intermediate-EP ValueError.
        with (
            _dsv4_flag(moe_layer, True),
            self.assertRaises(Exception) as ctx,
        ):
            moe_layer.MoELayer(
                config=_allgather_moe_init_config(), pg_collection=None
            )
        self.assertNotIn("forces the all-to-all token", str(ctx.exception))


class _StopForward(Exception):
    """Sentinel raised from a stubbed ``self.gate`` to halt ``forward`` early."""


class TestMoeForwardSequenceFirstInputTranspose(unittest.TestCase):
    """``MoELayer.forward`` sequence-first input transpose (lines 1878-1882).

    Only ``TransformerConfig.use_dsv4_accuracy`` and ``use_accuracy_compatible`` together (on
    a rank-3 input) flip the MoE into the Torch "sequence-first" layout, moving
    the sequence axis to the front of ``hidden_states``/``input_ids``/``residual``
    before routing. The transpose sits at the very top of ``forward`` -- ahead of
    dispatch/experts -- so it is isolated here by stubbing ``self.gate`` to halt
    the pass, then reading back what routing was handed. Flag off must leave the
    ``[batch, seq, hidden]`` layout untouched.
    """

    def _run(self, enabled):
        captured = {}

        def _prep_gate_input(hs, residual):
            captured["residual"] = residual
            return hs

        fake_self = types.SimpleNamespace(
            expert_model_parallel_size=1,
            sequence_parallel=False,
            use_accuracy_compatible=True,
            layer_number=0,
            _maybe_pre_allgather_overlap=lambda hs: None,
            _prepare_gate_input=_prep_gate_input,
            _supports_three_path_clone=lambda: False,
            gate=MagicMock(side_effect=_StopForward()),
        )
        hidden_states = paddle.zeros([2, 3, 4], dtype="float32")
        hidden_states.stop_gradient = True
        input_ids = paddle.zeros([2, 3], dtype="int64")
        residual = paddle.zeros([2, 3, 4], dtype="float32")
        with (
            _dsv4_flag(moe_layer, enabled),
            patch.object(moe_layer, "targets_hf", return_value=False),
            patch.object(
                moe_layer, "inspect_tensor", side_effect=lambda *a, **k: a[2]
            ),
            patch.object(moe_layer, "inspect_tensor_set_current_layer"),
        ):
            try:
                moe_layer.MoELayer.forward(
                    fake_self,
                    hidden_states,
                    input_ids=input_ids,
                    residual=residual,
                )
            except _StopForward:
                pass
        gate_input = fake_self.gate.call_args.args[0]
        routed_input_ids = fake_self.gate.call_args.kwargs["input_ids"]
        return gate_input, routed_input_ids, captured["residual"]

    def test_flag_on_moves_the_sequence_axis_to_front(self):
        gate_input, routed_input_ids, residual = self._run(True)
        self.assertEqual(gate_input.shape, [3, 2, 4])
        self.assertEqual(routed_input_ids.shape, [3, 2])
        self.assertEqual(residual.shape, [3, 2, 4])

    def test_flag_off_keeps_the_batch_first_layout(self):
        gate_input, routed_input_ids, residual = self._run(False)
        self.assertEqual(gate_input.shape, [2, 3, 4])
        self.assertEqual(routed_input_ids.shape, [2, 3])
        self.assertEqual(residual.shape, [2, 3, 4])


class TestMoeForwardSequenceFirstOutputTranspose(unittest.TestCase):
    """``MoELayer.forward`` sequence-first output transpose (line 2102).

    The sequence-first replay transposes the input to ``[seq, batch, hidden]``
    (so ``orig_shape`` and the flattened expert-token order follow that layout)
    and must transpose the expert output *back* to ``[batch, seq, hidden]`` at
    the end. Both ends are driven here on a single card by stubbing the router
    (``self.gate``) and the single-card expert compute; the stub tags each
    flattened token with its row index, so the sequence-first vs batch-first
    flatten order produces a different final tensor -- pinning that both the
    input transpose (via ``orig_shape``) and the output transpose (line 2102)
    executed only when the flag is on.
    """

    def _run(self, enabled):
        b, s, h = 2, 3, 4

        def _expert(reshaped_input, topk_indices, topk_weights):
            row = paddle.arange(reshaped_input.shape[0], dtype="float32")
            return reshaped_input + row.reshape([-1, 1])

        fake_self = types.SimpleNamespace(
            expert_model_parallel_size=1,
            sequence_parallel=False,
            use_accuracy_compatible=True,
            layer_number=0,
            training=False,
            router_aux_loss_coef=0.0,
            use_latent_moe=False,
            moe_expert_fusion=False,
            shared_experts=None,
            _maybe_pre_allgather_overlap=lambda hs: None,
            _prepare_gate_input=lambda hs, residual: hs,
            _prepare_expert_input=lambda hs, residual: hs,
            _post_routed_output=lambda o: o,
            _supports_three_path_clone=lambda: False,
            _forward_single_card_moe=_expert,
            gate=MagicMock(
                return_value=(
                    None,
                    paddle.ones([b * s, 1]),
                    paddle.zeros([b * s, 1], dtype="int64"),
                    paddle.ones([b * s, 4]),
                    paddle.ones([b * s, 4]),
                    None,
                    None,
                    None,
                )
            ),
        )
        hidden_states = paddle.zeros([b, s, h], dtype="float32")
        hidden_states.stop_gradient = True
        with (
            _dsv4_flag(moe_layer, enabled),
            patch.object(moe_layer, "targets_hf", return_value=False),
            patch.object(
                moe_layer, "inspect_tensor", side_effect=lambda *a, **k: a[2]
            ),
            patch.object(moe_layer, "inspect_tensor_set_current_layer"),
            patch.object(moe_layer, "log_moe_losses"),
        ):
            output, bias = moe_layer.MoELayer.forward(fake_self, hidden_states)
        return output

    def test_flag_on_transposes_output_back_to_batch_first(self):
        b, s, h = 2, 3, 4
        out = self._run(True)
        # Input was transposed to [s, b, h]; expert tags row r (== seq*b + batch)
        # and the output is transposed back, so element [batch, seq] carries
        # (seq * b + batch).
        self.assertEqual(out.shape, [b, s, h])
        expected = (
            paddle.arange(s * b, dtype="float32")
            .reshape([s, b, 1])
            .transpose([1, 0, 2])
            .broadcast_to([b, s, h])
        )
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_flag_off_keeps_batch_first_flatten_order(self):
        b, s, h = 2, 3, 4
        out = self._run(False)
        self.assertEqual(out.shape, [b, s, h])
        expected = (
            paddle.arange(b * s, dtype="float32")
            .reshape([b, s, 1])
            .broadcast_to([b, s, h])
        )
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_the_two_orders_are_actually_distinguishable(self):
        self.assertFalse(
            np.array_equal(self._run(True).numpy(), self._run(False).numpy())
        )


class TestTopKRouterMtpPaddingMaskGating(unittest.TestCase):
    """``TopKRouter.forward`` MTP padding-mask drop (line 1648).

    The router normally zeroes out padded tokens (``input_ids == pad``) from the
    routing map. On an MTP layer the DSV4 replay instead keeps them, because the
    MTP shift already realigned the labels and re-masking here would drop the
    wrong tokens; line 1648 sets ``input_ids_none_zero_mask = None`` only when
    the flag *and* ``is_mtp_layer`` are both set. The effect is observed through
    the hash-routing early-return path (the shortest single-card route that
    consumes the mask): the router matmul and hash lookup are stubbed so no gate
    weights or process group are needed, and the padded token's routing is the
    observable.
    """

    def _run(self, enabled, is_mtp_layer):
        fake_self = types.SimpleNamespace(
            sequence_parallel=False,
            config=types.SimpleNamespace(moe_router_force_load_balancing=False),
            is_mtp_layer=is_mtp_layer,
            is_hash_layer=True,
            moe_split_feature_routing=False,
            use_accuracy_compatible=False,
            weight=paddle.ones([4, 3], dtype="float32"),
            _layer_number=0,
            _hash_routing=lambda logits, flat_ids: (
                paddle.ones([2, 1], dtype="float32"),
                paddle.to_tensor([[0], [1]], dtype="int64"),
            ),
        )
        # [batch=1, seq=2, hidden=4]; second token is padding (id == 0).
        hidden = paddle.zeros([1, 2, 4], dtype="float32")
        input_ids = paddle.to_tensor([[5, 0]], dtype="int64")
        with (
            _dsv4_flag(moe_router, enabled),
            patch.object(
                moe_router, "get_context_parallel_world_size", return_value=1
            ),
            patch.object(
                moe_router,
                "gate_detach_matmul",
                return_value=paddle.ones([2, 3], dtype="float32"),
            ),
        ):
            _, top_gate, top_idx, probs, mask, *_ = (
                moe_router.TopKRouter.forward(
                    fake_self, hidden, input_ids=input_ids
                )
            )
        return top_idx, probs, mask

    def test_flag_on_mtp_layer_keeps_the_padded_token_routed(self):
        top_idx, probs, mask = self._run(True, is_mtp_layer=True)
        # Padded token (row 1) keeps its expert-1 assignment.
        np.testing.assert_array_equal(top_idx.numpy(), [[0], [1]])
        np.testing.assert_array_equal(mask.numpy(), [[1, 0, 0], [0, 1, 0]])

    def test_flag_off_masks_the_padded_token(self):
        top_idx, probs, mask = self._run(False, is_mtp_layer=True)
        # Padded token is dropped: index -> -1 and its row is zeroed.
        np.testing.assert_array_equal(top_idx.numpy(), [[0], [-1]])
        np.testing.assert_array_equal(mask.numpy(), [[1, 0, 0], [0, 0, 0]])

    def test_flag_on_non_mtp_layer_still_masks(self):
        # The drop requires *both* the flag and is_mtp_layer; a non-MTP layer
        # with the flag on must keep masking the padded token.
        top_idx, probs, mask = self._run(True, is_mtp_layer=False)
        np.testing.assert_array_equal(top_idx.numpy(), [[0], [-1]])
        np.testing.assert_array_equal(mask.numpy(), [[1, 0, 0], [0, 0, 0]])


class TestDeepEPDispatchOverlapResetsGlobalProbs(unittest.TestCase):
    """``_DeepEPManager.dispatch_overlap`` clears ``global_input_probs`` (line 899).

    ``dispatch_overlap`` records the fused-dispatch state and then resets
    ``global_input_probs`` to ``None`` so the expert path recaptures it fresh on
    the next ``get_permuted_hidden_states_by_experts``. The fused all-to-all is a
    collective, so it is stubbed here; the observable is that a stale
    ``global_input_probs`` is cleared while the dispatched state is taken from the
    returned handle. No process group is created.
    """

    def test_dispatch_overlap_resets_global_input_probs(self):
        states = {
            "handle": "HANDLE",
            "tokens_per_expert": "TPE",
            "dispatched_indices": "DINDICES",
        }
        owner = types.SimpleNamespace(
            num_experts=4,
            group=None,
            handle=None,
            tokens_per_expert=None,
            dispatched_indices=None,
            dispatched_probs=None,
            global_input_probs="STALE",
        )
        with patch.object(
            token_dispatcher,
            "fused_dispatch",
            return_value=("HS_OUT", "PROBS", states, "SCALE"),
        ) as fused:
            hs, scale = token_dispatcher._DeepEPManager.dispatch_overlap(
                owner,
                paddle.zeros([2, 4], dtype="float32"),
                paddle.zeros([2, 2], dtype="int64"),
                paddle.ones([2, 2], dtype="float32"),
            )

        fused.assert_called_once()
        self.assertIsNone(owner.global_input_probs)
        self.assertEqual(owner.handle, "HANDLE")
        self.assertEqual(owner.tokens_per_expert, "TPE")
        self.assertEqual(owner.dispatched_indices, "DINDICES")
        self.assertEqual(owner.dispatched_probs, "PROBS")
        self.assertEqual((hs, scale), ("HS_OUT", "SCALE"))


class TestFlexDispatchPostprocessCapturesGlobalProbs(unittest.TestCase):
    """``MoEFlexTokenDispatcher.dispatch_postprocess`` mirrors global probs (line 1253).

    After the comm manager permutes tokens into expert-major order,
    ``dispatch_postprocess`` copies the manager's ``global_input_probs`` up onto
    the dispatcher (defaulting to ``None`` when the manager never captured any)
    and returns the permuted tokens plus per-expert counts. The comm manager is
    faked here so the whole thing runs single-card with no collectives.
    """

    def _run(self, comm_manager):
        owner = types.SimpleNamespace(
            _comm_manager=comm_manager, global_input_probs="STALE"
        )
        tokens, tpe = (
            token_dispatcher.MoEFlexTokenDispatcher.dispatch_postprocess(
                owner, paddle.zeros([2, 4], dtype="float32")
            )
        )
        return owner, tokens, tpe

    def test_captures_the_managers_global_input_probs(self):
        comm = types.SimpleNamespace(
            get_permuted_hidden_states_by_experts=lambda hs: "GLOBAL_TOKENS",
            global_input_probs="GIP",
            get_number_of_tokens_per_expert=lambda: "TPE",
        )
        owner, tokens, tpe = self._run(comm)
        self.assertEqual(tokens, "GLOBAL_TOKENS")
        self.assertEqual(tpe, "TPE")
        self.assertEqual(owner.global_input_probs, "GIP")

    def test_defaults_to_none_when_manager_has_no_probs(self):
        comm = types.SimpleNamespace(
            get_permuted_hidden_states_by_experts=lambda hs: "GLOBAL_TOKENS",
            get_number_of_tokens_per_expert=lambda: "TPE",
        )
        owner, _, _ = self._run(comm)
        self.assertIsNone(owner.global_input_probs)


if __name__ == "__main__":
    unittest.main()
