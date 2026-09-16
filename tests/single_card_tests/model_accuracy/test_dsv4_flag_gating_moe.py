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

"""``FLAGS_use_dsv4_accuracy`` must gate the MoE dispatch/expert call sites.

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
from paddlefleet.transformer.moe import moe_layer, token_dispatcher


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


if __name__ == "__main__":
    unittest.main()
