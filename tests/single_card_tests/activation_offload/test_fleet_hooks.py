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
"""Tests for the micro-step anchors that drive grouping and prefetch.

Two things are pinned down here: which manager call each of the four anchors
makes, and the chunk id they pass along. Both are contracts that a pipeline run
cannot check on its own -- a wrong mapping or a wrong chunk id stays numerically
correct, because lazy reload covers for it, and only shows up as lost overlap.

Registration goes into a process-wide registry inside paddle that cannot be
undone, so it is intercepted here and the callbacks are invoked directly. That
the anchors really fire under an interleaved schedule is what the multi-card
pipeline test covers.
"""

from __future__ import annotations

import unittest
from unittest import mock

import paddle

from paddlefleet.activation_offload import fleet_hooks
from paddlefleet.activation_offload.manager import (
    OffloadManager,
    reset_offload_manager,
)

_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)

_LOC = fleet_hooks._Loc
ANCHORS = (
    _LOC.FORWARD_BEGIN,
    _LOC.FORWARD_END,
    _LOC.BACKWARD_BEGIN,
    _LOC.BACKWARD_END,
)


class _FakeInterleaved:
    """Stands in for PipelineParallelWithInterleave, which sets this per step."""

    def __init__(self, chunk):
        self._virtual_pp_rank = chunk


@unittest.skipUnless(_HAS_GPU, "Requires a CUDA device")
class TestFleetHooks(unittest.TestCase):
    def setUp(self):
        self._reset()

    def tearDown(self):
        self._reset()

    @staticmethod
    def _reset():
        reset_offload_manager()
        fleet_hooks._wired = False
        fleet_hooks._pp_model = None

    def _wire(self, pp_model):
        """Register the anchors, collecting them instead of installing them."""
        registered: dict = {}
        mgr = OffloadManager(numa_bind=False)
        with (
            mock.patch.object(
                fleet_hooks,
                "_register_hook",
                lambda loc, hook: registered.__setitem__(loc, hook),
            ),
            mock.patch.object(
                fleet_hooks, "get_offload_manager", return_value=mgr
            ),
        ):
            fleet_hooks.enable_fleet_prefetch(pp_model)
        return registered, mgr

    def test_all_four_anchors_are_registered(self):
        registered, _ = self._wire(_FakeInterleaved(0))
        self.assertEqual(set(registered), set(ANCHORS))

    def test_each_anchor_calls_its_own_manager_method(self):
        """The mapping from anchor to manager call, chunk id included.

        Swapping two of these, or passing the wrong chunk, leaves the numbers
        intact and only costs the overlap the anchors exist to buy.
        """
        chunk = 3
        registered, mgr = self._wire(_FakeInterleaved(chunk))
        expected = {
            _LOC.FORWARD_BEGIN: ("begin_forward_group", (chunk,)),
            _LOC.FORWARD_END: ("clear_current_group", ()),
            _LOC.BACKWARD_BEGIN: ("prefetch_next_group", (chunk,)),
            _LOC.BACKWARD_END: ("prefetch_next_group_head", (chunk,)),
        }
        for loc, (method, args) in expected.items():
            with (
                self.subTest(anchor=loc),
                mock.patch.object(mgr, method) as spy,
            ):
                # step_id is passed by paddle and deliberately ignored: forward
                # and backward number their steps independently of each other.
                registered[loc](step_id=7)
                spy.assert_called_once_with(*args)

    def test_the_chunk_id_comes_from_the_interleaved_schedule(self):
        """Groups must be keyed per chunk, or two chunks would share one group."""
        registered, mgr = self._wire(_FakeInterleaved(2))
        registered[_LOC.FORWARD_BEGIN]()
        self.assertEqual(mgr._cur_group_key, (2, 0))
        registered[_LOC.FORWARD_END]()
        self.assertIsNone(mgr._cur_group_key)

    def test_without_a_model_everything_collapses_into_one_chain(self):
        """No model to read means VPP=1, which is a single chunk keyed by order."""
        registered, mgr = self._wire(None)
        registered[_LOC.FORWARD_BEGIN]()
        registered[_LOC.FORWARD_END]()
        registered[_LOC.FORWARD_BEGIN]()
        self.assertEqual(mgr._cur_group_key, (0, 1))

    def test_a_schedule_without_interleaving_is_also_one_chain(self):
        """Non-interleaved schedules never set the attribute at all."""
        registered, mgr = self._wire(object())
        registered[_LOC.FORWARD_BEGIN]()
        self.assertEqual(mgr._cur_group_key, (0, 0))

    def test_registering_twice_rewires_nothing_but_updates_the_model(self):
        """Registration is once per process; the model reference is not.

        The callbacks live in paddle's registry for good, so a second call must
        not stack a second set of them -- but it still has to point at the model
        that is now being trained, or the chunk id would come from a stale one.
        """
        first_model = _FakeInterleaved(1)
        registered, mgr = self._wire(first_model)
        self.assertIs(fleet_hooks._pp_model, first_model)

        second_model = _FakeInterleaved(5)
        again, _ = self._wire(second_model)
        self.assertEqual(again, {}, "the anchors were registered a second time")
        self.assertIs(fleet_hooks._pp_model, second_model)

        # The surviving callbacks are the ones from the first call, so they still
        # hold the first manager -- but the chunk id has to come from the model
        # registered second.
        with mock.patch.object(mgr, "prefetch_next_group") as spy:
            registered[_LOC.BACKWARD_BEGIN]()
        spy.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
