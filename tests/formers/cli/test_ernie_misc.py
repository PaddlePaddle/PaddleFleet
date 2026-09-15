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

import unittest
from types import SimpleNamespace

# misc.py imports paddle and builds ``ZERO = paddle.zeros(...)`` at module
# import time, so the whole module is unimportable without paddle. Guard the
# import honestly; the local CI box has no paddle and will skip these tests.
try:
    import paddle  # noqa: F401

    from paddlefleet.cli.train.ernie_pretrain.src.utils.misc import (
        SmoothedValue,
        TrainingLogs,
        global_training_logs,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = (
    "paddle is required to import ernie_pretrain misc module; not installed"
)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSmoothedValue(unittest.TestCase):
    """Behavior of the running-average meter SmoothedValue."""

    def test_scalar_accumulation_and_average(self):
        # total and count accumulate; global_avg = total / count.
        sv = SmoothedValue(skip_zero=False)
        sv.update(4.0)
        sv.update(6.0)
        self.assertEqual(sv.total, 10.0)
        self.assertEqual(sv.count, 2)
        self.assertAlmostEqual(sv.global_avg, 5.0, places=6)

    def test_global_avg_without_data_is_zero(self):
        # count == 0 -> denominator clamps to 1e-6, numerator is 0.0.
        sv = SmoothedValue(skip_zero=False)
        self.assertEqual(sv.count, 0)
        self.assertAlmostEqual(sv.global_avg, 0.0, places=6)

    def test_reset_clears_total_and_count(self):
        sv = SmoothedValue(skip_zero=False)
        sv.update(7.0)
        sv.update(3.0)
        sv.reset()
        self.assertEqual(sv.total, 0.0)
        self.assertEqual(sv.count, 0)

    def test_skip_zero_does_not_skip_scalar_zero(self):
        # skip_zero only affects paddle tensors; python scalars always count.
        sv = SmoothedValue(skip_zero=True)
        sv.update(0.0)
        self.assertEqual(sv.count, 1)
        self.assertEqual(sv.total, 0.0)

    def test_skip_zero_skips_zero_tensor(self):
        # A zero tensor must not advance count; a nonzero one must.
        sv = SmoothedValue(skip_zero=True)
        sv.update(paddle.to_tensor([5.0]))  # nonzero -> counted
        sv.update(paddle.to_tensor([0.0]))  # zero -> skipped
        self.assertEqual(int(sv.count), 1)
        self.assertAlmostEqual(float(sv.total), 5.0, places=6)
        self.assertAlmostEqual(float(sv.global_avg), 5.0, places=6)

    def test_tensor_without_skip_zero_counts_every_update(self):
        # Without skip_zero, even a zero tensor advances count.
        sv = SmoothedValue(skip_zero=False)
        sv.update(paddle.to_tensor([3.0]))
        sv.update(paddle.to_tensor([0.0]))
        self.assertEqual(int(sv.count), 2)
        self.assertAlmostEqual(float(sv.total), 3.0, places=6)
        self.assertAlmostEqual(float(sv.global_avg), 1.5, places=6)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestTrainingLogs(unittest.TestCase):
    """Behavior of the TrainingLogs singleton meter registry."""

    def setUp(self):
        self.tl = TrainingLogs()
        self._reset()
        self.addCleanup(self._reset)

    def _reset(self):
        # TrainingLogs is a process-wide singleton; scrub shared state so
        # tests neither leak into nor depend on one another.
        self.tl.reset()
        self.tl.snapshot = None
        self.tl._skip_zero_keys = []
        self.tl._global_meters_keys = []
        self.tl.trainer = None
        self.tl.logging_interval = None

    def test_singleton_identity(self):
        # Every construction returns the same object, including the module
        # level global_training_logs.
        self.assertIs(TrainingLogs(), self.tl)
        self.assertIs(global_training_logs, self.tl)

    def test_setitem_accumulates_through_smoothed_value(self):
        self.tl["loss"] = 0.5
        self.tl["loss"] = 1.5
        # Same key feeds one SmoothedValue: avg of 0.5 and 1.5 is 1.0.
        self.assertAlmostEqual(self.tl["loss"].global_avg, 1.0, places=6)

    def test_update_accumulates_per_key(self):
        self.tl.update(a=2.0, b=4.0)
        self.tl.update(a=4.0)
        self.assertAlmostEqual(self.tl["a"].global_avg, 3.0, places=6)
        self.assertAlmostEqual(self.tl["b"].global_avg, 4.0, places=6)

    def test_skip_zero_keys_propagate_to_new_meter(self):
        # A new meter whose key matches a skip pattern is created with
        # skip_zero=True and therefore ignores a zero tensor.
        self.tl.enable_skip_zero(["grad_norm.*"])
        self.tl["grad_norm_x"] = paddle.to_tensor([0.0])
        self.assertTrue(self.tl.meters["grad_norm_x"]._skip_zero)
        self.assertEqual(int(self.tl.meters["grad_norm_x"].count), 0)

    def test_dict_returns_averaged_values(self):
        self.tl["a"] = 2.0
        self.tl["a"] = 4.0
        ret, global_info = self.tl.dict(use_async=False)
        self.assertEqual(ret, {"a": 3.0})
        self.assertEqual(global_info, {})

    def test_dict_filters_skip_zero_zero_values(self):
        # A skip_zero meter that averaged to 0.0 is dropped from dict();
        # a normal meter is retained.
        self.tl.enable_skip_zero(["z"])
        self.tl["z"] = 0.0
        self.tl["keep"] = 3.0
        ret, _ = self.tl.dict(use_async=False)
        self.assertEqual(ret, {"keep": 3.0})

    def test_take_and_restore_snapshot(self):
        self.tl["m1"] = 1.0
        self.tl.take_snapshot()
        self.tl["m2"] = 2.0
        self.tl.restore_snapshot()
        self.assertIn("m1", self.tl.meters)
        self.assertNotIn("m2", self.tl.meters)
        self.assertAlmostEqual(self.tl.meters["m1"].global_avg, 1.0, places=6)
        self.assertIsNone(self.tl.snapshot)

    def test_restore_without_take_raises(self):
        with self.assertRaises(AssertionError):
            self.tl.restore_snapshot()

    def test_getattr_returns_meter_or_raises(self):
        self.tl["x"] = 5.0
        self.assertIs(self.tl.x, self.tl.meters["x"])
        with self.assertRaises(AttributeError):
            _ = self.tl.definitely_missing_attr

    def test_is_enabled_without_trainer(self):
        self.tl.trainer = None
        self.assertTrue(self.tl.is_enabled())

    def test_is_enabled_respects_interval(self):
        trainer = SimpleNamespace(state=SimpleNamespace(global_step=4))
        self.tl.set_trainer_interval(trainer, 5)
        self.assertTrue(self.tl.is_enabled())  # (4 + 1) % 5 == 0
        trainer.state.global_step = 3
        self.assertFalse(self.tl.is_enabled())  # (3 + 1) % 5 != 0

    @unittest.expectedFailure
    def test_enable_skip_zero_marks_existing_meter(self):
        # REAL BUG: enable_skip_zero iterates meter *keys* (strings) and runs
        # ``m._skip_zero = True``, setting the attribute on the key string
        # instead of self.meters[m]. With a matching existing meter this
        # raises AttributeError and the meter's flag is never updated.
        # The assertions below state the CORRECT behavior; production code is
        # left unchanged and this is flagged as an expected failure.
        self.tl["loss"] = 1.0
        self.assertFalse(self.tl.meters["loss"]._skip_zero)
        self.tl.enable_skip_zero(["loss"])
        self.assertTrue(self.tl.meters["loss"]._skip_zero)


if __name__ == "__main__":
    unittest.main()
