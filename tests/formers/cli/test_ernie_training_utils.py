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

# training_utils.py itself is pure-python arithmetic, but importing it pulls in
# ``paddlefleet/__init__.py`` -> ``parallel_state`` -> ``import paddle`` at
# package import time, so the module is unimportable without paddle. Guard the
# import honestly; the local CI box has no paddle and will skip these tests.
try:
    import paddle  # noqa: F401

    from paddlefleet.cli.train.ernie_pretrain.src.utils.training_utils import (
        reset_per_device_batch_size,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = (
    "importing paddlefleet (parallel_state) requires paddle; not installed"
)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestResetPerDeviceBatchSize(unittest.TestCase):
    """Behavior of reset_per_device_batch_size.

    The function factors a global batch size into
    (per_device_train_batch_size, gradient_accumulation_steps) given a data
    parallel world size. Expected tuples below are derived by hand from the
    definition, not from the function under test.
    """

    def test_shrinks_per_device_when_batch_per_device_too_small(self):
        # batch_per_device = 16 // 8 = 2, which is < requested 4.
        # Contract: shrink per_device down to 2 and force accum to 1. The key
        # signal is that the returned per_device is 2 (the reset value), NOT
        # the requested 4 that was passed in.
        per_device, accum = reset_per_device_batch_size(16, 4, 8)
        self.assertEqual(per_device, 2)
        self.assertEqual(accum, 1)

    def test_computes_accumulation_when_batch_per_device_is_multiple(self):
        # batch_per_device = 64 // 8 = 8; 8 >= 4 and 8 % 4 == 0.
        # accum = 8 // 4 = 2, per_device stays 4.
        per_device, accum = reset_per_device_batch_size(64, 4, 8)
        self.assertEqual(per_device, 4)
        self.assertEqual(accum, 2)

    def test_large_accumulation_distinct_from_first_case(self):
        # batch_per_device = 256 // 4 = 64; 64 % 4 == 0 -> accum = 16.
        # Different (per_device, accum) magnitudes than the case above guard
        # against a swapped return order or a constant accum.
        per_device, accum = reset_per_device_batch_size(256, 4, 4)
        self.assertEqual(per_device, 4)
        self.assertEqual(accum, 16)

    def test_boundary_batch_per_device_equals_requested(self):
        # batch_per_device = 32 // 8 = 4 == requested 4. This lands in the
        # ">=" branch: 4 % 4 == 0 -> accum = 1, per_device unchanged at 4.
        per_device, accum = reset_per_device_batch_size(32, 4, 8)
        self.assertEqual(per_device, 4)
        self.assertEqual(accum, 1)

    def test_single_device_uses_full_global_batch(self):
        # world_size = 1 -> batch_per_device = 32; 32 % 8 == 0 -> accum = 4.
        per_device, accum = reset_per_device_batch_size(32, 8, 1)
        self.assertEqual(per_device, 8)
        self.assertEqual(accum, 4)

    def test_factorization_invariant_holds_for_valid_inputs(self):
        # The whole point of the function is to factor the global batch:
        # returned_per_device * accum * world_size must reconstruct the
        # global batch size for every successful call. Derived independently
        # of the return values by re-multiplying against the known global.
        cases = [
            (16, 4, 8),
            (64, 4, 8),
            (256, 4, 4),
            (32, 4, 8),
            (32, 8, 1),
            (48, 3, 4),
        ]
        for global_bsz, requested, world in cases:
            with self.subTest(global_bsz=global_bsz, world=world):
                per_device, accum = reset_per_device_batch_size(
                    global_bsz, requested, world
                )
                self.assertEqual(per_device * accum * world, global_bsz)
                # per_device never exceeds the requested cap.
                self.assertLessEqual(per_device, requested)
                self.assertGreaterEqual(accum, 1)

    def test_global_not_divisible_by_world_raises_first_assert(self):
        # 17 % 8 != 0 -> first assertion fires. Its message names world_size
        # but never mentions batch_per_device (that belongs to the 2nd assert).
        with self.assertRaisesRegex(AssertionError, r"world_size=8") as ctx:
            reset_per_device_batch_size(17, 4, 8)
        self.assertNotIn("batch_per_device", str(ctx.exception))

    def test_batch_per_device_not_divisible_raises_second_assert(self):
        # batch_per_device = 40 // 8 = 5; 5 >= 4 but 5 % 4 != 0 -> the second
        # assertion (inside the else branch) fires and reports the computed
        # batch_per_device value.
        with self.assertRaisesRegex(AssertionError, r"batch_per_device=5"):
            reset_per_device_batch_size(40, 4, 8)


if __name__ == "__main__":
    unittest.main()
