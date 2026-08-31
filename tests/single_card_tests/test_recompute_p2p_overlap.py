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

"""What ``install_recompute_p2p_overlap`` refuses, and what never enters the store.

Whether running a selective-recompute span early changes the gradients is settled
end to end by
``tests/multi_card_tests/pipeline_parallel/test_pp_dw_recompute_overlap.py``, on a
real pp=4/vpp=2 schedule against a reference with the filler off. What that test
cannot reach is what is left here:

* the install-time refusals -- wrong ``recompute_granularity``, or no pipeline
  parallel at all -- because a working schedule never has a bad config;
* a span registered outside a named chunk, because every forward there runs
  inside one. It must stay out of the store and still recompute from its own
  hook, since the scheduler can only ask for spans it can name.
"""

import unittest

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    RecomputeStore,
)

from paddlefleet.recompute_utils import install_recompute_p2p_overlap
from paddlefleet.tensor_parallel import RecomputeWithoutOutput


class _RcConfig:
    recompute_granularity = "selective"
    pipeline_model_parallel_size = 8
    virtual_pipeline_model_parallel_size = 2

    def __init__(self, enabled):
        self.p2p_overlap_recompute = enabled


def _build(seed=0):
    paddle.seed(seed)
    fc1 = nn.Linear(32, 64)
    fc2 = nn.Linear(64, 32)
    x = paddle.randn([8, 32])
    x.stop_gradient = False
    return fc1, fc2, x


def _run_span(fc1, fc2, x, key=(0, 0)):
    """One forward inside a named chunk, as the scheduler would drive it."""
    RecomputeStore.begin_chunk(key)
    span = RecomputeWithoutOutput()
    hidden = span.recompute(fc1, x, preserve_rng_state=False)
    out = fc2(hidden)
    span.discard_output_and_register_recompute(out)
    RecomputeStore.end_chunk()


class TestRecomputeStore(unittest.TestCase):
    def setUp(self):
        RecomputeStore.clear()
        RecomputeStore.enabled = False

    tearDown = setUp

    def test_off_registers_nothing(self):
        fc1, fc2, x = _build()
        _run_span(fc1, fc2, x)
        self.assertEqual(RecomputeStore.pending((0, 0)), 0)
        print("[rc store] disabled registers nothing OK")

    def test_install_requires_selective(self):
        cfg = _RcConfig(True)
        cfg.recompute_granularity = "full"
        with self.assertRaises(ValueError):
            install_recompute_p2p_overlap(cfg)
        self.assertFalse(RecomputeStore.enabled)

        install_recompute_p2p_overlap(_RcConfig(True))
        self.assertTrue(RecomputeStore.enabled)
        install_recompute_p2p_overlap(_RcConfig(False))
        self.assertFalse(RecomputeStore.enabled)

        # no pipeline parallel means no p2p window to fill
        cfg = _RcConfig(True)
        cfg.pipeline_model_parallel_size = 1
        with self.assertRaises(ValueError):
            install_recompute_p2p_overlap(cfg)
        cfg = _RcConfig(True)
        cfg.virtual_pipeline_model_parallel_size = None
        with self.assertRaises(ValueError):
            install_recompute_p2p_overlap(cfg)
        cfg = _RcConfig(True)
        cfg.virtual_pipeline_model_parallel_size = 1
        with self.assertRaises(ValueError):
            install_recompute_p2p_overlap(cfg)
        print("[rc store] install validates granularity and pp OK")

    def test_put_outside_a_named_chunk_is_ignored(self):
        """Nothing may enter the store that the scheduler cannot name."""
        install_recompute_p2p_overlap(_RcConfig(True))
        fc1, fc2, x = _build()
        span = RecomputeWithoutOutput()
        hidden = span.recompute(fc1, x, preserve_rng_state=False)
        out = fc2(hidden)
        span.discard_output_and_register_recompute(out)  # no begin_chunk
        self.assertEqual(RecomputeStore.groups, {})
        paddle.sum(out).backward()  # its own hook must still run it
        self.assertIsNotNone(x.grad)
        print("[rc store] put outside a named chunk ignored OK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
