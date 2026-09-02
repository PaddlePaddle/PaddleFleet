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

"""Multi-process VPP regression for the two p2p-window fillers.

The pp scheduler fills a p2p window with either a flushed dW batch or the
recompute spans of the chunk whose backward comes next. Both only move *when* a
computation runs, never what it computes, so loss and every gradient must match
the same model with both fillers off -- to the bit.

What needs more than one process to reach:

1. **The sync / async p2p branch is chosen per rank, per micro step.** The
   scheduler opens an asynchronous window only when it has something to run
   inside it -- a flushed dW batch (``have_dw``) or a chunk with pending
   recompute (``rc_key``) -- and takes the synchronous path otherwise. That
   decision is rank-local, so getting it wrong hangs the job instead of failing
   an assert. Chunk ``(vpp0, pp2)`` below is parameter free, so on that one rank
   the window has neither while the other three have both.

2. **The received activation is appended to a per-virtual-chunk FIFO.** A wrong
   chunk key does not raise; it silently routes gradients into the wrong virtual
   chunk. Comparing every gradient after backward catches exactly that.

3. **``WeightGradStore.flush()`` enqueues unconditionally**, so a blind call puts
   an empty batch on the queue and makes ``have_dw`` true with nothing to run.
   The parameter-free chunk is what exercises the ``if WeightGradStore.cache:``
   guard.

Unlike the single-process store tests, the recompute span here is the real
``RecomputeWithoutOutput``, so the ``run_recompute_now()`` contract the scheduler
depends on is the one that ships. Note the reference run recomputes too -- with
the filler off the span still runs, just from its own backward hook instead of
inside the window, which is exactly the difference being pinned down.

Layout: ``pp=4``, ``vpp=2`` -> 8 virtual chunks, one segmentable layer each.

    chunk (vpp, pp) | layer          | queues dW | registers recompute
    ----------------|----------------|-----------|--------------------
    (0, 0)          | LinearPipe     | yes       | no
    (0, 1)          | DeferredLinear | yes       | no
    (0, 2)          | NoParamPipe    | **no**    | **no**
    (0, 3)          | LinearPipe     | yes       | no
    (1, 0)          | RecomputePipe  | yes       | **yes**
    (1, 1)          | LinearPipe     | yes       | no
    (1, 2)          | RecomputePipe  | yes       | **yes**
    (1, 3)          | LinearPipe     | yes       | no

So ``rc_key`` is non-None on two of the four ranks only, and ``have_dw`` is false
on one -- both asymmetries hold at once.

Both schedulers that open such a window are covered in this one process:
``accumulate_steps`` is what picks between them in ``fleet/model.py``, and it is
read from the strategy when ``distributed_model`` runs, not when ``fleet.init``
does, so one init is enough.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    RecomputeStore,
    SplitBWLinear,
    WeightGradStore,
)
from paddle.io import DataLoader, Dataset
from paddle.nn import Layer, Linear

from paddlefleet.recompute_utils import install_recompute_p2p_overlap
from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.transformer.dw_overlap import DeferredWeightGradLinear

HIDDEN = 8
MICRO_BATCH_SIZE = 2
PP_DEGREE = 4
VPP = 2
STEPS = 4


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        image = rng.random([HIDDEN]).astype("float32")
        label = rng.randint(0, HIDDEN, (1,)).astype("int64")
        return image, label

    def __len__(self):
        return self.num_samples


class _RcConfig:
    """Minimal config for ``install_recompute_p2p_overlap``.

    Going through that entry point rather than setting ``RecomputeStore.enabled``
    by hand keeps the test on the switch the model library actually uses.
    """

    recompute_granularity = "selective"
    pipeline_model_parallel_size = PP_DEGREE
    virtual_pipeline_model_parallel_size = VPP

    def __init__(self, enabled):
        self.p2p_overlap_recompute = enabled


class LinearPipe(SplitBWLinear):
    """A linear whose dW goes through WeightGradStore when it is enabled.

    ``SplitBWLinear`` is Paddle's own split-backward linear: its PyLayer queues
    the weight-grad closure on ``WeightGradStore`` when ``enabled`` is set and
    computes it inline otherwise. That is what the scheduler pops inside a p2p
    window, so it is what the deferral path has to be tested against. Subclassed
    only to give ``seg_method`` a name to match on.
    """


class DeferredLinearPipe(Layer):
    """A real ``DeferredWeightGradLinear`` layer for the scheduler regression.

    The class switch makes the same model serve as the inline baseline and the
    deferred run. Unlike ``SplitBWLinear``, its backward is the implementation
    provided by this change, so the VPP scheduler must actually consume its
    queued dW for the run to match the baseline.
    """

    use_deferred = False

    def __init__(self, hidden):
        super().__init__()
        self.weight = self.create_parameter([hidden, hidden])
        self.weight.main_grad = None

    def forward(self, input):
        if self.use_deferred:
            return DeferredWeightGradLinear.apply(input, self.weight)
        return paddle.nn.functional.linear(input, self.weight)


class NoParamPipe(Layer):
    """A parameter-free chunk: queues no dW and registers no recompute.

    This is the asymmetric case. On the rank holding it the p2p window has
    nothing to fill it with (``have_dw`` false, ``rc_key`` None) while the other
    three ranks have both, so the four ranks must not disagree about whether the
    transfer is asynchronous. It is also what exercises the
    ``if WeightGradStore.cache:`` guard -- ``flush()`` enqueues unconditionally,
    so a blind call would put an empty batch on the queue and make ``have_dw``
    true here with nothing to run.
    """

    def forward(self, input):
        return paddle.tanh(input)


class RecomputePipe(Layer):
    """A chunk that both queues dW and registers a real recompute span.

    ``inner`` is a plain ``Linear`` on purpose: recompute replays the forward, so
    replaying a ``SplitBWLinear`` would queue its weight-grad closure a second
    time and double that gradient. The dW filler for this chunk comes from
    ``proj``, outside the span.
    """

    def __init__(self, hidden):
        super().__init__()
        self.inner = Linear(hidden, hidden, bias_attr=False)
        self.proj = SplitBWLinear(hidden, hidden, bias_attr=False)

    def forward(self, input):
        span = RecomputeWithoutOutput()
        hidden = span.recompute(self.inner, input, preserve_rng_state=False)
        out = self.proj(hidden)
        # Drops ``hidden``'s data and hooks ``out``: the span rebuilds it either
        # from that hook or, when the scheduler takes it, inside a p2p window.
        span.discard_output_and_register_recompute(out)
        return out


class CriterionPipe(Layer):
    def forward(self, logits, label):
        return paddle.nn.functional.cross_entropy(
            logits, label.reshape([-1]), reduction="mean"
        )


class ModelPipe(PipelineLayer):
    """8 segmentable layers -> exactly one per (vpp, pp) chunk at pp=4, vpp=2."""

    def __init__(self, **kwargs):
        decs = [
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (0, 0)
            LayerDesc(DeferredLinearPipe, HIDDEN),  # (0, 1), PR path
            LayerDesc(NoParamPipe),  # (0, 2)  no dW, no recompute
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (0, 3)
            LayerDesc(RecomputePipe, HIDDEN),  # (1, 0)  registers a span
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (1, 1)
            LayerDesc(RecomputePipe, HIDDEN),  # (1, 2)  registers a span
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (1, 3)
        ]
        super().__init__(
            layers=decs,
            loss_fn=CriterionPipe(),
            seg_method=(
                "layer:LinearPipe|DeferredLinearPipe|NoParamPipe|RecomputePipe"
            ),
            **kwargs,
        )


class TestPpDwRecomputeOverlap(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": PP_DEGREE,
            "pp_configs": {"best_unbalanced_scheduler": False},
        }
        fleet.init(is_collective=True, strategy=strategy)
        self._reset_stores()

    def tearDown(self):
        self._reset_stores()

    @staticmethod
    def _reset_stores():
        WeightGradStore.enabled = False
        install_recompute_p2p_overlap(_RcConfig(False))
        WeightGradStore.clear()
        RecomputeStore.clear()

    @staticmethod
    def _select_scheduler(acc_steps, best_unbalanced):
        """Point the live strategy at one of the two interleaved schedulers.

        ``distributed_model`` reads ``fleet.fleet._user_defined_strategy`` at call
        time, and the pp topology is the same either way, so switching schedulers
        needs no second ``fleet.init``.

        Only the one key is assigned: the setter merges what it is given, and
        feeding the getter's output back in raises on ``mp_configs`` fields it
        cannot round-trip.
        """
        strategy = fleet.fleet._user_defined_strategy
        strategy.hybrid_configs = {
            "pp_configs": {"best_unbalanced_scheduler": best_unbalanced}
        }
        strategy.pipeline_configs = {
            "accumulate_steps": acc_steps,
            "micro_batch_size": MICRO_BATCH_SIZE,
        }

    def _build(self):
        set_random_seed(1024)
        model = ModelPipe(
            num_stages=PP_DEGREE,
            num_virtual_pipeline_stages=VPP,
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        return fleet.distributed_model(model), fleet.distributed_optimizer(
            optimizer
        )

    def _reader(self, acc_steps, steps):
        # _check_data_valid wants exactly micro_batch_size * accumulate_steps.
        batch_size = MICRO_BATCH_SIZE * acc_steps
        return DataLoader(
            RandomDataset(batch_size * steps),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )

    def _run(self, acc_steps, defer_dw, early_recompute):
        """Train STEPS steps; return (losses, per-step grads, final params).

        Gradients are snapshotted after backward and before the optimizer step,
        which is the direct check: deferring dW only changes where the weight-grad
        GEMM runs and running a span early only changes when an activation is
        rebuilt, so every gradient must come out bit for bit identical.

        ``train_batch`` always steps the optimizer (and ``_optimizer_step``
        rescales the grads in place by 1/accumulate_steps), so the two halves are
        called separately to get at the grads in between. Final parameters are
        compared as well, which is what a gradient landing on the wrong parameter
        -- a wrong virtual-chunk key -- shows up as.
        """
        self._reset_stores()
        DeferredLinearPipe.use_deferred = defer_dw
        model, optimizer = self._build()

        WeightGradStore.enabled = defer_dw
        install_recompute_p2p_overlap(_RcConfig(early_recompute))
        losses = []
        grads = []
        try:
            for step, (img, label) in enumerate(
                self._reader(acc_steps, STEPS + 1)()
            ):
                if step >= STEPS:
                    break
                data = model._prepare_training([img, label], optimizer, None)
                losses.append(
                    np.array(model.forward_backward_pipeline(data, None))
                )
                grads.append(
                    [
                        (
                            name,
                            (
                                p.main_grad
                                if getattr(p, "main_grad", None) is not None
                                else p.grad
                            ),
                        )
                        for name, p in model.named_parameters()
                    ]
                )
                grads[-1] = [
                    (name, None if grad is None else grad.numpy().copy())
                    for name, grad in grads[-1]
                ]
                model._optimizer_step()
        finally:
            self._reset_stores()

        params = [
            (name, p.numpy().copy()) for name, p in model.named_parameters()
        ]
        return losses, grads, params

    def _assert_same(self, ref, got, label):
        ref_losses, ref_grads, ref_params = ref
        got_losses, got_grads, got_params = got

        self.assertEqual(len(ref_losses), STEPS)
        for step in range(STEPS):
            np.testing.assert_equal(
                ref_losses[step],
                got_losses[step],
                err_msg=f"{label} changed the loss at step {step}",
            )

            self.assertEqual(len(ref_grads[step]), len(got_grads[step]))
            for (name, ref_g), (got_name, got_g) in zip(
                ref_grads[step], got_grads[step]
            ):
                self.assertEqual(name, got_name)
                if ref_g is None or got_g is None:
                    self.assertIs(
                        ref_g,
                        got_g,
                        f"{label}: {name} has a gradient in one run only "
                        f"at step {step}",
                    )
                    continue
                np.testing.assert_equal(
                    ref_g,
                    got_g,
                    err_msg=(
                        f"{label} changed the gradient of {name} at step {step}"
                    ),
                )

        for (name, ref_p), (got_name, got_p) in zip(ref_params, got_params):
            self.assertEqual(name, got_name)
            np.testing.assert_equal(
                ref_p,
                got_p,
                err_msg=f"{label} moved parameter {name}",
            )

    def _check_fillers(self, acc_steps, best_unbalanced, scheduler):
        self._select_scheduler(acc_steps, best_unbalanced)
        baseline = self._run(acc_steps, False, False)
        self._assert_same(
            baseline,
            self._run(acc_steps, True, False),
            f"{scheduler}: dW deferral",
        )
        self._assert_same(
            baseline,
            self._run(acc_steps, True, True),
            f"{scheduler}: dW deferral + early recompute",
        )
        print(f"[pp dw/rc overlap] {scheduler} matches the baseline OK")

    def test_interleave(self):
        # accumulate_steps >= 2 * pp_degree -> PipelineParallelWithInterleave
        self._check_fillers(2 * PP_DEGREE, False, "interleave")

    def test_vpp_fthenb_balanced_memory(self):
        # pp_degree <= accumulate_steps < 2 * pp_degree, with
        # best_unbalanced_scheduler -> VPPFhenBInBalancedMemory, the schedule that
        # opens three separate windows and reorders which virtual chunk the last
        # stage's incoming gradient belongs to.
        self._check_fillers(PP_DEGREE, True, "vpp-fthenb balanced memory")

    def test_stores_are_drained_every_step(self):
        """Both stores must be empty when a step ends, on every rank.

        A leftover dW batch means some window popped nothing and the grad is
        missing; a leftover recompute group means a chunk key was built for a
        chunk whose backward never came. Either one is silent, so assert it.
        """
        acc_steps = 2 * PP_DEGREE
        self._select_scheduler(acc_steps, False)
        self._reset_stores()
        model, optimizer = self._build()
        WeightGradStore.enabled = True
        install_recompute_p2p_overlap(_RcConfig(True))
        try:
            for step, (img, label) in enumerate(self._reader(acc_steps, 3)()):
                if step >= 2:
                    break
                model.train_batch([img, label], optimizer)
                self.assertTrue(
                    WeightGradStore.funcs_queue.empty(),
                    "WeightGradStore.funcs_queue not drained at step end",
                )
                self.assertEqual(
                    WeightGradStore.cache,
                    [],
                    "WeightGradStore.cache not flushed at step end",
                )
                self.assertEqual(
                    RecomputeStore.groups,
                    {},
                    "RecomputeStore has spans left over at step end",
                )
        finally:
            self._reset_stores()
        print("[pp dw/rc overlap] both stores drained every step OK")


if __name__ == "__main__":
    unittest.main()
