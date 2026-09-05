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
"""Activation offloading under pipeline parallelism (PP=4, VPP=2, 4 GPUs).

What only shows up here, and not in the single-card component tests, is the part
driven by the pipeline schedule: the four micro-step anchors, the chunk id that
keys a group under virtual pipelining, the recorded backward group order that
cross-group prefetch depends on, and per-iteration bookkeeping surviving a whole
accumulation cycle.

The bar is the same as everywhere else: offloading only changes where an
activation lives between forward and backward, so loss and every gradient must
match a run with the feature off exactly. A control group runs the reference
twice first -- if that is not identical, the environment is non-deterministic and
a mismatch says nothing about offloading.

Run it the way CI does, as a script under the launcher::

    PYTHONPATH=$PWD:$PWD/src python -m paddle.distributed.launch \
        --gpus 0,1,2,3 \
        tests/multi_card_tests/activation_offload/test_activation_offload_pp.py
"""

from __future__ import annotations

import functools
import os
import random
import unittest

# Must be set before paddle initialises, or the control group will not be
# reproducible and every comparison below becomes meaningless.
os.environ.setdefault("FLAGS_cudnn_deterministic", "1")
os.environ.setdefault("FLAGS_embedding_deterministic", "1")

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet import distributed_model

import paddlefleet
from paddlefleet.activation_offload import (
    enable_fleet_prefetch,
    get_offload_manager,
    reset_offload_manager,
)
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4
VPP_DEGREE = 2
ACCUMULATE_STEPS = 8
MICRO_BATCH_SIZE = 1
SEQ_LEN = 128
VOCAB_SIZE = 1024
SEED = 46
OFFLOAD_MODULES = [
    "attn_norm",
    "qkv_linear",
    "core_attn",
    "attn_proj",
    "mlp_norm",
]

# Three iterations is the minimum that exercises the whole lifecycle: the first
# one is the learning iteration (every boundary offloaded, backward group order
# being recorded, no budget yet), the second is the first steady-state one, and
# the third is what shows that nothing accumulates across iterations.
NUM_ITERS = 3

# Every group of a chunk runs backward exactly once per iteration, so the
# recorded order has this many entries and one successor link fewer.
BWD_GROUPS_PER_ITER = VPP_DEGREE * ACCUMULATE_STEPS


def _set_random_seed(seed_: int):
    """Set random seed for reproducibility (per pipeline stage)."""
    seed = seed_ + (
        100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
    )
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    if paddle.distributed.is_initialized() and paddle.cuda.device_count() > 0:
        paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _make_config(offload: bool) -> GPTConfig:
    """A plain dense GPT sized so the layers divide evenly across the chunks.

    11 real layers plus 2 head and 3 tail empty ones is 16, which is
    ``PP_DEGREE * VPP_DEGREE * 2`` -- two layers per chunk, so every chunk on
    every stage holds real transformer layers and therefore real regions.
    """
    kwargs = {
        "vocab_size": VOCAB_SIZE,
        "max_sequence_length": SEQ_LEN,
        "num_hidden_layers": 11,
        "hidden_size": 512,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "normalization": "RMSNorm",
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "use_cpu_initialization": True,
        "parallel_output": True,
        "tie_word_embeddings": True,
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "rotary_base": 10000,
        "rope_scaling": 1.0,
        "init_method": functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        "output_layer_init_method": functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        "use_qk_norm": True,
        "num_empty_layers_add_in_head": 2,
        "num_empty_layers_add_in_tail": 3,
        "pipeline_model_parallel_size": PP_DEGREE,
        "virtual_pipeline_model_parallel_size": VPP_DEGREE,
    }
    if offload:
        kwargs.update(
            fine_grained_activation_offloading=True,
            offload_modules=list(OFFLOAD_MODULES),
            # Activations at this shape are a few hundred KB, so the 2MB default
            # threshold would filter out every one of them and nothing would be
            # offloaded at all.
            min_offloaded_tensor_bytes=1,
            # Binding NUMA changes the CPU affinity of the whole test process,
            # which a test must not do as a side effect.
            activation_offload_numa_bind=False,
        )
    return GPTConfig(**kwargs)


def _make_inputs():
    """One fixed batch per iteration, built once and reused by every run.

    The iterations must not be identical: a manager that silently reused the
    previous iteration's pinned contents would still match the baseline if every
    iteration saw the same data.
    """
    position_ids = (
        paddle.arange(SEQ_LEN, dtype=paddle.int64)
        .unsqueeze(0)
        .expand([MICRO_BATCH_SIZE, -1])
    )
    batches = []
    for _ in range(NUM_ITERS):
        data = paddle.randint(
            low=0,
            high=VOCAB_SIZE,
            shape=(MICRO_BATCH_SIZE, SEQ_LEN + 1),
        )
        batches.append(
            (
                {
                    "input_ids": [data[:, :-1]] * ACCUMULATE_STEPS,
                    "position_ids": [position_ids] * ACCUMULATE_STEPS,
                },
                [data[:, 1:]] * ACCUMULATE_STEPS,
            )
        )
    return batches


def _run(offload, batches):
    """Build a fresh model and run ``NUM_ITERS`` accumulation cycles.

    Returns ``(losses, grads, snapshots)``. ``snapshots`` holds, per iteration,
    the cumulative manager stats and the pinned bytes held afterwards; it is
    empty when offloading is off.
    """
    # The manager is a process-wide singleton built by TransformerLayer.__init__
    # from the config, and only the first construction's kwargs apply -- so it
    # must not survive from an earlier run in this process.
    reset_offload_manager()
    config = _make_config(offload)

    _set_random_seed(SEED)
    model = gpt_builder(
        config,
        num_stages=PP_DEGREE,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )
    pipe = distributed_model(model)

    mgr = None
    if offload:
        mgr = get_offload_manager()
        assert mgr.enabled, "offloading requested but the manager is disabled"
        # After distributed_model, because the hooks read _virtual_pp_rank off
        # the wrapper to key each group by chunk.
        enable_fleet_prefetch(pipe)

    losses, snapshots = [], []
    for inputs in batches:
        loss = pipe.forward_backward_pipeline(inputs, None)
        losses.append(
            None if loss is None else loss.astype("float32").numpy().copy()
        )
        if mgr is not None:
            mgr.end_iteration()
            snapshots.append((dict(mgr.stats), mgr.pool.total_bytes))

    # Gradients are never cleared, so these are accumulated over all iterations:
    # a divergence in any single one of them shows up here.
    grads = {
        name: param.grad.detach().astype("float32").numpy().copy()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return losses, grads, snapshots


def _delta(snapshots, key, i):
    """Per-iteration change of a cumulative stats counter."""
    prev = snapshots[i - 1][0][key] if i else 0
    return snapshots[i][0][key] - prev


class TestActivationOffloadPP(unittest.TestCase):
    """All three runs happen in ``setUpClass``, which is not a style choice.

    ``initialize_fleet`` may only be called once per process, and the global
    micro-step hooks ``enable_fleet_prefetch`` installs cannot be removed again,
    so the reference runs have to precede the offloading one and no test method
    may trigger a run of its own. Each method is therefore a pure assertion over
    stored results, and their order cannot matter.
    """

    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": PP_DEGREE,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
            # The offloading hooks are the four micro-step callbacks of
            # PipelineParallel. The overlap scheduler runs its own schedule and
            # does not go through them, so it has to stay off here.
            "pp_configs": {
                "forward_backward_overlap_scheduler": False,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": ACCUMULATE_STEPS,
            "micro_batch_size": MICRO_BATCH_SIZE,
        }
        initialize_fleet(strategy)

        _set_random_seed(SEED)
        batches = _make_inputs()

        cls.ref = _run(offload=False, batches=batches)
        cls.ref_again = _run(offload=False, batches=batches)
        cls.got = _run(offload=True, batches=batches)
        cls.stats = cls.got[2]

    # ---------------- numerical equivalence ----------------

    def _assert_bit_exact(self, ref, got, tag):
        ref_losses, ref_grads, _ = ref
        got_losses, got_grads, _ = got

        for i, (want, have) in enumerate(zip(ref_losses, got_losses)):
            if want is None or have is None:
                continue  # only the last stage produces a loss
            self.assertTrue(
                np.all(np.isfinite(want)) and np.all(np.isfinite(have)),
                f"{tag}: non-finite loss at iteration {i}: {want} vs {have}",
            )
            np.testing.assert_array_equal(
                have, want, err_msg=f"{tag}: loss differs at iteration {i}"
            )

        self.assertEqual(
            set(ref_grads),
            set(got_grads),
            f"{tag}: different sets of parameters received gradients",
        )
        self.assertGreater(len(got_grads), 0, f"{tag}: no gradients captured")
        for name in sorted(got_grads):
            np.testing.assert_array_equal(
                got_grads[name],
                ref_grads[name],
                err_msg=f"{tag}: gradient of {name} is not bit-exact",
            )

    def test_control_group_is_deterministic(self):
        """Two identical runs with the feature off must agree bit for bit.

        Without this, a mismatch in the test below could just as well come from
        a non-deterministic kernel as from offloading.
        """
        self._assert_bit_exact(self.ref, self.ref_again, "control group")

    def test_offloading_is_bit_exact(self):
        """Offloading only moves an activation, so nothing may change.

        Bit-exact rather than close: a looser bound cannot separate a harmless
        floating-point reordering from a real stream or event race, and races are
        exactly what this machinery can get wrong.
        """
        self._assert_bit_exact(self.ref, self.got, "offloading on")

    # ---------------- the schedule wiring ----------------

    def test_every_iteration_offloads_and_prefetches(self):
        """The existence proof for the forward and backward anchors.

        ``packed`` needs FORWARD_BEGIN to have opened a group, ``prefetched``
        needs BACKWARD_BEGIN to have queued it. A regression that dropped either
        callback would still produce correct numbers -- lazy reload covers for it
        -- and would only show up here.
        """
        for i in range(NUM_ITERS):
            with self.subTest(iteration=i):
                self.assertGreater(
                    _delta(self.stats, "packed", i),
                    0,
                    "nothing was offloaded: either no region was entered or the "
                    "size threshold filtered every activation out",
                )
                self.assertGreater(
                    _delta(self.stats, "prefetched", i),
                    0,
                    "nothing was prefetched, so every reload fell back to the "
                    "lazy path in unpack",
                )

    def test_backward_group_order_is_recorded(self):
        """The recorded successor table must span one whole iteration.

        Under interleaving the group that runs backward next is not this chunk's
        next micro-batch, so the order is observed during the first iteration.
        The count doubles as a check that the schedule really was interleaved with
        ``VPP_DEGREE`` chunks: a collapse to VPP=1 would halve it.
        """
        mgr = get_offload_manager()
        self.assertEqual(
            len(mgr._bwd_next),
            BWD_GROUPS_PER_ITER - 1,
            "the recorded backward group order does not cover one iteration",
        )

    def test_cross_group_prefetch_starts_after_the_learning_iteration(self):
        """Cross-group prefetch is off while learning and on afterwards.

        It needs both the successor table and a finite budget, and neither exists
        during the first iteration; the first iteration is therefore expected to
        report zero, and every later one to report some.
        """
        self.assertEqual(
            _delta(self.stats, "head_prefetch", 0),
            0,
            "cross-group prefetch ran during the learning iteration, where the "
            "successor of a group is not known yet",
        )
        for i in range(1, NUM_ITERS):
            with self.subTest(iteration=i):
                self.assertGreater(
                    _delta(self.stats, "head_prefetch", i),
                    0,
                    "BACKWARD_END never queued the next group, so the first "
                    "tensor of every group has nothing to hide its copy behind",
                )

    # ---------------- per-iteration bookkeeping ----------------

    def test_nothing_is_left_over_at_the_end_of_an_iteration(self):
        """Every offloaded activation must be consumed within its iteration.

        A leftover means the pinned buffer is recycled late, which is how the
        pool grows without bound over a long run.
        """
        for i in range(NUM_ITERS):
            with self.subTest(iteration=i):
                self.assertEqual(
                    _delta(self.stats, "not_consumed", i),
                    0,
                    "offloaded in forward but never consumed in backward",
                )

    def test_every_consumption_is_accounted_for(self):
        """``packed`` and ``unpacked`` must balance, per iteration.

        ``unpacked`` counts consumptions and ``packed`` counts records, so the
        difference is exactly the deduplicated slots. Separately, every consumed
        record is probed once, hence ``hit + late == packed``.

        Deliberately not asserted: ``late == 0``. Whether a copy has landed by the
        time backward asks for it depends on how fast the device is, so pinning
        that down is a recipe for an intermittent failure. What must hold is that
        each record lands in one bucket or the other.
        """
        for i in range(NUM_ITERS):
            with self.subTest(iteration=i):
                packed = _delta(self.stats, "packed", i)
                self.assertEqual(
                    _delta(self.stats, "unpacked", i),
                    packed + _delta(self.stats, "dedup_hits", i),
                    "the number of consumptions does not match the number of "
                    "offloaded activations plus the deduplicated slots",
                )
                self.assertEqual(
                    _delta(self.stats, "hit", i)
                    + _delta(self.stats, "late", i),
                    packed,
                    "some record was consumed without its reload event being "
                    "probed",
                )

    def test_pinned_memory_does_not_grow_across_iterations(self):
        """Pinned buffers must be recycled, not accumulated.

        The bound is one micro-batch group's worth of bytes. If recycling failed,
        the pool would have to grow by every group of the iteration, so anything
        under a single group means the buffers are coming back. It is not zero on
        purpose: a buffer whose reload event has not completed yet cannot be
        handed out again, so a request can still miss and add one bucket. That
        effect is bounded by how many copies are in flight, not by the number of
        iterations.

        Only the steady-state iterations are compared. The first one offloads
        every boundary in whole-group mode, which is a different regime.
        """
        held = [total for _, total in self.stats]
        self.assertGreater(held[1], 0, "no pinned memory was held at all")
        one_group = (
            _delta(self.stats, "packed_bytes", NUM_ITERS - 1)
            / BWD_GROUPS_PER_ITER
        )
        self.assertLess(
            held[-1] - held[1],
            one_group,
            f"pinned memory grew by a whole group across iterations: {held}, "
            f"one group is {one_group:.0f} bytes",
        )

    def test_the_pool_gave_out_every_activation_it_was_asked_for(self):
        """A full pool degrades silently: the activation just stays on device.

        This shape is far below any pool limit, so a non-zero count here means
        something is holding buffers past their consumption.
        """
        self.assertEqual(
            self.stats[-1][0]["pool_oom"],
            0,
            "the pinned pool hit its capacity, so some activations were never "
            "moved off the device",
        )


if __name__ == "__main__":
    unittest.main()
