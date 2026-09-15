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

"""Guard for ``RecomputeWithoutOutputManager`` and the mHC block recompute plan.

The manager exists for one reason: a ``RecomputeWithoutOutput`` whose input is
another's output cannot be replayed independently. Discarding an
output empties the alias ``save_for_backward`` kept -- ``detach_variable`` shares
the holder -- so a consumer replayed before its producer reads a cleared tensor.
Replaying in registration order (= forward order) fixes that, because
``_share_buffer_to`` propagates back to the holder-sharing aliases.

So the tests here are mostly about *order*:

  * a producer/consumer chain replayed in forward order reproduces the plain
    gradients bitwise;
  * the same chain replayed in reverse order fails -- which is what makes the
    ordering assertion non-vacuous;
  * the discard is skipped, not half-applied, when the boundary tensor carries no
    gradient, because a discarded-but-unhooked output is unrecoverable.

Plus the pure arithmetic of ``mhc_recompute_block_plan``, which decides who shares
a manager, and the layer-level wiring: exactly one ``[..., n*C]`` residual state
survives a block, and the block's gradients match the plain baseline.
"""

import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.recompute_utils import (
    mhc_recompute_block_plan,
    validate_recompute_modules,
)
from paddlefleet.tensor_parallel.random import (
    _MHC_RECOMPUTE_MANAGERS,
    RecomputeWithoutOutput,
    RecomputeWithoutOutputManager,
    finalize_mhc_recompute_block,
    get_mhc_recompute_manager,
    model_parallel_cuda_manual_seed,
)
from paddlefleet.transformer.hyper_connection import MhcAggregateRecompute
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
)

try:  # both import styles are in use in this directory
    from .test_mhc_fused_h_res_h_post_bda_recompute import (
        B,
        C,
        N,
        S,
        _make_config,
        _npy,
    )
except ImportError:
    from test_mhc_fused_h_res_h_post_bda_recompute import (
        B,
        C,
        N,
        S,
        _make_config,
        _npy,
    )


class TestManagerOrdering(unittest.TestCase):
    """The producer/consumer chain the manager exists for, in miniature.

    ``f`` stands in for a fused BDA: its output is the next one's input, so it is
    exactly the ``x_l -> x_(l+1)`` dependency that forbids independent replay.
    """

    @staticmethod
    def _chain(x, manager=None, recomputes=None):
        """Two chained recomputes; returns the tensor the block boundary would be."""

        def f(t):
            return t * 2.0 + 1.0

        out = x
        for _ in range(2):
            recompute = RecomputeWithoutOutput()
            out = recompute.recompute(f, out, preserve_rng_state=False)
            if manager is not None:
                manager.add(recompute)
            if recomputes is not None:
                recomputes.append(recompute)
        return f(out)

    def _grad(self, build):
        x = paddle.to_tensor(np.arange(6, dtype="float32").reshape(2, 3))
        x.stop_gradient = False
        build(x).sum().backward()
        return x.grad.numpy().copy()

    def test_forward_order_replay_matches_plain(self):
        def plain(x):
            out = x
            for _ in range(3):
                out = out * 2.0 + 1.0
            return out

        def managed(x):
            manager = RecomputeWithoutOutputManager()
            boundary = self._chain(x, manager=manager)
            manager.discard_all_outputs_and_register_unified_recompute(boundary)
            return boundary

        # Bitwise: the replay re-runs the same ops on the same values, so the
        # only thing that could move the result is a stale read.
        self.assertTrue(
            np.array_equal(self._grad(plain), self._grad(managed)),
            "managed block did not reproduce the plain gradient",
        )

    def test_discard_empties_the_consumers_saved_input(self):
        """The exact hazard the manager exists for, asserted on the mechanism.

        ``recompute.recompute`` saves ``detach_variable(args)``, which shares the
        producer's holder. So clearing the producer's output clears what the
        consumer will replay from, and only replaying the producer brings it
        back. This is the Paddle-side equivalent of Megatron re-pointing the
        output's ``StorageImpl``, and it is what makes forward order mandatory
        rather than merely tidy.
        """
        x = paddle.to_tensor(np.ones((2, 3), dtype="float32"))
        x.stop_gradient = False
        recomputes = []
        self._chain(x, recomputes=recomputes)
        producer, consumer = recomputes

        consumer_input = consumer.ctx.saved_tensor()[0]
        self.assertTrue(consumer_input._is_initialized())

        producer._discard_outputs()
        self.assertFalse(
            consumer_input._is_initialized(),
            "clearing the producer's output left the consumer's saved input "
            "intact, so the ordering constraint would not exist",
        )

        producer._recompute(None)
        self.assertTrue(
            consumer_input._is_initialized(),
            "_share_buffer_to did not propagate to the holder-sharing alias, "
            "which would make block replay impossible in Paddle",
        )

    def test_reverse_order_replay_does_not_reproduce_the_gradient(self):
        """Makes the forward-order assertion above non-vacuous.

        Replaying the consumer first reads its producer's already-cleared input.
        Paddle may surface that as an exception or as a silently wrong gradient
        (an emptied holder is not guaranteed to raise), so accept either -- what
        must not happen is matching the correct answer.
        """

        def plain(x):
            out = x
            for _ in range(3):
                out = out * 2.0 + 1.0
            return out

        def reversed_order(x):
            recomputes = []
            boundary = self._chain(x, recomputes=recomputes)
            for recompute in recomputes:
                recompute._discard_outputs()
            reversed_recomputes = recomputes[::-1]

            def hook(grad):
                for recompute in reversed_recomputes:
                    recompute._recompute(None)

            boundary.register_hook(hook)
            return boundary

        expected = self._grad(plain)
        try:
            got = self._grad(reversed_order)
        except Exception:
            return
        self.assertFalse(
            np.array_equal(expected, got),
            "reverse-order replay produced the correct gradient, so the "
            "forward-order test proves nothing",
        )

    def test_detached_boundary_skips_the_discard(self):
        """Discarding without a hook would be unrecoverable, not just wasteful.

        ``_recompute`` is the only place that assigns ``ctx.inputs``, so a member
        that is discarded but never replayed makes
        ``RecomputeWithoutOutputFunction.backward`` fail. Megatron discards
        unconditionally and only guards the hook; here both are guarded.
        """
        x = paddle.ones([2, 3])  # stop_gradient=True
        manager = RecomputeWithoutOutputManager()
        recomputes = []
        boundary = self._chain(x, manager=manager, recomputes=recomputes)
        self.assertTrue(boundary.stop_gradient)

        manager.discard_all_outputs_and_register_unified_recompute(boundary)
        for recompute in recomputes:
            for output in recompute.outputs:
                self.assertTrue(
                    output._is_initialized(),
                    "output was discarded with no hook to bring it back",
                )

    def test_repeat_hook_fire_is_a_noop(self):
        """Paddle can fire a hook twice for a multi-branch tensor."""
        x = paddle.to_tensor(np.ones((2, 3), dtype="float32"))
        x.stop_gradient = False
        manager = RecomputeWithoutOutputManager()
        recomputes = []
        boundary = self._chain(x, manager=manager, recomputes=recomputes)
        manager.discard_all_outputs_and_register_unified_recompute(boundary)

        boundary.sum().backward()
        # Second fire: every member already cleared its ctx, so this must not
        # raise and must not corrupt anything.
        for recompute in recomputes:
            recompute._recompute(None)


class TestBlockRegistry(unittest.TestCase):
    """Stands in for Megatron's ``TransformerBlock.forward`` layer loop."""

    def tearDown(self):
        _MHC_RECOMPUTE_MANAGERS.clear()

    def test_same_block_id_shares_one_manager(self):
        first = get_mhc_recompute_manager((0, 0), (0, 0))
        self.assertIs(first, get_mhc_recompute_manager((0, 0), (0, 1)))
        self.assertIsNot(first, get_mhc_recompute_manager((0, 1), (1, 0)))

    def test_finalize_pops_so_the_next_forward_gets_a_fresh_manager(self):
        first = get_mhc_recompute_manager((0, 0), (0, 0))
        boundary = paddle.ones([2])
        finalize_mhc_recompute_block((0, 0), boundary)
        self.assertNotIn((0, 0), _MHC_RECOMPUTE_MANAGERS)
        self.assertIsNot(first, get_mhc_recompute_manager((0, 0), (0, 0)))

    def test_a_position_that_does_not_advance_starts_a_new_manager(self):
        """A forward that never reached its block end must not be extended.

        The entry outlives that forward -- it is module state, not a local like
        Megatron's -- and its recomputes belong to a dead graph. Appending to it would
        make every later step replay them. The next forward is recognised by its
        position not advancing, since positions only increase within a block.
        """
        abandoned = get_mhc_recompute_manager((0, 0), (0, 0))
        abandoned.add(object())
        # Same layer, same half: a retry of the very first request.
        retried = get_mhc_recompute_manager((0, 0), (0, 0))
        self.assertIsNot(retried, abandoned)
        self.assertEqual(retried.recomputes, [])

    def test_a_single_layer_block_detects_the_retry(self):
        """Why the position carries the half-layer.

        With ``mhc_recompute_layer_num=1`` a block is one layer, so a retry's
        first request is ``(l, attention)`` while the abandoned manager's last was
        ``(l, mlp)``. Comparing layer numbers alone would call that "no change"
        and reuse the stale manager.
        """
        abandoned = get_mhc_recompute_manager((0, 0), (7, 0))
        self.assertIs(abandoned, get_mhc_recompute_manager((0, 0), (7, 1)))
        self.assertIsNot(abandoned, get_mhc_recompute_manager((0, 0), (7, 0)))

    def test_finalize_without_a_manager_is_a_noop(self):
        # Reachable when the block holds no mHC layer at all.
        finalize_mhc_recompute_block(("mtp", 3), paddle.ones([2]))


class _PlanConfig:
    """Just the fields ``mhc_recompute_block_plan`` reads."""

    def __init__(
        self,
        num_hidden_layers,
        pp=1,
        vpp=None,
        block=None,
        head=0,
        tail=0,
        blocks=None,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_empty_layers_add_in_head = head
        self.num_empty_layers_add_in_tail = tail
        self.pipeline_model_parallel_size = pp
        self.virtual_pipeline_model_parallel_size = vpp
        self.mhc_recompute_layer_num = block
        self.recompute_num_layers = None
        self.recompute_modules = (
            ["mhc_block"] if blocks is None else {"mhc_block": blocks}
        )


class TestBlockPlan(unittest.TestCase):
    def _plan(self, config, num_layers):
        return [mhc_recompute_block_plan(i, config) for i in range(num_layers)]

    def test_none_means_one_block_per_chunk(self):
        # Megatron's default: its blocks are per-TransformerBlock, i.e. per stage.
        config = _PlanConfig(4, pp=1)
        plan = self._plan(config, 4)
        self.assertEqual([block for block, _ in plan], [(0, 0)] * 4)
        self.assertEqual([end for _, end in plan], [False, False, False, True])

    def test_block_size_two_over_four_layers(self):
        # The worked example from Megatron's docstring.
        config = _PlanConfig(4, pp=1, block=2)
        plan = self._plan(config, 4)
        self.assertEqual(
            [block for block, _ in plan], [(0, 0), (0, 0), (0, 1), (0, 1)]
        )
        self.assertEqual([end for _, end in plan], [False, True, False, True])

    def test_blocks_never_span_a_pipeline_stage(self):
        config = _PlanConfig(8, pp=2, block=3)
        plan = self._plan(config, 8)
        chunks = [block[0] for block, _ in plan]
        self.assertEqual(chunks, [0, 0, 0, 0, 1, 1, 1, 1])
        # Layer 3 ends its chunk early even though 3 % 3 != 0, so the block does
        # not run into stage 1.
        self.assertEqual(
            [end for _, end in plan], [False, False, True, True] * 2
        )

    def test_short_final_chunk_still_ends_its_block(self):
        # 7 layers over pp=2 -> chunks of 4 and 3; the second is unfilled.
        config = _PlanConfig(7, pp=2, block=2)
        plan = self._plan(config, 7)
        self.assertEqual(
            [end for _, end in plan],
            [False, True, False, True, False, True, True],
        )

    def test_vpp_chunks_get_separate_blocks(self):
        config = _PlanConfig(8, pp=2, vpp=2)
        blocks = [block for block, _ in self._plan(config, 8)]
        self.assertEqual(
            blocks, [(0, 0)] * 2 + [(1, 0)] * 2 + [(2, 0)] * 2 + [(3, 0)] * 2
        )

    def test_mtp_layers_form_their_own_blocks(self):
        config = _PlanConfig(4, pp=1)
        for layer_number in range(2):
            block, end = mhc_recompute_block_plan(
                layer_number, config, is_mtp_layer=True
            )
            self.assertEqual(block, ("mtp", layer_number))
            self.assertTrue(end, "an MTP block must end on its own layer")

    def test_explicit_blocks_are_taken_as_written(self):
        config = _PlanConfig(6, pp=1, blocks=[[1, 2], [4]])
        plan = self._plan(config, 6)
        self.assertEqual(
            [block for block, _ in plan],
            [
                None,
                ("explicit", 0),
                ("explicit", 0),
                None,
                ("explicit", 1),
                None,
            ],
        )
        # Only the last layer of each block ends it; the gaps end nothing.
        self.assertEqual(
            [end for _, end in plan],
            [False, False, True, False, True, False],
        )

    def test_explicit_blocks_use_logical_layer_ids(self):
        """Ids skip the empty head layers, like every recompute_modules entry."""
        config = _PlanConfig(4, pp=1, head=2, blocks=[[0, 1]])
        # Physical layers 2 and 3 are logical 0 and 1.
        self.assertEqual(
            [block for block, _ in self._plan(config, 6)],
            [None, None, ("explicit", 0), ("explicit", 0), None, None],
        )

    def test_explicit_blocks_do_not_apply_to_mtp(self):
        config = _PlanConfig(4, pp=1, blocks=[[0]])
        block, end = mhc_recompute_block_plan(0, config, is_mtp_layer=True)
        self.assertEqual(block, ("mtp", 0))
        self.assertTrue(end)


class _ValidateConfig:
    def __init__(self, **kw):
        self.recompute_modules = kw.get("recompute_modules")
        self.recompute_granularity = kw.get(
            "recompute_granularity", "selective"
        )
        self.recompute_method = kw.get("recompute_method", "block")
        self.recompute_num_layers = kw.get("recompute_num_layers")
        self.enable_hyper_connections = kw.get("enable_hyper_connections", True)
        self.mhc_recompute_layer_num = kw.get("mhc_recompute_layer_num")
        self.num_hidden_layers = kw.get("num_hidden_layers", 4)
        self.num_empty_layers_add_in_head = 0
        self.num_empty_layers_add_in_tail = 0
        self.pipeline_model_parallel_size = kw.get(
            "pipeline_model_parallel_size", 1
        )
        self.virtual_pipeline_model_parallel_size = None
        self.num_nextn_predict_layers = 0


class TestBlockRecomputeValidation(unittest.TestCase):
    def test_accepts_the_intended_config(self):
        validate_recompute_modules(
            _ValidateConfig(
                recompute_modules=["mhc_block"], mhc_recompute_layer_num=2
            )
        )

    def test_requires_hyper_connections(self):
        with self.assertRaisesRegex(ValueError, "enable_hyper_connections"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules=["mhc_block"],
                    enable_hyper_connections=False,
                )
            )

    def test_rejects_full_granularity(self):
        # Full recompute replays the layer, so the manager would collect the same
        # outputs twice in one micro-batch.
        with self.assertRaisesRegex(ValueError, "selective"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules=["mhc_block"],
                    recompute_granularity="full",
                )
            )

    def test_rejects_mhc_forward_at_the_same_time(self):
        with self.assertRaisesRegex(ValueError, "mhc_forward"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules=["mhc_block", "mhc_forward"],
                )
            )

    def test_rejects_a_block_larger_than_a_pipeline_chunk(self):
        with self.assertRaisesRegex(ValueError, "pipeline"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules=["mhc_block"],
                    num_hidden_layers=8,
                    pipeline_model_parallel_size=4,
                    mhc_recompute_layer_num=3,
                )
            )

    def test_rejects_a_flat_layer_selector(self):
        # [0, 1] cannot say whether that is one block or two.
        with self.assertRaisesRegex(ValueError, "flat layer selector"):
            validate_recompute_modules(
                _ValidateConfig(recompute_modules={"mhc_block": [0, 1]})
            )

    def test_rejects_a_non_positive_block_size(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules=["mhc_block"], mhc_recompute_layer_num=0
                )
            )

    def test_accepts_explicit_blocks(self):
        validate_recompute_modules(
            _ValidateConfig(
                recompute_modules={"mhc_block": [[0, 1], [3]]},
                num_hidden_layers=4,
            )
        )

    def test_rejects_explicit_blocks_with_a_block_size(self):
        with self.assertRaisesRegex(ValueError, "mhc_recompute_layer_num"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules={"mhc_block": [[0, 1]]},
                    mhc_recompute_layer_num=2,
                )
            )

    def test_rejects_blocks_mixed_with_bare_layer_ids(self):
        with self.assertRaisesRegex(ValueError, "mixes blocks"):
            validate_recompute_modules(
                _ValidateConfig(recompute_modules={"mhc_block": [[0, 1], 3]})
            )

    def test_rejects_an_empty_block(self):
        with self.assertRaisesRegex(ValueError, "is empty"):
            validate_recompute_modules(
                _ValidateConfig(recompute_modules={"mhc_block": [[0], []]})
            )

    def test_rejects_a_non_int_layer_id(self):
        with self.assertRaisesRegex(ValueError, "must be ints"):
            validate_recompute_modules(
                _ValidateConfig(recompute_modules={"mhc_block": [["0"]]})
            )

    def test_rejects_a_block_with_a_gap(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validate_recompute_modules(
                _ValidateConfig(recompute_modules={"mhc_block": [[0, 2]]})
            )

    def test_rejects_an_out_of_range_block(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules={"mhc_block": [[3, 4]]},
                    num_hidden_layers=4,
                )
            )

    def test_rejects_a_layer_in_two_blocks(self):
        with self.assertRaisesRegex(ValueError, "repeats layer 1"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules={"mhc_block": [[0, 1], [1, 2]]}
                )
            )

    def test_rejects_a_block_crossing_a_pipeline_chunk(self):
        # 8 layers over pp=4 -> chunks of 2, boundaries at 2/4/6.
        with self.assertRaisesRegex(ValueError, "chunk boundary at layer 2"):
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules={"mhc_block": [[1, 2]]},
                    num_hidden_layers=8,
                    pipeline_model_parallel_size=4,
                )
            )

    def test_the_cross_chunk_error_prints_the_layout(self):
        """The message must show the legal spans, not just the failure.

        Deriving them by hand needs the chunk size *and* the empty-head shift,
        which is exactly what makes an explicit block list easy to get wrong.
        """
        with self.assertRaises(ValueError) as caught:
            validate_recompute_modules(
                _ValidateConfig(
                    recompute_modules={"mhc_block": [[1, 2]]},
                    num_hidden_layers=8,
                    pipeline_model_parallel_size=4,
                )
            )
        message = str(caught.exception)
        self.assertIn("chunk 0: layers 0-1", message)
        self.assertIn("chunk 3: layers 6-7", message)


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "requires CUDA")
class TestLayerLevelBlockRecompute(unittest.TestCase):
    """Two stacked mHC layers driven by ``recompute_modules=['mhc_block']``.

    Two layers is the smallest configuration that exercises the thing the
    manager is for: layer 1's recomputes consume layer 0's ``[..., n*C]``
    residual state, which the block discards.
    """

    def tearDown(self):
        _MHC_RECOMPUTE_MANAGERS.clear()

    def _build(self, **overrides):
        config = _make_config(num_hidden_layers=2, **overrides)
        model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)
        paddle.seed(42)
        spec = get_gpt_layer_local_spec(config)
        return [
            HyperConnectionTransformerLayer(
                config=config,
                sublayers_spec=spec.sublayers_spec,
                layer_number=layer_number,
            )
            for layer_number in range(2)
        ]

    @staticmethod
    def _x_np(seed):
        return np.random.RandomState(seed).randn(S, B, N * C).astype("float32")

    def _run(self, layers, x_np):
        x = paddle.to_tensor(x_np, dtype="float32")
        x.stop_gradient = False
        hidden_states = x
        for layer in layers:
            layer.train()
            hidden_states = layer.forward(
                {"hidden_states": hidden_states, "attention_mask": None}
            )["hidden_states"]
        loss = hidden_states.astype("float32").sum()
        loss.backward()
        grads = {"loss": _npy(loss), "x": _npy(x.grad)}
        for index, layer in enumerate(layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grads[f"{index}.{name}"] = _npy(param.grad)
        return grads

    def test_block_plan_marks_only_the_last_layer(self):
        layers = self._build(
            recompute_granularity="selective",
            recompute_modules=["mhc_block"],
            mhc_recompute_layer_num=2,
        )
        self.assertTrue(all(layer.recompute_mhc_block for layer in layers))
        self.assertEqual(
            [layer._mhc_block_id for layer in layers], [(0, 0), (0, 0)]
        )
        self.assertEqual(
            [layer._mhc_is_block_end for layer in layers], [False, True]
        )

    def test_gradients_match_the_plain_baseline(self):
        """The replay must be numerically free.

        Not bitwise: the mHC-aggregate replay reassociates a few fp32 sums
        inside the mHC projection, which predates block recompute (see
        ``test_mhc_fused_h_res_h_post_bda_recompute``). What must hold is that
        nothing reads a stale tensor, which shows up as a gross mismatch rather
        than as last-bit noise.
        """
        x_np = np.random.RandomState(0).randn(S, B, N * C).astype("float32")

        reference = self._run(self._build(), x_np)
        _MHC_RECOMPUTE_MANAGERS.clear()
        blocked = self._run(
            self._build(
                recompute_granularity="selective",
                recompute_modules=["mhc_block"],
                mhc_recompute_layer_num=2,
            ),
            x_np,
        )

        self.assertEqual(set(reference), set(blocked), "gradient sets differ")
        for key, expected in reference.items():
            self.assertFalse(np.isnan(expected).any(), f"{key} baseline is NaN")
            np.testing.assert_allclose(
                expected, blocked[key], rtol=2e-5, atol=2e-5, err_msg=key
            )

    def test_only_the_boundary_residual_survives_the_block(self):
        """The point of the whole exercise, asserted directly.

        Every member the block collected is discarded, and the tensor the next
        layer receives -- the block boundary -- is not one of them. So of the four
        ``[..., n*C]`` residual states two layers produce, one stays resident
        instead of all four.
        """
        collected = []
        real_add = RecomputeWithoutOutputManager.add

        def spy(manager, recompute):
            collected.append(recompute)
            return real_add(manager, recompute)

        layers = self._build(
            recompute_granularity="selective",
            recompute_modules=["mhc_block"],
            mhc_recompute_layer_num=2,
        )
        x = paddle.to_tensor(
            np.random.RandomState(1).randn(S, B, N * C).astype("float32")
        )
        x.stop_gradient = False

        with mock.patch.object(RecomputeWithoutOutputManager, "add", spy):
            hidden_states = x
            for layer in layers:
                layer.train()
                hidden_states = layer.forward(
                    {"hidden_states": hidden_states, "attention_mask": None}
                )["hidden_states"]

        # Per half-layer: the mHC aggregate, the layernorm and the fused BDA,
        # except the block-end MLP BDA, which produces the boundary tensor
        # instead.
        self.assertEqual(len(collected), 2 * 2 * 3 - 1)
        for recompute in collected:
            for output in recompute.outputs:
                if output is None:
                    continue
                self.assertFalse(
                    output._is_initialized(),
                    "a member registered with the manager was not discarded",
                )
        self.assertTrue(
            hidden_states._is_initialized(),
            "the block boundary must stay resident -- it is what the unified "
            "hook replays from",
        )

        hidden_states.astype("float32").sum().backward()
        self.assertIsNotNone(x.grad, "unified hook did not restore the block")

    def _released_shapes(self, **overrides):
        """Multiset of output shapes the block discards, from one forward."""
        collected = []
        real_add = RecomputeWithoutOutputManager.add

        def spy(manager, recompute):
            collected.append(recompute)
            return real_add(manager, recompute)

        layers = self._build(
            recompute_granularity="selective",
            recompute_modules=["mhc_block"],
            mhc_recompute_layer_num=2,
            **overrides,
        )
        x = paddle.to_tensor(
            np.random.RandomState(3).randn(S, B, N * C).astype("float32")
        )
        x.stop_gradient = False

        with mock.patch.object(RecomputeWithoutOutputManager, "add", spy):
            hidden_states = x
            for layer in layers:
                layer.train()
                hidden_states = layer.forward(
                    {"hidden_states": hidden_states, "attention_mask": None}
                )["hidden_states"]

        shapes = {}
        for recompute in collected:
            for output in recompute.outputs:
                if output is not None:
                    key = tuple(output.shape)
                    shapes[key] = shapes.get(key, 0) + 1
        return shapes, layers

    def test_release_set_matches_megatron(self):
        """Pin the exact set of tensors a block releases.

        Megatron's per-half-layer discard set is the ``aggregate`` output, the
        layernorm output and the fused-BDA output; ``h_res``/``h_post`` stay live
        because ``compute_mappings`` sits outside its checkpoint. We match that on
        both kernel paths: :class:`MhcAggregateRecompute` keeps
        ``h_res``/``h_post`` resident whether or not ``mappings_cache`` is
        available, which is what removes the replay ordering rule between the
        aggregate and the fused BDA. This test is what stops the set drifting.
        """
        # 2 layers x 2 half-layers = 4 aggregate + 4 layernorm recomputes, all
        # [S,B,C]; 3 BDA recomputes (the 4th is the boundary and stays live).
        megatron_set = {(S, B, C): 4 + 4, (S, B, N * C): 3}
        for use_fused_mhc in (True, False):
            with self.subTest(use_fused_mhc=use_fused_mhc):
                shapes, layers = self._released_shapes(
                    use_fused_mhc=use_fused_mhc
                )
                hc = layers[0].self_attention_hyper_connection
                # The two paths differ in whether the replay redoes the mapping
                # head, but not in what they release.
                self.assertEqual(hc.supports_mappings_cache, use_fused_mhc)
                self.assertEqual(shapes, megatron_set)

    def test_mapping_head_runs_once_per_half_layer(self):
        """Megatron's compute profile, without moving the recompute boundary.

        Without ``mappings_cache`` the replay covers the whole module, so
        ``compute_mappings`` runs twice per half-layer -- 0.231 ms of pure loss at
        the 43L dims for a ~50 element/token saving.
        """
        module = type(self._build()[0].self_attention_hyper_connection)
        real = module.compute_mappings

        for label, block_on, expected in (
            ("no recompute", False, 4),  # 2 layers x 2 half-layers
            ("mhc_block", True, 4),  # cached, so once each despite the replay
        ):
            with self.subTest(label):
                overrides = (
                    {
                        "recompute_granularity": "selective",
                        "recompute_modules": ["mhc_block"],
                        "mhc_recompute_layer_num": 2,
                    }
                    if block_on
                    else {}
                )
                layers = self._build(use_fused_mhc=True, **overrides)
                calls = []

                def counted(self_, x, real=real, calls=calls):
                    calls.append(None)
                    return real(self_, x)

                with mock.patch.object(module, "compute_mappings", counted):
                    self._run(layers, self._x_np(11))
                self.assertEqual(len(calls), expected)

    def test_h_res_h_post_stay_resident(self):
        """``MhcAggregateRecompute`` must never discard its trailing two outputs.

        ``fused_h_res_h_post_bda`` keeps holder-sharing aliases of
        ``h_res``/``h_post``, so discarding them would empty what it
        replays from -- silently, since a cleared tensor reads as garbage rather
        than raising. Keeping them resident is what removes the need to either
        clone them or reason about which one replays first, so it is asserted
        directly here rather than left to the release-set multiset above.
        """
        for use_fused_mhc in (True, False):
            with self.subTest(use_fused_mhc=use_fused_mhc):
                heads = []
                real = MhcAggregateRecompute.recompute

                def spy(agg, *args, heads=heads, real=real, **kwargs):
                    outputs = real(agg, *args, **kwargs)
                    heads.append(outputs)
                    return outputs

                layers = self._build(
                    use_fused_mhc=use_fused_mhc,
                    recompute_granularity="selective",
                    recompute_modules=["mhc_block"],
                    mhc_recompute_layer_num=2,
                )
                # Inspected between forward and backward: the replay refills a
                # discarded output, so after backward this test would pass even
                # if h_res/h_post were being discarded.
                with mock.patch.object(MhcAggregateRecompute, "recompute", spy):
                    hidden_states = paddle.to_tensor(self._x_np(17))
                    hidden_states.stop_gradient = False
                    for layer in layers:
                        layer.train()
                        hidden_states = layer.forward(
                            {
                                "hidden_states": hidden_states,
                                "attention_mask": None,
                            }
                        )["hidden_states"]

                self.assertEqual(len(heads), 4)  # 2 layers x 2 half-layers
                for aggregated, h_res, h_post in heads:
                    self.assertFalse(
                        aggregated._is_initialized(),
                        "the aggregate output should have been discarded",
                    )
                    self.assertTrue(
                        h_res._is_initialized(), "h_res was discarded"
                    )
                    self.assertTrue(
                        h_post._is_initialized(), "h_post was discarded"
                    )

    def test_clear_data_does_not_free_through_a_reshape_alias(self):
        """The Paddle semantics that decide where mHC recompute may put things.

        ``Tensor._clear_data()`` resets one ``DenseTensor``'s
        ``shared_ptr<Allocation>`` (paddle ``dense_tensor.h:244``). A ``detach()``
        alias shares that ``DenseTensor`` (``eager_method.cc:1455``) so it follows;
        a ``reshape()`` alias is a separate ``DenseTensor`` holding its own
        reference to the same ``Allocation``, so it does not -- and while it is
        alive nothing is freed. PyTorch has no equivalent problem because
        ``untyped_storage().resize_(0)`` mutates the shared StorageImpl.

        This is why ``FusedProjRms`` saves ``x`` rather than ``x.reshape(...)``:
        saving the reshape pinned every ``[..., n*C]`` residual state and cost the
        block 96 of the 160 MiB it frees.
        """
        held = paddle.randn([256, 256])
        detached = held.detach()
        reshaped = held.reshape([-1, 128])

        paddle.device.synchronize()
        before = paddle.device.memory_allocated()
        held._clear_data()
        paddle.device.synchronize()

        self.assertFalse(held._is_initialized())
        self.assertFalse(
            detached._is_initialized(), "detach alias should follow the clear"
        )
        self.assertTrue(
            reshaped._is_initialized(), "reshape alias unexpectedly cleared"
        )
        self.assertEqual(
            paddle.device.memory_allocated(),
            before,
            "a live reshape alias should keep the allocation",
        )

    def test_mapping_graph_does_not_pin_the_residual(self):
        """The ``FusedProjRms`` fix, asserted on the mechanism.

        The mapping cache keeps the projection's graph alive across the two
        passes, so whatever that graph saved stays reachable until backward. Its
        first saved tensor is the mHC input, i.e. an alias of the residual state
        the block discards -- so it must follow ``_clear_data()``. It only does
        because ``FusedProjRms`` saves ``x`` and not ``x.reshape(...)``; with the
        reshape it stayed initialized and pinned the whole ``[..., n*C]`` buffer,
        costing the block 96 of the 160 MiB it frees.
        """
        from paddlefleet.fusions import fused_mhc_kernels

        proj_rms = getattr(fused_mhc_kernels, "FusedProjRms", None)
        if proj_rms is None:
            self.skipTest("cuTile fused kernels unavailable")

        contexts = []
        real_forward = proj_rms.forward

        def spy(ctx, *args, **kwargs):
            out = real_forward(ctx, *args, **kwargs)
            contexts.append(ctx)
            return out

        layers = self._build(
            use_fused_mhc=True,
            recompute_granularity="selective",
            recompute_modules=["mhc_block"],
            mhc_recompute_layer_num=2,
        )
        x = paddle.to_tensor(self._x_np(13))
        x.stop_gradient = False

        with mock.patch.object(proj_rms, "forward", staticmethod(spy)):
            hidden_states = x
            for layer in layers:
                layer.train()
                hidden_states = layer.forward(
                    {"hidden_states": hidden_states, "attention_mask": None}
                )["hidden_states"]

        # One projection per half-layer, run once each thanks to the cache.
        self.assertEqual(len(contexts), 4)
        # The first half-layer reads the live block input; the other three read
        # residual states the block discarded.
        cleared = sum(
            not ctx.saved_tensor()[0]._is_initialized() for ctx in contexts
        )
        self.assertEqual(
            cleared,
            3,
            "the projection's saved input did not follow the block discard, so "
            "it is pinning the residual state",
        )

        hidden_states.astype("float32").sum().backward()
        self.assertIsNotNone(x.grad, "unified hook did not restore the block")

    def test_fp32_upcast_predicate(self):
        """``materializes_fp32_input`` is why the whole module must be wrapped.

        On the native path this hides an fp32 ``[..., n*C]`` copy -- twice
        the residual state, the largest tensor in the half-layer. Lifting
        ``compute_mappings`` out, Megatron-style, would pin it.
        """
        for use_fused_mhc in (False, True):
            layers = self._build(use_fused_mhc=use_fused_mhc)
            hc = layers[0].self_attention_hyper_connection
            self.assertEqual(hc._widen_in_kernel, use_fused_mhc)
            self.assertEqual(hc.materializes_fp32_input, not use_fused_mhc)

    def test_block_is_bitwise_on_the_fused_path(self):
        """The fused path must stay bitwise against the plain baseline.

        This is what ``_FixedOrderMappings`` buys and what any future attempt to
        narrow the recompute has to preserve: pinning ``dx``'s accumulation order is
        the whole reason ``compute_mappings`` and ``aggregate`` live in one node.
        A plain composition of the two measures 25/33 gradients non-bitwise.
        """
        x_np = np.random.RandomState(7).randn(S, B, N * C).astype("float32")

        reference = self._run(self._build(use_fused_mhc=True), x_np)
        _MHC_RECOMPUTE_MANAGERS.clear()
        blocked = self._run(
            self._build(
                use_fused_mhc=True,
                recompute_granularity="selective",
                recompute_modules=["mhc_block"],
                mhc_recompute_layer_num=2,
            ),
            x_np,
        )

        self.assertEqual(set(reference), set(blocked), "gradient sets differ")
        for key, expected in reference.items():
            self.assertFalse(np.isnan(expected).any(), f"{key} baseline is NaN")
            self.assertTrue(
                np.array_equal(expected, blocked[key]),
                f"{key} is not bitwise: max abs diff "
                f"{np.abs(expected - blocked[key]).max():.3e}",
            )

    def test_an_abandoned_forward_does_not_poison_the_next_one(self):
        """The registry-leak guard, end to end.

        A forward that stops inside a block -- it raised, or a layer skipped its
        mHC path -- leaves its manager in ``_MHC_RECOMPUTE_MANAGERS`` holding
        recomputes from a graph that is about to die. Here the abandoned forward runs
        only layer 0 of a two-layer block, so ``finalize`` never fires.

        The next full forward must not extend that manager. Its recomputes are the
        assertion: appending to it would let the block's finalize discard and
        replay them, which costs a whole layer's recompute on every later step and
        can fault on the dead graph. Numerically it goes unnoticed -- the stale
        recomputes feed nothing live -- so gradients alone would not catch this.
        """
        x_np = np.random.RandomState(23).randn(S, B, N * C).astype("float32")
        reference = self._run(self._build(use_fused_mhc=True), x_np)
        _MHC_RECOMPUTE_MANAGERS.clear()

        def block_layers():
            return self._build(
                use_fused_mhc=True,
                recompute_granularity="selective",
                recompute_modules=["mhc_block"],
                mhc_recompute_layer_num=2,
            )

        layer = block_layers()[0]
        layer.train()
        layer.forward(
            {"hidden_states": paddle.to_tensor(x_np), "attention_mask": None}
        )
        abandoned = _MHC_RECOMPUTE_MANAGERS[layer._mhc_block_id]
        stranded = list(abandoned.recomputes)
        self.assertTrue(
            stranded,
            "the abandoned forward was expected to register recomputes",
        )

        blocked = self._run(block_layers(), x_np)

        self.assertEqual(
            abandoned.recomputes,
            stranded,
            "the next forward extended the abandoned manager",
        )
        for recompute in stranded:
            for output in recompute.outputs:
                self.assertTrue(
                    output is None or output._is_initialized(),
                    "an abandoned member was discarded, so it will be replayed",
                )
        for key, expected in reference.items():
            self.assertTrue(
                np.array_equal(expected, blocked[key]),
                f"{key} is not bitwise after an abandoned forward: max abs diff "
                f"{np.abs(expected - blocked[key]).max():.3e}",
            )

    def test_explicit_blocks_leave_the_other_layers_alone(self):
        """A nested block list is the only way to opt a layer out entirely.

        With ``[[1]]`` only layer 1 recomputes, and being a one-layer block its
        MLP BDA is the boundary -- so it releases four ``[S,B,C]`` outputs (two
        aggregates, two layernorms) and one ``[S,B,N*C]`` residual state, while
        layer 0 releases nothing.
        """
        x_np = np.random.RandomState(31).randn(S, B, N * C).astype("float32")
        reference = self._run(self._build(), x_np)
        _MHC_RECOMPUTE_MANAGERS.clear()

        layers = self._build(
            recompute_granularity="selective",
            recompute_modules={"mhc_block": [[1]]},
        )
        self.assertEqual(
            [layer.recompute_mhc_block for layer in layers], [False, True]
        )
        self.assertEqual(
            [layer._mhc_block_id for layer in layers], [None, ("explicit", 0)]
        )
        self.assertEqual(
            [layer._mhc_is_block_end for layer in layers], [False, True]
        )

        shapes = {}
        real_add = RecomputeWithoutOutputManager.add

        def spy(manager, recompute_unit):
            # Recorded here rather than after the run: _recompute drops
            # ``outputs`` once it has replayed.
            for output in recompute_unit.outputs:
                if output is not None:
                    key = tuple(output.shape)
                    shapes[key] = shapes.get(key, 0) + 1
            return real_add(manager, recompute_unit)

        with mock.patch.object(RecomputeWithoutOutputManager, "add", spy):
            blocked = self._run(layers, x_np)

        self.assertEqual(shapes, {(S, B, C): 4, (S, B, N * C): 1})

        for key, expected in reference.items():
            np.testing.assert_allclose(
                expected, blocked[key], rtol=2e-5, atol=2e-5, err_msg=key
            )

    def test_layernorm_recompute_is_skipped_without_a_block(self):
        """``mhc_forward`` alone must not touch the layernorms.

        The norm recomputes depend on the manager's forward-order replay to see a
        restored ``aggregated``, so they only exist in block mode.
        """
        layers = self._build(
            recompute_granularity="selective", recompute_modules=["mhc_forward"]
        )
        for layer in layers:
            self.assertFalse(layer.recompute_mhc_block)
            self.assertIsNone(layer._mhc_block_manager(half_layer=0))
            # The flags are still computed -- they only say "the norm is real".
            self.assertTrue(layer.mhc_checkpoint_input_layernorm)
            self.assertTrue(layer.mhc_checkpoint_post_attention_layernorm)

    def test_block_size_one_makes_every_layer_its_own_block(self):
        layers = self._build(
            recompute_granularity="selective",
            recompute_modules=["mhc_block"],
            mhc_recompute_layer_num=1,
        )
        self.assertEqual(
            [layer._mhc_block_id for layer in layers], [(0, 0), (0, 1)]
        )
        self.assertEqual(
            [layer._mhc_is_block_end for layer in layers], [True, True]
        )

    def test_eval_mode_creates_no_manager(self):
        layers = self._build(
            recompute_granularity="selective",
            recompute_modules=["mhc_block"],
            mhc_recompute_layer_num=2,
        )
        hidden_states = paddle.to_tensor(
            np.random.RandomState(2).randn(S, B, N * C).astype("float32")
        )
        for layer in layers:
            layer.eval()
            hidden_states = layer.forward(
                {"hidden_states": hidden_states, "attention_mask": None}
            )["hidden_states"]
        self.assertEqual(_MHC_RECOMPUTE_MANAGERS, {})


if __name__ == "__main__":
    unittest.main()
