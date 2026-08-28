# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
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
"""``dsa_indexer_loss_bwd_p2p_overlap``: the deferred branch must match
the inline one.

The flag moves the whole indexer-loss branch out of the layer and into a
pipeline micro-step callback. The only thing that makes that safe is an
equivalence claim -- the five ``DSAIndexer`` gradients and the logged KL are the
same either way -- so that claim is what most of these tests pin down. How much
the move actually buys is a performance property and is not testable here;
:class:`TestForwardEndRegistration` covers the part that is, namely *which*
callback the drain is attached to.

The two paths are driven differently on purpose, because that asymmetry *is* the
feature:

* off -- one grad-enabled forward, then ``out.sum().backward()`` runs
  ``TileLangCSAIndexerLossAutoScaler.backward``;
* on, ``in_recompute=True`` -- one ``paddle.no_grad()`` forward (what the
  recompute wrapper's first pass looks like), which only enqueues, then an
  explicit ``drain()`` standing in for the pipeline forward hook;
* on, ``in_recompute=False`` -- one grad-enabled forward, which is the *only*
  pass an unwrapped layer gets, so that is the one that enqueues. Still no
  ``backward()``: the branch does not need one either way.

The last case is why the flag does not depend on ``recompute_granularity``. Both
spellings are asserted against the same inline baseline.
"""

from __future__ import annotations

import contextlib
import enum
import sys
import types
import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.transformer import indexer_loss_overlap
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper

from .hybrid_mla_utils import (
    _GPU,
    _PARENT_REPO_AVAILABLE,
    INDEX_TOPK,
    WINDOW,
    _add_repo_root_to_sys_path,
    _build_module,
    _create_mqa_config,
    _make_inputs,
    _row_end,
)

_INDEXER_PARAMS = (
    "wq_b.linear.weight",
    "wk.linear.weight",
    "weights_proj.linear.weight",
)

_SEQLEN = WINDOW + INDEX_TOPK


def _sparse_config(overlap: bool):
    config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
    # Phase 3: attention consumes window + top-k and the KL covers that same
    # set. The overlap flag is only implemented for this phase.
    config.dsa_indexer_use_sparse_loss = True
    config.dsa_indexer_loss_bwd_p2p_overlap = overlap
    # ``validate_config`` also insists on a pipeline and on the model really
    # having a ``-2`` (DSAIndexer) layer; ``_create_mqa_config`` leaves both
    # unset because the single-layer module fixture does not need them.
    config.pipeline_model_parallel_size = 4
    config.csa_compress_ratios = [-2] * config.num_hidden_layers
    return config


def _indexer_grads(module):
    """``{name: fp32 numpy}`` for the indexer weights that got a gradient."""
    out = {}
    for name, param in module.indexer.named_parameters():
        if param.grad is not None:
            out[name] = param.grad.cast("float32").numpy().copy()
    return out


def _logged_loss():
    values = DSAIndexerLossLoggingHelper.tracker.get("values")
    return None if values is None else values.cast("float32").numpy().copy()


@_GPU
class TestOverlapEquivalence(unittest.TestCase):
    """Deferred and inline paths agree on gradients and on the logged loss."""

    def setUp(self):
        indexer_loss_overlap.drain_all()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def tearDown(self):
        indexer_loss_overlap.drain_all()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def _run_inline(self, seed=0):
        module = _build_module(_sparse_config(overlap=False), bf16=True)
        query, key, w_v, x, qr = _make_inputs(
            _SEQLEN, seed=seed, with_hidden=True
        )
        for tensor in (query, key, x, qr):
            tensor.stop_gradient = False
        module.train()
        out = module(
            query,
            key,
            None,
            None,
            _row_end([_SEQLEN], _SEQLEN),
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        out.cast("float32").sum().backward()
        return module, _indexer_grads(module), _logged_loss()

    def _run_deferred(self, seed=0, drain=True, in_recompute=True):
        module = _build_module(_sparse_config(overlap=True), bf16=True)
        query, key, w_v, x, qr = _make_inputs(
            _SEQLEN, seed=seed, with_hidden=True
        )
        module.train()
        # ``in_recompute=True`` + ``paddle.no_grad()`` is what the recompute
        # wrapper's first pass looks like; ``in_recompute=False`` with grad on is
        # the single forward an unwrapped layer gets. Both are the pass the
        # deferred path claims. ``query``/``x``/``qr`` deliberately keep
        # ``stop_gradient=True``: the branch never touches the backbone graph, so
        # it must work without one.
        ctx = paddle.no_grad() if in_recompute else contextlib.nullcontext()
        with ctx:
            out = module(
                query,
                key,
                None,
                None,
                _row_end([_SEQLEN], _SEQLEN),
                v_b_proj_weight=w_v,
                x=x,
                qr=qr,
                in_recompute=in_recompute,
            )
        self.assertEqual(indexer_loss_overlap.pending(), 1)
        if drain:
            self.assertEqual(indexer_loss_overlap.drain(), 1)
            self.assertEqual(indexer_loss_overlap.pending(), 0)
        return module, out, _indexer_grads(module), _logged_loss()

    def _copy_indexer_weights(self, src, dst):
        """Make the two modules share weights so gradients are comparable."""
        state = dict(src.indexer.state_dict())
        dst.indexer.set_state_dict(state)

    def _assert_matches_inline(self, in_recompute):
        inline_mod, inline_grads, inline_loss = self._run_inline()
        self.assertTrue(inline_grads, "inline path produced no indexer grads")

        deferred_mod = _build_module(_sparse_config(overlap=True), bf16=True)
        self._copy_indexer_weights(inline_mod, deferred_mod)
        # Rebuild the whole layer from the inline one so every projection the
        # target depends on is identical, not just the indexer's.
        deferred_mod.set_state_dict(inline_mod.state_dict())
        DSAIndexerLossLoggingHelper.tracker.clear()

        query, key, w_v, x, qr = _make_inputs(_SEQLEN, seed=0, with_hidden=True)
        deferred_mod.train()
        ctx = paddle.no_grad() if in_recompute else contextlib.nullcontext()
        with ctx:
            deferred_mod(
                query,
                key,
                None,
                None,
                _row_end([_SEQLEN], _SEQLEN),
                v_b_proj_weight=w_v,
                x=x,
                qr=qr,
                in_recompute=in_recompute,
            )
        self.assertEqual(indexer_loss_overlap.drain(), 1)
        deferred_grads = _indexer_grads(deferred_mod)

        self.assertEqual(
            sorted(deferred_grads),
            sorted(inline_grads),
            "deferred path touched a different set of indexer weights",
        )
        for name in inline_grads:
            self.assertTrue(
                paddle.allclose(
                    paddle.to_tensor(inline_grads[name]),
                    paddle.to_tensor(deferred_grads[name]),
                    rtol=1e-5,
                    atol=1e-7,
                ).item(),
                f"{name} gradient differs between inline and deferred paths",
            )

    def test_deferred_gradients_match_the_inline_path(self):
        """Recompute-wrapped layer: the branch is claimed by the no-grad pass."""
        self._assert_matches_inline(in_recompute=True)

    def test_deferred_gradients_match_without_recompute(self):
        """Unwrapped layer: the single grad-enabled forward claims it instead.

        The case the old ``recompute_granularity == "full"`` gate rejected
        outright. Nothing about the branch needs the replay -- it only needs to
        run once, during the pipeline's forward phase -- so this has to produce
        the same five gradients as the inline path too.
        """
        self._assert_matches_inline(in_recompute=False)

    def test_no_grad_forward_enqueues_and_produces_no_backward_node(self):
        module, out, grads, loss = self._run_deferred(drain=True)
        # The layer output must come back untouched and gradient-free: with the
        # flag on there is no PyLayer wrapping it, which is the point.
        self.assertTrue(out.stop_gradient)
        self.assertTrue(bool(paddle.isfinite(out.cast("float32")).all()))
        self.assertTrue(grads, "drain produced no indexer gradients")
        for name in _INDEXER_PARAMS:
            self.assertIn(name, grads, f"{name} received no gradient")
            self.assertTrue(
                bool(paddle.isfinite(paddle.to_tensor(grads[name])).all()),
                f"{name} gradient is not finite",
            )
            self.assertGreater(
                float(abs(paddle.to_tensor(grads[name])).max()),
                0.0,
                f"{name} gradient is all zeros",
            )
        self.assertIsNotNone(loss, "drain logged no loss")

    def test_queue_stays_empty_until_drained(self):
        _, _, grads, _ = self._run_deferred(drain=False)
        # Nothing ran yet, so no gradient exists: this is what makes the
        # ``drain_all()`` safety net mandatory rather than cosmetic.
        self.assertFalse(grads)
        self.assertEqual(indexer_loss_overlap.pending(), 1)
        self.assertEqual(indexer_loss_overlap.drain_all(), 1)
        self.assertEqual(indexer_loss_overlap.pending(), 0)

    def test_grad_enabled_forward_is_a_no_op_when_deferring(self):
        """The replay pass must not enqueue or compute a second time.

        Only true for a recompute-wrapped layer (``in_recompute=True``), where the
        no-grad pass already claimed the branch. Without the wrapper the same
        grad-enabled forward is the one that enqueues -- see
        ``test_deferred_gradients_match_without_recompute``.
        """
        module = _build_module(_sparse_config(overlap=True), bf16=True)
        query, key, w_v, x, qr = _make_inputs(_SEQLEN, seed=0, with_hidden=True)
        for tensor in (query, key, x, qr):
            tensor.stop_gradient = False
        module.train()
        out = module(
            query,
            key,
            None,
            None,
            _row_end([_SEQLEN], _SEQLEN),
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
            in_recompute=True,
        )
        self.assertEqual(indexer_loss_overlap.pending(), 0)
        out.cast("float32").sum().backward()
        self.assertFalse(
            _indexer_grads(module),
            "the grad-enabled replay attached a loss it should have skipped",
        )

    def test_the_branch_is_claimed_exactly_once_either_way(self):
        """The enqueue predicate's truth table, straight from the layer.

        Two passes when wrapped, one when not, and exactly one enqueue in both
        cases. This is the whole reason the flag can ignore
        ``recompute_granularity``.
        """
        module = _build_module(_sparse_config(overlap=True), bf16=True)
        module.train()
        for in_recompute, expected in (
            # (wrapped?, which grad states enqueue)
            (True, {False}),
            (False, {True}),
        ):
            fired = {
                grad_on
                for grad_on in (True, False)
                if self._predicate(module, in_recompute, grad_on)
            }
            self.assertEqual(
                fired,
                expected,
                f"in_recompute={in_recompute} enqueues on the wrong pass",
            )

    @staticmethod
    def _predicate(module, in_recompute, grad_on):
        ctx = contextlib.nullcontext() if grad_on else paddle.no_grad()
        with ctx:
            return module._needs_indexer_loss(in_recompute)


@_GPU
class TestOverlapConfigValidation(unittest.TestCase):
    """The flag must refuse the configurations where it would lose the loss."""

    def test_recompute_granularity_is_not_a_constraint(self):
        """No granularity may be rejected -- the predicate adapts per layer.

        A config-level gate could not have been correct: with
        ``recompute_method`` ``first_n`` / ``block`` some layers are wrapped and
        some are not (``recompute_utils.py:180``), so the granularity says
        nothing about what any individual DSA layer does.
        """
        for granularity in ("full", "selective", "core_attn", None):
            config = _sparse_config(overlap=True)
            config.recompute_granularity = granularity
            indexer_loss_overlap.validate_config(config)

    def test_requires_the_sparse_phase(self):
        config = _sparse_config(overlap=True)
        config.dsa_indexer_use_sparse_loss = False
        with self.assertRaisesRegex(ValueError, "sparse"):
            indexer_loss_overlap.validate_config(config)

    def test_accepts_the_supported_combination(self):
        indexer_loss_overlap.validate_config(_sparse_config(overlap=True))

    def test_disabled_flag_validates_anything(self):
        config = _sparse_config(overlap=False)
        config.dsa_indexer_use_sparse_loss = False
        indexer_loss_overlap.validate_config(config)

    def test_warmup_phase_ignores_the_flag(self):
        """The module-level gate keeps phase 2 on its own loss path."""
        config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
        config.dsa_indexer_use_sparse_loss = False
        config.dsa_indexer_loss_bwd_p2p_overlap = True
        module = _build_module(config, bf16=True)
        self.assertFalse(module.indexer_loss_overlap)


@_GPU
class TestOverlapWithMainGrad(unittest.TestCase):
    """``amp_master_grad``: the deferred write must land in ``main_grad``.

    This is the interaction the flag is most likely to break, and the one a
    plain single-card test would miss. Production runs O2 + ``amp_master_grad``,
    where ``MixPrecisionLayer`` registers a per-parameter grad hook that asserts
    ``param.grad is None``, accumulates into an fp32 ``param.main_grad`` and then
    clears the incoming bf16 grad
    (``paddle/distributed/fleet/utils/mix_precision_utils.py:44-71``).

    The deferred path drives ``paddle.autograd.backward`` during the *forward*
    pass. What has to hold is that this still goes through that hook rather than
    around it: writing ``param.grad`` directly would trip the assert the moment
    the real backward runs, and rebinding ``main_grad`` would silently detach the
    parameter from the sharding fused buffer (``tensor_fusion_helper.py:711``).

    Sharding bucketing itself is not exercised here and does not need to be: with
    ``pipeline_model_parallel_size > 1`` the comm-overlap hooks are disabled and
    ``FusedCommBuffer.add_grad`` never runs, so the collective is a single
    ``reduce_gradients`` after every backward -- there is no per-parameter
    check-in counter for a forward-time write to desynchronise.
    """

    def setUp(self):
        indexer_loss_overlap.drain_all()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def tearDown(self):
        indexer_loss_overlap.drain_all()
        DSAIndexerLossLoggingHelper.tracker.clear()

    @staticmethod
    def _wrap(module):
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionLayer,
        )

        return MixPrecisionLayer(module, dtype="bfloat16")

    @staticmethod
    def _main_grads(module):
        out = {}
        for name, param in module.indexer.named_parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                out[name] = main_grad.cast("float32").numpy().copy()
        return out

    def test_deferred_write_goes_through_the_main_grad_hook(self):
        module = _build_module(_sparse_config(overlap=True), bf16=True)
        wrapped = self._wrap(module)
        query, key, w_v, x, qr = _make_inputs(_SEQLEN, seed=0, with_hidden=True)
        module.train()
        with paddle.no_grad():
            wrapped(
                query,
                key,
                None,
                None,
                _row_end([_SEQLEN], _SEQLEN),
                v_b_proj_weight=w_v,
                x=x,
                qr=qr,
                in_recompute=True,
            )
        self.assertEqual(indexer_loss_overlap.drain(), 1)

        main_grads = self._main_grads(module)
        for name in _INDEXER_PARAMS:
            param = dict(module.indexer.named_parameters())[name]
            # The assert that would fire on the next real backward.
            self.assertIsNone(
                param.grad,
                f"{name}.grad is set; MixPrecisionLayer's hook would assert",
            )
            self.assertIn(name, main_grads, f"{name} got no main_grad")
            self.assertEqual(main_grads[name].dtype.name, "float32")
            self.assertGreater(
                float(abs(paddle.to_tensor(main_grads[name])).max()),
                0.0,
                f"{name} main_grad is all zeros",
            )

    def test_main_grad_buffer_is_updated_in_place(self):
        """A rebind would drop the param out of the sharding fused buffer."""
        module = _build_module(_sparse_config(overlap=True), bf16=True)
        wrapped = self._wrap(module)
        params = dict(module.indexer.named_parameters())
        # Pre-seed main_grad so the hook takes its ``add_`` branch, which is the
        # branch production hits: the fused buffer is allocated at optimizer
        # construction, so main_grad is never None by the time a step runs.
        for name in _INDEXER_PARAMS:
            params[name].main_grad = paddle.zeros_like(
                params[name], dtype="float32"
            )
        addresses = {
            name: params[name].main_grad.data_ptr() for name in _INDEXER_PARAMS
        }

        query, key, w_v, x, qr = _make_inputs(_SEQLEN, seed=0, with_hidden=True)
        module.train()
        with paddle.no_grad():
            wrapped(
                query,
                key,
                None,
                None,
                _row_end([_SEQLEN], _SEQLEN),
                v_b_proj_weight=w_v,
                x=x,
                qr=qr,
                in_recompute=True,
            )
        self.assertEqual(indexer_loss_overlap.drain(), 1)

        for name in _INDEXER_PARAMS:
            self.assertEqual(
                params[name].main_grad.data_ptr(),
                addresses[name],
                f"{name}.main_grad was rebound instead of accumulated in place",
            )
            self.assertGreater(
                float(abs(params[name].main_grad).max()),
                0.0,
                f"{name}.main_grad was not accumulated into",
            )


class TestForwardEndRegistration(unittest.TestCase):
    """The drain must land on ``P2P_ISSUED``, with ``FORWARD_END`` as a fallback.

    ``P2P_ISSUED`` is raised by the schedule *after* the micro-step's p2p has been
    issued and before its wait handles are consumed. That side of the issue is the
    whole overlap: ``isend``/``irecv`` gate the NCCL kernel on an event recorded
    on the calculation stream at issue time, so a drain queued before the issue --
    which is what ``FORWARD_END`` (``pipeline_parallel.py:1690``) gives -- lands
    inside that event's reach and the send waits for the branch instead of running
    alongside it.

    ``FORWARD_END`` stays registered anyway, self-disarming on the first
    ``P2P_ISSUED``, because schedules that were never taught to defer their wait
    raise no ``P2P_ISSUED`` and losing the gradient is worse than losing the
    speedup.

    No GPU needed: this pins the wiring, not the numerics.
    """

    def setUp(self):
        import paddle.distributed.fleet.meta_parallel.pipeline_parallel as pp

        locations = pp.PipelineParallelMicroStepLocations
        self._registry = pp.pipeline_parallel_callbacks_.hooks
        if not hasattr(locations, "P2P_ISSUED"):
            locations = self._synthesise_p2p_issued(pp, locations)
        self._location = locations.FORWARD_END
        self._p2p_location = locations.P2P_ISSUED
        self._saved = list(self._registry[self._location])
        self._saved_p2p = list(self._registry[self._p2p_location])
        self._saved_flag = indexer_loss_overlap._HOOKS_REGISTERED
        self._saved_window = indexer_loss_overlap._P2P_WINDOW_SEEN
        self._saved_fe_drains = indexer_loss_overlap._FORWARD_END_DRAINS
        self._saved_warned = indexer_loss_overlap._FALLBACK_WARNED
        indexer_loss_overlap._HOOKS_REGISTERED = False
        indexer_loss_overlap._P2P_WINDOW_SEEN = False
        indexer_loss_overlap._FORWARD_END_DRAINS = 0
        indexer_loss_overlap._FALLBACK_WARNED = False

    def _synthesise_p2p_issued(self, pp, locations):
        """Stand in for the location on a Paddle that predates it.

        ``P2P_ISSUED`` reached Paddle together with the framework half of this
        feature, and this repository's CI may run against an older one. Skipping
        the class there would leave the p2p wiring -- the only thing it exists to
        pin -- untested on the runner that gates the PR, so the member is
        supplied instead. It behaves like the real one for these tests' purposes:
        they raise the location themselves through Paddle's own dispatcher, and
        the source resolves the member by name at registration time, so it picks
        the stand-in up. An ``Enum`` cannot be extended in place, hence the
        rebuild; the dispatcher's registry is keyed by member, so the rebuilt
        members need their own entries.
        """
        locations = enum.Enum(
            locations.__name__,
            [(m.name, m.value) for m in locations]
            + [("P2P_ISSUED", "p2p_issued")],
        )
        patcher = mock.patch.object(
            pp, "PipelineParallelMicroStepLocations", locations
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        added = [m for m in locations if m not in self._registry]
        for member in added:
            self._registry[member] = []
        self.addCleanup(lambda: [self._registry.pop(m, None) for m in added])
        return locations

    def tearDown(self):
        self._registry[self._location] = self._saved
        self._registry[self._p2p_location] = self._saved_p2p
        indexer_loss_overlap._HOOKS_REGISTERED = self._saved_flag
        indexer_loss_overlap._P2P_WINDOW_SEEN = self._saved_window
        indexer_loss_overlap._FORWARD_END_DRAINS = self._saved_fe_drains
        indexer_loss_overlap._FALLBACK_WARNED = self._saved_warned
        indexer_loss_overlap.drain_all()

    @staticmethod
    def _fake_pp_model():
        class _FakePipelineParallel:
            def _forward_step(self, *args, **kwargs):
                raise AssertionError("not meant to run")

        return _FakePipelineParallel()

    def test_registers_on_both_locations(self):
        self.assertTrue(
            indexer_loss_overlap.register_pipeline_hooks(self._fake_pp_model())
        )
        self.assertIn(
            indexer_loss_overlap._p2p_issued_hook,
            self._registry[self._p2p_location],
        )
        self.assertIn(
            indexer_loss_overlap._forward_end_hook,
            self._registry[self._location],
        )

    def test_registration_is_idempotent(self):
        """The trainer may wrap the model more than once (train then eval)."""
        model = self._fake_pp_model()
        self.assertTrue(indexer_loss_overlap.register_pipeline_hooks(model))
        self.assertTrue(indexer_loss_overlap.register_pipeline_hooks(model))
        self.assertEqual(
            self._registry[self._p2p_location].count(
                indexer_loss_overlap._p2p_issued_hook
            ),
            1,
            "the drain was registered twice; it would run the branch twice",
        )
        self.assertEqual(
            self._registry[self._location].count(
                indexer_loss_overlap._forward_end_hook
            ),
            1,
            "the fallback was registered twice",
        )

    def test_a_non_pipeline_model_is_declined(self):
        """PP=1 never enters ``_forward_step``, so the callback would never fire.

        Returning ``False`` is what makes the trainer warn and fall back to
        ``drain_all()`` instead of silently never draining.
        """
        self.assertFalse(indexer_loss_overlap.register_pipeline_hooks(object()))
        self.assertEqual(self._registry[self._location], self._saved)
        self.assertEqual(self._registry[self._p2p_location], self._saved_p2p)

    def test_an_older_paddle_registers_only_the_fallback(self):
        """No ``P2P_ISSUED`` in the enum: keep the pre-flag behaviour, but warn.

        Correct, un-overlapped. The alternative -- refusing to register at all --
        would silently stop supervising the indexer on any Paddle that predates
        the hook location. The warning is the only signal the user gets that the
        flag buys nothing here, so it is part of the contract.
        """
        import paddle.distributed.fleet.meta_parallel.pipeline_parallel as pp

        class _OldLocations:
            FORWARD_END = self._location

        with (
            mock.patch.object(
                pp, "PipelineParallelMicroStepLocations", _OldLocations
            ),
            self.assertLogs(
                indexer_loss_overlap.logger, level="WARNING"
            ) as logs,
        ):
            self.assertTrue(
                indexer_loss_overlap.register_pipeline_hooks(
                    self._fake_pp_model()
                )
            )
        self.assertIn("P2P_ISSUED", "\n".join(logs.output))
        self.assertIn(
            indexer_loss_overlap._forward_end_hook,
            self._registry[self._location],
        )
        self.assertNotIn(
            indexer_loss_overlap._p2p_issued_hook,
            self._registry[self._p2p_location],
        )

    def test_no_micro_step_callbacks_at_all_declines(self):
        """A Paddle without the callback machinery falls back to ``drain_all``."""
        with mock.patch.dict(
            sys.modules,
            {"paddle.distributed.fleet.meta_parallel.pipeline_parallel": None},
        ):
            self.assertFalse(
                indexer_loss_overlap.register_pipeline_hooks(
                    self._fake_pp_model()
                )
            )
        self.assertFalse(indexer_loss_overlap._HOOKS_REGISTERED)

    def _enqueue_recording(self):
        ran = []

        class _RecordingOwner:
            def _run_indexer_loss_branch(self, work):
                ran.append(work)

        work = object.__new__(indexer_loss_overlap._PendingWork)
        work.owner = _RecordingOwner()
        indexer_loss_overlap.enqueue(work)
        return ran

    def test_the_p2p_callback_drains_and_is_counted(self):
        """``on_location`` must reach the queue through Paddle's own dispatch.

        Driven with the keywords the schedule actually passes, so a hook
        signature that could not accept them would fail here.
        """
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            pipeline_parallel_callbacks_,
        )

        indexer_loss_overlap.register_pipeline_hooks(self._fake_pp_model())
        ran = self._enqueue_recording()
        before = indexer_loss_overlap.stats()

        pipeline_parallel_callbacks_.on_location(
            self._p2p_location,
            output_tensor=None,
            step_id=0,
        )

        self.assertEqual(len(ran), 1)
        self.assertEqual(indexer_loss_overlap.pending(), 0)
        self.assertEqual(
            indexer_loss_overlap.stats()["drained_in_hook"],
            before["drained_in_hook"] + 1,
        )
        self.assertEqual(
            indexer_loss_overlap.stats()["drained_in_p2p_window"],
            before["drained_in_p2p_window"] + 1,
        )

    def test_forward_end_drains_until_the_p2p_window_appears(self):
        """A schedule that raises no ``P2P_ISSUED`` must still get its gradient.

        The fallback is armed at first, so the very first micro-step drains at
        ``FORWARD_END`` (correct, un-overlapped); once ``P2P_ISSUED`` has fired
        the fallback must stand down, otherwise it would drain before the p2p is
        issued and the overlap would never happen.
        """
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            pipeline_parallel_callbacks_,
        )

        indexer_loss_overlap.register_pipeline_hooks(self._fake_pp_model())

        armed = self._enqueue_recording()
        pipeline_parallel_callbacks_.on_location(
            self._location, input_tensor=None, output_tensor=None, step_id=0
        )
        self.assertEqual(len(armed), 1, "fallback did not drain while armed")
        self.assertEqual(
            indexer_loss_overlap.stats()["drained_in_p2p_window"],
            0,
            "a FORWARD_END drain must not be counted as an overlapped one",
        )

        pipeline_parallel_callbacks_.on_location(
            self._p2p_location, output_tensor=None, step_id=0
        )
        self.assertTrue(indexer_loss_overlap._P2P_WINDOW_SEEN)

        disarmed = self._enqueue_recording()
        pipeline_parallel_callbacks_.on_location(
            self._location, input_tensor=None, output_tensor=None, step_id=1
        )
        self.assertEqual(
            len(disarmed), 0, "fallback still drained after P2P_ISSUED fired"
        )
        self.assertEqual(indexer_loss_overlap.pending(), 1)

    def test_a_schedule_that_never_raises_p2p_issued_warns_once(self):
        """The runtime half of the "no overlap window" diagnosis.

        A Paddle whose enum lacks ``P2P_ISSUED`` is caught at registration, but a
        schedule that simply never raises the location cannot be detected from
        there. It shows up as a *second* ``FORWARD_END`` drain: had the location
        been raised at all, it would have been raised inside the first micro-step
        and the latch would already have disarmed the fallback.

        Once only -- the fallback runs every micro-step, and a per-step warning
        would bury the training log.
        """
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            pipeline_parallel_callbacks_,
        )

        indexer_loss_overlap.register_pipeline_hooks(self._fake_pp_model())

        def _fallback_micro_step(step_id):
            ran = self._enqueue_recording()
            pipeline_parallel_callbacks_.on_location(
                self._location,
                input_tensor=None,
                output_tensor=None,
                step_id=step_id,
            )
            self.assertEqual(len(ran), 1, "fallback did not drain while armed")

        with self.assertNoLogs(indexer_loss_overlap.logger, level="WARNING"):
            _fallback_micro_step(0)

        with self.assertLogs(
            indexer_loss_overlap.logger, level="WARNING"
        ) as logs:
            _fallback_micro_step(1)
        self.assertIn("P2P_ISSUED", "\n".join(logs.output))
        self.assertTrue(indexer_loss_overlap._FALLBACK_WARNED)

        with self.assertNoLogs(indexer_loss_overlap.logger, level="WARNING"):
            _fallback_micro_step(2)


@unittest.skipUnless(
    _PARENT_REPO_AVAILABLE, "needs the erniebot parent repo for src.utils"
)
class TestConfigCheckRegistration(unittest.TestCase):
    """``dsa_indexer_loss_bwd_p2p_overlap`` must be a registered YAML field.

    ``ernie5/pretrain.py:1175`` runs
    ``config_check._check_no_unregistered_provider_fields`` over the fields the
    user actually configured and hard-errors on any that appear in neither list.
    So without this registration the flag cannot be switched on at all -- setting
    it in the YAML aborts startup.

    It belongs in ``_KNOWN_YAML_PROVIDER_FIELDS`` (YAML) rather than
    ``_MODEL_STRUCTURE_FIELDS`` (JSON) because it changes only *when* the branch
    runs. The gradients and the logged KL are identical either way -- that is
    what :class:`TestOverlapEquivalence` asserts -- so it is not part of the
    architecture and must not end up in a checkpoint's model_config.json, where
    it would make an overlap-on run look like a different model from an
    overlap-off one.
    """

    def test_registered_as_a_yaml_provider_field(self):
        _add_repo_root_to_sys_path()
        from src.utils.config_check import (
            _KNOWN_YAML_PROVIDER_FIELDS,
            _MODEL_STRUCTURE_FIELDS,
        )

        self.assertIn(
            "dsa_indexer_loss_bwd_p2p_overlap", _KNOWN_YAML_PROVIDER_FIELDS
        )
        self.assertNotIn(
            "dsa_indexer_loss_bwd_p2p_overlap",
            set(_MODEL_STRUCTURE_FIELDS),
            "a scheduling switch must not be a model-structure field",
        )


class _RecordingOwner:
    """Stands in for :class:`MQALatentAttention` in the queue-level tests.

    The queue only ever calls ``_run_indexer_loss_branch``, so the plumbing can
    be exercised without a GPU or a real layer -- which is the point: these are
    the paths a numerics test cannot reach (an empty queue, a failing branch, a
    postponed discard).
    """

    def __init__(self, events=None, boom=False):
        self.events = events if events is not None else []
        self._boom = boom

    def _run_indexer_loss_branch(self, work):
        self.events.append(("branch", work))
        if self._boom:
            raise RuntimeError("branch failed")


class _FakeSpan:
    """The ``RecomputeWithoutOutput`` span whose discard gets postponed."""

    def __init__(self, events):
        self.events = events

    def discard_output_and_register_recompute(self, tensor):
        self.events.append(("discard", tensor))


def _fake_work(owner):
    work = object.__new__(indexer_loss_overlap._PendingWork)
    work.owner = owner
    work.on_done = None
    return work


class _QueueIsolation(unittest.TestCase):
    """Base class: keep the module-level queue and counters out of each other."""

    def setUp(self):
        self._saved_queue = list(indexer_loss_overlap._QUEUE)
        indexer_loss_overlap._QUEUE.clear()

    def tearDown(self):
        indexer_loss_overlap._QUEUE.clear()
        indexer_loss_overlap._QUEUE.extend(self._saved_queue)


class TestQueueMechanics(_QueueIsolation):
    """:func:`drain` and its counters, without a GPU or a real layer."""

    def test_enqueue_is_counted_and_pending_reflects_the_queue(self):
        before = indexer_loss_overlap.stats()["enqueued"]
        self.assertEqual(indexer_loss_overlap.pending(), 0)
        indexer_loss_overlap.enqueue(_fake_work(_RecordingOwner()))
        self.assertEqual(indexer_loss_overlap.pending(), 1)
        self.assertEqual(indexer_loss_overlap.stats()["enqueued"], before + 1)

    def test_stats_hands_out_a_copy(self):
        """Callers must not be able to reach in and reset the counters."""
        snapshot = indexer_loss_overlap.stats()
        snapshot["drained"] = -1
        self.assertNotEqual(indexer_loss_overlap.stats()["drained"], -1)

    def test_draining_an_empty_queue_is_a_no_op(self):
        before = indexer_loss_overlap.stats()["drained"]
        self.assertEqual(indexer_loss_overlap.drain(), 0)
        self.assertEqual(indexer_loss_overlap.drain_all(), 0)
        self.assertEqual(indexer_loss_overlap.stats()["drained"], before)

    def test_drain_runs_every_queued_branch_once(self):
        events = []
        owner = _RecordingOwner(events)
        for _ in range(3):
            indexer_loss_overlap.enqueue(_fake_work(owner))
        before = indexer_loss_overlap.stats()["drained"]

        self.assertEqual(indexer_loss_overlap.drain(), 3)

        self.assertEqual(len(events), 3)
        self.assertEqual(indexer_loss_overlap.pending(), 0)
        self.assertEqual(indexer_loss_overlap.stats()["drained"], before + 3)

    def test_a_failing_branch_aborts_instead_of_training_unsupervised(self):
        """Swallowing this would leave the indexer silently un-gradiented."""
        events = []
        work = _fake_work(_RecordingOwner(events, boom=True))
        span = _FakeSpan(events)
        indexer_loss_overlap.enqueue(work)
        self.assertTrue(
            indexer_loss_overlap.defer_discard(work.owner, span, "qkv")
        )

        with self.assertRaises(RuntimeError):
            indexer_loss_overlap.drain()

        # The discard still has to happen: it is what arms the qkv recompute
        # hook, and dropping it would replace this error with a confusing one.
        self.assertEqual([name for name, _ in events], ["branch", "discard"])
        self.assertEqual(indexer_loss_overlap.pending(), 0)


class TestDeferDiscard(_QueueIsolation):
    """``mla_qkv_recompute``: the buffers the branch reads must outlive it.

    ``MultiLatentAttention.forward`` frees query / key / value the instant
    ``core_attention`` returns, and ``query.detach()`` is no protection because
    Paddle's ``detach`` shares the DenseTensor impl. The deferred branch reads
    them later in the same forward, so the discard is moved rather than skipped.
    """

    def test_nothing_pending_means_discard_immediately(self):
        span = _FakeSpan([])
        self.assertFalse(
            indexer_loss_overlap.defer_discard(_RecordingOwner(), span, "qkv")
        )
        self.assertEqual(span.events, [])

    def test_another_layers_pending_work_is_not_hijacked(self):
        """Only the layer that just enqueued may postpone its own discard."""
        indexer_loss_overlap.enqueue(_fake_work(_RecordingOwner()))
        self.assertFalse(
            indexer_loss_overlap.defer_discard(
                _RecordingOwner(), _FakeSpan([]), "qkv"
            )
        )

    def test_a_second_span_for_the_same_layer_is_declined(self):
        """One ``on_done`` slot: the second span discards on the spot."""
        work = _fake_work(_RecordingOwner())
        indexer_loss_overlap.enqueue(work)
        self.assertTrue(
            indexer_loss_overlap.defer_discard(
                work.owner, _FakeSpan([]), "first"
            )
        )
        self.assertFalse(
            indexer_loss_overlap.defer_discard(
                work.owner, _FakeSpan([]), "second"
            )
        )

    def test_the_discard_happens_after_the_branch_and_only_once(self):
        events = []
        work = _fake_work(_RecordingOwner(events))
        span = _FakeSpan(events)
        indexer_loss_overlap.enqueue(work)
        self.assertTrue(
            indexer_loss_overlap.defer_discard(work.owner, span, "qkv")
        )
        self.assertEqual(events, [], "discard was not postponed")

        indexer_loss_overlap.drain()

        self.assertEqual([name for name, _ in events], ["branch", "discard"])
        self.assertEqual(events[1][1], "qkv")
        # Cleared so a stray second drain cannot double-discard.
        self.assertIsNone(work.on_done)


class TestValidateConfig(unittest.TestCase):
    """``validate_config`` raises on wrong, warns on merely useless."""

    @staticmethod
    def _config(**kwargs):
        base = {
            "dsa_indexer_loss_bwd_p2p_overlap": True,
            "dsa_indexer_use_sparse_loss": True,
            "overlap_p2p_comm": True,
            "batch_p2p_comm": False,
            # the one supported shape: pipelined DSv4-hybrid with mqa_dsa
            # layers and a live indexer loss
            "pipeline_model_parallel_size": 4,
            "experimental_attention_variant": "dsv4_hybrid",
            "hybrid_mla_attention": "mqa_dsa",
            "csa_compress_ratios": [-2, 4, -1, -2],
            "dsa_indexer_loss_coeff": 0.01,
        }
        base.update(kwargs)
        return types.SimpleNamespace(**base)

    def test_enabled_reads_the_flag(self):
        self.assertTrue(indexer_loss_overlap.enabled(self._config()))
        self.assertFalse(
            indexer_loss_overlap.enabled(
                self._config(dsa_indexer_loss_bwd_p2p_overlap=False)
            )
        )
        # A provider config that predates the flag must not raise.
        self.assertFalse(indexer_loss_overlap.enabled(types.SimpleNamespace()))

    def test_a_disabled_flag_skips_every_check(self):
        """Off means off: a config that would fail all four gates is fine."""
        indexer_loss_overlap.validate_config(
            types.SimpleNamespace(dsa_indexer_loss_bwd_p2p_overlap=False)
        )
        indexer_loss_overlap.validate_config(types.SimpleNamespace())

    def test_batch_p2p_comm_only_warns(self):
        """The send/recv runs on the calc stream: correct, just no overlap."""
        with self.assertLogs(
            indexer_loss_overlap.logger, level="WARNING"
        ) as caught:
            indexer_loss_overlap.validate_config(
                self._config(batch_p2p_comm=True)
            )
        self.assertIn("nothing for it to overlap with", caught.output[0])

    def test_overlap_p2p_comm_off_only_warns(self):
        with self.assertLogs(indexer_loss_overlap.logger, level="WARNING"):
            indexer_loss_overlap.validate_config(
                self._config(overlap_p2p_comm=False)
            )

    def test_selective_core_attn_recompute_only_warns(self):
        """The enqueue slips into backward, so the window is lost, not the loss.

        ``recompute(self.core_attention, ...)`` is a second wrapper that the
        layer-level ``in_recompute`` marker does not describe, so the layer takes
        the grad-enabled row and enqueues on the replay. Still exactly once, and
        still drained before the optimizer -- only too late to overlap.
        """
        for modules in (
            ["core_attn", "mlp"],
            {"core_attn": 4},
        ):
            with self.subTest(modules=modules):
                with self.assertLogs(
                    indexer_loss_overlap.logger, level="WARNING"
                ) as caught:
                    indexer_loss_overlap.validate_config(
                        self._config(
                            recompute_granularity="selective",
                            recompute_modules=modules,
                        )
                    )
                self.assertIn("nothing overlaps", caught.output[0])

    def test_core_attn_outside_selective_is_not_flagged(self):
        """``recompute_core_attention`` is only set under ``selective``."""
        with self.assertNoLogs(indexer_loss_overlap.logger, level="WARNING"):
            indexer_loss_overlap.validate_config(
                self._config(
                    recompute_granularity="full",
                    recompute_modules=["core_attn"],
                )
            )
        with self.assertNoLogs(indexer_loss_overlap.logger, level="WARNING"):
            indexer_loss_overlap.validate_config(
                self._config(
                    recompute_granularity="selective",
                    recompute_modules=["mlp"],
                )
            )

    def test_the_supported_combination_is_silent(self):
        with self.assertNoLogs(indexer_loss_overlap.logger, level="WARNING"):
            indexer_loss_overlap.validate_config(self._config())

    def test_the_warmup_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sparse"):
            indexer_loss_overlap.validate_config(
                self._config(dsa_indexer_use_sparse_loss=False)
            )

    def test_no_pipeline_is_rejected(self):
        """The overlap window *is* the pipeline p2p; without one it is dead."""
        for pp in (1, 0, None):
            with (
                self.subTest(pipeline_model_parallel_size=pp),
                self.assertRaisesRegex(
                    ValueError, "pipeline_model_parallel_size"
                ),
            ):
                indexer_loss_overlap.validate_config(
                    self._config(pipeline_model_parallel_size=pp)
                )
        # missing altogether (a provider config with no PP field) => default 1
        cfg = self._config()
        del cfg.pipeline_model_parallel_size
        with self.assertRaisesRegex(ValueError, "pipeline"):
            indexer_loss_overlap.validate_config(cfg)

    def test_a_model_without_a_dsa_indexer_is_rejected(self):
        """Only ``-2`` layers under ``mqa_dsa`` build a ``DSAIndexer``."""
        dead = [
            {"experimental_attention_variant": "mla"},
            {"hybrid_mla_attention": "mha"},
            {"hybrid_mla_attention": "mqa_full_causal"},
            # CSA / HCA / window-only: an indexer, but not the DSA one
            {"csa_compress_ratios": [4, -1, 128, 0]},
            {"csa_compress_ratios": None},
            {"csa_compress_ratios": []},
        ]
        for kwargs in dead:
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "DSAIndexer"),
            ):
                indexer_loss_overlap.validate_config(self._config(**kwargs))

    def test_a_dead_loss_coefficient_is_rejected(self):
        """``_needs_indexer_loss`` gates on coeff > 0, so 0 enqueues nothing."""
        for coeff in (0.0, None, -1.0):
            with (
                self.subTest(dsa_indexer_loss_coeff=coeff),
                self.assertRaisesRegex(ValueError, "dsa_indexer_loss_coeff"),
            ):
                indexer_loss_overlap.validate_config(
                    self._config(dsa_indexer_loss_coeff=coeff)
                )

    def test_has_dsa_indexer_coerces_the_ratio_entries(self):
        """``csa_compress_ratios`` may arrive as np.int64 from ``np.load``."""
        cfg = self._config(csa_compress_ratios=[4.0, -2.0])
        self.assertTrue(indexer_loss_overlap._has_dsa_indexer(cfg))


class TestSharedGradientImplementation(unittest.TestCase):
    """Both paths go through :func:`compute_csa_indexer_grads`.

    The equivalence tests above assert the two paths agree numerically; this
    pins the reason they cannot drift -- there is one implementation -- and
    covers the backend guard, which no GPU path reaches.
    """

    def test_both_paths_call_the_same_function(self):
        from paddlefleet.transformer import csa_attention, mqa_latent_attention

        source = mqa_latent_attention.MQALatentAttention
        self.assertTrue(
            hasattr(csa_attention, "compute_csa_indexer_grads"),
            "the shared gradient body is gone",
        )
        self.assertIn(
            "compute_csa_indexer_grads",
            source._run_indexer_loss_branch.__code__.co_names,
            "the deferred path stopped using the shared gradient body",
        )
        self.assertIn(
            "compute_csa_indexer_grads",
            csa_attention.TileLangCSAIndexerLossAutoScaler.backward.__code__.co_names,
            "the inline path stopped using the shared gradient body",
        )

    def test_an_unknown_backend_is_rejected(self):
        from paddlefleet.transformer.csa_attention import (
            compute_csa_indexer_grads,
        )

        with self.assertRaisesRegex(NotImplementedError, "flash"):
            compute_csa_indexer_grads(
                None,
                None,
                None,
                None,
                None,
                None,
                loss_coeff=1.0,
                indexer_backend="flash",
                num_rows=1.0,
            )


class _FakeIndexerBwd:
    """Stands in for ``csa_indexer_bwd``, recording how it was called.

    Both real kernels need SM100, and neither is what the tests below are
    about: what they pin is the arithmetic :func:`compute_csa_indexer_grads`
    does *around* the kernel. Returning float64 for float32 inputs also forces
    the dtype normalisation at the end of the function.
    """

    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, index_q, weights, index_k_comp, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return (
            paddle.zeros(index_q.shape, dtype="float64"),
            paddle.zeros(weights.shape, dtype="float64"),
            paddle.zeros(index_k_comp.shape, dtype="float64"),
        )


class TestSharedGradientBackends(unittest.TestCase):
    """:func:`compute_csa_indexer_grads` around the kernel call.

    Three things happen there that no kernel is involved in: ``loss_mask`` is
    folded into what the kernel reduces, the externally-set main-loss scale is
    forwarded (each backend expects it in a different place), and the kernel's
    dtypes are brought back to the inputs'. The cuDNN half of that is
    unreachable on any machine without the kernel, which is every CI runner.
    """

    _ROWS, _TOPK, _DIM = 4, 3, 8

    def setUp(self):
        from paddlefleet.transformer import csa_attention

        self._csa = csa_attention
        scaler = csa_attention.DSAIndexerLossAutoScaler
        saved = scaler._main_loss_backward_scale
        self.addCleanup(setattr, scaler, "_main_loss_backward_scale", saved)

    def _scale(self, value):
        self._csa.DSAIndexerLossAutoScaler._main_loss_backward_scale = value

    def _call(self, backend, *, loss_mask=None, num_rows=None):
        """Run one gradient computation; returns the stand-in and the grads."""
        projections = [
            paddle.zeros([self._ROWS, self._DIM], dtype="float32")
            for _ in range(3)
        ]
        target = paddle.full(
            [1, self._ROWS, self._TOPK], 1.0 / self._TOPK, dtype="float32"
        )
        fake = _FakeIndexerBwd()
        ops = "cudnn_ops" if backend == "cudnn" else "tilelang_ops"
        with mock.patch(f"paddlefleet.{ops}.csa_indexer_bwd", fake):
            grads = self._csa.compute_csa_indexer_grads(
                *projections,
                target,
                paddle.zeros_like(target),
                paddle.zeros(target.shape, dtype="int32"),
                loss_coeff=0.5,
                indexer_backend=backend,
                num_rows=num_rows,
                loss_mask=loss_mask,
            )
        return fake, grads

    def test_grads_come_back_in_the_dtypes_of_the_projections(self):
        """The kernels do not promise the input dtype; the caller assumes it."""
        for backend in ("cudnn", "tilelang"):
            with self.subTest(backend=backend):
                _, grads = self._call(backend)
                for grad in grads:
                    self.assertEqual(grad.dtype, paddle.float32)

    def test_tilelang_gets_the_score_gradient_already_scaled(self):
        """``num_rows=None`` means the kernel's own mean over every row."""
        fake, _ = self._call("tilelang")
        grad_scores = fake.args[-1]
        # ``topk_probs - target`` is ``-target`` here, times
        # ``loss_coeff / rows``.
        expected = -0.5 / (1 * self._ROWS) / self._TOPK
        np.testing.assert_allclose(
            grad_scores.numpy(), np.full(grad_scores.shape, expected), rtol=1e-6
        )

    def test_a_row_mask_zeroes_that_rows_score_gradient(self):
        mask = paddle.to_tensor([1.0, 0.0, 1.0, 0.0], dtype="float32")
        fake, _ = self._call("tilelang", loss_mask=mask, num_rows=2.0)
        grad_scores = fake.args[-1].numpy()
        self.assertTrue((grad_scores[0, 1] == 0).all())
        self.assertTrue((grad_scores[0, 3] == 0).all())
        self.assertTrue((grad_scores[0, 0] != 0).all())

    def test_a_row_mask_reaches_cudnn_through_its_two_reduction_inputs(self):
        """cuDNN reduces ``target`` / ``topk_probs`` itself, so mask them."""
        mask = paddle.to_tensor([1.0, 0.0, 1.0, 0.0], dtype="float32")
        fake, _ = self._call("cudnn", loss_mask=mask, num_rows=2.0)
        bwd_target = fake.args[0].numpy()
        self.assertTrue((bwd_target[0, 1] == 0).all())
        self.assertTrue((bwd_target[0, 0] != 0).all())
        # The kernel divides by every row, so the coefficient carries the
        # correction from that count to the valid-row count.
        self.assertAlmostEqual(
            fake.kwargs["loss_coeff"], 0.5 * self._ROWS / 2.0, places=6
        )

    def test_the_main_loss_scale_reaches_each_backend_its_own_way(self):
        """Set by ``DSAIndexerLossAutoScaler`` outside this function's reach."""
        self._scale(None)
        self.assertIsNone(self._call("cudnn")[0].kwargs["grad_loss"])

        self._scale(4.0)
        as_tensor = self._call("cudnn")[0].kwargs["grad_loss"]
        self.assertEqual(float(as_tensor), 4.0)

        given = paddle.to_tensor(4.0, dtype="float32")
        self._scale(given)
        self.assertIs(self._call("cudnn")[0].kwargs["grad_loss"], given)

    def test_tilelang_folds_the_main_loss_scale_into_the_score_gradient(self):
        """It has no ``grad_loss`` argument, so the caller applies it."""
        unscaled = self._call("tilelang")[0].args[-1].numpy()
        self._scale(4.0)
        scaled = self._call("tilelang")[0].args[-1].numpy()
        np.testing.assert_allclose(scaled, unscaled * 4.0, rtol=1e-6)


class _PredicateOwner:
    """The three attributes ``_needs_indexer_loss`` reads."""

    def __init__(self, training=True, coeff=0.01, overlap=False):
        self.training = training
        self.indexer_loss_coeff = coeff
        self.indexer_loss_overlap = overlap


class TestNeedsIndexerLoss(unittest.TestCase):
    """Which forward pass owns the loss branch.

    This predicate is what makes the loss happen exactly once per
    ``(layer, micro-batch)`` under recompute, and the overlap inverts it: the
    deferred branch has to be queued by the pass that runs while the pipeline is
    in its forward phase, which is the *other* one. The whole table is checked
    here because a GPU fixture can only ever demonstrate one row at a time.
    """

    def _needs(self, owner, in_recompute=False, grad=True):
        from paddlefleet.transformer.mqa_latent_attention import (
            MQALatentAttention,
        )

        with paddle.set_grad_enabled(grad):
            return MQALatentAttention._needs_indexer_loss(owner, in_recompute)

    def test_eval_and_a_zero_coefficient_switch_the_branch_off(self):
        self.assertFalse(self._needs(_PredicateOwner(training=False)))
        self.assertFalse(self._needs(_PredicateOwner(coeff=0.0)))

    def test_the_inline_path_wants_the_grad_enabled_pass(self):
        owner = _PredicateOwner()
        for in_recompute in (False, True):
            with self.subTest(in_recompute=in_recompute):
                self.assertTrue(self._needs(owner, in_recompute, grad=True))
                self.assertFalse(self._needs(owner, in_recompute, grad=False))

    def test_a_wrapped_layer_queues_on_the_no_grad_pass(self):
        """Its grad-enabled pass is the replay, which runs in backward."""
        owner = _PredicateOwner(overlap=True)
        self.assertTrue(self._needs(owner, in_recompute=True, grad=False))
        self.assertFalse(self._needs(owner, in_recompute=True, grad=True))

    def test_an_unwrapped_layer_queues_on_its_only_pass(self):
        """Why the flag does not depend on ``recompute_granularity``."""
        owner = _PredicateOwner(overlap=True)
        self.assertTrue(self._needs(owner, in_recompute=False, grad=True))
        self.assertFalse(self._needs(owner, in_recompute=False, grad=False))


class _BranchOwner:
    """The layer attributes ``_run_indexer_loss_branch`` reads.

    The target definition and the row mask are the layer's, so they are stubbed
    rather than reimplemented: what is under test is the reduction the branch
    builds on top of them and the gradient hand-off that follows.
    """

    def __init__(self, loss_mask=None, valid_rows=None):
        self.indexer_loss_coeff = 0.5
        self.cp_size = 2
        self.indexer_backend = "tilelang"
        self.layer_number = 1
        self.config = types.SimpleNamespace(num_hidden_layers=2)
        self._loss_mask = loss_mask
        self._valid_rows = valid_rows

    def _attn_target(self, query, kv, topk_indices, lse_indexer):
        return paddle.full(
            topk_indices.shape, 1.0 / topk_indices.shape[-1], dtype="float32"
        )

    def _indexer_loss_mask(self, input_ids, batch, seqlen):
        return self._loss_mask, self._valid_rows


class TestDeferredBranchArithmetic(unittest.TestCase):
    """``_run_indexer_loss_branch`` without the sparse forward in front of it.

    The branch is reachable on its own -- that is the property the whole flag
    rests on -- so it can be driven from a hand-built work item on any machine.
    What the equivalence tests cannot show, because they need the kernels, is
    where the two reductions differ and what happens to a frozen indexer.
    """

    _ROWS, _TOPK, _DIM = 4, 3, 8
    _LAYER = 1
    # Deliberately not uniform: a flat score row is exactly the target, and a
    # zero KL would hide which of the two reductions ran.
    _SCORES = [0.0, 1.0, 2.0]

    def setUp(self):
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.addCleanup(DSAIndexerLossLoggingHelper.tracker.clear)

    def _kl_per_row(self):
        """The reference value of one row of ``kl_per_pos``."""
        exp = np.exp(self._SCORES)
        probs = exp / exp.sum()
        target = 1.0 / self._TOPK
        return float(
            (target * (np.log(target + 1e-10) - np.log(probs + 1e-10))).sum()
        )

    def _work(self, owner, frozen=False):
        """A work item whose projections carry a one-op autograd subgraph."""
        leaf = paddle.zeros([self._ROWS, self._DIM], dtype="float32")
        leaf.stop_gradient = frozen
        projections = [leaf * 2.0 for _ in range(3)]
        scores = paddle.to_tensor(
            [[self._SCORES] * self._ROWS], dtype="float32"
        )
        return leaf, indexer_loss_overlap._PendingWork(
            owner=owner,
            query=None,
            kv=None,
            lse_indexer=None,
            topk_indices=paddle.zeros(scores.shape, dtype="int32"),
            topk_scores=scores,
            index_q=projections[0],
            weights=projections[1],
            index_k=projections[2],
            input_ids=None,
            batch=1,
            seqlen=self._ROWS,
        )

    def _run(self, owner, frozen=False):
        """Drive the branch with a stand-in for the gradient kernel.

        The gradient it hands back is all-ones, so whatever reaches the leaf is
        the projection subgraph's own contribution and nothing else.
        """
        from paddlefleet.transformer.mqa_latent_attention import (
            MQALatentAttention,
        )

        leaf, work = self._work(owner, frozen=frozen)
        calls = []

        def fake_grads(index_q, weights, index_k, *args, **kwargs):
            calls.append(kwargs)
            return tuple(
                paddle.ones(t.shape, dtype=t.dtype)
                for t in (index_q, weights, index_k)
            )

        with mock.patch(
            "paddlefleet.transformer.csa_attention.compute_csa_indexer_grads",
            fake_grads,
        ):
            MQALatentAttention._run_indexer_loss_branch(owner, work)
        return leaf, calls[0]

    def _logged(self):
        return float(_logged_loss()[self._LAYER - 1])

    def test_without_a_row_mask_the_loss_is_the_row_mean_over_cp(self):
        """``cp_size`` divides here because each rank means over its own rows."""
        owner = _BranchOwner()
        self._run(owner)
        expected = self._kl_per_row() * owner.indexer_loss_coeff / owner.cp_size
        self.assertAlmostEqual(self._logged(), expected, places=5)

    def test_a_row_mask_replaces_the_mean_with_a_global_row_count(self):
        """The denominator is the count, not this rank's rows, so no ``cp_size``."""
        owner = _BranchOwner(
            loss_mask=paddle.to_tensor([[1.0, 0.0, 1.0, 0.0]]),
            valid_rows=2.0,
        )
        self._run(owner)
        expected = self._kl_per_row() * 2 / 2.0 * owner.indexer_loss_coeff
        self.assertAlmostEqual(self._logged(), expected, places=5)

    def test_the_mask_and_its_row_count_are_forwarded_to_the_kernel(self):
        """The loss and the gradient must reduce over the same rows."""
        mask = paddle.to_tensor([[1.0, 0.0, 1.0, 0.0]])
        owner = _BranchOwner(loss_mask=mask, valid_rows=2.0)
        _, kwargs = self._run(owner)
        self.assertIs(kwargs["loss_mask"], mask)
        self.assertEqual(kwargs["num_rows"], 2.0)
        self.assertEqual(kwargs["loss_coeff"], owner.indexer_loss_coeff)

    def test_the_gradient_reaches_the_indexer_projections(self):
        """What the deferral is for: the subgraph is walked here, not in backward."""
        leaf, _ = self._run(_BranchOwner())
        self.assertIsNotNone(leaf.grad)
        # Three projections, each ``leaf * 2``, each handed a gradient of ones.
        np.testing.assert_allclose(
            leaf.grad.numpy(), np.full(leaf.shape, 6.0), rtol=1e-6
        )

    def test_a_frozen_indexer_still_produces_the_loss(self):
        """No subgraph to walk, and ``paddle.autograd.backward`` would raise."""
        leaf, _ = self._run(_BranchOwner(), frozen=True)
        self.assertIsNone(leaf.grad)
        self.assertGreater(self._logged(), 0.0)


class TestForwardEndFallbackLatch(_QueueIsolation):
    """``_forward_end_hook`` on the micro-steps that have nothing to do.

    Every non-pipeline forward raises ``FORWARD_END``, and most of those queued
    nothing -- eval, a stage with no DSA layer, the flag off. The hook has to
    stay silent there: counting them would make the second drain, which is what
    the no-overlap warning keys on, mean nothing.
    """

    def setUp(self):
        super().setUp()
        for name in (
            "_P2P_WINDOW_SEEN",
            "_FORWARD_END_DRAINS",
            "_FALLBACK_WARNED",
        ):
            self.addCleanup(
                setattr,
                indexer_loss_overlap,
                name,
                getattr(indexer_loss_overlap, name),
            )
        indexer_loss_overlap._P2P_WINDOW_SEEN = False
        indexer_loss_overlap._FORWARD_END_DRAINS = 0

    def test_an_empty_queue_does_not_count_as_a_fallback_drain(self):
        before = indexer_loss_overlap.stats()["drained_in_hook"]
        indexer_loss_overlap._forward_end_hook()
        self.assertEqual(indexer_loss_overlap._FORWARD_END_DRAINS, 0)
        self.assertEqual(
            indexer_loss_overlap.stats()["drained_in_hook"], before
        )

    def test_work_left_over_is_drained_and_counted(self):
        indexer_loss_overlap.enqueue(_fake_work(_RecordingOwner()))
        before = indexer_loss_overlap.stats()["drained_in_hook"]
        indexer_loss_overlap._forward_end_hook()
        self.assertEqual(indexer_loss_overlap.pending(), 0)
        self.assertEqual(indexer_loss_overlap._FORWARD_END_DRAINS, 1)
        self.assertEqual(
            indexer_loss_overlap.stats()["drained_in_hook"], before + 1
        )

    def test_the_latch_hands_the_window_over_once_it_has_been_seen(self):
        """Draining here after that would be draining before the p2p issue."""
        indexer_loss_overlap._P2P_WINDOW_SEEN = True
        indexer_loss_overlap.enqueue(_fake_work(_RecordingOwner()))
        indexer_loss_overlap._forward_end_hook()
        self.assertEqual(indexer_loss_overlap.pending(), 1)


@_GPU
class TestOverlapBitwiseAlignment(unittest.TestCase):
    """Flipping the flag must not move a bit that the kernels can keep still.

    :class:`TestOverlapEquivalence` checks the two paths with ``allclose``, which
    is the wrong instrument for this flag: the branch is *moved*, not
    reformulated, so nothing about the arithmetic changes and a 1e-5 tolerance
    would happily accept a real drift (a scale applied twice, one gradient
    missing its last contribution, an extra bf16 round trip). This is the strict
    sentinel that runs on every commit.

    Two of the gradient families cannot be held to bit equality, and that is a
    property of the kernel rather than of the flag: ``csa_indexer_bwd`` reduces
    ``d_index_k`` with fp32 ``atomicAdd``, so everything downstream of it --
    ``wk`` (which produces ``index_k``) and the ``k_norm`` scale/bias applied to
    it -- lands in a different order from one launch to the next. Re-running the
    *inline* path against itself already perturbs them, so those two go through
    a budget instead; ``wq_b`` and ``weights_proj`` (fed by ``d_index_q`` /
    ``d_weights``, no atomics) and the KL written to the loss tracker are
    compared bit for bit.

    The budget counts *how many* elements differ, not by how much, because that
    is what separates the two populations. The atomics touch a small fraction of
    a large tensor and leave the small ones alone, whereas a real drift moves
    roughly half of *every* tensor at once. Magnitude cannot discriminate:
    perturbing ``dsa_indexer_loss_coeff`` by a fraction of a percent stays well
    inside the house ULP bound while changing the gradient everywhere, so a
    tolerance loose enough to absorb the reduction noise also absorbs the bug.
    """

    # Gradients that flow through ``d_index_k``, i.e. through the fp32
    # ``atomicAdd`` reduction; see the class docstring.
    _ATOMIC_REDUCED = ("wk.", "k_norm.")

    # Fraction of elements allowed to differ in the tensors above: several times
    # the worst reduction noise measured on this fixture, and several times below
    # the weakest signal a real drift produces.
    _NOISE_BUDGET = 0.10

    def setUp(self):
        indexer_loss_overlap.drain_all()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def tearDown(self):
        indexer_loss_overlap.drain_all()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def _forward(self, module, seed, in_recompute):
        """One micro-batch. ``in_recompute=None`` means the inline path."""
        query, key, w_v, x, qr = _make_inputs(
            _SEQLEN, seed=seed, with_hidden=True
        )
        inline = in_recompute is None
        if inline:
            # the inline branch hangs off the layer's autograd graph, so the
            # inputs have to be differentiable for `backward()` to reach it
            for tensor in (query, key, x, qr):
                tensor.stop_gradient = False
        module.train()
        ctx = paddle.no_grad() if in_recompute else contextlib.nullcontext()
        extra = {} if inline else {"in_recompute": in_recompute}
        with ctx:
            out = module(
                query,
                key,
                None,
                None,
                _row_end([_SEQLEN], _SEQLEN),
                v_b_proj_weight=w_v,
                x=x,
                qr=qr,
                **extra,
            )
        if inline:
            out.cast("float32").sum().backward()
        else:
            self.assertEqual(indexer_loss_overlap.drain(), 1)

    def _run(self, in_recompute, seeds, state):
        """``(grads, logged loss)`` after ``len(seeds)`` accumulated steps."""
        module = _build_module(
            _sparse_config(overlap=in_recompute is not None), bf16=True
        )
        module.set_state_dict(state)
        DSAIndexerLossLoggingHelper.tracker.clear()
        for seed in seeds:
            self._forward(module, seed, in_recompute)
        return _indexer_grads(module), _logged_loss()

    def _assert_within_noise_budget(self, actual, reference, name):
        """Only as many elements may move as the fp32 atomics can explain."""
        differing = int((actual != reference).sum())
        allowed = max(8, int(self._NOISE_BUDGET * reference.size))
        if differing <= allowed:
            return
        a = actual.astype("float64")
        b = reference.astype("float64")
        relative = float(
            np.linalg.norm((a - b).ravel())
            / (np.linalg.norm(b.ravel()) + 1e-45)
        )
        self.fail(
            f"{name}: {differing}/{reference.size} elements "
            f"({differing / reference.size:.2%}) differ between the inline and "
            f"the deferred path, at most {allowed} may; relative L2 "
            f"{relative:.3e}, max abs diff {float(np.abs(a - b).max()):.3e}. "
            "Reordering the fp32 atomicAdd in the dK reduction only ever "
            "touches a small fraction of the elements -- a population this "
            "large means the two paths no longer compute the same thing."
        )

    def _assert_aligned(self, seeds):
        state = _build_module(
            _sparse_config(overlap=False), bf16=True
        ).state_dict()

        base_grads, base_loss = self._run(None, seeds, state)
        self.assertTrue(base_grads, "inline path produced no indexer grads")
        self.assertIsNotNone(base_loss, "inline path logged no KL loss")
        strict = [
            name
            for name in base_grads
            if not name.startswith(self._ATOMIC_REDUCED)
        ]
        # If the indexer parameters are ever renamed, `_ATOMIC_REDUCED` would
        # start matching everything (or nothing) and this test would quietly
        # stop asserting anything. Pin that it still splits the set.
        self.assertTrue(
            strict, "no gradient left to compare bit for bit -- names changed?"
        )
        self.assertNotEqual(
            len(strict),
            len(base_grads),
            "the atomicAdd-reduced gradients disappeared -- if the kernel "
            "stopped using fp32 atomics, drop _ATOMIC_REDUCED and make every "
            "gradient a bitwise comparison",
        )

        for in_recompute in (True, False):
            with self.subTest(in_recompute=in_recompute):
                grads, loss = self._run(in_recompute, seeds, state)
                self.assertEqual(
                    sorted(grads),
                    sorted(base_grads),
                    "the deferred path touched a different set of indexer "
                    "weights",
                )
                for name, reference in base_grads.items():
                    if name in strict:
                        self.assertTrue(
                            np.array_equal(reference, grads[name]),
                            f"{name}: no kernel between the two paths is "
                            "order-dependent, so this gradient has to be bit "
                            "identical; max abs diff "
                            f"{float(np.abs(reference - grads[name]).max()):.3e}",
                        )
                    else:
                        self._assert_within_noise_budget(
                            grads[name], reference, name
                        )
                self.assertTrue(
                    np.array_equal(base_loss, loss),
                    "the KL written to the loss tracker differs between the "
                    "inline and the deferred path",
                )

    def test_one_micro_batch_is_bitwise_aligned(self):
        """The single-micro-batch case, both spellings of the deferred path."""
        self._assert_aligned(seeds=(0,))

    def test_accumulation_across_micro_batches_is_bitwise_aligned(self):
        """Two micro-batches: the accumulation order changes, the sum must not.

        With the flag off, micro-batch *i*'s indexer gradient is accumulated
        during *i*'s backward; with it on, during *i*'s forward. The visitation
        order over micro-batches is the same either way, so ``param.grad`` has
        to end up holding the same bits -- this is the property that a real
        pipeline run depends on and that a single-micro-batch test cannot see.
        """
        self._assert_aligned(seeds=(0, 1))


if __name__ == "__main__":
    unittest.main()
