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
"""Run the DSA indexer-loss branch inside the pipeline forward send/recv.

Gated by ``TransformerConfig.dsa_indexer_loss_bwd_p2p_overlap``; off by default,
and when off nothing in this module is reached.

Why the branch can move
-----------------------
The indexer loss is a *leaf subgraph* of the training graph:

* ``MQALatentAttention._indexer_projections`` detaches ``x`` / ``qr``, so no
  gradient reaches the backbone;
* ``TileLangCSAIndexerLossAutoScaler`` is an identity on the layer output, so
  the attention backward never waits on it;
* ``csa_indexer_bwd`` never reads ``grad_output`` -- every input it needs exists
  by the end of the layer's forward.

Its only effect is a gradient on the ``DSAIndexer`` weights, so both the forward
and the backward of the branch may run during the forward pass, at any point
after the layer produced its inputs.

Where it moves to
-----------------
The ``-2`` layer enqueues its inputs (:func:`enqueue`) on whichever forward pass
belongs to the pipeline's forward phase -- the no-grad one if the layer body is
recompute-wrapped, the only one if it is not; the choice lives in
``MQALatentAttention._needs_indexer_loss``. :func:`drain` then computes the loss
and the indexer gradients from Paddle's ``P2P_ISSUED`` callback, which the
schedule raises after issuing the micro-step's ``send_forward`` /
``send_forward_recv_forward`` and before consuming its wait handles.

Why that exact point and not ``FORWARD_END``
--------------------------------------------
``isend`` / ``irecv`` guarantee that the tensor being sent is finished by
recording an event on the *calculation* stream at issue time and having the
pipeline group's NCCL stream wait on it. Anything queued before the issue is
inside that event's reach, so draining at ``FORWARD_END`` makes the send wait for
the branch rather than run alongside it. Raising the hook after the issue puts
the branch's kernels behind the event record instead::

    calc  [chunk fwd][meta][EventRecord ev]      [drain branch][WaitEvent comm]
    comm                   [WaitEvent ev][SendRecv ......................]

which is the "issue, compute, then wait" shape the schedule already uses for
``WeightGradStore.pop()``.

Two conditions decide whether that is real overlap:

* **the p2p must not be on the compute stream.** ``overlap_p2p_comm=True``
  forces ``use_batch_p2p_comm`` off and thereby selects ``_p2p_ops`` -- ``isend``
  / ``irecv`` on the pipeline group's own stream -- over ``_batched_p2p_ops``,
  which runs on the calculation stream and leaves nothing to overlap with. The
  hook is raised either way, so only the speedup depends on this, never
  correctness.
* **the callback must fire on the schedule actually in use.** Only the schedules
  taught to defer their wait raise ``P2P_ISSUED``; for any other one
  :func:`_forward_end_hook` stays armed, so no schedule loses the loss. The latch
  flips on the first ``P2P_ISSUED``, which costs one un-overlapped micro-step per
  process and needs no knowledge of the schedule class. Both ways of ending up
  without a window are reported: a Paddle whose enum lacks the location warns at
  registration, a schedule that never raises it warns on the second fallback
  drain (:func:`_warn_if_fallback_is_permanent`).

Where the window does not exist
-------------------------------
In the steady 1F1B loop of ``VPPFhenBInBalancedMemory`` the last pipeline stage
has no window: ``send_forward`` and ``recv_backward`` are both no-ops there, so a
``-2`` layer living on it runs its branch earlier but overlaps nothing. That is a
property of the schedule, not something this module can fix. Without pipeline
parallelism there is no send/recv at all, which is why :func:`validate_config`
rejects ``pipeline_model_parallel_size == 1``.

Ordering guarantees
-------------------
The queue is drained inside the *same* micro-step that filled it, with no other
forward in between, so it is never more than one layer deep even when a rank owns
several ``-2`` layers -- those sit in different virtual chunks, i.e. different
micro-steps. ``query``, the largest tensor in the branch, is therefore released
shortly after it is produced. Nothing is carried across steps, and
:func:`drain_all` closes the queue before the optimizer as a net for the
non-pipeline case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import paddle

logger = logging.getLogger(__name__)


@dataclass
class _PendingWork:
    """One ``-2`` layer's forward pass, waiting for its whole loss branch.

    ``owner`` is the :class:`MQALatentAttention` instance. The target definition
    (which columns, which normalizer) and the row mask live there and must stay
    there -- duplicating them here is how the two paths would silently drift
    apart.

    ``index_q`` / ``weights`` / ``index_k`` carry the indexer projection's
    autograd subgraph, built by ``_indexer_projections(grad_enabled=True)``.
    Holding a reference is what keeps that subgraph alive until :func:`drain`
    drives it; because ``x`` / ``qr`` were detached, driving it cannot reach the
    backbone.

    ``on_done`` is set by :func:`defer_discard` when the layer's query/key came
    out of a ``RecomputeWithoutOutput`` span whose buffers must not be freed
    until this branch has read them. :func:`drain` calls it right after the
    branch, i.e. still inside the same micro-step.
    """

    owner: Any
    query: paddle.Tensor
    kv: paddle.Tensor
    lse_indexer: paddle.Tensor | None
    topk_indices: paddle.Tensor
    topk_scores: paddle.Tensor
    index_q: paddle.Tensor
    weights: paddle.Tensor
    index_k: paddle.Tensor
    input_ids: paddle.Tensor | None
    batch: int
    seqlen: int
    on_done: Any = None


_QUEUE: list[_PendingWork] = []
_HOOKS_REGISTERED = False
# Flips on the first ``P2P_ISSUED``. Until then ``_forward_end_hook`` keeps
# draining, so a schedule that never raises the new location still computes the
# loss; afterwards ``FORWARD_END`` stands down and the drain happens in the p2p
# window instead. See the "Why that exact point" section above.
_P2P_WINDOW_SEEN = False
# How many times ``FORWARD_END`` has had to do the work. Only the *second* one
# is diagnostic: see :func:`_warn_if_fallback_is_permanent`.
_FORWARD_END_DRAINS = 0
_FALLBACK_WARNED = False
_STATS = {
    "enqueued": 0,
    "drained": 0,
    "drained_in_hook": 0,
    "drained_in_p2p_window": 0,
}


def enabled(config) -> bool:
    """Whether the ``-2`` layers should defer their loss branch."""
    return bool(getattr(config, "dsa_indexer_loss_bwd_p2p_overlap", False))


def pending() -> int:
    """Number of queued layers. Zero everywhere outside a forward pass."""
    return len(_QUEUE)


def stats() -> dict[str, int]:
    """Cumulative counters, for tests and for a one-shot startup log."""
    return dict(_STATS)


def enqueue(work: _PendingWork) -> None:
    _QUEUE.append(work)
    _STATS["enqueued"] += 1


def defer_discard(owner, span, hook_tensor) -> bool:
    """Postpone a ``RecomputeWithoutOutput`` output discard past :func:`drain`.

    With ``mla_qkv_recompute``, ``MultiLatentAttention.forward`` frees query /
    key / value the instant ``core_attention`` returns and only restores them in
    backward. The deferred branch reads query and kv *later in the same forward*,
    and the ``query.detach()`` in the queue is no protection: Paddle's ``detach``
    shares the underlying ``DenseTensor``, so clearing the original nulls the view
    too and the branch dies on a missing holder.

    The discard is therefore moved, not skipped -- skipping would leave
    ``RecomputeWithoutOutputFunction.backward`` without the inputs/outputs that
    only this hook registers. :func:`drain` runs it in a ``finally`` right after
    the branch, still in the same micro-step and far ahead of any backward, so the
    recompute saving is kept in full; the cost is that one layer's ``query``
    survives until the drain.

    Returns ``False`` when this layer has nothing pending -- the caller then
    discards immediately, exactly as before.
    """
    if not _QUEUE:
        return False
    work = _QUEUE[-1]
    if work.owner is not owner or work.on_done is not None:
        return False
    work.on_done = lambda: span.discard_output_and_register_recompute(
        hook_tensor
    )
    return True


def drain() -> int:
    """Run every queued loss branch. Returns how many were run.

    The actual work lives on the layer
    (``MQALatentAttention._run_indexer_loss_branch``) so that the target
    definition, the KL reduction and the constants stay in one file and cannot
    drift from the inline path.

    Failures are not swallowed: a wrong gradient here is silent, so it is better
    to abort the run than to keep training an indexer that is no longer being
    supervised. The ``on_done`` callback still runs on that path, because it is
    what arms the qkv recompute hook and dropping it would replace a clear error
    with a confusing one in backward.
    """
    if not _QUEUE:
        return 0
    ran = 0
    while _QUEUE:
        work = _QUEUE.pop()
        try:
            work.owner._run_indexer_loss_branch(work)
        finally:
            if work.on_done is not None:
                work.on_done()
                work.on_done = None
        # Drop the references before the next item so the peak live set is one
        # layer's inputs, not the whole queue's.
        del work
        ran += 1
    _STATS["drained"] += ran
    return ran


def _p2p_issued_hook(**kwargs) -> None:
    """``P2P_ISSUED`` callback. Signature is ``hook(**kwargs)``.

    Kernels queued here sit *behind* the calc-stream event that gates the NCCL
    send/recv, so they can run concurrently with it. ``output_tensor`` /
    ``step_id`` are passed but unused -- the queue carries everything.

    Firing this location disarms :func:`_forward_end_hook` for the rest of the
    process, and the latch is set even on an empty queue: what it records is that
    this schedule provides the window, not that this micro-step had work. Were it
    conditional on work, a chunk with no ``-2`` layer would leave the fallback
    armed and it would keep taking the work away from this location.
    """
    global _P2P_WINDOW_SEEN
    _P2P_WINDOW_SEEN = True
    ran = drain()
    if ran:
        _STATS["drained_in_hook"] += ran
        _STATS["drained_in_p2p_window"] += ran


def _forward_end_hook(**kwargs) -> None:
    """``FORWARD_END`` fallback. Signature is ``hook(**kwargs)``.

    Drains only while ``_P2P_WINDOW_SEEN`` is unset, i.e. exactly in these
    situations:

    * **the first micro-step of the process, always.** ``FORWARD_END`` precedes
      the p2p issue, so it fires before the first ``P2P_ISSUED`` can flip the
      latch. Draining here is correct but not overlapped; the cost is one
      micro-step, once.
    * **the Paddle in use has no ``P2P_ISSUED``**, so
      :func:`register_pipeline_hooks` could not register the other hook at all.
    * **the schedule in use never raises it.** Only the schedules taught to defer
      their forward-send wait do; on any other one the location exists but stays
      silent.
    * **forward-only passes**, which raise ``FORWARD_END`` but not
      ``P2P_ISSUED``. Nothing is ever queued there -- ``_needs_indexer_loss``
      requires ``self.training`` -- so this is a no-op in practice.

    The last three would lose the indexer gradient outright without this hook,
    which is why it stays registered: draining here yields no overlap, but no
    overlap is much cheaper than an unsupervised indexer.

    Whether ``P2P_ISSUED`` will ever arrive cannot be decided in advance -- the
    enum member existing says nothing about the schedule raising it -- so it is
    decided by observation instead, and only the positive case is observable.
    That asymmetry is why the safe location is armed by default and stands down
    on evidence, rather than the other way round.

    ``**kwargs`` absorbs the keywords Paddle passes (``input_tensor`` /
    ``output_tensor`` / ``step_id``) and any it adds later.
    """
    global _FORWARD_END_DRAINS
    if _P2P_WINDOW_SEEN:
        return
    ran = drain()
    if not ran:
        return
    _STATS["drained_in_hook"] += ran
    _FORWARD_END_DRAINS += 1
    _warn_if_fallback_is_permanent()


def _warn_if_fallback_is_permanent() -> None:
    """Warn once when ``P2P_ISSUED`` is evidently never going to fire.

    The first ``FORWARD_END`` drain is expected of every run, so it says nothing.
    A *second* one does: had the location been raised at all, it would have been
    raised within the first micro-step and the latch would have disarmed this
    hook. So the flag is on, the loss is being computed, and none of it is
    overlapping -- worth a warning, because that is the entire point of the flag.
    Emitted once; per-micro-step logging would drown the training log.
    """
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED or _FORWARD_END_DRAINS < 2:
        return
    _FALLBACK_WARNED = True
    logger.warning(
        "indexer_loss_overlap: dsa_indexer_loss_bwd_p2p_overlap=True but this "
        "run never reaches the P2P_ISSUED hook location, so the indexer loss "
        "branch is draining at FORWARD_END -- before the pipeline issues its "
        "forward send/recv, i.e. with no overlap at all. Gradients and loss are "
        "unaffected. Either this Paddle predates the location or the pipeline "
        "schedule in use does not raise it (only the schedules that defer their "
        "forward-send wait do); switch schedule, upgrade Paddle, or turn the "
        "flag off to drop the deferral machinery."
    )


def drain_all() -> int:
    """Safety net: close the queue outside the callback.

    Needed wherever no ``FORWARD_END`` will fire before the gradients are read:
    ``pipeline_model_parallel_size == 1`` never enters ``_forward_step`` at all,
    and a non-pipeline eval loop calls the layer directly. Must be called once
    per step before the optimizer -- specifically before
    ``hybrid_parallel_scale_param_grad``, so a deferred contribution gets the
    same CP scaling as the inline path.
    """
    return drain()


def _has_dsa_indexer(config) -> bool:
    """Whether any layer of this model actually builds a ``DSAIndexer``.

    The same predicate ``TransformerConfig.__post_init__`` uses via
    ``has_mqa_indexer``, mirroring what ``gpt_layer_specs.py`` decides: only the
    ``csa_compress_ratios == -2`` layers of a ``dsv4_hybrid`` model under
    ``hybrid_mla_attention='mqa_dsa'``. The CSA indexer does not count -- its loss
    lives in ``csa_attention.py`` and has no deferred path.
    """
    if getattr(config, "experimental_attention_variant", None) != "dsv4_hybrid":
        return False
    if getattr(config, "hybrid_mla_attention", None) != "mqa_dsa":
        return False
    ratios = getattr(config, "csa_compress_ratios", None) or []
    return any(int(r) == -2 for r in ratios)


def _recomputes_core_attn(config) -> bool:
    """Whether ``Attention`` will wrap ``core_attention`` in its own recompute.

    Mirrors the ``selective`` branch of ``attention.py``, minus the per-layer
    ``first_n`` / ``block`` narrowing: any layer being wrapped is enough for the
    warning, and this predicate has no layer number.  ``in`` covers both spellings
    of ``recompute_modules`` -- the list and the ``{module: num_layers}`` dict.
    """
    if getattr(config, "recompute_granularity", None) != "selective":
        return False
    modules = getattr(config, "recompute_modules", None) or ()
    return "core_attn" in modules


def validate_config(config) -> None:
    """Reject the configurations where the flag would silently do nothing.

    The flag is a *scheduling* switch with one supported shape: DSA phase-3
    layers inside a pipeline that issues its p2p asynchronously. These
    combinations either lose the loss or are dead, so they raise:

    * ``pipeline_model_parallel_size == 1`` -- no ``P2P_ISSUED`` location and no
      micro-step callbacks, so only :func:`drain_all` would ever run the branch,
      i.e. strictly worse than the inline path.
    * no ``DSAIndexer`` (:func:`_has_dsa_indexer`) or
      ``dsa_indexer_loss_coeff <= 0`` -- nothing ever enqueues, so the flag is a
      typo or a leftover from another config.
    * ``dsa_indexer_use_sparse_loss=False`` -- only ``_forward_sparse`` has an
      enqueue path, so a warmup-phase run would lose its loss outright.

    Recompute of the *layer body* is deliberately not checked: the enqueue
    predicate adapts to it per layer
    (``MQALatentAttention._needs_indexer_loss``), and a config-level gate could
    not be right anyway, since ``recompute_method`` ``first_n`` / ``block`` leave
    part of the layers unwrapped. Selective recompute of ``core_attn`` is the one
    recompute shape that does cost the window, and it warns -- see below.

    ``overlap_p2p_comm`` / ``batch_p2p_comm`` only warn: with the batched form the
    send/recv runs on the calculation stream and the branch is serialised behind
    it, which is exactly the pre-flag cost, and no number changes. Raise when the
    config is wrong or dead, warn when only the overlap is missing.
    """
    if not enabled(config):
        return
    pp_size = getattr(config, "pipeline_model_parallel_size", 1) or 1
    if int(pp_size) <= 1:
        raise ValueError(
            "dsa_indexer_loss_bwd_p2p_overlap=True requires pipeline "
            "parallelism (pipeline_model_parallel_size > 1), got "
            f"{pp_size}. The overlap window is the pipeline's forward "
            "isend/irecv; without a pipeline there are no micro-step "
            "callbacks to drain the queue and the branch would only run in "
            "the end-of-step safety net. Turn the overlap off."
        )
    if not _has_dsa_indexer(config):
        raise ValueError(
            "dsa_indexer_loss_bwd_p2p_overlap=True but this model builds no "
            "DSAIndexer: it needs experimental_attention_variant="
            "'dsv4_hybrid' with hybrid_mla_attention='mqa_dsa' and at least "
            "one csa_compress_ratios entry equal to -2, got "
            f"experimental_attention_variant="
            f"{getattr(config, 'experimental_attention_variant', None)!r}, "
            f"hybrid_mla_attention="
            f"{getattr(config, 'hybrid_mla_attention', None)!r}. Nothing "
            "would ever be enqueued; turn the overlap off."
        )
    if float(getattr(config, "dsa_indexer_loss_coeff", 0.0) or 0.0) <= 0.0:
        raise ValueError(
            "dsa_indexer_loss_bwd_p2p_overlap=True but "
            "dsa_indexer_loss_coeff="
            f"{getattr(config, 'dsa_indexer_loss_coeff', 0.0)}, so no layer "
            "computes an indexer loss at all "
            "(MQALatentAttention._needs_indexer_loss gates on coeff > 0). "
            "Set a positive coefficient or turn the overlap off."
        )
    if not getattr(config, "dsa_indexer_use_sparse_loss", False):
        raise ValueError(
            "dsa_indexer_loss_bwd_p2p_overlap=True is only implemented for "
            "the sparse (phase-3) loss; set dsa_indexer_use_sparse_loss=True "
            "or turn the overlap off."
        )
    if _recomputes_core_attn(config):
        logger.warning(
            "indexer_loss_overlap: dsa_indexer_loss_bwd_p2p_overlap=True with "
            "recompute_granularity='selective' and 'core_attn' in "
            "recompute_modules. That wraps the core attention itself -- the "
            "module that owns the indexer -- in its own recompute, so the pass "
            "that runs inside the pipeline's forward is the no-grad one of that "
            "wrapper while the layer-level in_recompute marker stays False, and "
            "_needs_indexer_loss therefore enqueues on the grad-enabled replay, "
            "which happens in backward and past the p2p window. Loss and "
            "gradient are unaffected -- a later micro-step's callback, or "
            "drain_all() before the gradient scaling, runs the branch -- but "
            "nothing overlaps. Drop 'core_attn' from recompute_modules to get "
            "the speedup."
        )
    if not getattr(config, "overlap_p2p_comm", True) or getattr(
        config, "batch_p2p_comm", None
    ):
        logger.warning(
            "indexer_loss_overlap: dsa_indexer_loss_bwd_p2p_overlap=True but "
            "overlap_p2p_comm=%s / batch_p2p_comm=%s selects _batched_p2p_ops, "
            "which runs the NCCL send/recv on the calculation stream. The loss "
            "stays correct but there is nothing for it to overlap with; set "
            "overlap_p2p_comm=True to get the speedup.",
            getattr(config, "overlap_p2p_comm", True),
            getattr(config, "batch_p2p_comm", None),
        )


def register_pipeline_hooks(pp_model) -> bool:
    """Attach :func:`drain` to Paddle's micro-step callbacks.

    ``pp_model`` is the ``PipelineParallel`` wrapper and is only used to tell
    pipeline from non-pipeline: the callback registry itself is a module-level
    singleton, which is what ``register_global_pipeline_parallel_hook`` exists to
    reach, since trainers wrap the model and cannot hold the ``PipelineParallel``
    object.

    Two locations are registered, and they are not redundant:

    * ``P2P_ISSUED`` buys the overlap -- raised after the p2p is issued and before
      the wait handles are consumed.
    * ``FORWARD_END`` is the correctness fallback, self-disarming on the first
      ``P2P_ISSUED``. ``_forward_step`` raises it and every schedule funnels
      through there, and it is not gated by ``user_hooks_enabled``, so eval passes
      and schedules with no ``P2P_ISSUED`` stay correct without depending on the
      :func:`drain_all` net. :func:`_forward_end_hook` enumerates when it is the
      one doing the work.

    Neither is the ``PipelineHook`` family: that fires at the *top* of
    ``_forward_step``, one micro-step after the enqueue and after the preceding
    p2p has already been awaited, and it is suppressed on forward-only passes.

    Registration is idempotent and appends, so it coexists with whatever else the
    trainer put on the same location. A Paddle whose enum has no ``P2P_ISSUED``
    gets ``FORWARD_END`` only -- correct, never overlapped, and warned about here
    because the flag then buys nothing. The other way to end up with no overlap
    (the enum member exists but the schedule never raises it) cannot be seen from
    here; :func:`_warn_if_fallback_is_permanent` catches that one at runtime.

    Returns ``False`` when there is no pipeline to hook into; the caller then
    relies on :func:`drain_all` and gets correct-but-unoverlapped behaviour.
    """
    global _HOOKS_REGISTERED
    if _HOOKS_REGISTERED:
        return True
    try:
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepLocations,
            register_global_pipeline_parallel_hook,
        )
    except ImportError:
        logger.info(
            "indexer_loss_overlap: this Paddle has no pipeline micro-step "
            "callbacks; falling back to drain_all()"
        )
        return False
    # ``_forward_step`` is what raises FORWARD_END, so its absence means the
    # wrapper is not a pipeline model and the callback would never fire.
    if not hasattr(pp_model, "_forward_step"):
        logger.info(
            "indexer_loss_overlap: %s is not a pipeline model; "
            "falling back to drain_all()",
            type(pp_model).__name__,
        )
        return False
    p2p_location = getattr(
        PipelineParallelMicroStepLocations, "P2P_ISSUED", None
    )
    if p2p_location is not None:
        register_global_pipeline_parallel_hook(p2p_location, _p2p_issued_hook)
    else:
        logger.warning(
            "indexer_loss_overlap: this Paddle's "
            "PipelineParallelMicroStepLocations has no P2P_ISSUED, the only "
            "location from which the loss branch can overlap the pipeline "
            "send/recv. Registering the FORWARD_END fallback instead: the "
            "indexer loss and its gradient stay correct, but "
            "dsa_indexer_loss_bwd_p2p_overlap=True buys nothing on this "
            "Paddle. Upgrade it or turn the flag off."
        )
    register_global_pipeline_parallel_hook(
        PipelineParallelMicroStepLocations.FORWARD_END, _forward_end_hook
    )
    _HOOKS_REGISTERED = True
    logger.info(
        "indexer_loss_overlap: registered drain on %s of %s",
        "P2P_ISSUED (+FORWARD_END fallback)"
        if p2p_location is not None
        else "FORWARD_END",
        type(pp_model).__name__,
    )
    return True
