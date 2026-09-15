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

"""Optimizer update-magnitude monitor for the Megatron backend.

Always installed (no ``monitors`` opt-in), reported every ``monitor_interval`` steps
alongside grad-norm / param-norm:

    optim/update_rms        sqrt(mean((theta_new - theta_old)**2))
    optim/param_rms         sqrt(mean(theta_new**2))
    optim/update_param_ratio  update_rms / param_rms   (the "trust ratio", ~1e-3 healthy)

No copy of the parameters is kept. ``MixedPrecisionOptimizer.step_with_ready_grads``
runs ``self.optimizer.step()`` and then ``self._copy_main_params_to_model_params()``;
between those two calls the fp32 main param already holds ``theta_new`` while the bf16
model param still holds ``theta_old``. Wrapping the copy method therefore reads the
pre-update value for free.

The bf16 ``theta_old`` is rounded, which inflates the raw difference once the update is
small (measured 11x at lr 3e-6). ``mean_sq`` is debiased by subtracting the bf16
round-trip variance of ``theta_new``, which restores agreement to 1.00x across
lr 3e-4 .. 3e-6.
"""

import contextlib
import logging
from typing import Any

import torch
import torch.distributed as dist

from ...core.training_logs import training_logs

logger = logging.getLogger(__name__)

METRIC_PREFIX = "optim"

_CHUNK = 1 << 20


def _param_is_not_shared(param) -> bool:
    return not getattr(param, "shared", False)


def _param_is_not_tp_duplicate(param, tp_group) -> bool:
    """Whether this rank owns the canonical copy of ``param`` for TP purposes.

    A TP-sharded param is distinct on every TP rank, so every rank counts it. A
    replicated param would otherwise be counted once per TP rank, so only rank 0 does.
    Same rule as ``MegatronOptimizer._filter_grads_for_norm``.
    """
    if getattr(param, "tensor_model_parallel", False):
        return True
    try:
        from megatron.core.tensor_parallel.layers import (
            param_is_not_tensor_parallel_duplicate,
        )

        return bool(param_is_not_tensor_parallel_duplicate(param, tp_group))
    except Exception:
        if tp_group is not None:
            return tp_group.rank() == 0
        return True


def _pair_sums(
    main: torch.Tensor, model: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return ``(sum_sq_update, sum_sq_param, sum_sq_bf16_noise, numel)`` for one shard.

    ``main`` is fp32 ``theta_new``; ``model`` is the lower-precision ``theta_old`` view.
    Walked in ``_CHUNK`` slices so the elementwise temporary stays a few MB regardless of
    parameter size, instead of allocating a full fp32 copy of the shard.
    """
    flat_main = main.reshape(-1)
    flat_model = model.reshape(-1)
    n = flat_main.numel()
    ss_u = torch.zeros((), dtype=torch.float32, device=flat_main.device)
    ss_q = torch.zeros((), dtype=torch.float32, device=flat_main.device)
    for start in range(0, n, _CHUNK):
        a = flat_main[start : start + _CHUNK]
        b = flat_model[start : start + _CHUNK]
        ss_u += torch.linalg.vector_norm(a - b, dtype=torch.float32).square()
        ss_q += torch.linalg.vector_norm(
            a - a.to(model.dtype).to(a.dtype), dtype=torch.float32
        ).square()
    ss_p = torch.linalg.vector_norm(flat_main, dtype=torch.float32).square()
    return ss_u, ss_p, ss_q, n


class OptimUpdateMonitor:
    """Measure the RMS parameter update applied by the optimizer, once per step.

    Not a ``TorchProbe``: there are no forward hooks and no per-layer metrics. It wraps
    one optimizer method, accumulates 0-dim GPU tensors there, and reduces + logs in
    ``step()`` (which runs outside any forward, so collectives are free to happen).
    """

    def __init__(
        self,
        monitor_interval: int = 1,
        verbose: bool = False,
        debias_low_precision: bool = True,
    ):
        self.monitor_interval = monitor_interval
        self.step_count = 0
        self.verbose = verbose
        self.debias_low_precision = debias_low_precision
        self._patched: list[tuple[Any, str, Any, bool]] = []
        self._pending_dense: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = []
        self._pending_expert: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = []
        self._groups_resolved = False
        self._dp_cp_group = None
        self._expt_dp_group = None
        self._mp_group = None
        self._expert_group = None
        self._warned_fp32 = False
        self._warned_no_pairs = False
        self.latest: dict[str, float] = {}

    def _should_monitor(self) -> bool:
        """Same interval gate as ``BaseMonitor``, on the optimizer step counter.

        Gating in the WRAPPER (not just at report time) is what makes the interval
        actually cheap: on a skipped step no sums are computed, so the wrapper costs one
        boolean and the collectives in ``step`` never fire.
        """
        if not self.monitor_interval:
            return False
        return self.step_count % self.monitor_interval == 0

    def attach_optimizer(self, optimizer) -> bool:
        """Wrap this optimizer instance's main-param copy. Returns True if any wrapped."""
        self._resolve_groups()
        for opt in self._iter_optimizers(optimizer):
            for method in (
                "_copy_main_params_to_model_params",
                "_copy_main_params_to_param_buffer",
            ):
                original = getattr(opt, method, None)
                if original is None or getattr(
                    original, "_im_update_patched", False
                ):
                    continue
                self._install(opt, method, original, instance_level=True)
        if not self._patched and not self._warned_no_pairs:
            self._warned_no_pairs = True
            logger.warning(
                "[OptimUpdateMonitor] no main-param copy method found; update_rms unavailable"
            )
        return bool(self._patched)

    def attach_optimizer_classes(self) -> bool:
        """Wrap ``_copy_main_params_to_model_params`` on the mcore optimizer CLASSES.

        Patching the class rather than an instance is what lets this monitor be installed
        from the same model-side setup path as every other monitor: the optimizer does not
        exist yet at that point, and threading it in later would mean an extra callback in
        every training script. Every optimizer built afterwards is covered.

        Both concrete subclasses that define the copy are wrapped —
        ``Float16OptimizerWithFloat16Params`` (non-distributed) and
        ``DistributedOptimizer`` — since the copy is not defined on the shared
        ``MixedPrecisionOptimizer`` base. ``self`` is read from the call, so one wrapper
        serves all instances.
        """
        self._resolve_groups()
        classes = []
        try:
            from megatron.core.optimizer.distrib_optimizer import (
                DistributedOptimizer,
            )
            from megatron.core.optimizer.optimizer import (
                Float16OptimizerWithFloat16Params,
            )

            classes = [Float16OptimizerWithFloat16Params, DistributedOptimizer]
        except ImportError:
            logger.warning(
                "[OptimUpdateMonitor] megatron.core optimizer unavailable; update_rms disabled"
            )
            return False
        for cls in classes:
            for method in (
                "_copy_main_params_to_model_params",
                "_copy_main_params_to_param_buffer",
            ):
                original = cls.__dict__.get(method)
                if original is None or getattr(
                    original, "_im_update_patched", False
                ):
                    continue
                self._install(cls, method, original, instance_level=False)
        return bool(self._patched)

    def _iter_optimizers(self, optimizer):
        chained = getattr(optimizer, "chained_optimizers", None)
        if chained:
            for opt in chained:
                yield from self._iter_optimizers(opt)
        else:
            yield optimizer

    def _install(
        self, target, method: str, original, instance_level: bool
    ) -> None:
        """Wrap ``target.method`` so the update is measured just before the copy-back.

        ``instance_level`` selects how ``self`` reaches ``_measure``: an instance patch
        closes over the optimizer, a class patch reads it from the call's first argument.
        """
        monitor = self

        if instance_level:

            def patched(*args, **kwargs):
                monitor._measure_safe(target)
                return original(*args, **kwargs)

            was_instance_attr = method in vars(target)
        else:

            def patched(self, *args, **kwargs):
                monitor._measure_safe(self)
                return original(self, *args, **kwargs)

            was_instance_attr = False

        patched._im_update_patched = True
        # For an instance patch, remember whether the name already lived on the instance:
        # these are normally class methods, so the correct revert is to delete our
        # instance attribute and let class lookup resume, rather than leave a bound
        # method (an instance -> bound-method -> instance reference cycle) behind.
        setattr(target, method, patched)
        self._patched.append((target, method, original, was_instance_attr))

    def _measure_safe(self, opt) -> None:
        if not self._should_monitor():
            return
        try:
            self._measure(opt)
        except Exception as e:
            if self.verbose:
                logger.error(f"[OptimUpdateMonitor] measure error: {e}")

    @torch.no_grad()
    def _measure(self, opt) -> None:
        """Accumulate per-shard sums while ``theta_old`` is still in the model params.

        Dense and expert shards are bucketed separately because they reduce over
        different groups (see ``step``). The dedup predicates are evaluated on the
        ORIGINAL model param, not the shard view: ``allreduce`` (the dense/expert marker)
        is not among the attributes copied onto shard views by
        ``copy_optimizer_param_metadata``.
        """
        if getattr(opt, "is_stub_optimizer", False):
            return
        triples = self._param_triples(opt)
        if not triples:
            return
        if getattr(opt, "shard_fp32_groups", None) and not self._warned_fp32:
            self._warned_fp32 = True
            logger.warning(
                "[OptimUpdateMonitor] fp32 params are updated in place (no pre-update copy exists); "
                "they are excluded from update_rms."
            )
        tp_group = getattr(opt, "tp_group", None)
        for main, model_shard, model_param in triples:
            if (
                main is None
                or model_shard is None
                or main.numel() != model_shard.numel()
            ):
                continue
            owner = model_param if model_param is not None else model_shard
            if not _param_is_not_shared(
                owner
            ) or not _param_is_not_tp_duplicate(owner, tp_group):
                continue
            sums = _pair_sums(main.detach(), model_shard.detach())
            if getattr(owner, "allreduce", True):
                self._pending_dense.append(sums)
            else:
                self._pending_expert.append(sums)

    @staticmethod
    def _param_triples(opt):
        """``(fp32_main_shard, low_precision_shard, original_model_param)`` per param.

        The three group lists are index-parallel by construction: ``DistOpt``'s own
        ``_copy_main_params_to_model_params`` zips ``shard_fp32_from_float16_groups``
        against ``model_float16_groups`` the same way.
        """
        main_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
        shard_groups = getattr(opt, "shard_float16_groups", None)
        model_groups = getattr(opt, "model_float16_groups", None)
        if main_groups is None or shard_groups is None:
            main_groups = getattr(opt, "fp32_from_float16_groups", None)
            shard_groups = getattr(opt, "float16_groups", None)
            model_groups = shard_groups
        if main_groups is None or shard_groups is None:
            return []
        triples = []
        for gi, (main_group, shard_group) in enumerate(
            zip(main_groups, shard_groups, strict=False)
        ):
            model_group = (
                model_groups[gi]
                if model_groups is not None and gi < len(model_groups)
                else []
            )
            for pi, (main, shard) in enumerate(
                zip(main_group, shard_group, strict=False)
            ):
                model_param = model_group[pi] if pi < len(model_group) else None
                triples.append((main, shard, model_param))
        return triples

    def _resolve_groups(self) -> None:
        if self._groups_resolved:
            return
        self._groups_resolved = True
        try:
            from megatron.core import parallel_state as ps

            if not ps.model_parallel_is_initialized():
                return
            self._dp_cp_group = ps.get_data_parallel_group(
                with_context_parallel=True
            )
            self._mp_group = ps.get_model_parallel_group()
            for getter, attr in (
                (
                    getattr(ps, "get_expert_data_parallel_group", None),
                    "_expt_dp_group",
                ),
                (
                    getattr(
                        ps,
                        "get_expert_tensor_model_pipeline_parallel_group",
                        None,
                    ),
                    "_expert_group",
                ),
            ):
                if getter is None:
                    continue
                with contextlib.suppress(Exception):
                    setattr(self, attr, getter())
        except ImportError:
            pass

    def _all_reduce(self, vec: torch.Tensor, group) -> torch.Tensor:
        if (
            group is None
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return vec
        dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=group)
        return vec

    def _pool(self, pending, shard_group, model_group, device) -> torch.Tensor:
        """Reduce one bucket's ``(ss_update, ss_param, ss_bf16_noise, count)`` globally.

        Always issues both collectives, even with nothing pending, so every rank makes
        the same number of calls — a rank can legitimately own no expert shard while a
        peer does, and skipping the reduce there would hang.
        """
        acc = torch.zeros(4, dtype=torch.float64, device=device)
        for ss_u, ss_p, ss_q, n in pending:
            acc[0] += ss_u.to(torch.float64)
            acc[1] += ss_p.to(torch.float64)
            acc[2] += ss_q.to(torch.float64)
            acc[3] += float(n)
        self._all_reduce(acc, shard_group)
        self._all_reduce(acc, model_group)
        return acc

    @torch.no_grad()
    def step(self, global_step: int | None = None) -> dict[str, float]:
        """Reduce the step's sums, publish the scalars, and return them.

        Advances the interval counter first (mirroring ``BaseMonitor.step``), then reports
        only if the wrapper actually collected anything. On a skipped step the pending
        buckets are empty, so this returns ``{}`` without issuing any collective.

        Main params are sharded over DP+CP by the distributed optimizer and split again
        over TP/PP, so the sums pool over the shard group and then the model-parallel
        group — mirroring ``calc_params_l2_norm``. Expert params live on different groups
        (expert-DP, then expert-TP/PP), hence the separate bucket. RMS is nonlinear in the
        sums, so all pooling happens before the division.
        """
        if global_step is not None:
            self.step_count = global_step
        self.step_count += 1
        dense, self._pending_dense = self._pending_dense, []
        expert, self._pending_expert = self._pending_expert, []
        self.latest = {}
        if not dense and not expert:
            return self.latest

        device = None
        for bucket in (dense, expert):
            if bucket:
                device = bucket[0][0].device
                break
        if device is None:
            return self.latest

        acc = self._pool(dense, self._dp_cp_group, self._mp_group, device)
        acc += self._pool(
            expert,
            self._expt_dp_group or self._dp_cp_group,
            self._expert_group or self._mp_group,
            device,
        )

        if float(acc[3]) <= 0.0:
            return self.latest
        count = acc[3].clamp(min=1.0)
        update_ms = acc[0] / count
        param_ms = acc[1] / count
        if self.debias_low_precision:
            update_ms = (update_ms - acc[2] / count).clamp(min=0.0)
        update_rms = float(update_ms.sqrt())
        param_rms = float(param_ms.sqrt())
        self.latest = {
            f"{METRIC_PREFIX}/update_rms": update_rms,
            f"{METRIC_PREFIX}/param_rms": param_rms,
            f"{METRIC_PREFIX}/update_param_ratio": update_rms / param_rms
            if param_rms > 0
            else 0.0,
        }
        training_logs.update(**self.latest)
        return self.latest

    def remove_hooks(self) -> None:
        for target, method, original, was_instance_attr in self._patched:
            current = (
                target.__dict__.get(method)
                if isinstance(target, type)
                else getattr(target, method, None)
            )
            if not getattr(current, "_im_update_patched", False):
                continue
            if isinstance(target, type) or was_instance_attr:
                setattr(target, method, original)
            else:
                target.__dict__.pop(method, None)
        self._patched = []
        self._pending_dense = []
        self._pending_expert = []


def setup_optim_update_monitor(
    model=None,
    optimizer=None,
    monitor_dict: dict | None = None,
    monitor_interval: int = 1,
    verbose: bool = False,
    **_ignored,
):
    """Enable ``optim/update_rms`` + ``param_rms``. Always installed; no opt-in needed.

    ``model`` is accepted and ignored so this can sit in the same registry as the
    model-side monitors and be driven by the same ``setup_internal_medicine`` call. With
    no ``optimizer`` given (the normal path — it does not exist at model-setup time) the
    mcore optimizer CLASSES are patched instead, which covers whatever is built later.

    ``monitor_interval`` is honoured exactly like the model-side monitors: these are
    slow-moving scalars, so reporting them every step would only add log noise and two
    collectives per step.

    Registered under the ``optim`` key, matching ``METRIC_PREFIX``, so a caller that
    prints ``training_logs`` by iterating ``monitor_dict`` keys as prefixes picks these up
    with no extra wiring.
    """
    monitor = OptimUpdateMonitor(
        monitor_interval=monitor_interval, verbose=verbose
    )
    if optimizer is not None:
        monitor.attach_optimizer(optimizer)
    else:
        monitor.attach_optimizer_classes()
    if monitor_dict is not None:
        monitor_dict[METRIC_PREFIX] = monitor
    return model
