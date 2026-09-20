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


import functools
import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import ScheduleNode
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

from paddlefleet.accuracy_compatible_patch import LossScaleBeforeBackward
from paddlefleet.context_parallel_utils import (
    ContextParallelGatherOp,
    ContextParallelScatterOp,
    MTPDistillationLossShift,
)
from paddlefleet.parallel_state import (
    get_context_parallel_world_size,
    get_expert_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.recompute_utils import module_needs_recompute
from paddlefleet.training.global_vars import get_global_training_logs
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import use_dsv4_accuracy_compatible

# A replay must not notify either the original observer or one installed by a
# later micro-batch. Context-local state also keeps nested calls isolated.
_token_loss_replaying = ContextVar("token_loss_replaying", default=False)


# calculate_per_token_loss token accounting (E175/E176, loss_grad_normalization.md
# 6.1). The loss module runs once per microbatch and cannot see the trainer's step
# boundary, so it stashes the running count of THIS rank's shard non-padded label
# tokens. Counted BEFORE the CP gather (per-shard), so an all-reduce-SUM over the
# sharding group at the accumulation boundary yields the global token count with no
# cp double-count. Module-level rather than a forward() return value so the loss
# interface (and its many callers) stays unchanged. The trainer pops it per step.
_PER_TOKEN_LOCAL_COUNT = [0]


def accumulate_per_token_local_count(n):
    """Add this microbatch's per-shard non-padded token count to the step total."""
    _PER_TOKEN_LOCAL_COUNT[0] += int(n)


def pop_per_token_local_count():
    """Return the accumulated per-shard token count for the step and reset it."""
    n = _PER_TOKEN_LOCAL_COUNT[0]
    _PER_TOKEN_LOCAL_COUNT[0] = 0
    return n


# T_global must count the MAIN lm head's tokens ONLY (E180). Megatron's
# finalize_model_grads divides every gradient by total_num_tokens taken from the
# main loss alone; the MTP heads compensate inside themselves
# (Megatron-LM/.../multi_token_prediction.py:1781/1851-1852 `original/rolled`) and
# never enter that divisor. forward_impl is shared by the main head and by every
# MTP depth, so the MTP calls are wrapped in suppress_per_token_count() to keep
# their (rolled) label counts out of the step total. Without this T_global grows to
# ~(1+mtp_num_layers)x and the main-loss gradient is diluted by the same factor --
# and that happens even when add_mtp_loss=False, where MTP contributes no gradient
# at all to compensate. Context-local (not a forward() argument) so no call-site
# signature changes.
_PT_SUPPRESS_COUNT = ContextVar("pt_suppress_count", default=False)


@contextmanager
def suppress_per_token_count():
    """Keep the enclosed loss call out of the per-token T_global accounting."""
    token = _PT_SUPPRESS_COUNT.set(True)
    try:
        yield
    finally:
        _PT_SUPPRESS_COUNT.reset(token)


def _per_token_count_enabled(config):
    """True when this loss call should add its tokens to the step's T_global.

    Three conditions: per-token mode is on, this is not a recompute replay (a
    replay re-runs forward_impl in backward and would double the count), and the
    caller has not marked this call as an MTP head. Training-vs-eval is not checked
    here -- forward_impl asserts up front that per-token never runs in eval mode.
    """
    return (
        getattr(config, "calculate_per_token_loss", False)
        and not _token_loss_replaying.get()
        and not _PT_SUPPRESS_COUNT.get()
    )


# Display-only per-token accounting (print path only; detached, never touches
# grads). The returned loss is a raw per-token SUM, so tr_loss is token-scale and
# cannot be printed directly. forward() -- where the lm and mtp heads are already
# separate -- stashes, per microbatch, each head's post-CP-gather (masked-loss-sum,
# non-pad token count) here. The trainer prints two calibers:
#   * old caliber (comparable to baseline): `loss` = per-microbatch per-token mean
#     of the combined loss (lm [+ scaled mtp iff add_loss puts mtp in the value]),
#     equal-weight averaged over microbatches; `mtp_i_loss` via mtp_loss_tracker.
#   * new per-token caliber: `per_token_loss` = Σ lm_sum / Σ lm_tok and
#     `mtp_per_token_loss` = Σ mtp_sum / Σ mtp_tok. The trainer all-reduces
#     numerator and denominator separately over the whole world, so the cp/tp/pp
#     duplication of each sequence cancels in the ratio (also makes the printed
#     value correct on every rank, unlike a last-stage-only value).
# Splitting lm vs mtp is done in forward(), so forward_impl/_forward keep their
# original two-arg signatures and every caller (eval_acc monkey-patch, subbatch,
# ...) is unaffected.
_PER_TOKEN_DISP = {
    "lm_sum": 0.0,
    "lm_tok": 0,
    "mtp_sum": 0.0,
    "mtp_tok": 0,
    "loss_mean_sum": 0.0,
    "loss_mean_cnt": 0,
}


def accumulate_per_token_disp_lm(loss_sum, tok):
    """Add this microbatch's lm (masked-loss-sum, token-count) for new-caliber print."""
    _PER_TOKEN_DISP["lm_sum"] += float(loss_sum)
    _PER_TOKEN_DISP["lm_tok"] += int(tok)


def accumulate_per_token_disp_mtp(loss_sum, tok):
    """Add this microbatch's aggregated mtp (masked-loss-sum, token-count)."""
    _PER_TOKEN_DISP["mtp_sum"] += float(loss_sum)
    _PER_TOKEN_DISP["mtp_tok"] += int(tok)


def accumulate_per_token_disp_loss_mean(mean_value):
    """Add this microbatch's combined per-token mean (old caliber, comparable)."""
    _PER_TOKEN_DISP["loss_mean_sum"] += float(mean_value)
    _PER_TOKEN_DISP["loss_mean_cnt"] += 1


def pop_per_token_display():
    """Return the step's display stats (dict of python scalars) and reset."""
    d = dict(_PER_TOKEN_DISP)
    _PER_TOKEN_DISP.update(
        lm_sum=0.0,
        lm_tok=0,
        mtp_sum=0.0,
        mtp_tok=0,
        loss_mean_sum=0.0,
        loss_mean_cnt=0,
    )
    return d


def _loss_md5_enabled() -> bool:
    return os.environ.get("LOG_LOSS_MD5", "0") == "1"


def _use_accuracy_compatible_kernel() -> bool:
    """Switch for Megatron-aligned (accuracy-compatible) numeric paths.

    Controlled by the ``FLAGS_use_accuracy_compatible_kernel`` env variable.
    """
    return os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"


def _tensor_md5(tensor: Tensor, dtype: str = "float32") -> str:
    """Calculate MD5 hash of a tensor, **for debugging only**.

    Note: internally calls .numpy() which triggers GPU→CPU synchronization
    and blocks the async training pipeline. Do NOT use in the forward pass.
    """
    tensor_for_md5 = tensor.detach().cast(dtype)
    return hashlib.md5(tensor_for_md5.numpy().tobytes()).hexdigest()


def _print_scalar_loss_md5(prefix: str, name: str, loss: Tensor) -> None:
    if not _loss_md5_enabled():
        return
    rank = paddle.distributed.get_rank()
    loss_tensor = loss.detach().cast("float32").reshape([1])
    print(
        f"[{prefix}] rank={rank} {name}={loss_tensor.item():.20f} "
        f"{name}_md5={_tensor_md5(loss_tensor)}",
        flush=True,
    )


class DistributedSoftmaxOp(PyLayer):
    @staticmethod
    def forward(ctx, x, axis=-1, mp_group=None):
        ctx.axis = axis
        if mp_group is None:
            hcg = fleet.get_hybrid_communicate_group()
            mp_group = hcg.get_model_parallel_group()

        ctx.mp_group = mp_group

        local_max = paddle.max(x, axis=axis, keepdim=True)

        all_max = AllGatherOp.apply(local_max)

        global_max = paddle.max(all_max, axis=0, keepdim=True)

        x_stable = x - global_max

        exp_x = paddle.exp(x_stable.cast("float32"))

        local_sum_exp = paddle.sum(exp_x, axis=axis, keepdim=True)

        sum_exp = mp_ops._mp_allreduce(
            local_sum_exp,
            group=mp_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )

        softmax_output = exp_x / sum_exp

        ctx.save_for_backward(softmax_output, sum_exp)

        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, global_sum_exp = ctx.saved_tensor()
        axis = ctx.axis
        mp_group = ctx.mp_group

        grad_softmax = grad_output * softmax_output

        local_sum_grad = paddle.sum(grad_softmax, axis=axis, keepdim=True)

        all_sum_grad = AllGatherOp.apply(local_sum_grad)
        global_sum_grad = paddle.sum(all_sum_grad, axis=0, keepdim=True)

        grad_input = softmax_output * (grad_output - global_sum_grad)

        return grad_input


def subbatch(
    f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}
):
    """
    Converts a function to one that applies to subbatch of an input dimension.
    This is useful for processing large tensors in smaller chunks to reduce memory usage.

    Args:
        f (Callable): Original function to be converted to subbatch processing.
        arg_idx ([int]): Indices of the inputs to be subbatched.
        axis ([int]): Indices of the dimensions to be subbatched for each input.
        bs (int): Subbatch size (number of elements to process at once).
        out_idx (int): Index of the output dimension that needs stacking.
        use_recompute (bool, optional): Whether to use recomputation for memory savings. Defaults to False.
        same_arg_idx (dict, optional): Mapping of argument indices that share the same tensor.
                                     e.g. {1: 0} means args[1] == args[0], avoiding duplicate slicing.

    Returns:
        Callable: Converted function that processes inputs in subbatches.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        assert len(arg_idx) == len(axis), (
            "Number of batching args and number of batching dims should match."
        )

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = dict(zip(inps, axis))

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert i > same_arg_idx[i], (
                        f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    )
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(
                    f, *_args, **kwargs
                )
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.cat(outs, out_idx)

    return wrapper


class LanguageLoss(FleetLayer):
    """Language loss with an optional ``_eval_token_loss_hook(loss, labels)``.

    The observer receives unreduced CE before masking/normalization, in the
    same token layout as the accompanying labels. It runs for the main head
    and each CE-based MTP head (not the distillation objective), including
    all-masked inputs. Selective ``loss_fn`` backward recomputation does not
    notify again. Direct ``forward_impl`` calls each notify independently.

    Observers must not mutate the tensors; detach any values retained for
    metrics. Return values are ignored and observer exceptions propagate.
    Head selection/aggregation belongs to the caller.
    """

    # Class-level tracker for MTP loss, read by trainer for logging.
    mtp_loss_tracker: dict[str, float] = {}

    # Class-level stash for cu_seqlens_q under use_erndata=True.
    # Populated on every rank by the dataloader (ernie5
    # dist_data_loader.py — right after the three broadcast_data_obj calls),
    # and additionally by gpt_embedding.forward on the embedding stage as a
    # PP=1 safety net. Consumed here to drive strict per-doc `paddle.roll`
    # in the MTP label-rolling loop so EOS positions are zero-masked
    # correctly and match the embedding-side per-doc roll bit-exactly.
    #
    # PP=1: each micro-batch runs embedding→loss end-to-end before the next
    #   arrives, so the stash is race-free even without thread-locals.
    # PP>1: the last stage never runs embedding; the dataloader stash on every
    #   rank is what enables the loss stage to see cu_seqlens_q. cu_seqlens_q
    #   travels alongside `mtp_startend_row_indices_all` in every
    #   `broadcast_data_obj` tuple (shuffle / main / pp_data).
    _cu_seqlens_q_stash: "paddle.Tensor | None" = None

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.config = config
        self.use_accuracy_compatible = getattr(
            config, "use_accuracy_compatible", False
        )
        self.ignored_index = -100
        self.enable_parallel_cross_entropy = (
            paddle.distributed.is_initialized()
            and get_tensor_model_parallel_world_size() > 1
            and config.parallel_output
        )

        if self.enable_parallel_cross_entropy:
            self.loss_func = (
                paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy()
            )
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

        self.loss_subbatch_sequence_length = (
            config.loss_subbatch_sequence_length
        )
        self.use_subbatch = self.loss_subbatch_sequence_length > 0

        # Per-token loss + MTP: only the use_erndata=True (megatron-style, rolled
        # label) head layout has its per-head token normalisation worked out (E183).
        # Every other combination below would be *silently* mis-scaled, so reject the
        # config up front rather than train on a wrong objective. All of these are
        # fixed at construction time, and the guard is scoped to
        # calculate_per_token_loss so the legacy mean-of-means path is unaffected.
        mtp_enabled = (
            getattr(config, "num_nextn_predict_layers", 0)
            and config.num_nextn_predict_layers > 0
            and not getattr(config, "mtp_load_weight_only", False)
        )
        if getattr(config, "calculate_per_token_loss", False) and mtp_enabled:
            cp_size = (
                get_context_parallel_world_size()
                if paddle.distributed.is_initialized()
                else 1
            )
            # separate_mtp_headloss=True routes MTP through MTPLanguageLoss +
            # MainLanguageLoss instead of the fused forward() above. That pair derives
            # each head's token count from its own label slice/roll (both the erndata
            # roll and the L+K slice) and never reaches forward()'s
            # experimental-version MEAN branch, so the two guards that exist purely
            # because of the fused layout are scoped out of it (E194).
            separate = getattr(config, "separate_mtp_headloss", False)
            assert separate or getattr(config, "use_erndata", False), (
                "calculate_per_token_loss + MTP currently requires use_erndata=True "
                "(or separate_mtp_headloss=True). In the fused path the L+K-slicing "
                "layout leaves labels_ori as [B, L+K], so the main head's token count "
                "and the per-depth counts are not derivable there (E183)."
            )
            assert not config.train_mtp_only, (
                "calculate_per_token_loss + train_mtp_only is unsupported: the main "
                "head is skipped, so T_global would be 0 and the trainer's clamp "
                "would divide every gradient by 1 instead of the token count (E182)."
            )
            assert not config.mtp_distillation_loss, (
                "calculate_per_token_loss + mtp_distillation_loss is unsupported: "
                "that path returns a per-token MEAN per head and never enters the "
                "raw-sum caliber the trainer's 1/T_global assumes (E178)."
            )
            assert separate or not (
                config.gpt_model_use_experimental_version
                and config.fused_linear_ce_loss_chunk <= 0
            ), (
                "calculate_per_token_loss + gpt_model_use_experimental_version with "
                "fused_linear_ce_loss_chunk<=0 is unsupported: that branch reduces "
                "the MTP head to a CP-LOCAL mean, which the per-head renormalisation "
                "would scale cp_size times too large (E181)."
            )
            assert not (
                config.experimental_dataflow
                and cp_size > 1
                and getattr(config, "use_erndata", False)
            ), (
                "calculate_per_token_loss + use_erndata + experimental_dataflow with "
                "CP>1 is unsupported: per-depth labels are already CP-extracted when "
                "they are rolled, and _forward would scatter them a second time."
            )

    def forward_impl(self, logits: Tensor | tuple, labels: Tensor) -> Tensor:
        # Per-token loss is a TRAINING-ONLY caliber, so refuse any eval / no-grad
        # forward instead of returning a silently wrong number (E188). Two independent
        # reasons, either of which is enough:
        #   * the returned loss is a raw per-token SUM -- the trainer applies the single
        #     1/T_global to the GRADIENTS at the accumulation boundary, never to this
        #     scalar. erniebot's EvalPPLMixin.evaluate re-weights it as a per-group MEAN
        #     NLL (`nll_sum += loss * pending_tokens`), so eval loss / PPL would be wrong
        #     by ~the microbatch token count (usually overflowing).
        #   * the token counters below are popped only at the training accumulation
        #     boundary, so whatever an eval pass accumulated would leak into the NEXT
        #     train step's T_global and dilute that step's gradients.
        # Re-enabling eval means normalising the returned loss by its own valid-token
        # count at the eval call site, and saving/restoring the counters around it.
        assert self.training or not getattr(
            self.config, "calculate_per_token_loss", False
        ), (
            "eval / non-training forward is unsupported under "
            "calculate_per_token_loss: the returned loss is a raw per-token SUM (eval "
            "PPL would be wrong by ~the microbatch token count) and the token counters "
            "have no pop on that path (they would dilute the next train step). Disable "
            "eval for per-token runs."
        )
        # Fused linear + cross-entropy path: `logits` is actually a
        # (hidden_states, weight, bias) tuple emitted by GPTLMHead when
        # config.fused_linear_ce_loss_chunk > 0. Dispatch to the fused kernel
        # to avoid materializing the full [B, S, V] logits tensor.
        if isinstance(logits, tuple):
            assert not self.enable_parallel_cross_entropy, (
                "fused_linear_ce_loss_chunk is incompatible with tensor parallel "
                "parallel_output=True (ParallelCrossEntropy path)."
            )
            from paddlefleet.triton_ops.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyFunction,
            )

            hidden_states, weight, bias = logits[:3]
            # Multimax lm_head fused path: GPTLMHead emits a 5-tuple
            # (hidden_states, weight, bias, multimax_ranges, multimax_ts)
            # so SegLU is applied inside the chunked CE kernel without
            # materializing full [B, S, V] logits.
            multimax_ranges = logits[3] if len(logits) > 3 else None
            multimax_ts = logits[4] if len(logits) > 4 else None
            B, S, H = hidden_states.shape
            _input = hidden_states.reshape([-1, H])
            _labels = labels.reshape([-1])

            apply_args = [
                _input,
                weight,
                _labels,
                bias,
                self.ignored_index,
                "none",
                self.config.fused_linear_ce_loss_chunk,
                getattr(
                    self.config, "gpt_model_use_experimental_version", False
                ),
            ]
            if multimax_ranges is not None and multimax_ts is not None:
                apply_args.append(multimax_ranges)
                apply_args.append(multimax_ts)
            loss_1d = LigerFusedLinearCrossEntropyFunction.apply(*apply_args)
            # Reshape back to [B, S] so downstream CP gather / lossmask
            # handling matches the non-fused path exactly.
            loss = loss_1d.reshape([B, S])

            # Per-token loss: count THIS shard's non-padded tokens before the CP
            # gather (labels here are the local shard), so the trainer's sharding
            # all-reduce sums each token exactly once. See loss_grad_normalization.md 6.1.
            # MAIN head only -- MTP calls arrive under suppress_per_token_count(), and
            # recompute replays are skipped, so neither inflates T_global (E180).
            # Display stats are recorded separately in forward().
            if _per_token_count_enabled(self.config):
                accumulate_per_token_local_count((labels != self.ignored_index).sum())

            if get_context_parallel_world_size() > 1:
                loss = ContextParallelGatherOp.apply(
                    loss, axis=1, mode=self.config.cp_balance_mode
                )
                labels = ContextParallelGatherOp.apply(
                    labels, axis=1, mode=self.config.cp_balance_mode
                )

            observer = getattr(self, "_eval_token_loss_hook", None)
            if observer is not None and not _token_loss_replaying.get():
                observer(loss, labels)
            lossmask = labels != self.ignored_index
            if (~lossmask).all():
                return paddle.mean(loss) * 0.0

            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            loss = paddle.sum(
                loss.cast(paddle.float32).reshape([-1]) * lossmask
            )
            if not getattr(self.config, "calculate_per_token_loss", False):
                # mean-of-means: local per-token mean here, the trainer averages
                # over replicas/microbatches. Per-token instead keeps the raw sum
                # and divides by the global token count once (trainer, step end).
                loss = loss / lossmask.sum()
            return loss

        seq_len = logits.shape[1]

        # Loss-path MD5 probe: logits and labels before cross-entropy
        import os

        if (
            os.environ.get("LOG_LAYER_MD5", "0") == "1"
            or os.environ.get("LOG_LOSS_MD5", "0") == "1"
        ):
            import hashlib

            rank = paddle.distributed.get_rank()
            lg_md5 = hashlib.md5(
                logits.cast("float32").numpy().tobytes()
            ).hexdigest()
            lb_md5 = hashlib.md5(
                labels.cast("int64").numpy().tobytes()
            ).hexdigest()
            print(
                f"[LOSS_PATH_MD5] rank={rank} loss_input_logits shape={list(logits.shape)} md5={lg_md5}",
                flush=True,
            )
            print(
                f"[LOSS_PATH_MD5] rank={rank} loss_input_labels shape={list(labels.shape)} md5={lb_md5}",
                flush=True,
            )

        if self.use_subbatch and seq_len > self.loss_subbatch_sequence_length:

            def _cast_loss_func(logits, labels):
                return self.loss_func(logits.cast("float32"), labels)

            sb_loss_func = subbatch(
                _cast_loss_func,
                arg_idx=[0, 1],
                axis=[1, 1],
                bs=self.loss_subbatch_sequence_length,
                out_idx=1,
            )
            loss = sb_loss_func(logits, labels)
        else:
            if (
                self.config.gpt_model_use_experimental_version
                and self.config.sequence_parallel
            ):
                logits = logits.reshape([labels.shape[0], -1, logits.shape[-1]])
            loss = self.loss_func(logits.cast("float32"), labels)

        # Per-token loss: count this shard's non-padded tokens before the CP gather
        # (labels here are the local shard) so the trainer's sharding all-reduce
        # sums each token exactly once. See loss_grad_normalization.md 6.1.
        # MAIN head only -- MTP calls arrive under suppress_per_token_count(), and
        # recompute replays are skipped, so neither inflates T_global (E180).
        # Display stats are recorded separately in forward().
        if _per_token_count_enabled(self.config):
            accumulate_per_token_local_count((labels != self.ignored_index).sum())

        if get_context_parallel_world_size() > 1:
            loss = ContextParallelGatherOp.apply(
                loss, axis=1, mode=self.config.cp_balance_mode
            )
            labels = ContextParallelGatherOp.apply(
                labels, axis=1, mode=self.config.cp_balance_mode
            )

        if _use_accuracy_compatible_kernel():
            # 定位锚点 1：CP gather 后、mask/归一化前的 per-token CE，
            # 两侧语义唯一，未掺入归一化差异。
            print(
                f"\nper_token_loss: rank={dist.get_rank()} "
                f"shape={list(loss.shape)} md5={loss.cast('float32')._md5sum()}",
                flush=True,
            )

        observer = getattr(self, "_eval_token_loss_hook", None)
        if observer is not None and not _token_loss_replaying.get():
            observer(loss, labels)
        lossmask = labels != self.ignored_index
        if (~lossmask).all():
            loss = paddle.mean(loss) * 0.0
        else:
            lossmask = lossmask.reshape([-1]).cast(paddle.float32)

            # Loss-path MD5 probe: per-token loss and lossmask
            if (
                os.environ.get("LOG_LAYER_MD5", "0") == "1"
                or os.environ.get("LOG_LOSS_MD5", "0") == "1"
            ):
                import hashlib

                rank = paddle.distributed.get_rank()
                pt_md5 = hashlib.md5(
                    loss.cast("float32").reshape([-1]).numpy().tobytes()
                ).hexdigest()
                lm_md5 = hashlib.md5(lossmask.numpy().tobytes()).hexdigest()
                valid_count = lossmask.sum().item()
                loss_sum_val = paddle.sum(
                    loss.cast("float32").reshape([-1]) * lossmask
                ).item()
                print(
                    f"[LOSS_PATH_MD5] rank={rank} per_token_loss md5={pt_md5}",
                    flush=True,
                )
                print(
                    f"[LOSS_PATH_MD5] rank={rank} lossmask md5={lm_md5} valid_tokens={valid_count}",
                    flush=True,
                )
                print(
                    f"[LOSS_PATH_MD5] rank={rank} loss_sum={loss_sum_val} final_loss={loss_sum_val / valid_count}",
                    flush=True,
                )
                # Also compute line-wise loss (matches EC's _line_wise_loss) for exact comparison
                if self.config.gpt_model_use_experimental_version:
                    _probe_loss_2d = loss.cast(
                        paddle.float32
                    ) * lossmask.reshape(labels.shape)
                    _probe_lm_2d = lossmask.reshape(labels.shape)
                    _probe_tc = _probe_lm_2d.sum(-1)
                    _probe_inv = (_probe_tc == 0).astype(paddle.float32)
                    _probe_lpl = _probe_loss_2d.sum(-1) / (
                        _probe_tc + 1e-6 * _probe_inv
                    )
                    _probe_lpl = _probe_lpl * (1 - _probe_inv)
                    _probe_lw = _probe_lpl.sum() / (
                        (1 - _probe_inv).sum() + 1e-6
                    )
                    print(
                        f"[LOSS_PATH_MD5] rank={rank} line_wise_loss={_probe_lw.item():.20f}",
                        flush=True,
                    )

            # EC-compat: line-wise loss (per-sample mean then average across samples)
            # EC's ErniemmPretrainingCriterion recomputes loss as line-wise when task_id
            # is present, which changes the value due to division by (count + 1e-6).
            if getattr(self.config, "calculate_per_token_loss", False):
                # Per-token: raw non-padded token loss sum; the single global-token
                # denominator is applied later by the trainer (skip both the
                # mean-of-means /lossmask.sum() and the EC line-wise reweighting,
                # which are incompatible with a global-token objective).
                loss = paddle.sum(
                    loss.cast(paddle.float32).reshape([-1]) * lossmask
                )
            elif self.config.gpt_model_use_experimental_version:
                if max(get_tensor_model_parallel_world_size(), 1) > 1:
                    loss = loss.squeeze(-1)
                loss_2d = loss.cast(paddle.float32) * lossmask.reshape(
                    labels.shape
                )
                lossmask_2d = lossmask.reshape(labels.shape)
                token_count_per_line = lossmask_2d.sum(-1)
                is_invalid_line_float = (token_count_per_line == 0).astype(
                    paddle.float32
                )
                loss_per_line = loss_2d.sum(-1) / (
                    token_count_per_line + 1e-6 * is_invalid_line_float
                )
                loss_per_line = loss_per_line * (1 - is_invalid_line_float)
                loss = loss_per_line.sum() / (
                    (1 - is_invalid_line_float).sum() + 1e-6
                )
            else:
                if self.use_accuracy_compatible and not (
                    use_dsv4_accuracy_compatible()
                    and self.config.experimental_attention_variant
                    == "dsv4_hybrid"
                ):
                    _flat = loss.cast(paddle.float32).reshape([-1]) * lossmask
                    loss_sum = (
                        _flat.cast(paddle.float64).sum().cast(paddle.float32)
                    )
                    _count = lossmask.sum()
                    import paddle.distributed as _pdist

                    _pg_collection = getattr(self, "pg_collection", None)
                    _ep_group = getattr(_pg_collection, "ep", None)
                    if _ep_group is None:
                        _ep_group = get_expert_model_parallel_group(
                            check_initialized=False
                        )
                    _ep_size = (
                        _pdist.get_world_size(group=_ep_group)
                        if _ep_group is not None
                        else 1
                    )
                    _acc_sum = paddle.zeros([1], dtype=paddle.float32)
                    for _ in range(_ep_size):
                        _acc_sum = _acc_sum + loss_sum
                    loss = _acc_sum[0] / (_count * _ep_size)
                else:
                    loss = paddle.sum(
                        loss.cast(paddle.float32).reshape([-1]) * lossmask
                    )
                    loss = loss / lossmask.sum()

        if _use_accuracy_compatible_kernel():
            # 定位锚点 2：mask + 归一化后的标量 loss，与锚点 1 配合可切开
            # 「CE 上游差异」和「lossmask / valid_token / 除法差异」。
            print(
                f"\nfinal_loss: rank={dist.get_rank()} "
                f"val={float(loss):.20f} md5={loss.cast('float32')._md5sum()}",
                flush=True,
            )

        return loss

    def _forward(self, logits: Tensor | tuple, labels: Tensor):
        if (
            get_context_parallel_world_size() > 1
            and self.config.experimental_dataflow
        ):
            # In EB data flow and CP size > 1, scatter labels to cp local
            labels = ContextParallelScatterOp.apply(
                labels, axis=1, mode=self.config.cp_balance_mode
            )
        if module_needs_recompute("loss_fn", None, self.config):
            # One lifetime per CE invocation, not a flag on the shared layer:
            # several heads/micro-batches may await backward simultaneously.
            # Do not retain token losses or their autograd graphs in the closure.
            forward_active = True

            def loss_forward(logits, labels):
                token = _token_loss_replaying.set(
                    _token_loss_replaying.get() or not forward_active
                )
                try:
                    return self.forward_impl(logits, labels)
                finally:
                    _token_loss_replaying.reset(token)

            try:
                return recompute(loss_forward, logits, labels)
            finally:
                forward_active = False
        return self.forward_impl(logits, labels)

    def _megatron_label_for_depth(self, labels_ori, depth, extract_cp=True):
        """Megatron-style per-MTP-depth labels.

        Under use_erndata=True labels arrive length-L (no L+K append).
        ``depth < 0`` returns the main labels unchanged (length L); ``depth >= 0``
        rolls labels_ori left ``depth + 1`` times, filling ``ignored_index`` at
        every packed-document boundary (via the stashed cu_seqlens_q) so the
        boundary token is excluded from the loss. Under CP>1 this rank's local
        CP slice is extracted (per config.cp_balance_mode) so the label shape
        matches the local logits.

        Mirrors the megatron branch of ``LanguageLoss.forward`` so the separate
        Main/MTP head-loss path stays consistent with the fused path.

        ``extract_cp=False`` returns the FULL (un-scattered) labels; used by
        MainLanguageLoss to count per-head tokens in the same domain as the
        post-CP-gather loss sums.
        """
        if depth < 0:
            _lbl = labels_ori
        else:
            _cu = LanguageLoss._cu_seqlens_q_stash
            if _cu is None:
                raise RuntimeError(
                    "use_erndata=True requires cu_seqlens_q to be "
                    "stashed on LanguageLoss._cu_seqlens_q_stash before the loss "
                    "stage, but it is None on this rank. It should be set by "
                    "GPTEmbedding.forward (PP=1) or the LM head on the last PP "
                    "stage (GPTLMHead / GPTMainLMHead / GPTMTPLMHead)."
                )
            from paddlefleet.transformer.multi_token_prediction import (
                _roll_tensor_packed_seq,
            )

            _lbl = labels_ori
            for _ in range(depth + 1):
                _lbl, _ = _roll_tensor_packed_seq(
                    _lbl,
                    shifts=-1,
                    dims=1,
                    cu_seqlens_q=_cu,
                    pad_value=self.ignored_index,
                )
        if extract_cp and get_context_parallel_world_size() > 1:
            from paddlefleet.parallel_state import get_context_parallel_rank
            from paddlefleet.transformer.multi_token_prediction import (
                extract_local_cp_chunks,
            )

            _lbl = extract_local_cp_chunks(
                _lbl,
                get_context_parallel_rank(),
                get_context_parallel_world_size(),
                axis=1,
                mode=self.config.cp_balance_mode,
            )
        return _lbl

    def forward(self, logits: Tensor | list, labels: Tensor) -> Tensor:
        if isinstance(logits, list):
            assert (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
                and not self.config.mtp_load_weight_only
            )
            assert len(logits) == self.config.num_nextn_predict_layers + 1
            labels_ori = labels
            # Under use_erndata=True labels are already [B, L]
            # (no L+K trailing padding). The main-decoder logits also live
            # at length L (no L→L-K slicing was performed upstream), so
            # skip the L+K→L trim; likewise per-depth MTP labels come from
            # a per-depth roll — approximated here by shifting labels_ori
            # left by (depth+1) positions with -100 fill at the tail.
            _mtp_is_megatron = getattr(self.config, "use_erndata", False)
            # Under CP>1 the megatron path keeps labels_ori full-length on
            # every rank. The rank-local slice must be extracted here so that
            # shape matches the local logits produced by the embedding branch,
            # using the model's own CP layout (config.cp_balance_mode).
            _cp_size_for_extract = (
                get_context_parallel_world_size() if _mtp_is_megatron else 1
            )
            if _cp_size_for_extract > 1:
                from functools import partial

                from paddlefleet.parallel_state import (
                    get_context_parallel_rank as _get_cp_rank,
                )
                from paddlefleet.transformer.multi_token_prediction import (
                    extract_local_cp_chunks,
                )

                _extract_cp = partial(
                    extract_local_cp_chunks,
                    mode=self.config.cp_balance_mode,
                )
                _cp_rank_for_extract = _get_cp_rank()
            else:
                _extract_cp = None
                _cp_rank_for_extract = 0
            if _mtp_is_megatron:
                lm_labels = labels_ori
                if _cp_size_for_extract > 1:
                    # Extract this rank's local CP slice from full-length labels.
                    lm_labels = _extract_cp(
                        lm_labels,
                        _cp_rank_for_extract,
                        _cp_size_for_extract,
                        axis=1,
                    )
                seq_length = lm_labels.shape[1]
            else:
                lm_labels = labels[:, : -self.config.num_nextn_predict_layers]
                seq_length = lm_labels.shape[1]

            mtp_loss = []
            mtp_logits = logits[1:]
            # Display-only (calculate_per_token_loss): full non-pad token counts per
            # head, used by _record_per_token_display below. Detached; never affects
            # grads. lm_labels / labels_cur_depth here are the full (un-scattered)
            # slices, so these match the post-CP-gather loss sums forward_impl returns.
            _per_token_disp = getattr(self.config, "calculate_per_token_loss", False)
            _lm_tok = (
                (lm_labels != self.ignored_index).sum() if _per_token_disp else None
            )
            _mtp_tok_list = []

            # Full-sequence (pre-CP-extract) non-pad counts per head, used by the
            # per-token renormalisation below. forward_impl returns the CP-GATHERED loss
            # sum, so the matching denominator must be the full count, not this rank's
            # shard; and being identical on every CP rank it makes all ranks scale a
            # head the same way. Separate from _lm_tok / _mtp_tok_list, which stay on the
            # display path. Supported-config guards live in __init__ (E183).
            _lm_tok_full = (
                (labels_ori != self.ignored_index).sum() if _per_token_disp else None
            )
            _mtp_tok_full_list = []

            if not self.config.mtp_distillation_loss:
                if self.config.train_mtp_only:
                    lm_loss = 0.0
                else:
                    lm_loss = self._forward(logits[0], lm_labels)

                for depth in range(self.config.num_nextn_predict_layers):
                    logits_cur_depth = mtp_logits[depth]
                    if _mtp_is_megatron:
                        # Under use_erndata=True labels_ori is [B, L]
                        # (no L+K padding). MTP depth k predicts x[i+k+2],
                        # i.e. labels rolled left (k+1) times with per-doc
                        # boundary fill via cu_seqlens_q. For labels the
                        # boundary MUST be filled with ignored_index (not 0),
                        # otherwise the cross-doc position would train token 0.
                        #
                        # Strict per-doc parity: when cu_seqlens_q is
                        # available, use `_roll_tensor_packed_seq` with
                        # pad_value=ignored_index — same helper the embedding
                        # side uses (pad_value=0 there), so the EOS boundaries
                        # line up bit-exactly. When unavailable, fall back to
                        # plain `paddle.roll` + ignored_index tail.
                        _cu = LanguageLoss._cu_seqlens_q_stash
                        if _cu is not None:
                            from paddlefleet.transformer.multi_token_prediction import (
                                _roll_tensor_packed_seq,
                            )

                            _lbl = labels_ori
                            for _ in range(depth + 1):
                                _lbl, _ = _roll_tensor_packed_seq(
                                    _lbl,
                                    shifts=-1,
                                    dims=1,
                                    cu_seqlens_q=_cu,
                                    pad_value=self.ignored_index,
                                )
                        else:
                            # No cu_seqlens_q on this rank. A plain
                            # paddle.roll cannot respect packed-doc
                            # boundaries, so it would leak labels across
                            # documents (train the first token of doc N+1 as
                            # the target at the last position of doc N). Fail
                            # loudly instead of silently corrupting.
                            # cu_seqlens_q is normally stashed by
                            # GPTEmbedding.forward (PP=1 / first stage) and by
                            # GPTLMHead.forward on the last PP stage; reaching
                            # here means neither ran on this rank.
                            raise RuntimeError(
                                "use_erndata=True requires cu_seqlens_q "
                                "to be stashed on LanguageLoss._cu_seqlens_q_stash "
                                "before the loss stage, but it is None on this "
                                "rank. It should be set by GPTEmbedding.forward "
                                "(PP=1) or GPTLMHead.forward (last PP stage)."
                            )
                        if _per_token_disp:
                            # Count before the CP extract: matches the CP-gathered sum
                            # forward_impl returns, and is equal on every CP rank (E183).
                            _mtp_tok_full_list.append(
                                (_lbl != self.ignored_index).sum()
                            )
                        if _cp_size_for_extract > 1:
                            # Match local logits shape by extracting this
                            # rank's CP slice.
                            _lbl = _extract_cp(
                                _lbl,
                                _cp_rank_for_extract,
                                _cp_size_for_extract,
                                axis=1,
                            )
                        labels_cur_depth = _lbl
                    else:
                        labels_cur_depth = labels_ori[
                            :, (depth + 1) : (depth + 1 + seq_length)
                        ]
                    if _per_token_disp:
                        _mtp_tok_list.append(
                            (labels_cur_depth != self.ignored_index).sum()
                        )
                    if self.config.gpt_model_use_experimental_version:
                        # Align with EB: compute per-token loss matrix and reduce
                        # with global sum/count instead of going through forward_impl
                        # which applies line-wise loss.

                        if (
                            get_context_parallel_world_size() > 1
                            and not _mtp_is_megatron
                        ):
                            # In EB data flow and CP size > 1, since we do not use _forward
                            # we need to scatter labels to cp local here.
                            # Under use_erndata=True labels_cur_depth is
                            # already the local CP slice (extract_local_cp_chunks
                            # above), so skip the scatter to avoid double-scatter.
                            labels_cur_depth = ContextParallelScatterOp.apply(
                                labels_cur_depth,
                                axis=1,
                                mode=self.config.cp_balance_mode,
                            )

                        if self.config.fused_linear_ce_loss_chunk > 0:
                            # MTP head: its rolled labels must not enter T_global,
                            # which counts the main head only (E180).
                            with suppress_per_token_count():
                                loss_matrix_cur_depth = self._forward(
                                    logits_cur_depth,
                                    labels_cur_depth,
                                )
                        else:
                            if (
                                self.config.gpt_model_use_experimental_version
                                and self.config.sequence_parallel
                            ):
                                logits_cur_depth = logits_cur_depth.reshape(
                                    [
                                        labels_cur_depth.shape[0],
                                        -1,
                                        logits_cur_depth.shape[-1],
                                    ]
                                )
                            loss_matrix_cur_depth = self.loss_func(
                                logits_cur_depth.cast("float32"),
                                labels_cur_depth,
                            )

                        if (
                            get_context_parallel_world_size() > 1
                            and not _mtp_is_megatron
                        ):
                            # In EB data flow and CP size > 1, loss and labels need to be gathered back.
                            # Under use_erndata=True labels stay local — the
                            # subsequent lossmask/sum reduction is per-rank (allreduce
                            # happens implicitly via DP grad-averaging), so skip the
                            # gather to keep the length-L/cp local view.
                            loss_matrix_cur_depth = (
                                ContextParallelGatherOp.apply(
                                    loss_matrix_cur_depth,
                                    axis=1,
                                    mode=self.config.cp_balance_mode,
                                )
                            )
                            labels_cur_depth = ContextParallelGatherOp.apply(
                                labels_cur_depth,
                                axis=1,
                                mode=self.config.cp_balance_mode,
                            )

                        # Fused CE already notified inside _forward and returned
                        # a scalar. Only the direct, unreduced CE path needs this
                        # notification; callers may choose to consume main only.
                        if self.config.fused_linear_ce_loss_chunk <= 0:
                            observer = getattr(
                                self, "_eval_token_loss_hook", None
                            )
                            if (
                                observer is not None
                                and not _token_loss_replaying.get()
                            ):
                                observer(
                                    loss_matrix_cur_depth, labels_cur_depth
                                )

                        lossmask_cur_depth = (
                            labels_cur_depth != self.ignored_index
                        ).cast(paddle.float32)
                        loss_matrix_cur_depth = loss_matrix_cur_depth.cast(
                            paddle.float32
                        ).reshape([-1]) * lossmask_cur_depth.reshape([-1])
                        if lossmask_cur_depth.sum().item() > 0:
                            loss_cur_depth = (
                                loss_matrix_cur_depth.sum()
                                / lossmask_cur_depth.sum()
                            )
                        else:
                            loss_cur_depth = loss_matrix_cur_depth.sum() * 0.0
                    else:
                        # MTP head: its rolled labels must not enter T_global, which
                        # counts the main head only (E180).
                        with suppress_per_token_count():
                            loss_cur_depth = self._forward(
                                logits_cur_depth,
                                labels_cur_depth,
                            )
                    mtp_loss.append(loss_cur_depth)
            else:
                lm_loss = self._forward(logits[0], lm_labels)
                if get_tensor_model_parallel_world_size() > 1:
                    target_p_self_op_dist = DistributedSoftmaxOp.apply(
                        logits[0], axis=2
                    )
                else:
                    target_p_self_op_dist = nn.Softmax(axis=2)(logits[0])
                if get_context_parallel_world_size() > 1:
                    cp_balance_mode = self.config.cp_balance_mode
                    if cp_balance_mode == "contiguous_allgather":
                        target_p_self_op_dist = MTPDistillationLossShift.apply(
                            target_p_self_op_dist,
                            self.config.num_nextn_predict_layers,
                            mode=cp_balance_mode,
                        )
                    else:
                        target_p_self_op_dist = ContextParallelGatherOp.apply(
                            target_p_self_op_dist,
                            axis=1,
                            mode=cp_balance_mode,
                        )

                def padding(tensor, left=False, pad_len=1):
                    zeropadding = paddle.zeros_like(tensor[:, -pad_len:, :])
                    if left:
                        tensor = paddle.concat((zeropadding, tensor), axis=1)
                    else:
                        tensor = paddle.concat((tensor, zeropadding), axis=1)
                    return tensor

                if (
                    self.config.num_nextn_predict_layers > 0
                    and mtp_logits is not None
                ):
                    for depth in range(len(mtp_logits)):
                        prediction_scores_cur_depth = mtp_logits[depth]
                        if _mtp_is_megatron:
                            # Strict per-doc parity (mirror of the
                            # mtp_distillation_loss=False path above): use
                            # _roll_tensor_packed_seq with the cu_seqlens_q
                            # stashed by the dataloader / GPTEmbedding.forward,
                            # filling doc boundaries with ignored_index. If the
                            # stash is missing, raise instead of silently
                            # leaking labels across packed docs.
                            _cu = LanguageLoss._cu_seqlens_q_stash
                            if _cu is not None:
                                from paddlefleet.transformer.multi_token_prediction import (
                                    _roll_tensor_packed_seq,
                                )

                                _lbl = labels_ori
                                for _ in range(depth + 1):
                                    _lbl, _ = _roll_tensor_packed_seq(
                                        _lbl,
                                        shifts=-1,
                                        dims=1,
                                        cu_seqlens_q=_cu,
                                        pad_value=self.ignored_index,
                                    )
                            else:
                                # See the mtp_distillation_loss=False branch:
                                # without cu_seqlens_q a plain roll leaks
                                # labels across packed docs, so fail loudly
                                # rather than corrupt the loss silently.
                                raise RuntimeError(
                                    "use_erndata=True requires "
                                    "cu_seqlens_q to be stashed on "
                                    "LanguageLoss._cu_seqlens_q_stash before the "
                                    "loss stage, but it is None on this rank. "
                                    "It should be set by GPTEmbedding.forward "
                                    "(PP=1) or GPTLMHead.forward (last PP stage)."
                                )
                            if _cp_size_for_extract > 1:
                                _lbl = _extract_cp(
                                    _lbl,
                                    _cp_rank_for_extract,
                                    _cp_size_for_extract,
                                    axis=1,
                                )
                            labels_cur_depth = _lbl
                        else:
                            labels_cur_depth = labels_ori[
                                :, (depth + 1) : (depth + 1 + seq_length)
                            ]
                        lossmask = (
                            labels_cur_depth != self.ignored_index
                        ).cast(paddle.float32)
                        if get_tensor_model_parallel_world_size() > 1:
                            out_logp = paddle.log(
                                DistributedSoftmaxOp.apply(
                                    prediction_scores_cur_depth, axis=2
                                )
                            )
                        else:
                            out_logp = nn.LogSoftmax(axis=2)(
                                prediction_scores_cur_depth
                            )

                        if not (
                            get_context_parallel_world_size() > 1
                            and cp_balance_mode == "contiguous_allgather"
                        ):
                            target_p = target_p_self_op_dist[
                                :, (depth + 1) :, :
                            ].clone()
                            target_p = padding(
                                target_p, left=False, pad_len=depth + 1
                            )
                        if get_context_parallel_world_size() > 1:
                            if cp_balance_mode == "contiguous_allgather":
                                target_p = target_p_self_op_dist[
                                    :, depth : depth + out_logp.shape[1]
                                ]
                            else:
                                target_p = ContextParallelScatterOp.apply(
                                    target_p,
                                    axis=1,
                                    mode=cp_balance_mode,
                                )
                        plogp = target_p * out_logp

                        lossmask = lossmask[..., None]
                        xishu = lossmask.sum() + 1e-5
                        if get_context_parallel_world_size() > 1:
                            lossmask = ContextParallelScatterOp.apply(
                                lossmask,
                                axis=1,
                                mode=self.config.cp_balance_mode,
                            )

                        ploss = -paddle.sum(lossmask * plogp)
                        if get_tensor_model_parallel_world_size() > 1:
                            dist.all_reduce(
                                ploss,
                                group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                            )

                        if get_context_parallel_world_size() > 1:
                            dist.all_reduce(
                                ploss,
                                group=fleet.get_hybrid_communicate_group().get_context_parallel_group(),
                            )

                        ploss = ploss / xishu
                        mtp_loss.append(ploss)

            # Store detached MTP loss tensors into class-level tracker and global_training_logs.
            # Use .detach() instead of .item() to avoid GPU synchronization on every
            # micro-batch. The trainer will call .item() only at logging steps.
            for i, loss_val in enumerate(mtp_loss):
                LanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = (
                    loss_val.detach()
                )
                _print_scalar_loss_md5(
                    "MTP_LOSS_PATH_MD5",
                    f"mtp{i + 1}.final_loss",
                    loss_val,
                )

            logs = get_global_training_logs()
            if logs is not None and hasattr(logs, "update"):
                for i, loss_val in enumerate(mtp_loss):
                    logs.update(**{f"mtp_{i + 1}_loss": loss_val.detach()})

            def add_loss(main_loss, loss):
                if _use_accuracy_compatible_kernel():
                    # Megatron-aligned: MTP loss gradient flows but loss scalar unchanged.
                    # This matches Megatron's behavior where MTP contributes to training
                    # gradients without affecting the reported loss value.
                    if self.config.add_mtp_loss:
                        if use_dsv4_accuracy_compatible():
                            return loss - loss.detach() + main_loss
                        return main_loss + loss - loss.detach()
                    else:
                        return main_loss
                else:
                    # Original behavior
                    if self.config.add_mtp_loss:
                        return main_loss + loss
                    else:
                        return main_loss + loss - loss.detach()

            def renorm_mtp_head(idx, mtp_l):
                """Per-token: rescale MTP head `idx` from its own token count to the
                main head's -- the mirror of Megatron process_mtp_loss's
                `original_num_tokens / num_tokens`
                (Megatron-LM/megatron/core/transformer/multi_token_prediction.py:1851).
                The head returns a raw per-token SUM here, and the trainer applies a
                single 1/T_global (main-head tokens only) at the step boundary, so this
                factor is what leaves mtp_loss_scaling_factor/D x (per-token mean over
                THIS head's own valid tokens) -- the Megatron objective. Counts are
                full-sequence and equal on every CP rank. Returns mtp_l untouched
                outside per-token so the legacy path stays byte-identical.
                """
                if not _per_token_disp:
                    return mtp_l
                return mtp_l * (
                    _lm_tok_full.astype("float32")
                    / paddle.clip(
                        _mtp_tok_full_list[idx].astype("float32"), min=1.0
                    )
                )

            if self.config.gpt_model_use_experimental_version:
                # Align with EB: accumulate inside loop to match float32
                # arithmetic order: loss += scaling * loss_i / N
                loss = lm_loss
                if _use_accuracy_compatible_kernel():
                    # Megatron-aligned: only add MTP loss when add_mtp_loss=True.
                    # Use add_loss() to keep single maintenance point for compat
                    # behavior (loss + val - val.detach() for gradient-only flow).
                    if self.config.add_mtp_loss:
                        num_mtp = len(mtp_loss)
                        for i, mtp_l in enumerate(mtp_loss):
                            mtp_val = (
                                self.config.mtp_loss_scaling_factor
                                * renorm_mtp_head(i, mtp_l)
                                / num_mtp
                            )
                            loss = add_loss(loss, mtp_val)
                else:
                    # Original behavior: always use add_loss
                    num_mtp = len(mtp_loss)
                    for i, mtp_l in enumerate(mtp_loss):
                        loss = add_loss(
                            loss,
                            self.config.mtp_loss_scaling_factor
                            * renorm_mtp_head(i, mtp_l)
                            / num_mtp,
                        )
            else:
                loss = add_loss(
                    lm_loss,
                    self.config.mtp_loss_scaling_factor
                    * sum(
                        renorm_mtp_head(i, mtp_l)
                        for i, mtp_l in enumerate(mtp_loss)
                    )
                    / len(mtp_loss),
                )
            # per-token keeps the raw SUM (trainer divides by the global token count once
            # at step end, spanning all acc microbatches) -> must NOT also ÷acc_steps here.
            if use_dsv4_accuracy_compatible() and not _per_token_disp:
                loss = LossScaleBeforeBackward.scale(loss)

            # Per-token display stats (print only; detached). Split lm vs mtp here so
            # forward_impl keeps its original signature. Skipped for the distillation
            # path (mtp_loss there is already a mean, not a raw sum).
            if _per_token_disp and not self.config.mtp_distillation_loss:
                self._record_per_token_display(
                    lm_loss, _lm_tok, mtp_loss, _mtp_tok_list
                )

            return loss
        else:
            loss = self._forward(logits, labels)
            if getattr(self.config, "calculate_per_token_loss", False):
                self._record_per_token_display(
                    loss, (labels != self.ignored_index).sum(), [], []
                )
                # per-token: _forward returns the raw SUM; the trainer divides by the
                # global token count (spanning all acc microbatches) once at step end,
                # so must NOT also divide by acc_steps here (extra 1/acc under-scales grads).
            elif use_dsv4_accuracy_compatible():
                loss = LossScaleBeforeBackward.scale(loss)
            return loss

    def _record_per_token_display(self, lm_loss, lm_tok, mtp_loss, mtp_tok_list):
        """Stash detached display stats for calculate_per_token_loss (print only).

        lm_loss / mtp_loss[i] are the raw post-CP-gather masked-loss SUMS returned by
        forward_impl; lm_tok / mtp_tok_list[i] are the matching full non-pad token
        counts. Records the new per-token numerators/denominators (lm + aggregated
        mtp) and the old-caliber combined ``loss`` per-token mean (mirrors add_loss /
        add_mtp_loss). All values are detached and never affect gradients.
        """
        # lm head (lm_loss is 0.0 when train_mtp_only).
        lm_sum = 0.0 if isinstance(lm_loss, float) else float(lm_loss.detach())
        lm_tok_f = float(lm_tok)
        accumulate_per_token_disp_lm(lm_sum, lm_tok_f)
        lm_mean = lm_sum / max(lm_tok_f, 1.0)

        # mtp heads, aggregated over depths.
        mtp_sum = 0.0
        mtp_tok = 0.0
        mtp_means = []
        for ml, mt in zip(mtp_loss, mtp_tok_list):
            s = float(ml.detach())
            t = float(mt)
            mtp_sum += s
            mtp_tok += t
            mtp_means.append(s / max(t, 1.0))
        if mtp_loss:
            accumulate_per_token_disp_mtp(mtp_sum, mtp_tok)

        # Old-caliber combined `loss`: value includes mtp only when add_loss puts it
        # in the value (add_mtp_loss and not the accuracy-compatible kernel).
        combined = lm_mean
        if (
            mtp_means
            and self.config.add_mtp_loss
            and not _use_accuracy_compatible_kernel()
        ):
            combined = lm_mean + self.config.mtp_loss_scaling_factor * (
                sum(mtp_means) / len(mtp_means)
            )
        accumulate_per_token_disp_loss_mean(combined)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="LanguageLoss")


class MainLanguageLoss(LanguageLoss):
    # Class-level tracker for MTP loss, read by trainer for logging.
    mtp_loss_tracker: dict[str, float] = {}

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

    def forward(self, dict_args: dict | list, labels: Tensor) -> Tensor:
        assert (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        )
        labels_ori = labels
        if getattr(self.config, "use_erndata", False):
            # erndata: labels are length-L already; main logits are length-L
            # too, so keep the full labels (no L+K trim) and CP-extract to match.
            lm_labels = self._megatron_label_for_depth(labels_ori, -1)
        else:
            lm_labels = labels[:, : -self.config.num_nextn_predict_layers]
        seq_length = lm_labels.shape[1]

        mtp_loss = dict_args["mtp_loss"]
        logits = dict_args["logits"]

        assert not self.config.mtp_distillation_loss, (
            "separate mtp head & loss don't support mtp_distillation_loss"
        )

        if self.config.train_mtp_only:
            lm_loss = 0.0
        else:
            lm_loss = self._forward(logits, lm_labels)

        # Per-token: mtp_loss[i] is a raw masked SUM over head i's OWN rolled-label
        # token set, while the trainer applies a single 1/T_global counted from the
        # MAIN head only (E180). Recompute each head's FULL (pre-CP-scatter) non-pad
        # token count from the same slice/roll MTPLanguageLoss used, so it lives in the
        # same domain as the post-CP-gather sums. Used twice below: to recover the
        # baseline-comparable per-token mean for the mtp_i_loss display, and to
        # renormalise the head inside the loss value itself.
        per_token = getattr(self.config, "calculate_per_token_loss", False)
        mtp_tok_full_list = []
        lm_tok_full = None
        if per_token:
            if getattr(self.config, "use_erndata", False):
                lm_tok_full = (labels_ori != self.ignored_index).sum()
            else:
                lm_tok_full = (
                    labels[:, : -self.config.num_nextn_predict_layers]
                    != self.ignored_index
                ).sum()
            for depth in range(len(mtp_loss)):
                if getattr(self.config, "use_erndata", False):
                    labels_full_depth = self._megatron_label_for_depth(
                        labels_ori, depth, extract_cp=False
                    )
                else:
                    labels_full_depth = labels_ori[
                        :, (depth + 1) : (depth + 1 + seq_length)
                    ]
                mtp_tok_full_list.append(
                    (labels_full_depth != self.ignored_index).sum()
                )

        # Store detached MTP loss tensors into class-level tracker and global_training_logs.
        # Use .detach() instead of .item() to avoid GPU synchronization on every
        # micro-batch. The trainer will call .item() only at logging steps.
        if per_token:
            mtp_disp_vals = [
                mtp_loss[i].detach()
                / paddle.clip(
                    mtp_tok_full_list[i].astype("float32"), min=1.0
                )
                for i in range(len(mtp_loss))
            ]
        else:
            mtp_disp_vals = [loss_val.detach() for loss_val in mtp_loss]

        for i, disp_val in enumerate(mtp_disp_vals):
            MainLanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = disp_val
            _print_scalar_loss_md5(
                "MTP_LOSS_PATH_MD5",
                f"mtp{i + 1}.final_loss",
                mtp_loss[i],
            )

        # Also write to global_training_logs to read
        logs = get_global_training_logs()
        if logs is not None and hasattr(logs, "update"):
            for i, disp_val in enumerate(mtp_disp_vals):
                logs.update(**{f"mtp_{i + 1}_loss": disp_val})

        def add_loss(main_loss, loss):
            if _use_accuracy_compatible_kernel():
                # Megatron-aligned: MTP loss gradient flows but loss scalar unchanged.
                # This matches Megatron's behavior
                if self.config.add_mtp_loss:
                    return main_loss + loss - loss.detach()
                else:
                    return main_loss
            else:
                # Original behavior
                if self.config.add_mtp_loss:
                    return main_loss + loss
                else:
                    return main_loss + loss - loss.detach()

        def renorm_mtp_head(idx, mtp_l):
            """Per-token: rescale MTP head `idx` from its own token count to the main
            head's -- the mirror of Megatron process_mtp_loss's
            `original_num_tokens / num_tokens`
            (Megatron-LM/megatron/core/transformer/multi_token_prediction.py:1851).
            Without it the trainer's single 1/T_global would normalise every MTP token
            by the MAIN head's token count instead of its own, and the rolled labels
            have fewer valid tokens. Identity outside per-token.
            """
            if not per_token:
                return mtp_l
            return mtp_l * (
                lm_tok_full.astype("float32")
                / paddle.clip(
                    mtp_tok_full_list[idx].astype("float32"), min=1.0
                )
            )

        loss = add_loss(
            lm_loss,
            self.config.mtp_loss_scaling_factor
            * sum(
                renorm_mtp_head(i, mtp_l) for i, mtp_l in enumerate(mtp_loss)
            )
            / len(mtp_loss),
        )

        # Per-token display stats (print only; detached). Same accounting as
        # LanguageLoss.forward, which this separate-head path bypasses.
        if per_token:
            self._record_per_token_display(
                lm_loss, lm_tok_full, mtp_loss, mtp_tok_full_list
            )

        return loss

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MainLanguageLoss")


class MTPLanguageLoss(LanguageLoss):
    def __init__(
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

    def forward(self, dict_args: dict):
        mtp_logits = dict_args.get("mtp_logits")
        labels = dict_args.get("labels")
        assert mtp_logits is not None, (
            "separate mtp loss must provide mtp_logits"
        )
        assert labels is not None, "separate mtp loss must provide labels"
        assert (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        )
        labels_ori = labels
        _mtp_is_megatron = getattr(self.config, "use_erndata", False)
        if _mtp_is_megatron:
            # erndata: labels are length-L; per-depth labels come from a
            # per-doc roll (ignored_index at boundaries), NOT the ernie5 L+K
            # slice. seq_length is only used by the ernie5 slice path below.
            seq_length = labels_ori.shape[1]
        else:
            lm_labels = labels[:, : -self.config.num_nextn_predict_layers]
            seq_length = lm_labels.shape[1]

        mtp_loss = []

        assert not self.config.mtp_distillation_loss, (
            "separate mtp head & loss don't support mtp_distillation_loss"
        )

        for depth in range(self.config.num_nextn_predict_layers):
            logits_cur_depth = mtp_logits[depth]
            if _mtp_is_megatron:
                labels_cur_depth = self._megatron_label_for_depth(
                    labels_ori, depth
                )
            else:
                labels_cur_depth = labels_ori[
                    :, (depth + 1) : (depth + 1 + seq_length)
                ]
            # MTP head: its rolled labels must not enter T_global, which counts the
            # main head only (E180). Same rule as the fused LanguageLoss.forward path;
            # this separate-MTP-head class needs it independently.
            with suppress_per_token_count():
                loss_cur_depth = self._forward(
                    logits_cur_depth,
                    labels_cur_depth,
                )
            mtp_loss.append(loss_cur_depth)

        dict_args.pop("mtp_logits")
        dict_args["mtp_loss"] = mtp_loss

        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MTPLanguageLoss")
