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

    def forward_impl(self, logits: Tensor | tuple, labels: Tensor) -> Tensor:
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

            if get_context_parallel_world_size() > 1:
                loss = ContextParallelGatherOp.apply(
                    loss, axis=1, mode=self.config.cp_balance_mode
                )
                labels = ContextParallelGatherOp.apply(
                    labels, axis=1, mode=self.config.cp_balance_mode
                )

            lossmask = labels != self.ignored_index
            if (~lossmask).all():
                return paddle.mean(loss) * 0.0

            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            loss = paddle.sum(
                loss.cast(paddle.float32).reshape([-1]) * lossmask
            )
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
            if self.config.gpt_model_use_experimental_version:
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
                if self.use_accuracy_compatible:
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
            return recompute(self.forward_impl, logits, labels)
        return self.forward_impl(logits, labels)

    def _megatron_label_for_depth(self, labels_ori, depth):
        """Megatron-style per-MTP-depth labels.

        Under use_erndata=True labels arrive length-L (no L+K append).
        ``depth < 0`` returns the main labels unchanged (length L); ``depth >= 0``
        rolls labels_ori left ``depth + 1`` times, filling ``ignored_index`` at
        every packed-document boundary (via the stashed cu_seqlens_q) so the
        boundary token is excluded from the loss. Under CP>1 this rank's local
        zigzag chunk is extracted so the label shape matches the local logits.

        Mirrors the megatron branch of ``LanguageLoss.forward`` so the separate
        Main/MTP head-loss path stays consistent with the fused path.
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
        if get_context_parallel_world_size() > 1:
            from paddlefleet.parallel_state import get_context_parallel_rank
            from paddlefleet.transformer.multi_token_prediction import (
                extract_local_zigzag_chunks,
            )

            _lbl = extract_local_zigzag_chunks(
                _lbl,
                get_context_parallel_rank(),
                get_context_parallel_world_size(),
                axis=1,
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
            # every rank. Local zigzag chunks must be extracted here so that
            # shape matches the local logits produced by the embedding branch.
            _cp_size_for_extract = (
                get_context_parallel_world_size() if _mtp_is_megatron else 1
            )
            if _cp_size_for_extract > 1:
                from paddlefleet.parallel_state import (
                    get_context_parallel_rank as _get_cp_rank,
                )
                from paddlefleet.transformer.multi_token_prediction import (
                    extract_local_zigzag_chunks as _extract_cp,
                )

                _cp_rank_for_extract = _get_cp_rank()
            else:
                _extract_cp = None
                _cp_rank_for_extract = 0
            if _mtp_is_megatron:
                lm_labels = labels_ori
                if _cp_size_for_extract > 1:
                    # Extract this rank's local zigzag chunks from full-length labels.
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
                        if _cp_size_for_extract > 1:
                            # Match local logits shape by extracting this
                            # rank's zigzag chunks.
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
                            # already local zigzag chunks (extract_local_zigzag_chunks
                            # above), so skip the scatter to avoid double-scatter.
                            labels_cur_depth = ContextParallelScatterOp.apply(
                                labels_cur_depth,
                                axis=1,
                                mode=self.config.cp_balance_mode,
                            )

                        if self.config.fused_linear_ce_loss_chunk > 0:
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
                        return main_loss + loss - loss.detach()
                    else:
                        return main_loss
                else:
                    # Original behavior
                    if self.config.add_mtp_loss:
                        return main_loss + loss
                    else:
                        return main_loss + loss - loss.detach()

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
                        for mtp_l in mtp_loss:
                            mtp_val = (
                                self.config.mtp_loss_scaling_factor
                                * mtp_l
                                / num_mtp
                            )
                            loss = add_loss(loss, mtp_val)
                else:
                    # Original behavior: always use add_loss
                    num_mtp = len(mtp_loss)
                    for mtp_l in mtp_loss:
                        loss = add_loss(
                            loss,
                            self.config.mtp_loss_scaling_factor
                            * mtp_l
                            / num_mtp,
                        )
            else:
                loss = add_loss(
                    lm_loss,
                    self.config.mtp_loss_scaling_factor
                    * sum(mtp_loss)
                    / len(mtp_loss),
                )

            return loss
        else:
            return self._forward(logits, labels)

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

        # Store detached MTP loss tensors into class-level tracker and global_training_logs.
        # Use .detach() instead of .item() to avoid GPU synchronization on every
        # micro-batch. The trainer will call .item() only at logging steps.
        for i, loss_val in enumerate(mtp_loss):
            MainLanguageLoss.mtp_loss_tracker[f"mtp_{i + 1}_loss"] = (
                loss_val.detach()
            )
            _print_scalar_loss_md5(
                "MTP_LOSS_PATH_MD5",
                f"mtp{i + 1}.final_loss",
                loss_val,
            )

        # Also write to global_training_logs to read
        logs = get_global_training_logs()
        if logs is not None and hasattr(logs, "update"):
            for i, loss_val in enumerate(mtp_loss):
                logs.update(**{f"mtp_{i + 1}_loss": loss_val.detach()})

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

        loss = add_loss(
            lm_loss,
            self.config.mtp_loss_scaling_factor * sum(mtp_loss) / len(mtp_loss),
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
