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

# =============================================================================
# Multimax lm_head support
# -----------------------------------------------------------------------------
# When TransformerConfig.multimax_modules is a list containing "lm_head" (e.g.,
# ``multimax_modules: [lm_head]``), GPTLMHead adds two learnable [4]-shape
# parameters (multimax_ranges, multimax_ts) and applies a SegLU-style segmented
# modulation to logits before softmax/cross-entropy:
#
#     logits = SegLU(logits, multimax_ranges, multimax_ts)
#
# Both params are initialized to zero on cold start, which makes SegLU the
# identity at step 0 (safe for resume from checkpoints lacking these keys).
# Resumed training restores the trained values from the checkpoint -- the
# zero-init only applies to fresh runs.
#
# The "multimax" substring in the parameter names is matched by the
# pretraining trainers' no-weight-decay name filter, so these params are
# excluded from weight decay (timm-style convention).
#
# Sanity-check after a training launch:
#
#     grep -E "MULTIMAX-(CONFIG|LMHEAD-CONFIRM|LMHEAD-APPLIED)" <train.log>
#
# Expected output (per rank holding the LM head):
#   [MULTIMAX-CONFIG]            multimax_modules=['lm_head']
#   [MULTIMAX-LMHEAD-CONFIRM]    cls=... multimax_modules=['lm_head'] ranges.shape=[4] ...
#   [MULTIMAX-LMHEAD-APPLIED]    cls=... logits.shape=[...] ranges=[...] ts=[...]
#                                (or path=fused for the fused-CE branch)
#
# If [MULTIMAX-LMHEAD-APPLIED] is missing, the SegLU path did not execute
# (e.g., a different LM head class is used).
#
# Paths:
#   - fused_linear_ce_loss_chunk == 0: SegLU is applied here on the [B, S, V]
#     logits tensor right before the cross-entropy. Wrapped in `recompute` and
#     uses an in-place `+=` accumulator to bound peak activation memory.
#   - fused_linear_ce_loss_chunk  > 0: GPTLMHead emits a 5-tuple
#     (hidden, weight, bias, multimax_ranges, multimax_ts) and SegLU is
#     applied inside each CE chunk by LigerFusedLinearCrossEntropyFunction.
#     This avoids ever materializing the full [B*S, V] logits tensor and is
#     the recommended path at large vocab.
#
# Notes:
#   - SegLU is element-wise, so vocab-TP sharding is fine without collectives.
#   - Both paths share the same parameters (multimax_ranges/multimax_ts) and
#     produce identical math at step 0 (params init to zero -> SegLU = id).
# =============================================================================

import warnings

import paddle
from paddle.distributed.fleet.meta_parallel import (
    ScheduleNode,
)
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.flex_checkpoint.aoa.generation import (
    join_name,
    resolve_checkpoint_name_from_anchor,
    resolve_single_name,
    strip_name_suffix,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)

from paddlefleet.recompute_utils import module_needs_recompute
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from paddlefleet.train_infer_consistent_ops.inspect_util import inspect_tensor
from paddlefleet.utils import use_dsv4_accuracy_compatible

# Model-side head segment every LM head instance normalizes to before its
# checkpoint name is looked up. A model has at most one LM head tensor in the
# checkpoint, but the live tree names the head after the role it plays --
# ``lm_head``, ``shared_head`` once pipeline sharing splits it off, or
# ``shared_mtp_lm_head`` when MTP gets a head of its own -- and those are Fleet
# wiring facts, not checkpoint layout. Collapsing them here keeps the layout
# question ("what is the head called in the checkpoint?") answerable by one
# ``checkpoint_name_mapping`` entry per tensor instead of one per alias, and the
# collapsed name is a model-side name, so it stays a mapping key rather than a
# checkpoint name hardcoded in a component.
_AOA_CANONICAL_HEAD_SEGMENT = "lm_head"

# Model-side name of the token embedding weight relative to the root
# ``model_name_prefix`` declares. This is the live name the embedding leaf
# itself is resolved under, so a replica of that weight looks its checkpoint
# name up with the very same key and the two cannot drift apart.
CANONICAL_EMBEDDING_LOCAL_NAME = "embedding.embed_tokens.weight"


def resolve_embedding_checkpoint_name(ctx) -> str:
    """Resolves the checkpoint name of the token embedding weight.

    A replica of the embedding -- a tied LM head, an MTP block's own embedding
    -- has to name the single checkpoint tensor it is filled from, and that
    tensor is named after the embedding itself, not after the replica's position
    in the tree. Resolving it from the embedding's own model name keeps every
    replica on whatever layout ``checkpoint_name_mapping`` declares, so a model
    whose checkpoint calls it something else states that once, in the same entry
    the embedding leaf goes through, and every replica follows.

    Both directions call this: it only computes a name, so neither direction's
    statements are derived from the other's.
    """
    single_name = join_name(
        ctx.model_name_prefix, CANONICAL_EMBEDDING_LOCAL_NAME
    )
    return resolve_checkpoint_name_from_anchor(
        single_name,
        CANONICAL_EMBEDDING_LOCAL_NAME,
        CANONICAL_EMBEDDING_LOCAL_NAME,
        ctx.checkpoint_name_prefix,
        ctx.checkpoint_name_mapping,
        model_name_prefix=ctx.model_name_prefix,
    )


def SegLU(x, ranges, ts):
    """Learnable segmented activation applied to logits before softmax.

    Named ``SegLU`` ("segmented learnable unit") to avoid confusion with the
    framework's built-in ``SELU`` activation.

    Reference: ViT-style modulation provided by user.
        x: [..., V] logits (last dim may be vocab-parallel sharded; SegLU is
           element-wise so applying it on the sharded logits is equivalent to
           applying it on gathered logits).
        ranges: [4] learnable thresholds.
        ts:     [4] learnable scales.

    With ranges/ts initialized to zero, SegLU is the identity (logits unchanged),
    so adding this op is a no-op at start of training and safe for resume.

    Memory-efficient eager form: instead of building one giant expression
    ``out = x + a + b + c + d`` (which materializes 4 sum-intermediates each
    the size of logits), we accumulate into a single owned buffer with ``+=``.
    Combined with ``recompute`` at the call site, peak activation memory for
    this op drops from ~10x the logits tensor to ~1x.

    NOTE on jit fusion: an earlier version was wrapped with
    ``@paddle.jit.to_static`` to fuse the element-wise chain. That backend
    (CINN) was observed to OOM on full-vocab logits because CINN allocates
    all traced intermediates inside a single ``CinnJitInstruction::Run`` call,
    which defeats both ``recompute`` and the in-place ``+=`` hints. We use
    the eager path here.
    """
    relu = paddle.nn.functional.relu
    # Clone once so we own the buffer and in-place += is safe under autograd
    # (the original `x` -- the linear's output -- must not be mutated, since
    # it may still be referenced by the autograd graph upstream).
    out = x.clone()
    out += ts[0] * relu(ranges[0] - x)
    out += ts[1] * relu(x - ranges[1])
    out += ts[2] * relu(ranges[2] - x) ** 2
    out += ts[3] * relu(x - ranges[3]) ** 2
    return out


class GPTLMHead(ColumnParallelLinear):
    # Whether this head is the one that writes the head tensor to the
    # checkpoint. Several heads can hold the same tensor (``GPTMainLMHead`` and
    # ``GPTMTPLMHead`` under ``separate_mtp_headloss``) and the checkpoint keeps
    # one copy, so exactly one of them writes it. Loading is unaffected: every
    # head reads that one tensor. This head is the sole one on its own, so it
    # writes; the two roles that come as a pair elect their writer below.
    _aoa_writes_checkpoint_tensor = True

    def __init__(self, **kwargs):
        # Force-disable FP8 on the LM head.
        kwargs["disable_fp8"] = True
        self.config = kwargs["config"]
        self.skip_weight_param_allocation = kwargs[
            "skip_weight_param_allocation"
        ]
        self._dtype = self.config.params_dtype

        kwargs.pop("block_attn_res", None)

        kwargs["skip_weight_param_allocation"] = True
        if self.config.gpt_model_use_experimental_version:
            kwargs["bias"] = self.config.use_bias
        super().__init__(**kwargs)

        stride = kwargs["stride"] if "stride" in kwargs.keys() else 1
        init_method = kwargs["init_method"]
        keep_master_weight_for_test = (
            kwargs["keep_master_weight_for_test"]
            if "keep_master_weight_for_test" in kwargs.keys()
            else False
        )

        if not self.skip_weight_param_allocation:
            if self.config.use_cpu_initialization:
                self.weight = self.create_parameter(
                    shape=[self.output_size_per_partition, self.input_size],
                    dtype=self.config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                if self.config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=self.rank,
                        world_size=self.world_size,
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[self.output_size_per_partition, self.input_size],
                    dtype=self.config.params_dtype,
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )

                if self.config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )
            self.weight.is_distributed = True if self.world_size > 1 else False

        # Multimax: learnable SegLU-style modulation on logits before softmax.
        # Names contain the "multimax" substring so the trainer's no-decay
        # filter excludes them from weight decay (mirrors timm's convention
        # of not decaying scalar/1-D learnable coefficients).
        multimax_mode = getattr(self.config, "multimax_modules", None) or []
        self.use_multimax_lmhead = "lm_head" in multimax_mode
        if self.use_multimax_lmhead:
            # Cold-start init to zero -> SegLU is identity at step 0, so the
            # untrained model produces bit-identical logits with/without the
            # feature flag and resuming from a checkpoint that lacks these
            # params (loaded non-strictly) is safe. Resumed training restores
            # the trained values from the checkpoint -- the zero init only
            # applies on cold start.
            self.multimax_ranges = self.create_parameter(
                shape=[4],
                dtype=self.config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            self.multimax_ts = self.create_parameter(
                shape=[4],
                dtype=self.config.params_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            # Grep-friendly construction banner. See docstring at top of file
            # for the full sanity-check workflow.
            warnings.warn(
                f"[MULTIMAX-LMHEAD-CONFIRM] cls={type(self).__name__} "
                f"multimax_modules={multimax_mode} "
                f"ranges.shape={list(self.multimax_ranges.shape)} "
                f"ts.shape={list(self.multimax_ts.shape)} "
                f"dtype={self.config.params_dtype} "
                f"fused_linear_ce_loss_chunk="
                f"{getattr(self.config, 'fused_linear_ce_loss_chunk', 0)}"
            )
        # When the feature is disabled (default None / empty), stay silent --
        # no banner, no params -- so default training runs are not noisy.

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTLMHead")

    def _forward(self, hidden_states: paddle.Tensor):
        use_sequence_first_linear = (
            use_dsv4_accuracy_compatible() and not self.config.sequence_parallel
        )
        if use_sequence_first_linear:
            hidden_states = hidden_states.transpose([1, 0, 2]).contiguous()
        # Fused linear + cross-entropy path: skip materializing [B, S, V] logits
        # and delegate the linear projection into LanguageLoss, which will call
        # LigerFusedLinearCrossEntropyFunction.
        if getattr(self.config, "fused_linear_ce_loss_chunk", 0):
            if self.config.sequence_parallel:
                # [S, B, H] -> [B, S, H] to match the logits layout consumers expect.
                hidden_states = hidden_states.transpose([1, 0, 2]).contiguous()

            # Multimax lm_head + fused path: thread multimax_ranges/ts into
            # LanguageLoss so SegLU is applied inside each CE chunk and never
            # materializes a full [B*S, V] logits tensor. Also fire the same
            # one-shot [MULTIMAX-LMHEAD-APPLIED] grep banner used by the
            # unfused path so observability is consistent.
            if getattr(self, "use_multimax_lmhead", False):
                if not getattr(self, "_multimax_applied_logged", False):
                    warnings.warn(
                        f"[MULTIMAX-LMHEAD-APPLIED] cls={type(self).__name__} "
                        f"path=fused hidden.shape={list(hidden_states.shape)} "
                        f"ranges={self.multimax_ranges.detach().cast('float32').numpy().tolist()} "
                        f"ts={self.multimax_ts.detach().cast('float32').numpy().tolist()}"
                    )
                    self._multimax_applied_logged = True
                return (
                    hidden_states,
                    self.weight,
                    self.bias,
                    self.multimax_ranges,
                    self.multimax_ts,
                )

            return (hidden_states, self.weight, self.bias)

        hidden_states = inspect_tensor("lm_head_input", -1, hidden_states)

        if module_needs_recompute("lm_head", None, self.config):
            recompute_func = super().forward

            def recompute_handler(hidden_states, weight):
                logits, _ = recompute_func(hidden_states, weight)
                return logits

            logits = recompute_handler(hidden_states, self.weight.T)
        else:
            logits, _ = super().forward(hidden_states, self.weight.T)
        if use_sequence_first_linear:
            logits = logits.transpose([1, 0, 2]).contiguous()
        if (
            not self.config.gpt_model_use_experimental_version
            and self.config.sequence_parallel
        ):
            logits = logits.transpose([1, 0, 2]).contiguous()

        logits = inspect_tensor("lm_head_logits", -1, logits)

        # Multimax lm_head (unfused path): apply learnable SegLU modulation on
        # logits before softmax/cross-entropy. SegLU is element-wise so it works
        # correctly on vocab-parallel sharded logits without extra collectives.
        # The fused path (fused_linear_ce_loss_chunk > 0) returns early above
        # with a 5-tuple and applies SegLU inside the chunked CE kernel.
        if getattr(self, "use_multimax_lmhead", False):
            # One-shot confirmation (per rank) that the SegLU path actually
            # executed at runtime. Grep tag: [MULTIMAX-LMHEAD-APPLIED].
            if not getattr(self, "_multimax_applied_logged", False):
                warnings.warn(
                    f"[MULTIMAX-LMHEAD-APPLIED] cls={type(self).__name__} "
                    f"logits.shape={list(logits.shape)} "
                    f"ranges={self.multimax_ranges.detach().cast('float32').numpy().tolist()} "
                    f"ts={self.multimax_ts.detach().cast('float32').numpy().tolist()}"
                )
                self._multimax_applied_logged = True
            # Wrap SegLU in `recompute` to drop saved intermediates: only the
            # input logits (already kept by upstream autograd) are saved; SegLU
            # is re-run during backward. Avoids OOM on full-vocab logits when
            # `fused_linear_ce_loss_chunk=0` (required by multimax).
            logits = recompute(
                SegLU, logits, self.multimax_ranges, self.multimax_ts
            )
            logits = inspect_tensor("lm_head_seglu_output", -1, logits)

        return logits

    def _stash_cu_seqlens_q(self, dict_args):
        """Deliver cu_seqlens_q to the loss stage under use_erndata.

        ``LanguageLoss.forward`` only receives ``(logits, labels)`` and cannot
        see the pipeline dict. ``cu_seqlens_q`` (packed-doc boundaries) travels
        down the pipeline dict to this LM head, which runs on the LAST pipeline
        stage immediately before the loss. Stash it here — on the loss rank,
        per micro-batch — so ``LanguageLoss`` can roll labels per-doc even under
        PP>1 (GPTEmbedding.forward writes the same stash on the PP=1 / first
        stage). No external dataloader broadcast is required.

        Non-erndata runs never populate ``cu_seqlens_q`` (it is stripped from
        the dict in GPTEmbedding.forward), so this is a no-op there.
        """
        cu_seqlens_q = dict_args.get("cu_seqlens_q", None)
        if cu_seqlens_q is not None:
            from paddlefleet.models.common.language_loss.language_loss import (
                LanguageLoss as _LangLoss,
            )

            _LangLoss._cu_seqlens_q_stash = cu_seqlens_q

    def forward(self, dict_args: dict):
        self._stash_cu_seqlens_q(dict_args)
        hidden_states = dict_args["hidden_states"]

        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        ):
            tensor_list = paddle.split(
                hidden_states,
                self.config.num_nextn_predict_layers + 1,
            )
            logits = [self._forward(tensor_list[0])]
            for i in range(self.config.num_nextn_predict_layers):
                logits.append(self._forward(tensor_list[i + 1]))
            return logits
        else:
            return self._forward(hidden_states)

    @property
    def embedding_weight(self):
        return self.weight

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        """Sharding along axis 0, bias sharded.

        Multimax params (multimax_ranges, multimax_ts) are tiny [4]-shape
        replicated tensors; non-sharded parameters do not need to appear in
        ``shard_rules``, so we leave them out and let flex-checkpoint pick
        them up from the state dict directly.
        """
        state_dict = self.state_dict(structured_name_prefix="")
        if self.world_size == 1:
            shard_rules = None
        else:
            shard_rules = {"weight": 0, "bias": 0}
        return build_sharded_state_dict(
            state_dict, shard_rules, structured_name_prefix
        )

    def _resolve_aoa_names(
        self,
        ctx,
        local_name,
        structured_name_prefix,
        checkpoint_lookup_drop_segment,
    ):
        """Resolves one head tensor to its ``(checkpoint_name, model_name)``.

        The model side is the real live name, so a statement targets the key
        this rank actually holds. The checkpoint side is resolved from the
        alias-collapsed name instead (see ``_AOA_CANONICAL_HEAD_SEGMENT``),
        which is what lets the three head aliases share one mapping entry.

        Both directions call this: it only computes names, so neither
        direction's statements are derived from the other's.
        """
        model_name = resolve_single_name(
            local_name,
            structured_name_prefix,
            ctx.pp_to_single_mapping,
            ctx.model_name_prefix,
        )
        # A head sits one segment under its model root, so the segment right
        # before the tensor name is the alias to collapse.
        head_segment = strip_name_suffix(model_name, local_name).rsplit(".", 1)[
            -1
        ]
        checkpoint_name = resolve_checkpoint_name_from_anchor(
            model_name,
            join_name(head_segment, local_name),
            join_name(_AOA_CANONICAL_HEAD_SEGMENT, local_name),
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            model_name_prefix=ctx.model_name_prefix,
            checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
        )
        return checkpoint_name, model_name

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        """Checkpoint->model AOA for an LM head.

        No ``^T``, unlike the rest of the ``ColumnParallelLinear`` family: this
        head stores ``weight`` as ``[output_size_per_partition, input_size]``
        and transposes it in ``forward``, which is already the checkpoint
        layout.

        With ``tie_word_embeddings`` the head has no tensor of its own in the
        checkpoint: ``weight`` comes from the embedding's, and everything else
        the head carries comes from nothing, mirroring the inverse direction,
        which writes none of it. Otherwise every tensor is read from the head's
        checkpoint name -- by each head that holds it, which is how two heads
        sharing one tensor both get filled.
        """
        statements = []
        own_state_dict = self.state_dict(
            structured_name_prefix="", include_sublayers=False
        )
        for local_name in own_state_dict:
            checkpoint_name, model_name = self._resolve_aoa_names(
                ctx,
                local_name,
                structured_name_prefix,
                checkpoint_lookup_drop_segment,
            )
            if not ctx.config.tie_word_embeddings:
                statements.append(f"{checkpoint_name} -> {model_name}")
            elif local_name == "weight":
                statements.append(
                    f"{resolve_embedding_checkpoint_name(ctx)} -> {model_name}"
                )
            else:
                statements.append(f"_ -> {model_name}")
        return statements

    def gen_inv_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        """Inverse (model -> checkpoint) AOA, independently generated.

        Same reason as the forward direction for carrying no ``^T``. A tied head
        is not written at all -- the embedding already wrote that tensor -- and
        when several heads hold the same tensor only the writer emits it, so the
        checkpoint ends up with exactly one copy.
        """
        writes_checkpoint = (
            self._aoa_writes_checkpoint_tensor
            and not ctx.config.tie_word_embeddings
        )
        statements = []
        own_state_dict = self.state_dict(
            structured_name_prefix="", include_sublayers=False
        )
        for local_name in own_state_dict:
            checkpoint_name, model_name = self._resolve_aoa_names(
                ctx,
                local_name,
                structured_name_prefix,
                checkpoint_lookup_drop_segment,
            )
            if writes_checkpoint:
                statements.append(f"{model_name} -> {checkpoint_name}")
            else:
                statements.append(f"{model_name} -> _")
        return statements


class GPTMainLMHead(GPTLMHead):
    """Main LM Head, single prediction."""

    # This role only exists under ``separate_mtp_headloss``, i.e. never without
    # a ``GPTMTPLMHead`` beside it, so it is always the one that can be given
    # up. And it has to be: both roles are pipeline-registered under the one
    # shared key ``"embed"``, so when they land on the same rank only the first
    # desc in the list -- the MTP one -- is ever instantiated, and a writer flag
    # on this class would then belong to an object that does not exist.
    _aoa_writes_checkpoint_tensor = False

    def __init__(self, **kwargs):
        kwargs.pop("block_attn_res", None)
        super().__init__(**kwargs)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTMainLMHead")

    def forward(self, dict_args: dict):
        self._stash_cu_seqlens_q(dict_args)
        hidden_states = dict_args["hidden_states"]
        mtp_loss = dict_args.get("mtp_loss", None)

        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not getattr(self.config, "mtp_load_weight_only", False)
        ):
            tensor_list = paddle.split(
                hidden_states,
                self.config.num_nextn_predict_layers + 1,
            )
            logits = self._forward(tensor_list[0])
        else:
            logits = self._forward(hidden_states)
        ret = {
            "logits": logits,
            "mtp_loss": mtp_loss,
        }
        # Filter out None values to avoid AttributeError in
        # convert_tensor_dict_to_tuple when pipeline stage boundary
        # separates GPTMainLMHead from MTPLanguageLoss
        for key in list(ret.keys()):
            if ret[key] is None:
                ret.pop(key)
        return ret

    @property
    def embedding_weight(self):
        return self.weight


class GPTMTPLMHead(GPTLMHead):
    """MTP LM Head: splits concatenated hidden_states and computes per-MTP logits."""

    # Under ``separate_mtp_headloss`` this head and ``GPTMainLMHead`` hold the
    # same head tensor, and this one writes it. The choice is not free: both are
    # pipeline-registered under the one shared key ``"embed"``, whose earlier
    # desc wins per rank, and this head's desc comes first. So whenever the two
    # land on the same rank this is the only head object that exists, and
    # electing it keeps exactly one writer in every layout.
    _aoa_writes_checkpoint_tensor = True

    def __init__(self, **kwargs):
        kwargs.pop("block_attn_res", None)
        super().__init__(**kwargs)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTMTPLMHead")

    def forward(self, dict_args: dict):
        # Under separate_mtp_headloss this head runs on the last stage BEFORE
        # MTPLanguageLoss, which needs cu_seqlens_q to roll per-depth labels
        # under use_erndata; stash it here on the loss rank.
        self._stash_cu_seqlens_q(dict_args)
        hidden_states = dict_args["hidden_states"]
        num_mtp = self.config.num_nextn_predict_layers
        tensor_list = paddle.split(hidden_states, num_mtp + 1)

        mtp_logits = []
        for i in range(num_mtp):
            mtp_logits.append(self._forward(tensor_list[i + 1]))

        dict_args["mtp_logits"] = mtp_logits
        return dict_args

    @property
    def embedding_weight(self):
        return self.weight
