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
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

# (TODO): need adapt to flex_checkpoint
# dist_checkpoint in paddle is flex_checkpoint which have many difference.
# from paddlefleet.dist_checkpointing import ShardedTensor
# from paddlefleet.dist_checkpointing.mapping import (
#     ReplicaId,
#     ShardedStateDict,
#     ShardedTensorFactory,
# )
from paddlefleet.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from paddlefleet.fusions.fused_bias_gelu import bias_gelu_impl
from paddlefleet.fusions.fused_bias_swiglu import (
    bias_swiglu_impl,
    weighted_bias_swiglu_impl,
)
from paddlefleet.train_infer_consistent_ops.inspect_util import (
    get_current_layer,
    inspect_tensor,
)
from paddlefleet.transformer.activations import situ, situ_glu
from paddlefleet.transformer.dw_overlap import deferrable_linear
from paddlefleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig
from paddle.distributed.flex_checkpoint.aoa.generation import (
    format_dtype_cast_attr,
    format_inv_dtype_cast_attr,
    join_name,
    resolve_checkpoint_name_from_anchor,
    resolve_dtype_cast_rule,
    resolve_names,
    resolve_single_name,
    should_skip,
    strip_name_suffix,
)

from paddlefleet.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)


# pylint: disable=missing-class-docstring
@dataclass
class MLPSublayersSpec:
    """
    The dataclass for LayerSpecs of MLP sublayers_spec
    including  linear fc1, activation function, linear fc2.
    """

    up_gate_proj: LayerSpec | type = None
    hidden_act: LayerSpec | type = None
    down_proj: LayerSpec | type = None


class MLP(FleetLayer):
    # p2p_overlap_dw_calc 的延后点名。基类留空表示"这个调用点没有延后点"，
    # 子类（目前只有 StandardMLPSharedExpert）覆盖成具体名字。
    _dw_up_gate_point = None
    _dw_down_point = None

    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.use_bias is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLPSublayersSpec,
        is_expert: bool = False,
        input_size: int | None = None,
        intermediate_size: int | None = None,
        hidden_size: int | None = None,
        tp_group=None,
        disable_fp8: bool = False,
        inspect_name: str = "moe_shared",
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.use_accuracy_compatible = getattr(
            config, "use_accuracy_compatible", False
        )

        self.inspect_name = inspect_name

        self.input_size = (
            input_size if input_size is not None else self.config.hidden_size
        )

        tp_group = get_tensor_model_parallel_group_if_none(
            tp_group, is_expert=is_expert
        )
        if intermediate_size is None:
            if is_expert:
                raise ValueError(
                    "MoE MLP requires `intermediate_size`, but it was not provided."
                )
            warnings.warn(
                "MLP requires intermediate_size, but it was not provided. Using \
                    config.intermediate_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.config.intermediate_size is None:
                raise ValueError(
                    "MLP requires `config.intermediate_size` is not None, but it got None."
                )

            intermediate_size = self.config.intermediate_size

        self.hidden_size = (
            hidden_size if hidden_size is not None else self.config.hidden_size
        )
        skip_bias_add = (
            True
            if not self.config.gpt_model_use_experimental_version
            else False
        )

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            intermediate_size *= 2
        self.up_gate_proj = build_spec_layer(
            sublayers_spec.up_gate_proj,
            self.input_size,
            intermediate_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_group=tp_group,
            disable_fp8=disable_fp8,
        )

        # Ensure hidden_act is a callable function, not a bound method.
        # A spec-level ``hidden_act`` wins over ``config.hidden_act``: modules
        # such as the Qwen3-VL / Qwen3.5 patch merger and the Kimi-K2.5 tpool
        # merge declare their own activation in ``MLPSublayersSpec`` because it
        # differs from the model-wide ``config.hidden_act``.
        hidden_act_value = (
            sublayers_spec.hidden_act
            if sublayers_spec.hidden_act is not None
            else self.config.hidden_act
        )
        if hasattr(hidden_act_value, "__self__") and hasattr(
            hidden_act_value, "__func__"
        ):
            # If it's a bound method, use the unbound function
            self.hidden_act = hidden_act_value.__func__
        else:
            self.hidden_act = hidden_act_value

        if self.config.gated_linear_unit:
            intermediate_size //= 2

        self.down_proj = build_spec_layer(
            sublayers_spec.down_proj,
            intermediate_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_group=tp_group,
            disable_fp8=disable_fp8,
        )

    def muon_slice_specs(self, muon_configs):
        """Muon orthogonal-slice spec for the fused gate/up projection.

        Inherited by StandardMLPExpert / StandardMLPSharedExpert, so each expert
        (auto-prefixed by the module tree) gets its own spec. The gate/up split
        point is derived from the weight shape inside ``ortho_gate_up``.
        """
        from paddlefleet.transformer.muon_utils import ortho_gate_up

        if not self.config.gated_linear_unit or not muon_configs.get(
            "muon_ffn_split", False
        ):
            return {}
        return {"up_gate_proj.weight": (ortho_gate_up, {})}

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="up_gate_proj")
        intermediate_parallel, bias_parallel = deferrable_linear(
            self.config,
            self._dw_up_gate_point,
            self.up_gate_proj,
            hidden_states,
        )
        nvtx_range_pop(suffix="up_gate_proj")

        intermediate_parallel = inspect_tensor(
            f"{self.inspect_name}_ffn1_output",
            get_current_layer(),
            intermediate_parallel,
        )

        nvtx_range_push(suffix="activation")

        # Alignment mode: use Paddle native F.swiglu
        _use_paddle_swiglu = getattr(
            self.config, "gpt_model_use_experimental_version", False
        )
        if (
            self.config.use_bias
            and self.config.gpt_model_use_experimental_version
            and self.config.tensor_model_parallel_size == 1
            and self.hidden_act != situ
        ):
            hidden_states = paddle.incubate.nn.functional.fused_linear(
                hidden_states, self.up_gate_proj.weight, self.up_gate_proj.bias
            )
            hidden_states = F.swiglu(hidden_states)
            output = paddle.incubate.nn.functional.fused_linear(
                hidden_states, self.down_proj.weight, self.down_proj.bias
            )
            return output, None

        if self.hidden_act == situ and self.config.gated_linear_unit:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = situ_glu(
                intermediate_parallel,
                beta=self.config.activation_situ_beta,
                linear_beta=self.config.activation_situ_linear_beta,
                situ_glu_plain_fusion=getattr(
                    self.config, "situ_glu_plain_fusion", False
                ),
            )
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = (
                    intermediate_parallel * per_token_scale.unsqueeze(-1)
                )
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif (
            _use_paddle_swiglu
            and self.hidden_act == F.silu
            and self.config.gated_linear_unit
        ):
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = F.swiglu(intermediate_parallel)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.hidden_act == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        getattr(
                            self.config,
                            "activation_func_fp8_input_store",
                            False,
                        ),
                        self.config.activation_func_clamp_value,
                        use_accuracy_compatible=self.use_accuracy_compatible,
                    )
                elif (
                    self.hidden_act == quick_gelu
                    and self.config.gated_linear_unit
                ):
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        getattr(
                            self.config,
                            "activation_func_fp8_input_store",
                            False,
                        ),
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.hidden_act == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.use_bias is True
                        intermediate_parallel = bias_gelu_impl(
                            intermediate_parallel, bias_parallel
                        )
                elif (
                    self.hidden_act == F.silu and self.config.gated_linear_unit
                ):
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        fp8_input_store=getattr(
                            self.config,
                            "activation_func_fp8_input_store",
                            False,
                        ),
                        cpu_offload_input=False,
                        clamp_value=self.config.activation_func_clamp_value,
                        use_accuracy_compatible=self.use_accuracy_compatible,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = paddle.chunk(x, 2, axis=-1)
                    if (
                        val := self.config.activation_func_clamp_value
                    ) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.hidden_act(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.hidden_act(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = (
                    intermediate_parallel * per_token_scale.unsqueeze(-1)
                )
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        intermediate_parallel = inspect_tensor(
            f"{self.inspect_name}_swiglu_output",
            get_current_layer(),
            intermediate_parallel,
        )

        # [s, b, h]
        nvtx_range_push(suffix="down_proj")
        output, output_bias = deferrable_linear(
            self.config,
            self._dw_down_point,
            self.down_proj,
            intermediate_parallel,
        )
        nvtx_range_pop(suffix="down_proj")
        output = inspect_tensor(
            f"{self.inspect_name}_ffn2_output", get_current_layer(), output
        )

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    def backward_dw(self):
        self.down_proj.backward_dw()
        self.up_gate_proj.backward_dw()

    def gen_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        """Checkpoint->model generator for gated MLP.

        ``up_gate_proj`` fuses checkpoint ``gate_proj`` and ``up_proj`` (each
        transposed) via the ``fused_ffn`` macro; ``down_proj`` is delegated to
        the Linear family. Any own directly-held param/buffer of this MLP (not a
        sublayer) is caught by a generic fallback loop between the fused body
        and the ``down_proj`` delegation, mirroring ``SelfAttention``. Statement
        order is semantic. Inverse is implemented independently in
        :meth:`gen_inv_aoa_statements` (no cross-call).
        """
        if not ctx.mlp_gate_up_fused:
            return self._gen_nongated_mlp_aoa_statements(
                ctx, structured_name_prefix, aoa_name_scope
            )

        fused_local_name = "up_gate_proj.weight"
        fused_model_name = resolve_single_name(
            fused_local_name,
            structured_name_prefix,
            ctx.pp_to_single_mapping,
            model_name_prefix=ctx.model_name_prefix,
        )
        gate_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "gate_proj.weight",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        up_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "up_proj.weight",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        # ``excluded_names`` holds model names another statement already
        # produces, so re-emitting one here would duplicate the target. A
        # fusion is all-or-nothing: the whole statement goes or stays. Weight
        # and bias are excluded independently.
        if structured_name_prefix + fused_local_name in ctx.excluded_names:
            statements = []
        else:
            statements = self._gen_up_gate_fusion_aoa_statements(
                gate_checkpoint_name, up_checkpoint_name, fused_model_name
            )
        if self.up_gate_proj.bias is not None:
            statements += self._gen_up_gate_bias_aoa_statements(
                ctx, structured_name_prefix, aoa_name_scope
            )
        local_state_dict = self.state_dict(
            structured_name_prefix="", include_sublayers=False
        )
        for name in local_state_dict:
            if structured_name_prefix + name in ctx.excluded_names:
                continue
            source_name, target_name = resolve_names(
                name,
                ctx.checkpoint_name_prefix,
                structured_name_prefix,
                ctx.pp_to_single_mapping,
                ctx.checkpoint_name_mapping,
                aoa_name_scope=aoa_name_scope,
                model_name_prefix=ctx.model_name_prefix,
            )
            cast = format_dtype_cast_attr(
                resolve_dtype_cast_rule(
                    target_name,
                    ctx.dtype_cast_rules,
                    ctx.model_name_prefix,
                )
            )
            if should_skip(source_name, target_name, cast):
                continue
            statements.append(f"{source_name} -> {target_name}{cast}")
        statements += self.down_proj.gen_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}down_proj.",
            aoa_name_scope=aoa_name_scope,
        )
        return statements

    def gen_inv_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        """Inverse (model -> checkpoint) generator for gated MLP.

        Independently splits the fused ``up_gate_proj`` back into checkpoint
        ``gate_proj`` and ``up_proj`` via ``fused_ffn``, catches any own
        directly-held param/buffer via a generic fallback loop, then delegates
        ``down_proj``. Not derived from the checkpoint->model text. The inverse
        ``fused_ffn`` carries no ``^T`` (matches the validated existing
        generators in ``aoa_config_base``/``qwen3_moe``/``glm4_moe``; the fused
        macro handles the layout, so no explicit inverse transpose is emitted).
        """
        if not ctx.mlp_gate_up_fused:
            return self._gen_inv_nongated_mlp_aoa_statements(
                ctx, structured_name_prefix, aoa_name_scope
            )

        fused_local_name = "up_gate_proj.weight"
        fused_model_name = resolve_single_name(
            fused_local_name,
            structured_name_prefix,
            ctx.pp_to_single_mapping,
            model_name_prefix=ctx.model_name_prefix,
        )
        # Split targets use the model-side gate/up half names, derived from the
        # fused anchor's scope (never sent through pp_to_single_mapping, since
        # the halves are not real params under the fused up_gate_proj). Same
        # anchor-scope mechanism used for the checkpoint names below.
        scope_single = strip_name_suffix(fused_model_name, fused_local_name)
        gate_model_name = join_name(scope_single, "gate_proj.weight")
        up_model_name = join_name(scope_single, "up_proj.weight")
        gate_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "gate_proj.weight",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        up_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "up_proj.weight",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        if structured_name_prefix + fused_local_name in ctx.excluded_names:
            statements = []
        else:
            statements = self._gen_inv_up_gate_fusion_aoa_statements(
                fused_model_name,
                gate_model_name,
                up_model_name,
                gate_checkpoint_name,
                up_checkpoint_name,
            )
        if self.up_gate_proj.bias is not None:
            statements += self._gen_inv_up_gate_bias_aoa_statements(
                ctx, structured_name_prefix, aoa_name_scope
            )
        local_state_dict = self.state_dict(
            structured_name_prefix="", include_sublayers=False
        )
        for name in local_state_dict:
            if structured_name_prefix + name in ctx.excluded_names:
                continue
            target_name, source_name = resolve_names(
                name,
                ctx.checkpoint_name_prefix,
                structured_name_prefix,
                ctx.pp_to_single_mapping,
                ctx.checkpoint_name_mapping,
                aoa_name_scope=aoa_name_scope,
                model_name_prefix=ctx.model_name_prefix,
            )
            cast = format_inv_dtype_cast_attr(
                resolve_dtype_cast_rule(
                    source_name,
                    ctx.dtype_cast_rules,
                    ctx.model_name_prefix,
                )
            )
            if should_skip(source_name, target_name, cast):
                continue
            statements.append(f"{source_name} -> {target_name}{cast}")
        statements += self.down_proj.gen_inv_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}down_proj.",
            aoa_name_scope=aoa_name_scope,
        )
        return statements

    def _gen_nongated_mlp_aoa_statements(
        self, ctx, structured_name_prefix, aoa_name_scope
    ):
        """Checkpoint->model for a non-gated MLP (e.g. Qwen3-VL vision tower):
        ``up_gate_proj`` / ``down_proj`` are plain Linears with no gate/up
        split, so delegate both to the generic Linear family (per-tensor ``^T``
        weight rename + identity bias) and emit no ``fused_ffn``."""
        statements = self.up_gate_proj.gen_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}up_gate_proj.",
            aoa_name_scope=aoa_name_scope,
        )
        statements += self.down_proj.gen_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}down_proj.",
            aoa_name_scope=aoa_name_scope,
        )
        return statements

    def _gen_inv_nongated_mlp_aoa_statements(
        self, ctx, structured_name_prefix, aoa_name_scope
    ):
        """Model->checkpoint for a non-gated MLP. Independently mirrors
        :meth:`_gen_nongated_mlp_aoa_statements` by delegating both plain
        Linears to their inverse generators."""
        statements = self.up_gate_proj.gen_inv_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}up_gate_proj.",
            aoa_name_scope=aoa_name_scope,
        )
        statements += self.down_proj.gen_inv_aoa_statements(
            ctx,
            structured_name_prefix=f"{structured_name_prefix}down_proj.",
            aoa_name_scope=aoa_name_scope,
        )
        return statements

    def _gen_up_gate_fusion_aoa_statements(
        self, gate_checkpoint_name, up_checkpoint_name, fused_model_name
    ):
        """Checkpoint->model fusion of gate/up into ``up_gate_proj``.

        Default uses the ``fused_ffn`` macro, whose TP-rank interleaving is
        required for dense and shared MLPs that are tensor-parallel sharded.
        Routed experts are expert-parallel (their fused weight is not
        TP-interleaved), so they override this with a plain ``axis=1`` concat.
        """
        return [
            f"{gate_checkpoint_name}^T, {up_checkpoint_name}^T "
            f"-> {fused_model_name}, fused_ffn"
        ]

    def _gen_inv_up_gate_fusion_aoa_statements(
        self,
        fused_model_name,
        gate_model_name,
        up_model_name,
        gate_checkpoint_name,
        up_checkpoint_name,
    ):
        """Model->checkpoint split of ``up_gate_proj`` back into gate/up.

        Split the fused weight via the TP-interleaving ``fused_ffn`` macro onto
        the model-side gate/up half names (still in model ``(out,in)`` layout),
        then transpose each half back to its checkpoint ``(in,out)`` name. The
        transpose is a separate statement because ``^T`` binds to the source
        (left) side while ``fused_ffn`` emits its halves on the target side.
        Routed experts override only the fusion macro (``axis=1``); the
        split-then-transpose shape is identical.
        """
        return [
            f"{fused_model_name} -> "
            f"{gate_model_name}, {up_model_name}, fused_ffn",
            f"{gate_model_name}^T -> {gate_checkpoint_name}",
            f"{up_model_name}^T -> {up_checkpoint_name}",
        ]

    def _gen_up_gate_bias_fusion_aoa_statements(
        self, gate_checkpoint_name, up_checkpoint_name, fused_model_name
    ):
        """Checkpoint->model fusion of the gate/up 1-D bias.

        Same TP-interleaving requirement as the weight, on the single bias axis
        (``axis=0``); routed experts are expert-parallel and override this with
        a plain concat.
        """
        return [
            f"{gate_checkpoint_name}, {up_checkpoint_name} "
            f"-> {fused_model_name}, fused_ffn, axis=0"
        ]

    def _gen_inv_up_gate_bias_fusion_aoa_statements(
        self, fused_model_name, gate_checkpoint_name, up_checkpoint_name
    ):
        """Model->checkpoint split of the fused 1-D bias.

        No transpose on either side, so unlike the weight this is a single
        statement straight onto the checkpoint names.
        """
        return [
            f"{fused_model_name} -> "
            f"{gate_checkpoint_name}, {up_checkpoint_name}, fused_ffn, axis=0"
        ]

    def _gen_up_gate_bias_aoa_statements(
        self, ctx, structured_name_prefix, aoa_name_scope
    ):
        """Checkpoint->model helper: fuse checkpoint gate/up 1-D bias into the
        model fused bias (no transpose). Anchored on the real
        ``up_gate_proj.bias`` key."""
        fused_local_name = "up_gate_proj.bias"
        if structured_name_prefix + fused_local_name in ctx.excluded_names:
            return []
        fused_model_name = resolve_single_name(
            fused_local_name,
            structured_name_prefix,
            ctx.pp_to_single_mapping,
            model_name_prefix=ctx.model_name_prefix,
        )
        gate_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "gate_proj.bias",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        up_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "up_proj.bias",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        return self._gen_up_gate_bias_fusion_aoa_statements(
            gate_checkpoint_name, up_checkpoint_name, fused_model_name
        )

    def _gen_inv_up_gate_bias_aoa_statements(
        self, ctx, structured_name_prefix, aoa_name_scope
    ):
        """Inverse-only helper: split the model fused 1-D bias back into
        checkpoint gate/up bias (no transpose)."""
        fused_local_name = "up_gate_proj.bias"
        if structured_name_prefix + fused_local_name in ctx.excluded_names:
            return []
        fused_model_name = resolve_single_name(
            fused_local_name,
            structured_name_prefix,
            ctx.pp_to_single_mapping,
            model_name_prefix=ctx.model_name_prefix,
        )
        gate_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "gate_proj.bias",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        up_checkpoint_name = resolve_checkpoint_name_from_anchor(
            fused_model_name,
            fused_local_name,
            "up_proj.bias",
            ctx.checkpoint_name_prefix,
            ctx.checkpoint_name_mapping,
            aoa_name_scope=aoa_name_scope,
            model_name_prefix=ctx.model_name_prefix,
        )
        return self._gen_inv_up_gate_bias_fusion_aoa_statements(
            fused_model_name, gate_checkpoint_name, up_checkpoint_name
        )
