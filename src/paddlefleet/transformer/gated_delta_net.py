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

# Ported from NVIDIA Megatron-LM megatron/core/ssm/gated_delta_net.py
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd.py_layer import PyLayer
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
)

from paddlefleet.accuracy_target import targets_hf
from paddlefleet.jit import jit_fuser
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import (
    get_pg_rank,
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
)

from .paddle_norm import get_norm_extra_args

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule = None
    HAVE_FLA = False

logger = logging.getLogger(__name__)

# Run the depthwise QKV convolution in FP32 and round once, instead of letting
# cuDNN pick its own BF16 accumulation order. The reference implementation
# (transformers' causal_conv1d_fn fallback) produces bit-for-bit the FP32 sum
# rounded to BF16, while paddle's native BF16 conv1d differs in the last
# mantissa bits on ~37% of elements -- the first divergence in the whole
# forward pass. See paddlefleet/hf_bitexact.py for the opt-in flag.


def _hf_conv1d_wgrad(x, grad_out, kernel_size, padding):
    """Depthwise conv1d weight gradient matching the reference accumulation.

    Each element is ``sum_t grad_out[c, t] * x_padded[c, t + j]``; both operands
    are BF16, so every product is exact in FP32 and only the summation order can
    differ. torch's depthwise wgrad reduces the time axis with 32 lanes -- each
    lane walks its own strided slice, then the 32 partials are combined by a
    binary tree. Paddle's cuDNN path uses a different order and lands one BF16
    ULP away on 1-3 of the 32768 elements per layer, which then propagates.

    Only ever reached from ``_HFCausalConv1d.backward``, i.e. under the ``"hf"``
    accuracy target, so it is written for the shape that mode actually runs:
    micro-batch 1. A weight gradient has to be summed over the batch as well, and
    the reference's reduction order across that axis has not been established
    against a capture, so a larger batch is rejected rather than silently reduced
    in a made-up order -- which would look aligned and not be.
    """
    assert x.shape[0] == 1, (
        "_hf_conv1d_wgrad only supports micro-batch 1, got "
        f"{x.shape[0]}. The HF-aligned depthwise conv weight gradient would "
        "additionally need a batch reduction whose order is not yet pinned "
        "against the reference; run the alignment mode with "
        "micro_batch_size=1, or use the default/Megatron accuracy target."
    )
    channels = x.shape[1]
    out_len = grad_out.shape[-1]
    padded = F.pad(x.astype(paddle.float32), [padding, padding])
    grad_f32 = grad_out.astype(paddle.float32)
    products = paddle.stack(
        [grad_f32 * padded[:, :, j : j + out_len] for j in range(kernel_size)],
        axis=2,
    )
    lanes = 32
    tail = (-out_len) % lanes
    if tail:
        products = F.pad(products, [0, tail])
    rows = products.shape[-1] // lanes
    lane_view = products.reshape([1, channels, kernel_size, rows, lanes])
    acc = lane_view[:, :, :, 0]
    for row in range(1, rows):
        acc = acc + lane_view[:, :, :, row]
    width = lanes
    while width > 1:
        half = width // 2
        acc = acc[..., :half] + acc[..., half:width]
        width = half
    return acc.reshape([channels, 1, kernel_size])


class _HFCausalConv1d(PyLayer):
    """FP32 depthwise conv1d whose weight gradient matches the reference.

    The forward is the reference's ``F.conv1d`` fallback (FP32 accumulation,
    rounded once by the caller). The input gradient is already bit-exact through
    paddle's own conv backward, so it is reused; only the weight gradient needs
    the explicit reduction in ``_hf_conv1d_wgrad``.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, padding, groups):
        ctx.save_for_backward(x, weight)
        ctx.padding = padding
        ctx.groups = groups
        ctx.has_bias = bias is not None
        with paddle.amp.auto_cast(False):
            return F.conv1d(
                x.astype(paddle.float32),
                weight.astype(paddle.float32),
                bias=None if bias is None else bias.astype(paddle.float32),
                padding=padding,
                groups=groups,
            )

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensor()
        grad_out = grad_out.detach()
        with paddle.amp.auto_cast(False), paddle.enable_grad():
            x_f32 = x.astype(paddle.float32).detach()
            x_f32.stop_gradient = False
            out = F.conv1d(
                x_f32,
                weight.astype(paddle.float32).detach(),
                padding=ctx.padding,
                groups=ctx.groups,
            )
            (grad_x,) = paddle.grad([out], [x_f32], grad_outputs=[grad_out])
        with paddle.amp.auto_cast(False):
            grad_weight = _hf_conv1d_wgrad(
                x, grad_out, weight.shape[-1], ctx.padding
            )
        grad_x = grad_x.astype(x.dtype)
        grad_weight = grad_weight.reshape(weight.shape).astype(weight.dtype)
        if ctx.has_bias:
            with paddle.amp.auto_cast(False):
                grad_bias = grad_out.sum(axis=[0, 2])
            return grad_x, grad_weight, grad_bias
        return grad_x, grad_weight


def _hf_cumsum(x, axis=-1):
    """Inclusive prefix sum matching ``torch.cumsum`` bit-for-bit on FP32.

    Paddle's default ``cumsum`` and torch's produce different last-mantissa bits,
    and the delta-rule recurrence exponentiates that difference. With
    ``FLAGS_use_accuracy_compatible_kernel=1`` -- which the rest of this alignment
    already requires -- paddle's forward ``cumsum`` *does* match torch at every
    shape measured (``tools/search_cumsum_kernel.py``: 1, 28, 32, 96 and 256 rows
    of 64/128/2048 elements, 0 differing elements each), so the forward needs no
    replacement.

    The *backward* does not match: paddle's ``cumsum_grad`` differs from torch's
    on ~17% of the elements (1025/6144 per layer). torch computes the gradient of
    an inclusive prefix sum as a reversed inclusive prefix sum, and spelling that
    out as ``flip -> cumsum -> flip`` is bit-exact
    (``tools/search_cumsum_backward.py``), so only that is overridden.

    Note: a hand-written Sklansky scan was used here previously. It is bit-exact
    only while the tensor has at most 32 rows -- torch switches association order
    above that -- so it silently broke as soon as a sequence spanned more than one
    64-token chunk. Do not reintroduce it.
    """
    if axis not in (-1, x.ndim - 1):
        return x.cumsum(axis=axis)
    return _HFCumsum.apply(x)


def _sklansky_scan(x):
    """Inclusive Sklansky prefix sum over the last (power-of-two) axis.

    Kept only for ``tools/search_cumsum_kernel.py``, which uses it to show that
    torch's innermost-dim scan follows this association order for at most 32 rows
    and a different one above that. Not used by the model path.
    """
    length = x.shape[-1]
    lead = x.shape[:-1]
    out = x
    stride = 1
    while stride < length:
        out = out.reshape([*lead, length // (2 * stride), 2 * stride])
        carry = out[..., stride - 1 : stride]
        out = paddle.concat(
            [out[..., :stride], out[..., stride:] + carry], axis=-1
        ).reshape([*lead, length])
        stride *= 2
    return out


class _HFCumsum(PyLayer):
    """``cumsum`` whose backward matches torch's bit-for-bit.

    The forward is paddle's own accuracy-compatible ``cumsum``, which already
    agrees with torch. The gradient of an inclusive prefix sum is a reversed
    inclusive prefix sum; torch computes it that way, and paddle's
    ``cumsum_grad`` uses a different association order that moves ~17% of the
    elements. Spelling the reversal out restores the reference exactly.
    """

    @staticmethod
    def forward(ctx, x):
        return x.cumsum(axis=-1)

    @staticmethod
    def backward(ctx, grad):
        return grad.flip(-1).cumsum(axis=-1).flip(-1)


class _HFStateFanout(PyLayer):
    """Fan ``last_recurrent_state`` out to its three consumers in torch's order.

    Inside the chunk loop the running state is read three times::

        v_prime    = k_cumdecay[:, :, i] @ state
        attn_inter = (q_i * g_i.exp())   @ state
        state_next = state * g_last.exp() + ...

    so its gradient is a sum of three FP32 contributions, and that sum is not
    associative. The two engines drain their ready queues differently --
    ``tools/probe_accum_order.py`` measures torch accumulating in **reverse**
    consumer-creation order and paddle in creation order -- which moves ~10% of
    the state gradient by 1 ULP on the first chunk and then feeds the rest of the
    recurrence, ultimately reaching ``in_q``/``in_k`` at 1e-03.

    Routing the three reads through one node makes the accumulation explicit and
    engine-independent. Verified against the recorded torch trace by
    ``tools/search_gdr_state_grad.py``: reverse order is bit-exact (0/524288),
    creation order differs on 54539 elements.
    """

    @staticmethod
    def forward(ctx, state):
        # Distinct buffers: paddle rejects a PyLayer that returns one tensor
        # several times, and aliasing would defeat the point of separate grads.
        return state.clone(), state.clone(), state.clone()

    @staticmethod
    def backward(ctx, grad_v_prime, grad_attn_inter, grad_decay):
        return grad_decay + grad_attn_inter + grad_v_prime


class _HFL2Norm(PyLayer):
    """L2 normalization whose backward matches torch's autograd bit-for-bit.

    The reference is ``x * rsqrt((x * x).sum(-1, keepdim=True) + eps)``. ``x``
    feeds two consumers there, so its gradient is the sum of three terms: one
    from the outer multiply and two symmetric ones from ``x * x``. Paddle's
    autograd groups those three adds differently depending on
    ``FLAGS_use_accuracy_compatible_kernel`` -- with the flag on (which the rest
    of this alignment requires) ~28% of the elements move by up to 8e-03, which
    the delta-rule recurrence then amplifies. Writing the backward out fixes the
    grouping.

    The exact recipe, verified bit-exact on the real layer-6 operands for both
    ``query`` and ``key`` (0/225280 differing elements each):

    * ``g_from_y = grad_out * inv_norm`` in FP32, then rounded to the input dtype
      *before* being accumulated -- rounding after the adds instead moves nearly
      every element;
    * the reduction gradient is rounded to the input dtype before forming the two
      ``x * x`` operand gradients, because ``x * x`` is a BF16 tensor;
    * the ``rsqrt`` gradient cubes ``inv_norm`` as a **group**
      (``-0.5 * g_inv * (inv * inv * inv)``, i.e. torch's ``-0.5 * grad *
      result.pow(3)``). Multiplying left to right
      (``-0.5 * g_inv * inv * inv * inv``) differs on a handful of rows whose
      norm is large enough for the intermediate to lose a bit
      (``tools/search_l2norm_backward2.py``: 5 elements of layer 5's ``key``);
    * the outer-multiply term must not be added last (``y + s + s``, ``s + y + s``
      both work; ``s + s + y`` does not).
    """

    @staticmethod
    def forward(ctx, x, eps=1e-6):
        with paddle.amp.auto_cast(False):
            inv_norm = paddle.rsqrt(
                (x * x).sum(-1, keepdim=True, dtype=paddle.float32) + eps
            )
            out = x * inv_norm
        ctx.save_for_backward(x, inv_norm)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, inv_norm = ctx.saved_tensor()
        with paddle.amp.auto_cast(False):
            grad_from_y = (grad_out * inv_norm).astype(x.dtype)
            grad_inv = (grad_out * x.astype(grad_out.dtype)).sum(
                -1, keepdim=True
            )
            # torch's rsqrt backward is ``-0.5 * grad * result.pow(3)``; the cube
            # has to be formed before it multiplies ``grad_inv``.
            grad_total = -0.5 * grad_inv * (inv_norm * inv_norm * inv_norm)
            # ``x * x`` is BF16, so its output gradient is BF16 as well.
            grad_from_sq = grad_total.astype(x.dtype) * x
            return grad_from_y + grad_from_sq + grad_from_sq


def _l2norm(x, accuracy_target=True):
    """Apply L2 normalization along the last dimension.

    Equivalent to fla.modules.l2norm.l2norm for paddle tensors.

    ``accuracy_target`` is the caller's ``use_accuracy_compatible`` value; it
    defaults to the Megatron arithmetic so a bare one-argument call is unchanged.
    """
    if targets_hf(accuracy_target):
        # ``Qwen3_5MoeGatedDeltaNet`` runs under ``torch.autocast(bfloat16)``,
        # where
        #     inv_norm = rsqrt((x * x).sum(-1, keepdim=True) + eps); x * inv_norm
        # squares in BF16 and reduces those BF16 values with an FP32
        # *accumulator*. Paddle black-lists ``reduce_sum`` in AMP, so inside an
        # ``auto_cast`` region the BF16 squares are first materialized as FP32 and
        # that FP32 buffer is reduced -- a different rounding that moves ~24% of
        # the elements by up to 5 ULP, which then feeds the whole delta-rule
        # recurrence. Stepping out of AMP for the reduction restores the
        # reference accumulation exactly. ``_HFL2Norm`` additionally pins the
        # backward's accumulation order (see its docstring).
        return _HFL2Norm.apply(x)
    x_float = x.astype(paddle.float32)
    inv_norm = paddle.rsqrt(x_float.pow(2).sum(-1, keepdim=True) + 1e-6)
    return (x_float * inv_norm).astype(x.dtype)


@dataclass
class GatedDeltaNetSublayersSpec:
    """Contains the layer specs for the input linear, output norm, and output linear layers."""

    in_proj: LayerSpec | type = IdentityOp
    out_norm: LayerSpec | type = IdentityOp
    out_proj: LayerSpec | type = IdentityOp


class GatedDeltaNet(FleetLayer):
    """Gated Delta Net (GDN) layer class.

    GDN layer takes input with size [b, s, h] (or [s, b, h] when sequence_parallel is enabled)
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: GatedDeltaNetSublayersSpec,
        layer_number: int | None = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: float | None = None,
        use_qk_l2norm: bool = True,
        A_init_range: tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        conv_kernel_dim: int = 4,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        num_key_heads: int = 16,
        num_value_heads: int = 32,
    ):
        """
        Args:
            config: The transformer config of the model.
            sublayers_spec: Contains the layer specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization on query and key.
            A_init_range: The initialization range for the A parameter.
            pg_collection: The required process groups for tensor model parallel.
            conv_kernel_dim: Kernel size for the causal convolution.
            key_head_dim: Dimension of each query/key head.
            value_head_dim: Dimension of each value/gate head.
            num_key_heads: Number of query/key heads.
            num_value_heads: Number of value/gate heads.
        """
        super().__init__(config=config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
        assert pg_collection is not None, (
            "pg_collection must be provided for GatedDeltaNet"
        )
        self.pg_collection = pg_collection
        self.tp_size = get_pg_size(self.pg_collection.tp)
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.hidden_size = config.hidden_size
        # HF bit-exact arithmetic is a reference-target override *inside* the
        # accuracy-compatible mode, so the config layer gates it as well. The
        # module-level helpers (``_l2norm``, ``_hf_cumsum``,
        # ``paddle_chunk_gated_delta_rule``) receive no config and rely on the
        # env flag pair alone, as ``use_accuracy_compatible_kernel()`` does.
        self.accuracy_target = getattr(config, "use_accuracy_compatible", False)
        self.hf_bitexact = targets_hf(self.accuracy_target)
        self.act_fn = config.hidden_act
        self.activation = getattr(self.act_fn, "__name__", "silu")
        self.conv_kernel_dim = conv_kernel_dim
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        self.in_proj_dim = (
            self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        )

        self.in_proj = build_spec_layer(
            sublayers_spec.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        self._maybe_tag_in_proj_dgrad_groups()

        # Conv1D for QKV
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size

        # weight shape: [conv_dim, 1, d_conv], bias shape: [conv_dim]
        self.conv1d = nn.Conv1D(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            bias_attr=conv_bias,
            data_format="NCL",
        )
        self.conv1d.weight.is_distributed = True if self.tp_size > 1 else False
        if conv_bias and self.conv1d.bias is not None:
            self.conv1d.bias.is_distributed = (
                True if self.tp_size > 1 else False
            )

        # Time step projection (discretization)
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size

        # The reference implementation declares ``dt_bias``/``A_log`` as plain
        # parameters, so they live in the model dtype (BF16) and are promoted to
        # FP32 only at the softplus/exp computation boundary below. Creating FP32
        # leaves instead changes the parameter, gradient and optimizer-state
        # dtype relative to the official checkpoint, so honor the declared dtype
        # in accuracy-compatible mode. The default path keeps the FP32 leaves.
        state_param_dtype = (
            config.params_dtype
            if getattr(config, "use_accuracy_compatible", False)
            else "float32"
        )

        self.dt_bias = self.create_parameter(
            shape=[self.num_v_heads_local_tp],
            dtype=state_param_dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.dt_bias.is_distributed = True if self.tp_size > 1 else False

        self.A_log = self.create_parameter(
            shape=[self.num_v_heads_local_tp],
            dtype=state_param_dtype,
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.A_log.is_distributed = True if self.tp_size > 1 else False

        # Output layernorm before projection (per-head norm)
        # out_norm weight is replicated (not TP-sharded), but each TP rank only
        # processes its local value heads, so the gradient is a partial sum.
        # Mark the parameter so that register_sequence_parallel_allreduce_hooks
        # will all-reduce its gradient across the TP group.
        input_is_parallel = True if self.tp_size > 1 else False
        extra_args = get_norm_extra_args(
            sublayers_spec.out_norm,
            self.config,
            self.value_head_dim,
            self.config.rms_norm_eps,
            input_is_parallel,
        )
        self.out_norm = build_spec_layer(
            sublayers_spec.out_norm,
            **extra_args,
        )

        self.out_proj = build_spec_layer(
            sublayers_spec.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True if self.tp_size > 1 else False,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    def _maybe_tag_in_proj_dgrad_groups(self) -> None:
        """Tag ``in_proj.weight`` with the reference's projection column groups.

        The reference ``Qwen3_5MoeGatedDeltaNet`` splits this projection into
        four ``nn.Linear`` modules -- ``in_proj_qkv``, ``in_proj_z``,
        ``in_proj_b`` and ``in_proj_a`` -- so the gradient w.r.t. their shared
        input is a chain of four narrow GEMMs. One fused wide-K GEMM computes
        the same sum, but BF16 GEMM K splitting is not associative, so the
        results differ in the last mantissa bit. Recording each reference
        projection's fused columns lets the linear backward reproduce the split;
        the order is reversed because torch accumulates a multiply-used
        tensor's gradient in reverse module-creation order.

        Each group also records whether the reference gradient reaches cuBLAS
        column-major. ``qkv``, ``b`` and ``a`` feed tensors that the reference
        transposes downstream (``qkv`` through the conv, ``b``/``a`` through
        ``beta``/``g``), so their gradients arrive as ``[K, M]`` views; ``z``
        stays row-major. Those two operand layouts make cuBLAS pick different
        reduction splits, which changes the last mantissa bit on ~0.04% of the
        elements.
        """
        if not self.hf_bitexact:
            return
        weight = getattr(self.in_proj, "weight", None)
        if weight is None:
            return
        # (columns, gradient reaches the GEMM column-major)
        specs = [
            ((self.qk_dim * 2 + self.v_dim) // self.tp_size, True),
            (self.v_dim // self.tp_size, False),
            (self.num_value_heads // self.tp_size, True),
            (self.num_value_heads // self.tp_size, True),
        ]
        groups = []
        offset = 0
        for size, column_major in specs:
            groups.append(
                (
                    paddle.to_tensor(
                        list(range(offset, offset + size)), dtype="int64"
                    ),
                    column_major,
                )
            )
            offset += size
        weight.hf_dgrad_groups = list(reversed(groups))
        # The reference's *gradient-clipping* partition is the same split, but in
        # forward order and without the layout tag: torch takes one per-tensor
        # norm per ``nn.Linear``, so a fused projection contributes four BF16
        # norms rather than one. See paddleformers/utils/hf_bitexact_clip.py.
        weight.hf_norm_groups = [columns for columns, _ in groups]

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            # conv1d.weight
            if self.conv_init is not None:
                nn.initializer.Uniform(
                    low=-self.conv_init, high=self.conv_init
                )(self.conv1d.weight)

            # dt_bias: initialize to ones
            nn.initializer.Constant(1.0)(self.dt_bias)

            # A_log: initialize to log(uniform(A_init_range))
            A = paddle.empty([self.num_v_heads_local_tp], dtype="float32")
            nn.initializer.Uniform(
                low=self.A_init_range[0], high=self.A_init_range[1]
            )(A)
            paddle.assign(paddle.log(A).astype(self.A_log.dtype), self.A_log)

    def _build_padding_mask(
        self,
        attention_mask: paddle.Tensor | None,
        attn_mask_startend_row_indices: paddle.Tensor | None,
        batch: int,
        seq_len: int,
    ) -> paddle.Tensor | None:
        """Derive a padding mask (1.0=valid, 0.0=padding) for GDN."""
        is_sp = self.config.sequence_parallel and self.sp_size > 1

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                if attention_mask.shape[-2:] != [seq_len, seq_len]:
                    # Shape mismatch – fall through to startend indices.
                    pass
                else:
                    # The multimodal collator provides a block-causal
                    # ``[b, 1, s, s]`` mask, but GDN is a recurrence and only
                    # needs per-token validity. A token is valid iff its own
                    # causal diagonal entry is enabled; padding rows have a zero
                    # diagonal. Reduce to ``[b, s]`` and reuse the 2D/SP logic
                    # below instead of rejecting the mask. ``paddle.diagonal``
                    # returns a strided view, so materialize it — reductions
                    # over a non-contiguous float buffer read the wrong memory.
                    attention_mask = paddle.diagonal(
                        attention_mask[:, 0, :, :], axis1=-2, axis2=-1
                    ).contiguous()

            if attention_mask.ndim == 2:
                full_seq = attention_mask.shape[-1]
                if is_sp:
                    if full_seq != seq_len:
                        # Shape mismatch under SP – fall through to startend indices.
                        pass
                    else:
                        # attention_mask is [b, full_s], slice to local chunk
                        seq_len_local = seq_len // self.sp_size
                        tp_rank = get_pg_rank(self.pg_collection.tp)
                        offset = tp_rank * seq_len_local
                        local_mask = attention_mask[
                            :, offset : offset + seq_len_local
                        ]
                        if local_mask.astype("bool").all():
                            return None
                        return local_mask.astype(paddle.float32).T.unsqueeze(-1)
                else:
                    if full_seq == seq_len:
                        mask = attention_mask.unsqueeze(-1).astype(
                            paddle.float32
                        )
                        # Reduce through bool like the SP branch above: ``all()``
                        # on a float tensor is not reliable, so a float mask of
                        # all-ones would otherwise fail this check and skip the
                        # ``None`` fast path.
                        if mask.astype("bool").all():
                            return None
                        return mask
                    # full_seq != seq_len: attention_mask shape does not match the
                    # current sequence length (e.g. stale mask from a previous stage).
                    # Fall through to try attn_mask_startend_row_indices instead.

        if attn_mask_startend_row_indices is not None:
            indices = attn_mask_startend_row_indices[:, 0, :, 0]
            full_seq = indices.shape[-1]

            if is_sp:
                if full_seq != seq_len:
                    # Shape mismatch under SP – cannot derive valid mask.
                    pass
                else:
                    seq_len_local = seq_len // self.sp_size
                    tp_rank = get_pg_rank(self.pg_collection.tp)
                    offset = tp_rank * seq_len_local
                    local_indices = indices[:, offset : offset + seq_len_local]
                    seq_positions = paddle.arange(
                        offset,
                        offset + seq_len_local,
                        dtype=local_indices.dtype,
                    )
                    valid = (local_indices > seq_positions.unsqueeze(0)).astype(
                        paddle.float32
                    )
                    if valid.all():
                        return None
                    return valid.T.unsqueeze(-1)
            else:
                seq_positions = paddle.arange(full_seq, dtype=indices.dtype)
                valid = (indices > seq_positions.unsqueeze(0)).astype(
                    paddle.float32
                )
                if valid.all():
                    return None
                return valid.unsqueeze(-1)

        if (
            attention_mask is not None
            or attn_mask_startend_row_indices is not None
        ):
            raise ValueError(
                f"GatedDeltaNet._build_padding_mask: could not derive a valid "
                f"padding mask from the provided inputs "
                f"(attention_mask.shape={list(attention_mask.shape) if attention_mask is not None else None}, "
                f"attn_mask_startend_row_indices.shape="
                f"{list(attn_mask_startend_row_indices.shape) if attn_mask_startend_row_indices is not None else None}, "
                f"seq_len={seq_len})."
            )
        return None

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        key_value_states: paddle.Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        packed_seq_params=None,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states: Hidden states [b, s, h] or [s, b, h] with sequence_parallel.
            attention_mask: Attention mask.
            key_value_states: Key/value states (for cross attention, not supported).
            attention_bias: Attention bias.
            packed_seq_params: Parameters used for THD format (not supported).

        Returns:
            Tuple of (output, output_bias).
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "GDN does not support packed sequence for now."
            )

        hidden_states = hidden_states.contiguous()
        # Determine sequence layout
        if self.config.sequence_parallel and self.sp_size > 1:
            # Input is [s, b, h] with sequence parallel
            seq_len_local, batch, _ = hidden_states.shape
            seq_len = seq_len_local * self.sp_size
        else:
            # Input is [b, s, h]
            batch, seq_len, _ = hidden_states.shape

        attn_mask_startend_row_indices = kwargs.get(
            "attn_mask_startend_row_indices", None
        )
        padding_mask = self._build_padding_mask(
            attention_mask, attn_mask_startend_row_indices, batch, seq_len
        )
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask.astype(
                hidden_states.dtype
            )

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # Ensure [b, s, x] format for the rest of computation
        if self.config.sequence_parallel:
            # [s, b, x] -> [b, s, x]
            qkvzba = qkvzba.transpose([1, 0, 2])

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        qkv, gate, beta, alpha = paddle.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.tp_size,
                self.v_dim // self.tp_size,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            axis=-1,
        )
        gate = gate.reshape([batch, seq_len, -1, self.value_head_dim])
        beta = beta.reshape([batch, seq_len, -1])
        alpha = alpha.reshape([batch, seq_len, -1])

        # Convolution on qkv
        qkv = qkv.transpose([0, 2, 1]).contiguous()  # b, s, d -> b, d, s
        nvtx_range_push(suffix="conv1d")
        # Always use Conv1D + activation path (causal_conv1d not available for Paddle)
        if self.hf_bitexact and qkv.dtype != paddle.float32:
            # FP32 accumulation, then a single round back to the activation
            # dtype -- matches the reference F.conv1d fallback bit-for-bit.
            # ``_HFCausalConv1d`` additionally pins the weight gradient's
            # reduction order (see its docstring).
            qkv_dtype = qkv.dtype
            conv_out = _HFCausalConv1d.apply(
                qkv,
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv_kernel_dim - 1,
                self.conv_dim_local_tp,
            ).astype(qkv_dtype)
            qkv = self.act_fn(conv_out[..., :seq_len])
        else:
            qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        nvtx_range_pop(suffix="conv1d")

        # Split qkv into query, key, and value
        qkv = qkv.transpose([0, 2, 1])  # b, d, s -> b, s, d
        query, key, value = paddle.split(
            qkv,
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
            ],
            axis=-1,
        )
        query = query.reshape([batch, seq_len, -1, self.key_head_dim])
        key = key.reshape([batch, seq_len, -1, self.key_head_dim])
        value = value.reshape([batch, seq_len, -1, self.value_head_dim])

        # Apply L2 norm to query and key
        if self.use_qk_l2norm and not self.hf_bitexact:
            query = _l2norm(query.contiguous(), self.accuracy_target)
            key = _l2norm(key.contiguous(), self.accuracy_target)

        # GQA repeat if num_value_heads > num_key_heads
        if self.num_value_heads // self.num_key_heads > 1:
            query = paddle.repeat_interleave(
                query, self.num_value_heads // self.num_key_heads, axis=2
            )
            key = paddle.repeat_interleave(
                key, self.num_value_heads // self.num_key_heads, axis=2
            )

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.astype(paddle.float32).exp() * F.softplus(
            alpha.astype(paddle.float32) + self.dt_bias.astype(paddle.float32)
        )
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        if (not HAVE_FLA) or self.config.deterministic_mode:
            core_attn_out, last_recurrent_state = paddle_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                # Qwen3_5MoeGatedDeltaNet passes use_qk_l2norm_in_kernel=True, so
                # the reference kernel captures ``initial_dtype`` from the *raw*
                # BF16 query and normalizes afterwards. Normalizing outside (the
                # default path above) makes ``initial_dtype`` FP32 whenever AMP
                # promotes the l2norm, so core_attn_out -- and with it the gated
                # norm and out_proj input -- stays FP32 instead of being rounded
                # back to BF16. L2 norm acts on the last axis only, so it
                # commutes with the GQA repeat_interleave above either way.
                use_qk_l2norm_in_kernel=bool(
                    self.hf_bitexact and self.use_qk_l2norm
                ),
                accuracy_target=self.accuracy_target,
            )
        else:
            raise NotImplementedError("FLA not supported yet.")
            # core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            #     query,
            #     key,
            #     value,
            #     g=g,
            #     beta=beta,
            #     initial_state=None,
            #     output_final_state=False,
            #     use_qk_l2norm_in_kernel=False,
            # )
        nvtx_range_pop(suffix="gated_delta_rule")

        # Gated norm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Reshape: [b, s, num_heads, head_dim] -> [b, s, v_dim]
        norm_out = norm_out.reshape([batch, seq_len, -1])

        # Transpose back if sequence parallel: [b, s, x] -> [s, b, x]
        if self.config.sequence_parallel:
            norm_out = norm_out.transpose([1, 0, 2]).contiguous()

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        """Apply output normalization with gating."""
        # x: [b, s, num_heads, head_dim], gate: [b, s, num_heads, head_dim]
        x_dtype = x.dtype
        x = x.reshape([-1, x.shape[-1]])
        gate = gate.reshape([-1, gate.shape[-1]])
        if self.hf_bitexact and hasattr(self.out_norm, "weight"):
            # Reproduce Qwen3_5MoeRMSNormGated exactly: FP32 variance and rsqrt,
            # round the normalized value back to the activation dtype, and only
            # then scale by the weight. Paddle's fused rms_norm keeps the weight
            # multiply in FP32, which shifts the last mantissa bits on ~26% of
            # elements before the gate is applied.
            h = x.astype(paddle.float32)
            variance = h.pow(2).mean(-1, keepdim=True)
            h = h * paddle.rsqrt(variance + self.out_norm.variance_epsilon)
            y = self.out_norm.weight * h.astype(x_dtype)
        else:
            y = self.out_norm(x)
        # Output gate
        y = y * self.act_fn(gate.astype(paddle.float32))
        y = y.astype(x_dtype)
        return y

    def sharded_state_dict(self, structured_name_prefix: str = ""):
        """Provide a sharded state dictionary for distributed checkpointing."""
        try:
            from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
                build_sharded_state_dict,
            )
        except ImportError:
            return {}

        sharded_sd = {}

        # in_proj (ColumnParallelLinear) — delegate to its own sharded_state_dict
        if hasattr(self.in_proj, "sharded_state_dict"):
            sharded_sd.update(
                self.in_proj.sharded_state_dict(
                    structured_name_prefix=f"{structured_name_prefix}in_proj."
                )
            )

        # conv1d — TP-sharded along axis 0
        conv_sd = self.conv1d.state_dict(structured_name_prefix="")
        conv_shard_rules = None
        if self.tp_size > 1:
            conv_shard_rules = {"weight": 0}
            if self.conv_bias and "bias" in conv_sd:
                conv_shard_rules["bias"] = 0
        sharded_sd.update(
            build_sharded_state_dict(
                conv_sd,
                conv_shard_rules,
                f"{structured_name_prefix}conv1d.",
            )
        )

        # dt_bias and A_log — TP-sharded along axis 0
        param_sd = {"dt_bias": self.dt_bias, "A_log": self.A_log}
        param_shard_rules = None
        if self.tp_size > 1:
            param_shard_rules = {"dt_bias": 0, "A_log": 0}
        sharded_sd.update(
            build_sharded_state_dict(
                param_sd,
                param_shard_rules,
                structured_name_prefix,
            )
        )

        # out_norm — not TP-sharded (per-head norm)
        if hasattr(self.out_norm, "sharded_state_dict"):
            sharded_sd.update(
                self.out_norm.sharded_state_dict(
                    structured_name_prefix=f"{structured_name_prefix}out_norm."
                )
            )
        else:
            out_norm_sd = self.out_norm.state_dict(structured_name_prefix="")
            sharded_sd.update(
                build_sharded_state_dict(
                    out_norm_sd,
                    None,
                    f"{structured_name_prefix}out_norm.",
                )
            )

        # out_proj (RowParallelLinear) — delegate to its own sharded_state_dict
        if hasattr(self.out_proj, "sharded_state_dict"):
            sharded_sd.update(
                self.out_proj.sharded_state_dict(
                    structured_name_prefix=f"{structured_name_prefix}out_proj."
                )
            )

        return sharded_sd

    # def backward_dw(self):
    #     """Execute weight gradient computation for all linear layers."""
    #     self._backward_in_proj()
    #     self._backward_out_proj()

    # def _backward_in_proj(self):
    #     """Computes weight gradients of input projection layer."""
    #     self.in_proj.backward_dw()

    # def _backward_out_proj(self):
    #     """Computes weight gradients of output projection layer."""
    #     self.out_proj.backward_dw()


def paddle_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    accuracy_target=True,
):
    """
    Paddle-native implementation of chunked gated delta rule for deterministic mode.

    This is a direct port from Megatron-LM.

    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/
        src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, accuracy_target)
        key = _l2norm(key, accuracy_target)

    # Convert to [b, num_heads, s, head_dim] and float32
    query, key, value, beta, g = [
        x.transpose([0, 2, 1, 3]).contiguous().astype(paddle.float32)
        if x.ndim == 4
        else x.transpose([0, 2, 1]).contiguous().astype(paddle.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    if pad_size > 0:
        query = F.pad(query, [0, 0, 0, pad_size])
        key = F.pad(key, [0, 0, 0, pad_size])
        value = F.pad(value, [0, 0, 0, pad_size])
        beta = F.pad(beta, [0, pad_size])
        g = F.pad(g, [0, pad_size])

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks: [b, h, num_chunks, chunk_size, dim]
    query, key, value, k_beta, v_beta = [
        x.reshape([x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape([g.shape[0], g.shape[1], -1, chunk_size])

    mask = paddle.triu(
        paddle.ones([chunk_size, chunk_size], dtype=paddle.bool), diagonal=0
    )

    # Chunk decay
    _hf = targets_hf(accuracy_target)
    g = _hf_cumsum(g, axis=-1) if _hf else g.cumsum(axis=-1)
    decay_mask = (
        (g.unsqueeze(-1) - g.unsqueeze(-2))
        .tril()
        .exp()
        .astype(paddle.float32)
        .tril()
    )

    # attn = -((k_beta @ key^T) * decay_mask), masked to lower triangular
    attn = -(
        (k_beta @ key.transpose([0, 1, 2, 4, 3])) * decay_mask
    ).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + paddle.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = paddle.zeros(
            [batch_size, num_heads, k_head_dim, v_head_dim],
            dtype=value.dtype,
        )
    else:
        last_recurrent_state = initial_state.astype(value.dtype)

    core_attn_out = paddle.zeros_like(value)

    mask = paddle.triu(
        paddle.ones([chunk_size, chunk_size], dtype=paddle.bool), diagonal=1
    )

    # For each chunk
    num_chunks = total_sequence_length // chunk_size
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (
            q_i @ k_i.transpose([0, 1, 3, 2]) * decay_mask[:, :, i]
        ).masked_fill_(mask, 0)
        if _hf and not last_recurrent_state.stop_gradient:
            # One node feeds all three reads so the three gradient contributions
            # are summed in torch's order rather than paddle's (see
            # ``_HFStateFanout``). Only needed once the state carries a gradient,
            # i.e. from the second chunk on.
            state_v_prime, state_attn_inter, state_decay = _HFStateFanout.apply(
                last_recurrent_state
            )
        else:
            state_v_prime = state_attn_inter = state_decay = (
                last_recurrent_state
            )
        v_prime = k_cumdecay[:, :, i] @ state_v_prime
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state_attn_inter
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            state_decay * g[:, :, i, -1, None, None].exp()
            + (
                k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
            ).transpose([0, 1, 3, 2])
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        [
            core_attn_out.shape[0],
            core_attn_out.shape[1],
            -1,
            core_attn_out.shape[-1],
        ]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = (
        core_attn_out.transpose([0, 2, 1, 3]).contiguous().astype(initial_dtype)
    )
    return core_attn_out, last_recurrent_state
