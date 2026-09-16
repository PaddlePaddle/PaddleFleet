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

"""RMSNorm variant used by HyperEncoder.

This module keeps the HyperEncoder-specific RMSNorm in its own file because it
is only used by HyperEncoder (``models/hyperencoder/layer_specs.py``). Keeping
it separate avoids adding a single-model class to the shared norm module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
from paddle import Tensor

# Optional import: older Paddle versions may not ship this utility.
try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        mark_as_sequence_parallel_parameter,
    )
except ImportError:  # older Paddle without this helper

    def mark_as_sequence_parallel_parameter(parameter):
        return parameter


if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig

__all__ = ["HyperEncoderRMSNorm"]


class HyperEncoderRMSNorm(paddle.nn.Layer):
    """RMSNorm that always accumulates in fp32.

    Reference computation:

    .. code-block:: python

        x_float = x.float()
        x_norm  = x_float * paddle.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
        return (x_norm * self.weight).astype(x.dtype)

    This class is used instead of a generic RMSNorm for three reasons:

    1. A generic RMSNorm typically exposes its high-precision behavior as a
       forward argument rather than a constructor option; since layers are
       instantiated by the spec system, the call site cannot pass that flag.
    2. A generic RMSNorm may route through a fused ``rms_norm`` kernel whose
       ``mean(x**2)`` reduction order is not controllable. This class uses
       ``paddle.mean`` directly instead.
    3. A generic RMSNorm may derive its return dtype from the weight dtype,
       whereas this class derives it from the input dtype ``x``. These agree
       under bf16 training, but a fp32 weight would otherwise silently upcast
       the output.

    Two intentional (mathematically equivalent) differences from a plain
    reference formulation:

    * ``x_float.pow(2)`` is written as ``x_float * x_float``.
    * The weight multiply casts the weight to fp32 explicitly. bf16 -> fp32 is
      an exact conversion, so this only avoids relying on implicit type
      promotion.

    Sequence-parallel marking: the weight is tagged via
    :func:`mark_as_sequence_parallel_parameter`, triggered by
    ``input_is_parallel``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        normalized_shape=None,
        norm_eps=None,
        input_is_parallel=False,
        **kwargs,
    ):
        super().__init__()
        self.normalized_shape = (
            config.hidden_size if normalized_shape is None else normalized_shape
        )
        self.variance_epsilon = (
            config.rms_norm_eps if norm_eps is None else norm_eps
        )
        # Explicitly reject unsupported configurations instead of ignoring them.
        if getattr(config, "normalization", "RMSNorm") != "RMSNorm":
            raise ValueError(
                f"HyperEncoderRMSNorm only supports RMSNorm, got {config.normalization!r}"
            )
        if getattr(config, "layernorm_zero_centered_gamma", False):
            raise ValueError(
                "HyperEncoderRMSNorm does not support zero-centered gamma"
            )
        if getattr(config, "persist_layer_norm", False):
            raise ValueError(
                "HyperEncoderRMSNorm does not support persistent layer norm"
            )

        self.weight = paddle.create_parameter(
            shape=[self.normalized_shape],
            dtype=config.params_dtype
            if config.params_dtype is not None
            else paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.config = config

        if input_is_parallel:
            self.enable_sequence_parallel()

    def forward(self, hidden_states: Tensor, **kwargs):
        """Apply the RMSNorm.

        ``**kwargs`` absorbs any extra flags a generic RMSNorm might accept;
        this class always computes in fp32, so they have no effect. The
        reduction goes through ``paddle.mean`` directly (not a fused kernel),
        and the return dtype follows the input dtype.
        """
        t1 = hidden_states.astype(paddle.float32)
        t4 = paddle.mean(t1 * t1, axis=-1, keepdim=True) + self.variance_epsilon
        t5 = paddle.rsqrt(t4)
        out = (t1 * t5) * self.weight.astype(paddle.float32)
        return out.astype(hidden_states.dtype)

    def enable_sequence_parallel(self):
        """Tag the weight for sequence-parallel replication.

        This mirrors the sequence-parallel marking used by the other norm
        layers. It must be called explicitly for this class; it is only
        exercised when tensor parallelism is enabled.
        """
        mark_as_sequence_parallel_parameter(self.weight)
