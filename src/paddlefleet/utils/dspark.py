# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional, read-only target token I/O for a detached DSpark branch.

No target parameters are frozen, registered on a second Layer, or copied at
initialization. Construct a projection each forward from the current target
weights, and complete both backward passes before updating those weights.
These helpers require full-vocabulary tensors (TP=1 or externally gathered).
"""

from dataclasses import dataclass
from typing import Any


def readonly_embedding(token_ids, weight):
    """Read the current [vocab, hidden] weight without a target gradient."""
    import paddle.nn.functional as F

    if weight.ndim != 2:
        raise ValueError("embedding weight must have shape [vocab, hidden]")
    return F.embedding(token_ids, weight.detach())


@dataclass(frozen=True)
class ReadOnlyProjection:
    """Full-vocabulary projection, including optional bias and Multimax.

    ``transpose_y`` is explicit even for square heads: True means [V, H],
    False means [H, V]. Detached views preserve the hidden-state gradient.
    Mutating the underlying target weights before backward is unsupported.
    """

    dspark_readonly = True

    weight: Any
    transpose_y: bool
    bias: Any = None
    multimax_ranges: Any = None
    multimax_ts: Any = None

    def __post_init__(self):
        if self.weight.ndim != 2 or not isinstance(self.transpose_y, bool):
            raise ValueError(
                "provide a matrix weight and explicit boolean transpose_y"
            )
        vocab = self.weight.shape[0 if self.transpose_y else 1]
        if self.bias is not None and list(self.bias.shape) != [vocab]:
            raise ValueError("bias must have shape [vocab]")
        if (self.multimax_ranges is None) != (self.multimax_ts is None):
            raise ValueError("Multimax requires both ranges and ts")
        for name in ("multimax_ranges", "multimax_ts"):
            value = getattr(self, name)
            if value is not None and list(value.shape) != [4]:
                raise ValueError(f"{name} must have shape [4]")
        for name in ("weight", "bias", "multimax_ranges", "multimax_ts"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.detach())

    def transform_logits(self, logits):
        """Apply target bias and the same elementwise SegLU as GPTLMHead."""
        import paddle.nn.functional as F

        if self.bias is not None:
            # Match linear's autocast bias dtype, including streamed callers
            # that project first and call transform_logits directly.
            logits = logits + self.bias.astype(logits.dtype)
        if self.multimax_ranges is None:
            return logits
        r, t = self.multimax_ranges, self.multimax_ts
        return (
            logits
            + t[0] * F.relu(r[0] - logits)
            + t[1] * F.relu(logits - r[1])
            + t[2] * F.relu(r[2] - logits) ** 2
            + t[3] * F.relu(logits - r[3]) ** 2
        )

    def __call__(self, hidden_states):
        import paddle

        return self.transform_logits(
            paddle.matmul(
                hidden_states, self.weight, transpose_y=self.transpose_y
            )
        )
