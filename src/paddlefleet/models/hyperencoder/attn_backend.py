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

"""Attention backend switches for HyperEncoder.

## Environment variables

| env | values | effect |
|---|---|---|
| ``HYPERBODY_ENCODER_ATTN_BACKEND`` | ``dp`` (default) / ``flex`` / ``triton`` | selects the ``core_attention`` implementation |
| ``HYPERBODY_PACKED_FLEX_DECODER`` | ``0`` (default) / ``1`` | run the trunk with a single packed call |

**The ``flex`` backend is not implemented here**: it is a compiled artifact of
``torch.nn.attention.flex_attention`` and has no Paddle equivalent. The ``flex``
and ``triton`` backends are equivalent under the packed layout, so this module
supports ``dp`` and ``triton`` only and **raises on ``flex``** instead of
silently falling back to ``dp`` (a silent downgrade would produce misleading
results).

## Functions instead of module-level constants

The backend selection is read on every call rather than fixed at import time.
Some scripts run several configurations back-to-back in one process; fixing the
switch at import time would make a later configuration silently inherit the
first one's backend.
"""

from __future__ import annotations

import os

__all__ = [
    "encoder_attn_backend",
    "use_triton_encoder_attn",
    "use_packed_decoder",
]

_TRUE = ("1", "true", "on")
_FALSE = ("0", "false", "off")


def encoder_attn_backend() -> str:
    """Return ``dp`` or ``triton``. ``flex`` raises (see module docstring)."""
    v = os.environ.get("HYPERBODY_ENCODER_ATTN_BACKEND", "dp").lower()
    if v == "flex":
        raise ValueError(
            "HYPERBODY_ENCODER_ATTN_BACKEND=flex is not implemented here: "
            "flex is a compiled artifact of torch.nn.attention.flex_attention "
            "and has no Paddle equivalent. Use triton for packed semantics "
            "(it is equivalent to flex)."
        )
    if v not in ("dp", "triton"):
        raise ValueError(
            f"Unknown HYPERBODY_ENCODER_ATTN_BACKEND={v!r}; expected 'dp' or 'triton'"
        )
    return v


def use_triton_encoder_attn() -> bool:
    """Whether to replace ``core_attention`` with ``PrefixLMTritonCore``."""
    return encoder_attn_backend() == "triton"


def use_packed_decoder() -> bool:
    """Whether the trunk runs as a single packed call."""
    v = os.environ.get("HYPERBODY_PACKED_FLEX_DECODER", "0").lower()
    if v not in _TRUE + _FALSE:
        raise ValueError(
            f"Unknown HYPERBODY_PACKED_FLEX_DECODER={v!r}; expected 0/1/false/true/off/on"
        )
    on = v in _TRUE
    # The packed path attaches the segment layout to packed_seq_params, which
    # only the triton backend reads. Enabling packed while staying on the dp
    # backend would silently ignore the layout and degrade the mask, so this is
    # a hard error.
    if on and not use_triton_encoder_attn():
        raise RuntimeError(
            "HYPERBODY_PACKED_FLEX_DECODER=1 requires HYPERBODY_ENCODER_ATTN_BACKEND=triton: "
            "the packed segment layout is only read by the triton core and would "
            "be silently ignored on the dp backend."
        )
    return on
