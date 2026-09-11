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

"""Attention backend selection for HyperEncoder.

The backend and packed-decoder path are driven by config fields rather than
environment variables, so they are declared with defaults, validated centrally
(in ``HyperEncoderProvider.__post_init__``, which calls these functions), and
serialized with the rest of the config:

| field | values | effect |
|---|---|---|
| ``hyperencoder_attn_backend`` | ``dp`` (default) / ``triton`` | selects the ``core_attention`` implementation |
| ``hyperencoder_packed_decoder`` | ``False`` (default) / ``True`` | run the trunk with a single packed call |

**The ``flex`` backend is not implemented here**: it is a compiled artifact of
``torch.nn.attention.flex_attention`` and has no Paddle equivalent. The ``flex``
and ``triton`` backends are equivalent under the packed layout, so this module
supports ``dp`` and ``triton`` only and **raises on ``flex``** instead of
silently falling back to ``dp`` (a silent downgrade would produce misleading
results).

The readers use ``getattr`` with defaults so any ``TransformerConfig`` works;
the fields are authoritatively declared and validated on the HyperEncoder
config.
"""

from __future__ import annotations

__all__ = [
    "encoder_attn_backend",
    "use_triton_encoder_attn",
    "use_packed_decoder",
]


def encoder_attn_backend(config) -> str:
    """Return ``dp`` or ``triton``. ``flex`` raises (see module docstring)."""
    v = str(getattr(config, "hyperencoder_attn_backend", "dp")).lower()
    if v == "flex":
        raise ValueError(
            "hyperencoder_attn_backend='flex' is not implemented here: "
            "flex is a compiled artifact of torch.nn.attention.flex_attention "
            "and has no Paddle equivalent. Use triton for packed semantics "
            "(it is equivalent to flex)."
        )
    if v not in ("dp", "triton"):
        raise ValueError(
            f"Unknown hyperencoder_attn_backend={v!r}; expected 'dp' or 'triton'"
        )
    return v


def use_triton_encoder_attn(config) -> bool:
    """Whether to replace ``core_attention`` with ``PrefixLMTritonCore``."""
    return encoder_attn_backend(config) == "triton"


def use_packed_decoder(config) -> bool:
    """Whether the trunk runs as a single packed call."""
    on = bool(getattr(config, "hyperencoder_packed_decoder", False))
    # The packed path attaches the segment layout to packed_seq_params, which
    # only the triton backend reads. Enabling packed while staying on the dp
    # backend would silently ignore the layout and degrade the mask, so this is
    # a hard error.
    if on and not use_triton_encoder_attn(config):
        raise RuntimeError(
            "hyperencoder_packed_decoder=True requires "
            "hyperencoder_attn_backend='triton': the packed segment layout is "
            "only read by the triton core and would be silently ignored on the "
            "dp backend."
        )
    return on
