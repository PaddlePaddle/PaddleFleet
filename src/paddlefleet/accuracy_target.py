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
"""Which reference the accuracy-compatible mode reproduces bit-for-bit.

PaddleFleet's default kernels are chosen for throughput.
``TransformerConfig.use_accuracy_compatible`` switches to bit-reproducible
kernels, but "bit-reproducible" is only meaningful relative to a reference, and
the two references in use do not agree. They differ in *where* a computation is
promoted to FP32: Megatron's router gating linear runs in FP32
(``router_dtype``) while ``Qwen3_5MoeTopKRouter`` keeps the projection in the
activation dtype and only upcasts inside the softmax. Both are legitimate; they
simply produce different last mantissa bits.

``use_accuracy_compatible`` is therefore a *tri-state* rather than a boolean:

``False``
    default throughput kernels.
``"megatron"`` (also accepted as ``True``)
    bit-reproducible against Megatron-LM. This is what ``True`` has always
    meant, so existing configs and checkpoints keep working unchanged.
``"hf"``
    bit-reproducible against the HuggingFace/Torch reference implementation.

Both non-default values are truthy, so the many ``if config.use_accuracy_compatible:``
sites that only ask "am I in an alignment mode at all" need no change. Only the
handful of sites where the two references demand *different* arithmetic call
:func:`targets_hf`.

Modules and ``PyLayer``s that have no config object receive the value as a
parameter rather than reading a global, so a single config field stays the only
source of truth and a mixed state (module on one target, helper on the other) is
not representable.
"""

from __future__ import annotations

__all__ = [
    "ACCURACY_TARGETS",
    "ACCURACY_TARGET_HF",
    "ACCURACY_TARGET_MEGATRON",
    "AccuracyTarget",
    "normalize_accuracy_target",
    "targets_hf",
]

#: Bit-reproducible against Megatron-LM; the meaning of a bare ``True``.
ACCURACY_TARGET_MEGATRON = "megatron"

#: Bit-reproducible against the HuggingFace/Torch reference implementation.
ACCURACY_TARGET_HF = "hf"

ACCURACY_TARGETS = (ACCURACY_TARGET_MEGATRON, ACCURACY_TARGET_HF)

#: ``False`` (default kernels) or one of :data:`ACCURACY_TARGETS`.
AccuracyTarget = bool | str


#: Spellings of "off" and "on" that config layers produce when they stringify a
#: boolean. YAML, argparse and env plumbing all do this somewhere, and a value
#: that survived as ``"True"`` must not be mistaken for an unknown target.
_FALSE_WORDS = frozenset({"false", "0", "no", "off", "none", "null"})
_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})


def normalize_accuracy_target(value: AccuracyTarget) -> AccuracyTarget:
    """Canonicalize a ``use_accuracy_compatible`` value.

    ``True`` becomes ``"megatron"`` so that the stored value always names its
    reference; every falsy input becomes ``False``. Raises ``ValueError`` on an
    unknown target rather than silently degrading to the default kernels, which
    would turn a typo into a slow, wrong-reference run that still looks aligned.
    """
    if not value:
        # Covers False, None, "" and 0 -- the YAML, CLI and dataclass layers each
        # produce a different spelling of "off" and they must not diverge.
        return False
    if value is True or value == 1:
        # ``True`` predates the "hf" target; a YAML scalar ``1`` means the same.
        return ACCURACY_TARGET_MEGATRON
    if isinstance(value, str):
        target = value.strip().lower()
        if target in ACCURACY_TARGETS:
            return target
        if target in _TRUE_WORDS:
            return ACCURACY_TARGET_MEGATRON
        if target in _FALSE_WORDS:
            return False
        raise ValueError(
            f"use_accuracy_compatible must be False, True, or one of "
            f"{list(ACCURACY_TARGETS)}; got {value!r}."
        )
    raise TypeError(
        f"use_accuracy_compatible must be a bool or str, got "
        f"{type(value).__name__}."
    )


def targets_hf(value: AccuracyTarget) -> bool:
    """Whether ``value`` selects the HuggingFace/Torch reference.

    Accepts the raw field value so call sites can pass
    ``config.use_accuracy_compatible`` (or a threaded parameter) directly.
    """
    return value == ACCURACY_TARGET_HF
