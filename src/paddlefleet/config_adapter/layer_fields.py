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

"""Per-layer list fields that must follow ``num_hidden_layers``.

Shrinking PP shrinks ``num_hidden_layers``, and every per-layer list in
``model_config.json`` has to shrink with it or the framework refuses to start::

    csa_compress_ratios    len == num_hidden_layers + mtp layers
    window_attn_skip_freq  len == num_hidden_layers + mtp layers (list form)
    layer_types            len == num_hidden_layers
    moe_layer_freq         len == num_hidden_layers (list form)

The rewrite keeps the first ``layers_new`` entries -- so the dense prefix
(``first_k_dense_replace``) and the leading attention pattern survive -- and
re-appends the trailing MTP entries verbatim for the fields whose length counts
them.  ``csa_compress_ratios`` then gets one extra fix: with
``mtp_shared_last_layer`` the MTP layer aliases the last decoder layer's
attention weights, so the truncated tail is forced back to the MTP layer's own
attention type (see :func:`_shares_mtp_attention`).

Truncation is refused, rather than silently producing an unusable config, when

* a source list length does not match the source ``num_hidden_layers`` (the
  input config is already inconsistent, so there is nothing safe to derive), or
* it would drop the last layer of an attention family that the source used
  (``csa_compress_ratios`` only).  The framework has several "at least one
  layer must be X" rules, and a truncation that erases a family turns a
  successful adaptation into a startup failure.
"""

from __future__ import annotations

from collections import namedtuple

# includes_mtp: the expected length counts the trailing MTP layers.
# families: entries encode an attention family whose presence must survive.
LayerField = namedtuple("LayerField", ["aliases", "includes_mtp", "families"])

LAYER_FIELDS = (
    LayerField(("csa_compress_ratios",), includes_mtp=True, families=True),
    LayerField(("window_attn_skip_freq",), includes_mtp=True, families=False),
    LayerField(("layer_types",), includes_mtp=False, families=False),
    LayerField(("moe_layer_freq",), includes_mtp=False, families=False),
)


class StaleMtpKeyError(ValueError):
    """A ``model_config.json`` still carries the removed ``mtp_num_layers``."""


def reject_stale_mtp_key(model_config):
    """Fail loudly when a *meaningful* ``mtp_num_layers`` is still present.

    ``TransformerConfig`` rejects the key via ``renamed_config_keys_when_set``
    whenever it is non-zero, so a JSON that still carries a real value cannot
    start training. The adapter reads the same JSON and would otherwise happily
    rewrite it while silently treating K as 0 -- handing back a config the
    framework then refuses. Surface it here instead.

    ``mtp_num_layers: 0`` is tolerated for the same reason the framework
    tolerates it: it means "MTP off", which is exactly what the key's absence
    means, and PaddleFormers stamps that default onto every config it produces.
    """
    if model_config is None:
        return
    if not model_config.get("mtp_num_layers"):
        return
    raise StaleMtpKeyError(
        "model_config.json still sets the removed `mtp_num_layers` "
        f"(={model_config.get('mtp_num_layers')!r}). Use "
        "`num_nextn_predict_layers` instead: it is the only field the MTP "
        "consumers read. TransformerConfig rejects a non-zero "
        "`mtp_num_layers` outright, so this config cannot start training "
        "until the key is migrated."
    )


def effective_mtp_layers(model_config):
    """MTP layer count the framework actually uses.

    ``num_nextn_predict_layers`` is the single source of truth; the historical
    ``mtp_num_layers`` alias has been removed from ``TransformerConfig``.
    """
    if model_config is None:
        return 0
    reject_stale_mtp_key(model_config)
    return int(model_config.get("num_nextn_predict_layers") or 0)


def _csa_family(ratio):
    """Attention family encoded by one ``csa_compress_ratios`` entry."""
    value = int(ratio)
    if value == -2:
        return "MLA(-2)"
    if value == -1:
        return "full-causal MQA(-1)"
    if value == 0:
        return "window(0)"
    if value == 128:
        return "HCA(128)"
    return "CSA(2..127)"


def _shares_mtp_attention(model_config, mtp_layers):
    """Whether the MTP layer aliases the last decoder layer's attention.

    ``GPTModel.get_sequential_layers`` wraps both the backbone's last
    transformer layer and every MTP layer in
    ``SharedLayerDesc("mtp_reuse_transformer", ...,
    shared_weight_attr="transformer_layer_weights")`` once
    ``mtp_shared_last_layer`` is set and MTP is present.  Paddle then
    broadcasts that parameter list across the two pipeline stages attribute by
    attribute, so the two layers must be the same kind of attention: MLA and
    HCA/CSA expose different parameters, and the mismatched broadcast hangs the
    job instead of raising.
    """
    return bool(model_config.get("mtp_shared_last_layer")) and mtp_layers > 0


def plan_layer_field_shrink(model_config, layers_old, layers_new, mtp_layers):
    """Plan the per-layer list rewrites implied by a layer-count shrink.

    Returns ``(changes, error)`` where ``changes`` is
    ``[(key, value, reason), ...]`` ready to apply to ``model_config.json``.
    A non-empty ``error`` means this layer count cannot be adapted safely and
    the caller must reject the candidate.
    """
    changes = []
    if model_config is None or layers_new >= layers_old:
        return changes, None

    shares_mtp = _shares_mtp_attention(model_config, mtp_layers)

    for field in LAYER_FIELDS:
        key = next((a for a in field.aliases if a in model_config), None)
        if key is None:
            continue
        value = model_config[key]
        # Scalar forms (e.g. an int window_attn_skip_freq) are layer-count
        # independent and need no rewrite.
        if not isinstance(value, list):
            continue

        tail = mtp_layers if field.includes_mtp else 0
        expected_old = layers_old + tail
        if len(value) != expected_old:
            return [], (
                f"{key} 长度为 {len(value)}，与源 num_hidden_layers"
                f"{'+MTP' if tail else ''}={expected_old} 不一致，"
                f"源 model_config.json 本身不自洽，无法安全裁剪逐层配置"
            )

        layer_part, mtp_part = value[:layers_old], value[layers_old:]
        kept = layer_part[:layers_new]
        extra = ""

        if field.families:
            if shares_mtp and kept and mtp_part and kept[-1] != mtp_part[0]:
                extra = (
                    f"；并把末层的 {kept[-1]} 改成 MTP 层的 {mtp_part[0]}"
                    f"（mtp_shared_last_layer 让末层与 MTP 层共享 attention "
                    f"权重，两者注意力类型必须一致，"
                    f"否则共享权重跨 stage broadcast 时几何不匹配、训练挂死）"
                )
                kept = [*kept[:-1], mtp_part[0]]
            lost = {_csa_family(r) for r in layer_part} - {
                _csa_family(r) for r in kept
            }
            if lost:
                return [], (
                    f"把层数缩到 {layers_new} 会让 {key} 丢掉注意力类型 "
                    f"{sorted(lost)}（框架要求这些类型至少各有一层），"
                    f"该缩容方案不安全"
                )

        changes.append(
            (
                key,
                kept + mtp_part,
                f"逐层配置随层数 {layers_old} -> {layers_new} 裁剪："
                f"保留前 {layers_new} 层"
                + (f" + 末尾 {len(mtp_part)} 个 MTP 层" if mtp_part else "")
                + f"，长度 {len(value)} -> {len(kept + mtp_part)}"
                + extra,
            )
        )

    return changes, None
