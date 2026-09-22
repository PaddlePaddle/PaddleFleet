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

"""Unit coverage for ``get_hyperbody_decoder_layer_specs``.

The HyperBody decoder-layers spec now builds the same layers as the shared
``get_gpt_decoder_layers_spec`` (so the decoder can be configured into the
ernie5_v2 / dsv4_hybrid architecture) while keeping the stricter per-layer 0/1
list ``moe_layer_freq`` contract. This module asserts:

  * ``_is_moe_layer`` reads the per-layer 0/1 list and rejects int / non-0/1 /
    length-mismatch;
  * the previous pure-MHA guards (multi_latent_attention /
    experimental_attention_variant / use_vha_attention) are gone;
  * the spec list length + per-layer dense/MoE pattern match
    ``get_gpt_decoder_layers_spec`` on the same config.

Pure spec-construction assertions (no GPU / no distributed init).
"""

import sys
from types import SimpleNamespace

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_decoder_layers_spec
from paddlefleet.models.hyperbody.decoder_layer_specs import (
    _is_moe_layer,
    get_hyperbody_decoder_layer_specs,
)
from paddlefleet.transformers.hyperbody.configuration import HyperBodyConfig
from paddlefleet.transformers.hyperbody.modeling import _build_decoder_view

_DECODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
}
_ENCODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "hyperencoder_query_lengths": (16, 64),
}


def _decoder_view():
    cfg = HyperBodyConfig(
        decoder_config={**_DECODER_GEOMETRY},
        encoder_config={**_ENCODER_GEOMETRY},
    )
    return _build_decoder_view(cfg)


def _decoder_view_with(decoder_overrides):
    cfg = HyperBodyConfig(
        decoder_config={**_DECODER_GEOMETRY, **decoder_overrides},
        encoder_config={**_ENCODER_GEOMETRY},
    )
    return _build_decoder_view(cfg)


def _mlp_names(specs):
    """Per-layer MLP class name: 'MLP' (dense) vs 'MoELayer' (expert)."""
    return [s.sublayers_spec.mlp.layer.__name__ for s in specs]


# Full geometry for the DSV4-hybrid attention family (CSA + hybrid-MLA).
_DSV4_OVERRIDES = {
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "experimental_attention_variant": "dsv4_hybrid",
    "gated_attention": True,
    "use_qk_norm": True,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "qk_nope_head_dim": 24,
    "v_head_dim": 32,
    "hybrid_mla_q_lora_rank": 32,
    "hybrid_mla_kv_lora_rank": 16,
    "hybrid_mla_qk_nope_head_dim": 24,
    "hybrid_mla_qk_rope_head_dim": 8,
    "hybrid_mla_v_head_dim": 16,
    "hybrid_mla_num_attention_heads": 8,
    "hybrid_mla_num_key_value_heads": 8,
    # ratio -2 marks the hybrid-MLA layer; 128 marks the CSA layers.
    "csa_compress_ratios": [128, 128, -2],
    "csa_window_size": 128,
}
# Multi-latent-attention (MLA) geometry.
_MLA_OVERRIDES = {
    "multi_latent_attention": True,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "qk_nope_head_dim": 8,
    "v_head_dim": 16,
}


def test_is_moe_layer_reads_per_layer_list():
    """Per-layer 0/1 list: element i decides dense(0)/MoE(1)."""
    cfg = SimpleNamespace(moe_layer_freq=[0, 1, 1, 1], num_hidden_layers=4)
    assert [_is_moe_layer(cfg, i) for i in range(4)] == [
        False,
        True,
        True,
        True,
    ]


def test_is_moe_layer_rejects_int_and_bad_values():
    """int (i%N semantics), non-0/1 entries, and length mismatch all raise."""
    raised = 0
    for cfg, exc in [
        (SimpleNamespace(moe_layer_freq=1, num_hidden_layers=4), TypeError),
        (
            SimpleNamespace(moe_layer_freq=[0, 2, 1, 1], num_hidden_layers=4),
            ValueError,
        ),
        (
            SimpleNamespace(moe_layer_freq=[0, 1], num_hidden_layers=4),
            ValueError,
        ),
    ]:
        try:
            _is_moe_layer(cfg, 1)
        except exc:
            raised += 1
    assert raised == 3, raised


def test_supports_dsv4_hybrid_variant():
    """Decoder now BUILDS the dsv4_hybrid family (old code hard-raised on it).

    Positively assert a real build: ``num_hidden_layers`` specs with the
    dense-first MLP pattern (layer 0 dense, rest MoE). This is the change's goal
    -- not merely the absence of the old error string.
    """
    view = _decoder_view_with(_DSV4_OVERRIDES)
    specs = get_hyperbody_decoder_layer_specs(view)
    assert len(specs) == view.num_hidden_layers == 3
    assert _mlp_names(specs) == ["MLP"] + ["MoELayer"] * 2


def test_supports_multi_latent_attention():
    """Decoder now BUILDS with multi_latent_attention=True (old code raised)."""
    view = _decoder_view_with(_MLA_OVERRIDES)
    specs = get_hyperbody_decoder_layer_specs(view)
    assert len(specs) == view.num_hidden_layers == 4


def test_len_and_pattern_match_gpt_spec():
    """Same length AND same per-layer dense/MoE MLP class as the shared GPT spec.

    Locks the core equivalence claim: HyperBody's spec builds the *same layers*
    as ``get_gpt_decoder_layers_spec`` -- compared per layer by MLP class
    (dense ``MLP`` vs expert ``MoELayer``), not just by list length.
    """
    for overrides in ({}, _DSV4_OVERRIDES):
        view = _decoder_view_with(overrides)
        hb = get_hyperbody_decoder_layer_specs(view)
        gpt = get_gpt_decoder_layers_spec(view)
        assert len(hb) == view.num_hidden_layers == len(gpt), (
            overrides,
            len(hb),
            len(gpt),
        )
        assert _mlp_names(hb) == _mlp_names(gpt), (
            overrides,
            _mlp_names(hb),
            _mlp_names(gpt),
        )
        # dense-first: layer 0 dense, the rest MoE.
        assert _mlp_names(hb) == ["MLP"] + ["MoELayer"] * (
            view.num_hidden_layers - 1
        )


if __name__ == "__main__":
    try:
        test_is_moe_layer_reads_per_layer_list()
        test_is_moe_layer_rejects_int_and_bad_values()
        test_supports_dsv4_hybrid_variant()
        test_supports_multi_latent_attention()
        test_len_and_pattern_match_gpt_spec()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
