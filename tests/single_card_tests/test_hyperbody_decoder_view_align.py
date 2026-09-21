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

"""Unit coverage for the HyperBody decoder-view alignment knobs.

Pure config-view assertions (no GPU / no distributed init) for the fixes carried
onto the nested (``sub_configs``) config:

  * ``_attn_implementation`` is yaml/config-configurable and passes through onto
    the decoder view (a fixed decoder kernel is required for bit-exact alignment
    with standalone ernie5_v2 -- the MLA core diverges under ``eager`` vs the
    fused ``default`` path);
  * ``first_k_dense_replace`` truthy no longer collides with a list
    ``moe_layer_freq`` (the view is materialized with an int freq so the
    provider ``__post_init__`` builds the ``[0] + [1]*(L-1)`` dense-first table);
  * nested ``decoder_config`` / ``encoder_config`` dicts are promoted to their
    sub-config objects by ``HyperBodyConfig``.

Geometry is REQUIRED by the sensitive-info policy, so configs are built with
explicit ``decoder_config=`` / ``encoder_config=`` sub-configs.
"""

import sys

from paddlefleet.transformers.hyperbody.configuration import (
    HyperBodyConfig,
    HyperBodyDecoderConfig,
    HyperBodyEncoderConfig,
)
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


def _make_config(decoder_overrides=None, **top_level):
    decoder_config = {**_DECODER_GEOMETRY, **(decoder_overrides or {})}
    return HyperBodyConfig(
        decoder_config=decoder_config,
        encoder_config={**_ENCODER_GEOMETRY},
        **top_level,
    )


def test_nested_dict_promoted_to_subconfigs():
    """Nested ``decoder_config`` / ``encoder_config`` dicts become sub-config objects."""
    cfg = _make_config()
    assert isinstance(cfg.decoder_config, HyperBodyDecoderConfig)
    assert isinstance(cfg.encoder_config, HyperBodyEncoderConfig)
    assert cfg.decoder_config.num_hidden_layers == 4
    assert cfg.encoder_config.num_hidden_layers == 2


def test_attn_implementation_passthrough_to_view():
    """``_attn_implementation`` flows from config onto the decoder view.

    The fused ``default`` kernel is what standalone ernie5_v2(lite) uses; the
    HyperBody decoder view must be able to select it (else the MLA core runs
    ``eager`` and diverges from lite).
    """
    for impl in ("default", "eager"):
        cfg = _make_config(_attn_implementation=impl)
        view = _build_decoder_view(cfg)
        assert view._attn_implementation == impl, (
            impl,
            view._attn_implementation,
        )


def test_attn_implementation_falls_back_to_default_when_absent():
    """When config carries no truthy ``_attn_implementation`` the view uses 'default'."""
    cfg = _make_config()
    # Simulate an absent/empty attribute (older/hand-built configs).
    cfg._attn_implementation = None
    view = _build_decoder_view(cfg)
    assert view._attn_implementation == "default", view._attn_implementation


def test_first_k_dense_replace_builds_dense_first_without_conflict():
    """``first_k_dense_replace`` truthy must not collide with a list moe_layer_freq.

    The view builder passes an int ``moe_layer_freq`` to the provider so the
    provider builds the ``[0] + [1]*(L-1)`` dense-first table (layer 0 dense).
    """
    cfg = _make_config(decoder_overrides={"first_k_dense_replace": 1})
    view = _build_decoder_view(cfg)  # must not raise
    freq = view.moe_layer_freq
    assert isinstance(freq, (list, tuple)), freq
    assert freq[0] == 0, freq  # layer 0 dense
    assert all(x == 1 for x in freq[1:]), freq  # rest MoE


if __name__ == "__main__":
    try:
        test_nested_dict_promoted_to_subconfigs()
        test_attn_implementation_passthrough_to_view()
        test_attn_implementation_falls_back_to_default_when_absent()
        test_first_k_dense_replace_builds_dense_first_without_conflict()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
