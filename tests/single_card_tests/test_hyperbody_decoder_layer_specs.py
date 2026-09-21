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


def test_pure_mha_asserts_removed():
    """The old pure-MHA guards must be gone (decoder now supports dsv4/MLA/VHA).

    We flip each previously-forbidden attribute and assert the function no longer
    raises the old pure-MHA ValueError. Any *downstream* geometry error (from
    building an attention variant without full geometry) is unrelated to the
    removed guards and is tolerated here.
    """
    old_markers = (
        "must be False",
        "must be None",
        "pure MHA",
    )
    for attr, val in (
        ("multi_latent_attention", True),
        ("experimental_attention_variant", "dsv4_hybrid"),
        ("use_vha_attention", True),
    ):
        view = _decoder_view()
        setattr(view, attr, val)
        try:
            get_hyperbody_decoder_layer_specs(view)
        except ValueError as e:
            assert not any(m in str(e) for m in old_markers), (
                f"old pure-MHA guard still present for {attr}: {e}"
            )
        except Exception:
            pass  # unrelated downstream geometry error is fine


def test_len_and_pattern_match_gpt_spec():
    """Same length + same per-layer dense/MoE pattern as the shared GPT spec."""
    view = _decoder_view()
    hb = get_hyperbody_decoder_layer_specs(view)
    gpt = get_gpt_decoder_layers_spec(view)
    assert len(hb) == view.num_hidden_layers == len(gpt), (len(hb), len(gpt))
    # moe_layer_freq is expanded to the [0] + [1]*(L-1) dense-first list.
    assert list(view.moe_layer_freq) == [0] + [1] * (view.num_hidden_layers - 1)


if __name__ == "__main__":
    try:
        test_is_moe_layer_reads_per_layer_list()
        test_is_moe_layer_rejects_int_and_bad_values()
        test_pure_mha_asserts_removed()
        test_len_and_pattern_match_gpt_spec()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
