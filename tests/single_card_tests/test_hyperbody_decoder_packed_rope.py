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

"""Unit coverage for the ``decoder_packed_rope`` switch on ``HyperBodyConfig``.

``decoder_packed_rope`` decouples the decoder's per-segment RoPE reset from
``hyperencoder_packed_decoder``:
  * unset (``None``)  -> falls back to ``hyperencoder_packed_decoder`` so the
    historical behavior is unchanged;
  * set explicitly    -> overrides, independently of the encoder flag.

These are pure config-view assertions (no GPU / no distributed init): they
exercise ``modeling._build_decoder_view`` which only materializes the decoder
provider view and sets ``packed_decoder_rope``.
"""

import sys

from paddlefleet.transformers.hyperbody.configuration import HyperBodyConfig
from paddlefleet.transformers.hyperbody.modeling import _build_decoder_view


def test_decoder_packed_rope_defaults_to_hyperencoder_packed_decoder():
    """When unset, ``packed_decoder_rope`` mirrors ``hyperencoder_packed_decoder``."""
    for packed in (True, False):
        cfg = HyperBodyConfig(hyperencoder_packed_decoder=packed)
        assert cfg.decoder_packed_rope is None
        view = _build_decoder_view(cfg)
        assert view.packed_decoder_rope is packed, (
            packed,
            view.packed_decoder_rope,
        )


def test_decoder_packed_rope_explicit_overrides_fallback():
    """An explicit value wins regardless of ``hyperencoder_packed_decoder``."""
    for explicit in (True, False):
        for packed in (True, False):
            cfg = HyperBodyConfig(
                hyperencoder_packed_decoder=packed, decoder_packed_rope=explicit
            )
            assert cfg.decoder_packed_rope is explicit
            view = _build_decoder_view(cfg)
            assert view.packed_decoder_rope is explicit, (
                explicit,
                packed,
                view.packed_decoder_rope,
            )


def test_triton_backend_requires_packed_decoder():
    """The 'triton' frontend without packed decoding is rejected at construction."""
    raised = False
    try:
        HyperBodyConfig(
            hyperencoder_attn_backend="triton",
            hyperencoder_packed_decoder=False,
        )
    except ValueError as e:
        assert "requires hyperencoder_packed_decoder" in str(e)
        raised = True
    assert raised, "triton + non-packed should raise ValueError"

    # The guard is case-insensitive (mirrors encoder_attn_backend), so an
    # upper-cased 'TRITON' + non-packed must also be rejected at construction.
    raised = False
    try:
        HyperBodyConfig(
            hyperencoder_attn_backend="TRITON",
            hyperencoder_packed_decoder=False,
        )
    except ValueError as e:
        assert "requires hyperencoder_packed_decoder" in str(e)
        raised = True
    assert raised, "TRITON + non-packed should raise ValueError"

    # triton + packed is accepted.
    cfg = HyperBodyConfig(
        hyperencoder_attn_backend="triton", hyperencoder_packed_decoder=True
    )
    assert cfg.hyperencoder_attn_backend == "triton"


if __name__ == "__main__":
    try:
        test_decoder_packed_rope_defaults_to_hyperencoder_packed_decoder()
        test_decoder_packed_rope_explicit_overrides_fallback()
        test_triton_backend_requires_packed_decoder()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
