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

"""Unit tests for the HyperEncoder attention-backend config helpers."""

import types
import unittest

from paddlefleet.models.hyperencoder import attn_backend


def _cfg(**kw):
    """A duck-typed config carrying only the hyperencoder_* fields under test."""
    return types.SimpleNamespace(**kw)


class TestEncoderAttnBackend(unittest.TestCase):
    def test_default_is_dp(self):
        # A config without the field falls back to the "dp" default.
        self.assertEqual(attn_backend.encoder_attn_backend(_cfg()), "dp")
        self.assertFalse(attn_backend.use_triton_encoder_attn(_cfg()))

    def test_triton_selected(self):
        c = _cfg(hyperencoder_attn_backend="triton")
        self.assertEqual(attn_backend.encoder_attn_backend(c), "triton")
        self.assertTrue(attn_backend.use_triton_encoder_attn(c))

    def test_case_insensitive(self):
        c = _cfg(hyperencoder_attn_backend="TRITON")
        self.assertEqual(attn_backend.encoder_attn_backend(c), "triton")

    def test_flex_raises(self):
        with self.assertRaises(ValueError):
            attn_backend.encoder_attn_backend(
                _cfg(hyperencoder_attn_backend="flex")
            )

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            attn_backend.encoder_attn_backend(
                _cfg(hyperencoder_attn_backend="nope")
            )


class TestUsePackedDecoder(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(attn_backend.use_packed_decoder(_cfg()))

    def test_on_with_triton(self):
        c = _cfg(
            hyperencoder_packed_decoder=True,
            hyperencoder_attn_backend="triton",
        )
        self.assertTrue(attn_backend.use_packed_decoder(c))

    def test_off_with_dp(self):
        c = _cfg(
            hyperencoder_packed_decoder=False,
            hyperencoder_attn_backend="dp",
        )
        self.assertFalse(attn_backend.use_packed_decoder(c))

    def test_on_without_triton_raises(self):
        c = _cfg(
            hyperencoder_packed_decoder=True,
            hyperencoder_attn_backend="dp",
        )
        with self.assertRaises(RuntimeError):
            attn_backend.use_packed_decoder(c)


if __name__ == "__main__":
    unittest.main()
