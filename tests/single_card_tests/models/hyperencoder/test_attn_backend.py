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

"""Unit tests for the HyperEncoder attention-backend env switches."""

import os
import unittest
from contextlib import contextmanager

from paddlefleet.models.hyperencoder import attn_backend


@contextmanager
def _env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestEncoderAttnBackend(unittest.TestCase):
    def test_default_is_dp(self):
        with _env(HYPERBODY_ENCODER_ATTN_BACKEND=None):
            self.assertEqual(attn_backend.encoder_attn_backend(), "dp")
            self.assertFalse(attn_backend.use_triton_encoder_attn())

    def test_triton_selected(self):
        with _env(HYPERBODY_ENCODER_ATTN_BACKEND="triton"):
            self.assertEqual(attn_backend.encoder_attn_backend(), "triton")
            self.assertTrue(attn_backend.use_triton_encoder_attn())

    def test_case_insensitive(self):
        with _env(HYPERBODY_ENCODER_ATTN_BACKEND="TRITON"):
            self.assertEqual(attn_backend.encoder_attn_backend(), "triton")

    def test_flex_raises(self):
        with (
            _env(HYPERBODY_ENCODER_ATTN_BACKEND="flex"),
            self.assertRaises(ValueError),
        ):
            attn_backend.encoder_attn_backend()

    def test_unknown_backend_raises(self):
        with (
            _env(HYPERBODY_ENCODER_ATTN_BACKEND="nope"),
            self.assertRaises(ValueError),
        ):
            attn_backend.encoder_attn_backend()


class TestUsePackedDecoder(unittest.TestCase):
    def test_default_off(self):
        with _env(
            HYPERBODY_PACKED_FLEX_DECODER=None,
            HYPERBODY_ENCODER_ATTN_BACKEND=None,
        ):
            self.assertFalse(attn_backend.use_packed_decoder())

    def test_on_with_triton(self):
        with _env(
            HYPERBODY_PACKED_FLEX_DECODER="1",
            HYPERBODY_ENCODER_ATTN_BACKEND="triton",
        ):
            self.assertTrue(attn_backend.use_packed_decoder())

    def test_truthy_variants(self):
        for v in ("1", "true", "on", "TRUE", "On"):
            with _env(
                HYPERBODY_PACKED_FLEX_DECODER=v,
                HYPERBODY_ENCODER_ATTN_BACKEND="triton",
            ):
                self.assertTrue(attn_backend.use_packed_decoder())

    def test_falsy_variants(self):
        for v in ("0", "false", "off"):
            with _env(
                HYPERBODY_PACKED_FLEX_DECODER=v,
                HYPERBODY_ENCODER_ATTN_BACKEND="dp",
            ):
                self.assertFalse(attn_backend.use_packed_decoder())

    def test_unknown_value_raises(self):
        with (
            _env(HYPERBODY_PACKED_FLEX_DECODER="maybe"),
            self.assertRaises(ValueError),
        ):
            attn_backend.use_packed_decoder()

    def test_on_without_triton_raises(self):
        with (
            _env(
                HYPERBODY_PACKED_FLEX_DECODER="1",
                HYPERBODY_ENCODER_ATTN_BACKEND="dp",
            ),
            self.assertRaises(RuntimeError),
        ):
            attn_backend.use_packed_decoder()


if __name__ == "__main__":
    unittest.main()
