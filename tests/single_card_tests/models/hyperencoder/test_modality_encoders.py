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

"""Unit tests for the HyperEncoder image / audio modality towers."""

import unittest

import paddle

# The accuracy-compatible conv path routes conv2d to a kernel registered only
# for GPU; disable it so the towers exercise the standard kernel.
paddle.set_flags({"FLAGS_use_accuracy_compatible_kernel": False})

from paddlefleet.models.hyperencoder.modality_encoders import (
    AudioEncoderConv,
    ImageEncoderConv,
    MlpProjector,
    PatchEmbed,
    get_abs_pos_1d,
    get_abs_pos_2d,
)


class TestMlpProjector(unittest.TestCase):
    def test_identity(self):
        proj = MlpProjector(
            {"projector_type": "identity", "input_dim": 4, "n_embed": 4}
        )
        x = paddle.randn([2, 4])
        self.assertTrue(paddle.allclose(proj(x), x))

    def test_linear_forward_and_backward(self):
        proj = MlpProjector(
            {"projector_type": "linear", "input_dim": 4, "n_embed": 6}
        )
        x = paddle.randn([3, 4])
        x.stop_gradient = False
        out = proj(x)
        self.assertEqual(out.shape, [3, 6])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(proj.layers.weight.grad)
        self.assertIsNotNone(proj.layers.bias.grad)

    def test_mlp_gelu(self):
        proj = MlpProjector(
            {
                "projector_type": "mlp_gelu",
                "input_dim": 4,
                "n_embed": 5,
                "depth": 2,
            }
        )
        x = paddle.randn([2, 4])
        out = proj(x)
        self.assertEqual(out.shape, [2, 5])

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            MlpProjector(
                {"projector_type": "nope", "input_dim": 4, "n_embed": 4}
            )

    def test_linear_no_bias(self):
        proj = MlpProjector(
            {"projector_type": "linear", "input_dim": 4, "n_embed": 6}
        )
        proj.layers.bias = None
        x = paddle.randn([3, 4])
        out = proj(x)
        self.assertEqual(out.shape, [3, 6])


class TestPatchEmbed(unittest.TestCase):
    def test_output_is_channel_last(self):
        pe = PatchEmbed(
            kernel_size=(2, 2), stride=(2, 2), in_chans=3, embed_dim=8
        )
        x = paddle.randn([1, 3, 4, 4])
        out = pe(x)
        # [B, H//2, W//2, embed_dim]
        self.assertEqual(out.shape, [1, 2, 2, 8])


class TestAbsPos2d(unittest.TestCase):
    def test_short_circuit_when_sizes_match(self):
        abs_pos = paddle.randn([1, 4, 4, 8])
        out = get_abs_pos_2d(abs_pos, (4, 4))
        self.assertTrue(out is abs_pos)

    def test_interpolates_when_sizes_differ(self):
        abs_pos = paddle.randn([1, 4, 4, 8])
        out = get_abs_pos_2d(abs_pos, (2, 2))
        self.assertEqual(out.shape, [1, 2, 2, 8])


class TestAbsPos1d(unittest.TestCase):
    def test_short_circuit_when_sizes_match(self):
        abs_pos = paddle.randn([1, 10, 8])
        out = get_abs_pos_1d(abs_pos, 10)
        self.assertTrue(out is abs_pos)

    def test_interpolates_when_sizes_differ(self):
        abs_pos = paddle.randn([1, 10, 8])
        out = get_abs_pos_1d(abs_pos, 5)
        self.assertEqual(out.shape, [1, 5, 8])


class TestImageEncoderConv(unittest.TestCase):
    def test_forward_and_backward(self):
        enc = ImageEncoderConv(
            img_size=8, patch_size=2, in_chans=3, embed_dim=8, out_chans=16
        )
        x = paddle.randn([1, 3, 8, 8])
        x.stop_gradient = False
        out = enc(x)
        # grid = 8 // 2 = 4
        self.assertEqual(out.shape, [1, 4, 4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_pos_embed_interpolated_when_grid_differs(self):
        enc = ImageEncoderConv(
            img_size=8, patch_size=2, in_chans=3, embed_dim=8, out_chans=16
        )
        # A 12x12 image -> 6x6 grid, forcing the pos-embed interpolation branch.
        x = paddle.randn([1, 3, 12, 12])
        out = enc(x)
        self.assertEqual(out.shape, [1, 6, 6, 8])


class TestAudioEncoderConv(unittest.TestCase):
    def test_forward_and_backward(self):
        enc = AudioEncoderConv(
            num_mel_bins=4, embed_dim=8, max_position_embeddings=64
        )
        x = paddle.randn([1, 4, 20])
        x.stop_gradient = False
        out = enc(x)
        # T' = (20 + 1) // 2 = 10 ; shape [B, 1, T', embed_dim]
        self.assertEqual(out.shape, [1, 1, 10, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
