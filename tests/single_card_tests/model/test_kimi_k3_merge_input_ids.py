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

import unittest

import numpy as np
import paddle

from paddlefleet.models.kimi_k3 import merge_input_ids_with_image_features

IMG = 100
PAD = 0
IGNORE = -100
EMBED_DIM = 4


class TestKimiK3Fusion(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        np.random.seed(0)
        paddle.set_device("cpu")

    def test_single_sample_exact(self):
        # input: [tok5, tok6, IMG(->3 tokens), tok7] -> expanded length 6
        input_ids = paddle.to_tensor([[5, 6, IMG, 7]], dtype="int64")
        attention_mask = paddle.ones([1, 4], dtype="int64")
        inputs_embeds = paddle.arange(
            1, 4 * EMBED_DIM + 1, dtype="float32"
        ).reshape([1, 4, EMBED_DIM])
        img_feat = paddle.arange(
            100, 100 + 3 * EMBED_DIM, dtype="float32"
        ).reshape([3, EMBED_DIM])

        emb, mask, labels, pos = merge_input_ids_with_image_features(
            [img_feat],
            inputs_embeds,
            input_ids,
            attention_mask,
            image_token_index=IMG,
            pad_token_id=PAD,
            ignore_index=IGNORE,
        )

        self.assertEqual(list(emb.shape), [1, 6, EMBED_DIM])
        # Expected order: emb(5), emb(6), img0, img1, img2, emb(7)
        expected = paddle.concat(
            [
                inputs_embeds[0, 0:2],
                img_feat,
                inputs_embeds[0, 3:4],
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            emb[0].numpy(), expected.numpy(), rtol=0, atol=0
        )
        # Full attention, contiguous position ids
        self.assertTrue(bool((mask[0] == 1).all().item()))
        np.testing.assert_array_equal(pos[0].numpy(), np.arange(6))
        self.assertIsNone(labels)

    def test_labels_ignore_on_image(self):
        input_ids = paddle.to_tensor([[5, IMG, 7]], dtype="int64")
        attention_mask = paddle.ones([1, 3], dtype="int64")
        inputs_embeds = paddle.randn([1, 3, EMBED_DIM])
        labels = paddle.to_tensor([[5, 55, 7]], dtype="int64")
        img_feat = paddle.randn([2, EMBED_DIM])

        _, _, out_labels, _ = merge_input_ids_with_image_features(
            [img_feat],
            inputs_embeds,
            input_ids,
            attention_mask,
            image_token_index=IMG,
            pad_token_id=PAD,
            ignore_index=IGNORE,
            labels=labels,
        )
        # Expanded: [lbl5, ignore, ignore, lbl7]
        expected = np.array([5, IGNORE, IGNORE, 7])
        np.testing.assert_array_equal(out_labels[0].numpy(), expected)

    def test_right_padding_exact_order(self):
        # sample0: [5,6,IMG,7] img_len 3 -> 6 slots (no padding)
        # sample1: [8,IMG,9,PAD] img_len 2 -> 5 used slots, 1 trailing pad slot
        input_ids = paddle.to_tensor(
            [[5, 6, IMG, 7], [8, IMG, 9, PAD]], dtype="int64"
        )
        attention_mask = paddle.to_tensor(
            [[1, 1, 1, 1], [1, 1, 1, 0]], dtype="int64"
        )
        inputs_embeds = paddle.arange(
            1, 2 * 4 * EMBED_DIM + 1, dtype="float32"
        ).reshape([2, 4, EMBED_DIM])
        feats = [
            paddle.full([3, EMBED_DIM], 100.0),
            paddle.full([2, EMBED_DIM], 200.0),
        ]

        emb, mask, _, pos = merge_input_ids_with_image_features(
            feats,
            inputs_embeds,
            input_ids,
            attention_mask,
            image_token_index=IMG,
            pad_token_id=PAD,
        )

        self.assertEqual(list(emb.shape), [2, 6, EMBED_DIM])
        # row0: emb(5), emb(6), img0 x3, emb(7)
        expected0 = paddle.concat(
            [inputs_embeds[0, 0:2], feats[0], inputs_embeds[0, 3:4]], axis=0
        )
        # row1: emb(8), img1 x2, emb(9), zeroed pad, unused slot
        expected1 = paddle.concat(
            [
                inputs_embeds[1, 0:1],
                feats[1],
                inputs_embeds[1, 2:3],
                paddle.zeros([2, EMBED_DIM]),
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            emb.numpy(),
            paddle.stack([expected0, expected1]).numpy(),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            mask.numpy(), np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        )
        np.testing.assert_array_equal(pos[0].numpy(), np.arange(6))
        np.testing.assert_array_equal(pos[1, :4].numpy(), np.arange(4))

    def test_left_padding_exact_order(self):
        # sample0: [5,6,IMG,7] img_len 3 -> 6 slots
        # sample1: [PAD,8,IMG,9] img_len 2 -> 5 used slots, 1 leading pad slot
        input_ids = paddle.to_tensor(
            [[5, 6, IMG, 7], [PAD, 8, IMG, 9]], dtype="int64"
        )
        attention_mask = paddle.to_tensor(
            [[1, 1, 1, 1], [0, 1, 1, 1]], dtype="int64"
        )
        inputs_embeds = paddle.arange(
            1, 2 * 4 * EMBED_DIM + 1, dtype="float32"
        ).reshape([2, 4, EMBED_DIM])
        feats = [
            paddle.full([3, EMBED_DIM], 100.0),
            paddle.full([2, EMBED_DIM], 200.0),
        ]

        emb, mask, _, pos = merge_input_ids_with_image_features(
            feats,
            inputs_embeds,
            input_ids,
            attention_mask,
            image_token_index=IMG,
            pad_token_id=PAD,
        )

        # row1: unused slot, zeroed pad, emb(8), img1 x2, emb(9)
        expected1 = paddle.concat(
            [
                paddle.zeros([2, EMBED_DIM]),
                inputs_embeds[1, 1:2],
                feats[1],
                inputs_embeds[1, 3:4],
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            emb[1].numpy(), expected1.numpy(), rtol=0, atol=0
        )
        np.testing.assert_array_equal(
            mask.numpy(), np.array([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
        )
        np.testing.assert_array_equal(pos[1, 2:].numpy(), np.arange(4))

    def test_two_images_one_sample(self):
        # two image placeholders in one sample, different feature lengths
        input_ids = paddle.to_tensor([[5, IMG, 6, IMG, 7]], dtype="int64")
        attention_mask = paddle.ones([1, 5], dtype="int64")
        inputs_embeds = paddle.randn([1, 5, EMBED_DIM])
        feats = [paddle.randn([2, EMBED_DIM]), paddle.randn([3, EMBED_DIM])]

        emb, mask, _, pos = merge_input_ids_with_image_features(
            feats,
            inputs_embeds,
            input_ids,
            attention_mask,
            image_token_index=IMG,
            pad_token_id=PAD,
        )
        # 3 text + 2 + 3 image = 8 tokens
        self.assertEqual(list(emb.shape), [1, 8, EMBED_DIM])
        np.testing.assert_array_equal(pos[0].numpy(), np.arange(8))


if __name__ == "__main__":
    unittest.main()
