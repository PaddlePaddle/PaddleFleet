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

"""Tests for the Kimi-K3 vision tower assembled through the Fleet LayerSpecs."""

import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

import paddlefleet.parallel_state as ps
from paddlefleet.models.kimi_k3 import (
    KimiK3VisionEmbedding,
    KimiK3VisionModel,
    build_kimi_k3_vision_config,
    kimi_k3_vision_builder,
    merge_vision_block_diag_mask,
)

PATCH_SIZE = 4
IN_CHANNELS = 3
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 128
TEXT_HIDDEN_SIZE = 96
MERGE_KERNEL = (2, 2)


def _init_distributed():
    # rms_norm / flash attention need GPU; a CPU-only sibling test may have
    # switched the default device in this process.
    paddle.set_device("gpu")
    if ps.have_global_memory_buffer():
        return
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())


def _attn_mask_startend_row_indices(grid_thws: paddle.Tensor):
    """Block-diagonal varlen mask bounds: each media attends only to itself."""
    lengths = paddle.concat(
        [
            paddle.zeros([1], dtype=grid_thws.dtype),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        ]
    )
    cu_seqlens = lengths.cumsum(axis=0).cast("int32")
    repeats = cu_seqlens[1:] - cu_seqlens[:-1]
    lts = paddle.repeat_interleave(cu_seqlens[1:], repeats).reshape(
        [1, 1, -1, 1]
    )
    ute = paddle.repeat_interleave(cu_seqlens[:-1], repeats).reshape(
        [1, 1, -1, 1]
    )
    return paddle.concat([lts, ute], axis=-1)


def _dense_block_diag_mask(grid_thws: paddle.Tensor):
    """Reference dense mask (True == masked out) for the float32 branch."""
    lengths = (
        (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).numpy().tolist()
    )
    total = sum(lengths)
    mask = np.ones([total, total], dtype=bool)
    start = 0
    for length in lengths:
        mask[start : start + length, start : start + length] = False
        start += length
    return mask


class _KimiK3VisionTestMixin:
    """Shared model construction / forward helpers."""

    @classmethod
    def setUpClass(cls):
        _init_distributed()

    def _create_model(self, **overrides):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        kwargs = {
            "patch_size": PATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_attention_heads": NUM_HEADS,
            "num_hidden_layers": NUM_LAYERS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "qkv_hidden_size": HIDDEN_SIZE,
            "init_pos_emb_height": 8,
            "init_pos_emb_width": 8,
            "init_pos_emb_time": 4,
            "merge_kernel_size": MERGE_KERNEL,
            "mm_hidden_size": HIDDEN_SIZE,
            "text_hidden_size": TEXT_HIDDEN_SIZE,
            "max_height": 64,
            "max_width": 64,
        }
        kwargs.update(overrides)
        config = build_kimi_k3_vision_config(**kwargs)
        model = kimi_k3_vision_builder(
            config,
            seg_method="layer:TransformerLayer|EmptyLayer",
            num_stages=config.pipeline_model_parallel_size,
        )
        return model, config

    def _forward(self, model, grid_thws, pixel_values=None, dict_args=None):
        if pixel_values is None:
            num_patches = int(
                (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2])
                .sum()
                .item()
            )
            pixel_values = paddle.randn(
                [num_patches, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
            )
        args = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
        }
        if dict_args is None:
            # Most tests keep supplying the bounds explicitly; the tests that
            # exercise the derived masks pass their own dict_args instead.
            dict_args = {
                "attn_mask_startend_row_indices": _attn_mask_startend_row_indices(
                    grid_thws
                )
            }
        args.update(dict_args)
        return model(args)["hidden_states"]


class TestKimiK3VisionModel(_KimiK3VisionTestMixin, unittest.TestCase):
    def test_model_construction(self):
        model, _ = self._create_model()
        self.assertIsInstance(model, KimiK3VisionModel)
        self.assertGreater(sum(p.numel().item() for p in model.parameters()), 0)

    def test_forward_single_image(self):
        model, _ = self._create_model()
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int32")
        out = self._forward(model, grid_thws)
        self.assertEqual(len(out), 1)
        # one 2x2 merge: (4/2) * (4/2) tokens of text hidden size
        self.assertEqual(list(out[0].shape), [4, TEXT_HIDDEN_SIZE])
        self.assertTrue(paddle.isfinite(out[0]).all().item())

    def test_temporal_pooling_collapses_frames(self):
        model, _ = self._create_model()
        single = self._forward(
            model, paddle.to_tensor([[1, 4, 4]], dtype="int32")
        )
        video = self._forward(
            model, paddle.to_tensor([[2, 4, 4]], dtype="int32")
        )
        # sd2_tpool averages over all frames, so the token count is unchanged
        self.assertEqual(list(video[0].shape), list(single[0].shape))

    def test_forward_multi_image_variable_size(self):
        model, _ = self._create_model()
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 2, 6]], dtype="int32")
        out = self._forward(model, grid_thws)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out[0].shape), [4, TEXT_HIDDEN_SIZE])
        self.assertEqual(list(out[1].shape), [3, TEXT_HIDDEN_SIZE])

    def test_qkv_hidden_size_differs_from_hidden_size(self):
        model, config = self._create_model(qkv_hidden_size=96)
        self.assertEqual(config.head_dim, 96 // NUM_HEADS)
        out = self._forward(model, paddle.to_tensor([[1, 4, 4]], dtype="int32"))
        self.assertEqual(list(out[0].shape), [4, TEXT_HIDDEN_SIZE])

    def test_backward(self):
        model, _ = self._create_model()
        out = self._forward(model, paddle.to_tensor([[1, 4, 4]], dtype="int32"))
        sum(item.sum() for item in out).backward()

        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
                self.assertTrue(
                    paddle.isfinite(param.grad).all().item(),
                    f"non-finite grad for {name}",
                )
        self.assertGreater(params_with_grad, 0)

    def test_rejects_mismatched_grid(self):
        model, _ = self._create_model()
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int32")
        with self.assertRaisesRegex(ValueError, "describes 16 patches"):
            self._forward(
                model,
                grid_thws,
                pixel_values=paddle.randn(
                    [8, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
                ),
            )


class TestKimiK3VisionModelConfigVariations(
    _KimiK3VisionTestMixin, unittest.TestCase
):
    """Config-space coverage, mirroring the Kimi-K2.5 vision tests."""

    def test_construction_with_different_layers(self):
        for num_layers in [1, 2, 4]:
            model, _ = self._create_model(num_hidden_layers=num_layers)
            self.assertIsInstance(model, KimiK3VisionModel)
            out = self._forward(
                model, paddle.to_tensor([[1, 4, 4]], dtype="int32")
            )
            self.assertEqual(list(out[0].shape), [4, TEXT_HIDDEN_SIZE])

    def test_construction_with_different_hidden_size(self):
        for hidden_size in [32, 64, 128]:
            model, config = self._create_model(
                hidden_size=hidden_size,
                qkv_hidden_size=hidden_size,
                mm_hidden_size=hidden_size,
                num_attention_heads=hidden_size // 16,  # head_dim = 16
            )
            self.assertEqual(config.head_dim, 16)
            out = self._forward(
                model, paddle.to_tensor([[1, 4, 4]], dtype="int32")
            )
            self.assertEqual(list(out[0].shape), [4, TEXT_HIDDEN_SIZE])

    def test_different_merge_kernel_size(self):
        # grid is 4x4, so the merged token count is (4 / kh) * (4 / kw)
        for kernel_size, expected_tokens in [
            ((1, 1), 16),
            ((2, 2), 4),
            ((2, 4), 2),
        ]:
            model, _ = self._create_model(merge_kernel_size=kernel_size)
            out = self._forward(
                model, paddle.to_tensor([[1, 4, 4]], dtype="int32")
            )
            self.assertEqual(
                list(out[0].shape),
                [expected_tokens, TEXT_HIDDEN_SIZE],
                f"merge kernel {kernel_size}",
            )

    def test_params_dtype_float32(self):
        model, _ = self._create_model(params_dtype=paddle.float32)
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int32")
        out = self._forward(
            model,
            grid_thws,
            pixel_values=paddle.randn(
                [16, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE], dtype=paddle.float32
            ),
        )
        self.assertEqual(out[0].dtype, paddle.float32)

    def test_params_dtype_bfloat16(self):
        model, _ = self._create_model(params_dtype=paddle.bfloat16)
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int32")
        out = self._forward(
            model,
            grid_thws,
            pixel_values=paddle.randn(
                [16, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
            ).astype(paddle.bfloat16),
        )
        self.assertEqual(out[0].dtype, paddle.bfloat16)
        self.assertTrue(paddle.isfinite(out[0].astype("float32")).all().item())

    def test_empty_layers_in_head_and_tail(self):
        model, _ = self._create_model(
            num_empty_layers_add_in_head=1,
            num_empty_layers_add_in_tail=1,
        )
        out = self._forward(model, paddle.to_tensor([[1, 4, 4]], dtype="int32"))
        self.assertEqual(list(out[0].shape), [4, TEXT_HIDDEN_SIZE])

    def test_no_qk_norm_even_when_config_enables_it(self):
        # MoonViT has no qk norm, so the spec pins it to IdentityOp regardless.
        model, _ = self._create_model(use_qk_norm=True)
        qk_norm_params = [
            name
            for name, _ in model.named_parameters()
            if "q_norm" in name or "k_norm" in name
        ]
        self.assertEqual(qk_norm_params, [])

    def test_forward_with_attention_mask(self):
        model, _ = self._create_model()
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int32")
        out = model(
            {
                "pixel_values": paddle.randn(
                    [16, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
                ),
                "grid_thws": grid_thws,
                "attention_mask": paddle.ones([1, 16]),
                "attn_mask_startend_row_indices": _attn_mask_startend_row_indices(
                    grid_thws
                ),
            }
        )["hidden_states"]
        self.assertEqual(list(out[0].shape), [4, TEXT_HIDDEN_SIZE])

    def test_train_and_eval_mode(self):
        model, _ = self._create_model()
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int32")
        pixel_values = paddle.randn([16, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE])

        model.train()
        train_out = self._forward(model, grid_thws, pixel_values=pixel_values)
        model.eval()
        eval_out = self._forward(model, grid_thws, pixel_values=pixel_values)

        # no dropout is configured, so both modes must agree
        np.testing.assert_allclose(
            train_out[0].numpy(),
            eval_out[0].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_media_are_isolated_by_varlen_mask(self):
        """Patches of one image must not attend to another image.

        The flashmask path (and therefore `attn_mask_startend_row_indices`) is
        only taken for bf16/fp16 inputs; in float32 the attention falls back to
        a dense implementation that ignores the varlen bounds, so this must be
        checked in bf16.
        """
        model, _ = self._create_model(params_dtype=paddle.bfloat16)
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 4, 4]], dtype="int32")
        pixel_values = paddle.randn(
            [32, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
        ).astype(paddle.bfloat16)

        baseline = self._forward(model, grid_thws, pixel_values=pixel_values)

        perturbed_pixels = pixel_values.clone()
        perturbed_pixels[16:] = paddle.randn(
            [16, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
        ).astype(paddle.bfloat16)
        perturbed = self._forward(
            model, grid_thws, pixel_values=perturbed_pixels
        )

        np.testing.assert_allclose(
            baseline[0].astype("float32").numpy(),
            perturbed[0].astype("float32").numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        self.assertFalse(
            np.allclose(
                baseline[1].astype("float32").numpy(),
                perturbed[1].astype("float32").numpy(),
            )
        )


class TestKimiK3VisionDefaultMask(_KimiK3VisionTestMixin, unittest.TestCase):
    """The tower must derive per-media bounds itself, whatever the caller sends.

    `DotProductAttention` reads `attn_mask_startend_row_indices` only on the
    bf16/fp16 flashmask branch and `attention_mask` only on the dense branch, so
    a caller passing just one of them must still get media isolation.
    """

    def _embedding_stage(self, model):
        stages = [
            layer
            for layer in model.sublayers()
            if isinstance(layer, KimiK3VisionEmbedding)
        ]
        self.assertEqual(len(stages), 1)
        return stages[0]

    def _isolation_diff(self, model, grid_thws, dict_args):
        """Max change in media 0 output when media 1 pixels are replaced."""
        first = int(
            (grid_thws[0, 0] * grid_thws[0, 1] * grid_thws[0, 2]).item()
        )
        total = int(
            (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).sum().item()
        )
        pixel_values = paddle.randn(
            [total, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
        )
        perturbed = pixel_values.clone()
        perturbed[first:] = (
            paddle.randn([total - first, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE])
            * 5
        )

        baseline = self._forward(
            model, grid_thws, pixel_values=pixel_values, dict_args=dict_args
        )
        changed = self._forward(
            model, grid_thws, pixel_values=perturbed, dict_args=dict_args
        )
        return float((baseline[0] - changed[0]).abs().max())

    def test_no_mask_isolates_media_in_float32(self):
        model, _ = self._create_model()
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 2, 6]], dtype="int32")
        self.assertEqual(self._isolation_diff(model, grid_thws, {}), 0.0)

    def test_startend_only_still_isolates_media_in_float32(self):
        # float32 takes the dense branch, which ignores the varlen bounds, so
        # the tower has to add the equivalent dense mask on top of them.
        model, _ = self._create_model()
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 2, 6]], dtype="int32")
        dict_args = {
            "attn_mask_startend_row_indices": _attn_mask_startend_row_indices(
                grid_thws
            )
        }
        self.assertEqual(self._isolation_diff(model, grid_thws, dict_args), 0.0)

    def test_attention_mask_only_still_builds_startend_bounds(self):
        # bf16 takes the flashmask branch, which ignores attention_mask, so the
        # bounds must be filled in even though a mask was supplied.
        model, _ = self._create_model(params_dtype=paddle.bfloat16)
        embedding = self._embedding_stage(model)
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 2, 6]], dtype="int32")
        total = 28
        out = embedding(
            {
                "pixel_values": paddle.randn(
                    [total, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
                ).astype(paddle.bfloat16),
                "grid_thws": grid_thws,
                "attention_mask": paddle.ones(
                    [1, 1, total, total], dtype="bool"
                ),
            }
        )
        np.testing.assert_array_equal(
            out["attn_mask_startend_row_indices"].numpy(),
            _attn_mask_startend_row_indices(grid_thws).numpy(),
        )

    def test_dense_mask_merges_caller_mask(self):
        grid_thws = paddle.to_tensor([[1, 2, 2], [1, 1, 2]], dtype="int32")
        expected = _dense_block_diag_mask(grid_thws)

        derived = merge_vision_block_diag_mask(None, grid_thws)
        np.testing.assert_array_equal(derived.numpy()[0, 0], expected)

        # Float masks use 1.0 == attend, so masking token 0 must survive the or.
        caller = paddle.ones([1, 1, 6, 6], dtype="float32")
        caller[:, :, :, 0] = 0.0
        merged = merge_vision_block_diag_mask(caller, grid_thws).numpy()[0, 0]
        self.assertTrue(bool(merged[:, 0].all()))
        np.testing.assert_array_equal(merged[:, 1:], expected[:, 1:])

    def test_no_mask_isolates_media_in_bfloat16(self):
        # bf16 takes the flashmask branch; with nothing supplied the tower has
        # to derive the bounds, otherwise SDPA would apply a causal mask.
        model, _ = self._create_model(params_dtype=paddle.bfloat16)
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 4, 4]], dtype="int32")
        pixel_values = paddle.randn(
            [32, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
        ).astype(paddle.bfloat16)
        perturbed = pixel_values.clone()
        perturbed[16:] = paddle.randn(
            [16, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]
        ).astype(paddle.bfloat16)

        baseline = self._forward(
            model, grid_thws, pixel_values=pixel_values, dict_args={}
        )
        changed = self._forward(
            model, grid_thws, pixel_values=perturbed, dict_args={}
        )

        np.testing.assert_array_equal(
            baseline[0].astype("float32").numpy(),
            changed[0].astype("float32").numpy(),
        )
        # Sanity check that the perturbation was visible somewhere at all.
        self.assertFalse(
            np.allclose(
                baseline[1].astype("float32").numpy(),
                changed[1].astype("float32").numpy(),
            )
        )

    def test_dense_mask_merges_padding_mask(self):
        # A [batch, kv] padding mask is broadcast over the query axis.
        grid_thws = paddle.to_tensor([[1, 2, 2], [1, 1, 2]], dtype="int32")
        expected = _dense_block_diag_mask(grid_thws)
        caller = paddle.ones([1, 6], dtype="float32")
        caller[:, 0] = 0.0
        merged = merge_vision_block_diag_mask(caller, grid_thws).numpy()[0, 0]
        self.assertTrue(bool(merged[:, 0].all()))
        np.testing.assert_array_equal(merged[:, 1:], expected[:, 1:])

    def test_dense_mask_rejects_3d_caller_mask(self):
        grid_thws = paddle.to_tensor([[1, 2, 2]], dtype="int32")
        with self.assertRaisesRegex(ValueError, r"\[batch, heads, q, kv\]"):
            merge_vision_block_diag_mask(
                paddle.ones([1, 4, 4], dtype="bool"), grid_thws
            )


if __name__ == "__main__":
    unittest.main()
