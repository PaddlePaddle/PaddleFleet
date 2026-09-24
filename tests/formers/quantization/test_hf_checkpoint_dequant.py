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
"""Behavior tests for checkpoint dequantization and HF quantized-checkpoint loading.

Environment: 无卡 (CPU). Every tensor op exercised here (LUT gather, block-scale
broadcast, nibble unpacking, element-wise multiply) runs on CPU paddle, so the
numeric contracts are genuinely executed without a GPU. No GPU-only kernel is
claimed. Tests pin ``paddle.set_device("cpu")`` so results never depend on the
CI device.

Expected values are hand-derived from the fp8-e4m3 / ue8m0 / fp4-e2m1 storage
definitions below, never by calling the function under test.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import paddle
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorMetadata,
    Metadata,
)

from paddlefleet.quantization.checkpoint_dequant import (
    get_checkpoint_dequantizer,
)
from paddlefleet.quantization.hf_checkpoint import (
    HFDequantLoadTransform,
    QuanDescriptor,
    build_hf_dequant_load_transform,
    hf_checkpoint_is_quantized,
)

# ---------------------------------------------------------------------------
# Hand-derived raw storage codes (independent of the module under test).
#
# e4m3 byte -> value, using magnitude = (1 + m/8) * 2**(e-7) for e != 0:
#   0x38 (e=7,m=0) -> 1.0        0x30 (e=6,m=0) -> 0.5
#   0x40 (e=8,m=0) -> 2.0        0x3C (e=7,m=4) -> 1.5
#   0xB8 -> sign bit set on 0x38 -> -1.0        0x00 -> 0.0
# ue8m0 byte -> 2**(code-127):
#   125 -> 0.25   126 -> 0.5   127 -> 1.0   128 -> 2.0   129 -> 4.0
# fp4-e2m1 nibble -> value (low nibble unpacked before high nibble):
#   1 -> 0.5   2 -> 1.0   6 -> 4.0   12 (0xC) -> -2.0
# ---------------------------------------------------------------------------

WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".scale"
FP8_WEIGHT = "layers.0.attn.wq_a.weight"
FP8_SCALE = "layers.0.attn.wq_a.scale"
MXFP4_WEIGHT = "layers.0.mlp.experts.0.w1.weight"
MXFP4_SCALE = "layers.0.mlp.experts.0.w1.scale"
NORM_WEIGHT = "layers.0.norm.weight"


def physical_metadata(entries):
    """Build a Paddle DCP Metadata from {key: (shape, dtype)} entries."""
    return Metadata(
        state_dict_metadata={
            key: [
                LocalTensorMetadata(
                    global_offset=(0,) * len(shape),
                    local_shape=tuple(shape),
                    global_shape=tuple(shape),
                    dtype=dtype,
                )
            ]
            for key, (shape, dtype) in entries.items()
        },
        storage_metadata={},
    )


def descriptor_dict(groups, logic_name_suffix=WEIGHT_SUFFIX):
    return {
        "schema_version": 1,
        "component_pairing": {
            "weight_suffix": WEIGHT_SUFFIX,
            "scale_suffix": SCALE_SUFFIX,
        },
        "logic_name_suffix": logic_name_suffix,
        "groups": groups,
    }


FP8_GROUP = {
    "name": "fp8",
    "targets": [r"re:.*\.attn\.wq_a\.weight$"],
    "quant_method": "fp8_block",
    "value_format": "e4m3",
    "scale_format": "ue8m0",
    "block_shape": [2, 2],
}
MXFP4_GROUP = {
    "name": "mxfp4",
    "targets": [r"re:.*\.experts\.[0-9]+\.w[123]\.weight$"],
    "quant_method": "mxfp4_group",
    "value_format": "e2m1",
    "scale_format": "ue8m0",
    "block_shape": [2],
}


class _CPUTestCase(unittest.TestCase):
    def setUp(self):
        self._original_device = paddle.get_device()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.set_device(self._original_device)


class TestFP8BlockDequantMath(_CPUTestCase):
    """dequant = decoded_e4m3(qweight) * block_broadcast(decoded_ue8m0(scale))."""

    def setUp(self):
        super().setUp()
        self.dequantizer = (
            get_checkpoint_dequantizer("fp8_block")
            .configure_formats("e4m3", "ue8m0")
            .configure_geometry((0, 1), (2, 2))
        )

    def test_dequant_matches_hand_derived_values(self):
        # Distinct codes so a byte swap, sign drop, or scale/axis error shows up.
        qweight = paddle.to_tensor(
            [
                [0x38, 0x30, 0x40, 0x3C],  # -> [ 1.0, 0.5, 2.0, 1.5]
                [0xB8, 0x00, 0x40, 0x38],  # -> [-1.0, 0.0, 2.0, 1.0]
            ],
            dtype="uint8",
        )
        # Two 2x2 blocks along axis 1: left block scale 1.0, right block 4.0.
        scale = paddle.to_tensor([[127, 129]], dtype="uint8")

        output = self.dequantizer.dequantize(
            {"qweight": qweight, "scale": scale}, paddle.float32
        )

        # decoded values * block-broadcast scale [[1,1,4,4],[1,1,4,4]]
        expected = np.array(
            [
                [1.0, 0.5, 8.0, 6.0],
                [-1.0, 0.0, 8.0, 4.0],
            ],
            dtype="float32",
        )
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_partial_trailing_block_scale_is_cropped(self):
        # 3x3 logical weight, 2x2 blocks -> 2x2 scale grid cropped back to 3x3.
        qweight = paddle.full([3, 3], 0x38, dtype="uint8")  # all 1.0
        scale = paddle.to_tensor([[127, 128], [129, 125]], dtype="uint8")
        # ue8m0 -> [[1.0, 2.0], [4.0, 0.25]] broadcast/cropped to 3x3:
        #   [[1,1,2],[1,1,2],[4,4,0.25]]
        output = self.dequantizer.dequantize(
            {"qweight": qweight, "scale": scale}, paddle.float32
        )
        expected = np.array(
            [
                [1.0, 1.0, 2.0],
                [1.0, 1.0, 2.0],
                [4.0, 4.0, 0.25],
            ],
            dtype="float32",
        )
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_requested_output_dtype_is_honored(self):
        output = self.dequantizer.dequantize(
            {
                "qweight": paddle.full([2, 2], 0x38, dtype="uint8"),
                "scale": paddle.full([1, 1], 127, dtype="uint8"),
            },
            paddle.bfloat16,
        )
        self.assertEqual(output.dtype, paddle.bfloat16)
        np.testing.assert_array_equal(
            output.astype("float32").numpy(),
            np.ones((2, 2), dtype="float32"),
        )


class TestMXFP4GroupDequantMath(_CPUTestCase):
    """Each stored byte unpacks low nibble then high nibble; every 2 share a scale."""

    def setUp(self):
        super().setUp()
        self.dequantizer = (
            get_checkpoint_dequantizer("mxfp4_group")
            .configure_formats("e2m1", "ue8m0")
            .configure_geometry((1,), (2,))
        )

    def test_dequant_matches_hand_derived_values(self):
        # byte 0x21 -> low 1 (0.5), high 2 (1.0); byte 0x6C -> low 12 (-2.0), high 6 (4.0)
        qweight = paddle.to_tensor([[0x21, 0x6C]], dtype="uint8")
        # ue8m0 [128, 126] -> [2.0, 0.5]; group size 2 -> [2.0, 2.0, 0.5, 0.5]
        scale = paddle.to_tensor([[128, 126]], dtype="uint8")

        output = self.dequantizer.dequantize(
            {"qweight": qweight, "scale": scale}, paddle.float32
        )

        # [0.5, 1.0, -2.0, 4.0] * [2.0, 2.0, 0.5, 0.5]
        expected = np.array([[1.0, 2.0, -1.0, 2.0]], dtype="float32")
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_logical_shape_doubles_the_packed_axis(self):
        # One physical byte per two logical values on the last axis.
        self.assertEqual(self.dequantizer.logical_shape((2, 3)), (2, 6))
        self.assertEqual(
            self.dequantizer.physical_qweight_shape((2, 6)), (2, 3)
        )


class TestQuanDescriptorKeyMapping(unittest.TestCase):
    """build_metadata() binds descriptor rules to physical tensor names/shapes."""

    def test_logical_names_and_component_keys_are_mapped(self):
        metadata = QuanDescriptor.from_dict(
            descriptor_dict([FP8_GROUP, MXFP4_GROUP])
        ).build_metadata(
            physical_metadata(
                {
                    FP8_WEIGHT: ((2, 4), "uint8"),
                    FP8_SCALE: ((1, 2), "uint8"),
                    MXFP4_WEIGHT: ((2, 2), "uint8"),
                    # physical (2, 2) -> logical (2, 4); block_shape [2] groups
                    # along the last (packed) axis, so the scale grid is
                    # (2, ceil(4/2)) = (2, 2). A (2, 1) scale would imply a
                    # block size of 4 and is rejected by _infer_block_axes.
                    MXFP4_SCALE: ((2, 2), "uint8"),
                    NORM_WEIGHT: ((4,), "bfloat16"),
                }
            ),
            output_dtype=paddle.float16,
        )

        # logic_name_suffix == weight_suffix, so the logical key equals the
        # physical weight key here; the unquantized norm weight stays out.
        self.assertEqual(set(metadata.relations), {FP8_WEIGHT, MXFP4_WEIGHT})
        self.assertEqual(
            set(metadata.logical_metadata), {FP8_WEIGHT, MXFP4_WEIGHT}
        )

        fp8 = metadata.relations[FP8_WEIGHT]
        # qweight/scale must map to their own physical keys, not be swapped.
        self.assertEqual(
            fp8.components, {"qweight": FP8_WEIGHT, "scale": FP8_SCALE}
        )
        self.assertEqual(fp8.group_name, "fp8")
        self.assertEqual(fp8.logical_shape, (2, 4))
        self.assertEqual(metadata.groups["fp8"].block_axes, (0, 1))

        mxfp4 = metadata.relations[MXFP4_WEIGHT]
        self.assertEqual(
            mxfp4.components, {"qweight": MXFP4_WEIGHT, "scale": MXFP4_SCALE}
        )
        # Packed axis doubles: physical (2, 2) -> logical (2, 4).
        self.assertEqual(mxfp4.logical_shape, (2, 4))

        logical = metadata.logical_metadata[FP8_WEIGHT]
        self.assertEqual(logical.dtype, "float16")
        self.assertEqual(tuple(logical.global_shape), (2, 4))

    def test_logic_name_suffix_actually_rewrites_the_key(self):
        # A distinct suffix proves the logical name is derived, not echoed.
        metadata = QuanDescriptor.from_dict(
            descriptor_dict([FP8_GROUP], logic_name_suffix=".dequant_weight")
        ).build_metadata(
            physical_metadata(
                {FP8_WEIGHT: ((2, 4), "uint8"), FP8_SCALE: ((1, 2), "uint8")}
            )
        )
        logical_key = "layers.0.attn.wq_a.dequant_weight"
        self.assertEqual(set(metadata.relations), {logical_key})
        spec = metadata.relations[logical_key]
        self.assertEqual(spec.logical_name, logical_key)
        # Physical component keys keep the checkpoint's real names.
        self.assertEqual(
            spec.components, {"qweight": FP8_WEIGHT, "scale": FP8_SCALE}
        )

    def test_weight_without_paired_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "paired scale .* is missing"):
            QuanDescriptor.from_dict(
                descriptor_dict([FP8_GROUP])
            ).build_metadata(physical_metadata({FP8_WEIGHT: ((2, 4), "uint8")}))

    def test_unmatched_quantized_pair_is_rejected(self):
        # Silently dropping a quantized pair would load raw bytes as weights.
        with self.assertRaisesRegex(ValueError, "did not match all quantized"):
            QuanDescriptor.from_dict(
                descriptor_dict([FP8_GROUP])
            ).build_metadata(
                physical_metadata(
                    {
                        FP8_WEIGHT: ((2, 4), "uint8"),
                        FP8_SCALE: ((1, 2), "uint8"),
                        "layers.0.attn.wq_b.weight": ((2, 4), "uint8"),
                        "layers.0.attn.wq_b.scale": ((1, 2), "uint8"),
                    }
                )
            )


class TestHFDequantLoadTransformReload(_CPUTestCase):
    """Full path: descriptor -> metadata -> transform -> apply, keyed correctly."""

    def setUp(self):
        super().setUp()
        self.transform = HFDequantLoadTransform(
            QuanDescriptor.from_dict(
                descriptor_dict([FP8_GROUP])
            ).build_metadata(
                physical_metadata(
                    {
                        FP8_WEIGHT: ((2, 4), "uint8"),
                        FP8_SCALE: ((1, 2), "uint8"),
                    }
                )
            )
        )

    def test_source_keys_list_qweight_before_scale(self):
        self.assertEqual(
            self.transform.source_keys(FP8_WEIGHT), [FP8_WEIGHT, FP8_SCALE]
        )

    def test_logical_metadata_is_a_defensive_copy(self):
        logical = self.transform.logical_metadata()
        self.assertEqual(set(logical), {FP8_WEIGHT})
        logical.clear()
        self.assertEqual(set(self.transform.logical_metadata()), {FP8_WEIGHT})

    def test_unmanaged_key_is_rejected(self):
        with self.assertRaisesRegex(
            KeyError, "is not managed by this load transform"
        ):
            self.transform.source_keys("layers.0.attn.wq_b.weight")

    def test_apply_dequantizes_full_tensor_from_named_sources(self):
        # No cached read_plan -> apply must reconstruct the whole logical tensor.
        output = self.transform.apply(
            FP8_WEIGHT,
            {
                FP8_WEIGHT: paddle.to_tensor(
                    [
                        [0x38, 0x30, 0x40, 0x3C],
                        [0xB8, 0x00, 0x40, 0x38],
                    ],
                    dtype="uint8",
                ),
                FP8_SCALE: paddle.to_tensor([[127, 129]], dtype="uint8"),
            },
            paddle.float32,
        )
        expected = np.array(
            [
                [1.0, 0.5, 8.0, 6.0],
                [-1.0, 0.0, 8.0, 4.0],
            ],
            dtype="float32",
        )
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_apply_rejects_a_shard_without_a_plan(self):
        # A partial shard with no cached plan is a bug: expect the full shape.
        with self.assertRaisesRegex(
            ValueError, r"expected \(2, 4\), got \(1, 4\)"
        ):
            self.transform.apply(
                FP8_WEIGHT,
                {
                    FP8_WEIGHT: paddle.full([1, 4], 0x38, dtype="uint8"),
                    FP8_SCALE: paddle.full([1, 2], 127, dtype="uint8"),
                },
                paddle.float32,
            )


class TestHFCheckpointDetection(unittest.TestCase):
    """config.json is the authority on whether a checkpoint is quantized."""

    def setUp(self):
        self.checkpoint_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.checkpoint_path, True)

    def _write_config(self, config):
        with open(
            os.path.join(self.checkpoint_path, "config.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file)

    def test_quantization_config_present_is_detected(self):
        self._write_config({"quantization_config": {"quant_method": "fp8"}})
        self.assertTrue(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_empty_or_missing_quantization_config_is_not_quantized(self):
        self._write_config({"quantization_config": {}})
        self.assertFalse(hf_checkpoint_is_quantized(self.checkpoint_path))
        self._write_config({"hidden_size": 8})
        self.assertFalse(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_missing_config_file_is_not_quantized(self):
        # No config.json written at all.
        self.assertFalse(hf_checkpoint_is_quantized(self.checkpoint_path))

    def test_no_descriptor_means_plain_load_path(self):
        # Without model-defined quant rules there is nothing to dequantize.
        self.assertIsNone(
            build_hf_dequant_load_transform(
                self.checkpoint_path, quan_config=None
            )
        )


if __name__ == "__main__":
    unittest.main()
