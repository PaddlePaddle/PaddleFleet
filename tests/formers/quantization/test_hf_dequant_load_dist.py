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
"""Behavior tests for the HF fp8_block dequantization load path.

Two independent slices of behavior are covered:

* CPU (无卡): the real ``FP8BlockCheckpointDequantizer`` dequant math and the
  real ``HFDequantLoadTransform.read_plan`` shard/rank -> qweight/scale mapping.
  Both run on CPU with hand-derived expected values; no collective, no GPU, and
  no faked ``world_size`` are involved, so these prove local math and mapping
  only -- not cross-rank communication.

* Multi-card (多卡-only): a real two-rank ``paddle.distributed.launch`` of the
  sibling worker ``hf_dequant_load_dist_logic.py``, which loads a sharded
  fp8_block checkpoint and checks each rank owns its own dequantized rows. This
  is skipped without >=2 GPUs; it is the only genuine cross-rank evidence and is
  never emulated on a single process.
"""

import inspect
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
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
)

# The sibling worker script (real cross-rank load) lives next to this file.
from tests.formers.quantization import hf_dequant_load_dist_logic as dist_logic

WORKER_SCRIPT = dist_logic.__file__

# -----------------------------------------------------------------------------
# Fixtures.  Defined inline (not imported) so the expected values below stay
# independent of the worker helper's own reference implementation.
# -----------------------------------------------------------------------------

WEIGHT_KEY = "layers.0.attn.wq_a.weight"
SCALE_KEY = "layers.0.attn.wq_a.scale"
LOGICAL_SHAPE = (4, 4)
BLOCK_SHAPE = (2, 2)
BLOCK_AXES = (0, 1)

# e4m3 codes, one per logical row: 0x38->2**0, 0x40->2**1, 0x48->2**2, 0x50->2**3
# (exponent bias 7, zero mantissa, positive sign).  A wrong row mapping would
# therefore change the magnitude by a factor of two and be visible.
E4M3_ROW_CODES = (0x38, 0x40, 0x48, 0x50)
# ue8m0 codes are biased exponents: 127->2**0, 128->2**1.  The 2x2 grid is
# asymmetric so reading the wrong scale block cannot silently cancel out.
SCALE_GRID = ((127, 128), (128, 127))

DESCRIPTOR = {
    "schema_version": 1,
    "component_pairing": {
        "weight_suffix": ".weight",
        "scale_suffix": ".scale",
    },
    "logic_name_suffix": ".weight",
    "groups": [
        {
            "name": "fp8",
            "targets": [r"re:.*\.attn\.wq_a\.weight$"],
            "quant_method": "fp8_block",
            "value_format": "e4m3",
            "scale_format": "ue8m0",
            "block_shape": list(BLOCK_SHAPE),
        }
    ],
}

# Hand-derived reference for the full 4x4 logical weight.
#   value(row) = 2**row (from the e4m3 code above)
#   scale(row, col) = 2**(SCALE_GRID[row // 2][col // 2] - 127)
#     decoded scale grid = [[1, 2], [2, 1]] then block-expanded (2x2) to 4x4:
#       rows 0-1 -> [1, 1, 2, 2];  rows 2-3 -> [2, 2, 1, 1]
#   logical[row, col] = value(row) * scale(row, col)
EXPECTED_LOGICAL = np.array(
    [
        [1.0, 1.0, 2.0, 2.0],  # 2**0 * [1, 1, 2, 2]
        [2.0, 2.0, 4.0, 4.0],  # 2**1 * [1, 1, 2, 2]
        [8.0, 8.0, 4.0, 4.0],  # 2**2 * [2, 2, 1, 1]
        [16.0, 16.0, 8.0, 8.0],  # 2**3 * [2, 2, 1, 1]
    ],
    dtype="float32",
)


def _fp8_block_dequantizer():
    """The real registered dequantizer, configured for this descriptor group."""
    return (
        get_checkpoint_dequantizer("fp8_block")
        .configure_formats("e4m3", "ue8m0")
        .configure_geometry(BLOCK_AXES, BLOCK_SHAPE)
    )


def _uint8(array_like):
    return paddle.to_tensor(np.array(array_like, dtype="uint8"))


def _physical_metadata():
    """Physical safetensors metadata: raw uint8 qweight (4x4) + scale grid (2x2)."""
    weight_md = LocalTensorMetadata(
        global_offset=(0, 0),
        local_shape=LOGICAL_SHAPE,
        global_shape=LOGICAL_SHAPE,
        dtype="uint8",
    )
    scale_md = LocalTensorMetadata(
        global_offset=(0, 0),
        local_shape=(2, 2),
        global_shape=(2, 2),
        dtype="uint8",
    )
    return Metadata({WEIGHT_KEY: [weight_md], SCALE_KEY: [scale_md]}, {}, None)


def _build_transform():
    """Build the real transform from the real descriptor + physical metadata."""
    descriptor = QuanDescriptor.from_dict(DESCRIPTOR)
    quan_metadata = descriptor.build_metadata(_physical_metadata())
    return HFDequantLoadTransform(quan_metadata)


def _shard(global_offset, local_shape):
    return LocalTensorMetadata(
        global_offset=global_offset,
        local_shape=local_shape,
        global_shape=LOGICAL_SHAPE,
        dtype="bfloat16",
    )


class TestFP8BlockDequantMathCPU(unittest.TestCase):
    """Real dequant math on CPU, compared against hand-derived values."""

    @classmethod
    def setUpClass(cls):
        # 无卡: force CPU so this proves the CPU math path, not a GPU kernel.
        paddle.set_device("cpu")

    def test_full_tensor_dequant_matches_hand_derived(self):
        deq = _fp8_block_dequantizer()
        qweight = _uint8([[code] * 4 for code in E4M3_ROW_CODES])  # (4, 4)
        scale = _uint8(SCALE_GRID)  # (2, 2) block grid
        out = deq.dequantize(
            {"qweight": qweight, "scale": scale}, paddle.float32
        )
        # Powers of two are exact in float32, so require exact equality.
        np.testing.assert_array_equal(out.numpy(), EXPECTED_LOGICAL)

    def test_row_block_shard_dequant_owns_its_rows(self):
        # Rows [2, 4): e4m3 codes 0x48 (2**2) and 0x50 (2**3); the rank reads
        # only scale-grid row 1 = (128, 127) -> decoded [2, 1], expanded to
        # [[2, 2, 1, 1], [2, 2, 1, 1]].  Result must equal the full reference's
        # rows 2-3, proving per-shard scale selection and block expansion.
        deq = _fp8_block_dequantizer()
        qweight = _uint8([[0x48] * 4, [0x50] * 4])  # (2, 4)
        scale = _uint8([[128, 127]])  # (1, 2): only this shard's scale block
        out = deq.dequantize(
            {"qweight": qweight, "scale": scale}, paddle.float32
        )
        np.testing.assert_array_equal(out.numpy(), EXPECTED_LOGICAL[2:4])

    def test_first_row_block_shard_differs_from_second(self):
        # Rows [0, 2): scale-grid row 0 = (127, 128) -> [1, 2] -> [[1,1,2,2]...].
        # A confirmation that a different shard yields a different (correct)
        # block, so the mapping is not degenerate to a single scale block.
        deq = _fp8_block_dequantizer()
        qweight = _uint8([[0x38] * 4, [0x40] * 4])  # (2, 4)
        scale = _uint8([[127, 128]])  # (1, 2)
        out = deq.dequantize(
            {"qweight": qweight, "scale": scale}, paddle.float32
        )
        np.testing.assert_array_equal(out.numpy(), EXPECTED_LOGICAL[0:2])

    def test_dequant_requires_both_components(self):
        deq = _fp8_block_dequantizer()
        qweight = _uint8([[code] * 4 for code in E4M3_ROW_CODES])
        with self.assertRaises(KeyError):
            deq.dequantize({"qweight": qweight}, paddle.float32)

    def test_configure_rejects_unsupported_formats(self):
        with self.assertRaises(ValueError):
            get_checkpoint_dequantizer("fp8_block").configure_formats(
                "e4m3", "definitely_not_a_scale_format"
            )
        with self.assertRaises(ValueError):
            get_checkpoint_dequantizer("no_such_quant_method")


class TestHFReadPlanShardMappingCPU(unittest.TestCase):
    """Real read_plan shard/rank -> qweight/scale slice mapping, on CPU.

    This exercises the load-distribution logic (which physical slices a given
    logical shard owns) without any process group.  It proves the mapping
    arithmetic only; genuine cross-rank load is covered by the multi-card test.
    """

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def setUp(self):
        # Fresh transform per test: build_metadata configures group geometry.
        self.transform = _build_transform()

    def test_second_row_shard_reads_its_own_qweight_and_scale(self):
        # A two-way Shard(0) split gives rank 1 rows [2, 4).  That is block
        # aligned, so the plan must read local slices, not the whole tensor.
        plan = self.transform.read_plan(WEIGHT_KEY, _shard((2, 0), (2, 4)))
        self.assertEqual(plan.mode, "local")
        self.assertEqual(plan.logical_local_shape, (2, 4))
        self.assertEqual(plan.logical_global_offset, (2, 0))

        qweight_slice = plan.source_slices[WEIGHT_KEY]
        self.assertEqual(tuple(qweight_slice.global_offset), (2, 0))
        self.assertEqual(tuple(qweight_slice.local_shape), (2, 4))
        # Rows [2, 4) map onto scale-grid row 1 only: offset 2//2=1, one block.
        scale_slice = plan.source_slices[SCALE_KEY]
        self.assertEqual(tuple(scale_slice.global_offset), (1, 0))
        self.assertEqual(tuple(scale_slice.local_shape), (1, 2))

    def test_first_row_shard_maps_to_first_scale_block(self):
        plan = self.transform.read_plan(WEIGHT_KEY, _shard((0, 0), (2, 4)))
        self.assertEqual(plan.mode, "local")
        self.assertEqual(plan.logical_global_offset, (0, 0))

        qweight_slice = plan.source_slices[WEIGHT_KEY]
        self.assertEqual(tuple(qweight_slice.global_offset), (0, 0))
        self.assertEqual(tuple(qweight_slice.local_shape), (2, 4))
        scale_slice = plan.source_slices[SCALE_KEY]
        # Distinct from rank 1's (1, 0): the scale origin tracks the shard.
        self.assertEqual(tuple(scale_slice.global_offset), (0, 0))
        self.assertEqual(tuple(scale_slice.local_shape), (1, 2))

    def test_whole_tensor_target_plans_a_global_read(self):
        # A single-card whole-tensor target (local == global) reads in full.
        plan = self.transform.read_plan(WEIGHT_KEY, _shard((0, 0), (4, 4)))
        self.assertEqual(plan.mode, "global")
        self.assertEqual(plan.logical_local_shape, LOGICAL_SHAPE)
        qweight_slice = plan.source_slices[WEIGHT_KEY]
        self.assertEqual(tuple(qweight_slice.global_offset), (0, 0))
        self.assertEqual(tuple(qweight_slice.local_shape), LOGICAL_SHAPE)
        scale_slice = plan.source_slices[SCALE_KEY]
        self.assertEqual(tuple(scale_slice.global_offset), (0, 0))
        self.assertEqual(tuple(scale_slice.local_shape), (2, 2))

    def test_block_unaligned_shard_falls_back_to_global(self):
        # Offset 1 is not block aligned (block size 2), so a local slice would
        # need a fractional scale origin; the plan must read globally instead.
        plan = self.transform.read_plan(WEIGHT_KEY, _shard((1, 0), (1, 4)))
        self.assertEqual(plan.mode, "global")
        # Global mode exposes the whole logical tensor and reads full sources.
        self.assertEqual(plan.logical_local_shape, LOGICAL_SHAPE)
        qweight_slice = plan.source_slices[WEIGHT_KEY]
        self.assertEqual(tuple(qweight_slice.local_shape), LOGICAL_SHAPE)
        scale_slice = plan.source_slices[SCALE_KEY]
        self.assertEqual(tuple(scale_slice.local_shape), (2, 2))

    def test_logical_shard_alignment_predicate(self):
        deq = _fp8_block_dequantizer()
        # Interior shard on a block boundary: aligned.
        self.assertTrue(deq.logical_shard_is_aligned((4, 4), (2, 4), (2, 0)))
        # Boundary shard whose partial block ends at the tensor edge: aligned.
        self.assertTrue(deq.logical_shard_is_aligned((5, 4), (1, 4), (4, 0)))
        # Start offset 1 is off a block boundary: not aligned.
        self.assertFalse(deq.logical_shard_is_aligned((4, 4), (1, 4), (1, 0)))


def _load_transform_is_supported():
    """Whether the installed Paddle shards a ``load_transform`` target by shard.

    Older wheels described a distributed target by its global shape, so
    ``read_plan()`` never saw a local shard and the local path could not run.
    ``_apply_load_transform`` gained a ``read_plans`` argument in the same change
    that fixed this, so it marks a Paddle new enough to run the worker.  Both
    checks are precise signature probes, not broad capability guessing.
    """
    if (
        "load_transform"
        not in inspect.signature(dist.load_state_dict).parameters
    ):
        return False
    from paddle.distributed.flex_checkpoint.dcp import load_state_dict as dcp

    return (
        "read_plans" in inspect.signature(dcp._apply_load_transform).parameters
    )


@unittest.skipUnless(
    paddle.device.cuda.device_count() > 1, "test requires multiple GPUs"
)
@unittest.skipUnless(
    _load_transform_is_supported(),
    "paddle is too old to shard a load_transform target",
)
class TestHFDequantLoadDistMultiCard(unittest.TestCase):
    """Genuine two-rank load of a sharded fp8_block HF checkpoint.

    多卡-only: the "local" read mode is unreachable with a single card, so this
    launches two real ranks via ``paddle.distributed.launch``.  Each rank owns
    two of the four logical rows (block aligned) and the worker asserts that its
    dequantized local rows equal the independent reference for exactly those
    rows.  There is no faked ``world_size`` and no mocked collective; without
    >=2 GPUs the case is skipped, never emulated.
    """

    def test_block_aligned_shard_loads_its_own_rows(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            env = dict(os.environ, ckpt_path=ckpt_dir)
            command = [
                sys.executable,
                "-m",
                "paddle.distributed.launch",
                "--devices",
                "0,1",
                "--log_dir",
                os.path.join(ckpt_dir, "log"),
                WORKER_SCRIPT,
            ]
            process = subprocess.run(
                command, env=env, capture_output=True, text=True
            )
            # Propagate a non-zero worker exit (a failed rank must fail here).
            self.assertEqual(
                process.returncode,
                0,
                f"two-card load failed:\n{process.stdout}\n{process.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
