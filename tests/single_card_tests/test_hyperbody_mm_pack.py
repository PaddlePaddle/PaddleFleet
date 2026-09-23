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

"""Unit coverage for the HyperBody multimodal packing helpers (``mm_pack``).

Packed multimodal inputs are per-segment lists of optional (variable-shape)
tensors; the fleet pipeline micro-batch loader only carries Tensor/None. These
helpers pack such lists into pure Tensors and losslessly restore them. This
module asserts the round-trips (values / shapes / seg_index), the empty-list
edge case, the mixed-None layout, and that ``unpack_hyperbody_mm`` cleans up the
packing keys (and is a no-op when they are absent). Pure CPU tensor ops.
"""

import sys

import numpy as np
import paddle

from paddlefleet.transformers.hyperbody.mm_pack import (
    HB_AUDIO_PREFIX,
    HB_IMAGE_PREFIX,
    pack_hyperbody_mm,
    pack_optional_tensor_list,
    unpack_hyperbody_mm,
    unpack_optional_tensor_list,
)


def _same(a, b):
    return (
        a is None
        and b is None
        or (
            a is not None
            and b is not None
            and list(a.shape) == list(b.shape)
            and np.array_equal(a.numpy(), b.numpy())
        )
    )


def test_pack_unpack_roundtrip_with_none():
    """A mixed list[Tensor|None] round-trips exactly (shapes + values + gaps).

    Present tensors in one list share a modality rank (self-describing shapes),
    so we use same-rank (but different-size) tensors here.
    """
    items = [
        paddle.to_tensor(np.arange(6, dtype="float32").reshape(2, 3)),
        None,
        paddle.to_tensor(np.arange(20, dtype="float32").reshape(4, 5)),
    ]
    packed = pack_optional_tensor_list(items, "px")
    assert set(packed) == {"px_values", "px_shapes", "px_seg_index"}
    out = unpack_optional_tensor_list(packed, "px")
    assert len(out) == 3
    assert _same(out[0], items[0])
    assert out[1] is None
    assert _same(out[2], items[2])


def test_pack_unpack_empty_and_all_none():
    """Empty list and all-None list both round-trip to the same-length list."""
    assert (
        unpack_optional_tensor_list(pack_optional_tensor_list([], "e"), "e")
        == []
    )
    allnone = [None, None]
    out = unpack_optional_tensor_list(
        pack_optional_tensor_list(allnone, "n"), "n"
    )
    assert out == [None, None]


def test_pack_hyperbody_mm_roundtrip_and_key_cleanup():
    """pack_hyperbody_mm -> unpack_hyperbody_mm restores image/audio and cleans keys."""
    image = [paddle.to_tensor(np.ones((3, 2, 2), dtype="float32")), None]
    audio = [None, paddle.to_tensor(np.arange(5, dtype="float32"))]
    batch = {"input_ids": paddle.to_tensor(np.zeros((1, 4), dtype="int64"))}
    batch.update(pack_hyperbody_mm(image, audio))
    # Only Tensors in the batch now (pipeline-safe): no python lists.
    assert not any(isinstance(v, list) for v in batch.values())
    assert f"{HB_IMAGE_PREFIX}_seg_index" in batch

    unpack_hyperbody_mm(batch)
    assert _same(batch["image"][0], image[0]) and batch["image"][1] is None
    assert batch["audio"][0] is None and _same(batch["audio"][1], audio[1])
    # packing keys removed after restore
    for pfx in (HB_IMAGE_PREFIX, HB_AUDIO_PREFIX):
        for suf in ("_values", "_shapes", "_seg_index"):
            assert pfx + suf not in batch


def test_unpack_hyperbody_mm_noop_without_keys():
    """unpack is a no-op (returns as-is) when packing keys are absent."""
    batch = {"input_ids": paddle.to_tensor(np.zeros((1, 3), dtype="int64"))}
    out = unpack_hyperbody_mm(batch)
    assert out is batch and "image" not in batch


if __name__ == "__main__":
    try:
        test_pack_unpack_roundtrip_with_none()
        test_pack_unpack_empty_and_all_none()
        test_pack_hyperbody_mm_roundtrip_and_key_cleanup()
        test_unpack_hyperbody_mm_noop_without_keys()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
