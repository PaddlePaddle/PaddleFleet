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

"""Self-contained coverage for
``paddlefleet.transformers.model_utils.save_full_param``.

Drives the saver with a synthetic in-memory weight iterator -- no parent-repo
configs, no distributed launch -- so it runs in the standalone coverage CI that
the erniebot-config-driven roundtrip test skips. ``max_shard_size`` and
``sync_copy_threshold_bytes`` are injected to steer the code under test:

* a small ``max_shard_size`` over many tensors makes the pinned pool fill past
  its capacity (async D2H eviction) and shards flush mid-iteration;
* ``sync_copy_threshold_bytes`` below a tensor, a non-contiguous tensor, or a
  CPU-resident tensor forces the synchronous ``param.cpu()`` fallback.

The invariant checked is ``save_full_param``'s contract: every tensor it is
handed lands on disk exactly once, byte-for-byte, and the returned total equals
the summed tensor bytes. The async pinned-offload branch only exists on GPU, so
the tests that must exercise it are gated on a usable CUDA device; the sync
fallbacks run everywhere.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import paddle
from safetensors.paddle import load_file

from paddlefleet.transformers.model_utils import save_full_param


def _init_gpu():
    """Make GPU the default device so the saver takes the async path.

    ``save_full_param`` keys its async loader off ``paddle.get_device()``, so a
    GPU tensor alone is not enough -- the default device must be gpu too.
    Returns whether a CUDA device is actually usable.
    """
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        paddle.set_device("gpu")
        paddle.zeros([1])  # force context init; raises if no usable GPU
        return True
    except Exception:
        return False


_HAS_GPU = _init_gpu()
_GPU_ONLY = unittest.skipUnless(_HAS_GPU, "requires a usable CUDA device")


def _tensor(shape, dtype="float32", seed=0, place=None):
    """A deterministic tensor on the current device (or ``place``)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(size=shape).astype("float32")
    t = paddle.to_tensor(arr, place=place)
    if dtype != "float32":
        t = t.astype(dtype)
    return t


def _nbytes(t):
    # Mirror save_full_param's own accounting: param.size * param.itemsize.
    return int(t.size) * int(t.itemsize)


def _save(originals, tmp, **save_kwargs):
    """Run the saver over ``originals``; return (total, sorted shard paths)."""
    total = save_full_param(
        itr=iter(list(originals.items())),
        save_dir=tmp,
        rank=0,
        moe_sharding_world_size=1,
        num_saver_ranks=1,
        **save_kwargs,
    )
    return total, sorted(Path(tmp).glob("shard_*.safetensors"))


def _load(shards):
    """Merge every written shard into one ``{key: paddle tensor}``."""
    out = {}
    for shard in shards:
        out.update(load_file(str(shard)))
    return out


class TestSaveFullParam(unittest.TestCase):
    """save_full_param writes back every tensor once, byte-for-byte."""

    def _check_lossless(self, originals, total, loaded):
        self.assertEqual(
            set(loaded),
            set(originals),
            "on-disk keys do not match what was handed to the saver",
        )
        self.assertEqual(
            total,
            sum(_nbytes(t) for t in originals.values()),
            "reported total bytes do not match the input",
        )
        for key, ref in originals.items():
            np.testing.assert_array_equal(
                loaded[key].astype("float32").numpy(),
                ref.astype("float32").numpy(),
                err_msg=f"{key} changed on save",
            )

    def _roundtrip(self, originals, **save_kwargs):
        with TemporaryDirectory() as tmp:
            total, shards = _save(originals, tmp, **save_kwargs)
            loaded = _load(shards)
            num_shards = len(shards)
        self._check_lossless(originals, total, loaded)
        return num_shards

    @_GPU_ONLY
    def test_async_offload_pool_eviction_and_shard_flush(self):
        # 20 tensors under a shard cap that holds several of them: within a
        # shard the pinned pool climbs past its capacity (async eviction), and
        # the tensors span multiple shards (mid-iteration flush).
        dtypes = ["float32", "bfloat16", "float16"]
        originals = {
            f"w{i:02d}": _tensor([256, 128], dtypes[i % 3], seed=i)
            for i in range(20)
        }
        num_shards = self._roundtrip(originals, max_shard_size="1MB")
        self.assertGreaterEqual(
            num_shards, 2, "small max_shard_size should split into shards"
        )

    @_GPU_ONLY
    def test_async_preserves_bf16_values(self):
        # bf16 rides a pinned buffer, then _copy_to(CPUPlace) into pageable
        # memory; that hop must keep the raw bits (numpy has no bf16 to
        # round-trip through), which is the whole point of not using .numpy().
        originals = {
            "bf16": _tensor([64, 64], "bfloat16", seed=1),
            "fp16": _tensor([64, 64], "float16", seed=2),
            "fp32": _tensor([64, 64], "float32", seed=3),
        }
        self._roundtrip(originals)

    def test_sync_fallback_over_threshold(self):
        # A tensor above sync_copy_threshold_bytes skips the pinned pool (so one
        # huge tensor never needs an equally huge pinned buffer); a small one
        # below the threshold still takes the fast path.
        originals = {
            "big": _tensor([256, 128], "float32", seed=4),  # 128 KiB
            "small": _tensor([4, 4], "float32", seed=5),  # 64 B
        }
        self._roundtrip(originals, sync_copy_threshold_bytes=1024)

    def test_sync_fallback_non_contiguous(self):
        # A non-contiguous tensor must take param.cpu(): async_offload copies
        # raw storage and would otherwise scramble the logical layout.
        base = _tensor([256, 128], "float32", seed=6)
        nc = paddle.transpose(base, perm=[1, 0])
        if nc.is_contiguous():
            self.skipTest("could not build a non-contiguous tensor")
        self._roundtrip({"nc": nc, "c": _tensor([8, 8], seed=7)})

    @_GPU_ONLY
    def test_sync_fallback_cpu_resident(self):
        # On a GPU box a CPU-resident param is not is_gpu_place(), so it must
        # fall back to param.cpu() rather than the GPU-only async path.
        cpu_t = _tensor([32, 32], "float32", seed=8, place=paddle.CPUPlace())
        self.assertFalse(cpu_t.place.is_gpu_place())
        self._roundtrip({"cpu": cpu_t, "gpu": _tensor([32, 32], seed=9)})


if __name__ == "__main__":
    unittest.main()
