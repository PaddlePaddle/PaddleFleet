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

"""Coverage for ``paddlefleet.transformers.model_utils.save_full_param``.

Mirrors the save half of ``test_hybrid_mla_hf_roundtrip.py`` but drives the
**paddlefleet** saver (the copy this repo owns and this PR changed) instead of
the paddleformers one, so the async pinned-offload path runs on the real
``sharded_state_dict`` + ``full_param`` output rather than a synthetic iterator.
Reuses that module's erniebot-config-driven fixtures; like it, skips without the
parent-repo configs and requires a CUDA device.

The invariant asserted is ``save_full_param``'s own contract -- every tensor it
is handed lands on disk exactly once, unchanged -- not the AOA name mapping
(``test_hybrid_mla_hf_roundtrip.py`` owns that), so it holds for both configs.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .hybrid_mla_utils import (
    _CONFIG_DIR,
    _DSA_CFG as _DSA,
    _MHA_CFG as _MHA,
    _PARENT_REPO_AVAILABLE,
)
from .test_hybrid_mla_hf_roundtrip import (
    PREFIX,
    _aoa,
    _attn,
    _filter,
    _requires_cuda,
    _single_process_world,
)


def setUpModule():
    if not _PARENT_REPO_AVAILABLE:
        raise unittest.SkipTest(
            f"requires the erniebot parent repo configs at {_CONFIG_DIR}"
        )


def _save_hf_fleet(attn, inv_stmts, path):
    """The ``HFFormatFullParamSaver`` chain through the *paddlefleet* saver.

    ``sharded_state_dict`` -> ``full_param(aoa_config=inverse)`` ->
    ``paddlefleet.transformers.model_utils.save_full_param`` ->
    ``replace_name_and_gen_index``. Returns ``{key: fp32 ndarray}`` for every
    tensor handed to the saver, so the caller can check it survived the trip.
    """
    from paddle.distributed.flex_checkpoint.dcp.full_param import full_param

    from paddlefleet.transformers.model_utils import (
        replace_name_and_gen_index,
        save_full_param,
    )

    path.mkdir(parents=True, exist_ok=True)
    emitted = {}

    def tap():
        sd = attn.sharded_state_dict(PREFIX)
        for key, tensor in full_param(
            sd, aoa_config={"aoa_statements": inv_stmts}
        ):
            emitted[key] = tensor.astype("float32").numpy()
            yield key, tensor

    with _single_process_world():
        total = save_full_param(
            itr=tap(),
            save_dir=str(path),
            rank=0,
            moe_sharding_world_size=1,
            num_saver_ranks=1,
        )
        replace_name_and_gen_index(str(path), total)
    return emitted, total


def _load_disk(path):
    """``{key: paddle CPU tensor}`` across the written shards (bf16-safe)."""
    from safetensors.paddle import load_file

    out = {}
    for shard in sorted(Path(path).glob("model-*.safetensors")):
        out.update(load_file(str(shard)))
    return out


@_requires_cuda
class TestFleetSaveFullParam(unittest.TestCase):
    """The paddlefleet saver writes back every tensor once, bit-for-bit."""

    def _save_and_check(self, cfg_name):
        attn = _attn(cfg_name, seed=11)
        module_keys = {PREFIX + k for k in attn.state_dict()}
        _, inverse = _aoa(cfg_name, indexer_init_from_scratch=True)
        inv = _filter(inverse, module_keys, fleet_on_left=True)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hf"
            emitted, total = _save_hf_fleet(attn, inv, path)
            disk = _load_disk(path)

        self.assertTrue(emitted, f"{cfg_name}: nothing was emitted to save")
        self.assertGreater(total, 0)
        # save_full_param's contract: write exactly what it was handed, once.
        self.assertEqual(
            set(disk),
            set(emitted),
            f"{cfg_name}: on-disk keys do not match what was saved",
        )
        for key, ref in emitted.items():
            np.testing.assert_array_equal(
                disk[key].astype("float32").numpy(),
                ref,
                err_msg=f"{cfg_name}: {key} changed on save",
            )

    def test_mha_save_is_complete_and_lossless(self):
        self._save_and_check(_MHA)

    def test_mqa_dsa_save_is_complete_and_lossless(self):
        self._save_and_check(_DSA)


if __name__ == "__main__":
    unittest.main()
