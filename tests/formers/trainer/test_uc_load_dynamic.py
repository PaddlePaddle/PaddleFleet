# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""CPU-observable behavior tests for
``trainer/unified_checkpoint/load_dynamic.py``.

Scope note (无卡 / CPU-only). This file targets ``distributed_send_recv``,
the routine that actually moves checkpoint tensors from safetensors shards
into the model/optimizer state dict. With a single logical rank
(``PADDLE_LOCAL_SIZE=1`` and ``dist.get_rank() == 0``) the send/recv peer
branches are never taken, so the *local* read + tensor-parallel dispatch +
placement logic runs on CPU without a process group:

* file-to-key ownership -- each key is read from the shard that
  ``file_keyname_mappings``/``file_machine_mappings`` route it to; same-shape
  but distinct-content tensors catch a mapping swap that a shape-only check
  would miss;
* content fidelity -- the value landed in ``state_dict[key]`` equals the exact
  bytes written to the shard (independent numpy oracle via
  ``safetensors.numpy.save_file``, never the production writer);
* tensor-parallel split selection -- when ``recv_table[key]`` carries a split
  index, the placed slice is the one at that index (not index 0 / not the
  whole tensor).

Deliberately NOT claimed here: cross-rank ``dist.stream.send/recv`` numerics,
``create_dispatch_table`` / ``create_optimizer_dispatch_table`` (both call
``fleet.get_hybrid_communicate_group`` + ``all_gather_object``) and the
``load_unified_*`` orchestrators. Those require a real hybrid-parallel process
group and belong to the multi-card suite; a single documented ``skipTest``
records that they are unverified on CPU rather than faking a pass.

Expected values are hand-derived; the tensor-parallel action passed in is a
genuine, input-dependent collaborator (it splits the real slice it receives),
so a wrong file, wrong key, or wrong split index is rejected. All
Paddle-dependent imports are guarded and only ``ImportError`` is swallowed, so
a missing dependency skips cleanly without masking a real regression.
"""

import os
import tempfile
import unittest

import numpy as np

try:
    from safetensors.numpy import save_file as st_save_file

    _SAFETENSORS_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _SAFETENSORS_ERROR = exc

try:
    import paddle  # noqa: F401

    from paddlefleet.trainer.unified_checkpoint.load_dynamic import (
        distributed_send_recv,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = exc


def _write_shard(directory, filename, arrays):
    """Write ``arrays`` (name -> np.ndarray) to a real safetensors shard.

    Uses the upstream ``safetensors`` writer, which is independent of the
    PaddleFleet save path, so it is a valid oracle for the loader under test.
    """
    path = os.path.join(directory, filename)
    st_save_file(arrays, path)
    return path


class _RequiresRealImports(unittest.TestCase):
    """Skip when paddle / paddlefleet / safetensors are unavailable."""

    def setUp(self):
        if _SAFETENSORS_ERROR is not None:
            self.skipTest(f"safetensors unavailable: {_SAFETENSORS_ERROR!r}")
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet load_dynamic import failed "
                f"(needs paddle + deps): {_IMPORT_ERROR!r}"
            )
        # distributed_send_recv reads PADDLE_LOCAL_SIZE / PADDLE_RANK_IN_NODE
        # from the environment. Pin a single-node, single-local-rank layout so
        # rank 0 owns and reads every shard locally, then restore afterwards
        # (avoid leaking global env state into sibling tests).
        for name, value in (
            ("PADDLE_LOCAL_SIZE", "1"),
            ("PADDLE_RANK_IN_NODE", "0"),
        ):
            original = os.environ.get(name)
            self.addCleanup(self._restore_env, name, original)
            os.environ[name] = value

    @staticmethod
    def _restore_env(name, original):
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


class TestDistributedSendRecvSingleRank(_RequiresRealImports):
    """Local (rank-0-owns-all) path of ``distributed_send_recv``."""

    def test_reads_exact_content_and_resists_key_swap(self):
        """Each key is filled with its own shard bytes, not another key's.

        Two keys share the same shape but hold different values; a mapping that
        swapped them would still pass a shape-only assertion, so the values are
        compared element-wise.
        """
        w0 = np.arange(6, dtype=np.float32).reshape(2, 3)
        w1 = np.arange(6, dtype=np.float32).reshape(2, 3) + 100.0
        with tempfile.TemporaryDirectory() as tmp:
            _write_shard(tmp, "model-00001.safetensors", {"a.w": w0, "b.w": w1})

            file_keyname_mappings = {
                "model-00001.safetensors": ["a.w", "b.w"],
            }
            file_machine_mappings = {"model-00001.safetensors": [0]}
            send_table = {"a.w": 0, "b.w": 0}
            recv_table = {"a.w": [(0, -1)], "b.w": [(0, -1)]}

            state_dict = {}
            out = distributed_send_recv(
                state_dict,
                {},  # no tensor-parallel actions
                send_table,
                recv_table,
                tmp,
                file_keyname_mappings,
                file_machine_mappings,
            )

            self.assertIs(out, state_dict)
            self.assertEqual(set(out.keys()), {"a.w", "b.w"})
            np.testing.assert_array_equal(out["a.w"].numpy(), w0)
            np.testing.assert_array_equal(out["b.w"].numpy(), w1)

    def test_key_routed_to_correct_shard(self):
        """A key must be read from the shard that owns it, across two files.

        ``x`` lives in shard 1 and ``y`` in shard 2 with identical shape but
        distinct content; reading from the wrong shard would swap the values.
        """
        x = np.full((3,), 7.0, dtype=np.float32)
        y = np.full((3,), -7.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            _write_shard(tmp, "shard-1.safetensors", {"x": x})
            _write_shard(tmp, "shard-2.safetensors", {"y": y})

            file_keyname_mappings = {
                "shard-1.safetensors": ["x"],
                "shard-2.safetensors": ["y"],
            }
            file_machine_mappings = {
                "shard-1.safetensors": [0],
                "shard-2.safetensors": [0],
            }
            send_table = {"x": 0, "y": 0}
            recv_table = {"x": [(0, -1)], "y": [(0, -1)]}

            out = distributed_send_recv(
                {},
                {},
                send_table,
                recv_table,
                tmp,
                file_keyname_mappings,
                file_machine_mappings,
            )

            np.testing.assert_array_equal(out["x"].numpy(), x)
            np.testing.assert_array_equal(out["y"].numpy(), y)

    def test_tp_action_split_index_is_selected(self):
        """With a split index in ``recv_table``, the placed slice is that part.

        The tensor-parallel action is a real collaborator: it receives the
        safetensors slice, materializes it, and returns row-wise halves. The
        loader must (a) feed the correct slice content to the action and
        (b) store the half at the receiver's split index -- here index 1.
        """
        full = np.arange(8, dtype=np.float32).reshape(4, 2)
        top_half = full[0:2]
        bottom_half = full[2:4]
        seen = {}

        def split_rows(safe_slice):
            materialized = np.asarray(safe_slice[:], dtype=np.float32)
            seen["content"] = materialized.copy()
            parts = np.array_split(materialized, 2, axis=0)
            return [np.ascontiguousarray(p) for p in parts]

        with tempfile.TemporaryDirectory() as tmp:
            _write_shard(tmp, "tp.safetensors", {"proj.w": full})

            file_keyname_mappings = {"tp.safetensors": ["proj.w"]}
            file_machine_mappings = {"tp.safetensors": [0]}
            send_table = {"proj.w": 0}
            # split index 1 -> the receiver expects the second (bottom) half
            recv_table = {"proj.w": [(0, 1)]}

            out = distributed_send_recv(
                {},
                {"proj.w": split_rows},
                send_table,
                recv_table,
                tmp,
                file_keyname_mappings,
                file_machine_mappings,
            )

            # The action saw the exact bytes of the whole tensor.
            np.testing.assert_array_equal(seen["content"], full)
            # Index 1 selected -> bottom half, and it is distinguishable from
            # the top half so a wrong index would be caught.
            np.testing.assert_array_equal(out["proj.w"].numpy(), bottom_half)
            self.assertFalse(np.array_equal(out["proj.w"].numpy(), top_half))

    def test_tp_action_split_index_zero_selects_first_part(self):
        """Split index 0 selects the first part (guards against a fixed pick)."""
        full = np.arange(8, dtype=np.float32).reshape(4, 2)
        top_half = full[0:2]

        def split_rows(safe_slice):
            materialized = np.asarray(safe_slice[:], dtype=np.float32)
            parts = np.array_split(materialized, 2, axis=0)
            return [np.ascontiguousarray(p) for p in parts]

        with tempfile.TemporaryDirectory() as tmp:
            _write_shard(tmp, "tp.safetensors", {"proj.w": full})

            out = distributed_send_recv(
                {},
                {"proj.w": split_rows},
                {"proj.w": 0},
                {"proj.w": [(0, 0)]},
                tmp,
                {"tp.safetensors": ["proj.w"]},
                {"tp.safetensors": [0]},
            )

            np.testing.assert_array_equal(out["proj.w"].numpy(), top_half)


class TestProcessGroupOnlyPathsDocumented(_RequiresRealImports):
    """The remaining public entry points need a real process group."""

    def test_dispatch_and_load_paths_need_real_process_group(self):
        """Document (via skip) the CPU-unverifiable, PG-dependent surface.

        ``create_dispatch_table`` / ``create_optimizer_dispatch_table`` build
        the send/recv tables through ``fleet.get_hybrid_communicate_group`` and
        ``dist.all_gather_object``; ``load_unified_checkpoint_dynamically`` /
        ``load_unified_optimizer_dynamically`` additionally drive real
        ``dist.stream.send/recv`` across ranks. None of these can be exercised
        as a single CPU process without faking collectives (which would prove
        nothing), so their cross-rank numerics are intentionally left to the
        multi-card suite.
        """
        self.skipTest(
            "create_dispatch_table / create_optimizer_dispatch_table / "
            "load_unified_checkpoint_dynamically / "
            "load_unified_optimizer_dynamically require a real "
            "hybrid-parallel process group; verified in multi-card tests only."
        )


if __name__ == "__main__":
    unittest.main()
