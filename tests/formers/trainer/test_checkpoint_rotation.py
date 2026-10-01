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

"""Checkpoint rotation is deterministic by step and safe to run once per node.

One process per node rotates, so the same logic must hold when nodes share
output_dir (e.g. GPFS) and when each node writes to its own local disk. The
methods under test are invoked on a bare object so the test does not construct
Trainer or a process group.
"""

import os
import tempfile
import threading
import unittest
from types import SimpleNamespace

from paddlefleet.trainer.trainer import Trainer


def _names(path):
    return sorted(os.listdir(path))


def _ckpts(*steps, prefix="checkpoint"):
    # Same lexicographic order as _names().
    return sorted(f"{prefix}-{step}" for step in steps)


class CheckpointRotationTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self._tmpdir.name, "output")
        self.signal_dir = os.path.join(self._tmpdir.name, "signal")
        os.makedirs(self.output_dir)
        os.makedirs(self.signal_dir)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _trainer(
        self, step, limit, local_process_index=0, best=None, hf_limit=None
    ):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            local_process_index=local_process_index,
            save_total_limit=limit,
            save_hf_total_limit=hf_limit,
            output_dir=self.output_dir,
            output_signal_dir=self.signal_dir,
        )
        trainer.state = SimpleNamespace(
            global_step=step, best_model_checkpoint=best
        )
        return trainer

    def _make_checkpoints(self, root, steps, prefix="checkpoint"):
        for step in steps:
            path = os.path.join(root, f"{prefix}-{step}")
            os.makedirs(path)
            open(os.path.join(path, "model.pdparams"), "w").close()

    def _rotate_on_node(self, trainer):
        # Mirrors the gating in Trainer._save_checkpoint.
        if trainer.is_local_process_zero():
            trainer._rotate_checkpoints(output_dir=self.output_dir)
            trainer._rotate_checkpoints(output_dir=self.signal_dir)

    def test_non_local_zero_process_does_not_delete(self):
        steps = [16000, 20000, 24000, 28000]
        self._make_checkpoints(self.output_dir, steps)
        self._make_checkpoints(self.signal_dir, steps)

        self._rotate_on_node(self._trainer(28000, 3, local_process_index=1))

        self.assertEqual(_names(self.output_dir), _ckpts(*steps))
        self.assertEqual(_names(self.signal_dir), _ckpts(*steps))

    def test_deletes_only_oldest_step_and_ignores_other_entries(self):
        self._make_checkpoints(self.output_dir, [16000, 20000, 24000, 28000])
        self._make_checkpoints(self.signal_dir, [16000, 20000, 24000, 28000])
        open(os.path.join(self.output_dir, "checkpoint-99999.tmp"), "w").close()
        os.makedirs(os.path.join(self.output_dir, "checkpoint-foo"))
        open(os.path.join(self.output_dir, "notes.txt"), "w").close()

        self._rotate_on_node(self._trainer(28000, limit=3))

        self.assertEqual(
            _names(self.output_dir),
            _ckpts(20000, 24000, 28000)
            + ["checkpoint-99999.tmp", "checkpoint-foo", "notes.txt"],
        )
        self.assertEqual(_names(self.signal_dir), _ckpts(20000, 24000, 28000))

    def test_order_is_by_step_not_name(self):
        # Lexicographically "checkpoint-12000" < "checkpoint-4000".
        self._make_checkpoints(self.output_dir, [4000, 8000, 12000, 16000])

        self._trainer(16000, limit=2)._rotate_checkpoints(
            output_dir=self.output_dir
        )

        self.assertEqual(_names(self.output_dir), _ckpts(12000, 16000))

    def test_backlog_is_trimmed_to_the_limit(self):
        self._make_checkpoints(
            self.output_dir, [4000, 8000, 12000, 16000, 20000, 24000]
        )

        self._trainer(24000, limit=3)._rotate_checkpoints(
            output_dir=self.output_dir
        )

        self.assertEqual(_names(self.output_dir), _ckpts(16000, 20000, 24000))

    def test_current_checkpoint_not_yet_visible_reserves_a_slot(self):
        # checkpoint-28000 is still being created by other ranks.
        self._make_checkpoints(self.output_dir, [16000, 20000, 24000])

        self._trainer(28000, limit=3)._rotate_checkpoints(
            output_dir=self.output_dir
        )

        self.assertEqual(_names(self.output_dir), _ckpts(20000, 24000))

    def test_at_limit_and_limit_one(self):
        self._make_checkpoints(self.output_dir, [20000, 24000, 28000])
        trainer = self._trainer(28000, limit=3)
        trainer._rotate_checkpoints(output_dir=self.output_dir)
        self.assertEqual(_names(self.output_dir), _ckpts(20000, 24000, 28000))

        trainer.args.save_total_limit = 1
        trainer._rotate_checkpoints(output_dir=self.output_dir)
        self.assertEqual(_names(self.output_dir), _ckpts(28000))

    def test_disabled_limit_keeps_everything(self):
        self._make_checkpoints(self.output_dir, [4000, 8000, 12000])
        for limit in (None, 0):
            self._trainer(12000, limit=limit)._rotate_checkpoints(
                output_dir=self.output_dir
            )
            self.assertEqual(_names(self.output_dir), _ckpts(4000, 8000, 12000))

    def _rotate_with_best(self, steps, current, limit, best_step):
        self._make_checkpoints(self.output_dir, steps)
        best = os.path.join(self.output_dir, f"checkpoint-{best_step}")
        self._trainer(current, limit=limit, best=best)._rotate_checkpoints(
            output_dir=self.output_dir
        )
        return _names(self.output_dir)

    def test_best_checkpoint_stays_out_of_the_deletion_window(self):
        self.assertEqual(
            self._rotate_with_best(
                [16000, 20000, 24000, 28000], 28000, 3, 16000
            ),
            _ckpts(16000, 24000, 28000),
        )

    def test_best_checkpoint_kept_with_limit_two(self):
        self.assertEqual(
            self._rotate_with_best([20000, 24000, 28000], 28000, 2, 20000),
            _ckpts(20000, 28000),
        )

    def test_best_checkpoint_kept_with_limit_one(self):
        # Keep both the best and the latest so training can still resume.
        self.assertEqual(
            self._rotate_with_best([20000, 24000, 28000], 28000, 1, 20000),
            _ckpts(20000, 28000),
        )

    def test_best_checkpoint_is_current(self):
        self.assertEqual(
            self._rotate_with_best([20000, 24000, 28000], 28000, 2, 28000),
            _ckpts(24000, 28000),
        )

    def test_hf_rotation_uses_its_own_limit_and_prefix(self):
        steps = [4000, 8000, 12000]
        self._make_checkpoints(self.output_dir, steps)
        self._make_checkpoints(self.output_dir, steps, prefix="hf_checkpoint")

        self._trainer(12000, limit=1, hf_limit=2)._rotate_hf_checkpoints(
            output_dir=self.output_dir
        )

        self.assertEqual(
            _names(self.output_dir),
            _ckpts(*steps) + _ckpts(8000, 12000, prefix="hf_checkpoint"),
        )

    def test_shared_dir_rotated_concurrently_by_every_node(self):
        # GPFS: every node's local rank 0 rotates the same directory while
        # other ranks are still writing the current checkpoint.
        self._make_checkpoints(self.output_dir, range(4000, 32001, 4000))
        current = os.path.join(self.output_dir, "checkpoint-32000")
        stop = threading.Event()
        errors = []

        def write_current_shards():
            i = 0
            while not stop.is_set():
                open(os.path.join(current, f"shard_{i % 64}"), "w").close()
                i += 1

        def rotate():
            try:
                self._trainer(32000, limit=3)._rotate_checkpoints(
                    output_dir=self.output_dir
                )
            except Exception as e:
                errors.append(e)

        writer = threading.Thread(target=write_current_shards)
        writer.start()
        nodes = [threading.Thread(target=rotate) for _ in range(8)]
        for node in nodes:
            node.start()
        for node in nodes:
            node.join()
        stop.set()
        writer.join()

        self.assertEqual(errors, [])
        self.assertEqual(_names(self.output_dir), _ckpts(24000, 28000, 32000))
        self.assertIn("model.pdparams", os.listdir(current))

    def test_node_local_dirs_are_each_rotated(self):
        # Node-local disks: each node holds its own shards of every step.
        node_dirs = []
        for node in range(4):
            node_dir = os.path.join(self._tmpdir.name, f"node{node}")
            os.makedirs(node_dir)
            self._make_checkpoints(node_dir, range(4000, 32001, 4000))
            node_dirs.append(node_dir)

        for node_dir in node_dirs:
            self._trainer(32000, limit=3)._rotate_checkpoints(
                output_dir=node_dir
            )

        for node_dir in node_dirs:
            self.assertEqual(_names(node_dir), _ckpts(24000, 28000, 32000))


if __name__ == "__main__":
    unittest.main()
