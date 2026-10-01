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
"""Behavioral unit tests for the auto_parallel pretraining workflow.

Module under test: ``paddlefleet.cli.train.auto_parallel.workflow`` (Trainer
training engine / configuration-and-run infrastructure per the PaddleFleet unit
test rules). These are no-card, CPU-runnable tests.

Independent oracle. None of the expected values below are produced by calling
the function under test. They are hand-derived from the source contract:

* ``get_train_data_file``: expected path lists are built by hand from the
  documented rules -- a whitespace-split ``input_dir`` is a verbatim
  ``weight prefix ...`` list; a single directory is scanned for names
  containing ``_idx.npz`` or ``.idx``, those suffixes are stripped, and the
  result is (a) an interleaved ``[1.0, prefix, ...]`` list when more than one
  dataset is found, (b) a bare ``[prefix]`` list (no weight) for exactly one,
  or (c) ``[]`` for none. Real files/dirs are created under a tempdir so the
  genuine ``os.listdir``/``os.path.isfile`` path executes; expected prefixes
  are assembled independently with ``os.path.join``.
* ``create_pretrained_dataset``: the ``train/valid/test`` sample counts are
  recomputed by hand (770 / 390 / 255) from distinct, non-coinciding operands
  so that swapping any field is observable; the non-tested data builder is
  spied to capture the actually-forwarded arguments.
* ``_collate_data``: expected shifted ``input_ids``/``labels`` are written out
  by hand for distinguishable token ids so a labels/tokens swap is rejected.
* ``PretrainingTrainer``: the only behavior it adds over ``Trainer`` is setting
  ``is_pretraining = True`` after forwarding to ``super().__init__``; the heavy
  base initializer is spied (not the tested subclass initializer).

Paddle-only import paths are guarded: if ``paddlefleet`` cannot be imported
because ``paddle`` (or another dependency) is absent, every test is skipped
with a recorded reason rather than being faked green. Only ``ImportError`` is
treated as "missing dependency".
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import paddlefleet.cli.train.auto_parallel.workflow as workflow_mod
    from paddlefleet.cli.train.auto_parallel.workflow import (
        PretrainingTrainer,
        create_pretrained_dataset,
        get_train_data_file,
    )
    from paddlefleet.trainer.trainer import Trainer

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddlefleet workflow import failed (missing dependency): {_IMPORT_ERROR!r}"


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestGetTrainDataFile(unittest.TestCase):
    """get_train_data_file: real filesystem discovery and weighting rules."""

    def test_whitespace_input_is_split_verbatim(self):
        # Multiple whitespace-separated tokens => returned unchanged, as
        # `weight prefix weight prefix ...`. No filesystem access happens.
        args = SimpleNamespace(input_dir="0.3 /data/a 0.7 /data/b")
        result = get_train_data_file(args)
        self.assertEqual(result, ["0.3", "/data/a", "0.7", "/data/b"])
        # Weights arrive as the raw strings the user typed, not floats.
        self.assertIsInstance(result[0], str)

    def test_whitespace_input_collapses_runs_of_spaces(self):
        args = SimpleNamespace(input_dir="1   /x/y")
        self.assertEqual(get_train_data_file(args), ["1", "/x/y"])

    def test_multiple_index_files_interleaved_with_unit_weights(self):
        with tempfile.TemporaryDirectory() as d:
            # Two valid datasets, one non-index decoy, one directory decoy.
            open(os.path.join(d, "alpha.idx"), "w").close()
            open(os.path.join(d, "beta_idx.npz"), "w").close()
            open(os.path.join(d, "gamma.bin"), "w").close()  # not an index
            os.mkdir(
                os.path.join(d, "subdir.idx")
            )  # matches name but not a file

            result = get_train_data_file(SimpleNamespace(input_dir=d))

            expected_prefixes = {
                os.path.join(d, "alpha"),
                os.path.join(d, "beta"),
            }
            # Interleaved layout: [1.0, prefix, 1.0, prefix]; listdir order is
            # not guaranteed, so verify the pairing/weights order-independently.
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0::2], [1.0, 1.0])
            for w in result[0::2]:
                self.assertIsInstance(w, float)
            self.assertEqual(set(result[1::2]), expected_prefixes)
            # Decoys excluded entirely (neither the .bin file nor the dir).
            self.assertNotIn(os.path.join(d, "gamma.bin"), result)
            self.assertNotIn(os.path.join(d, "subdir"), result)

    def test_single_index_file_returns_bare_prefix_without_weight(self):
        # Exactly one dataset takes the `len(files) > 1` == False path and is
        # returned as a bare [prefix] list -- crucially WITHOUT a 1.0 weight.
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "solo_idx.npz"), "w").close()
            result = get_train_data_file(SimpleNamespace(input_dir=d))
            self.assertEqual(result, [os.path.join(d, "solo")])
            self.assertNotIn(1.0, result)

    def test_single_idx_suffix_stripped(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "train.idx"), "w").close()
            result = get_train_data_file(SimpleNamespace(input_dir=d))
            self.assertEqual(result, [os.path.join(d, "train")])

    def test_no_index_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "data.bin"), "w").close()
            open(os.path.join(d, "notes.txt"), "w").close()
            self.assertEqual(
                get_train_data_file(SimpleNamespace(input_dir=d)), []
            )


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestPretrainingTrainer(unittest.TestCase):
    """PretrainingTrainer adds is_pretraining=True atop the real Trainer."""

    def test_is_subclass_of_trainer(self):
        self.assertTrue(issubclass(PretrainingTrainer, Trainer))

    def test_init_forwards_to_super_and_sets_flag(self):
        captured = {}

        def spy_base_init(self, *args, **kwargs):
            # Stand in for the heavy base Trainer.__init__ (non-tested
            # collaborator). It deliberately does NOT set is_pretraining, so a
            # True value can only come from the subclass under test.
            captured["args"] = args
            captured["kwargs"] = kwargs

        with mock.patch.object(Trainer, "__init__", spy_base_init):
            obj = PretrainingTrainer(
                "model-obj", args="ARGS", data_collator="dc"
            )

        self.assertTrue(obj.is_pretraining)
        self.assertEqual(captured["args"], ("model-obj",))
        self.assertEqual(
            captured["kwargs"], {"args": "ARGS", "data_collator": "dc"}
        )


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestCreatePretrainedDataset(unittest.TestCase):
    """create_pretrained_dataset: sample-count math and collate label shift."""

    def _make_args(self):
        # Distinct operands chosen so no two products coincide and any field
        # swap changes the result: train=2*5*7*11=770,
        # valid=3*5*13*(7//4 + 1)=3*5*13*2=390, test=3*5*17=255.
        data_args = SimpleNamespace(
            split="949,50,1",  # real check_data_split accepts this
            data_impl="mmap",
            max_seq_len=1024,
            seed=999,  # unused by builder (builder takes training_args.seed)
            skip_warmup=True,
            share_folder=False,
            data_cache="/tmp/does-not-matter",
        )
        training_args = SimpleNamespace(
            do_train=True,
            do_eval=True,
            do_predict=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=3,
            dataset_world_size=5,
            max_steps=7,
            gradient_accumulation_steps=11,
            eval_iters=13,
            eval_steps=4,
            test_iters=17,
            seed=123,
        )
        return data_args, training_args

    def test_forwards_hand_derived_sample_counts_and_prefix(self):
        data_args, training_args = self._make_args()
        captured = {}

        def spy_build(**kwargs):
            captured.update(kwargs)
            return ("train_ds", "valid_ds", "test_ds")

        with (
            mock.patch.object(
                workflow_mod, "build_train_valid_test_datasets", spy_build
            ),
            mock.patch.object(
                workflow_mod, "print_rank_0", lambda *a, **k: None
            ),
        ):
            train, valid, test, collator = create_pretrained_dataset(
                data_args,
                training_args,
                data_file="/prefix/dataset",
                tokenizer=object(),
                need_data=False,
            )

        # Independently hand-derived expected sample counts.
        self.assertEqual(
            captured["train_val_test_num_samples"], [770, 390, 255]
        )
        # Key arguments are actually forwarded to the data builder.
        self.assertEqual(captured["data_prefix"], "/prefix/dataset")
        self.assertEqual(captured["seq_length"], 1024)
        self.assertEqual(captured["splits_string"], "949,50,1")
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["data_impl"], "mmap")
        self.assertIs(captured["need_data"], False)
        # The builder's three datasets are returned in order.
        self.assertEqual(
            (train, valid, test), ("train_ds", "valid_ds", "test_ds")
        )
        self.assertTrue(callable(collator))

    def test_collate_shifts_tokens_and_labels_by_one(self):
        data_args, training_args = self._make_args()

        def spy_build(**kwargs):
            return ("t", "v", "te")

        with (
            mock.patch.object(
                workflow_mod, "build_train_valid_test_datasets", spy_build
            ),
            mock.patch.object(
                workflow_mod, "print_rank_0", lambda *a, **k: None
            ),
        ):
            _, _, _, collator = create_pretrained_dataset(
                data_args,
                training_args,
                data_file="/prefix/dataset",
                tokenizer=object(),
                need_data=False,
            )

        batch = [
            {"text": np.array([10, 11, 12, 13])},
            {"text": np.array([20, 21, 22, 23])},
        ]
        out = collator(batch)

        # labels are the next token; input_ids drop the last token.
        np.testing.assert_array_equal(
            np.asarray(out["input_ids"]), [[10, 11, 12], [20, 21, 22]]
        )
        np.testing.assert_array_equal(
            np.asarray(out["labels"]), [[11, 12, 13], [21, 22, 23]]
        )
        # Guard against a labels/tokens swap regression.
        self.assertFalse(
            np.array_equal(
                np.asarray(out["input_ids"]), np.asarray(out["labels"])
            )
        )


if __name__ == "__main__":
    unittest.main()
