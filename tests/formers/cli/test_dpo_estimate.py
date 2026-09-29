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

"""Behavior tests for
``paddlefleet.cli.train.dpo.dpo_estimate_training``.

Two production entry points are exercised:

* ``calculate_acc_steps`` -- recommends a gradient-accumulation count from the
  dataset size, target global batch, dataset-parallel world size and per-device
  batch. Every expected number below is hand-derived from the documented
  formula
  ``min(ceil(recommend_bs / (per_device * world * num_samples / train_batch)), 32)``
  with the ``recommend_bs`` tier table (8 / 16 / 32 / 64 / 128 keyed on the
  ``100 / 1000 / 10000 / 100000`` sample thresholds). Inputs are chosen so the
  result is a distinctive non-trivial value (not the degenerate ``1``), so a
  wrong tier boundary, a dropped world-size/per-device factor or a missing cap
  would change the asserted number.

* ``dpo_estimate_training`` -- walks a real (test-double) dataset, counts
  batches / samples / tokens, optionally recomputes
  ``gradient_accumulation_steps``, derives ``max_steps`` and token totals, flags
  too-small datasets as invalid and persists the result JSON. The dataset is a
  small hand-built collaborator (not a mock of the code under test) whose token
  lengths are distinguishable so a miscount is observable; the produced dict and
  the written ``dpo_train_args.json`` are both checked against independently
  computed expectations.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so when
Paddle / paddlefleet is unavailable the tests skip with a recorded reason
(never faked green). Only ``ImportError`` is treated as "missing dependency".
The dpo_estimate_training cases always pass ``num_of_gpus > 0`` so the code path
never calls ``paddle.distributed.get_world_size()``; they run for real on any
CPU where Paddle is installed.
"""

import json
import os
import tempfile
import types
import unittest

try:
    from paddlefleet.cli.train.dpo.dpo_estimate_training import (
        calculate_acc_steps,
        dpo_estimate_training,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed in this env
    calculate_acc_steps = None
    dpo_estimate_training = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.cli.train.dpo.dpo_estimate_training is unavailable "
    f"(ImportError: {_IMPORT_ERROR}); Paddle is not installed in this env"
)


# --- lightweight, real dataset collaborator (NOT a mock of the tested code) ---


class _Seq:
    """A single sequence exposing ``token_ids`` like the real dataset yields."""

    def __init__(self, token_ids):
        self.token_ids = list(token_ids)


class _FakeDPODataset:
    """Iterable dataset double.

    ``mix_datasets`` length drives ``max_samples``; iterating yields ``batches``
    where each batch is a list of ``_Seq``. This reproduces exactly the two
    surfaces ``dpo_estimate_training`` consumes (``len(mix_datasets)`` and the
    nested ``for sequences ... for sequence in sequences`` walk) without faking
    any of the arithmetic under test.
    """

    def __init__(self, batches, num_mix=None):
        self._batches = batches
        n = len(batches) if num_mix is None else num_mix
        self.mix_datasets = [object() for _ in range(n)]

    def __iter__(self):
        return iter(self._batches)


def _make_training_args(**overrides):
    args = types.SimpleNamespace(
        should_save=True,
        should_save_model_state=False,
        output_dir=None,
        num_train_epochs=1,
        max_steps=-1,
        gradient_accumulation_steps=1,
        num_of_gpus=1,
        per_device_train_batch_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        seed=42,
        dataset_world_size=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_data_args(**overrides):
    args = types.SimpleNamespace(
        max_seq_len=2048,
        max_prompt_len=1024,
        num_samples_each_epoch=6000000,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@unittest.skipUnless(calculate_acc_steps is not None, _SKIP_REASON)
class TestCalculateAccSteps(unittest.TestCase):
    """Hand-derived checks of the recommend/accumulate formula."""

    def test_returns_one_when_batch_dominates(self):
        # samples_per_batch = 2 * 1 * 500 / 4 = 250; num_samples 500 -> rb 16
        # ceil(16 / 250) = ceil(0.064) = 1
        result = calculate_acc_steps(
            num_samples=500,
            train_batch=4,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        self.assertEqual(int(result), 1)

    def test_recommend_bs_boundary_at_100(self):
        # 99 < 100 -> rb 8; 100 is NOT < 100 -> rb 16.
        # With samples_per_batch = 1 (num == train_batch) the result equals rb,
        # so the boundary is directly observable: 8 vs 16.
        below = calculate_acc_steps(
            num_samples=99,
            train_batch=99,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        at = calculate_acc_steps(
            num_samples=100,
            train_batch=100,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        self.assertEqual(int(below), 8)
        self.assertEqual(int(at), 16)

    def test_recommend_bs_boundary_at_1000(self):
        # 999 -> rb 16 ; 1000 -> rb 32. samples_per_batch = 1 in both.
        below = calculate_acc_steps(
            num_samples=999,
            train_batch=999,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        at = calculate_acc_steps(
            num_samples=1000,
            train_batch=1000,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        self.assertEqual(int(below), 16)
        self.assertEqual(int(at), 32)

    def test_recommend_bs_tiers_32_64_128(self):
        # Keep samples_per_batch = 10 (num/train_batch, per_device=world=1) so
        # results stay below the cap and the rb tier is directly visible.
        # 5000 -> rb 32 : ceil(32/10) = ceil(3.2) = 4
        # 50000 -> rb 64 : ceil(64/10) = ceil(6.4) = 7
        # 200000 -> rb 128 : ceil(128/10) = ceil(12.8) = 13
        r_32 = calculate_acc_steps(
            num_samples=5000,
            train_batch=500,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        r_64 = calculate_acc_steps(
            num_samples=50000,
            train_batch=5000,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        r_128 = calculate_acc_steps(
            num_samples=200000,
            train_batch=20000,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        self.assertEqual(int(r_32), 4)
        self.assertEqual(int(r_64), 7)
        self.assertEqual(int(r_128), 13)

    def test_cap_at_32(self):
        # samples_per_batch = 1 * 1 * 50 / 1000 = 0.05 ; rb 8
        # ceil(8 / 0.05) = ceil(160) = 160 -> capped to 32
        result = calculate_acc_steps(
            num_samples=50,
            train_batch=1000,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        self.assertEqual(int(result), 32)

    def test_world_size_and_per_device_both_scale_samples_per_batch(self):
        # Baseline: num 500 -> rb 16 ; samples_per_batch = 1*1*500/100 = 5
        #           ceil(16/5) = ceil(3.2) = 4
        base = calculate_acc_steps(
            num_samples=500,
            train_batch=100,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        # Doubling world size doubles samples_per_batch -> 10 ; ceil(16/10) = 2
        dbl_world = calculate_acc_steps(
            num_samples=500,
            train_batch=100,
            dataset_world_size=2,
            per_device_train_batch_size=1,
        )
        # Doubling per_device batch has the identical multiplicative effect.
        dbl_pdev = calculate_acc_steps(
            num_samples=500,
            train_batch=100,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        self.assertEqual(int(base), 4)
        self.assertEqual(int(dbl_world), 2)
        self.assertEqual(int(dbl_pdev), 2)


@unittest.skipUnless(dpo_estimate_training is not None, _SKIP_REASON)
class TestDpoEstimateTraining(unittest.TestCase):
    """End-to-end estimate over a real (double) dataset + JSON persistence."""

    def _load_written_json(self, output_dir):
        path = os.path.join(output_dir, "dpo_train_args.json")
        self.assertTrue(os.path.isfile(path), "estimate JSON was not written")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_fixed_accumulation_counts_and_estimate(self):
        # 3 batches, one sequence each -> max_samples == 3 == mix_datasets len.
        # token lengths 5,7,3 -> train_tokens = 15 (pre-epoch).
        dataset = _FakeDPODataset(
            batches=[[_Seq(range(5))], [_Seq(range(7))], [_Seq(range(3))]]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = _make_training_args(
                output_dir=tmpdir,
                num_train_epochs=2,
                gradient_accumulation_steps=1,  # >= 0 : keep as-is (no recompute)
                num_of_gpus=1,  # > 0 : avoids paddle.distributed.get_world_size
                per_device_train_batch_size=2,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                seed=42,
            )
            data_args = _make_data_args(
                max_seq_len=2048,
                max_prompt_len=1024,
                num_samples_each_epoch=6000000,
            )

            returned_args, res = dpo_estimate_training(
                tokenizer=None,
                data_args=data_args,
                training_args=training_args,
                dataset_config={"stage": "dpo"},
                train_dataset=dataset,
            )

            # dataset_world_size = num_of_gpus // tp // pp = 1 // 1 // 1 = 1
            # global_batch_size  = pdtbs(2) * gas(1) * dws(1) = 2
            # train_batch (post-epoch) = 3 * 2 = 6
            # max_steps = ceil(6 / 2) = 3
            # total_tokens = 3 * 2048 * 2 = 12288
            # train_tokens (post-epoch) = 15 * 2 = 30
            # valid check: 6 / 2 / 2 = 1.5 >= 1 -> valid True
            self.assertEqual(res["num_train_epochs"], 2)
            self.assertEqual(res["max_steps"], 3)
            self.assertEqual(res["train_samples"], 6)
            self.assertEqual(res["gradient_accumulation_steps"], 1)
            self.assertEqual(res["num_of_gpus"], 1)
            self.assertEqual(res["per_device_train_batch_size"], 2)
            self.assertEqual(res["pipeline_model_parallel_size"], 1)
            self.assertEqual(res["tensor_model_parallel_size"], 1)
            self.assertEqual(res["seed"], 42)
            self.assertEqual(res["num_samples_each_epoch"], 6000000)
            self.assertEqual(res["max_seq_len"], 2048)
            self.assertEqual(res["max_prompt_len"], 1024)
            self.assertEqual(res["total_tokens"], 12288)
            self.assertEqual(res["train_tokens"], 30)
            self.assertTrue(res["valid"])

            # max_steps is coerced to a plain int and mirrored onto the args.
            self.assertIsInstance(res["max_steps"], int)
            self.assertEqual(int(returned_args.max_steps), 3)

            # The persisted JSON must equal the returned estimate exactly.
            self.assertEqual(self._load_written_json(tmpdir), res)

    def test_negative_accumulation_is_recomputed_and_flags_too_small(self):
        # 3 single-sequence batches, 4 tokens each. gradient_accumulation_steps
        # starts negative, so it is replaced via calculate_acc_steps using the
        # PRE-epoch counts (num_samples=3, train_batch=3, dws=1, pdtbs=1):
        #   samples_per_batch = 1*1*3/3 = 1 ; num 3 < 100 -> rb 8
        #   gas = ceil(8 / 1) = 8
        # global_batch_size = pdtbs(1) * gas(8) * dws(1) = 8
        # max_steps = ceil(3 / 8) = 1
        # valid check: 3 / 1 / 8 = 0.375 < 1 -> valid False (dataset too small)
        dataset = _FakeDPODataset(
            batches=[[_Seq(range(4))], [_Seq(range(4))], [_Seq(range(4))]]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = _make_training_args(
                output_dir=tmpdir,
                num_train_epochs=1,
                gradient_accumulation_steps=-1,  # < 0 -> recompute
                num_of_gpus=1,
                per_device_train_batch_size=1,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
            data_args = _make_data_args(max_seq_len=2048)

            _, res = dpo_estimate_training(
                tokenizer=None,
                data_args=data_args,
                training_args=training_args,
                dataset_config={},
                train_dataset=dataset,
            )

            self.assertEqual(res["gradient_accumulation_steps"], 8)
            self.assertEqual(res["max_steps"], 1)
            self.assertEqual(res["train_samples"], 3)
            # total_tokens = max_steps(1) * max_seq_len(2048) * gbs(8) = 16384
            self.assertEqual(res["total_tokens"], 16384)
            self.assertFalse(res["valid"])
            self.assertEqual(self._load_written_json(tmpdir), res)

    def test_dataset_world_size_derived_from_gpus_tp_pp(self):
        # num_of_gpus=8, tp=2, pp=2 -> dataset_world_size = 8 // 2 // 2 = 2
        # (NOT 8). 8 single-seq batches -> train_batch = 8, max_samples = 8.
        # global_batch_size = pdtbs(1) * gas(1) * dws(2) = 2
        # max_steps = ceil(8 / 2) = 4 ; had the code used num_of_gpus (8) as the
        # world size, global_batch_size would be 8 and max_steps 1.
        dataset = _FakeDPODataset(batches=[[_Seq(range(10))] for _ in range(8)])
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = _make_training_args(
                output_dir=tmpdir,
                num_train_epochs=1,
                gradient_accumulation_steps=1,
                num_of_gpus=8,
                per_device_train_batch_size=1,
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=2,
            )
            data_args = _make_data_args(max_seq_len=1024)

            _, res = dpo_estimate_training(
                tokenizer=None,
                data_args=data_args,
                training_args=training_args,
                dataset_config={},
                train_dataset=dataset,
            )

            self.assertEqual(res["num_of_gpus"], 8)
            self.assertEqual(res["tensor_model_parallel_size"], 2)
            self.assertEqual(res["pipeline_model_parallel_size"], 2)
            self.assertEqual(res["max_steps"], 4)
            self.assertEqual(res["train_samples"], 8)
            # train_tokens = 8 batches * 10 tokens * 1 epoch = 80
            self.assertEqual(res["train_tokens"], 80)
            # total_tokens = 4 * 1024 * 2 = 8192
            self.assertEqual(res["total_tokens"], 8192)
            self.assertTrue(res["valid"])
            self.assertEqual(self._load_written_json(tmpdir), res)

    def test_invalid_parallel_config_raises(self):
        # num_of_gpus=2, tp=4 -> 2 // 4 // 1 = 0 < 1 -> ValueError guard.
        dataset = _FakeDPODataset(batches=[[_Seq(range(4))]])
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = _make_training_args(
                output_dir=tmpdir,
                num_of_gpus=2,
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
            )
            data_args = _make_data_args()
            with self.assertRaises(ValueError):
                dpo_estimate_training(
                    tokenizer=None,
                    data_args=data_args,
                    training_args=training_args,
                    dataset_config={},
                    train_dataset=dataset,
                )

    def test_empty_dataset_reports_invalid_and_hardcoded_epoch_samples(self):
        # max_samples == 0 -> else branch: max_steps 0, train_samples 0,
        # valid False, and num_samples_each_epoch is the hardcoded 6000000
        # regardless of the value supplied on data_args (here 123).
        dataset = _FakeDPODataset(batches=[], num_mix=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = _make_training_args(
                output_dir=tmpdir,
                num_train_epochs=3,
                gradient_accumulation_steps=4,
                num_of_gpus=2,
                per_device_train_batch_size=2,
                seed=7,
            )
            data_args = _make_data_args(
                max_seq_len=512, max_prompt_len=256, num_samples_each_epoch=123
            )

            _, res = dpo_estimate_training(
                tokenizer=None,
                data_args=data_args,
                training_args=training_args,
                dataset_config={},
                train_dataset=dataset,
            )

            self.assertEqual(res["max_steps"], 0)
            self.assertEqual(res["train_samples"], 0)
            self.assertFalse(res["valid"])
            self.assertEqual(res["num_train_epochs"], 3)
            self.assertEqual(res["gradient_accumulation_steps"], 4)
            self.assertEqual(res["num_of_gpus"], 2)
            self.assertEqual(res["seed"], 7)
            self.assertEqual(res["max_seq_len"], 512)
            self.assertEqual(res["max_prompt_len"], 256)
            self.assertEqual(res["num_samples_each_epoch"], 6000000)
            self.assertNotIn("train_tokens", res)
            self.assertEqual(self._load_written_json(tmpdir), res)


if __name__ == "__main__":
    unittest.main()
