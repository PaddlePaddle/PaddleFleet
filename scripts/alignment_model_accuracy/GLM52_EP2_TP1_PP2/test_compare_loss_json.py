#!/usr/bin/env python3

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

import json
import os
import tempfile
import unittest

from compare_loss import compare_loss_json, validate_loss_artifact


def _art(
    framework,
    n=2,
    losses=None,
    steps=None,
    extra=None,
    schema="glm52-machine-loss/v1",
):
    obj = {
        "schema": schema,
        "raw": True,
        "stage": "training_callback_complete",
        "framework": framework,
        "steps": steps if steps is not None else list(range(1, n + 1)),
        "losses": losses if losses is not None else [1.0] * n,
    }
    if extra:
        obj.update(extra)
    return obj


def _write(dirpath, name, obj):
    p = os.path.join(dirpath, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return p


class TestValidateLossArtifact(unittest.TestCase):
    def test_ok(self):
        losses = validate_loss_artifact(
            _art("paddle", 2, [0.1, -0.0]),
            expected_framework="paddle",
            required_steps=2,
        )
        self.assertEqual(len(losses), 2)

    def test_wrong_framework(self):
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("torch"), expected_framework="paddle", required_steps=2
            )

    def test_wrong_schema(self):
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("paddle", extra=None, schema="other"),
                expected_framework="paddle",
                required_steps=2,
            )

    def test_empty_arrays(self):
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("paddle", 0, [], []),
                expected_framework="paddle",
                required_steps=1,
            )

    def test_unequal_length(self):
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("paddle", losses=[1.0], steps=[1, 2]),
                expected_framework="paddle",
                required_steps=2,
            )

    def test_wrong_count(self):
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("paddle", 2),
                expected_framework="paddle",
                required_steps=100,
            )

    def test_noncontiguous_steps(self):
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("paddle", steps=[1, 3], losses=[1.0, 2.0]),
                expected_framework="paddle",
                required_steps=2,
            )

    def test_nan_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                validate_loss_artifact(
                    _art("paddle", losses=[1.0, bad]),
                    expected_framework="paddle",
                    required_steps=2,
                )

    def test_boolean_rejected(self):
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("paddle", losses=[True, 1.0]),
                expected_framework="paddle",
                required_steps=2,
            )
        with self.assertRaises(ValueError):
            validate_loss_artifact(
                _art("paddle", steps=[True, 2], losses=[1.0, 2.0]),
                expected_framework="paddle",
                required_steps=2,
            )


class TestCompareLossJson(unittest.TestCase):
    def _pair(self, pf, mg):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        td = temp.name
        return _write(td, "pf.json", pf), _write(td, "mg.json", mg)

    def test_bit_identical_pass(self):
        pf, mg = self._pair(
            _art("paddle", 2, [1.25, -0.0]), _art("torch", 2, [1.25, -0.0])
        )
        self.assertEqual(compare_loss_json(pf, mg, 2), 0)

    def test_signed_zero_mismatch(self):
        pf, mg = self._pair(
            _art("paddle", 2, [0.0, 1.0]), _art("torch", 2, [-0.0, 1.0])
        )
        self.assertEqual(compare_loss_json(pf, mg, 2), 1)

    def test_ulp_mismatch(self):
        a = 1.0
        b = float.fromhex("0x1.0000000000001p+0")
        pf, mg = self._pair(
            _art("paddle", 1, [a], [1]), _art("torch", 1, [b], [1])
        )
        self.assertEqual(compare_loss_json(pf, mg, 1), 1)

    def test_missing_file(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        td = temp.name
        pf = _write(td, "pf.json", _art("paddle", 1, [1.0], [1]))
        self.assertEqual(
            compare_loss_json(pf, os.path.join(td, "nope.json"), 1), 1
        )

    def test_truncated_json(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        td = temp.name
        pf = _write(td, "pf.json", _art("paddle", 1, [1.0], [1]))
        bad = os.path.join(td, "mg.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write('{"schema": "glm52-machine-loss/v1", "framework": "torch"')
        self.assertEqual(compare_loss_json(pf, bad, 1), 1)

    def test_empty_file(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        td = temp.name
        pf = _write(td, "pf.json", _art("paddle", 1, [1.0], [1]))
        bad = os.path.join(td, "mg.json")
        open(bad, "w").close()
        self.assertEqual(compare_loss_json(pf, bad, 1), 1)

    def test_swapped_framework_fail_closed(self):
        pf, mg = self._pair(
            _art("torch", 1, [1.0], [1]), _art("paddle", 1, [1.0], [1])
        )
        self.assertEqual(compare_loss_json(pf, mg, 1), 1)


class TestIncompleteAndStaleArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name

    def test_incomplete_native_callback_is_rejected(self):
        for patch in ({"raw": False}, {"stage": "training"}, {"losses": None}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_loss_artifact(
                    _art("paddle", extra=patch),
                    expected_framework="paddle",
                    required_steps=2,
                )

    def test_does_not_reuse_previous_success(self):
        old = os.path.join(self.root, "old-success")
        os.mkdir(old)
        _write(old, "loss.json", _art("paddle"))
        torch = _write(self.root, "torch.json", _art("torch"))
        self.assertEqual(compare_loss_json(self.root, torch, 2), 1)

    def test_wrong_root_and_overflow_fail_closed(self):
        torch = _write(self.root, "torch.json", _art("torch"))
        for bad in ([], _art("paddle", losses=[10**400, 1.0])):
            with self.subTest(bad_type=type(bad).__name__):
                paddle = _write(self.root, "paddle.json", bad)
                self.assertEqual(compare_loss_json(paddle, torch, 2), 1)


if __name__ == "__main__":
    unittest.main()
