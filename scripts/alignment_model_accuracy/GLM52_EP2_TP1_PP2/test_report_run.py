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

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from report_run import input_manifest, report_receipts


class ReportRunTest(unittest.TestCase):
    def inputs(self, root):
        model, tokenizer, data = [
            root / name for name in ("model", "tokenizer", "data")
        ]
        for path in (model, tokenizer, data):
            path.mkdir()
        for path in (
            model / "config.json",
            tokenizer / "tokenizer.json",
            data / "alignment_paddle.jsonl",
            data / "alignment_torch.jsonl",
        ):
            path.write_text("{}\n")
        (model / "model.safetensors").write_bytes(
            b"fingerprint fixture, not training weights"
        )
        return model, tokenizer, data

    def test_indexed_weights_are_hashed_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, tokenizer, data = self.inputs(root)
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model.safetensors",
                            "b": "model.safetensors",
                        }
                    }
                )
            )
            rows = input_manifest(model, tokenizer, data)["files"]
            weights = [
                row for row in rows if row["path"].endswith(".safetensors")
            ]
            self.assertEqual(len(weights), 1)
            self.assertEqual(
                weights[0]["sha256"],
                hashlib.sha256(
                    (model / "model.safetensors").read_bytes()
                ).hexdigest(),
            )

    def test_receipts_keep_loss_bits_and_do_not_dump_environment_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "torch").mkdir()
            (root / "torch/env.json").write_text(
                json.dumps(
                    {
                        "framework": "torch",
                        "device_name": "test-device",
                        "access_token": "not-for-logs",
                    }
                )
            )
            losses = [12.167675018310547, 12.166175842285156]
            (root / "torch/loss.json").write_text(
                json.dumps({"losses": losses})
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                report_receipts(root)
            text = stream.getvalue()
            self.assertNotIn("not-for-logs", text)
            rows = [
                json.loads(line.split(" ", 1)[1]) for line in text.splitlines()
            ]
            self.assertEqual(rows[-1]["receipt"]["losses"], losses)

    def test_reporting_preserves_training_and_comparison_failures(self):
        source = Path(__file__).resolve().parent
        for paddle_exit, compare_exit, broken_reporter in (
            (17, 0, False),
            (0, 23, False),
            (17, 0, True),
            (0, 0, False),
            (0, 0, True),
        ):
            with (
                self.subTest(
                    paddle_exit=paddle_exit,
                    compare_exit=compare_exit,
                    broken_reporter=broken_reporter,
                ),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                model, tokenizer, data = self.inputs(root)
                for name in ("run_alignment.sh", "report_run.py"):
                    shutil.copyfile(source / name, root / name)
                if broken_reporter:
                    (root / "report_run.py").write_text(
                        "raise SystemExit(19)\n"
                    )
                (root / "run_paddle_glm52.sh").write_text(
                    f"exit {paddle_exit}\n"
                )
                (root / "run_torch_glm52.sh").write_text(
                    'touch "${GLM52_TEST_MARKER}"\n'
                )
                (root / "compare_loss.py").write_text(
                    f"raise SystemExit({compare_exit})\n"
                )
                venv = root / "venv/paddle/bin"
                venv.mkdir(parents=True)
                (venv / "python").symlink_to(sys.executable)
                env = dict(
                    os.environ,
                    GLM52_VENV_ROOT=str(root / "venv"),
                    GLM52_MODEL_DIR=str(model),
                    GLM52_TOKENIZER_DIR=str(tokenizer),
                    GLM52_DATA_DIR=str(data),
                    ALIGNMENT_RUN_TAG="test",
                    GLM52_TEST_MARKER=str(root / "torch-started"),
                )
                result = subprocess.run(
                    ["bash", str(root / "run_alignment.sh")],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode,
                    paddle_exit or compare_exit,
                    result.stderr,
                )
                self.assertEqual(
                    (root / "torch-started").exists(), paddle_exit == 0
                )


if __name__ == "__main__":
    unittest.main()
