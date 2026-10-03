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

"""Exercise the real multi-card shell runner with controlled process outcomes."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestMultiCardRunner(unittest.TestCase):
    def run_scenario(self, scenario):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_dir = root / "tests/multi_card_tests"
            test_dir.mkdir(parents=True)
            for name in ["test_a.py", "test_b.py"]:
                (test_dir / name).touch()
            (root / "ci").mkdir()
            shutil.copy2(source / "check_log_for_exitcode.py", root / "ci")
            executables = root / "bin"
            executables.mkdir()
            scripts = {
                "coverage": """import os, sys
from pathlib import Path
first = Path(sys.argv[-1]).name == "test_a.py"
print("OK", flush=True)
if first and os.environ["RUNNER_SCENARIO"] == "rank_failure":
    print("FAILED (failures=1)", file=sys.stderr, flush=True)
    sys.exit(1)
if first and os.environ["RUNNER_SCENARIO"] == "crash":
    sys.exit(139)
""",
                "timeout": """import os, sys
from pathlib import Path
if os.environ["RUNNER_SCENARIO"] == "timeout" and Path(sys.argv[-1]).name == "test_a.py":
    print("OK", flush=True)
    sys.exit(124)
os.execvp(sys.argv[3], sys.argv[3:])
""",
                "yq": 'print("tests/multi_card_tests/*|2")\n',
                "find": """import sys
from pathlib import Path
for path in sorted(Path(sys.argv[1]).rglob("test_*.py")):
    print(path)
""",
            }
            if scenario == "log_failure":
                scripts["tee"] = (
                    "import sys\nsys.stdout.write(sys.stdin.read())\nsys.exit(1)\n"
                )
            for name, script in scripts.items():
                path = executables / name
                path.write_text(f"#!{sys.executable}\n{script}")
                path.chmod(0o755)
            result = subprocess.run(
                ["bash", str(source / "multi-card_test.sh")],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": f"{executables}{os.pathsep}{os.environ['PATH']}",
                    "work_dir": str(root),
                    "RUNNER_SCENARIO": scenario,
                },
                capture_output=True,
                text=True,
                timeout=30,
            )
            log = root / "test_a_multi_card.log"
            return result, log.read_text() if log.exists() else ""

    def test_success_requires_successful_processes(self):
        result, _ = self.run_scenario("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Tests executed: 2", result.stdout)

    def test_failed_rank_after_ok_fails_and_keeps_running_other_cases(self):
        result, log = self.run_scenario("rank_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("FAILED (failures=1)", log)
        self.assertIn("test_b.py", result.stdout)
        self.assertIn("Tests executed: 2", result.stdout)

    def test_crash_after_ok_still_fails(self):
        result, _ = self.run_scenario("crash")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_timeout_after_ok_still_fails(self):
        result, _ = self.run_scenario("timeout")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Test TIMEOUT:", result.stdout)

    def test_unwritable_test_log_is_a_failure(self):
        result, _ = self.run_scenario("log_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
