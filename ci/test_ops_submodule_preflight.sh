#!/usr/bin/env bash
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

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON_BIN:-python3}" - "$ROOT" <<'PY'
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1])
workflow = (root / '.github/workflows/Test-release.yml').read_text()
start = workflow.index('          bash ci/prepare_ops_submodules.sh')
end = workflow.index('          OPS_WHL=', start)
commands = workflow[start:end]
assert commands.count('pip install') == 2
assert 'uv build --wheel --package paddlefleet-ops' in commands
assert "/bin/bash -ce '" in workflow[workflow.rfind('      - name:', 0, start):start]
real_timeout = shutil.which('timeout')
assert real_timeout
with tempfile.TemporaryDirectory() as directory:
    work = pathlib.Path(directory)
    (work / 'bin').mkdir()
    (work / 'ci').symlink_to(root / 'ci')
    mocks = {
        'git': '''#!/usr/bin/env bash
set -eu
[[ "$*" == "submodule update --init --recursive" ]]
echo prepare >> "$TRACE"
n=$(grep -c prepare "$TRACE")
case "$CASE" in
 success) exit 0;;
 transient) [[ $n -ge 2 ]];;
 persistent) exit 23;;
 timeout) /bin/sleep 5;;
esac
''',
        'sleep': '#!/usr/bin/env bash\necho "sleep:$1" >> "$TRACE"\n',
        'pip': '#!/usr/bin/env bash\necho install >> "$TRACE"\n',
        'uv': '#!/usr/bin/env bash\necho build >> "$TRACE"\n',
        'timeout': '''#!/usr/bin/env bash
set -eu
[[ $1 == --kill-after=30s && $2 == 15m ]]
echo bound >> "$TRACE"
shift 2
if [[ $CASE == timeout ]]; then exec "$REAL_TIMEOUT" --kill-after=0.1s 0.1s "$@"; fi
exec "$@"
''',
    }
    for name, source in mocks.items():
        command = work / 'bin' / name
        command.write_text(source)
        command.chmod(0o755)
    env = dict(os.environ, PATH=str(work / 'bin') + ':' + os.environ['PATH'],
               TRACE=str(work / 'trace'), REAL_TIMEOUT=real_timeout)
    for case, count, code in [('success', 1, 0), ('transient', 2, 0),
                              ('persistent', 3, 23), ('timeout', 3, 124)]:
        trace = work / 'trace'
        trace.write_text('')
        env['CASE'] = case
        result = subprocess.run(['bash', '-ce', commands], cwd=work,
                                env=env, capture_output=True, text=True, timeout=10)
        lines = trace.read_text().splitlines()
        assert result.returncode == code, (case, result.returncode, result.stderr)
        assert lines.count('prepare') == count, (case, lines)
        assert lines.count('bound') == count, (case, lines)
        assert [x for x in lines if x.startswith('sleep:')] == ['sleep:15', 'sleep:30'][:max(0, count - 1)], (case, lines)
        assert lines.count('install') == (2 if code == 0 else 0), (case, lines)
        assert lines.count('build') == (1 if code == 0 else 0), (case, lines)
        print(f'PASS: extracted workflow {case}, attempts={count}, exit={code}')
print('All ops submodule preflight fixtures passed')
PY
