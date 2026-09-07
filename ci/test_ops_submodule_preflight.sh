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
[[ "$*" == "submodule update --init --recursive --force" ]]
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
# Exercise Git itself: an aborted clone can leave an apparently current HEAD
# with no checked-out files. A normal retry returns success without repairing it.
with tempfile.TemporaryDirectory() as directory:
    work = pathlib.Path(directory)
    env = dict(os.environ, GIT_CONFIG_COUNT='1',
               GIT_CONFIG_KEY_0='protocol.file.allow', GIT_CONFIG_VALUE_0='always')

    def git(path, *args, check=True):
        return subprocess.run(['git', '-C', str(path), *args], env=env,
                              capture_output=True, text=True, check=check, timeout=10)

    for name in ['leaf', 'good', 'bad', 'super']:
        repo = work / name
        repo.mkdir()
        git(repo, 'init', '-q')
        git(repo, 'config', 'user.name', 'fixture')
        git(repo, 'config', 'user.email', 'fixture@example.invalid')
        git(repo, 'config', 'commit.gpgsign', 'false')
        (repo / 'payload').write_text(name)
        git(repo, 'add', 'payload')
        git(repo, 'commit', '-qm', 'fixture')

    def add_links(repo, links):
        (repo / '.gitmodules').write_text(''.join(
            f'[submodule "{name}"]\n path = {name}\n url = {url}\n'
            for name, url, sha in links))
        git(repo, 'add', '.gitmodules')
        for name, url, sha in links:
            git(repo, 'update-index', '--add', '--cacheinfo', f'160000,{sha},{name}')
        git(repo, 'commit', '-qm', 'submodules')

    leaf_sha = git(work / 'leaf', 'rev-parse', 'HEAD').stdout.strip()
    add_links(work / 'good', [('nested', work / 'leaf', leaf_sha)])
    good_sha = git(work / 'good', 'rev-parse', 'HEAD').stdout.strip()
    bad_sha = git(work / 'bad', 'rev-parse', 'HEAD').stdout.strip()
    repo = work / 'super'
    add_links(repo, [('a_good', work / 'good', good_sha),
                     ('z_bad', work / 'missing', bad_sha)])
    first = git(repo, 'submodule', 'update', '--init', '--recursive', check=False)
    assert first.returncode != 0
    assert not (repo / 'a_good/payload').exists()
    git(repo, 'config', 'submodule.z_bad.url', str(work / 'bad'))
    retry = git(repo, 'submodule', 'update', '--init', '--recursive', check=False)
    assert retry.returncode == 0, retry.stderr
    assert not (repo / 'a_good/payload').exists()
    print('PASS: real Git ordinary retry returns 0 with incomplete checkout')
    prepared = subprocess.run(['bash', str(root / 'ci/prepare_ops_submodules.sh')],
                              cwd=repo, env=env, capture_output=True, text=True, timeout=10)
    assert prepared.returncode == 0, prepared.stderr
    for name, sha, payload in [('a_good', good_sha, 'good'),
                               ('a_good/nested', leaf_sha, 'leaf'),
                               ('z_bad', bad_sha, 'bad')]:
        checkout = repo / name
        assert (checkout / 'payload').read_text() == payload
        assert git(checkout, 'rev-parse', 'HEAD').stdout.strip() == sha
        git(checkout, 'diff', '--exit-code')
    print('PASS: real preflight restores all pinned parent/child worktrees')
print('All ops submodule preflight fixtures passed')
PY
