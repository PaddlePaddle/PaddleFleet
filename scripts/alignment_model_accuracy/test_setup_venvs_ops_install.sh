#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# CPU fixture for setup_venvs.sh ops install argv.
# Directory paths must get --no-build-isolation; wheel/url paths must not.
# Does not run uv, compile ops, or touch GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SETUP="${SCRIPT_DIR}/setup_venvs.sh"
PY="/nonexistent/python"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

dir_path="${tmp}/paddlefleet_ops"
mkdir -p "${dir_path}"
wheel_path="${tmp}/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl"
echo dummy >"${wheel_path}"
url_path="https://example.invalid/paddlefleet_ops-0.0.0-cp312-cp312-linux_x86_64.whl"

dump_argv() {
    local kind="$1"
    local path="$2"
    bash "${SETUP}" --ops-install-argv "${PY}" "${path}" >"${tmp}/${kind}.out"
}

has_flag() {
    grep -qx "ARG:${1}" "${2}"
}

dump_argv dir "${dir_path}"
grep -qx "KIND=dir" "${tmp}/dir.out"
has_flag --no-build-isolation "${tmp}/dir.out"
has_flag --python "${tmp}/dir.out"
has_flag --force-reinstall "${tmp}/dir.out"
grep -qx "ARG:${dir_path}" "${tmp}/dir.out"
# --python stays before the path; isolation flag is extra, not a replacement.
grep -q "ARG:--python" "${tmp}/dir.out"

dump_argv wheel "${wheel_path}"
grep -qx "KIND=wheel" "${tmp}/wheel.out"
if has_flag --no-build-isolation "${tmp}/wheel.out"; then
    echo "FAIL: wheel path gained --no-build-isolation" >&2
    cat "${tmp}/wheel.out" >&2
    exit 1
fi
has_flag --python "${tmp}/wheel.out"
has_flag --force-reinstall "${tmp}/wheel.out"
grep -qx "ARG:${wheel_path}" "${tmp}/wheel.out"

dump_argv url "${url_path}"
grep -qx "KIND=url" "${tmp}/url.out"
if has_flag --no-build-isolation "${tmp}/url.out"; then
    echo "FAIL: url path gained --no-build-isolation" >&2
    cat "${tmp}/url.out" >&2
    exit 1
fi
has_flag --python "${tmp}/url.out"
has_flag --force-reinstall "${tmp}/url.out"
grep -qx "ARG:${url_path}" "${tmp}/url.out"

# Historical wheel argv shape: --python PY --force-reinstall PATH
mapfile -t wheel_args < <(grep '^ARG:' "${tmp}/wheel.out" | sed 's/^ARG://')
if [[ "${wheel_args[*]}" != "--python ${PY} --force-reinstall ${wheel_path}" ]]; then
    echo "FAIL: wheel argv degraded: ${wheel_args[*]}" >&2
    exit 1
fi

# CPU mock: a source tree that is not a git worktree must fail closed
# before any uv / compile step. Does not clone, fetch, or build.
bare="${tmp}/bare_ops"
mkdir -p "${bare}"
set +e
bash "${SETUP}" --prepare-ops-submodules "${bare}" >"${tmp}/bare.out" 2>"${tmp}/bare.err"
bare_rc=$?
set -e
if [[ "${bare_rc}" -eq 0 ]]; then
    echo "FAIL: bare ops tree was allowed to continue" >&2
    cat "${tmp}/bare.err" >&2
    exit 1
fi
if ! grep -q "not inside a git work tree" "${tmp}/bare.err"; then
    echo "FAIL: bare ops tree did not fail closed on missing git" >&2
    cat "${tmp}/bare.err" >&2
    exit 1
fi
if grep -q "Building paddlefleet-ops" "${tmp}/bare.out" "${tmp}/bare.err"; then
    echo "FAIL: bare ops tree continued into a build" >&2
    exit 1
fi

# CPU mock: a git tree without .gitmodules / recorded gitlinks must
# refuse rather than call `git submodule update` on empty paths.
empty_repo="${tmp}/empty.git"
mkdir -p "${empty_repo}/packages/paddlefleet_ops"
echo placeholder >"${empty_repo}/packages/paddlefleet_ops/README"
git -C "${empty_repo}" init -q
git -C "${empty_repo}" config user.email "fixture@example.invalid"
git -C "${empty_repo}" config user.name "fixture"
git -C "${empty_repo}" add packages/paddlefleet_ops/README
git -C "${empty_repo}" commit -qm "empty ops tree"
set +e
bash "${SETUP}" --prepare-ops-submodules "${empty_repo}/packages/paddlefleet_ops" \
    >"${tmp}/empty.out" 2>"${tmp}/empty.err"
empty_rc=$?
set -e
if [[ "${empty_rc}" -eq 0 ]]; then
    echo "FAIL: empty git tree was allowed to continue" >&2
    cat "${tmp}/empty.err" >&2
    exit 1
fi
if ! grep -Eq "missing .*\\.gitmodules|no gitmodules entry" "${tmp}/empty.err"; then
    echo "FAIL: empty git tree did not fail closed on missing gitmodules" >&2
    cat "${tmp}/empty.err" >&2
    exit 1
fi

# CPU mock: parent DeepGEMM .git is present but nested cutlass cannot be
# initialized (missing nested URL). Parent-only init would pass
# check_submodule_updated and then fail at compile; prepare must refuse
# before any uv build.
nested="${tmp}/nested.git"
deepgemm_url="${tmp}/deepgemm.git"
mkdir -p "${deepgemm_url}"
git -C "${deepgemm_url}" init -q
git -C "${deepgemm_url}" config user.email "fixture@example.invalid"
git -C "${deepgemm_url}" config user.name "fixture"
mkdir -p "${deepgemm_url}/deep_gemm"
echo parent >"${deepgemm_url}/deep_gemm/README"
git -C "${deepgemm_url}" add deep_gemm
git -C "${deepgemm_url}" commit -qm "deepgemm parent"
printf '[submodule "third-party/cutlass"]\n\tpath = third-party/cutlass\n\turl = %s/missing-cutlass.git\n' "${tmp}" \
    >"${deepgemm_url}/.gitmodules"
git -C "${deepgemm_url}" add .gitmodules
git -C "${deepgemm_url}" update-index --add --cacheinfo \
    160000,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,third-party/cutlass
git -C "${deepgemm_url}" commit -qm "nested cutlass gitlink to a missing url"
if ! git -C "${deepgemm_url}" ls-tree HEAD -- third-party/cutlass | grep -q '^160000'; then
    echo "FAIL: DeepGEMM fixture did not record nested cutlass gitlink" >&2
    git -C "${deepgemm_url}" ls-tree HEAD >&2
    exit 1
fi

mkdir -p "${nested}"
git -C "${nested}" init -q
git -C "${nested}" config user.email "fixture@example.invalid"
git -C "${nested}" config user.name "fixture"
mkdir -p "${nested}/packages/paddlefleet_ops/third_party"
{
    printf '[submodule "third_party/DeepGEMM"]\n\tpath = packages/paddlefleet_ops/third_party/DeepGEMM\n\turl = %s\n' "${deepgemm_url}"
    for name in DeepEP HybridEP quack sonic-moe flash-attention flash-linear-attention FlashMLA fast-hadamard-transform; do
        printf '[submodule "third_party/%s"]\n\tpath = packages/paddlefleet_ops/third_party/%s\n\turl = %s/%s.git\n' \
            "${name}" "${name}" "${tmp}" "${name}"
    done
} >"${nested}/.gitmodules"
git -C "${nested}" add .gitmodules
git -C "${nested}" commit -qm "gitmodules"
GIT_ALLOW_PROTOCOL=file git -C "${nested}" submodule add --quiet "${deepgemm_url}" \
    packages/paddlefleet_ops/third_party/DeepGEMM
# Drop any nested checkout submodule add may have attempted; parent .git stays.
rm -rf "${nested}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass"
mkdir -p "${nested}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass"
git -C "${nested}" add packages .gitmodules
git -C "${nested}" commit -qm "superproject with DeepGEMM parent only"
if [[ -e "${nested}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass/.git" ]]; then
    echo "FAIL: nested fixture still has cutlass .git" >&2
    exit 1
fi
if [[ ! -e "${nested}/packages/paddlefleet_ops/third_party/DeepGEMM/.git" ]]; then
    echo "FAIL: nested fixture lost parent DeepGEMM .git" >&2
    exit 1
fi

set +e
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always \
    GIT_ALLOW_PROTOCOL=file \
    NVIDIA_OPS_BUILD_SUBMODULES_OVERRIDE=DeepGEMM \
    bash "${SETUP}" --prepare-ops-submodules "${nested}/packages/paddlefleet_ops" \
    >"${tmp}/nested.out" 2>"${tmp}/nested.err"
nested_rc=$?
set -e
if [[ "${nested_rc}" -eq 0 ]]; then
    echo "FAIL: parent-only DeepGEMM was allowed to continue into build" >&2
    cat "${tmp}/nested.err" >&2
    exit 1
fi
if ! grep -Eq "nested DeepGEMM/third-party/cutlass|recursive failed" "${tmp}/nested.err"; then
    echo "FAIL: nested cutlass miss did not fail closed" >&2
    cat "${tmp}/nested.err" >&2
    exit 1
fi
if grep -q "Building paddlefleet-ops" "${tmp}/nested.out" "${tmp}/nested.err"; then
    echo "FAIL: nested miss continued into a build" >&2
    exit 1
fi

# CPU mock: a legal source tree with a recorded nested gitlink that is
# actually checked out must pass prepare (no uv / compile).
ok="${tmp}/ok.git"
cutlass_url="${tmp}/cutlass.git"
mkdir -p "${cutlass_url}/include/cutlass"
echo cutlass >"${cutlass_url}/include/cutlass/version.h"
git -C "${cutlass_url}" init -q
git -C "${cutlass_url}" config user.email "fixture@example.invalid"
git -C "${cutlass_url}" config user.name "fixture"
git -C "${cutlass_url}" add include
git -C "${cutlass_url}" commit -qm "cutlass headers"
cutlass_sha="$(git -C "${cutlass_url}" rev-parse HEAD)"

deepgemm_ok="${tmp}/deepgemm-ok.git"
mkdir -p "${deepgemm_ok}/deep_gemm"
echo parent >"${deepgemm_ok}/deep_gemm/README"
git -C "${deepgemm_ok}" init -q
git -C "${deepgemm_ok}" config user.email "fixture@example.invalid"
git -C "${deepgemm_ok}" config user.name "fixture"
git -C "${deepgemm_ok}" add deep_gemm
git -C "${deepgemm_ok}" commit -qm "deepgemm parent"
GIT_ALLOW_PROTOCOL=file git -C "${deepgemm_ok}" submodule add --quiet "${cutlass_url}" third-party/cutlass
git -C "${deepgemm_ok}" commit -qm "nested cutlass gitlink"
if ! git -C "${deepgemm_ok}" ls-tree HEAD -- third-party/cutlass | grep -q '^160000'; then
    echo "FAIL: success fixture DeepGEMM did not record nested cutlass gitlink" >&2
    git -C "${deepgemm_ok}" ls-tree HEAD >&2
    exit 1
fi
recorded_cutlass="$(git -C "${deepgemm_ok}" ls-tree HEAD -- third-party/cutlass | awk '{print $3}')"
if [[ "${recorded_cutlass}" != "${cutlass_sha}" ]]; then
    echo "FAIL: success fixture cutlass gitlink ${recorded_cutlass} != ${cutlass_sha}" >&2
    exit 1
fi

mkdir -p "${ok}"
git -C "${ok}" init -q
git -C "${ok}" config user.email "fixture@example.invalid"
git -C "${ok}" config user.name "fixture"
printf '[submodule "third_party/DeepGEMM"]\n\tpath = packages/paddlefleet_ops/third_party/DeepGEMM\n\turl = %s\n' \
    "${deepgemm_ok}" >"${ok}/.gitmodules"
git -C "${ok}" add .gitmodules
git -C "${ok}" commit -qm "gitmodules"
GIT_ALLOW_PROTOCOL=file git -C "${ok}" submodule add --quiet "${deepgemm_ok}" \
    packages/paddlefleet_ops/third_party/DeepGEMM
git -C "${ok}" add packages .gitmodules
git -C "${ok}" commit -qm "superproject with DeepGEMM and nested cutlass"
# Leave nested cutlass uninitialized so prepare must fetch it recursively.
rm -rf "${ok}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass"
mkdir -p "${ok}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass"
if [[ -e "${ok}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass/.git" ]]; then
    echo "FAIL: success fixture started with nested cutlass already present" >&2
    exit 1
fi

set +e
# Local file:// remotes are a fixture, not a production transport.
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always \
    GIT_ALLOW_PROTOCOL=file \
    NVIDIA_OPS_BUILD_SUBMODULES_OVERRIDE=DeepGEMM \
    bash "${SETUP}" --prepare-ops-submodules "${ok}/packages/paddlefleet_ops" \
    >"${tmp}/ok.out" 2>"${tmp}/ok.err"
ok_rc=$?
set -e
if [[ "${ok_rc}" -ne 0 ]]; then
    echo "FAIL: legal source tree did not pass prepare" >&2
    cat "${tmp}/ok.out" >&2
    cat "${tmp}/ok.err" >&2
    exit 1
fi
if [[ ! -e "${ok}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass/.git" ]]; then
    echo "FAIL: legal prepare did not check out nested cutlass" >&2
    cat "${tmp}/ok.out" >&2
    exit 1
fi
actual_cutlass="$(git -C "${ok}/packages/paddlefleet_ops/third_party/DeepGEMM/third-party/cutlass" rev-parse HEAD)"
if [[ "${actual_cutlass}" != "${cutlass_sha}" ]]; then
    echo "FAIL: nested cutlass HEAD ${actual_cutlass} != pin ${cutlass_sha}" >&2
    exit 1
fi
if ! grep -q "third-party/cutlass @ ${cutlass_sha}" "${tmp}/ok.out"; then
    echo "FAIL: success prepare did not report nested gitlink SHA" >&2
    cat "${tmp}/ok.out" >&2
    exit 1
fi
if grep -q "Building paddlefleet-ops" "${tmp}/ok.out" "${tmp}/ok.err"; then
    echo "FAIL: success prepare continued into a build" >&2
    exit 1
fi

# Default required set from get_libs() AST: top-level parent of each
# source_rel_path under third_party/, excluding the ENABLE_MOONEP branch.
# Do not hand-copy NVIDIA_OPS_BUILD_SUBMODULES. MoonEP stays opt-in.
BUILD_UTILS="${SCRIPT_DIR}/../../packages/paddlefleet_ops/build_utils.py"
expected="$(python3 - "${BUILD_UTILS}" <<'PY'
import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))


class Collect(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: list[str] = []
        self._skip = 0

    def visit_If(self, node: ast.If) -> None:
        skip = "ENABLE_MOONEP" in ast.dump(node.test)
        if skip:
            self._skip += 1
            for child in node.body:
                self.visit(child)
            self._skip -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_lib = isinstance(func, ast.Name) and func.id == "EcosystemLibrary"
        if is_lib and self._skip == 0:
            for kw in node.keywords:
                if kw.arg == "source_rel_path" and isinstance(kw.value, ast.Constant):
                    parts = str(kw.value.value).split("/")
                    if len(parts) >= 2 and parts[0] == "third_party":
                        self.parents.append(parts[1])
        self.generic_visit(node)


for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "get_libs":
        visitor = Collect()
        visitor.visit(node)
        seen: set[str] = set()
        for parent in visitor.parents:
            if parent not in seen:
                seen.add(parent)
                print(parent)
PY
)"
if [[ -z "${expected}" ]]; then
    echo "FAIL: get_libs AST produced no third_party parents" >&2
    exit 1
fi
if grep -Fxq MoonEP <<<"${expected}"; then
    echo "FAIL: AST extract included opt-in MoonEP" >&2
    exit 1
fi
if ! grep -Fxq cudnn-frontend <<<"${expected}"; then
    echo "FAIL: get_libs AST missing cudnn-frontend parent" >&2
    echo "${expected}" >&2
    exit 1
fi
listed="$(awk '
    $0 ~ /^NVIDIA_OPS_BUILD_SUBMODULES=\(/ {inlist=1; next}
    inlist && $0 ~ /^\)/ {exit}
    inlist {gsub(/[[:space:]]/,""); if ($0!="") print}
' "${SETUP}")"
if [[ "$(sort <<<"${listed}")" != "$(sort <<<"${expected}")" ]]; then
    echo "FAIL: NVIDIA_OPS_BUILD_SUBMODULES != get_libs AST parents" >&2
    echo "listed:" >&2
    echo "${listed}" >&2
    echo "expected:" >&2
    echo "${expected}" >&2
    exit 1
fi

git_file() {
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always \
        GIT_ALLOW_PROTOCOL=file "$@"
}

# One empty legal child repo reused as every default parent gitlink.
leaf="${tmp}/leaf.git"
mkdir -p "${leaf}"
git -C "${leaf}" init -q
git -C "${leaf}" config user.email "fixture@example.invalid"
git -C "${leaf}" config user.name "fixture"
echo leaf >"${leaf}/README"
git -C "${leaf}" add README
git -C "${leaf}" commit -qm "legal empty parent"
leaf_sha="$(git -C "${leaf}" rev-parse HEAD)"

record_parents() {
    local super="$1"
    shift
    local name
    : >"${super}/.gitmodules"
    mkdir -p "${super}/packages/paddlefleet_ops"
    echo ops >"${super}/packages/paddlefleet_ops/README"
    git -C "${super}" add packages/paddlefleet_ops/README
    for name in "$@"; do
        printf '[submodule "third_party/%s"]\n\tpath = packages/paddlefleet_ops/third_party/%s\n\turl = %s\n' \
            "${name}" "${name}" "${leaf}" >>"${super}/.gitmodules"
        git -C "${super}" update-index --add --cacheinfo \
            160000,"${leaf_sha}",packages/paddlefleet_ops/third_party/"${name}"
    done
    git -C "${super}" add .gitmodules
    git -C "${super}" commit -qm "default parents"
    for name in "$@"; do
        if ! git -C "${super}" ls-tree HEAD -- "packages/paddlefleet_ops/third_party/${name}" | grep -q '^160000'; then
            echo "FAIL: missing mode 160000 gitlink for ${name}" >&2
            git -C "${super}" ls-tree HEAD >&2
            exit 1
        fi
    done
}

mapfile -t expected_names <<<"${expected}"
without_cudnn=()
for name in "${expected_names[@]}"; do
    if [[ "${name}" != "cudnn-frontend" ]]; then
        without_cudnn+=("${name}")
    fi
done

# Fail: every default parent except cudnn-frontend is a real gitlink.
# Default prepare (no OVERRIDE) must refuse on cudnn-frontend, not DeepGEMM.
missing="${tmp}/missing-cudnn.git"
mkdir -p "${missing}"
git -C "${missing}" init -q
git -C "${missing}" config user.email "fixture@example.invalid"
git -C "${missing}" config user.name "fixture"
git -C "${missing}" commit --allow-empty -qm "init"
record_parents "${missing}" "${without_cudnn[@]}"
set +e
unset NVIDIA_OPS_BUILD_SUBMODULES_OVERRIDE || true
git_file bash "${SETUP}" --prepare-ops-submodules "${missing}/packages/paddlefleet_ops" \
    >"${tmp}/missing.out" 2>"${tmp}/missing.err"
missing_rc=$?
set -e
if [[ "${missing_rc}" -eq 0 ]]; then
    echo "FAIL: default list skipped cudnn-frontend and continued" >&2
    cat "${tmp}/missing.err" >&2
    exit 1
fi
if ! grep -q "cudnn-frontend" "${tmp}/missing.err"; then
    echo "FAIL: default prepare did not fail on cudnn-frontend" >&2
    cat "${tmp}/missing.err" >&2
    exit 1
fi
if grep -q "Building paddlefleet-ops" "${tmp}/missing.out" "${tmp}/missing.err"; then
    echo "FAIL: missing cudnn-frontend continued into a build" >&2
    exit 1
fi

# Success: every default parent including cudnn-frontend is a real gitlink.
# Default prepare (no OVERRIDE) must init them and not build.
complete="${tmp}/complete.git"
mkdir -p "${complete}"
git -C "${complete}" init -q
git -C "${complete}" config user.email "fixture@example.invalid"
git -C "${complete}" config user.name "fixture"
git -C "${complete}" commit --allow-empty -qm "init"
record_parents "${complete}" "${expected_names[@]}"
set +e
unset NVIDIA_OPS_BUILD_SUBMODULES_OVERRIDE || true
git_file bash "${SETUP}" --prepare-ops-submodules "${complete}/packages/paddlefleet_ops" \
    >"${tmp}/complete.out" 2>"${tmp}/complete.err"
complete_rc=$?
set -e
if [[ "${complete_rc}" -ne 0 ]]; then
    echo "FAIL: default prepare of the full AST parent set failed" >&2
    cat "${tmp}/complete.out" >&2
    cat "${tmp}/complete.err" >&2
    exit 1
fi
for name in "${expected_names[@]}"; do
    if [[ ! -e "${complete}/packages/paddlefleet_ops/third_party/${name}/.git" ]]; then
        echo "FAIL: default prepare did not check out ${name}" >&2
        cat "${tmp}/complete.out" >&2
        exit 1
    fi
    actual="$(git -C "${complete}/packages/paddlefleet_ops/third_party/${name}" rev-parse HEAD)"
    if [[ "${actual}" != "${leaf_sha}" ]]; then
        echo "FAIL: ${name} HEAD ${actual} != leaf pin ${leaf_sha}" >&2
        exit 1
    fi
done
if grep -q "Building paddlefleet-ops" "${tmp}/complete.out" "${tmp}/complete.err"; then
    echo "FAIL: default success prepare continued into a build" >&2
    exit 1
fi

echo "setup_venvs ops-install argv PASS (dir isolated-off, wheel/url unchanged)"
echo "setup_venvs ops-submodule prepare PASS (bare/empty/nested fail closed; legal nested gitlink prepares; default AST list fail/success on cudnn-frontend)"
