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

echo "setup_venvs ops-install argv PASS (dir isolated-off, wheel/url unchanged)"
echo "setup_venvs ops-submodule prepare PASS (bare/empty/nested fail closed; legal nested gitlink prepares)"
