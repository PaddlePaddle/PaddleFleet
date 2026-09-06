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

echo "setup_venvs ops-install argv PASS (dir isolated-off, wheel/url unchanged)"
