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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"

readonly PYTHON_VERSION="3.12"
readonly TORCH_VERSION="2.12.0+cu130"
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
readonly TE_VERSION="2.17.1"
readonly TRANSFORMERS_VERSION="4.57.1"
readonly PADDLE_INDEX_URL="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
readonly NIGHTLY_WHL_BASE="https://paddle-whl.bj.bcebos.com/nightly/cu130"
# readonly PADDLE_VERSION="xx"
# PaddleFleet will install default paddle"
readonly PADDLEFLEET_WHEEL="${PADDLEFLEET_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/paddlefleet/paddlefleet-0.4.0.dev20260807+d01517879a3-py3-none-any.whl}"
readonly PADDLEFLEET_OPS_WHEEL="${PADDLEFLEET_OPS_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/paddlefleet-ops/paddlefleet_ops-0.4.0.dev20260807+d0151787-cp312-cp312-linux_x86_64.whl}"
readonly PADDLEFORMERS_WHEEL="${NIGHTLY_WHL_BASE}/paddleformers/paddleformers-0.0.0.dev-py3-none-any.whl"
readonly MEGATRON_CORE_WHEEL="${MEGATRON_CORE_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/megatron_core-0.19.0+f2706b6f3-cp312-cp312-linux_x86_64.whl}"
readonly MS_SWIFT_WHEEL="${MS_SWIFT_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/ms_swift-4.5.0.dev0-py3-none-any.whl}"
readonly MCORE_BRIDGE_WHEEL="${MCORE_BRIDGE_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/mcore_bridge-1.7.0.dev0-py3-none-any.whl}"
readonly NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com,baidu-int.com,.baidu.com,.bcebos.com"
# readonly PROXY_URL="set your proxy"
readonly UV_BIN_DIR="/home/.local/bin"
readonly UV_CACHE_DIR_PATH="/home/.cache/uv"

usage() {
    cat <<'EOF'
Usage: setup_venvs.sh

Create or reuse:
  - venv/torch   (torch + Megatron-LM + ms-swift)
  - venv/paddle  (paddlepaddle-gpu + PaddleFleet + PaddleFormers)

Both venvs are created next to this script, and the four sibling repositories
are installed editable from the same directory.
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[setup_venvs] missing required command: $1" >&2
        exit 1
    fi
}

# Populate OPS_UV_KIND and OPS_UV_ARGS for `uv pip install`.
# dir  → --no-build-isolation so setup.py can `import paddle` from this venv.
# file → local wheel; same argv as before (no --no-build-isolation).
# url  → default nightly / remote wheel; same argv as file.
collect_ops_uv_pip_args() {
    local paddle_py="$1"
    local ops_path="$2"
    OPS_UV_ARGS=(--python "${paddle_py}" --force-reinstall)
    if [[ -d "${ops_path}" ]]; then
        OPS_UV_KIND="dir"
        OPS_UV_ARGS+=(--no-build-isolation "${ops_path}")
        return 0
    fi
    if [[ -f "${ops_path}" ]]; then
        OPS_UV_KIND="wheel"
        OPS_UV_ARGS+=("${ops_path}")
        return 0
    fi
    OPS_UV_KIND="url"
    OPS_UV_ARGS+=("${ops_path}")
    return 0
}

# NVIDIA source-tree ops build required gitlinks. This is the union of
# check_submodule_updated() and get_libs() EcosystemLibrary source_rel_path
# parents under third_party/ (build_utils.py). The guard list is a subset:
# Python >= 3.12 always registers name="cudnn" at third_party/cudnn-frontend
# and pip-installs that directory; missing setup.py/pyproject.toml there
# fails the wheel (Swift 34035298069). MoonEP stays opt-in via ENABLE_MOONEP=1.
NVIDIA_OPS_BUILD_SUBMODULES=(
    DeepGEMM
    DeepEP
    HybridEP
    quack
    sonic-moe
    flash-attention
    flash-linear-attention
    FlashMLA
    fast-hadamard-transform
    cudnn-frontend
)

_gitlink_sha() {
    local repo="$1"
    local path="$2"
    git -C "${repo}" ls-tree HEAD -- "${path}" | awk '{print $3}'
}

# Walk every mode-160000 gitlink recorded at HEAD, then recurse. Include
# paths and vendored trees are not gitlinks and are not required here.
# A recorded gitlink whose worktree has no .git, or whose HEAD differs
# from the recorded SHA, is a hard error — never rewrite the pin.
_verify_recorded_gitlinks() {
    local repo="$1"
    local prefix="$2"
    local meta path mode recorded actual
    while IFS=$'\t' read -r meta path; do
        [[ -n "${meta}" && -n "${path}" ]] || continue
        mode="${meta%% *}"
        [[ "${mode}" == "160000" ]] || continue
        recorded="$(printf '%s\n' "${meta}" | awk '{print $3}')"
        if [[ ! "${recorded}" =~ ^[0-9a-fA-F]{40}$ ]]; then
            echo "[setup_venvs] ${prefix}${path} gitlink SHA is unusable; refuse source-tree paddlefleet-ops build" >&2
            return 1
        fi
        if [[ ! -e "${repo}/${path}/.git" ]]; then
            echo "[setup_venvs] nested ${prefix}${path} still has no .git after recursive init; refuse source-tree paddlefleet-ops build" >&2
            return 1
        fi
        actual="$(git -C "${repo}/${path}" rev-parse HEAD)"
        if [[ "${actual}" != "${recorded}" ]]; then
            echo "[setup_venvs] nested ${prefix}${path} HEAD ${actual} != recorded gitlink ${recorded}; refuse to rewrite the pin" >&2
            return 1
        fi
        echo "[setup_venvs] ${prefix}${path} @ ${recorded}"
        _verify_recorded_gitlinks "${repo}/${path}" "${prefix}${path}/"
    done < <(git -C "${repo}" ls-tree -r HEAD)
}

# Initialize the NVIDIA build-required gitlinks at the commits recorded in
# the superproject (the source pin), including nested gitlinks required by
# the NVIDIA compile. Does not rewrite gitlinks. Missing git, missing
# .gitmodules, or an uninitialized nested path after update is a hard
# error — never continue into uv build.
prepare_ops_build_submodules() {
    local ops_path="$1"
    local repo_root
    local name
    local rel
    local recorded
    local actual
    local -a names
    if [[ -n "${NVIDIA_OPS_BUILD_SUBMODULES_OVERRIDE:-}" ]]; then
        # CPU fixtures name a subset. Production leaves this unset.
        read -r -a names <<<"${NVIDIA_OPS_BUILD_SUBMODULES_OVERRIDE}"
    else
        names=("${NVIDIA_OPS_BUILD_SUBMODULES[@]}")
        if [[ "${ENABLE_MOONEP:-0}" == "1" ]]; then
            names+=("MoonEP")
        fi
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "[setup_venvs] git is required to initialize paddlefleet_ops submodules" >&2
        return 1
    fi
    repo_root="$(git -C "${ops_path}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -z "${repo_root}" ]]; then
        echo "[setup_venvs] ${ops_path} is not inside a git work tree; refuse source-tree paddlefleet-ops build" >&2
        return 1
    fi
    if [[ ! -f "${repo_root}/.gitmodules" ]]; then
        echo "[setup_venvs] missing ${repo_root}/.gitmodules; refuse source-tree paddlefleet-ops build" >&2
        return 1
    fi

    echo "[setup_venvs] initializing paddlefleet_ops build submodules at recorded gitlinks"
    for name in "${names[@]}"; do
        rel="packages/paddlefleet_ops/third_party/${name}"
        if ! git -C "${repo_root}" config -f .gitmodules --get-regexp '^submodule\..*\.path$' \
            | awk '{print $2}' | grep -Fxq "${rel}"; then
            echo "[setup_venvs] no gitmodules entry for ${rel}; refuse source-tree paddlefleet-ops build" >&2
            return 1
        fi
        recorded="$(_gitlink_sha "${repo_root}" "${rel}")"
        if [[ ! "${recorded}" =~ ^[0-9a-fA-F]{40}$ ]]; then
            echo "[setup_venvs] no recorded gitlink for ${rel} at HEAD; refuse source-tree paddlefleet-ops build" >&2
            return 1
        fi
        # Recursive so nested cutlass (DeepGEMM, flash-attention/flashmask)
        # is present before check_submodule_updated only looks at parent .git.
        # No --depth: a shallow fetch of the default branch can miss the
        # gitlink commit recorded in the superproject pin.
        if ! git -C "${repo_root}" submodule update --init --recursive -- "${rel}"; then
            echo "[setup_venvs] git submodule update --init --recursive failed for ${rel} (recorded ${recorded})" >&2
            return 1
        fi
        if [[ ! -e "${repo_root}/${rel}/.git" ]]; then
            echo "[setup_venvs] ${rel} still has no .git after init; refuse source-tree paddlefleet-ops build" >&2
            return 1
        fi
        actual="$(git -C "${repo_root}/${rel}" rev-parse HEAD)"
        if [[ "${actual}" != "${recorded}" ]]; then
            echo "[setup_venvs] ${rel} HEAD ${actual} != recorded gitlink ${recorded}; refuse to rewrite the pin" >&2
            return 1
        fi
        echo "[setup_venvs] ${rel} @ ${recorded}"
        _verify_recorded_gitlinks "${repo_root}/${rel}" "${rel}/"
    done
}

setup_proxy() {
    if [[ -z "${PROXY_URL:-}" ]]; then
        echo "[setup_venvs] warning: PROXY_URL is not set, continuing without a proxy." >&2
        echo "  export PROXY_URL=http://<proxy-host>:<proxy-port> to use one." >&2
        return
    fi

    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export no_proxy="${NO_PROXY_LIST}"
    export HTTP_PROXY="${http_proxy}"
    export HTTPS_PROXY="${https_proxy}"
    export NO_PROXY="${no_proxy}"
}

setup_cache() {
    export UV_CACHE_DIR="${UV_CACHE_DIR_PATH}"
    mkdir -p "${UV_CACHE_DIR}"
}

ensure_venv() {
    local venv_dir="$1"

    if [[ -d "${venv_dir}" ]]; then
        echo "[setup_venvs] reusing ${venv_dir}"
        return
    fi

    uv venv --relocatable --seed -p "${PYTHON_VERSION}" "${venv_dir}"
}

setup_torch_venv() {
    local torch_py="$1"

    echo "[setup_venvs] torch python : ${torch_py}"
    uv pip install --python "${torch_py}" --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}"

    uv pip install --python "${torch_py}" \
        "setuptools>=66.1.0" pip wheel packaging cmake "ninja==1.11.1.1" \
        "pybind11[global]>=2.13,<3" Pillow

    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${torch_py}" --index-strategy unsafe-best-match \
        --force-reinstall --no-deps \
        "${MEGATRON_CORE_WHEEL}" "${MS_SWIFT_WHEEL}" "${MCORE_BRIDGE_WHEEL}"
    # uv pip install --python "${torch_py}" --index-strategy unsafe-best-match \
    #     -e ./ms-swift -e ./Megatron-LM -e ./mcore-bridge

    uv pip install --python "${torch_py}" --index-strategy unsafe-best-match \
        omegaconf tensor-spec-worker datasets transformers_stream_generator tensorboard json_repair matplotlib modelscope peft \
        "transformers==${TRANSFORMERS_VERSION}" \
        "transformer-engine[core_cu13]==${TE_VERSION}"

    # transformer_engine_torch
    uv cache clean transformer-engine-torch
    NVTE_FRAMEWORK=pytorch NVTE_PYTORCH_FORCE_BUILD=TRUE \
        uv pip install --python "${torch_py}" \
        --index-strategy unsafe-best-match --no-build-isolation \
        --no-binary transformer-engine-torch \
        --reinstall-package transformer-engine-torch \
        "transformer_engine_torch==${TE_VERSION}"
}

setup_paddle_venv() {
    local paddle_py="$1"

    echo "[setup_venvs] paddle python: ${paddle_py}"
    local -a paddle_index=(
        --no-config
        --index-url "${PADDLE_INDEX_URL}"
        --extra-index-url https://pypi.org/simple/
        --index-strategy unsafe-best-match
    )

    # uv pip install --python "${paddle_py}" "${paddle_index[@]}" \
    #     "${PADDLE_VERSION}" --force-reinstall

    # Build-time deps must live in the venv so `uv sync --no-build-isolation`
    # can compile paddlefleet-ops against the paddle installed above.
    uv pip install --python "${paddle_py}" \
        "setuptools>=66.1.0" pip wheel packaging "ninja==1.11.1.1" \
        "pybind11[global]>=2.13,<3" "paddle-nvidia-nvshmem-cu13>=3.3.9,<3.5" \
        "tensor-spec-worker"

    # PaddleFleet. --no-deps is intentionally dropped: the wheel's pinned
    # paddlepaddle-gpu dependency must be installed here, otherwise
    # venv/paddle/bin/paddleformers-cli fails to import paddle at runtime.
    uv pip install --python "${paddle_py}" "${paddle_index[@]}" \
        --force-reinstall \
        "${PADDLEFLEET_WHEEL}"
    # (
    #     cd ./PaddleFleet
    #     git submodule update --init --recursive
    #     VIRTUAL_ENV="${WORKSPACE_DIR}/venv/paddle" \
    #         uv sync --python "${paddle_py}" --inexact --active --no-build-isolation -v \
    #             --index "paddlepaddle-gpu=${PADDLE_INDEX_URL}"
    # )

    # paddlefleet_ops: a prebuilt wheel (or URL) stays on the historical
    # isolated install. A source tree must compile against paddle already
    # in this venv — isolated uv builds fail with No module named paddle
    # (paired Mega 33968412986).
    collect_ops_uv_pip_args "${paddle_py}" "${PADDLEFLEET_OPS_WHEEL}"
    if [[ "${OPS_UV_KIND}" == "dir" ]]; then
        if ! "${paddle_py}" -c "import paddle" >/dev/null 2>&1; then
            echo "[setup_venvs] paddle not importable in ${paddle_py}; refuse source-tree paddlefleet-ops build" >&2
            exit 1
        fi
        prepare_ops_build_submodules "${PADDLEFLEET_OPS_WHEEL}"
        echo "[setup_venvs] paddlefleet_ops source tree ${PADDLEFLEET_OPS_WHEEL} (--no-build-isolation)"
    else
        echo "[setup_venvs] paddlefleet_ops ${OPS_UV_KIND} ${PADDLEFLEET_OPS_WHEEL}"
    fi
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install "${OPS_UV_ARGS[@]}"

    # PaddleFormers
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${paddle_py}" --force-reinstall \
        "${PADDLEFORMERS_WHEEL}"
    # uv pip install --python "${paddle_py}" -v -e ./PaddleFormers
}

print_installed_versions() {
    local torch_py="$1"
    local pkg line version

    echo "[setup_venvs] installed versions (venv/torch):"
    for pkg in megatron-core ms-swift mcore-bridge; do
        version=""
        while IFS= read -r line; do
            case "${line}" in
                "Version: "*) version="${line#Version: }" ;;
            esac
        done < <(uv pip show --python "${torch_py}" "${pkg}" 2>/dev/null)
        printf '  %-14s %s\n' "${pkg}" "${version:-not installed}"
    done
}

main() {
    if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
        usage
        exit 0
    fi
    if [[ ${1:-} == "--ops-install-argv" ]]; then
        collect_ops_uv_pip_args "${2:?python}" "${3:?ops-path}"
        printf 'KIND=%s\n' "${OPS_UV_KIND}"
        printf 'ARG:%s\n' "${OPS_UV_ARGS[@]}"
        exit 0
    fi
    if [[ ${1:-} == "--prepare-ops-submodules" ]]; then
        prepare_ops_build_submodules "${2:?ops-path}"
        exit $?
    fi

    setup_proxy
    setup_cache
    export UV_NO_PROGRESS=1
    export PATH="${UV_BIN_DIR}:${PATH}"
    require_command "uv"

    cd "${WORKSPACE_DIR}"
    uv python install "${PYTHON_VERSION}"
    uv tool install tensor-spec

    ensure_venv "venv/torch"
    ensure_venv "venv/paddle"

    setup_torch_venv "${WORKSPACE_DIR}/venv/torch/bin/python"
    setup_paddle_venv "${WORKSPACE_DIR}/venv/paddle/bin/python"

    print_installed_versions "${WORKSPACE_DIR}/venv/torch/bin/python"
}

main "$@"
