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

# Only for fresh CI sources, before installation/build can modify submodules.
# Interrupted cloning can leave HEAD at the gitlink but the worktree empty;
# --force ensures a retry materializes that pinned checkout, including children.
set -euo pipefail
for attempt in 1 2 3; do
    echo "Ops submodule preparation: attempt ${attempt}/3"
    if timeout --kill-after=30s 15m git submodule update --init --recursive --force; then
        exit 0
    else
        status=$?
    fi
    echo "Ops submodule preparation failed with exit ${status}" >&2
    if [[ $attempt -eq 3 ]]; then
        exit "$status"
    fi
    sleep "$((attempt * 15))"
done
