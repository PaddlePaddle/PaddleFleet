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

# Fleet CI accuracy jobs use docker --network host. A hardcoded
# MASTER_PORT (29500) can already be taken by another job on the
# same runner (EADDRINUSE). Bind an ephemeral localhost port instead.
_py="python3"
if command -v python >/dev/null 2>&1; then
    _py="python"
fi
MASTER_PORT="$("${_py}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
export MASTER_PORT
echo "[align] MASTER_PORT=${MASTER_PORT}"
