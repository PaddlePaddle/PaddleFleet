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

ILE="paddleformers/utils/import_utils.py"
LINE=15
TEXT="import importlib.metadata"

# 已存在就跳过
if grep -q "^${TEXT}$" "$FILE"; then
  echo "Already patched, skip"
  exit 0
fi

# 插入到第 15 行后
awk -v line="$LINE" -v text="$TEXT" '
NR==line { print; print text; next }
{ print }
' "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
