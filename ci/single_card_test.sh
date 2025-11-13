# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

disable_file="$work_dir/tests/single_card_tests/disable_single_card_uts.txt"

disabled=()
if [ -f "$disable_file" ]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        filename=$(basename "$line")
        disabled+=("$filename")
    done < "$disable_file"
fi

echo -e "\033[34mDisabled tests:\033[0m ${disabled[@]}"

uv run pytest tests/single_card_tests $(sed 's/^/--ignore=/' tests/single_card_tests/disable_single_card_uts.txt)
