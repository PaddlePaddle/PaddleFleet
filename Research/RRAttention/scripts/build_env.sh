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

PADDLEFORMERS_DIR=${PADDLEFORMERS_DIR:-.deps/PaddleFormers}
PADDLEFORMERS_COMMIT=${PADDLEFORMERS_COMMIT:-12de383451cc795830de01032a3dd36a9af71796}
PADDLE_EXTRA_INDEX_URL=${PADDLE_EXTRA_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/nightly/cu129/}
PADDLE_WHEEL=${PADDLE_WHEEL:-paddlepaddle-gpu==3.4.0.post20260417+616f7c19d12}

rm -rf "${PADDLEFORMERS_DIR}"
mkdir -p "$(dirname "${PADDLEFORMERS_DIR}")"
git clone https://github.com/PaddlePaddle/PaddleFormers.git "${PADDLEFORMERS_DIR}"
git -C "${PADDLEFORMERS_DIR}" checkout "${PADDLEFORMERS_COMMIT}"

pip install -e "${PADDLEFORMERS_DIR}[paddlefleet]" --extra-index-url "${PADDLE_EXTRA_INDEX_URL}"

# we recommend installing the latest paddlepaddle-gpu now
pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu129/ --force-reinstall
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install triton ninja packaging einops
pip install -r eval/HELMET/requirements.txt
pip install tokenizers protobuf dill pyyaml nltk pandas tabulate tiktoken torchcodec pytest
pip install -e .
