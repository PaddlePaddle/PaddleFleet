#!/usr/bin/env bash
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
pip install --force-reinstall --no-deps "${PADDLE_WHEEL}" -i "${PADDLE_EXTRA_INDEX_URL}"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install triton ninja packaging einops
pip install -r eval/HELMET/requirements.txt
pip install tokenizers protobuf dill pyyaml nltk pandas tabulate tiktoken torchcodec pytest
pip install -e .
