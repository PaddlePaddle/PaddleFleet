#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HELMET_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${HELMET_DIR}"
wget -c https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz
tar -xvzf data.tar.gz

mkdir -p models
hf download gaotianyu1350/roberta-large-squad --local-dir models/roberta-large-squad
hf download google/t5_xxl_true_nli_mixture --local-dir models/t5_xxl_true_nli_mixture
