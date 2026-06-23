<div align="center">

# RRAttention

### Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference

[![Paper](https://img.shields.io/badge/ACL%202026-Paper-red)](https://arxiv.org/abs/2602.05853)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../../LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-yellow)](https://www.python.org/)
[![PaddlePaddle](https://img.shields.io/badge/Paddle-nightly%203.4-orange)](https://www.paddlepaddle.org.cn/)

</div>

---

## Overview

The quadratic complexity of attention mechanisms poses a critical bottleneck for large language models (LLMs) processing long contexts. While dynamic sparse attention methods offer input-adaptive efficiency, they face fundamental trade-offs: requiring preprocessing, lacking global evaluation, violating query independence, or incurring high computational overhead.

**RRAttention** is a novel dynamic sparse attention method that simultaneously achieves all desirable properties through a **head round-robin (RR) sampling strategy**. By rotating query sampling positions across attention heads within each stride, RRAttention maintains query independence while enabling efficient global pattern discovery with stride-level aggregation.

## 🔔 News

- **[2026.04.19]** Paddle open-source code, HELMET evaluation entry, and release scripts are now available in this repository.
- **[2026.04.06]** 🎉 Our paper has been accepted by **ACL 2026**!


## 📌 TODO

- [x] Release PaddlePaddle evaluation code
- [ ] Release PyTorch evaluation code

## Quick Start

### Build Environment

RRAttention currently requires the latest PaddlePaddle GPU nightly. The helper script below installs the tested environment.

```bash
cd Research/RRAttention
uv venv --python 3.10 --seed .venv
source .venv/bin/activate
bash ./scripts/build_env.sh
```

### Download HELMET Data

```bash
bash ./eval/HELMET/scripts/download_data.sh
```

This also downloads the local ALCE judge models to `eval/HELMET/models/roberta-large-squad` and `eval/HELMET/models/t5_xxl_true_nli_mixture`.

### Prepare Checkpoints

Hugging Face safetensors checkpoints are supported, but for repeated HELMET or speed benchmarks we recommend converting them once to Paddle checkpoints for better loading efficiency and reproducibility.

```bash
python scripts/convert_hf_to_paddle.py \
  --input /path/to/hf-checkpoint \
  --output /path/to/paddle-checkpoint
```

### Python API

`rrattn_prefill` expects query/key/value tensors in `[batch, heads, seq_len, head_dim]` layout:

```python
from rrattn import rrattn_prefill

attn_output, sparse_ratio = rrattn_prefill(
    query_states,
    key_states,
    value_states,
    threshold=0.95,
    stride=8,
    block_size=128,
    use_triton=True,
)
```

For model-level use, patch a supported PaddleFormers model before generation:

```python
from rrattn import patch_llama_attention

model = patch_llama_attention(
    model,
    method="rrattn",
    threshold=0.95,
    stride=8,
    use_triton=True,
    keep_sink=True,
    keep_recent=True,
)
```

### Run HELMET Evaluation

The release script runs the default HELMET sweep for `rrattn` and full attention on the short configs and the 128k configs. Set `model_name_or_paths` to one or more local checkpoint paths before running it:

```bash
model_name_or_paths=/path/to/local-model-or-paddle-checkpoint \
data_root_dir=eval/HELMET \
qa_model_name_or_path=eval/HELMET/models/roberta-large-squad \
autoais_model_name_or_path=eval/HELMET/models/t5_xxl_true_nli_mixture \
bash ./scripts/run_helmet.sh
```

`model_name_or_path` must be a local checkpoint path: either a local Hugging Face safetensors directory or a converted Paddle checkpoint path. Remote Hugging Face repo IDs are not loaded directly; download or convert the model first.

For a single task/config, run HELMET directly:

```bash
cd eval/HELMET
python eval.py \
  --model_name_or_path /path/to/local-model-or-paddle-checkpoint \
  --data_root_dir . \
  --qa_model_name_or_path ./models/roberta-large-squad \
  --autoais_model_name_or_path ./models/t5_xxl_true_nli_mixture \
  --config configs/recall_short.yaml \
  --method rrattn \
  --threshold 0.95 \
  --stride 8 \
  --tag rrattn_0.95
```

### Run Speed Test

`scripts/speed_test.py` benchmarks the model prefill path on synthetic key-value retrieval prompts and records total, attention, and estimate time.

```bash
python scripts/speed_test.py \
  --model-name /path/to/local-model-or-paddle-checkpoint \
  --method rrattn \
  --threshold 0.95 \
  --stride 8 \
  --seq-lens 8000,32000,64000,128000 \
  --n-times 3 \
  --output-dir speed_results
```

`--model-name` must be a local checkpoint path: either a local Hugging Face safetensors directory or a converted Paddle checkpoint path. Remote Hugging Face repo IDs are not loaded directly; download or convert the model first. Supported methods are `rrattn`, `full`, `xattn`, and `flex`. If model auto-detection is not sufficient, pass `--model-type llama`, `--model-type qwen`, or `--model-type ernie`.

### Repository Layout

- `rrattn/`: Paddle RRAttention implementation and PaddleFormers patch entrypoints
- `eval/HELMET/`: HELMET evaluation code, configs, prompts, data/cache/output layout
- `scripts/`: environment setup and evaluation launch scripts
- `tests/`: public API import checks

## Method

<p align="center">
  <img src="./assets/rrattn.png" width="90%" alt="RRAttention Overview">
</p>

RRAttention achieves efficient dynamic sparse attention through three stages:

| Stage | Name | Description |
|:---:|---|---|
| **1** | **Query Sampling with Head Round-Robin** | For each stride of *S* tokens, one representative query is sampled per head. The sampling position rotates across heads via round-robin, ensuring all positions within a stride are eventually covered. |
| **2** | **Stride-level Importance Estimation** | Attention scores are aggregated at stride granularity through dimension reduction, reducing computational cost from O(L²) to O(L²/S²). |
| **3** | **Block-level Top-τ Selection** | Stride-level scores are aggregated to block level, and a top-τ strategy selects the minimal set of key blocks whose cumulative importance exceeds threshold τ. |

---

## Results

### :books: HELMET Benchmark (Natural Language Understanding)

RRAttention recovers **over 99%** of full attention performance while computing only half of the attention blocks.

| Model | Method | Avg. Sparsity | Avg. Score |
|---|---|:---:|:---:|
| Llama-3.1-8B | FullAttention | 0% | 56.42 |
| | FlexPrefill (γ=0.95) | 66.07% | 52.54 |
| | XAttention (τ=0.95) | 47.50% | 55.74 |
| | **RRAttention (τ=0.95)** | **48.68%** | **56.24** |
| Qwen2.5-7B | FullAttention | 0% | 47.68 |
| | FlexPrefill (γ=0.95) | 63.21% | 38.27 |
| | XAttention (τ=0.95) | 47.35% | 46.64 |
| | **RRAttention (τ=0.95)** | **47.56%** | **47.17** |

### :movie_camera: Video-MME Benchmark (Multimodal Video Comprehension)

| Method | Avg. Sparsity | Short | Medium | Long | Avg. |
|---|:---:|:---:|:---:|:---:|:---:|
| FullAttention | 0% | 72.90 | 63.60 | 55.20 | 63.90 |
| FlexPrefill (γ=0.95) | 46.70% | 72.80 | 62.00 | 54.00 | 62.90 |
| XAttention (τ=0.95) | 37.50% | 72.70 | 63.60 | 56.10 | 64.10 |
| **RRAttention (τ=0.95)** | **34.70%** | **72.60** | **64.00** | **56.20** | **64.30** |

---


## Hyperparameters

| Parameter | Default | Description |
|---|:---:|---|
| `block_size` (*B*) | 128 | Block size for sparse attention computation |
| `stride` (*S*) | 8 | Stride size for query sampling and importance estimation |
| `tau` (*τ*) | 0.95 | Cumulative importance threshold for Top-τ block selection |

> **Recommended:** Conservative `tau=0.95` recovers 99%+ accuracy (~48% sparsity). Aggressive `tau=0.90` gives stronger speedup (~61% sparsity) with minor accuracy trade-off.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{liu2026rrattention,
    title={RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference},
    author={Liu, Siran and Wang, Guoxia and Wang, Sa and Zeng, Jinle and Xie, HaoYang and Lou, Siyu and Yang, JiaBin and Yu, DianHai and Wang, Haifeng and Yang, Chao},
    booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](../../LICENSE) file for details.
