# HELMET Evaluation

This directory contains the HELMET evaluation code adapted for the RRAttention Paddle release.

Use the repository-level README for environment setup and evaluation launch:

```bash
bash ./scripts/build_env.sh
bash ./eval/HELMET/scripts/download_data.sh
bash ./scripts/run_helmet.sh
```

Only the local data download helper is retained under `eval/HELMET/scripts/`.
The upstream API, Slurm, vLLM/Gaudi, GPT-4 scoring, and result collection helper scripts are not part of this release.
