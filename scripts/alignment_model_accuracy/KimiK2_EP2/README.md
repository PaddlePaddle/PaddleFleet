# KimiK2_EP2 对齐用例

Kimi-K2 在 **PaddleFormers+PaddleFleet ↔ ms-swift+Megatron-LM** 下的精度对齐监控用例，
接入 `scripts/alignment_model_accuracy/` 框架（见上级 `README.md`）。
配置约定对标同目录 `GLM45Air_EP2`（TP=1/EP=2/PP=1，单机 2 卡）。

## 文件

| 文件 | 说明 |
|---|---|
| `KimiK2.yaml` | paddle 侧训练配置（TP=1/EP=2/PP=1，SFT，`use_accuracy_compatible: true`，单 step）|
| `run_paddle_kimik2.sh` | paddle 侧启动：清分布式 env + 对齐 flags + `paddleformers-cli train` |
| `run_torch_kimik2.sh` | torch 侧启动：对齐 flags + `megatron sft <ARGS>` |

## 注册到测试入口

`scripts/alignment_model_accuracy/run_alignment_test.sh` 的 `CASES` 数组中本用例
**当前是注释状态**，待首次 smoke 确立 step1 md5 基线后再放开：

```bash
CASES=(
    "MinimaxV2.5_EP2 ./MinimaxV2.5_EP2/run_paddle_minimax.sh ./MinimaxV2.5_EP2/run_torch_minimax.sh"
    "GLM45Air_EP2 ./GLM45Air_EP2/run_paddle_glm45.sh ./GLM45Air_EP2/run_torch_glm45.sh"
    # "KimiK2_EP2 ./KimiK2_EP2/run_paddle_kimik2.sh ./KimiK2_EP2/run_torch_kimik2.sh"
)
```

也可绕过入口单独跑：先 `bash KimiK2_EP2/run_paddle_kimik2.sh`，再
`bash KimiK2_EP2/run_torch_kimik2.sh`，最后 `python3 compare_loss.py logs/paddle logs/torch`。

## 判定口径：单 step md5

- 判定依据：`compare_loss.py` 解析两侧日志的 `per_token_loss`/`final_loss` md5，逐 step 要求完全一致。
- 本用例固定 `max_steps=1`/`train_iters=1`，只监控首个 step 的前向 loss md5；多 step 反向的
  bf16 累加差异（bf16 精度极限，非算法 bug）会被严格 md5 判为不一致。
- 注：2 卡 EP2 配置的 step1 md5 基线需首次 smoke 后确立（既往 bit-exact 结论来自 8 卡 TP8/EP8 复现）。

## 两侧关键配置对应关系

为让严格 md5 可比，以下项两侧必须成对出现，改动任一侧都要同步另一侧：

| 项 | paddle (`KimiK2.yaml`) | torch (`run_torch_kimik2.sh`) |
|---|---|---|
| attention | `_attn_implementation: eager` | `--attention_backend unfused` + `NVTE_FLASH_ATTN=0` / `NVTE_FUSED_ATTN=0` |
| template | `use_template: false` | `--template dummy --template_backend swift` |
| 截断 | `truncation_strategy: right` | `--truncation_strategy right` |
| 样本顺序 | `mix_strategy: concat` / `random_shuffle: false` / `dataloader_shuffle: false` | `--dataset_shuffle False --train_dataloader_shuffle False` |
| MoE aux loss | `router_aux_loss_coef: 0.0` | `--moe_aux_loss_coeff 0` |
| MoE fusion | `moe_expert_fusion: false` / `moe_shared_expert_overlap: false` | `--moe_grouped_gemm False --moe_permute_fusion False` |
| dispatcher | `moe_token_dispatcher_type: alltoall` | `--moe_token_dispatcher_type alltoall` |
| rope | `apply_rope_fusion: false` | 未开 rope fusion |
| 梯度累加 | `amp_master_grad: true` | `--accumulate_allreduce_grads_in_fp32 True` |
| 梯度裁剪 | 默认 `max_grad_norm: 1.0` | `--clip_grad 1.0` |
| 优化器 | 不开 offload / 不开 sharding | `--use_precision_aware_optimizer False --use_distributed_optimizer False` |

## 端口分配

框架内各用例串行执行但端口独占，避免残留连接互相干扰：

| 用例 | paddle | torch |
|---|---|---|
| MinimaxV2.5_EP2 | 29501 | 29500 |
| GLM45Air_EP2 | 29503 | 29502 |
| KimiK2_EP2 | 29505 | 29504 |

## 逐层 md5 hook（默认关闭）

排查首个 diff 层时按需打开，会产生大量 IO：

```bash
ENABLE_SAVE_HOOK=1 SAVE_TENSOR_GRAD=1 bash KimiK2_EP2/run_paddle_kimik2.sh
```

paddle 侧落到 `logs/pf/`，torch 侧落到 `logs/mg/`。

## 依赖资产（运行前需 stage 到缓存目录，可用环境变量覆盖）

| 资产 | 默认路径 | 覆盖方式 |
|---|---|---|
| PF ckpt（flex_checkpoint）| `/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP` | 改 `KimiK2.yaml` 的 `model_name_or_path` |
| PF 对齐数据 | `/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP/alignment_paddle.jsonl` | 改 `KimiK2.yaml` 的 `train/eval_dataset_path` |
| MG ckpt（mcore）| `/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP` | 环境变量 `KIMIK2_MODEL` |
| MG 对齐数据 | `/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP/alignment_torch.jsonl` | 环境变量 `KIMIK2_DATASET` |
| Megatron-LM | `${WORKSPACE_DIR}/Megatron-LM` | 环境变量 `KIMIK2_MEGATRON_LM_PATH` |

> 路径沿用框架内既有用例的 `/home/.cache/PaddleFormers/` 约定，不写死任何个人共享盘路径；
> 运行方按上表把 Kimi-K2 权重/数据 stage 到对应缓存目录，或用环境变量指向实际位置。

## 已知前置 / 未验证项

- 需单机 2 卡空闲；框架 venv（`venv/paddle`、`venv/torch`）由上级 `setup_venvs.sh` 构建。
- 本用例在既有 Kimi-K2 PF↔MG 精度对齐基础上，按 2 卡 TP=1/EP=2/PP=1 整理而来，
  但**尚未在本框架 venv 下端到端跑通验证**；落地时需先 smoke 一次确认 md5 锚点正常打印并确立 step1 基线，
  之后才放开 `run_alignment_test.sh` 里的 `CASES` 注册。
- 若两侧日志未出现 `per_token_loss:`/`final_loss:` 锚点，检查 paddle 侧
  `use_accuracy_compatible`/`FLAGS_use_accuracy_compatible_kernel` 与 torch 侧 `--use_accuracy_compatible`。
- paddle 侧脚本先 `unset MASTER_PORT` 再取默认值，因此外部传入的 `MASTER_PORT` 不生效（与同目录既有用例同源行为）。
- torch 侧 `--torch_dtype float32` 配 `--bf16 True` 沿用 Megatron master weight 惯例，未单独验证可否去掉。
