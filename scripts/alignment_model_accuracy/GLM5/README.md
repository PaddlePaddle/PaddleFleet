# GLM5 对齐用例

GLM-5 在 **PaddleFormers+PaddleFleet ↔ ms-swift+Megatron-LM** 下的精度对齐监控用例，
接入 `scripts/alignment_model_accuracy/` 框架（见上级 `README.md`）。
配置约定对标同目录 `GLM45Air_EP2`，并行度为单机 **2 卡：TP=1 / EP=1 / PP=1 + sharding stage1 degree2**
（paddle 侧 ZeRO-1 优化器状态分片，torch 侧对应 DP2 + `--use_distributed_optimizer True`）。

## 文件

| 文件 | 说明 |
|---|---|
| `GLM5.yaml` | paddle 侧训练配置（TP=1/EP=1/PP=1 + `sharding: stage1` / `sharding_parallel_size: 2`，PT，`use_accuracy_compatible: true`，10 step）|
| `run_paddle_glm5.sh` | paddle 侧启动：清分布式 env + 确定性/对齐 flags + `paddleformers-cli train` |
| `run_torch_glm5.sh` | torch 侧启动：对齐 flags + `megatron sft <ARGS>` |

## 注册到测试入口

`scripts/alignment_model_accuracy/run_alignment_test.sh` 的 `CASES` 数组中本用例**已放开**：

```bash
CASES=(
    "MinimaxV2.5_EP2 ./MinimaxV2.5_EP2/run_paddle_minimax.sh ./MinimaxV2.5_EP2/run_torch_minimax.sh"
    "GLM45Air_EP2 ./GLM45Air_EP2/run_paddle_glm45.sh ./GLM45Air_EP2/run_torch_glm45.sh"
    "GLM5 ./GLM5/run_paddle_glm5.sh ./GLM5/run_torch_glm5.sh"
)
```

也可绕过入口单独跑：先 `bash GLM5/run_paddle_glm5.sh`，再
`bash GLM5/run_torch_glm5.sh`，最后 `python3 compare_loss.py logs/paddle logs/torch`。

## 判定口径：10 step per_token_loss md5

- 判定依据：`compare_loss.py` 解析两侧日志的 `per_token_loss`/`final_loss` md5，逐 step 要求完全一致。
- 本用例 `max_steps=10`/`train_iters=10`，**rank0 / rank1 各 10 个 step 的 `per_token_loss` md5 两侧逐位一致**（cu130 环境下已验证）。
- `per_token_loss` 是 CE 直出张量（mask/归一化前），驱动反向的就是它；它全 step 一致即说明
  前向→反向→优化器 step 的整条轨迹 bit-exact。
- **已知残余**：`final_loss`（打印用标量）在个别 step 会差 1 个 fp32 ULP（$2^{-20}\approx9.54\text{e-}7$），
  原因是两侧 fp64 求和后 cast 回 fp32 的 double-rounding 落到相邻值；**不进反向、不随 step 累积**。
  因此判定应以 `per_token_loss` 为准；若要 `final_loss` 也严格全绿，需再统一该标量末端的 reduce/cast。
- cu129 时代 step2/3 会稳定漂 ~1.4e-4 且发散（paddle 融合 Adam kernel 的 fp32 master 更新 FMA 次序），
  **cu130 下该地板消失**，故本用例要求 cu130 环境。

## 两侧关键配置对应关系

为让严格 md5 可比，以下项两侧必须成对出现，改动任一侧都要同步另一侧：

| 项 | paddle (`GLM5.yaml`) | torch (`run_torch_glm5.sh`) |
|---|---|---|
| 并行 | `tensor/pipeline/expert_model_parallel_size: 1`、`use_expert_parallel: false` | `--tensor/pipeline/expert_model_parallel_size 1` |
| 分片 | `sharding: stage1` / `sharding_parallel_size: 2`（2 卡 ZeRO-1）| DP=2（`NPROC_PER_NODE=2`）+ `--use_distributed_optimizer True` |
| 模型结构 | `multi_latent_attention: true` / `use_qk_norm: true` / `num_hidden_layers: 3` | 由 ckpt config 决定（tiny 3 层） |
| attention | 默认 flash 路径 | `--attention_backend flash` |
| template | `use_template: false` | `--template dummy --template_backend swift` |
| 样本顺序 | `mix_strategy: concat` / `random_shuffle: false` / `dataloader_shuffle: false` | `--dataset_shuffle False --train_dataloader_shuffle False` |
| packing | `packing: false` / `truncate_packing: false` | `--packing False --padding_free False` |
| batch | `per_device_train_batch_size: 1` / `gradient_accumulation_steps: 1`（2 shard → global 2）| `--micro_batch_size 1 --global_batch_size 2` |
| MoE aux loss | `router_aux_loss_coef: 1.0e-4` / `moe_router_load_balancing_type: seq_aux_loss` | `--moe_aux_loss_coeff 1e-4 --moe_router_load_balancing_type seq_aux_loss` |
| MoE 其他 | `moe_deep_gemm: false` | `--moe_grouped_gemm True --moe_permute_fusion True --moe_router_dtype fp32` |
| dispatcher | 默认 alltoall | `--moe_token_dispatcher_type alltoall` |
| DSA indexer loss | `dsa_indexer_loss_coeff: 0.0` | `--dsa_indexer_loss_coeff 0.0` |
| 重计算 | `recompute_granularity: none` | `--recompute_granularity none` |
| loss fusion | 走非 fused 交叉熵 | `--cross_entropy_loss_fusion False --calculate_per_token_loss False` |
| 梯度累加 | `amp_master_grad: true` | `--accumulate_allreduce_grads_in_fp32 True` |
| 梯度裁剪 | `max_grad_norm: 0.0` | `--clip_grad 0.0` |
| 优化器 | 不开 offload / `sharding: stage1` | `--use_precision_aware_optimizer False --use_distributed_optimizer True` |
| lr | `5.0e-5` / `min_lr 1.0e-5` / cosine / `warmup_steps: 0` | `--lr 5e-5 --min_lr 1e-5 --lr_decay_style cosine --lr_warmup_iters 0` |
| 确定性 | `FLAGS_cudnn_deterministic=1` / `FLAGS_embedding_deterministic=1` | `NVIDIA_TF32_OVERRIDE=0` / `TORCHDYNAMO_DISABLE=1` |

## 端口分配

框架内各用例串行执行但端口独占，避免残留连接互相干扰：

| 用例 | paddle | torch |
|---|---|---|
| MinimaxV2.5_EP2 | 29501 | 29500 |
| GLM45Air_EP2 | 29503 | 29502 |
| GLM5 | 29507 | 29506 |

## 逐层 md5 hook（默认关闭）

排查首个 diff 层时按需打开，会产生大量 IO：

```bash
ENABLE_SAVE_HOOK=1 SAVE_TENSOR_GRAD=1 bash GLM5/run_paddle_glm5.sh
```

paddle 侧落到 `logs/pf/`，torch 侧落到 `logs/mg/`。

## 依赖资产（运行前需 stage 到缓存目录，可用环境变量覆盖）

| 资产 | 默认路径 | 覆盖方式 |
|---|---|---|
| PF ckpt（flex_checkpoint）| `/home/.cache/PaddleFormers/GLM-5-bf16_1Card` | 改 `GLM5.yaml` 的 `model_name_or_path` |
| PF 对齐数据 | `/home/.cache/PaddleFormers/GLM-5-bf16_1Card/alignment_paddle.jsonl` | 改 `GLM5.yaml` 的 `train/eval_dataset_path` |
| MG ckpt（mcore）| `/home/.cache/PaddleFormers/GLM-5-bf16_1Card` | 改 `run_torch_glm5.sh` 的 `--model` |
| MG 对齐数据 | `/home/.cache/PaddleFormers/GLM-5-bf16_1Card/alignment_torch.jsonl` | 改 `run_torch_glm5.sh` 的 `--dataset` |
| Megatron-LM | `${WORKSPACE_DIR}/Megatron-LM` | `MEGATRON_LM_PATH` |

> 路径沿用框架内既有用例的 `/home/.cache/PaddleFormers/` 约定，不写死任何个人共享盘路径。

## 已知前置 / 未验证项

- 需 2 卡空闲（两侧均 `CUDA_VISIBLE_DEVICES=0,1`，用例串行执行；每卡约 60GB 显存，8192 序列长 3 层 tiny ckpt）；框架 venv 由上级 `setup_venvs.sh` 构建（**cu130**）。
- 2 卡 DP 的梯度 all-reduce 是单次成对求和（可交换、确定性），优化器状态分片只改「在哪算」不改数学，
  因此分片路径不引入新的跨框架分叉——这一点已与单卡结果对照验证过。
- **模型 `config.json` 需满足 `rope_parameters.rope_type: "rope"`**：GLM-5 原始 config 写的是 `"default"`，
  而 PF 对 `"default"` 不改写、会落到 `TransformerConfig` 默认 `"yarn"`（NTK-by-parts scaling），
  导致 MLA 误建 YarnRotaryEmbedding、rotary 与 MG 分叉。`"default"` 与 `"rope"` 对 GLM-5 语义等价，
  故资产侧显式写成 `"rope"` 即可，无需任何代码 hack。
- 两侧 `dsa_indexer_loss_coeff` 必须同为 `0.0`（PF yaml 与 torch `--dsa_indexer_loss_coeff`），否则 loss 口径不一致。
- 本用例配置来自线下已复现的 GLM-5 PF↔MG 对齐（cu130 下 10 step `per_token_loss` 逐位一致），
  但**尚未在本框架 venv + `/home/.cache/PaddleFormers/` 资产下端到端跑通**；首次落地建议先 smoke 一次确认
  md5 锚点正常打印。
- 若两侧日志未出现 `per_token_loss:`/`final_loss:` 锚点，检查 paddle 侧
  `use_accuracy_compatible`/`FLAGS_use_accuracy_compatible_kernel` 与 torch 侧 `--use_accuracy_compatible`。
- paddle 侧脚本先 `unset MASTER_PORT` 再取默认值，因此外部传入的 `MASTER_PORT` 不生效（与同目录既有用例同源行为）。
- 线下复现用的是 3 层 tiny GLM-5；换全量 ckpt 需重新确立 md5 基线。
