# KimiK2_EP2 对齐用例

Kimi-K2 在 **PaddleFormers+PaddleFleet ↔ ms-swift+Megatron-LM** 下的精度对齐监控用例，
接入 `scripts/alignment_model_accuracy/` 框架（见上级 `README.md` 与 PR #1751）。

## 文件

| 文件 | 说明 |
|---|---|
| `KimiK2.yaml` | paddle 侧训练配置（TP=1/EP=2/PP=1，SFT，`use_accuracy_compatible: true`，单 step）|
| `run_paddle_kimik2.sh` | paddle 侧启动：清分布式 env + 对齐 flags + md5 hooks + `paddleformers-cli train` |
| `run_torch_kimik2.sh` | torch 侧启动：对齐 flags + md5 hooks + `megatron sft <ARGS>` |

## 注册到测试入口

在 `scripts/alignment_model_accuracy/run_alignment_test.sh` 的 `CASES` 数组加一行：

```bash
CASES=(
    "MinimaxV2.5_EP2 ./MinimaxV2.5_EP2/run_paddle_minimax.sh ./MinimaxV2.5_EP2/run_torch_minimax.sh"
    "KimiK2_EP2 ./KimiK2_EP2/run_paddle_kimik2.sh ./KimiK2_EP2/run_torch_kimik2.sh"
)
```

## 判定口径：单 step md5

- 判定依据：`compare_loss.py` 解析两侧日志的 `per_token_loss`/`final_loss` md5，逐 step 要求完全一致。
- 本用例固定 `max_steps=1`/`train_iters=1`，只监控首个 step 的前向 loss md5；多 step 反向的
  bf16 累加差异（bf16 精度极限，非算法 bug）会被严格 md5 判为不一致。
- 注：2 卡 EP2 配置的 step1 md5 基线需首次 smoke 后确立（既往 bit-exact 结论来自 8 卡 TP8/EP8 复现）。

## 依赖资产（运行前需 stage 到缓存目录，可用环境变量覆盖）

| 资产 | 默认路径 | 覆盖方式 |
|---|---|---|
| PF ckpt（flex_checkpoint）| `/root/.cache/PaddleFormers/Kimi-K2-bf16_2EP` | 改 `KimiK2.yaml` 的 `model_name_or_path` |
| PF 对齐数据 | `/root/.cache/PaddleFormers/alignment_paddle.jsonl` | 改 `KimiK2.yaml` 的 `train/eval_dataset_path` |
| MG ckpt（mcore）| `/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP` | 环境变量 `KIMIK2_MODEL` |
| MG 对齐数据 | `/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP/alignment_torch.jsonl` | 环境变量 `KIMIK2_DATASET` |
| Megatron-LM | `${WORKSPACE_DIR}/Megatron-LM` | 环境变量 `KIMIK2_MEGATRON_LM_PATH` |

> 说明：路径沿用框架内 MiniMax 用例的 `~/.cache/PaddleFormers/` 约定，不写死任何个人共享盘路径；
> 运行方按上表把 Kimi-K2 权重/数据 stage 到对应缓存目录，或用环境变量指向实际位置。

## 已知前置 / 未验证项

- 需单机 2 卡空闲；框架 venv（`venv/paddle`、`venv/torch`）由上级 `setup_venvs.sh` 构建。
- 本用例在既有 Kimi-K2 PF↔MG 精度对齐基础上，按 2 卡 TP=1/EP=2/PP=1 整理而来，
  但**尚未在本框架 venv 下端到端跑通验证**；落地时需先 smoke 一次确认 md5 锚点正常打印并确立 step1 基线。
- 若两侧日志未出现 `per_token_loss:`/`final_loss:` 锚点，检查 paddle 侧
  `use_accuracy_compatible`/`FLAGS_use_accuracy_compatible_kernel` 与 torch 侧 `--use_accuracy_compatible`。
