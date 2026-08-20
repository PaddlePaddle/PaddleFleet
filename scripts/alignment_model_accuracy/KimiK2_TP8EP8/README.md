# KimiK2_TP8EP8 对齐用例

Kimi-K2 在 **PaddleFormers+PaddleFleet ↔ ms-swift+Megatron-LM** 下的精度对齐监控用例，
接入 `scripts/alignment_model_accuracy/` 框架（见上级 `README.md` 与 PR #1751）。

## 文件

| 文件 | 说明 |
|---|---|
| `KimiK2.yaml` | paddle 侧训练配置（TP=8/EP=8/PP=1，SFT，`use_accuracy_compatible: true`，单 step）|
| `run_paddle_kimik2.sh` | paddle 侧启动：清分布式 env + 对齐 flags + md5 hooks + `paddleformers-cli train` |
| `run_torch_kimik2.sh` | torch 侧启动：对齐 flags + md5 hooks + `megatron sft <ARGS>` |

## 注册到测试入口

在 `scripts/alignment_model_accuracy/run_alignment_test.sh` 的 `CASES` 数组加一行：

```bash
CASES=(
    "MinimaxV2.5_EP2 ./MinimaxV2.5_EP2/run_paddle_minimax.sh ./MinimaxV2.5_EP2/run_torch_minimax.sh"
    "KimiK2_TP8EP8 ./KimiK2_TP8EP8/run_paddle_kimik2.sh ./KimiK2_TP8EP8/run_torch_kimik2.sh"
)
```

## 判定口径：单 step md5

- 判定依据：`compare_loss.py` 解析两侧日志的 `per_token_loss`/`final_loss` md5，逐 step 要求完全一致。
- Kimi-K2 已验证 **step1 loss bit-exact = 13.11262703**；step2 起 0.0066% 为 bf16 精度极限（非算法 bug，
  源于两套 grouped-GEMM 内核的 bf16 权重梯度累加舍入差异）。严格 md5 会把 step2+ 判为不一致，
  故本用例固定 `max_steps=1`/`train_iters=1`，只监控已对齐的 step1。

## 依赖资产（运行前需 stage 到缓存目录，可用环境变量覆盖）

| 资产 | 默认路径 | 覆盖方式 |
|---|---|---|
| PF ckpt（flex_checkpoint）| `/root/.cache/PaddleFormers/Kimi-K2-bf16_TP8EP8` | 改 `KimiK2.yaml` 的 `model_name_or_path` |
| PF 对齐数据 | `/root/.cache/PaddleFormers/alignment_paddle.jsonl` | 改 `KimiK2.yaml` 的 `train/eval_dataset_path` |
| MG ckpt（mcore）| `/home/.cache/PaddleFormers/Kimi-K2-bf16_TP8EP8` | 环境变量 `KIMIK2_MODEL` |
| MG 对齐数据 | `/home/.cache/PaddleFormers/Kimi-K2-bf16_TP8EP8/alignment_torch.jsonl` | 环境变量 `KIMIK2_DATASET` |
| Megatron-LM | `${WORKSPACE_DIR}/Megatron-LM` | 环境变量 `KIMIK2_MEGATRON_LM_PATH` |

> 说明：路径沿用框架内 MiniMax 用例的 `~/.cache/PaddleFormers/` 约定，不写死任何个人共享盘路径；
> 运行方按上表把 Kimi-K2 权重/数据 stage 到对应缓存目录，或用环境变量指向实际位置。

## 已知前置 / 未验证项

- 需单机 8 卡全空闲；框架 venv（`venv/paddle`、`venv/torch`）由上级 `setup_venvs.sh` 构建。
- 本用例的训练配置基于已完成的 Kimi-K2 PF↔MG 精度对齐（TP=8/EP=8/PP=1 SFT）整理而来，
  但**尚未在本框架 venv 下端到端跑通验证**；落地时需先 smoke 一次确认 md5 锚点正常打印。
- 若两侧日志未出现 `per_token_loss:`/`final_loss:` 锚点，检查 paddle 侧
  `use_accuracy_compatible`/`FLAGS_use_accuracy_compatible_kernel` 与 torch 侧 `--use_accuracy_compatible`。
