# Alignment Model Accuracy

用于验证 PaddleFleet（Paddle 侧）与 Megatron-LM/ms-swift（Torch 侧）在同一份训练配置下
逐 step 的 loss 是否精度对齐。

## 目录结构

- `setup_venvs.sh`：创建/复用 `venv/torch`、`venv/paddle` 两个虚拟环境，分别安装
  Torch 侧（Megatron-LM + ms-swift）与 Paddle 侧（PaddleFleet + PaddleFormers）依赖。
- `run_alignment_test.sh`：对齐测试入口，依次跑每个用例的 paddle/torch 训练脚本，
  再用 `compare_loss.py` 对比两侧日志中的 loss md5，汇总所有用例的成功/失败。
- `compare_loss.py`：解析训练日志里的 `per_token_loss` / `final_loss` 锚点并逐 step
  比较 md5，支持直接指定某次运行目录，也支持指定日志根目录自动取最新一次运行。
- `<CaseName>/`（如 `MinimaxV2.5_EP2/`）：每个对齐用例一个目录，包含该用例的
  `run_paddle_*.sh`、`run_torch_*.sh` 和训练配置文件。

## 本地运行

```bash
cd scripts/alignment_model_accuracy

# 1. 准备环境（首次运行较慢，之后会复用已存在的 venv）
bash setup_venvs.sh

# 2. 跑对齐测试（内部会执行 CASES 列表中的每个用例并汇总结果）
bash run_alignment_test.sh
```

`setup_venvs.sh` 默认从 nightly 镜像下载 `paddlefleet` / `paddlefleet-ops` wheel；
如果本地已有构建好的 wheel，可通过环境变量覆盖：

```bash
export PADDLEFLEET_WHEEL_PATH=/path/to/paddlefleet-xxx.whl
export PADDLEFLEET_OPS_WHEEL_PATH=/path/to/paddlefleet_ops-xxx.whl
export PADDLEFORMERS_WHEEL_PATH=/path/to/paddleformers-xxx.whl
export MEGATRON_CORE_WHEEL_PATH=/path/to/megatron_core.whl
export MS_SWIFT_WHEEL_PATH=/path/to/ms_swift.whl
bash setup_venvs.sh
```

## 新增对齐用例

1. 在本目录下新建 `<CaseName>/` 子目录，放入该用例的 `run_paddle_*.sh`、
   `run_torch_*.sh` 和训练配置。两个脚本需要各自把训练日志写到
   `logs/paddle/<时间戳>/`、`logs/torch/<时间戳>/`（沿用现有脚本里
   `WORKSPACE_DIR` 的写法，指向本目录）。
2. 在 `run_alignment_test.sh` 的 `CASES` 数组里加一行
   `"<CaseName> ./<CaseName>/run_paddle_xxx.sh ./<CaseName>/run_torch_xxx.sh"`。

## GLM-5.2 的 IEEE EP2/TP1 对齐配置

GLM-5.2 在 `use_accuracy_compatible: true` 且
`MODEL_REPRO_IEEE_KERNEL=1` 时，保留 padding token 的 embedding 与 MoE routing，
使 MTP 的移位输入在 EP/TP 布局之间一致；实验版模型仍沿用其 padding 策略。

使用 PaddleFormers 的 deferred token normalization 路径验证 EP2/TP1/PP2 时，
显式设置以下现有训练参数：

```yaml
hybrid_parallel_expert_grad_scale: 1.0
```

该布局下自动值 `TP * CP / EP` 为 `0.5`，会在 token normalization 之前额外
缩放 routed expert 梯度。这里的显式配置保留已经对齐的原始专家梯度，之后由
有效 token 数完成归一化。

## GLM-5.2 精度监控提测

`GLM52_EP2_TP1_PP2` 使用四张 GPU（TP1、PP2、EP2、SP 关闭），运行原生
PaddleFormers / Megatron CLI 各 100 步。模型是 GLM-5.2 官方权重的
`minimum_complete` 子集：3 个 dense 层、1 个 MoE 层、1 个 MTP 层、16 个专家；
不是完整的 GLM-5.2 模型。

### 模型与数据缓存

请先联系模型仓库管理员创建 `PaddleFormers/GLM-5.2-minimum-complete-bf16`，
确认实际 repo_id 后再上传。缓存目标为
`/home/.cache/PaddleFormers/GLM-5.2-minimum-complete-bf16/`，需包含：

- `config.json`、`model.safetensors`、`model.safetensors.index.json`
- `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`
- `generation_config.json`、`LICENSE`、`extraction_manifest.json`

权重来源为 `zai-org/GLM-5.2` revision
`b4734de4facf877f85769a911abafc5283eab3d9`，187 个张量；原始抽取文件的
SHA256 为 `53ad1565fe8173420db4b3ef53db3bddef0afda4b43f859aacf16826700916c2`。
保留抽取来源记录，不上传训练后的 checkpoint 或自动生成的加载缓存。
上传完成后将 repo_id、文件 SHA256 清单和缓存路径同步给张军军，待 CI 机器缓存完成再提测。
仓库创建、上传和 CI 缓存是外部准备步骤，添加用例不表示它们已完成。

数据复用现有 `MinimaxV2.5_EP2` 的
`/home/.cache/PaddleFormers/MiniMax-V2.5-bf16_2EP/alignment_paddle.jsonl` 和
`alignment_torch.jsonl`。两侧保持样本顺序一致，由各自原生 tokenizer/template 处理。

### 配套依赖与运行

PaddleFleet #1961、PaddleFormers #4956、PFCCLab/Megatron-LM #4、
PFCCLab/ms-swift #3 的 GLM-5.2 原生训练实现需要一起使用。在这些改动进入默认
wheel 前，使用上文的 wheel 环境变量指定对应构建，尤其是新增的
`PADDLEFORMERS_WHEEL_PATH`；不要用默认 nightly 的成功安装代替配套源码验证。

```bash
cd scripts/alignment_model_accuracy
bash run_alignment_test.sh --case GLM52_EP2_TP1_PP2
```

不指定 `--case` 时会运行全部注册用例。已有 Minimax / GLM45 用例继续使用日志
MD5 比较。GLM52 的 PP2/MTP 路径使用原生训练回调写出的 `loss.json`，通过
`compare_loss.py --loss-json --required-steps 100` 检查主训练 loss 的 IEEE 位模式。
比较器要求正确的 framework、完成状态、连续的 1–100 步和有限数值；缺失、截断、
步数不足或任何一位不同都会失败。该检查不代表所有中间张量或 MTP loss 均逐位一致。
每次用例的原始日志、loss、环境记录及 checkpoint 保留在
`results/<本次运行标识>/<CaseName>/`，训练失败也保留日志并计入总失败结果。

已有本地环境可直接执行两个 `run_*_glm52.sh`，通过 `GLM52_VENV_ROOT` 指向包含
`paddle/` 和 `torch/` 的 venv 父目录；模型、tokenizer 和数据路径分别用
`GLM52_MODEL_DIR`、`GLM52_TOKENIZER_DIR`、`GLM52_DATA_DIR` 指定。
两个脚本设置相同的 `ALIGNMENT_RUN_TAG` 后，可对明确的本次文件执行比较，
避免选择到历史成功结果。正式提测仍需验证 CI 的实际 GPU、CUDA 与配套 wheel，
本地环境的通过结果不能替代 CI 机器实测。
