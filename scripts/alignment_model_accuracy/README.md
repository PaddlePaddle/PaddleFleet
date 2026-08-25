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
export MEGATRON_CORE_WHEEL=/path/to/megatron_core.whl
export MS_SWIFT_WHEEL=/path/to/ms_swift.whl
bash setup_venvs.sh
```

## 新增对齐用例

1. 在本目录下新建 `<CaseName>/` 子目录，放入该用例的 `run_paddle_*.sh`、
   `run_torch_*.sh` 和训练配置。两个脚本需要各自把训练日志写到
   `logs/paddle/<时间戳>/`、`logs/torch/<时间戳>/`（沿用现有脚本里
   `WORKSPACE_DIR` 的写法，指向本目录）。
2. 在 `run_alignment_test.sh` 的 `CASES` 数组里加一行
   `"<CaseName> ./<CaseName>/run_paddle_xxx.sh ./<CaseName>/run_torch_xxx.sh"`。
