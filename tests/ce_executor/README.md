# CE 执行器

## 简介

简化的 CE (Continuous Evaluation) 执行器，用于在本地 Docker 环境中运行 PaddleFleet CE 测试用例。

## 文件结构

```
ce_executor/
├── run_ce.py          # 主执行脚本
├── upload_bos.py       # 上传文件到 BOS
├── insert_db.py         # 插入数据到数据库
├── run.sh              # 快速启动脚本
└── README.md           # 本文件
```

## 功能

1. **执行 CE cases** - 通过 Docker 容器运行测试
2. **收集日志** - 从容器提取日志文件
3. **上传 BOS** - 使用 bos_tools.py 上传日志
4. **写入数据库** - 使用 HTTP POST 接口插入数据
5. **生成 Allure 报告** - 可选生成测试报告

## 使用方式

### 快速开始

```bash
# 进入目录
cd ce_executor

# 执行所有 cases（不上传）
./run.sh

# 执行并启用 BOS 上传
./run.sh --upload

# 执行特定 case
./run.sh --case glm45_single_card

# 执行并生成 Allure 报告
./run.sh --upload --allure
```

### 直接运行 Python 脚本

```bash
# 执行所有 cases
python run_ce.py

# 启用 BOS 上传
python run_ce.py --upload

# 指定 case
python run_ce.py --case glm45_single_card

# 指定环境
python run_ce.py --env cu130-py310

# 只打印不执行（dry run）
python run_ce.py --dry-run
```

### 独立功能脚本

```bash
# 上传单个文件到 BOS
python upload_bos.py /path/to/file.log cases/20260415/

# 上传整个目录
python upload_bos.py /path/to/logs/ cases/20260415/

# 注意：上传使用 bos_tools.py（自动下载）
# 实际路径：paddle-github-action/PaddleFleet/ce/{你指定的路径}

# 测试数据库接口
python insert_db.py
```

## 配置

### 上传方式

BOS 上传使用 **bos_tools.py** 脚本（从百度 BOS 自动下载）：

### Case 配置

在 `run_ce.py` 中定义：

- **DOCKER_IMAGES** - Docker 镜像映射（cu129, cu130）
- **ENVIRONMENTS** - 环境组合（CUDA + Python）
- **CASES** - CE case 列表（3个）

### 可用的 Cases

| 名称 | 脚本 | 超时 |
|------|-------|--------|
| `glm45_single_card` | PaddleFormers/tests/integration_test/glm45_pt_single_card.sh | 3600s |
| `qwen3_single_card` | PaddleFormers/tests/integration_test/qwen3_single_card.sh | 3600s |
| `qwen3vl_sft_single_card` | PaddleFormers/tests/integration_test/qwen3vl_sft_single_card.sh | 300s |

## 输出

### 工作目录

```
workspace/
├── results/
│   ├── execution_report.json      # 执行报告
│   └── {case_name}/
│       └── logs/                # 收集的日志
└── ce_data.json               # 数据库记录（JSON）
```

### BOS 结构（启用上传后）

通过 bos_tools.py 上传：
```
paddle-github-action/PaddleFleet/ce/{YYYYMMDD}/
└── cases/
    └── {case_name}/
        └── {case_name}.log
```

## 命令行参数

| 参数 | 简写 | 说明 |
|------|-------|------|
| `--case` | `-k` | 指定 case 名称（逗号分隔） |
| `--env` | `-e` | 指定环境（如 cu130-py310） |
| `--upload` | `-u` | 启用 BOS 上传 |
| `--allure` | | 生成 Allure 报告 |
| `--dry-run` | | 只打印命令不执行 |
| `--work-dir` | | 指定工作目录 |

## 故障排查

### Docker 相关

```bash
# 检查 Docker 是否运行
docker info

# 检查 GPU 可用性
nvidia-smi

# 清理旧容器
docker ps -a | grep ce- | awk '{print $1}' | xargs docker rm -f
```

### 日志查看

```bash
# 查看执行日志
tail -f workspace/results/{case_name}/{case_name}.log

# 查看 CE 执行器输出
tail -f workspace/results/execution_report.json

# 打开 Allure 报告
# 如果使用 --allure 参数，报告生成在：
# workspace/results/allure-report/index.html
```

## 执行流程

```
1. 生成执行 ID
2. 按环境分组 cases
3. 对每个环境:
   ├─ 拉取 Docker 镜像
   ├─ 启动容器
   ├─ 设置环境（安装依赖）
   ├─ 运行每个 case:
   │   ├─ 执行测试脚本
   │   ├─ 收集日志
   │   ├─ 上传日志到 BOS（如启用）
   │   └─ 插入数据库
   └─ 清理容器
4. 生成执行报告
```

## License

Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
