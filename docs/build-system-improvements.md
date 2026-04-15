# PaddleFleet 构建系统优化说明

## 概述

本文档记录了对 PaddleFleet monorepo 构建系统的三项优化，涉及 CUDA 架构编译控制、子包构建环境隔离、ops 版本自动绑定以及依赖清理。

---

## 优化一：CUDA 架构编译自动检测

### 背景

`packages/paddlefleet_ops/setup.py` 中原先硬编码了三个 CUDA 架构的编译目标：

```python
# 改动前
nvcc_args = [
    ...
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-gencode=arch=compute_100,code=sm_100",
]
```

这带来两个问题：

1. **本地开发编译慢**：开发者在 H800（sm_90a）机器上开发，不需要同时编译 sm_80 和 sm_100，但每次都会浪费大量时间。
2. **缺乏灵活性**：新增 sm_103（Blackwell）架构时需要修改源码，且无法按需控制编译目标。

### 方案

在 `setup.py` 中引入三级优先级的动态 gencode 解析逻辑：

```
优先级（从高到低）：
1. PADDLE_CUDA_ARCH_LIST 环境变量（手动指定，CI 控制入口）
2. nvidia-smi 自动检测当前机器 GPU 架构
3. CUDA 版本兜底：< 12.8 → sm_90 / ≥ 12.8 → sm_90 + sm_100 + sm_103
```

架构字符串到 gencode flags 的映射：

| 架构字符串 | gencode flag |
|-----------|-------------|
| `8.0` | `-gencode=arch=compute_80,code=sm_80` |
| `9.0` | `-gencode=arch=compute_90a,code=sm_90a` |
| `10.0` | `-gencode=arch=compute_100,code=sm_100` |
| `10.3` | `-gencode=arch=compute_103,code=sm_103` |

> **注意**：`PADDLE_CUDA_ARCH_LIST` 被 Paddle 的 `CUDAExtension` 内部读取，会额外追加 gencode flags。因此开发者直接 `unset PADDLE_CUDA_ARCH_LIST` 即可让 nvidia-smi 自动检测生效，无需额外操作。

### 各场景行为

| 场景 | 实际编译架构 | 说明 |
|------|------------|------|
| 本地开发（H800，无环境变量） | `sm_90a` | nvidia-smi 检测到 9.0，只编一个架构 |
| 本地开发（手动指定） | 任意 | `export PADDLE_CUDA_ARCH_LIST=9.0;10.0` |
| nightly 编包（CUDA 12.9，无 GPU） | `sm_90a + sm_100 + sm_103` | nvidia-smi 失败，走 CUDA 版本兜底 |
| nightly 编包（CUDA 12.6，无 GPU） | `sm_90a` | CUDA < 12.8，sm_100/sm_103 不支持 |

### 收益

- **本地开发编译时间大幅缩短**：只编当前机器架构，避免编译无用 PTX
- **新架构支持零成本**：在 `_ARCH_TO_GENCODE` 字典中加一行即可
- **CI 无需配置**：nightly 编包机无 GPU，自动走版本兜底全量编译

---

## 优化二：子包构建环境隔离（PYTHONPATH 污染修复）

### 背景

`paddlefleet_ops` 的 `build_backend.py` 在构建时会以子进程方式编译三方库（`quack`、`sonic-moe` 等）：

```python
cmd = [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation", ...]
subprocess.check_call(cmd, cwd=self.source_dir, env=_env)
```

由于 `paddlefleet_ops/pyproject.toml` 声明了 `backend-path = ["."]`，pip 在 `--no-build-isolation` 模式下会将该路径注入到子进程的 `PYTHONPATH`。子包（如 `quack`）查找自己的 `build-backend = "setuptools.build_meta"` 时，会优先搜索 `packages/paddlefleet_ops/`，而该目录只有自定义的 `build_backend.py`，没有 `setuptools`，导致报错：

```
BackendUnavailable: Cannot find module 'setuptools.build_meta'
in ['.../packages/paddlefleet_ops']
```

### 方案

在 `build_utils.py` 的子进程启动前，从 `PYTHONPATH` 中过滤掉父包根目录：

```python
pythonpath = _env.get("PYTHONPATH", "")
cleaned = os.pathsep.join(
    p for p in pythonpath.split(os.pathsep)
    if p and str(PKG_ROOT) not in p
)
if cleaned:
    _env["PYTHONPATH"] = cleaned
else:
    _env.pop("PYTHONPATH", None)
```

### 收益

- **修复三方库子包构建报错**：quack、sonic-moe 等子包能正确找到 setuptools
- **不影响三方库自身的构建逻辑**：仅清理父包路径，不干扰子包的其他依赖

---

## 优化三：paddlefleet-ops 版本自动精确绑定

### 背景

`paddlefleet`（纯 Python 包）依赖 `paddlefleet-ops`（CUDA 算子包）。原先 `pyproject.toml` 中写的是无版本约束的裸名：

```toml
dependencies = [
    "colorlog>=6.10.1",
    "paddlefleet-ops",   # ← 无版本约束
]
```

这在本地开发（uv workspace）时没有问题，因为 `[tool.uv.sources]` 会直接用本地源码。但当两个包分开打成 wheel 发布后，`paddlefleet` wheel 的 `Requires-Dist: paddlefleet-ops` 没有版本约束，用户安装时可能拿到任意版本的 ops，导致接口不兼容。

### 方案

参考 **xFormers** 的做法：在 `build_backend.py` 的构建阶段动态读取 ops 的实际版本，并 patch setuptools 的 `Distribution.finalize_options`，在 wheel metadata 写入前将裸名替换为精确版本约束。

```python
# build_backend.py（主包）

def _get_ops_version() -> str | None:
    """读取 workspace 中 paddlefleet-ops 的实际版本。"""
    if not _ops_version_py.exists():
        return None
    globs: dict = {}
    exec(_ops_version_py.read_text(), globs)
    return globs.get("__version__")

def _pin_ops_dependency() -> None:
    ops_version = _get_ops_version()
    if ops_version is None:
        return  # 降级为裸名（sdist 场景）

    pinned = f"paddlefleet-ops=={ops_version}"
    _orig_finalize = Distribution.finalize_options

    def _patched_finalize(self):
        _orig_finalize(self)
        if self.install_requires:
            self.install_requires = [
                pinned if req.strip() == "paddlefleet-ops" else req
                for req in self.install_requires
            ]

    Distribution.finalize_options = _patched_finalize
```

### 各场景行为

| 场景 | paddlefleet wheel 中的依赖约束 |
|------|------------------------------|
| `uv sync` / `uv pip install -e .` | workspace 直接用源码，版本号无关 |
| `uv build --wheel`（ops 先构建） | `Requires-Dist: paddlefleet-ops==0.3.0.dev20260415` |
| 单独构建 paddlefleet（ops 未构建） | 降级为 `Requires-Dist: paddlefleet-ops`，并打印 warning |

### 收益

- **开发者零感知**：无需手动修改任何版本号，两包同批次构建天然版本一致
- **生产环境安全**：发布的 wheel 携带精确版本约束，杜绝版本漂移
- **降级兼容**：在无 ops 源码的环境（如单独构建）自动退为裸名，不影响正常安装

---

## 优化四：依赖归属清理

### 背景

`paddlefleet-ops` 是纯 CUDA 算子包，不应依赖 `colorlog`（日志库）。原先 `setup.py` 中错误地将 `colorlog` 列为 `paddlefleet-ops` 的运行时依赖：

```python
common_dependencies = [
    "colorlog>=6.10.1",   # ← 不应属于 ops 包
]
```

这导致从 paddle 专用源安装 `paddlefleet-ops` wheel 时报错：

```
ERROR: Could not find a version that satisfies the requirement colorlog>=6.10.1
```

因为 paddle 的源里没有 `colorlog`。

### 方案

将 `colorlog` 从 `paddlefleet-ops` 的依赖中移除，仅保留在上层的 `paddlefleet` 主包中：

```python
# packages/paddlefleet_ops/setup.py
common_dependencies: list[str] = []   # colorlog 不属于 ops 包
```

### 收益

- **修复从 paddle 专用源安装失败的问题**
- **依赖关系清晰**：ops 包只依赖 CUDA 相关库，`colorlog` 由上层 `paddlefleet` 管理

---

## 文件变更汇总

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `packages/paddlefleet_ops/setup.py` | 修改 | CUDA 架构动态检测；移除 colorlog 依赖 |
| `packages/paddlefleet_ops/build_utils.py` | 修改 | 子进程构建前清理 PYTHONPATH 中的父包路径 |
| `build_backend.py` | 修改 | 构建时动态绑定 paddlefleet-ops 精确版本 |
| `.github/workflows/publish_wheel_nightly.yml` | 修改 | 移除硬编码 PADDLE_CUDA_ARCH_LIST，走自动兜底 |
| `.github/workflows/manual_build_wheel.yml` | 修改 | 同上 |
