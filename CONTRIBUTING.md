# PaddleFleet 贡献指南

很高兴你对参与 PaddleFleet 的贡献感兴趣，在提交你的贡献之前，请花一点点时间阅读本指南

## 开发工具链

为了获得最佳的开发体验，希望你能够安装一些开发工具

这些工具都是可选的，都有一定的替代方案，不过可能会稍微麻烦些……

### 项目管理工具 uv

[uv](https://docs.astral.sh/uv/) 是 PaddleFleet 用来进行项目管理的工具，你可以从[安装指南](https://docs.astral.sh/uv/getting-started/installation/)找到合适的安装方式～

## 本地调试

### Fork Repo 到自己 GitHub 账户

为了方便提交 PR，建议你在 clone 之前先在自己的 GitHub 创建一个 fork，你可以前往 [PaddleFleet/fork](https://github.com/PaddlePaddle/PaddleFleet.git/fork) 来创建一个 Fork。

### Clone Repo 到本地调试

<!-- TODO: Need to update the following commands. -->
```bash
git clone git@github.com:<YOUR_USER_NAME>/PaddleFleet.git  # 将你的 repo clone 到本地
cd PaddleFleet/  # cd 到该目录
git remote add upstream https://github.com/PaddlePaddle/PaddleFleet.git  # 将原分支绑定在 upstream
```

### 编包与开发安装

PaddleFleet 现在使用 uv workspace 方式管理项目，请在 PaddleFleet 仓库根目录执行以下命令。

这里涉及两种常见使用方式：

- **编包**：将当前代码构建成 `.whl` 安装包，适合发布、部署，或在其他环境中安装验证。
- **开发模式（editable）**：将源码目录以可编辑方式安装到当前 Python 环境中，适合本地开发调试。Python 代码修改后通常无需重新安装；如果修改了 C++ 自定义算子，则需要重新编译或重新安装对应子包。

#### 编包

```bash
# 只编根包 paddlefleet，产出纯 Python wheel 包
uv build --wheel -vv

# 只编子包 paddlefleet_ops，产出 C++ wheel 包
# 需要先在当前环境中准备好 paddlepaddle-gpu 等依赖包
uv build --package paddlefleet-ops --wheel -vv --no-build-isolation
```

#### 开发模式安装

##### 安装根包 `paddlefleet`

```bash
uv pip install -e . -vv
```

##### 安装 C++ 包 `paddlefleet_ops`

如果不需要自己开发自定义算子，可以执行以下脚本自动安装匹配当前代码的 `paddlefleet_ops`：

```bash
bash scripts/install_ops_wheel.sh
```

如果需要开发自定义算子，可以使用 editable 方式安装。

###### 系统环境

```bash
uv pip install -e packages/paddlefleet_ops -vv --no-build-isolation --system
```

###### 虚拟环境

```bash
# 需要先安装 paddlepaddle-gpu
uv pip install -e packages/paddlefleet_ops -vv --no-build-isolation
```
