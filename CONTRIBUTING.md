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

```bash
git clone git@github.com:<YOUR_USER_NAME>/PaddleFleet.git               # 将你的 repo clone 到本地
cd PaddleFleet/                                                         # cd 到该目录
git remote add upstream https://github.com/PaddlePaddle/PaddleFleet.git     # 将原分支绑定在 upstream
uv sync # 本地构建环境
uv build # 构建wheel包
uv run pytest #跑单卡单测
uv run python -m paddle.distributed.launch --nnodes=1 --log_dir=log --devices=0,1,2,3,4,5,6,7 tests/unit_tests/test_parallel_states.py #跑多卡单测

```
