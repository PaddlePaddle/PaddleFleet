# Train/Infer Consistent Ops Tests / 训推一致性打点测试

Unit tests for the `paddlefleet.train_infer_consistent_ops` package -- the tensor
probe used by train_infer_consistent_inspect (K3) runs: `inspect_tensor` and the
layout helpers that turn framework-specific buffers into comparable views.
`paddlefleet.train_infer_consistent_ops` 包的单元测试——训推对拍（K3）用的张量探针
`inspect_tensor`，以及把各框架私有布局转成可比视图的辅助函数。

Every function in the package is covered (100% line coverage), private helpers
included. The `ABLATION_*` configuration is snapshotted once at import, so each
test flips the environment, calls `refresh_env_cache()`, and restores both in
`tearDown` -- otherwise the probes would stay on for the rest of the suite.
包内每个函数都有覆盖（行覆盖率 100%，含私有辅助函数）。`ABLATION_*` 配置在 import 时快照
一次，所以每个用例改完环境变量都要调 `refresh_env_cache()`，并在 `tearDown` 里还原，
否则探针会一直开着影响后续用例。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_ai_train_infer_consistent_ops.py` | Unit tests for the whole train_infer_consistent_ops package / 覆盖整个 train_infer_consistent_ops 包 |

## Covered Areas / 覆盖范围

| Module / 模块 | What is covered / 覆盖内容 |
|------|-------------------|
| `inspect_util.py` | env snapshot, current-layer context, `_stats` (±0 normalization), `_squeeze_shape` / `_load_shape_ok` shape gate, `inspect_tensor` stages 1-7, `index=` into a tuple / list / dict / 环境快照、当前层上下文、`_stats`（±0 归一）、形状门禁、`inspect_tensor` 七个阶段、`index=` 取 tuple / list / dict 里的张量 |
| `permute.py` | canonical (token, expert) row order and its inverse, padding-row isolation / canonical 行序与逆变换、padding 行隔离 |
| `ffn_act.py` | unit routing weights, fp8 dequant, `1x128` requant recipe, the two `post_load_func` inverses / 强制单位 routing 权重、fp8 反量化、`1x128` 再量化配方、两个 `post_load_func` 逆变换 |
| `slice_util.py` | last-dim segment view and its inverse, and that neither the probes-off path nor a dumpless load touches the live buffer's inplace version / 末维分段视图与逆变换，以及关闭探针、无 dump 时都不动 live buffer 的 inplace version |
| composite probes / 组合探针 | the network-definition call sites spelled verbatim: dispatched activation, SwiGLU+quant output, fused gate logits (each one `inspect_tensor` + a pre/post pair) / 逐字照抄组网调用处：dispatch 激活、SwiGLU+量化输出、融合 gate logits（都是一个 `inspect_tensor` 加一对 pre/post） |
| `transformer/mlp.py` | `MLP.inspect_name` + `get_current_layer()` tag composition (`moe_shared_*` vs `dense_mlp_*`) / MLP 探针 tag 拼接（共享专家 vs dense MLP） |
