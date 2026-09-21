# AOA 模块化生成

## 1. 背景

原来每接一个模型，就要在它的 `modeling.py` 里手写一整套 AOA 生成函数：按层循环与层号
offset、前缀与子模块名替换、Linear 转置、QKV / FFN / 专家融合拆分、MTP 等特殊结构、PP
与 EP 下的 key 映射，正向逆向各一份。

模块化 AOA 不再为每个模型手写一份整模 AOA 生成器，而是让组件自己产出 AOA 语句，并沿真实
的 Layer 模块树递归组装。

## 2. 架构概览

生成能力由三层承担：

1. **整模入口**：`GPTModel.gen_(inv_)aoa_statements`（`models/gpt/gpt_model.py`），对外
   唯一 API。
2. **上下文与工具**
   - `AOAContext` / `resolve_names`（Paddle `flex_checkpoint/aoa/generation.py`）：只读
     上下文与名称解析。
   - `models/gpt/aoa_generator.py`：建上下文、遍历流水线单元、流水线全局化。
3. **组件方法**
   - `Layer.gen_aoa_statements` / `gen_inv_aoa_statements`（Paddle 基类）：沿模型树的基
     础递归。
   - Fleet 各组件 override 实现特殊布局。

两个方向**各自独立生成**，逆向不从正向的语句文本反推：

| 方法 | 方向 | 语句形状 |
|---|---|---|
| `gen_aoa_statements` | checkpoint -> model（加载） | `ckpt_name -> model_name` |
| `gen_inv_aoa_statements` | model -> checkpoint（保存） | `model_name -> ckpt_name` |

### 调用链

```
model.gen_aoa_statements(config=None)               # GPTModel：整模入口
  → build_aoa_context(model, config)                # 构建 AOA 上下文（aoa_generator.py）
    → config.aoa_checkpoint_name_mapping or DEFAULT_CHECKPOINT_NAME_MAPPING
    → config.aoa_checkpoint_name_prefix or DEFAULT_CHECKPOINT_NAME_PREFIX
    → validate_checkpoint_name_mapping(...)         # 模板一次性校验
    → AOAContext(...)                               # frozen，全程原样下传
  → gen_whole_model_aoa(model, ctx)                 # 收集本 stage 的语句
    → _iter_pipeline_units(model)                   # 活着的顶层子层（跳过 None）
    → unit.gen_aoa_statements(ctx, structured_name_prefix=...)
      → Layer.gen_aoa_statements                    # 基类恒等递归
        → state_dict(include_sublayers=False)       # 本层张量，与 sharded_state_dict 同源
        → resolve_names(...) -> (checkpoint_name, model_name)
        → 两名相同则不发语句
        → 逐 self._sub_layers 递归，逐位置参数原样下传
      → 组件 override                               # 融合 / 转置 / 丢弃等
    → _globalize_statements(...)                    # 按流水线组 all_gather_object 拼接
```

逆向链路同形，入口为 `gen_inv_aoa_statements` → `gen_whole_model_inv_aoa`。

两名相同就不发语句，依据是引擎的收口行为：没有被任何语句消费的源张量会按原名直接进入输
出。因此语句集合恰好等于「真正需要变换的张量」，规模远小于参数总数。

### 只读上下文

`AOAContext` 是 frozen dataclass，整个生成过程只建一次：

| 字段 | 含义 |
|---|---|
| `config` | 当前结构的 config |
| `model_name_prefix` | 模型侧单卡名的根前缀，由模型声明 |
| `checkpoint_name_prefix` | checkpoint 侧共享根前缀，恒等回退时使用 |
| `checkpoint_name_mapping` | 绝对模型名模板 -> 绝对 checkpoint 名模板 |
| `pp_to_single_mapping` | 结构名 -> 单卡名 |

## 3. 名字解析：三步，一个函数

一个张量有三个名字：structured name（活模块树里的路径，随 PP / VPP 切分而变）、single
name（模型侧的单卡规范名，也是语句里模型侧的最终名）、checkpoint name。`resolve_names`
是组件唯一需要调用的名字入口，内部三步：

1. **structured name -> single name**：`structured_name_prefix + local_name` 过
   `pp_to_single_mapping` 查表。
2. **（可选）丢掉一段路径**：把 `checkpoint_lookup_drop_segment` 指定的那一段从名字里去
   掉，用于适配 MTP 内层 transformer_layer 参数。
3. **single name -> checkpoint name**：命中 `checkpoint_name_mapping` 时 value 就是最终
   的 checkpoint 名；未命中走恒等回退，只把模型根换成 checkpoint 根。

模板两侧都可以用占位符 `$LAYER_ID` / `$EXPERT_ID`。建 ctx 时
`validate_checkpoint_name_mapping` 会一次性校验：非空、key 必须带模型根、`$` 必须是整段
且是已知占位符、value 用到的占位符必须被 key 捕获。

### 示例

以 MTP 块内层的某个专家权重为例，MTP 块为整个子树声明了 drop segment
`transformer_layer`，命中的是 `aoa_generator.py` 默认表里这条为普通层写的条目：

```python
DEFAULT_CHECKPOINT_NAME_MAPPING = {
    # ...
    "model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight":
        "model.layers.$LAYER_ID.block_sparse_moe.experts.$EXPERT_ID.w2.weight",
    # ...
}
```

| 步骤 | 名字 |
|---|---|
| single name（模型侧最终名） | `model.layers.3.transformer_layer.mlp.experts.5.down_proj.weight` |
| 查表输入（丢掉 `transformer_layer`） | `model.layers.3.mlp.experts.5.down_proj.weight` |
| 命中条目的 key | `model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight` |
| 渲染出的 checkpoint 名 | `model.layers.3.block_sparse_moe.experts.5.w2.weight` |

最终语句就是这两个名字相连，模型侧保留 `transformer_layer`，checkpoint 侧不带。
`down_proj` 是 `RowParallelLinear`，Linear 家族的 weight 一律带 `^T`：

```
# 正向（加载）
model.layers.3.block_sparse_moe.experts.5.w2.weight^T -> model.layers.3.transformer_layer.mlp.experts.5.down_proj.weight

# 逆向（保存）
model.layers.3.transformer_layer.mlp.experts.5.down_proj.weight^T -> model.layers.3.block_sparse_moe.experts.5.w2.weight
```

checkpoint 里独有、模型侧没有对应张量的名字（融合前的 Q/K/V、gate/up 等）用
`resolve_checkpoint_name_from_anchor` 从一个真实张量（anchor）推出来，这类名字不过
`pp_to_single_mapping`。

## 4. 模型侧声明

模型可以在传入的 config（provider）上声明两个覆盖点：

| 属性 | 作用 |
|---|---|
| `aoa_checkpoint_name_mapping` | 替换默认映射表（**整体替换**，要保留默认条目需自己并进去） |
| `aoa_checkpoint_name_prefix` | 替换 checkpoint 根前缀 |

未声明时用 `aoa_generator.py` 里的 `DEFAULT_CHECKPOINT_NAME_MAPPING` /
`DEFAULT_CHECKPOINT_NAME_PREFIX`。默认表只收**命名分叉**：组件已经能产出相同名字的张量
不在表里；层级根不带 `transformer_layer`，所以同一条条目既覆盖普通层也覆盖 MTP 块的内层
transformer；输出头是唯一 value 落在 checkpoint 共享根之外的条目。

## 5. 接入一个组件的规范

1. 按需生成：布局与恒等一致的组件不写代码。
2. 两个方向各写一个 override，签名固定为
   `(self, ctx, *, structured_name_prefix="", checkpoint_lookup_drop_segment=None)`。
3. 逆向独立实现，不从正向语句文本反推。
4. 只描述本组件自己的张量；子层交给递归，`structured_name_prefix` 与
   `checkpoint_lookup_drop_segment` **原样下传**。
5. checkpoint 名一律由 `resolve_names`（或 anchor 版本）得到，不在组件里第二次拼
   checkpoint 根。
6. 不读 rank、不读 shape、不做 dtype 转换：语句只描述名字与结构上的变换。
