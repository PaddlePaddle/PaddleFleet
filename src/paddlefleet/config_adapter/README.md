# config_adapter

把面向大集群的训练 YAML 自动改写成能在**更小机器规模**上跑起来的配置：重算
sharding 与 batch，必要时缩小 EP/PP 并联动改写 `model_config.json`，并保证改写
后的配置满足 Fleet 的通信组约束。

`--test-performance` 与 `--test-accuracy` 是**两个正交的、可选的**测试维度，各管
一件事，可以单独给、同时给、也可以都不给：

| 组合 | 并行度 | acc | batch 策略 | 精度开关 |
|---|---|---|---|---|
| 都不给 | 需要时缩小 EP/PP | 不变 | `scale_batch` | 不注入 |
| `--test-performance` | 全部冻结 | 冻结 | `scale_batch` | 不注入 |
| `--test-accuracy` | 需要时缩小 EP/PP | 等比放大 | `scale_accumulation` | 注入 |
| 两个都给 | 全部冻结 | 冻结 | `scale_batch` | 注入 |

- `--test-performance` 负责「测出来的单步耗时可比」：冻结 TP/PP/EP/CP/SEP 与
  `gradient_accumulation_steps`，只动 sharding 和 `global_batch_size`。
- `--test-accuracy` 负责「结果可复现、不 aadiff」：注入确定性开关；只要没同时给
  `--test-performance`，就用放大 acc 的方式保持等效 batch。

两种组合都遵守同一条硬规则：**源配置里大于 1 的并行度，最小只能缩到 2，不允许
缩成 1**（EP8 最多缩到 EP2，不会变成 EP1）。把一个维度缩成 1 等于把待测的通信
组直接去掉，测出来的结果没有参考价值。唯一例外是 VPP：框架硬断言 VPP>1 需要
PP>2，PP 缩到 2 时 VPP 只能是 1。

TP 与 SEP 永远不改：减小 TP 会增大单卡显存占用，有 OOM 风险。EP 与 PP 都要缩时按
**EP 优先于 PP** 的顺序缩（缩 EP 只损失路由专家数，缩 PP 会连带改层数与逐层配置）。

---

## 用法

```bash
# 1) 只看这份配置能跑在哪些机器规模上（不生成任何文件）
python -m paddlefleet.config_adapter --input config.yaml

# 2) 只要能跑起来：适配到 2 台机器（默认每台 8 卡），必要时自动缩 EP/PP
python -m paddlefleet.config_adapter --input config.yaml --target-nodes 2

# 3) 测速：冻结并行策略与 acc，只改 sharding 和 GBS
python -m paddlefleet.config_adapter --input config.yaml \
    --target-nodes 2 --test-performance

# 4) 精度测试：注入避免 aadiff 的开关，并保持等效 batch
python -m paddlefleet.config_adapter --input config.yaml \
    --target-nodes 1 --test-accuracy

# 5) 两个维度同时给：既冻结并行策略，又注入精度开关
python -m paddlefleet.config_adapter --input config.yaml \
    --target-nodes 8 --test-performance --test-accuracy

# 6) 非 8 卡机型用 --cards-per-node 表达（这里是单机 2 卡）
python -m paddlefleet.config_adapter --input config.yaml \
    --target-nodes 1 --cards-per-node 2 --test-accuracy

# 7) 就地改写源文件，并额外生成 <input>.patch
python -m paddlefleet.config_adapter --input config.yaml \
    --target-nodes 1 --test-accuracy --in-place

# 8) 自定义字段：不带前缀时工具自己判断该改 yaml 还是 model_config.json
python -m paddlefleet.config_adapter --input config.yaml \
    --target-nodes 1 --test-accuracy \
    --set max_steps=10 --set n_routed_experts=32
```

改写 YAML 时用 `ruamel.yaml` 保留注释与字段顺序，它是 paddlefleet 的运行时依赖，随包一起安装，无需额外操作。

### 命令行参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--input` | 是 | — | 源 YAML 路径 |
| `--target-nodes N` | 否 | — | 目标机器台数；总卡数 = N × `--cards-per-node`。不传 = 只列出合法规模，不生成文件 |
| `--cards-per-node` | 否 | 8 | 每台机器的卡数 |
| `--test-performance` | 否 | 关 | 测速维度，可与 `--test-accuracy` 叠加 |
| `--test-accuracy` | 否 | 关 | 精度维度，可与 `--test-performance` 叠加 |
| `--output-dir` | 否 | `./adapted_configs` | 输出目录；`--in-place` 时忽略 |
| `--set [yaml:\|json:]KEY=VALUE` | 否 | — | 自定义覆盖，可重复 |
| `-i` / `--in-place` | 否 | 关 | 就地改写源文件，并生成 `<input>.patch` |
| `-f` / `--force` | 否 | 关 | 允许覆盖已存在的 model_config 生成目录 |

`--set` 的前缀是可选的：

- `yaml:KEY=VALUE` / `json:KEY=VALUE`：显式指定改哪个文件；
- `KEY=VALUE`：工具扫描两个文件自己判断 —— 谁声明了这个 key 就改谁，两边都声明
  就**两边都改**（框架两边都读），两边都没有则**默认新增到 yaml**；想把新字段加到
  `model_config.json` 必须显式写 `json:KEY=VALUE`。

取值类型自动推断（整数→int、`true/false`→bool、`null/none`→None、其余→字符串）。
被 `--set` 指定的字段是**受保护**的：后续任何自动规则都不会再覆盖它。但由适配器
统一计算的字段（TP/PP/EP/CP/SEP、VPP、`num_empty_layers_add_in_tail`、
`sharding_parallel_size`、`data_parallel_size`、`model_name_or_path`）**不允许**
用 `--set` 锁定（带不带前缀都一样），否则生成的文件会与报告里的并行度对不上 ——
遇到这种组合会直接报错。`model_name_or_path` 即使在 `--in-place` 下也不允许锁定：
改它会决定「到底加载/缩容/快照哪一份 `model_config.json`」。当某个模型
结构字段在 `model_config.json` 里读不到（不同模型家族命名不一致）时，也用
`--set <字段>=<值>` 兜底。

源作业卡数按两条证据推断并交叉校验：通信组
（`DP × sharding × TP × SEP × PP`）与 batch 字段
（`GBS / (micro_bs × acc) × TP × SEP × PP × CP`）。两者都能算且不一致时，取「没漏因子」
的那个 —— 源 YAML 声明了 `data_parallel_size` 就用通信组，没声明则用 batch 字段
（未声明的 DP 正是通信组公式缺的那一项）—— 并在报告里给出 WARNING。

C3/C5 只取决于并行度本身，与卡数无关：这两项冲突时不会再列出「合法节点数」，而是
直接说明必须先改 TP/SEP/EP/CP 的组合。

`model_config.json` 的位置永远从 YAML 的 `model_name_or_path` 推断：绝对路径直接
用，相对路径先按 YAML 所在目录、再按当前工作目录（Fleet 的解析方式）尝试。

---

## 适配流程

```
加载 YAML
  -> 应用 --set yaml:（并锁定这些字段）
  -> 删除 fa_version（与环境强绑定的 flash-attention 版本 pin）
  -> 需要时加载 model_config.json 并应用 --set json:
  -> 扫描两个文件，落定不带前缀的 --set
  -> 规划最终 TP/PP/EP/CP/SEP（冻结或缩容）
  -> 写入 model_config.json 的结构改动（专家数 / 层数）
  -> 注入精度开关（给了 --test-accuracy 时）
  -> model_config.json 有改动时另存一份，并把 model_name_or_path 指过去
  -> 应用并行度改动并复核 C1..C4
  -> 缩放 batch、sharding、data_parallel_size
  -> sharding 路数变小时注入补偿开关（数据流路数 / 优化器 offload）
  -> 写出 YAML 并打印报告
```

不冻结并行度时（默认，或只给了 `--test-accuracy`），按「改动越小越优先」的顺序
尝试，命中第一个可行方案就停：

```
维度已合法  <  只缩 EP  <  只缩 PP  <  EP+PP 联合缩
```

- 缩 EP：专家数等比缩减，要求每个 EP rank 专家数相等（M1）且不小于 top-k（M2）。
- 缩 PP：`num_hidden_layers` 等比缩减，再用 VPP/空尾层重新对齐（M3）；层数低于
  推荐值只告警不拒绝（M4）；dense 前缀层必须放得下（M5）。同时**逐层配置列表**
  （`csa_compress_ratios` / `window_attn_skip_freq` / `layer_types` / 列表形式的
  `moe_layer_freq`）按新层数裁剪：保留前 N 层，并原样接回末尾的 MTP 项，否则框架会
  因长度不符直接启动失败。源列表长度本身与 `num_hidden_layers` 不一致，或裁剪会让
  `csa_compress_ratios` 丢掉某一类注意力层（框架要求每类至少一层）时，该候选会被
  拒绝而不是硬改。
- 联合缩：EP × PP 笛卡尔积，按 **EP 优先于 PP** 的缩容顺序取点 —— 先把 EP 压到可行
  下限，再取该 EP 下最大的 PP。缩 EP 只损失路由专家数，缩 PP 会连带改层数、逐层配置、
  VPP 和空尾层，所以宁可多缩 EP、少缩 PP。

两条按框架真实行为兜住的不变量：

- `mtp_shared_last_layer` 为真且有 MTP 层时，末层 decoder 与 MTP 层通过
  `SharedLayerDesc` 共享 attention 权重，两者注意力类型必须一致；裁剪后末层
  `csa_compress_ratios` 会被**改写成** MTP 项的值，否则共享权重跨 stage broadcast
  时几何不匹配，训练直接挂死（不报错）。
- VPP>1 需要 PP>2（框架硬断言 `virtual pipeline must run under pp degree > 2`）。
  PP 缩到 2 时 VPP 一并置 1，冻结 / 只缩 EP / 维度已合法这几条不动 PP 的路径同样会
  兜一次，避免把源里 PP≤2 且 VPP>1 的组合原样透传成启动即挂的配置。

写盘时机：所有规划与校验都在内存里完成后才落盘，中途任何一步失败都不改动源文件；
写文件走临时文件 + 原子替换，`--in-place` 下若 YAML 写失败会回滚已写的
`model_config.json`。

### 约束体系

| 族 | 约束 | 含义 |
|---|---|---|
| C1 | `N % (TP·SEP·PP) == 0` | sharding 为正整数 |
| C2 | `N % (PP·EP) == 0`（EP>1） | moe_sharding 为正整数 |
| C3 | `EP % (TP·SEP) == 0`（EP>1） | dense_sharding = EP/(TP·SEP) 为正整数 |
| C4 | `sharding % CP == 0` | cp_sharding 为正整数 |
| C5 | 不允许 SEP>1 且 CP>1 | 框架禁止 sep 与 context 并行同时使用 |
| E1 | 跨机必须整节点分配 | Fleet 的 rank 映射要求各机对称 |
| E2 | `TP·SEP` 放得进单机 | TP/SEP 不缩，缩了有 OOM 风险 |
| E3 | 所有并行度是 ≥1 的整数 | — |

C/E/M 全部不通过时，报错会给出候选淘汰明细和最近的合法机器规模。

---

## 精度开关（避免 aadiff）

给了 `--test-accuracy` 就会把下面这些开关钉到确定性实现上；键在 YAML 和
`model_config.json` 里都声明时，两份都会改（框架两边都读）。

| 目标 | 字段 | 值 | 原因 |
|---|---|---|---|
| yaml | `csa_sparse_attn_backend` | `tilelang` | HCA/CSA 的 cudnn 后端（FlashMLA 前向 + cuDNN 反向）与 tilelang 结果不一致 |
| yaml | `csa_indexer_backend` | `tilelang` | indexer 的 cudnn top-k 与 tilelang 有差异，与上一项取齐 |
| yaml | `mqa_sparse_attn_backward_backend` | `tilelang` | absorbed-MQA（`csa_compress_ratios=-2`）层的 dKV 反向默认走 cuDNN，原子累加不可逐位复现 |
| json | `multimax_modules` | `null` | multimax 的可学习 range/ts 引入非确定性 |

新增开关只需在 `precision.py` 的 `PRECISION_SWITCHES` 里加一行，报告与日志会自动
带上原因。

---

## sharding 缩容的补偿开关

卡数变少 → `sharding` 变小，有两件事按缩容倍数一起变差：数据流被切成
`dataset_world_size = sharding / CP` 份，每份要走更长的文件列表才拿到第一个 batch；
优化器状态按同一个度切分，单卡优化器显存按倍数放大。目标路数比源窄时自动注入：

| 字段 | 值 | 原因 |
|---|---|---|
| `debug_reeao_dataset_world_size` | 源路数 | 把**数据切分**的份数固定回源规模（rank 仍是真实 rank，各 rank 读到的切片依旧互不重叠） |
| `tensorwise_offload_optimizer` | `true` | 优化器状态放到 host 内存，防止单卡放大若干倍后 OOM |

offload 有一条全由框架硬报错撑起来的依赖链，按顺序一并关掉；只改源里**显式声明且
取值冲突**的键（框架默认 `false` / `0` 本来就兼容），`--set tensorwise_offload_optimizer=false`
时整条链都不动：

```
tensorwise_offload_optimizer=true
  -> fuse_optimizer_states=false        # fuse 与 offload 冲突，框架直接报错
  -> enable_zero_cost_checkpoint=false  # ZCC 断言 fuse_optimizer_states=true
  -> flash_device_save_steps=0          # >0 断言 enable_zero_cost_checkpoint=true
```

线上 YAML 常把 `global_batch_size` 注释掉、也不写 `sharding_parallel_size`，源规模推
不出来；这两个开关在那种配置上同样必要，所以此时按「源配置自己的 `dense_sharding`
（= `EP/(TP·SEP)`，实际加载数据的卡数）× 96」估计源路数。路数往大估只会让每路切片更
短，没有代价。

---

## 输出

只列**真正发生了变化**的字段（值没变的写入不会出现在日志里），每条都带上
「为什么改」，并保留可被上游脚本正则匹配的形状（`CHANGE field=… old=… new=…` /
`ADD` / `DELETE`，以及末尾的 `ORIGINAL_CARDS=` / `TARGET_CARDS=` / `OUTPUT=` /
`MODEL_CONFIG_OUTPUT=`）：

```
========================================================================================
config_adapter: 适配成功
========================================================================================
输入      ：big.yaml
模式      ：accuracy（--test-accuracy），batch 策略 scale_accumulation
机器规模  ：96 节点 / 768 卡 -> 1 节点 / 8 卡（每节点 8 卡）
并行度    ：TP 1->1  PP 8->4  EP 64->2  CP 1->1  SEP 1->1
sharding  ：96 -> 2（moe_sharding=1, dense_sharding=2）
缩容方案  ：EP+PP 联合缩容：EP 64 -> 2，PP 8 -> 4

--- YAML 改动（9 项） ------------------------------------------------------------------
写入      ：adapted_configs/big_adapted_8cards.yaml

  CHANGE field=expert_model_parallel_size old=64 new=2
      原因：缩小 EP 64 -> 2：单独缩 EP 或单独缩 PP 都无法适配 8 卡，改为联合缩容；按 EP
            优先于 PP 的缩容顺序，先把 EP 压到可行下限，再尽量少缩 PP（缩 PP 会连带改层
            数/VPP/尾部空层/逐层配置）

  CHANGE field=gradient_accumulation_steps old=2 new=192
      原因：保持等效 batch：GBS 保持 1536 不变，acc 放大为 2 × 768 / 8 = 192

  ADD field=tensorwise_offload_optimizer new=True
      原因：sharding 路数 96 -> 2，单卡优化器状态放大 48 倍，offload 到 host 内存防 OOM

  ...

--- model_config.json 改动（3 项） -----------------------------------------------------
写入      ：adapted_configs/model_config_separated/model_dir_adapted_8cards/model_config.json

  CHANGE field=n_routed_experts old=256 new=8
      原因：随 EP 64 -> 2 等比缩减专家数 256 -> 8（>= top-k=8）

  ...

--- 机器可读摘要 -----------------------------------------------------------------------
ORIGINAL_CARDS=768
ORIGINAL_NODES=96
TARGET_CARDS=8
OUTPUT=adapted_configs/big_adapted_8cards.yaml
MODEL_CONFIG_OUTPUT=adapted_configs/model_config_separated/model_dir_adapted_8cards/model_config.json
```

---

## 目录结构

```
config_adapter/
├── cli.py                    # 参数解析 + main()
├── core.py                   # ConfigAdapter：编排整个适配流程
├── options.py                # AdaptOptions：两个正交开关的派生行为
├── planner.py                # 并行度规划：冻结 / EP-PP 分层缩容
├── plan.py                   # ParallelismPlan
├── precision.py              # 精度开关表
├── sharding_shrink.py        # sharding 缩容的补偿开关
├── strategies.py             # scale_batch / scale_accumulation
├── topology.py               # C1..C4 通信组约束
├── constraints.py            # E/M 族约束 + 候选枚举 + 层对齐
├── layer_fields.py           # 逐层配置列表的裁剪与校验
├── field_spec.py             # model_config.json 字段别名注册表
├── model_config_resolver.py  # 定位/改写 model_config 路径
├── io_writers.py             # YAML / JSON 读改写
├── report.py                 # 改动记录与报告渲染
└── utils.py                  # 通用小工具
```

## 测试

```bash
pytest -s tests/single_card_tests/config_adapter/test_config_adapter.py
```
