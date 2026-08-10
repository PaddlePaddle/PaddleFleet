# PaddleFleet 用户配置开关规则

- 适用路径：`src/paddlefleet/transformer/transformer_config.py`、`src/paddlefleet/model_parallel_config.py`
- 触发条件：在上述文件中新增、重命名或删除用户配置开关，修改开关默认值、取值集合或行为，或在 `__post_init__` 中新增/修改开关之间的约束
- 规则来源：`ci/check_approval.sh` 将这两个文件列为需指定审批人的受控配置文件；其余检查项来自两文件现有的命名分区、docstring 和 `__post_init__` 校验约定
- 引用约定：本文档只用符号名（字段名、函数名、分区注释）定位代码，不写行号
- 评审范围：只评审本 PR diff 中新增或修改的开关。存量开关（含已有的 `enable_*`、`use_*` 字段和重复声明）不因不符合本规则而报告，除非本 PR 正在改动它们。

## 规则优先级

必要性、跨版本兼容性、二次解析、校验与测试缺失优先于命名、分段与文档问题。生态一致性优先于本文档的命名偏好：上游框架已有等价开关时沿用上游命名，即使其形式与“命名与声明规范”一节冲突。

## 开关必要性

- 新增开关前用 `rg` 检索同一功能域的现有开关。若新开关的行为可由现有一个或多个开关的组合等效表达，应复用现有开关；若确需新增且旧开关因此变为冗余，必须在同一 PR 内删除旧开关并给出迁移方式，保持开关组合低冗余。
- 若开关实际只有一个取值会被使用，应在代码中实现固定分支而不是新增开关。判定依据：PR 未新增任何使用非默认取值的测试或配置，且 PR 描述未说明其他取值带来的功能点或性能收益。
- docstring 必须写清每个取值的适用场景。适用场景不清晰的取值视为非必要，报告时引用具体字段位置并指出缺失的说明。
- 仅用于定位问题的调试开关（例如后端对照、参考实现比对）必须在 docstring 声明它是临时开关还是长期用户接口；临时开关需说明回收条件。

## 生态一致性

为保持跨框架使用体验一致，对外部生态已有的功能点，开关命名、取值语义和默认行为应尽量与外部框架一致：

- 模型结构与算法策略相关开关对齐 Huggingface Transformers 或 ms-swift。
- 并行配置与性能优化策略相关开关对齐 Megatron-LM（本仓库配置类本身派生自 Megatron-LM，见 `transformer_config.py` 文件头的 `Referred to NVIDIA Megatron-LM` 说明）。

对齐方式：先确认该结构或策略最早由哪个模型/框架提出，再检索上游对应的配置项名称与默认值。若上游存在等价开关而 PR 使用了不同命名或不同默认值，报告时给出上游名称，要求对齐或在 docstring 说明差异原因。沿用上游命名但行为不同时，必须在 docstring 显式标注差异，不得让同名开关产生不同语义。

参考来源：

- Huggingface Transformers：https://github.com/huggingface/transformers 、https://github.com/huggingface/transformers/blob/8e7d47a325d647d9e78b832b8e003c9d676b658a/docs/source/en/custom_models.md?plain=1#L26
- ms-swift：https://github.com/modelscope/ms-swift 、https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html
- Megatron-LM：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py

## 跨版本兼容性

- 默认禁止不兼容变更：重命名开关、删除开关、修改默认值、改变同一取值下的行为。这类改动会让既有 YAML/JSON 配置行为静默改变。
- 确需变更时，同一 PR 必须对旧开关配置显式报错拦截，不允许旧配置静默失效。重命名或删除开关时把旧键登记到 `TransformerConfig.renamed_config_keys`（`_process_attribute` 会据此抛 `ValueError` 并给出迁移提示）；`_process_attribute` 的兜底分支是 `setattr`，未登记的旧键会被当成无用属性吸收，旧配置想打开的功能会静默保持关闭。特殊场景可参考 `sonicmoe_quant_format` 的单独 guard。
- 拦截必须无条件可达。不要把旧开关检查放在某个取值的条件分支内，否则其他取值下旧配置仍会静默通过；现有对 `csa_tilelang_backend` / `csa_tilelang_enable_indexer` / `csa_tilelang_enable_sparse_attn` 的拦截嵌在 `__post_init__` 里 `experimental_attention_variant == "dsv4_hybrid"` 的分支内部，属于反例。新增拦截应放在 `renamed_config_keys`、`_process_attribute` 或 `__post_init__` 顶层。
- 仅在 docstring 标注 `Deprecated` / `This flag is ignored` 而保留字段、不给运行时信号的做法，只适用于从 Megatron-LM 继承的存量字段（如 `moe_extended_tp`、`async_tensor_model_parallel_allreduce`），不得用于本 PR 新引入的废弃。
- 修改默认值必须在 PR 描述中说明对既有配置的影响，并与 PR 模板中“是否引起精度变化”一节一致。

## 命名与声明规范

- 与所在功能域分段中现有开关的命名风格保持一致，沿用既有功能域前缀（`moe_`、`csa_`、`dsa_`、`fp8_`、`cp_`、`tp_comm_`、`recompute_` 等）。
- 布尔开关使用名词或形容词短语命名，True 表示打开、False 表示关闭，例如 `gated_attention`、`csa_dense_mode`、`qk_norm_fusion`。避免 `enable_xxx` / `disable_xxx`：`enable` 与布尔 True 语义重复，`enable` 与 `disable` 并存会造成语义混乱。名词结构过短不足以表达时，用“功能域前缀 + 动宾”构成名词化短语，例如 `moe_dequant_input`、`train_indexer_only`。
- 传值开关用 `<功能域>_<对象>` 加类型后缀，例如 `_size`、`_ratio`、`_coeff`、`_backend`、`_mode`、`_type`。多个互斥实现用单个字符串枚举字段表达（参考 `csa_indexer_backend`），不要拆成多个互斥布尔开关。
- 配置命名空间是扁平的：`TransformerConfig` 继承 `ModelParallelConfig`，所有字段处于同一层命名空间。不要新增子配置 dataclass 制造嵌套。确需结构化取值时用 dict/list 字段，嵌套不超过两级，并在 docstring 列出全部合法键。
- 禁止重复声明同名字段。新增字段前用 `rg` 确认本类与基类中不存在同名字段；重复声明会让后一次声明静默覆盖前一次，且可能与 docstring 描述的默认值不一致。
- 不要新增下划线前缀字段作为用户可配置开关。

## 分段与放置

- 两个配置文件用分区注释按功能域分段（例如 `# model architecture`、`# mixed-precision`、`# fusion`、`# activation recomputation`、`# MoE related`、`# Context Parallel`、`# fp8`、`# MLA`、`# DSA`、`# CSA / DSv4 Hybrid Attention`）。新增开关必须插入到所属功能域的分段内、紧邻同类开关，不得追加到类体末尾、插到文件里第一个能编译通过的位置，或塞进不相关分段。
- 一组相关开关连续声明：主开关在前，依赖它的参数开关紧随其后，使依赖关系在阅读时即可看出。例如某功能的 backend 选择与该 backend 的参数应相邻，不要被无关字段隔开。
- 若新增开关属于一个尚不存在的功能域，应新增一条分区注释，并把该功能域的开关集中在这一段内，而不是分散插入多个已有分段。
- 评审时发现新增字段所在位置与其功能域不匹配，要指出应归入的分段名称，并说明分散声明会导致同类开关难以检索、后续容易重复新增等价开关。

## 反面模式：语义不清、结构混乱、配置复杂、二次解析

以下四类问题即使功能正确也应报告，并给出具体的替代设计。

- 命名语义不清：从字段名加 docstring 无法判断“取该值会发生什么”。典型表现：名字描述实现细节而非用户意图；同一概念在不同开关里混用 `_mode` / `_type` / `_variant` / `_backend`；缩写未在 docstring 展开；布尔名带否定语义导致双重否定。报告时给出建议命名和判断依据。
- 结构混乱：一个开关同时承担多个正交语义（既选实现又调参数），或多个开关的取值互相隐式改写。新增开关必须做到：正交语义拆成正交字段；组合非法时在 `__post_init__` 抛 `ValueError`，而不是静默把某个字段改写成兼容值；确有必要的自动改写要在 docstring 写明改写条件和结果。
- 配置复杂：启用一个功能需要用户同时正确设置三个以上开关，或需要用户了解内部实现（kernel 名、stage 编号、内部阈值）才能填对取值。这类情况应提供单一入口开关，其余参数给出可用默认值或从已有配置推导。
- 二次解析：开关取值必须在配置层完成归一化，禁止把多形态取值透传给消费方、让 trainer、model、layer 或算子封装各自再写一遍解析和分支。
  - 判定条件：同一字段接受多种类型（`bool | str`、`int | list`、`list | dict` 等）、需要按分隔符切分的字符串、或需要消费方补齐缺省键的 dict。
  - 现网反例：`recompute_modules` 声明为 `list[str] | dict`，`attention.py`、`moe/moe_layer.py` 等多个消费点各自重复 `isinstance(..., list)` / `isinstance(..., dict)` 分支；`moe_layer_freq` 声明为 `int | list`，在 `models/gpt/gpt_layer_specs.py` 里再次按类型分支。新增开关不得引入同类形态。
  - 要求：新增开关取值类型单一；确需兼容多形态输入时，在 `__post_init__` 中一次性归一化成唯一内部表示，消费方只读归一化后的字段；不得在配置类之外用 `split`、正则、`json.loads`、`literal_eval` 解析开关取值。
  - 评审动作：对新增开关用 `rg <switch_name>` 列出全部消费点，若出现类型分支、字符串切分或缺省键补齐等重复逻辑，报告并要求把归一化上移到配置层。

## 文档与默认值

- 每个新增字段紧随其后必须有 docstring，说明：语义、完整取值集合（枚举需逐项说明）、默认值及其选择理由、与其他开关的依赖或互斥关系。仅有字段声明而无 docstring 的新增开关必须报告。
- 默认值必须保持现有行为不变，新功能默认关闭。除非 PR 明确说明行为变更并评估精度影响。
- 若开关对应 Huggingface `config.json` 中的字段名，需在 `TransformerConfig.transform_rules` 中登记映射，否则从 HF 配置加载时该开关不生效。重命名已登记的开关时必须同步更新 `transform_rules`。

## 校验与测试

- 开关之间的依赖与互斥必须在 `__post_init__` 中校验并抛出 `ValueError`，不使用 `assert`（与基础规则一致）。错误信息要包含冲突的开关名、当前值和期望值。
- 新增或修改开关必须补充测试，覆盖默认取值路径、至少一个非默认取值路径，以及用 `pytest.raises` 验证非法组合：
  - `transformer_config.py` 的通用校验放在 `tests/single_card_tests/test_transformer_config.py`；特定功能的开关放在 `tests/single_card_tests/transformer/` 下对应功能的测试文件。
  - `model_parallel_config.py` 的校验放在 `tests/single_card_tests/ai_edited_test/distributed/test_ai_model_parallel_config.py`。
  - 开关影响并行行为时需要 `tests/multi_card_tests/` 下的代表性多卡覆盖。
- `TransformerConfig.from_config` 走 `object.__new__` 并手动调用 `__post_init__`，不经过 dataclass 构造函数。新增字段需确认在配置字典缺少该键时仍能取到声明的默认值，且 `__post_init__` 中对该字段的校验不会因缺键而异常。
