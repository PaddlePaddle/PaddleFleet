# PaddleFleet 单元测试编写与评审规范

按仓库的主要模块组织 PaddleFleet 单测要求：数据层、模型层、Trainer、分布式训练、优化器、计算优化、Checkpoint、推理、配置与运行基础设施。采样、路由、回调、具体算法等作为所属模块内的测试要点，不单独划分章节。

本文不设置统一数值误差、覆盖率、用例数量或完成率门槛。具体数值测试仍需说明独立参考及适合所测 dtype、后端、精度目标的比较方式。例子用于说明如何发现业务错误，不要求照抄其输入或组织形式。


## PaddleFleet仓库模块

生产路径省略 `src/paddlefleet/` 前缀，标注“仓库根”的除外。同一目录中的功能按实际职责归属；`utils/` 随消费者归类。当前仓库数据存储主要是文件与索引，未发现独立 SQL 数据库服务，不套用事务、SQL 或连接池规范。

| 仓库模块 | 关键模块 | 职责范围与边界 | PaddleFleet 对应代码 |
| --- | --- | --- | --- |
| 数据层 | **开源数据流**：Dataset、Sampler、Collator、DataLoader、Tokenizer、Processor | 从开源数据集、本地文件或远端存储读取数据，完成预处理、采样、打包及模型输入构造；数据来源不改变归属 | `data/`、`datasets/`、`utils/batch_sampler.py`、`transformers/` 中 tokenizer/processor；`cli/train/sft/make_data_utils.py`、`dataset_formatting.py` 等数据工具 |
| 模型层 | Model、Attention、Loss、MoE、PEFT | 模型结构、参数连接、前后向、训练目标、专家数学与适配器；分布式通信和加速 kernel 按对应模块规则验证 | `models/`、`transformer/`、`transformers/`、`nn/`、`peft/`，以及 CLI 目录中的模型实现 |
| Trainer 训练引擎 | **Trainer**、Callback、SFTTrainer、DPOTrainer、Tuner、Workflow | 训练循环、梯度累积、状态推进、回调与评估；tuner/workflow 选择训练任务、组装模型和数据并启动 Trainer；任务内损失计算同时适用模型层规则 | `trainer/trainer.py`、`trainer/trainer_callback.py`、`cli/train/tuner.py`、`cli/train/*/workflow.py`、`cli/train/sft/sft_trainer.py`、`cli/train/dpo/dpo_trainer.py` |
| 分布式训练 | ParallelState、TP、PP、CP、EP、Dispatcher | 并行拓扑、参数/序列分片、通信、流水线与专家并行；配置对象中真实执行并行初始化的部分也适用本模块规则 | `tensor_parallel/`、`pipeline_parallel/`、`distributed/`、`parallel_state.py`、`context_parallel_utils.py`、MoE dispatcher |
| 优化器 | Optimizer、GradClip、LRScheduler、Muon | 参数更新、梯度消费、优化器状态与 Muon 切分；scheduler 被 Trainer 推进的时机按训练引擎规则验证 | `utils/optimizer.py`、`utils/moe_hybrid_parallel_optimizer.py`、`transformer/muon_utils.py` 及参数 spec、`transformers/optimization.py` |
| 计算优化 | FP8/Quantization、Fused Ops、Recompute、Offload | 低精度、融合算子、后端加速、重计算及显存管理，核对目标数值与资源生命周期 | `fp8/`、`quantization/`、`triton_ops/`、`triton_kernels/`、`tilelang_ops/`、`cudnn_ops/`、`fusions/`、`train_infer_consistent_ops/`、`refined_recompute/`、`trainer/utils/offload_optimizer.py`、仓库根 `packages/paddlefleet_ops/` |
| Checkpoint 与权重管理层 | Checkpoint、Serialization、Reshard、MergeKit | 状态持久化、加载、重分片、恢复与模型权重合并；恢复后训练推进交由 Trainer 验证 | `trainer/unified_checkpoint/`、`trainer/utils/zero_cost_checkpoint.py`、`trainer/utils/reshard/`、`mergekit/`、配置保存接口 |
| 推理层 | Generation、LogitsProcessor、KV Cache | 解码策略、生成约束、缓存、结束状态及请求隔离 | `generation/`、模型的 cache/生成接口 |
| 配置与运行基础设施 | **training_args**、**CLI**、Hparams、ConfigAdapter、Launcher、AutoConfigurator、CI | 参数定义、默认值、归一化、校验及配置传播；CLI 负责命令分派，launcher 负责环境和进程启动；搜索与 CI 负责运行结果处理。training_args 中 flags/Fleet 初始化同时按对应执行模块验证 | `trainer/training_args.py`、`trainer/argparser.py`、`cli/cli.py`、`cli/launcher.py`、`cli/hparams/`、`config_adapter/`、`training/`、`transformer/transformer_config.py`、`model_parallel_config.py`；仓库根 `auto_configurator/`、`ci/`、`scripts/`、`.github/workflows/` |

主表按职责划分模块，不按目录整体归类。`training_args.py` 位于 `trainer/`，主归配置与运行基础设施；`cli/` 内的数据、模型、通信和 kernel 分别归其所属模块，训练 workflow 归训练引擎。混合文件按本次修改的函数和行为选择规则。

跨模块变更检查连接处的契约。例如数据模块保证标签与文档边界正确，模型模块保证正确消费；Checkpoint 保证状态还原，Trainer 保证按恢复后的状态继续推进。局部数学单测、模块编排测试和真实多卡验证各自证明对应范围的行为。

## 测试环境描述

- **无卡测试**：不使用加速卡，以 CPU、临时文件和本地进程验证可在 CPU 执行的逻辑；不等于没有 Paddle 依赖。代码若只能在特定设备运行，可以无卡验证其控制逻辑，但不能宣称已验证设备数值。
- **单卡测试**：在一张 H20 GPU 上执行测试
- **多卡测试**：在多张 H20 GPU 上执行测试，使用目标业务所需的真实多卡进程组与 launcher，卡数服从所测并行拓扑，不统一规定所有模块的卡数；无卡多进程和 mock collective 不属于多卡数值测试。

下面每个模块都列出三个环境的适用范围和证据要求。按变更触发相应环境，不要求每次修改都运行三套测试。代码路径不支持某种环境时说明不适用及依据；环境缺硬件/依赖属于未运行，不属于业务不适用。已有支持条件和 skip 应保留，不能用“通过”概括全 skip。

各模块的反模式统一汇总在 [典型单测反例](unit-test-antipatterns.md)，包含反例代码、漏检原因和修正写法。所示代码为代表性示意，输入和比较参数只适用于对应示例；评审仍需核对真实入口与独立预期，并按本节要求选择测试环境。

## 数据层

**测试目标：** 数据内容、样本身份和监督信息经过存储、编码、采样及批处理后仍正确对应，最终输入符合模型约定。

- 使用内容与长度可区分的小样本，检查 token、label、position、attention mask、文档边界和样本顺序。SFT 打包与 padding-free 变化应验证这些字段同步变换，padding 不进入监督，局部文档 mask 不发生跨文档串扰。按所选 packing 策略判断分组，不要求不同算法输出相同排列。
- 文件与索引变化用临时目录完成真实写入、关闭、重新打开和读取，检查内容、dtype、长度、文档边界与偏移。`MMapIndexedDataset.get` 返回 token 和可选 loss mask，字节偏移与 token 偏移需分别推导；不能只检查文件存在。
- 采样测试观察完整样本序列、尾批策略与各 rank 分配。当前 sampler 可以重复末尾样本补齐分片；同 seed 的新实例按确定性契约复现，同一实例的随机状态会推进。排序按真实 key 判断，不能把默认按样本长度排序当成按索引排序。
- 分词验证特殊 token、词表映射、截断/padding 及保存重载；decode 按对应词表规范化语义判断。多模态输入验证文本占位符与图片/帧特征、网格元数据及 batch 顺序的对应，具体协议跟随所改 processor。
- 隔离下载和外部数据服务，使用本地小词表、内存数据集或合成媒体。若测试编码、预处理或 I/O 本身，应保留真实实现；组合层可替换其非被测依赖。空样本、无有效监督样本的行为按所测入口处理，不能统一假定返回空 tensor。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 使用内存样本、临时索引文件、本地词表或合成媒体运行真实读取、采样、编码和 collate；检查内容、顺序、标签、位置及 mask。需要 Paddle 的 CPU 计算仍需安装相应依赖，不用 mock 被测转换来规避依赖。 |
| 单卡 | 修改上卡、异步预取、设备侧 padding/mask 或输入消费接口时，在实际设备检查 dtype、device、内容与小模型消费结果；保留真实搬运/计算。只改纯数据规则且无设备差异时可复用无卡证据。 |
| 多卡 | 修改分布式采样或加载时，以真实 rank 运行并收集样本身份，检查分片、补齐、epoch/随机状态及整体消费对应；允许当前策略规定的末样本重复。CPU 上手算 shard 不能证明设备通信或分布式加载已执行。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，数据层常见于只数批次而遗漏样本顺序与身份、多模态对应关系未核对、保存重载只检查文件存在。

**常规测试位置：** `tests/formers/data/`、`tests/formers/dataset/`、`tests/formers/transformers/` 中 tokenizer/processor 测试。

## 模型与训练目标

**测试目标：** 配置构建出的模型结构、参数连接、前后向及训练目标符合该模型与算法的定义。

- 使用小配置、本地初始化权重和独立基础运算建立参考，检查层装配、输出结构、权重共享与可选分支。不同层使用可区分内容，避免只靠参数数量或模型能实例化判断正确。
- 注意力用可区分的 Q/K/V、mask 和位置验证可见范围、输出布局及相关梯度；RoPE 检查位置偏移、旋转范围和未旋转部分。参考路径与目标实现的 dtype、mask、精度模式和必要 cast 应一致。
- loss 从实际入口确认 label shift、有效 token 与归约方式。`LanguageLoss.forward_impl` 消费已对齐标签，部分模型内部移动标签；token 平均与逐样本平均不能共用公式。DPO 检查 chosen/rejected 与 policy/reference 的对应，KTO 按非成对类别及 KL 分支推导，避免把所有偏好目标套成同一关系。
- MoE 路由和专家数学在本模块验证选择、原始门控值、归一化、缩放及专家输出。用于选择的 correction bias 不应被误当成最终权重；开启 routed scaling 后不能统一要求权重和为一。跨 rank 分发与还原按分布式训练规则验证。
- 训练路径按名称核对预期参数集合，再比较输出和相关梯度，避免 `zip` 截短漏检。冻结或未参与计算的参数可以没有梯度，融合分支可能用 `main_grad`。PEFT 同时验证目标层适配、基础权重冻结和 adapter 更新；LoRA 的 merge、unmerge、disable 按实际状态语义分别检查。
- 隔离大权重下载与非被测模块，保留真实被测层或 criterion。人工 logits 可以验证 loss 数学，不能单独证明模型到 loss 的标签传递正确；接口变化需补相应连接验证。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 对支持 CPU 的小模型/层/criterion 保留真实前后向，检查结构、参数名、共享/冻结、损失定义及独立参考；配置、module spec 可用轻量对象验证。只能在 GPU 运行的数学不以替身输出充当无卡数值验证。 |
| 单卡 | 运行本次涉及的真实模型层、loss 或 adapter，检查输出、相关梯度及必要参数更新；选择实际使用的 dtype、mask、可选分支和精度模式。按真实梯度落点检查 grad/main_grad，不要求冻结参数都有梯度。 |
| 多卡 | 修改并行模型、共享权重、专家或 loss 跨 rank 归约时，在受支持 TP/PP/CP/EP 配置下运行真实模型，以相同初始权重和数据的完整参考核对局部/还原输出及梯度。纯局部模型改动不机械遍历所有并行组合。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，模型层常见于只检查形状和存在性、测试内重写生产公式自证、空用例或异常吞掉真实回归。

**常规测试位置：** `tests/single_card_tests/model/`、`tests/single_card_tests/models/`、`tests/single_card_tests/transformer/`、`tests/single_card_tests/moe/`；`tests/formers/transformers/`、`tests/formers/nn/`、`tests/formers/peft/`。

## Trainer 训练引擎

**测试目标：** 训练循环在正确时机消费数据、累积梯度、更新参数并推进训练状态，日志、评估、保存和停止行为与配置一致。

- 使用相同初始权重和相同有效样本，将累积微批与等价参考训练比较，观察优化器消费时的梯度及更新后参数。当前普通路径先对未除累积因子的 loss 做 backward，再在更新前平均 `.grad` 或 `main_grad`；日志 loss 单独缩放，不能强制规定缩放必须发生在 backward 前。
- 结合 `global_step`、epoch 和样本消费位置检查训练推进。普通更新、AMP overflow 和冻结路径分别验证 scheduler 推进与梯度清理；`global_step` 变化不代表参数一定成功更新。
- 回调测试检查事件、control flags、动作触发和状态重置，再验证 Trainer 如何消费动作、传递评估结果。区分微批事件与优化器更新事件，修改日志时检查对应更新窗口及无更新窗口。
- 恢复训练检查已恢复状态如何决定数据位置与后续调度；`ignore_data_skip` 按其配置语义验证。对照连续训练与中断续训时保持 scheduler 总计划一致，持久化内容由 Checkpoint 测试负责。
- 使用小型真实模型和内存数据集验证训练数值；外部日志平台、远程存储和与本次变更无关的耗时 evaluate/save 可替换。只 mock `training_step` 可以观察编排，不能证明梯度累积正确。DPO 等专用 Trainer 还应保留实际参考/策略阶段切换与样本对应逻辑。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 以真实状态类、callback 和控制逻辑验证事件、flags、日志/评估/保存触发及恢复进度计算；可替换耗时模型和外部服务，检查传递的状态。调用记录仅证明编排，不证明 backward、梯度缩放或更新数值。 |
| 单卡 | 用小型真实模型和内存数据执行训练循环，检查累积梯度、参数更新、scheduler、清梯度和样本推进；修改 AMP/冻结分支时验证对应跳步行为。不能把仅回调单测或 training_step 替身当训练数值验证。 |
| 多卡 | 修改分布式 Trainer 时，用真实 launcher 执行相关训练流程，按 rank 职责检查训练状态、数据消费、梯度同步及日志/保存归属；涉及续训时核对分布式恢复后推进，确保任一 worker 失败可传播。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，Trainer 常见于测试内重写编排逻辑、回调只检查被调用、拼批只检查形状、绕过真实初始化。

**常规测试位置：** `tests/formers/trainer/`，回调可参考 `test_trainer_callback.py`。

## 分布式训练

**测试目标：** 并行拓扑、数据与参数分片、通信和执行调度保持完整训练语义，各 rank 正确接收其结果和梯度。

- 从实际 rank 所有权推导分片和通信前后的布局，各 rank 使用不同内容及上游梯度。TP 的 Gather 与 AllGather 可能具有不同反向归约，CP 的 contiguous 与 dualchunk 位置映射也不同，不能仅按算子名称套同一参考。
- 拓扑依据真实组成员关系检查；EP、sharding、CP 等维度可能重叠，不要求所有组互斥，也不凭所有 degree 的简单乘积新增合法性规则。遵循仓库现有 Fleet 初始化生命周期。
- 流水线以同权重、同数据、同 loss 归约的非流水模型为参考，验证微批传递、梯度累积、共享权重与更新。调度事件用于定位问题，不能替代数值结果；动态 shape、状态缓存和异步变更用连续输入检查状态刷新，读取结果前等待通信完成。
- MoE 分发给 token 和专家可识别身份，核对各专家内容、padding 排除、输出归位和概率应用。AllToAll 按专家所有权推导，概率可能位于 combine 或精度兼容的专家端；SonicMoE AllGather 的 rank 持有专家中间分片并通过 ReduceScatter 求和，不能套用同一回环参考。
- 通信和跨 rank 数值变化需要真实进程组运行。mock collective 仅证明参数传递或编排；保留各 rank 的失败传播证据，不能用某个 rank 的成功日志代表整体成功。后端开关在导入时缓存的路径，应在正确时机设置或隔离进程。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 验证不依赖通信的配置、rank 映射、分片元数据和调度状态；允许通信替身检查 group、参数及调用顺序。说明真实进程组/通信未执行，不能把 mock collective 结果称为多卡数值证据。 |
| 单卡 | 验证接口已有的 world-size=1、无通信或本地分片路径及对应前后向，不人为给不支持单 rank 的后端增加成功保证。单进程初始化或单卡回退只能证明该本地路径。 |
| 多卡 | 对通信、分片、拓扑和调度变化运行真实相关进程组；各 rank 使用可区分输入与梯度，核对独立推导的聚合/还原及相关更新。卡数按待测拓扑选取，读取前等待异步任务，验证所有参与 rank 的结果与失败传播。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，分布式训练常见于形状断言漏掉 rank 顺序、余弦比较漏掉幅值错误、未执行反向、替换集合通信后只检查未报错。

**常规测试位置：** `tests/multi_card_tests/` 中对应 TP、PP、CP、MoE 和拓扑测试；卡数与启动方式以目标测试和作业为准。

## 优化器

**测试目标：** 正确消费梯度并按算法更新参数与内部状态，分组、冻结、精度和分片条件下保持预期更新语义。

- 从明确初始参数、梯度及状态推导更新，检查本次改动涉及的动量、方差、master weight、step 和参数值。沿实际调用链确认消费 `.grad` 还是 `main_grad`，不要以所有参数梯度非空为统一条件。
- 使用可区分的参数组和梯度验证学习率、权重衰减、裁剪、冻结等被修改行为。区分未参与计算、跳过更新与正常更新，防止旧梯度或状态污染后续步骤。
- 状态保存或更新公式改变时，在新优化器实例恢复，再输入相同后续梯度与连续更新对照。`AdamWCustom` 的 step 状态参与恢复，只比较模型权重不能证明优化器恢复完整。
- Muon 从真实 `named_parameters` 核对 spec 名称、切分布局和拼回。用可识别分块变换验证 helper 实际逐块生效；EP 按完整中间维与本地 shard 的真实语义检查。具体数学见 [Muon 扩展规则](muon-slice-specs.md)。
- 纯更新可用小 tensor 隔离大模型；涉及真实分片归约时使用对应分布式环境。优化器状态卸载的搬运与生命周期按计算优化规则验证，保存格式按 Checkpoint 规则验证。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 对支持 CPU 的更新算法或 Muon helper 使用小 tensor，检查参数组、状态、真实 spec 匹配和分块行为；纯配置/选择逻辑可轻量验证。CPU 的 marker 只能证明分块调用，不能证明 GPU 正交化数值。 |
| 单卡 | 运行真实设备更新，按被改分支检查 grad/main_grad、裁剪、低精度 master weight、动量/step 和冻结行为；从明确状态建立参考，必要时新优化器加载后继续更新。 |
| 多卡 | 修改 sharding、混合并行更新、梯度归约或 Muon EP 路径时，在真实组中核对局部参数与状态，并与完整参考或独立分片推导比较。仅单卡 helper 测试不能证明跨 rank 重分布后更新正确。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，优化器常见于分块只检查形状或次数、参数更新只检查非零、调度只取退化点、同一对象保存恢复自证。

**常规测试位置：** `tests/formers/utils/test_hf_bitexact_optimizer.py`、`tests/single_card_tests/transformer/test_muon_slice_specs.py`、`tests/multi_card_tests/moe/test_muon_ep_slice.py`。

## 计算优化

**测试目标：** 加速或节省显存后的计算仍满足目标精度与状态语义，支持条件和资源生命周期正确。

- 融合算子与目标后端使用独立基础运算或已验证参考，比较相关输出和梯度；对齐 dtype、mask、布局、cast 与归约方式。修改 inplace 行为还要检查 buffer、别名及需要保留的输入；特殊 stride/offset 按接口支持范围选取。
- 量化围绕 scale 的轴与形状、块布局、打包、舍入/饱和和反量化关系设计用例。FP8、FP4、UE8M0 按各自协议判断；区分量化近似与融合实现相对同协议参考的差异。`fp8_simulate_qat` 的前向按模拟量化对照，反向按 STE 契约检查。
- 重计算在相同权重、输入和随机状态下比较输出及命名梯度。验证随机重放时保持 dropout 等随机行为，管理实际参与的 Python、NumPy、Paddle 和模型并行 RNG 状态；保存独立参考，防止空梯度比较或参考被覆盖。
- 内存优化按状态创建、搬回、计算与释放的生命周期观察。当前 offload 包装在 accumulator 创建时有独立行为，验证更新阶段开关应从初始化后观察；确认恢复的是最新状态，参数或梯度未被错误搬运，连续迭代无遗留引用。
- 包装层测试可替换昂贵内部 kernel，但保留真实 PyLayer、冻结判断、hook 与梯度路由；其结论限于包装层。真实 kernel 数值、显存收益或异步时序需对应设备与受控测量，不能由 mock 调用次数证明。
- 按实际硬件、扩展依赖及 dtype 限制运行或标记 skip。数值容差和性能基准依据目标契约设置，变更已有容差或参考结果时说明来源；不能统一放宽误差，也不能只凭 `empty_cache` 被调用认定显存已释放。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 检查支持条件、元数据、layout/scale 推导、CPU 参考及后端选择；包装层可用 kernel 替身验证真实分派、冻结与 hook。明确替身覆盖范围，不将 CPU/mock 结果表述为 GPU kernel、异步或显存验证。 |
| 单卡 | 在目标硬件、dtype 和后端运行真实算子/量化/重计算，独立比较相关输出与梯度；inplace、随机重放、卸载和显存变化分别核对别名、RNG、状态内容与受控资源生命周期。 |
| 多卡 | 修改与通信融合、CP/PP 重计算、EP 量化或分片卸载耦合的路径时，运行真实并行配置，以对应未优化参考检查数值和状态；跨 rank 随机重放、通信等待与连续迭代缓存不得由单卡测试替代。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，计算优化常见于忽略量化数值和掩码、丢弃返回值、与被测路径共用参考实现、过宽容差放过数值错误。

**常规测试位置：** `tests/single_card_tests/custom_ops/`、`tests/single_card_tests/fp8/`、重计算专项，`tests/formers/quantization/`；多卡优化放在对应 `tests/multi_card_tests/` 专项。包装层可参考 `tests/single_card_tests/model/test_fused_linear_ce_pylayer_freeze.py`。

## Checkpoint 与权重管理

**测试目标：** 模型和训练状态被准确保存、映射及恢复，分片或权重处理后参数身份与内容保持对应。

- 按改动选择元数据、真实张量 I/O、重分片或完整训练恢复的测试范围。让不同参数使用相同 shape、不同内容，检查键、所有权、分片偏移及 master weight 区别，避免映射交换仍通过 shape 检查。
- 使用临时路径和新对象恢复，检查保存入口承诺的模型、optimizer、scheduler、RNG 等状态。异步保存验证完成时机与读取关系；缺失或损坏格式按实际加载接口验证错误处理。
- 完整续训同时验证恢复状态被 Trainer 正确消费，数据位置和 scheduler 总计划与参考相符；权重导出不机械要求完整训练。元数据替身可以验证映射逻辑，不能证明真实持久化或跨 rank 重分片数值。
- 配置保存允许按当前策略省略默认值，检查加载后的语义及类型。`to_json_string` 默认快照与保存文件时的字段过滤不同，不能要求所有 JSON 都保留全部默认键或删去相同字段。
- 模型合并按实际 linear、SLERP、TIES 等算法建立独立参考，检查参数键、dtype、分片顺序及元数据；随机 sparsify 单独管理随机性。adapter 保存还应核对适配器配置与基础模型身份。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 使用临时文件运行真实配置/元数据读写、键映射和支持 CPU 的权重合并；新对象加载后比较语义、参数身份与内容，异常格式按契约处理。配置 JSON 往返只能证明配置保存。 |
| 单卡 | 在实际设备保存并恢复小模型及该入口承诺的优化器状态，检查新对象参数内容、device、master weight 和后续更新；异步保存须确认完成。不能仅在原对象 load 或只查 shape。 |
| 多卡 | 修改分布式保存、重分片或并行恢复时，实际生成并加载各 rank 的 shard，检查所有权、复制参数过滤、偏移和还原内容；按本次支持的保存/加载拓扑验证，续训再核对 Trainer 消费。元数据 mock 不证明张量重分片。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，Checkpoint 与权重管理常见于切片和拼接只检查形状、恢复路径被空实现替代、测试只操作自造文件而未调用生产入口。

**常规测试位置：** `tests/formers/trainer/` 中 checkpoint 测试、`tests/formers/transformers/test_configuration_utils.py`、`tests/formers/mergekit/`。

## 推理与生成

**测试目标：** 解码选择、生成约束、序列结束与 cache 状态符合接口约定，多样本和连续请求间不串状态。

- 使用确定 logits 验证 greedy 或所改 sampling 策略，检查 logits processor 的约束和顺序、重复 token 的处理及 EOS 屏蔽。mask 的实际表示跟随实现，不统一要求负无穷。
- 构造 batch 内不同结束进度，检查 EOS 后的 padding、输出拼接和完成状态。区分包含 prompt 的最大长度与仅限制新增 token 的参数，已结束序列不能被意外重新激活。
- cache 变化用同一小模型的完整前向与 prefill/decode 对照，检查位置偏移、有效长度与请求重置。流式输出变化验证 token 顺序、取消后的状态和资源处理。
- 策略测试可替换大模型并提供确定 logits；验证 KV cache 时保留真实模型计算。随机采样按明确 RNG 契约或适当分布测试判断，不能用一次随机文本当所有实现的固定答案。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 对 CPU 可执行的 logits processor、终止判定和状态维护使用确定 logits/token，检查 EOS、重复惩罚、长度语义和 batch 结束状态；可替换模型提供 logits，但不据此证明 cache 计算。 |
| 单卡 | 用真实小模型运行 prefill/decode 与 cache，和相同权重的完整前向比较对应位置；检查批内结束、连续请求重置及本次涉及的 dtype/后端。随机采样采用所选策略的确定性或分布性证据。 |
| 多卡 | 仅在修改仓库已有且受支持的并行生成路径时，验证真实参数分片、logits 聚合、各 rank 的 token/结束状态以及必要 cache 对应。无并行生成入口的功能注明不适用，不臆造服务能力，也不用单卡结果冒充多卡验证。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，推理与生成常见于空方法覆盖生成契约、全量跳过被计为通过、新参数只检查候选数量、KV 缓存只检查形状。

**常规测试位置：** `tests/formers/generation/`、`tests/single_card_tests/generation/` 及对应模型测试。

## 配置与运行基础设施

**测试目标：** 用户配置真实影响目标行为，搜索和启动流程选对配置并传递结果，测试与 CI 能准确反映执行状态。

- 配置从真实 CLI/YAML/转换入口构造并追踪到消费者，检查默认值、显式关闭、别名、类型及必要约束。精度目标区分布尔 `False`、`"megatron"`、`"hf"`，防止字符串真值错误；`TransformerConfig.from_config` 与直接构造路径不同，不能只测 helper。
- 当前 `--configs` 用 YAML 替换参数对象，`LlmMetaConfig.set_llm_config` 跳过 `None`，保存配置又可能省略默认值。用例应按入口真实语义设计，检查转换后的字段被消费者使用，而非仅作为无用属性存在。专项要求见 [配置扩展规则](user-configuration-rules.md)。
- 自动配置搜索从模型信息、资源约束和所选算法推导候选，检查评分、去重和限制实际生效。当前 GPT 规模估算有适用范围，不能当任意模型精确计数；`max_configs=0` 当前表示不限量，不凭通常含义写预期。
- 用临时配置与替身进程检查 argv、env、cwd、结果解析和失败传播，保留被测解析、搜索和启动编排；真实训练吞吐或可训练性由实际训练验证。
- 修改测试发现、过滤或汇总时，使用隔离测试树检查同 basename、并发日志、收集失败、断言失败和子进程失败。某个 rank 的 `OK`、零收集或全 skip 不能证明目标行为通过，文件数量与用例数量分开解释。
- 覆盖率按实际工作流及 `ci/.coveragerc` 执行，确认 worker 数据被汇总；Python 行覆盖率不能代替业务断言或 GPU kernel 验证。临时路径、端口和日志应隔离，避免依赖共享环境状态。

**分环境要求：**

| 环境 | 适用范围与编写/评审要求 |
| --- | --- |
| 无卡 | 用真实解析、配置转换、搜索和 runner 判定逻辑，配合临时 YAML/JSON、argv/env 和替身子进程验证配置传播、排序去重及失败处理；可直接加载独立源码 helper，但须说明未验证包导入和完整启动。 |
| 单卡 | 修改设备配置生效、后端选择或单卡启动时，运行相应最小设备任务，检查最终消费者实际选中的行为及错误传播。纯字段归一化或搜索规则变化无需为增加卡数强行启动训练。 |
| 多卡 | 修改分布式配置、launcher 或多卡 CI 时，验证 rank/world size、设备分配、环境传播、输出隔离和失败汇总；调度逻辑可无卡测试，但真正训练/通信成功必须由受支持多卡任务证明。某 rank 打印 OK 不抵消其他 rank 失败。 |

**反模式：** [典型单测反例](unit-test-antipatterns.md)，配置与运行基础设施常见于配置自赋值后读回、测试内重写 CLI 分派、入口只检查被调用、参数量和 FLOPs 只断言正数。

**常规测试位置：** `tests/single_card_tests/test_accuracy_target.py`、`tests/single_card_tests/test_transformer_config.py`、`auto_configurator/tests/`、`ci/rule-tests/`。

## 用例放置与运行方式

新增测试放在最接近所属模块的常规目录，沿用邻近 unittest/pytest 风格。选择测试形式时核对实际执行入口：

| 测试范围 | 当前入口与注意事项 |
| --- | --- |
| 无卡 | 在所属常规目录中定向运行明确支持 CPU 的用例；如使用 Paddle，应选择 CPU 并隔离可见加速卡。仓库未建立统一无卡专用目录，不将放在 single_card_tests 内当作必需 GPU 的证明 |
| Fleet 单卡 | `ci/single_card_test.sh` 逐文件运行 pytest，并发执行文件；本地可定向运行文件或用例，避免共享临时资源 |
| Fleet 多卡 | `ci/multi-card_test.sh` 通过 `paddle.distributed.launch` 直接执行 Python 文件；需要 `unittest.main` 或等效 runner，卡数和禁用配置核对 `tests/test_configs.yaml` 与实际脚本 |
| Formers | `scripts/unit_test/ci_unittest.sh` 包含特定工作目录设置；本地可定向运行 `tests/formers/`，导入时读取的 slow 等开关需提前设置 |
| SonicMoE 专项 | `ci/single_card_sonic.sh` 使用显式文件列表，新增相关文件需检查目标作业注册 |
| 自动配置 | 定向运行 `auto_configurator/tests/`，区分纯搜索与真实训练启动 |

默认 pytest 收集不能证明多卡文件已通过 launcher 执行。由 launcher 直接运行的文件若只定义 pytest 函数，可能没有执行任何断言。依赖或硬件不足时记录 skip 原因和未运行部分，不当作目标行为已验证。

## 评审要求

先定位所属模块，再结合本次变更的真实入口、状态与消费者检查相关测试。独立参考可以是手工推导的小数据、基础运算或已验证实现，不能再调用被测函数生成 expected。mock 只能支持其保留逻辑范围内的结论。

评审意见说明哪项行为缺少验证、什么错误会漏检以及应观察的结果。例如：“MoE 测试只检查总 token 数，交换专家分片仍会通过；应核对专家收到的 token 身份与还原后的加权结果。”不因未照抄本文例子或没有新增测试文件而直接判错。

已有断言能验证相同契约时可以复用，纯文档或格式修改不机械要求训练测试。报告实际命令、相关环境、执行结果及未运行部分；静态核对、mock 编排检查、真实数值运行分别说明证据范围。已证实的生产缺陷与测试证据不足分开表述，不维护统一场景编号或完成率台账。
