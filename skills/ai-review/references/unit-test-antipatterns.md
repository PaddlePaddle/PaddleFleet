# 典型单测反例

本文给出代表性的反例，判据见[单测规范](unit-test-rules.md)、[基础规则](base-rules.md)。

**示例约定。** 代码展示反例与修正方向，省略公共 import、设备准备及部分 fixture。
`cfg`、`model_stub`、`build_*`、`fixed_*`、`independent_*` 等名称可以是示意性辅助对象，不承诺仓库有同名 API。
落地时须绑定实际入口和签名：构造辅助函数应执行真实初始化，参考辅助函数应有独立的公式或已验证锚点，
不能把待验证逻辑藏进替身。代码块可独立做 Python 语法检查，不代表可脱离这些依赖直接运行。
数值阈值须按 dtype、输入规模和算法误差校准；mock 通信示例提供本地编排证据，真实跨 rank 语义仍需实际进程组运行。

shape、dtype、类型、调用次数、异常和 skip 都可能是有效契约。问题是用它们替代了所声称验证的内容、
参数消费或数值行为。随机输入也不是天然无效：保存输入并独立计算期望仍可形成证据；
固定且可区分的输入更便于暴露交换、错位和退化分支。

## 类型目录

| 类型 | 典型信号 | 重点查找 |
| --- | --- | --- |
| [1. 生产入口不在验证链上](#entry) | 本地重写公式、自赋值自断言、patch 被测方法 | 初始化、workflow、CLI、源码文本断言 |
| [2. 输出被压成弱断言](#weak-output) | 只比 shape、数量、sum、非空 | 专家路由、样本顺序、权重布局、缓存内容、对象身份 |
| [3. 参数与编排没有被观察](#consumption) | 只查 called、参数传了却无效果断言 | attention 分派、归约系数、peer、长度惩罚、回调 |
| [4. 参考与被测同源](#reference) | x 与 x 比、共用故障实现、恢复结果自洽 | 量化锚点、PPL、投影与生成路径对照的边界 |
| [5. 输入退化或未触发分支](#degenerate) | 全零、相同 rank、相等维度、只取交点 | 分块重组、调度、chosen/rejected、音频边界 |
| [6. 数值比较丢失错误](#numeric) | 只比 cosine/norm、宽 atol、合并非有限值 | 幅值错误、NaN/+inf、独立 mask 比对 |
| [7. 反向路径或梯度集合漏检](#backward) | 只 forward、遇 None 跳过、zip 截短 | dx/dw、冻结与未选专家、grad/main_grad |
| [8. 保存恢复由原状态自证](#restore) | 同对象往返、只查文件或恢复后内部一致 | 新模型/优化器、续训轨迹、词表和配置 |
| [9. 断言未执行或失败未传播](#execution) | pass 覆写、未收集、空循环、提前 skip | mixin、初始化时机、runner、退出码 |
| [10. 异常被吞成成功或 skip](#exceptions) | except Exception/pass、错误导入当缺依赖 | 精确能力探测、明确异常契约 |
| [11. 共享状态污染或自造前提](#state) | 全局 patch 不恢复、清空后断言为空 | RNG、环境变量、类属性、tracker、缓存 |
| [12. 随意 mock 盖住被测逻辑](#mock) | patch 被测方法/构造、替身返回与输入无关的张量 | 被测 kernel 数值、构造装配与参数校验 |
| [13. 多卡行为只在单卡/单进程取证](#single-card) | 伪造 world_size+mock collective、CPU 手算模拟 collective、只跑单卡 helper | 通信方向与 peer、切分尺寸、专家归位、跨 rank 拼回与归约、reshard 张量 |

<a id="entry"></a>
## 1. 生产入口不在验证链上

**典型写法**

```python
def test_fused_norm_math(self):
    var = (x * x).mean(axis=-1, keepdim=True)
    actual = x * paddle.rsqrt(var + 1e-6) * w
    expected = x * paddle.rsqrt(var + 1e-6) * w
    self.assertTrue(paddle.allclose(actual, expected))

def test_dtype_selection(self):
    mixed_precision = "fp16"
    dtype = "float16" if mixed_precision == "fp16" else "bfloat16"
    self.assertEqual(dtype, "float16")

def test_init_validation(self):
    with mock.patch.object(SFTTrainer, "__init__", return_value=None):
        trainer = SFTTrainer()
    self.assertIsNotNone(trainer)
```

**为什么漏检。** RMSNorm、workflow 和 Trainer 初始化都没有进入被观察的调用链；
删除真实公式、配置分支或参数校验，这些用例仍通过。`__new__` 后手动填属性再读回、
测试自己建立 `COMMAND_MAP` 或合法 `loss_type` 集合，也只证明了测试自己的安排。
`inspect.getsource` 字符串匹配和 `.co_code` 比较观察的是实现文本，不能替代分支行为验证；
改注释可能使它失败，保留关键字却写错行为又可能通过。

**修正写法**

```python
def test_fused_norm_against_independent_reference(self):
    x, w = fixed_norm_inputs()
    out = RMSNormFusionTriton.apply(x, w, 1e-6)  # actual 来自真实融合入口
    ref = x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-5, rtol=1e-4)
    # 此处只证明前向；反向须另按第 7 类实际驱动和比较。

def test_workflow_forwards_config(self):
    captured = {}

    def build_trainer(**kwargs):
        captured.update(kwargs)
        return trainer_stub

    run_sft_workflow(cfg, build_trainer=build_trainer)
    self.assertEqual(captured["dtype"], "float16")
    self.assertEqual(captured["max_seq_len"], 4096)

def test_dpo_rejects_conflicting_flags(self):
    with self.assertRaises(ValueError):
        DPOTrainer(model=model_stub, args=args, loss_type="sigmoid",
                   reference_free=True, ref_model=ref_stub)

def test_yaml_override_reaches_consumer(self):
    path = self._write_yaml({"training": {"hidden_size": 1024}})
    args = parse_cli(["--configs", path])
    cfg = CoreTransformerConfig.from_args(args)
    self.assertEqual(cfg.hidden_size, 1024)
    self.assertEqual(build_layer_plan(cfg).projection_in_features, 1024)
```

外围构建、模型、writer 和通信协作者可以替换，真实配置转换、校验和分派必须保留。
只测内部 helper 可以证明该 helper 的契约；若用例声称验证公开 `forward` 中的 separate-MTP 守卫、
MLA `softmax_scale` 传播、MoE decoder 配置或 recompute 初始化，还应从对应入口触发并观察下游。
例如用 spy 包装 `core_attention`，记录真实收到的 scale，或从真实 `TransformerLayer.forward`
观察切分后的内容；不要在测试内复制同一个 `if` 来代替执行。

只创建再删除一个临时 `.safetensors` 文件也没有验证本地权重入口。应准备可读取的配置，
调用 `check_download_repo`，检查返回本地目录且下载协作者未被调用。

<a id="weak-output"></a>
## 2. 输出被压成 shape、数量、sum 或非空

### 内容、位置和配对关系丢失

**典型写法**

```python
def test_topk_routing(self):
    expert_id, weights, aux_loss = gate(logits)
    self.assertEqual(list(expert_id.shape), [num_tokens, top_k])
    self.assertEqual(expert_id.dtype, paddle.int64)
    self.assertTrue((weights >= 0).all().item())
    self.assertIsNotNone(aux_loss)

def test_split_ratios(self):
    train, valid, test = get_train_valid_test_split(n=100, ratio=(8, 1, 1))
    self.assertEqual(len(train) + len(valid) + len(test), 100)

def test_image_component(self):
    out = processor(images=image, text="<image> describe")
    self.assertAlmostEqual(float(out["pixel_values"].sum()), expected_sum)
```

**为什么漏检。** 所有 token 都路由到专家 0、权重恒为 `1/top_k`、辅助损失恒为零，
仍满足形态、非负和非空检查。把全部样本给 train，总数仍是 100；像素 patch/channel 置换或正负误差
抵消，sum 也可以不变。类似地，label 偏移、媒体与样本错配、padding 位置、切片起点和权重拼接顺序，
都可能不改变输出 shape。

**修正写法**

```python
def test_topk_routing(self):
    logits = paddle.to_tensor(
        [[0.0, 3.0, 1.0, -1.0], [2.0, 0.0, -1.0, 1.0]], dtype="float32"
    )
    expert_id, weights, aux_loss = gate(logits, top_k=2)
    self.assertEqual(expert_id.tolist(), [[1, 2], [0, 3]])
    probs = paddle.nn.functional.softmax(logits, axis=-1)
    expected = paddle.stack([
        probs[0, [1, 2]] / probs[0, [1, 2]].sum(),
        probs[1, [0, 3]] / probs[1, [0, 3]].sum(),
    ])
    np.testing.assert_allclose(weights.numpy(), expected.numpy(), atol=1e-6)
    np.testing.assert_allclose(aux_loss.numpy(), independent_aux_loss(logits), atol=1e-6)

def test_split_ratios(self):
    train, valid, test = get_train_valid_test_split(n=100, ratio=(8, 1, 1))
    self.assertEqual(list(train), list(range(80)))
    self.assertEqual(list(valid), list(range(80, 90)))
    self.assertEqual(list(test), list(range(90, 100)))

def test_padding_order_and_value(self):
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[8.0, 9.0, 10.0, 11.0]])
    out = pad_and_concatenate([a, b], padding_index=0, max_length=4)
    np.testing.assert_array_equal(out, [[1.0, 2.0, 3.0, 0.0],
                                        [8.0, 9.0, 10.0, 11.0]])

def test_image_component_matches_reference(self):
    out = processor(images=image, text="<image> describe")
    ref = reference_image_encoder(image)  # 独立参考
    self.assertEqual(set(out), {"pixel_values", "image_grid_thw"})
    np.testing.assert_allclose(out["pixel_values"], ref, rtol=1e-5)
    self.assertEqual(out["image_grid_thw"].tolist(), [[1, 2, 3]])
```

容量裁剪还须单独设置“同一专家收到 3 个 token、capacity=2”的输入，
逐位比对独立期望 mask、保留 token 身份及丢弃 token 的路由哨兵约定。
仅检查 `mask.sum() == 2` 仍不能区分保留的是哪两个 token。

### 只看前缀、总量或存在性，漏掉整组契约

**典型写法**

```python
def test_shuffle(self):
    out = list(build_sampler(records, shuffle=True, seed=102))
    self.assertEqual(out[:2], [4, 9])

def test_interleave(self):
    out = list(interleave(source_a, source_b, stopping_strategy="first_exhausted"))
    self.assertLessEqual(len(out), 10)

def test_single_element_state(self):
    result = merge_splited_param(state_dict, [], {}, {}, {})
    self.assertIn("beta1_pow_acc_0", result)
```

shuffle 后半段丢失或重复、interleave 提前结束、单元素状态被置零，都可以通过。
修正时要检查完整结果或足以限定目标契约的独立不变量；数量是附加条件。

```python
def test_shuffle_conserves_samples(self):
    first = list(build_sampler(records, shuffle=True, seed=102))
    second = list(build_sampler(records, shuffle=True, seed=102))
    self.assertEqual(first, second)  # 同 seed 新实例复现
    self.assertCountEqual(first, range(len(records)))  # 多重集：不丢不重
    self.assertNotEqual(first, sorted(first))  # 此固定 fixture 已知应发生打乱
    # 若契约规定确切排列，还应与独立期望排列比较。

def test_interleave_stops_at_exhaustion(self):
    out = list(interleave(source_a, source_b, stopping_strategy="first_exhausted"))
    self.assertEqual(out, [("a", 0), ("b", 0), ("a", 1), ("b", 1)])

def test_single_element_state(self):
    state_dict = {"beta1_pow_acc_0": paddle.to_tensor([0.81])}
    result = merge_splited_param(state_dict, [], {}, {}, {})
    self.assertEqual(list(result["beta1_pow_acc_0"].shape), [1])
    np.testing.assert_allclose(result["beta1_pow_acc_0"].numpy(), [0.81])
```

| 短变体 | 会漏掉的改坏方式 | 修正落点 |
| --- | --- | --- |
| 目录发现只数文件，多轮转换只看 `role` 键 | 文件反序、路径裁剪错、prompt/completion 或图片错配 | 完整文件与权重配对；逐轮角色、内容与媒体 |
| repeat 只断言两倍长度 | interleave/非 interleave 顺序互换、label 不随样本 | 逐项核对 ID 和 label |
| merge/prepare/slice 只比 shape | 转置、拼接次序、截断头尾或 pad 值错 | `arange` 输入、完整重排期望、原值区与 padding 区分别比较 |
| 文档窗口只检查非 `-1` 项小于上界 | 全部返回 `-1` 也通过 | 独立逐行可见索引、下界、可见数量 |
| `text_ok or (encoded, expected)` | 非空 tuple 恒真，token 内容错也通过 | 用真实等值比较；同时满足的契约用 `and` |
| `hasattr(cls, "__new__")` / `callable(fn)` | 普通继承方法、错误分派函数仍满足 | 真实构造的关键结构；明确函数身份或可区分输出 |

### 类型、设备、存储身份和缓存内容是不同契约

```python
def test_cache_static_only_shape(self):
    cache = mha.gen_cache(key, value, type=MultiHeadAttention.StaticCache)
    self.assertIsNotNone(cache.k)
    self.assertEqual(cache.k.shape, [2, 4, 8, 16])

def test_device_and_buffer_only_shape(self):
    self.assertIsNotNone(to_device(t, target))
    self.assertEqual(get_tensor("a", (2, 2)).shape, get_tensor("a", (2, 2)).shape)
```

同形常量缓存、k/v 投影混用、错误设备和每次重分配 buffer 都可能被放过。
具体 shape 检查可以发现空的增量缓存与静态缓存之间的长度差异，但不能据此证明缓存内容正确。

```python
def test_cache_static_content(self):
    key = paddle.arange(2 * 8 * 64, dtype="float32").reshape([2, 8, 64])
    value = key + 1000.0
    cache = mha.gen_cache(key, value, type=MultiHeadAttention.StaticCache)
    # 此处 MultiHeadAttention 的 Linear3D 投影直接产生缓存布局。
    np.testing.assert_allclose(cache.k.numpy(), mha.k_proj(key).numpy())
    np.testing.assert_allclose(cache.v.numpy(), mha.v_proj(value).numpy())
    self.assertIsInstance(cache, MultiHeadAttention.StaticCache)
    # 这验证封装和投影选择；共享投影本身的数学正确性见第 4 类的参考边界。

def test_same_place_preserves_identity_and_value(self):
    t = paddle.arange(6, dtype="float32").reshape([2, 3])
    before = t.numpy().copy()
    result = to_device(t, t.place)
    self.assertIs(result, t)  # 仅在同 place 原样返回属于契约时要求
    self.assertTrue(result.place._equals(t.place))
    np.testing.assert_array_equal(result.numpy(), before)

def test_buffer_reuses_storage(self):
    first = get_tensor("a", (2, 2))
    second = get_tensor("a", (2, 2))
    self.assertEqual(first.data_ptr(), second.data_ptr())

def test_detach_keeps_value_and_stops_gradient(self):
    t = paddle.to_tensor([1.0, 2.0])
    t.stop_gradient = False
    out = nested_detach((t,))
    self.assertTrue(out[0].stop_gradient)
    np.testing.assert_array_equal(out[0].numpy(), [1.0, 2.0])
```

跨 place 迁移另查目标 place 和完整内容；offload/reload 另查迁移阶段的状态和恢复后的内容。
仅比较同一个局部变量的 `id(t)` 前后相等，不能证明 offload/reload 真正发生。
如果工厂的目标就是类型选择，断言明确的 `timers._Timer` 有效；
拿 `type(timers("同名缓存"))` 当参考则可能只是与同一个对象自比。

<a id="consumption"></a>
## 3. 参数传入了，编排与消费结果却没被观察

### called 无法单独证明实参与返回值被使用

**典型写法**

```python
def test_fused_attention_forward(self):
    with mock.patch.object(fused_attn, "dispatch", return_value=paddle.zeros_like(q)) as dispatch:
        fused_attn.apply(q, k, v, causal=True, dropout=0.1, softmax_scale=0.125)
    dispatch.assert_called_once()
```

把 causal 写反、忽略 dropout、传错 scale，调用次数都不变；返回值被丢弃后，
调用方即使不使用协作者输出也能通过。mock 编排协作者是合法的，关键在于给它可区分的响应，
并观察保留下来的真实编排逻辑。

**修正写法**

```python
def test_fused_attention_forward_contract(self):
    q, k, v = fixed_nonuniform_qkv()
    marker = paddle.arange(q.numel(), dtype="float32").reshape(q.shape)
    with mock.patch.object(fused_attn, "dispatch", return_value=marker) as dispatch:
        out = fused_attn.apply(q, k, v, causal=True, dropout=0.0, softmax_scale=0.125)

    dispatch.assert_called_once()
    args, kwargs = dispatch.call_args
    self.assertIs(args[0], q)
    self.assertIs(args[1], k)
    self.assertIs(args[2], v)
    self.assertTrue(kwargs["causal"])
    self.assertEqual(kwargs["dropout"], 0.0)
    self.assertEqual(kwargs["softmax_scale"], 0.125)
    np.testing.assert_array_equal(out.numpy(), marker.numpy())
```

### 归约、peer 和轴的编排必须有具体期望

只查 `all_reduce.called` 可以发现调用被删，却不能发现 group/op 错误或忘记除以 `nranks`；
也不能把本地假 collective 作为真实多卡归约正确的证据。

```python
def test_reduce_loss_consumes_sum_and_group(self):
    from types import SimpleNamespace
    from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper

    values = paddle.to_tensor([2.0, 4.0])
    dp_group = SimpleNamespace(nranks=2)
    captured = {}

    def fake_all_reduce(tensor, group=None, op=paddle.distributed.ReduceOp.SUM):
        captured.update(group=group, op=op, tensor=tensor)
        tensor.set_value(paddle.to_tensor([10.0, 18.0]))

    with mock.patch.object(DSAIndexerLossLoggingHelper, "tracker", {"values": values}), \
         mock.patch.object(parallel_state, "get_pipeline_model_parallel_group", return_value=None), \
         mock.patch.object(parallel_state, "get_context_parallel_world_size", return_value=1), \
         mock.patch.object(parallel_state, "get_data_parallel_group", return_value=dp_group), \
         mock.patch.object(paddle.distributed, "all_reduce", side_effect=fake_all_reduce) as ar:
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(num_layers=2)

    ar.assert_called_once()
    self.assertIs(captured["group"], dp_group)
    self.assertIs(captured["tensor"], values)
    self.assertEqual(captured["op"], paddle.distributed.ReduceOp.SUM)
    self.assertEqual(values.tolist(), [5.0, 9.0])  # 实际消费求和结果并除以 2

def test_send_meta_selects_prev_peer(self):
    captured = {}

    def fake_send(tensor, dst, group):
        captured.update(tensor=tensor.numpy().copy(), dst=dst, group=group)

    with mock.patch.object(transport, "raw_send", side_effect=fake_send):
        meta_send_meta(meta, group)  # 保留真实 peer 选择与元数据编码

    self.assertEqual(captured["dst"], group.prev_rank)
    self.assertIs(captured["group"], group)
    np.testing.assert_array_equal(captured["tensor"], expected_meta)
```

CP scatter 同理：逐字段记录输入、axis 和 mode，让替身返回依赖输入的不同标记，
检查 `input_ids`、`labels`、`position_ids` 各自消费了对应结果，非目标 mask 保持契约要求的身份或内容。
只保留最后一次调用记录会漏掉前面字段传错轴；identity 替身也区分不了“使用结果”和“直接返回原输入”。

### 计数、正数和字段存在性没有证明参数效果

**典型写法**

```python
def test_length_penalty_consumed(self):
    hyps = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
    hyps.add(hyp, sum_logprobs=-1.0, origin_len=5)
    self.assertEqual(len(hyps), 1)

def test_num_params(self):
    self.assertGreater(mfu.get_num_params(cfg, lora=True), 0)
```

忽略 `origin_len`、改变 beam 得分公式仍会 append 一个候选；遗漏 embedding、LoRA 或专家贡献，
参数量仍可为正。应观察精确得分或单一开关产生的独立增量。

```python
def test_length_penalty_consumed(self):
    hyp = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8])
    hyps = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
    hyps.add(hyp, sum_logprobs=-1.0, origin_len=5)
    expected = -1.0 / (((8 - 5 + 5) / 6) ** 1.0)
    score, stored = hyps.beams[0]
    self.assertAlmostEqual(score, expected, places=6)
    self.assertAlmostEqual(hyps.worst_score, expected, places=6)
    np.testing.assert_array_equal(stored.numpy(), hyp.numpy())

    other = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
    other.add(hyp, sum_logprobs=-1.0, origin_len=2)
    self.assertNotAlmostEqual(other.beams[0][0], expected, places=6)

def test_parameter_and_flop_deltas(self):
    base = mfu.get_num_params(small_cfg(embedding=False))
    full = mfu.get_num_params(small_cfg(embedding=True))
    self.assertEqual(full - base, vocab_size * hidden_size)

    f2 = mfu.get_num_flop_fwd(cfg, batch=2)
    f4 = mfu.get_num_flop_fwd(cfg, batch=4)
    self.assertEqual(f4, 2 * f2)

    extra = mfu.get_num_flop_bwd(cfg) - mfu.get_num_flop_bwd(cfg, no_fused=True)
    self.assertEqual(extra, independent_attention_recompute_flops(cfg))
```

参数量总数仍应由完整组件公式锚定；只做差值或倍数关系可能漏掉两边共同缺项。
MLA 低秩投影的参数增量应按维度带符号推导，不能一概要求 LoRA 增加参数；
`top_k` 加一只增加相应路由专家贡献，固定注意力与共享专家项不会一起翻倍。
融合注意力额外 QK 重计算属于 backward，不能误要求 forward FLOPs 增加。

### 回调要观察触发和不触发，以及真正输出的内容

```python
def test_gc_callback_without_observation(self):
    callback.on_step_end(args, state, control)  # 没有观察动作

def test_gc_fires_only_at_interval(self):
    callback = GCCallback(gc_interval=2)
    with mock.patch("gc.collect") as collect:
        callback.on_step_end(args, make_state(global_step=2), control)
        collect.assert_called_once()
        callback.on_step_end(args, make_state(global_step=3), control)
        collect.assert_called_once()  # 非命中 step 不重复

def test_gc_interval_zero(self):
    callback = GCCallback(gc_interval=0)
    with mock.patch("gc.collect") as collect:
        callback.on_step_end(args, make_state(global_step=10), control)
        collect.assert_not_called()

def test_writer_receives_scalar(self):
    writer = WriterStub()
    callback = TensorBoardCallback(output_dir=tmp, writer=writer)
    callback.on_log(args, make_state(global_step=7), control, logs={"loss": 0.5})
    self.assertEqual(writer.scalars, [("loss", 0.5, 7)])
```

相关短变体：`normalizer=2` 要将受控耗时 10 记录为 5；`set_logging(DEBUG)` 要观察目标 logger level；
`remove_loss_function` 声称告警就用 `assertWarns` 捕获；rank 0 打印同时验证 rank 1 无输出；
CLI 通过真实 `main` 分派 `argv`，核对命令协作者收到的参数。
`record_event` 要记录 push/body/pop 的完整顺序，必要时检查异常退出仍 pop，
不能只进入空 context manager 或只看最终 pop 次数。若目标就是缓存避免重复查询，
清洁初始缓存后断言查询协作者只调用一次是有效证据；两次返回值相等则不足。

<a id="reference"></a>
## 4. 参考与被测同源，共同错误两边一起通过

**典型写法**

```python
def test_int8_linear(self):
    q_w, scale = quantizer.quantize(weight)
    ref = manual_matmul(x, quantizer.dequantize(q_w, scale))
    got = int8_linear(x, weight, None)
    np.testing.assert_allclose(got.numpy(), ref.numpy(), atol=1e-2)

def test_log_ppl(self):
    loss = self.trainer.prediction_step(inputs)
    expected = np.exp(loss)
    self.assertAlmostEqual(np.exp(loss), expected)
```

**为什么漏检。** 若 int8 路径也调用同一 quantizer，scale 轴、饱和和位序错误可能同时出现在两侧。
第二个用例根本没有观察 logger 产生的 PPL，生产代码忘记 `exp`、甚至不写 `ppl` 都不影响断言。
恢复后的词表正反映射相互自洽，也无法证明原 token 没有丢失或改 ID。

**修正写法**

```python
def test_int8_linear_against_independent_anchor(self):
    x, weight, bias = fixed_linear_inputs()
    q_ref, scale_ref = hand_derived_quant(weight)  # 不调用产品 quantize/dequantize
    ref = (x @ q_ref.astype("float32").T) * scale_ref + bias
    got = int8_linear(x, weight, bias)
    np.testing.assert_allclose(got.numpy(), ref.numpy(), atol=1e-4)

def test_quantize_activation_numeric(self):
    x = paddle.to_tensor([[-3.0, -1.0, 0.0, 2.0]], dtype="float32")
    x_h = apply_hadamard_explicit(x)  # 按该量化协议独立展开小矩阵
    scale_ref = x_h.abs().max() / 127.0
    q_ref = paddle.round(x_h / scale_ref).clip(-127, 127).cast("int8")
    q, scale = quantize_activation(x, hadamard=True)
    np.testing.assert_allclose(scale.numpy(), scale_ref.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(q.numpy(), q_ref.numpy())
    np.testing.assert_allclose(
        dequantize(q, scale).numpy(),
        (q_ref.astype("float32") * scale_ref).numpy(),
        atol=1e-6,
    )

def test_log_adds_ppl(self):
    logs = {}
    self.trainer.log({"loss": 2.0}, logs)
    self.assertAlmostEqual(logs["ppl"], 7.389056, places=5)
```

量化小例只锚定其所用协议。不同块的非均匀 scale、舍入临界值、可饱和输入、位打包顺序和
Hadamard 左右乘/分块仍需对应的独立期望；不能因为调用了 `clip` 就声称覆盖了饱和分支。
偏置可在同一输入下比较“有 bias − 无 bias”与广播后的 bias，再保留整体数值锚点。
RoPE 的 offset、interpolation、partial-rot 尾部透传和 MLA 通道重排也要有可区分位置的参考。

**同路径对照的能力边界。** 复用已验证投影来检查 StaticCache 的 k/v 选择，
或对比 `generate()` 与 greedy/beam/sample 入口，可以验证封装、分派和两条路径的一致性。
它们不能排除共享投影、beam scorer 或核心解码器内部的共同错误。
保留这些对照，并为所声称的底层数值、长度惩罚、EOS 后完成状态等补独立锚点；
无需把每一个合法共享协作者都认定为错误参考。

<a id="degenerate"></a>
## 5. 输入退化，错误实现与正确实现在观测点重合

### identity、全零与单点采样掩盖分块错误

**典型写法**

```python
def test_per_head_marker_sampling(self):
    weight = paddle.zeros([rows, heads * cols])
    seen = []

    def marker(block):
        seen.append(list(block.shape))
        return paddle.full_like(block, len(seen))

    out = ortho_per_head(weight, marker, heads=heads)
    self.assertEqual(seen, [[rows, cols]] * heads)
    self.assertEqual([float(out[0, i * cols]) for i in range(heads)],
                     [1.0, 2.0, 3.0, 4.0])
```

**为什么漏检。** marker 抹掉原块内容，而断言每块只取第一个点；
送入错块、只正确处理首行、清空其余行列，都可能通过。identity 协作者加总 shape/调用次数
也无法约束切片起点、轴、Q/K/V 段宽及输出拼回位置。

**修正写法**

```python
def test_per_head_full_content(self):
    rows, heads, cols = 3, 4, 2
    weight = paddle.arange(rows * heads * cols, dtype="float32").reshape(
        [rows, heads * cols]
    )
    expected_blocks = [weight[:, i * cols:(i + 1) * cols] for i in range(heads)]
    calls = []

    def marker(block):
        i = len(calls)
        np.testing.assert_array_equal(block.numpy(), expected_blocks[i].numpy())
        calls.append(i)
        return block * (i + 1) + 1000.0 * (i + 1)  # 内容与块序号都可区分

    out = ortho_per_head(weight, marker, heads=heads)
    expected = paddle.concat(
        [block * (i + 1) + 1000.0 * (i + 1)
         for i, block in enumerate(expected_blocks)],
        axis=-1,
    )
    self.assertEqual(calls, list(range(heads)))
    np.testing.assert_array_equal(out.numpy(), expected.numpy())
```

按实际 per-head/per-role、gate、MLA 吸收权重或专家 MLP 布局分别手写预期区间；
leading-batch 3D 输入也要核对批维与 head 维不串。不要用被测切片 helper 生成 expected_blocks。

### rank、样本侧别和维度必须可区分

**典型写法与漏检原因。** 两个 rank 都输入 `ones`，all-gather 本地重复或 rank 反序可能得到相同输出；
chosen/rejected 和每个 microbatch 都是全零，交换两侧也不变；
`s == n` 时，`[b, s, n, d]` 的转置错误连 shape 都可能不变。

```python
def test_all_gather_rank_order(self):
    # fixture 建立真实、恰好两个 rank 的 tp_group；此处不 mock collective。
    rank = dist.get_rank()
    values = ([[0.0, 1.0], [2.0, 3.0]] if rank == 0
              else [[10.0, 11.0], [12.0, 13.0]])
    local = paddle.to_tensor(values, dtype="float32")
    out = all_gather_from_tensor_parallel_region(local, group=tp_group)
    expected = np.array([[0.0, 1.0, 10.0, 11.0],
                         [2.0, 3.0, 12.0, 13.0]], dtype=np.float32)
    np.testing.assert_array_equal(out.numpy(), expected)

def test_merge_preserves_side_and_microbatch(self):
    chosen = [paddle.to_tensor([[1.0, 2.0]]), paddle.to_tensor([[3.0, 4.0]])]
    rejected = [paddle.to_tensor([[9.0, 8.0]]), paddle.to_tensor([[7.0, 6.0]])]
    merged = merge(chosen, rejected)
    self.assertEqual(len(merged), 2)
    np.testing.assert_array_equal(merged[0].numpy(), [[1.0, 2.0, 9.0, 8.0]])
    np.testing.assert_array_equal(merged[1].numpy(), [[3.0, 4.0, 7.0, 6.0]])
```

转置与通道重排用 `b=2, s=5, n=3, d=4` 和位置唯一输入，对完整内容作独立比较；
有 stride 契约时还应选择实际非连续布局，不能只改形状标签。

### 边界名称不能代替边界输入

**典型写法**

```python
def test_schedule_at_decay_start(self):
    advance_to(sched, decay_start)
    self.assertEqual(sched.get_lr()[0], 1.0)

def test_db_range(self):
    db = power_to_db(signal_with_60db_range, db_range=80)
    self.assertLessEqual(db.max() - db.min(), 80)

def test_short_waveform(self):
    out = frame_wave(np.ones(1600), frame_length=400)
    self.assertEqual(out.shape[1], 400)
```

调度的所有分支可能在 step 0 或 decay 起点重合；未超过 80 dB 的输入无需裁剪；
1600 点输入没有进入“短于 400 点一帧”的分支。这些观测点无法证明参数被消费。

```python
def test_schedule_in_differing_region(self):
    for step in [warmup_end + 1, mid_decay, decay_end]:
        self.assertAlmostEqual(
            lr_at(step),
            independent_cosine_formula(step, num_cycles=num_cycles),
            places=6,
        )
    self.assertNotAlmostEqual(lr_at(mid_decay, num_cycles=0.5),
                              lr_at(mid_decay, num_cycles=1.0), places=6)

def test_db_range_clips(self):
    db = power_to_db(np.array([1.0, 1e-10]), db_range=80)
    self.assertAlmostEqual(db.max() - db.min(), 80)

def test_short_waveform_padding(self):
    out = frame_wave(np.ones(100), frame_length=400, pad_mode="constant")
    self.assertEqual(out.shape, (1, 400))
    np.testing.assert_array_equal(out[0, :100], np.ones(100))
    np.testing.assert_array_equal(out[0, 100:], np.zeros(300))

def test_spectrogram_impulse_anchor(self):
    waveform = np.zeros(400)
    waveform[200] = 1.0
    window = np.hanning(400)
    out = spectrogram(waveform, window=window, frame_length=400,
                      hop_length=400, center=False)
    np.testing.assert_allclose(out, np.full((201, 1), window[200]), rtol=1e-6)
```

调度 fixture 应保证选取的中间点确实区分目标 cycles/衰减分支。频谱例以原协议的幅值输出为前提，
脉冲放在非零窗口系数处；若把脉冲放到零系数端点，窗口失效也可能不可见。
图像 reflect/replicate/symmetric padding 用坐标和通道可区分的小图，分别比内部与边缘内容。

<a id="numeric"></a>
## 6. 数值比较只保留方向、范数或宽容差

### 整倍数幅值错误和小梯度被放过

**典型写法**

```python
def test_context_parallel_grads(self):
    self.assertGreater(cos_sim(loss_cp, loss_ref).item(), 0.95)
    self.assertGreater(cos_sim(dw, dw_ref).item(), 0.999)

def test_small_gradient(self):
    np.testing.assert_allclose(dx.numpy(), dx_ref.numpy(), atol=0.08, rtol=0.08)
```

**为什么漏检。** `0.5 * ref` 或 `2 * ref` 的 cosine 都是 1，CP/TP 的 sum/average 系数错误
因此可能完全不可见。只比 norm 又会漏掉符号翻转、元素重排和切片错位。
当 `atol=rtol=0.08` 时，绝对值约不超过 `0.08 / 0.92` 的参考元素被错误置零仍可满足容差；
缩小上游 cotangent 会进一步弱化检查。共享 helper 默认关闭相对 L2/allclose，
只传最小 cosine，也属于同一机制。

**修正写法**

```python
def assert_scale_sensitive_close(actual, ref):
    actual = np.asarray(actual, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert np.isfinite(actual).all()
    assert np.isfinite(ref).all()
    np.testing.assert_allclose(actual, ref, rtol=5e-2, atol=1e-4)
    norm = np.linalg.norm(ref)
    assert norm > 0  # 本 fixture 专门验证非零梯度；零梯度另按其契约判断
    assert np.linalg.norm(actual - ref) / norm < 0.05

def test_backward_is_scale_sensitive(self):
    dx, dw = fused_w4a8_group_gemm(inputs)
    dx_ref, dw_ref = independent_reference_grads(inputs)
    self.assertGreater(np.abs(dw_ref.numpy()).max(), 1e-3)
    assert_scale_sensitive_close(dx.numpy(), dx_ref.numpy())
    assert_scale_sensitive_close(dw.numpy(), dw_ref.numpy())

    for bad in (0.5 * dw_ref.numpy(), 2.0 * dw_ref.numpy(),
                np.zeros_like(dw_ref.numpy())):
        with self.assertRaises(AssertionError):
            assert_scale_sensitive_close(bad, dw_ref.numpy())
```

方向指标可作补充。阈值应有精度依据，参考梯度本身要能区分目标错误；
前向 allclose 不能补偿反向只比 cosine 的不足。优化器 `step` 后只看“参数变了、动量非零”
同样没有验证更新量，应独立推导一步参数、moment1、moment2、bias correction 与 master weight。
例如 AdamW 一步的 m/v 来自固定梯度，参数同时包含 decoupled decay 与归一化后的梯度项；
对 no_decay 分组再检查精确差值，不能用任意非零变化作结论。

### NaN、+inf 与合法 mask 不能统统归为“非有限”

**典型写法**

```python
def weak_compare_logits(cand, ref):
    cand_invalid = ~np.isfinite(cand)
    ref_invalid = ~np.isfinite(ref)
    np.testing.assert_array_equal(cand_invalid, ref_invalid)
    np.testing.assert_allclose(np.where(cand_invalid, 0.0, cand),
                               np.where(ref_invalid, 0.0, ref))
```

参考的 `-inf` mask 被候选的 NaN 或 +inf 替代，非有限位置仍相同，置零后比较就通过。
即使正确区分 `-inf`，只比较两个输出导出的 mask 也无法发现两边共同屏蔽错位；
只检查 `result[mask][0]` 还会漏掉其余屏蔽位置和误清零的未屏蔽值。

```python
def assert_masked_logits(cand, ref, expected_mask, neg_sentinel=None):
    cand, ref = np.asarray(cand), np.asarray(ref)
    expected_mask = np.asarray(expected_mask, dtype=bool)
    assert cand.shape == ref.shape == expected_mask.shape
    for values in (cand, ref):
        assert not np.isnan(values).any()
        assert not np.isposinf(values).any()

    def mask_of(values):
        mask = np.isneginf(values)
        if neg_sentinel is not None:
            mask = mask | (values <= neg_sentinel)
        return mask

    # expected_mask 由可见窗口/因果位置等输入规则独立构造，不能由任一输出生成。
    np.testing.assert_array_equal(mask_of(cand), expected_mask)
    np.testing.assert_array_equal(mask_of(ref), expected_mask)
    np.testing.assert_allclose(cand[~expected_mask], ref[~expected_mask],
                               atol=1e-5, rtol=1e-4)
```

只有协议允许负大哨兵且正常有限值不会与其混淆时，才传入 `neg_sentinel` 做上述归一化。
零值 mask 的场景则比较所有应清零位置，并对未屏蔽位置另作内容比较。
对比较 helper 可用 NaN/+inf、错一位 mask、整倍数幅值等负对照，确认它确实会拒绝目标错误。

<a id="backward"></a>
## 7. 声称验证反向，却未执行反向或漏掉梯度集合

**典型写法**

```python
def test_backward_calls_all_to_all(self):
    x = paddle.randn([2, 4], dtype="float32")  # 未开启梯度
    y = AllToAllAsyncPyLayer.apply(x, group)
    self.assertIsNotNone(y)

def compare_grads(self, grads_cp, grads_ref):
    for actual, expected in zip(grads_cp, grads_ref):
        if actual is None or expected is None:
            continue
        self.assertAlmostEqual(actual.norm().item(), expected.norm().item())
```

**为什么漏检。** 第一条即使删除 `backward` 也不影响用例；
第二条 `zip` 会截短列表，任一 None 直接跳过使缺失梯度被放行，甚至没有一次数值断言。
只取参数名交集也会掩盖整条分支少产出的梯度。返回零张量不会触发 None 分支，
但仍需有非零参考及幅值敏感比较才能拒绝；两种缺失形式不能混为一谈。

**修正写法**

```python
def test_fused_norm_forward_and_backward(self):
    x, w = fixed_norm_inputs()
    x.stop_gradient = False
    w.stop_gradient = False
    x_ref = x.detach().clone()
    w_ref = w.detach().clone()
    x_ref.stop_gradient = False
    w_ref.stop_gradient = False
    upstream = nonuniform_upstream_grad(x.shape)

    out = RMSNormFusionTriton.apply(x, w, 1e-6)
    ref = x_ref * paddle.rsqrt(x_ref.pow(2).mean(-1, keepdim=True) + 1e-6) * w_ref
    out.backward(upstream)
    ref.backward(upstream)
    for actual, expected in ((out, ref), (x.grad, x_ref.grad), (w.grad, w_ref.grad)):
        self.assertIsNotNone(actual)
        self.assertIsNotNone(expected)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-5)

def test_complete_gradient_contract(self):
    _, grads_cp = run_context_parallel(x)
    _, grads_ref = run_full_reference(x)
    expected_names = expected_with_grad | expected_without_grad
    self.assertEqual(set(grads_cp), expected_names)
    self.assertEqual(set(grads_ref), expected_names)

    for name in expected_names:
        actual, expected = grads_cp[name], grads_ref[name]
        if name in expected_without_grad:
            self.assertIsNone(actual)
            self.assertIsNone(expected)
        else:
            self.assertIsNotNone(actual)
            self.assertIsNotNone(expected)
            np.testing.assert_allclose(actual.numpy(), expected.numpy(),
                                       atol=1e-3, rtol=1e-2)
            if name in expected_zero_grad:
                np.testing.assert_array_equal(actual.numpy(),
                                               np.zeros_like(actual.numpy()))
```

`expected_with_grad`、`expected_without_grad` 和 `expected_zero_grad` 应由该用例的训练/冻结/路由
契约指定，不能从实测梯度推出来。未选专家可能要求零梯度，也可能按实现契约允许 None；
需区分这些约定，不能强制所有参数都非空，或一概遇 None 就跳过。

AllToAll 反向应开启梯度、实际调用 backward，并核对按 rank 可区分的独立本地梯度切片；
非均匀上游梯度有助于暴露逆序或错误归位。测试反向保存张量时，也要真正开启相关路径，
核对保存的对象/内容和反向消费，而不是仅断言前向 shape 或一个名字存在。
优化器用例须沿真实调用链确认消费的是 `.grad` 还是 `main_grad`，把参考更新量与实际被消费的
梯度对应起来，避免给错梯度槽却宣称验证了优化器算法。

<a id="restore"></a>
## 8. 保存恢复由原状态自证

**典型写法**

```python
def test_prepare_tensor_transpose(self):
    t = paddle.randn([8, 4])
    out = prepare_tensor(t, need_transpose=True)
    self.assertEqual(out.shape, [4, 8])   # 只看形状

def test_optimizer_state_unchanged(self):
    state = {"beta1_pow_acc": paddle.randn([4]), "moment1": paddle.randn([4])}
    restored = dequant_unified_optimizer(state, stage="O0")
    self.assertIn("moment1", restored)    # 只看键还在
```

**为什么漏检。** 转置只查 shape，把值搬错、转置方向反、漏掉复制都不影响 `[4, 8]`；
恢复只查键存在，同键换成任意张量、moment 数值恢复错、量化误差超限都能通过。
`save` 后在同一对象上 `load`、或恢复后只比较对象内部字段彼此一致，都是拿被测状态给自己背书；
`test_save_load` 整个函数体 `pass` 更是连保存恢复都没执行。

**修正写法**

```python
def test_prepare_tensor_transpose(self):
    t = paddle.arange(32, dtype="float32").reshape([8, 4])  # 可区分内容
    out = prepare_tensor(t, need_transpose=True)
    np.testing.assert_array_equal(out.numpy(), t.numpy().T)  # 值与方向都核对

def test_optimizer_roundtrip_restores_values(self):
    ckpt = tempfile.mkdtemp()
    m = SmallModel(); opt = AdamWCustom(m.parameters())
    step_once(m, opt, fixed_batch())          # 让 moment/step 非平凡
    save_checkpoint(m, opt, ckpt)
    m2 = SmallModel(); opt2 = AdamWCustom(m2.parameters())
    load_checkpoint(m2, opt2, ckpt)           # 新对象加载
    for (n, p), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        np.testing.assert_allclose(p.numpy(), p2.numpy())
    self.assertEqual(opt2.state_dict()["LR_Scheduler"]["last_epoch"],
                     opt.state_dict()["LR_Scheduler"]["last_epoch"])
    grad = fixed_batch_grad()
    ref = expected_update_from(opt.state_dict(), grad)   # 独立推导下一步更新
    opt2.step_with(grad)
    np.testing.assert_allclose(m2.param.numpy(), ref, rtol=1e-5)
```

恢复必须用新模型/新优化器加载，比较参数与优化器状态（moment、step、master weight）的真实值，
再输入相同后续梯度确认恢复后能按算法继续更新；续训还要核对数据位置和 scheduler 总计划。
只查文件存在、键存在或恢复后内部自洽，都不能证明持久化与跨 rank 重分片的数值正确。

<a id="execution"></a>
## 9. 断言未执行或失败未传播

**典型写法**

```python
def test_save_load(self):
    pass                                   # 覆写基类用例，什么都不跑

class MixinChecks:                          # 不继承 TestCase，用例不会被收集
    def test_merge_single_element(self):
        out = merge_splited_param(state)
        self.assertIn("beta1_pow_acc", out)  # 只查键，且这个类根本不被执行

def run_all(self):
    for f in self.collected_files:
        subprocess.run([sys.executable, f])  # 不看返回码
```

**为什么漏检。** `pass` 覆写把声称验证 save/load 的用例清空，实现坏掉也“通过”；
断言写在没被 runner 收集的 mixin/裸类里，永远不执行；launcher 直接跑文件却不校验退出码，
子进程里 `assert` 失败、collect 为零、全 skip 都被吞成成功。多卡文件只定义 pytest 函数、
由 `paddle.distributed.launch` 直接执行时，同样可能一条断言都没跑。

**修正写法**

```python
def test_save_load(self):
    self._run_save_load_contract(self.model_cls, atol=1e-5)  # 真正执行契约

class MergeParamTest(unittest.TestCase):    # 继承 TestCase，确保被收集
    def test_merge_single_element(self):
        out = merge_splited_param(make_state(beta1=paddle.to_tensor([1., 2.])))
        np.testing.assert_array_equal(out["beta1_pow_acc"].numpy(), [1., 2.])

def run_all(self):
    for f in self.collected_files:
        r = subprocess.run([sys.executable, f], capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr.decode())  # 失败必须传播
        self.assertIn("Ran", r.stderr.decode())               # 确认真的跑了用例
```

覆写基类用例要么实现真实契约、要么按不支持条件显式 `skipTest` 并记录原因，不能留空体。
断言必须落在会被 runner 收集执行的用例里；由 launcher/子进程执行的路径要校验退出码和实际
运行的用例数，某个 rank 的 `OK`、零收集或全 skip 都不能当作目标行为通过。

<a id="exceptions"></a>
## 10. 异常被吞成成功或 skip

**典型写法**

```python
def test_recv_meta_reverse(self):
    try:
        obj._recv_meta(reverse=True)       # 抛错也没关系
    except Exception:
        pass
    self.assertFalse(obj.has_cache)         # 断言的还是构造时就为 False 的标记

def test_init_with_tensorboard(self):
    try:
        cb = TensorBoardCallback(output_dir=tmp)
    except Exception:
        pass                                # 初始化失败被吞，无成功路径观察

@unittest.skipUnless(_can_import(), "no dep")
def _can_import():
    try:
        import paddlefleet.triton_ops  # noqa
        return True
    except Exception:                       # 任何错误都当“缺依赖”跳过
        return False
```

**为什么漏检。** `except Exception: pass` 把被测入口的真实错误全部吞掉，实现第一行抛异常也“通过”；
断言的对象要么是构造时早已确定的值，要么根本没有。用宽泛 `except` 探测能力会把
真实的 import 错误、编译失败、API 变更误判成“环境缺依赖”而 skip，掩盖回归。

**修正写法**

```python
def test_recv_meta_reverse(self):
    meta = encode_meta(shapes=[[2, 3]], dtypes=["float32"], src=1)
    decoded = obj._recv_meta(meta, reverse=True)   # 真实解码，不吞异常
    self.assertEqual(decoded.src_rank, 1)
    self.assertEqual(decoded.shapes, [[2, 3]])

def test_recv_meta_rejects_corrupt(self):
    with self.assertRaises(MetaDecodeError):        # 明确异常契约
        obj._recv_meta(b"\x00\x01", reverse=True)

def test_triton_ops_importable(self):
    import importlib
    mod = importlib.import_module("paddlefleet.triton_ops")  # 精确探测
    self.assertTrue(hasattr(mod, "fused_rms_norm"))
```

不要用 `try/except: pass` 包住被测调用。预期成功就断言真实输出，预期失败就用 `assertRaises`
锁定具体异常类型；能力探测只捕获精确的 `ImportError`/`ModuleNotFoundError` 并记录 skip 原因，
不把编译或 API 错误也当作缺依赖。

<a id="state"></a>
## 11. 共享状态污染或自造前提

**典型写法**

```python
def test_register_cls_attr(self):
    GeneralModelForCausalLMPipe.config_class = DummyConfig   # 直接改生产类属性
    GeneralModelForCausalLMPipe._init_weights = "noop"
    self.assertEqual(GeneralModelForCausalLMPipe.config_class, DummyConfig)
    # 没有 tearDown / addCleanup，污染后续真实 PP 模型构造

def test_expert_rank(self):
    parallel_state._EXPERT_PARALLEL_RANK = 3                 # 设全局拓扑不还原
    self.assertEqual(get_expert_model_parallel_rank(), 3)

def test_tracker_empty(self):
    tracker.clear()
    self.assertEqual(tracker.values, [])                    # 清空后断言为空，恒真
```

**为什么漏检。** 前两例直接改生产类属性/全局拓扑且无 `tearDown`、`addCleanup` 或 `finally`
恢复，即使本例断言成立，也会把伪配置、伪 rank 留给同进程后续测试，制造隐性串扰；
断言的还是自己刚设进去的值，等于自证。最后一例先 `clear()` 再断言为空是自造前提，
无论生产逻辑对错都成立。RNG seed、环境变量、导入时缓存的后端开关同理。

**修正写法**

```python
def test_register_cls_attr(self):
    class _Probe(GeneralModelForCausalLMPipe):   # 在独立子类上验证，不碰生产类
        pass
    register_cls_attr(_Probe, config_class=DummyConfig)
    self.assertIs(_Probe.config_class, DummyConfig)

def test_expert_rank(self):
    orig = parallel_state._EXPERT_PARALLEL_RANK
    self.addCleanup(lambda: setattr(parallel_state, "_EXPERT_PARALLEL_RANK", orig))
    parallel_state._EXPERT_PARALLEL_RANK = 3
    self.assertEqual(get_expert_model_parallel_rank(), 3)   # 失败时也会清理

def test_tracker_accumulates(self):
    tracker.clear()
    self.addCleanup(tracker.clear)
    helper.record(0.5); helper.record(1.5)      # 走真实生产写入路径
    self.assertEqual(tracker.values, [0.5, 1.5])  # 断言生产行为，而非自己塞的值
```

修改全局/类属性、RNG、环境变量或缓存的用例必须保存原值并在 `addCleanup`/`tearDown` 里恢复，
断言失败时也要执行清理；不要直接改生产类，改用独立子类或 `patch.object`。若目标是“清空后
避免重复查询”，清洁初始状态后应断言生产逻辑产生的结果，而不是断言刚被自己清空的容器为空。

<a id="mock"></a>
## 12. 随意 mock 盖住被测逻辑

mock 用来隔离**不被测**的协作者是合法的（见[类型 3](#consumption)）；本类型针对另一种失效：
替身盖住的恰好是本用例声称要验证的行为，于是无论断言怎么写都无法拒绝错误实现。两个高频信号：

**（1）patch 掉被测对象的构造或被测方法本身**

```python
def test_trainer_setup(self):
    with mock.patch.object(SFTTrainer, "__init__", return_value=None):
        trainer = SFTTrainer.__new__(SFTTrainer)   # __init__ 被架空
        trainer.args = FakeArgs()
        trainer.setup_something()
    self.assertEqual(trainer.args.x, 1)             # 只断言自己塞进去的属性
```

`__init__` 被替换成 `return None`，对象成了空壳；真实构造里的参数校验、组件装配、状态初始化
一行都没跑，改坏 `__init__` 用例照过。criterion、量化 linear 的构造被 patch 成 `None` 同理。
应保留真实构造，用小配置实例化后断言可观察的组件/参数/状态；确要跳过某个昂贵子步骤时，
只 mock 那个子步骤并核对它被真实构造以正确参数调用。

**（2）mock 掉被测 kernel/算子，替身返回与输入无关的张量，只查 shape 或 called**

```python
@mock.patch("....attention.scaled_dot_product_attention")
def test_attention_forward(self, sdpa):
    sdpa.return_value = paddle.randn([2, 8, 4, 16])   # 与 q/k/v/mask 无关
    out = sdpa_attention(module, q, k, v, mask, is_causal=None)
    self.assertEqual(out.shape, [2, 8, 4, 16])
    sdpa.assert_called_once()
```

被测的正是这条注意力路径，核心计算却被随机张量顶替：`is_causal` 推断、mask 传递、
`softmax_scale`、GQA 展开是否正确，输出全由 `randn` 决定，测不出来。
应对支持 CPU 的算子跑真实计算并与独立参考比数值；确需替身时，让替身按输入返回可区分标记，
并断言传入 kernel 的 q/k/v/mask/scale/causal 等实参精确正确（见[类型 3](#consumption) 的 marker 写法），
同时声明未验证真实 kernel 数值。

**边界。** mock 一个真正不被测的协作者、给它可区分响应再核对消费，是[类型 3](#consumption)的正确做法；
本类型的错误在于 mock 的对象就是被测逻辑或决定其正确性的依赖，此时任何断言都无法证明目标行为。
伪造 world_size、mock 集合通信把单卡冒充多卡的写法归入[类型 13](#single-card)。

<a id="single-card"></a>
## 13. 多卡行为只在单卡/单进程取证

切分、跨 rank 通信、专家分发、重分片、并行恢复等行为的正确性取决于**多个 rank 各自持有不同数据后交换的结果**。
若只在单卡、单进程或 CPU 手算里取证再宣称这些行为已验证，则发错方向、peer 取错、切分尺寸错、
归约漏项都不会被拒绝——本需真实进程组的语义被本地路径冒充。三个高频信号：

**（1）伪造 world_size/拓扑 + mock collective，只 assert_called**

```python
@mock.patch("....all_to_all.dist.get_world_size", return_value=4)
@mock.patch("....all_to_all.stream.alltoall_single")
def test_forward_multi_rank(self, alltoall, _ws):
    alltoall.return_value = None
    out = AllToAll.apply(x, group=MagicMock(), sync_op=True)
    alltoall.assert_called_once()

def test_p2p_ops_send_recv(self):
    with mock.patch("paddle.distributed.isend", return_value=MagicMock()), \
         mock.patch("paddle.distributed.irecv", return_value=MagicMock()):
        reqs = _p2p_ops(tensor, None, tensor, None, MagicMock())
    self.assertGreater(len(reqs), 0)      # 只数排队的通信请求
```

`get_world_size` 写成 4 以进入“多卡”分支，实际单进程；alltoall/all_gather/reduce_scatter/isend
被替身吞掉，只断言“被调用”或“请求数 > 0”。切分尺寸、专家归位、group/轴、收发方向、peer 取错、
发错 tensor 都不影响这些断言。

**（2）CPU 手算或本地循环模拟 collective，替代真实进程组**

```python
def test_all_gather_shard(self):
    shards = [make_shard(r) for r in range(4)]     # 单进程造出 4 份
    gathered = paddle.concat(shards, axis=0)       # 手动拼接“模拟” all_gather
    expected = paddle.concat(shards, axis=0)       # 与上面同一路拼接
    np.testing.assert_array_equal(gathered.numpy(), expected.numpy())
```

拼接、比较全在本地完成，`all_gather` 本身、rank 排序、通信后 contiguous/dualchunk 的位置映射
一次都没跑；expected 又与被测走同一条本地拼接（叠加[类型 4](#reference)），改坏真实聚合仍过。

**（3）只跑单卡 helper/局部分片函数，宣称跨 rank 重分布正确**

```python
def test_muon_ep_slice(self):
    local = slice_expert_weight(full_w, ep_rank=0, ep_size=4)   # 只取 rank0 的本地块
    self.assertEqual(local.shape, [full_w.shape[0] // 4, full_w.shape[1]])
```

只验证单个 rank 的本地切分形状，各 rank 内容是否互不重叠、拼回是否等于完整权重、
EP 归约后的更新是否正确都没触及；reshard 只查元数据键映射同理，真实张量重分片没有发生。

**修正方向。** 通信、分片、专家分发、重分片、并行恢复在受支持的真实进程组里运行（卡数按待测拓扑选取），
各 rank 用可区分内容，读取前等待异步任务完成，核对**对端实际收到**的分片/张量与方向、还原后的聚合结果
及相关梯度，并验证任一 rank 失败可传播（见[分布式训练模块](unit-test-rules.md)的多卡要求）。

**边界。** 接口本身支持的 world-size=1、本地分片或单卡回退路径，单卡取证是恰当的，但只能声明验证了该本地路径；
错误在于把单卡/单进程/CPU 手算结果表述为跨 rank 通信或分布式数值已验证。无卡下允许用通信替身检查
group/axis/peer 与被发送内容，但须显式声明真实进程组未运行（见[类型 12](#mock) 的 mock 边界）。
