# Pipeline Parallel Tests / 流水线并行模块测试

Unit tests for PaddleFleet pipeline parallel communication, P2P operations, schedule nodes, and VPP simulator. / PaddleFleet 流水线并行通信、P2P 操作、调度节点和 VPP 模拟器的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_forward_backward_overlap_utils.py` | Tests for FakeClone in forward_backward_overlap_utils / 测试前向反向重叠工具中的 FakeClone |
| `test_coverage_forward_backward_overlap_utils_2.py` |  |
| `test_coverage_forward_backward_overlap_utils_3.py` |  |
| `test_coverage_forward_backward_overlap_utils_4.py` |  |
| `test_coverage_four_directions_p2p_communication.py` | Tests for XPU communication group management / 测试四方向 P2P 通信组管理 |
| `test_coverage_four_directions_p2p_communication_2.py` |  |
| `test_coverage_four_directions_p2p_communication_3.py` |  |
| `test_coverage_four_directions_p2p_communication_4.py` |  |
| `test_coverage_four_directions_p2p_communication_5.py` |  |
| `test_coverage_four_directions_p2p_communication_6.py` |  |
| `test_coverage_four_directions_p2p_communication_7.py` |  |
| `test_coverage_moe_router.py` | Tests for the p2p_overlap_dw_calc deferral points in FusedGateDetachMatmul, TopKRouter, DeferredWeightGradLinear / 测试 dW 延后计算掩盖 P2P 的各个生效点 |
| `test_coverage_overlap_edge_cases.py` | Edge case tests for detach_and_requires_grad / 测试 detach_and_requires_grad 边界情况 |
| `test_coverage_p2p_communication.py` | Tests for SendRecvMeta class / 测试收发元数据类 |
| `test_coverage_p2p_communication_2.py` |  |
| `test_coverage_p2p_communication_3.py` |  |
| `test_coverage_p2p_communication_4.py` |  |
| `test_coverage_p2p_communication_5.py` |  |
| `test_coverage_p2p_communication_6.py` |  |
| `test_coverage_p2p_communication_7.py` |  |
| `test_coverage_p2p_communication_8.py` |  |
| `test_coverage_p2p_communication_9.py` |  |
| `test_coverage_pipeline_hooks.py` | Unit tests for pipeline_hooks module / 测试流水线钩子模块 |
| `test_coverage_pipeline_parallel.py` | Tests for get_action function / 测试流水线动作获取函数 |
| `test_coverage_pipeline_parallel_2.py` |  |
| `test_coverage_pipeline_parallel_withinterleave.py` | Tests for P2PAsyncHandle dataclass / 测试 P2P 异步句柄数据类 |
| `test_coverage_pipeline_parallel_withinterleave_fthenb.py` | Tests for PipelineParallelWithInterleaveFthenB / 测试交错流水线前向后向并行 |
| `test_coverage_pp_layers.py` | Tests for ScheduleChunk / 测试调度块 |
| `test_coverage_pp_utils.py` | Tests for paddle_2_number conversion / 测试 Paddle 数值转换 |
| `test_coverage_utils.py` |  |
| `test_coverage_utils_2.py` |  |
| `test_coverage_utils_3.py` |  |
| `test_coverage_utils_4.py` |  |
| `test_coverage_utils_5.py` |  |
| `test_coverage_utils_6.py` |  |
| `test_coverage_vpp_balanced_memory.py` | Tests for OffloadQueue / 测试卸载队列 |
| `test_coverage_vpp_simulator.py` | Tests for ChunkType enum / 测试块类型枚举 |
| `test_coverage_vpp_simulator_2.py` |  |
| `test_coverage_vpp_simulator_3.py` |  |
| `test_coverage_vpp_simulator_4.py` |  |
