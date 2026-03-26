#!/usr/bin/env python3
"""单配置预测示例。"""

import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
pkg = importlib.import_module(PACKAGE_NAME)

ModelConfig = pkg.ModelConfig
ParallelConfig = pkg.ParallelConfig
PDCostModel = pkg.PDCostModel
get_hardware_config = pkg.get_hardware_config


MODEL_NAME_OR_PATH = Path("/root/paddlejob/workspace/env_run/zhangdongqi/Qwen3-30B-A3B-Base")
NODE_COUNT = 1

PARALLEL = ParallelConfig(tp=1, pp=2, dp=4, ep=4, sharding="stage1")

PREDICT_ARGS = {
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 64,
    "max_seq_len": 8192,
    "recompute_granularity": "none",
    # "recompute_method": "uniform",
    # "recompute_num_layers": 1,
    "tensorwise_offload_optimizer": True,
    "split_param": True,
    "sd_release_grads": True,
    "overlap_p2p_comm": True,
    "stage1_overlap": True,
    "variable_seq_lengths": True,
    "best_unbalanced_scheduler": True,
    "attn_implementation": "flashmask",
    "apply_rope_fusion": True,
    "moe_grouped_gemm": True,
    "moe_router_fusion": True,
    "moe_ep_barrier": False,
}


def _load_model():
    model_json = MODEL_NAME_OR_PATH / "config.json"
    if model_json.exists():
        return ModelConfig.from_json(str(model_json))
    return ModelConfig.from_name("glm-4.5-air")


def main() -> None:
    model = _load_model()
    hardware = get_hardware_config(node_count=NODE_COUNT, verbose=False)
    costmodel = PDCostModel(model, hardware_config=hardware)

    result = costmodel.predict(PARALLEL, **PREDICT_ARGS)

    print("=== Predict Demo ===")
    print(f"Model:      {MODEL_NAME_OR_PATH}")
    print(f"Parallel:   {PARALLEL}")
    print(
        f"Hardware:   {hardware.gpu.name} "
        f"{hardware.num_nodes}x{hardware.gpus_per_node}"
    )
    print(f"Step(ms):   {result.step_time_ms:.2f}")
    print(f"Compute(ms): {result.compute_time_ms:.2f}")
    print(f"Optimizer(ms): {result.optimizer_step_time_ms:.2f}")
    print(f"Comm(ms):   {result.total_comm_time_ms:.2f}")
    print(f"EP Comm(ms): {result.ep_comm_time_ms:.2f}")
    print(f"PP Comm(ms): {result.pp_comm_time_ms:.2f}")
    print(f"DP Exp(ms): {result.dp_exposed_comm_time_ms:.2f}")
    print(f"Allocated:  {result.allocated_memory_gb:.2f} GB")
    print(f"Reserved:   {result.reserved_memory_gb:.2f} GB")
    print(f"Tok/s/GPU:  {result.tokens_per_second_per_gpu:.0f}")
    print(f"Fits:       {result.fits_memory}")


if __name__ == "__main__":
    main()
