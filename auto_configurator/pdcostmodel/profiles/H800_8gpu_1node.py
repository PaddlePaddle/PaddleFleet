#!/usr/bin/env python3
"""
硬件校准配置文件
GPU: NVIDIA H800
数量: 8
节点: 1
校准时间: 2026-02-27T12:49:39.929541
"""

HARDWARE_PROFILE = {
    "gpu_name": "NVIDIA H800",
    "gpu_memory_gb": 79.65,
    "gpu_count": 8,
    "fp32_tflops": 51.63,
    "fp16_tflops": 765.12,
    "bf16_tflops": 798.07,
    "memory_bandwidth_gbps": 2788.47,
    "intra_node_bandwidth_gbps": 0.0,
    "calibrated": True,
    "bf16_curve": {
        "dtype": "bfloat16",
        "peak_tflops": 798.07,
        "fit_a": 0.246402,
        "fit_b": -1.248085,
        "fit_max": 1.0,
        "points": [
            {
                "size": 64,
                "tflops": 0.04,
                "efficiency": 0.0001,
                "time_ms": 0.012,
            },
            {
                "size": 128,
                "tflops": 0.34,
                "efficiency": 0.0004,
                "time_ms": 0.012,
            },
            {
                "size": 256,
                "tflops": 2.75,
                "efficiency": 0.0034,
                "time_ms": 0.012,
            },
            {
                "size": 512,
                "tflops": 21.84,
                "efficiency": 0.0274,
                "time_ms": 0.012,
            },
            {
                "size": 1024,
                "tflops": 171.36,
                "efficiency": 0.2147,
                "time_ms": 0.013,
            },
            {
                "size": 2048,
                "tflops": 621.37,
                "efficiency": 0.7786,
                "time_ms": 0.028,
            },
            {
                "size": 4096,
                "tflops": 782.29,
                "efficiency": 0.9802,
                "time_ms": 0.176,
            },
            {
                "size": 8192,
                "tflops": 790.69,
                "efficiency": 0.9907,
                "time_ms": 1.391,
            },
        ],
    },
    "node_count": 1,
    "calibrated_at": "2026-02-27T12:49:39.929541",
    "pdcost_version": "1.0.0",
}
