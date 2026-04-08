#!/usr/bin/env python3
"""Similarity-based residual calibration for costmodel outputs.

This module deliberately avoids direct config->log lookup. Instead it stores a
small set of calibration samples in feature space and applies a smooth kernel
residual correction on top of the analytic cost model.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

from .runtime_model_calibration import canonicalize_model_name, default_model_key_from_config

logger = logging.getLogger(__name__)

_SUPPORTED_VERSION = 1


def _safe_log(value: float, eps: float = 1e-6) -> float:
    return math.log(max(float(value or 0.0), eps))


def _safe_log2_ratio(value: float, reference: float) -> float:
    value = max(float(value or 0.0), 1e-12)
    reference = max(float(reference or 0.0), 1e-12)
    return math.log(value / reference, 2.0)


def build_similarity_feature_vector(result: Any, context: Optional[Dict[str, Any]]) -> List[float]:
    ctx = dict(context or {})
    pp = max(1, int(ctx.get("pp", 1) or 1))
    tp = max(1, int(ctx.get("tp", 1) or 1))
    ep = max(1, int(ctx.get("ep", 1) or 1))
    dp = max(1, int(ctx.get("dp", 1) or 1))
    sharding_degree = max(1, int(ctx.get("sharding_degree", dp) or dp or 1))
    mbs = max(1, int(ctx.get("micro_batch_size", 1) or 1))
    seq = max(1, int(ctx.get("max_seq_len", 4096) or 4096))
    offload = 1.0 if bool(ctx.get("tensorwise_offload_optimizer", False)) else 0.0
    rec_g = str(ctx.get("recompute_granularity", "") or "").strip().lower()
    rec_m = str(ctx.get("recompute_method", "") or "").strip().lower()
    rec_none = 1.0 if rec_g in {"", "none", "null"} or rec_m in {"", "none", "null"} else 0.0
    rec_block = 1.0 if rec_m == "block" else 0.0
    rec_uniform = 1.0 if rec_m == "uniform" else 0.0
    sp = 1.0 if bool(ctx.get("sp", False)) else 0.0
    return [
        math.log2(pp) if pp > 1 else 0.0,
        math.log2(tp) if tp > 1 else 0.0,
        math.log2(ep) if ep > 1 else 0.0,
        math.log2(dp) if dp > 1 else 0.0,
        math.log2(sharding_degree) if sharding_degree > 1 else 0.0,
        _safe_log2_ratio(seq, 4096.0),
        math.log2(mbs) if mbs > 1 else 0.0,
        offload,
        sp,
        rec_none,
        rec_block,
        rec_uniform,
        _safe_log(getattr(result, "step_time_ms", 0.0)),
        _safe_log(getattr(result, "allocated_memory_gb", 0.0)),
        _safe_log(getattr(result, "reserved_memory_gb", 0.0)),
    ]




def _normalized_cluster_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 8)
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s


def _cluster_signature_from_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ctx = dict(context or {})
    fields = [
        "micro_batch_size",
        "max_seq_len",
        "split_param",
        "stage1_overlap",
        "sd_release_grads",
        "apply_rope_fusion",
        "moe_grouped_gemm",
        "moe_ep_barrier",
        "moe_router_fusion",
        "moe_router_force_load_balancing",
        "router_aux_loss_coef",
        "pp_delay_scale_loss",
        "overlap_p2p_comm",
        "variable_seq_lengths",
        "best_unbalanced_scheduler",
    ]
    signature: Dict[str, Any] = {}
    for field in fields:
        signature[field] = _normalized_cluster_value(ctx.get(field))
    return signature


def _matching_cluster_indices(entry: Dict[str, Any], runtime_context: Optional[Dict[str, Any]]):
    cluster_fields = list(entry.get("cluster_fields") or [])
    sample_clusters = list(entry.get("sample_clusters") or [])
    if not cluster_fields or not sample_clusters:
        return None
    query = _cluster_signature_from_context(runtime_context)
    matched = []
    for idx, cluster in enumerate(sample_clusters):
        if not isinstance(cluster, dict):
            continue
        ok = True
        for field in cluster_fields:
            if cluster.get(field) != query.get(field):
                ok = False
                break
        if ok:
            matched.append(idx)
    return matched if matched else None


class SimilarityCalibrationStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or str(Path(__file__).with_name("similarity_calibrations.json"))
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        path_obj = Path(self.path)
        if not path_obj.exists():
            return {"version": _SUPPORTED_VERSION, "models": {}}
        try:
            data = json.loads(path_obj.read_text())
            if not isinstance(data, dict):
                raise ValueError("similarity calibration store must be a JSON object")
            data.setdefault("version", _SUPPORTED_VERSION)
            data.setdefault("models", {})
            return data
        except Exception as exc:
            logger.warning("failed to load similarity calibration store %s: %s", self.path, exc)
            return {"version": _SUPPORTED_VERSION, "models": {}}

    def save(self) -> None:
        Path(self.path).write_text(json.dumps(self.data, indent=2, sort_keys=True))

    def persist(self) -> None:
        self.save()

    def get(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.data.get("models", {}).get(canonicalize_model_name(model_name))

    def upsert(self, model_name: str, entry: Dict[str, Any], persist: bool = True) -> None:
        self.data.setdefault("models", {})[canonicalize_model_name(model_name)] = entry
        if persist:
            self.save()


class SimilarityCalibrationFitter:
    def __init__(self, costmodel: Any, store: SimilarityCalibrationStore):
        if np is None:
            raise RuntimeError(f"numpy is required for similarity calibration: {_NUMPY_IMPORT_ERROR}")
        self.costmodel = costmodel
        self.store = store

    def _build_parallel(self, obs: Any) -> Any:
        from ..config import ParallelConfig
        return ParallelConfig(
            pp=int(obs.pp or 1),
            tp=int(obs.tp or 1),
            dp=int(obs.dp or 1),
            ep=int(obs.ep or 1),
            sp=bool(obs.sp),
            cp=int(obs.cp or 1),
            sharding=str(obs.sharding or "stage1"),
            sharding_degree=int(obs.sharding_degree if obs.sharding_degree is not None else (obs.dp or 1)),
        )

    def _predict_raw(self, obs: Any):
        parallel = self._build_parallel(obs)
        result = self.costmodel.predict(
            parallel,
            micro_batch_size=obs.micro_batch_size,
            max_seq_len=obs.max_seq_len,
            gradient_accumulation_steps=obs.gradient_accumulation_steps,
            recompute_granularity=obs.recompute_granularity,
            recompute_method=obs.recompute_method,
            recompute_num_layers=obs.recompute_num_layers,
            tensorwise_offload_optimizer=obs.tensorwise_offload_optimizer,
            tensorwise_offload_ratio=obs.tensorwise_offload_ratio,
            split_param=obs.runtime_context.get("split_param", True),
            sd_release_grads=obs.runtime_context.get("sd_release_grads", None),
            apply_runtime_calibration=False,
            apply_similarity_calibration=False,
        )
        context = {
            "pp": int(parallel.pp),
            "tp": int(parallel.tp),
            "dp": int(parallel.dp),
            "ep": int(parallel.ep),
            "sp": bool(parallel.sp),
            "sharding_degree": int(getattr(parallel, "sharding_degree", obs.sharding_degree or obs.dp or 1) or 1),
            "micro_batch_size": int(obs.micro_batch_size or 1),
            "gradient_accumulation_steps": int(obs.gradient_accumulation_steps or 1),
            "max_seq_len": int(obs.max_seq_len or 4096),
            "tensorwise_offload_optimizer": bool(obs.tensorwise_offload_optimizer),
            "recompute_granularity": obs.recompute_granularity,
            "recompute_method": obs.recompute_method,
        }
        return parallel, result, context

    def fit_from_observations(self, model_name: str, observations: Sequence[Any], persist: bool = True) -> Dict[str, Any]:
        rows = []
        for obs in observations:
            if str(getattr(obs, "status", "")).upper() != "OK":
                continue
            actual_step_s = obs.preferred_step_time_s()
            if not actual_step_s or not obs.peak_allocated_gb or not obs.peak_reserved_gb:
                continue
            try:
                _, result, context = self._predict_raw(obs)
            except Exception:
                continue
            feature = build_similarity_feature_vector(result, context)
            rows.append({
                "source": str(getattr(obs, "source", "") or getattr(obs, "config_name", "")),
                "feature": feature,
                "cluster": _cluster_signature_from_context(context),
                "actual_step_time_ms": float(actual_step_s) * 1000.0,
                "actual_allocated_gb": float(obs.peak_allocated_gb),
                "actual_reserved_gb": float(obs.peak_reserved_gb),
                "base_step_time_ms": float(getattr(result, "step_time_ms", 0.0) or 0.0),
                "base_allocated_gb": float(getattr(result, "allocated_memory_gb", 0.0) or 0.0),
                "base_reserved_gb": float(getattr(result, "reserved_memory_gb", 0.0) or 0.0),
            })
        if not rows:
            raise ValueError("no valid OK observations available for similarity calibration")
        X = np.asarray([row["feature"] for row in rows], dtype=float)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        X_norm = ((X - mean) / std).tolist()
        entry = {
            "version": _SUPPORTED_VERSION,
            "fit_kind": "kernel_residual_v1",
            "feature_names": [
                "log2_pp", "log2_tp", "log2_ep", "log2_dp", "log2_sharding_degree",
                "log2_seq_over_4096", "log2_mbs", "offload", "sp", "recompute_none",
                "recompute_block", "recompute_uniform", "log_base_step_ms",
                "log_base_allocated_gb", "log_base_reserved_gb",
            ],
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "samples_norm": X_norm,
            "residual_logs": {
                "step_time_ms": [math.log(max(row["actual_step_time_ms"], 1e-6) / max(row["base_step_time_ms"], 1e-6)) for row in rows],
                "allocated_memory_gb": [math.log(max(row["actual_allocated_gb"], 1e-6) / max(row["base_allocated_gb"], 1e-6)) for row in rows],
                "reserved_memory_gb": [math.log(max(row["actual_reserved_gb"], 1e-6) / max(row["base_reserved_gb"], 1e-6)) for row in rows],
            },
            "bandwidths": {
                "step_time_ms": 0.35,
                "allocated_memory_gb": 0.35,
                "reserved_memory_gb": 0.35,
            },
            "top_k": 8,
            "cluster_fields": [
                "micro_batch_size",
                "max_seq_len",
                "split_param",
                "stage1_overlap",
                "sd_release_grads",
                "apply_rope_fusion",
                "moe_grouped_gemm",
                "moe_ep_barrier",
                "moe_router_fusion",
                "moe_router_force_load_balancing",
                "router_aux_loss_coef",
                "pp_delay_scale_loss",
                "overlap_p2p_comm",
                "variable_seq_lengths",
                "best_unbalanced_scheduler",
            ],
            "sample_clusters": [row["cluster"] for row in rows],
            "fit_summary": {
                "num_ok_observations": len(rows),
                "sources": [row["source"] for row in rows],
                "cluster_counts": {
                    json.dumps(cluster, sort_keys=True): sum(1 for row in rows if row["cluster"] == cluster)
                    for cluster in [dict(x) for x in {tuple(sorted(row["cluster"].items())) for row in rows}]
                },
            },
        }
        self.store.upsert(model_name, entry, persist=False)
        # Also store under the architecture-derived default key so callers that
        # build ModelConfig directly (without a human-readable model name field)
        # can still load the calibration automatically.
        try:
            default_key = default_similarity_model_key_from_config(getattr(self.costmodel, "model_config", None))
        except Exception:
            default_key = None
        if default_key and default_key != canonicalize_model_name(model_name):
            self.store.upsert(default_key, entry, persist=False)
        if persist:
            self.store.save()
        return entry


def _weighted_residual_log(entry: Dict[str, Any], metric_name: str, feature_vector: Sequence[float], runtime_context: Optional[Dict[str, Any]] = None) -> Optional[float]:
    if np is None:
        return None
    samples = np.asarray(entry.get("samples_norm") or [], dtype=float)
    if samples.size == 0:
        return None
    mean = np.asarray(entry.get("feature_mean") or [], dtype=float)
    std = np.asarray(entry.get("feature_std") or [], dtype=float)
    if mean.size == 0 or std.size == 0:
        return None
    feature = np.asarray(feature_vector, dtype=float)
    if feature.shape[-1] != mean.shape[-1]:
        return None
    feature_norm = (feature - mean) / np.where(std < 1e-9, 1.0, std)
    residual_logs = np.asarray((entry.get("residual_logs") or {}).get(metric_name) or [], dtype=float)
    if residual_logs.size != samples.shape[0]:
        return None

    matched_indices = _matching_cluster_indices(entry, runtime_context)
    if matched_indices:
        samples = samples[matched_indices]
        residual_logs = residual_logs[matched_indices]
        if residual_logs.size == 0:
            return None

    bandwidth = float((entry.get("bandwidths") or {}).get(metric_name, 0.8) or 0.8)
    bandwidth = max(1e-3, bandwidth)
    diff = samples - feature_norm.reshape(1, -1)
    dist2 = np.sum(diff * diff, axis=1)
    weights = np.exp(-dist2 / (2.0 * bandwidth * bandwidth))
    top_k = int(entry.get("top_k", 12) or 12)
    if 0 < top_k < weights.size:
        order = np.argsort(dist2)[:top_k]
        weights = weights[order]
        residual_logs = residual_logs[order]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None
    return float(np.sum(weights * residual_logs) / weight_sum)


def apply_similarity_calibration_to_result(*,
                                           result: Any,
                                           calibration_entry: Optional[Dict[str, Any]],
                                           runtime_context: Optional[Dict[str, Any]],
                                           gpu_memory_gb: float = 0.0) -> Any:
    if not calibration_entry:
        return result
    if str(calibration_entry.get("fit_kind", "") or "") != "kernel_residual_v1":
        return result
    feature_vector = build_similarity_feature_vector(result, runtime_context)
    metrics = [
        ("step_time_ms", None),
        ("allocated_memory_gb", "memory_gb"),
        ("reserved_memory_gb", "memory_gb"),
    ]
    for metric_name, coupled_name in metrics:
        residual_log = _weighted_residual_log(calibration_entry, metric_name, feature_vector, runtime_context)
        if residual_log is None:
            continue
        base_value = float(getattr(result, metric_name, 0.0) or 0.0)
        corrected = max(1e-9, base_value * math.exp(residual_log))
        setattr(result, metric_name, corrected)
        if coupled_name == "memory_gb":
            result.memory_gb = max(float(getattr(result, "memory_gb", 0.0) or 0.0), corrected)
    result.fits_memory = True if gpu_memory_gb <= 0 else (float(getattr(result, "reserved_memory_gb", 0.0) or 0.0) <= float(gpu_memory_gb))
    result.parallel_config["similarity_calibration_applied"] = True
    return result


def default_similarity_model_key_from_config(model_config: Any) -> str:
    return canonicalize_model_name(default_model_key_from_config(model_config))
