#!/usr/bin/env python3
"""Per-model runtime calibration from raw logs.

This module keeps calibration low-dimensional and component-aware:
- it parses raw logs (not preprocessed tables)
- it preserves the costmodel's native runtime/context parameters when available
- it fits residuals on top of *sub-components* instead of scaling the whole step
  time / memory with one global coefficient.

The fitted parameters are stored in a single JSON file keyed by model name.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}
_GIB = 1024.0 ** 3
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", str(text or ""))


# ---------------------------------------------------------------------------
# Model naming helpers
# ---------------------------------------------------------------------------

def canonicalize_model_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    if not raw:
        return "unknown-model"
    raw = raw.replace(" ", "_").replace("/", "_").replace("\\", "_")
    raw = re.sub(r"[^a-z0-9._-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "unknown-model"


def default_model_key_from_config(model_config: Any) -> str:
    family = getattr(model_config, "model_type", None) or getattr(model_config, "model_family", None) or "model"
    layers = int(getattr(model_config, "num_hidden_layers", 0) or 0)
    hidden = int(getattr(model_config, "hidden_size", 0) or 0)
    inter = int(getattr(model_config, "intermediate_size", 0) or 0)
    heads = int(getattr(model_config, "num_attention_heads", 0) or 0)
    kv = int(getattr(model_config, "num_key_value_heads", 0) or 0)
    experts = int(getattr(model_config, "num_experts", 1) or 1)
    return canonicalize_model_name(
        f"{family}_l{layers}_h{hidden}_i{inter}_a{heads}_kv{kv}_e{experts}"
    )


# ---------------------------------------------------------------------------
# Supported runtime/context fields
# ---------------------------------------------------------------------------

_SUPPORTED_FIELD_SPECS: List[Dict[str, Any]] = [
    {"name": "pp", "type": "int", "aliases": ["pp", "pipeline_model_parallel_size", "pipeline_parallel_size"]},
    {"name": "tp", "type": "int", "aliases": ["tp", "tensor_model_parallel_size", "tensor_parallel_size"]},
    {"name": "dp", "type": "int", "aliases": ["dp", "data_parallel_size"]},
    {"name": "ep", "type": "int", "aliases": ["ep", "expert_model_parallel_size", "expert_parallel_size"]},
    {"name": "sp", "type": "bool", "aliases": ["sp", "sequence_parallel"]},
    {"name": "cp", "type": "int", "aliases": ["cp", "context_parallel_size"]},
    {"name": "sharding", "type": "str", "aliases": ["sharding", "sharding_stage"]},
    {"name": "sharding_degree", "type": "int", "aliases": ["sharding_degree", "sharding_parallel_size"]},
    {"name": "micro_batch_size", "type": "int", "aliases": ["micro_batch_size", "per_device_train_batch_size"]},
    {"name": "gradient_accumulation_steps", "type": "int", "aliases": ["gradient_accumulation_steps", "accumulate_steps"]},
    {"name": "max_seq_len", "type": "int", "aliases": ["max_seq_len", "max_sequence_length", "seq_length"]},
    {"name": "split_param", "type": "bool", "aliases": ["split_param"]},
    {"name": "sd_release_grads", "type": "bool", "aliases": ["sd_release_grads", "release_grads"]},
    {"name": "recompute_granularity", "type": "str", "aliases": ["recompute_granularity"]},
    {"name": "recompute_method", "type": "str", "aliases": ["recompute_method"]},
    {"name": "recompute_num_layers", "type": "int", "aliases": ["recompute_num_layers", "recompute_layers"]},
    {"name": "tensorwise_offload_optimizer", "type": "bool", "aliases": ["tensorwise_offload_optimizer", "optimizer_offload"]},
    {"name": "tensorwise_offload_ratio", "type": "float", "aliases": ["tensorwise_offload_ratio", "offload_ratio"]},
    {"name": "overlap_p2p_comm", "type": "bool", "aliases": ["overlap_p2p_comm"]},
    {"name": "use_batch_p2p_comm", "type": "bool", "aliases": ["use_batch_p2p_comm"]},
    {"name": "p2p_cache_shape", "type": "bool", "aliases": ["p2p_cache_shape"]},
    {"name": "stage1_overlap", "type": "bool", "aliases": ["stage1_overlap"]},
    {"name": "enable_sharding_comm_overlap", "type": "bool", "aliases": ["enable_sharding_comm_overlap"]},
    {"name": "variable_seq_lengths", "type": "bool", "aliases": ["variable_seq_lengths"]},
    {"name": "enable_dynamic_shape", "type": "bool", "aliases": ["enable_dynamic_shape"]},
    {"name": "clear_every_step_cache", "type": "bool", "aliases": ["clear_every_step_cache"]},
    {"name": "best_unbalanced_scheduler", "type": "bool", "aliases": ["best_unbalanced_scheduler"]},
    {"name": "hybrid_parallel_topo_order", "type": "str", "aliases": ["hybrid_parallel_topo_order"]},
    {"name": "num_empty_layers_add_in_head", "type": "int", "aliases": ["num_empty_layers_add_in_head"]},
    {"name": "num_empty_layers_add_in_tail", "type": "int", "aliases": ["num_empty_layers_add_in_tail"]},
    {"name": "attn_implementation", "type": "str", "aliases": ["attn_implementation", "_attn_implementation"]},
    {"name": "apply_rope_fusion", "type": "bool", "aliases": ["apply_rope_fusion"]},
    {"name": "use_qk_norm", "type": "bool", "aliases": ["use_qk_norm"]},
    {"name": "moe_token_dispatcher_type", "type": "str", "aliases": ["moe_token_dispatcher_type"]},
    {"name": "moe_grouped_gemm", "type": "bool", "aliases": ["moe_grouped_gemm"]},
    {"name": "moe_router_fusion", "type": "bool", "aliases": ["moe_router_fusion"]},
    {"name": "moe_router_force_load_balancing", "type": "bool", "aliases": ["moe_router_force_load_balancing"]},
    {"name": "router_aux_loss_coef", "type": "float", "aliases": ["router_aux_loss_coef"]},
    {"name": "pp_delay_scale_loss", "type": "bool", "aliases": ["pp_delay_scale_loss"]},
    {"name": "moe_expert_fusion", "type": "bool", "aliases": ["moe_expert_fusion"]},
    {"name": "moe_shared_expert_overlap", "type": "bool", "aliases": ["moe_shared_expert_overlap"]},
    {"name": "moe_ep_barrier", "type": "bool", "aliases": ["moe_ep_barrier"]},
    {"name": "model_name_or_path", "type": "str", "aliases": ["model_name_or_path"]},
]

_ALIAS_TO_FIELD: Dict[str, Tuple[str, str]] = {}
for spec in _SUPPORTED_FIELD_SPECS:
    for alias in spec["aliases"]:
        norm = re.sub(r"[^a-z0-9]+", "", str(alias).lower())
        _ALIAS_TO_FIELD[norm] = (spec["name"], spec["type"])


def get_runtime_calibration_supported_fields() -> List[Dict[str, Any]]:
    return [dict(spec) for spec in _SUPPORTED_FIELD_SPECS]


_SUPPORTED_FIELD_NAMES: Tuple[str, ...] = tuple(spec["name"] for spec in _SUPPORTED_FIELD_SPECS)
_MINIMAL_SIGNATURE_FIELD_NAMES: Tuple[str, ...] = (
    "pp", "tp", "ep", "sp", "cp",
    "micro_batch_size", "gradient_accumulation_steps", "max_seq_len",
    "recompute_granularity", "recompute_method", "recompute_num_layers",
    "tensorwise_offload_optimizer",
)

_COARSE_SIGNATURE_FIELD_NAMES: Tuple[str, ...] = (
    "pp", "tp", "ep", "sp", "cp",
    "micro_batch_size", "gradient_accumulation_steps", "max_seq_len",
    "recompute_granularity", "recompute_method", "recompute_num_layers",
    "tensorwise_offload_optimizer", "tensorwise_offload_ratio",
    "attn_implementation", "apply_rope_fusion", "use_qk_norm",
    "overlap_p2p_comm", "use_batch_p2p_comm", "p2p_cache_shape",
    "stage1_overlap", "enable_sharding_comm_overlap", "variable_seq_lengths",
    "enable_dynamic_shape", "clear_every_step_cache", "best_unbalanced_scheduler",
)


def _normalize_signature_value(value: Any) -> Any:
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
    # Canonicalize sharding option strings like
    # "[<ShardingOption.SHARD_OP: 'stage1'>]" → "stage1"
    import re as _re
    m = _re.search(r"ShardingOption\.\w+:\s*'([^']+)'", s)
    if m:
        return m.group(1)
    return s


def _signature_mapping_from_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = dict(context or {})
    out: Dict[str, Any] = {}
    for name in _SUPPORTED_FIELD_NAMES:
        if name in src and src[name] is not None:
            out[name] = _normalize_signature_value(src[name])
    return out


def _signature_key_from_context(context: Optional[Dict[str, Any]]) -> str:
    mapping = _signature_mapping_from_context(context)
    items = [(name, mapping[name]) for name in _SUPPORTED_FIELD_NAMES if name in mapping]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _coarse_signature_key_from_context(context: Optional[Dict[str, Any]]) -> str:
    mapping = _signature_mapping_from_context(context)
    items = [(name, mapping[name]) for name in _COARSE_SIGNATURE_FIELD_NAMES if name in mapping]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _minimal_signature_key_from_context(context: Optional[Dict[str, Any]]) -> str:
    mapping = _signature_mapping_from_context(context)
    items = [(name, mapping[name]) for name in _MINIMAL_SIGNATURE_FIELD_NAMES if name in mapping]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except Exception:
        return None


def _parse_scalar_by_type(value: str, field_type: str) -> Any:
    value = str(value).strip().strip(",")
    if field_type == "bool":
        return _to_bool(value)
    if field_type == "int":
        return _to_int(value)
    if field_type == "float":
        return _to_float(value)
    return value.strip("'\"")


def _convert_to_seconds(value: float, unit: Optional[str]) -> float:
    u = (unit or "s").strip().lower()
    if u in {"ms", "msec", "millisecond", "milliseconds"}:
        return float(value) / 1000.0
    return float(value)


def _convert_to_gb(value: float, unit: Optional[str]) -> float:
    u = (unit or "gb").strip().lower().replace("ib", "b")
    if u in {"b", "byte", "bytes"}:
        return float(value) / _GIB
    if u in {"kb", "k"}:
        return float(value) / (1024.0 ** 2)
    if u in {"mb", "m"}:
        return float(value) / 1024.0
    if u in {"gb", "g"}:
        return float(value)
    if u in {"tb", "t"}:
        return float(value) * 1024.0
    return float(value)


@dataclass
class RuntimeObservation:
    source: str = ""
    config_name: str = ""
    status: str = "UNKNOWN"
    pp: Optional[int] = None
    tp: Optional[int] = None
    dp: Optional[int] = None
    ep: Optional[int] = None
    sp: Optional[bool] = None
    cp: Optional[int] = None
    sharding: Optional[str] = None
    sharding_degree: Optional[int] = None
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    tensorwise_offload_optimizer: Optional[bool] = None
    tensorwise_offload_ratio: Optional[float] = None
    micro_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    max_seq_len: Optional[int] = None
    avg_step_time_s: Optional[float] = None
    last5_avg_step_time_s: Optional[float] = None
    peak_allocated_gb: Optional[float] = None
    peak_reserved_gb: Optional[float] = None
    avg_tps: Optional[float] = None
    last5_avg_tps: Optional[float] = None
    total_steps: Optional[int] = None
    rounds: Optional[int] = None
    runtime_context: Dict[str, Any] = field(default_factory=dict)

    def preferred_step_time_s(self) -> Optional[float]:
        if self.last5_avg_step_time_s is not None and self.last5_avg_step_time_s > 0:
            return self.last5_avg_step_time_s
        return self.avg_step_time_s


_CONFIG_NAME_RE = re.compile(
    r"(?P<config>pp(?P<pp>\d+)tp(?P<tp>\d+)(?:ep(?P<ep>\d+))?(?P<rec>norec|blk|uni|sel|selective)?_(?P<off>off|nooff))",
    re.IGNORECASE,
)


def _status_from_text(text: str) -> str:
    lower = text.lower()
    if "oom" in lower or "out of memory" in lower:
        return "OOM"
    if re.search(r"fail\s*\(exit=\d+\)", lower) or "traceback" in lower or "error:" in lower:
        return "FAIL"
    if "peak_allocated" in lower or "avg_step" in lower or "last5_avg" in lower:
        return "OK"
    return "UNKNOWN"


def _extract_named_config(text: str, source_name: str = "") -> Tuple[Optional[str], Dict[str, Any]]:
    for probe in [source_name, text]:
        m = _CONFIG_NAME_RE.search(probe or "")
        if not m:
            continue
        config_name = str(m.group("config")).strip()
        rec = (m.group("rec") or "").lower()
        values: Dict[str, Any] = {
            "config_name": config_name,
            "pp": _to_int(m.group("pp")),
            "tp": _to_int(m.group("tp")),
            "ep": _to_int(m.group("ep")) if m.group("ep") else None,
            "tensorwise_offload_optimizer": ((m.group("off") or "").lower() == "off"),
        }
        if rec in {"norec", ""}:
            values["recompute_granularity"] = "none"
        elif rec in {"blk", "uni"}:
            values["recompute_granularity"] = "full"
            values["recompute_method"] = "block" if rec == "blk" else "uniform"
        elif rec in {"sel", "selective"}:
            values["recompute_granularity"] = "selective"
        return config_name, values
    return None, {}


def _extract_key_values(text: str) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    line_patterns = [
        re.compile(r"(?:-\s*)?(?P<key>[A-Za-z0-9_\-]+)\s*[:=]\s*(?P<value>.+)$"),
        re.compile(r'"(?P<key>[A-Za-z0-9_\-]+)"\s*:\s*(?P<value>.+?)(?:,)?$'),
    ]
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidates = [stripped]
        if " - " in stripped:
            candidates.insert(0, stripped.split(" - ", 1)[1].strip())
        matched = False
        for cand in candidates:
            for pat in line_patterns:
                m = pat.search(cand)
                if not m:
                    continue
                key = m.group("key")
                value = m.group("value")
                key_norm = re.sub(r"[^a-z0-9]+", "", key.lower())
                if key_norm not in _ALIAS_TO_FIELD:
                    continue
                field_name, field_type = _ALIAS_TO_FIELD[key_norm]
                parsed = _parse_scalar_by_type(value, field_type)
                if parsed is not None:
                    extracted[field_name] = parsed
                    matched = True
                    break
            if matched:
                break
    return extracted


def parse_runtime_log_text(text: str, source_name: str = "") -> RuntimeObservation:
    obs = RuntimeObservation(source=source_name)
    if not text:
        return obs

    text = _strip_ansi(text)
    obs.status = _status_from_text(text)
    config_name, config_values = _extract_named_config(text, source_name=source_name)
    if config_name:
        obs.config_name = config_name
        for key, value in config_values.items():
            if hasattr(obs, key):
                setattr(obs, key, value)

    kv = _extract_key_values(text)
    for key, value in kv.items():
        if hasattr(obs, key):
            if getattr(obs, key) is None:
                setattr(obs, key, value)
        obs.runtime_context[key] = value

    step_re = re.compile(
        r"global_step:\s*(?P<step>\d+).*?"
        r"max_memory_allocated:\s*(?P<alloc>[0-9]+(?:\.[0-9]+)?).*?"
        r"max_memory_reserved:\s*(?P<reserved>[0-9]+(?:\.[0-9]+)?).*?"
        r"interval_runtime:\s*(?P<runtime>[0-9]+(?:\.[0-9]+)?).*?"
        r"interval_tokens_per_second_per_device:\s*(?P<tps>[0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    step_rows = []
    for line in text.splitlines():
        m = step_re.search(line)
        if not m:
            continue
        step_rows.append({
            "step": int(m.group("step")),
            "alloc": float(m.group("alloc")),
            "reserved": float(m.group("reserved")),
            "runtime": float(m.group("runtime")),
            "tps": float(m.group("tps")),
        })
    if step_rows:
        stable = step_rows[1:] if len(step_rows) > 1 else step_rows
        tail = step_rows[-5:] if len(step_rows) >= 5 else step_rows
        obs.total_steps = len(step_rows)
        obs.avg_step_time_s = sum(r["runtime"] for r in stable) / len(stable)
        obs.last5_avg_step_time_s = sum(r["runtime"] for r in tail) / len(tail)
        obs.avg_tps = sum(r["tps"] for r in stable) / len(stable)
        obs.last5_avg_tps = sum(r["tps"] for r in tail) / len(tail)
        obs.peak_allocated_gb = max(r["alloc"] for r in step_rows)
        obs.peak_reserved_gb = max(r["reserved"] for r in step_rows)
        if obs.status == "UNKNOWN":
            obs.status = "OK"

    extra_patterns = {
        "micro_batch_size": [r'micro_batch_size[\'"]?\s*[:=]\s*(\d+)', r'per_device_train_batch_size\s*[:=]\s*(\d+)'],
        "gradient_accumulation_steps": [r'gradient_accumulation_steps\s*[:=]\s*(\d+)', r'accumulate_steps[\'"]?\s*[:=]\s*(\d+)'],
        "max_seq_len": [r'max_seq_len\s*[:=]\s*(\d+)', r'max_sequence_length\s*[:=]\s*(\d+)', r'seq_length\s*[:=]\s*(\d+)'],
    }
    for field_name, regexes in extra_patterns.items():
        if getattr(obs, field_name) is not None:
            continue
        for regex in regexes:
            m = re.search(regex, text, re.IGNORECASE)
            if m:
                setattr(obs, field_name, int(m.group(1)))
                obs.runtime_context[field_name] = int(m.group(1))
                break

    # Summary metrics.
    patterns = {
        "avg_step_time_s": [
            r"avg[_\s-]*steptime[_\s-]*s\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"avg[_\s-]*step[_\s-]*time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|seconds)?",
        ],
        "last5_avg_step_time_s": [
            r"last5[_\s-]*avg[_\s-]*steptime[_\s-]*s\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"last5[_\s-]*avg[_\s-]*step[_\s-]*time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|seconds)?",
        ],
        "peak_allocated_gb": [
            r"peak[_\s-]*allocated(?:[_\s-]*memory)?(?:[_\s-]*gb)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(gb|g|mb|m|kb|k|tb|t|bytes|b)?",
        ],
        "peak_reserved_gb": [
            r"peak[_\s-]*reserved(?:[_\s-]*memory)?(?:[_\s-]*gb)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(gb|g|mb|m|kb|k|tb|t|bytes|b)?",
        ],
        "avg_tps": [r"avg[_\s-]*tps\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "last5_avg_tps": [r"last5[_\s-]*avg[_\s-]*tps\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "total_steps": [r"total[_\s-]*steps\s*[:=]\s*(\d+)"],
        "rounds": [r"rounds\s*[:=]\s*(\d+)"],
    }
    for field_name, regexes in patterns.items():
        for regex in regexes:
            m = re.search(regex, text, re.IGNORECASE)
            if not m:
                continue
            value = _to_float(m.group(1))
            if value is None:
                continue
            if field_name.endswith("_s"):
                unit = m.group(2) if len(m.groups()) >= 2 else None
                setattr(obs, field_name, _convert_to_seconds(value, unit))
            elif field_name.endswith("_gb"):
                unit = m.group(2) if len(m.groups()) >= 2 else None
                setattr(obs, field_name, _convert_to_gb(value, unit))
            elif field_name in {"total_steps", "rounds"}:
                setattr(obs, field_name, int(value))
            else:
                setattr(obs, field_name, float(value))
            break

    # Fill commonly used direct fields from context if still missing.
    for field in [
        "pp", "tp", "dp", "ep", "sp", "cp", "sharding", "sharding_degree",
        "recompute_granularity", "recompute_method", "recompute_num_layers",
        "tensorwise_offload_optimizer", "tensorwise_offload_ratio",
        "micro_batch_size", "gradient_accumulation_steps", "max_seq_len",
    ]:
        if getattr(obs, field) is None and field in obs.runtime_context:
            setattr(obs, field, obs.runtime_context[field])

    if obs.recompute_granularity is None:
        obs.recompute_granularity = "none"
    if obs.dp is None and obs.pp and obs.tp:
        # Leave unresolved if not in log; costmodel's training_config/world_size may decide later.
        obs.dp = 1
    if obs.ep is None:
        obs.ep = 1
    if obs.sharding is None:
        obs.sharding = "stage1"
    if obs.sharding_degree is None and obs.dp is not None:
        obs.sharding_degree = int(obs.dp)
    if obs.sp is None:
        obs.sp = False
    if obs.tensorwise_offload_optimizer is None:
        obs.tensorwise_offload_optimizer = False
    if obs.tensorwise_offload_ratio is None:
        obs.tensorwise_offload_ratio = 0.95
    return obs


def parse_runtime_log_file(path: str) -> RuntimeObservation:
    path_obj = Path(path)
    return parse_runtime_log_text(path_obj.read_text(errors="ignore"), source_name=str(path_obj))


def parse_runtime_logs(paths: Sequence[str]) -> List[RuntimeObservation]:
    return [parse_runtime_log_file(path) for path in paths]

def _normalize_status_token(value: Any) -> str:
    s = str(value or '').strip().upper()
    if not s:
        return 'UNKNOWN'
    if s.startswith('OK'):
        return 'OK'
    if s.startswith('OOM'):
        return 'OOM'
    if s.startswith('FAIL'):
        return 'FAIL'
    if s.startswith('INCOMPLETE'):
        return 'UNKNOWN'
    return s


def _infer_recompute_method_from_token(token: Any) -> Optional[str]:
    s = str(token or '').strip().lower()
    if s in {'uni', 'uniform', 'uniform1'}:
        return 'uniform'
    if s in {'blk', 'block'}:
        return 'block'
    if s in {'norec', 'none', 'no', 'null', ''}:
        return 'none'
    return s or None


def _infer_offload_bool_from_token(token: Any) -> Optional[bool]:
    s = str(token or '').strip().lower()
    if s in {'off', 'true', '1', 'yes', 'y', 'on'}:
        return True
    if s in {'nooff', 'false', '0', 'no', 'n', 'offload_off'}:
        return False
    b = _to_bool(token)
    return b


def _load_yaml_runtime_defaults(yaml_path: Optional[str]) -> Dict[str, Any]:
    if not yaml_path:
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception:
        return {}


def _normalize_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in row.items():
        norm_key = str(key).strip() if key is not None else ''
        if not norm_key:
            continue
        if isinstance(value, str):
            norm_value = value.strip()
            clean[norm_key] = None if norm_value == '' else norm_value
        else:
            clean[norm_key] = value
    return clean


def _infer_dense_stage1_sharding_degree(pp: Optional[int], tp: Optional[int], cp: Optional[int], ep: Optional[int], *, total_gpus: Optional[int]) -> Optional[int]:
    if not total_gpus or total_gpus <= 0:
        return None
    pp_v = max(1, int(pp or 1))
    tp_v = max(1, int(tp or 1))
    cp_v = max(1, int(cp or 1))
    ep_v = max(1, int(ep or 1))
    # For dense models EP=1. When EP>1 (MoE), avoid inferring from total_gpus here because
    # EP grouping semantics differ from plain world-size factorization.
    if ep_v != 1:
        return None
    denom = pp_v * tp_v * cp_v
    if denom <= 0 or total_gpus % denom != 0:
        return None
    inferred = total_gpus // denom
    return inferred if inferred >= 1 else None




def _candidate_yaml_paths_for_config(yaml_root_obj: Path, config_name: str) -> List[Path]:
    candidates: List[Path] = []
    base = str(config_name or '').strip()
    if not base:
        return candidates
    seen = set()
    def add(name: str):
        path = yaml_root_obj / f'{name}.yaml'
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    add(base)
    # Dense CSVs may use sd{n} while YAML filenames use dp{n}.
    if 'sd' in base:
        add(base.replace('sd', 'dp'))
    if 'dp' in base:
        add(base.replace('dp', 'sd'))
    return candidates


def parse_runtime_metrics_csv(path: str, *, yaml_root: Optional[str] = None, default_model_name: Optional[str] = None,
                              default_num_nodes: Optional[int] = None, default_gpus_per_node: Optional[int] = None) -> List[RuntimeObservation]:
    path_obj = Path(path)
    yaml_root_obj = Path(yaml_root) if yaml_root else None
    total_gpus = None
    if default_num_nodes and default_gpus_per_node:
        total_gpus = int(default_num_nodes) * int(default_gpus_per_node)
    observations: List[RuntimeObservation] = []
    with open(path_obj, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [str(name).strip() if name is not None else '' for name in reader.fieldnames]
        for raw_row in reader:
            row = _normalize_csv_row(raw_row)
            config_name = str(row.get('config') or '').strip()
            yaml_path = None
            yaml_cfg: Dict[str, Any] = {}
            if yaml_root_obj and config_name:
                for candidate in _candidate_yaml_paths_for_config(yaml_root_obj, config_name):
                    if candidate.exists():
                        yaml_path = str(candidate)
                        yaml_cfg = _load_yaml_runtime_defaults(yaml_path)
                        break

            obs = RuntimeObservation()
            obs.source = f'{path_obj}::{config_name}' if config_name else str(path_obj)
            obs.config_name = config_name
            obs.status = _normalize_status_token(row.get('status'))
            obs.pp = _to_int(row.get('pp')) or _to_int(yaml_cfg.get('pipeline_model_parallel_size')) or 1
            obs.tp = _to_int(row.get('tp')) or _to_int(yaml_cfg.get('tensor_model_parallel_size')) or 1
            obs.ep = _to_int(row.get('ep')) or _to_int(yaml_cfg.get('expert_model_parallel_size')) or 1
            obs.sp = bool(yaml_cfg.get('sequence_parallel', False))
            obs.cp = _to_int(yaml_cfg.get('context_parallel_size')) or 1
            obs.sharding = str(yaml_cfg.get('sharding', 'stage1') or 'stage1')
            explicit_sharding_degree = _to_int(yaml_cfg.get('sharding_degree')) or _to_int(row.get('dp'))
            inferred_dense_sharding_degree = None
            if explicit_sharding_degree is None and str(obs.sharding).lower() == 'stage1':
                inferred_dense_sharding_degree = _infer_dense_stage1_sharding_degree(obs.pp, obs.tp, obs.cp, obs.ep, total_gpus=total_gpus)
            obs.sharding_degree = explicit_sharding_degree or inferred_dense_sharding_degree or 1
            # For these Paddle runs, stage1 sharding usually absorbs the non-PP/TP remainder.
            # Keep effective DP at 1 unless it is explicitly specified in YAML.
            obs.dp = _to_int(yaml_cfg.get('data_parallel_degree')) or 1
            obs.recompute_granularity = str(yaml_cfg.get('recompute_granularity', 'none') or 'none')
            obs.recompute_method = _infer_recompute_method_from_token(yaml_cfg.get('recompute_method') or row.get('recompute'))
            obs.recompute_num_layers = _to_int(yaml_cfg.get('recompute_num_layers'))
            obs.tensorwise_offload_optimizer = _infer_offload_bool_from_token(yaml_cfg.get('tensorwise_offload_optimizer') if 'tensorwise_offload_optimizer' in yaml_cfg else row.get('offload'))
            obs.tensorwise_offload_ratio = _to_float(yaml_cfg.get('tensorwise_offload_ratio'))
            obs.micro_batch_size = _to_int(yaml_cfg.get('per_device_train_batch_size')) or 1
            obs.gradient_accumulation_steps = _to_int(yaml_cfg.get('gradient_accumulation_steps')) or 1
            obs.max_seq_len = _to_int(yaml_cfg.get('max_seq_len')) or 4096
            obs.avg_step_time_s = _to_float(row.get('avg_steptime_s'))
            obs.last5_avg_step_time_s = _to_float(row.get('last5_avg_steptime_s'))
            obs.peak_allocated_gb = _to_float(row.get('peak_allocated_GB'))
            obs.peak_reserved_gb = _to_float(row.get('peak_reserved_GB'))
            obs.avg_tps = _to_float(row.get('avg_tps'))
            obs.last5_avg_tps = _to_float(row.get('last5_avg_tps'))
            obs.total_steps = _to_int(row.get('total_steps'))
            obs.rounds = _to_int(row.get('rounds'))
            obs.runtime_context = {
                'pp': int(obs.pp or 1),
                'tp': int(obs.tp or 1),
                'dp': int(obs.dp or 1),
                'ep': int(obs.ep or 1),
                'sp': bool(obs.sp),
                'cp': int(obs.cp or 1),
                'sharding': str(obs.sharding or 'stage1'),
                'sharding_degree': int(obs.sharding_degree or 1),
                'micro_batch_size': int(obs.micro_batch_size or 1),
                'gradient_accumulation_steps': int(obs.gradient_accumulation_steps or 1),
                'max_seq_len': int(obs.max_seq_len or 4096),
                'recompute_granularity': obs.recompute_granularity,
                'recompute_method': obs.recompute_method,
                'recompute_num_layers': obs.recompute_num_layers,
                'tensorwise_offload_optimizer': bool(obs.tensorwise_offload_optimizer),
                'tensorwise_offload_ratio': obs.tensorwise_offload_ratio,
                'split_param': _to_bool(yaml_cfg.get('split_param')),
                'sd_release_grads': _to_bool(yaml_cfg.get('sd_release_grads')),
                'stage1_overlap': _to_bool(yaml_cfg.get('stage1_overlap')),
                'overlap_p2p_comm': _to_bool(yaml_cfg.get('overlap_p2p_comm')),
                'variable_seq_lengths': _to_bool(yaml_cfg.get('variable_seq_lengths')),
                'best_unbalanced_scheduler': _to_bool(yaml_cfg.get('best_unbalanced_scheduler')),
                'enable_dynamic_shape': _to_bool(yaml_cfg.get('enable_dynamic_shape')),
                'clear_every_step_cache': _to_bool(yaml_cfg.get('clear_every_step_cache')),
                'apply_rope_fusion': _to_bool(yaml_cfg.get('apply_rope_fusion')),
                'use_qk_norm': _to_bool(yaml_cfg.get('use_qk_norm')),
                'attn_implementation': yaml_cfg.get('attn_implementation', yaml_cfg.get('_attn_implementation')),
                'yaml_path': yaml_path,
                'model_name_or_path': yaml_cfg.get('model_name_or_path', default_model_name),
            }
            observations.append(obs)
    return observations


def parse_runtime_metric_sources(log_paths: Optional[Sequence[str]] = None, *, csv_specs: Optional[Sequence[Tuple[str, Optional[str]]]] = None) -> List[RuntimeObservation]:
    observations: List[RuntimeObservation] = []
    for path in (log_paths or []):
        observations.append(parse_runtime_log_file(path))
    for csv_path, yaml_root in (csv_specs or []):
        observations.extend(parse_runtime_metrics_csv(csv_path, yaml_root=yaml_root))
    return observations



# ---------------------------------------------------------------------------
# Context snapshot helpers
# ---------------------------------------------------------------------------

def _clean_context_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if value is None:
        return None
    return str(value)


def build_runtime_context_snapshot(
    *,
    parallel: Any,
    runtime_training_config: Optional[Any],
    recompute_granularity: Optional[str] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    tensorwise_offload_optimizer: Optional[bool] = None,
    tensorwise_offload_ratio: Optional[float] = None,
    split_param: Optional[bool] = None,
    sd_release_grads: Optional[bool] = None,
    micro_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "pp": int(getattr(parallel, "pp", 1) or 1),
        "tp": int(getattr(parallel, "tp", 1) or 1),
        "dp": int(getattr(parallel, "dp", 1) or 1),
        "ep": int(getattr(parallel, "ep", 1) or 1),
        "sp": bool(getattr(parallel, "sp", False)),
        "cp": int(getattr(parallel, "cp", 1) or 1),
        "sharding": str(getattr(parallel, "sharding", "stage1") or "stage1"),
        "sharding_degree": int(getattr(parallel, "sharding_degree", -1) or -1),
        "micro_batch_size": int(micro_batch_size if micro_batch_size is not None else getattr(runtime_training_config, "micro_batch_size", 1) or 1),
        "gradient_accumulation_steps": int(gradient_accumulation_steps if gradient_accumulation_steps is not None else getattr(runtime_training_config, "gradient_accumulation_steps", 1) or 1),
        "max_seq_len": int(max_seq_len if max_seq_len is not None else getattr(runtime_training_config, "sequence_length", 0) or 0),
        "recompute_granularity": str(recompute_granularity or getattr(runtime_training_config, "recompute_granularity", "none") or "none"),
        "recompute_method": str(recompute_method or getattr(runtime_training_config, "recompute_method", "uniform") or "uniform"),
        "recompute_num_layers": int(recompute_num_layers if recompute_num_layers is not None else getattr(runtime_training_config, "recompute_num_layers", 1) or 1),
        "tensorwise_offload_optimizer": bool(
            tensorwise_offload_optimizer
            if tensorwise_offload_optimizer is not None
            else getattr(runtime_training_config, "tensorwise_offload_optimizer", False)
        ),
        "tensorwise_offload_ratio": float(
            tensorwise_offload_ratio
            if tensorwise_offload_ratio is not None
            else getattr(runtime_training_config, "tensorwise_offload_ratio", 0.95)
        ),
        "split_param": bool(split_param if split_param is not None else True),
        "sd_release_grads": bool(
            sd_release_grads if sd_release_grads is not None else getattr(runtime_training_config, "sd_release_grads", False)
        ),
    }
    if runtime_training_config is not None:
        for spec in _SUPPORTED_FIELD_SPECS:
            name = spec["name"]
            if name in snapshot:
                continue
            if hasattr(runtime_training_config, name):
                snapshot[name] = _clean_context_value(getattr(runtime_training_config, name))
    return {k: _clean_context_value(v) for k, v in snapshot.items() if v is not None}


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def _ridge_fit(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    feature_dim = int(X.shape[1])
    eye = np.eye(feature_dim, dtype=float)
    # Keep base columns slightly freer than interaction columns.
    if feature_dim > 0:
        eye[0, 0] = 0.25
    return np.linalg.solve(X.T @ X + ridge_lambda * eye, X.T @ y)


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1e-9)


def _bool_feature(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _context_numeric_features(context: Dict[str, Any]) -> Dict[str, float]:
    tp = max(1, int(context.get("tp", 1) or 1))
    pp = max(1, int(context.get("pp", 1) or 1))
    dp = max(1, int(context.get("dp", 1) or 1))
    ep = max(1, int(context.get("ep", 1) or 1))
    rec_layers = max(0, int(context.get("recompute_num_layers", 0) or 0))
    out: Dict[str, float] = {
        "tp_log2": math.log2(tp),
        "pp_minus1": float(pp - 1),
        "dp_log2": math.log2(dp),
        "ep_log2": math.log2(ep),
        "offload_ratio": float(context.get("tensorwise_offload_ratio", 0.95) or 0.95),
        "recompute_num_layers": float(rec_layers),
        "empty_layers_head": float(context.get("num_empty_layers_add_in_head", 0) or 0),
        "empty_layers_tail": float(context.get("num_empty_layers_add_in_tail", 0) or 0),
        "sp": _bool_feature(context.get("sp")),
        "offload": _bool_feature(context.get("tensorwise_offload_optimizer")),
        "split_param": _bool_feature(context.get("split_param")),
        "sd_release_grads": _bool_feature(context.get("sd_release_grads")),
        "overlap_p2p_comm": _bool_feature(context.get("overlap_p2p_comm")),
        "use_batch_p2p_comm": _bool_feature(context.get("use_batch_p2p_comm")),
        "p2p_cache_shape": _bool_feature(context.get("p2p_cache_shape")),
        "stage1_overlap": _bool_feature(context.get("stage1_overlap")),
        "enable_sharding_comm_overlap": _bool_feature(context.get("enable_sharding_comm_overlap")),
        "variable_seq_lengths": _bool_feature(context.get("variable_seq_lengths")),
        "enable_dynamic_shape": _bool_feature(context.get("enable_dynamic_shape")),
        "clear_every_step_cache": _bool_feature(context.get("clear_every_step_cache")),
        "best_unbalanced_scheduler": _bool_feature(context.get("best_unbalanced_scheduler")),
        "apply_rope_fusion": _bool_feature(context.get("apply_rope_fusion")),
        "full_recompute": _bool_feature(str(context.get("recompute_granularity", "none")).lower() == "full"),
        "selective_recompute": _bool_feature(str(context.get("recompute_granularity", "none")).lower() == "selective"),
        "uniform_recompute": _bool_feature(str(context.get("recompute_method", "")).lower() == "uniform"),
        "block_recompute": _bool_feature(str(context.get("recompute_method", "")).lower() == "block"),
    }
    for cat_field in ["attn_implementation", "hybrid_parallel_topo_order", "sharding", "moe_token_dispatcher_type"]:
        value = context.get(cat_field)
        if value is not None:
            out[f"cat::{cat_field}::{str(value).lower()}"] = 1.0
    return out


def _discover_active_context_features(contexts: Sequence[Dict[str, Any]]) -> List[str]:
    values: Dict[str, set] = {}
    for ctx in contexts:
        feats = _context_numeric_features(ctx)
        for name, value in feats.items():
            if abs(float(value)) < 1e-12:
                values.setdefault(name, set()).add(0.0)
            else:
                values.setdefault(name, set()).add(float(value))
    active = [name for name, uniq in values.items() if len(uniq) > 1]
    # Keep a small stable subset always, even if constant during one fit, to preserve metadata.
    for name in ["tp_log2", "pp_minus1", "offload", "full_recompute", "selective_recompute", "uniform_recompute", "block_recompute"]:
        if name not in active and name in values:
            active.append(name)
    return sorted(set(active))


def _time_component_exposures(result: Any, total_layers: int) -> Dict[str, float]:
    layers = max(1, int(total_layers or 1))
    return {
        "forward_layer_ms": float(getattr(result, "forward_time_ms", 0.0) or 0.0) / layers,
        "backward_layer_ms": float(getattr(result, "backward_time_ms", 0.0) or 0.0) / layers,
        "recompute_layer_ms": float(getattr(result, "recompute_time_ms", 0.0) or 0.0) / layers,
        "tp_comm_layer_ms": float(getattr(result, "tp_comm_time_ms", 0.0) or 0.0) / layers,
        "ep_comm_layer_ms": float(getattr(result, "ep_comm_time_ms", 0.0) or 0.0) / layers,
        "sp_comm_layer_ms": float(getattr(result, "sp_comm_time_ms", 0.0) or 0.0) / layers,
        "bubble_ms": float(getattr(result, "bubble_time_ms", 0.0) or 0.0),
        "framework_ms": float(getattr(result, "framework_overhead_ms", 0.0) or 0.0),
        "runtime_ms": float(getattr(result, "runtime_overhead_ms", 0.0) or 0.0),
        "pp_comm_ms": float(getattr(result, "pp_comm_time_ms", 0.0) or 0.0),
        "dp_exposed_ms": float(getattr(result, "dp_exposed_comm_time_ms", 0.0) or 0.0),
        "offload_ms": float(getattr(result, "offload_overhead_ms", 0.0) or 0.0),
        "optimizer_ms": float(getattr(result, "optimizer_step_time_ms", 0.0) or 0.0),
        "layers": float(layers),
    }


def _alloc_component_exposures(result: Any) -> Dict[str, float]:
    bd = getattr(result, "memory_breakdown", None)
    if bd is None:
        return {}
    return {
        "param_gb": float(getattr(bd, "parameter_memory_gb", 0.0) or 0.0),
        "grad_gb": float(getattr(bd, "gradient_memory_gb", 0.0) or 0.0),
        "optimizer_gb": float(getattr(bd, "optimizer_memory_gb", 0.0) or 0.0),
        "activation_gb": float(getattr(bd, "activation_memory_gb", 0.0) or 0.0),
        "master_grad_gb": float(getattr(bd, "master_grad_memory_gb", 0.0) or 0.0),
        "tensor_fusion_gb": float(getattr(bd, "tensor_fusion_buffer_gb", 0.0) or 0.0),
        "comm_buffer_gb": float(getattr(bd, "communication_buffer_gb", 0.0) or 0.0),
        "temp_buffer_gb": float(getattr(bd, "temporary_buffer_gb", 0.0) or 0.0),
        "optimizer_ws_gb": float(getattr(bd, "optimizer_update_workspace_gb", 0.0) or 0.0),
    }


def _reserved_extra_component_exposures(result: Any) -> Dict[str, float]:
    bd = getattr(result, "memory_breakdown", None)
    if bd is None:
        return {}
    return {
        "temp_buffer_gb": float(getattr(bd, "temporary_buffer_gb", 0.0) or 0.0),
        "activation_pool_gb": float(getattr(bd, "activation_buffer_pool_gb", 0.0) or 0.0),
        "comm_runtime_pool_gb": float(getattr(bd, "communication_runtime_pool_gb", 0.0) or 0.0),
        "overlap_runtime_pool_gb": float(getattr(bd, "overlap_runtime_pool_gb", 0.0) or 0.0),
        "comm_fragmentation_gb": float(getattr(bd, "communication_fragmentation_gb", 0.0) or 0.0),
        "reserved_candidate_optimizer_gb": float(getattr(bd, "reserved_candidate_optimizer_gb", 0.0) or 0.0),
    }


def _build_basis(
    exposures: Dict[str, float],
    context: Dict[str, Any],
    *,
    active_context_features: Sequence[str],
    include_constant: bool = False,
) -> Tuple[List[str], List[float], Dict[str, float]]:
    ctx_features = _context_numeric_features(context)
    feature_names: List[str] = []
    values: List[float] = []
    per_component_delta_basis: Dict[str, float] = {}
    if include_constant:
        feature_names.append("const")
        values.append(1.0)
    for comp_name, comp_value in exposures.items():
        if comp_name == "layers":
            continue
        comp_val = float(comp_value)
        if abs(comp_val) < 1e-12:
            continue
        base_name = f"{comp_name}|base"
        feature_names.append(base_name)
        values.append(comp_val)
        per_component_delta_basis[base_name] = comp_val
        for ctx_name in active_context_features:
            ctx_val = float(ctx_features.get(ctx_name, 0.0) or 0.0)
            if abs(ctx_val) < 1e-12:
                continue
            name = f"{comp_name}|{ctx_name}"
            feature_names.append(name)
            values.append(comp_val * ctx_val)
            per_component_delta_basis[name] = comp_val * ctx_val
    return feature_names, values, per_component_delta_basis


def _align_matrix(rows: Sequence[Tuple[List[str], List[float]]]) -> Tuple[List[str], np.ndarray]:
    name_order: List[str] = []
    seen = set()
    for names, _ in rows:
        for name in names:
            if name not in seen:
                seen.add(name)
                name_order.append(name)
    X = np.zeros((len(rows), len(name_order)), dtype=float)
    name_to_idx = {name: idx for idx, name in enumerate(name_order)}
    for row_idx, (names, values) in enumerate(rows):
        for name, value in zip(names, values):
            X[row_idx, name_to_idx[name]] = float(value)
    return name_order, X




def _fit_compact_residual(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Fit y ≈ X @ beta using ridge regression with column normalization."""
    if X.size == 0:
        return np.zeros((0,), dtype=float)
    n, p = X.shape
    col_scales = np.sqrt(np.sum(X ** 2, axis=0) + 1e-12)
    col_scales = np.where(col_scales < 1e-12, 1.0, col_scales)
    X_norm = X / col_scales
    beta_norm = np.linalg.solve(
        X_norm.T @ X_norm + ridge_lambda * np.eye(p, dtype=float),
        X_norm.T @ y,
    )
    return beta_norm / col_scales


def _compact_time_feature_names() -> List[str]:
    """Config-based features for error-ratio regression.

    Target: actual_step / predicted_step ≈ X @ beta.
    """
    return [
        "log2_pp", "log2_pp_sq", "log2_tp", "log2_tp_sq", "offload",
        "recompute_none", "recompute_block",
        "log2_pp_x_offload", "log2_tp_x_offload",
        "log2_pp_x_log2_tp",
        "rec_none_x_log2_tp", "rec_block_x_log2_tp",
        "rec_none_x_log2_pp", "rec_block_x_log2_pp",
        "rec_none_x_offload", "rec_block_x_offload",
        "log2_mbs", "log2_seq", "log2_ep",
        "log2_mbs_x_offload", "log2_seq_x_offload", "log2_ep_x_offload",
        "log2_mbs_x_log2_pp", "log2_seq_x_log2_pp",
        "bias",
    ]


# Legacy mapping kept for reference; not used in ratio-based apply.
_TIME_FEATURE_TO_RESULT_FIELD: Dict[str, str] = {}


def _compact_time_feature_values(result: Any, context: Dict[str, Any]) -> np.ndarray:
    pp = max(1, int(context.get("pp", 1) or 1))
    tp = max(1, int(context.get("tp", 1) or 1))
    ep = max(1, int(context.get("ep", 1) or 1))
    mbs = max(1, int(context.get("micro_batch_size", 1) or 1))
    seq = max(1, int(context.get("max_seq_len", 4096) or 4096))
    off = 1.0 if bool(context.get("tensorwise_offload_optimizer", False)) else 0.0
    rec_g = str(context.get("recompute_granularity", "") or "").lower()
    rec_m = str(context.get("recompute_method", "") or "").lower()
    rec_none = 1.0 if (rec_g in ("", "none", "null") or rec_m in ("", "none", "null")) else 0.0
    rec_block = 1.0 if rec_m == "block" else 0.0

    lpp = math.log2(pp) if pp > 1 else 0.0
    ltp = math.log2(tp) if tp > 1 else 0.0
    lep = math.log2(ep) if ep > 1 else 0.0
    lmbs = math.log2(mbs) if mbs > 1 else 0.0
    lseq = math.log2(seq) - 12.0  # normalize: log2(4096)=12, so seq=4096 -> 0, seq=8192 -> 1

    return np.asarray([
        lpp, lpp * lpp, ltp, ltp * ltp, off,
        rec_none, rec_block,
        lpp * off, ltp * off,
        lpp * ltp,
        rec_none * ltp, rec_block * ltp,
        rec_none * lpp, rec_block * lpp,
        rec_none * off, rec_block * off,
        lmbs, lseq, lep,
        lmbs * off, lseq * off, lep * off,
        lmbs * lpp, lseq * lpp,
        1.0,
    ], dtype=float)


def _compact_alloc_feature_names() -> List[str]:
    """Config-based features for memory error-ratio regression.

    Simplified to 8 core features to avoid overfitting with small observation
    counts (target: features/observations <= 0.2).
    """
    return [
        "log2_pp", "log2_tp", "log2_ep",
        "offload",
        "recompute_none", "recompute_block",
        "log2_seq",
        "bias",
    ]


# Legacy mapping kept for reference; not used in ratio-based apply.
_ALLOC_FEATURE_TO_BREAKDOWN_FIELD: Dict[str, str] = {}


def _compact_alloc_feature_values(result: Any, context: Dict[str, Any]) -> np.ndarray:
    pp = max(1, int(context.get("pp", 1) or 1))
    tp = max(1, int(context.get("tp", 1) or 1))
    ep = max(1, int(context.get("ep", 1) or 1))
    mbs = max(1, int(context.get("micro_batch_size", 1) or 1))
    seq = max(1, int(context.get("max_seq_len", 4096) or 4096))
    off = 1.0 if bool(context.get("tensorwise_offload_optimizer", False)) else 0.0
    rec_g = str(context.get("recompute_granularity", "") or "").lower()
    rec_m = str(context.get("recompute_method", "") or "").lower()
    rec_none = 1.0 if (rec_g in ("", "none", "null") or rec_m in ("", "none", "null")) else 0.0
    rec_block = 1.0 if rec_m == "block" else 0.0
    lpp = math.log2(pp) if pp > 1 else 0.0
    ltp = math.log2(tp) if tp > 1 else 0.0
    lep = math.log2(ep) if ep > 1 else 0.0
    lmbs = math.log2(mbs) if mbs > 1 else 0.0
    lseq = math.log2(seq) - 12.0
    return np.asarray([
        lpp, ltp, lep,
        off,
        rec_none, rec_block,
        lseq,
        1.0,
    ], dtype=float)


def _compact_reserved_feature_names() -> List[str]:
    """Config-based features for reserved memory error-ratio regression."""
    return [
        "log2_pp", "log2_tp", "log2_ep",
        "offload",
        "recompute_none", "recompute_block",
        "log2_seq",
        "bias",
    ]


def _compact_reserved_feature_values(result: Any, context: Dict[str, Any]) -> np.ndarray:
    pp = max(1, int(context.get("pp", 1) or 1))
    tp = max(1, int(context.get("tp", 1) or 1))
    ep = max(1, int(context.get("ep", 1) or 1))
    mbs = max(1, int(context.get("micro_batch_size", 1) or 1))
    seq = max(1, int(context.get("max_seq_len", 4096) or 4096))
    off = 1.0 if bool(context.get("tensorwise_offload_optimizer", False)) else 0.0
    rec_g = str(context.get("recompute_granularity", "") or "").lower()
    rec_m = str(context.get("recompute_method", "") or "").lower()
    rec_none = 1.0 if (rec_g in ("", "none", "null") or rec_m in ("", "none", "null")) else 0.0
    rec_block = 1.0 if rec_m == "block" else 0.0
    lpp = math.log2(pp) if pp > 1 else 0.0
    ltp = math.log2(tp) if tp > 1 else 0.0
    lep = math.log2(ep) if ep > 1 else 0.0
    lmbs = math.log2(mbs) if mbs > 1 else 0.0
    lseq = math.log2(seq) - 12.0
    return np.asarray([
        lpp, ltp, lep,
        off,
        rec_none, rec_block,
        lseq,
        1.0,
    ], dtype=float)

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class RuntimeCalibrationStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or str(Path(__file__).with_name("runtime_model_calibrations.json"))
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        path_obj = Path(self.path)
        if not path_obj.exists():
            return {"version": 2, "models": {}}
        try:
            data = json.loads(path_obj.read_text())
            if not isinstance(data, dict):
                raise ValueError("Calibration store must be a JSON object")
            data.setdefault("version", 2)
            data.setdefault("models", {})
            return data
        except Exception as exc:
            logger.warning("Failed to load runtime calibration store %s: %s", self.path, exc)
            return {"version": 2, "models": {}}

    def save(self) -> None:
        Path(self.path).write_text(json.dumps(self.data, indent=2, sort_keys=True))

    def get(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.data.get("models", {}).get(canonicalize_model_name(model_name))

    def upsert(self, model_name: str, entry: Dict[str, Any], persist: bool = True) -> None:
        key = canonicalize_model_name(model_name)
        self.data.setdefault("models", {})[key] = entry
        if persist:
            self.save()


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def _aggregate_observations(observations: Sequence[RuntimeObservation]) -> List[RuntimeObservation]:
    groups: Dict[Tuple[Any, ...], List[RuntimeObservation]] = {}
    for obs in observations:
        key = (obs.config_name or "", obs.pp, obs.tp, obs.dp, obs.ep, obs.sp, obs.cp, obs.sharding, obs.sharding_degree, obs.recompute_granularity, obs.recompute_method, obs.recompute_num_layers, obs.tensorwise_offload_optimizer, obs.micro_batch_size, obs.max_seq_len, obs.runtime_context.get("sd_release_grads"), obs.runtime_context.get("stage1_overlap"), obs.runtime_context.get("overlap_p2p_comm"))
        groups.setdefault(key, []).append(obs)
    out: List[RuntimeObservation] = []
    for items in groups.values():
        base = copy.deepcopy(items[0])
        def _mean(vals):
            vals=[v for v in vals if v is not None]
            return float(sum(vals)/len(vals)) if vals else None
        def _median(vals):
            vals=sorted(v for v in vals if v is not None)
            if not vals: return None
            n=len(vals)
            return float(vals[n//2]) if n%2 else float((vals[n//2-1]+vals[n//2])/2.0)
        base.avg_step_time_s = _mean([x.avg_step_time_s for x in items])
        base.last5_avg_step_time_s = _mean([x.last5_avg_step_time_s for x in items])
        base.avg_tps = _mean([x.avg_tps for x in items])
        base.last5_avg_tps = _mean([x.last5_avg_tps for x in items])
        base.peak_allocated_gb = _median([x.peak_allocated_gb for x in items])
        base.peak_reserved_gb = _median([x.peak_reserved_gb for x in items])
        base.total_steps = max([x.total_steps or 0 for x in items] + [0]) or None
        base.rounds = len(items)
        merged = dict(base.runtime_context)
        for k in list(merged.keys()):
            vals = {json.dumps(x.runtime_context.get(k), sort_keys=True) for x in items}
            if len(vals) > 1:
                merged.pop(k, None)
        base.runtime_context = merged
        out.append(base)
    return out

class RuntimeCalibrationFitter:
    def __init__(self, costmodel: Any, store: RuntimeCalibrationStore):
        if np is None:
            raise RuntimeError(f"numpy is required for runtime calibration: {_NUMPY_IMPORT_ERROR}")
        self.costmodel = costmodel
        self.store = store

    def _build_parallel(self, obs: RuntimeObservation) -> Any:
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

    def _predict_raw(self, obs: RuntimeObservation) -> Tuple[Any, Any, Dict[str, Any]]:
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
        )
        runtime_training_config = getattr(self.costmodel, "training_config", None)
        context = build_runtime_context_snapshot(
            parallel=parallel,
            runtime_training_config=runtime_training_config,
            recompute_granularity=obs.recompute_granularity,
            recompute_method=obs.recompute_method,
            recompute_num_layers=obs.recompute_num_layers,
            tensorwise_offload_optimizer=obs.tensorwise_offload_optimizer,
            tensorwise_offload_ratio=obs.tensorwise_offload_ratio,
            split_param=obs.runtime_context.get("split_param", True),
            sd_release_grads=obs.runtime_context.get("sd_release_grads"),
        )
        # Merge obs.runtime_context for feature extraction (e.g. tensorwise_offload
        # flags); but the signature context used for exact matching is already built
        # above from build_runtime_context_snapshot.
        feature_context = dict(context)
        feature_context.update({k: v for k, v in obs.runtime_context.items() if v is not None})
        return parallel, result, context, feature_context

    def fit_from_observations(
        self,
        model_name: str,
        observations: Sequence[RuntimeObservation],
        persist: bool = True,
        min_ok_observations: int = 4,
    ) -> Dict[str, Any]:
        observations = _aggregate_observations(list(observations))
        ok_obs = [obs for obs in observations if str(obs.status).upper() == "OK" and obs.pp and obs.tp and obs.preferred_step_time_s()]
        oom_obs = [obs for obs in observations if str(obs.status).upper() == "OOM" and obs.pp and obs.tp]
        fail_obs = [obs for obs in observations if str(obs.status).upper() == "FAIL" and obs.pp and obs.tp]
        unknown_obs = [obs for obs in observations if str(obs.status).upper() not in {"OK", "OOM", "FAIL"}]
        if len(ok_obs) < int(min_ok_observations):
            raise ValueError(f"Need at least {min_ok_observations} OK observations, got {len(ok_obs)}")

        total_layers = int(getattr(self.costmodel.model_config, "num_hidden_layers", 1) or 1)
        pred_records: List[Dict[str, Any]] = []
        contexts: List[Dict[str, Any]] = []
        for obs in ok_obs:
            parallel, pred, context, feature_context = self._predict_raw(obs)
            contexts.append(context)
            pred_records.append({"obs": obs, "parallel": parallel, "pred": pred, "context": context, "feature_context": feature_context})
        status_records: List[Dict[str, Any]] = []
        for obs in (ok_obs + oom_obs):
            try:
                parallel, pred, context, feature_context = self._predict_raw(obs)
            except Exception:
                continue
            status_records.append({"obs": obs, "parallel": parallel, "pred": pred, "context": context})

        X_time = []
        X_alloc = []
        X_reserved = []
        y_time = []
        y_alloc = []
        y_reserved = []
        for record in pred_records:
            obs = record["obs"]
            pred = record["pred"]
            fctx = record["feature_context"]
            X_time.append(_compact_time_feature_values(pred, fctx))
            X_alloc.append(_compact_alloc_feature_values(pred, fctx))
            X_reserved.append(_compact_reserved_feature_values(pred, fctx))
            actual_step_ms = float(obs.preferred_step_time_s() or 0.0) * 1000.0
            actual_alloc = float(obs.peak_allocated_gb or 0.0)
            actual_reserved = float(obs.peak_reserved_gb or 0.0)
            predicted_step_ms = float(pred.step_time_ms or 0.0)
            predicted_alloc = float(pred.allocated_memory_gb or 0.0)
            predicted_reserved = float(pred.reserved_memory_gb or 0.0)
            # Error-ratio targets: actual / predicted
            y_time.append(actual_step_ms / predicted_step_ms if predicted_step_ms > 1e-6 else 1.0)
            y_alloc.append(actual_alloc / predicted_alloc if predicted_alloc > 1e-6 else 1.0)
            y_reserved.append(actual_reserved / predicted_reserved if predicted_reserved > 1e-6 else 1.0)

        X_time = np.asarray(X_time, dtype=float)
        X_alloc = np.asarray(X_alloc, dtype=float)
        X_reserved = np.asarray(X_reserved, dtype=float)
        y_time = np.asarray(y_time, dtype=float)
        y_alloc = np.asarray(y_alloc, dtype=float)
        y_reserved = np.asarray(y_reserved, dtype=float)

        gpu_memory_gb = float(getattr(self.costmodel.hardware_config.gpu, "memory_gb", 0.0) or 0.0)
        oom_guard_gb = 0.0
        status_train_summary: Dict[str, Any] = {}
        if gpu_memory_gb > 0.0 and status_records:
            candidate_margins = {0.0}
            for record in status_records:
                obs = record["obs"]
                pred = record["pred"]
                pred_reserved = float(getattr(pred, "reserved_memory_gb", 0.0) or 0.0)
                need = max(0.0, gpu_memory_gb - pred_reserved + 1e-6)
                candidate_margins.add(round(need, 6))
            best = None
            for margin in sorted(candidate_margins):
                tp = fp = tn = fn = 0
                for record in status_records:
                    obs = record["obs"]
                    pred = record["pred"]
                    pred_oom = bool((float(getattr(pred, "reserved_memory_gb", 0.0) or 0.0) + margin) > gpu_memory_gb)
                    actual_oom = str(getattr(obs, "status", "")).upper() == "OOM"
                    if pred_oom and actual_oom:
                        tp += 1
                    elif pred_oom and not actual_oom:
                        fp += 1
                    elif (not pred_oom) and actual_oom:
                        fn += 1
                    else:
                        tn += 1
                tpr = tp / max(tp + fn, 1)
                tnr = tn / max(tn + fp, 1)
                score = 0.5 * (tpr + tnr)
                acc = (tp + tn) / max(tp + tn + fp + fn, 1)
                candidate = (score, acc, -margin)
                if best is None or candidate > best[0]:
                    best = (candidate, margin, {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "balanced_accuracy": score, "accuracy": acc})
            if best is not None:
                oom_guard_gb = float(best[1])
                status_train_summary = dict(best[2])

        exact_ok_signatures: Dict[str, Any] = {}
        exact_oom_signatures: Dict[str, Any] = {}
        exact_fail_signatures: Dict[str, Any] = {}
        coarse_ok_signatures: Dict[str, Any] = {}
        coarse_oom_signatures: Dict[str, Any] = {}
        coarse_fail_signatures: Dict[str, Any] = {}
        minimal_ok_signatures: Dict[str, Any] = {}
        minimal_oom_signatures: Dict[str, Any] = {}
        minimal_fail_signatures: Dict[str, Any] = {}
        for record in pred_records:
            obs = record["obs"]
            context = record["context"]
            sig = _signature_key_from_context(context)
            payload = {
                "step_time_ms": float(obs.preferred_step_time_s() or 0.0) * 1000.0,
                "allocated_memory_gb": float(obs.peak_allocated_gb or 0.0),
                "reserved_memory_gb": float(obs.peak_reserved_gb or 0.0),
                "avg_tps": float(obs.last5_avg_tps if obs.last5_avg_tps is not None else (obs.avg_tps or 0.0)),
                "status": "OK",
                "signature_fields": _signature_mapping_from_context(context),
                "config_name": str(obs.config_name or ""),
                "source": str(obs.source or obs.config_name or ""),
            }
            exact_ok_signatures[sig] = payload
            coarse_ok_signatures[_coarse_signature_key_from_context(context)] = payload
            minimal_ok_signatures[_minimal_signature_key_from_context(context)] = payload
        for obs in oom_obs:
            try:
                _, pred, context, _ = self._predict_raw(obs)
            except Exception:
                context = build_runtime_context_snapshot(
                    parallel=type('P', (), {
                        'pp': int(obs.pp or 1), 'tp': int(obs.tp or 1), 'dp': int(obs.dp or 1), 'ep': int(obs.ep or 1),
                        'sp': bool(obs.sp or False), 'cp': int(obs.cp or 1), 'sharding': str(obs.sharding or 'stage1'),
                        'sharding_degree': int(obs.sharding_degree or (obs.dp or 1)),
                    })(),
                    runtime_training_config=getattr(self.costmodel, 'training_config', None),
                    recompute_granularity=obs.recompute_granularity,
                    recompute_method=obs.recompute_method,
                    recompute_num_layers=obs.recompute_num_layers,
                    tensorwise_offload_optimizer=obs.tensorwise_offload_optimizer,
                    tensorwise_offload_ratio=obs.tensorwise_offload_ratio,
                    split_param=obs.runtime_context.get('split_param', True),
                    sd_release_grads=obs.runtime_context.get('sd_release_grads'),
                )
            sig = _signature_key_from_context(context)
            payload = {
                "status": "OOM",
                "signature_fields": _signature_mapping_from_context(context),
                "config_name": str(obs.config_name or ""),
                "source": str(obs.source or obs.config_name or ""),
            }
            exact_oom_signatures[sig] = payload
            coarse_oom_signatures[_coarse_signature_key_from_context(context)] = payload
            minimal_oom_signatures[_minimal_signature_key_from_context(context)] = payload
        for obs in fail_obs:
            try:
                _, pred, context, _ = self._predict_raw(obs)
            except Exception:
                context = build_runtime_context_snapshot(
                    parallel=type('P', (), {
                        'pp': int(obs.pp or 1), 'tp': int(obs.tp or 1), 'dp': int(obs.dp or 1), 'ep': int(obs.ep or 1),
                        'sp': bool(obs.sp or False), 'cp': int(obs.cp or 1), 'sharding': str(obs.sharding or 'stage1'),
                        'sharding_degree': int(obs.sharding_degree or (obs.dp or 1)),
                    })(),
                    runtime_training_config=getattr(self.costmodel, 'training_config', None),
                    recompute_granularity=obs.recompute_granularity,
                    recompute_method=obs.recompute_method,
                    recompute_num_layers=obs.recompute_num_layers,
                    tensorwise_offload_optimizer=obs.tensorwise_offload_optimizer,
                    tensorwise_offload_ratio=obs.tensorwise_offload_ratio,
                    split_param=obs.runtime_context.get('split_param', True),
                    sd_release_grads=obs.runtime_context.get('sd_release_grads'),
                )
            sig = _signature_key_from_context(context)
            payload = {
                "status": "FAIL",
                "signature_fields": _signature_mapping_from_context(context),
                "config_name": str(obs.config_name or ""),
                "source": str(obs.source or obs.config_name or ""),
            }
            exact_fail_signatures[sig] = payload
            coarse_fail_signatures[_coarse_signature_key_from_context(context)] = payload
            minimal_fail_signatures[_minimal_signature_key_from_context(context)] = payload

        time_coefs = _fit_compact_residual(X_time, y_time, 1e-8)
        alloc_coefs = _fit_compact_residual(X_alloc, y_alloc, 1.0)
        reserved_coefs = _fit_compact_residual(X_reserved, y_reserved, 1.0)

        # Compute interpretable ratio model summary.
        time_feature_names = _compact_time_feature_names()
        component_scale_factors: Dict[str, Any] = {}
        if X_time.shape[0] > 0 and time_coefs.size == X_time.shape[1]:
            mean_features = np.mean(X_time, axis=0)
            mean_ratio = float(np.mean(y_time))
            for i, fname in enumerate(time_feature_names):
                coef = float(time_coefs[i])
                component_scale_factors[fname] = {
                    "coefficient": round(coef, 6),
                    "mean_feature_value": round(float(mean_features[i]), 4),
                    "mean_contribution_to_ratio": round(coef * float(mean_features[i]), 4),
                }
            component_scale_factors["_mean_actual_ratio"] = round(mean_ratio, 4)

        entry = {
            "schema_version": 7,
            "fit_kind": "config_ratio_regression_v1",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model_name": canonicalize_model_name(model_name),
            "supported_runtime_fields": get_runtime_calibration_supported_fields(),
            "exact_signature_model": {
                "signature_field_names": list(_SUPPORTED_FIELD_NAMES),
                "ok_signatures": exact_ok_signatures,
                "oom_signatures": exact_oom_signatures,
                "fail_signatures": exact_fail_signatures,
                "ok_signatures_coarse": coarse_ok_signatures,
                "oom_signatures_coarse": coarse_oom_signatures,
                "fail_signatures_coarse": coarse_fail_signatures,
                "ok_signatures_minimal": minimal_ok_signatures,
                "oom_signatures_minimal": minimal_oom_signatures,
                "fail_signatures_minimal": minimal_fail_signatures,
            },
            "time_model": {
                "feature_names": time_feature_names,
                "coefficients": [float(v) for v in time_coefs],
                "component_scale_factors": component_scale_factors,
            },
            "allocated_model": {
                "feature_names": _compact_alloc_feature_names(),
                "coefficients": [float(v) for v in alloc_coefs],
            },
            "reserved_extra_model": {
                "feature_names": _compact_reserved_feature_names(),
                "coefficients": [float(v) for v in reserved_coefs],
            },
            "status_model": {
                "oom_reserved_guard_gb": float(oom_guard_gb),
                "num_oom_observations": len(oom_obs),
                "num_fail_observations": len(fail_obs),
                "num_unknown_observations": len(unknown_obs),
                "seen_fail_signatures": sorted({obs.config_name for obs in fail_obs if obs.config_name}),
                "train_summary": status_train_summary,
            },
            "fit_summary": {
                "num_ok_observations": len(ok_obs),
                "num_oom_observations": len(oom_obs),
                "num_fail_observations": len(fail_obs),
                "num_unknown_observations": len(unknown_obs),
                "num_total_observations": len(observations),
                "uses_all_available_observations": True,
                "ok_sources": [obs.source or obs.config_name for obs in ok_obs],
                "oom_sources": [obs.source or obs.config_name for obs in oom_obs],
                "fail_sources": [obs.source or obs.config_name for obs in fail_obs],
            },
        }

        # Train error summary + per-target safety gating.
        base_step_errs = []
        base_alloc_errs = []
        base_reserved_errs = []
        step_errs = []
        alloc_errs = []
        reserved_errs = []
        for record in pred_records:
            obs = record["obs"]
            base_pred = record["pred"]
            pred = copy.deepcopy(base_pred)
            apply_runtime_calibration_to_result(
                result=pred,
                calibration_entry=entry,
                parallel=record["parallel"],
                runtime_context=record["context"],
                gpu_memory_gb=float(getattr(self.costmodel.hardware_config.gpu, "memory_gb", 0.0) or 0.0),
                total_layers=int(getattr(self.costmodel.model_config, "num_hidden_layers", 1) or 1),
            )
            actual_step_ms = float(obs.preferred_step_time_s() or 0.0) * 1000.0
            actual_alloc = float(obs.peak_allocated_gb or 0.0)
            actual_reserved = float(obs.peak_reserved_gb or 0.0)
            base_step_errs.append(abs(float(base_pred.step_time_ms) - actual_step_ms) / max(actual_step_ms, 1e-6))
            base_alloc_errs.append(abs(float(base_pred.allocated_memory_gb) - actual_alloc) / max(actual_alloc, 1e-6))
            base_reserved_errs.append(abs(float(base_pred.reserved_memory_gb) - actual_reserved) / max(actual_reserved, 1e-6))
            step_errs.append(abs(float(pred.step_time_ms) - actual_step_ms) / max(actual_step_ms, 1e-6))
            alloc_errs.append(abs(float(pred.allocated_memory_gb) - actual_alloc) / max(actual_alloc, 1e-6))
            reserved_errs.append(abs(float(pred.reserved_memory_gb) - actual_reserved) / max(actual_reserved, 1e-6))

        base_step = float(np.mean(base_step_errs)) if base_step_errs else None
        base_alloc = float(np.mean(base_alloc_errs)) if base_alloc_errs else None
        base_reserved = float(np.mean(base_reserved_errs)) if base_reserved_errs else None
        cal_step = float(np.mean(step_errs)) if step_errs else None
        cal_alloc = float(np.mean(alloc_errs)) if alloc_errs else None
        cal_reserved = float(np.mean(reserved_errs)) if reserved_errs else None

        entry["fit_summary"].update({
            "base_train_mape_step": base_step,
            "base_train_mape_allocated": base_alloc,
            "base_train_mape_reserved": base_reserved,
            "train_mape_step": cal_step,
            "train_mape_allocated": cal_alloc,
            "train_mape_reserved": cal_reserved,
        })

        self.store.upsert(model_name, entry, persist=persist)
        return entry

    def fit_from_log_files(self, model_name: str, log_paths: Sequence[str], persist: bool = True, min_ok_observations: int = 4) -> Dict[str, Any]:
        observations = [parse_runtime_log_file(path) for path in log_paths]
        return self.fit_from_observations(model_name, observations, persist=persist, min_ok_observations=min_ok_observations)

    def fit_from_log_texts(
        self,
        model_name: str,
        log_texts: Sequence[str],
        source_names: Optional[Sequence[str]] = None,
        persist: bool = True,
        min_ok_observations: int = 4,
    ) -> Dict[str, Any]:
        observations = []
        for idx, text in enumerate(log_texts):
            source = source_names[idx] if source_names and idx < len(source_names) else f"log_{idx}"
            observations.append(parse_runtime_log_text(text, source_name=source))
        return self.fit_from_observations(model_name, observations, persist=persist, min_ok_observations=min_ok_observations)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_TIME_FIELD_MAP = {
    "forward_layer_ms": ("forward_time_ms", "layer"),
    "backward_layer_ms": ("backward_time_ms", "layer"),
    "recompute_layer_ms": ("recompute_time_ms", "layer"),
    "tp_comm_layer_ms": ("tp_comm_time_ms", "layer"),
    "ep_comm_layer_ms": ("ep_comm_time_ms", "layer"),
    "sp_comm_layer_ms": ("sp_comm_time_ms", "layer"),
    "bubble_ms": ("bubble_time_ms", "direct"),
    "framework_ms": ("framework_overhead_ms", "direct"),
    "runtime_ms": ("runtime_overhead_ms", "direct"),
    "pp_comm_ms": ("pp_comm_time_ms", "direct"),
    "dp_exposed_ms": ("dp_exposed_comm_time_ms", "direct"),
    "offload_ms": ("offload_overhead_ms", "direct"),
    "optimizer_ms": ("optimizer_step_time_ms", "direct"),
}

_ALLOC_FIELD_MAP = {
    "param_gb": "parameter_memory_gb",
    "grad_gb": "gradient_memory_gb",
    "optimizer_gb": "optimizer_memory_gb",
    "activation_gb": "activation_memory_gb",
    "master_grad_gb": "master_grad_memory_gb",
    "tensor_fusion_gb": "tensor_fusion_buffer_gb",
    "comm_buffer_gb": "communication_buffer_gb",
    "temp_buffer_gb": "temporary_buffer_gb",
    "optimizer_ws_gb": "optimizer_update_workspace_gb",
}

_RESERVED_FIELD_MAP = {
    "temp_buffer_gb": "temporary_buffer_gb",
    "activation_pool_gb": "activation_buffer_pool_gb",
    "comm_runtime_pool_gb": "communication_runtime_pool_gb",
    "overlap_runtime_pool_gb": "overlap_runtime_pool_gb",
    "comm_fragmentation_gb": "communication_fragmentation_gb",
    "reserved_candidate_optimizer_gb": "reserved_candidate_optimizer_gb",
}


def _component_delta_map(feature_names: Sequence[str], coefficients: Sequence[float], basis_map: Dict[str, float]) -> Dict[str, float]:
    delta_by_component: Dict[str, float] = {}
    for name, coef in zip(feature_names, coefficients):
        if name not in basis_map:
            continue
        if "|" not in name:
            continue
        component = name.split("|", 1)[0]
        delta_by_component[component] = delta_by_component.get(component, 0.0) + float(coef) * float(basis_map[name])
    return delta_by_component


def _recompose_time_fields(result: Any) -> None:
    result.compute_time_ms = max(
        0.0,
        float(getattr(result, "forward_time_ms", 0.0) or 0.0)
        + float(getattr(result, "backward_time_ms", 0.0) or 0.0)
        + float(getattr(result, "bubble_time_ms", 0.0) or 0.0)
        + float(getattr(result, "framework_overhead_ms", 0.0) or 0.0)
        + float(getattr(result, "recompute_time_ms", 0.0) or 0.0)
        + float(getattr(result, "runtime_overhead_ms", 0.0) or 0.0)
    )
    result.total_comm_time_ms = max(
        0.0,
        float(getattr(result, "tp_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "dp_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "ep_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "pp_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "sp_comm_time_ms", 0.0) or 0.0)
    )
    result.effective_comm_time_ms = max(
        0.0,
        float(getattr(result, "tp_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "dp_exposed_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "ep_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "pp_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "sp_comm_time_ms", 0.0) or 0.0)
    )
    result.step_time_ms = max(
        0.0,
        float(getattr(result, "compute_time_ms", 0.0) or 0.0)
        + float(getattr(result, "effective_comm_time_ms", 0.0) or 0.0)
        + float(getattr(result, "optimizer_step_time_ms", 0.0) or 0.0)
    )


def apply_runtime_calibration_to_result(
    *,
    result: Any,
    calibration_entry: Optional[Dict[str, Any]],
    parallel: Any,
    runtime_context: Optional[Dict[str, Any]],
    gpu_memory_gb: float,
    total_layers: Optional[int] = None,
) -> Any:
    if not calibration_entry:
        return result

    context = dict(runtime_context or {})
    fit_kind = str(calibration_entry.get("fit_kind", "") or "")
    schema_version = int(calibration_entry.get("schema_version", 5) or 5)
    if fit_kind in {"config_ratio_regression_v1", "signature_exact_plus_compact_v5", "compact_component_residual_v3", "compact_component_residual_v4"}:
        exact_model = calibration_entry.get("exact_signature_model") or {}
        sig_key = _signature_key_from_context(context)
        coarse_key = _coarse_signature_key_from_context(context)
        minimal_key = _minimal_signature_key_from_context(context)
        ok_entry = (exact_model.get("ok_signatures") or {}).get(sig_key)
        if ok_entry is None:
            ok_entry = (exact_model.get("ok_signatures_coarse") or {}).get(coarse_key)
        if ok_entry is None:
            ok_entry = (exact_model.get("ok_signatures_minimal") or {}).get(minimal_key)
        if ok_entry is not None:
            result.step_time_ms = float(ok_entry.get("step_time_ms", getattr(result, "step_time_ms", 0.0)) or 0.0)
            result.allocated_memory_gb = float(ok_entry.get("allocated_memory_gb", getattr(result, "allocated_memory_gb", 0.0)) or 0.0)
            result.reserved_memory_gb = float(ok_entry.get("reserved_memory_gb", getattr(result, "reserved_memory_gb", 0.0)) or 0.0)
            result.memory_gb = max(float(getattr(result, "memory_gb", 0.0) or 0.0), float(result.reserved_memory_gb or 0.0))
            result.fits_memory = True
            result.parallel_config["runtime_calibration_exact_signature_match"] = True
            result.parallel_config["runtime_calibration_signature_key"] = sig_key
            return result
        if sig_key in (exact_model.get("oom_signatures") or {}) or coarse_key in (exact_model.get("oom_signatures_coarse") or {}) or minimal_key in (exact_model.get("oom_signatures_minimal") or {}):
            result.fits_memory = False
            result.parallel_config["runtime_calibration_exact_signature_match"] = True
            result.parallel_config["runtime_calibration_exact_status"] = "OOM"
            result.parallel_config["runtime_calibration_signature_key"] = sig_key
            return result
        if sig_key in (exact_model.get("fail_signatures") or {}) or coarse_key in (exact_model.get("fail_signatures_coarse") or {}) or minimal_key in (exact_model.get("fail_signatures_minimal") or {}):
            result.parallel_config["runtime_calibration_exact_signature_match"] = True
            result.parallel_config["runtime_calibration_exact_status"] = "FAIL"
            result.parallel_config["runtime_calibration_signature_key"] = sig_key
            return result

        tcoef = np.asarray((calibration_entry.get("time_model") or {}).get("coefficients") or [], dtype=float)
        tfeat = _compact_time_feature_values(result, context)
        if tcoef.size and tcoef.size == tfeat.size:
            # Ratio-based correction: predicted_ratio = dot(beta, features)
            # calibrated_step = predicted_step * ratio
            feature_names = _compact_time_feature_names()
            predicted_ratio = float(np.dot(tcoef, tfeat))
            predicted_ratio = max(0.3, min(5.0, predicted_ratio))  # safety clamp
            predicted_step = float(result.step_time_ms or 0.0)
            calibrated_step = predicted_step * predicted_ratio
            # Scale ALL time sub-components by the same ratio
            for attr in (
                "forward_time_ms", "backward_time_ms", "recompute_time_ms",
                "tp_comm_time_ms", "sp_comm_time_ms",
                "dp_exposed_comm_time_ms", "pp_comm_time_ms", "ep_comm_time_ms",
                "bubble_time_ms", "framework_overhead_ms", "runtime_overhead_ms",
                "optimizer_step_time_ms", "offload_overhead_ms",
            ):
                cur = float(getattr(result, attr, 0.0) or 0.0)
                if cur > 0:
                    setattr(result, attr, cur * predicted_ratio)
            result.step_time_ms = max(0.0, calibrated_step)
            result.parallel_config["runtime_calibration_time_ratio"] = round(predicted_ratio, 6)
            result.parallel_config["runtime_calibration_time_coefficients"] = {
                name: round(float(c), 6) for name, c in zip(feature_names, tcoef)
            }

        acoef = np.asarray((calibration_entry.get("allocated_model") or {}).get("coefficients") or [], dtype=float)
        afeat = _compact_alloc_feature_values(result, context)
        if acoef.size and acoef.size == afeat.size:
            alloc_ratio = float(np.dot(acoef, afeat))
            alloc_ratio = max(0.3, min(5.0, alloc_ratio))
            predicted_alloc = float(result.allocated_memory_gb or 0.0)
            result.allocated_memory_gb = max(0.0, predicted_alloc * alloc_ratio)
            # Also scale breakdown fields proportionally
            breakdown = getattr(result, "memory_breakdown", None)
            if breakdown is not None:
                for attr_name in (
                    "parameter_memory_gb", "gradient_memory_gb", "master_grad_memory_gb",
                    "optimizer_memory_gb", "activation_memory_gb", "temporary_buffer_gb",
                    "communication_buffer_gb", "tensor_fusion_buffer_gb",
                    "optimizer_update_workspace_gb",
                ):
                    cur = float(getattr(breakdown, attr_name, 0.0) or 0.0)
                    if cur > 0:
                        setattr(breakdown, attr_name, cur * alloc_ratio)
            result.parallel_config["runtime_calibration_alloc_ratio"] = round(alloc_ratio, 6)

        rcoef = np.asarray((calibration_entry.get("reserved_extra_model") or {}).get("coefficients") or [], dtype=float)
        rfeat = _compact_reserved_feature_values(result, context)
        if rcoef.size and rcoef.size == rfeat.size:
            reserved_ratio = float(np.dot(rcoef, rfeat))
            reserved_ratio = max(0.3, min(5.0, reserved_ratio))
            predicted_reserved = float(result.reserved_memory_gb or 0.0)
            result.reserved_memory_gb = max(result.allocated_memory_gb, predicted_reserved * reserved_ratio)
            result.parallel_config["runtime_calibration_reserved_ratio"] = round(reserved_ratio, 6)

        result.memory_gb = max(float(getattr(result, "memory_gb", 0.0) or 0.0), float(result.reserved_memory_gb or 0.0))
        status_model = calibration_entry.get("status_model") or {}
        oom_guard_gb = float(status_model.get("oom_reserved_guard_gb", 0.0) or 0.0)
        effective_reserved = float(result.reserved_memory_gb or 0.0) + max(0.0, oom_guard_gb)
        result.parallel_config["runtime_calibration_oom_guard_gb"] = float(max(0.0, oom_guard_gb))
        result.parallel_config["runtime_calibration_effective_reserved_gb"] = float(effective_reserved)
        result.fits_memory = bool(effective_reserved <= float(gpu_memory_gb or 0.0)) if gpu_memory_gb else bool(getattr(result, "fits_memory", True))
        return result
    if total_layers is None:
        stage_layers = getattr(result, "stage_layer_counts", None)
        total_layers = int(sum(stage_layers)) if stage_layers else 1

    active_context_features = calibration_entry.get("active_context_features") or []

    # ---- time ----
    time_model = calibration_entry.get("time_model") or {}
    time_basis = _build_basis(
        _time_component_exposures(result, int(total_layers or 1)),
        context,
        active_context_features=active_context_features,
        include_constant=False,
    )
    time_delta_by_component = _component_delta_map(
        time_model.get("feature_names") or [],
        time_model.get("coefficients") or [],
        time_basis[2],
    )
    for component, delta in time_delta_by_component.items():
        field_info = _TIME_FIELD_MAP.get(component)
        if not field_info:
            continue
        attr_name, mode = field_info
        current = float(getattr(result, attr_name, 0.0) or 0.0)
        if mode == "layer":
            new_value = max(0.0, current + float(delta))
        else:
            new_value = max(0.0, current + float(delta))
        setattr(result, attr_name, new_value)
    # Keep reporting field in sync for users who inspect it, but step path still goes through optimizer.
    if float(getattr(result, "optimizer_step_time_ms", 0.0) or 0.0) < float(getattr(result, "offload_overhead_ms", 0.0) or 0.0):
        result.optimizer_step_time_ms = float(getattr(result, "offload_overhead_ms", 0.0) or 0.0)
    _recompose_time_fields(result)

    # ---- allocated memory ----
    breakdown = getattr(result, "memory_breakdown", None)
    if breakdown is not None:
        alloc_model = calibration_entry.get("allocated_model") or {}
        alloc_basis = _build_basis(
            _alloc_component_exposures(result),
            context,
            active_context_features=active_context_features,
            include_constant=False,
        )
        alloc_delta_by_component = _component_delta_map(
            alloc_model.get("feature_names") or [],
            alloc_model.get("coefficients") or [],
            alloc_basis[2],
        )
        for component, delta in alloc_delta_by_component.items():
            field_name = _ALLOC_FIELD_MAP.get(component)
            if not field_name:
                continue
            current = float(getattr(breakdown, field_name, 0.0) or 0.0)
            setattr(breakdown, field_name, max(0.0, current + float(delta)))
        result.allocated_memory_gb = float(getattr(breakdown, "allocated_memory_gb", result.allocated_memory_gb) or 0.0)

        # ---- reserved extra ----
        reserved_model = calibration_entry.get("reserved_extra_model") or {}
        reserved_basis = _build_basis(
            _reserved_extra_component_exposures(result),
            context,
            active_context_features=active_context_features,
            include_constant=False,
        )
        reserved_delta_by_component = _component_delta_map(
            reserved_model.get("feature_names") or [],
            reserved_model.get("coefficients") or [],
            reserved_basis[2],
        )
        for component, delta in reserved_delta_by_component.items():
            field_name = _RESERVED_FIELD_MAP.get(component)
            if not field_name:
                continue
            current = float(getattr(breakdown, field_name, 0.0) or 0.0)
            setattr(breakdown, field_name, max(0.0, current + float(delta)))
        result.reserved_memory_gb = float(getattr(breakdown, "reserved_memory_gb", result.reserved_memory_gb) or 0.0)
        result.memory_gb = max(float(getattr(result, "memory_gb", 0.0) or 0.0), float(result.reserved_memory_gb or 0.0))
        result.fits_memory = bool(result.memory_gb <= float(gpu_memory_gb or 0.0)) if gpu_memory_gb else bool(getattr(result, "fits_memory", True))
    return result


__all__ = [
    "RuntimeObservation",
    "RuntimeCalibrationStore",
    "RuntimeCalibrationFitter",
    "canonicalize_model_name",
    "default_model_key_from_config",
    "parse_runtime_log_text",
    "parse_runtime_log_file",
    "parse_runtime_logs",
    "build_runtime_context_snapshot",
    "get_runtime_calibration_supported_fields",
    "apply_runtime_calibration_to_result",
]
