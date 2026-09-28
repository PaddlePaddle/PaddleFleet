# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Framework-agnostic training metrics store.

All values stored as Python floats. Framework-specific tensor conversion
is the caller's responsibility.
"""

import logging
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ("ReduceSchema", "SmoothedValue", "TrainingLogs", "training_logs")


@dataclass(frozen=True)
class ReduceSchema:
    """One rank's metrics, split by how they reduce across ranks.

    ``mean_keys`` / ``max_keys`` / ``min_keys`` are sorted so that two ranks
    holding the same key set produce byte-identical layouts, which is what makes
    the fingerprint below a usable agreement check.
    """

    values: dict
    mean_keys: tuple[str, ...] = ()
    max_keys: tuple[str, ...] = ()
    min_keys: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> int:
        """Stable hash of the local key layout.

        ``hash()`` is salted per process (PYTHONHASHSEED), so it would disagree
        between ranks that hold identical keys and trigger a re-alignment every
        step. CRC32 is stable across processes and interpreter runs.
        """
        blob = "\u0000".join(
            ("M", *self.mean_keys, "X", *self.max_keys, "N", *self.min_keys)
        )
        return zlib.crc32(blob.encode()) & 0x7FFFFFFF

    def all_keys(self) -> tuple[str, ...]:
        return self.mean_keys + self.max_keys + self.min_keys


MAX_AGGREGATED_SUFFIXES = (
    "topk_channel_norm",
    "channel_max_ratio",
    "channel_median",
    "channel_p95",
    "channel_p99",
    "activation_rms",
    "massive_act_channel_count",
)

# Absolute-threshold channel counts: the threshold is the trailing token
# (``channel_count_gt_10``), so this cannot be a fixed suffix.
CHANNEL_COUNT_GT_MARKER = "channel_count_gt_"


class SmoothedValue:
    """Track a series of scalar values and provide smoothed access."""

    def __init__(self, mode="mean"):
        self.total = 0.0
        self.count = 0
        self.mode = mode
        self.latest_value = 0.0
        if self.mode == "max":
            self.max_value = float("-inf")
        if self.mode == "min":
            self.min_value = float("inf")

    def update(self, value: float):
        self.latest_value = value
        self.count += 1
        self.total += value
        if self.mode == "max":
            self.max_value = max(self.max_value, value)
        if self.mode == "min":
            self.min_value = min(self.min_value, value)

    @property
    def global_avg(self):
        return self.total / max(self.count, 1e-6)

    @property
    def log(self):
        if self.mode == "max":
            return self.max_value
        if self.mode == "min":
            return self.min_value
        return self.global_avg

    @property
    def latest(self):
        return self.latest_value

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.latest_value = 0.0
        if self.mode == "max":
            self.max_value = float("-inf")
        if self.mode == "min":
            self.min_value = float("inf")


class TrainingLogs:
    """Singleton metric store. All monitors write here."""

    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, gather_fn: Callable | None = None):
        if not hasattr(self, "meters"):
            self.meters = {}
        if gather_fn is not None:
            self._gather_fn = gather_fn

    def set_gather_fn(self, fn: Callable):
        """Set the distributed gather function (backend provides this)."""
        self._gather_fn = fn

    def set_reduce_fn(self, fn: Callable):
        """Install a numeric cross-rank reducer, preferred over ``gather_fn``.

        ``all_gather_object`` pickles the whole ``{key: value}`` dict, so its
        payload — and the receive buffer, and the number of unpickle calls on
        every rank — grows with world size: at 6144 ranks a 155 KB dict means
        ~930 MB of traffic per rank and 6144 deserialisations, none of which the
        GPU can overlap. The aggregation only ever needs mean/max/min, which a
        reduction can do in one pass with a payload that does not depend on world
        size at all. Backends that cannot reduce (or want the old semantics) just
        leave this unset and keep using ``gather_fn``.

        ``fn(schema) -> dict[str, float] | None`` receives a ``ReduceSchema`` and
        returns the aggregated metrics, or None to fall back to the gather path.
        """
        self._reduce_fn = fn

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, k, v):
        val = (
            float(v)
            if isinstance(v, int | float)
            else float(v.item() if hasattr(v, "item") else v)
        )
        if k not in self.meters:
            if self._is_max_metric(k):
                mode = "max"
            elif self._is_min_metric(k):
                mode = "min"
            else:
                mode = "mean"
            self.meters[k] = SmoothedValue(mode=mode)
        self.meters[k].update(val)

    @staticmethod
    def _is_max_metric(key: str) -> bool:
        return (
            "/max" in key
            or key.endswith("_max")
            or key.endswith(MAX_AGGREGATED_SUFFIXES)
            or CHANNEL_COUNT_GT_MARKER in key
        )

    @staticmethod
    def _is_min_metric(key: str) -> bool:
        return "/min" in key or key.endswith("_min")

    def __getitem__(self, v):
        return self.meters[v]

    def dict(self, smoothed: bool = False):
        return {
            k: (v.log if smoothed else v.latest) for k, v in self.meters.items()
        }

    def get_latest(self, prefix=None, smoothed: bool = False):
        result = {}
        for k, v in self.meters.items():
            if prefix is None or k.startswith(prefix):
                result[k] = v.log if smoothed else v.latest
        return result

    def print_metrics(self, metrics=None, prefix=None, format_fn=None):
        if format_fn is None:
            format_fn = logger.info
        if metrics is None:
            metrics = self.get_latest(prefix)
        elif prefix:
            metrics = {k: v for k, v in metrics.items() if k.startswith(prefix)}
        if not metrics:
            return
        grouped = defaultdict(dict)
        for k, v in metrics.items():
            if "/" in k:
                parts = k.split("/")
                category = parts[0]
                metric_name = "/".join(parts[1:])
                grouped[category][metric_name] = v
            else:
                grouped["other"][k] = v
        for category, items in sorted(grouped.items()):
            format_fn(f"[{category}]")
            for name, value in sorted(items.items()):
                format_fn(f"  {name}: {value:.4f}")

    def reset(self):
        self.meters.clear()

    def gather_and_aggregate(self):
        """Aggregate metrics across ranks, by reduction when the backend can."""
        all_metrics = self.get_latest()

        reduce_fn = getattr(self, "_reduce_fn", None)
        if reduce_fn is not None:
            reduced = reduce_fn(self.build_reduce_schema(all_metrics))
            if reduced is not None:
                return reduced

        gather_fn = getattr(self, "_gather_fn", None)
        if gather_fn is None:
            return all_metrics

        info_list = gather_fn(all_metrics)
        if info_list is None:
            return all_metrics

        aggregated = {}
        all_keys = {key for item in info_list for key in item}
        for k in all_keys:
            values = [v[k] for v in info_list if k in v]
            if not values:
                continue
            if self._is_max_metric(k):
                aggregated[k] = max(values)
            elif self._is_min_metric(k):
                aggregated[k] = min(values)
            else:
                aggregated[k] = sum(values) / len(values)
        return aggregated

    def build_reduce_schema(self, metrics: dict) -> ReduceSchema:
        """Describe this rank's metrics for a numeric cross-rank reduction.

        The mode split has to happen here, not in the backend: ``_is_max_metric``
        and ``_is_min_metric`` are the single source of truth for how a key
        reduces, and the gather path already agrees with them.
        """
        mean_keys, max_keys, min_keys = [], [], []
        for key in metrics:
            if self._is_max_metric(key):
                max_keys.append(key)
            elif self._is_min_metric(key):
                min_keys.append(key)
            else:
                mean_keys.append(key)
        return ReduceSchema(
            values=metrics,
            mean_keys=tuple(sorted(mean_keys)),
            max_keys=tuple(sorted(max_keys)),
            min_keys=tuple(sorted(min_keys)),
        )


training_logs = TrainingLogs()
