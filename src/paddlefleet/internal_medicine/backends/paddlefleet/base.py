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

"""PaddleFleet-specific Probe base class with GPU-buffer recording API.

Hot-path discipline: see ``.claude/skills/monitor-hook-perf-rules``.
``declare_*`` is the schema gate; ``record_*`` only has a disabled-key guard.
"""

import logging

import paddle

from ...core.base_monitor import Probe
from ...core.training_logs import training_logs

logger = logging.getLogger(__name__)


class PaddleProbe(Probe):
    """Probe with paddle-backed GPU-buffer accumulator + recompute guard.

    GPU-buffer API — ``declare_mean`` / ``declare_max`` / ``declare_min``
    at hook-registration time, then ``record_mean`` / ``record_max`` /
    ``record_min`` inside hooks with **GPU 0-dim tensors**. No D2H sync
    fires until ``step()`` does a single batched flush.
    """

    def __init__(
        self,
        log_per_layer=True,
        log_global=True,
        monitor_interval=1,
        verbose=False,
        exclude_families=None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            exclude_families=exclude_families,
        )
        self._mean_keys: set[str] = set()  # 需要求平均的 metric keys
        self._max_keys: set[str] = set()  # 需要取最大值的 metric keys
        self._min_keys: set[str] = set()  # 需要取最小值的 metric keys
        self._gpu_acc: dict[str, paddle.Tensor] = {}  # GPU 0-dim 累加器
        self._gpu_cnt: dict[str, int] = {}  # 每个 key 被 record 的次数
        # global_key → (聚合方式, [对应的 layer keys])，flush 时从 layer 推导 global
        self._layer_metric_groups: dict[str, tuple[str, list[str]]] = {}
        self._layer_metric_keys: set[str] = (
            set()
        )  # 所有 per-layer key，用于 flush 时判断是否输出
        self._disabled_keys: set[str] = (
            set()
        )  # log_global=False 时被禁用的 global keys
        self._vector_keys: dict[str, int] = {}  # per-layer 向量 key → 长度
        self._gpu_vec: dict[
            str, paddle.Tensor
        ] = {}  # per-layer 向量累加器 [size]
        self._gpu_vec_cnt: dict[str, int] = {}
        self._vector_elem_keys: dict[
            str, list[str]
        ] = {}  # 向量 key → 展开后的逐元素 key
        self._mtp_layer_ids: set[int] = set()
        self._buffers_allocated = False

    def _should_monitor(self) -> bool:
        if not paddle.is_grad_enabled():
            return False
        return super()._should_monitor()

    def _flush_buffers(self) -> None:
        """由 step() 调用，执行唯一的 D2H 传输并写入 training_logs。"""
        if not self._buffers_allocated:
            return
        flushed = self._flush_gpu_buffer()
        if flushed:
            training_logs.update(**flushed)

    # ------------------------------------------------------------------
    # GPU-buffer API: declare → allocate → record → flush
    # ------------------------------------------------------------------

    def declare_mean(self, key: str) -> None:
        assert not self._buffers_allocated, (
            f"declare_mean({key!r}) after allocate_buffers"
        )
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert (
            key not in self._mean_keys
            and key not in self._max_keys
            and key not in self._min_keys
        )
        self._mean_keys.add(key)

    def declare_max(self, key: str) -> None:
        assert not self._buffers_allocated, (
            f"declare_max({key!r}) after allocate_buffers"
        )
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert (
            key not in self._mean_keys
            and key not in self._max_keys
            and key not in self._min_keys
        )
        self._max_keys.add(key)

    def declare_min(self, key: str) -> None:
        assert not self._buffers_allocated, (
            f"declare_min({key!r}) after allocate_buffers"
        )
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert (
            key not in self._mean_keys
            and key not in self._max_keys
            and key not in self._min_keys
        )
        self._min_keys.add(key)

    def allocate_buffers(self, dtype=None) -> None:
        """物化所有已声明的累加器为 GPU 0-dim tensor。幂等，调用后 schema 冻结。"""
        if self._buffers_allocated:
            return
        if dtype is None:
            dtype = "float32"
        # mean: 初始化为 0，record 时累加，flush 时除以 count
        for k in self._mean_keys:
            self._gpu_acc[k] = paddle.zeros((), dtype=dtype)
            self._gpu_cnt[k] = 0
        # max: 初始化为 -inf，record 时取 maximum
        for k in self._max_keys:
            self._gpu_acc[k] = paddle.full((), float("-inf"), dtype=dtype)
            self._gpu_cnt[k] = 0
        # min: 初始化为 +inf，record 时取 minimum
        for k in self._min_keys:
            self._gpu_acc[k] = paddle.full((), float("inf"), dtype=dtype)
            self._gpu_cnt[k] = 0
        # vector: 每层一条 [size] 累加器，flush 时按元素展开成独立 key
        for k, size in self._vector_keys.items():
            self._gpu_vec[k] = paddle.zeros([size], dtype=dtype)
            self._gpu_vec_cnt[k] = 0
        self._buffers_allocated = True
        if self.verbose:
            logger.info(
                f"[{self.METRIC_PREFIX}] GPU buffer: "
                f"mean={len(self._mean_keys)} max={len(self._max_keys)} min={len(self._min_keys)} "
                f"vector={len(self._vector_keys)}"
            )

    def record_mean(self, key: str, val: paddle.Tensor) -> None:
        """热路径：GPU 上就地累加，不触发 D2H 同步。"""
        if key in self._disabled_keys:
            return
        self._gpu_acc[key].add_(val.detach())
        self._gpu_cnt[key] += 1

    def record_max(self, key: str, val: paddle.Tensor) -> None:
        """热路径：GPU 上取 maximum，不触发 D2H 同步。"""
        if key in self._disabled_keys:
            return
        # paddle 不支持 maximum 的 out= 参数，用 assign 实现就地写入
        paddle.assign(
            paddle.maximum(self._gpu_acc[key], val.detach()), self._gpu_acc[key]
        )
        self._gpu_cnt[key] += 1

    def record_min(self, key: str, val: paddle.Tensor) -> None:
        """热路径：GPU 上取 minimum，不触发 D2H 同步。"""
        if key in self._disabled_keys:
            return
        paddle.assign(
            paddle.minimum(self._gpu_acc[key], val.detach()), self._gpu_acc[key]
        )
        self._gpu_cnt[key] += 1

    # ------------------------------------------------------------------
    # Convenience: declare/record using class-level aggregation rules
    # ------------------------------------------------------------------

    def mark_mtp_layers(self, layer_ids) -> None:
        """Mark MTP layers before declaring the metric schema."""
        assert not self._buffers_allocated, (
            "mark_mtp_layers after allocate_buffers"
        )
        self._mtp_layer_ids.update(int(layer_idx) for layer_idx in layer_ids)

    def _layer_key(
        self, layer_idx: int, metric_name: str, attn_type: str | None = None
    ) -> str:
        # Attention type stays on the metric name so window/full charts split
        # naturally. MTP identity stays on the layer token and follows the
        # existing metric gather/JSONL path without separate metadata.
        layer_token = f"layer_{layer_idx}"
        if layer_idx in self._mtp_layer_ids:
            layer_token += "_mtp"
        if attn_type is not None:
            return (
                f"{self.METRIC_PREFIX}/{layer_token}/{attn_type}_{metric_name}"
            )
        return f"{self.METRIC_PREFIX}/{layer_token}/{metric_name}"

    def _global_key(
        self, metric_name: str, attn_type: str | None = None
    ) -> str:
        if attn_type is not None:
            return f"{self.METRIC_PREFIX}/global_{attn_type}_{metric_name}"
        return f"{self.METRIC_PREFIX}/global_{metric_name}"

    def _should_disable_explicit_key(self, key: str) -> bool:
        if (
            key.startswith(f"{self.METRIC_PREFIX}/global_")
            and not self.log_global
        ):
            return True
        return not self.family_allows(key)

    def declare_layer_metric(
        self, layer_idx: int, metric_name: str, attn_type: str | None = None
    ) -> None:
        """声明一个 per-layer 指标。

        1. 根据 MAX_AGGREGATED/MIN_AGGREGATED 选择聚合方式，注册 layer key
        2. 建立 layer_key → global_key 的分组映射，flush 时自动推导 global

        ``attn_type`` (optional): when set (``"mla"`` / ``"hca"`` / ``"csa"`` /
        ``"window"`` / ``"mqa"`` on ``csa_compress_ratios`` stacks, ``"swa"`` /
        ``"full"`` on ``sliding_window`` stacks), the tag is prepended to
        ``metric_name`` in both layer and global keys so the viewer renders each
        attention kind in a separate chart. When ``None``, legacy key layout is
        preserved.
        """
        if not (self.log_per_layer or self.log_global):
            return
        layer_key = self._layer_key(layer_idx, metric_name, attn_type=attn_type)
        global_key = self._global_key(metric_name, attn_type=attn_type)
        if not self.family_allows(layer_key):
            # Registered as disabled rather than merely skipped: record_* looks the
            # key up here to stay a cheap no-op instead of a KeyError.
            self._disabled_keys.add(layer_key)
            return
        all_declared = self._mean_keys | self._max_keys | self._min_keys
        assert global_key not in all_declared
        # 根据类级别聚合规则选择 declare 方式 (基于原始 metric_name，attn_type 不改变聚合语义)
        if self._is_max_aggregated(metric_name):
            agg = "max"
            if layer_key not in all_declared:
                self.declare_max(layer_key)
        elif metric_name in self.MIN_AGGREGATED:
            agg = "min"
            if layer_key not in all_declared:
                self.declare_min(layer_key)
        else:
            agg = "mean"
            if layer_key not in all_declared:
                self.declare_mean(layer_key)
        self._layer_metric_keys.add(layer_key)
        if not self.log_global:
            return
        # 注册 global 分组: flush 时从这些 layer keys 聚合出 global 值
        existing = self._layer_metric_groups.get(global_key)
        if existing is None:
            self._layer_metric_groups[global_key] = (agg, [layer_key])
        else:
            assert existing[0] == agg
            existing[1].append(layer_key)

    def record_layer_metric(
        self,
        layer_idx: int,
        metric_name: str,
        val: paddle.Tensor,
        attn_type: str | None = None,
    ) -> None:
        """热路径：只写 per-layer 累加器，global 在 flush 时从各层推导。"""
        if not (self.log_per_layer or self.log_global):
            return
        layer_key = self._layer_key(layer_idx, metric_name, attn_type=attn_type)
        if self._is_max_aggregated(metric_name):
            self.record_max(layer_key, val)
        elif metric_name in self.MIN_AGGREGATED:
            self.record_min(layer_key, val)
        else:
            self.record_mean(layer_key, val)

    # ------------------------------------------------------------------
    # Per-layer vector metrics (one curve per element, e.g. per expert)
    # ------------------------------------------------------------------

    def declare_layer_vector(
        self, layer_idx: int, metric_name: str, size: int, elem_tag: str = "e"
    ) -> None:
        """声明一个 per-layer 向量指标，mean 聚合，flush 时展开为 ``{key}_{elem_tag}{i}``。

        用于「每层每个元素一条曲线」的指标（例如每个专家的占比）：热路径上整条
        向量只有一次 ``add_``，而不是每个元素一个 kernel。向量指标不参与 global
        派生 —— 跨层视图由 viewer 自己从各层曲线汇总。
        """
        assert not self._buffers_allocated, (
            f"declare_layer_vector({metric_name!r}) after allocate_buffers"
        )
        if not self.log_per_layer:
            return
        key = self._layer_key(layer_idx, metric_name)
        # 判族要用展开后的元素名：per-unit 族（cell / stream）只在 ``_{elem_tag}{i}``
        # 后缀上才匹配得到，base name 会落到聚合族（mix / gate）而误判为保留。
        if not self.family_allows(f"{key}_{elem_tag}0"):
            # No buffer is allocated, so record_layer_vector already no-ops on it.
            return
        assert key not in self._vector_keys, (
            f"declare_layer_vector({key!r}) declared twice"
        )
        size = int(size)
        assert size > 0
        self._vector_keys[key] = size
        self._vector_elem_keys[key] = [
            f"{key}_{elem_tag}{i}" for i in range(size)
        ]

    def record_layer_vector(
        self, layer_idx: int, metric_name: str, vec: paddle.Tensor
    ) -> None:
        """热路径：整条 ``[size]`` 向量一次就地累加，不触发 D2H 同步。"""
        key = self._layer_key(layer_idx, metric_name)
        buf = self._gpu_vec.get(key)
        if buf is None:  # log_per_layer=False 或该层未声明
            return
        buf.add_(vec.detach().astype(buf.dtype))
        self._gpu_vec_cnt[key] += 1

    def _emits_layer_key(self, key: str) -> bool:
        """per-layer key 是否写进日志（global 派生不受此影响）。"""
        if key not in self._layer_metric_keys:
            return True  # 显式 declare 的非逐层 key
        return self.log_per_layer

    def _flush_gpu_buffer(self) -> dict[str, float]:
        """单次批量 D2H：收集所有累加器 → 推导 global → concat→cpu→tolist → 重置。"""
        keys: list[str] = []
        tensors: list[paddle.Tensor] = []

        # 1) 收集 per-layer mean 指标: acc / count
        for k in self._mean_keys:
            cnt = self._gpu_cnt[k]
            if cnt == 0:
                continue
            if self._emits_layer_key(k):
                keys.append(k)
                tensors.append(self._gpu_acc[k] / cnt)

        # 2) 收集 per-layer max 指标: 直接取累加器值
        for k in self._max_keys:
            if self._gpu_cnt[k] == 0:
                continue
            if self._emits_layer_key(k):
                keys.append(k)
                tensors.append(self._gpu_acc[k].clone())

        # 3) 收集 per-layer min 指标
        for k in self._min_keys:
            if self._gpu_cnt[k] == 0:
                continue
            if self._emits_layer_key(k):
                keys.append(k)
                tensors.append(self._gpu_acc[k].clone())

        # 4) 从各层累加器推导 global 值（不需要 hook 时双写）
        if self.log_global:
            for global_key, (
                agg,
                layer_keys,
            ) in self._layer_metric_groups.items():
                active = [
                    lk for lk in layer_keys if self._gpu_cnt.get(lk, 0) > 0
                ]
                if not active:
                    continue
                if agg == "mean":
                    total_sum = paddle.stack(
                        [self._gpu_acc[lk] for lk in active]
                    ).sum()
                    total_cnt = sum(self._gpu_cnt[lk] for lk in active)
                    tensors.append(total_sum / total_cnt)
                elif agg == "max":
                    tensors.append(
                        paddle.stack([self._gpu_acc[lk] for lk in active]).max()
                    )
                else:
                    tensors.append(
                        paddle.stack([self._gpu_acc[lk] for lk in active]).min()
                    )
                keys.append(global_key)

        # 5) 唯一 D2H 同步点：一次 concat → cpu → tolist（标量与向量共用）
        vec_key_groups: list[list[str]] = []
        vec_tensors: list[paddle.Tensor] = []
        for k, elem_keys in self._vector_elem_keys.items():
            cnt = self._gpu_vec_cnt[k]
            if cnt == 0:
                continue
            vec_key_groups.append(elem_keys)
            vec_tensors.append(self._gpu_vec[k] / cnt)

        out: dict[str, float] = {}
        if tensors or vec_tensors:
            flat = paddle.concat(
                [t.reshape([-1]) for t in tensors] + vec_tensors
            )
            vals = flat.cpu().tolist()
            out = dict(zip(keys, vals, strict=False))
            offset = len(keys)
            for elem_keys in vec_key_groups:
                out.update(
                    zip(
                        elem_keys,
                        vals[offset : offset + len(elem_keys)],
                        strict=False,
                    )
                )
                offset += len(elem_keys)

        # 6) 重置所有累加器，为下一个 step 准备
        for k in self._mean_keys:
            self._gpu_acc[k].zero_()
            self._gpu_cnt[k] = 0
        for k in self._max_keys:
            paddle.assign(
                paddle.full((), float("-inf"), dtype=self._gpu_acc[k].dtype),
                self._gpu_acc[k],
            )
            self._gpu_cnt[k] = 0
        for k in self._min_keys:
            paddle.assign(
                paddle.full((), float("inf"), dtype=self._gpu_acc[k].dtype),
                self._gpu_acc[k],
            )
            self._gpu_cnt[k] = 0
        for k in self._vector_keys:
            self._gpu_vec[k].zero_()
            self._gpu_vec_cnt[k] = 0
        return out
