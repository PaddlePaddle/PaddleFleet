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

"""PaddleFleet cross-rank aggregation for training_logs.

Two paths live here. ``_paddle_gather`` is the original object gather, kept as a
fallback. ``_PaddleReducer`` is the numeric path: it exchanges a float64 vector
whose length is the number of metric keys, so the payload no longer scales with
world size and no rank has to unpickle world_size dicts.
"""

import logging

logger = logging.getLogger(__name__)


def _paddle_gather(local_metrics: dict) -> list:
    """Gather metrics from all ranks using paddle.distributed."""
    import paddle.distributed as dist

    if not dist.is_initialized():
        return None
    info_list = []
    dist.all_gather_object(info_list, local_metrics)
    return info_list


class _PaddleReducer:
    """Reduce metrics across ranks with a layout agreed once, then reused.

    ``all_reduce`` needs every rank to pass the same shape, so the key layout has
    to be global. Deriving it is a collective operation, which means the
    *decision* to re-derive it must be collective too — a rank that re-aligns
    alone would hang the job. Hence every call first agrees, in two tiny
    reductions, on whether the cached layout still holds.
    """

    def __init__(self):
        self._layout: tuple[str, ...] = ()
        self._modes: tuple[
            int, ...
        ] = ()  # 0=mean, 1=max, 2=min, parallel to _layout
        self._fingerprint: int | None = None

    def __call__(self, schema):
        import paddle
        import paddle.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return None

        if not self._agree_on_layout(schema, paddle, dist):
            self._align(schema, dist)
        if not self._layout:
            return {}
        return self._reduce(schema, paddle, dist)

    # ------------------------------------------------------------------
    # Layout agreement
    # ------------------------------------------------------------------

    def _agree_on_layout(self, schema, paddle, dist) -> bool:
        """True when every rank still shares the cached layout."""
        local_fp = schema.fingerprint
        stale = 1 if (not self._layout or local_fp != self._fingerprint) else 0
        probe = paddle.to_tensor([local_fp, stale], dtype="int64")
        dist.all_reduce(probe, op=dist.ReduceOp.MAX)
        fp_max, any_stale = (int(v) for v in probe.numpy())
        fp_only = paddle.to_tensor([local_fp], dtype="int64")
        dist.all_reduce(fp_only, op=dist.ReduceOp.MIN)
        fp_min = int(fp_only.numpy()[0])
        return fp_max == fp_min and not any_stale

    def _align(self, schema, dist) -> None:
        """Derive the global layout from every rank's key names.

        Only the names travel here, and only when the schema actually changed —
        under a fixed model that is once per process.
        """
        payload = {
            "mean": schema.mean_keys,
            "max": schema.max_keys,
            "min": schema.min_keys,
        }
        gathered: list = []
        dist.all_gather_object(gathered, payload)

        merged: dict[int, set] = {0: set(), 1: set(), 2: set()}
        for item in gathered:
            merged[0].update(item.get("mean", ()))
            merged[1].update(item.get("max", ()))
            merged[2].update(item.get("min", ()))

        layout: list[str] = []
        modes: list[int] = []
        for mode in (0, 1, 2):
            for key in sorted(merged[mode]):
                layout.append(key)
                modes.append(mode)
        self._layout = tuple(layout)
        self._modes = tuple(modes)
        self._fingerprint = schema.fingerprint
        logger.info(
            f"[internal_medicine] reduce layout aligned: {len(self._layout)} keys"
        )

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------

    def _reduce(self, schema, paddle, dist) -> dict:
        values = schema.values
        n = len(self._layout)
        # float64: a float32 sum over thousands of ranks loses digits these
        # metrics are read at, and doubling a few-KB payload costs nothing.
        sums = [0.0] * n
        counts = [0.0] * n
        maxs = [float("-inf")] * n
        mins = [float("inf")] * n

        for idx, (key, mode) in enumerate(
            zip(self._layout, self._modes, strict=True)
        ):
            if key not in values:
                continue
            val = values[key]
            if mode == 0:
                sums[idx] = val
                counts[idx] = 1.0
            elif mode == 1:
                maxs[idx] = val
            else:
                mins[idx] = val

        # mean needs sum and count together, so pack them into one collective.
        sum_t = paddle.to_tensor(sums + counts, dtype="float64")
        max_t = paddle.to_tensor(maxs, dtype="float64")
        min_t = paddle.to_tensor(mins, dtype="float64")
        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(min_t, op=dist.ReduceOp.MIN)

        # Single D2H for the whole result, matching the probes' one-sync-per-step rule.
        flat = paddle.concat([sum_t, max_t, min_t]).numpy().tolist()
        sum_out, cnt_out = flat[:n], flat[n : 2 * n]
        max_out, min_out = flat[2 * n : 3 * n], flat[3 * n :]

        out: dict[str, float] = {}
        for idx, (key, mode) in enumerate(
            zip(self._layout, self._modes, strict=True)
        ):
            if mode == 0:
                count = cnt_out[idx]
                if count > 0:
                    out[key] = sum_out[idx] / count
            elif mode == 1:
                if max_out[idx] != float("-inf"):
                    out[key] = max_out[idx]
            elif min_out[idx] != float("inf"):
                out[key] = min_out[idx]
        return out


def install_gather_fn():
    """Install the paddle-based aggregation into the global training_logs.

    ``IM_DISABLE_REDUCE=1`` keeps the object-gather path. It exists so the
    reduction can be switched off in a running job — and so an A/B can hold
    everything else fixed — without shipping a different build.
    """
    import os

    from ...core.training_logs import training_logs

    try:
        import paddle.distributed as dist

        if dist.is_initialized():
            training_logs.set_gather_fn(_paddle_gather)
            if os.environ.get("IM_DISABLE_REDUCE", "") == "1":
                logger.info(
                    "[internal_medicine] IM_DISABLE_REDUCE=1: using all_gather_object"
                )
            else:
                training_logs.set_reduce_fn(_PaddleReducer())
    except ImportError:
        pass
