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

"""Sparse-attention forward kernels for the "cudnn" CSA sparse-attn backend.

Two forward implementations live here; the backward half is in
``csa_sparse_attn_bwd_cudnn.py``.

* ``flash_mla_sparse_attn`` -- the FlashMLA sparse prefill kernel plus the
  index/alignment helpers it needs. This is the historical default and the
  only path that covers every shape.
* ``cudnn_sparse_attn_fwd`` -- cuDNN Frontend's own DSA sparse prefill
  (``cudnn.DSA.sparse_attention_forward_wrapper``, added upstream in
  cudnn-frontend PR #569). Only the three ``(H, D_qk)`` variants in
  ``_CUDNN_FWD_VARIANTS`` are supported, so it is opt-in and falls back.

``sparse_attn_fwd`` dispatches between them. Both return the same
``(out, lse, lse_indexer)`` triple in the same layouts, so call sites only
need to thread the backend name through.
"""

import logging

import paddle
import paddle.nn.functional as F

from paddlefleet.fusions.csa_sparse_attn_utils import local_to_global_flat

_logger = logging.getLogger(__name__)
# One-time "cuDNN forward is really running" marker, see cudnn_sparse_attn_fwd.
_logged_cudnn_fwd_active = False
# Tri-state cache for _cudnn_fwd_module_available(): None = not probed yet.
_cudnn_fwd_module_cached = None

try:
    from paddlefleet_ops.flash_mla import (
        flash_mla_sparse_fwd as _flash_mla_sparse_fwd,
    )
except (ImportError, RuntimeError):
    _flash_mla_sparse_fwd = None


# ---------------------------------------------------------------------------
# Paddle <-> cuDNN Frontend bridging for the DSA sparse **forward**
#
# ``csa_sparse_attn_bwd_cudnn`` already installs four such patches at import
# (memory_format sentinels, cutlass nvgpu, nvtx range, Stream.cuda_stream).
# The forward kernels need two more; both are no-ops once applied.
# ---------------------------------------------------------------------------
def _patch_paddle_record_stream():
    """Give ``paddle.Tensor`` a torch-named ``record_stream``.

    ``sparse_attention_forward``'s ``_interface_sm100._record_stream`` calls
    ``tensor.record_stream(consumer)`` on every input so the caching allocator
    knows the raw kernel pointers are still in flight on that stream. Paddle
    only exposes the underscored ``_record_stream``, so the torch-proxied call
    raises ``AttributeError``. Forward the name; prefer passing the stream
    through and fall back to the no-arg form (which records on the current
    stream -- the same stream cuDNN's ``resolve_stream()`` resolves to under
    the bwd module's ``Stream.cuda_stream`` patch).
    """
    if hasattr(paddle.Tensor, "record_stream"):
        return

    def record_stream(self, stream=None):
        try:
            return self._record_stream(stream)
        except Exception:
            return self._record_stream()

    paddle.Tensor.record_stream = record_stream


def _alias_cudnn_modules():
    """Re-register the loaded cuDNN modules under a top-level ``cudnn`` name.

    ``paddlefleet_ops._safe_load_ecosystem_lib`` only puts the ops dir on
    ``sys.path`` inside a ``ModuleContext`` window, then renames everything to
    ``paddlefleet_ops.cudnn.*``. Existing DSA modules import their kernels
    eagerly at module scope, i.e. inside that window, so their absolute
    ``from cudnn.api_base import ...`` imports resolve.

    ``sparse_attention_forward._interface_sm100._make_kernel`` instead imports
    its kernel lazily at execute time, which lands outside the window and
    fails with ``ModuleNotFoundError: No module named 'cudnn'``. Alias the
    already-loaded module objects (``setdefault``, never re-import) so the lazy
    import hits the same instances -- a second copy would create a second
    ``APIBase`` and break identity checks.
    """
    import sys

    import paddlefleet_ops

    prefix = "paddlefleet_ops.cudnn"
    cudnn_mod = getattr(paddlefleet_ops, "cudnn", None)
    if cudnn_mod is None:
        return
    sys.modules.setdefault("cudnn", cudnn_mod)
    for name, mod in list(sys.modules.items()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.setdefault("cudnn" + name[len(prefix) :], mod)


def _get_topk_alignment() -> int:
    """Minimum ``TopK`` alignment required by the current GPU architecture.

    * SM90 : dual-warpgroup loop steps by 2 blocks → ``2 * B_TOPK = 128``
    * SM100: single-pipeline loop steps by 1 block → ``B_TOPK`` (64 for
      head64, 128 for head128). DSA uses ``D = 512`` which maps to the
      head64 kernel path → 64.
    """
    sm = paddle.cuda.get_device_capability()
    if sm[0] >= 10:
        return 64
    return 128


def flash_mla_sparse_attn(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    indexer_topk: int = 0,
    d_v=None,
    topk_length=None,
    global_kv_idx_remap_fusion: bool = False,
):
    if _flash_mla_sparse_fwd is None:
        raise RuntimeError("flash_mla is not available")

    b, sq, h, d = q.shape
    _, skv, _ = kv.shape
    topk = topk_idxs.shape[-1]
    # Value dim may be smaller than the query/key dim (absorbed MLA MQA uses
    # d_qk=576 / d_v=512). Default to a symmetric d_v=d.
    if d_v is None:
        d_v = d

    q_flat = q.reshape([b * sq, h, d])
    kv_flat = kv.reshape([b * skv, d])
    global_idxs = local_to_global_flat(
        topk_idxs, skv, fused=global_kv_idx_remap_fusion
    )
    # [b, sq] -> [b * sq]: one valid-prefix length per flattened query row.
    topk_length_flat = (
        None
        if topk_length is None
        else topk_length.reshape([b * sq]).cast("int32")
    )

    topk_align = _get_topk_alignment()
    topk_padded = (topk + topk_align - 1) // topk_align * topk_align
    if topk_padded != topk:
        global_idxs = F.pad(global_idxs, (0, topk_padded - topk), value=-1)

    res = _flash_mla_sparse_fwd(
        q_flat,
        kv_flat.unsqueeze(1),
        global_idxs.unsqueeze(1),
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length_flat,
        indexer_topk=indexer_topk,
    )
    if indexer_topk > 0:
        out_flat, _max_logits, lse, lse_indexer = res
        lse_indexer = lse_indexer.reshape([b, sq, h])
    else:
        out_flat, _max_logits, lse = res
        lse_indexer = None
    return (
        out_flat.reshape([b, sq, h, d_v]),
        lse.reshape([b, sq, h]),
        lse_indexer,
    )


# ---------------------------------------------------------------------------
# cuDNN Frontend DSA sparse prefill forward (cudnn-frontend PR #569)
# ---------------------------------------------------------------------------
# ``api._SUPPORTED_VARIANTS``: (H, D_qk) -> allowed ``indexer_topk`` values.
# Mirrored here so the dispatch can pre-filter without importing the kernel
# (the import is what pulls in CuTe DSL). Widen together with upstream.
_CUDNN_FWD_VARIANTS = {
    (64, 512): (0, 512, 1024, 2048),
    (64, 576): (0, 512, 1024, 2048),
    (128, 512): (0, 512, 1024),
}
# The kernels fix the value width at 512 (``check_support`` sets head_dim_v).
_CUDNN_FWD_D_V = 512
_CUDNN_FWD_DTYPES = (paddle.bfloat16, paddle.float16)


def _cudnn_fwd_module_available() -> bool:
    """Whether the bundled cuDNN Frontend actually ships the forward kernels.

    ``sparse_attention_forward`` landed upstream in cudnn-frontend PR #569 and
    is absent from every release the vendored fork has picked up so far, so the
    submodule bump lags this dispatch. Probe the module instead of assuming it,
    otherwise ``sparse_attn_forward_backend="cudnn"`` would raise
    ``ModuleNotFoundError`` mid-forward on an older frontend rather than falling
    back. ``find_spec`` avoids importing CuTe DSL just to answer the question.

    Cached because this runs per layer per step.
    """
    global _cudnn_fwd_module_cached
    if _cudnn_fwd_module_cached is None:
        from importlib.util import find_spec

        _patch_paddle_record_stream()
        _alias_cudnn_modules()
        try:
            _cudnn_fwd_module_cached = (
                find_spec(
                    "paddlefleet_ops.cudnn.deepseek_sparse_attention"
                    ".sparse_attention_forward"
                )
                is not None
            )
        except (ImportError, AttributeError, ValueError):
            _cudnn_fwd_module_cached = False
    return _cudnn_fwd_module_cached


def cudnn_sparse_attn_fwd_supported(h, d_qk, d_v, indexer_topk, dtype):
    """Whether ``cudnn_sparse_attn_fwd`` can serve this shape.

    Returns ``(ok, reason)``; ``reason`` is empty when ok, else a short string
    for the caller to log before falling back to FlashMLA.
    """
    try:
        from paddlefleet_ops import is_cudnn_frontend_available

        if not is_cudnn_frontend_available():
            return False, "cudnn frontend unavailable"
    except ImportError:
        return False, "paddlefleet_ops.is_cudnn_frontend_available missing"

    if not _cudnn_fwd_module_available():
        return False, (
            "cudnn frontend has no deepseek_sparse_attention."
            "sparse_attention_forward (needs the PR #569 kernels)"
        )

    # ``api.check_support`` gates on the major only, i.e. the whole SM100
    # family (10.0 / 10.3 / 10.7) is admitted. Pre-filter identically.
    if paddle.cuda.get_device_capability()[0] != 10:
        return (
            False,
            f"needs SM100 family, got {paddle.cuda.get_device_capability()}",
        )
    if dtype not in _CUDNN_FWD_DTYPES:
        return False, f"dtype {dtype} not in {_CUDNN_FWD_DTYPES}"
    if int(d_v) != _CUDNN_FWD_D_V:
        return False, f"d_v must be {_CUDNN_FWD_D_V}, got {d_v}"
    allowed = _CUDNN_FWD_VARIANTS.get((int(h), int(d_qk)))
    if allowed is None:
        return False, (
            f"(H, D_qk)=({h}, {d_qk}) not in {tuple(_CUDNN_FWD_VARIANTS)}"
        )
    if int(indexer_topk) not in allowed:
        return False, (
            f"indexer_topk={indexer_topk} not in {allowed} for (H, D_qk)="
            f"({h}, {d_qk})"
        )
    return True, ""


def cudnn_sparse_attn_fwd(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    indexer_topk: int = 0,
    d_v=None,
    topk_length=None,
    global_kv_idx_remap_fusion: bool = False,
):
    """Drop-in replacement for ``flash_mla_sparse_attn`` on supported shapes.

    Same arguments, same ``(out, lse, lse_indexer)`` return in the same
    layouts. ``cudnn_sparse_attn_fwd_supported`` must pass first; this function
    does not fall back on its own.

    Two differences from the FlashMLA path:

    * no ``topk`` width padding -- the kernel pads logical K to a multiple of
      64 internally, so the ``_get_topk_alignment`` / ``F.pad`` step is skipped;
    * ``out`` / ``lse`` / ``lse_indexer`` are freshly allocated per call rather
      than reused. ``execute()`` accepts caller-owned buffers, but the caller's
      autograd node saves ``out`` and ``lse`` for backward, so a reused buffer
      would be overwritten by the next layer's forward.
    """
    _patch_paddle_record_stream()
    _alias_cudnn_modules()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.sparse_attention_forward.api import (
        sparse_attention_forward_wrapper,
    )

    b, sq, h, d = q.shape
    _, skv, _ = kv.shape
    if d_v is None:
        d_v = d

    q_flat = q.reshape([b * sq, h, d])
    kv_flat = kv.reshape([b * skv, d])
    global_idxs = local_to_global_flat(
        topk_idxs, skv, fused=global_kv_idx_remap_fusion
    ).reshape([b * sq, topk_idxs.shape[-1]])
    topk_length_flat = (
        None
        if topk_length is None
        else topk_length.reshape([b * sq]).cast("int32")
    )

    res = sparse_attention_forward_wrapper(
        q_flat,
        kv_flat,
        global_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length_flat,
        softmax_scale=sm_scale,
        indexer_topk=int(indexer_topk),
    )

    # Logged **after** the kernel returned, so the line's presence means the
    # cuDNN kernel really executed -- not merely that this wrapper was entered.
    # Without it a silent fallback would make an A/B measurement compare the
    # baseline against itself. Emitted once per process.
    global _logged_cudnn_fwd_active
    if not _logged_cudnn_fwd_active:
        _logged_cudnn_fwd_active = True
        _logger.info(
            "sparse-attn forward: cuDNN Frontend DSA sparse prefill RAN "
            "(H=%d, D_qk=%d, D_v=%d, K=%d, indexer_topk=%d) -> out%s lse%s",
            h,
            d,
            d_v,
            topk_idxs.shape[-1],
            int(indexer_topk),
            tuple(res["out"].shape),
            tuple(res["lse"].shape),
        )

    lse_indexer = res["lse_indexer"]
    return (
        res["out"].reshape([b, sq, h, d_v]),
        res["lse"].reshape([b, sq, h]),
        None if lse_indexer is None else lse_indexer.reshape([b, sq, h]),
    )


def sparse_attn_fwd(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    indexer_topk: int = 0,
    d_v=None,
    topk_length=None,
    global_kv_idx_remap_fusion: bool = False,
    forward_backend: str = "flash_mla",
):
    """Pick the sparse-attention forward kernel.

    ``forward_backend`` is ``"flash_mla"`` (default, every shape) or
    ``"cudnn"`` (cudnn-frontend PR #569, only the ``_CUDNN_FWD_VARIANTS``
    shapes -- unsupported shapes warn once and fall back rather than failing a
    training run over a config typo).
    """
    if forward_backend not in ("flash_mla", "cudnn"):
        raise ValueError(
            f"sparse_attn_fwd forward_backend must be 'flash_mla' or 'cudnn', "
            f"got {forward_backend!r}"
        )

    if forward_backend == "cudnn":
        d_qk = q.shape[3]
        ok, reason = cudnn_sparse_attn_fwd_supported(
            q.shape[2],
            d_qk,
            d_qk if d_v is None else d_v,
            indexer_topk,
            q.dtype,
        )
        if ok:
            return cudnn_sparse_attn_fwd(
                q,
                kv,
                attn_sink,
                topk_idxs,
                sm_scale=sm_scale,
                indexer_topk=indexer_topk,
                d_v=d_v,
                topk_length=topk_length,
                global_kv_idx_remap_fusion=global_kv_idx_remap_fusion,
            )
        _warn_cudnn_fwd_fallback(reason)

    return flash_mla_sparse_attn(
        q,
        kv,
        attn_sink,
        topk_idxs,
        sm_scale=sm_scale,
        indexer_topk=indexer_topk,
        d_v=d_v,
        topk_length=topk_length,
        global_kv_idx_remap_fusion=global_kv_idx_remap_fusion,
    )


_warned_cudnn_fwd_fallback = set()


def _warn_cudnn_fwd_fallback(reason: str) -> None:
    """Warn once per distinct reason; this runs per layer per step."""
    if reason in _warned_cudnn_fwd_fallback:
        return
    _warned_cudnn_fwd_fallback.add(reason)
    import warnings

    warnings.warn(
        f"sparse_attn_forward_backend='cudnn' requested but unsupported "
        f"({reason}); falling back to FlashMLA.",
        stacklevel=3,
    )
