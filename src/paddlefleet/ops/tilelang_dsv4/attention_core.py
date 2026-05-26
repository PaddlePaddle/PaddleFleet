import hashlib
import json
import os
import time

import paddle

from .compat import enable_tilelang_paddle_compat_before_import


DEFAULT_TOPK_PAD_TO = 64
_PROFILE_COUNTERS = {}


def _profile_enabled():
    return os.getenv("DSV4_TILELANG_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def _profile_limit():
    try:
        return int(os.getenv("DSV4_TILELANG_PROFILE_STEPS", "20"))
    except ValueError:
        return 20


def _profile_sync():
    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
    except Exception:
        pass


def _profile_time_ms(fn):
    _profile_sync()
    start = time.perf_counter()
    result = fn()
    _profile_sync()
    return result, (time.perf_counter() - start) * 1000.0


def _profile_should_log(key):
    if not _profile_enabled():
        return False
    count = _PROFILE_COUNTERS.get(key, 0)
    if count >= _profile_limit():
        return False
    _PROFILE_COUNTERS[key] = count + 1
    return True


def _profile_log(phase, elapsed_ms=None, **kwargs):
    fields = [f"phase={phase}"]
    if elapsed_ms is not None:
        fields.append(f"elapsed_ms={elapsed_ms:.3f}")
    fields.extend(f"{key}={value}" for key, value in kwargs.items())
    print("[TileLangProfile] " + " ".join(fields), flush=True)


def _digest_enabled():
    return os.getenv("DSV4_TILELANG_DIGEST", "0").lower() in {"1", "true", "yes", "on"}


def _digest_limit():
    try:
        return int(os.getenv("DSV4_TILELANG_DIGEST_LIMIT", "40"))
    except ValueError:
        return 40


def _tensor_digest(name, tensor):
    tensor_f32 = tensor.detach().cast("float32").cpu()
    array = tensor_f32.numpy()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "sum_f32": float(array.sum()),
        "max_abs_f32": float(abs(array).max()) if array.size else 0.0,
    }


def _digest_log(op, **tensors):
    if not _digest_enabled():
        return
    count = _PROFILE_COUNTERS.get(f"digest:{op}", 0)
    if count >= _digest_limit():
        return
    _PROFILE_COUNTERS[f"digest:{op}"] = count + 1
    payload = {"op": op, "call": count, "tensors": [_tensor_digest(name, tensor) for name, tensor in tensors.items()]}
    print("[TileLangDigest] " + json.dumps(payload, sort_keys=True), flush=True)


def _topk_invalid_ratio(topk_idxs):
    if topk_idxs.numel() == 0:
        return 0.0
    return float((topk_idxs == -1).sum().item()) / float(topk_idxs.numel())


def _ensure_paddle_tensor(name, tensor):
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"{name} must be a paddle.Tensor, got {type(tensor)!r}")


def _contiguous(tensor):
    if hasattr(tensor, "contiguous"):
        return tensor.contiguous()
    return tensor


def _prepare_topk_idxs(topk_idxs, *, topk_pad_to=DEFAULT_TOPK_PAD_TO):
    _ensure_paddle_tensor("topk_idxs", topk_idxs)
    if len(topk_idxs.shape) != 3:
        raise ValueError(f"topk_idxs must have shape [B, S, topk], got {tuple(topk_idxs.shape)}")
    if topk_pad_to <= 0:
        raise ValueError(f"topk_pad_to must be positive, got {topk_pad_to}")

    if topk_idxs.dtype != paddle.int32:
        topk_idxs = topk_idxs.cast("int32")

    topk = topk_idxs.shape[-1]
    padded_topk = (topk + topk_pad_to - 1) // topk_pad_to * topk_pad_to
    if padded_topk != topk:
        pad = paddle.full(
            [topk_idxs.shape[0], topk_idxs.shape[1], padded_topk - topk],
            -1,
            dtype="int32",
        )
        topk_idxs = paddle.concat([topk_idxs, pad], axis=-1)
    return _contiguous(topk_idxs)


def _prepare_attn_sink(attn_sink):
    _ensure_paddle_tensor("attn_sink", attn_sink)
    if len(attn_sink.shape) != 1:
        raise ValueError(f"attn_sink must have shape [H], got {tuple(attn_sink.shape)}")
    if attn_sink.dtype != paddle.float32:
        attn_sink = attn_sink.cast("float32")
    return _contiguous(attn_sink)


def _prepare_inputs(q, kv, attn_sink, topk_idxs, *, topk_pad_to=DEFAULT_TOPK_PAD_TO):
    _ensure_paddle_tensor("q", q)
    _ensure_paddle_tensor("kv", kv)
    if len(q.shape) != 4:
        raise ValueError(f"q must have shape [B, S, H, D], got {tuple(q.shape)}")
    if len(kv.shape) != 3:
        raise ValueError(f"kv must have shape [B, S_kv, D], got {tuple(kv.shape)}")
    if q.shape[0] != kv.shape[0] or q.shape[-1] != kv.shape[-1]:
        raise ValueError(f"q shape {tuple(q.shape)} is incompatible with kv shape {tuple(kv.shape)}")

    topk_idxs = _prepare_topk_idxs(topk_idxs, topk_pad_to=topk_pad_to)
    if topk_idxs.shape[0] != q.shape[0] or topk_idxs.shape[1] != q.shape[1]:
        raise ValueError(f"topk_idxs shape {tuple(topk_idxs.shape)} is incompatible with q shape {tuple(q.shape)}")

    attn_sink = _prepare_attn_sink(attn_sink)
    if attn_sink.shape[0] != q.shape[2]:
        raise ValueError(f"attn_sink shape {tuple(attn_sink.shape)} is incompatible with q shape {tuple(q.shape)}")

    return (
        _contiguous(q),
        _contiguous(kv),
        attn_sink,
        topk_idxs,
    )


def sparse_attn_paddle(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Paddle reference implementation for DSv4 sparse attention."""
    q, kv, attn_sink, topk_idxs = _prepare_inputs(q, kv, attn_sink, topk_idxs)
    q_dtype = q.dtype
    q = q.cast("float32")
    kv = kv.cast("float32")

    b, m, h, d = q.shape
    k_len = kv.shape[1]

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    if bool(paddle.any((topk_idxs >= k_len) & (topk_idxs != -1)).item()):
        raise ValueError(f"topk_idxs contains index >= kv length {k_len}")

    mask = topk_idxs != -1
    safe_idxs = paddle.where(mask, topk_idxs, paddle.zeros_like(topk_idxs)).cast("int64")
    batch_idx = paddle.arange(b, dtype="int64").reshape([b, 1, 1])
    batch_idx = paddle.expand(batch_idx, [b, m, safe_idxs.shape[-1]])
    kv_gathered = kv[batch_idx, safe_idxs]

    scores = paddle.einsum("bmhd,bmkd->bmhk", q, kv_gathered) * sm_scale
    mask_expanded = paddle.expand(mask.unsqueeze(2), [b, m, h, mask.shape[-1]])
    scores = paddle.where(mask_expanded, scores, paddle.full_like(scores, float("-inf")))

    scores = scores.cast("float32")
    scores_max = paddle.maximum(
        paddle.max(scores, axis=-1),
        paddle.full([b, m, h], -1e30, dtype="float32"),
    )
    exp_scores = paddle.exp(scores - scores_max.unsqueeze(-1))

    numerator = paddle.einsum("bmhk,bmkd->bmhd", exp_scores, kv_gathered.cast("float32"))
    sum_exp = paddle.sum(exp_scores, axis=-1)
    sink_term = paddle.exp(_prepare_attn_sink(attn_sink).reshape([1, 1, h]) - scores_max)
    denominator = sum_exp + sink_term

    return (numerator / denominator.unsqueeze(-1)).cast(q_dtype)


def _get_sparse_mla_bwd():
    enable_tilelang_paddle_compat_before_import()
    from .kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd

    return sparse_mla_bwd


def _get_sparse_attn_tilelang_paddle():
    enable_tilelang_paddle_compat_before_import()
    from .kernel.tilelang_sparse_mla import _prepare_inputs_paddle, sparse_attn_tilelang_paddle

    return _prepare_inputs_paddle, sparse_attn_tilelang_paddle


def _compat_backward_kernel(query, kv_full, attn_sink, output, grad_output, topk_idxs, lse, softmax_scale):
    sparse_mla_bwd = _get_sparse_mla_bwd()
    return sparse_mla_bwd.sparse_mqa_bwd_interface(
        query,
        kv_full,
        attn_sink,
        output,
        grad_output,
        topk_idxs,
        lse,
        sm_scale=softmax_scale,
    )


class TileLangCompressedSparseAttentionPaddleCompatPyLayer(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, query, kv_full, attn_sink, topk_idxs, softmax_scale, topk_pad_to):
        if int(topk_pad_to) != DEFAULT_TOPK_PAD_TO:
            raise ValueError(f"topk_pad_to must be {DEFAULT_TOPK_PAD_TO}, got {topk_pad_to}")
        b, sq, np_heads, hn = query.shape
        ctx.query_shape = (b, sq, np_heads, hn)
        ctx.softmax_scale = float(softmax_scale)
        ctx.topk_pad_to = int(topk_pad_to)
        ctx.attn_sink_dtype = attn_sink.dtype
        _prepare_inputs_paddle, sparse_attn_tilelang_paddle = _get_sparse_attn_tilelang_paddle()
        profile = _profile_should_log("paddle_compat_forward")
        if profile:
            (query, kv_full, attn_sink, topk_idxs), elapsed_ms = _profile_time_ms(
                lambda: _prepare_inputs_paddle(
                    query,
                    kv_full,
                    attn_sink,
                    topk_idxs,
                    topk_pad_to=ctx.topk_pad_to,
                )
            )
            _profile_log("compat_forward_prepare", elapsed_ms, q_shape=tuple(query.shape), kv_shape=tuple(kv_full.shape), topk=topk_idxs.shape[-1])
            (output, lse), elapsed_ms = _profile_time_ms(
                lambda: sparse_attn_tilelang_paddle(
                    query,
                    kv_full,
                    attn_sink,
                    topk_idxs,
                    sm_scale=ctx.softmax_scale,
                    topk_pad_to=ctx.topk_pad_to,
                )
            )
            _profile_log("compat_forward_kernel", elapsed_ms)
        else:
            query, kv_full, attn_sink, topk_idxs = _prepare_inputs_paddle(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                topk_pad_to=ctx.topk_pad_to,
            )
            output, lse = sparse_attn_tilelang_paddle(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                sm_scale=ctx.softmax_scale,
                topk_pad_to=ctx.topk_pad_to,
            )
        ctx.save_for_backward(query, kv_full, attn_sink, topk_idxs, output, lse)
        return output.reshape([b, sq, np_heads * hn])

    @staticmethod
    def backward(ctx, grad_output):
        query, kv_full, attn_sink, topk_idxs, output, lse = ctx.saved_tensor()
        b, sq, np_heads, hn = ctx.query_shape
        grad_output = grad_output.reshape([b, sq, np_heads, hn])
        profile = _profile_should_log("paddle_compat_backward")
        if profile:
            (dq, dkv, d_attn_sink), elapsed_ms = _profile_time_ms(
                lambda: _compat_backward_kernel(
                    query,
                    kv_full,
                    attn_sink,
                    output,
                    grad_output,
                    topk_idxs,
                    lse,
                    ctx.softmax_scale,
                )
            )
            _profile_log("compat_backward_total", elapsed_ms)
        else:
            dq, dkv, d_attn_sink = _compat_backward_kernel(
                query,
                kv_full,
                attn_sink,
                output,
                grad_output,
                topk_idxs,
                lse,
                ctx.softmax_scale,
            )
        dq = dq.reshape(query.shape)
        dkv = dkv.reshape(kv_full.shape)
        d_attn_sink = d_attn_sink.reshape(attn_sink.shape).cast(ctx.attn_sink_dtype)
        _digest_log("sparse_mla_bwd", dq=dq, dkv=dkv, d_attn_sink=d_attn_sink)
        return (
            dq,
            dkv,
            d_attn_sink,
            None,
        )


def tilelang_compressed_sparse_attn_paddle_compat_autograd(
    query,
    kv_full,
    attn_sink,
    topk_idxs,
    softmax_scale,
    topk_pad_to=DEFAULT_TOPK_PAD_TO,
):
    return TileLangCompressedSparseAttentionPaddleCompatPyLayer.apply(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_pad_to,
    )
