import paddle

from ..compat import paddle_tilelang_compat_guard
from . import tilelang_sparse_mla_fwd as sparse_mla_fwd


def _prepare_inputs_paddle(q, kv, attn_sink, topk_idxs, topk_pad_to=64):
    if len(q.shape) != 4:
        raise ValueError(f"q must have shape [B, S, H, D], got {q.shape}")
    if len(kv.shape) != 3:
        raise ValueError(f"kv must have shape [B, S_kv, D], got {kv.shape}")
    if len(topk_idxs.shape) != 3:
        raise ValueError(f"topk_idxs must have shape [B, S, topk], got {topk_idxs.shape}")
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

    if attn_sink.dtype != paddle.float32:
        attn_sink = attn_sink.cast("float32")
    return q, kv, attn_sink, topk_idxs


def sparse_attn_tilelang_paddle(q, kv, attn_sink, topk_idxs, sm_scale=None, topk_pad_to=64):
    q, kv, attn_sink, topk_idxs = _prepare_inputs_paddle(q, kv, attn_sink, topk_idxs, topk_pad_to=topk_pad_to)
    with paddle_tilelang_compat_guard():
        out, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)
    if not isinstance(out, paddle.Tensor) or not isinstance(lse, paddle.Tensor):
        raise RuntimeError(
            "attention_paddle_compat requires TileLang to return Paddle tensors, "
            f"but got output={type(out)!r}, lse={type(lse)!r}. "
            "Paddle torch proxy did not take over the already-imported TileLang runtime; "
            "refusing to fall back to DLPack bridge."
        )
    return out, lse
