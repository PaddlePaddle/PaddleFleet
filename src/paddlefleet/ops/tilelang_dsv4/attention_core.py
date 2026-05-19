import os
import time

import paddle
import torch
import torch.nn.functional as F

from .compat import paddle_tilelang_compat_guard
from .kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd
from .kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd
from .kernel.tilelang_sparse_mla import _prepare_inputs_paddle, sparse_attn_tilelang_paddle


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
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def _topk_invalid_ratio(topk_idxs):
    if topk_idxs.numel() == 0:
        return 0.0
    return float((topk_idxs == -1).sum().item()) / float(topk_idxs.numel())


def _ensure_torch_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")


def _prepare_topk_idxs(topk_idxs, *, topk_pad_to=DEFAULT_TOPK_PAD_TO):
    _ensure_torch_tensor("topk_idxs", topk_idxs)
    if topk_idxs.dim() != 3:
        raise ValueError(f"topk_idxs must have shape [B, S, topk], got {tuple(topk_idxs.shape)}")
    if topk_pad_to <= 0:
        raise ValueError(f"topk_pad_to must be positive, got {topk_pad_to}")

    if topk_idxs.dtype != torch.int32:
        topk_idxs = topk_idxs.to(torch.int32)
    if not topk_idxs.is_contiguous():
        topk_idxs = topk_idxs.contiguous()

    topk = topk_idxs.shape[-1]
    padded_topk = (topk + topk_pad_to - 1) // topk_pad_to * topk_pad_to
    if padded_topk != topk:
        topk_idxs = F.pad(topk_idxs, (0, padded_topk - topk), value=-1)
    return topk_idxs.contiguous()


def _prepare_attn_sink(attn_sink):
    _ensure_torch_tensor("attn_sink", attn_sink)
    if attn_sink.dim() != 1:
        raise ValueError(f"attn_sink must have shape [H], got {tuple(attn_sink.shape)}")
    if attn_sink.dtype != torch.float32:
        attn_sink = attn_sink.float()
    return attn_sink.contiguous()


def _prepare_inputs(q, kv, attn_sink, topk_idxs, *, topk_pad_to=DEFAULT_TOPK_PAD_TO):
    _ensure_torch_tensor("q", q)
    _ensure_torch_tensor("kv", kv)
    if q.dim() != 4:
        raise ValueError(f"q must have shape [B, S, H, D], got {tuple(q.shape)}")
    if kv.dim() != 3:
        raise ValueError(f"kv must have shape [B, S_kv, D], got {tuple(kv.shape)}")
    if q.shape[0] != kv.shape[0] or q.shape[-1] != kv.shape[-1]:
        raise ValueError(f"q shape {tuple(q.shape)} is incompatible with kv shape {tuple(kv.shape)}")

    return (
        q.contiguous(),
        kv.contiguous(),
        _prepare_attn_sink(attn_sink),
        _prepare_topk_idxs(topk_idxs, topk_pad_to=topk_pad_to),
    )


def sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """
    Torch reference implementation for DSv4 sparse attention.

    Args:
        q: [B, S, H, D]
        kv: [B, S_kv, D]
        attn_sink: [H]
        topk_idxs: [B, S, topk], -1 means masked entry
        sm_scale: optional softmax scale
    Returns:
        o: [B, S, H, D]
    """
    q_dtype = q.dtype
    q = q.float()
    kv = kv.float()

    b, m, h, d = q.shape
    k_len = kv.shape[1]

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    topk_idxs = _prepare_topk_idxs(topk_idxs)
    if (topk_idxs >= k_len).any():
        raise ValueError(f"topk_idxs contains index >= kv length {k_len}")

    mask = topk_idxs != -1
    safe_idxs = topk_idxs.masked_fill(~mask, 0)

    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1)
    kv_gathered = kv[batch_idx, safe_idxs.long()]

    scores = torch.einsum("bmhd,bmkd->bmhk", q, kv_gathered) * sm_scale
    mask_expanded = mask.unsqueeze(2).expand(-1, -1, h, -1)
    scores = scores.masked_fill(~mask_expanded, float("-inf"))

    scores = scores.to(torch.float32)
    scores_max = scores.max(dim=-1).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))

    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, kv_gathered.to(torch.float32))
    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(_prepare_attn_sink(attn_sink).view(1, 1, h) - scores_max)
    denominator = sum_exp + sink_term

    return (numerator / denominator.unsqueeze(-1)).to(q_dtype)


def dense_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Dense reference implementation using a mask generated from topk_idxs."""
    q_dtype = q.dtype
    b, m, h, d = q.shape
    n = kv.shape[1]

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    topk_idxs = _prepare_topk_idxs(topk_idxs)
    attn_mask = torch.zeros(b, m, n, device=q.device, dtype=torch.bool)

    _, _, topk = topk_idxs.shape
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, m, topk)
    seq_idx = torch.arange(m, device=q.device).view(1, m, 1).expand(b, m, topk)
    valid_mask = topk_idxs != -1

    attn_mask[batch_idx[valid_mask], seq_idx[valid_mask], topk_idxs[valid_mask].long()] = True

    scores = torch.einsum("bmhd,bnd->bmhn", q.float(), kv.float()).to(torch.float32) * sm_scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(2).expand(-1, -1, h, -1), float("-inf"))

    scores_max = scores.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max)

    numerator = torch.einsum("bmhn,bnd->bmhd", exp_scores, kv.float())
    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(_prepare_attn_sink(attn_sink).view(1, 1, h) - scores_max.squeeze(-1))
    denominator = sum_exp + sink_term

    return (numerator / denominator.unsqueeze(-1)).to(q_dtype)


def _compat_backward_kernel(query, kv_full, attn_sink, output, grad_output, topk_idxs, lse, softmax_scale):
    with paddle_tilelang_compat_guard():
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
        return (
            dq.reshape(query.shape),
            dkv.reshape(kv_full.shape),
            d_attn_sink.reshape(attn_sink.shape).cast(ctx.attn_sink_dtype),
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
