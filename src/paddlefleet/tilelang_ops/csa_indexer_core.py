import hashlib
import json
import os

import paddle


DEFAULT_INDEXER_BLOCK = 32
_DIGEST_COUNTERS = {}


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
    count = _DIGEST_COUNTERS.get(op, 0)
    if count >= _digest_limit():
        return
    _DIGEST_COUNTERS[op] = count + 1
    payload = {"op": op, "call": count, "tensors": [_tensor_digest(name, tensor) for name, tensor in tensors.items()]}
    print("[TileLangDigest] " + json.dumps(payload, sort_keys=True), flush=True)


def _get_csa_indexer_topk_fwd_interface():
    from .kernel.tilelang_csa_indexer_fwd import csa_indexer_topk_fwd_interface

    return csa_indexer_topk_fwd_interface


def _get_csa_indexer_bwd_interface():
    from .kernel.tilelang_csa_indexer_bwd import csa_indexer_bwd_interface

    return csa_indexer_bwd_interface


def _contiguous(tensor):
    return tensor.contiguous()


def _cast_int32(tensor):
    return tensor.cast("int32") if tensor.dtype != paddle.int32 else tensor


def _cast_float32(tensor):
    return tensor.cast("float32") if tensor.dtype != paddle.float32 else tensor


def _validate_indexer_inputs(index_q, index_k_comp, weights):
    if not isinstance(index_q, paddle.Tensor):
        raise TypeError(f"index_q must be a paddle.Tensor, got {type(index_q)!r}")
    if not isinstance(index_k_comp, paddle.Tensor):
        raise TypeError(f"index_k_comp must be a paddle.Tensor, got {type(index_k_comp)!r}")
    if not isinstance(weights, paddle.Tensor):
        raise TypeError(f"weights must be a paddle.Tensor, got {type(weights)!r}")
    if len(index_q.shape) != 4:
        raise ValueError(f"index_q must have shape [B, S, H_i, D_i], got {tuple(index_q.shape)}")
    if len(index_k_comp.shape) != 3:
        raise ValueError(f"index_k_comp must have shape [B, S_comp, D_i], got {tuple(index_k_comp.shape)}")
    if len(weights.shape) != 3:
        raise ValueError(f"weights must have shape [B, S, H_i], got {tuple(weights.shape)}")

    batch, seq_len, heads, dim = tuple(index_q.shape)
    batch_k, _, dim_k = tuple(index_k_comp.shape)
    batch_w, seq_len_w, heads_w = tuple(weights.shape)
    if batch != batch_k or batch != batch_w:
        raise ValueError(
            f"batch mismatch: index_q={tuple(index_q.shape)}, index_k_comp={tuple(index_k_comp.shape)}, weights={tuple(weights.shape)}"
        )
    if seq_len != seq_len_w or heads != heads_w or dim != dim_k:
        raise ValueError(
            f"shape mismatch: index_q={tuple(index_q.shape)}, index_k_comp={tuple(index_k_comp.shape)}, weights={tuple(weights.shape)}"
        )


def _validate_topk_and_grad(index_q, topk_indices, grad_scores):
    if not isinstance(topk_indices, paddle.Tensor):
        raise TypeError(f"topk_indices must be a paddle.Tensor, got {type(topk_indices)!r}")
    if not isinstance(grad_scores, paddle.Tensor):
        raise TypeError(f"grad_scores must be a paddle.Tensor, got {type(grad_scores)!r}")
    if len(topk_indices.shape) != 3:
        raise ValueError(f"topk_indices must have shape [B, S, topk], got {tuple(topk_indices.shape)}")
    if len(grad_scores.shape) != 3:
        raise ValueError(f"grad_scores must have shape [B, S, topk], got {tuple(grad_scores.shape)}")
    batch, seq_len, _, _ = tuple(index_q.shape)
    if tuple(topk_indices.shape) != tuple(grad_scores.shape):
        raise ValueError(f"topk_indices shape {tuple(topk_indices.shape)} must match grad_scores shape {tuple(grad_scores.shape)}")
    if tuple(topk_indices.shape)[0] != batch or tuple(topk_indices.shape)[1] != seq_len:
        raise ValueError(f"topk_indices shape {tuple(topk_indices.shape)} is incompatible with index_q shape {tuple(index_q.shape)}")
    if tuple(topk_indices.shape)[-1] <= 0:
        raise ValueError("topk_indices last dimension must be positive")


def _prepare_forward_inputs(index_q, index_k_comp, weights, topk_effective):
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    if int(topk_effective) <= 0:
        raise ValueError(f"topk_effective must be positive, got {topk_effective}")
    return _contiguous(index_q), _contiguous(index_k_comp), _contiguous(_cast_float32(weights)), int(topk_effective)


def _prepare_backward_inputs(index_q, weights, index_k_comp, topk_indices, grad_scores):
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    _validate_topk_and_grad(index_q, topk_indices, grad_scores)
    topk_indices = _contiguous(_cast_int32(topk_indices))
    grad_scores = _contiguous(_cast_float32(grad_scores))
    grad_scores = paddle.where(topk_indices >= 0, grad_scores, paddle.zeros_like(grad_scores))
    return (
        _contiguous(index_q),
        _contiguous(_cast_float32(weights)),
        _contiguous(index_k_comp),
        topk_indices,
        _contiguous(grad_scores),
    )


def tilelang_csa_compressed_indexer_topk_paddle(
    index_q,
    index_k_comp,
    weights,
    ratio: int,
    topk_effective: int,
    block_K: int = DEFAULT_INDEXER_BLOCK,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Paddle entry for V4 CSA compressed indexer forward.

    Args:
        index_q: [B, S, H_i, D_i] indexer queries.
        index_k_comp: [B, S_comp, D_i] compressed indexer keys.
        weights: [B, S, H_i] per-head weights for score aggregation.
        ratio: compression ratio (e.g. 4). Causal range: [0, (t+1)//ratio).
        topk_effective: number of top-k entries to select per query position.
            - Phase 2 (dsa_indexer_use_sparse_loss=False): set to n_compressed
              = floor(S / ratio) for full-candidate selection.
            - Phase 3 (dsa_indexer_use_sparse_loss=True): set to
              min(index_topk, n_compressed), typically 512.
        block_K: tile size for streaming over compressed keys (default 32).

    Returns:
        topk_indices: [B, S, topk_effective] int32, invalid slots are -1.
        topk_scores: [B, S, topk_effective] fp32 top-k softmax probabilities.
    """
    index_q, index_k_comp, weights, topk_effective = _prepare_forward_inputs(
        index_q,
        index_k_comp,
        weights,
        topk_effective,
    )
    csa_indexer_topk_fwd_interface = _get_csa_indexer_topk_fwd_interface()
    topk_indices, topk_scores = csa_indexer_topk_fwd_interface(
        index_q,
        index_k_comp,
        weights,
        ratio=int(ratio),
        topk_effective=topk_effective,
        block_K=int(block_K),
        num_stages=int(num_stages),
        num_threads=int(num_threads),
    )
    expected_shape = (tuple(index_q.shape)[0], tuple(index_q.shape)[1], topk_effective)
    if tuple(topk_indices.shape) != expected_shape or tuple(topk_scores.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected CSA indexer forward output shapes: indices={tuple(topk_indices.shape)}, scores={tuple(topk_scores.shape)}, expected={expected_shape}"
        )
    if not isinstance(topk_indices, paddle.Tensor) or not isinstance(topk_scores, paddle.Tensor):
        raise RuntimeError(
            "TileLang must return Paddle tensors. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    return topk_indices, topk_scores


def tilelang_csa_compressed_indexer_bwd_paddle(
    index_q,
    weights,
    index_k_comp,
    topk_indices,
    grad_scores,
    block_I: int = DEFAULT_INDEXER_BLOCK,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Paddle entry for V4 CSA compressed indexer backward.

    Computes gradients for IndexQ, Weights, and IndexKComp given the selected
    top-k indices and the gradient of the loss w.r.t. the selected scores.

    Args:
        index_q: [B, S, H_i, D_i] indexer queries (same as forward input).
        weights: [B, S, H_i] per-head weights (same as forward input).
        index_k_comp: [B, S_comp, D_i] compressed indexer keys.
        topk_indices: [B, S, topk_effective] int32, from forward output.
            Invalid slots must be -1 (they are masked to zero gradient).
        grad_scores: [B, S, topk_effective] fp32, gradient of loss w.r.t.
            the selected indexer scores (typically ``(probs - target) * coeff``).

    Returns:
        grad_q: [B, S, H_i, D_i] gradient for indexer queries.
        grad_weights: [B, S, H_i] gradient for per-head weights.
        grad_k_comp: [B, S_comp, D_i] gradient for compressed indexer keys.
    """
    index_q, weights, index_k_comp, topk_indices, grad_scores = _prepare_backward_inputs(
        index_q,
        weights,
        index_k_comp,
        topk_indices,
        grad_scores,
    )
    csa_indexer_bwd_interface = _get_csa_indexer_bwd_interface()
    grad_q, grad_weights, grad_k_comp = csa_indexer_bwd_interface(
        index_q,
        weights,
        index_k_comp,
        topk_indices,
        grad_scores,
        block_I=int(block_I),
        num_stages=int(num_stages),
        num_threads=int(num_threads),
    )
    if tuple(grad_q.shape) != tuple(index_q.shape) or tuple(grad_weights.shape) != tuple(weights.shape) or tuple(grad_k_comp.shape) != tuple(index_k_comp.shape):
        raise RuntimeError(
            "unexpected CSA indexer backward output shapes: "
            f"grad_q={tuple(grad_q.shape)}, grad_weights={tuple(grad_weights.shape)}, grad_k_comp={tuple(grad_k_comp.shape)}"
        )
    if not isinstance(grad_q, paddle.Tensor) or not isinstance(grad_weights, paddle.Tensor) or not isinstance(grad_k_comp, paddle.Tensor):
        raise RuntimeError(
            "TileLang must return Paddle tensors. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    return grad_q, grad_weights, grad_k_comp
