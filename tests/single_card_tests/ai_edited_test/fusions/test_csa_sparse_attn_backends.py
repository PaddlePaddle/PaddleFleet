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

"""Unit test: verify tilelang vs cudnn backends for CSA sparse attention.

Tests various shapes (small/large batch, seq, heads, topk) and checks:
  1. Both backends run without error
  2. Numerical agreement between backends (cosine sim, max abs diff)
  3. Backward gradients agree
"""

import paddle

paddle.set_device("gpu:0")


def cosine_sim(a, b):
    a_f = a.flatten().cast("float32")
    b_f = b.flatten().cast("float32")
    return float(
        paddle.nn.functional.cosine_similarity(
            a_f.unsqueeze(0), b_f.unsqueeze(0)
        )
    )


def max_abs_diff(a, b):
    return float((a.cast("float32") - b.cast("float32")).abs().max())


def make_inputs(B, S, S_kv, H, D, topk, dtype="bfloat16"):
    """MQA layout: q=[B,S,H,D], kv=[B,S_kv,D] (single KV head, compressed)."""
    q = paddle.randn([B, S, H, D]).cast(dtype)
    q.stop_gradient = False
    # kv is MQA-style: [B, S_kv, D] where D == head_dim
    kv = paddle.randn([B, S_kv, D]).cast(dtype)
    kv.stop_gradient = False
    attn_sink = paddle.randn([H]).cast("float32") * 0.1
    attn_sink.stop_gradient = False
    # Random topk indices in [0, S_kv)
    topk_idxs = paddle.randint(0, S_kv, [B, S, topk]).cast("int32")
    softmax_scale = 1.0 / (D**0.5)
    return q, kv, attn_sink, topk_idxs, softmax_scale


def run_forward_backward(q, kv, attn_sink, topk_idxs, softmax_scale, backend):
    from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

    # Clone inputs for independent gradient computation
    q_c = q.detach().clone()
    q_c.stop_gradient = False
    kv_c = kv.detach().clone()
    kv_c.stop_gradient = False
    attn_sink_c = attn_sink.detach().clone()
    attn_sink_c.stop_gradient = False

    out = csa_sparse_attn(
        q_c, kv_c, attn_sink_c, topk_idxs, softmax_scale, backend=backend
    )
    loss = out.sum()
    loss.backward()

    return out, q_c.grad, kv_c.grad, attn_sink_c.grad


def test_single_shape(B, S, S_kv, H, D, topk, label=""):
    print(f"\n{'=' * 60}")
    print(
        f"Test: {label}  B={B}, S={S}, S_kv={S_kv}, H={H}, D={D}, topk={topk}"
    )
    print(f"{'=' * 60}")

    q, kv, attn_sink, topk_idxs, softmax_scale = make_inputs(
        B, S, S_kv, H, D, topk
    )

    # --- tilelang ---
    try:
        out_tl, dq_tl, dkv_tl, dsink_tl = run_forward_backward(
            q, kv, attn_sink, topk_idxs, softmax_scale, backend="tilelang"
        )
        print(f"  tilelang forward OK, out shape={list(out_tl.shape)}")
    except Exception as e:
        print(f"  tilelang FAILED: {e}")
        return False

    # --- cudnn ---
    try:
        out_cu, dq_cu, dkv_cu, dsink_cu = run_forward_backward(
            q, kv, attn_sink, topk_idxs, softmax_scale, backend="cudnn"
        )
        print(f"  cudnn forward OK, out shape={list(out_cu.shape)}")
    except Exception as e:
        print(f"  cudnn FAILED: {e}")
        return False

    # --- Compare forward ---
    cos_out = cosine_sim(out_tl, out_cu)
    mad_out = max_abs_diff(out_tl, out_cu)
    print(f"  Forward: cosine={cos_out:.6f}, max_abs_diff={mad_out:.6f}")

    # --- Compare backward ---
    cos_dq = cosine_sim(dq_tl, dq_cu)
    mad_dq = max_abs_diff(dq_tl, dq_cu)
    print(f"  dq:      cosine={cos_dq:.6f}, max_abs_diff={mad_dq:.6f}")

    cos_dkv = cosine_sim(dkv_tl, dkv_cu)
    mad_dkv = max_abs_diff(dkv_tl, dkv_cu)
    print(f"  dkv:     cosine={cos_dkv:.6f}, max_abs_diff={mad_dkv:.6f}")

    if dsink_tl is not None and dsink_cu is not None:
        cos_dsink = cosine_sim(dsink_tl, dsink_cu)
        mad_dsink = max_abs_diff(dsink_tl, dsink_cu)
        print(
            f"  d_sink:  cosine={cos_dsink:.6f}, max_abs_diff={mad_dsink:.6f}"
        )

    # Thresholds for bf16
    passed = cos_out > 0.99 and cos_dq > 0.95
    print(f"  PASS: {passed}")
    return passed


if __name__ == "__main__":
    # DSv4 uses H=64 heads, D=512 (head_dim for kv is H*D compressed)
    # topk alignment: SM100 requires multiples of 64
    test_cases = [
        # (B, S, S_kv, H, D, topk, label)
        # Basic shapes
        (1, 128, 256, 64, 512, 64, "small-basic"),
        (2, 128, 256, 64, 512, 64, "batch2"),
        (4, 128, 256, 64, 512, 64, "batch4"),
        # Varying sequence lengths
        (1, 64, 128, 64, 512, 64, "tiny-seq"),
        (1, 256, 512, 64, 512, 128, "medium-seq"),
        (1, 512, 1024, 64, 512, 128, "large-seq"),
        # Edge: single query token
        (1, 1, 256, 64, 512, 64, "single-token"),
        (2, 1, 128, 64, 512, 64, "single-token-batch2"),
        # Large topk (non-aligned to 64)
        (2, 256, 512, 64, 512, 192, "large-topk-192"),
        (1, 128, 512, 64, 512, 256, "large-topk-256"),
        # Stress: large batch + large seq
        (8, 64, 128, 64, 512, 64, "batch8"),
        # Minimal: smallest valid config
        (1, 1, 64, 64, 512, 64, "minimal"),
    ]

    results = []
    for B, S, S_kv, H, D, topk, label in test_cases:
        try:
            passed = test_single_shape(B, S, S_kv, H, D, topk, label)
            results.append((label, passed))
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append((label, False))
        # Free GPU memory
        paddle.device.cuda.empty_cache()

    print(f"\n\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for label, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
    total_pass = sum(1 for _, p in results if p)
    print(f"\n  {total_pass}/{len(results)} passed")
