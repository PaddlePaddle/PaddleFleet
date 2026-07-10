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

"""Autograd-contract tests for the HySparse MQA ``PyLayer`` wrappers.

These validate two properties the low-level fwd/bwd interface tests do not:

* the ``PyLayer`` backward honours the Paddle contract -- it returns a gradient
  for exactly the Tensor inputs whose ``stop_gradient`` is False and ``None``
  for every frozen input (and for the non-differentiable ``indices`` /
  ``valid_range`` slots), so partial-freeze training graphs do not error and
  do not leak gradient into detached tensors;
* the backward accepts a **non-contiguous** upstream gradient (the kernel
  interfaces assert ``do.is_contiguous()`` -- the wrapper must materialise it).
"""

import unittest

import numpy as np
import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

from paddlefleet.tilelang_ops.hysparse.differentiable_ops import (
    block_score_mha_attention,
    block_score_mqa_attention,
    block_sparse_mqa_attention,
    hysparse_mqa_attention,
)
from paddlefleet.tilelang_ops.hysparse.reference import (
    make_causal_valid_range,
    ref_block_score_attn_mha,
    ref_block_score_attn_mqa,
)


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _maxerr(a, b):
    a = a.astype("float32").numpy()
    b = b.astype("float32").numpy()
    return float(np.abs(a - b).max())


def _grad_leaves(*tensors):
    """Independent float32 leaf copies (stop_gradient=False) for reference
    gradients. ``detach`` first so each is a graph leaf whose ``.grad`` is
    retained even when the source tensor already tracks gradient."""
    out = []
    for t in tensors:
        tf = t.detach().astype("float32")
        tf.stop_gradient = False
        out.append(tf)
    return out


def _leaves(b, s, h, d, seed=0, freeze=()):
    """q [B,S,H,D] and shared K/V [B,S,D]; ``freeze`` names get
    stop_gradient=True."""
    paddle.seed(seed)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, d], dtype="bfloat16")
    v = paddle.randn([b, s, d], dtype="bfloat16")
    tensors = {"q": q, "k": k, "v": v}
    for name, t in tensors.items():
        t.stop_gradient = name in freeze
    return q, k, v


def _leaves_mha(b, s, h, d, dv=None, seed=0, freeze=()):
    """q [B,S,H,D], per-head k [B,S,H,D], v [B,S,H,Dv]; ``freeze`` -> frozen."""
    paddle.seed(seed)
    dv = d if dv is None else dv
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, h, d], dtype="bfloat16")
    v = paddle.randn([b, s, h, dv], dtype="bfloat16")
    for name, t in (("q", q), ("k", k), ("v", v)):
        t.stop_gradient = name in freeze
    return q, k, v


OUT_ATOL = 5e-2
GRAD_ATOL = 8e-2


class TestAutogradStopGradient(unittest.TestCase):
    """PyLayer backward must return None for frozen inputs, Tensor otherwise."""

    B, S, H, D = 2, 128, 4, 64

    def _score(self, freeze):
        q, k, v = _leaves(self.B, self.S, self.H, self.D, seed=3, freeze=freeze)
        vr = make_causal_valid_range(self.S, batch=self.B)
        out, _, _ = block_score_mqa_attention(q, k, v, vr, block_B=64)
        out.sum().backward()
        return q, k, v

    def _sparse(self, freeze):
        q, k, v = _leaves(self.B, self.S, self.H, self.D, seed=4, freeze=freeze)
        vr = make_causal_valid_range(self.S, batch=self.B)
        num_blocks = (self.S + 63) // 64
        idx = paddle.arange(num_blocks, dtype="int32")
        idx = idx.reshape([1, 1, num_blocks]).expand(
            [self.B, self.S, num_blocks]
        )
        out, _ = block_sparse_mqa_attention(q, k, v, idx, vr, block_B=64)
        out.sum().backward()
        return q, k, v

    def _check(self, q, k, v, freeze):
        for name, t in (("q", q), ("k", k), ("v", v)):
            if name in freeze:
                self.assertIsNone(t.grad, f"{name} frozen -> grad must be None")
            else:
                self.assertIsNotNone(
                    t.grad, f"{name} trainable -> grad must exist"
                )

    def test_score_all_trainable(self):
        _cuda_or_skip(self)
        self._check(*self._score(freeze=()), freeze=())

    def test_score_freeze_q(self):
        _cuda_or_skip(self)
        self._check(*self._score(freeze=("q",)), freeze=("q",))

    def test_score_freeze_kv(self):
        _cuda_or_skip(self)
        self._check(*self._score(freeze=("k", "v")), freeze=("k", "v"))

    def test_sparse_all_trainable(self):
        _cuda_or_skip(self)
        self._check(*self._sparse(freeze=()), freeze=())

    def test_sparse_freeze_q(self):
        _cuda_or_skip(self)
        self._check(*self._sparse(freeze=("q",)), freeze=("q",))

    def test_sparse_freeze_kv(self):
        _cuda_or_skip(self)
        self._check(*self._sparse(freeze=("k", "v")), freeze=("k", "v"))


class TestAutogradNonContiguousGrad(unittest.TestCase):
    """Backward must accept a non-contiguous upstream gradient."""

    B, S, H, D = 2, 128, 4, 64

    def _non_contig_grad_loss(self, out):
        # Transposing the PyLayer output makes the gradient that flows back
        # into ``out`` non-contiguous, exercising the wrapper's .contiguous().
        t = out.transpose([0, 2, 1, 3])
        self.assertFalse(t.is_contiguous())
        return t.sum()

    def test_score_non_contiguous(self):
        _cuda_or_skip(self)
        q, k, v = _leaves(self.B, self.S, self.H, self.D, seed=5)
        vr = make_causal_valid_range(self.S, batch=self.B)
        out, _, _ = block_score_mqa_attention(q, k, v, vr, block_B=64)
        self._non_contig_grad_loss(out).backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    def test_sparse_non_contiguous(self):
        _cuda_or_skip(self)
        q, k, v = _leaves(self.B, self.S, self.H, self.D, seed=6)
        vr = make_causal_valid_range(self.S, batch=self.B)
        num_blocks = (self.S + 63) // 64
        idx = paddle.arange(num_blocks, dtype="int32")
        idx = idx.reshape([1, 1, num_blocks]).expand(
            [self.B, self.S, num_blocks]
        )
        out, _ = block_sparse_mqa_attention(q, k, v, idx, vr, block_B=64)
        self._non_contig_grad_loss(out).backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    def test_mha_score_non_contiguous(self):
        _cuda_or_skip(self)
        q, k, v = _leaves_mha(self.B, self.S, self.H, self.D, seed=7)
        vr = make_causal_valid_range(self.S, batch=self.B)
        out, _, _ = block_score_mha_attention(q, k, v, vr, block_B=64)
        self._non_contig_grad_loss(out).backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)


class TestMHAScoreStopGradient(unittest.TestCase):
    """stop_gradient contract for the per-head MHA block-score wrapper."""

    B, S, H, D = 2, 128, 4, 64

    def _run(self, freeze):
        q, k, v = _leaves_mha(
            self.B, self.S, self.H, self.D, seed=8, freeze=freeze
        )
        vr = make_causal_valid_range(self.S, batch=self.B)
        out, _, _ = block_score_mha_attention(q, k, v, vr, block_B=64)
        out.sum().backward()
        for name, t in (("q", q), ("k", k), ("v", v)):
            if name in freeze:
                self.assertIsNone(t.grad, f"{name} frozen -> grad None")
            else:
                self.assertIsNotNone(t.grad, f"{name} trainable -> grad exists")

    def test_all_trainable(self):
        _cuda_or_skip(self)
        self._run(())

    def test_freeze_q(self):
        _cuda_or_skip(self)
        self._run(("q",))

    def test_freeze_kv(self):
        _cuda_or_skip(self)
        self._run(("k", "v"))


class TestDifferentiableCorrectness(unittest.TestCase):
    """Wrappers must match the naive reference forward and backward."""

    B, S, H, D = 2, 128, 4, 64

    def test_mqa_score_matches_reference(self):
        _cuda_or_skip(self)
        q, k, v = _leaves(self.B, self.S, self.H, self.D, seed=20)
        for t in (q, k, v):
            t.stop_gradient = False
        vr = make_causal_valid_range(self.S, batch=self.B)
        qf, kf, vf = _grad_leaves(q, k, v)
        sm = self.D**-0.5
        ref_out, _, _ = ref_block_score_attn_mqa(qf, kf, vf, vr, sm_scale=sm)
        out, _, _ = block_score_mqa_attention(q, k, v, vr, sm_scale=sm)
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        do = paddle.randn([self.B, self.S, self.H, self.D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        (out.astype("float32") * do.astype("float32")).sum().backward()
        self.assertLess(_maxerr(q.grad, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(k.grad, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(v.grad, vf.grad), GRAD_ATOL)

    def test_mha_score_matches_reference(self):
        _cuda_or_skip(self)
        q, k, v = _leaves_mha(self.B, self.S, self.H, self.D, seed=21)
        for t in (q, k, v):
            t.stop_gradient = False
        vr = make_causal_valid_range(self.S, batch=self.B)
        qf, kf, vf = _grad_leaves(q, k, v)
        sm = self.D**-0.5
        ref_out, _, _ = ref_block_score_attn_mha(qf, kf, vf, vr, sm_scale=sm)
        out, _, _ = block_score_mha_attention(q, k, v, vr, sm_scale=sm)
        self.assertLess(_maxerr(out, ref_out), OUT_ATOL)
        do = paddle.randn([self.B, self.S, self.H, self.D], dtype="bfloat16")
        (ref_out * do.astype("float32")).sum().backward()
        (out.astype("float32") * do.astype("float32")).sum().backward()
        self.assertLess(_maxerr(q.grad, qf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(k.grad, kf.grad), GRAD_ATOL)
        self.assertLess(_maxerr(v.grad, vf.grad), GRAD_ATOL)


class TestHySparsePipeline(unittest.TestCase):
    """The score -> TopK -> sparse chain runs forward and backward."""

    B, S, H, D = 2, 128, 4, 64

    def test_forward_backward(self):
        _cuda_or_skip(self)
        q, k, v = _leaves(self.B, self.S, self.H, self.D, seed=30)
        for t in (q, k, v):
            t.stop_gradient = False
        vr = make_causal_valid_range(self.S, batch=self.B)
        sparse_out, sparse_lse, indices, full_out, full_lse = (
            hysparse_mqa_attention(q, k, v, vr, topk=2, block_B=64)
        )
        self.assertEqual(
            list(sparse_out.shape), [self.B, self.S, self.H, self.D]
        )
        self.assertEqual(list(indices.shape), [self.B, self.S, 2])
        self.assertTrue(indices.stop_gradient)
        # both differentiable outputs flow gradient back to q/k/v
        (
            sparse_out.astype("float32").sum()
            + full_out.astype("float32").sum()
        ).backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)


if __name__ == "__main__":
    unittest.main()
