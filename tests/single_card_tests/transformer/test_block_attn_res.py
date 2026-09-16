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

"""Behaviour tests for paddlefleet.transformer.block_attn_res (Block AttnRes).

Block AttnRes replaces fixed-weight residual accumulation with a learned
softmax attention over block-level representations. For the reduction

    all_repr = [*blocks, partial_block]        # each [B, S, H]
    scores[t, r] = sum_h norm(all_repr[r])[t, h] * proj_weight[h]
    prob[t, :]   = softmax_r(scores[t, :])
    out[t, h]    = sum_r prob[t, r] * all_repr[r][t, h]   # values are RAW

the expected values below are derived BY HAND in float64 (numpy), independent
of the module under test, then compared against three production entry points
that must agree with that derivation:

  * ``BlockAttnRes.forward`` else-branch with IdentityOp norm (no rms, no
    norm_weight in the score);
  * ``BlockAttnRes.forward`` else-branch with a real RMSNorm in eval mode;
  * ``BlockAttnRes.forward`` training path, which dispatches through the
    ``BlockAttnResFunc`` PyLayer (RMSNorm math is hardcoded there).

Gradients from the PyLayer path are checked against an independently-written
autograd reference so the PyLayer's save/recompute/return-ordering wiring is
observed, not just the forward value.

Environment: this suite needs paddle (CPU is enough for the exercised math).
It is skipped honestly when paddle / numpy cannot be imported; the fused FLA
Triton kernel (``paddlefleet_ops``) is intentionally NOT required — without it
the module falls back to the PyLayer path, which is exactly what is tested.
"""

import unittest

try:
    import numpy as np
    import paddle
    from paddle.nn import functional as F

    from paddlefleet.transformer.block_attn_res import (
        HAVE_FUSED_ATTNRES,
        BlockAttnRes,
        BlockAttnResSublayersSpec,
        _block_attn_res_rmsnorm,
    )
    from paddlefleet.transformer.identity_op import IdentityOp
    from paddlefleet.transformer.paddle_norm import RMSNorm, WrappedPaddleNorm
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
    HAS_DEPS = True
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    # Only genuine missing-dependency errors are swallowed into a skip; any
    # other exception (API drift, compile failure) must surface as an error.
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    HAS_DEPS = False

_SKIP_REASON = (
    f"paddle/numpy not importable in this environment ({_IMPORT_ERROR})"
)

# Small, fixed, non-degenerate problem sizes. hidden_size is divisible by the
# head count so TransformerConfig construction stays valid; heads are otherwise
# irrelevant to BlockAttnRes.
_B, _S, _H = 2, 3, 8
_NUM_HEADS = 2
_EPS = 1e-5


def _make_config(**overrides):
    defaults = {
        "hidden_size": _H,
        "num_attention_heads": _NUM_HEADS,
        "rms_norm_eps": _EPS,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _distinct(seed, shape):
    """Deterministic, non-uniform array so token/position/channel swaps show."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape).astype(np.float64)


def _ref_reduction(partial_block, blocks, proj_flat, norm_weight, eps, rms):
    """Independent float64 reference for the Block AttnRes reduction.

    ``rms=True`` mirrors the RMSNorm path (score uses rms-normalized reps times
    norm_weight*proj_weight); ``rms=False`` mirrors the IdentityOp path (score
    uses raw reps times proj_weight only, no norm_weight). The *values* summed
    with the softmax weights are the RAW representations in both cases.
    """
    all_repr = [np.asarray(b, np.float64) for b in blocks]
    all_repr.append(np.asarray(partial_block, np.float64))
    lead = all_repr[0].shape[:-1]
    h = all_repr[0].shape[-1]
    flat = [r.reshape(-1, h) for r in all_repr]
    values = np.stack(flat, axis=1)  # [T, R, H]
    proj_flat = np.asarray(proj_flat, np.float64)
    if rms:
        var = np.mean(values**2, axis=-1, keepdims=True)
        normalized = values / np.sqrt(var + eps)
        score_weight = np.asarray(norm_weight, np.float64) * proj_flat
        scores = np.sum(normalized * score_weight, axis=-1)  # [T, R]
    else:
        scores = np.sum(values * proj_flat, axis=-1)  # [T, R]
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    prob = e / e.sum(axis=-1, keepdims=True)  # [T, R]
    out = np.einsum("tr,trh->th", prob, values)  # [T, H]
    return out.reshape(*lead, h)


def _set_param(param, array):
    with paddle.no_grad():
        param.set_value(paddle.to_tensor(np.asarray(array, np.float32)))


def _leaf(array):
    return paddle.to_tensor(np.asarray(array, np.float32), stop_gradient=False)


@unittest.skipUnless(HAS_DEPS, _SKIP_REASON)
class TestSublayersSpecDefaults(unittest.TestCase):
    def test_default_norm_is_identity_op(self):
        # The dataclass default norm must be IdentityOp: this is what selects
        # the non-RMSNorm (standard autograd) forward branch downstream.
        self.assertIs(BlockAttnResSublayersSpec().norm, IdentityOp)

    def test_custom_norm_is_preserved(self):
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)
        self.assertIs(spec.norm, WrappedPaddleNorm)


@unittest.skipUnless(HAS_DEPS, _SKIP_REASON)
class TestConstruction(unittest.TestCase):
    def setUp(self):
        self._orig_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")
        self.addCleanup(paddle.set_default_dtype, self._orig_dtype)

    def test_proj_weight_layout_and_zero_init(self):
        # proj_weight keeps the Linear(hidden_size, 1) checkpoint layout [1, H]
        # and is zero-initialized; both are load-bearing (zero init makes the
        # very first step a plain uniform residual mean, see forward tests).
        block = BlockAttnRes(
            config=_make_config(), sublayers_spec=BlockAttnResSublayersSpec()
        )
        self.assertEqual(list(block.proj_weight.shape), [1, _H])
        np.testing.assert_array_equal(
            block.proj_weight.numpy(), np.zeros([1, _H], np.float32)
        )

    def test_identity_norm_selects_non_pylayer_path(self):
        block = BlockAttnRes(
            config=_make_config(), sublayers_spec=BlockAttnResSublayersSpec()
        )
        self.assertIsInstance(block.norm, IdentityOp)
        self.assertFalse(block._use_pylayer)
        # No fused kernel is installed here, so the fused path must stay off.
        self.assertFalse(block._use_fused)

    def test_rmsnorm_selects_pylayer_path(self):
        block = BlockAttnRes(
            config=_make_config(),
            sublayers_spec=BlockAttnResSublayersSpec(norm=WrappedPaddleNorm),
        )
        self.assertIsInstance(block.norm, RMSNorm)
        self.assertTrue(block._use_pylayer)
        # RMSNorm makes the instance fused-eligible (attn_res_fusion default on,
        # deterministic_mode off), so _use_fused tracks HAVE_FUSED_ATTNRES
        # exactly: True when the paddlefleet_ops fused kernel is built (real
        # GPU runner), False when it is absent and the PyLayer path is used.
        self.assertEqual(block._use_fused, HAVE_FUSED_ATTNRES)


@unittest.skipUnless(HAS_DEPS, _SKIP_REASON)
class TestForwardIdentityNorm(unittest.TestCase):
    """else-branch forward with IdentityOp norm (always taken: not RMSNorm)."""

    def setUp(self):
        self._orig_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")
        self.addCleanup(paddle.set_default_dtype, self._orig_dtype)

    def _build(self):
        return BlockAttnRes(
            config=_make_config(), sublayers_spec=BlockAttnResSublayersSpec()
        )

    def test_nonuniform_proj_weight_matches_reference(self):
        # Non-uniform proj_weight and distinct representations produce
        # per-token, non-uniform softmax weights: a swap of representations,
        # a transposed proj_weight, or dropping the softmax would all change
        # the output and be caught here.
        partial = _distinct(1, [_B, _S, _H])
        blocks = [_distinct(2, [_B, _S, _H]), _distinct(3, [_B, _S, _H])]
        proj = _distinct(9, [1, _H])

        block = self._build()
        _set_param(block.proj_weight, proj)

        out = block(
            paddle.to_tensor(partial, "float32"),
            [paddle.to_tensor(b, "float32") for b in blocks],
        )
        expected = _ref_reduction(
            partial, blocks, proj.reshape(-1), None, _EPS, rms=False
        )
        self.assertEqual(list(out.shape), [_B, _S, _H])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_zero_init_gives_uniform_residual_mean(self):
        # Production default: proj_weight == 0 => all scores 0 => softmax is
        # uniform => output is the plain mean over [*blocks, partial]. This
        # pins the first-step behaviour that the zero init is chosen for.
        partial = _distinct(4, [_B, _S, _H])
        blocks = [_distinct(5, [_B, _S, _H]), _distinct(6, [_B, _S, _H])]

        block = self._build()  # proj_weight left at its zero default
        out = block(
            paddle.to_tensor(partial, "float32"),
            [paddle.to_tensor(b, "float32") for b in blocks],
        )
        expected = np.mean(np.stack([*blocks, partial], axis=0), axis=0)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_no_completed_blocks_returns_partial_unchanged(self):
        # Boundary: with a single representation the softmax over one element is
        # 1.0, so the output must equal partial_block exactly regardless of the
        # (here non-zero) proj_weight.
        partial = _distinct(7, [_B, _S, _H])
        block = self._build()
        _set_param(block.proj_weight, _distinct(8, [1, _H]))

        out = block(paddle.to_tensor(partial, "float32"), [])
        np.testing.assert_allclose(out.numpy(), partial, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(HAS_DEPS, _SKIP_REASON)
class TestForwardRMSNorm(unittest.TestCase):
    """RMSNorm path: eval else-branch and training PyLayer must both match."""

    def setUp(self):
        self._orig_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")
        self.addCleanup(paddle.set_default_dtype, self._orig_dtype)

    def _build_and_seed(self, proj, norm_w):
        block = BlockAttnRes(
            config=_make_config(),
            sublayers_spec=BlockAttnResSublayersSpec(norm=WrappedPaddleNorm),
        )
        _set_param(block.proj_weight, proj)
        _set_param(block.norm.weight, norm_w)
        return block

    def test_pure_reduction_function_matches_reference(self):
        # Exercises the shared _block_attn_res_rmsnorm helper directly against
        # the independent float64 reference (score uses norm_weight*proj_weight
        # over rms-normalized reps; values are raw).
        partial = _distinct(11, [_B, _S, _H])
        blocks = [_distinct(12, [_B, _S, _H]), _distinct(13, [_B, _S, _H])]
        proj = _distinct(14, [1, _H])
        norm_w = _distinct(15, [_H])

        out = _block_attn_res_rmsnorm(
            paddle.to_tensor(partial, "float32"),
            [paddle.to_tensor(b, "float32") for b in blocks],
            paddle.to_tensor(proj, "float32"),
            paddle.to_tensor(norm_w, "float32"),
            _EPS,
        )
        expected = _ref_reduction(
            partial, blocks, proj.reshape(-1), norm_w, _EPS, rms=True
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_eval_and_training_paths_match_reference(self):
        partial = _distinct(21, [_B, _S, _H])
        blocks = [_distinct(22, [_B, _S, _H]), _distinct(23, [_B, _S, _H])]
        proj = _distinct(24, [1, _H])
        norm_w = _distinct(25, [_H])
        expected = _ref_reduction(
            partial, blocks, proj.reshape(-1), norm_w, _EPS, rms=True
        )

        # eval() forces the standard autograd else-branch (real RMSNorm layer).
        eval_block = self._build_and_seed(proj, norm_w)
        eval_block.eval()
        eval_out = eval_block(
            paddle.to_tensor(partial, "float32"),
            [paddle.to_tensor(b, "float32") for b in blocks],
        )
        np.testing.assert_allclose(
            eval_out.numpy(), expected, rtol=1e-4, atol=1e-5
        )

        # train() dispatches through BlockAttnResFunc (PyLayer, RMSNorm math
        # hardcoded). The two independent code paths must agree with the same
        # reference (and hence with each other).
        train_block = self._build_and_seed(proj, norm_w)
        train_block.train()
        self.assertTrue(train_block._use_pylayer)
        train_out = train_block(
            paddle.to_tensor(partial, "float32"),
            [paddle.to_tensor(b, "float32") for b in blocks],
        )
        np.testing.assert_allclose(
            train_out.numpy(), expected, rtol=1e-4, atol=1e-5
        )


@unittest.skipUnless(HAS_DEPS, _SKIP_REASON)
class TestPyLayerBackward(unittest.TestCase):
    """PyLayer backward wiring: every input gets the autograd-correct grad."""

    def setUp(self):
        self._orig_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")
        self.addCleanup(paddle.set_default_dtype, self._orig_dtype)

    def _reference_grads(self, partial, blocks, proj, norm_w, upstream):
        """Independently-written autograd reference for the reduction.

        Mirrors the reduction math but is authored here (does not call the
        production helper), so a mis-routed / mis-ordered / dropped gradient in
        the PyLayer is caught while the shared forward math is validated
        separately by the forward tests.
        """
        rp = _leaf(partial)
        rb = [_leaf(b) for b in blocks]
        rproj = _leaf(proj)
        rnw = _leaf(norm_w)

        all_repr = [*rb, rp]
        flat = [r.reshape([-1, _H]) for r in all_repr]
        values = paddle.stack(flat, axis=1).astype("float32")
        var = values.pow(2).mean(axis=-1, keepdim=True)
        normalized = values * paddle.rsqrt(var + _EPS)
        score_weight = rnw.astype("float32") * rproj.reshape([-1]).astype(
            "float32"
        )
        scores = (normalized * score_weight).sum(axis=-1)
        prob = F.softmax(scores, axis=-1).unsqueeze(1)
        out = paddle.matmul(prob, values).squeeze(1).reshape(rp.shape)
        (out * paddle.to_tensor(upstream, "float32")).sum().backward()
        return {
            "partial": rp.grad.numpy(),
            "blocks": [b.grad.numpy() for b in rb],
            "proj": rproj.grad.numpy(),
            "norm": rnw.grad.numpy(),
        }

    def test_grads_match_independent_autograd_reference(self):
        partial = _distinct(31, [_B, _S, _H])
        blocks = [_distinct(32, [_B, _S, _H]), _distinct(33, [_B, _S, _H])]
        proj = _distinct(34, [1, _H])
        norm_w = _distinct(35, [_H])
        upstream = _distinct(
            36, [_B, _S, _H]
        )  # non-uniform: exposes misrouting

        ref = self._reference_grads(partial, blocks, proj, norm_w, upstream)

        block = BlockAttnRes(
            config=_make_config(),
            sublayers_spec=BlockAttnResSublayersSpec(norm=WrappedPaddleNorm),
        )
        block.train()
        _set_param(block.proj_weight, proj)
        _set_param(block.norm.weight, norm_w)

        p_in = _leaf(partial)
        b_in = [_leaf(b) for b in blocks]
        out = block(p_in, b_in)
        (out * paddle.to_tensor(upstream, "float32")).sum().backward()

        # Every forward tensor input must receive a gradient (no None, no
        # zip-truncation): partial_block, both blocks, proj_weight, norm.weight.
        got = {
            "partial": p_in.grad,
            "blocks": [b.grad for b in b_in],
            "proj": block.proj_weight.grad,
            "norm": block.norm.weight.grad,
        }
        self.assertIsNotNone(got["partial"])
        self.assertIsNotNone(got["proj"])
        self.assertIsNotNone(got["norm"])
        for g in got["blocks"]:
            self.assertIsNotNone(g)

        # Reference grads are non-trivial, so a zeroed / dropped grad fails.
        self.assertGreater(np.abs(ref["proj"]).max(), 1e-3)
        self.assertGreater(np.abs(ref["norm"]).max(), 1e-3)
        self.assertGreater(np.abs(ref["partial"]).max(), 1e-3)

        np.testing.assert_allclose(
            got["partial"].numpy(), ref["partial"], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            got["proj"].numpy(), ref["proj"], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            got["norm"].numpy(), ref["norm"], rtol=1e-4, atol=1e-5
        )
        for g, gref in zip(got["blocks"], ref["blocks"]):
            self.assertGreater(np.abs(gref).max(), 1e-3)
            np.testing.assert_allclose(g.numpy(), gref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
