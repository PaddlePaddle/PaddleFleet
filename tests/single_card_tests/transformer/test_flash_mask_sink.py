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

"""Single-card tests for FA4 attention-sink (learnable_sink) support.

Covers PR "Support FA4 sink." (ab8c450) on the non-CP paths:
  1. Low-level cute ``flashmask_attention(..., learnable_sink=)`` fwd+bwd vs an
     inline fp32 golden (trainable sink / sink=None / GQA / dsink dtype).
  2. Facade ``paddlefleet_ops.flash_mask_facade.flashmask_attention`` (+2 lines:
     forwards learnable_sink), exercising its two quirks (clone() on
     startend_row_indices; lse reshape only valid for nheads==1).
  3. ``DotProductAttention`` softmax_offset construction (vanilla/off-by-one/
     learnable + sink-bias promotion) and the fwd/bwd sink branch.
  4. Refined-recompute (rr) FlashMask attention with learnable_sink, driven
     through ``recompute`` to force both forward passes, vs the non-rr path.
"""

import contextlib
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle
from paddlefleet_ops import is_flash_mask_available

# Force FA4 (cute) backend by default for the existing FA4 cases. FA3 cases
# switch the flag locally and restore it after each run.
paddle.set_flags({"FLAGS_flash_attn_version": 4})

DTYPE = paddle.bfloat16
SEED = 2026


def _is_sm100_or_newer():
    try:
        return (
            paddle.cuda.is_available()
            and paddle.cuda.get_device_capability()[0] >= 10
        )
    except (AttributeError, RuntimeError, ValueError):
        return False


# FA4 attention-sink lives only in the cute backend, which is built solely for
# compute capability >= 10 (sm100/Blackwell). Elsewhere the cute kernels are
# absent and the facade falls back to a sink-less backend, so skip these tests.
_SINK_AVAILABLE = is_flash_mask_available()
_SKIP_REASON = (
    "FA4 attention-sink requires the cute backend (sm100, capability >= 10)"
)
_IS_SM100_OR_NEWER = _is_sm100_or_newer()
_FA3_SINK_AVAILABLE = (
    not _IS_SM100_OR_NEWER
    and hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_v3")
    and hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_v3_grad")
    and hasattr(paddle.base.libpaddle.pir.ops, "flashmask_attention_v2_grad")
)
_FA3_SKIP_REASON = "FA3 sink attention requires FA3 flash attention ops and is disabled on sm100+"


@contextlib.contextmanager
def _flash_attn_version(version):
    old = paddle.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]
    paddle.set_flags({"FLAGS_flash_attn_version": version})
    try:
        yield
    finally:
        paddle.set_flags({"FLAGS_flash_attn_version": old})


def _startend_row_indices(batch_size, seq_len, causal):
    """Build a [b, 1, seq_len, 2] int32 startend_row_indices tensor.

    causal=True  -> start=0,       end=arange(1, seq+1)  (lower-triangular)
    causal=False -> start=seq_len, end=0                 (full attention)

    For the non-causal 2-col case the columns are (down_start, up_end):
    rows >= down_start and rows < up_end are masked. "Mask nothing" (full
    attention) is therefore down_start=seq_len, up_end=0 -- matching the repo's
    generate_non_causal_mask. (start=0, end=seq_len would mask EVERY row.)
    """
    if causal:
        start = np.zeros((batch_size, 1, seq_len, 1), dtype=np.int32)
        end = np.arange(1, seq_len + 1, dtype=np.int32).reshape(
            1, 1, seq_len, 1
        )
        end = np.broadcast_to(end, (batch_size, 1, seq_len, 1))
    else:
        start = np.full((batch_size, 1, seq_len, 1), seq_len, dtype=np.int32)
        end = np.zeros((batch_size, 1, seq_len, 1), dtype=np.int32)
    indices = np.concatenate([start, end], axis=-1)
    return paddle.to_tensor(indices, dtype=paddle.int32)


def _attn_bias_from_indices(startend_row_indices, seqlen_q, nheads, causal):
    """Dense additive bias (-inf masked) from a 2-col startend_row_indices.

    Mirrors generate_startend_row_indices.startend_row_indices_to_attn_bias for
    the causal-2col / non-causal-2col cases used in these tests. Returns a fp32
    tensor broadcastable to [b, nheads, seqlen_q, seqlen_k].
    """
    bz, num_head, seqlen_k, bound_num = startend_row_indices.shape
    assert nheads % num_head == 0
    idx = startend_row_indices.numpy()
    m = np.zeros((bz, num_head, seqlen_q, seqlen_k), dtype=np.float32)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seqlen_k):
                downstart = int(idx[bi, hi, j, 0])
                if causal:
                    downend = int(idx[bi, hi, j, 1])
                    m[bi, hi, downstart:downend, j] = -np.inf
                    # bottom-right aligned causal mask
                    top = max(0, j - (seqlen_k - seqlen_q))
                    m[bi, hi, :top, j] = -np.inf
                else:
                    upend = int(idx[bi, hi, j, 1])
                    m[bi, hi, downstart:, j] = -np.inf
                    m[bi, hi, :upend, j] = -np.inf
    m = np.repeat(m, nheads // num_head, axis=1)
    return paddle.to_tensor(m, dtype=paddle.float32)


def attention_ref_with_sink(q, k, v, attn_bias, learnable_sink):
    """fp32 reference for flashmask + attention-sink.

    q,k,v: [b, s, h, d] (GQA: h_kv may be < h_q). attn_bias: additive [-inf] mask
    broadcastable to [b, h_q, sq, sk]. learnable_sink: [h_q] per-q-head logit
    (un-scaled) competing only in the softmax denominator, or None.

    sink formula:
        sink   = sink[h].reshape(1, h, 1, 1)            # fp32, un-scaled logit
        row_max= max(scores.max(-1, keepdim), sink)
        denom  = exp(scores-row_max).sum(-1, keepdim) + exp(sink-row_max)
        attn   = exp(scores-row_max) / denom
        out    = attn @ v
    """
    qf = q.astype("float32").transpose([0, 2, 1, 3])  # b h s d
    kf = k.astype("float32").transpose([0, 2, 1, 3])
    vf = v.astype("float32").transpose([0, 2, 1, 3])

    h_q = qf.shape[1]
    g = h_q // kf.shape[1]
    if g > 1:
        kf = paddle.repeat_interleave(kf, g, axis=1)
        vf = paddle.repeat_interleave(vf, g, axis=1)

    d = qf.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    scores = paddle.matmul(qf * softmax_scale, kf, transpose_y=True)

    if attn_bias is not None:
        bias = attn_bias
        if bias.shape[1] != h_q:
            bias = paddle.repeat_interleave(bias, h_q // bias.shape[1], axis=1)
        scores = scores + bias
        all_inf = (bias == -np.inf).all(axis=-1, keepdim=True)
        scores = paddle.where(all_inf, paddle.full_like(scores, -1e30), scores)
    else:
        all_inf = None

    row_max = scores.max(axis=-1, keepdim=True)
    if learnable_sink is not None:
        sink = learnable_sink.astype("float32").reshape([1, h_q, 1, 1])
        row_max = paddle.maximum(row_max, sink)
        exp_sink = paddle.exp(sink - row_max)
    else:
        exp_sink = 0.0

    exp_scores = paddle.exp(scores - row_max)
    denom = exp_scores.sum(axis=-1, keepdim=True) + exp_sink
    attention = exp_scores / denom
    if all_inf is not None:
        attention = paddle.where(
            all_inf, paddle.zeros_like(attention), attention
        )

    out = paddle.matmul(attention.astype(vf.dtype), vf)
    out = out.transpose([0, 2, 1, 3])  # b s h d
    return out


class TestSinkImplMaskAndHelpers(unittest.TestCase):
    """CPU-friendly coverage for sink_impl mask and tensor helpers."""

    @staticmethod
    def _assert_array_equal(actual, expected):
        np.testing.assert_array_equal(actual.numpy(), np.asarray(expected))

    def test_gen_dense_mask_supported_bound_layouts(self):
        from paddlefleet.transformer.sink_impl import (
            gen_dense_mask_from_startend_row_indices,
        )

        cases = (
            (np.array([[[3, 2, 1]]], dtype=np.int32), None),
            (
                np.array([[[[1, 2], [1, 3], [2, 3]]]], dtype=np.int32),
                True,
            ),
            (
                np.array([[[[3, 0], [2, 1], [3, 2]]]], dtype=np.int32),
                False,
            ),
            (
                np.array(
                    [[[[1, 2, 0, 1], [2, 3, 0, 2], [1, 3, 1, 2]]]],
                    dtype=np.int32,
                ),
                None,
            ),
        )
        for indices, is_causal in cases:
            with self.subTest(bound_num=indices.shape[-1], causal=is_causal):
                tensor = paddle.to_tensor(indices)
                result = gen_dense_mask_from_startend_row_indices(
                    tensor, dtype=paddle.float32, is_causal=is_causal
                )

                normalized = (
                    indices[..., None] if indices.ndim == 3 else indices
                )
                causal = normalized.shape[-1] == 1 or is_causal is True
                expected_allowed = np.ones((1, 1, 3, 3), dtype=bool)
                for row in range(3):
                    for col in range(3):
                        bounds = normalized[0, 0, col]
                        allowed = not causal or row >= col
                        if causal:
                            allowed &= row < bounds[0] or (
                                len(bounds) == 2 and row >= bounds[1]
                            )
                        else:
                            lower = row < bounds[0]
                            if len(bounds) == 2:
                                upper = row >= bounds[1]
                            else:
                                lower |= row >= bounds[1]
                                upper = row >= bounds[3] or row < bounds[2]
                            allowed &= lower and upper
                        expected_allowed[0, 0, row, col] = allowed
                expected = np.where(expected_allowed, 0.0, -1_000_000.0)
                np.testing.assert_array_equal(result.numpy(), expected)

    def test_gen_dense_mask_requires_causal_for_ambiguous_layout(self):
        from paddlefleet.transformer.sink_impl import (
            gen_dense_mask_from_startend_row_indices,
        )

        indices = paddle.zeros([1, 1, 2, 2], dtype=paddle.int32)
        with self.assertRaisesRegex(ValueError, "is_causal.*must be specified"):
            gen_dense_mask_from_startend_row_indices(indices)

    def test_dense_mask_conversion_valid_masks(self):
        from paddlefleet.transformer.sink_impl import (
            _dense_mask_to_startend_row_indices,
        )

        query = paddle.zeros([2, 3, 2, 4])
        key = paddle.zeros_like(query)
        causal = np.triu(np.ones((3, 3), dtype=bool), k=1)

        bool_result = _dense_mask_to_startend_row_indices(
            paddle.to_tensor(causal).reshape([1, 1, 3, 3]),
            query,
            key,
            True,
        )
        self.assertEqual(list(bool_result.shape), [2, 1, 3, 1])
        self._assert_array_equal(bool_result, np.full((2, 1, 3, 1), 3))

        additive = np.where(causal, -1e4, 0.0).astype("float32")
        additive = np.broadcast_to(additive, (2, 1, 3, 3)).copy()
        additive_result = _dense_mask_to_startend_row_indices(
            paddle.to_tensor(additive), query, key, True
        )
        self._assert_array_equal(additive_result, bool_result.numpy())

        keep_mask = paddle.ones([1, 1, 3, 3], dtype=paddle.float32)
        full_result = _dense_mask_to_startend_row_indices(
            keep_mask, query, key, False
        )
        expected_full = np.stack(
            [np.full((2, 1, 3, 1), 3), np.zeros((2, 1, 3, 1))],
            axis=-1,
        ).reshape(2, 1, 3, 2)
        self._assert_array_equal(full_result, expected_full)
        self.assertIsNone(
            _dense_mask_to_startend_row_indices(None, query, key, False)
        )

    def test_dense_mask_conversion_rejects_unsupported_masks(self):
        from paddlefleet.transformer.sink_impl import (
            _dense_mask_to_startend_row_indices,
        )

        query = paddle.zeros([2, 3, 2, 4])
        key = paddle.zeros_like(query)
        valid = paddle.zeros([1, 1, 3, 3], dtype=paddle.bool)
        cases = (
            (valid, query, paddle.zeros([2, 4, 2, 4]), "q_len == kv_len"),
            (paddle.zeros([3, 3]), query, key, "4-D dense masks"),
            (paddle.zeros([1, 2, 3, 3]), query, key, "shape.*seq, seq"),
            (
                paddle.to_tensor(
                    np.stack(
                        [np.zeros((3, 3), dtype=bool), np.eye(3, dtype=bool)]
                    )[:, None]
                ),
                query,
                key,
                "per-sample dense mask differences",
            ),
            (
                paddle.to_tensor(np.eye(3, dtype=bool)).reshape([1, 1, 3, 3]),
                query,
                key,
                "equivalent to causal or full attention",
            ),
        )
        for mask, q_value, k_value, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(NotImplementedError, message),
            ):
                _dense_mask_to_startend_row_indices(
                    mask, q_value, k_value, False
                )

    def test_repeat_kv_identity_and_repetition(self):
        from paddlefleet.transformer.sink_impl import _repeat_kv

        hidden = paddle.arange(12, dtype=paddle.float32).reshape([1, 2, 2, 3])
        self.assertIs(_repeat_kv(hidden, 1), hidden)
        repeated = _repeat_kv(hidden, 3)
        expected = np.repeat(hidden.numpy(), 3, axis=2)
        self._assert_array_equal(repeated, expected)

    def test_get_fa_version_environment_branches(self):
        from paddlefleet.transformer import sink_impl

        with (
            mock.patch.object(
                sink_impl.paddle, "get_device", return_value="xpu:0"
            ),
            mock.patch.object(
                sink_impl.paddle.base.framework, "get_flags"
            ) as get_flags,
        ):
            self.assertEqual(sink_impl._get_fa_version(256), 2)
            get_flags.assert_not_called()

        def flags(names):
            name = names[0]
            return {name: 3 if name == "FLAGS_flash_attn_version" else True}

        with (
            mock.patch.object(
                sink_impl.paddle, "get_device", return_value="gpu:0"
            ),
            mock.patch.object(
                sink_impl.paddle.base.framework,
                "get_flags",
                side_effect=flags,
            ),
        ):
            self.assertEqual(sink_impl._get_fa_version(256), 2)
            self.assertEqual(sink_impl._get_fa_version(128), 3)

        with (
            mock.patch.object(
                sink_impl.paddle, "get_device", return_value="gpu:0"
            ),
            mock.patch.object(
                sink_impl.paddle.base.framework,
                "get_flags",
                return_value={"FLAGS_flash_attn_version": 4},
            ),
        ):
            self.assertEqual(sink_impl._get_fa_version(256), 4)

    def test_full_indices_logaddexp_and_merge_grad(self):
        from paddlefleet.transformer import sink_impl

        _full_startend_row_indices = sink_impl._full_startend_row_indices
        _merge_kv_grad = sink_impl._merge_kv_grad
        _stable_logaddexp = sink_impl._stable_logaddexp

        full = _full_startend_row_indices(2, 3)
        expected_full = np.concatenate(
            [np.full((2, 1, 3, 1), 3), np.zeros((2, 1, 3, 1))], axis=-1
        )
        self.assertEqual(full.dtype, paddle.int32)
        self._assert_array_equal(full, expected_full)

        lhs = paddle.to_tensor([[1000.0], [-1000.0], [0.0]])
        rhs = paddle.to_tensor([[999.0, -1000.0]])
        actual = _stable_logaddexp(lhs, rhs)
        np.testing.assert_allclose(
            actual.numpy(), np.logaddexp(lhs.numpy(), rhs.numpy()), rtol=1e-6
        )

        original = paddle.zeros([1, 2, 2, 1])
        repeated = paddle.arange(8, dtype=paddle.float32).reshape([1, 2, 4, 1])
        merged = _merge_kv_grad(repeated, original, 2)
        expected = repeated.numpy().reshape(1, 2, 2, 2, 1).sum(axis=3)
        self._assert_array_equal(merged, expected)
        self.assertIs(_merge_kv_grad(repeated, original, 1), repeated)
        mismatched = paddle.zeros([1, 2, 3, 1])
        self.assertIs(_merge_kv_grad(mismatched, original, 2), mismatched)


class TestPrepareFA3SinkAttention(unittest.TestCase):
    """CPU-only routing tests for FA3 sink preparation."""

    def setUp(self):
        self.q = paddle.zeros([2, 3, 2, 4], dtype=paddle.float32)
        self.k = paddle.zeros_like(self.q)
        self.v = paddle.zeros_like(self.q)
        self.sink = paddle.zeros([2], dtype=paddle.float32)

    def test_routes_only_non_none_fa3_sink(self):
        from paddlefleet.transformer import sink_impl

        indices = paddle.zeros([2, 1, 3, 1], dtype=paddle.int32)
        for version, sink, expected in (
            (2, self.sink, False),
            (3, None, False),
        ):
            with (
                self.subTest(version=version, sink=sink),
                mock.patch.object(
                    sink_impl, "_get_fa_version", return_value=version
                ),
            ):
                enabled, returned = sink_impl.prepare_fa3_sink_attention(
                    self.q,
                    self.k,
                    self.v,
                    sink,
                    startend_row_indices=indices,
                )
                self.assertEqual(enabled, expected)
                self.assertIs(returned, indices)

        with mock.patch.object(sink_impl, "_get_fa_version", return_value=3):
            enabled, returned = sink_impl.prepare_fa3_sink_attention(
                self.q,
                self.k,
                self.v,
                self.sink,
                startend_row_indices=indices,
            )
        self.assertTrue(enabled)
        self.assertIs(returned, indices)

    def test_converts_dense_mask_and_forwards_causal(self):
        from paddlefleet.transformer import sink_impl

        converted = paddle.ones([2, 1, 3, 1], dtype=paddle.int32)
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=3),
            mock.patch.object(
                sink_impl,
                "_dense_mask_to_startend_row_indices",
                return_value=converted,
            ) as convert,
        ):
            enabled, returned = sink_impl.prepare_fa3_sink_attention(
                self.q,
                self.k,
                self.v,
                self.sink,
                attention_mask="mask",
                causal=True,
            )
        self.assertTrue(enabled)
        self.assertIs(returned, converted)
        convert.assert_called_once_with("mask", self.q, self.k, True)

    def test_rejects_unsupported_modes_and_decode(self):
        from paddlefleet.transformer import sink_impl

        cases = (
            ({"context_parallel_size": 2}, "non-context-parallel"),
            ({"use_rr_flash_attention": True}, "refined recompute"),
            ({"flashmask_use_varlen": True}, "flashmask_use_varlen"),
        )
        with mock.patch.object(sink_impl, "_get_fa_version", return_value=3):
            for kwargs, message in cases:
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(NotImplementedError, message),
                ):
                    sink_impl.prepare_fa3_sink_attention(
                        self.q, self.k, self.v, self.sink, **kwargs
                    )

            short_key = paddle.zeros([2, 2, 2, 4], dtype=paddle.float32)
            with self.assertRaisesRegex(NotImplementedError, "KV-cache decode"):
                sink_impl.prepare_fa3_sink_attention(
                    self.q, short_key, self.v, self.sink
                )


class TestFlashAttentionDispatch(unittest.TestCase):
    """Mock every FA dispatch backend without invoking a device kernel."""

    def setUp(self):
        self.q = paddle.zeros([1, 2, 2, 4], dtype=paddle.float32)
        self.k = paddle.ones_like(self.q)
        self.v = paddle.full_like(self.q, 2.0)
        self.out = paddle.full_like(self.q, 3.0)
        self.lse = paddle.zeros([1, 2, 4], dtype=paddle.float32)
        self.grad = paddle.ones_like(self.q)

    @staticmethod
    def _ops_patch(sink_impl, *names):
        fake_ops = SimpleNamespace(**{name: object() for name in names})
        return mock.patch.object(
            sink_impl.paddle.base,
            "libpaddle",
            SimpleNamespace(pir=SimpleNamespace(ops=fake_ops)),
        )

    def test_forward_fa2_mock_and_lse_slice(self):
        from paddlefleet.transformer import sink_impl

        c_ops = SimpleNamespace(
            flash_attn=mock.Mock(return_value=(self.out, None, self.lse, None))
        )
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=2),
            mock.patch.object(sink_impl, "_C_ops", c_ops),
            self._ops_patch(sink_impl, "flash_attn"),
        ):
            result, lse = sink_impl._flash_attention_forward_dispatch(
                self.q,
                self.k,
                self.v,
                dropout=0.25,
                causal=True,
                fixed_seed_offset="seed",
                rng_name="rng",
                training=False,
            )
        self.assertIs(result, self.out)
        self.assertEqual(list(lse.shape), [1, 2, 2])
        args = c_ops.flash_attn.call_args.args
        self.assertEqual(args[3:7], ("seed", None, 0.25, True))
        self.assertEqual(args[8:10], (True, "rng"))

    def test_forward_fa3_mock_and_parameter_errors(self):
        from paddlefleet.transformer import sink_impl

        c_ops = SimpleNamespace(
            flash_attn_v3=mock.Mock(return_value=(self.out, self.lse))
        )
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=3),
            mock.patch.object(sink_impl, "_C_ops", c_ops),
            self._ops_patch(sink_impl, "flash_attn_v3"),
        ):
            result = sink_impl._flash_attention_forward_dispatch(
                self.q, self.k, self.v, causal=True, softmax_scale=0.7
            )
            self.assertIs(result[0], self.out)
            self.assertIs(result[1], self.lse)
            self.assertEqual(
                c_ops.flash_attn_v3.call_args.args[7:9], (0.7, True)
            )
            with self.assertRaisesRegex(NotImplementedError, "dropout"):
                sink_impl._flash_attention_forward_dispatch(
                    self.q, self.k, self.v, dropout=0.1, training=True
                )
            with self.assertRaisesRegex(AssertionError, "dense mask"):
                sink_impl._flash_attention_forward_dispatch(
                    self.q, self.k, self.v, attention_mask=self.lse
                )

    def test_forward_fa4_mock_and_errors(self):
        from paddlefleet.transformer import sink_impl

        kernel = mock.Mock(return_value=(self.out, self.lse))
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=4),
            mock.patch.object(sink_impl, "_flash_attn_fwd", kernel),
        ):
            self.assertEqual(
                sink_impl._flash_attention_forward_dispatch(
                    self.q, self.k, self.v, softmax_scale=0.25
                ),
                (self.out, self.lse),
            )
        kernel.assert_called_once_with(
            self.q,
            self.k,
            self.v,
            softmax_scale=0.25,
            causal=False,
            return_lse=True,
            startend_row_indices=None,
            pack_gqa=False,
        )

        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=4),
            self.assertRaisesRegex(AssertionError, "not available"),
        ):
            sink_impl._flash_attention_forward_dispatch(self.q, self.k, self.v)

    def test_forward_common_validation_and_unsupported_version(self):
        from paddlefleet.transformer import sink_impl

        with self.assertRaisesRegex(AssertionError, "return_softmax"):
            sink_impl._flash_attention_forward_dispatch(
                self.q, self.k, self.v, return_softmax=True
            )
        with self.assertRaisesRegex(AssertionError, "equal sequence lengths"):
            sink_impl._flash_attention_forward_dispatch(
                self.q, self.k[:, :1], self.v
            )
        with self.assertRaisesRegex(NotImplementedError, "head_dim to match"):
            sink_impl._flash_attention_forward_dispatch(
                self.q, self.k, self.v[..., :2]
            )
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=9),
            self.assertRaisesRegex(ValueError, "Unsupported"),
        ):
            sink_impl._flash_attention_forward_dispatch(self.q, self.k, self.v)

    def test_backward_fa2_and_fa3_mocks(self):
        from paddlefleet.transformer import sink_impl

        grads = (self.q + 1, self.k + 2, self.v + 3)
        cases = (
            (2, "flash_attn_grad", 10),
            (3, "flash_attn_v3_grad", 12),
        )
        for version, op_name, arg_count in cases:
            op = mock.Mock(return_value=grads)
            with (
                self.subTest(version=version),
                mock.patch.object(
                    sink_impl, "_get_fa_version", return_value=version
                ),
                mock.patch.object(
                    sink_impl, "_C_ops", SimpleNamespace(**{op_name: op})
                ),
                self._ops_patch(sink_impl, op_name),
            ):
                actual = sink_impl._flash_attention_backward_dispatch(
                    self.grad,
                    self.q,
                    self.k,
                    self.v,
                    self.out,
                    self.lse,
                    causal=True,
                    softmax_scale=0.5,
                )
            for returned, expected in zip(actual, grads):
                self.assertIs(returned, expected)
            self.assertEqual(len(op.call_args.args), arg_count)

    def test_backward_fa4_and_error_branches(self):
        from paddlefleet.transformer import sink_impl

        grads = (self.q + 1, self.k + 2, self.v + 3)
        kernel = mock.Mock(return_value=grads)
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=4),
            mock.patch.object(sink_impl, "_flash_attn_bwd", kernel),
            mock.patch.object(
                sink_impl.paddle,
                "get_flags",
                return_value={"FLAGS_cudnn_deterministic": True},
            ),
        ):
            actual = sink_impl._flash_attention_backward_dispatch(
                self.grad, self.q, self.k, self.v, self.out, self.lse
            )
        for returned, expected in zip(actual, grads):
            self.assertIs(returned, expected)
        self.assertIsNone(kernel.call_args.args[6])
        self.assertTrue(kernel.call_args.kwargs["deterministic"])

        for version, message in ((4, "not available"), (8, "Unsupported")):
            with (
                self.subTest(version=version),
                mock.patch.object(
                    sink_impl, "_get_fa_version", return_value=version
                ),
                mock.patch.object(sink_impl, "_flash_attn_bwd", None),
            ):
                error = AssertionError if version == 4 else ValueError
                with self.assertRaisesRegex(error, message):
                    sink_impl._flash_attention_backward_dispatch(
                        self.grad,
                        self.q,
                        self.k,
                        self.v,
                        self.out,
                        self.lse,
                    )


class TestFlashMaskDispatch(unittest.TestCase):
    """CPU-only FlashMask dispatch and compatibility-signature coverage."""

    def setUp(self):
        self.q = paddle.zeros([1, 2, 2, 4], dtype=paddle.float32)
        self.k = paddle.ones_like(self.q)
        self.v = paddle.full_like(self.q, 2.0)
        self.out = paddle.full_like(self.q, 3.0)
        self.lse = paddle.zeros([1, 2, 2], dtype=paddle.float32)
        self.grad = paddle.ones_like(self.q)
        self.indices = paddle.zeros([1, 1, 2, 1], dtype=paddle.int32)

    def test_forward_fa2_fa3_functional_mocks(self):
        from paddlefleet.transformer import sink_impl

        for version in (2, 3):
            function = mock.Mock(return_value=(self.out, self.lse))
            with (
                self.subTest(version=version),
                mock.patch.object(
                    sink_impl, "_get_fa_version", return_value=version
                ),
                mock.patch.object(
                    sink_impl.paddle.nn.functional,
                    "flashmask_attention",
                    function,
                ),
            ):
                actual = sink_impl._flashmask_attention_forward_dispatch(
                    self.q,
                    self.k,
                    self.v,
                    self.indices,
                    dropout=0.0,
                    causal=True,
                    training=False,
                    softmax_scale=0.25,
                )
            self.assertIs(actual[0], self.out)
            self.assertIs(actual[1], self.lse)
            kwargs = function.call_args.kwargs
            self.assertIs(kwargs["startend_row_indices"], self.indices)
            self.assertEqual("softmax_scale" in kwargs, version == 3)

        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=3),
            self.assertRaisesRegex(NotImplementedError, "dropout"),
        ):
            sink_impl._flashmask_attention_forward_dispatch(
                self.q, self.k, self.v, self.indices, dropout=0.1
            )

    def test_forward_fa4_and_unsupported(self):
        from paddlefleet.transformer import sink_impl

        kernel = mock.Mock(return_value=(self.out, self.lse))
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=4),
            mock.patch.object(sink_impl, "_flash_attn_fwd", kernel),
        ):
            actual = sink_impl._flashmask_attention_forward_dispatch(
                self.q, self.k, self.v, self.indices, softmax_scale=0.3
            )
        self.assertIs(actual[0], self.out)
        self.assertIs(actual[1], self.lse)
        self.assertIs(
            kernel.call_args.kwargs["startend_row_indices"], self.indices
        )

        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=7),
            self.assertRaisesRegex(ValueError, "Unsupported"),
        ):
            sink_impl._flashmask_attention_forward_dispatch(
                self.q, self.k, self.v, self.indices
            )

    def test_backward_fa2_mock(self):
        from paddlefleet.transformer import sink_impl

        grads = (self.q + 1, self.k + 2, self.v + 3)
        op = mock.Mock(return_value=grads)
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=2),
            mock.patch.object(
                sink_impl,
                "_C_ops",
                SimpleNamespace(flashmask_attention_grad=op),
            ),
            mock.patch.object(
                sink_impl.paddle.base,
                "libpaddle",
                SimpleNamespace(
                    pir=SimpleNamespace(
                        ops=SimpleNamespace(flashmask_attention_grad=object())
                    )
                ),
            ),
        ):
            actual = sink_impl._flashmask_attention_backward_dispatch(
                self.grad,
                self.q,
                self.k,
                self.v,
                self.out,
                self.lse,
                self.indices,
            )
        for returned, expected in zip(actual, grads):
            self.assertIs(returned, expected)
        self.assertIs(op.call_args.args[3], self.indices)

    def test_backward_fa3_signature_variants(self):
        from paddlefleet.transformer import sink_impl

        grads = (self.q + 1, self.k + 2, self.v + 3)
        for parameter, expected_count in (
            ("group", 12),
            ("block_mask", 10),
            ("legacy", 9),
        ):
            op = mock.Mock(return_value=grads)
            parameters = {} if parameter == "legacy" else {parameter: object()}
            with (
                self.subTest(parameter=parameter),
                mock.patch.object(sink_impl, "_get_fa_version", return_value=3),
                mock.patch.object(
                    sink_impl,
                    "_C_ops",
                    SimpleNamespace(flashmask_attention_v2_grad=op),
                ),
                mock.patch.object(
                    sink_impl.paddle.base,
                    "libpaddle",
                    SimpleNamespace(
                        pir=SimpleNamespace(
                            ops=SimpleNamespace(
                                flashmask_attention_v2_grad=object()
                            )
                        )
                    ),
                ),
                mock.patch.object(
                    sink_impl.inspect,
                    "signature",
                    return_value=SimpleNamespace(parameters=parameters),
                ),
            ):
                actual = sink_impl._flashmask_attention_backward_dispatch(
                    self.grad,
                    self.q,
                    self.k,
                    self.v,
                    self.out,
                    self.lse,
                    self.indices,
                    softmax_scale=0.4,
                )
            for returned, expected in zip(actual, grads):
                self.assertIs(returned, expected)
            self.assertEqual(len(op.call_args.args), expected_count)
            scale_index = -4 if parameter == "group" else -2
            self.assertEqual(op.call_args.args[scale_index], 0.4)

    def test_backward_fa4_builds_flashmask_info(self):
        from paddlefleet.transformer import sink_impl

        info = object()
        info_factory = mock.Mock(return_value=info)
        kernel = mock.Mock(return_value=(self.q, self.k, self.v))
        with (
            mock.patch.object(sink_impl, "_get_fa_version", return_value=4),
            mock.patch.object(sink_impl, "_FlashMaskInfoPaddle", info_factory),
            mock.patch.object(sink_impl, "_flash_attn_bwd", kernel),
            mock.patch.object(
                sink_impl.paddle,
                "get_flags",
                return_value={"FLAGS_cudnn_deterministic": False},
            ),
        ):
            sink_impl._flashmask_attention_backward_dispatch(
                self.grad,
                self.q,
                self.k,
                self.v,
                self.out,
                self.lse,
                self.indices,
                causal=True,
            )
        info_factory.assert_called_once_with(
            startend_row_indices=self.indices, is_causal=True
        )
        self.assertIs(kernel.call_args.args[6], info)


class TestFlashMaskSinkPyLayerMocked(unittest.TestCase):
    """Exercise sink correction and gradients with fake contexts and kernels."""

    class FakeContext:
        def save_for_backward(self, *values):
            self.values = values

        def saved_tensor(self):
            return self.values

    def setUp(self):
        self.q = paddle.zeros([1, 2, 4, 2], dtype=paddle.float32)
        self.k = paddle.ones([1, 2, 2, 2], dtype=paddle.float32)
        self.v = paddle.full([1, 2, 2, 2], 2.0, dtype=paddle.float32)
        self.sink = paddle.zeros([4], dtype=paddle.float32)
        self.sink.stop_gradient = False

    def test_forward_sink_correction_gqa_and_saved_state(self):
        from paddlefleet.transformer import sink_impl

        raw = paddle.ones_like(self.q)
        lse = paddle.full([1, 4, 3], math.log(2.0), dtype=paddle.float32)
        ctx = self.FakeContext()
        dispatch = mock.Mock(return_value=(raw, lse))
        with mock.patch.object(
            sink_impl, "_flash_attention_forward_dispatch", dispatch
        ):
            result = sink_impl.FlashMaskSinkPyLayer.forward(
                ctx,
                self.q,
                self.k,
                self.v,
                self.sink,
                None,
                dropout=0.2,
                causal=True,
                training=False,
                softmax_scale=0.3,
            )
        np.testing.assert_allclose(result.numpy(), np.full(result.shape, 2 / 3))
        self.assertEqual(list(dispatch.call_args.args[1].shape), [1, 2, 4, 2])
        self.assertEqual(list(dispatch.call_args.args[2].shape), [1, 2, 4, 2])
        self.assertEqual(ctx.num_key_value_groups, 2)
        self.assertEqual(ctx.softmax_scale, 0.3)
        self.assertEqual(list(ctx.values[6].shape), [1, 4, 2])

    def test_backward_merges_gqa_and_computes_sink_gradient(self):
        from paddlefleet.transformer import sink_impl

        ctx = self.FakeContext()
        with mock.patch.object(
            sink_impl,
            "_flash_attention_forward_dispatch",
            return_value=(
                paddle.ones_like(self.q),
                paddle.full([1, 4, 2], math.log(2.0)),
            ),
        ):
            output = sink_impl.FlashMaskSinkPyLayer.forward(
                ctx, self.q, self.k, self.v, self.sink, None
            )

        repeated_k = paddle.arange(16, dtype=paddle.float32).reshape(
            [1, 2, 4, 2]
        )
        repeated_v = repeated_k + 20
        kernel = mock.Mock(
            return_value=(paddle.ones_like(self.q), repeated_k, repeated_v)
        )
        with mock.patch.object(
            sink_impl, "_flash_attention_backward_dispatch", kernel
        ):
            grads = sink_impl.FlashMaskSinkPyLayer.backward(
                ctx, paddle.ones_like(output)
            )
        self.assertEqual(len(grads), 4)
        np.testing.assert_array_equal(
            grads[1].numpy(), repeated_k.numpy().reshape(1, 2, 2, 2, 2).sum(3)
        )
        np.testing.assert_array_equal(
            grads[2].numpy(), repeated_v.numpy().reshape(1, 2, 2, 2, 2).sum(3)
        )
        np.testing.assert_allclose(
            grads[3].numpy(), np.full([4], -8 / 9), atol=1e-6
        )
        self.assertEqual(list(kernel.call_args.args[2].shape), [1, 2, 4, 2])

    def test_flashmask_forward_backward_and_fixed_sink(self):
        from paddlefleet.transformer import sink_impl

        indices = paddle.zeros([1, 1, 2, 1], dtype=paddle.int32)
        ctx = self.FakeContext()
        self.sink.stop_gradient = True
        forward = mock.Mock(
            return_value=(
                paddle.ones_like(self.q),
                paddle.zeros([1, 4, 2], dtype=paddle.float32),
            )
        )
        with mock.patch.object(
            sink_impl, "_flashmask_attention_forward_dispatch", forward
        ):
            output = sink_impl.FlashMaskSinkPyLayer.forward(
                ctx, self.q, self.k, self.v, self.sink, indices
            )
        backward = mock.Mock(
            return_value=(
                paddle.ones_like(self.q),
                paddle.ones_like(self.k),
                paddle.ones_like(self.v),
            )
        )
        with mock.patch.object(
            sink_impl, "_flashmask_attention_backward_dispatch", backward
        ):
            grads = sink_impl.FlashMaskSinkPyLayer.backward(
                ctx, paddle.ones_like(output)
            )
        self.assertEqual(len(grads), 5)
        self.assertIsNone(grads[3])
        self.assertIs(grads[4], None)
        self.assertIs(forward.call_args.args[3], indices)
        self.assertIs(backward.call_args.args[6], indices)

    def test_forward_validation(self):
        from paddlefleet.transformer import sink_impl

        cases = (
            (
                (self.q.reshape([2, 4, 2]), self.k, self.v, self.sink),
                "Query must be 4D",
            ),
            (
                (self.q, self.k, self.v, self.sink.reshape([2, 2])),
                "Sink must be 1D",
            ),
            ((self.q, self.k, self.v[..., :1], self.sink), "value head_dim"),
            (
                (self.q, self.k[:, :, :1], self.v, self.sink),
                "same number of heads",
            ),
            (
                (self.q, self.k, self.v, paddle.zeros([3])),
                "Sink parameter size",
            ),
        )
        for args, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex((AssertionError, ValueError), message),
            ):
                sink_impl.FlashMaskSinkPyLayer.forward(
                    self.FakeContext(), *args, None
                )

        indices = paddle.zeros([1, 1, 2, 1], dtype=paddle.int32)
        with self.assertRaisesRegex(AssertionError, "dense mask"):
            sink_impl.FlashMaskSinkPyLayer.forward(
                self.FakeContext(),
                self.q,
                self.k,
                self.v,
                self.sink,
                indices,
                attention_mask=paddle.zeros([1]),
            )

    def test_sink_attention_forward_forwards_arguments(self):
        from paddlefleet.transformer import sink_impl

        indices = paddle.zeros([1, 1, 2, 1], dtype=paddle.int32)
        apply = mock.Mock(return_value="result")
        with mock.patch.object(sink_impl.FlashMaskSinkPyLayer, "apply", apply):
            result = sink_impl.sink_attention_forward(
                self.q,
                self.k,
                self.v,
                self.sink,
                attention_mask="mask",
                startend_row_indices=indices,
                dropout_p=0.2,
                softmax_scale=0.7,
                causal=True,
                training=False,
            )
        self.assertEqual(result, "result")
        self.assertEqual(
            apply.call_args.args, (self.q, self.k, self.v, self.sink, indices)
        )
        self.assertEqual(
            apply.call_args.kwargs,
            {
                "attention_mask": "mask",
                "dropout": 0.2,
                "causal": True,
                "return_softmax": False,
                "training": False,
                "softmax_scale": 0.7,
            },
        )


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestCuteFlashmaskSink(unittest.TestCase):
    """Low-level cute flashmask_attention numerical correctness with sink."""

    def _run(
        self,
        batch_size,
        seq_len,
        nheads,
        nheads_kv,
        d,
        causal,
        use_sink,
    ):
        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )

        paddle.seed(SEED)
        np.random.seed(SEED)

        q_ref = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        k_ref = paddle.randn([batch_size, seq_len, nheads_kv, d], dtype=DTYPE)
        v_ref = paddle.randn([batch_size, seq_len, nheads_kv, d], dtype=DTYPE)
        for t in (q_ref, k_ref, v_ref):
            t.stop_gradient = False

        q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]
        for t in (q, k, v):
            t.stop_gradient = False

        if use_sink:
            sink_ref = paddle.randn([nheads], dtype=DTYPE)
            sink_ref.stop_gradient = False
            sink = sink_ref.detach().clone()
            sink.stop_gradient = False
        else:
            sink_ref = None
            sink = None

        startend_row_indices = _startend_row_indices(
            batch_size, seq_len, causal
        )
        attn_bias = _attn_bias_from_indices(
            startend_row_indices, seq_len, nheads, causal
        )

        out_ref = attention_ref_with_sink(
            q_ref, k_ref, v_ref, attn_bias, sink_ref
        )

        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            causal=causal,
            learnable_sink=sink,
        )

        # Forward tolerance: 2x the bf16->fp32 round-trip noise of the reference.
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        max_diff = (out - out_ref).abs().max().item()
        self.assertLessEqual(
            max_diff,
            2e-2 + fwd_atol,
            f"fwd max diff {max_diff} too large (atol={fwd_atol})",
        )

        # Backward
        g = paddle.randn(out.shape, dtype=out.dtype)
        out.backward(g.clone())
        out_ref.backward(g.clone())

        for name, a, b in (
            ("dq", q.grad, q_ref.grad),
            ("dk", k.grad, k_ref.grad),
            ("dv", v.grad, v_ref.grad),
        ):
            atol = 2 * (b + 0.3 - 0.3 - b).abs().max().item()
            diff = (a - b).abs().max().item()
            self.assertLessEqual(
                diff, 5e-2 + atol, f"{name} max diff {diff} too large"
            )

        if use_sink:
            self.assertIsNotNone(sink.grad)
            # Kernel computes dsink in fp32 then casts back to sink dtype (bf16).
            self.assertEqual(sink.grad.dtype, DTYPE)
            self.assertEqual(list(sink.grad.shape), [nheads])

    def test_causal_with_trainable_sink(self):
        self._run(2, 256, 4, 4, 128, causal=True, use_sink=True)

    def test_noncausal_with_trainable_sink(self):
        self._run(2, 256, 4, 4, 128, causal=False, use_sink=True)

    def test_sink_none_matches_plain_softmax(self):
        self._run(2, 256, 4, 4, 128, causal=True, use_sink=False)

    def test_gqa_with_sink(self):
        # nheads_kv < nheads exercises the GQA repeat path with sink.
        self._run(2, 256, 8, 2, 128, causal=True, use_sink=True)

    def test_small_head_dim_sink(self):
        self._run(2, 128, 4, 4, 64, causal=False, use_sink=True)

    def test_fixed_sink_backward_returns_dsink_slot(self):
        # A FIXED (stop_gradient=True) bf16 sink is a valid forward input, but
        # FlashMaskFunc.backward chooses its return arity from
        # ``learnable_sink is None`` alone (interface.py:1770-1772) -- it does
        # NOT consult stop_gradient. So a non-None fixed sink makes backward
        # return the 4-tuple (dq, dk, dv, dsink); Paddle's PyLayer then rejects
        # it because the sink forward input has stop_gradient=True and its slot
        # must be None. We assert the forward is still numerically correct and
        # that backward raises this ValueError, documenting the limitation.
        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )

        paddle.seed(SEED)
        np.random.seed(SEED)
        b, s, h, d = 2, 256, 4, 128
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        k = paddle.randn([b, s, h, d], dtype=DTYPE)
        v = paddle.randn([b, s, h, d], dtype=DTYPE)
        for t in (q, k, v):
            t.stop_gradient = False

        # off-by-one == a fixed all-zeros sink logit (stop_gradient=True).
        sink = paddle.zeros([h], dtype=DTYPE)
        sink.stop_gradient = True

        idx = _startend_row_indices(b, s, causal=True)
        attn_bias = _attn_bias_from_indices(idx, s, h, causal=True)
        out_ref = attention_ref_with_sink(q, k, v, attn_bias, sink)

        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=idx,
            causal=True,
            learnable_sink=sink,
        )
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        self.assertLessEqual(
            (out - out_ref).abs().max().item(), 2e-2 + fwd_atol
        )

        with self.assertRaises(ValueError):
            out.backward(paddle.randn(out.shape, dtype=out.dtype))

    def test_sink_dtype_assert(self):
        # The cute kernel asserts learnable_sink is bf16; fp32 must raise.
        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )

        paddle.seed(SEED)
        b, s, h, d = 1, 64, 2, 64
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        k = paddle.randn([b, s, h, d], dtype=DTYPE)
        v = paddle.randn([b, s, h, d], dtype=DTYPE)
        sink_fp32 = paddle.zeros([h], dtype=paddle.float32)
        idx = _startend_row_indices(b, s, causal=True)
        with self.assertRaises(AssertionError):
            flashmask_attention(
                q,
                k,
                v,
                startend_row_indices=idx,
                causal=True,
                learnable_sink=sink_fp32,
            )


@unittest.skipUnless(_FA3_SINK_AVAILABLE, _FA3_SKIP_REASON)
class TestFA3SinkAttention(unittest.TestCase):
    """FA3 sink_attention_forward numerical correctness with sink."""

    def _run(
        self,
        batch_size,
        seq_len,
        nheads,
        nheads_kv,
        d,
        causal,
        use_startend=True,
    ):
        from paddlefleet.transformer.sink_impl import sink_attention_forward

        with _flash_attn_version(3):
            paddle.seed(SEED)
            np.random.seed(SEED)

            q_ref = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
            k_ref = paddle.randn(
                [batch_size, seq_len, nheads_kv, d], dtype=DTYPE
            )
            v_ref = paddle.randn(
                [batch_size, seq_len, nheads_kv, d], dtype=DTYPE
            )
            for t in (q_ref, k_ref, v_ref):
                t.stop_gradient = False

            q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]
            for t in (q, k, v):
                t.stop_gradient = False

            sink_ref = paddle.randn([nheads], dtype=DTYPE)
            sink_ref.stop_gradient = False
            sink = sink_ref.detach().clone()
            sink.stop_gradient = False

            startend_row_indices = None
            if use_startend:
                startend_row_indices = _startend_row_indices(
                    batch_size, seq_len, causal
                )
                attn_bias = _attn_bias_from_indices(
                    startend_row_indices, seq_len, nheads, causal
                )
            elif causal:
                causal_mask = np.triu(
                    np.ones((seq_len, seq_len), dtype=bool), k=1
                )
                attn_bias = paddle.to_tensor(
                    np.where(causal_mask, -np.inf, 0.0).astype("float32")
                ).reshape([1, 1, seq_len, seq_len])
            else:
                attn_bias = paddle.zeros(
                    [1, 1, seq_len, seq_len], dtype=paddle.float32
                )

            out_ref = attention_ref_with_sink(
                q_ref, k_ref, v_ref, attn_bias, sink_ref
            )
            out = sink_attention_forward(
                q,
                k,
                v,
                sink=sink,
                startend_row_indices=startend_row_indices,
                causal=causal,
            )

            fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
            max_diff = (out - out_ref).abs().max().item()
            self.assertLessEqual(
                max_diff,
                2e-2 + fwd_atol,
                f"fwd max diff {max_diff} too large (atol={fwd_atol})",
            )

            g = paddle.randn(out.shape, dtype=out.dtype)
            out.backward(g.clone())
            out_ref.backward(g.clone())

            for name, a, b in (
                ("dq", q.grad, q_ref.grad),
                ("dk", k.grad, k_ref.grad),
                ("dv", v.grad, v_ref.grad),
            ):
                atol = 2 * (b + 0.3 - 0.3 - b).abs().max().item()
                diff = (a - b).abs().max().item()
                self.assertLessEqual(
                    diff, 5e-2 + atol, f"{name} max diff {diff} too large"
                )

            self.assertIsNotNone(sink.grad)
            self.assertEqual(list(sink.grad.shape), [nheads])

    def test_causal_with_trainable_sink(self):
        self._run(2, 256, 4, 4, 128, causal=True)

    def test_noncausal_with_trainable_sink(self):
        self._run(2, 256, 4, 4, 128, causal=False)

    def test_dense_causal_with_trainable_sink(self):
        self._run(2, 256, 4, 4, 128, causal=True, use_startend=False)

    def test_gqa_with_sink(self):
        self._run(2, 256, 8, 2, 128, causal=True)

    def test_small_head_dim_sink(self):
        self._run(2, 128, 4, 4, 64, causal=False)

    def test_fixed_sink_backward_no_dsink(self):
        from paddlefleet.transformer.sink_impl import sink_attention_forward

        with _flash_attn_version(3):
            paddle.seed(SEED)
            b, s, h, d = 2, 256, 4, 128
            q = paddle.randn([b, s, h, d], dtype=DTYPE)
            k = paddle.randn([b, s, h, d], dtype=DTYPE)
            v = paddle.randn([b, s, h, d], dtype=DTYPE)
            for t in (q, k, v):
                t.stop_gradient = False
            sink = paddle.zeros([h], dtype=DTYPE)
            sink.stop_gradient = True
            idx = _startend_row_indices(b, s, causal=True)
            out = sink_attention_forward(
                q, k, v, sink=sink, startend_row_indices=idx, causal=True
            )
            out.backward(paddle.randn(out.shape, dtype=out.dtype))
            self.assertIsNone(sink.grad)
            self.assertIsNotNone(q.grad)


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestFacadeFlashmaskSink(unittest.TestCase):
    """paddlefleet_ops.flash_mask_facade.flashmask_attention sink forwarding.

    The PR adds learnable_sink to the facade signature and forwards it to the
    cute kernel. Note two facade quirks exercised here:
      - startend_row_indices.clone() is called unconditionally -> must pass a
        non-None tensor.
      - return_softmax_lse reshapes lse to [bsz, q_len] -> only valid nheads==1.
    """

    def _run(self, nheads, nheads_kv, use_sink, return_lse=False, causal=False):
        from paddlefleet_ops.flash_mask_facade import flashmask_attention

        paddle.seed(SEED)
        b, s, d = 2, 128, 128
        q = paddle.randn([b, s, nheads, d], dtype=DTYPE)
        k = paddle.randn([b, s, nheads_kv, d], dtype=DTYPE)
        v = paddle.randn([b, s, nheads_kv, d], dtype=DTYPE)
        for t in (q, k, v):
            t.stop_gradient = False

        sink = None
        if use_sink:
            sink = paddle.randn([nheads], dtype=DTYPE)
            sink.stop_gradient = False

        idx = _startend_row_indices(b, s, causal)
        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=idx,
            causal=causal,
            return_softmax_lse=return_lse,
            learnable_sink=sink,
        )
        if return_lse:
            out, lse = out
            self.assertEqual(list(lse.shape), [b, s])
        self.assertEqual(list(out.shape), [b, s, nheads, d])

        # Reference + numerical check (facade should be a thin wrapper).
        attn_bias = _attn_bias_from_indices(idx, s, nheads, causal)
        out_ref = attention_ref_with_sink(q, k, v, attn_bias, sink)
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        diff = (out - out_ref).abs().max().item()
        self.assertLessEqual(diff, 2e-2 + fwd_atol)

        out.sum().backward()
        self.assertIsNotNone(q.grad)
        if use_sink:
            self.assertIsNotNone(sink.grad)

    def test_facade_with_sink(self):
        self._run(nheads=4, nheads_kv=4, use_sink=True)

    def test_facade_sink_none(self):
        self._run(nheads=4, nheads_kv=4, use_sink=False)

    def test_facade_lse_single_head(self):
        # lse reshape [bsz, q_len] only works for nheads == 1 (facade quirk 2).
        self._run(nheads=1, nheads_kv=1, use_sink=True, return_lse=True)

    def test_facade_gqa_with_sink(self):
        self._run(nheads=8, nheads_kv=2, use_sink=True)


def _make_config(
    softmax_type="vanilla",
    add_full_attention_sink_bias=False,
    add_swa_attention_sink_bias=False,
    num_attention_heads=4,
    head_dim=128,
    hidden_size=512,
):
    """TransformerConfig for DotProductAttention sink tests (bf16, no CP)."""
    from paddlefleet.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    config.head_dim = head_dim
    config.num_key_value_heads = num_attention_heads
    config.softmax_scale = None
    config.use_bias = False
    config.context_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.sliding_window = None
    config.window_attn_skip_freq = None
    config.fp16 = False
    config.bf16 = True
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.softmax_type = softmax_type
    config.add_full_attention_sink_bias = add_full_attention_sink_bias
    config.add_swa_attention_sink_bias = add_swa_attention_sink_bias
    # learnable softmax_offset is created with config.params_dtype; the cute
    # kernel requires bf16 sink, so params_dtype must be bf16.
    config.params_dtype = paddle.bfloat16
    config.perform_initialization = False
    config.flashmask_use_varlen = False
    config.experimental_dataflow = False
    return config


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestDotProductAttentionSinkInit(unittest.TestCase):
    """softmax_offset construction across softmax_type / sink-bias promotion."""

    def _build(self, **cfg_kwargs):
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType

        config = _make_config(**cfg_kwargs)
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def test_vanilla_offset_none(self):
        attn = self._build(softmax_type="vanilla")
        self.assertIsNone(attn.softmax_offset)

    def test_offbyone_zeros(self):
        attn = self._build(softmax_type="off-by-one")
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(
            list(attn.softmax_offset.shape),
            [attn.num_attention_heads_per_partition],
        )
        self.assertTrue(bool((attn.softmax_offset == 0).all().item()))

    def test_learnable_is_parameter(self):
        attn = self._build(softmax_type="learnable")
        self.assertIsNotNone(attn.softmax_offset)
        # A create_parameter() result is trainable (stop_gradient False).
        self.assertFalse(attn.softmax_offset.stop_gradient)
        self.assertEqual(attn.softmax_offset.dtype, paddle.bfloat16)

    def test_full_attention_sink_bias_promotes_to_learnable(self):
        # add_full_attention_sink_bias + non-SWA -> promoted to learnable.
        attn = self._build(
            softmax_type="vanilla",
            add_full_attention_sink_bias=True,
        )
        self.assertIsNotNone(attn.softmax_offset)
        self.assertFalse(attn.softmax_offset.stop_gradient)


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestDotProductAttentionSinkForward(unittest.TestCase):
    """Full fwd/bwd through the flashmask sink branch of DotProductAttention."""

    def _run(self, softmax_type, **cfg_kwargs):
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType

        paddle.seed(SEED)
        config = _make_config(softmax_type=softmax_type, **cfg_kwargs)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        b, s = 2, 64
        h = config.num_attention_heads
        d = config.head_dim
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        k = paddle.randn([b, s, h, d], dtype=DTYPE)
        v = paddle.randn([b, s, h, d], dtype=DTYPE)
        for t in (q, k, v):
            t.stop_gradient = False

        idx = _startend_row_indices(b, s, causal=True)
        out = attn(
            query=q,
            key=k,
            value=v,
            attention_mask=None,
            attn_mask_startend_row_indices=idx,
            attn_mask_type=AttnMaskType.causal,
        )
        self.assertEqual(list(out.shape), [b, s, h * d])

        out.sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

        if softmax_type == "learnable":
            self.assertIsNotNone(attn.softmax_offset.grad)
            self.assertEqual(attn.softmax_offset.grad.dtype, paddle.bfloat16)

    def test_forward_vanilla(self):
        self._run("vanilla")

    def test_forward_learnable_sink(self):
        self._run("learnable")

    def test_forward_offbyone(self):
        # off-by-one builds an fp32 zeros offset; the cute kernel asserts bf16,
        # so this path is expected to raise on the fa4 sink branch.
        with self.assertRaises(AssertionError):
            self._run("off-by-one")


@unittest.skipUnless(_FA3_SINK_AVAILABLE, _FA3_SKIP_REASON)
class TestDotProductAttentionFA3SinkForward(unittest.TestCase):
    """Full fwd/bwd through the FA3 sink branch of DotProductAttention."""

    def _run(self, softmax_type, use_startend=True, **cfg_kwargs):
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType

        with _flash_attn_version(3):
            paddle.seed(SEED)
            config = _make_config(softmax_type=softmax_type, **cfg_kwargs)
            attn = DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )

            b, s = 2, 64
            h = config.num_attention_heads
            d = config.head_dim
            q = paddle.randn([b, s, h, d], dtype=DTYPE)
            k = paddle.randn([b, s, h, d], dtype=DTYPE)
            v = paddle.randn([b, s, h, d], dtype=DTYPE)
            for t in (q, k, v):
                t.stop_gradient = False

            idx = (
                _startend_row_indices(b, s, causal=True)
                if use_startend
                else None
            )
            out = attn(
                query=q,
                key=k,
                value=v,
                attention_mask=None,
                attn_mask_startend_row_indices=idx,
                attn_mask_type=AttnMaskType.causal,
            )
            self.assertEqual(list(out.shape), [b, s, h * d])

            out.sum().backward()
            self.assertIsNotNone(q.grad)
            self.assertIsNotNone(k.grad)
            self.assertIsNotNone(v.grad)

            if softmax_type == "learnable":
                self.assertIsNotNone(attn.softmax_offset.grad)
                self.assertEqual(
                    attn.softmax_offset.grad.dtype, paddle.bfloat16
                )
            elif softmax_type == "off-by-one":
                self.assertIsNone(attn.softmax_offset.grad)

    def test_forward_vanilla(self):
        self._run("vanilla")

    def test_forward_learnable_sink(self):
        self._run("learnable")

    def test_forward_learnable_sink_dense_causal(self):
        self._run("learnable", use_startend=False)

    def test_forward_learnable_sink_additive_dense_mask(self):
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType

        with _flash_attn_version(3):
            paddle.seed(SEED)
            config = _make_config(softmax_type="learnable")
            attn = DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )

            b, s = 2, 64
            h = config.num_attention_heads
            d = config.head_dim
            q = paddle.randn([b, s, h, d], dtype=DTYPE)
            k = paddle.randn([b, s, h, d], dtype=DTYPE)
            v = paddle.randn([b, s, h, d], dtype=DTYPE)
            for t in (q, k, v):
                t.stop_gradient = False

            causal_mask = np.triu(np.ones((s, s), dtype=bool), k=1)
            attention_mask = paddle.to_tensor(
                np.where(causal_mask, -1e6, 0.0).astype("float32")
            ).reshape([1, 1, s, s])
            out = attn(
                query=q,
                key=k,
                value=v,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=None,
                attn_mask_type=AttnMaskType.causal,
            )
            self.assertEqual(list(out.shape), [b, s, h * d])

            out.sum().backward()
            self.assertIsNotNone(q.grad)
            self.assertIsNotNone(k.grad)
            self.assertIsNotNone(v.grad)
            self.assertIsNotNone(attn.softmax_offset.grad)
            self.assertEqual(attn.softmax_offset.grad.dtype, paddle.bfloat16)

    def test_forward_learnable_sink_value_head_dim_padding(self):
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType

        with _flash_attn_version(3):
            paddle.seed(SEED)
            b, s, h = 2, 64, 4
            qk_dim, v_dim = 192, 128
            config = _make_config(
                softmax_type="learnable",
                num_attention_heads=h,
                head_dim=qk_dim,
                hidden_size=h * qk_dim,
            )
            attn = DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                k_channels=qk_dim,
                v_channels=v_dim,
            )

            q = paddle.randn([b, s, h, qk_dim], dtype=DTYPE)
            k = paddle.randn([b, s, h, qk_dim], dtype=DTYPE)
            v = paddle.randn([b, s, h, v_dim], dtype=DTYPE)
            for t in (q, k, v):
                t.stop_gradient = False

            out = attn(
                query=q,
                key=k,
                value=v,
                attention_mask=None,
                attn_mask_startend_row_indices=None,
                attn_mask_type=AttnMaskType.causal,
            )
            self.assertEqual(list(out.shape), [b, s, h * v_dim])

            out.sum().backward()
            self.assertIsNotNone(q.grad)
            self.assertIsNotNone(k.grad)
            self.assertIsNotNone(v.grad)
            self.assertIsNotNone(attn.softmax_offset.grad)
            self.assertEqual(attn.softmax_offset.grad.dtype, paddle.bfloat16)

    def test_decode_shape_raises_for_fa3_sink(self):
        from paddlefleet.transformer.sink_impl import prepare_fa3_sink_attention

        with _flash_attn_version(3):
            b, h, d = 2, 4, 128
            q = paddle.randn([b, 1, h, d], dtype=DTYPE)
            k = paddle.randn([b, 8, h, d], dtype=DTYPE)
            v = paddle.randn([b, 8, h, d], dtype=DTYPE)
            sink = paddle.randn([h], dtype=DTYPE)
            with self.assertRaisesRegex(
                NotImplementedError,
                "does not support KV-cache decode with unequal q/k/v sequence lengths",
            ):
                prepare_fa3_sink_attention(
                    q, k, v, sink, attention_mask=None, causal=False
                )

    def test_forward_offbyone(self):
        self._run("off-by-one")


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestRefinedRecomputeFlashMaskSink(unittest.TestCase):
    """Refined-recompute (rr) non-CP path with learnable_sink.

    The rr FlashMask attention keys off ``framework._dygraph_tracer()._has_grad``
    to pick between two forward passes: the first (``_has_grad`` False) runs the
    real cute kernel under no_grad and stashes tensors; the second (``_has_grad``
    True) rebuilds the graph via ``FlashMaskAttnFunctor``, whose custom backward
    computes the grads. We drive both passes manually so the test exercises the
    rr functor's fwd/bwd directly, and compare against the non-rr cute
    ``flashmask_attention`` sink path.
    """

    def _run(self, causal, use_sink, sink_trainable=True):
        from paddle import framework
        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )

        from paddlefleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        paddle.seed(SEED)
        np.random.seed(SEED)
        b, s, h, d = 2, 256, 4, 128

        q_ref = paddle.randn([b, s, h, d], dtype=DTYPE)
        k_ref = paddle.randn([b, s, h, d], dtype=DTYPE)
        v_ref = paddle.randn([b, s, h, d], dtype=DTYPE)
        for t in (q_ref, k_ref, v_ref):
            t.stop_gradient = False

        q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]
        for t in (q, k, v):
            t.stop_gradient = False

        if use_sink:
            sink_ref = paddle.randn([h], dtype=DTYPE)
            sink_ref.stop_gradient = not sink_trainable
            sink = sink_ref.detach().clone()
            sink.stop_gradient = not sink_trainable
        else:
            sink_ref = sink = None

        idx = _startend_row_indices(b, s, causal)

        # Non-rr reference through the plain cute flashmask_attention.
        out_ref = flashmask_attention(
            q_ref,
            k_ref,
            v_ref,
            startend_row_indices=idx,
            causal=causal,
            learnable_sink=sink_ref,
        )

        # Drive the rr two-pass mechanism MANUALLY (no recompute) so the test
        # exercises FlashMaskAttnFunctor's fwd/bwd directly. The first call runs
        # with _has_grad False (stash pass, no grad tracked); the second runs
        # with _has_grad True (graph-rebuild pass) so `out` carries the functor
        # grad node and out.backward() invokes the rr custom backward directly.
        rr_attn = RefinedRcomputeFlashMaskAttention()
        tracer = framework._dygraph_tracer()
        prev_has_grad = tracer._has_grad

        tracer._has_grad = False
        try:
            rr_attn(q, k, v, idx, causal=causal, learnable_sink=sink)
        finally:
            tracer._has_grad = prev_has_grad

        tracer._has_grad = True
        try:
            out = rr_attn(q, k, v, idx, causal=causal, learnable_sink=sink)
        finally:
            tracer._has_grad = prev_has_grad
        self.assertEqual(list(out.shape), [b, s, h, d])

        # rr should match the non-rr sink path exactly (same kernel).
        max_diff = (out - out_ref).abs().max().item()
        self.assertLessEqual(
            max_diff, 1e-2, f"rr fwd max diff {max_diff} vs non-rr too large"
        )

        # A fixed (stop_gradient) sink makes the non-rr flashmask_attention
        # PyLayer return a dsink slot that Paddle rejects on backward (a known
        # limitation of that op, covered by test_fixed_sink_backward_returns_
        # dsink_slot). So only run the reference backward when the sink is not a
        # fixed tensor; the rr backward is always exercised below.
        ref_backward_safe = not (use_sink and not sink_trainable)

        g = paddle.randn(out.shape, dtype=out.dtype)
        out.backward(g.clone())
        if ref_backward_safe:
            out_ref.backward(g.clone())
            for name, a, ref in (
                ("dq", q.grad, q_ref.grad),
                ("dk", k.grad, k_ref.grad),
                ("dv", v.grad, v_ref.grad),
            ):
                self.assertIsNotNone(a, f"{name} grad is None")
                diff = (a - ref).abs().max().item()
                self.assertLessEqual(
                    diff, 2e-2, f"rr {name} max diff {diff} vs non-rr too large"
                )
        else:
            # Still assert the rr backward produced q/k/v grads.
            for name, a in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
                self.assertIsNotNone(a, f"{name} grad is None")

        if use_sink and sink_trainable:
            self.assertIsNotNone(sink.grad)
            self.assertEqual(sink.grad.dtype, DTYPE)
            self.assertEqual(list(sink.grad.shape), [h])
            sink_diff = (sink.grad - sink_ref.grad).abs().max().item()
            self.assertLessEqual(
                sink_diff,
                2e-2,
                f"rr dsink max diff {sink_diff} vs non-rr too large",
            )
        elif use_sink and not sink_trainable:
            # Fixed sink is stop_gradient -> rr returns no sink grad.
            self.assertIsNone(sink.grad)

    def test_rr_causal_trainable_sink(self):
        self._run(causal=True, use_sink=True)

    def test_rr_noncausal_trainable_sink(self):
        self._run(causal=False, use_sink=True)

    def test_rr_sink_none(self):
        self._run(causal=True, use_sink=False)

    def test_rr_fixed_sink_no_grad(self):
        self._run(causal=True, use_sink=True, sink_trainable=False)

    def test_rr_non_fa4_sink_raises(self):
        # Sink is only supported on the fa_version==4 cute backend; forcing v3
        # must make the rr entry point reject a non-None sink.
        from paddlefleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        paddle.seed(SEED)
        b, s, h, d = 2, 128, 4, 128
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        idx = _startend_row_indices(b, s, causal=True)
        sink = paddle.randn([h], dtype=DTYPE)
        rr_attn = RefinedRcomputeFlashMaskAttention()

        old = paddle.get_flags(["FLAGS_flash_attn_version"])[
            "FLAGS_flash_attn_version"
        ]
        paddle.set_flags({"FLAGS_flash_attn_version": 3})
        try:
            with self.assertRaises(NotImplementedError):
                rr_attn.forward(q, q, q, idx, learnable_sink=sink)
        finally:
            paddle.set_flags({"FLAGS_flash_attn_version": old})


if __name__ == "__main__":
    unittest.main()
