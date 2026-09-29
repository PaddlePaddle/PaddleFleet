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

"""Behavior tests for paddlefleet.transformer.attention.

Covered production behavior:
  * ``_md5`` computes an MD5 hex digest over the *detached* tensor's raw bytes.
  * ``SelfAttention.get_query_key_value_tensors`` (default, non-experimental
    path) splits the fused QKV projection into query/key/value with the correct
    per-group -> per-head reshape, applies q_norm/k_norm and v_scale, and returns
    an extra gate tensor when gated attention is enabled.

The method under test reads a handful of pre-computed integer attributes that
``Attention.__init__`` would otherwise set. To keep the test on CPU and avoid
the heavy real constructor (process-group / Fleet init), we invoke the *real*
unbound method with a lightweight ``self`` stub that carries exactly those
attributes. The method body itself is executed unchanged; only genuine
not-under-test collaborators (``qkv_proj``, ``q_norm``, ``k_norm``) are stubbed
with distinguishable, input-dependent responses so their consumption can be
observed. No paddle/GPU numerics are required.
"""

import hashlib
import types
import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.attention import (
        SelfAttention,
        SelfAttentionSublayersSpec,
        _md5,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)


class _RecordingNorm:
    """Not-under-test collaborator: records its input and returns a
    distinguishable, input-dependent tensor (x + 1000) so both the exact
    argument fed to the norm and the consumption of its output are observable."""

    def __init__(self):
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return x + 1000.0


def _make_attn_stub(
    *,
    num_attention_heads_per_partition,
    num_query_groups_per_partition,
    hidden_size_per_attention_head,
    value_hidden_size_per_attention_head,
    mixed_qkv,
    gated_attention=False,
    qk_norm_type="per_head",
    q_norm=None,
    k_norm=None,
    v_scale=None,
):
    """Build a ``self`` stub exposing exactly the attributes the default
    (non-experimental) path of ``get_query_key_value_tensors`` reads. ``qkv_proj``
    is a genuine not-under-test collaborator returning a fixed fused tensor."""
    return types.SimpleNamespace(
        qkv_proj=lambda hidden_states: (mixed_qkv, None),
        config=types.SimpleNamespace(
            gpt_model_use_experimental_version=False,
            qk_norm_type=qk_norm_type,
        ),
        gated_attention=gated_attention,
        num_attention_heads_per_partition=num_attention_heads_per_partition,
        num_query_groups_per_partition=num_query_groups_per_partition,
        hidden_size_per_attention_head=hidden_size_per_attention_head,
        value_hidden_size_per_attention_head=value_hidden_size_per_attention_head,
        v_scale=v_scale,
        q_norm=q_norm,
        k_norm=k_norm,
    )


def _call_qkv(stub, split_qkv=True):
    # Invoke the real unbound method with the stub as ``self``; the method body
    # (split boundaries, reshape, norm selection, gate, v_scale) runs unchanged.
    hidden = paddle.zeros([1, 1, 1], dtype="float32")
    return SelfAttention.get_query_key_value_tensors(
        stub, hidden, split_qkv=split_qkv
    )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestMd5(unittest.TestCase):
    """_md5 must digest the tensor's actual detached byte content."""

    def test_matches_independent_hashlib_over_raw_bytes(self):
        # Distinguishable, non-degenerate content; float32 so tensor bytes equal
        # the numpy source bytes exactly (no dtype conversion).
        arr = np.arange(12, dtype=np.float32).reshape([3, 4]) * 0.5 - 1.0
        expected = hashlib.md5(arr.tobytes()).hexdigest()
        t = paddle.to_tensor(arr)
        self.assertEqual(_md5(t), expected)

    def test_content_sensitive(self):
        base = np.arange(6, dtype=np.float32).reshape([2, 3])
        changed = base.copy()
        changed[1, 2] += (
            1.0  # single-element perturbation must change the digest
        )
        self.assertNotEqual(
            _md5(paddle.to_tensor(base)),
            _md5(paddle.to_tensor(changed)),
        )

    def test_dtype_affects_bytes(self):
        # Same numeric values, different dtype -> different raw bytes -> different digest.
        vals = [1.0, 2.0, 3.0, 4.0]
        d32 = _md5(paddle.to_tensor(np.array(vals, dtype=np.float32)))
        d64 = _md5(paddle.to_tensor(np.array(vals, dtype=np.float64)))
        self.assertNotEqual(d32, d64)

    def test_detach_is_load_bearing_for_grad_tensor(self):
        # A tensor with stop_gradient=False cannot go through .numpy() directly;
        # _md5 relies on .detach(). Result must still equal the raw-byte digest.
        arr = np.arange(8, dtype=np.float32)
        expected = hashlib.md5(arr.tobytes()).hexdigest()
        t = paddle.to_tensor(arr)
        t.stop_gradient = False
        self.assertEqual(_md5(t), expected)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestGetQKVSplitDefaultPath(unittest.TestCase):
    """Default (non-experimental) get_query_key_value_tensors split/reshape.

    Layout: num_query_groups=2, num_heads=4 (heads_per_group=2), head_dim=3,
    v_head_dim=3. Fused width per group = q(6)+k(3)+v(3)=12; total last dim=24.
    Expected mapping hand-derived from arange(24)."""

    def _base_stub(self, mixed_qkv, **kw):
        return _make_attn_stub(
            num_attention_heads_per_partition=4,
            num_query_groups_per_partition=2,
            hidden_size_per_attention_head=3,
            value_hidden_size_per_attention_head=3,
            mixed_qkv=mixed_qkv,
            **kw,
        )

    def test_split_and_per_head_reshape_content(self):
        mixed = paddle.arange(24, dtype="float32").reshape([1, 1, 24])
        query, key, value = _call_qkv(self._base_stub(mixed))

        # query: [1,1,2,6] regrouped to per-head [1,1,4,3]
        np.testing.assert_array_equal(
            query.numpy(),
            np.array(
                [[[[0, 1, 2], [3, 4, 5], [12, 13, 14], [15, 16, 17]]]],
                dtype=np.float32,
            ),
        )
        # key stays grouped [1,1,2,3]
        np.testing.assert_array_equal(
            key.numpy(),
            np.array([[[[6, 7, 8], [18, 19, 20]]]], dtype=np.float32),
        )
        # value stays grouped [1,1,2,3]
        np.testing.assert_array_equal(
            value.numpy(),
            np.array([[[[9, 10, 11], [21, 22, 23]]]], dtype=np.float32),
        )

    def test_norms_receive_and_feed_correct_tensors(self):
        mixed = paddle.arange(24, dtype="float32").reshape([1, 1, 24])
        q_norm, k_norm = _RecordingNorm(), _RecordingNorm()
        query, key, value = _call_qkv(
            self._base_stub(mixed, q_norm=q_norm, k_norm=k_norm)
        )

        # q_norm is fed the reshaped per-head query; k_norm the grouped key.
        self.assertEqual(len(q_norm.inputs), 1)
        self.assertEqual(len(k_norm.inputs), 1)
        expected_q_in = np.array(
            [[[[0, 1, 2], [3, 4, 5], [12, 13, 14], [15, 16, 17]]]],
            dtype=np.float32,
        )
        expected_k_in = np.array(
            [[[[6, 7, 8], [18, 19, 20]]]], dtype=np.float32
        )
        np.testing.assert_array_equal(q_norm.inputs[0].numpy(), expected_q_in)
        np.testing.assert_array_equal(k_norm.inputs[0].numpy(), expected_k_in)

        # Outputs of the norms (x + 1000) must actually be consumed / returned.
        np.testing.assert_array_equal(query.numpy(), expected_q_in + 1000.0)
        np.testing.assert_array_equal(key.numpy(), expected_k_in + 1000.0)
        # value untouched by the norm branch.
        np.testing.assert_array_equal(
            value.numpy(),
            np.array([[[[9, 10, 11], [21, 22, 23]]]], dtype=np.float32),
        )

    def test_v_scale_scales_value_only(self):
        mixed = paddle.arange(24, dtype="float32").reshape([1, 1, 24])
        query, key, value = _call_qkv(self._base_stub(mixed, v_scale=2.0))
        np.testing.assert_array_equal(
            value.numpy(),
            np.array([[[[18, 20, 22], [42, 44, 46]]]], dtype=np.float32),
        )
        # query/key are not scaled.
        np.testing.assert_array_equal(
            query.numpy(),
            np.array(
                [[[[0, 1, 2], [3, 4, 5], [12, 13, 14], [15, 16, 17]]]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            key.numpy(),
            np.array([[[[6, 7, 8], [18, 19, 20]]]], dtype=np.float32),
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestGetQKVGatedPath(unittest.TestCase):
    """Gated attention returns (query, key, value, gate) with a flattened gate.

    Layout: groups=2, heads=4 (hpg=2), head_dim=3, v_head_dim=3.
    Per group = q(6)+gate(6)+k(3)+v(3)=18; total last dim = 36. Hand-derived
    from arange(36)."""

    def test_returns_four_tensors_with_correct_content(self):
        mixed = paddle.arange(36, dtype="float32").reshape([1, 1, 36])
        stub = _make_attn_stub(
            num_attention_heads_per_partition=4,
            num_query_groups_per_partition=2,
            hidden_size_per_attention_head=3,
            value_hidden_size_per_attention_head=3,
            mixed_qkv=mixed,
            gated_attention=True,
        )
        result = _call_qkv(stub)
        self.assertEqual(len(result), 4)
        query, key, value, gate = result

        np.testing.assert_array_equal(
            query.numpy(),
            np.array(
                [[[[0, 1, 2], [3, 4, 5], [18, 19, 20], [21, 22, 23]]]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            key.numpy(),
            np.array([[[[12, 13, 14], [30, 31, 32]]]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            value.numpy(),
            np.array([[[[15, 16, 17], [33, 34, 35]]]], dtype=np.float32),
        )
        # gate: grouped [1,1,2,6] flattened to [1,1,12].
        np.testing.assert_array_equal(
            gate.numpy(),
            np.array(
                [[[6, 7, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29]]],
                dtype=np.float32,
            ),
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestSublayersSpecDefaults(unittest.TestCase):
    """SelfAttentionSublayersSpec is a dataclass with all-None slots by default."""

    def test_all_fields_default_none(self):
        spec = SelfAttentionSublayersSpec()
        self.assertIsNone(spec.qkv_proj)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)
        self.assertIsNone(spec.q_norm)
        self.assertIsNone(spec.k_norm)
        self.assertIsNone(spec.gate_proj)

    def test_fields_are_assignable(self):
        sentinel = object()
        spec = SelfAttentionSublayersSpec(qkv_proj=sentinel)
        self.assertIs(spec.qkv_proj, sentinel)
        self.assertIsNone(spec.core_attention)


if __name__ == "__main__":
    unittest.main()
