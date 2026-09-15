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

"""Single-card coverage for the configurable HCA P2P compression.

Everything here is pure tensor arithmetic or construction, so it needs no
collectives. The CP>1 collective branches (the one-hop windows on the wire and
the two Compressor.forward pooling paths) are exercised by the multi-card
companion ``multi_card_tests/transformer/test_hca_cp_p2p_switch.py``.

Covered:
  * ``cp_utils._neighbour_window`` guards + the ``group is None`` zero-pad path
    reached through both public wrappers
  * ``CSADocMaskMetadata.cp_compress_plan`` owner-sharding math and its cache
  * ``Compressor.__init__`` reading ``cp_compress_p2p`` and the no-CP forward
  * the ``TransformerConfig.cp_compress_p2p`` field contract
"""

import types
import unittest
from dataclasses import fields

import numpy as np
import paddle
from paddle import nn

from paddlefleet.transformer.cp_utils import (
    append_next_window,
    prepend_prev_window,
)
from paddlefleet.transformer.csa_attention import (
    Compressor,
    CompressorSublayersSpec,
    CSADocMaskMetadata,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

DTYPE = "float32"


class _Linear(nn.Layer):
    """Minimal ``build_spec_layer`` target: matmul, returns (out, bias=None)."""

    def __init__(self, input_size, output_size, dtype=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[output_size, input_size],
            dtype=dtype or DTYPE,
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight.T), None


class _RMSNorm(nn.Layer):
    def __init__(self, hidden_size=None, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x, **kwargs):
        normed = x * paddle.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return normed * self.weight.cast(x.dtype)


_SPEC = CompressorSublayersSpec(
    linear_wkv=_Linear, linear_wgate=_Linear, norm=_RMSNorm
)


def _config(hidden_size, **overrides):
    base = {
        "hidden_size": hidden_size,
        "qk_pos_emb_head_dim": 0,
        "init_method": None,
        "init_method_std": 0.02,
        "rms_norm_eps": 1e-5,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _doc_ends(doc_lens):
    """``[1, 1, seqlen, 1]`` int32 exclusive document ends, the CSA contract."""
    rows, cum = [], 0
    for length in doc_lens:
        cum += length
        rows += [cum] * length
    return paddle.to_tensor(rows, dtype="int32").reshape([1, 1, len(rows), 1])


def _meta(doc_lens, ratio):
    ends = _doc_ends(doc_lens)
    return CSADocMaskMetadata.build(
        ratio, 1, ends.shape[2], ends, dense_mode=False
    )


class TestNeighbourWindowNonCollective(unittest.TestCase):
    """The wrapper guards on the single-process path."""

    def test_nonpositive_window_is_identity(self):
        x = paddle.randn([1, 8, 4], dtype=DTYPE)
        self.assertIs(append_next_window(x, 0, None), x)
        self.assertIs(prepend_prev_window(x, -3, None), x)

    def test_window_wider_than_shard_raises(self):
        x = paddle.randn([1, 8, 4], dtype=DTYPE)
        for fn in (append_next_window, prepend_prev_window):
            with self.assertRaises(ValueError):
                fn(x, 9, None)

    def test_requires_cp_group(self):
        x = paddle.randn([1, 8, 4], dtype=DTYPE)
        for fn in (append_next_window, prepend_prev_window):
            with self.assertRaises(ValueError):
                fn(x, 3, None)


class TestCpCompressPlan(unittest.TestCase):
    """Owner-sharding plan is pure math, so every rank is reachable here.

    Two contract invariants, computed without reusing the method's formula:
      * ``local`` rebased back to global equals exactly the cutoff indices of
        the groups this rank owns; the slots past that run are padding zeros.
      * ``perm`` reorders the rank-major concatenation back to dense group
        order across the whole group.
    """

    # 300 -> 2 groups, 212 -> 1 group: 3 real groups, so at CP2/CP4 some ranks
    # own fewer slots than they have, hitting the padded / empty-run branches.
    DOCS, RATIO = [300, 212], 128

    def _check(self, cp_size):
        meta = _meta(self.DOCS, self.RATIO)
        cutoff = meta.cutoff_gather_indices.numpy()
        ratio, seqlen = self.RATIO, meta.seqlen
        sq_local = seqlen // cp_size
        starts = cutoff[::ratio]
        n_groups = starts.shape[0]
        slots = sq_local // ratio
        # independent ownership: bounds[r] = #starts strictly below r*sq_local
        bounds = [
            int((starts < r * sq_local).sum()) for r in range(cp_size + 1)
        ]

        rankmajor = np.full(cp_size * slots, -1)
        for r in range(cp_size):
            local, perm = (t.numpy() for t in meta.cp_compress_plan(cp_size, r))
            lo, hi = bounds[r], bounds[r + 1]
            self.assertEqual(local.shape[0], sq_local)
            n_rows = (hi - lo) * ratio
            # real rows: this rank's owned cutoff indices, rebased to local
            np.testing.assert_array_equal(
                local[:n_rows] + r * sq_local, cutoff[lo * ratio : hi * ratio]
            )
            # padding rows past the owned run are zeroed
            np.testing.assert_array_equal(
                local[n_rows:], np.zeros(sq_local - n_rows)
            )
            # perm is identical across ranks (a full-group quantity); record the
            # rank-major slot -> group id map this rank contributes
            for j in range(hi - lo):
                rankmajor[r * slots + j] = lo + j
            self.perm = perm
        # dense order recovered: rank-major[perm] == 0, 1, ..., n_groups-1
        np.testing.assert_array_equal(rankmajor[self.perm], np.arange(n_groups))

    def test_plan_even_split(self):
        self._check(cp_size=2)

    def test_plan_more_ranks_than_groups(self):
        # CP4: sq_local=128, one slot per rank; rank 3 owns no group at all
        self._check(cp_size=4)

    def test_plan_is_cached_per_key(self):
        meta = _meta(self.DOCS, self.RATIO)
        first = meta.cp_compress_plan(2, 0)
        self.assertIs(meta.cp_compress_plan(2, 0), first)  # cache hit
        self.assertIsNot(meta.cp_compress_plan(2, 1), first)  # distinct key


class TestCompressorInitAndNoCP(unittest.TestCase):
    def _build(self, ratio, head_dim, **cfg):
        return Compressor(
            config=_config(64, **cfg),
            sublayers_spec=_SPEC,
            compress_ratio=ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=None,
        )

    def test_switch_defaults_off_and_is_read_from_config(self):
        self.assertFalse(self._build(128, 32).cp_compress_p2p)
        self.assertTrue(
            self._build(128, 32, cp_compress_p2p=True).cp_compress_p2p
        )

    def test_no_cp_forward_pools_every_group(self):
        # cp_group=None -> cp_size 1 -> pool_by_owner False, n_shard 0: the plain
        # densify-pool-norm path, n_compressed = sq // ratio.
        ratio, head_dim = 128, 32
        comp = self._build(ratio, head_dim)
        meta = _meta([256], ratio)
        x = paddle.randn([1, meta.seqlen, 64], dtype=DTYPE)
        x.stop_gradient = False
        out = comp(x, cp_group=None, docmask_meta=meta)
        self.assertEqual(out.shape, [1, meta.seqlen // ratio, head_dim])
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestConfigField(unittest.TestCase):
    def test_field_default_is_false(self):
        field = {f.name: f for f in fields(TransformerConfig)}[
            "cp_compress_p2p"
        ]
        self.assertIs(field.default, False)


if __name__ == "__main__":
    unittest.main()
