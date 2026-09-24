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

"""Behavioural tests for the configurable HCA compress-P2P path (module:
distributed training / CP).

What is verified here (single process, no collective needed):
  * ``cp_utils._neighbour_window`` argument guards reached through both public
    wrappers (``append_next_window`` / ``prepend_prev_window``): the
    non-positive-window identity short-circuit, and the two *distinct*
    ``ValueError`` branches (window wider than the shard vs. missing/degenerate
    CP group). Error text is asserted so a swapped guard order is caught.
  * ``CSADocMaskMetadata.cp_compress_plan`` owner-sharding index math: for a
    fixed cutoff layout the per-rank local (rebased) cutoff columns and the
    rank-major -> dense-group permutation are compared against a hand-derived
    expectation (NOT the method's own formula), including ranks that own no
    group and therefore return an all-zero padded run. The per-key cache
    identity contract is checked too.
  * The ``TransformerConfig.cp_compress_p2p`` declared default and its actual
    consumption by ``Compressor.__init__`` (config field -> ``self`` attribute).

What is NOT verified here, and why: the real one-hop window exchange
(``NeighbourWindow`` forward/backward) and the pool-by-owner ``Compressor``
forward numerics require a live CP process group and the Triton document-mask
kernels, so they belong to the multi-card companion
``tests/multi_card_tests/transformer/test_hca_cp_p2p_switch.py`` and are not
run in this single-card file.
"""

import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

try:
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

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    nn = None
    _IMPORT_ERROR = exc

_REQUIRES = unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"requires paddle + paddlefleet.transformer: {_IMPORT_ERROR}",
)

DTYPE = "float32"


# The following helpers only materialise when paddle imported successfully; the
# tests that reference them are skipped otherwise, so the names are never used
# in an environment where paddle is missing.
if _IMPORT_ERROR is None:

    class _Linear(nn.Layer):
        """Minimal ``build_spec_layer`` target: returns ``(out, bias=None)``."""

        def __init__(self, input_size, output_size, dtype=None, **kwargs):
            super().__init__()
            self.weight = self.create_parameter(
                shape=[output_size, input_size],
                dtype=dtype or DTYPE,
                default_initializer=nn.initializer.Constant(0.0),
            )

        def forward(self, x):
            return paddle.matmul(x, self.weight.T), None

    class _RMSNorm(nn.Layer):
        def __init__(self, hidden_size=None, eps=1e-5, **kwargs):
            super().__init__()
            self.eps = eps
            self.weight = self.create_parameter(
                shape=[hidden_size],
                dtype=DTYPE,
                default_initializer=nn.initializer.Constant(1.0),
            )

        def forward(self, x, **kwargs):
            normed = x * paddle.rsqrt(
                x.square().mean(-1, keepdim=True) + self.eps
            )
            return normed * self.weight.cast(x.dtype)

    def _spec():
        return CompressorSublayersSpec(
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
        return SimpleNamespace(**base)

    def _make_meta(cutoff_list, ratio, seqlen):
        """Construct a real ``CSADocMaskMetadata`` with only the fields
        ``cp_compress_plan`` consumes (ratio, seqlen, cutoff_gather_indices);
        the remaining dataclass fields are genuinely unused by that method.

        This bypasses ``CSADocMaskMetadata.build`` on purpose: ``build`` derives
        the cutoff via Triton kernels (GPU only), whereas the plan under test is
        pure index arithmetic over an already-known cutoff layout.
        """
        cutoff = paddle.to_tensor(cutoff_list, dtype="int64")
        return CSADocMaskMetadata(
            startend_row_indices=None,
            ratio=ratio,
            batch_size=1,
            seqlen=seqlen,
            n_compressed=seqlen // ratio,
            doc_lens=None,
            doc_starts=None,
            doc_start_per_pos=None,
            doc_len_per_pos=None,
            is_valid=None,
            pos_in_doc=None,
            cutoff_gather_indices=cutoff,
            cutoff_pos_in_doc=None,
            compressed_is_first=None,
            compressed_pos_in_doc=None,
        )


@_REQUIRES
class TestNeighbourWindowGuards(unittest.TestCase):
    """The single-process argument guards inside ``_neighbour_window``.

    These run before any ``NeighbourWindow.apply`` collective, so they are the
    portion of the CP window primitive that is genuinely exercisable without a
    process group. Both public wrappers share the guard, so both are checked.
    """

    def test_nonpositive_window_returns_same_object(self):
        # window <= 0 must be an exact identity: the very tensor is returned
        # (no copy, no group required), for both wrappers and window 0 / < 0.
        x = paddle.arange(1 * 8 * 4, dtype=DTYPE).reshape([1, 8, 4])
        self.assertIs(append_next_window(x, 0, None), x)
        self.assertIs(prepend_prev_window(x, -3, None), x)

    def test_window_wider_than_shard_is_rejected(self):
        # x.shape[1] == 8, window 9 exceeds one shard; this guard is checked
        # *before* the group guard, so a None group must not mask it.
        x = paddle.zeros([1, 8, 4], dtype=DTYPE)
        for fn in (append_next_window, prepend_prev_window):
            with self.assertRaisesRegex(
                ValueError, "exceeds the local sequence length"
            ):
                fn(x, 9, None)

    def test_valid_window_requires_multi_rank_group(self):
        # A within-shard window (3 <= 8) makes the size guard pass, so the
        # missing-group guard is the one that must fire -- and identically for a
        # degenerate single-rank group.
        x = paddle.zeros([1, 8, 4], dtype=DTYPE)
        single_rank = SimpleNamespace(nranks=1)
        for fn in (append_next_window, prepend_prev_window):
            with self.assertRaisesRegex(
                ValueError, "requires a context-parallel group"
            ):
                fn(x, 3, None)
            with self.assertRaisesRegex(
                ValueError, "requires a context-parallel group"
            ):
                fn(x, 3, single_rank)


@_REQUIRES
class TestCpCompressPlan(unittest.TestCase):
    """Owner-sharding math of ``CSADocMaskMetadata.cp_compress_plan``.

    Fixed layout: ratio=2, seqlen=16, so ``sq_local = 16 // cp_size``.
    ``cutoff_gather_indices = [0,1, 2,3, 8,9, 10,11]`` -> 4 groups whose starts
    are ``[0, 2, 8, 10]``. Groups 0,1 live in the first half [0,8); groups 2,3
    in the second half [8,16). Every expected array below is derived by hand
    from that ownership, never from the method's own formula.
    """

    CUTOFF = [0, 1, 2, 3, 8, 9, 10, 11]
    RATIO = 2
    SEQLEN = 16
    # perm maps rank-major group slots back to dense group order; it is a
    # whole-group quantity, so it must be identical on every rank.
    EXPECTED_PERM = [0, 1, 4, 5]

    def _plan(self, cp_size, cp_rank):
        meta = _make_meta(self.CUTOFF, self.RATIO, self.SEQLEN)
        local, perm = meta.cp_compress_plan(cp_size, cp_rank)
        return local.numpy(), perm.numpy()

    def test_even_split_two_ranks(self):
        # sq_local = 8, slots = 4. rank 0 owns groups {0,1} (starts 0,2),
        # rank 1 owns groups {2,3} (starts 8,10). Each owns 2 groups -> 4 real
        # rows, then 4 padded zero rows up to sq_local = 8. Real rows are the
        # owned cutoff indices rebased by -cp_rank*sq_local.
        expected_local = {
            0: [0, 1, 2, 3, 0, 0, 0, 0],  # cutoff[0:4] - 0
            1: [0, 1, 2, 3, 0, 0, 0, 0],  # cutoff[4:8] - 8 == [8,9,10,11]-8
        }
        for rank in (0, 1):
            local, perm = self._plan(cp_size=2, cp_rank=rank)
            self.assertEqual(local.shape[0], self.SEQLEN // 2)
            np.testing.assert_array_equal(local, expected_local[rank])
            np.testing.assert_array_equal(perm, self.EXPECTED_PERM)

    def test_more_ranks_than_groups_pads_empty_owners(self):
        # sq_local = 4, slots = 2. Ownership by half: rank 0 -> {0,1},
        # rank 2 -> {2,3}; ranks 1 and 3 own no group at all and must return an
        # all-zero padded run (the branch that reads row 0 then masks it out).
        expected_local = {
            0: [0, 1, 2, 3],  # cutoff[0:4] - 0
            1: [0, 0, 0, 0],  # empty run
            2: [0, 1, 2, 3],  # cutoff[4:8] - 8
            3: [0, 0, 0, 0],  # empty run
        }
        for rank in range(4):
            local, perm = self._plan(cp_size=4, cp_rank=rank)
            self.assertEqual(local.shape[0], self.SEQLEN // 4)
            np.testing.assert_array_equal(local, expected_local[rank])
            np.testing.assert_array_equal(perm, self.EXPECTED_PERM)

    def test_plan_is_cached_per_key(self):
        meta = _make_meta(self.CUTOFF, self.RATIO, self.SEQLEN)
        first = meta.cp_compress_plan(2, 0)
        # Same (cp_size, cp_rank) -> the identical tuple object (cache hit).
        self.assertIs(meta.cp_compress_plan(2, 0), first)
        # Different rank -> a distinct entry, not the cached one.
        self.assertIsNot(meta.cp_compress_plan(2, 1), first)


@_REQUIRES
class TestCompressCpP2pSwitch(unittest.TestCase):
    """The ``cp_compress_p2p`` config field and its consumption."""

    def _build(self, **cfg):
        return Compressor(
            config=_config(64, **cfg),
            sublayers_spec=_spec(),
            compress_ratio=128,
            head_dim=32,
            rotate=False,
            rotary_pos_emb=None,
        )

    def test_declared_default_is_false(self):
        field = {f.name: f for f in fields(TransformerConfig)}[
            "cp_compress_p2p"
        ]
        self.assertIs(field.default, False)

    def test_init_reads_switch_from_config(self):
        # Absent on the config object -> compressor defaults the switch off.
        self.assertFalse(self._build().cp_compress_p2p)
        # Present and True -> honoured (config field flows to the consumer).
        self.assertTrue(self._build(cp_compress_p2p=True).cp_compress_p2p)


if __name__ == "__main__":
    unittest.main()
