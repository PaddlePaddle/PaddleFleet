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

"""Tests for ``rotary_embed_cache``.

The memo hands the *same tensor object* to callers that would otherwise build
their own, so it must be bit-identical to the unmemoised path and must never
serve a table built for a different configuration.

``rotary_embed_cache`` keeps the angle table in a process-wide store keyed by
each ``RotaryEmbedding`` instance's configuration signature, so the ~50 rotary
instances of a 44-layer model retain one table per distinct configuration
instead of one table each.

The signature tests are the ones that matter: the only way the memo can be wrong
is by serving a table that was built for something else, and neither the forward
output nor the loss would look suspicious if it did.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    _SHARED_EMB_CACHE,
    RotaryEmbedding,
    clear_shared_rotary_embed_cache,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

HEAD_DIM = 64
BASE = 160000.0
RATIO = 4


def _bits_equal(a: paddle.Tensor, b: paddle.Tensor) -> bool:
    """Bit-for-bit comparison, so a 1-ULP drift cannot pass as equal."""
    x, y = a.numpy(), b.numpy()
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return np.array_equal(x.view(np.uint8), y.view(np.uint8))


def _build(**kwargs) -> RotaryEmbedding:
    """A rotary module with the memo on (shared across instances)."""
    kwargs.setdefault("head_dim", HEAD_DIM)
    kwargs.setdefault("rotary_percent", 1.0)
    kwargs.setdefault("rotary_base", BASE)
    return RotaryEmbedding(rotary_embed_cache=True, **kwargs)


def _unmemoised(**kwargs) -> RotaryEmbedding:
    """The baseline path: no memo at all."""
    kwargs.setdefault("head_dim", HEAD_DIM)
    kwargs.setdefault("rotary_percent", 1.0)
    kwargs.setdefault("rotary_base", BASE)
    return RotaryEmbedding(rotary_embed_cache=False, **kwargs)


class SharedRotaryCacheTestBase(unittest.TestCase):
    """Process-wide state, so every test starts from an empty store."""

    def setUp(self) -> None:
        clear_shared_rotary_embed_cache()

    def tearDown(self) -> None:
        clear_shared_rotary_embed_cache()


class TestSharedRotaryEmbedCache(SharedRotaryCacheTestBase):
    """Sharing must be invisible except for how many tables are retained."""

    def test_off_by_default(self) -> None:
        rope = RotaryEmbedding(head_dim=HEAD_DIM, rotary_percent=1.0)
        self.assertFalse(rope.rotary_embed_cache)
        rope(512, 0)
        self.assertEqual(len(_SHARED_EMB_CACHE), 0)

    def test_shared_across_instances(self) -> None:
        """The point of the cache: one table for identically built layers."""
        layers = [_build() for _ in range(8)]
        tables = [layer(512, 0) for layer in layers]
        for table in tables[1:]:
            self.assertIs(table, tables[0])
        self.assertEqual(len(_SHARED_EMB_CACHE), 1)

    def test_bit_exact_against_unmemoised(self) -> None:
        reference = _unmemoised()
        shared = _build()
        for max_seq_len, offset in [(512, 0), (256, 0), (512, 7), (1, 1024)]:
            with self.subTest(max_seq_len=max_seq_len, offset=offset):
                want = reference(max_seq_len, offset)
                # Call twice so the compared value comes from the store, not
                # from the build that populated it.
                shared(max_seq_len, offset)
                self.assertTrue(_bits_equal(want, shared(max_seq_len, offset)))

    def _assert_not_confused(self, **difference) -> None:
        """Two layers differing only in ``difference`` must not share a table.

        Both are also checked against their own unmemoised reference, so a miss
        that silently returns the wrong table cannot pass.
        """
        left, right = _build(), _build(**difference)
        table_left, table_right = left(512, 0), right(512, 0)
        self.assertIsNot(table_left, table_right)
        self.assertNotEqual(left._emb_cache_sig, right._emb_cache_sig)
        self.assertEqual(len(_SHARED_EMB_CACHE), 2)
        self.assertTrue(_bits_equal(_unmemoised()(512, 0), table_left))
        self.assertTrue(
            _bits_equal(_unmemoised(**difference)(512, 0), table_right)
        )

    def test_rotary_base_not_confused(self) -> None:
        self._assert_not_confused(rotary_base=2500000.0)

    def test_head_dim_not_confused(self) -> None:
        self._assert_not_confused(head_dim=32)

    def test_rotary_interleaved_not_confused(self) -> None:
        self._assert_not_confused(rotary_interleaved=True)

    def test_interpolation_factor_not_confused(self) -> None:
        self._assert_not_confused(seq_len_interpolation_factor=2.0)

    def test_rope_scaling_not_confused(self) -> None:
        self._assert_not_confused(rope_scaling=True)

    def test_rope_scaling_factor_not_confused(self) -> None:
        left = _build(rope_scaling=True, rope_scaling_factor=8.0)
        right = _build(rope_scaling=True, rope_scaling_factor=4.0)
        self.assertIsNot(left(512, 0), right(512, 0))
        self.assertEqual(len(_SHARED_EMB_CACHE), 2)

    def test_use_accuracy_compatible_not_confused(self) -> None:
        """The flag only moves where the exponent is evaluated, which can change
        its last bits -- so it must still split the signature."""
        self._assert_not_confused(use_accuracy_compatible=True)

    def test_equal_effective_dim_does_share(self) -> None:
        """The signature keys on the *effective* dim, so ``head_dim=64,
        percent=0.5`` and ``head_dim=32, percent=1.0`` are the same table."""
        left = _build(head_dim=64, rotary_percent=0.5)
        right = _build(head_dim=32, rotary_percent=1.0)
        self.assertEqual(left._emb_cache_sig, right._emb_cache_sig)
        self.assertIs(left(512, 0), right(512, 0))
        self.assertEqual(len(_SHARED_EMB_CACHE), 1)
        self.assertTrue(
            _bits_equal(
                _unmemoised(head_dim=32, rotary_percent=1.0)(512, 0),
                right(512, 0),
            )
        )

    def test_yarn_signature_differs_from_rope(self) -> None:
        """``YarnRotaryEmbedding`` overrides the table, so its signature must
        never match a base-class one."""
        yarn = YarnRotaryEmbedding(
            HEAD_DIM,
            rotary_base=BASE,
            scaling_factor=16.0,
            original_max_position_embeddings=512,
        )
        rope = _build()
        self.assertNotEqual(yarn._emb_cache_sig, rope._emb_cache_sig)

    def test_position_ids_never_shared(self) -> None:
        """A ``position_ids`` result depends on a runtime tensor, so it must not
        enter the store nor be served from it."""
        rope = _build()
        plain = rope(128, 0)
        self.assertEqual(len(_SHARED_EMB_CACHE), 1)
        shuffled = paddle.concat([paddle.arange(64, 128), paddle.arange(0, 64)])
        with_ids = rope(128, 0, position_ids=shuffled)
        self.assertIsNot(plain, with_ids)
        self.assertFalse(_bits_equal(plain, with_ids))
        self.assertEqual(len(_SHARED_EMB_CACHE), 1)
        self.assertTrue(
            _bits_equal(_unmemoised()(128, 0, position_ids=shuffled), with_ids)
        )

    def test_varying_key_replaces_rather_than_accumulates(self) -> None:
        """One slot per signature, as with the per-instance memo: decode walks
        the key space and must not retain a table per step."""
        shared = _build()
        reference = _unmemoised()
        for max_seq_len in range(1, 25):
            got = shared(max_seq_len, 0)
            self.assertTrue(_bits_equal(reference(max_seq_len, 0), got))
            self.assertEqual(len(_SHARED_EMB_CACHE), 1)

    def test_two_layers_alternating_keys_stay_correct(self) -> None:
        """Sharing means one layer can evict what another just wrote. The result
        must still be right, only rebuilt."""
        left, right = _build(), _build()
        reference = _unmemoised()
        for _ in range(3):
            self.assertTrue(_bits_equal(reference(256, 0), left(256, 0)))
            self.assertTrue(_bits_equal(reference(512, 0), right(512, 0)))
        self.assertEqual(len(_SHARED_EMB_CACHE), 1)


class TestSharedRotaryEmbedCacheBackward(SharedRotaryCacheTestBase):
    """Gradients must be unchanged when one table object is reused by many
    rope applications in a single autograd graph -- what N layers now do."""

    @staticmethod
    def _apply_layers(ropes, x: paddle.Tensor) -> paddle.Tensor:
        out = x
        for rope in ropes:
            freqs = rope(x.shape[1], 0)
            out = _apply_rotary_pos_emb_bshd(
                out,
                freqs,
                mscale=1.0,
                rotary_interleaved=False,
                multi_latent_attention=True,
                mla_output_remove_interleaving=True,
            )
        return out

    def _run(self, use_cache: bool, n_layers: int, dtype: str):
        clear_shared_rotary_embed_cache()
        paddle.seed(1234)
        # One module per layer, exactly as the model builds them.
        ropes = [
            (_build() if use_cache else _unmemoised()) for _ in range(n_layers)
        ]
        x = paddle.randn([1, 256, 4, HEAD_DIM], dtype="float32").astype(dtype)
        x.stop_gradient = False
        out = self._apply_layers(ropes, x)
        out.sum().astype("float32").backward()
        return out, x.grad

    def test_backward_bit_exact(self) -> None:
        for dtype in ("float32", "bfloat16"):
            for n_layers in (1, 8):
                with self.subTest(dtype=dtype, n_layers=n_layers):
                    want_out, want_grad = self._run(False, n_layers, dtype)
                    got_out, got_grad = self._run(True, n_layers, dtype)
                    self.assertTrue(_bits_equal(want_out, got_out))
                    self.assertTrue(_bits_equal(want_grad, got_grad))

    def test_shared_table_not_mutated_by_consumers(self) -> None:
        ropes = [_build() for _ in range(4)]
        before = ropes[0](256, 0).clone()
        x = paddle.randn([1, 256, 4, HEAD_DIM], dtype="float32")
        x.stop_gradient = False
        self._apply_layers(ropes, x).sum().backward()
        self.assertTrue(_bits_equal(before, ropes[0](256, 0)))


class TestRotaryEmbedCacheConfig(unittest.TestCase):
    """``rotary_embed_cache`` is off by default."""

    def test_off_by_default(self) -> None:
        config = TransformerConfig()
        self.assertFalse(config.rotary_embed_cache)


if __name__ == "__main__":
    unittest.main()
