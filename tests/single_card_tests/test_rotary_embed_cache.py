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

"""Tests for ``TransformerConfig.rotary_embed_cache``.

``RotaryEmbedding.forward`` is a pure function of ``(max_seq_len, offset)`` when
``position_ids`` is None, so memoising its angle table must be observationally
invisible. The tests pin that down:

  - Forward bit-exact against the uncached path, over several keys and dtypes.
  - Backward bit-exact, including the case that actually risks breaking: one
    cached table object shared by several rope applications in a single autograd
    graph (what 37 attention layers do per microbatch).
  - Off by default, and off means no caching at all -- empty dict, fresh object
    per call, so the baseline code path is untouched.
  - Calls that pass ``position_ids`` are never cached: the result then depends on
    a runtime tensor rather than on the key.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)

HEAD_DIM = 64
BASE = 160000.0


def _build(cache: bool, **kwargs) -> RotaryEmbedding:
    return RotaryEmbedding(
        head_dim=HEAD_DIM,
        rotary_percent=1.0,
        rotary_base=BASE,
        rotary_embed_cache=cache,
        **kwargs,
    )


def _bits_equal(a: paddle.Tensor, b: paddle.Tensor) -> bool:
    """Bit-for-bit comparison, so a 1-ULP drift cannot pass as equal."""
    x, y = a.numpy(), b.numpy()
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return np.array_equal(x.view(np.uint8), y.view(np.uint8))


class TestRotaryEmbedCache(unittest.TestCase):
    """Forward / backward equivalence of the memoised angle table."""

    def test_off_by_default(self) -> None:
        rope = RotaryEmbedding(head_dim=HEAD_DIM, rotary_percent=1.0)
        self.assertFalse(rope.rotary_embed_cache)
        rope(8192, 0)
        rope(8192, 0)
        self.assertIsNone(rope._emb_cache_key)

    def test_off_does_not_cache(self) -> None:
        rope = _build(False)
        first = rope(8192, 0)
        second = rope(8192, 0)
        self.assertIsNone(rope._emb_cache_key)
        self.assertIsNot(first, second)
        self.assertTrue(_bits_equal(first, second))

    def test_on_returns_same_object_on_hit(self) -> None:
        rope = _build(True)
        first = rope(8192, 0)
        second = rope(8192, 0)
        self.assertIs(first, second)
        self.assertEqual(rope._emb_cache_key, (8192, 0))

    def test_distinct_keys_are_not_confused(self) -> None:
        """Only one table is retained, but a different key must never be served
        the previous one."""
        rope = _build(True)
        a = rope(8192, 0)
        b = rope(4096, 0)
        c = rope(8192, 7)
        self.assertIsNot(a, b)
        self.assertIsNot(a, c)
        self.assertEqual(a.shape[1], 8192)
        self.assertEqual(b.shape[1], 4096)
        off = _build(False)
        self.assertTrue(_bits_equal(off(8192, 7), c))

    def test_forward_bit_exact(self) -> None:
        off, on = _build(False), _build(True)
        for max_seq_len, offset in [
            (8192, 0),
            (4096, 0),
            (8192, 7),
            (1, 1024),
        ]:
            with self.subTest(max_seq_len=max_seq_len, offset=offset):
                ref = off(max_seq_len, offset)
                # Second call so the cached branch, not the build, is compared.
                on(max_seq_len, offset)
                got = on(max_seq_len, offset)
                self.assertTrue(_bits_equal(ref, got))

    def test_forward_bit_exact_interleaved(self) -> None:
        off = _build(False, rotary_interleaved=True)
        on = _build(True, rotary_interleaved=True)
        on(8192, 0)
        self.assertTrue(_bits_equal(off(8192, 0), on(8192, 0)))

    def test_position_ids_never_cached(self) -> None:
        rope = _build(True)
        position_ids = paddle.arange(128)
        first = rope(128, 0, position_ids=position_ids)
        second = rope(128, 0, position_ids=position_ids)
        self.assertIsNone(rope._emb_cache_key)
        self.assertIsNot(first, second)
        self.assertTrue(_bits_equal(first, second))

    def test_position_ids_result_not_served_from_cache(self) -> None:
        """A cached (128, 0) entry must not be returned for a position_ids call."""
        rope = _build(True)
        plain = rope(128, 0)
        shuffled = paddle.concat([paddle.arange(64, 128), paddle.arange(0, 64)])
        with_ids = rope(128, 0, position_ids=shuffled)
        self.assertIsNot(plain, with_ids)
        self.assertFalse(_bits_equal(plain, with_ids))


class TestRotaryEmbedCacheBackward(unittest.TestCase):
    """Gradients must be unchanged, including across a shared table object."""

    @staticmethod
    def _rope_layers(rope: RotaryEmbedding, x: paddle.Tensor, n_layers: int):
        """Apply RoPE n_layers times, the way n attention layers would.

        Each application calls ``rope(...)`` afresh, so with the cache on every
        layer receives the *same* table object inside one autograd graph -- the
        case that would break if sharing were unsafe.
        """
        out = x
        for _ in range(n_layers):
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

    def _run(self, cache: bool, n_layers: int, dtype: str):
        paddle.seed(1234)
        rope = _build(cache)
        x = paddle.randn([1, 256, 4, HEAD_DIM], dtype="float32").astype(dtype)
        x.stop_gradient = False
        out = self._rope_layers(rope, x, n_layers)
        out.sum().astype("float32").backward()
        return out, x.grad

    def test_backward_bit_exact_single_layer(self) -> None:
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                ref_out, ref_grad = self._run(False, 1, dtype)
                got_out, got_grad = self._run(True, 1, dtype)
                self.assertTrue(_bits_equal(ref_out, got_out))
                self.assertTrue(_bits_equal(ref_grad, got_grad))

    def test_backward_bit_exact_shared_table_across_layers(self) -> None:
        """37 HCA layers reuse one cached table per microbatch; emulate that."""
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                ref_out, ref_grad = self._run(False, 8, dtype)
                got_out, got_grad = self._run(True, 8, dtype)
                self.assertTrue(_bits_equal(ref_out, got_out))
                self.assertTrue(_bits_equal(ref_grad, got_grad))

    def test_cached_table_not_mutated_by_consumer(self) -> None:
        """The shared object must survive being consumed by a backward pass."""
        rope = _build(True)
        before = rope(256, 0).clone()
        x = paddle.randn([1, 256, 4, HEAD_DIM], dtype="float32")
        x.stop_gradient = False
        self._rope_layers(rope, x, 4).sum().backward()
        self.assertTrue(_bits_equal(before, rope(256, 0)))


class TestRotaryEmbedCacheBounded(unittest.TestCase):
    """The cache must stay bounded when the key varies (incremental decode)."""

    def test_growing_max_seq_len_does_not_grow_cache(self) -> None:
        """``_build_rope_freqs`` asks for ``sq + position_offset`` without
        ``position_ids``, so decode walks the key space. Retaining one table per
        step would OOM; the bound must hold and the results stay correct."""
        rope = _build(True)
        off = _build(False)
        for seq_len in range(1, 40):
            got = rope(seq_len, 0)
            self.assertTrue(_bits_equal(off(seq_len, 0), got))
            self.assertEqual(rope._emb_cache_key, (seq_len, 0))

    def test_only_the_latest_key_is_retained(self) -> None:
        """A single slot cannot accumulate tables, whatever the caller asks for."""
        rope = _build(True)
        rope(16, 0)
        rope(32, 0)
        self.assertEqual(rope._emb_cache_key, (32, 0))
        rope(64, 7)
        self.assertEqual(rope._emb_cache_key, (64, 7))

    def test_repeated_single_key_never_evicts(self) -> None:
        """The training pattern: one key, unlimited hits, one entry."""
        rope = _build(True)
        first = rope(8192, 0)
        for _ in range(50):
            self.assertIs(rope(8192, 0), first)
        self.assertEqual(rope._emb_cache_key, (8192, 0))


if __name__ == "__main__":
    unittest.main()
