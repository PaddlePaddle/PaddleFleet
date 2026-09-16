# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""No-card behavior tests for ``paddlefleet.data.tokenizer``.

The module under test exposes exactly two public objects:

* ``get_idx_from_word(word, word_to_idx, unk_word)`` -- a vocab lookup that
  maps a token to its integer id, falling back to the id of ``unk_word`` when
  the token is out of vocabulary.
* ``BaseTokenizer`` -- a base class that stores a vocab and exposes
  ``get_tokenizer`` (returns the ``tokenizer`` attribute). Its ``cut`` and
  ``encode`` methods are unimplemented stubs in this module, so there is no
  segmentation / encoding behavior to verify here.

These tests build a small, local, hand-written vocab whose ids are all
distinct and independent of the implementation, so that a swapped mapping, a
truthiness bug on id ``0``, or a hard-coded unk key would be rejected. No
downloads, no accelerator, and no Paddle tensors are involved.
"""

import unittest

from paddlefleet.data.tokenizer import BaseTokenizer, get_idx_from_word

# A small local vocab with deliberately distinguishable, non-contiguous ids so
# that a constant / shifted / swapped mapping cannot pass silently. The unk id
# (7) is distinct from every real-token id, so an unk fallback is observable.
LOCAL_VOCAB = {
    "[UNK]": 7,
    "[PAD]": 8,
    "zero_tok": 0,  # a real, in-vocab token whose id is the falsy value 0
    "hello": 5,
    "world": 9,
    "foo": 3,
}


class TestGetIdxFromWord(unittest.TestCase):
    """Behavior of the vocab lookup with independent expected ids."""

    def test_known_words_map_to_their_own_distinct_ids(self):
        # Independent expectations: each token resolves to exactly its own id,
        # so a constant or swapped mapping would fail here.
        self.assertEqual(get_idx_from_word("hello", LOCAL_VOCAB, "[UNK]"), 5)
        self.assertEqual(get_idx_from_word("world", LOCAL_VOCAB, "[UNK]"), 9)
        self.assertEqual(get_idx_from_word("foo", LOCAL_VOCAB, "[UNK]"), 3)
        self.assertEqual(get_idx_from_word("[PAD]", LOCAL_VOCAB, "[UNK]"), 8)

    def test_in_vocab_token_with_id_zero_is_not_treated_as_unknown(self):
        # Membership must be decided by ``in``, not by truthiness of the id.
        # ``zero_tok`` is in-vocab with id 0 and must return 0, NOT the unk id.
        result = get_idx_from_word("zero_tok", LOCAL_VOCAB, "[UNK]")
        self.assertEqual(result, 0)
        self.assertNotEqual(result, LOCAL_VOCAB["[UNK]"])

    def test_unknown_word_falls_back_to_unk_id(self):
        # A token absent from the vocab resolves to the unk id (7), which is
        # distinct from every real-token id, so the fallback is observable.
        result = get_idx_from_word("out_of_vocab", LOCAL_VOCAB, "[UNK]")
        self.assertEqual(result, 7)
        self.assertNotIn(result, (0, 3, 5, 8, 9))

    def test_unk_token_itself_resolves_to_its_own_id(self):
        # The unk token is itself in the vocab, so it maps directly to its id.
        self.assertEqual(get_idx_from_word("[UNK]", LOCAL_VOCAB, "[UNK]"), 7)

    def test_fallback_uses_the_supplied_unk_word_argument(self):
        # The unk key is a parameter, not hard-coded: switching the unk_word
        # argument changes which id is returned for an unknown token.
        via_unk = get_idx_from_word("missing", LOCAL_VOCAB, "[UNK]")
        via_pad = get_idx_from_word("missing", LOCAL_VOCAB, "[PAD]")
        self.assertEqual(via_unk, 7)
        self.assertEqual(via_pad, 8)
        self.assertNotEqual(via_unk, via_pad)

    def test_missing_word_and_missing_unk_raises_key_error(self):
        # When both the token and the unk key are absent, the lookup surfaces
        # a KeyError rather than silently returning a default.
        vocab = {"hello": 5, "world": 9}
        with self.assertRaises(KeyError):
            get_idx_from_word("unknown", vocab, "[UNK]")


class TestBaseTokenizer(unittest.TestCase):
    """Behavior of the BaseTokenizer container and its accessor."""

    def test_init_stores_the_exact_vocab_object(self):
        vocab = {"hello": 5, "world": 9, "[UNK]": 7}
        tokenizer = BaseTokenizer(vocab)
        # Stored by reference and with identical content.
        self.assertIs(tokenizer.vocab, vocab)
        self.assertEqual(tokenizer.vocab, {"hello": 5, "world": 9, "[UNK]": 7})

    def test_get_tokenizer_returns_the_assigned_tokenizer_attribute(self):
        tokenizer = BaseTokenizer({"a": 0})
        sentinel = object()
        tokenizer.tokenizer = sentinel
        # get_tokenizer must return exactly the object held in .tokenizer.
        self.assertIs(tokenizer.get_tokenizer(), sentinel)

    def test_get_tokenizer_raises_when_no_tokenizer_is_set(self):
        # No tokenizer attribute is assigned by __init__, so accessing it
        # through get_tokenizer surfaces an AttributeError.
        tokenizer = BaseTokenizer({"a": 0})
        with self.assertRaises(AttributeError):
            tokenizer.get_tokenizer()


if __name__ == "__main__":
    unittest.main()
