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

"""Behavior tests for KimiK2TikTokenTokenizer (data layer, CPU / no-GPU).

Design notes (independent of the coverage_test source):
- A tiny, hand-derivable tiktoken BPE vocab is built locally so encode/decode
  results can be predicted by an independent BPE derivation, not by calling the
  tokenizer back on itself.
- Special-token IDs are checked against the deterministic offset contract in
  ``__init__`` (ids start right after the base vocab), so an off-by-one in the
  reserved-token range or a swapped bos/eos/unk/pad mapping is caught.
- Save + reload is verified by loading a *new* tokenizer object from the copied
  vocab file and re-running behavior, not by inspecting the original object.
- No network download is used; the vocab asset is synthesized locally.
"""

import base64
import os
import tempfile
import unittest

# Precise capability probe: only real missing third-party deps should skip.
# tiktoken / tokenizers / transformers are required by the production module.
_IMPORT_ERROR = None
try:
    from tokenizers import AddedToken

    from paddlefleet.transformers.kimi_k2.tokenizer import (
        KimiK2TikTokenTokenizer,
    )
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = repr(exc)


# --- Independently defined minimal tiktoken BPE vocab ------------------------
# Ranks are the token ids. Single bytes a/b/c plus one merge "ab".
# This is the objective tiktoken .model line format: "<b64(token)> <rank>".
_BPE_RANKS = {b"a": 0, b"b": 1, b"c": 2, b"ab": 3}
_NUM_BASE_TOKENS = len(_BPE_RANKS)  # 4
_NUM_RESERVED = 256  # KimiK2 num_reserved_special_tokens
# special-token id range is [num_base, num_base + num_reserved + 2) -> 258 ids
_NUM_SPECIAL = _NUM_RESERVED + 2  # 258
_EXPECTED_N_WORDS = _NUM_BASE_TOKENS + _NUM_SPECIAL  # 262

# Deterministic special-token id assignment (see __init__):
_BOS_ID = _NUM_BASE_TOKENS  # 4
_EOS_ID = _NUM_BASE_TOKENS + 1  # 5
_UNK_ID = _NUM_BASE_TOKENS + 2  # 6
_PAD_ID = _NUM_BASE_TOKENS + 3  # 7
_UNK_TOKEN = f"<|reserved_token_{_UNK_ID}|>"
_PAD_TOKEN = f"<|reserved_token_{_PAD_ID}|>"


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"KimiK2 tokenizer deps unavailable: {_IMPORT_ERROR}",
)
class KimiK2TikTokenTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.vocab_file = os.path.join(cls.tmpdir, "tiktoken.model")
        with open(cls.vocab_file, "w") as f:
            f.writelines(
                f"{base64.b64encode(token).decode()} {rank}\n"
                for token, rank in _BPE_RANKS.items()
            )

    def _make_tokenizer(self, vocab_file=None):
        # Only bos/eos are given explicit content; unk/pad fall back to the
        # default "<|reserved_token_{i}|>" names, matching the ids we assert.
        added_tokens_decoder = {
            _BOS_ID: AddedToken("[BOS]", special=True),
            _EOS_ID: AddedToken("[EOS]", special=True),
        }
        return KimiK2TikTokenTokenizer(
            vocab_file=vocab_file or self.vocab_file,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token=_UNK_TOKEN,
            pad_token=_PAD_TOKEN,
            added_tokens_decoder=added_tokens_decoder,
        )

    # -- special tokens / vocab size -----------------------------------------
    def test_special_token_ids_follow_base_offset(self):
        tok = self._make_tokenizer()
        # ids are pinned to the base-vocab offset; a swap or off-by-one fails.
        self.assertEqual(tok.bos_id, _BOS_ID)
        self.assertEqual(tok.eos_id, _EOS_ID)
        self.assertEqual(tok.unk_id, _UNK_ID)
        self.assertEqual(tok.pad_id, _PAD_ID)
        # the four ids must be distinct (guards against all-mapping-to-one bug)
        self.assertEqual(
            len({tok.bos_id, tok.eos_id, tok.unk_id, tok.pad_id}), 4
        )
        # special-token table covers exactly the reserved range
        self.assertEqual(len(tok.special_tokens), _NUM_SPECIAL)
        self.assertEqual(tok.special_tokens["[BOS]"], _BOS_ID)
        self.assertEqual(tok.special_tokens["[EOS]"], _EOS_ID)
        self.assertEqual(tok.special_tokens[_UNK_TOKEN], _UNK_ID)
        # last reserved id is num_base + num_special - 1
        last_id = _NUM_BASE_TOKENS + _NUM_SPECIAL - 1
        self.assertEqual(
            tok.special_tokens[f"<|reserved_token_{last_id}|>"], last_id
        )

    def test_vocab_size_matches_n_words(self):
        tok = self._make_tokenizer()
        self.assertEqual(tok.vocab_size, tok.n_words)
        # base vocab + reserved specials, exact count.
        self.assertEqual(tok.vocab_size, _EXPECTED_N_WORDS)

    def test_get_vocab_maps_base_tokens(self):
        tok = self._make_tokenizer()
        vocab = tok.get_vocab()
        self.assertIsInstance(vocab, dict)
        # printable-ASCII bytes map to themselves under bytes_to_unicode,
        # so the base byte tokens are addressable by their literal string.
        self.assertEqual(vocab["a"], 0)
        self.assertEqual(vocab["b"], 1)
        self.assertEqual(vocab["c"], 2)
        self.assertEqual(vocab["ab"], 3)

    # -- encode: independent BPE derivation ----------------------------------
    def test_encode_matches_independent_bpe(self):
        tok = self._make_tokenizer()
        # "ab": pair (a,b) has rank 3 -> merges to single token "ab"=3.
        self.assertEqual(tok.encode("ab"), [3])
        # "abc": only (a,b) is mergeable -> ["ab", "c"] = [3, 2].
        self.assertEqual(tok.encode("abc"), [3, 2])
        # "cba": no adjacent pair exists in ranks -> [c, b, a] = [2, 1, 0].
        self.assertEqual(tok.encode("cba"), [2, 1, 0])
        # single byte
        self.assertEqual(tok.encode("a"), [0])

    # -- decode / round-trip --------------------------------------------------
    def test_decode_round_trip(self):
        tok = self._make_tokenizer()
        self.assertEqual(tok.decode([3, 2]), "abc")
        for text in ("ab", "abc", "cba", "a"):
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_decode_accepts_scalar_int(self):
        tok = self._make_tokenizer()
        # a bare int is wrapped into a single-element list before decoding.
        self.assertEqual(tok.decode(2), "c")
        self.assertEqual(tok.decode(3), "ab")

    # -- token <-> id mapping -------------------------------------------------
    def test_convert_token_id_mapping(self):
        tok = self._make_tokenizer()
        self.assertEqual(tok._convert_token_to_id("a"), 0)
        self.assertEqual(tok._convert_token_to_id("ab"), 3)
        self.assertEqual(tok._convert_id_to_token(0), "a")
        self.assertEqual(tok._convert_id_to_token(3), "ab")
        # unknown token falls back to unk_id (not a silent 0/None).
        self.assertEqual(
            tok._convert_token_to_id("this-token-does-not-exist"),
            _UNK_ID,
        )

    # -- kwargs branches actually route through the transformers plumbing -----
    def test_encode_with_kwargs_routes_to_super(self):
        tok = self._make_tokenizer()
        # with extra kwargs, encode() delegates to super().encode which walks
        # _tokenize -> _convert_token_to_id; result must still equal the direct
        # tiktoken encode for the same text.
        out = tok.encode("abc", add_special_tokens=False)
        self.assertEqual(out, [3, 2])

    def test_decode_with_kwargs_routes_to_super(self):
        tok = self._make_tokenizer()
        # extra kwargs delegate to super().decode -> _convert_id_to_token +
        # convert_tokens_to_string; ids 0,1,2 are non-special so nothing is
        # skipped and the bytes reassemble to "abc".
        out = tok.decode([0, 1, 2], skip_special_tokens=True)
        self.assertEqual(out, "abc")

    # -- whitespace / non-whitespace slicing ---------------------------------
    def test_split_whitespaces_or_nonwhitespaces(self):
        split = KimiK2TikTokenTokenizer._split_whitespaces_or_nonwhitespaces
        # a run of 3 identical-class chars with limit 2 splits after 2.
        self.assertEqual(list(split("aaa", 2)), ["aa", "a"])
        # class is whitespace-vs-not; "aaabbb" is one 6-long non-space run.
        self.assertEqual(list(split("aaabbb", 2)), ["aa", "ab", "bb"])
        # transitions between classes reset the counter, so runs under the
        # limit are NOT split even across a class boundary.
        self.assertEqual(list(split("aa  bb", 5)), ["aa  bb"])

    def test_pre_tokenizer_and_clean_up_are_identity(self):
        tok = self._make_tokenizer()
        self.assertEqual(
            tok.pre_tokenizer_process("hello world"), ["hello world"]
        )
        self.assertEqual(
            KimiK2TikTokenTokenizer.clean_up_tokenization("hi  there"),
            "hi  there",
        )

    # -- save + reload (new object) ------------------------------------------
    def test_save_and_reload_preserves_behavior(self):
        tok = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as save_dir:
            (out_file,) = tok.save_vocabulary(save_dir)
            self.assertTrue(os.path.isfile(out_file))
            self.assertEqual(os.path.basename(out_file), "tiktoken.model")
            # reload from the copied file into a brand-new tokenizer object.
            reloaded = self._make_tokenizer(vocab_file=out_file)
            self.assertEqual(reloaded.vocab_size, tok.vocab_size)
            self.assertEqual(reloaded.encode("abc"), [3, 2])
            self.assertEqual(reloaded.decode([3, 2]), "abc")
            self.assertEqual(reloaded.bos_id, _BOS_ID)
            self.assertEqual(reloaded.pad_id, _PAD_ID)

    def test_save_vocabulary_rejects_non_directory(self):
        tok = self._make_tokenizer()
        with self.assertRaises(ValueError):
            tok.save_vocabulary(os.path.join(self.tmpdir, "nope", "missing"))

    # -- padding placement / attention mask ----------------------------------
    def test_pad_places_pad_tokens_and_builds_mask(self):
        tok = self._make_tokenizer()
        pad_id = tok.pad_token_id
        if pad_id is None:
            self.skipTest(
                "pad_token_id not resolved by base tokenizer; padding "
                "placement cannot be verified without it."
            )
        batch = {"input_ids": [[3, 2], [3]]}

        right = tok.pad(dict(batch), padding="longest")
        self.assertEqual(right["input_ids"], [[3, 2], [3, pad_id]])
        self.assertEqual(right["attention_mask"], [[1, 1], [1, 0]])

        tok.padding_side = "left"
        left = tok.pad({"input_ids": [[3, 2], [3]]}, padding="longest")
        self.assertEqual(left["input_ids"], [[3, 2], [pad_id, 3]])
        self.assertEqual(left["attention_mask"], [[1, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
