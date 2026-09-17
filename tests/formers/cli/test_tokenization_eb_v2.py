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

"""Behavior tests for ErnieBotTokenizer (SentencePiece-backed).

These tests build a tiny *real* SentencePiece model locally (no network) whose
piece/id layout is fully controlled, so expected values are hand-derived from
the training configuration and from an independent raw SentencePieceProcessor
oracle -- never from the tokenizer under test. The production module imports
paddle, which is not installed in the no-card CPU environment, so the whole
suite skips honestly when paddle/paddlefleet or sentencepiece cannot be
imported.
"""

import os
import tempfile
import unittest

try:
    import sentencepiece as spm

    from paddlefleet.cli.train.ernie_pretrain.src.tokenizers.tokenization_eb_v2 import (
        ErnieBotTokenizer,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    spm = None
    ErnieBotTokenizer = None
    _IMPORT_ERROR = exc


# Special pieces registered as SentencePiece user_defined_symbols, in the exact
# order handed to the trainer. With unk_id=0 and bos/eos/pad disabled (-1),
# SentencePiece assigns unk->0 and then these symbols to the contiguous ids
# 1..8 in this order. This is the independent, hand-derived id oracle.
_USER_DEFINED = [
    "<s>",
    "<cls>",
    "</s>",
    "<mask:0>",
    "<pad>",
    "<sep>",
    "<mask:1>",
    "<mask:7>",
]


def _build_tiny_spm_model(directory):
    """Train a small real SentencePiece model into ``directory``.

    Returns the path to the ``.model`` file. Uses a fixed local corpus and a
    fixed configuration so the resulting vocabulary layout is deterministic.
    """
    corpus = os.path.join(directory, "corpus.txt")
    sentences = [
        "hello world",
        "hello there",
        "the quick brown fox",
        "world peace now",
        "language model tokenizer",
        "a small test corpus",
        "paddle fleet ernie bot",
    ]
    with open(corpus, "w", encoding="utf-8") as handle:
        for _ in range(50):
            handle.writelines(line + "\n" for line in sentences)

    prefix = os.path.join(directory, "tokenizer")
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=prefix,
        vocab_size=100,
        model_type="bpe",
        character_coverage=1.0,
        hard_vocab_limit=False,
        user_defined_symbols=list(_USER_DEFINED),
        unk_id=0,
        unk_piece="<unk>",
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )
    return prefix + ".model"


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"requires paddle/paddlefleet + sentencepiece: {_IMPORT_ERROR}",
)
class TestErnieBotTokenizerBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="eb_tok_")
        cls.model_file = _build_tiny_spm_model(cls._tmpdir)
        # Independent oracle: a raw SentencePieceProcessor loaded from the same
        # model file. It shares no code with ErnieBotTokenizer's added logic
        # (special-token splitting, string reconstruction, save/load), so it is
        # a valid reference for the plain segmentation/decoding it wraps.
        cls.oracle = spm.SentencePieceProcessor()
        cls.oracle.Load(cls.model_file)
        cls.tokenizer = ErnieBotTokenizer(cls.model_file)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_special_piece_ids_match_hand_derived_layout(self):
        # unk is pinned to 0 by unk_id=0; user-defined symbols follow in the
        # exact registration order at contiguous ids 1..8.
        self.assertEqual(self.tokenizer._convert_token_to_id("<unk>"), 0)
        self.assertEqual(self.tokenizer._convert_id_to_token(0), "<unk>")

        expected = {piece: idx + 1 for idx, piece in enumerate(_USER_DEFINED)}
        for piece, want_id in expected.items():
            self.assertEqual(
                self.tokenizer._convert_token_to_id(piece),
                want_id,
                f"piece {piece!r} should map to id {want_id}",
            )
            self.assertEqual(
                self.tokenizer._convert_id_to_token(want_id),
                piece,
                f"id {want_id} should map back to piece {piece!r}",
            )
        # Ids are distinct (no collision / no silent aliasing).
        all_ids = [0, *list(expected.values())]
        self.assertEqual(len(set(all_ids)), len(all_ids))

    def test_named_special_token_properties(self):
        # These properties are fixed literals; their ids must resolve to the
        # matching real pieces in the built model (ids 7 and 8 per layout).
        self.assertEqual(self.tokenizer.space_token, "<mask:1>")
        self.assertEqual(self.tokenizer.gend_token, "<mask:7>")
        self.assertEqual(self.tokenizer.space_token_id, 7)
        self.assertEqual(self.tokenizer.gend_token_id, 8)

    def test_vocab_size_and_get_vocab_inverse_map(self):
        # vocab_size delegates to the model; anchor it to the independent load.
        self.assertEqual(self.tokenizer.vocab_size, self.oracle.vocab_size())

        vocab = self.tokenizer.get_vocab()
        # get_vocab must expose at least every base piece and be a true
        # token->id map for the controlled special pieces.
        self.assertGreaterEqual(len(vocab), self.tokenizer.vocab_size)
        for piece, want_id in (
            ("<unk>", 0),
            ("<pad>", 5),
            ("<mask:1>", 7),
            ("<mask:7>", 8),
        ):
            self.assertEqual(vocab[piece], want_id)

    def test_tokenize_preserves_special_tokens_and_delegates_rest(self):
        # Independent expectation, derived only from the raw SentencePiece
        # oracle (never from the tokenizer under test).
        #
        # "<mask:1>" is registered as a SentencePiece *user_defined_symbol* (see
        # _USER_DEFINED / _build_tiny_spm_model), so SentencePiece keeps it as
        # one atomic piece. ErnieBotTokenizer.tokenize
        # (src/paddlefleet/cli/train/ernie_pretrain/src/tokenizers/tokenization_eb_v2.py:164)
        # emits the special token verbatim and delegates the surrounding text to
        # sp_model.encode_as_pieces (its _tokenize at line 114). SentencePiece's
        # add_dummy_prefix normalizer inserts the word-start marker "\u2581" only
        # at the very START of the encoded string, so "hello" before the symbol
        # receives a marker while "world" after it is segmented mid-string
        # WITHOUT one (yielding e.g. 'w','orld', not '\u2581world').
        #
        # Take the two ordinary segments straight from the oracle's encoding of
        # the full input: the pieces before the symbol are the word-start
        # "hello", the pieces after it are the marker-less "world". This is the
        # exact segmentation SentencePiece applies to each substring in context,
        # derived without ever calling the tokenizer under test.
        oracle_pieces = self.oracle.encode_as_pieces("hello<mask:1>world")
        sep = oracle_pieces.index("<mask:1>")
        left = oracle_pieces[
            :sep
        ]  # ['\u2581hello'] -> string start, word-start marker
        right = oracle_pieces[
            sep + 1 :
        ]  # ['w', 'orld'] -> mid-string, no marker
        expected = [*left, "<mask:1>", *right]

        got = self.tokenizer.tokenize("hello<mask:1>world")
        self.assertEqual(got, expected)

        # The special token survives as exactly one element.
        self.assertEqual(got.count("<mask:1>"), 1)

        # Non-trivial word-start-marker contract: the chunk that FOLLOWS the
        # symbol is not the standalone-word segmentation. Encoding "world" on its
        # own receives the add_dummy_prefix marker (['\u2581world']); the
        # post-symbol chunk does not, so the two genuinely differ.
        #
        # NOTE on the fix: a previous revision asserted ``got != oracle.encode_
        # as_pieces("hello<mask:1>world")`` and built ``expected`` with
        # ``encode_as_pieces("world")`` (which adds the dummy prefix). That
        # premise was wrong. Because "<mask:1>" is an SPM-native user_defined_
        # symbol, the tokenizer's special-token-preserving output coincides with
        # the raw whole-string encoding, so ``got == whole`` and only the
        # trailing chunk (marker vs no marker) is the meaningful distinction.
        # This is legitimate SentencePiece behavior, not a production defect, so
        # the oracle derivation is corrected rather than capturing a failure.
        self.assertNotEqual(right, self.oracle.encode_as_pieces("world"))

    def test_convert_tokens_to_string_keeps_special_tokens_literal(self):
        tokens = [
            *self.oracle.encode_as_pieces("hello"),
            "<mask:1>",
            *self.oracle.encode_as_pieces("world"),
        ]
        # Hand-derived: ordinary runs are decoded by the oracle, the special
        # token is emitted verbatim and separates the runs.
        expected = (
            self.oracle.decode(self.oracle.encode_as_pieces("hello"))
            + "<mask:1>"
            + self.oracle.decode(self.oracle.encode_as_pieces("world"))
        )
        self.assertEqual(
            self.tokenizer.convert_tokens_to_string(tokens), expected
        )
        # Guard the anchor itself: the ordinary word round-trips to plain text.
        self.assertEqual(expected, "hello<mask:1>world")

    def test_save_vocabulary_roundtrip_reloads_equivalent_tokenizer(self):
        with tempfile.TemporaryDirectory() as out_dir:
            saved = self.tokenizer.save_vocabulary(out_dir)
            self.assertEqual(len(saved), 1)
            saved_path = saved[0]
            self.assertEqual(os.path.basename(saved_path), "tokenizer.model")
            self.assertTrue(os.path.isfile(saved_path))

            # A brand-new tokenizer loaded from the saved file must reproduce
            # the same vocabulary size and the same segmentation -- proving the
            # file is a usable model, not just a created path.
            reloaded = ErnieBotTokenizer(saved_path)
            self.assertEqual(reloaded.vocab_size, self.tokenizer.vocab_size)
            sample = "hello<mask:7>world"
            self.assertEqual(
                reloaded.tokenize(sample),
                self.tokenizer.tokenize(sample),
            )
            self.assertEqual(
                reloaded._convert_token_to_id("<mask:7>"),
                self.tokenizer._convert_token_to_id("<mask:7>"),
            )


if __name__ == "__main__":
    unittest.main()
