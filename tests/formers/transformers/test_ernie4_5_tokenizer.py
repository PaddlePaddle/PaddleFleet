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

"""Behavior tests for ``Ernie4_5Tokenizer``.

``Ernie4_5Tokenizer`` is produced by ``warp_tokenizer(hf.LlamaTokenizer)``,
i.e. a dynamic subclass mixing ``PaddleTokenizerMixin`` into HuggingFace's
``LlamaTokenizer``. On the pinned ``transformers>=5.0`` line ``LlamaTokenizer``
is a fast ``tokenizers``-backed BPE model (``TokenizersBackend``) constructed
from an in-memory ``vocab`` + ``merges`` pair -- it no longer wraps a
SentencePiece ``.model`` file. These tests therefore build a *real* small BPE
tokenizer locally with the ``tokenizers`` library (no network / no hub
download, CPU only) and feed its ``vocab``/``merges`` into the wrapped
tokenizer, then exercise the data-layer contracts of the wrapper: special-token
ids, vocab mapping, truncation, right padding + attention mask, save/reload
round-trip, streaming ``decode_token`` and the PaddleFleet-specific
``apply_chat_template`` default.

Independent oracle: the raw ``tokenizers.Tokenizer`` BPE that produced the
vocab/merges is used to check tokenization and vocab id mapping, so the
reference never goes through the code under test.
"""

import json
import os
import tempfile
import unittest

import transformers as hf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from paddlefleet.transformers.ernie4_5.tokenizer import Ernie4_5Tokenizer
from paddlefleet.transformers.tokenizer_utils import PaddleTokenizerMixin

# Fixed special-token layout; ids are pinned as the first four BPE special
# tokens so expected values below are independent of the wrapper.
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3
UNK_PIECE, BOS_PIECE, EOS_PIECE, PAD_PIECE = "<unk>", "<s>", "</s>", "<pad>"
# SentencePiece-style word-boundary marker used by the Metaspace pre-tokenizer.
_SPACE = "\u2581"

_TRAIN_LINES = [
    "hello world this is a paddle tokenizer test",
    "the quick brown fox jumps over the lazy dog",
    "paddlefleet ernie tokenizer wrapping llama model",
    "banana apple orange fruit basket grocery store",
    "streaming decode round trip padding truncation length",
    "special tokens vocabulary mapping save reload directory",
    "abcdefghijklmnopqrstuvwxyz numbers 0123456789 done",
]


def _build_bpe(directory):
    """Train a tiny Metaspace BPE with the ``tokenizers`` library.

    Returns ``(bpe, vocab, merges)`` where ``bpe`` is the raw
    ``tokenizers.Tokenizer`` (used below as an independent oracle) and
    ``vocab``/``merges`` are the exact pair consumed by the ``transformers``
    fast ``LlamaTokenizer`` constructor. The four special tokens are declared
    first so they occupy the pinned ids 0..3.
    """
    bpe = Tokenizer(models.BPE(unk_token=UNK_PIECE))
    bpe.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=_SPACE, prepend_scheme="always", split=False
    )
    bpe.decoder = decoders.Metaspace(
        replacement=_SPACE, prepend_scheme="always", split=False
    )
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=[UNK_PIECE, BOS_PIECE, EOS_PIECE, PAD_PIECE],
        show_progress=False,
    )
    bpe.train_from_iterator(
        [line for line in _TRAIN_LINES for _ in range(60)], trainer=trainer
    )
    saved = os.path.join(directory, "bpe.json")
    bpe.save(saved)
    vocab = bpe.get_vocab()
    raw_merges = json.load(open(saved, encoding="utf-8"))["model"]["merges"]
    # tokenizers may serialize merges as ["a", "b"] pairs or "a b" strings;
    # the transformers constructor wants a list of (left, right) tuples.
    merges = [
        tuple(m) if isinstance(m, (list, tuple)) else tuple(m.split(" "))
        for m in raw_merges
    ]
    return bpe, vocab, merges


class TestErnie4_5TokenizerConstruction(unittest.TestCase):
    """Structural contracts of ``warp_tokenizer(hf.LlamaTokenizer)``."""

    def test_class_mixes_paddle_mixin_over_llama(self):
        # warp_tokenizer must place PaddleTokenizerMixin ahead of the HF base
        # so the Paddle overrides win; breaking the base tuple would fail here.
        self.assertTrue(issubclass(Ernie4_5Tokenizer, PaddleTokenizerMixin))
        self.assertTrue(issubclass(Ernie4_5Tokenizer, hf.LlamaTokenizer))
        mro = Ernie4_5Tokenizer.__mro__
        self.assertLess(
            mro.index(PaddleTokenizerMixin), mro.index(hf.LlamaTokenizer)
        )
        # The wrapper adopts the HF class name.
        self.assertEqual(Ernie4_5Tokenizer.__name__, "LlamaTokenizer")
        # decode_token is contributed by the mixin, not the HF base.
        self.assertTrue(hasattr(Ernie4_5Tokenizer, "decode_token"))


class TestErnie4_5TokenizerBehavior(unittest.TestCase):
    """Data-layer behavior on a real, locally-built BPE tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp(prefix="ernie45_tok_")
        cls.bpe, cls.vocab, cls.merges = _build_bpe(cls._dir)
        # PaddleFleet trains with right padding, which is not the HF fast
        # default ('left'); set it explicitly so the padding contract below is
        # exercised as production configures it.
        cls.tokenizer = Ernie4_5Tokenizer(
            vocab=cls.vocab,
            merges=cls.merges,
            unk_token=UNK_PIECE,
            bos_token=BOS_PIECE,
            eos_token=EOS_PIECE,
            pad_token=PAD_PIECE,
            padding_side="right",
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls._dir, ignore_errors=True)

    def test_special_token_ids_match_pinned_layout(self):
        tok = self.tokenizer
        self.assertEqual(tok.unk_token_id, UNK_ID)
        self.assertEqual(tok.bos_token_id, BOS_ID)
        self.assertEqual(tok.eos_token_id, EOS_ID)
        self.assertEqual(tok.pad_token_id, PAD_ID)
        # id -> piece must invert the pinned assignment as well.
        self.assertEqual(tok.convert_ids_to_tokens(BOS_ID), BOS_PIECE)
        self.assertEqual(tok.convert_ids_to_tokens(EOS_ID), EOS_PIECE)
        self.assertEqual(tok.convert_ids_to_tokens(PAD_ID), PAD_PIECE)

    def test_vocab_mapping_matches_raw_bpe(self):
        tok = self.tokenizer
        text = "hello world padding banana"
        pieces = tok.tokenize(text)
        # Non-degenerate: the sample must exercise several distinct pieces so a
        # collapsed / constant mapping cannot pass.
        self.assertGreater(len(pieces), 3)
        ids = tok.convert_tokens_to_ids(pieces)
        self.assertGreater(len(set(ids)), 1)
        # Independent oracle: raw tokenizers BPE piece->id, not the wrapper.
        expected = [self.bpe.token_to_id(p) for p in pieces]
        self.assertEqual(ids, expected)
        # The oracle's own tokenization must also agree end-to-end.
        oracle = self.bpe.encode(text, add_special_tokens=False)
        self.assertEqual(pieces, oracle.tokens)
        self.assertEqual(ids, oracle.ids)
        # Round-trip id -> piece recovers the original pieces exactly.
        self.assertEqual(tok.convert_ids_to_tokens(ids), pieces)
        # An out-of-vocabulary piece must fall back to the unk id.
        self.assertEqual(tok.convert_tokens_to_ids("qzxwvk_absent"), UNK_ID)

    def test_encode_respects_special_token_flags(self):
        tok = self.tokenizer
        text = "hello world padding"
        with_special = tok(text)["input_ids"]
        without_special = tok(text, add_special_tokens=False)["input_ids"]
        # Derive the expected decorated ids from the tokenizer's *own*
        # add_bos_token / add_eos_token flags rather than hard-coding a base
        # default; that default has varied across transformers versions, so
        # pinning it makes the test brittle. Whatever the flags report, the
        # full encoding must equal the no-special encoding wrapped by exactly
        # those markers.
        expected = list(without_special)
        if getattr(tok, "add_bos_token", False):
            expected = [BOS_ID, *expected]
        if getattr(tok, "add_eos_token", False):
            expected = [*expected, EOS_ID]
        self.assertEqual(with_special, expected)
        # The no-special payload is a contiguous interior slice of the full
        # encoding at the position implied by the leading-bos flag.
        start = 1 if getattr(tok, "add_bos_token", False) else 0
        end = len(with_special) - (
            1 if getattr(tok, "add_eos_token", False) else 0
        )
        self.assertEqual(with_special[start:end], without_special)

    def test_truncation_keeps_leading_prefix(self):
        tok = self.tokenizer
        text = "hello world padding truncation length banana apple orange"
        full = tok(text)["input_ids"]
        self.assertGreater(len(full), 4)
        truncated = tok(text, max_length=4, truncation=True)["input_ids"]
        # Right-truncation keeps exactly the first max_length ids.
        self.assertEqual(len(truncated), 4)
        self.assertEqual(truncated, full[:4])

    def test_right_padding_and_attention_mask(self):
        tok = self.tokenizer
        self.assertEqual(tok.padding_side, "right")
        text = "hello world padding"
        full = tok(text)["input_ids"]
        target = len(full) + 3
        enc = tok(text, max_length=target, padding="max_length")
        pad_count = target - len(full)
        # Independent expectation hand-derived from the unpadded ids + pad id.
        self.assertEqual(enc["input_ids"], full + [PAD_ID] * pad_count)
        self.assertEqual(
            enc["attention_mask"], [1] * len(full) + [0] * pad_count
        )

    def test_decode_round_trip_is_lossless(self):
        tok = self.tokenizer
        text = "hello world padding banana apple"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        self.assertGreater(len(ids), 0)
        # Prove losslessness by re-encoding the decoded text and requiring
        # identical ids; the trained Metaspace BPE makes this piece->text->piece
        # mapping stable for the ASCII input.
        decoded = tok.decode(ids, skip_special_tokens=False)
        reencoded = tok(decoded, add_special_tokens=False)["input_ids"]
        self.assertEqual(reencoded, ids)
        # When the tokenizer prepends BOS, it must appear in the full decode.
        with_bos = tok(text)["input_ids"]
        if getattr(tok, "add_bos_token", False):
            self.assertIn(
                BOS_PIECE, tok.decode(with_bos, skip_special_tokens=False)
            )

    def test_decode_token_streaming_matches_full_decode(self):
        tok = self.tokenizer
        text = "hello world padding banana"
        ids = tok(text)["input_ids"]
        full = tok.decode(ids)
        # Drive the mixin's streaming decoder incrementally; its offset logic
        # must reconstruct the full decode without dropping or duplicating.
        accumulated = ""
        prefix_offset = 0
        read_offset = 0
        for i in range(len(ids)):
            piece, prefix_offset, read_offset = tok.decode_token(
                ids[: i + 1], prefix_offset, read_offset
            )
            accumulated += piece
        self.assertEqual(accumulated, full)

    def test_save_pretrained_round_trip_preserves_encoding(self):
        tok = self.tokenizer
        save_dir = tempfile.mkdtemp(prefix="ernie45_tok_save_")
        try:
            tok.save_pretrained(save_dir)
            # The fast tokenizer persists its vocab/merges inside tokenizer.json.
            self.assertTrue(
                os.path.isfile(os.path.join(save_dir, "tokenizer.json"))
            )
            reloaded = Ernie4_5Tokenizer.from_pretrained(save_dir)
            text = "hello world padding banana"
            # Reload recovers identical ids and special-token layout, proving
            # the saved vocabulary (not just files) is intact.
            self.assertEqual(
                reloaded(text)["input_ids"], tok(text)["input_ids"]
            )
            self.assertEqual(reloaded.bos_token_id, BOS_ID)
            self.assertEqual(reloaded.eos_token_id, EOS_ID)
            self.assertEqual(reloaded.pad_token_id, PAD_ID)
        finally:
            import shutil

            shutil.rmtree(save_dir, ignore_errors=True)

    def test_apply_chat_template_defaults_generation_prompt_true(self):
        tok = self.tokenizer
        original = tok.chat_template
        tok.chat_template = (
            "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}"
            "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        )
        try:
            chat = [{"role": "user", "content": "hi"}]
            explicit_true = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            explicit_false = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
            defaulted = tok.apply_chat_template(chat, tokenize=False)
            # The PaddleFleet override flips HF's default (False) to True; if
            # that default were removed the omitted call would match False.
            self.assertNotEqual(explicit_true, explicit_false)
            self.assertEqual(defaulted, explicit_true)
            self.assertIn("<|assistant|>", defaulted)
            self.assertNotIn("<|assistant|>", explicit_false)
        finally:
            tok.chat_template = original

    def test_return_tensors_pd_wraps_ids_in_paddle_tensor(self):
        try:
            import paddle
        except ImportError:
            self.skipTest("paddle not installed; pd tensor conversion unrun")
        tok = self.tokenizer
        text = "hello world padding"
        list_ids = tok(text)["input_ids"]
        self.assertGreater(len(list_ids), 0)
        enc = tok(text, return_tensors="pd")
        # The mixin converts a 1-D id list into a batched [1, seq] paddle tensor
        # while leaving the underlying ids unchanged.
        self.assertIsInstance(enc["input_ids"], paddle.Tensor)
        self.assertEqual(list(enc["input_ids"].shape), [1, len(list_ids)])
        self.assertEqual(enc["input_ids"].numpy().tolist(), [list_ids])


if __name__ == "__main__":
    unittest.main()
