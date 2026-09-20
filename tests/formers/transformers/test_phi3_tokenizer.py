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

"""Behavior tests for ``Phi3Tokenizer`` (data layer, CPU / no-GPU).

``Phi3Tokenizer`` is produced by ``warp_tokenizer(hf.GPT2Tokenizer)``: a
dynamic subclass mixing ``PaddleTokenizerMixin`` into HuggingFace's
byte-level BPE ``GPT2Tokenizer``.  These tests build a *tiny, hand-derivable*
byte-level BPE vocab locally (no network / no hub download) and exercise the
data-layer contracts the standard asks for on a tokenizer: special-token
ids, vocab mapping, encode against an independent BPE derivation, truncation,
right/left padding + attention mask, and save + reload round-trip content.

Independence: expected ids and token sequences are derived by hand from the
byte-level BPE algorithm and the vocab/merges files this test writes -- never
by calling the tokenizer to build its own oracle.  Save/reload is verified by
loading a *new* tokenizer object (both via the constructor and via the
production ``from_pretrained`` entry) and re-running behavior, not by
inspecting the original object or merely checking that files exist.

Environment note (无卡, but Paddle is not optional here): importing
``paddlefleet`` pulls in ``parallel_state`` which imports ``paddle`` at module
load, so the whole tokenizer import requires Paddle even though byte-level BPE
runs purely on CPU.  When Paddle (or another hard dependency of the production
module) is missing, the suite skips with the precise import error rather than
silently passing.  A separate ``return_tensors="pd"`` test is additionally
gated on Paddle being importable at run time, because ``PaddleTokenizerMixin``
converts outputs to Paddle tensors on that path.
"""

import json
import os
import tempfile
import unittest

# Precise capability probe: only a genuinely missing hard dependency (e.g.
# Paddle, transformers) should skip.  API changes would surface as a real
# error here instead of being swallowed as "missing dep".
_IMPORT_ERROR = None
try:
    import transformers as hf

    from paddlefleet.transformers.phi3.tokenizer import Phi3Tokenizer
    from paddlefleet.transformers.tokenizer_utils import PaddleTokenizerMixin
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = repr(exc)

# Independent run-time probe for the Paddle-tensor return path only.
_HAS_PADDLE = False
try:
    import paddle  # noqa: F401

    _HAS_PADDLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    _HAS_PADDLE = False


# --- Hand-derived minimal byte-level BPE vocab ------------------------------
# All three base characters are printable ASCII, so under GPT2's
# bytes_to_unicode they map to themselves and are addressable by their literal
# string.  A single merge rule "a b" (rank 0) is the only merge, which makes
# every encode result derivable by hand (see per-test comments).
_VOCAB = {
    "a": 0,
    "b": 1,
    "c": 2,
    "ab": 3,
    "<unk>": 4,
    "<s>": 5,
    "</s>": 6,
    "<pad>": 7,
}
_MERGES = "#version: 0.2\na b\n"

_A_ID, _B_ID, _C_ID, _AB_ID = 0, 1, 2, 3
_UNK_ID, _BOS_ID, _EOS_ID, _PAD_ID = 4, 5, 6, 7
_VOCAB_SIZE = 8  # len(encoder), specials included in the vocab file


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"Phi3 tokenizer deps unavailable (Paddle/transformers): {_IMPORT_ERROR}",
)
class Phi3TokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.vocab_file = os.path.join(cls.tmpdir, "vocab.json")
        cls.merges_file = os.path.join(cls.tmpdir, "merges.txt")
        with open(cls.vocab_file, "w", encoding="utf-8") as f:
            json.dump(_VOCAB, f)
        with open(cls.merges_file, "w", encoding="utf-8") as f:
            f.write(_MERGES)

    def _make_tokenizer(self, vocab_file=None, merges_file=None):
        return Phi3Tokenizer(
            vocab_file=vocab_file or self.vocab_file,
            merges_file=merges_file or self.merges_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
        )

    # -- wrapped-class identity ----------------------------------------------
    def test_wrapped_class_identity(self):
        # warp_tokenizer mixes PaddleTokenizerMixin into GPT2Tokenizer; both
        # bases must be present.  (The garbage coverage test skipped this
        # claiming issubclass fails -- it does not.)
        self.assertTrue(issubclass(Phi3Tokenizer, PaddleTokenizerMixin))
        self.assertTrue(issubclass(Phi3Tokenizer, hf.GPT2Tokenizer))
        # warp_tokenizer names the dynamic class after the wrapped HF class.
        self.assertEqual(Phi3Tokenizer.__name__, "GPT2Tokenizer")

    # -- special tokens / vocab size -----------------------------------------
    def test_special_token_ids_pinned_and_distinct(self):
        tok = self._make_tokenizer()
        # ids are pinned in the vocab file this test wrote (independent oracle).
        self.assertEqual(tok.bos_token_id, _BOS_ID)
        self.assertEqual(tok.eos_token_id, _EOS_ID)
        self.assertEqual(tok.unk_token_id, _UNK_ID)
        self.assertEqual(tok.pad_token_id, _PAD_ID)
        # the four special ids must be distinct (guards all-map-to-one bug).
        self.assertEqual(
            len(
                {
                    tok.bos_token_id,
                    tok.eos_token_id,
                    tok.unk_token_id,
                    tok.pad_token_id,
                }
            ),
            4,
        )
        # id -> piece round-trips through the real decoder tables.
        self.assertEqual(tok.convert_ids_to_tokens(_BOS_ID), "<s>")
        self.assertEqual(tok.convert_ids_to_tokens(_PAD_ID), "<pad>")

    def test_vocab_size_and_get_vocab_mapping(self):
        tok = self._make_tokenizer()
        self.assertEqual(tok.vocab_size, _VOCAB_SIZE)
        vocab = tok.get_vocab()
        self.assertIsInstance(vocab, dict)
        # base byte tokens and specials map to exactly the hand-assigned ids.
        self.assertEqual(vocab["a"], _A_ID)
        self.assertEqual(vocab["b"], _B_ID)
        self.assertEqual(vocab["c"], _C_ID)
        self.assertEqual(vocab["ab"], _AB_ID)
        self.assertEqual(vocab["<s>"], _BOS_ID)
        self.assertEqual(vocab["</s>"], _EOS_ID)
        self.assertEqual(vocab["<unk>"], _UNK_ID)
        self.assertEqual(vocab["<pad>"], _PAD_ID)

    # -- token <-> id mapping -------------------------------------------------
    def test_convert_token_id_mapping(self):
        tok = self._make_tokenizer()
        self.assertEqual(tok.convert_tokens_to_ids("a"), _A_ID)
        self.assertEqual(tok.convert_tokens_to_ids("ab"), _AB_ID)
        self.assertEqual(tok.convert_ids_to_tokens(_AB_ID), "ab")
        self.assertEqual(tok.convert_ids_to_tokens(_C_ID), "c")
        # an unknown piece falls back to unk_id, not a silent 0/None.
        self.assertEqual(tok.convert_tokens_to_ids("zzz"), _UNK_ID)

    # -- encode: independent byte-level BPE derivation -----------------------
    def test_encode_matches_independent_bpe(self):
        tok = self._make_tokenizer()
        # "ab": the single pair (a,b) has rank 0 -> merges to "ab" = 3.
        self.assertEqual(tok.encode("ab", add_special_tokens=False), [_AB_ID])
        # "abc": only (a,b) is mergeable -> ["ab", "c"] = [3, 2].
        self.assertEqual(
            tok.encode("abc", add_special_tokens=False), [_AB_ID, _C_ID]
        )
        # "cba": no adjacent pair exists in ranks -> [c, b, a] = [2, 1, 0].
        self.assertEqual(
            tok.encode("cba", add_special_tokens=False),
            [_C_ID, _B_ID, _A_ID],
        )
        # "abcab": both (a,b) pairs merge -> ["ab","c","ab"] = [3, 2, 3].
        self.assertEqual(
            tok.encode("abcab", add_special_tokens=False),
            [_AB_ID, _C_ID, _AB_ID],
        )
        # single byte token.
        self.assertEqual(tok.encode("a", add_special_tokens=False), [_A_ID])

    def test_add_special_tokens_is_noop_for_gpt2(self):
        tok = self._make_tokenizer()
        # GPT2 does not inject bos/eos by default; add_special_tokens must not
        # change the id sequence (guards accidental bos/eos insertion).
        self.assertEqual(
            tok.encode("abcab", add_special_tokens=True),
            tok.encode("abcab", add_special_tokens=False),
        )

    def test_truncation_keeps_leading_tokens(self):
        tok = self._make_tokenizer()
        full = tok.encode("abcab", add_special_tokens=False)
        self.assertEqual(full, [_AB_ID, _C_ID, _AB_ID])
        truncated = tok.encode(
            "abcab", add_special_tokens=False, max_length=2, truncation=True
        )
        # right-truncation drops the tail, keeping the first two tokens.
        self.assertEqual(truncated, [_AB_ID, _C_ID])

    # -- decode / round-trip --------------------------------------------------
    def test_decode_round_trip(self):
        tok = self._make_tokenizer()
        # byte-level decode reassembles the original bytes.
        self.assertEqual(tok.decode([_A_ID, _B_ID, _C_ID]), "abc")
        self.assertEqual(tok.decode([_AB_ID, _C_ID]), "abc")
        for text in ("ab", "abc", "cba", "a", "abcab"):
            self.assertEqual(
                tok.decode(tok.encode(text, add_special_tokens=False)), text
            )

    # -- padding placement / attention mask ----------------------------------
    def test_pad_right_places_pad_and_builds_mask(self):
        tok = self._make_tokenizer()
        tok.padding_side = "right"
        out = tok.pad({"input_ids": [[3, 2, 3], [3]]}, padding="longest")
        self.assertEqual(out["input_ids"], [[3, 2, 3], [3, _PAD_ID, _PAD_ID]])
        self.assertEqual(out["attention_mask"], [[1, 1, 1], [1, 0, 0]])

    def test_pad_left_places_pad_and_builds_mask(self):
        tok = self._make_tokenizer()
        tok.padding_side = "left"
        out = tok.pad({"input_ids": [[3, 2, 3], [3]]}, padding="longest")
        self.assertEqual(out["input_ids"], [[3, 2, 3], [_PAD_ID, _PAD_ID, 3]])
        self.assertEqual(out["attention_mask"], [[1, 1, 1], [0, 0, 1]])

    # -- save + reload (new object) ------------------------------------------
    def test_save_reload_via_constructor(self):
        tok = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as save_dir:
            tok.save_pretrained(save_dir)
            saved_vocab = os.path.join(save_dir, "vocab.json")
            saved_merges = os.path.join(save_dir, "merges.txt")
            self.assertTrue(os.path.isfile(saved_vocab))
            self.assertTrue(os.path.isfile(saved_merges))
            # reload into a brand-new object from the saved assets.
            reloaded = self._make_tokenizer(
                vocab_file=saved_vocab, merges_file=saved_merges
            )
            self.assertEqual(reloaded.vocab_size, _VOCAB_SIZE)
            self.assertEqual(
                reloaded.encode("abcab", add_special_tokens=False),
                [_AB_ID, _C_ID, _AB_ID],
            )
            self.assertEqual(reloaded.decode([_AB_ID, _C_ID]), "abc")
            self.assertEqual(reloaded.bos_token_id, _BOS_ID)
            self.assertEqual(reloaded.pad_token_id, _PAD_ID)

    def test_save_reload_via_from_pretrained(self):
        # Exercises the production ``from_pretrained`` override on a local dir
        # (offline: existing local files skip the download resolver).
        tok = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as save_dir:
            tok.save_pretrained(save_dir)
            reloaded = Phi3Tokenizer.from_pretrained(save_dir)
            self.assertEqual(
                reloaded.encode("abcab", add_special_tokens=False),
                [_AB_ID, _C_ID, _AB_ID],
            )
            self.assertEqual(reloaded.decode([_AB_ID, _C_ID]), "abc")
            self.assertEqual(reloaded.bos_token_id, _BOS_ID)
            self.assertEqual(reloaded.pad_token_id, _PAD_ID)
            self.assertEqual(reloaded.vocab_size, _VOCAB_SIZE)

    # -- Paddle-tensor return path (mixin value-add) -------------------------
    @unittest.skipUnless(
        _HAS_PADDLE, "Paddle not importable; return_tensors='pd' path skipped."
    )
    def test_call_return_tensors_pd_yields_paddle_tensor(self):
        tok = self._make_tokenizer()
        # __call__ on the mixin converts list outputs to Paddle tensors when
        # return_tensors="pd"; the underlying ids must still be the BPE result.
        out = tok("abcab", return_tensors="pd", add_special_tokens=False)
        self.assertEqual(
            out["input_ids"].numpy().tolist(), [[_AB_ID, _C_ID, _AB_ID]]
        )


if __name__ == "__main__":
    unittest.main()
