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

"""Behavior tests for paddlefleet.datasets.data_utils.

These tests exercise the real data-utility functions with content
distinguishable inputs and independently computed expectations. Tokenizers
are replaced by small, real (non-mock) stubs whose token/id/decode behavior
is deterministic and reproducible in the test, so that the assertions observe
the orchestration performed by the production code (join, tokenize, truncate,
accumulate, pack, shard) rather than the stub itself. All tests run on CPU
and require no accelerator.
"""

import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

from paddlefleet.datasets.data_utils import (
    calculate_matched_group,
    convert_to_input_ids,
    convert_to_tokens_for_pt,
    convert_to_tokens_for_sft,
    generate_greedy_packs_from_sequences,
    get_worker_sliced_iterator,
    print_debug_info,
    round_up_to_multiple_of_8,
)


class CharTokenizer:
    """Character-level tokenizer stub with reproducible behavior.

    Each character of a string is one token; its id is ``ord(char)``. This
    lets the test recompute the expected ids independently while the
    production code performs the join/tokenize/convert orchestration.
    """

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        return [ord(t) for t in tokens]


class RecordingDecodeTokenizer:
    """Records the argument passed to ``decode`` and optionally raises."""

    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.received = None

    def decode(self, data):
        self.received = data
        if self.raise_exc is not None:
            raise self.raise_exc
        return "TOKENS<" + "-".join(str(x) for x in data) + ">"


class ChatTokenizer:
    """Returns a fixed list of (src, target) turn encodings.

    ``chat_template`` is truthy so the production code must not call
    ``init_chat_template``; doing so is treated as a failure.
    """

    def __init__(self, encoded_messages):
        self.chat_template = "dummy-template"
        self._encoded = encoded_messages
        self.seen_messages = None

    def init_chat_template(self, template):
        raise AssertionError(
            "init_chat_template must not be called when chat_template is set"
        )

    def encode_chat_inputs(self, payload):
        self.seen_messages = payload["messages"]
        return self._encoded


class Seq:
    """Minimal sequence object exposing ``token_ids`` for greedy packing."""

    def __init__(self, token_ids):
        self.token_ids = token_ids


class TestRoundUpToMultipleOf8(unittest.TestCase):
    def test_rounds_up_against_independent_arithmetic(self):
        # Independent reference uses integer arithmetic; production uses
        # bit ops. They must agree for every value, catching off-by-one.
        for n in [0, 1, 5, 7, 8, 9, 15, 16, 17, 100, 1000, 1001, 1023]:
            expected = ((n + 7) // 8) * 8
            self.assertEqual(round_up_to_multiple_of_8(n), expected)

    def test_specific_boundaries(self):
        self.assertEqual(round_up_to_multiple_of_8(0), 0)
        self.assertEqual(round_up_to_multiple_of_8(1), 8)
        self.assertEqual(round_up_to_multiple_of_8(8), 8)
        self.assertEqual(round_up_to_multiple_of_8(9), 16)
        self.assertEqual(round_up_to_multiple_of_8(1023), 1024)


class TestPrintDebugInfo(unittest.TestCase):
    def test_decodes_data_and_logs_decoded_text(self):
        tokenizer = RecordingDecodeTokenizer()
        data = [72, 73, 33]
        with self.assertLogs("PaddleFleet", level="INFO") as cm:
            print_debug_info(tokenizer, data, "prompt")
        # The exact token ids reach decode ...
        self.assertEqual(tokenizer.received, data)
        # ... and the decoded text plus the label appear in the log line.
        joined = "\n".join(cm.output)
        self.assertIn("TOKENS<72-73-33>", joined)
        self.assertIn("prompt", joined)

    def test_swallows_typeerror_from_decode(self):
        tokenizer = RecordingDecodeTokenizer(raise_exc=TypeError("bad"))
        # TypeError from decode must be caught, not propagated, and the
        # call must actually have reached decode.
        with self.assertLogs("PaddleFleet", level="INFO"):
            result = print_debug_info(tokenizer, [1, 2, 3], "label")
        self.assertIsNone(result)
        self.assertEqual(tokenizer.received, [1, 2, 3])

    def test_does_not_swallow_unlisted_exception(self):
        tokenizer = RecordingDecodeTokenizer(raise_exc=KeyError("boom"))
        # KeyError is outside the caught set and must propagate.
        with self.assertRaises(KeyError):
            print_debug_info(tokenizer, [1, 2, 3], "label")


class TestConvertToTokensForPT(unittest.TestCase):
    def test_joins_contents_with_newline_then_tokenizes(self):
        tokenizer = CharTokenizer()
        dial = [{"content": "ab"}, {"content": "c"}]
        tokens = convert_to_tokens_for_pt(dial, tokenizer, max_src_len=1024)
        # "ab" + "\n" + "c" -> characters; newline joiner must be present.
        self.assertEqual(tokens, ["a", "b", "\n", "c"])

    def test_truncation_keeps_head_half_and_tail(self):
        tokenizer = CharTokenizer()
        text = "abcdefghijklmnopqrst"  # 20 distinct characters
        self.assertEqual(len(text), 20)
        dial = [{"content": text}]
        result = convert_to_tokens_for_pt(dial, tokenizer, max_src_len=10)
        full = list(text)
        # Documented head+tail strategy: first max_src_len//2, then last
        # max_src_len tokens (the two windows overlap by construction).
        expected = full[: 10 // 2] + full[-10:]
        self.assertEqual(result, expected)
        self.assertEqual(result, list("abcde") + list("klmnopqrst"))

    def test_no_truncation_when_within_limit(self):
        tokenizer = CharTokenizer()
        dial = [{"content": "hello"}]
        result = convert_to_tokens_for_pt(dial, tokenizer, max_src_len=1024)
        self.assertEqual(result, list("hello"))


class TestConvertToTokensForSFT(unittest.TestCase):
    def test_assembles_all_turns_when_budget_allows(self):
        encoded = [([1, 2], [3, 4]), ([5, 6], [7, 8]), ([9], [10])]
        tokenizer = ChatTokenizer(encoded)
        dial = [{"role": "user", "content": "x"}]
        tokens = convert_to_tokens_for_sft(dial, tokenizer, max_src_len=100)
        # Last turn contributes only its src ([9]); earlier turns prepend
        # src+target in order.
        self.assertEqual(tokens, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        # The dialogue is forwarded verbatim to encode_chat_inputs.
        self.assertEqual(tokenizer.seen_messages, dial)

    def test_partial_inclusion_when_budget_limited(self):
        encoded = [([1], [2]), ([3], [4]), ([5], [6])]
        tokenizer = ChatTokenizer(encoded)
        dial = [{"role": "user", "content": "x"}]
        # With max_src_len=11 the second turn fits but the first does not,
        # so only turn[1] is prepended onto the last turn's src.
        tokens = convert_to_tokens_for_sft(dial, tokenizer, max_src_len=11)
        self.assertEqual(tokens, [3, 4, 5])

    def test_only_last_turn_src_when_budget_tiny(self):
        encoded = [([1, 2], [3, 4]), ([5, 6], [7, 8]), ([9], [10])]
        tokenizer = ChatTokenizer(encoded)
        dial = [{"role": "user", "content": "x"}]
        tokens = convert_to_tokens_for_sft(dial, tokenizer, max_src_len=5)
        self.assertEqual(tokens, [9])


class TestConvertToInputIds(unittest.TestCase):
    def test_base_format_maps_each_dialogue_to_ids(self):
        tokenizer = CharTokenizer()
        dials = [[{"content": "ab"}], [{"content": "cde"}]]
        input_ids, num_tokens = convert_to_input_ids(
            dials, tokenizer, "base", 1024
        )
        # ids are ord() of each character; distinct dialogues -> distinct ids.
        self.assertEqual(input_ids, [[97, 98], [99, 100, 101]])
        self.assertEqual(num_tokens, 5)

    def test_base_format_joins_multi_content_before_ids(self):
        tokenizer = CharTokenizer()
        dials = [[{"content": "ab"}, {"content": "c"}]]
        input_ids, num_tokens = convert_to_input_ids(
            dials, tokenizer, "base", 1024
        )
        # "ab\nc" -> ['a','b','\n','c'] -> [97, 98, 10, 99]
        self.assertEqual(input_ids, [[97, 98, 10, 99]])
        self.assertEqual(num_tokens, 4)

    def test_chat_format_delegates_to_sft_assembly(self):
        encoded = [([1, 2], [3, 4]), ([9], [10])]
        tokenizer = ChatTokenizer(encoded)
        dials = [[{"role": "user", "content": "x"}]]
        input_ids, num_tokens = convert_to_input_ids(
            dials, tokenizer, "chat", 100
        )
        self.assertEqual(input_ids, [[1, 2, 3, 4, 9]])
        self.assertEqual(num_tokens, 5)

    def test_invalid_format_raises_value_error(self):
        tokenizer = CharTokenizer()
        dials = [[{"content": "ab"}]]
        with self.assertRaises(ValueError):
            convert_to_input_ids(dials, tokenizer, "unsupported", 1024)


class TestCalculateMatchedGroup(unittest.TestCase):
    @staticmethod
    def _flatten(bins):
        ids = []
        for group in bins:
            for item in group:
                ids.append(item[0])
        return ids

    def test_empty_returns_two_empty_lists(self):
        packed, ret = calculate_matched_group([], 500)
        self.assertEqual(packed, [])
        self.assertEqual(ret, [])

    def test_packing_preserves_all_items_and_respects_capacity(self):
        sequences = [
            ("s0", 100, None),
            ("s1", 200, None),
            ("s2", 300, None),
            ("s3", 400, None),
        ]
        packed, ret = calculate_matched_group(sequences, 500, is_finished=True)
        # is_finished=True keeps nothing as carry-over.
        self.assertEqual(ret, [])
        # Every original sequence appears exactly once (no loss / dup).
        self.assertCountEqual(self._flatten(packed), ["s0", "s1", "s2", "s3"])
        # Total weight 1000 with volume 500 forces more than one bin.
        self.assertGreater(len(packed), 1)
        # Each produced bin fits within the packing length (all items <= 500).
        for group in packed:
            self.assertLessEqual(sum(item[1] for item in group), 500)

    def test_unfinished_carries_last_bin_as_ret(self):
        sequences = [
            ("s0", 100, None),
            ("s1", 200, None),
            ("s2", 300, None),
            ("s3", 400, None),
        ]
        packed, ret = calculate_matched_group(sequences, 500, is_finished=False)
        # The carried-over group is a non-empty bin, excluded from packed.
        self.assertIsInstance(ret, list)
        self.assertGreater(len(ret), 0)
        # packed + ret together still account for all items exactly once.
        recombined = self._flatten(packed) + [item[0] for item in ret]
        self.assertCountEqual(recombined, ["s0", "s1", "s2", "s3"])
        # ret's items are not also present inside packed.
        ret_ids = {item[0] for item in ret}
        self.assertTrue(ret_ids.isdisjoint(self._flatten(packed)))


class TestGenerateGreedyPacksFromSequences(unittest.TestCase):
    def test_fills_open_pack_before_opening_new_one(self):
        s0 = Seq([10, 11, 12, 13])  # length 4
        s1 = Seq([20, 21, 22])  # length 3
        s2 = Seq([30, 31])  # length 2
        packs = generate_greedy_packs_from_sequences(8, [s0, s1, s2])
        # s0+s1 (=7) fit the first pack; s2 (would overflow to 9) opens a new
        # pack. Identity is checked to confirm placement, not just counts.
        self.assertEqual(len(packs), 2)
        self.assertEqual(packs[0], [s0, s1])
        self.assertEqual(packs[1], [s2])
        for pack in packs:
            self.assertLessEqual(sum(len(seq.token_ids) for seq in pack), 8)

    def test_oversized_first_item_forces_second_pack(self):
        s0 = Seq(list(range(6)))  # length 6
        s1 = Seq(list(range(100, 106)))  # length 6
        packs = generate_greedy_packs_from_sequences(8, [s0, s1])
        self.assertEqual(len(packs), 2)
        self.assertEqual(packs[0], [s0])
        self.assertEqual(packs[1], [s1])

    def test_all_sequences_preserved_across_packs(self):
        seqs = [Seq([i]) for i in range(5)]
        packs = generate_greedy_packs_from_sequences(2, seqs)
        placed = [seq for pack in packs for seq in pack]
        self.assertCountEqual(placed, seqs)

    def test_empty_sequences_raise_index_error(self):
        with self.assertRaises(IndexError):
            generate_greedy_packs_from_sequences(8, [])


class TestGetWorkerSlicedIterator(unittest.TestCase):
    def test_no_worker_info_cycles_the_dataset(self):
        with mock.patch(
            "paddlefleet.datasets.data_utils.paddle.io.get_worker_info",
            return_value=None,
        ):
            iterator = get_worker_sliced_iterator([1, 2, 3])
            first_seven = list(itertools.islice(iterator, 7))
        # Without workers the iterator repeats the dataset endlessly.
        self.assertEqual(first_seven, [1, 2, 3, 1, 2, 3, 1])

    def test_workers_receive_disjoint_strided_shards(self):
        dataset = [10, 11, 12, 13, 14, 15]

        def sliced(worker_id):
            worker = SimpleNamespace(id=worker_id, num_workers=2)
            with mock.patch(
                "paddlefleet.datasets.data_utils.paddle.io.get_worker_info",
                return_value=worker,
            ):
                iterator = get_worker_sliced_iterator(dataset)
                return list(itertools.islice(iterator, 5))

        # Worker 0 starts at offset 0, worker 1 at offset 1, both step by 2
        # over the (infinitely cycled) dataset.
        self.assertEqual(sliced(0), [10, 12, 14, 10, 12])
        self.assertEqual(sliced(1), [11, 13, 15, 11, 13])


if __name__ == "__main__":
    unittest.main()
