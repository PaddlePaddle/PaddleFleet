# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest
from collections import Counter

import numpy as np

from paddlefleet.data import Vocab
from tests.formers import testing_utils
from tests.formers.common_test import CpuCommonTest


class TestVocab(CpuCommonTest):
    def create_counter(self):
        counter = Counter()
        counter["一万七千多"] = 2
        counter["一万七千余"] = 3
        counter["一万万"] = 1
        counter["一万七千多户"] = 3
        counter["一万七千"] = 4
        counter["一万七"] = 0
        self.counter = counter

    def setUp(self):
        self.create_counter()

    @testing_utils.assert_raises(ValueError)
    def test_invalid_specail_token(self):
        Vocab(wrong_kwarg="")

    @testing_utils.assert_raises(ValueError)
    def test_invalid_identifier(self):
        Vocab(counter=self.counter, _special_token="")

    @testing_utils.assert_raises(ValueError)
    def test_sort_index_value_error1(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "IP地址": 3}
        Vocab(
            counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx
        )

    @testing_utils.assert_raises(ValueError)
    def test_sort_index_value_error2(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万七千": 2}
        Vocab(
            counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx
        )

    @testing_utils.assert_raises(ValueError)
    def test_sort_index_value_error3(self):
        token_to_idx = {"一万七千多": -1, "一万七千余": 2, "一万七千": 3}
        Vocab(
            counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx
        )

    @testing_utils.assert_raises(ValueError)
    def test_to_token_excess_size(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万万": 3}
        vocab = Vocab(
            counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx
        )
        vocab.to_tokens(len(vocab))

    def test_counter(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万万": 3}
        vocab = Vocab(
            counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx
        )
        self.check_output_equal(vocab.to_tokens(1), "一万七千多")
        self.check_output_equal(vocab.to_tokens(2), "一万七千余")
        self.check_output_equal(vocab.to_tokens(3), "一万万")

    def test_json(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万万": 3}
        vocab = Vocab(
            counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx
        )
        json_str = vocab.to_json()
        copied_vocab = Vocab.from_json(json_str)
        for key, value in copied_vocab.token_to_idx.items():
            self.check_output_equal(value, vocab[key])

    def test_counter_frequency_then_alphabetical_order(self):
        counter = Counter()
        counter["b"] = 3
        counter["c"] = 2
        counter["a"] = 1
        counter["d"] = 1
        vocab = Vocab(counter=counter)
        # freq desc, ties broken alphabetically asc: b(3), c(2), a(1), d(1)
        self.check_output_equal(vocab.to_indices("b"), 0)
        self.check_output_equal(vocab.to_indices("c"), 1)
        self.check_output_equal(vocab.to_indices("a"), 2)
        self.check_output_equal(vocab.to_indices("d"), 3)
        self.check_output_equal(vocab.to_tokens(0), "b")
        self.check_output_equal(vocab.to_tokens(3), "d")
        self.check_output_equal(len(vocab), 4)

    def test_special_tokens_indexed_alphabetically_first(self):
        counter = Counter()
        counter["zebra"] = 1
        vocab = Vocab(counter=counter, unk_token="[UNK]", pad_token="[PAD]")
        # special names sorted: pad_token before unk_token
        self.check_output_equal(vocab.to_indices("[PAD]"), 0)
        self.check_output_equal(vocab.to_indices("[UNK]"), 1)
        self.check_output_equal(vocab.to_indices("zebra"), 2)
        self.check_output_equal(vocab.unk_token, "[UNK]")
        self.check_output_equal(vocab.pad_token, "[PAD]")
        self.check_output_equal(len(vocab), 3)

    def test_unknown_token_maps_to_unk_index(self):
        counter = Counter()
        counter["hello"] = 2
        counter["world"] = 1
        vocab = Vocab(counter=counter, unk_token="[UNK]")
        unk_id = vocab.to_indices("[UNK]")
        self.check_output_equal(vocab.to_indices("not_in_vocab"), unk_id)
        self.check_output_equal(vocab["another_oov"], unk_id)

    def test_contains_reflects_membership(self):
        counter = Counter()
        counter["hello"] = 2
        counter["world"] = 1
        vocab = Vocab(counter=counter, unk_token="[UNK]")
        self.assertIn("hello", vocab)
        self.assertIn("[UNK]", vocab)
        self.assertNotIn("missing", vocab)

    def test_max_size_keeps_highest_frequency_regular_tokens(self):
        counter = Counter()
        counter["a"] = 5
        counter["b"] = 4
        counter["c"] = 3
        counter["d"] = 2
        counter["e"] = 1
        vocab = Vocab(counter=counter, max_size=2, unk_token="[UNK]")
        # special token excluded from max_size budget: 1 special + 2 regular
        self.check_output_equal(len(vocab), 3)
        self.check_output_equal(vocab.to_indices("[UNK]"), 0)
        self.check_output_equal(vocab.to_indices("a"), 1)
        self.check_output_equal(vocab.to_indices("b"), 2)
        self.assertNotIn("c", vocab)

    def test_min_freq_filters_low_frequency_tokens(self):
        counter = Counter()
        counter["a"] = 3
        counter["b"] = 3
        counter["c"] = 1
        vocab = Vocab(counter=counter, min_freq=2)
        self.check_output_equal(len(vocab), 2)
        self.assertIn("a", vocab)
        self.assertIn("b", vocab)
        self.assertNotIn("c", vocab)

    def test_to_indices_and_to_tokens_roundtrip_list(self):
        counter = Counter()
        counter["b"] = 3
        counter["c"] = 2
        counter["a"] = 1
        vocab = Vocab(counter=counter)
        # order: b=0, c=1, a=2
        self.assertEqual(vocab.to_indices(["a", "b", "c"]), [2, 0, 1])
        self.assertEqual(vocab.to_tokens([0, 1, 2]), ["b", "c", "a"])
        self.assertEqual(vocab.to_tokens((2, 0)), ["a", "b"])
        self.assertEqual(vocab.to_tokens(np.array([1, 2])), ["c", "a"])

    @testing_utils.assert_raises(ValueError)
    def test_to_tokens_2d_array_raises(self):
        counter = Counter()
        counter["b"] = 3
        counter["c"] = 2
        vocab = Vocab(counter=counter)
        vocab.to_tokens(np.array([[0, 1], [1, 0]]))

    def test_build_vocab_frequency_order(self):
        vocab = Vocab.build_vocab([["a", "b", "a"], ["c"]])
        # counter a=2, b=1, c=1 -> a(0), b(1), c(2)
        self.check_output_equal(vocab.to_indices("a"), 0)
        self.check_output_equal(vocab.to_indices("b"), 1)
        self.check_output_equal(vocab.to_indices("c"), 2)
        self.check_output_equal(vocab.to_tokens(0), "a")

    def test_from_dict_preserves_mapping_and_unk_fallback(self):
        token_to_idx = {"[UNK]": 0, "foo": 1, "bar": 2}
        vocab = Vocab.from_dict(token_to_idx, unk_token="[UNK]")
        self.check_output_equal(vocab.to_indices("foo"), 1)
        self.check_output_equal(vocab.to_indices("bar"), 2)
        self.check_output_equal(vocab.to_indices("out_of_vocab"), 0)
        self.check_output_equal(vocab.to_tokens(2), "bar")
        self.check_output_equal(len(vocab), 3)

    def test_get_special_token_ids(self):
        counter = Counter()
        counter["x"] = 1
        vocab = Vocab(counter=counter, unk_token="[UNK]", pad_token="[PAD]")
        # sorted special names: pad_token(0) before unk_token(1)
        self.check_output_equal(vocab.get_pad_token_id(), 0)
        self.check_output_equal(vocab.get_unk_token_id(), 1)
        self.assertIsNone(vocab.get_bos_token_id())
        self.assertIsNone(vocab.get_eos_token_id())

    def test_save_and_load_vocabulary_preserves_order(self):
        counter = Counter()
        counter["apple"] = 2
        counter["banana"] = 1
        vocab = Vocab(counter=counter, unk_token="[UNK]", pad_token="[PAD]")
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        vocab.save_vocabulary(path)
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]
        # idx order: [PAD]=0, [UNK]=1, apple=2, banana=3
        self.assertEqual(lines, ["[PAD]", "[UNK]", "apple", "banana"])
        reloaded = Vocab.load_vocabulary(path)
        self.check_output_equal(reloaded.to_indices("[PAD]"), 0)
        self.check_output_equal(reloaded.to_indices("apple"), 2)
        self.check_output_equal(reloaded.to_indices("banana"), 3)
        self.check_output_equal(len(reloaded), 4)


if __name__ == "__main__":
    unittest.main()
