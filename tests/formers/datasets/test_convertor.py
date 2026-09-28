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

"""Behavior tests for paddlefleet.datasets.reader.convertor.

The convertor functions turn ErnieKit-style raw examples (DPO / SFT / MM /
pretraining) into the OpenAI ``messages`` layout. These tests feed small,
content-distinguishable inputs and compare against expectations derived by
hand from the conversion contract (role alternation, chosen/rejected
selection by sort order, media placeholders, tool-call serialization, and the
documented validation errors). Assertions check exact message content,
ordering and role -- not shapes, key-presence or call counts -- so that a
swapped side, mis-placed placeholder or dropped turn would fail. All tests run
on CPU and require no accelerator.
"""

import unittest

from paddlefleet.datasets.reader.convertor import (
    convert_dpo_txt_data,
    convert_mm_data,
    convert_pretraining_data,
    convert_txt_data,
    erniekit_convertor,
)


class TestConvertDpoTxtData(unittest.TestCase):
    def test_higher_sort_index_selects_chosen(self):
        # sort[0] > sort[1] => response[0] is the chosen (preferred) branch.
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": ["chosentxt", "rejectedtxt"],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
        )
        self.assertEqual(
            result["chosen_response"],
            [{"role": "assistant", "content": "chosentxt"}],
        )
        self.assertEqual(
            result["rejected_response"],
            [{"role": "assistant", "content": "rejectedtxt"}],
        )

    def test_reversed_sort_swaps_chosen_and_rejected(self):
        # sort[0] < sort[1] => response[1] becomes chosen. Same inputs as the
        # previous test except the sort order, so a broken selection branch
        # cannot pass both.
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": ["firsttxt", "secondtxt"],
            "sort": [0, 1],
        }
        result = convert_dpo_txt_data(data)
        self.assertEqual(
            result["chosen_response"],
            [{"role": "assistant", "content": "secondtxt"}],
        )
        self.assertEqual(
            result["rejected_response"],
            [{"role": "assistant", "content": "firsttxt"}],
        )

    def test_multi_turn_response_alternates_roles(self):
        # A session-level response is List[List[str]]; within a branch the
        # 0th/2nd entries are assistant turns and odd entries are user turns.
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": [
                ["a1_good", "q2_good", "a2_good"],
                ["a1_bad", "q2_bad", "a2_bad"],
            ],
            "sort": [1, 0],
        }
        result = convert_dpo_txt_data(data)
        self.assertEqual(
            result["chosen_response"],
            [
                {"role": "assistant", "content": "a1_good"},
                {"role": "user", "content": "q2_good"},
                {"role": "assistant", "content": "a2_good"},
            ],
        )
        self.assertEqual(
            result["rejected_response"],
            [
                {"role": "assistant", "content": "a1_bad"},
                {"role": "user", "content": "q2_bad"},
                {"role": "assistant", "content": "a2_bad"},
            ],
        )

    def test_is_system_flag_lifts_first_src_turn_to_system(self):
        # is_system == 1 => src[0] becomes the system prompt and is removed
        # from the conversation turns.
        data = {
            "src": ["sysprompt", "q1"],
            "tgt": ["a1"],
            "response": ["chosentxt", "rejectedtxt"],
            "sort": [1, 0],
            "is_system": 1,
        }
        result = convert_dpo_txt_data(data)
        self.assertEqual(result["system"], "sysprompt")
        self.assertEqual(result["src"], ["q1"])
        self.assertEqual(
            result["messages"],
            [
                {"role": "system", "content": "sysprompt"},
                {"role": "user", "content": "q1"},
            ],
        )

    def test_explicit_system_field_prepends_system_message(self):
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": ["chosentxt", "rejectedtxt"],
            "sort": [1, 0],
            "system": "Helpful assistant",
        }
        result = convert_dpo_txt_data(data)
        self.assertEqual(
            result["messages"],
            [
                {"role": "system", "content": "Helpful assistant"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
        )
        self.assertEqual(
            result["chosen_response"],
            [{"role": "assistant", "content": "chosentxt"}],
        )

    def test_src_tgt_length_mismatch_raises(self):
        # Contract: len(src) must equal len(tgt) + 1.
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1", "a2"],
            "response": ["c", "r"],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_equal_sort_values_raise(self):
        # Preference cannot be decided when the two sort keys are equal.
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": ["c", "r"],
            "sort": [1, 1],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_even_number_of_responses_in_branch_raises(self):
        # Each response branch must hold an odd number of turns.
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": [["r1", "r2"], ["b1", "b2"]],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_blank_response_string_raises(self):
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": [" ", "bad"],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)

    def test_non_list_response_branch_raises(self):
        # When the branches are not both plain strings, each must already be a
        # list; a stray scalar is rejected.
        data = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": ["good", 123],
            "sort": [1, 0],
        }
        with self.assertRaises(ValueError):
            convert_dpo_txt_data(data)


class TestConvertTxtData(unittest.TestCase):
    def test_multi_turn_produces_alternating_messages(self):
        item = {"src": ["q1", "q2"], "tgt": ["a1", "a2"]}
        result = convert_txt_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        )

    def test_explicit_system_prepends_system_message(self):
        item = {"src": ["q1"], "tgt": ["a1"], "system": "You are helpful."}
        result = convert_txt_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        )

    def test_is_system_flag_lifts_first_turn_to_system(self):
        # is_system == 1 => the first (src, tgt) turn is consumed as the
        # system prompt (src[0]) and dropped from the conversation.
        item = {
            "src": ["sys_q", "q1"],
            "tgt": ["sys_a", "a1"],
            "is_system": 1,
        }
        result = convert_txt_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "system", "content": "sys_q"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        )

    def test_empty_system_string_adds_no_system_message(self):
        item = {"src": ["q1"], "tgt": ["a1"], "system": ""}
        result = convert_txt_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        )

    def test_empty_src_raises(self):
        with self.assertRaises(ValueError):
            convert_txt_data({"src": [], "tgt": ["a1"]})

    def test_whitespace_only_turn_raises(self):
        with self.assertRaises(ValueError):
            convert_txt_data({"src": ["  "], "tgt": ["a1"]})

    def test_src_tgt_label_length_mismatch_raises(self):
        item = {"src": ["q1", "q2"], "tgt": ["a1", "a2"], "label": [1]}
        with self.assertRaises(ValueError):
            convert_txt_data(item)

    def test_non_string_system_raises(self):
        item = {"src": ["q1"], "tgt": ["a1"], "system": 123}
        with self.assertRaises(ValueError):
            convert_txt_data(item)


class TestConvertMmData(unittest.TestCase):
    def test_image_placeholder_inserted_and_url_collected(self):
        # The image is anchored at matched_text_index 0, so "<image>" is
        # prepended to the masked (user) turn and its url is collected.
        item = {
            "text_info": [
                {"tag": "mask", "text": "what is this"},
                {"tag": "no_mask", "text": "a cat"},
            ],
            "image_info": [
                {"image_url": "http://img", "matched_text_index": 0}
            ],
        }
        result = convert_mm_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "<image>what is this"},
                {"role": "assistant", "content": "a cat"},
            ],
        )
        self.assertEqual(result["images"], ["http://img"])
        self.assertNotIn("videos", result)

    def test_video_placeholder_inserted_and_url_collected(self):
        item = {
            "text_info": [
                {"tag": "mask", "text": "what is this"},
                {"tag": "no_mask", "text": "a video"},
            ],
            "video_info": [
                {"image_url": "http://vid", "matched_text_index": 0}
            ],
        }
        result = convert_mm_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "<video>what is this"},
                {"role": "assistant", "content": "a video"},
            ],
        )
        self.assertEqual(result["videos"], ["http://vid"])
        self.assertNotIn("images", result)

    def test_tool_response_mask_becomes_observation(self):
        # A masked turn flagged tool_response=True is emitted with the
        # "observation" role; a plain masked turn stays "user".
        item = {
            "text_info": [
                {"tag": "mask", "text": "q"},
                {"tag": "no_mask", "text": "ans"},
                {"tag": "mask", "text": "tool out", "tool_response": True},
            ],
        }
        result = convert_mm_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "ans"},
                {"role": "observation", "content": "tool out"},
            ],
        )

    def test_list_tool_calls_serialized_to_json(self):
        # A list-valued tool_calls is json-serialized and attached to the
        # assistant message alongside its content.
        item = {
            "text_info": [
                {
                    "tag": "no_mask",
                    "text": "call",
                    "tool_calls": [{"name": "f"}],
                },
            ],
        }
        result = convert_mm_data(item)
        self.assertEqual(
            result["messages"],
            [
                {
                    "role": "assistant",
                    "content": "call",
                    "tool_calls": '[{"name": "f"}]',
                },
            ],
        )

    def test_system_field_prepended(self):
        item = {
            "system": "SYS helper",
            "text_info": [{"tag": "no_mask", "text": "hi"}],
        }
        result = convert_mm_data(item)
        self.assertEqual(
            result["messages"],
            [
                {"role": "system", "content": "SYS helper"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_image_and_video_without_order_raises(self):
        # When both media kinds are present the caller must supply "order".
        item = {
            "text_info": [{"tag": "no_mask", "text": "describe"}],
            "image_info": [{"image_url": "i", "matched_text_index": 0}],
            "video_info": [{"image_url": "v", "matched_text_index": 0}],
        }
        with self.assertRaises(AssertionError):
            convert_mm_data(item)


class TestConvertPretrainingData(unittest.TestCase):
    def test_string_text_becomes_single_assistant_message(self):
        result = convert_pretraining_data({"text": "hello world"})
        self.assertEqual(
            result["messages"],
            [{"role": "assistant", "content": "hello world"}],
        )

    def test_list_text_uses_only_first_element(self):
        # A list value collapses to its first element; the second is ignored.
        result = convert_pretraining_data({"text": ["first", "second"]})
        self.assertEqual(
            result["messages"],
            [{"role": "assistant", "content": "first"}],
        )

    def test_blank_text_raises(self):
        with self.assertRaises(ValueError):
            convert_pretraining_data({"text": "   "})

    def test_non_string_text_raises(self):
        with self.assertRaises(AssertionError):
            convert_pretraining_data({"text": 123})


class TestErniekitConvertorDispatch(unittest.TestCase):
    def test_routes_dpo_when_response_present(self):
        item = {
            "src": ["q1", "q2"],
            "tgt": ["a1"],
            "response": ["chosentxt", "rejectedtxt"],
            "sort": [1, 0],
        }
        result = erniekit_convertor(item)
        # DPO path produces chosen/rejected with the higher-sort branch chosen.
        self.assertEqual(
            result["chosen_response"],
            [{"role": "assistant", "content": "chosentxt"}],
        )
        self.assertEqual(
            result["rejected_response"],
            [{"role": "assistant", "content": "rejectedtxt"}],
        )

    def test_routes_sft_when_only_src_tgt(self):
        result = erniekit_convertor({"src": ["q1"], "tgt": ["a1"]})
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        )
        # SFT output must not carry DPO-only fields.
        self.assertNotIn("chosen_response", result)

    def test_routes_pretraining_when_text_present(self):
        result = erniekit_convertor({"text": "pretrain body"})
        self.assertEqual(
            result["messages"],
            [{"role": "assistant", "content": "pretrain body"}],
        )

    def test_routes_mm_when_no_text_keys(self):
        result = erniekit_convertor(
            {"text_info": [{"tag": "no_mask", "text": "hi"}]}
        )
        self.assertEqual(
            result["messages"],
            [{"role": "assistant", "content": "hi"}],
        )


if __name__ == "__main__":
    unittest.main()
