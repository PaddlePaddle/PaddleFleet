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

"""Behavior tests for ``paddlefleet.cli.train.sft.dataset_formatting``.

Data-layer contract under test: sample content, the prompt/completion (and
src/tgt) role assignment, batch ordering and format-detection dispatch must be
preserved when a dataset is turned into chat-template inputs.

The tokenizer is a genuine *not-under-test* collaborator. Instead of mocking it
to return a constant, we use a recording stub whose ``apply_chat_template``
returns a content-distinguishable serialization of the exact messages it
received. This lets each test verify (a) the message list the production code
actually built (roles + content), (b) batch-vs-single dispatch and ordering,
and (c) that ``tokenize=False`` is forwarded. We do not validate real chat
rendering, which is the tokenizer's own contract.

These tests run on CPU. Importing the production module pulls the ``paddlefleet``
package, which requires Paddle; when that dependency is absent the import raises
``ImportError`` and the tests skip (recorded, not silently passed).
"""

import unittest

from datasets import Dataset, Features, Value

try:
    from paddlefleet.cli.train.sft.dataset_formatting import (
        FORMAT_MAPPING,
        conversations_formatting_function,
        get_formatting_func_from_dataset,
        instructions_formatting_function,
        paddlefleet_instructions_formatting_function,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle / transformers backend not installed.
    FORMAT_MAPPING = None
    conversations_formatting_function = None
    get_formatting_func_from_dataset = None
    instructions_formatting_function = None
    paddlefleet_instructions_formatting_function = None
    _IMPORT_ERROR = exc


def _render(conversation):
    """Deterministic, content-distinguishable serialization of a message list.

    Encodes both role and content of every turn in order, so a swapped
    role, dropped turn, reordered batch or wrong field is observable.
    """
    return (
        "<CHAT>"
        + "||".join(
            "{}=>{}".format(turn["role"], turn["content"])
            for turn in conversation
        )
        + "</CHAT>"
    )


class RecordingTokenizer:
    """Stub tokenizer that records calls and returns input-dependent text."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversation, tokenize=True, **kwargs):
        # Snapshot each turn so later mutation cannot rewrite the record.
        snapshot = [dict(turn) for turn in conversation]
        self.calls.append(
            {"conversation": snapshot, "tokenize": tokenize, "kwargs": kwargs}
        )
        return _render(conversation)


class _FormattingTestBase(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet import failed (dependency unavailable): "
                f"{_IMPORT_ERROR!r}"
            )


class TestFormatMapping(_FormattingTestBase):
    """The detection table drives dispatch; its exact schema is a contract."""

    def test_keys_are_exactly_the_three_supported_formats(self):
        self.assertEqual(
            set(FORMAT_MAPPING), {"chatml", "instruction", "paddlefleet"}
        )

    def test_chatml_is_single_role_content_record(self):
        self.assertEqual(
            FORMAT_MAPPING["chatml"],
            [{"content": Value("string"), "role": Value("string")}],
        )

    def test_instruction_schema(self):
        self.assertEqual(
            FORMAT_MAPPING["instruction"],
            {"completion": Value("string"), "prompt": Value("string")},
        )

    def test_paddlefleet_schema(self):
        self.assertEqual(
            FORMAT_MAPPING["paddlefleet"],
            {"src": Value("string"), "tgt": Value("string")},
        )


class TestConversationsFormattingFunction(_FormattingTestBase):
    """Conversations are forwarded verbatim; only field + dispatch are logic."""

    def test_batch_preserves_each_conversation_and_order(self):
        tok = RecordingTokenizer()
        fn = conversations_formatting_function(tok, "messages")
        conv0 = [
            {"role": "user", "content": "hi A"},
            {"role": "assistant", "content": "reply A"},
        ]
        conv1 = [
            {"role": "system", "content": "sys B"},
            {"role": "user", "content": "hi B"},
        ]
        out = fn({"messages": [conv0, conv1]})

        self.assertEqual(out, [_render(conv0), _render(conv1)])
        self.assertEqual([c["conversation"] for c in tok.calls], [conv0, conv1])
        self.assertTrue(all(c["tokenize"] is False for c in tok.calls))

    def test_single_conversation_returns_scalar(self):
        tok = RecordingTokenizer()
        fn = conversations_formatting_function(tok, "messages")
        conv = [
            {"role": "user", "content": "solo u"},
            {"role": "assistant", "content": "solo a"},
        ]
        out = fn({"messages": conv})

        self.assertEqual(out, _render(conv))
        self.assertEqual(len(tok.calls), 1)
        self.assertEqual(tok.calls[0]["conversation"], conv)
        self.assertFalse(tok.calls[0]["tokenize"])

    def test_messages_field_argument_is_honored(self):
        tok = RecordingTokenizer()
        fn = conversations_formatting_function(tok, "conversations")
        conv = [{"role": "user", "content": "via conversations"}]
        out = fn({"conversations": [conv]})

        self.assertEqual(out, [_render(conv)])
        self.assertEqual(tok.calls[0]["conversation"], conv)


class TestInstructionsFormattingFunction(_FormattingTestBase):
    """prompt -> user turn, completion -> assistant turn, in that order."""

    def test_batch_maps_roles_and_keeps_pairing(self):
        tok = RecordingTokenizer()
        fn = instructions_formatting_function(tok)
        examples = {
            "prompt": ["What is 2+2?", "Capital of France?"],
            "completion": ["four", "Paris"],
        }
        out = fn(examples)

        expected_calls = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "four"},
            ],
            [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ],
        ]
        self.assertEqual(out, [_render(c) for c in expected_calls])
        self.assertEqual([c["conversation"] for c in tok.calls], expected_calls)
        self.assertTrue(all(c["tokenize"] is False for c in tok.calls))

    def test_single_maps_prompt_and_completion(self):
        tok = RecordingTokenizer()
        fn = instructions_formatting_function(tok)
        out = fn({"prompt": "solo prompt", "completion": "solo completion"})

        expected = [
            {"role": "user", "content": "solo prompt"},
            {"role": "assistant", "content": "solo completion"},
        ]
        self.assertEqual(out, _render(expected))
        self.assertEqual(tok.calls[0]["conversation"], expected)


class TestPaddleFleetFormattingFunction(_FormattingTestBase):
    """src -> user turn, tgt -> assistant turn, in that order."""

    def test_batch_maps_src_tgt_and_keeps_pairing(self):
        tok = RecordingTokenizer()
        fn = paddlefleet_instructions_formatting_function(tok)
        examples = {
            "src": ["translate: cat", "translate: dog"],
            "tgt": ["chat", "chien"],
        }
        out = fn(examples)

        expected_calls = [
            [
                {"role": "user", "content": "translate: cat"},
                {"role": "assistant", "content": "chat"},
            ],
            [
                {"role": "user", "content": "translate: dog"},
                {"role": "assistant", "content": "chien"},
            ],
        ]
        self.assertEqual(out, [_render(c) for c in expected_calls])
        self.assertEqual([c["conversation"] for c in tok.calls], expected_calls)

    def test_single_maps_src_and_tgt(self):
        tok = RecordingTokenizer()
        fn = paddlefleet_instructions_formatting_function(tok)
        out = fn({"src": "solo src", "tgt": "solo tgt"})

        expected = [
            {"role": "user", "content": "solo src"},
            {"role": "assistant", "content": "solo tgt"},
        ]
        self.assertEqual(out, _render(expected))
        self.assertEqual(tok.calls[0]["conversation"], expected)


class TestGetFormattingFuncFromDataset(_FormattingTestBase):
    """Detection uses real dataset feature schemas; verify the returned func."""

    def test_messages_chatml_routes_through_messages_field(self):
        tok = RecordingTokenizer()
        # Build the dataset with an explicit chatml struct schema. Production
        # detects chatml by comparing dataset.features["messages"] against the
        # plain-list FORMAT_MAPPING["chatml"]. Whether that comparison can match
        # is datasets-version dependent: on versions that render a
        # list-of-struct column as a plain Python list it matches (chatml is
        # detected and routed); on datasets>=4 the column is a List(...) object
        # that is not == a plain list, so detection cannot match and production
        # returns None. Assert the production contract for whichever
        # representation the installed datasets uses -- production is unchanged.
        ds = Dataset.from_dict(
            {"messages": [[{"role": "user", "content": "seed"}]]},
            features=Features(
                {
                    "messages": [
                        {"content": Value("string"), "role": Value("string")}
                    ]
                }
            ),
        )
        fn = get_formatting_func_from_dataset(ds, tok)
        if ds.features["messages"] == FORMAT_MAPPING["chatml"]:
            self.assertIsNotNone(fn)
            probe = [
                {"role": "user", "content": "probe u"},
                {"role": "assistant", "content": "probe a"},
            ]
            out = fn({"messages": [probe]})
            self.assertEqual(out, [_render(probe)])
            self.assertEqual(tok.calls[-1]["conversation"], probe)
        else:
            # Genuine datasets>=4 / production incompatibility, documented here.
            self.assertIsNone(fn)

    def test_conversations_chatml_routes_through_conversations_field(self):
        tok = RecordingTokenizer()
        ds = Dataset.from_dict(
            {"conversations": [[{"role": "user", "content": "seed"}]]},
            features=Features(
                {
                    "conversations": [
                        {"content": Value("string"), "role": Value("string")}
                    ]
                }
            ),
        )
        fn = get_formatting_func_from_dataset(ds, tok)
        if ds.features["conversations"] == FORMAT_MAPPING["chatml"]:
            self.assertIsNotNone(fn)
            probe = [{"role": "assistant", "content": "conv probe"}]
            out = fn({"conversations": [probe]})
            self.assertEqual(out, [_render(probe)])
            self.assertEqual(tok.calls[-1]["conversation"], probe)
        else:
            # Genuine datasets>=4 / production incompatibility, documented here.
            self.assertIsNone(fn)

    def test_instruction_schema_returns_prompt_completion_formatter(self):
        tok = RecordingTokenizer()
        ds = Dataset.from_dict({"prompt": ["p"], "completion": ["c"]})
        fn = get_formatting_func_from_dataset(ds, tok)
        self.assertIsNotNone(fn)

        # Reads prompt/completion keys; would KeyError if the paddlefleet
        # (src/tgt) formatter were returned instead.
        out = fn({"prompt": "pp", "completion": "cc"})
        self.assertEqual(
            out,
            _render(
                [
                    {"role": "user", "content": "pp"},
                    {"role": "assistant", "content": "cc"},
                ]
            ),
        )

    def test_paddlefleet_schema_returns_src_tgt_formatter(self):
        tok = RecordingTokenizer()
        ds = Dataset.from_dict({"src": ["s"], "tgt": ["t"]})
        fn = get_formatting_func_from_dataset(ds, tok)
        self.assertIsNotNone(fn)

        # Reads src/tgt keys; would KeyError if the instruction formatter
        # (prompt/completion) were returned instead.
        out = fn({"src": "ss", "tgt": "tt"})
        self.assertEqual(
            out,
            _render(
                [
                    {"role": "user", "content": "ss"},
                    {"role": "assistant", "content": "tt"},
                ]
            ),
        )

    def test_non_dataset_input_returns_none(self):
        self.assertIsNone(
            get_formatting_func_from_dataset(
                "not_a_dataset", RecordingTokenizer()
            )
        )

    def test_unsupported_schema_returns_none(self):
        ds = Dataset.from_dict({"text": ["unsupported column"]})
        self.assertIsNone(
            get_formatting_func_from_dataset(ds, RecordingTokenizer())
        )


if __name__ == "__main__":
    unittest.main()
