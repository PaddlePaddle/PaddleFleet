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

import unittest

from paddlefleet.datasets.template.formatter import (
    EmptyFormatter,
    FunctionFormatter,
    StringFormatter,
    ThinkingFormatter,
    ToolFormatter,
)


def default_tool_prompt(tool_text, tool_names):
    """Independently assemble the ``default`` tool-system prompt wrapper.

    Mirrors DEFAULT_TOOL_PROMPT by hand so the expected value never calls
    the production tool_formatter. ``tool_text`` is hand-derived per test.
    """
    return (
        "You have access to the following tools:\n"
        + tool_text
        + "Use the following format if using a tool:\n```\n"
        "Action: tool name (one of [" + tool_names + "])\n"
        "Action Input: the input to the tool, in a JSON format representing "
        'the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)\n'
        "```\n"
    )


class TestEmptyFormatter(unittest.TestCase):
    """EmptyFormatter passes its slots through unchanged and rejects
    slots that contain template placeholders."""

    def test_apply_passes_slots_through_verbatim(self):
        # Mixed str / set / dict slots must be returned in the same order
        # and with the same content and types.
        formatter = EmptyFormatter(
            slots=["<bos>", {"eos_token"}, {"role": "system"}, "<eos>"]
        )
        result = formatter.apply()
        self.assertEqual(
            result, ["<bos>", {"eos_token"}, {"role": "system"}, "<eos>"]
        )

    def test_apply_empty_slots_returns_empty_list(self):
        formatter = EmptyFormatter(slots=[])
        self.assertEqual(formatter.apply(), [])

    def test_placeholder_in_string_slot_is_rejected(self):
        with self.assertRaises(ValueError):
            EmptyFormatter(slots=["prefix {{name}} suffix"])

    def test_set_slot_is_not_placeholder_checked(self):
        # Only string slots are scanned for placeholders; a set slot with a
        # brace-like token must not trip the check and is passed through.
        formatter = EmptyFormatter(slots=[{"bos_token"}])
        self.assertEqual(formatter.apply(), [{"bos_token"}])


class TestStringFormatter(unittest.TestCase):
    """StringFormatter substitutes ``{{name}}`` placeholders with the
    matching keyword argument, replacing each placeholder at most once
    per slot and passing non-string slots through unchanged."""

    def test_single_placeholder_substituted_by_value(self):
        formatter = StringFormatter(slots=["Question: {{prompt}}"])
        result = formatter.apply(prompt="how tall is the tower")
        self.assertEqual(result, ["Question: how tall is the tower"])

    def test_each_placeholder_replaced_only_once_per_slot(self):
        # replace(..., 1) means only the FIRST occurrence of each placeholder
        # is filled; the trailing "{{a}}" must survive untouched.
        formatter = StringFormatter(slots=["{{a}}-{{b}}-{{a}}"])
        result = formatter.apply(a="X", b="Y")
        self.assertEqual(result, ["X-Y-{{a}}"])

    def test_placeholders_are_applied_per_slot(self):
        # Every kwarg is offered to every string slot; a placeholder only
        # changes the slot that actually contains it.
        formatter = StringFormatter(slots=["User: {{q}}", "Assistant: {{a}}"])
        result = formatter.apply(q="hello", a="world")
        self.assertEqual(result, ["User: hello", "Assistant: world"])

    def test_non_string_slots_pass_through_in_order(self):
        formatter = StringFormatter(slots=["{{content}}", {"eos_token"}])
        result = formatter.apply(content="payload")
        self.assertEqual(result, ["payload", {"eos_token"}])

    def test_missing_placeholder_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            StringFormatter(slots=["no placeholder here"])

    def test_non_string_value_for_placeholder_raises(self):
        formatter = StringFormatter(slots=["{{content}}"])
        with self.assertRaises(RuntimeError):
            formatter.apply(content=123)


class TestFunctionFormatter(unittest.TestCase):
    """FunctionFormatter parses a JSON tool-call payload and renders it
    through the ``default`` tool utils as ``Action`` / ``Action Input``
    lines, one per function, then fills the ``{{content}}`` slot."""

    def _formatter(self):
        return FunctionFormatter(slots=["{{content}}"], tool_format="default")

    def test_single_call_maps_name_and_arguments(self):
        result = self._formatter().apply(
            content='[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        )
        # arguments dict is re-serialized with json.dumps (space after ':').
        self.assertEqual(
            result, ['Action: get_weather\nAction Input: {"city": "Paris"}']
        )

    def test_parallel_calls_are_joined_in_order(self):
        result = self._formatter().apply(
            content='[{"name":"f1","arguments":{"a":1}},'
            '{"name":"f2","arguments":{"b":2}}]'
        )
        self.assertEqual(
            result,
            [
                'Action: f1\nAction Input: {"a": 1}\n'
                'Action: f2\nAction Input: {"b": 2}'
            ],
        )

    def test_type_function_wrapper_is_unwrapped(self):
        result = self._formatter().apply(
            content='[{"type": "function", "function": '
            '{"name": "myfunc", "arguments": {"x": 1}}}]'
        )
        self.assertEqual(result, ['Action: myfunc\nAction Input: {"x": 1}'])

    def test_single_object_is_treated_as_one_call(self):
        # A non-list JSON object is wrapped into a single-element call list.
        result = self._formatter().apply(
            content='{"name": "solo", "arguments": {"k": "v"}}'
        )
        self.assertEqual(result, ['Action: solo\nAction Input: {"k": "v"}'])

    def test_string_arguments_are_not_reserialized(self):
        # Already-string arguments are forwarded verbatim (no json.dumps).
        result = self._formatter().apply(
            content='[{"name": "f", "arguments": "raw_args"}]'
        )
        self.assertEqual(result, ["Action: f\nAction Input: raw_args"])

    def test_thought_block_is_extracted_and_prepended(self):
        result = self._formatter().apply(
            content="<think>reasoning</think>"
            '[{"name": "f", "arguments": {"a": 1}}]',
            thought_words=("<think>", "</think>"),
        )
        self.assertEqual(
            result,
            ['<think>reasoning</think>Action: f\nAction Input: {"a": 1}'],
        )

    def test_invalid_json_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self._formatter().apply(content="not valid json")


class TestToolFormatter(unittest.TestCase):
    """ToolFormatter renders a JSON tool schema into the ``default``
    system prompt, including parameter type, required flag and enum
    hints; an empty tool list yields an empty string."""

    def test_schema_renders_full_default_prompt(self):
        tools_json = (
            '[{"name": "get_weather", "description": "Query the weather", '
            '"parameters": {"type": "object", "properties": '
            '{"city": {"type": "string", "description": "Target city"}, '
            '"unit": {"type": "string", "description": "Unit", '
            '"enum": ["c", "f"]}}, "required": ["city"]}}]'
        )
        result = ToolFormatter(tool_format="default").apply(content=tools_json)

        # Hand-derived tool_text: required flag only on "city", enum hint
        # only on "unit", parameters kept in declared order.
        tool_text = (
            "> Tool Name: get_weather\n"
            "Tool Description: Query the weather\n"
            "Tool Args:\n"
            "  - city (string, required): Target city\n"
            "  - unit (string): Unit, should be one of [c, f]\n"
            "\n"
        )
        self.assertEqual(
            result, [default_tool_prompt(tool_text, "get_weather")]
        )

    def test_type_function_wrapper_is_unwrapped(self):
        tools_json = (
            '[{"type": "function", "function": {"name": "noop", '
            '"description": "does nothing", "parameters": '
            '{"type": "object", "properties": {}}}}]'
        )
        result = ToolFormatter(tool_format="default").apply(content=tools_json)

        tool_text = (
            "> Tool Name: noop\nTool Description: does nothing\nTool Args:\n\n"
        )
        self.assertEqual(result, [default_tool_prompt(tool_text, "noop")])

    def test_empty_tool_list_returns_empty_string(self):
        result = ToolFormatter(tool_format="default").apply(content="[]")
        self.assertEqual(result, [""])

    def test_invalid_json_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            ToolFormatter(tool_format="default").apply(content="not json")


class TestThinkingFormatter(unittest.TestCase):
    """ThinkingFormatter splits a response into a normalized reasoning
    block and the remaining content, emitting each as its own slot."""

    def _formatter(self):
        return ThinkingFormatter(slots=["{{content}}"])

    def test_reasoning_and_content_split_into_two_slots(self):
        result = self._formatter().apply(
            content="<think>\nmy reasoning\n</think>\nfinal answer",
            thought_words=("<think>", "</think>"),
        )
        # Reasoning is re-wrapped as "<think>\n...\n</think>\n"; the text
        # after the thought (leading newline kept) becomes the content slot.
        self.assertEqual(
            result,
            ["<think>\nmy reasoning\n</think>\n", "\nfinal answer"],
        )

    def test_content_without_thought_is_passed_through(self):
        result = self._formatter().apply(
            content="just answer, no thinking",
            thought_words=("<think>", "</think>"),
        )
        self.assertEqual(result, ["just answer, no thinking"])

    def test_thought_only_yields_only_reasoning_slot(self):
        # When nothing follows the thought block, only the reasoning slot
        # is returned (no empty content slot appended).
        result = self._formatter().apply(
            content="<think>\nonly reasoning\n</think>",
            thought_words=("<think>", "</think>"),
        )
        self.assertEqual(result, ["<think>\nonly reasoning\n</think>\n"])

    def test_empty_content_returns_empty_list(self):
        result = self._formatter().apply(
            content="", thought_words=("<think>", "</think>")
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
