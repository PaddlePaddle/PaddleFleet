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

import json
import unittest
from datetime import datetime

from paddlefleet.datasets.template.tool_utils import (
    DefaultToolUtils,
    ERNIEToolUtils,
    ERNIEVLToolUtils,
    FunctionCall,
    GLM4MOEToolUtils,
    GLM4ToolUtils,
    GLM_5ToolUtils,
    Llama3ToolUtils,
    QwenToolUtils,
    get_tool_utils,
)

# A single tool described in the OpenAI "type/function" wrapped form, with
# content-distinguishable names, descriptions and per-parameter attributes so
# that a formatter dropping a field or mixing parameters is observable.
WRAPPED_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Query current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Target city"},
                "unit": {
                    "type": "string",
                    "description": "Temperature unit",
                    "enum": ["celsius", "fahrenheit"],
                },
                "days": {
                    "type": "array",
                    "description": "Forecast days",
                    "items": {"type": "integer"},
                },
            },
            "required": ["city"],
        },
    },
}

# The same tool without the outer "type"/"function" envelope. Templates that
# accept bare tools must wrap it themselves; keeping the payload identical lets
# the wrapping (not the content) be the thing under test.
BARE_ADD_TOOL = {
    "name": "add",
    "description": "Add numbers",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
    },
}


class TestFunctionCall(unittest.TestCase):
    def test_fields_and_positional_order(self):
        fc = FunctionCall(name="search", arguments='{"q": "hello"}')
        self.assertEqual(fc.name, "search")
        self.assertEqual(fc.arguments, '{"q": "hello"}')
        # The formatters iterate ``for name, arguments in functions``; the tuple
        # order must be (name, arguments), not the reverse.
        self.assertEqual(tuple(fc), ("search", '{"q": "hello"}'))


class TestDefaultToolUtils(unittest.TestCase):
    def test_tool_formatter_renders_full_prompt(self):
        # Expected string derived by hand from the ReAct-style template:
        # required / enum / items each contribute a distinct suffix, and the
        # tool name is echoed into the "one of [...]" action line.
        expected = (
            "You have access to the following tools:\n"
            "> Tool Name: get_weather\n"
            "Tool Description: Query current weather\n"
            "Tool Args:\n"
            "  - city (string, required): Target city\n"
            "  - unit (string): Temperature unit, "
            "should be one of [celsius, fahrenheit]\n"
            "  - days (array): Forecast days, "
            "where each item should be integer\n"
            "\n"
            "Use the following format if using a tool:\n"
            "```\n"
            "Action: tool name (one of [get_weather])\n"
            "Action Input: the input to the tool, in a JSON format "
            "representing the kwargs "
            '(e.g. ```{"input": "hello world", "num_beams": 5}```)\n'
            "```\n"
        )
        self.assertEqual(
            DefaultToolUtils.tool_formatter([WRAPPED_WEATHER_TOOL]), expected
        )

    def test_tool_formatter_lists_every_tool_name_in_order(self):
        tools = [
            {
                "name": "alpha",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "beta",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
        result = DefaultToolUtils.tool_formatter(tools)
        # Both names appear, joined in input order inside the action line.
        self.assertIn("Action: tool name (one of [alpha, beta])\n", result)
        self.assertIn("> Tool Name: alpha\n", result)
        self.assertIn("> Tool Name: beta\n", result)

    def test_function_formatter_joins_action_blocks(self):
        functions = [
            FunctionCall(name="get_weather", arguments='{"city": "Paris"}'),
            FunctionCall(name="lookup", arguments='{"q": "x"}'),
        ]
        expected = (
            'Action: get_weather\nAction Input: {"city": "Paris"}\n'
            'Action: lookup\nAction Input: {"q": "x"}'
        )
        self.assertEqual(
            DefaultToolUtils.function_formatter(functions), expected
        )


class TestQwenToolUtils(unittest.TestCase):
    def test_tool_formatter_wraps_bare_tool_as_function(self):
        result = QwenToolUtils.tool_formatter([BARE_ADD_TOOL])
        # The bare tool must be re-wrapped in a {"type": "function", ...}
        # envelope and embedded verbatim (compact JSON) inside <tools>.
        expected_tool_json = (
            '{"type": "function", "function": '
            '{"name": "add", "description": "Add numbers", '
            '"parameters": {"type": "object", '
            '"properties": {"a": {"type": "integer"}}}}}'
        )
        self.assertIn("<tools>\n" + expected_tool_json + "\n</tools>", result)
        self.assertTrue(result.startswith("\n\n# Tools\n\n"))

    def test_tool_formatter_keeps_already_wrapped_tool(self):
        result = QwenToolUtils.tool_formatter([WRAPPED_WEATHER_TOOL])
        # An already-wrapped tool must not be double-wrapped.
        self.assertNotIn('"function": {"type": "function"', result)
        self.assertIn('"name": "get_weather"', result)

    def test_function_formatter_reparses_arguments_json(self):
        # arguments arrive as a JSON *string*; the Qwen template parses it and
        # re-emits an object under the "arguments" key inside <tool_call>.
        functions = [FunctionCall(name="add", arguments='{"a": 1, "b": 2}')]
        expected = '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>'
        self.assertEqual(QwenToolUtils.function_formatter(functions), expected)


class TestGLM4ToolUtils(unittest.TestCase):
    def test_tool_formatter_uses_glm_persona_and_indented_json(self):
        result = GLM4ToolUtils.tool_formatter([BARE_ADD_TOOL])
        self.assertIn("你是一个名为 ChatGLM 的人工智能助手", result)
        self.assertIn("# 可用工具\n\n## add\n\n", result)
        # GLM-4 dumps the tool body with indent=4 (pretty-printed), unlike the
        # compact JSON used by Qwen.
        self.assertIn('{\n    "name": "add",', result)
        self.assertIn(
            "在调用上述函数时，请使用 Json 格式表示调用的参数。", result
        )

    def test_function_formatter_single_call_name_newline_arguments(self):
        functions = [FunctionCall(name="add", arguments='{"a": 1}')]
        self.assertEqual(
            GLM4ToolUtils.function_formatter(functions), 'add\n{"a": 1}'
        )

    def test_function_formatter_rejects_parallel_calls(self):
        functions = [
            FunctionCall(name="f1", arguments='{"a": 1}'),
            FunctionCall(name="f2", arguments='{"b": 2}'),
        ]
        with self.assertRaises(ValueError):
            GLM4ToolUtils.function_formatter(functions)


class TestGLM4MOEToolUtils(unittest.TestCase):
    def test_function_formatter_serializes_only_non_string_values(self):
        # String argument values are inlined raw; non-string values (here a
        # list) are JSON-serialized. Mixing both in one call distinguishes the
        # two branches and preserves key order.
        functions = [
            FunctionCall(
                name="run", arguments='{"path": "/tmp/x", "flags": [1, 2]}'
            )
        ]
        expected = (
            "\n<tool_call>run"
            "\n<arg_key>path</arg_key>\n<arg_value>/tmp/x</arg_value>"
            "\n<arg_key>flags</arg_key>\n<arg_value>[1, 2]</arg_value>"
        )
        self.assertEqual(
            GLM4MOEToolUtils.function_formatter(functions), expected
        )


class TestGLM5ToolUtils(unittest.TestCase):
    def test_function_formatter_has_no_newline_separators(self):
        # GLM-5 differs from GLM-4-MOE precisely by dropping every "\n"
        # separator and closing each call with </tool_call>; parallel calls are
        # concatenated with no delimiter.
        functions = [
            FunctionCall(name="add", arguments='{"a": 1}'),
            FunctionCall(name="sub", arguments='{"b": 2}'),
        ]
        expected = (
            "<tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"
            "<tool_call>sub<arg_key>b</arg_key><arg_value>2</arg_value></tool_call>"
        )
        result = GLM_5ToolUtils.function_formatter(functions)
        self.assertEqual(result, expected)
        self.assertNotIn("\n", result)


class TestLlama3ToolUtils(unittest.TestCase):
    def test_function_formatter_single_call_is_object(self):
        functions = [FunctionCall(name="add", arguments='{"a": 1}')]
        result = Llama3ToolUtils.function_formatter(functions)
        # A lone call collapses to a single JSON object (not a list) with the
        # arguments re-keyed under "parameters".
        self.assertEqual(result, '{"name": "add", "parameters": {"a": 1}}')
        self.assertEqual(
            json.loads(result), {"name": "add", "parameters": {"a": 1}}
        )

    def test_function_formatter_multiple_calls_is_list(self):
        functions = [
            FunctionCall(name="add", arguments='{"a": 1}'),
            FunctionCall(name="sub", arguments='{"b": 2}'),
        ]
        result = Llama3ToolUtils.function_formatter(functions)
        self.assertEqual(
            json.loads(result),
            [
                {"name": "add", "parameters": {"a": 1}},
                {"name": "sub", "parameters": {"b": 2}},
            ],
        )

    def test_tool_formatter_embeds_current_date(self):
        result = Llama3ToolUtils.tool_formatter([BARE_ADD_TOOL])
        today = datetime.now().strftime("%d %b %Y")
        self.assertIn("Cutting Knowledge Date: December 2023\n", result)
        self.assertIn("Today Date: {}\n".format(today), result)
        # Llama3 pretty-prints the wrapped tool with indent=4.
        self.assertIn('"type": "function"', result)
        self.assertIn('{\n    "type": "function",', result)


class TestERNIEToolUtils(unittest.TestCase):
    def test_tool_formatter_wraps_in_tool_list(self):
        result = ERNIEToolUtils.tool_formatter([BARE_ADD_TOOL])
        expected = (
            "\n\n<tool_list>\n["
            '{"type": "function", "function": '
            '{"name": "add", "description": "Add numbers", '
            '"parameters": {"type": "object", '
            '"properties": {"a": {"type": "integer"}}}}}'
            "]\n</tool_list>"
        )
        self.assertEqual(result, expected)

    def test_function_formatter_keeps_trailing_newline_per_call(self):
        functions = [FunctionCall(name="add", arguments='{"a": 1}')]
        expected = '<tool_call>\n{"name": "add", "arguments": {"a": 1}}\n</tool_call>\n'
        self.assertEqual(ERNIEToolUtils.function_formatter(functions), expected)


class TestERNIEVLToolUtils(unittest.TestCase):
    def test_tool_formatter_wraps_in_tool_list_with_outer_newlines(self):
        result = ERNIEVLToolUtils.tool_formatter([BARE_ADD_TOOL])
        # ERNIE-VL differs from ERNIE only in the surrounding newlines: a single
        # leading "\n" and a trailing "\n" after </tool_list>.
        expected = (
            "\n<tool_list>\n["
            '{"type": "function", "function": '
            '{"name": "add", "description": "Add numbers", '
            '"parameters": {"type": "object", '
            '"properties": {"a": {"type": "integer"}}}}}'
            "]\n</tool_list>\n"
        )
        self.assertEqual(result, expected)

    def test_function_formatter_has_no_trailing_newline(self):
        # The one behavioral difference from ERNIEToolUtils.function_formatter:
        # no "\n" after the closing </tool_call>.
        functions = [FunctionCall(name="add", arguments='{"a": 1}')]
        expected = (
            '<tool_call>\n{"name": "add", "arguments": {"a": 1}}\n</tool_call>'
        )
        self.assertEqual(
            ERNIEVLToolUtils.function_formatter(functions), expected
        )
        self.assertNotEqual(
            ERNIEToolUtils.function_formatter(functions),
            ERNIEVLToolUtils.function_formatter(functions),
        )


class TestGetToolUtils(unittest.TestCase):
    def test_returns_instance_with_expected_behavior(self):
        # Verify the registry maps each key to a utility whose *behavior*
        # matches the corresponding class, not merely that something non-None
        # comes back.
        one = [FunctionCall(name="add", arguments='{"a": 1}')]

        self.assertEqual(
            get_tool_utils("default").function_formatter(one),
            'Action: add\nAction Input: {"a": 1}',
        )
        self.assertEqual(
            get_tool_utils("glm4").function_formatter(one), 'add\n{"a": 1}'
        )
        # glm4_moe keeps the newline-separated arg_key/arg_value form ...
        self.assertEqual(
            get_tool_utils("glm4_moe").function_formatter(one),
            "\n<tool_call>add\n<arg_key>a</arg_key>\n<arg_value>1</arg_value>",
        )
        # ... while glm_moe_dsa (GLM-5) drops the newlines and closes the tag.
        self.assertEqual(
            get_tool_utils("glm_moe_dsa").function_formatter(one),
            "<tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
        )
        # Both qwen and qwen3_5 alias the Qwen template.
        qwen_expected = (
            '<tool_call>\n{"name": "add", "arguments": {"a": 1}}\n</tool_call>'
        )
        self.assertEqual(
            get_tool_utils("qwen").function_formatter(one), qwen_expected
        )
        self.assertEqual(
            get_tool_utils("qwen3_5").function_formatter(one), qwen_expected
        )
        # ERNIE vs ERNIE-VL trailing-newline distinction survives the lookup.
        self.assertTrue(
            get_tool_utils("ernie")
            .function_formatter(one)
            .endswith("</tool_call>\n")
        )
        self.assertTrue(
            get_tool_utils("ernie_vl")
            .function_formatter(one)
            .endswith("</tool_call>")
        )
        self.assertEqual(
            get_tool_utils("llama3").function_formatter(one),
            '{"name": "add", "parameters": {"a": 1}}',
        )

    def test_unknown_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_tool_utils("nonexistent_tool")


if __name__ == "__main__":
    unittest.main()
