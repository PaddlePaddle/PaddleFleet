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
    ToolFormatter,
)
from paddlefleet.datasets.template.template import (
    TEMPLATES,
    GLM5ReasoningTemplate,
    ReasoningTemplate,
    Role,
    Template,
    get_template_and_fix_tokenizer,
    register_template,
)

# Production default thought markers (see register_template). Kept here as an
# independent copy so expected strings never read them from the template.
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"


def char_ids(text):
    """Independently encode a string the same way CharTokenizer does.

    Each character maps to its unicode code point, so an id sequence is a
    reproducible, content-distinguishable encoding of ``text``. Used to build
    expected values by hand without calling any template method.
    """
    return [ord(c) for c in text]


class CharTokenizer:
    """Deterministic stand-in for a real HuggingFace tokenizer.

    This is a genuine collaborator substitute, not the code under test: the
    Template assembly logic needs a tokenizer to turn slot strings and special
    tokens into ids. ``encode`` maps every character to its code point so the
    resulting ids can be reproduced independently via ``char_ids``. Special
    token ids are injected verbatim, which lets the tests observe exactly where
    bos / eos land in the prompt vs. response streams.
    """

    def __init__(
        self,
        bos_token_id=None,
        eos_token_id=None,
        eos_token="<eos>",
        chat_template=None,
        special=None,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.eos_token = eos_token
        self.chat_template = chat_template
        self._special = special or {}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return self._special[token]


# Real plugin objects built by the template module at import time. Reused so the
# hand-built templates carry genuine (mm / grounding) collaborators rather than
# ad-hoc doubles; the encode paths under test never invoke them.
_BASE_PLUGIN = TEMPLATES["empty"].mm_plugin
_GROUNDING_PLUGIN = TEMPLATES["empty"].grounding_plugin


def build_template(template_class=Template, **overrides):
    """Construct a real Template with content-distinguishable slots.

    Every formatter is a genuine production formatter. Distinct angle-bracket
    markers per role (``<sys>``/``<usr>``/``<ast>``/``<obs>``) make role turns,
    system/user/assistant assembly and chat separators individually observable
    in the emitted id stream.
    """
    kwargs = {
        "format_user": StringFormatter(slots=["<usr>{{content}}</usr>"]),
        "format_assistant": StringFormatter(
            slots=["<ast>{{content}}", {"eos_token"}]
        ),
        "format_system": StringFormatter(slots=["<sys>{{content}}</sys>"]),
        "format_function": FunctionFormatter(
            slots=["{{content}}"], tool_format="default"
        ),
        "format_observation": StringFormatter(slots=["<obs>{{content}}</obs>"]),
        "format_tools": ToolFormatter(tool_format="default"),
        "format_prefix": EmptyFormatter(slots=[{"bos_token"}]),
        "default_system": "SYS",
        "chat_sep": "<sep>",
        "suffix": [],
        "stop_words": [],
        "thought_words": (THINK_OPEN, THINK_CLOSE),
        "efficient_eos": True,
        "auto_add_bos": False,
        "enable_thinking": True,
        "mm_plugin": _BASE_PLUGIN,
        "grounding_plugin": _GROUNDING_PLUGIN,
    }
    kwargs.update(overrides)
    return template_class(**kwargs)


class TestRoleEnum(unittest.TestCase):
    """Role values feed the dict-key dispatch inside Template._encode."""

    def test_role_values_and_string_identity(self):
        self.assertEqual(Role.USER.value, "user")
        self.assertEqual(Role.ASSISTANT.value, "assistant")
        self.assertEqual(Role.SYSTEM.value, "system")
        self.assertEqual(Role.FUNCTION.value, "function")
        self.assertEqual(Role.OBSERVATION.value, "observation")
        self.assertIsInstance(Role.USER, str)
        self.assertEqual(Role.USER, "user")
        self.assertEqual(
            {r.value for r in Role},
            {"user", "assistant", "system", "function", "observation"},
        )


class TestTemplateEncoding(unittest.TestCase):
    """Prompt/response assembly, special-token placement and label masking."""

    def test_encode_oneturn_assembles_roles_and_masks_prompt_vs_response(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template()
        messages = [
            {"role": Role.USER, "content": "hi"},
            {"role": Role.ASSISTANT, "content": "world"},
        ]
        prompt_ids, response_ids = template.encode_oneturn(tok, messages)

        expected_prompt = [1, *char_ids("<sys>SYS</sys><usr>hi</usr>")]
        expected_response = [*char_ids("<ast>world"), 2]
        self.assertEqual(prompt_ids, expected_prompt)
        self.assertEqual(response_ids, expected_response)
        # Label-masking boundary: bos and system/user are prompt-only (masked);
        # eos and the assistant body are response-only (loss computed there).
        self.assertEqual(prompt_ids[0], 1)
        self.assertEqual(response_ids[-1], 2)
        self.assertNotIn(2, prompt_ids)
        self.assertNotIn(1, response_ids)

    def test_encode_oneturn_system_argument_overrides_default_system(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template(default_system="SYS")
        messages = [
            {"role": Role.USER, "content": "hi"},
            {"role": Role.ASSISTANT, "content": "world"},
        ]
        prompt_ids, _ = template.encode_oneturn(tok, messages, system="OVR")
        self.assertEqual(
            prompt_ids, [1, *char_ids("<sys>OVR</sys><usr>hi</usr>")]
        )
        # The default system text must not leak in when overridden.
        self.assertEqual(
            "".join(chr(c) for c in prompt_ids[1:]),
            "<sys>OVR</sys><usr>hi</usr>",
        )

    def test_encode_oneturn_omits_system_block_when_empty(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template(default_system="")
        messages = [
            {"role": Role.USER, "content": "hi"},
            {"role": Role.ASSISTANT, "content": "world"},
        ]
        prompt_ids, _ = template.encode_oneturn(tok, messages)
        # Empty system => no <sys> block, only prefix + user.
        self.assertEqual(prompt_ids, [1, *char_ids("<usr>hi</usr>")])

    def test_encode_multiturn_inserts_chat_sep_between_turns_only(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template(chat_sep="<sep>")
        messages = [
            {"role": Role.USER, "content": "a"},
            {"role": Role.ASSISTANT, "content": "b"},
            {"role": Role.USER, "content": "c"},
            {"role": Role.ASSISTANT, "content": "d"},
        ]
        pairs = template.encode_multiturn(tok, messages)
        self.assertEqual(len(pairs), 2)
        # First assistant turn is followed by <sep>; the last one is not.
        self.assertEqual(
            pairs[0][0], [1, *char_ids("<sys>SYS</sys><usr>a</usr>")]
        )
        self.assertEqual(
            pairs[0][1], [*char_ids("<ast>b"), 2, *char_ids("<sep>")]
        )
        self.assertEqual(pairs[1][0], char_ids("<usr>c</usr>"))
        self.assertEqual(pairs[1][1], [*char_ids("<ast>d"), 2])
        self.assertNotIn("<sep>", "".join(chr(c) for c in pairs[1][1]))

    def test_default_registered_template_encodes_expected_ids(self):
        # Exercises the actually shipped "default" template end to end.
        tok = CharTokenizer(eos_token_id=2)
        template = TEMPLATES["default"]
        messages = [
            {"role": Role.USER, "content": "hi"},
            {"role": Role.ASSISTANT, "content": "ok"},
        ]
        prompt_ids, response_ids = template.encode_oneturn(tok, messages)
        # default format_user = ["Human: {{content}}", {eos}, "\nAssistant:"]
        # default format_assistant = ["{{content}}", {eos}, "\n"]
        self.assertEqual(
            prompt_ids, [*char_ids("Human: hi"), 2, *char_ids("\nAssistant:")]
        )
        self.assertEqual(response_ids, [*char_ids("ok"), 2, *char_ids("\n")])


class TestThoughtWords(unittest.TestCase):
    """add_thought / remove_thought text transforms on the base Template."""

    def test_add_thought_wraps_content_with_both_markers(self):
        template = build_template()
        self.assertEqual(
            template.add_thought("answer"), THINK_OPEN + THINK_CLOSE + "answer"
        )
        self.assertEqual(template.add_thought(), THINK_OPEN + THINK_CLOSE)

    def test_remove_thought_strips_block_and_leading_newlines(self):
        template = build_template()
        content = THINK_OPEN + "reasoning here" + THINK_CLOSE + "answer"
        self.assertEqual(template.remove_thought(content), "answer")

    def test_remove_thought_inverts_empty_add_thought(self):
        template = build_template()
        wrapped = template.add_thought("final")
        self.assertEqual(template.remove_thought(wrapped), "final")


class TestReasoningTemplate(unittest.TestCase):
    """Empty-CoT injection differs by enable_thinking, which changes whether
    the thought tokens are masked (prompt) or supervised (response)."""

    def _messages(self):
        return [
            {"role": Role.USER, "content": "hi"},
            {"role": Role.ASSISTANT, "content": "world"},
        ]

    def test_empty_thought_goes_to_response_when_thinking_enabled(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template(ReasoningTemplate, enable_thinking=True)
        prompt_ids, response_ids = template.encode_oneturn(
            tok, self._messages()
        )

        thought_ids = char_ids(THINK_OPEN + THINK_CLOSE)
        base_response = [*char_ids("<ast>world"), 2]
        # Loss is computed on the CoT: it prefixes the response, not the prompt.
        self.assertEqual(response_ids, thought_ids + base_response)
        self.assertEqual(
            prompt_ids, [1, *char_ids("<sys>SYS</sys><usr>hi</usr>")]
        )

    def test_empty_thought_goes_to_prompt_when_thinking_disabled(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template(ReasoningTemplate, enable_thinking=False)
        prompt_ids, response_ids = template.encode_oneturn(
            tok, self._messages()
        )

        thought_ids = char_ids(THINK_OPEN + THINK_CLOSE)
        base_prompt = [1, *char_ids("<sys>SYS</sys><usr>hi</usr>")]
        base_response = [*char_ids("<ast>world"), 2]
        # No supervision on the CoT: it is appended to the (masked) prompt.
        self.assertEqual(prompt_ids, base_prompt + thought_ids)
        self.assertEqual(response_ids, base_response)

    def test_thinking_flag_flips_thought_between_prompt_and_response(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        on_prompt, on_resp = build_template(
            ReasoningTemplate, enable_thinking=True
        ).encode_oneturn(tok, self._messages())
        off_prompt, off_resp = build_template(
            ReasoningTemplate, enable_thinking=False
        ).encode_oneturn(tok, self._messages())
        # The two modes must produce genuinely different masking boundaries.
        self.assertNotEqual(on_prompt, off_prompt)
        self.assertNotEqual(on_resp, off_resp)
        self.assertGreater(len(off_prompt), len(on_prompt))
        self.assertGreater(len(on_resp), len(off_resp))

    def test_existing_thought_is_not_duplicated(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template(ReasoningTemplate, enable_thinking=True)
        content = THINK_OPEN + "kept" + THINK_CLOSE + "done"
        messages = [
            {"role": Role.USER, "content": "hi"},
            {"role": Role.ASSISTANT, "content": content},
        ]
        _, response_ids = template.encode_oneturn(tok, messages)
        # Content already carries a thought block, so no empty CoT is prepended.
        self.assertEqual(response_ids, [*char_ids("<ast>" + content), 2])


class TestGLM5ReasoningTemplate(unittest.TestCase):
    """GLM-5 emits only the closing marker for an empty CoT."""

    def test_add_thought_uses_closing_marker_only(self):
        template = build_template(GLM5ReasoningTemplate)
        self.assertEqual(template.add_thought("x"), THINK_CLOSE + "x")
        self.assertFalse(template.add_thought("x").startswith(THINK_OPEN))

    def test_empty_thought_in_response_uses_closing_marker_only(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2)
        template = build_template(GLM5ReasoningTemplate, enable_thinking=True)
        messages = [
            {"role": Role.USER, "content": "hi"},
            {"role": Role.ASSISTANT, "content": "world"},
        ]
        _, response_ids = template.encode_oneturn(tok, messages)
        close_ids = char_ids(THINK_CLOSE)
        base_response = [*char_ids("<ast>world"), 2]
        self.assertEqual(response_ids, close_ids + base_response)
        # Must NOT prepend the opening marker that the base template would use.
        self.assertNotEqual(
            response_ids, char_ids(THINK_OPEN + THINK_CLOSE) + base_response
        )


class TestRegisterTemplate(unittest.TestCase):
    """register_template wiring and duplicate protection."""

    def test_duplicate_name_raises(self):
        # "default" is registered at import time.
        with self.assertRaises(ValueError):
            register_template(name="default")

    def test_registers_template_with_working_formatters(self):
        name = "_indep_register_probe"
        self.assertNotIn(name, TEMPLATES)
        self.addCleanup(lambda: TEMPLATES.pop(name, None))
        register_template(
            name=name,
            format_user=StringFormatter(slots=["Q:{{content}}"]),
            format_assistant=StringFormatter(slots=["A:{{content}}"]),
        )
        tpl = TEMPLATES[name]
        self.assertIsInstance(tpl, Template)
        # The supplied formatters are stored and apply the real substitution.
        self.assertEqual(tpl.format_user.apply(content="hello"), ["Q:hello"])
        self.assertEqual(tpl.format_assistant.apply(content="hi"), ["A:hi"])
        # An unspecified formatter falls back to the passthrough default.
        self.assertEqual(tpl.format_system.apply(content="s"), ["s"])


class TestGetTemplateAndFixTokenizer(unittest.TestCase):
    """Config-driven template selection and field propagation."""

    def _register(self, name, **over):
        self.assertNotIn(name, TEMPLATES)
        self.addCleanup(lambda: TEMPLATES.pop(name, None))
        register_template(name=name, **over)
        return TEMPLATES[name]

    def test_unknown_template_name_raises(self):
        tok = CharTokenizer(eos_token_id=2)
        config = {
            "tokenizer": tok,
            "template": "_no_such_template_",
            "tool_format": None,
            "default_system": None,
        }
        with self.assertRaises(ValueError):
            get_template_and_fix_tokenizer(config)

    def test_returns_named_template_and_propagates_system_and_suffix(self):
        tpl = self._register("_indep_gtx_probe")
        tok = CharTokenizer(eos_token_id=2, eos_token="<eos>")
        config = {
            "tokenizer": tok,
            "template": "_indep_gtx_probe",
            "tool_format": None,
            "default_system": "Indep system",
        }
        result = get_template_and_fix_tokenizer(config)
        self.assertIs(result, tpl)
        # default_system from config is consumed onto the template.
        self.assertEqual(result.default_system, "Indep system")
        # Empty suffix is backfilled from the tokenizer eos token.
        self.assertEqual(result.suffix, ["<eos>"])

    def test_tool_format_replaces_function_and_tool_formatters(self):
        self._register("_indep_toolfmt_probe")
        tok = CharTokenizer(eos_token_id=2, eos_token="<eos>")
        config = {
            "tokenizer": tok,
            "template": "_indep_toolfmt_probe",
            "tool_format": "qwen",
            "default_system": None,
        }
        result = get_template_and_fix_tokenizer(config)
        self.assertIsInstance(result.format_function, FunctionFormatter)
        self.assertIsInstance(result.format_tools, ToolFormatter)
        self.assertEqual(result.format_function.tool_format, "qwen")
        self.assertEqual(result.format_tools.tool_format, "qwen")

    def test_none_template_without_chat_template_falls_back_to_empty(self):
        empty_tpl = TEMPLATES["empty"]
        # get_template_and_fix_tokenizer mutates suffix on the shared object;
        # snapshot and restore so the fallback template is not polluted.
        orig_suffix = list(empty_tpl.suffix)
        self.addCleanup(lambda: setattr(empty_tpl, "suffix", orig_suffix))
        tok = CharTokenizer(
            eos_token_id=2, eos_token="<eos>", chat_template=None
        )
        config = {
            "tokenizer": tok,
            "template": None,
            "tool_format": None,
            "default_system": None,
        }
        result = get_template_and_fix_tokenizer(config)
        self.assertIs(result, empty_tpl)


if __name__ == "__main__":
    unittest.main()
