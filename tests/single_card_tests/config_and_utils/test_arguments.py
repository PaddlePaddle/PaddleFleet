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

"""Behavior tests for paddlefleet.training.arguments.

These exercise the CPU-observable pure logic of ``parse_args`` (argparse
wiring: defaults, extra-arg providers, unknown-arg handling, abbreviation
disabling, and YAML-config dispatch) and ``core_transformer_config_from_args``
(dataclass-field copy / default / non-field filtering).

Every expected value below is hand-derived from the production source, not by
re-running the code under test.

The production module does ``from paddlefleet.transformer import
TransformerConfig`` at import time, so the whole module is unimportable when
paddle / paddlefleet are not installed. The import is guarded and the tests are
skipped with an honest reason in that case rather than faking a pass.
"""

import os
import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

# Allow ``import paddlefleet`` from the in-repo source tree when the package is
# not pip-installed. tests/single_card_tests/config_and_utils/ -> repo root.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_IMPORT_ERROR = None
try:
    from paddlefleet.training.arguments import (
        core_transformer_config_from_args,
        parse_args,
    )

    _HAVE_MODULE = True
except ImportError as exc:  # honest: real dependency missing, do not fake pass
    parse_args = None
    core_transformer_config_from_args = None
    _HAVE_MODULE = False
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.training.arguments not importable in this environment: "
    f"{_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAVE_MODULE, _SKIP_REASON)
class TestParseArgs(unittest.TestCase):
    """parse_args argparse wiring, derived from arguments.parse_args source."""

    def test_configs_defaults_to_none(self):
        # Source: parser.add_argument("--configs", type=str, default=None) and
        # no --configs on argv => args.configs is the default None, and because
        # it is None the yaml-load branch is skipped so the argparse namespace
        # itself is returned.
        with mock.patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        self.assertIsNone(args.configs)

    def test_extra_args_provider_parses_typed_value(self):
        # Source: parser = extra_args_provider(parser); the returned parser is
        # used for parsing. A --width int arg given "128" must be converted to
        # the int 128 (hand-derived, not read back from a prior set).
        def add_width(parser):
            parser.add_argument("--width", type=int, default=7)
            return parser

        with mock.patch.object(sys, "argv", ["prog", "--width", "128"]):
            args = parse_args(extra_args_provider=add_width)
        self.assertEqual(args.width, 128)
        self.assertIsInstance(args.width, int)
        self.assertIsNone(args.configs)

    def test_extra_args_provider_default_used_when_absent(self):
        # Source: when the provider-registered arg is not on argv, argparse
        # supplies its declared default. Hand-derived default is "fallback".
        def add_label(parser):
            parser.add_argument("--label", type=str, default="fallback")
            return parser

        with mock.patch.object(sys, "argv", ["prog"]):
            args = parse_args(extra_args_provider=add_label)
        self.assertEqual(args.label, "fallback")

    def test_ignore_unknown_keeps_known_and_drops_unknown(self):
        # Source: ignore_unknown_args=True selects parse_known_args, so unknown
        # flags land in the discarded leftover list (no SystemExit) while the
        # provider-registered known arg is still parsed. Hand-derived: --depth
        # "3" -> int 3; --mystery is unknown and must not appear on args.
        def add_depth(parser):
            parser.add_argument("--depth", type=int, default=1)
            return parser

        with mock.patch.object(
            sys, "argv", ["prog", "--mystery", "zzz", "--depth", "3"]
        ):
            args = parse_args(
                extra_args_provider=add_depth, ignore_unknown_args=True
            )
        self.assertEqual(args.depth, 3)
        self.assertIsNone(args.configs)
        self.assertFalse(hasattr(args, "mystery"))

    def test_strict_unknown_flag_raises_system_exit(self):
        # Source: ignore_unknown_args=False selects parser.parse_args(), which
        # errors on unrecognized arguments and calls sys.exit -> SystemExit.
        with mock.patch.object(sys, "argv", ["prog", "--mystery", "zzz"]):  # noqa: SIM117
            with self.assertRaises(SystemExit):
                parse_args(ignore_unknown_args=False)

    def test_abbreviation_disabled_config_not_expanded(self):
        # Source: ArgumentParser(..., allow_abbrev=False). With abbreviation
        # disabled, "--config foo" is NOT expanded to the "--configs" option.
        # Under ignore_unknown_args=True it is therefore treated as unknown and
        # dropped, so args.configs stays at its None default and the yaml-load
        # branch is skipped. (If allow_abbrev were True, --config would set
        # configs="foo" and trigger a yaml load instead.)
        with mock.patch.object(sys, "argv", ["prog", "--config", "foo"]):
            args = parse_args(ignore_unknown_args=True)
        self.assertIsNone(args.configs)

    def test_configs_set_dispatches_to_load_yaml_and_returns_result(self):
        # Source: when args.configs is not None, parse_args does
        # ``from .yaml_arguments import load_yaml`` then
        # ``args = load_yaml(args.configs)`` and returns that. load_yaml is a
        # genuine non-under-test collaborator; replace its module so we can
        # observe the exact path forwarded and that its return value replaces
        # the argparse namespace.
        sentinel = object()
        received = []

        def fake_load_yaml(path):
            received.append(path)
            return sentinel

        fake_module = types.ModuleType("paddlefleet.training.yaml_arguments")
        fake_module.load_yaml = fake_load_yaml

        with (
            mock.patch.dict(
                sys.modules,
                {"paddlefleet.training.yaml_arguments": fake_module},
            ),
            mock.patch.object(
                sys, "argv", ["prog", "--configs", "cfg/exp.yaml"]
            ),
        ):
            result = parse_args()

        self.assertEqual(received, ["cfg/exp.yaml"])
        self.assertIs(result, sentinel)


@dataclass
class _StubConfig:
    """Independent dataclass standing in for a config target.

    Declared defaults are the hand-derived reference for the "field missing on
    args" branch of core_transformer_config_from_args.
    """

    hidden: int = 1
    name: str = "base"
    ratio: float = 2.5


@unittest.skipUnless(_HAVE_MODULE, _SKIP_REASON)
class TestCoreTransformerConfigFromArgs(unittest.TestCase):
    """Field-copy semantics of core_transformer_config_from_args.

    Source logic: for each dataclasses.field(config_class), copy
    getattr(args, name) into kw_args only if hasattr(args, name); then
    config_class(**kw_args). So present fields override, missing fields fall
    back to the dataclass default, and args attributes that are not fields are
    never read.
    """

    def test_present_copied_missing_defaulted_extra_ignored(self):
        # args provides hidden and ratio (fields) plus an unrelated attribute
        # that is NOT a dataclass field; name is absent.
        # Hand-derived expected config: hidden=10 (copied), ratio=9.0 (copied),
        # name="base" (dataclass default, since args has no name). The
        # non-field "unrelated" must be filtered out entirely -- had it been
        # forwarded, _StubConfig(**kw_args) would raise TypeError.
        args = SimpleNamespace(hidden=10, ratio=9.0, unrelated=999)
        config = core_transformer_config_from_args(
            args, config_class=_StubConfig
        )
        self.assertEqual(config.hidden, 10)
        self.assertEqual(config.ratio, 9.0)
        self.assertEqual(config.name, "base")
        self.assertFalse(hasattr(config, "unrelated"))

    def test_all_fields_present_are_all_copied(self):
        # Every field supplied by args; none should fall back to a default.
        # Hand-derived: config mirrors args field-for-field.
        args = SimpleNamespace(hidden=64, name="probe", ratio=0.25)
        config = core_transformer_config_from_args(
            args, config_class=_StubConfig
        )
        self.assertEqual(config.hidden, 64)
        self.assertEqual(config.name, "probe")
        self.assertEqual(config.ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
