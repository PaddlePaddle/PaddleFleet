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

"""Behavior tests for paddlefleet.training.initialize.

Scope: the CPU-observable pure logic.
  * set_logging: the args.logging_level -> logger level mapping (including the
    getattr default), and the colorlog handler/formatter it installs.
  * initialize_fleet: only the arg-selection branch (parse_args vs a supplied
    parsed_args) and the fixed fleet.init contract. The distributed numerics
    (fleet.init / hybrid communicate group / parallel-state init) need a real
    process group and are NOT exercised here; those external collaborators are
    replaced so the branch/forwarding logic can be observed on CPU. This file
    therefore does NOT claim to verify any distributed behavior.

set_logging mutates the shared "paddlefleet" logger (level + handlers); every
test snapshots that logger in setUp and fully restores it in tearDown so no
state leaks between tests or into concurrent siblings.
"""

import logging
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe: initialize -> paddle.distributed / parallel_state ->
# paddle. Only a genuine missing dependency (ImportError) may skip; any other
# error must surface as a real failure rather than a fake pass.
try:
    import paddlefleet.training.initialize as initmod

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    initmod = None
    _IMPORT_ERROR = exc

_LOGGER_NAME = "paddlefleet"

# Hand-copied from the set_logging body in the module under test, so that any
# change to the installed log format is caught here. Split across two literals;
# concatenation reproduces the production string exactly (single space between
# "[%(levelname)8s]" and "%(filename)s").
_EXPECTED_FORMAT = (
    "%(log_color)s[%(asctime)-15s] [%(levelname)8s] "
    "%(filename)s:%(lineno)d%(reset)s - %(message)s"
)

_SKIP_REASON = (
    f"paddlefleet.training.initialize not importable: {_IMPORT_ERROR}"
)


class _LoggerStateMixin:
    """Snapshot/restore the shared 'paddlefleet' logger around each test."""

    def setUp(self):
        self._logger = logging.getLogger(_LOGGER_NAME)
        self._orig_handlers = list(self._logger.handlers)
        self._orig_level = self._logger.level
        self._orig_propagate = self._logger.propagate

    def tearDown(self):
        # Fully restore pre-test logger state even if the test failed.
        self._logger.handlers = list(self._orig_handlers)
        self._logger.setLevel(self._orig_level)
        self._logger.propagate = self._orig_propagate

    def _added_handlers(self):
        return [
            h for h in self._logger.handlers if h not in self._orig_handlers
        ]


@unittest.skipUnless(initmod is not None, _SKIP_REASON)
class TestSetLoggingLevel(_LoggerStateMixin, unittest.TestCase):
    """set_logging propagates args.logging_level onto the logger."""

    def test_maps_each_explicit_level(self):
        # Distinct, non-degenerate levels with hand-derived integer values
        # (logging.DEBUG == 10, WARNING == 30, ERROR == 40). Before each case
        # the logger is forced to CRITICAL (50) so the observed value can only
        # come from set_logging, not from a leftover default.
        cases = [
            (logging.DEBUG, 10),
            (logging.WARNING, 30),
            (logging.ERROR, 40),
        ]
        for input_level, expected_int in cases:
            with self.subTest(level=expected_int):
                self._logger.setLevel(logging.CRITICAL)  # 50, differs from all
                initmod.set_logging(SimpleNamespace(logging_level=input_level))
                self.assertEqual(self._logger.level, expected_int)

    def test_defaults_to_info_when_attr_absent(self):
        # args has no logging_level -> getattr(..., logging.INFO) -> 20.
        self._logger.setLevel(logging.CRITICAL)  # 50, so INFO is attributable
        initmod.set_logging(SimpleNamespace())
        self.assertEqual(self._logger.level, 20)


@unittest.skipUnless(initmod is not None, _SKIP_REASON)
class TestSetLoggingHandler(_LoggerStateMixin, unittest.TestCase):
    """set_logging installs a colorlog handler with the expected format."""

    def test_adds_single_colored_stream_handler_with_expected_format(self):
        import colorlog

        initmod.set_logging(SimpleNamespace(logging_level=logging.INFO))

        added = self._added_handlers()
        self.assertEqual(len(added), 1)
        handler = added[0]
        # Genuine collaborators (real colorlog / logging), not mocked.
        self.assertIsInstance(handler, colorlog.StreamHandler)
        self.assertIsInstance(handler, logging.StreamHandler)

        formatter = handler.formatter
        self.assertIsInstance(formatter, colorlog.ColoredFormatter)
        # The exact configured format string is part of the contract.
        self.assertEqual(formatter._fmt, _EXPECTED_FORMAT)


@unittest.skipUnless(initmod is not None, _SKIP_REASON)
class TestInitializeFleetArgSelection(_LoggerStateMixin, unittest.TestCase):
    """initialize_fleet's parse_args-vs-parsed_args branch and fleet.init call.

    The distributed collaborators (fleet.init, hybrid communicate group,
    dist.get_rank/get_world_size, parallel_state init) are external to the
    module and cannot run on CPU, so they are replaced. set_global_variables is
    replaced with a recorder to observe which args object is forwarded. The real
    set_logging still runs, giving an independent CPU-observable signal that the
    selected args reached a downstream consumer. No distributed numerics are
    claimed.
    """

    def _patch_distributed(self, captured):
        fake_set_global = mock.patch.object(
            initmod,
            "set_global_variables",
            side_effect=lambda a: captured.append(a),
        )
        return [
            fake_set_global,
            mock.patch.object(initmod.fleet, "init"),
            mock.patch.object(
                initmod.fleet,
                "get_hybrid_communicate_group",
                return_value=object(),
            ),
            mock.patch.object(initmod.dist, "get_rank", return_value=0),
            mock.patch.object(initmod.dist, "get_world_size", return_value=1),
            mock.patch.object(initmod.ps, "initialize_model_parallel"),
        ]

    def test_uses_provided_parsed_args_and_skips_parse(self):
        captured = []
        strategy = object()
        args = SimpleNamespace(logging_level=logging.WARNING)  # 30

        set_global_p, init_p, hcg_p, rank_p, world_p, ps_p = (
            self._patch_distributed(captured)
        )
        with (
            mock.patch.object(initmod, "parse_args") as m_parse,
            set_global_p,
            init_p as m_init,
            hcg_p,
            rank_p,
            world_p,
            ps_p,
        ):
            initmod.initialize_fleet(strategy, parsed_args=args)
            # parse_args must be bypassed entirely.
            m_parse.assert_not_called()

        # The supplied object itself is forwarded to set_global_variables.
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0], args)
        # fleet.init receives the fixed is_collective flag and the strategy.
        m_init.assert_called_once_with(is_collective=True, strategy=strategy)
        # The real set_logging consumed the same args (WARNING -> 30).
        self.assertEqual(self._logger.level, 30)

    def test_parses_args_when_none_and_forwards_result(self):
        captured = []
        strategy = object()
        parsed = SimpleNamespace(logging_level=logging.ERROR)  # 40

        set_global_p, init_p, hcg_p, rank_p, world_p, ps_p = (
            self._patch_distributed(captured)
        )
        with (
            mock.patch.object(
                initmod, "parse_args", return_value=parsed
            ) as m_parse,
            set_global_p,
            init_p,
            hcg_p,
            rank_p,
            world_p,
            ps_p,
        ):
            initmod.initialize_fleet(strategy, some_flag=123)

        # parse_args is called once with the forwarded kwargs plus the fixed
        # ignore_unknown_args=True; no positional args.
        m_parse.assert_called_once()
        pos, kwargs = m_parse.call_args
        self.assertEqual(pos, ())
        self.assertEqual(
            kwargs, {"some_flag": 123, "ignore_unknown_args": True}
        )
        # The parse_args result flows to set_global_variables (identity)...
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0], parsed)
        # ...and to the real set_logging (ERROR -> 40).
        self.assertEqual(self._logger.level, 40)


if __name__ == "__main__":
    unittest.main()
