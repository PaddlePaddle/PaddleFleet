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

"""Behavior tests for ernie_pretrain logging utilities.

Module under test:
    paddlefleet.cli.train.ernie_pretrain.src.utils.logging

Oracle policy
-------------
The module has two observable behaviors, both verified here against
hand-derived expectations:

* Import-time configuration of the ``logging`` registry: the root logger is
  set to DEBUG with a single stderr ``StreamHandler`` whose formatter renders
  a specific layout; the ``baidubce`` logger is silenced (no handlers, no
  propagation); ``bce_bns_proxy.wrapper`` and ``filelock`` are disabled; and
  the shared ``PaddleFleet`` logger is stripped of handlers and set to
  propagate.  We assert each of these directly and render a constructed
  ``LogRecord`` through the real formatter to check the exact layout (the
  time field is variable, so only the fixed prefix/suffix are pinned).

* ``setup_logger_output_file(outputpath, local_rank)`` creates ``<out>/log``
  and attaches a per-rank ``FileHandler`` (append mode) to the root logger
  alongside the existing stream handler, with a formatter that embeds the
  rank tag.  We verify the directory/file, the exact handler list and file
  path, the append mode, the rendered layout of a constructed record, and
  end-to-end that a message logged through the configured root logger is
  actually written to the rank file.

Both the formatter layouts and the padding used in the expected strings are
derived by hand from the logging format contract, not copied from the module.

Paddle skip
-----------
The module itself only needs the stdlib ``logging`` package, but importing it
drags in ``paddlefleet/__init__.py`` -> ``parallel_state``, which imports
``paddle`` at top level.  Paddle is not installed in this environment, so the
import raises ``ImportError`` and every test skips with that honest reason.
"""

import logging
import os
import tempfile
import unittest

try:
    import paddle  # heavy dep pulled in by paddlefleet/__init__

    from paddlefleet.cli.train.ernie_pretrain.src.utils import (
        logging as ernie_logging,
    )
    from paddlefleet.cli.train.ernie_pretrain.src.utils.logging import (
        setup_logger_output_file,
    )
    from paddlefleet.utils.log import logger as paddlenlp_logger

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the module) unavailable in this env
    paddle = None
    ernie_logging = None
    setup_logger_output_file = None
    paddlenlp_logger = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "importing paddlefleet.cli.train.ernie_pretrain.src.utils.logging requires "
    "paddle (pulled in by paddlefleet/__init__ -> parallel_state), which is not "
    f"installed here: {_IMPORT_ERROR!r}"
)


def _make_record(level, filename, lineno, msg, args):
    """Build a LogRecord with fully controlled filename/lineno/message.

    ``LogRecord`` derives ``filename`` from ``basename(pathname)`` and renders
    the message via ``msg % args``; both are pinned so the formatted output is
    deterministic apart from the timestamp field.
    """
    return logging.LogRecord(
        name="probe",
        level=level,
        pathname="/some/dir/" + filename,
        lineno=lineno,
        msg=msg,
        args=args,
        exc_info=None,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class ImportSideEffectTest(unittest.TestCase):
    """Configuration applied to the logging registry when the module loads."""

    def test_root_logger_is_debug_with_single_stream_handler(self):
        root = ernie_logging.logger
        # ``logging.getLogger()`` with no name is the root logger.
        self.assertIs(root, logging.getLogger())
        self.assertEqual(root.level, 10)  # 10 == logging.DEBUG
        self.assertEqual(logging.getLevelName(10), "DEBUG")
        # The module installs exactly its own stderr handler at import time.
        self.assertIn(ernie_logging.hdl, root.handlers)
        self.assertIsInstance(ernie_logging.hdl, logging.StreamHandler)
        # StreamHandler with no explicit stream targets stderr.
        import sys

        self.assertIs(ernie_logging.hdl.stream, sys.stderr)

    def test_baidubce_logger_is_silenced(self):
        bce = logging.getLogger("baidubce")
        self.assertEqual(bce.handlers, [])
        self.assertFalse(bce.propagate)

    def test_bns_proxy_and_filelock_loggers_disabled(self):
        self.assertTrue(logging.getLogger("bce_bns_proxy.wrapper").disabled)
        self.assertTrue(logging.getLogger("filelock").disabled)
        # A logger the module never touches must stay enabled: this rules out
        # a blanket "disable everything" implementation passing the two above.
        self.assertFalse(
            logging.getLogger("some.untouched.logger.name").disabled
        )

    def test_paddlefleet_logger_reconfigured_to_propagate(self):
        # The module strips the shared PaddleFleet logger's own handlers and
        # turns propagation back on so its records reach the root handlers.
        self.assertEqual(paddlenlp_logger.logger.handlers, [])
        self.assertTrue(paddlenlp_logger.logger.propagate)

    def test_stream_formatter_layout(self):
        # Format contract:
        #   "[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:    %(message)s"
        # filename "probe.py" (8 chars) right-justified to width 12 -> 4 spaces;
        # lineno 42 right-justified to width 5 -> 3 spaces; message "hi x".
        record = _make_record(logging.WARNING, "probe.py", 42, "hi %s", ("x",))
        rendered = ernie_logging.formatter.format(record)
        self.assertTrue(rendered.startswith("[WARNING] "))
        self.assertTrue(
            rendered.endswith(" [    probe.py:   42]:    hi x"),
            rendered,
        )


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class SetupLoggerOutputFileTest(unittest.TestCase):
    """setup_logger_output_file wiring, per-rank file path and content."""

    def setUp(self):
        # setup_logger_output_file mutates global state: it replaces the root
        # logger's handler list and rebinds the module stream handler's
        # formatter. Snapshot and restore both, closing any file handler we
        # open, so tests don't pollute each other or the process.
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        self._saved_hdl_formatter = ernie_logging.hdl.formatter
        self.addCleanup(self._restore, root)

    def _restore(self, root):
        for h in root.handlers:
            if (
                isinstance(h, logging.FileHandler)
                and h not in self._saved_handlers
            ):
                h.close()
        root.handlers = self._saved_handlers
        ernie_logging.hdl.setFormatter(self._saved_hdl_formatter)

    def test_creates_log_dir_file_and_wires_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_logger_output_file(tmp, 3)
            root = logging.getLogger()

            logdir = os.path.join(tmp, "log")
            logfile = os.path.join(logdir, "workerlog.3")
            self.assertTrue(os.path.isdir(logdir))
            self.assertTrue(os.path.isfile(logfile))

            # Root logger now carries exactly the stream handler followed by
            # the new file handler (order matters: it is set as [hdl, file]).
            self.assertEqual(len(root.handlers), 2)
            self.assertIs(root.handlers[0], ernie_logging.hdl)
            file_hdl = root.handlers[1]
            self.assertIsInstance(file_hdl, logging.FileHandler)
            self.assertEqual(file_hdl.baseFilename, os.path.abspath(logfile))
            # Append mode preserves earlier ranks' logs across restarts.
            self.assertEqual(file_hdl.mode, "a")

    def test_rank_formatter_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_logger_output_file(tmp, 7)
            file_hdl = logging.getLogger().handlers[1]
            # Rank format contract embeds "[rank-7]" before the "]:    " gap.
            # filename "probe.py" -> 4 leading spaces; lineno 5 -> 4 spaces.
            record = _make_record(
                logging.INFO, "probe.py", 5, "payload-%d", (99,)
            )
            rendered = file_hdl.format(record)
            self.assertTrue(rendered.startswith("[INFO] "))
            self.assertTrue(
                rendered.endswith(
                    " [    probe.py:    5][rank-7]:    payload-99"
                ),
                rendered,
            )

    def test_message_reaches_rank_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_logger_output_file(tmp, 2)
            marker = "ernie-logging-endtoend-marker-4242"
            # Log through the module's configured root logger (DEBUG level, so
            # INFO passes) and confirm it is actually written to the rank file.
            ernie_logging.logger.info(marker)
            logging.getLogger().handlers[1].flush()

            logfile = os.path.join(tmp, "log", "workerlog.2")
            with open(logfile, encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn(marker, content)
            self.assertIn("[rank-2]", content)
            self.assertIn("[INFO]", content)

    def test_distinct_file_per_rank_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_logger_output_file(tmp, 0)
            setup_logger_output_file(tmp, 1)
            # Re-invoking with an existing "log" dir must not raise (exist_ok).
            setup_logger_output_file(tmp, 1)

            logdir = os.path.join(tmp, "log")
            self.assertTrue(os.path.isfile(os.path.join(logdir, "workerlog.0")))
            self.assertTrue(os.path.isfile(os.path.join(logdir, "workerlog.1")))
            self.assertFalse(
                os.path.exists(os.path.join(logdir, "workerlog.2"))
            )
            # The live handler points at the most recent rank's file.
            self.assertEqual(
                logging.getLogger().handlers[1].baseFilename,
                os.path.abspath(os.path.join(logdir, "workerlog.1")),
            )


if __name__ == "__main__":
    unittest.main()
