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

"""Behavior tests for ``paddlefleet.cli.train.sft.sft_config.SFTConfig``.

``SFTConfig`` is a ``@dataclass`` subclass of ``TrainingArguments``. Its only
self-contained logic lives in ``__post_init__``: after delegating to
``super().__post_init__()`` it, *and only when* ``autotuner_benchmark`` is
truthy, forcibly overrides a fixed set of training flags (``max_steps=5``,
``do_train=True``, ``do_export/do_predict/do_eval=False``,
``overwrite_output_dir=True``, ``load_best_model_at_end=False``,
``report_to=[]`` and both ``save_strategy``/``evaluation_strategy`` set to
``IntervalStrategy.NO``).

Rather than assign a field and read it straight back (the "配置自赋值后读回"
antipattern), these tests drive the object through that real ``__post_init__``
and check the *derived* behavior:

* In benchmark mode the overrides win even against a conflicting user value
  (``max_steps`` is passed as 7 yet must come out as the hand-derived 5 that
  the user never set), and the fields the discarded coverage test ignored
  (``report_to``, ``save_strategy``, ``evaluation_strategy``) are asserted.
* With ``autotuner_benchmark`` off, the guard branch must NOT run, so a
  user-supplied ``max_steps`` survives unchanged. This pair pins the branch:
  deleting the ``if self.autotuner_benchmark`` guard breaks one test, and
  deleting the override body breaks the other.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so tests
skip with an honest reason when Paddle (and therefore the package) is
unavailable; they run for real on any CPU where Paddle is installed.
"""

import os
import unittest

try:
    from paddlefleet.cli.train.sft.sft_config import SFTConfig
    from paddlefleet.trainer.trainer_utils import IntervalStrategy

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed
    SFTConfig = None
    IntervalStrategy = None
    _IMPORT_ERROR = exc


class SFTConfigPostInitTest(unittest.TestCase):
    """Drive SFTConfig through its real __post_init__ override logic."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")
        # SFTConfig.__post_init__ chains into TrainingArguments.__post_init__,
        # which on a CUDA-compiled Paddle (a) probes
        # paddle_device.get_device_capability() on the current place
        # (training_args.py:2071 -- a CPU place raises ValueError) and (b)
        # drives initialize_fleet() -> fleet.init() -> ParallelEnv(), reading
        # int(FLAGS_selected_gpus[0]) (training_args.py:3201). A launcher/fleet
        # normally selects a GPU place and exports FLAGS_selected_gpus; the CI
        # runner leaves the latter as an empty string. Replicate the
        # launcher-provided environment here. These are GPU-only paths, so skip
        # honestly on a CPU-only build.
        import paddle

        if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() == 0
        ):
            self.skipTest(
                "SFTConfig construction probes GPU device capability and "
                "initializes a single-card fleet; this build has no usable "
                "CUDA device"
            )
        self._orig_device = paddle.get_device()
        self._orig_selected_gpus = os.environ.get("FLAGS_selected_gpus")
        os.environ["FLAGS_selected_gpus"] = "0"
        paddle.set_device("gpu:0")
        self.addCleanup(self._restore_gpu_env)

    def _restore_gpu_env(self):
        import paddle

        paddle.set_device(self._orig_device)
        if self._orig_selected_gpus is None:
            os.environ.pop("FLAGS_selected_gpus", None)
        else:
            os.environ["FLAGS_selected_gpus"] = self._orig_selected_gpus

    def test_benchmark_mode_overrides_conflicting_user_values(self):
        """autotuner_benchmark=True must force the benchmark flag set.

        ``max_steps`` is deliberately passed as 7. __post_init__ overwrites it
        with 5 (a value the user never supplied), so observing 5 proves the
        override body actually executed rather than echoing an input.
        """
        config = SFTConfig(
            output_dir="/tmp/test_sft_output",
            autotuner_benchmark=True,
            max_steps=7,
            bf16=True,
        )

        # max_steps==5 is produced solely by the override, not by the caller.
        self.assertEqual(config.max_steps, 5)
        self.assertTrue(config.do_train)
        self.assertFalse(config.do_export)
        self.assertFalse(config.do_predict)
        self.assertFalse(config.do_eval)
        self.assertTrue(config.overwrite_output_dir)
        self.assertFalse(config.load_best_model_at_end)
        # Fields the discarded coverage test never checked:
        self.assertEqual(config.report_to, [])
        self.assertEqual(config.save_strategy, IntervalStrategy.NO)
        self.assertEqual(config.evaluation_strategy, IntervalStrategy.NO)

    def test_non_benchmark_mode_leaves_user_value_untouched(self):
        """With the default (False) guard, the override branch must not run.

        A caller-supplied max_steps=7 must survive; if it silently became 5 the
        benchmark override would be leaking into the normal path.
        """
        config = SFTConfig(
            output_dir="/tmp/test_sft_output",
            autotuner_benchmark=False,
            max_steps=7,
            bf16=True,
        )

        self.assertFalse(config.autotuner_benchmark)
        self.assertEqual(config.max_steps, 7)


if __name__ == "__main__":
    unittest.main()
