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

"""Behavior tests for ``paddlefleet.cli.hparams.model_args.ModelArguments``.

The only non-trivial logic in this module is ``ModelArguments.__post_init__``,
which *derives* the ``lora`` flag from the ``fine_tuning`` string:

    if self.fine_tuning.lower() == "lora": self.lora = True
    else:                                  self.lora = False

That derivation is the real consumer of the ``fine_tuning`` field, so the tests
drive it by actually constructing ``ModelArguments`` (which runs
``__post_init__``) rather than assigning a field and reading it straight back
(the "配置自赋值后读回" antipattern). The load-bearing checks are:

* the declared ``lora`` field default is ``False`` yet a default instance ends
  up ``True`` -- only possible if ``__post_init__`` really runs; and
* an *explicitly passed* ``lora`` value that conflicts with ``fine_tuning`` is
  discarded in favour of the derived value -- proving the assertion observes
  production derivation, not a value the test itself stored.

Expected values are hand-derived from the field declarations in
``model_args.py`` (e.g. ``VisionArguments.depth == 32``,
``ErniePretrainArgument.num_hidden_layers == 2``).

The whole ``paddlefleet`` package imports ``paddle`` at import time, so these
tests skip when Paddle (and therefore the package) is unavailable; they run for
real on any CPU where Paddle is installed.
"""

import dataclasses
import unittest

try:
    from paddlefleet.cli.hparams.model_args import (
        ErniePretrainArgument,
        ModelArguments,
        VisionArguments,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed
    ErniePretrainArgument = None
    ModelArguments = None
    VisionArguments = None
    _IMPORT_ERROR = exc


class ModelArgumentsPostInitLoraTest(unittest.TestCase):
    """__post_init__ derives ``lora`` solely from ``fine_tuning``."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def _declared_default(self, name):
        """Return the *declared* field default (pre ``__post_init__``)."""
        field = next(
            f for f in dataclasses.fields(ModelArguments) if f.name == name
        )
        return field.default

    def test_post_init_overrides_field_default_for_lora(self):
        """Default instance is lora=True even though the field default is False.

        ``fine_tuning`` defaults to "LoRA", so __post_init__ must flip the
        declared ``lora`` default (False) to True. If __post_init__ were removed
        the default instance would keep the declared False and this fails.
        """
        self.assertIs(self._declared_default("lora"), False)
        self.assertEqual(self._declared_default("fine_tuning"), "LoRA")
        args = ModelArguments()
        self.assertIs(args.lora, True)

    def test_full_fine_tuning_yields_lora_false(self):
        args = ModelArguments(fine_tuning="Full")
        self.assertIs(args.lora, False)

    def test_case_insensitive_lora_matching(self):
        """Matching is case-insensitive via ``fine_tuning.lower() == "lora"``."""
        for value in ("LoRA", "lora", "LORA", "lOrA"):
            with self.subTest(fine_tuning=value):
                self.assertIs(ModelArguments(fine_tuning=value).lora, True)
        for value in ("Full", "full", "FULL"):
            with self.subTest(fine_tuning=value):
                self.assertIs(ModelArguments(fine_tuning=value).lora, False)

    def test_non_lora_strings_map_to_false(self):
        """Any string other than a case-folded "lora" derives lora=False."""
        for value in ("DoRA", "sft", "", "lora "):
            with self.subTest(fine_tuning=value):
                self.assertIs(ModelArguments(fine_tuning=value).lora, False)

    def test_post_init_discards_conflicting_explicit_lora(self):
        """An explicit ``lora`` argument is overridden by the derived value.

        This is the decisive check that the test observes production derivation
        rather than a value it set itself: the explicit input is the opposite of
        the outcome in both directions.
        """
        # explicit True is discarded because fine_tuning != lora
        self.assertIs(ModelArguments(fine_tuning="Full", lora=True).lora, False)
        # explicit False is discarded because fine_tuning == lora
        self.assertIs(ModelArguments(fine_tuning="LoRA", lora=False).lora, True)


class ModelArgumentsNestedConfigTest(unittest.TestCase):
    """Nested configs come from ``default_factory`` -> fresh, independent objects."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def test_nested_configs_are_per_instance_not_shared(self):
        """Two instances get distinct nested objects; mutating one is isolated.

        A shared mutable default would make both instances alias the same
        VisionArguments / ErniePretrainArgument. The mutation-isolation checks
        below (hand-derived defaults: depth==32, num_hidden_layers==2) fail if
        the nested objects were shared.
        """
        a = ModelArguments()
        b = ModelArguments()

        self.assertIsInstance(a.vision_config, VisionArguments)
        self.assertIsInstance(a.ernie_model_config, ErniePretrainArgument)
        self.assertIsNot(a.vision_config, b.vision_config)
        self.assertIsNot(a.ernie_model_config, b.ernie_model_config)

        a.vision_config.depth = 999
        a.ernie_model_config.num_hidden_layers = 777
        # b keeps the declared defaults from model_args.py
        self.assertEqual(b.vision_config.depth, 32)
        self.assertEqual(b.ernie_model_config.num_hidden_layers, 2)


if __name__ == "__main__":
    unittest.main()
