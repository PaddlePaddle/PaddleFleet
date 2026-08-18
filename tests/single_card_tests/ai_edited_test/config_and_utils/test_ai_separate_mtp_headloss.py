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
import warnings


class TestSeparateMtpHeadlossValidation(unittest.TestCase):
    """Cover the separate_mtp_headloss validation in
    TransformerConfig.__post_init__ (lines 1491-1564)."""

    def _make_config_kwargs(self, **overrides):
        """Minimal kwargs to construct a TransformerConfig that reaches the
        separate_mtp_headloss validation branch."""
        defaults = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "separate_mtp_headloss": True,
            # MTP enabled via num_nextn_predict_layers (mtp_num_layers stays 0).
            "num_nextn_predict_layers": 1,
            "pipeline_model_parallel_size": 4,
            "num_empty_layers_add_in_head": 0,
            "num_empty_layers_add_in_tail": 3,
        }
        defaults.update(overrides)
        return defaults

    def _build(self, **overrides):
        from paddlefleet.transformer.transformer_config import TransformerConfig

        return TransformerConfig(**self._make_config_kwargs(**overrides))

    def test_valid_config_keeps_separate_mtp_headloss_true(self):
        """Both MTP and PP enabled, tail >= 3, and total layers split as
        exactly 1 layer per pp*vpp stage -> flag stays True.

        total_layers = num_hidden(1) + mtp(1) + num_empty(head 0 + tail-1=2) = 4
        denom = pp(4) * vpp(1) = 4 ; 4 % 4 == 0 and 4 // 4 == 1.
        """
        config = self._build()
        self.assertTrue(config.separate_mtp_headloss)

    def test_pp_disabled_forces_false(self):
        """Line 1507: pipeline parallel not enabled -> force-disable."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._build(pipeline_model_parallel_size=1)
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(
            any("both MTP and pipeline" in str(w.message) for w in caught)
        )

    def test_mtp_disabled_forces_false(self):
        """Line 1507: MTP not enabled (no mtp / nextn layers) -> force-disable."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._build(num_nextn_predict_layers=0, mtp_num_layers=0)
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(
            any("both MTP and pipeline" in str(w.message) for w in caught)
        )

    def test_indivisible_total_layers_forces_false(self):
        """Line 1541: total layers not evenly split as 1 layer per stage.

        num_hidden(2) + mtp(1) + num_empty(2) = 5, denom = 4 -> 5 % 4 != 0.
        tail >= 3 so only the divisibility warning fires.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._build(num_hidden_layers=2)
        self.assertFalse(config.separate_mtp_headloss)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("divisible" in m for m in messages))
        self.assertFalse(
            any("num_empty_layers_add_in_tail >= 3" in m for m in messages)
        )

    def test_insufficient_tail_empty_layers_forces_false(self):
        """Line 1557: tail empty layers < 3 -> force-disable.

        pp=2, num_hidden(1) + mtp(1) + num_empty(head 0 + tail 0) = 2,
        denom = 2 -> divisibility passes, only the tail warning fires.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._build(
                pipeline_model_parallel_size=2,
                num_empty_layers_add_in_tail=0,
            )
        self.assertFalse(config.separate_mtp_headloss)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("num_empty_layers_add_in_tail >= 3" in m for m in messages)
        )
        self.assertFalse(any("divisible" in m for m in messages))

    def test_both_checks_can_fire_independently(self):
        """The two checks are independent `if`s: when both divisibility and
        tail constraints fail, both warnings are emitted.

        pp=4, num_hidden(1) + mtp(1) + num_empty(0) = 2, denom = 4 ->
        indivisible; tail = 0 < 3.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._build(num_empty_layers_add_in_tail=0)
        self.assertFalse(config.separate_mtp_headloss)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("divisible" in m for m in messages))
        self.assertTrue(
            any("num_empty_layers_add_in_tail >= 3" in m for m in messages)
        )
