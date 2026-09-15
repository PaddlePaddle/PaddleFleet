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

"""Behavior tests for the ``separate_mtp_headloss`` guard in
``TransformerConfig.__post_init__``.

The guard normalizes a user-requested ``separate_mtp_headloss=True`` into the
actually-supported set of layouts. It force-disables the flag (and warns) when:

  1. MTP or pipeline parallel is not enabled
     (``num_nextn_predict_layers <= 0`` or
     ``pipeline_model_parallel_size <= 1``);
  2. once both are enabled, the seg-weight-bearing layer count
     ``num_hidden_layers + num_nextn_predict_layers + num_empty_layers`` is not
     exactly one layer per ``pp_degree * vpp_degree`` stage, where
     ``num_empty_layers = num_empty_layers_add_in_head +
     max(0, num_empty_layers_add_in_tail - 1)`` and ``vpp`` normalizes to 1 when
     unset; and/or
  3. fewer than 3 tail EmptyLayer slots are reserved
     (``num_empty_layers_add_in_tail < 3``).

Checks 2 and 3 are two independent ``if`` statements, so both warnings can fire
for the same config. Every expected value below is hand-derived from that
production logic, not read back from what the test set.

Constructing ``TransformerConfig`` requires paddle (the module does
``import paddle.nn.functional`` at import time). When paddle / paddlefleet are
not installed the import is guarded and the tests are skipped with an honest
reason rather than faking a pass.
"""

import os
import sys
import unittest
import warnings

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
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _HAVE_MODULE = True
except ImportError as exc:  # honest: real dependency missing, do not fake pass
    TransformerConfig = None
    _HAVE_MODULE = False
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.transformer.transformer_config not importable in this "
    f"environment: {_IMPORT_ERROR!r}"
)

# Substrings that uniquely identify each of the three force-disable warnings in
# the production source. Kept as constants so the negative assertions (a given
# warning must NOT fire) match the same text as the positive ones.
_PP_MTP_MSG = "both MTP and pipeline parallel"
_DIVISIBLE_MSG = "to be divisible by"
_TAIL_MSG = "num_empty_layers_add_in_tail >= 3"
_FORCING_MSG = "Forcing separate_mtp_headloss=False."


@unittest.skipUnless(_HAVE_MODULE, _SKIP_REASON)
class TestSeparateMtpHeadlossGuard(unittest.TestCase):
    """Force-disable behavior of the separate_mtp_headloss guard."""

    def _build(self, **overrides):
        """Construct a TransformerConfig that reaches the guard.

        Only the fields the guard reads are varied by the tests; the rest are
        fixed, valid values needed to build the config object.
        """
        kwargs = {
            "hidden_size": 64,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "separate_mtp_headloss": True,
            "num_nextn_predict_layers": 1,
            "pipeline_model_parallel_size": 4,
            "num_empty_layers_add_in_head": 0,
            "num_empty_layers_add_in_tail": 3,
        }
        kwargs.update(overrides)
        return TransformerConfig(**kwargs)

    def _build_capturing(self, **overrides):
        """Build a config and return (config, [warning message strings])."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._build(**overrides)
        return config, [str(w.message) for w in caught]

    def test_all_guards_pass_keeps_flag_enabled(self):
        # pp=4, vpp unset->1 ; num_empty = head 0 + max(0, tail 3 - 1) = 2 ;
        # total = num_hidden 1 + mtp 1 + 2 = 4 ; denom = 4*1 = 4 ;
        # 4 % 4 == 0 and 4 // 4 == 1 -> divisibility passes ;
        # tail 3 >= 3 -> tail check passes. Nothing force-disables it.
        config, messages = self._build_capturing()
        self.assertTrue(config.separate_mtp_headloss)
        # The flag surviving must be because the guard *ran and approved*, not
        # because it was skipped: assert no force-disable warning was emitted.
        self.assertFalse(
            any(_FORCING_MSG in m for m in messages),
            msg=f"unexpected force-disable warning(s): {messages}",
        )

    def test_pp_disabled_forces_disable(self):
        # pipeline_model_parallel_size = 1 -> pp_enabled False -> check 1 fires
        # and short-circuits into the force-disable-and-warn branch; the
        # divisibility / tail checks in the else-branch are never reached.
        config, messages = self._build_capturing(pipeline_model_parallel_size=1)
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(any(_PP_MTP_MSG in m for m in messages), msg=messages)
        self.assertFalse(any(_DIVISIBLE_MSG in m for m in messages))
        self.assertFalse(any(_TAIL_MSG in m for m in messages))

    def test_mtp_disabled_forces_disable(self):
        # num_nextn_predict_layers = 0 -> mtp_enabled False -> check 1 fires.
        # Same short-circuit: no divisibility / tail warning.
        config, messages = self._build_capturing(num_nextn_predict_layers=0)
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(any(_PP_MTP_MSG in m for m in messages), msg=messages)
        self.assertFalse(any(_DIVISIBLE_MSG in m for m in messages))
        self.assertFalse(any(_TAIL_MSG in m for m in messages))

    def test_indivisible_remainder_forces_disable(self):
        # num_hidden = 2 : num_empty = 0 + max(0, 3 - 1) = 2 ;
        # total = 2 + 1 + 2 = 5 ; denom = 4*1 = 4 ; 5 % 4 == 1 != 0 ->
        # divisibility fails via the remainder sub-condition. tail 3 >= 3 so the
        # tail check passes and only the divisibility warning fires.
        config, messages = self._build_capturing(num_hidden_layers=2)
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(
            any(_DIVISIBLE_MSG in m for m in messages), msg=messages
        )
        self.assertFalse(any(_TAIL_MSG in m for m in messages))
        self.assertFalse(any(_PP_MTP_MSG in m for m in messages))

    def test_divisible_but_quotient_not_one_forces_disable(self):
        # num_hidden = 5 : num_empty = 0 + max(0, 3 - 1) = 2 ;
        # total = 5 + 1 + 2 = 8 ; denom = 4*1 = 4 ; 8 % 4 == 0 (divisible) but
        # 8 // 4 == 2 != 1 -> divisibility fails via the quotient sub-condition,
        # exercising the "exactly one layer per stage" part specifically.
        # tail 3 >= 3 so only the divisibility warning fires.
        config, messages = self._build_capturing(num_hidden_layers=5)
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(
            any(_DIVISIBLE_MSG in m for m in messages), msg=messages
        )
        self.assertFalse(any(_TAIL_MSG in m for m in messages))
        self.assertFalse(any(_PP_MTP_MSG in m for m in messages))

    def test_insufficient_tail_forces_disable(self):
        # pp = 2, tail = 0 : num_empty = 0 + max(0, 0 - 1) = 0 ;
        # total = 1 + 1 + 0 = 2 ; denom = 2*1 = 2 ; 2 % 2 == 0 and 2 // 2 == 1
        # -> divisibility passes. tail 0 < 3 -> only the tail warning fires.
        config, messages = self._build_capturing(
            pipeline_model_parallel_size=2,
            num_empty_layers_add_in_tail=0,
        )
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(any(_TAIL_MSG in m for m in messages), msg=messages)
        self.assertFalse(any(_DIVISIBLE_MSG in m for m in messages))
        self.assertFalse(any(_PP_MTP_MSG in m for m in messages))

    def test_both_divisibility_and_tail_fail_emit_both_warnings(self):
        # pp = 4, tail = 0 : num_empty = 0 + max(0, 0 - 1) = 0 ;
        # total = 1 + 1 + 0 = 2 ; denom = 4*1 = 4 ; 2 % 4 == 2 != 0 ->
        # divisibility fails ; tail 0 < 3 -> tail check also fails. The two
        # checks are independent ``if`` statements, so both warnings fire.
        config, messages = self._build_capturing(num_empty_layers_add_in_tail=0)
        self.assertFalse(config.separate_mtp_headloss)
        self.assertTrue(
            any(_DIVISIBLE_MSG in m for m in messages), msg=messages
        )
        self.assertTrue(any(_TAIL_MSG in m for m in messages), msg=messages)
        self.assertFalse(any(_PP_MTP_MSG in m for m in messages))


if __name__ == "__main__":
    unittest.main()
