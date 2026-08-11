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
"""Tests for the MTP split guard in ``GPTMainLMHead.forward``.

``GPTMainLMHead`` used to split ``hidden_states`` into
``num_nextn_predict_layers + 1`` chunks unconditionally, which crashes when MTP
is off (split into 1 chunk is fine, but ``num_nextn_predict_layers`` may be
``None``) and silently drops 1/(n+1) of the sequence when MTP weights are only
loaded, not trained (``mtp_load_weight_only=True``). The split now happens only
when MTP is actually active.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import paddle

from paddlefleet.models.gpt.lm_head import GPTMainLMHead


def _make_head(**config_attrs):
    head = GPTMainLMHead.__new__(GPTMainLMHead)
    head.__dict__.setdefault("_parameters", {})
    head.__dict__.setdefault("_buffers", {})
    head.__dict__.setdefault("_sub_layers", {})
    head.__dict__.setdefault("_loaddict_holder", {})
    head.__dict__.setdefault("_non_persistable_buffers", set())
    defaults = {
        "num_nextn_predict_layers": None,
        "mtp_load_weight_only": False,
    }
    defaults.update(config_attrs)
    object.__setattr__(head, "config", MagicMock(**defaults))
    object.__setattr__(
        head, "_forward", MagicMock(side_effect=lambda x: x * 2.0)
    )
    return head


class TestGPTMainLMHeadMTPSplit(unittest.TestCase):
    def test_no_mtp_uses_full_hidden_states(self):
        head = _make_head(num_nextn_predict_layers=None)
        hidden_states = paddle.randn([4, 2, 8])

        ret = head.forward({"hidden_states": hidden_states})

        head._forward.assert_called_once()
        self.assertIs(head._forward.call_args[0][0], hidden_states)
        self.assertEqual(list(ret["logits"].shape), [4, 2, 8])
        # mtp_loss was None and must be filtered out
        self.assertNotIn("mtp_loss", ret)

    def test_zero_mtp_layers_uses_full_hidden_states(self):
        head = _make_head(num_nextn_predict_layers=0)
        hidden_states = paddle.randn([4, 2, 8])

        ret = head.forward({"hidden_states": hidden_states})

        self.assertEqual(list(ret["logits"].shape), [4, 2, 8])

    def test_mtp_load_weight_only_uses_full_hidden_states(self):
        """Weights-only MTP must not truncate the main branch."""
        head = _make_head(num_nextn_predict_layers=2, mtp_load_weight_only=True)
        hidden_states = paddle.randn([6, 2, 8])

        ret = head.forward({"hidden_states": hidden_states})

        self.assertEqual(list(ret["logits"].shape), [6, 2, 8])

    def test_active_mtp_keeps_only_the_main_chunk(self):
        head = _make_head(
            num_nextn_predict_layers=2, mtp_load_weight_only=False
        )
        # distinct per-chunk values so that picking the wrong (same-shaped)
        # chunk cannot pass: rows 0-1 = 1.0, rows 2-3 = 2.0, rows 4-5 = 3.0
        hidden_states = paddle.concat(
            [
                paddle.full([2, 2, 8], 1.0, dtype="float32"),
                paddle.full([2, 2, 8], 2.0, dtype="float32"),
                paddle.full([2, 2, 8], 3.0, dtype="float32"),
            ]
        )

        ret = head.forward({"hidden_states": hidden_states})

        # 6 rows == main + 2 MTP chunks -> logits come from the first 2 rows
        self.assertEqual(list(ret["logits"].shape), [2, 2, 8])
        np.testing.assert_allclose(
            ret["logits"].numpy(),
            (hidden_states[:2] * 2.0).numpy(),
            err_msg="logits must come from the main chunk, not an MTP chunk",
        )

    def test_mtp_loss_passthrough(self):
        """``mtp_loss`` is forwarded untouched when present."""
        head = _make_head()
        hidden_states = paddle.randn([4, 2, 8])
        mtp_loss = paddle.to_tensor(1.5)

        ret = head.forward(
            {"hidden_states": hidden_states, "mtp_loss": mtp_loss}
        )

        self.assertIs(head._forward.call_args[0][0], hidden_states)
        self.assertIs(ret["mtp_loss"], mtp_loss)


if __name__ == "__main__":
    unittest.main()
