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

"""GPTLMHead._stash_cu_seqlens_q: deliver cu_seqlens_q to the loss stage.

Under use_erndata=True and PP>1, the last pipeline stage runs the
LM head (which sees the pipeline dict) immediately before LanguageLoss (which
does not). GPTLMHead.forward must stash cu_seqlens_q onto
LanguageLoss._cu_seqlens_q_stash so the loss rank can roll labels per packed
document. These tests exercise that hand-off directly (no fleet init) via
GPTLMHead.__new__.
"""

from __future__ import annotations

import unittest

import paddle

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.lm_head import GPTLMHead


class TestLMHeadCuSeqlensStash(unittest.TestCase):
    def setUp(self) -> None:
        paddle.set_device("cpu")
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_stash_written_when_present(self) -> None:
        head = GPTLMHead.__new__(GPTLMHead)
        cu = paddle.to_tensor([0, 4, 10, 16], dtype="int32")
        head._stash_cu_seqlens_q({"hidden_states": None, "cu_seqlens_q": cu})
        self.assertIs(LanguageLoss._cu_seqlens_q_stash, cu)

    def test_noop_when_absent(self) -> None:
        head = GPTLMHead.__new__(GPTLMHead)
        # Non-megatron: cu_seqlens_q is stripped from the dict upstream.
        head._stash_cu_seqlens_q({"hidden_states": None})
        self.assertIsNone(LanguageLoss._cu_seqlens_q_stash)

    def test_noop_when_none(self) -> None:
        head = GPTLMHead.__new__(GPTLMHead)
        head._stash_cu_seqlens_q({"hidden_states": None, "cu_seqlens_q": None})
        self.assertIsNone(LanguageLoss._cu_seqlens_q_stash)


if __name__ == "__main__":
    unittest.main()
