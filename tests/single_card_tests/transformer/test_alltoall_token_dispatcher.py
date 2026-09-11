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
from unittest.mock import patch

import paddle

from paddlefleet.transformer.moe.token_dispatcher import AllToAllTokenDispatcher


class TestAllToAllTokenDispatcher(unittest.TestCase):
    @patch(
        "paddlefleet.transformer.moe.token_dispatcher._AllToAll.apply",
        new=lambda shape, value, **kwargs: value,
    )
    def test_weighted_roundtrip_and_backward(self):
        for accuracy_compatible in (False, True):
            with self.subTest(use_accuracy_compatible=accuracy_compatible):
                dispatcher = AllToAllTokenDispatcher(
                    moe_group=None,
                    expert_model_parallel_size=1,
                    num_experts_per_device=3,
                    local_expert_indices=[0, 1, 2],
                    use_accuracy_compatible=accuracy_compatible,
                )
                tokens = paddle.to_tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    stop_gradient=False,
                )
                probs = paddle.to_tensor(
                    [[0.25, 0.75, 0.0], [0.0, 0.5, 0.5], [0.75, 0.0, 0.25]]
                )
                routing = probs > 0
                # Emulate the collectives in this single-rank test.
                # Permutation, probability routing and autograd execute normally.
                with patch(
                    "paddlefleet.transformer.moe.token_dispatcher.AllGatherGroupOp.apply",
                    side_effect=lambda value, group=None: value,
                ):
                    dispatched = dispatcher.dispatch_preprocess(
                        tokens, probs, routing
                    )
                dispatched, _ = dispatcher.token_dispatch(dispatched)
                dispatched, counts = dispatcher.dispatch_postprocess(dispatched)
                self.assertEqual(counts.tolist(), [2, 2, 2])
                # Accuracy mode weights expert outputs before the combine;
                # the default path applies probabilities during unpermutation.
                if accuracy_compatible:
                    dispatched = (
                        dispatched * dispatcher.global_input_probs.unsqueeze(-1)
                    )
                combined = dispatcher.combine_preprocess(dispatched)
                combined = dispatcher.token_combine(combined)
                output = dispatcher.combine_postprocess(combined)
                self.assertTrue(paddle.equal_all(output, tokens).item())
                output.sum().backward()
                self.assertTrue(
                    paddle.equal_all(
                        tokens.grad, paddle.ones_like(tokens)
                    ).item()
                )


if __name__ == "__main__":
    unittest.main()
