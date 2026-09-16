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

import inspect
import os
import sys
import unittest

import numpy as np
import paddle

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridAttention,
)


class _QKVProbe(paddle.nn.Layer):
    """A qkv segment whose ``get_query_key_value_tensors`` takes exactly the
    parameters ``DSv4HybridAttention`` declares -- no ``**kwargs`` to absorb an
    extra keyword. Both branches read ``hidden_states``, which is what makes the
    summation order into its gradient observable."""

    def __init__(self, width):
        super().__init__()
        self.q_weight = self.create_parameter([width, width], dtype="bfloat16")
        self.kv_weight = self.create_parameter([width, width], dtype="bfloat16")

    def get_query_key_value_tensors(
        self,
        hidden_states,
        startend_row_indices=None,
        position_offset=0,
        docmask_meta=None,
    ):
        del startend_row_indices, position_offset, docmask_meta
        query = paddle.matmul(hidden_states, self.q_weight)
        key = paddle.matmul(hidden_states, self.kv_weight).unsqueeze(2)
        q_compressed = hidden_states * 3
        return query, key, key, q_compressed, hidden_states

    def _qkv_forward(self, hidden_states, position_offset, docmask_meta):
        return DSv4HybridAttention._qkv_forward(
            self, hidden_states, position_offset, docmask_meta
        )


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "requires CUDA")
class TestDSv4FixedOrderQKV(unittest.TestCase):
    def _run_direct(self, module, hidden_states):
        query, key, q_compressed = module._qkv_forward(hidden_states, 0, None)
        loss = query.sum() + 2 * key.sum() + 3 * q_compressed.sum()
        loss.backward()
        return (
            tuple(
                output.numpy().copy() for output in (query, key, q_compressed)
            ),
            hidden_states.grad.numpy().copy(),
            tuple(
                parameter.grad.numpy().copy()
                for parameter in module.parameters()
            ),
        )

    def _run_recompute(self, module, hidden_states):
        span = RecomputeWithoutOutput()
        query, key, q_compressed = span.recompute(
            module._qkv_forward,
            hidden_states,
            0,
            None,
            preserve_rng_state=False,
            share_grad_holder=True,
        )
        loss = query.sum() + 2 * key.sum() + 3 * q_compressed.sum()
        span.discard_output_and_register_recompute(loss)
        loss.backward()
        return (
            tuple(
                output.numpy().copy() for output in (query, key, q_compressed)
            ),
            hidden_states.grad.numpy().copy(),
            tuple(
                parameter.grad.numpy().copy()
                for parameter in module.parameters()
            ),
        )

    def test_recompute_replay_is_bitwise_identical(self):
        paddle.seed(2026)
        module = _QKVProbe(32)
        hidden_states = paddle.randn([3, 2, 32]).astype("bfloat16")
        hidden_states.stop_gradient = False

        direct = self._run_direct(module, hidden_states)
        for parameter in module.parameters():
            parameter.clear_gradient()
        hidden_states.clear_gradient()

        recomputed = self._run_recompute(module, hidden_states)
        for direct_values, recomputed_values in zip(direct, recomputed):
            if isinstance(direct_values, tuple):
                for direct_value, recomputed_value in zip(
                    direct_values, recomputed_values
                ):
                    np.testing.assert_array_equal(
                        direct_value, recomputed_value
                    )
            else:
                np.testing.assert_array_equal(direct_values, recomputed_values)

    def test_segment_passes_only_the_declared_parameters(self):
        """The segment must call the override through the base contract.

        A subclass implementing exactly ``DSv4HybridAttention``'s signature has
        no ``**kwargs`` to swallow an extra keyword, so an added one is a
        TypeError -- on every differentiable call, since ``_qkv_forward``
        branches on ``stop_gradient`` rather than on the recompute switch.
        """
        declared = set(
            inspect.signature(
                DSv4HybridAttention.get_query_key_value_tensors
            ).parameters
        )
        probed = set(
            inspect.signature(_QKVProbe.get_query_key_value_tensors).parameters
        )
        self.assertEqual(probed, declared)

        seen = []
        module = _QKVProbe(32)
        original = module.get_query_key_value_tensors

        def _recording(*args, **kwargs):
            seen.append(set(kwargs))
            return original(*args, **kwargs)

        module.get_query_key_value_tensors = _recording

        hidden_states = paddle.randn([3, 2, 32]).astype("bfloat16")
        hidden_states.stop_gradient = False
        module._qkv_forward(hidden_states, 0, None)

        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0], declared - {"self", "hidden_states"})


if __name__ == "__main__":
    unittest.main()
