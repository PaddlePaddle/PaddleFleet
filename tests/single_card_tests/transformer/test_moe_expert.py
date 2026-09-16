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
"""Behavior tests for paddlefleet.transformer.moe.moe_expert.

Scope is the CPU-constructible / CPU-executable surface of the grouped-GEMM
MoE expert, plus the GPU-only batched-GEMM PyLayer that backs it:

  * ``GroupedMLPExpert.__init__`` parameter geometry (GLU doubling of the fc1
    output width, per-partition intermediate override, bfloat16 dtype), the
    ``use_bias`` guard, activation-function selection (SwiGLU closure vs the
    raw activation vs the unsupported-activation reject), and the
    ``moe_act`` + fp8 recompute reject in ``update_activation_recompute``.
  * The SwiGLU activation closure's math, checked against an independent numpy
    ``silu(gate) * up`` reference on distinguishable, non-square input.
  * Process-group wiring (``pg_collection.ep`` -> ``self.ep_group``) and the
    ``intermediate_ep_sharded`` AND-guard against ``expert_parallel``.
  * ``sharded_state_dict`` (ep_group is None branch): the fused 3-D weights are
    flattened to 2-D with names preserved before being handed to the DCP
    builder, verified by spying on the builder collaborator.
  * ``BMMFunction`` forward and backward against an independent per-group numpy
    matmul reference with uneven group sizes (CUDA only).

Every numeric expectation is derived from an independent numpy computation,
never by calling the production path under test. Heavy imports (paddle +
paddlefleet + paddlefleet_ops) are guarded; only ImportError /
ModuleNotFoundError is treated as "runtime absent" so genuine API breaks still
surface instead of being silently skipped.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import numpy as np
    import paddle
    import paddle.nn.functional as F

    from paddlefleet.transformer.moe import moe_expert as moe_expert_mod
    from paddlefleet.transformer.moe.moe_expert import (
        BMMFunction,
        GroupedMLPExpert,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle/paddlefleet/paddlefleet_ops/numpy not importable in this "
    f"environment: {_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)
_HAS_CUDA = bool(_HAS_DEPS and paddle.is_compiled_with_cuda())


def _make_config(**overrides):
    """Build a small CPU-friendly TransformerConfig for the grouped MoE expert.

    Defaults mirror a SwiGLU MoE expert (gated_linear_unit + silu). Individual
    tests override only the knob they exercise.
    """
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "moe_intermediate_size": 128,
        "use_bias": False,
        "gated_linear_unit": True,
        "hidden_act": F.silu,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _numpy_silu(x):
    """Independent SiLU: x * sigmoid(x). Not the production activation."""
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def _grouped_matmul(x, y, batch_sizes, trans_y=False):
    """Independent per-group matmul reference for batched_gemm.

    ``x`` is [total_tokens, K]; group g owns ``batch_sizes[g]`` consecutive
    rows and multiplies them by ``y[g]`` ([K, N], or [N, K] when trans_y).
    Written as an explicit per-group loop so it shares no code with the
    production PyLayer.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    outs = []
    start = 0
    for g, n in enumerate(batch_sizes):
        block = x[start : start + n]
        mat = y[g].T if trans_y else y[g]
        outs.append(block @ mat)
        start += n
    return np.concatenate(outs, axis=0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGroupedMLPExpertConstruction(unittest.TestCase):
    """Parameter geometry and constructor guards (CPU-constructible)."""

    def test_glu_doubles_fc1_output_width(self):
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=3, config=config, moe_deep_gemm=False
        )
        # weight1 projects hidden -> 2*I because SwiGLU needs gate and up.
        self.assertEqual(expert.weight1.shape, [3, 64, 2 * 128])
        # weight2 projects the (post-GLU) intermediate I back to hidden.
        self.assertEqual(expert.weight2.shape, [3, 128, 64])
        self.assertEqual(expert.weight1.dtype, paddle.bfloat16)
        self.assertEqual(expert.weight2.dtype, paddle.bfloat16)

    def test_non_glu_keeps_single_fc1_width(self):
        config = _make_config(gated_linear_unit=False)
        expert = GroupedMLPExpert(
            num_local_experts=2, config=config, moe_deep_gemm=False
        )
        # No GLU -> fc1 output width is exactly the intermediate size.
        self.assertEqual(expert.weight1.shape, [2, 64, 128])
        self.assertEqual(expert.weight2.shape, [2, 128, 64])

    def test_intermediate_size_per_partition_override(self):
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            intermediate_size_per_partition=32,
        )
        # The local shard replaces moe_intermediate_size in both weights;
        # weight1 still doubles for GLU, weight2 uses the shard directly.
        self.assertEqual(expert.intermediate_size_per_partition, 32)
        self.assertEqual(expert.weight1.shape, [2, 64, 64])
        self.assertEqual(expert.weight2.shape, [2, 32, 64])

    def test_use_bias_is_rejected(self):
        config = _make_config(use_bias=True)
        with self.assertRaises(AssertionError):
            GroupedMLPExpert(
                num_local_experts=2, config=config, moe_deep_gemm=False
            )

    def test_unsupported_activation_with_glu_is_rejected(self):
        # relu is neither silu, gelu, nor situ -> GroupedMLP refuses it.
        config = _make_config(hidden_act=F.relu)
        with self.assertRaises(ValueError):
            GroupedMLPExpert(
                num_local_experts=2, config=config, moe_deep_gemm=False
            )

    def test_moe_act_recompute_with_fp8_is_rejected(self):
        # moe_act selective recompute has no fp8 path on the legacy GroupedMLP.
        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["moe_act"],
            fp8=True,
        )
        with self.assertRaises(ValueError):
            GroupedMLPExpert(
                num_local_experts=2, config=config, moe_deep_gemm=False
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGroupedMLPExpertActivation(unittest.TestCase):
    """The activation closure selected in __init__ is exercised for real."""

    def test_swiglu_closure_matches_independent_reference(self):
        config = _make_config()  # gated_linear_unit + silu
        expert = GroupedMLPExpert(
            num_local_experts=1, config=config, moe_deep_gemm=False
        )
        # Distinguishable, non-square input: last dim 6 -> gate/up halves of 3,
        # so a wrong chunk axis or a gate/up swap changes the result.
        raw = np.array(
            [
                [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0],
                [0.1, -0.3, 4.0, -4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        x = paddle.to_tensor(raw, dtype="float32")
        out = expert.activation_func(x).numpy().astype(np.float64)

        gate = raw[:, :3].astype(np.float64)
        up = raw[:, 3:].astype(np.float64)
        expected = _numpy_silu(gate) * up
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_non_glu_uses_config_activation_directly(self):
        config = _make_config(gated_linear_unit=False, hidden_act=F.silu)
        expert = GroupedMLPExpert(
            num_local_experts=1, config=config, moe_deep_gemm=False
        )
        # Without GLU the expert applies the configured activation unchanged;
        # it must be the very same callable, not a wrapping closure.
        self.assertIs(expert.activation_func, F.silu)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGroupedMLPExpertProcessGroupWiring(unittest.TestCase):
    """pg_collection wiring and the intermediate-EP-sharded guard."""

    def test_ep_group_taken_from_pg_collection(self):
        config = _make_config()
        sentinel_ep = object()
        pg = mock.MagicMock()
        pg.ep = sentinel_ep
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=pg,
        )
        # The constructor must store pg_collection.ep verbatim as ep_group.
        self.assertIs(expert.ep_group, sentinel_ep)
        # Distributed is not initialised in this single process, so get_pg_size
        # returns 1 and expert parallelism stays off regardless of the group.
        self.assertFalse(expert.expert_parallel)
        self.assertFalse(expert.weight1.is_distributed)
        self.assertFalse(expert.weight2.is_distributed)

    def test_intermediate_ep_sharded_requires_expert_parallel(self):
        # allgather is the dispatcher that *would* trigger intermediate
        # sharding, but only together with expert parallelism. With EP off the
        # property must stay False (guards against dropping the AND term).
        for dispatcher in ("alltoall", "allgather"):
            config = _make_config(moe_token_dispatcher_type=dispatcher)
            expert = GroupedMLPExpert(
                num_local_experts=2, config=config, moe_deep_gemm=False
            )
            self.assertFalse(expert.expert_parallel)
            self.assertFalse(
                expert.intermediate_ep_sharded,
                msg=f"dispatcher={dispatcher}",
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGroupedMLPExpertShardedStateDict(unittest.TestCase):
    """sharded_state_dict (ep_group is None) flattening + name preservation."""

    def test_weights_flattened_and_named_before_builder(self):
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=3, config=config, moe_deep_gemm=False
        )
        self.assertIsNone(expert.ep_group)

        w1_ref = expert.weight1.astype("float32").numpy()
        w2_ref = expert.weight2.astype("float32").numpy()
        name1 = expert.weight1.name
        name2 = expert.weight2.name

        captured = {}
        sentinel = object()

        def fake_builder(state_dict, arg, prefix):
            captured["state_dict"] = state_dict
            captured["arg"] = arg
            captured["prefix"] = prefix
            return sentinel

        with mock.patch.object(
            moe_expert_mod, "build_sharded_state_dict", side_effect=fake_builder
        ) as builder:
            result = expert.sharded_state_dict()

        # The builder's return value is passed straight back to the caller.
        self.assertIs(result, sentinel)
        builder.assert_called_once()
        self.assertIsNone(captured["arg"])
        self.assertEqual(captured["prefix"], "")

        sd = captured["state_dict"]
        self.assertEqual(set(sd), {"weight1", "weight2"})
        # Fused 3-D [E,H,2I] / [E,I,H] weights are flattened to 2-D on the
        # leading dims; a transpose or wrong reshape would change the content.
        self.assertEqual(list(sd["weight1"].shape), [3 * 64, 2 * 128])
        self.assertEqual(list(sd["weight2"].shape), [3 * 128, 64])
        np.testing.assert_array_equal(
            sd["weight1"].astype("float32").numpy(),
            w1_ref.reshape(3 * 64, 2 * 128),
        )
        np.testing.assert_array_equal(
            sd["weight2"].astype("float32").numpy(),
            w2_ref.reshape(3 * 128, 64),
        )
        # Names must be carried over so the checkpoint keys stay stable.
        self.assertEqual(sd["weight1"].name, name1)
        self.assertEqual(sd["weight2"].name, name2)


@unittest.skipUnless(_HAS_CUDA, "BMMFunction.batched_gemm requires CUDA")
class TestBMMFunction(unittest.TestCase):
    """Grouped batched-GEMM PyLayer forward/backward (GPU only)."""

    def _inputs(self):
        # Uneven groups: rows 0,1 -> y[0]; row 2 -> y[1]. K=3, N=2.
        x_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        y_np = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        batch_sizes = [2, 1]
        return x_np, y_np, batch_sizes

    def test_forward_matches_grouped_matmul(self):
        x_np, y_np, batch_sizes = self._inputs()
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)

        out = BMMFunction.apply(x, y, batch_sizes, False, None)

        expected = _grouped_matmul(x_np, y_np, batch_sizes, trans_y=False)
        self.assertEqual(list(out.shape), [3, 2])
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), expected, rtol=1e-5, atol=1e-5
        )

    def test_backward_matches_independent_grads(self):
        x_np, y_np, batch_sizes = self._inputs()
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False

        out = BMMFunction.apply(x, y, batch_sizes, False, None)
        g_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        out.backward(paddle.to_tensor(g_np))

        # dx: per group, grad_block @ y[g].T
        dx_expected = _grouped_matmul(g_np, y_np, batch_sizes, trans_y=True)
        # dy[g]: x_block.T @ grad_block, assembled per group.
        dy_blocks = []
        start = 0
        for g, n in enumerate(batch_sizes):
            xb = x_np[start : start + n].astype(np.float64)
            gb = g_np[start : start + n].astype(np.float64)
            dy_blocks.append(xb.T @ gb)
            start += n
        dy_expected = np.stack(dy_blocks, axis=0)

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        # Non-zero references so a zeroed/halved gradient cannot pass.
        self.assertGreater(np.abs(dx_expected).max(), 1.0)
        self.assertGreater(np.abs(dy_expected).max(), 1.0)
        np.testing.assert_allclose(
            x.grad.numpy().astype(np.float64), dx_expected, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            y.grad.numpy().astype(np.float64), dy_expected, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
