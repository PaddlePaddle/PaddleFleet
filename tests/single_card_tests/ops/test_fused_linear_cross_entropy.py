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

"""Behavior tests for the fused linear cross-entropy Triton op.

Module under test:
``paddlefleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy``.
In the repository module map this is "计算优化 / Fused Ops": a memory-efficient
fused linear + cross-entropy that chunks the [BT, V] logits and lets a Triton
kernel write grad_logits in place.

Scope of this file (CPU-observable *pure logic only*, no GPU kernel numerics):

* ``_select_ce_launch_config`` — the register-budget launch tiling math: the
  block size is capped and ``num_warps`` is derived from a per-thread element
  budget. This is plain integer arithmetic (uses the real ``triton`` power-of-2
  helper) and is hand-derived here for a spread of vocab sizes, including the
  large-vocab cap and the small-vocab warp floor.
* ``fused_linear_cross_entropy_backward`` — the ``grad_output == 1.0`` scalar
  short-circuit control-flow: it must return the saved grads unchanged (same
  objects) *without* touching any Triton element-mul kernel.
* ``LigerFusedLinearCrossEntropyFunction.backward`` — the PyLayer ctx
  save/restore control-flow: main_grad accumulation (default vs ``ec_align``
  transpose), main_grad lazy creation, the target-slot ``None`` contract, the
  bias slot, and the frozen-multimax-param -> ``None`` slot contract. All of
  these run on CPU tensors when ``grad_output`` is the scalar 1.0 (so the
  scalar short-circuit above skips every kernel launch).

The real GPU kernels (loss/softmax-grad/SegLU numerics) are intentionally NOT
driven here — that requires a real device and is out of scope for a
CPU-observable unit test. When Paddle/Triton/the kernel siblings are not
importable, the whole file honestly skips (it does not fake-pass).
"""

import os
import sys
import types
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Import the production module FOR REAL. Only genuine missing-dependency errors
# (Paddle / Triton / the CE kernel siblings absent) count as a skip condition;
# any other error surfaces instead of being swallowed as "no dep" (a blanket
# except would hide real regressions such as an API change or a syntax error).
try:
    import paddle

    from paddlefleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
        CE_BLOCK_SIZE_CAP,
        CE_ELEMENTS_PER_THREAD,
        MAX_FUSED_SIZE,
        LigerFusedLinearCrossEntropyFunction,
        _select_ce_launch_config,
        fused_linear_cross_entropy_backward,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    "requires paddle + triton + paddlefleet.triton_ops kernels "
    f"(real import failed: {_IMPORT_ERROR})"
)


def _make_ctx(**attrs):
    """Build a stand-in PyLayer ctx (a genuine not-under-test collaborator).

    ``LigerFusedLinearCrossEntropyFunction.backward`` reads its inputs and
    control flags off ``ctx``; in production these are populated by ``forward``.
    We supply them directly so the *real* backward control-flow (main_grad
    accumulation, slot routing) is exercised — we never patch ``backward``
    itself. ``saved_tensor`` is exposed as a callable returning the 5-tuple the
    current backward unpacks.
    """
    saved = attrs.pop("saved")
    ns = types.SimpleNamespace(**attrs)
    ns.saved_tensor = lambda: saved
    return ns


@unittest.skipUnless(_MODULE_AVAILABLE, _SKIP_REASON)
class TestSelectCeLaunchConfig(unittest.TestCase):
    """Hand-derived checks of the register-budget launch tiling math."""

    def test_config_matches_hand_derived_values(self):
        # Independently derived: block = min(MAX_FUSED_SIZE, next_pow2(V), CAP);
        # warps = max(1, min(32, block // (32 * ELEMENTS_PER_THREAD))).
        # With CAP=2048 and ELEMENTS_PER_THREAD=16 the divisor is 512.
        cases = {
            1: (1, 1),  # next_pow2=1;   1//512=0 -> floored to 1
            100: (128, 1),  # next_pow2=128; 128//512=0 -> 1
            256: (256, 1),  # 256//512=0 -> 1
            512: (512, 1),  # 512//512=1
            513: (1024, 2),  # next_pow2=1024; 1024//512=2
            1024: (1024, 2),
            2000: (2048, 4),  # next_pow2=2048; 2048//512=4
            2048: (2048, 4),
            4096: (2048, 4),  # next_pow2=4096 capped to 2048
            32768: (2048, 4),
            201216: (2048, 4),  # huge vocab: still capped, warps stay 4
        }
        for n_cols, expected in cases.items():
            self.assertEqual(
                _select_ce_launch_config(n_cols),
                expected,
                msg=f"n_cols={n_cols}",
            )

    def test_block_size_is_capped_regardless_of_vocab(self):
        # The whole point of the cap is that a large vocab does NOT scale the
        # register tile (that was the local-memory-spill regression). No V may
        # produce a block above the cap, and it must never exceed MAX_FUSED_SIZE.
        for n_cols in (2049, 5000, 50000, 201216, 1_000_000):
            block, warps = _select_ce_launch_config(n_cols)
            self.assertEqual(block, CE_BLOCK_SIZE_CAP)
            self.assertLessEqual(block, MAX_FUSED_SIZE)
            self.assertGreaterEqual(warps, 1)

    def test_warp_budget_gives_target_columns_per_thread(self):
        # Where warps are not floored (block >= 32*ELEMENTS_PER_THREAD = 512),
        # each of the warps*32 threads must cover exactly ELEMENTS_PER_THREAD
        # columns of the tile: block == warps * 32 * ELEMENTS_PER_THREAD.
        for n_cols in (512, 1024, 2048, 4096, 201216):
            block, warps = _select_ce_launch_config(n_cols)
            self.assertEqual(block, warps * 32 * CE_ELEMENTS_PER_THREAD)

    def test_warp_count_floored_to_one_for_tiny_vocab(self):
        # A launch with 0 warps is invalid; tiny vocab must floor to 1 warp.
        for n_cols in (1, 2, 8, 100, 256):
            _, warps = _select_ce_launch_config(n_cols)
            self.assertEqual(warps, 1)


@unittest.skipUnless(_MODULE_AVAILABLE, _SKIP_REASON)
class TestBackwardScalarShortCircuit(unittest.TestCase):
    """The grad_output==1.0 scalar path returns saved grads untouched."""

    def test_scalar_one_returns_saved_grads_by_identity(self):
        # grad_output is the 0-d scalar 1.0 -> the function must short-circuit
        # and return the exact saved-grad objects, never launching the Triton
        # element-mul kernel (which would need a device). Object identity proves
        # nothing was recomputed or copied.
        grad_input = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        grad_weight = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])
        grad_bias = paddle.to_tensor([5.0, 6.0])
        grad_output = paddle.to_tensor(1.0)
        self.assertEqual(grad_output.shape, [])

        out = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        self.assertIs(out[0], grad_input)
        self.assertIs(out[1], grad_weight)
        self.assertIs(out[2], grad_bias)

    def test_scalar_one_with_none_grads_passthrough(self):
        # None slots (frozen input/weight/bias) must survive the short-circuit.
        out = fused_linear_cross_entropy_backward(
            paddle.to_tensor(1.0), None, None, None
        )
        self.assertEqual(out, (None, None, None))


@unittest.skipUnless(_MODULE_AVAILABLE, _SKIP_REASON)
class TestPyLayerBackwardControlFlow(unittest.TestCase):
    """ctx save/restore control-flow of the custom PyLayer backward.

    All cases use grad_output == scalar 1.0 so the inner
    ``fused_linear_cross_entropy_backward`` short-circuits and no kernel is
    launched; the accumulation math below is pure CPU tensor arithmetic.
    """

    def _base_ctx(self, **overrides):
        attrs = {
            "weight_requires_grad": False,
            "ec_align": False,
            "has_bias": False,
            "has_multimax": False,
            "weight_ref": None,
            "multimax_ranges_ref": None,
            "multimax_ts_ref": None,
            "multimax_ranges_requires_grad": False,
            "multimax_ts_requires_grad": False,
        }
        attrs.update(overrides)
        return _make_ctx(**attrs)

    def test_default_mode_accumulates_grad_weight_into_main_grad(self):
        grad_input = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        grad_weight = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])  # [V, H]
        weight = paddle.to_tensor([[0.0, 0.0], [0.0, 0.0]])
        weight.main_grad = paddle.zeros([2, 2], dtype=paddle.float32)

        ctx = self._base_ctx(
            weight_requires_grad=True,
            ec_align=False,
            weight_ref=weight,
            saved=(grad_input, grad_weight, None, None, None),
        )
        result = LigerFusedLinearCrossEntropyFunction.backward(
            ctx, paddle.to_tensor(1.0)
        )

        # main_grad += grad_weight (direct, no transpose).
        self.assertEqual(
            weight.main_grad.tolist(), [[10.0, 20.0], [30.0, 40.0]]
        )
        # grad_weight was consumed into main_grad -> returned None at its slot.
        # No bias / multimax -> 3-tuple (grad_input, grad_weight, target=None).
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], grad_input)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])

    def test_ec_align_mode_transposes_before_accumulating(self):
        grad_input = paddle.to_tensor([[1.0, 1.0]])
        # ec_align: grad_weight is [H, V]; main_grad is [V, H] -> add grad_weight.T
        grad_weight = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weight = paddle.zeros([3, 2], dtype=paddle.float32)
        weight.main_grad = paddle.zeros([3, 2], dtype=paddle.float32)

        ctx = self._base_ctx(
            weight_requires_grad=True,
            ec_align=True,
            weight_ref=weight,
            saved=(grad_input, grad_weight, None, None, None),
        )
        LigerFusedLinearCrossEntropyFunction.backward(
            ctx, paddle.to_tensor(1.0)
        )

        # Hand-derived transpose of grad_weight.
        self.assertEqual(
            weight.main_grad.tolist(),
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        )

    def test_main_grad_created_when_absent(self):
        grad_input = paddle.to_tensor([[1.0, 2.0]])
        grad_weight = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])
        weight = paddle.zeros([2, 2], dtype=paddle.float32)
        weight.main_grad = None  # not yet allocated

        ctx = self._base_ctx(
            weight_requires_grad=True,
            weight_ref=weight,
            saved=(grad_input, grad_weight, None, None, None),
        )
        LigerFusedLinearCrossEntropyFunction.backward(
            ctx, paddle.to_tensor(1.0)
        )

        self.assertIsNotNone(weight.main_grad)
        # Freshly created zeros + grad_weight == grad_weight.
        self.assertEqual(
            weight.main_grad.tolist(), [[10.0, 20.0], [30.0, 40.0]]
        )

    def test_bias_slot_appended_and_target_slot_none(self):
        grad_input = paddle.to_tensor([[1.0, 2.0]])
        grad_weight = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])
        grad_bias = paddle.to_tensor([7.0, 8.0])

        # weight frozen -> grad_weight passes straight through unchanged.
        ctx = self._base_ctx(
            weight_requires_grad=False,
            has_bias=True,
            saved=(grad_input, grad_weight, grad_bias, None, None),
        )
        result = LigerFusedLinearCrossEntropyFunction.backward(
            ctx, paddle.to_tensor(1.0)
        )

        # Layout: (grad_input, grad_weight, target=None, grad_bias).
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], grad_input)
        self.assertIs(result[1], grad_weight)
        self.assertIsNone(result[2])
        self.assertIs(result[3], grad_bias)

    def test_frozen_multimax_params_return_none_at_their_slots(self):
        # multimax present but BOTH params frozen: even though the kernel
        # produced grad tensors, the PyLayer contract requires None at the
        # frozen slots (must not touch main_grad / fire hooks).
        grad_input = paddle.to_tensor([[1.0, 2.0]])
        grad_weight = paddle.to_tensor([[1.0, 1.0], [1.0, 1.0]])
        grad_mm_ranges = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
        grad_mm_ts = paddle.to_tensor([0.5, 0.6, 0.7, 0.8])
        mm_ranges = paddle.zeros([4], dtype=paddle.float32)
        mm_ts = paddle.zeros([4], dtype=paddle.float32)

        ctx = self._base_ctx(
            weight_requires_grad=False,
            has_bias=False,
            has_multimax=True,
            multimax_ranges_ref=mm_ranges,
            multimax_ts_ref=mm_ts,
            multimax_ranges_requires_grad=False,
            multimax_ts_requires_grad=False,
            saved=(
                grad_input,
                grad_weight,
                None,
                grad_mm_ranges,
                grad_mm_ts,
            ),
        )
        result = LigerFusedLinearCrossEntropyFunction.backward(
            ctx, paddle.to_tensor(1.0)
        )

        # Layout: (grad_input, grad_weight, target=None, mm_ranges, mm_ts).
        self.assertEqual(len(result), 5)
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])


if __name__ == "__main__":
    unittest.main()
