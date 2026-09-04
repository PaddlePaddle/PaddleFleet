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

"""Automatic use of cuDNN's high-precision sparse indexer backward.

There is no config switch: ``csa_indexer_bwd`` requests the v2 GEMM stage
whenever the vendored cuDNN accepts it, because it is strictly more accurate.
``_select_backend`` therefore carries the whole contract, and it fails quietly in
both directions -- requesting outside cuDNN's envelope aborts the step (the
backend is request-or-fail, it does not degrade to a slower kernel), while
failing to request inside it silently keeps the old accuracy. So every condition
is tested on both sides of its boundary, including the device family and a
wrapper that predates the ``backend`` parameter.

The rest pins the argument contract: an accepted call must hand cuDNN an fp32
``d_weights`` buffer -- without it the fp32 accumulator is rounded straight back
to the bf16 floor, which is what made an earlier version of this change a no-op
on that output -- a rejected call must be argument-for-argument what it was
before, and either way the caller gets its own dtypes back.

Host-side only: the wrapper is mocked, so no cuDNN build and no kernel launch.
"""

import unittest
from unittest.mock import patch

import paddle

from paddlefleet.cudnn_ops.indexer import csa_indexer_bwd_cudnn as mod

_API_PATH = (
    "paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api"
)
_SM100 = (10, 0)
_SM103 = (10, 3)


def _with_backend(*args, backend="default", **kwargs):
    """Stand-in for a cuDNN wrapper that accepts ``backend``."""
    raise AssertionError("not meant to be called")


def _without_backend(*args, **kwargs):
    """Stand-in for a wheel that predates the vendored bump."""
    raise AssertionError("not meant to be called")


def _api(wrapper):
    return type("_Api", (), {"indexer_backward_wrapper": staticmethod(wrapper)})


class _Base(unittest.TestCase):
    def setUp(self):
        # Cached on shape; the device query and wrapper signature are ambient, so
        # a decision cached under one patch would leak into the next test.
        mod._select_backend.cache_clear()
        self.addCleanup(mod._select_backend.cache_clear)
        self._refused = mod._V2_REFUSED
        mod._V2_REFUSED = False
        self.addCleanup(lambda: setattr(mod, "_V2_REFUSED", self._refused))


class TestSelectBackend(_Base):
    """``_select_backend(heads, head_dim, topk, block_I)``."""

    IN = (64, 128, 2048, 128)

    def _select(self, args, capability=_SM100, wrapper=_with_backend):
        with (
            patch.dict("sys.modules", {_API_PATH: _api(wrapper)}),
            patch.object(
                paddle.device.cuda,
                "get_device_capability",
                lambda *a, **k: capability,
            ),
        ):
            return mod._select_backend(*args)

    def test_production_shape_selects_v2(self):
        self.assertEqual(self._select(self.IN), "sm100_v2")

    def test_sm100_family_all_qualify(self):
        """The vendored cuDNN gates on the family, so SM103 qualifies too."""
        for capability in (_SM100, _SM103, (10, 7)):
            with self.subTest(capability=capability):
                self.assertEqual(
                    self._select(self.IN, capability=capability), "sm100_v2"
                )

    def test_other_families_do_not(self):
        for capability in ((9, 0), (8, 0), (12, 0)):
            with self.subTest(capability=capability):
                self.assertIsNone(self._select(self.IN, capability=capability))

    def test_wrapper_without_backend_parameter(self):
        """Passing ``backend=`` to an older wrapper is a TypeError, not a fallback."""
        self.assertIsNone(self._select(self.IN, wrapper=_without_backend))

    def test_head_and_dim_specialization(self):
        for heads, head_dim in ((32, 128), (128, 128), (64, 64), (64, 576)):
            with self.subTest(heads=heads, head_dim=head_dim):
                self.assertIsNone(self._select((heads, head_dim, 2048, 128)))

    def test_block_i_must_be_128(self):
        for block_I in (64, 256):
            with self.subTest(block_I=block_I):
                self.assertIsNone(self._select((64, 128, 2048, block_I)))

    def test_topk_boundaries(self):
        # 2048 fills the SM100 dynamic-smem budget exactly; 128 is the 1-tile floor.
        for topk in (128, 256, 1024, 2048):
            with self.subTest(inside=topk):
                self.assertEqual(self._select((64, 128, topk, 128)), "sm100_v2")
        for topk in (0, 64, 1000, 2176, 4096):
            with self.subTest(outside=topk):
                self.assertIsNone(self._select((64, 128, topk, 128)))


class TestCall(_Base):
    """What ``csa_indexer_bwd`` forwards, per selection outcome."""

    B, S, H, D, TOPK, SCOMP = 1, 8, 64, 128, 2048, 8

    def _inputs(self):
        paddle.seed(7)
        return {
            "index_q": paddle.randn(
                [self.B, self.S, self.H, self.D], dtype="float32"
            ).cast("bfloat16"),
            "weights": paddle.randn(
                [self.B, self.S, self.H], dtype="float32"
            ).cast("bfloat16"),
            "index_k_comp": paddle.randn(
                [self.B, self.SCOMP, self.D], dtype="float32"
            ).cast("bfloat16"),
            "target": paddle.nn.functional.softmax(
                paddle.randn([self.B, self.S, self.TOPK], dtype="float32"),
                axis=-1,
            ),
            "topk_probs": paddle.nn.functional.softmax(
                paddle.randn([self.B, self.S, self.TOPK], dtype="float32"),
                axis=-1,
            ),
            "topk_indices": paddle.zeros(
                [self.B, self.S, self.TOPK], dtype="int32"
            ),
            "loss_coeff": 0.01,
        }

    def _outs(self, kwargs, echo=False):
        dw = kwargs.get(
            "d_weights",
            paddle.zeros([self.B, self.S, self.H], dtype="bfloat16"),
        )
        dk = kwargs["d_index_k"]
        return {
            "d_index_q": paddle.zeros(
                [self.B, self.S, self.H, self.D], dtype="bfloat16"
            ),
            "d_weights": dw,
            "d_index_k": dk if echo else dk.cast("bfloat16"),
        }

    def _run(self, capability, inputs=None, echo=False, raise_on_v2=None):
        seen = []

        def wrapper(*args, backend="default", **kwargs):
            seen.append((backend, args[3].clone(), args[4].clone(), kwargs))
            if backend == "sm100_v2" and raise_on_v2 is not None:
                raise raise_on_v2
            return self._outs(kwargs, echo=echo)

        with (
            patch.dict("sys.modules", {_API_PATH: _api(wrapper)}),
            patch.object(
                paddle.device.cuda,
                "get_device_capability",
                lambda *a, **k: capability,
            ),
            patch.object(mod, "_require_cudnn_frontend", lambda: None),
        ):
            grads = mod.csa_indexer_bwd(**(inputs or self._inputs()))
        return seen, grads

    def test_accepted_call_passes_backend_and_fp32_d_weights(self):
        for capability in (_SM100, _SM103):
            with self.subTest(capability=capability):
                mod._select_backend.cache_clear()
                seen, _ = self._run(capability)
                self.assertEqual(len(seen), 1)
                backend, _, _, kwargs = seen[0]
                self.assertEqual(backend, "sm100_v2")
                self.assertEqual(kwargs["d_weights"].dtype, paddle.float32)
                self.assertEqual(
                    list(kwargs["d_weights"].shape),
                    [self.B, self.S, self.H],
                )
                # fp32 d_index_k predates this change and must stay fp32.
                self.assertEqual(kwargs["d_index_k"].dtype, paddle.float32)

    def test_rejected_call_is_unchanged(self):
        """The pre-existing call: no ``backend=``, no ``d_weights`` buffer."""
        seen, _ = self._run((9, 0))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "default")
        self.assertNotIn("d_weights", seen[0][3])
        self.assertEqual(seen[0][3]["d_index_k"].dtype, paddle.float32)

    def test_shape_outside_envelope_is_also_unchanged(self):
        inputs = self._inputs()
        # 1000 is not a multiple of 128, so the envelope rejects it.
        for key in ("target", "topk_probs", "topk_indices"):
            inputs[key] = inputs[key][..., :1000].contiguous()
        seen, _ = self._run(_SM103, inputs=inputs)
        self.assertEqual(seen[0][0], "default")
        self.assertNotIn("d_weights", seen[0][3])

    def test_caller_dtypes_are_restored(self):
        """The fp32 buffers are internal: bf16 in, bf16 out."""
        _, (grad_q, grad_weights, grad_k) = self._run(_SM103, echo=True)
        self.assertEqual(grad_q.dtype, paddle.bfloat16)
        self.assertEqual(grad_weights.dtype, paddle.bfloat16)
        self.assertEqual(grad_k.dtype, paddle.bfloat16)
        self.assertEqual(list(grad_weights.shape), [self.B, self.S, self.H])

    def test_device_refusal_falls_back_with_intact_buffers(self):
        """A stale wheel refuses; the retry must see byte-identical score buffers.

        cuDNN raises this from ``check_support`` / ``compile``, both before
        ``execute`` consumes the buffers, which is what makes the retry exact.
        """
        refusal = RuntimeError(
            "backend='sm100_v2' requires an SM100 device (capability (10, 0)); "
            "the plan device reports (10, 3) (use backend='default' on non-SM100)"
        )
        seen, grads = self._run(_SM103, raise_on_v2=refusal)
        self.assertEqual([s[0] for s in seen], ["sm100_v2", "default"])
        for i in (1, 2):
            self.assertTrue(
                bool(
                    (
                        seen[0][i].cast("float32") == seen[1][i].cast("float32")
                    ).all()
                ),
                f"score buffer {i} differed between attempt and retry",
            )
        self.assertNotIn("d_weights", seen[1][3])
        self.assertEqual(len(grads), 3)
        self.assertTrue(mod._V2_REFUSED)

    def test_factory_refusal_is_also_recognised(self):
        refusal = RuntimeError(
            "indexer_backward_v2_sm100 requires SM100; "
            "use backend='default' elsewhere"
        )
        seen, _ = self._run(_SM103, raise_on_v2=refusal)
        self.assertEqual([s[0] for s in seen], ["sm100_v2", "default"])

    def test_other_runtime_errors_propagate(self):
        """The narrow match: a real failure must not be downgraded to a fallback.

        Anything raised after kernel 1 has overwritten ``attn_score`` would make
        a retry silently wrong, so only the two device refusals are caught.
        """
        for exc in (
            RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            ),
            RuntimeError("cuTe compilation failed"),
            ValueError("d_weights must be bfloat16"),
        ):
            with self.subTest(exc=type(exc).__name__ + ": " + str(exc)[:24]):
                mod._select_backend.cache_clear()
                mod._V2_REFUSED = False
                with self.assertRaises(type(exc)):
                    self._run(_SM103, raise_on_v2=exc)
                self.assertFalse(mod._V2_REFUSED)

    def test_refusal_is_latched_for_the_process(self):
        refusal = RuntimeError("backend='sm100_v2' requires an SM100 device")
        seen1, _ = self._run(_SM103, raise_on_v2=refusal)
        seen2, _ = self._run(_SM103, raise_on_v2=refusal)
        self.assertEqual([s[0] for s in seen1], ["sm100_v2", "default"])
        self.assertEqual([s[0] for s in seen2], ["default"])


if __name__ == "__main__":
    unittest.main()
