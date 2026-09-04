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

"""Coverage for the dense score path's query-head padding.

``dense_attn_kl_scores`` widens its query/LSE to a head count the kernel's MMA
``M`` tile can express, because on SM100+ ``m_block_size // 4`` has to be a valid
``tcgen05.copy.Repetition``. The cuDNN wrapper is mocked here: what needs
guarding is *what the caller hands the kernel* (the padded width, zeros in the
query, ``+inf`` in the LSE), which is exactly what a mock can observe. The
numerics of the padded kernel itself need a device and live with the op tests.

The supported width depends on the device, since ``score_recompute/api.py``
dispatches on ``get_device_capability`` and the SM90 kernel tiles by 64 heads
instead. The capability is therefore patched rather than read, so both branches
are covered on whichever card the suite happens to run on.
"""

import sys
import types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.cudnn_ops.indexer import dense_indexer_kl_cudnn as mod

_API = "paddlefleet_ops.cudnn.deepseek_sparse_attention.score_recompute.api"


def _as_arch(major):
    """Pretend the running device is ``major``.0, so the tables are fixed."""
    return patch.object(
        paddle.device.cuda, "get_device_capability", lambda: (major, 0)
    )


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, q, k, per_head, *args, **kwargs):
        self.calls.append((q, k, per_head, kwargs))
        total_q = int(q.shape[0])
        return {
            "out": paddle.zeros([total_q, 16], dtype="float32"),
            "denom": paddle.zeros([total_q], dtype="float32"),
        }


class TestDenseScoreQheads(unittest.TestCase):
    def test_sm100_width_is_a_power_of_two_of_at_least_16(self):
        with _as_arch(10):
            for heads, want in (
                (1, 16),
                (8, 16),
                (16, 16),
                (24, 32),
                (32, 32),
                (40, 64),
                (64, 64),
                (128, 128),
                (192, 256),
            ):
                self.assertEqual(mod._dense_score_qheads(heads), want)

    def test_sm90_serves_any_width_up_to_64_and_multiples_above(self):
        # ``_interface_sm90._compute_tile_m`` caps ``tile_m`` at 64 and loops
        # ``qhpkv // tile_m`` head tiles, so nothing under 64 needs padding and
        # ``192`` is three whole tiles rather than an untileable width. Only a
        # single head is out: that same helper asserts ``qhead_per_kvhead > 1``.
        with _as_arch(9):
            for heads, want in (
                (1, 2),
                (2, 2),
                (8, 8),
                (16, 16),
                (24, 24),
                (40, 40),
                (64, 64),
                (96, 128),
                (128, 128),
                (192, 192),
            ):
                self.assertEqual(mod._dense_score_qheads(heads), want)

    def test_indexer_rejects_a_width_sm100_cannot_tile(self):
        with _as_arch(10):
            for heads in (24, 40, 96, 192):
                with self.assertRaises(ValueError):
                    mod._require_dense_score_qheads(heads, "index_n_heads")

    def test_indexer_accepts_on_sm90_what_its_kernel_tiles(self):
        # The regression this guards: an unconditional power-of-two rule would
        # reject ``index_n_heads=192``, which the SM90 wrapper runs natively.
        with _as_arch(9):
            for heads in (24, 40, 64, 192):
                mod._require_dense_score_qheads(heads, "index_n_heads")
            for heads in (1, 96):
                with self.assertRaises(ValueError):
                    mod._require_dense_score_qheads(heads, "index_n_heads")

    def test_single_head_is_rejected_on_both_arches(self):
        # ``_interface_sm90._compute_tile_m`` asserts ``qhead_per_kvhead > 1``
        # and the validators re-derive the ratio behind
        # ``assert num_head > num_head_kv``, so a single head has no tile on
        # either path -- SM100+ already needs 16.
        for major in (9, 10):
            with _as_arch(major), self.assertRaises(ValueError):
                mod._require_dense_score_qheads(1, "index_n_heads")

    def test_indexer_accepts_the_production_width_on_both_arches(self):
        for major in (9, 10):
            with _as_arch(major):
                mod._require_dense_score_qheads(64, "index_n_heads")


class TestDenseAttnKlScoresHeadPadding(unittest.TestCase):
    def _call(self, heads, head_dim=576, total_q=4, total_k=16, major=10):
        query = paddle.randn([total_q, heads, head_dim]).astype("bfloat16")
        kv = paddle.randn([total_k, head_dim]).astype("bfloat16")
        lse = paddle.randn([total_q, heads]).astype("float32")
        recorder = _Recorder()

        fake_api = types.ModuleType(_API)
        fake_api.dense_attn_score_recompute_wrapper = recorder
        with (
            patch.dict(sys.modules, {_API: fake_api}),
            patch.object(mod, "_require_cudnn_frontend", lambda: None),
            _as_arch(major),
        ):
            out, denom = mod.dense_attn_kl_scores(
                query,
                kv,
                lse,
                head_dim**-0.5,
                paddle.to_tensor([0, total_q], dtype="int32"),
                paddle.to_tensor([0, total_k], dtype="int32"),
                total_q,
                total_k,
            )
        q_seen, _, lse_seen, kwargs = recorder.calls[0]
        return query, lse, q_seen, lse_seen, kwargs, out, denom

    def test_non_power_of_two_width_is_padded(self):
        query, lse, q_seen, lse_seen, kwargs, out, denom = self._call(24)
        self.assertEqual(int(q_seen.shape[1]), 32)
        self.assertEqual(int(lse_seen.shape[1]), 32)
        # The kernel tiles on what it is told, so the two must agree.
        self.assertEqual(kwargs["qhead_per_kv_head"], 32)
        # Real heads untouched.
        self.assertTrue(
            paddle.equal_all(
                q_seen[:, :24].astype("float32"), query.astype("float32")
            )
        )
        self.assertTrue(paddle.equal_all(lse_seen[:, :24], lse))
        # Pad heads: zero query, infinite LSE, so they contribute
        # ``exp(0 - inf)`` to the head sum instead of a finite score.
        self.assertEqual(float(paddle.abs(q_seen[:, 24:]).max()), 0.0)
        self.assertTrue(bool(paddle.isinf(lse_seen[:, 24:]).all()))
        self.assertTrue(bool((lse_seen[:, 24:] > 0).all()))
        # Head-reduced outputs, so nothing has to be sliced back.
        self.assertEqual(list(out.shape), [4, 16])
        self.assertEqual(list(denom.shape), [4])

    def test_supported_width_is_passed_through_untouched(self):
        query, lse, q_seen, lse_seen, kwargs, _, _ = self._call(64)
        self.assertEqual(int(q_seen.shape[1]), 64)
        self.assertEqual(kwargs["qhead_per_kv_head"], 64)
        self.assertTrue(
            paddle.equal_all(q_seen.astype("float32"), query.astype("float32"))
        )
        self.assertTrue(paddle.equal_all(lse_seen, lse))

    def test_narrow_width_is_padded_up_to_the_floor(self):
        _, _, q_seen, lse_seen, kwargs, _, _ = self._call(8)
        self.assertEqual(int(q_seen.shape[1]), 16)
        self.assertEqual(int(lse_seen.shape[1]), 16)
        self.assertEqual(kwargs["qhead_per_kv_head"], 16)

    def test_sm90_is_handed_the_unpadded_width(self):
        query, lse, q_seen, lse_seen, kwargs, _, _ = self._call(24, major=9)
        self.assertEqual(int(q_seen.shape[1]), 24)
        self.assertEqual(kwargs["qhead_per_kv_head"], 24)
        self.assertTrue(
            paddle.equal_all(q_seen.astype("float32"), query.astype("float32"))
        )
        self.assertTrue(paddle.equal_all(lse_seen, lse))

    def test_sm90_still_pads_a_single_head_up_to_two(self):
        # ``_compute_tile_m`` asserts ``qhead_per_kvhead > 1``, so the one width
        # SM90 does have to widen is ``h == 1``.
        query, lse, q_seen, lse_seen, kwargs, _, _ = self._call(1, major=9)
        self.assertEqual(int(q_seen.shape[1]), 2)
        self.assertEqual(int(lse_seen.shape[1]), 2)
        self.assertEqual(kwargs["qhead_per_kv_head"], 2)
        self.assertTrue(
            paddle.equal_all(
                q_seen[:, :1].astype("float32"), query.astype("float32")
            )
        )
        self.assertTrue(paddle.equal_all(lse_seen[:, :1], lse))
        self.assertEqual(float(paddle.abs(q_seen[:, 1:]).max()), 0.0)
        self.assertTrue(bool((lse_seen[:, 1:] > 0).all()))
        self.assertTrue(bool(paddle.isinf(lse_seen[:, 1:]).all()))


if __name__ == "__main__":
    unittest.main()
