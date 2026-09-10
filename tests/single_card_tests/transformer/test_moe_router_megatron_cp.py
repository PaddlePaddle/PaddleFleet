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

"""Single-card coverage for TopKRouter.forward's megatron-MTP CP branch.

The router's ``forward`` has an ``elif`` that, under CP>1 +
use_erndata=True, slices ``input_ids`` down to this rank's chunk to match the
embedding — in whatever layout ``config.cp_balance_mode`` names. We reach it
single-card by:

- ``TopKRouter.__new__`` + MagicMock config (experimental_dataflow=False so
  the preceding ``if`` is skipped and the ``elif`` is evaluated;
  use_erndata=True);
- monkeypatching the module-level ``get_context_parallel_world_size`` -> 2
  and ``get_context_parallel_rank`` -> 0;
- feeding a 3-D ``input`` and an ``input_ids`` whose seq length differs from
  ``input``'s, so the ``elif`` condition is True;
- replacing the (locally-imported) ``extract_local_cp_chunks`` with a recorder
  that captures its kwargs and then raises a sentinel, so execution stops right
  after the slice (before the full MoE routing, which needs real gate weights).
  The sentinel proves the slice line was reached; the captured ``mode`` proves
  the router read ``config.cp_balance_mode`` instead of assuming a layout.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock
from unittest.mock import MagicMock

import paddle

import paddlefleet.transformer.moe.moe_router as mr
import paddlefleet.transformer.multi_token_prediction as mtp
from paddlefleet.transformer.moe.moe_router import TopKRouter


class _Sentinel(Exception):
    pass


@contextlib.contextmanager
def _fake_cp_and_extract(cp_size=2):
    """Fake a CP group and swap ``extract_local_cp_chunks`` for a recorder.

    Yields the ``calls`` list so a test can assert on the kwargs the router
    passed -- ``mode`` in particular. Recording and *then* raising keeps the
    original behaviour: execution stops right after the slice line, before the
    real MoE routing that would need gate weights.
    """
    calls = []

    def _record_and_raise(*a, **k):
        calls.append((a, k))
        raise _Sentinel

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                mr, "get_context_parallel_world_size", lambda: cp_size
            )
        )
        stack.enter_context(
            mock.patch.object(mr, "get_context_parallel_rank", lambda: 0)
        )
        stack.enter_context(
            mock.patch.object(mtp, "extract_local_cp_chunks", _record_and_raise)
        )
        yield calls


class TestTopKRouterMegatronCPSlice(unittest.TestCase):
    def _run_forward(self, cp_balance_mode):
        router = TopKRouter.__new__(TopKRouter)
        cfg = MagicMock()
        cfg.experimental_dataflow = False  # skip the preceding `if`
        cfg.use_erndata = True
        cfg.cp_balance_mode = cp_balance_mode
        router.config = cfg
        router.sequence_parallel = False

        B, seq_len, d_model = 1, 4, 8
        inp = paddle.randn([B, seq_len, d_model], dtype="float32")
        # input_ids seq length (6) != input seq_len (4) -> elif condition True.
        input_ids = paddle.zeros([B, 6], dtype="int64")

        with (
            _fake_cp_and_extract(cp_size=2) as calls,
            self.assertRaises(_Sentinel),
        ):
            router.forward(inp, input_ids=input_ids)
        return calls

    def test_elif_slices_input_ids(self) -> None:
        self.assertEqual(len(self._run_forward("dualchunk_allgather")), 1)

    def test_slice_uses_configured_cp_balance_mode(self) -> None:
        # The router must forward config.cp_balance_mode rather than assume a
        # layout: picking the wrong one hands this rank input_ids belonging to
        # other ranks' tokens, which shifts the MoE mask silently. Asserting on
        # the recorded kwarg is what makes a hard-coded mode fail here -- the
        # reachability check above passes either way.
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            with self.subTest(mode=mode):
                ((_args, kwargs),) = self._run_forward(mode)
                self.assertEqual(kwargs.get("mode"), mode)


if __name__ == "__main__":
    unittest.main()
