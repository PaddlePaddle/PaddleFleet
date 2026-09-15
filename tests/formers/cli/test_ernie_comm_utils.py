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

"""Behavior tests for ernie_pretrain comm_utils.

Module under test:
    paddlefleet.cli.train.ernie_pretrain.models.comm_utils

Scope and oracle policy
-----------------------
These tests exercise only the parts of ``comm_utils`` whose behavior is fully
determined on a single process WITHOUT a real collective:

* ``scatter`` -- the ``nranks == 1`` clone path and the rank-based *local*
  slicing (``scatter`` hands each rank its own contiguous slice; no
  communication happens). Expected slices are derived by hand from
  ``arange`` inputs.
* ``all_gather`` / ``reduce_scatter`` -- only the ``nranks == 1`` fallback,
  which is documented to return an independent clone of the input.
* ``subbatch`` -- a pure decorator that splits the batched arg(s) along an
  axis in chunks of ``bs`` and concatenates the per-chunk outputs along
  ``out_idx``. Expected chunk boundaries, per-chunk contents and the final
  concatenation are hand-derived.
* ``profile`` -- a context manager that, when ``get_timers()`` yields a timer
  factory, starts the named timer before the body and stops it after. We
  observe the real orchestration (factory args + start/stop ordering) with a
  distinguishable timer stub; ``get_timers`` is an isolated collaborator, not
  the code under test.

Multi-rank behavior of ``all_gather``/``reduce_scatter``/``scatter`` (real
``dist.stream`` collectives) is NOT verified here: it requires a real process
group and is out of scope for a no-card single-process test.

Paddle is an optional heavy dependency and the production module imports it at
top level, so every test skips (with an honest reason) when the import fails.
The local environment has no paddle installed.
"""

import unittest
import unittest.mock

import numpy as np

try:
    import paddle

    from paddlefleet.cli.train.ernie_pretrain.models import comm_utils
    from paddlefleet.cli.train.ernie_pretrain.models.comm_utils import (
        all_gather,
        reduce_scatter,
        scatter,
        subbatch,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the module) unavailable in this env
    paddle = None
    comm_utils = None
    all_gather = reduce_scatter = scatter = subbatch = None
    _IMPORT_ERROR = exc


class _Group:
    """Minimal topology descriptor used as a collaborator.

    ``scatter``/``all_gather``/``reduce_scatter`` read ``.nranks`` (and, for
    ``scatter``, ``.rank``) off the group object. This is a plain data holder,
    not a mock of the code under test.
    """

    def __init__(self, nranks, rank=0):
        self.nranks = nranks
        self.rank = rank


class ScatterTest(unittest.TestCase):
    """scatter: nranks==1 clone contract and rank-based local slicing."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                f"paddle/comm_utils import unavailable: {_IMPORT_ERROR!r}"
            )
        paddle.set_device("cpu")

    def test_single_rank_returns_independent_clone(self):
        # nranks == 1 must return a *distinct* object with identical content.
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = scatter(x, group=_Group(nranks=1, rank=0), axis=0)
        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_axis0_slices_are_rank_local_halves(self):
        # 8 rows split across 2 ranks -> each rank keeps its contiguous 4 rows.
        x = paddle.arange(24, dtype="float32").reshape([8, 3])
        full = x.numpy()

        out0 = scatter(x, group=_Group(nranks=2, rank=0), axis=0)
        out1 = scatter(x, group=_Group(nranks=2, rank=1), axis=0)

        np.testing.assert_array_equal(out0.numpy(), full[0:4])
        np.testing.assert_array_equal(out1.numpy(), full[4:8])
        # The two halves are disjoint and reassemble the original in order.
        np.testing.assert_array_equal(
            np.concatenate([out0.numpy(), out1.numpy()], axis=0), full
        )

    def test_axis1_slices_columns_by_rank(self):
        # 6 columns split across 2 ranks along axis=1 -> 3 columns each.
        x = paddle.arange(24, dtype="float32").reshape([4, 6])
        full = x.numpy()

        out0 = scatter(x, group=_Group(nranks=2, rank=0), axis=1)
        out1 = scatter(x, group=_Group(nranks=2, rank=1), axis=1)

        np.testing.assert_array_equal(out0.numpy(), full[:, 0:3])
        np.testing.assert_array_equal(out1.numpy(), full[:, 3:6])

    def test_indivisible_length_raises(self):
        # 5 rows cannot be split evenly across 2 ranks.
        x = paddle.arange(15, dtype="float32").reshape([5, 3])
        with self.assertRaises(AssertionError):
            scatter(x, group=_Group(nranks=2, rank=0), axis=0)


class AllGatherReduceScatterLocalTest(unittest.TestCase):
    """all_gather / reduce_scatter single-rank clone fallback."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                f"paddle/comm_utils import unavailable: {_IMPORT_ERROR!r}"
            )
        paddle.set_device("cpu")

    def test_all_gather_single_rank_returns_independent_clone(self):
        x = paddle.arange(20, dtype="float32").reshape([4, 5])
        out = all_gather(x, group=_Group(nranks=1), axis=0)
        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_reduce_scatter_single_rank_returns_independent_clone(self):
        x = paddle.arange(20, dtype="float32").reshape([4, 5])
        out = reduce_scatter(x, group=_Group(nranks=1))
        self.assertIsNot(out, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())


class SubbatchTest(unittest.TestCase):
    """subbatch: pass-through, chunk boundaries, contents and concatenation."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                f"paddle/comm_utils import unavailable: {_IMPORT_ERROR!r}"
            )
        paddle.set_device("cpu")

    def test_small_input_passes_through_in_single_call(self):
        # axis_width (5) < bs (8): f is called exactly once on the full input.
        seen = []

        def record(a):
            seen.append(a.numpy().copy())
            return a * 2.0

        x = paddle.arange(20, dtype="float32").reshape([5, 4])
        wrapped = subbatch(record, arg_idx=[0], axis=[0], bs=8, out_idx=0)
        out = wrapped(x)

        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], x.numpy())
        np.testing.assert_array_equal(out.numpy(), x.numpy() * 2.0)

    def test_large_input_splits_into_expected_chunks(self):
        # axis_width 8, bs 3 -> chunks at 0,3,6 with rows [0:3],[3:6],[6:8].
        seen = []

        def record(a):
            seen.append(a.numpy().copy())
            return a * 2.0

        x = paddle.arange(32, dtype="float32").reshape([8, 4])
        full = x.numpy()
        wrapped = subbatch(record, arg_idx=[0], axis=[0], bs=3, out_idx=0)
        out = wrapped(x)

        # Three chunks, hand-derived boundaries and contents.
        self.assertEqual([s.shape for s in seen], [(3, 4), (3, 4), (2, 4)])
        np.testing.assert_array_equal(seen[0], full[0:3])
        np.testing.assert_array_equal(seen[1], full[3:6])
        np.testing.assert_array_equal(seen[2], full[6:8])
        # Concatenation along out_idx=0 reproduces f applied to whole input.
        np.testing.assert_array_equal(out.numpy(), full * 2.0)

    def test_multiple_batched_args_split_together(self):
        # Both args sliced with the same boundaries and fed pairwise to f.
        pairs = []

        def add(a, b):
            pairs.append((float(a.numpy()[0, 0]), float(b.numpy()[0, 0])))
            return a + b

        a = paddle.arange(24, dtype="float32").reshape([6, 4])
        b = paddle.arange(24, 48, dtype="float32").reshape([6, 4])
        wrapped = subbatch(add, arg_idx=[0, 1], axis=[0, 0], bs=4, out_idx=0)
        out = wrapped(a, b)

        # Chunk 0 starts at a-row0 (0)/b-row0 (24); chunk 1 at a-row4 (16)/b-row4 (40).
        self.assertEqual(pairs, [(0.0, 24.0), (16.0, 40.0)])
        np.testing.assert_array_equal(out.numpy(), a.numpy() + b.numpy())

    def test_kwargs_forwarded_to_each_chunk(self):
        def scaled(x, scale=1.0):
            return x * scale

        x = paddle.arange(24, dtype="float32").reshape([6, 4])
        wrapped = subbatch(scaled, arg_idx=[0], axis=[0], bs=4, out_idx=0)
        out = wrapped(x, scale=3.0)
        np.testing.assert_array_equal(out.numpy(), x.numpy() * 3.0)

    def test_same_arg_idx_reuses_sliced_argument(self):
        # same_arg_idx maps positional arg 1 onto sliced arg 0, so f sees
        # (slice, slice) each chunk -> output is 2*x.
        def add(a, b):
            return a + b

        x = paddle.arange(24, dtype="float32").reshape([6, 4])
        wrapped = subbatch(
            add, arg_idx=[0], axis=[0], bs=3, out_idx=0, same_arg_idx={1: 0}
        )
        out = wrapped(x, x)
        np.testing.assert_array_equal(out.numpy(), x.numpy() * 2.0)

    def test_mismatched_arg_idx_and_axis_raises(self):
        def identity(a):
            return a

        x = paddle.arange(32, dtype="float32").reshape([8, 4])
        wrapped = subbatch(identity, arg_idx=[0], axis=[0, 1], bs=4, out_idx=0)
        with self.assertRaises(AssertionError):
            wrapped(x)

    def test_unequal_batched_dims_raises(self):
        def add(a, b):
            return a + b

        a = paddle.arange(24, dtype="float32").reshape([6, 4])
        b = paddle.arange(20, dtype="float32").reshape([5, 4])
        wrapped = subbatch(add, arg_idx=[0, 1], axis=[0, 0], bs=4, out_idx=0)
        with self.assertRaises(AssertionError):
            wrapped(a, b)


class ProfileTest(unittest.TestCase):
    """profile: timer factory args and start/stop ordering around the body."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                f"paddle/comm_utils import unavailable: {_IMPORT_ERROR!r}"
            )

    def test_no_timers_runs_body_without_timing(self):
        events = []
        with unittest.mock.patch.object(comm_utils, "get_timers", lambda: None):
            with comm_utils.profile("op"):
                events.append("body")
        self.assertEqual(events, ["body"])

    def test_timer_started_before_body_and_stopped_after(self):
        events = []

        class _Timer:
            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")

        timer = _Timer()
        factory_calls = []

        def factory(name, use_event=True):
            factory_calls.append((name, use_event))
            return timer

        with unittest.mock.patch.object(
            comm_utils, "get_timers", lambda: factory
        ):
            with comm_utils.profile("attn", use_event=False):
                events.append("body")

        # Factory queried once for start and once for stop, both with the name
        # and use_event that profile received.
        self.assertEqual(factory_calls, [("attn", False), ("attn", False)])
        # Timer starts before the body executes and stops only afterwards.
        self.assertEqual(events, ["start", "body", "stop"])

    def test_default_use_event_is_true(self):
        factory_calls = []

        class _Timer:
            def start(self):
                pass

            def stop(self):
                pass

        def factory(name, use_event=True):
            factory_calls.append((name, use_event))
            return _Timer()

        with unittest.mock.patch.object(
            comm_utils, "get_timers", lambda: factory
        ):
            with comm_utils.profile("layer"):
                pass

        self.assertEqual(factory_calls, [("layer", True), ("layer", True)])


if __name__ == "__main__":
    unittest.main()
