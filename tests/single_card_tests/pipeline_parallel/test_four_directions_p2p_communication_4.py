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

"""CPU-observable behavior tests for the process-group-free logic in
``paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication``.

The real four-direction schedule (``_p2p_helper`` and every ``P2pHelper``
send_*/recv_* method) drives NCCL send/recv/all-gather over a live pipeline
process group. Its correctness -- peer selection, partial-send slicing across
mp ranks, and tensor reconstruction -- can only be proven by a real multi-rank
job. Faking ``world_size`` and mocking the collectives (as the coverage source
does) would only re-assert the test's own scaffolding, so those paths are
intentionally NOT covered here (see antipattern 13).

Complementary to the base file ``test_four_directions_p2p_communication.py``
(which pins ``_is_valid_send_recv_partial``, ``SendRecvMeta.set_send_message``
and the meta initial state), this file pins:

  * the dtype <-> wire-code collaborator table (``paddle_2_number`` /
    ``number_2_dtype``) that serialization depends on, against a hand-written
    mapping;
  * ``initialize_p2p_groups`` propagating the ``enable_partial_send_recv`` flag
    all the way to the ``_is_valid_send_recv_partial`` consumer;
  * ``P2pHelper.__init__`` state;
  * ``allgather_partial`` no-op identity return on the not-valid-partial path;
  * the non-member short-circuit of ``send_partial`` / ``recv_partial`` (returns
    without touching ``_hcg``).

Expected dtype codes are written independently from the on-wire protocol, not
read back from the production tables, so a swapped or dropped entry is
observable. Import of paddle / paddlefleet is guarded so a host without those
packages skips honestly (with the captured error) instead of passing vacuously.
"""

import unittest
from types import SimpleNamespace

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        four_directions_p2p_communication as p2p,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        number_2_dtype,
        paddle_2_number,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # paddle/paddlefleet absent
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)

# Independently written dtype <-> wire-code mapping (derived from the protocol,
# NOT from the production dicts) so a swapped/dropped code is caught.
_EXPECTED_DTYPE_CODES = [
    ("float16", 0),
    ("float32", 1),
    ("float64", 2),
    ("int32", 3),
    ("int64", 4),
    ("bfloat16", 5),
    ("bool", 6),
]


class _Tripwire:
    """Stand-in for ``_hcg`` that fails if any attribute is touched.

    Used to prove a short-circuit returned before the module ever consulted
    the (real, out-of-scope) hybrid communication group.
    """

    def __getattr__(self, name):
        raise AssertionError(f"_hcg.{name} must not be accessed on this path")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDtypeWireCodes(unittest.TestCase):
    """The dtype <-> wire-code collaborator table used to (de)serialize
    tensors sent between pipeline stages."""

    def test_paddle_2_number_matches_independent_table(self):
        for name, code in _EXPECTED_DTYPE_CODES:
            self.assertEqual(
                paddle_2_number(getattr(paddle, name)),
                code,
                msg=f"dtype {name} should encode to {code}",
            )

    def test_number_2_dtype_matches_independent_table(self):
        for name, code in _EXPECTED_DTYPE_CODES:
            self.assertEqual(
                number_2_dtype(code),
                name,
                msg=f"code {code} should decode to {name}",
            )

    def test_roundtrip_is_lossless_and_codes_unique(self):
        codes = []
        for name, _ in _EXPECTED_DTYPE_CODES:
            code = paddle_2_number(getattr(paddle, name))
            self.assertEqual(number_2_dtype(code), name)
            codes.append(code)
        # A reused code (two dtypes -> same wire value) would corrupt rebuild.
        self.assertEqual(len(set(codes)), len(codes))

    def test_paddle_2_number_rejects_unmapped_dtype(self):
        # complex64 is deliberately outside the transmissible dtype set.
        with self.assertRaises(AssertionError):
            paddle_2_number(paddle.complex64)

    def test_number_2_dtype_rejects_unknown_code(self):
        with self.assertRaises(AssertionError):
            number_2_dtype(99)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestInitializeP2pGroupsPropagatesFlag(unittest.TestCase):
    """``initialize_p2p_groups`` stores ``_hcg`` and, crucially, propagates
    ``enable_partial_send_recv`` to the global that ``_is_valid_send_recv_partial``
    later consumes -- observed via the downstream predicate, not just readback."""

    def setUp(self):
        paddle.set_device("cpu")
        # Capture and restore every global the entry point mutates, so a
        # False flag or an installed fake hcg does not leak to other tests.
        self.addCleanup(setattr, p2p, "_hcg", p2p._hcg)
        self.addCleanup(
            setattr,
            p2p,
            "_enable_partial_send_recv",
            p2p._enable_partial_send_recv,
        )
        self.addCleanup(setattr, p2p, "_timers", p2p._timers)

    def _fake_hcg(self):
        groups = ("g_send_next", "g_send_prev", "g_recv_next", "g_recv_prev")
        return SimpleNamespace(get_p2p_groups=lambda: groups)

    def test_disabled_flag_reaches_predicate(self):
        t = paddle.zeros([2, 4], dtype="float32")  # numel 8, divisible by 4
        p2p.initialize_p2p_groups(
            self._fake_hcg(), enable_partial_send_recv=False
        )
        self.assertIs(p2p._hcg.get_p2p_groups()[0], "g_send_next")
        # With the flag off, an otherwise-valid partial must be rejected: the
        # argument was consumed by the predicate, not merely stored.
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 4))

    def test_enabled_flag_reaches_predicate(self):
        t = paddle.zeros([2, 4], dtype="float32")
        p2p.initialize_p2p_groups(
            self._fake_hcg(), enable_partial_send_recv=True
        )
        # Same tensor + degree now passes precisely because the flag flipped.
        self.assertTrue(p2p._is_valid_send_recv_partial(t, 4))

    def test_enable_timer_false_leaves_timers_unset(self):
        p2p.initialize_p2p_groups(
            self._fake_hcg(),
            enable_partial_send_recv=True,
            enable_timer=False,
        )
        self.assertIsNone(p2p._timers)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestP2pHelperInit(unittest.TestCase):
    """``P2pHelper.__init__`` builds a fresh, empty meta and stores the cache
    flag verbatim."""

    def test_stores_use_cache_and_fresh_meta(self):
        for use_cache in (True, False):
            helper = p2p.P2pHelper(use_cache=use_cache)
            self.assertEqual(helper._use_cache, use_cache)
            meta = helper._send_recv_meta
            self.assertIsInstance(meta, p2p.SendRecvMeta)
            self.assertIsNone(meta.send_shape_message)
            self.assertIsNone(meta.recv_shape_message)
            self.assertFalse(meta.has_send_meta)
            self.assertFalse(meta.has_recv_meta)

    def test_default_use_cache_is_true(self):
        self.assertTrue(p2p.P2pHelper()._use_cache)

    def test_each_helper_gets_independent_meta(self):
        a = p2p.P2pHelper()
        b = p2p.P2pHelper()
        self.assertIsNot(a._send_recv_meta, b._send_recv_meta)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAllgatherPartialNoOpPath(unittest.TestCase):
    """``allgather_partial`` returns the input tensor unchanged when the input
    is not eligible for partial transfer (mp_degree == 1). Only this local
    no-op branch is CPU-provable; the real all-gather needs a process group."""

    def setUp(self):
        paddle.set_device("cpu")
        self.addCleanup(
            setattr,
            p2p,
            "_enable_partial_send_recv",
            p2p._enable_partial_send_recv,
        )
        p2p._enable_partial_send_recv = True

    def test_nranks_one_returns_same_object(self):
        t = paddle.arange(6, dtype="float32").reshape([2, 3])
        # Independent check: this input is genuinely not a valid partial.
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 1))
        out = p2p.allgather_partial(t, nranks=1, rank_id=0, group=None)
        # Contract is identity return, not a copy or None.
        self.assertIs(out, t)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPartialSendRecvNonMemberShortCircuit(unittest.TestCase):
    """``send_partial`` / ``recv_partial`` return immediately for a group the
    current rank does not belong to, before any peer lookup on ``_hcg``."""

    def setUp(self):
        paddle.set_device("cpu")
        self.addCleanup(setattr, p2p, "_hcg", p2p._hcg)
        # A tripwire proves the short-circuit fired before _hcg was consulted.
        p2p._hcg = _Tripwire()

    def _non_member_group(self):
        return SimpleNamespace(is_member=lambda: False, id=7)

    def test_send_partial_non_member_returns_none(self):
        t = paddle.zeros([2, 4], dtype="float32")
        result = p2p.send_partial(
            t, dst=1, nranks=2, rank_id=0, group=self._non_member_group()
        )
        self.assertIsNone(result)

    def test_recv_partial_non_member_returns_none(self):
        t = paddle.zeros([2, 4], dtype="float32")
        result = p2p.recv_partial(
            t, src=0, nranks=2, rank_id=0, group=self._non_member_group()
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
