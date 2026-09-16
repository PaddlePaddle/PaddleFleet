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
"""Behavior tests for the DeepSeek-V3 pretrain MoE token dispatcher.

Production entries under test:
    src/paddlefleet/cli/train/deepseek_v3_pretrain/moe_utils.py
        * topk_to_permuted_indices  -- group tokens by expert ownership,
          with capacity clipping via num_tokens_per_expert_list.
        * permute_fast              -- gather/reposition tokens by the
          per-expert permutation index (index_select).
        * unpermute_fast            -- probability-weighted scatter-add back
          to each token's original row (output repositioning + prob apply).
    src/paddlefleet/cli/train/deepseek_v3_pretrain/token_dispatcher.py
        * _DeepepManager._indices_to_multihot -- padding (-1) exclusion and
          multihot expert-slot repositioning.
        * MoETokenDispatcher        -- abstract dispatch/restore contract.

These belong to the *distributed training / MoE dispatcher* module. Per the
repository unit-test rules the CPU-observable permutation / indexing / capacity
logic is verified against a HAND-DERIVED oracle using content-distinguishable
tokens (identity, not shape). The cross-rank AllToAll numerics of dispatch() /
combine() (fused_dispatch / fused_combine) require a real process group and are
NOT faked with world_size + a mocked collective (antipattern 13); they are
covered by a dedicated skipTest that states why.

All numeric paths depend on paddle. Import is attempted lazily and guarded with
try/except ImportError -> skipTest, so this file still collects on a CPU-only
box that lacks paddle / paddlefleet.
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
_PKG_DIR = os.path.join(
    _SRC_DIR, "paddlefleet", "cli", "train", "deepseek_v3_pretrain"
)
_PKG_NAME = "paddlefleet.cli.train.deepseek_v3_pretrain"

_moe_utils = None
_token_dispatcher = None
_MOE_UTILS_ERR = None
_TD_ERR = None


def _ensure_pkg_node():
    """Register the package node so relative source loads resolve.

    Only the namespace node is stubbed (to bypass the heavyweight package
    __init__ -> workflow -> AutoTokenizer import); the modules themselves are
    the genuine production source files, not re-implementations.
    """
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    if _PKG_NAME not in sys.modules:
        pkg = types.ModuleType(_PKG_NAME)
        pkg.__path__ = [_PKG_DIR]
        pkg.__package__ = _PKG_NAME
        sys.modules[_PKG_NAME] = pkg


def _load_source(name):
    """Load a single production source file as a real module object."""
    full = f"{_PKG_NAME}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(_PKG_DIR, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_moe_utils():
    """Return the real moe_utils module (package import, else source)."""
    try:
        return importlib.import_module(f"{_PKG_NAME}.moe_utils")
    except ImportError:
        _ensure_pkg_node()
        return _load_source("moe_utils")


def _load_token_dispatcher():
    """Return the real token_dispatcher module (package import, else source)."""
    try:
        return importlib.import_module(f"{_PKG_NAME}.token_dispatcher")
    except ImportError:
        _ensure_pkg_node()
        # token_dispatcher imports `.moe_utils`; make sure it is resolvable.
        _load_source("moe_utils")
        return _load_source("token_dispatcher")


try:
    import paddle  # noqa: F401

    try:
        _moe_utils = _load_moe_utils()
    except ImportError as exc:
        _MOE_UTILS_ERR = exc
    try:
        _token_dispatcher = _load_token_dispatcher()
    except ImportError as exc:
        _TD_ERR = exc
except ImportError as exc:  # paddle itself is unavailable
    _MOE_UTILS_ERR = exc
    _TD_ERR = exc


class _CpuPaddleBase(unittest.TestCase):
    """Force CPU so the permutation/indexing kernels stay deterministic and do
    not assert on a missing accelerator in a no-card environment."""

    def setUp(self):
        try:
            import paddle
        except ImportError as exc:  # pragma: no cover - guarded above too
            self.skipTest(f"paddle unavailable: {exc}")
        self.paddle = paddle
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")

    def tearDown(self):
        if getattr(self, "paddle", None) is not None:
            self.paddle.set_device(self._orig_device)


class _GpuRestrictNonzeroBase(unittest.TestCase):
    """Force GPU for cases that go through ``topk_to_permuted_indices``.

    That helper builds its permutation via
    ``paddle.tensor.search._restrict_nonzero``, whose kernel is registered for
    the GPU backend only; on a CPU place it raises "kernel ... not registered.
    Selected wrong Backend CPU". Run on GPU (skip when CUDA is not compiled in)
    so the real op executes. No production change -- the op is legitimately
    GPU-only.
    """

    def setUp(self):
        try:
            import paddle
        except ImportError as exc:  # pragma: no cover - guarded above too
            self.skipTest(f"paddle unavailable: {exc}")
        if not paddle.is_compiled_with_cuda():
            self.skipTest(
                "topk_to_permuted_indices uses the GPU-only kernel "
                "_restrict_nonzero; this build has no CUDA support"
            )
        self.paddle = paddle
        self._orig_device = paddle.get_device()
        paddle.set_device("gpu")

    def tearDown(self):
        if getattr(self, "paddle", None) is not None:
            self.paddle.set_device(self._orig_device)


class TestTopkToPermutedIndices(_GpuRestrictNonzeroBase):
    """Oracle for topk_to_permuted_indices: group (token, slot) entries by the
    expert they were dispatched to, in expert order, preserving identity."""

    def setUp(self):
        super().setUp()
        if _moe_utils is None:
            self.skipTest(f"moe_utils unavailable: {_MOE_UTILS_ERR}")

    def test_groups_slots_by_expert_ownership(self):
        """dispatched_indices (per-token top-k expert ids), topk=2:

            [[0, 1],
             [1, 0],
             [2, 1]]  -> flattened positions 0..5 = [0,1,1,0,2,1]

        Grouping by expert (counts [2,3,1]) yields the flat prob indices
        expert0:[0,3] ++ expert1:[1,2,5] ++ expert2:[4] = [0,3,1,2,5,4];
        token indices = prob // topk = [0,1,0,1,2,2].
        """
        paddle = self.paddle
        dispatched = paddle.to_tensor([[0, 1], [1, 0], [2, 1]], dtype="int64")
        tok_idx, prob_idx = _moe_utils.topk_to_permuted_indices(
            dispatched, [2, 3, 1], topk=2
        )
        self.assertEqual(prob_idx.tolist(), [0, 3, 1, 2, 5, 4])
        self.assertEqual(tok_idx.tolist(), [0, 1, 0, 1, 2, 2])

    def test_capacity_clip_drops_trailing_tokens(self):
        """With expert1 capacity clipped to 2 (< its 3 real assignments), the
        third occurrence (flat position 5, token 2) must be dropped:

            counts [2,2,1] -> expert0:[0,3] ++ expert1:[1,2] ++ expert2:[4]
            prob indices = [0,3,1,2,4]; token indices = [0,1,0,1,2].

        Position 5 is absent -> token 2's second routed slot is excluded.
        """
        paddle = self.paddle
        dispatched = paddle.to_tensor([[0, 1], [1, 0], [2, 1]], dtype="int64")
        tok_idx, prob_idx = _moe_utils.topk_to_permuted_indices(
            dispatched, [2, 2, 1], topk=2
        )
        self.assertEqual(prob_idx.tolist(), [0, 3, 1, 2, 4])
        self.assertEqual(tok_idx.tolist(), [0, 1, 0, 1, 2])
        self.assertNotIn(5, prob_idx.tolist())


class TestPermuteFast(_CpuPaddleBase):
    """Oracle for permute_fast: rows are gathered/repositioned by
    token_permuted_indices; token content/identity must be preserved."""

    def setUp(self):
        super().setUp()
        if _moe_utils is None:
            self.skipTest(f"moe_utils unavailable: {_MOE_UTILS_ERR}")

    def test_permute_repositions_rows_by_index(self):
        """tokens rows are content-distinguishable; index [0,1,0,1,2,2] must
        reproduce those exact rows in that order (not merely the shape)."""
        paddle = self.paddle
        tokens = paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype="float32"
        )
        index = paddle.to_tensor([0, 1, 0, 1, 2, 2], dtype="int64")
        out = _moe_utils.permute_fast(tokens, index)
        self.assertEqual(out.shape, [6, 2])
        self.assertEqual(
            out.tolist(),
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [30.0, 31.0],
            ],
        )

    def test_permute_rejects_drop_and_pad(self):
        """drop_and_pad path is explicitly unsupported (real assertion)."""
        paddle = self.paddle
        tokens = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        index = paddle.to_tensor([0], dtype="int64")
        with self.assertRaises(AssertionError):
            _moe_utils.permute_fast(tokens, index, drop_and_pad=True)


class TestUnpermuteFast(_CpuPaddleBase):
    """Oracle for unpermute_fast: scatter-add expert outputs back to each
    token's original row, applying per-slot probabilities when supplied."""

    def setUp(self):
        super().setUp()
        if _moe_utils is None:
            self.skipTest(f"moe_utils unavailable: {_MOE_UTILS_ERR}")

    def test_unpermute_repositions_without_probs(self):
        """No probs -> plain scatter-add of permuted rows back to origin rows.

        permuted rows 1..6, token_permuted_indices [0,1,0,1,2,2] ->
            row0 = perm0 + perm2 = [1,1]+[3,3] = [4,4]
            row1 = perm1 + perm3 = [2,2]+[4,4] = [6,6]
            row2 = perm4 + perm5 = [5,5]+[6,6] = [11,11]
        """
        paddle = self.paddle
        permuted = paddle.to_tensor(
            [[float(i)] * 2 for i in range(1, 7)], dtype="float32"
        )
        tok_idx = paddle.to_tensor([0, 1, 0, 1, 2, 2], dtype="int64")
        prob_idx = paddle.to_tensor([0, 3, 1, 2, 5, 4], dtype="int64")
        out = _moe_utils.unpermute_fast(
            permuted, tok_idx, prob_idx, restore_shape=[3, 2], probs=None
        )
        self.assertEqual(out.shape, [3, 2])
        self.assertEqual(out.tolist(), [[4.0, 4.0], [6.0, 6.0], [11.0, 11.0]])

    def test_unpermute_applies_probs_then_scatters(self):
        """probs are gathered by prob_permuted_indices, applied per row, then
        scatter-added to origin rows.

        probs flat = [0.5,0.3,0.2,0.4,0.1,0.9], prob_idx [0,3,1,2,5,4]
            -> per-slot weights [0.5,0.4,0.3,0.2,0.9,0.1]
        weighted rows (perm rows 1..6):
            j0 0.5*[1,1]=[0.5,0.5] (tok0)   j2 0.3*[3,3]=[0.9,0.9] (tok0)
            j1 0.4*[2,2]=[0.8,0.8] (tok1)   j3 0.2*[4,4]=[0.8,0.8] (tok1)
            j4 0.9*[5,5]=[4.5,4.5] (tok2)   j5 0.1*[6,6]=[0.6,0.6] (tok2)
        origin sums: tok0=[1.4,1.4] tok1=[1.6,1.6] tok2=[5.1,5.1]
        """
        import numpy as np

        paddle = self.paddle
        permuted = paddle.to_tensor(
            [[float(i)] * 2 for i in range(1, 7)], dtype="float32"
        )
        tok_idx = paddle.to_tensor([0, 1, 0, 1, 2, 2], dtype="int64")
        prob_idx = paddle.to_tensor([0, 3, 1, 2, 5, 4], dtype="int64")
        probs = paddle.to_tensor(
            [[0.5, 0.3], [0.2, 0.4], [0.1, 0.9]], dtype="float32"
        )
        out = _moe_utils.unpermute_fast(
            permuted, tok_idx, prob_idx, restore_shape=[3, 2], probs=probs
        )
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[1.4, 1.4], [1.6, 1.6], [5.1, 5.1]], dtype="float32"),
            rtol=1e-6,
            atol=1e-6,
        )


class TestIndicesToMultihot(_CpuPaddleBase):
    """Oracle for _DeepepManager._indices_to_multihot: -1 padding is excluded
    and selected experts are repositioned into their local-expert slots."""

    def setUp(self):
        super().setUp()
        if _token_dispatcher is None:
            self.skipTest(f"token_dispatcher unavailable: {_TD_ERR}")

    def _manager(self, num_local_experts):
        # _indices_to_multihot only reads num_local_experts; the real __init__
        # requires DeepEP (fused_dispatch) which is not the logic under test.
        # __new__ + setting only that attribute keeps the REAL method running
        # against an independent oracle (not a mock of the method itself).
        mgr = _token_dispatcher._DeepepManager.__new__(
            _token_dispatcher._DeepepManager
        )
        mgr.num_local_experts = num_local_experts
        return mgr

    def test_padding_excluded_and_experts_repositioned(self):
        """indices [[0,2],[1,-1],[3,0]] over 4 local experts, probs
        [[0.5,0.3],[0.7,0.0],[0.2,0.8]]:

            token0 -> experts {0,2}         probs slot0=0.5 slot2=0.3
            token1 -> expert  {1} (-1 drop) prob  slot1=0.7
            token2 -> experts {3,0}         probs slot3=0.2 slot0=0.8
        """
        import numpy as np

        paddle = self.paddle
        mgr = self._manager(num_local_experts=4)
        indices = paddle.to_tensor([[0, 2], [1, -1], [3, 0]], dtype="int64")
        probs = paddle.to_tensor(
            [[0.5, 0.3], [0.7, 0.0], [0.2, 0.8]], dtype="float32"
        )
        routing_map, multihot_probs = mgr._indices_to_multihot(indices, probs)
        self.assertEqual(routing_map.dtype, paddle.bool)
        self.assertEqual(
            routing_map.astype("int64").tolist(),
            [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
        )
        np.testing.assert_allclose(
            multihot_probs.numpy(),
            np.array(
                [
                    [0.5, 0.0, 0.3, 0.0],
                    [0.0, 0.7, 0.0, 0.0],
                    [0.8, 0.0, 0.0, 0.2],
                ],
                dtype="float32",
            ),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_all_padding_yields_empty_routing(self):
        """A fully masked row (all -1) selects no expert."""
        paddle = self.paddle
        mgr = self._manager(num_local_experts=4)
        indices = paddle.to_tensor([[-1, -1]], dtype="int64")
        probs = paddle.to_tensor([[0.0, 0.0]], dtype="float32")
        routing_map, _ = mgr._indices_to_multihot(indices, probs)
        self.assertEqual(routing_map.astype("int64").tolist(), [[0, 0, 0, 0]])
        self.assertFalse(bool(routing_map.any()))


class TestMoETokenDispatcherContract(_CpuPaddleBase):
    """Base dispatcher contract: dispatch/restore are abstract and the ep_size
    property reflects the real expert-parallel group world size."""

    def setUp(self):
        super().setUp()
        if _token_dispatcher is None:
            self.skipTest(f"token_dispatcher unavailable: {_TD_ERR}")

    def test_ep_size_reads_group_world_size(self):
        paddle = self.paddle
        group = types.SimpleNamespace(world_size=4)
        disp = _token_dispatcher.MoETokenDispatcher(ep_group=group)
        self.assertIs(disp.ep_group, group)
        self.assertEqual(disp.ep_size, 4)

    def test_token_permutation_is_abstract(self):
        paddle = self.paddle
        group = types.SimpleNamespace(world_size=2)
        disp = _token_dispatcher.MoETokenDispatcher(ep_group=group)
        with self.assertRaises(NotImplementedError):
            disp.token_permutation(
                paddle.zeros([2, 4], dtype="float32"),
                paddle.zeros([2, 4], dtype="float32"),
                paddle.zeros([2, 4], dtype="int64"),
            )

    def test_token_unpermutation_is_abstract(self):
        paddle = self.paddle
        group = types.SimpleNamespace(world_size=2)
        disp = _token_dispatcher.MoETokenDispatcher(ep_group=group)
        with self.assertRaises(NotImplementedError):
            disp.token_unpermutation(
                paddle.zeros([2, 4], dtype="float32"), bias=None
            )


class TestCrossRankAllToAll(unittest.TestCase):
    """Dispatch()/combine() route tokens across expert-parallel ranks via
    fused_dispatch/fused_combine (DeepEP AllToAll). Their cross-rank numerics
    -- each rank owning different experts, then exchanging tokens and combining
    weighted results -- can only be proven with a REAL multi-rank process group
    where every rank holds distinguishable content.

    Faking world_size and mocking the collective would only exercise local
    plumbing and would pass even if peer selection, split sizes, expert homing,
    or the cross-rank sum were wrong (unit-test-antipatterns.md #13). It is
    therefore skipped here rather than asserted with a single-process stand-in.
    """

    def test_alltoall_dispatch_combine_needs_real_process_group(self):
        self.skipTest(
            "Cross-rank AllToAll dispatch/combine numerics require a real "
            "expert-parallel process group (EP>1) run under "
            "tests/multi_card_tests via paddle.distributed.launch; a "
            "single-process world_size fake + mocked collective cannot prove "
            "peer/split/expert-homing correctness (antipattern 13)."
        )


if __name__ == "__main__":
    unittest.main()
