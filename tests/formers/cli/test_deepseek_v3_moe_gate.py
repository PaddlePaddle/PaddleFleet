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
"""Behavior tests for the DeepSeek-V3 pretrain MoE gate (``PretrainedMoEGate``).

Production entry under test:
    src/paddlefleet/cli/train/deepseek_v3_pretrain/moe_gate.py

These are model-layer (MoE routing) tests. Following the repository unit-test
rules, they verify against a *hand-derived oracle*:

  * expert SELECTION (which experts win top-k), not just tensor shape;
  * the raw gating VALUES returned for the selected experts;
  * NORMALIZATION (norm_topk_prob renormalizes the selected weights so they sum
    to 1 across the top-k for each token);
  * routed SCALING (routed_scaling_factor); with scaling on, weights need not
    sum to 1;
  * the noaux_tc rule: e_score_correction_bias only shifts the *selection*
    affinity; it is NOT the final gating weight (in eval the returned weight is
    the un-biased raw score at the selected indices).

All numeric paths depend on paddle. Every such test imports paddle lazily and
skips with a clear reason when paddle (or the real gate module) is unavailable,
so the file still collects on a CPU-only box without paddle installed.
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest

# --- Resolve the real production module (no mocks, no re-implementation). ---
# tests/formers/cli/ -> repo root is three levels up; production src lives under
# src/paddlefleet/...  We import the genuine paddlefleet API.  If the eager
# package __init__ (which pulls workflow -> AutoTokenizer) or paddle itself is
# missing, we fall back to loading the real source file directly, and finally
# record the failure so dependent tests skip rather than error.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
_GATE_SRC = os.path.join(
    _SRC_DIR,
    "paddlefleet",
    "cli",
    "train",
    "deepseek_v3_pretrain",
    "moe_gate.py",
)

PretrainedMoEGate = None
_IMPORT_ERROR = None


def _load_real_gate():
    """Return the real PretrainedMoEGate class from the production source.

    Tries the ordinary package import first; on failure loads the genuine
    moe_gate.py file object (still the production code) while stubbing only the
    ``deepseek_v3_pretrain`` package node so its heavyweight ``__init__`` /
    workflow import is bypassed.
    """
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)

    try:
        mod = importlib.import_module(
            "paddlefleet.cli.train.deepseek_v3_pretrain.moe_gate"
        )
        return mod.PretrainedMoEGate
    except ImportError:
        pass

    pkg_name = "paddlefleet.cli.train.deepseek_v3_pretrain"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [os.path.dirname(_GATE_SRC)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
    full = pkg_name + ".moe_gate"
    spec = importlib.util.spec_from_file_location(full, _GATE_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod.PretrainedMoEGate


try:
    import paddle  # noqa: F401

    PretrainedMoEGate = _load_real_gate()
except ImportError as exc:  # paddle or a real dependency is unavailable
    _IMPORT_ERROR = exc


class _GateConfig:
    """Minimal real config object (not a mock) consumed by the gate.

    Only ``seq_aux`` / ``sequence_parallel`` are read on the CPU no-drop paths
    exercised here; using a plain object keeps the production code under test
    instead of a MagicMock that would silently satisfy any attribute access.
    """

    def __init__(self, seq_aux=False, sequence_parallel=False):
        self.seq_aux = seq_aux
        self.sequence_parallel = sequence_parallel


class _MoEGateOracleBase(unittest.TestCase):
    NUM_EXPERTS = 4
    HIDDEN = 8

    def setUp(self):
        if _IMPORT_ERROR is not None or PretrainedMoEGate is None:
            self.skipTest(f"paddle/gate module unavailable: {_IMPORT_ERROR}")
        try:
            import paddle
        except ImportError as exc:  # pragma: no cover - defensive
            self.skipTest(f"paddle import failed: {exc}")
        self.paddle = paddle
        self._orig_device = paddle.get_device()
        # one_hot and topk are exercised on CPU to stay deterministic and to
        # avoid GPU-only kernel assertions in a no-card environment.
        paddle.set_device("cpu")

    def tearDown(self):
        if getattr(self, "paddle", None) is not None:
            self.paddle.set_device(self._orig_device)

    def _make_gate(self, num_experts=None, seq_aux=False, **kwargs):
        return PretrainedMoEGate(
            config=_GateConfig(seq_aux=seq_aux),
            num_experts=num_experts or self.NUM_EXPERTS,
            expert_hidden_size=self.HIDDEN,
            **kwargs,
        )


class TestTopkSelectionPrimitives(_MoEGateOracleBase):
    """Oracle: greedy / group-limited top-k pick the correct experts+values."""

    def test_topk_greedy_selects_highest_scoring_experts(self):
        """Oracle: for row [0.10,0.40,0.25,0.05,0.20], top-2 = idx[1,2],
        values [0.40,0.25] in descending order."""
        paddle = self.paddle
        gate = self._make_gate(num_experts=5)
        scores = paddle.to_tensor(
            [[0.10, 0.40, 0.25, 0.05, 0.20]], dtype="float32"
        )
        weight, idx = gate._topk_greedy(scores, k=2)
        self.assertEqual(idx.tolist(), [[1, 2]])
        self.assertTrue(
            paddle.allclose(
                weight,
                paddle.to_tensor([[0.40, 0.25]], dtype="float32"),
                atol=1e-6,
            )
        )

    def test_group_limited_greedy_excludes_unselected_group(self):
        """Oracle: 8 experts in 4 groups of 2, topk_group=2, k=3.

        group maxes = [0.95,0.50,0.92,0.91] -> groups {0,2} selected. Expert 6
        (score 0.91, group 3) would be a global top-3 pick but is masked out by
        group limiting, so selection = idx[0,4,1] with values [0.95,0.92,0.90].
        This proves group limiting changes the routing result.
        """
        paddle = self.paddle
        gate = self._make_gate(
            num_experts=8, topk_method="group_limited_greedy"
        )
        scores = paddle.to_tensor(
            [[0.95, 0.90, 0.50, 0.45, 0.92, 0.88, 0.91, 0.10]],
            dtype="float32",
        )
        weight, idx = gate._topk_group_limited_greedy(
            scores, k=3, n_group=4, topk_group=2
        )
        self.assertEqual(idx.tolist(), [[0, 4, 1]])
        self.assertNotIn(6, idx.tolist()[0])
        self.assertTrue(
            paddle.allclose(
                weight,
                paddle.to_tensor([[0.95, 0.92, 0.90]], dtype="float32"),
                atol=1e-6,
            )
        )

    def test_group_limited_greedy_requires_divisible_experts(self):
        """Contract: n_experts must be divisible by n_group (real assertion)."""
        paddle = self.paddle
        gate = self._make_gate(
            num_experts=8, topk_method="group_limited_greedy"
        )
        scores = paddle.to_tensor([[0.1, 0.2, 0.3]], dtype="float32")
        with self.assertRaises(AssertionError):
            gate._topk_group_limited_greedy(
                scores, k=1, n_group=2, topk_group=1
            )


class TestNoauxTcCorrectionBias(_MoEGateOracleBase):
    """Oracle: correction bias steers SELECTION but is not the final weight."""

    def test_bias_changes_selection_not_returned_weight(self):
        """Oracle: raw scores [0.30,0.35,0.60,0.55]; bias [1.0,0,0,0].

        With n_group=2, topk_group=2 (all groups kept) and k=2:
          * biased scores_for_choice = [1.30,0.35,0.60,0.55] -> top-2 idx[0,2]
            (expert 0 wins only because of the bias; un-biased top-2 is [2,3]).
          * In eval mode the returned weight is the *un-biased* score at the
            selected indices: [0.30, 0.60] -- NOT the biased [1.30, 0.60].
        Confirms e_score_correction_bias is a selection-time affinity shift,
        not the gating weight applied downstream.
        """
        paddle = self.paddle
        gate = self._make_gate(
            num_experts=4, topk_method="noaux_tc", n_group=2, topk_group=2
        )
        gate.e_score_correction_bias = paddle.to_tensor(
            [1.0, 0.0, 0.0, 0.0], dtype="float32"
        )
        gate.eval()  # not training -> weight taken from raw (un-biased) scores
        scores = paddle.to_tensor([[0.30, 0.35, 0.60, 0.55]], dtype="float32")
        weight, idx = gate._topk_noaux_tc(scores, k=2, n_group=2, topk_group=2)

        self.assertEqual(idx.tolist(), [[0, 2]])
        self.assertTrue(
            paddle.allclose(
                weight,
                paddle.to_tensor([[0.30, 0.60]], dtype="float32"),
                atol=1e-6,
            )
        )
        # Guard the specific antipattern: weight must not equal biased value.
        self.assertAlmostEqual(weight.tolist()[0][0], 0.30, places=6)
        self.assertNotAlmostEqual(weight.tolist()[0][0], 1.30, places=6)

    def test_noaux_tc_requires_correction_bias(self):
        """Contract: missing e_score_correction_bias raises (real assertion)."""
        paddle = self.paddle
        gate = self._make_gate(
            num_experts=4, topk_method="noaux_tc", n_group=2, topk_group=2
        )
        gate.e_score_correction_bias = None
        scores = paddle.to_tensor([[0.30, 0.35, 0.60, 0.55]], dtype="float32")
        with self.assertRaises(AssertionError):
            gate._topk_noaux_tc(scores, k=2, n_group=2, topk_group=2)


class TestTopkgating(_MoEGateOracleBase):
    """Oracle for the full topkgating() no-drop routing pipeline."""

    def _gates(self):
        # shape [batch=1, seq=2, experts=4]; reshaped to [2,4] inside the gate.
        return self.paddle.to_tensor(
            [[[0.10, 0.40, 0.25, 0.05], [0.50, 0.10, 0.15, 0.60]]],
            dtype="float32",
        )

    def test_selection_and_raw_weights_without_norm_or_scaling_applied(self):
        """Oracle (greedy, top_k=2, norm_topk_prob=False, scaling=3.0):

          token0 [0.10,0.40,0.25,0.05] -> idx[1,2], weights [0.40,0.25]
          token1 [0.50,0.10,0.15,0.60] -> idx[3,0], weights [0.60,0.50]
          all experts chosen once -> capacity == 1.

        NOTE: the returned combine weights equal the RAW masked gate values;
        routed_scaling_factor (3.0) is NOT applied to them in this method (see
        report). The oracle therefore expects the un-scaled values.
        """
        paddle = self.paddle
        gate = self._make_gate(
            topk_method="greedy",
            top_k=2,
            norm_topk_prob=False,
            routed_scaling_factor=3.0,
        )
        capacity, weights, top_idx, token_prio, l_aux, l_zloss = (
            gate.topkgating(self._gates())
        )
        self.assertEqual(top_idx.tolist(), [[1, 2], [3, 0]])
        self.assertTrue(
            paddle.allclose(
                weights,
                paddle.to_tensor([[0.40, 0.25], [0.60, 0.50]], dtype="float32"),
                atol=1e-6,
            )
        )
        self.assertEqual(capacity, 1)
        self.assertTrue(paddle.isfinite(l_aux).item())
        self.assertTrue(paddle.isfinite(l_zloss).item())

    def test_norm_topk_prob_renormalizes_selected_weights_to_sum_one(self):
        """Oracle (greedy, top_k=2, norm_topk_prob=True, scaling=1.0, training):

          token0 selected {1,2} raw [0.40,0.25] sum 0.65 -> [0.61538,0.38462]
          token1 selected {3,0} raw [0.60,0.50] sum 1.10 -> [0.54545,0.45455]
        Each token's returned weights sum to exactly 1 after normalization.
        """
        paddle = self.paddle
        gate = self._make_gate(
            topk_method="greedy",
            top_k=2,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        gate.train()
        _, weights, top_idx, _, _, _ = gate.topkgating(self._gates())
        self.assertEqual(top_idx.tolist(), [[1, 2], [3, 0]])
        expected = paddle.to_tensor(
            [[0.40 / 0.65, 0.25 / 0.65], [0.60 / 1.10, 0.50 / 1.10]],
            dtype="float32",
        )
        self.assertTrue(paddle.allclose(weights, expected, atol=1e-6))
        row_sums = weights.sum(axis=-1)
        self.assertTrue(
            paddle.allclose(
                row_sums, paddle.ones([2], dtype="float32"), atol=1e-6
            )
        )


class TestTopkgatingNodropScaling(_MoEGateOracleBase):
    """Oracle: topkgating_nodrop DOES apply routed scaling; sum(weights)!=1."""

    def test_routed_scaling_applied_and_breaks_sum_to_one(self):
        """Oracle (greedy, top_k=2, norm_topk_prob=False, scaling=2.0):

          gates_masked = (gates * top-k mask) * 2.0
          token0 -> [0, 0.80, 0.50, 0]  (selected sum 1.30, != 1)
          token1 -> [1.00, 0, 0, 1.20]
          mask   -> [[0,1,1,0],[1,0,0,1]]
        Demonstrates that with routed scaling enabled the weights need not sum
        to 1, and contrasts with topkgating() which leaves scaling unapplied.
        """
        paddle = self.paddle
        gate = self._make_gate(
            topk_method="greedy",
            top_k=2,
            norm_topk_prob=False,
            routed_scaling_factor=2.0,
        )
        gates = paddle.to_tensor(
            [[[0.10, 0.40, 0.25, 0.05], [0.50, 0.10, 0.15, 0.60]]],
            dtype="float32",
        )
        gates_masked, mask, exp_counts, l_aux, l_zloss = gate.topkgating_nodrop(
            gates
        )
        self.assertTrue(
            paddle.allclose(
                gates_masked,
                paddle.to_tensor(
                    [[0.0, 0.80, 0.50, 0.0], [1.00, 0.0, 0.0, 1.20]],
                    dtype="float32",
                ),
                atol=1e-6,
            )
        )
        self.assertEqual(mask.tolist(), [[0, 1, 1, 0], [1, 0, 0, 1]])
        self.assertEqual(exp_counts.tolist(), [1, 1, 1, 1])
        # token0 selected weights sum to 1.30 (scaled), not 1.0.
        self.assertAlmostEqual(
            float(gates_masked[0].sum().item()), 1.30, places=5
        )


if __name__ == "__main__":
    unittest.main()
