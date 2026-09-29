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

"""Behavior tests for paddlefleet.models.backends.LocalSpecProvider.

This module is a backend "spec provider": each method selects *which* concrete
layer class (or spec) the local backend wires into a model. The contract under
test is therefore the routing decision itself -- the exact class object each
method returns and the way grouped-MLP wiring is assembled. Asserting the exact
class identity (and cross-checking that distinct slots resolve to distinct
classes) catches the realistic failure mode for such selectors: returning the
wrong-but-plausible neighbour (e.g. a row-parallel linear where a
column-parallel one is required).

The module imports paddle-backed layer classes at import time, so the whole
suite is skipped with an honest reason when paddle / paddlefleet is not
importable. The selection logic itself is pure Python and is exercised for
real; no method under test is mocked.
"""

import unittest

try:
    import paddlefleet.models.backends as backends_mod
    from paddlefleet.models.backends import (
        GroupedMLP,
        LocalSpecProvider,
        SequentialMLP,
    )
    from paddlefleet.tensor_parallel.layers import (
        ColumnParallelLinear,
        Linear,
        RowParallelLinear,
    )
    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
    )
    from paddlefleet.transformer.mlp import MLPSublayersSpec
    from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuine missing-dependency signal
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.models.backends not importable (missing paddle?): "
    f"{_IMPORT_ERROR}",
)
class TestLocalSpecProviderSelection(unittest.TestCase):
    """Each selector must resolve to the exact class the local backend wires."""

    def setUp(self):
        self.provider = LocalSpecProvider()

    def test_linear_selects_plain_linear(self):
        # The non-parallel linear slot must be the plain Linear, and must not
        # be confused with either parallel variant.
        selected = self.provider.linear()
        self.assertIs(selected, Linear)
        self.assertIsNot(selected, ColumnParallelLinear)
        self.assertIsNot(selected, RowParallelLinear)

    def test_column_parallel_linear_selects_column_class(self):
        selected = self.provider.column_parallel_linear()
        self.assertIs(selected, ColumnParallelLinear)
        # Guard against a column/row swap -- the two are distinct classes.
        self.assertIsNot(selected, RowParallelLinear)

    def test_row_parallel_linear_selects_row_class(self):
        selected = self.provider.row_parallel_linear()
        self.assertIs(selected, RowParallelLinear)
        self.assertIsNot(selected, ColumnParallelLinear)

    def test_column_and_row_parallel_linears_are_distinct(self):
        # A single wrong return that aliased both slots to one class would pass
        # each individual selector test above only if they returned the same
        # object; pin the invariant that the two slots differ.
        self.assertIsNot(
            self.provider.column_parallel_linear(),
            self.provider.row_parallel_linear(),
        )

    def test_fuse_layernorm_and_linear_is_false(self):
        result = self.provider.fuse_layernorm_and_linear()
        # Must be the boolean False, not just any falsy value (None, 0, "").
        self.assertIs(result, False)

    def test_column_parallel_layer_norm_linear_is_none(self):
        # The local backend has no fused layernorm+linear layer.
        self.assertIsNone(self.provider.column_parallel_layer_norm_linear())

    def test_core_attention_selects_dot_product_attention(self):
        self.assertIs(self.provider.core_attention(), DotProductAttention)

    def test_hidden_act_returns_none(self):
        # Local backend defers activation selection; hidden_act yields None.
        self.assertIsNone(self.provider.hidden_act())


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.models.backends not importable (missing paddle?): "
    f"{_IMPORT_ERROR}",
)
class TestLocalSpecProviderLayerNorm(unittest.TestCase):
    """layer_norm returns WrappedPaddleNorm and, only for rms_norm, resets the
    module-level LNImpl global."""

    def setUp(self):
        self.provider = LocalSpecProvider()
        # layer_norm(rms_norm=True) mutates the module global LNImpl; snapshot
        # and restore it so this test cannot leak state to other tests.
        self._orig_lnimpl = backends_mod.LNImpl
        self.addCleanup(setattr, backends_mod, "LNImpl", self._orig_lnimpl)

    def test_layer_norm_non_rms_returns_current_global(self):
        # The non-rms branch returns the module global as-is, WITHOUT resetting
        # it. Install a sentinel as the global and confirm it is returned
        # unchanged -- proving this branch does not force WrappedPaddleNorm.
        sentinel = object()
        backends_mod.LNImpl = sentinel
        result = self.provider.layer_norm(rms_norm=False)
        self.assertIs(result, sentinel)
        # Global must be untouched by the non-rms path.
        self.assertIs(backends_mod.LNImpl, sentinel)

    def test_layer_norm_rms_resets_global_to_wrapped_paddle_norm(self):
        # The rms branch reassigns the global to WrappedPaddleNorm and returns
        # it, regardless of the prior global value. Start from a sentinel to
        # prove the reset actually happens.
        backends_mod.LNImpl = object()
        result = self.provider.layer_norm(rms_norm=True)
        self.assertIs(result, WrappedPaddleNorm)
        self.assertIs(backends_mod.LNImpl, WrappedPaddleNorm)

    def test_layer_norm_default_and_for_qk_do_not_alter_selection(self):
        # for_qk is accepted but must not change the returned class; with the
        # global at its default WrappedPaddleNorm, every combination resolves
        # to WrappedPaddleNorm.
        backends_mod.LNImpl = WrappedPaddleNorm
        self.assertIs(self.provider.layer_norm(), WrappedPaddleNorm)
        self.assertIs(self.provider.layer_norm(for_qk=True), WrappedPaddleNorm)
        self.assertIs(
            self.provider.layer_norm(rms_norm=True, for_qk=True),
            WrappedPaddleNorm,
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.models.backends not importable (missing paddle?): "
    f"{_IMPORT_ERROR}",
)
class TestLocalSpecProviderGroupedMLP(unittest.TestCase):
    """grouped_mlp_layers routes on moe_use_grouped_gemm and assembles the
    sequential-MLP spec with the correct up/down projection wiring."""

    def setUp(self):
        self.provider = LocalSpecProvider()

    def test_grouped_gemm_returns_grouped_mlp_and_no_spec(self):
        layer, spec = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=True, moe_use_legacy_grouped_gemm=False
        )
        self.assertIs(layer, GroupedMLP)
        # Grouped GEMM path carries no sublayer spec.
        self.assertIsNone(spec)

    def test_sequential_path_wires_column_up_and_row_down(self):
        layer, spec = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=False, moe_use_legacy_grouped_gemm=False
        )
        self.assertIs(layer, SequentialMLP)
        self.assertIsInstance(spec, MLPSublayersSpec)
        # The exact projection wiring is the contract: up-gate must be the
        # column-parallel linear and down must be the row-parallel linear.
        # A swap here would still yield an MLPSublayersSpec, so pin identities.
        self.assertIs(spec.up_gate_proj, ColumnParallelLinear)
        self.assertIs(spec.down_proj, RowParallelLinear)
        self.assertIsNot(spec.up_gate_proj, spec.down_proj)
        # hidden_act is left unset (dataclass default None) on this path.
        self.assertIsNone(spec.hidden_act)

    def test_legacy_flag_is_currently_ignored(self):
        # Documented current behavior: only moe_use_grouped_gemm drives the
        # decision; moe_use_legacy_grouped_gemm is not consumed, so toggling it
        # leaves the selected layer unchanged for both branches.
        grouped_a, spec_a = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=True, moe_use_legacy_grouped_gemm=False
        )
        grouped_b, spec_b = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=True, moe_use_legacy_grouped_gemm=True
        )
        self.assertIs(grouped_a, grouped_b)
        self.assertIsNone(spec_a)
        self.assertIsNone(spec_b)

        seq_a, _ = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=False, moe_use_legacy_grouped_gemm=False
        )
        seq_b, _ = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=False, moe_use_legacy_grouped_gemm=True
        )
        self.assertIs(seq_a, seq_b)
        self.assertIs(seq_a, SequentialMLP)

    def test_grouped_and_sequential_select_different_layers(self):
        # The two branches must not collapse to the same layer class.
        grouped_layer, _ = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=True, moe_use_legacy_grouped_gemm=False
        )
        seq_layer, _ = self.provider.grouped_mlp_layers(
            moe_use_grouped_gemm=False, moe_use_legacy_grouped_gemm=False
        )
        self.assertIsNot(grouped_layer, seq_layer)


if __name__ == "__main__":
    unittest.main()
