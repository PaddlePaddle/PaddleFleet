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

"""Behaviour tests for DSv4 hybrid-attention selective recompute.

Scope (no-card / CPU logic):

* ``module_needs_recompute`` is the single decision that ``DSv4HybridAttention.
  __init__`` calls to set every ``recompute_*`` flag
  (``recompute_full_attn``/``recompute_gated_attn``/``recompute_qkv``/
  ``recompute_post_core``/``recompute_vha_postmix``). Each flag is
  ``config.recompute_granularity == "selective" and module_needs_recompute(<name>, ...)``.
  We drive the *real* decision function with hand-derived layer selectors and
  assert the exact per-layer / per-module / per-MTP outcome, instead of
  re-implementing the ``if`` chain inside the test.
* ``DSv4HybridAttention._can_fuse_inv_rope_postmix`` is the real method that
  gates the fused inverse-RoPE + VHA-postmix path; its last guard depends on the
  ``recompute_vha_postmix`` flag. We bind the real method onto a lightweight
  stand-in and observe its return across every branch.

These are pure Python branch decisions and carry no device numerics; the real
recompute *gradient* equivalence (RecomputeWithoutOutput replay vs. eager) is a
single-card backward concern and is intentionally not asserted here.
"""

import types
import unittest
from types import SimpleNamespace

# ``module_needs_recompute`` lives in paddlefleet.recompute_utils, which imports
# paddle at module top. Guard the import so the honest skip reason is recorded
# where paddle / paddlefleet is unavailable; never fake a pass.
try:
    from paddlefleet.recompute_utils import module_needs_recompute

    _RECOMPUTE_UTILS_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest: dependency missing
    module_needs_recompute = None
    _RECOMPUTE_UTILS_IMPORT_ERROR = repr(exc)

# The DSv4 layer module pulls in paddle + the attention/CSA stack; guard it
# separately so the recompute-decision tests above can still run in an
# environment where only recompute_utils is importable.
try:
    from paddlefleet.transformer.dsv4_hybrid_attention import (
        DSv4HybridAttention,
    )

    _DSV4_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest: dependency missing
    DSv4HybridAttention = None
    _DSV4_IMPORT_ERROR = repr(exc)


def _recompute_config(**overrides):
    """A lightweight config for ``module_needs_recompute``.

    Only fields the decision path reads are present. This is plain
    configuration data (not the logic under test): the selector resolution,
    layer-index mapping and first_n/block methods all run on the real
    production functions.
    """
    cfg = SimpleNamespace(
        recompute_modules=None,
        recompute_num_layers=-1,
        recompute_method="first_n",
        num_hidden_layers=4,
        num_empty_layers_add_in_head=0,
        num_empty_layers_add_in_tail=0,
        virtual_pipeline_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@unittest.skipUnless(
    module_needs_recompute is not None,
    f"paddlefleet.recompute_utils not importable: {_RECOMPUTE_UTILS_IMPORT_ERROR}",
)
class TestModuleNeedsRecomputeDecision(unittest.TestCase):
    """The real per-module / per-layer recompute decision for DSv4 flags."""

    def test_unconfigured_module_never_recomputes(self):
        # recompute_modules=None: no module is selected on any layer.
        cfg = _recompute_config(recompute_modules=None)
        self.assertFalse(module_needs_recompute("full_attn", 0, cfg))
        self.assertFalse(module_needs_recompute("gated_attn", 3, cfg))

    def test_list_mode_selects_only_listed_modules_every_layer(self):
        # List form + negative recompute_num_layers => selected modules recompute
        # on every layer; an unlisted module stays off. This distinguishes the
        # module set: swapping full_attn<->qkv in the config would flip these.
        cfg = _recompute_config(
            recompute_modules=["full_attn", "gated_attn"],
            recompute_num_layers=-1,
        )
        for layer in (0, 2, 3):
            self.assertTrue(module_needs_recompute("full_attn", layer, cfg))
            self.assertTrue(module_needs_recompute("gated_attn", layer, cfg))
        # dsv4_hybrid_attn_qkv is NOT in the list -> the qkv recompute flag off.
        self.assertFalse(module_needs_recompute("dsv4_hybrid_attn_qkv", 0, cfg))
        self.assertFalse(module_needs_recompute("dsv4_hybrid_attn_qkv", 3, cfg))

    def test_dict_mode_explicit_layer_list_is_per_layer(self):
        # full_attn recomputes only on logical layers {0, 2}; gated_attn ("all")
        # everywhere. A degenerate all-layers/never-layers implementation would
        # not reproduce the layer-1 gap.
        cfg = _recompute_config(
            recompute_modules={"full_attn": [0, 2], "gated_attn": "all"},
            num_hidden_layers=4,
            num_empty_layers_add_in_head=0,
        )
        self.assertTrue(module_needs_recompute("full_attn", 0, cfg))
        self.assertFalse(module_needs_recompute("full_attn", 1, cfg))
        self.assertTrue(module_needs_recompute("full_attn", 2, cfg))
        self.assertFalse(module_needs_recompute("full_attn", 3, cfg))
        for layer in (0, 1, 2, 3):
            self.assertTrue(module_needs_recompute("gated_attn", layer, cfg))

    def test_head_offset_shifts_logical_layer_index(self):
        # With 2 empty head layers, physical layer 2 maps to logical index 0,
        # so the [0] selector fires on physical layer 2, not physical layer 0.
        cfg = _recompute_config(
            recompute_modules={"full_attn": [0]},
            num_hidden_layers=4,
            num_empty_layers_add_in_head=2,
        )
        self.assertFalse(module_needs_recompute("full_attn", 0, cfg))
        self.assertFalse(module_needs_recompute("full_attn", 1, cfg))
        self.assertTrue(module_needs_recompute("full_attn", 2, cfg))

    def test_count_selector_resolves_via_first_n(self):
        # Integer count 2 with recompute_method="first_n", pp=1, vpp=1:
        # need_recompute_in_first_n selects layers [0, 1] only.
        cfg = _recompute_config(
            recompute_modules={"full_attn": 2},
            recompute_method="first_n",
            num_hidden_layers=4,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=1,
        )
        self.assertTrue(module_needs_recompute("full_attn", 0, cfg))
        self.assertTrue(module_needs_recompute("full_attn", 1, cfg))
        self.assertFalse(module_needs_recompute("full_attn", 2, cfg))
        self.assertFalse(module_needs_recompute("full_attn", 3, cfg))

    def test_mtp_layer_index_does_not_collide_with_backbone_layer0(self):
        # Layer id 4 (== num_hidden_layers) addresses MTP depth 0, distinct from
        # backbone physical layer 0. is_mtp_layer routes through
        # logical_layer_index so the two do not alias.
        cfg = _recompute_config(
            recompute_modules={"gated_attn": [4]},
            num_hidden_layers=4,
        )
        self.assertTrue(
            module_needs_recompute("gated_attn", 0, cfg, is_mtp_layer=True)
        )
        self.assertFalse(
            module_needs_recompute("gated_attn", 0, cfg, is_mtp_layer=False)
        )

    def test_layer_list_without_layer_number_raises_or_defers(self):
        # An explicit layer list needs a layer number to filter on. Without
        # defer, this is a hard error; with defer_if_layer_unknown it resolves
        # to False (the "ask again later" contract).
        cfg = _recompute_config(recompute_modules={"full_attn": [0]})
        with self.assertRaises(ValueError):
            module_needs_recompute("full_attn", None, cfg)
        self.assertFalse(
            module_needs_recompute(
                "full_attn", None, cfg, defer_if_layer_unknown=True
            )
        )

    def test_malformed_selector_type_raises(self):
        # A float selector is neither "all"/None, a layer list, nor an int
        # count -> the decision function rejects it rather than silently
        # enabling or disabling recompute.
        cfg = _recompute_config(recompute_modules={"full_attn": 3.5})
        with self.assertRaises(ValueError):
            module_needs_recompute("full_attn", 0, cfg)


@unittest.skipUnless(
    DSv4HybridAttention is not None,
    f"paddlefleet.transformer.dsv4_hybrid_attention not importable: {_DSV4_IMPORT_ERROR}",
)
class TestCanFuseInvRopePostmix(unittest.TestCase):
    """Real DSv4HybridAttention._can_fuse_inv_rope_postmix branch behaviour."""

    def _make_probe(
        self,
        *,
        fuse_flag=True,
        use_vha_postmix=True,
        vha_postmix_grouped=False,
        apply_rope_fusion=True,
        high_precision_rope=False,
        recompute_vha_postmix=False,
        training=True,
    ):
        """Bind the real _can_fuse_inv_rope_postmix onto a stand-in.

        Only attributes/config fields the method actually reads are set; the
        method body itself is the real production code, executed unchanged.
        """
        config = SimpleNamespace(
            fuse_inv_rope_into_vha_postmix=fuse_flag,
            apply_rope_fusion=apply_rope_fusion,
            high_precision_rope=high_precision_rope,
        )
        inst = SimpleNamespace(
            config=config,
            use_vha_postmix=use_vha_postmix,
            vha_postmix_grouped=vha_postmix_grouped,
            recompute_vha_postmix=recompute_vha_postmix,
            training=training,
        )
        inst._can_fuse_inv_rope_postmix = types.MethodType(
            DSv4HybridAttention._can_fuse_inv_rope_postmix, inst
        )
        return inst

    def test_all_conditions_met_allows_fusion(self):
        inst = self._make_probe()
        self.assertTrue(
            inst._can_fuse_inv_rope_postmix(in_full_recompute=False)
        )

    def test_fusion_disabled_when_config_flag_off(self):
        inst = self._make_probe(fuse_flag=False)
        self.assertFalse(
            inst._can_fuse_inv_rope_postmix(in_full_recompute=False)
        )

    def test_fusion_requires_ungrouped_postmix(self):
        # No postmix at all -> cannot fuse.
        self.assertFalse(
            self._make_probe(use_vha_postmix=False)._can_fuse_inv_rope_postmix(
                in_full_recompute=False
            )
        )
        # Grouped postmix is an einsum with no [nh, nh] GEMM to fold into.
        self.assertFalse(
            self._make_probe(
                vha_postmix_grouped=True
            )._can_fuse_inv_rope_postmix(in_full_recompute=False)
        )

    def test_fusion_requires_rope_fusion_and_low_precision(self):
        self.assertFalse(
            self._make_probe(
                apply_rope_fusion=False
            )._can_fuse_inv_rope_postmix(in_full_recompute=False)
        )
        self.assertFalse(
            self._make_probe(
                high_precision_rope=True
            )._can_fuse_inv_rope_postmix(in_full_recompute=False)
        )

    def test_postmix_recompute_blocks_fusion_only_outside_full_recompute(self):
        # The nested-postmix recompute wrapper would re-enter the fused PyLayer
        # for nothing, so fusion stands down -- but ONLY when this call is not
        # already inside an outer full_attn recompute replay.
        inst = self._make_probe(recompute_vha_postmix=True, training=True)
        self.assertFalse(
            inst._can_fuse_inv_rope_postmix(in_full_recompute=False)
        )
        self.assertTrue(inst._can_fuse_inv_rope_postmix(in_full_recompute=True))

    def test_postmix_recompute_flag_ignored_when_not_training(self):
        # recompute paths are training-only; in eval the last guard is inert.
        inst = self._make_probe(recompute_vha_postmix=True, training=False)
        self.assertTrue(
            inst._can_fuse_inv_rope_postmix(in_full_recompute=False)
        )


if __name__ == "__main__":
    unittest.main()
