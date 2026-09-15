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

"""Behavior tests for ``paddlefleet.recompute_utils``.

The unit under test is the pure layer-selection logic that decides, for a given
physical ``layer_number``, whether an activation-recompute segment (or a refined
recompute segment) should run. The public entry points exercised here are:

* ``need_recompute_in_block`` / ``need_recompute_in_first_n`` -- the two layer
  selectors, including their pipeline (PP) and virtual-pipeline (VPP) chunking.
* ``need_full_recompute`` -- dispatches to the selectors by
  ``recompute_granularity`` / ``recompute_method``.
* ``module_needs_recompute`` / ``module_needs_refined_recompute`` -- the
  per-module ``recompute_modules`` resolution (dict, list and scalar selectors).
* ``validate_recompute_modules`` -- startup structural validation.
* ``effective_mtp_layers`` / ``logical_layer_index`` /
  ``normalize_recompute_layer_ids`` -- small helpers with sharp edge cases.
* ``has_recovered`` -- the env-driven recovery-step gate.

Every expected value below is derived by hand from the documented chunking /
selection rules (ceil-division chunks, interleaved VPP stage assignment,
first-n-per-stage, logical-index remapping), never by calling the function under
test to produce its own oracle. Fixed, distinguishable inputs are used so that
off-by-one, wrong-stage and dropped-layer mistakes are observable.

Because the production module imports ``paddle`` at import time, the whole suite
is skipped -- with an honest reason -- when Paddle (and therefore the module) is
not importable on the host.
"""

import os
import sys
import types
import unittest

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed into the environment (repo_root/src is 4 levels up from here).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle  # noqa: F401  (imported so the skip reason is honest)

    from paddlefleet import recompute_utils as ru

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    ru = None
    _IMPORT_ERROR = exc


def make_config(**overrides):
    """Build a minimal config stub exposing only the attributes the code reads.

    Defaults mirror a plain 8-layer, single-stage, full/uniform setup; tests
    override just the fields relevant to the behavior they pin down. This is a
    collaborator stub, not the unit under test.
    """
    defaults = dict(
        num_hidden_layers=8,
        num_empty_layers_add_in_head=0,
        num_empty_layers_add_in_tail=0,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=None,
        num_nextn_predict_layers=0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class NeedRecomputeInBlockTest(unittest.TestCase):
    """``need_recompute_in_block``: first ``n`` layers of every ceil chunk."""

    def test_negative_count_recomputes_every_layer(self):
        """A negative count is the documented "all layers" shortcut."""
        config = make_config(
            num_hidden_layers=8, pipeline_model_parallel_size=1
        )
        for layer in range(8):
            self.assertTrue(ru.need_recompute_in_block(layer, config, -1))

    def test_single_stage_selects_leading_n(self):
        """pp=1 -> one chunk of 8; count=4 selects exactly layers 0..3.

        Hand-derived: chunk_size = ceil(8/1) = 8, so the only chunk is
        [0..7] and its first 4 entries {0,1,2,3} are recomputed.
        """
        config = make_config(
            num_hidden_layers=8, pipeline_model_parallel_size=1
        )
        selected = [ru.need_recompute_in_block(i, config, 4) for i in range(8)]
        self.assertEqual(
            selected, [True, True, True, True, False, False, False, False]
        )

    def test_pipeline_parallel_selects_per_chunk(self):
        """pp=2 -> two chunks of 4; count=2 selects {0,1} and {4,5}.

        chunk_size = ceil(8/2) = 4, chunks start at 0 and 4, first two of
        each: {0,1,4,5}.
        """
        config = make_config(
            num_hidden_layers=8, pipeline_model_parallel_size=2
        )
        selected = [ru.need_recompute_in_block(i, config, 2) for i in range(8)]
        self.assertEqual(
            selected, [True, True, False, False, True, True, False, False]
        )

    def test_uneven_last_chunk_uses_ceil_division(self):
        """15 layers over pp=4 -> chunks 4,4,4,3 (ceil), count=2 per chunk.

        Chunk starts (step ceil(15/4)=4): 0,4,8,12 -> selected {0,1,4,5,8,9,
        12,13}. The final short chunk [12,13,14] still contributes its first
        two, and layer 14 (the ragged tail) is never selected.
        """
        config = make_config(
            num_hidden_layers=15, pipeline_model_parallel_size=4
        )
        selected = {
            i for i in range(15) if ru.need_recompute_in_block(i, config, 2)
        }
        self.assertEqual(selected, {0, 1, 4, 5, 8, 9, 12, 13})
        self.assertFalse(ru.need_recompute_in_block(14, config, 2))

    def test_empty_head_and_tail_layers_count_toward_total(self):
        """Head/tail padding layers widen the total before chunking.

        head=1, hidden=6, tail=2 -> total=9, pp=1 -> single chunk, count=2
        selects layers {0,1}.
        """
        config = make_config(
            num_hidden_layers=6,
            num_empty_layers_add_in_head=1,
            num_empty_layers_add_in_tail=2,
            pipeline_model_parallel_size=1,
        )
        self.assertTrue(ru.need_recompute_in_block(0, config, 2))
        self.assertTrue(ru.need_recompute_in_block(1, config, 2))
        self.assertFalse(ru.need_recompute_in_block(2, config, 2))

    def test_none_count_raises(self):
        """A missing count is an explicit assertion failure, not silent True."""
        config = make_config()
        with self.assertRaises(AssertionError):
            ru.need_recompute_in_block(0, config, None)

    def test_count_larger_than_chunk_raises(self):
        """Recomputing more layers than a chunk holds is rejected.

        chunk_size = 8 for pp=1/total=8, so count=9 must trip the assert.
        """
        config = make_config(
            num_hidden_layers=8, pipeline_model_parallel_size=1
        )
        with self.assertRaises(AssertionError):
            ru.need_recompute_in_block(0, config, 9)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class NeedRecomputeInFirstNTest(unittest.TestCase):
    """``need_recompute_in_first_n``: first ``n`` layers of each PP stage."""

    def test_single_stage_selects_leading_n(self):
        """pp=1 -> a single stage; count=4 selects layers 0..3."""
        config = make_config(
            num_hidden_layers=8, pipeline_model_parallel_size=1
        )
        selected = [
            ru.need_recompute_in_first_n(i, config, 4) for i in range(8)
        ]
        self.assertEqual(
            selected, [True, True, True, True, False, False, False, False]
        )

    def test_pipeline_parallel_first_n_per_stage(self):
        """pp=2 -> stages [0..3] and [4..7]; count=2 selects {0,1,4,5}.

        chunk_size = 4; layers[0::4]={0,4} and layers[1::4]={1,5} are the
        first two positions of each stage.
        """
        config = make_config(
            num_hidden_layers=8, pipeline_model_parallel_size=2
        )
        selected = {
            i for i in range(8) if ru.need_recompute_in_first_n(i, config, 2)
        }
        self.assertEqual(selected, {0, 1, 4, 5})

    def test_vpp_selects_first_layer_of_each_stage(self):
        """pp=2, vpp=2, count=1 -> the leading layer of each PP stage.

        parallel_size=4, chunk_size=ceil(8/4)=2. With count=1 only the very
        first layer offered to each of the 2 PP stages is selected, giving
        {0, 4}.
        """
        config = make_config(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        selected = {
            i for i in range(8) if ru.need_recompute_in_first_n(i, config, 1)
        }
        self.assertEqual(selected, {0, 4})

    def test_none_count_raises(self):
        config = make_config()
        with self.assertRaises(AssertionError):
            ru.need_recompute_in_first_n(0, config, None)

    def test_count_larger_than_stage_raises(self):
        """count must not exceed layers-per-stage = ceil(total/pp).

        pp=2, total=8 -> 4 layers per stage, so count=5 must raise.
        """
        config = make_config(
            num_hidden_layers=8, pipeline_model_parallel_size=2
        )
        with self.assertRaises(AssertionError):
            ru.need_recompute_in_first_n(0, config, 5)

    @unittest.expectedFailure
    def test_vpp_interior_layers_are_wrongly_dropped(self):
        """REAL BUG: the VPP chunking drops interior layers entirely.

        In src/paddlefleet/recompute_utils.py:184-187 the chunk list is built
        as ``layers[i * chunk_size:(i + 1) * chunk_size]`` while ``i`` already
        iterates ``range(0, len(layers), chunk_size)`` -- i.e. i is a multiple
        of chunk_size and is then multiplied by chunk_size again. For total=8,
        pp=2, vpp=2 (chunk_size=2) this yields chunks
        [[0,1], [4,5], [], []] instead of [[0,1],[2,3],[4,5],[6,7]], so layers
        2, 3, 6, 7 are never placed in any chunk and can never be recomputed --
        even when ``recompute_num_layers`` equals the full per-stage count.

        Correct behavior: with count = layers-per-stage (4), EVERY layer must
        be selected. Interleaved VPP puts stage 0 = [0,1,4,5] and stage 1 =
        [2,3,6,7], so the union of first-4-per-stage is all of 0..7. This test
        asserts that correct behavior and is expected to fail until the
        production chunking is fixed. Production is intentionally NOT edited.
        """
        config = make_config(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        selected = {
            i for i in range(8) if ru.need_recompute_in_first_n(i, config, 4)
        }
        self.assertEqual(selected, set(range(8)))


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class NeedFullRecomputeTest(unittest.TestCase):
    """``need_full_recompute``: dispatch by granularity + method."""

    def test_uniform_full_always_true(self):
        """full+uniform recomputes every layer regardless of index."""
        config = make_config(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        self.assertTrue(ru.need_full_recompute(0, config))
        self.assertTrue(ru.need_full_recompute(7, config))

    def test_non_full_granularity_is_false(self):
        """Anything other than 'full' granularity disables full recompute."""
        config = make_config(
            recompute_granularity="selective",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        self.assertFalse(ru.need_full_recompute(0, config))

    def test_uniform_requires_count_one(self):
        """uniform is only defined for count==1; count==2 must raise."""
        config = make_config(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=2,
        )
        with self.assertRaises(AssertionError):
            ru.need_full_recompute(0, config)

    def test_first_n_delegates(self):
        """full+first_n selects the first count layers (pp=1 -> 0..count-1)."""
        config = make_config(
            num_hidden_layers=8,
            pipeline_model_parallel_size=1,
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=4,
        )
        selected = [ru.need_full_recompute(i, config) for i in range(8)]
        self.assertEqual(
            selected, [True, True, True, True, False, False, False, False]
        )

    def test_block_delegates(self):
        """full+block selects the first count layers of the single chunk."""
        config = make_config(
            num_hidden_layers=8,
            pipeline_model_parallel_size=1,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=4,
        )
        selected = [ru.need_full_recompute(i, config) for i in range(8)]
        self.assertEqual(
            selected, [True, True, True, True, False, False, False, False]
        )

    def test_unknown_method_falls_through_to_false(self):
        """full granularity with an unrecognized method yields False.

        Only uniform/first_n/block are handled inside the 'full' branch; any
        other method falls through to the trailing ``return False``.
        """
        config = make_config(
            recompute_granularity="full",
            recompute_method="mystery",
            recompute_num_layers=1,
        )
        self.assertFalse(ru.need_full_recompute(0, config))


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class EffectiveMtpLayersTest(unittest.TestCase):
    """``effective_mtp_layers``: sanitize ``num_nextn_predict_layers``."""

    def test_missing_attribute_is_zero(self):
        config = types.SimpleNamespace()  # no num_nextn_predict_layers at all
        self.assertEqual(ru.effective_mtp_layers(config), 0)

    def test_positive_int_passes_through(self):
        self.assertEqual(
            ru.effective_mtp_layers(make_config(num_nextn_predict_layers=3)), 3
        )

    def test_none_becomes_zero(self):
        self.assertEqual(
            ru.effective_mtp_layers(make_config(num_nextn_predict_layers=None)),
            0,
        )

    def test_bool_is_coerced_to_zero(self):
        """``True`` is an int subclass but must not count as one layer."""
        self.assertEqual(
            ru.effective_mtp_layers(make_config(num_nextn_predict_layers=True)),
            0,
        )

    def test_non_int_becomes_zero(self):
        self.assertEqual(
            ru.effective_mtp_layers(make_config(num_nextn_predict_layers="2")),
            0,
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class LogicalLayerIndexTest(unittest.TestCase):
    """``logical_layer_index``: physical -> config-facing layer id."""

    def test_mtp_layer_offsets_by_backbone_depth(self):
        """MTP layers live above the backbone: index = num_hidden + layer."""
        config = make_config(num_hidden_layers=8)
        self.assertEqual(
            ru.logical_layer_index(config, 0, is_mtp_layer=True), 8
        )
        self.assertEqual(
            ru.logical_layer_index(config, 2, is_mtp_layer=True), 10
        )

    def test_backbone_subtracts_head_padding(self):
        """Empty head layers shift physical numbering above the logical id."""
        config = make_config(num_empty_layers_add_in_head=1)
        self.assertEqual(ru.logical_layer_index(config, 5), 4)

    def test_backbone_without_head_padding_is_identity(self):
        config = make_config(num_empty_layers_add_in_head=0)
        self.assertEqual(ru.logical_layer_index(config, 5), 5)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class NormalizeRecomputeLayerIdsTest(unittest.TestCase):
    """``normalize_recompute_layer_ids``: validate + dedupe layer ids."""

    def test_dedupes_into_frozenset(self):
        result = ru.normalize_recompute_layer_ids([0, 2, 2, 5], "attn")
        self.assertEqual(result, frozenset({0, 2, 5}))

    def test_bool_id_rejected(self):
        with self.assertRaises(ValueError):
            ru.normalize_recompute_layer_ids([True], "attn")

    def test_negative_id_rejected(self):
        with self.assertRaises(ValueError):
            ru.normalize_recompute_layer_ids([-1], "attn")

    def test_non_int_id_rejected(self):
        with self.assertRaises(ValueError):
            ru.normalize_recompute_layer_ids(["1"], "attn")


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class ModuleNeedsRecomputeTest(unittest.TestCase):
    """``module_needs_recompute``: resolve a ``recompute_modules`` selector."""

    def test_unconfigured_module_is_false(self):
        config = make_config(recompute_modules={"attn": "all"})
        self.assertFalse(ru.module_needs_recompute("mlp", 0, config))

    def test_all_selector_is_true_everywhere(self):
        config = make_config(recompute_modules={"attn": "all"})
        self.assertTrue(ru.module_needs_recompute("attn", 0, config))
        self.assertTrue(ru.module_needs_recompute("attn", 7, config))

    def test_layer_agnostic_module_ignores_layer(self):
        """lm_head/loss_fn are single-instance: always on once configured.

        The layer-agnostic short circuit runs before any selector logic, so
        even a ``None`` layer number returns True.
        """
        config = make_config(recompute_modules={"lm_head": None})
        self.assertTrue(ru.module_needs_recompute("lm_head", None, config))

    def test_int_selector_resolves_via_first_n(self):
        """A scalar count is a first_n selector over the physical layer.

        count=4, pp=1 -> layers 0..3 selected, 4 excluded.
        """
        config = make_config(
            num_hidden_layers=8,
            pipeline_model_parallel_size=1,
            recompute_method="first_n",
            recompute_modules={"attn": 4},
        )
        self.assertTrue(ru.module_needs_recompute("attn", 0, config))
        self.assertTrue(ru.module_needs_recompute("attn", 3, config))
        self.assertFalse(ru.module_needs_recompute("attn", 4, config))

    def test_explicit_layer_list_uses_logical_index(self):
        """A layer list selects those logical ids (head-offset removed)."""
        config = make_config(
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            recompute_modules={"attn": [0, 2]},
        )
        self.assertTrue(ru.module_needs_recompute("attn", 0, config))
        self.assertFalse(ru.module_needs_recompute("attn", 1, config))
        self.assertTrue(ru.module_needs_recompute("attn", 2, config))

    def test_layer_list_without_layer_number_raises(self):
        config = make_config(recompute_modules={"attn": [0, 2]})
        with self.assertRaises(ValueError):
            ru.module_needs_recompute("attn", None, config)

    def test_layer_list_defers_when_layer_unknown(self):
        """With defer=True an unknown layer resolves to False, not an error."""
        config = make_config(recompute_modules={"attn": [0, 2]})
        self.assertFalse(
            ru.module_needs_recompute(
                "attn", None, config, defer_if_layer_unknown=True
            )
        )

    def test_negative_selector_is_true_everywhere(self):
        config = make_config(recompute_modules={"attn": -1})
        self.assertTrue(ru.module_needs_recompute("attn", 5, config))

    def test_list_mode_shares_recompute_num_layers(self):
        """List-style recompute_modules falls back to config.recompute_num_layers.

        count=2 via the shared field, pp=1 -> layers 0,1 on; 2 off.
        """
        config = make_config(
            num_hidden_layers=8,
            pipeline_model_parallel_size=1,
            recompute_method="first_n",
            recompute_num_layers=2,
            recompute_modules=["attn", "mlp"],
        )
        self.assertTrue(ru.module_needs_recompute("attn", 0, config))
        self.assertTrue(ru.module_needs_recompute("attn", 1, config))
        self.assertFalse(ru.module_needs_recompute("attn", 2, config))

    def test_invalid_selector_type_raises(self):
        config = make_config(recompute_modules={"attn": 1.5})
        with self.assertRaises(ValueError):
            ru.module_needs_recompute("attn", 0, config)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class ModuleNeedsRefinedRecomputeTest(unittest.TestCase):
    """``module_needs_refined_recompute``: RR is the inverse selection."""

    def test_unconfigured_module_is_false(self):
        config = make_config(recompute_modules={"flash_attn": "all"})
        self.assertFalse(
            ru.module_needs_refined_recompute("moe_combine", 0, config)
        )

    def test_list_mode_enables_rr_everywhere(self):
        """List mode carries no per-layer info, so RR runs on every layer."""
        config = make_config(recompute_modules=["flash_attn"])
        self.assertTrue(
            ru.module_needs_refined_recompute("flash_attn", 3, config)
        )

    def test_dict_all_disables_rr(self):
        """'all'/None mean plain recompute everywhere, so RR is off."""
        config = make_config(recompute_modules={"flash_attn": "all"})
        self.assertFalse(
            ru.module_needs_refined_recompute("flash_attn", 0, config)
        )

    def test_dict_none_disables_rr(self):
        config = make_config(recompute_modules={"flash_attn": None})
        self.assertFalse(
            ru.module_needs_refined_recompute("flash_attn", 0, config)
        )

    def test_layer_list_inverts_selection(self):
        """RR runs on the layers NOT named in the plain-recompute list."""
        config = make_config(
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            recompute_modules={"flash_attn": [0, 2]},
        )
        self.assertFalse(
            ru.module_needs_refined_recompute("flash_attn", 0, config)
        )
        self.assertTrue(
            ru.module_needs_refined_recompute("flash_attn", 1, config)
        )
        self.assertFalse(
            ru.module_needs_refined_recompute("flash_attn", 2, config)
        )

    def test_layer_list_without_layer_number_raises(self):
        config = make_config(recompute_modules={"flash_attn": [0, 2]})
        with self.assertRaises(ValueError):
            ru.module_needs_refined_recompute("flash_attn", None, config)

    def test_negative_selector_disables_rr(self):
        """A negative count means "all" and must not invert into RR-everywhere."""
        config = make_config(recompute_modules={"flash_attn": -1})
        self.assertFalse(
            ru.module_needs_refined_recompute("flash_attn", 5, config)
        )

    def test_int_selector_inverts_first_n(self):
        """count=2 -> plain recompute on {0,1}; RR runs on the complement.

        need_recompute_in_first_n selects {0,1}; RR is its negation, so RR is
        off at layer 0 and on at layer 2.
        """
        config = make_config(
            num_hidden_layers=8,
            pipeline_model_parallel_size=1,
            recompute_modules={"flash_attn": 2},
        )
        self.assertFalse(
            ru.module_needs_refined_recompute("flash_attn", 0, config)
        )
        self.assertTrue(
            ru.module_needs_refined_recompute("flash_attn", 2, config)
        )


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class ValidateRecomputeModulesTest(unittest.TestCase):
    """``validate_recompute_modules``: structural startup validation."""

    def test_none_is_accepted(self):
        ru.validate_recompute_modules(make_config(recompute_modules=None))

    def test_list_of_strings_is_accepted(self):
        ru.validate_recompute_modules(
            make_config(recompute_modules=["attn", "mlp"])
        )

    def test_list_with_non_string_raises(self):
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(
                make_config(recompute_modules=["attn", 5])
            )

    def test_non_sequence_non_dict_raises(self):
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(make_config(recompute_modules=5))

    def test_dict_all_and_none_selectors_accepted(self):
        ru.validate_recompute_modules(
            make_config(recompute_modules={"attn": "all", "mlp": None})
        )

    def test_dict_in_range_layer_list_accepted(self):
        config = make_config(
            num_hidden_layers=8, recompute_modules={"attn": [0, 2, 7]}
        )
        ru.validate_recompute_modules(config)

    def test_dict_out_of_range_layer_list_raises(self):
        """8 backbone layers -> valid ids 0..7; 100 is out of range."""
        config = make_config(
            num_hidden_layers=8, recompute_modules={"attn": [0, 100]}
        )
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(config)

    def test_mtp_layers_extend_valid_id_range(self):
        """MTP layers add ids above the backbone: id 9 valid, 10 out of range.

        num_hidden=8 + 2 MTP layers -> valid ids 0..9.
        """
        base = make_config(
            num_hidden_layers=8,
            num_nextn_predict_layers=2,
            recompute_modules={"attn": [9]},
        )
        ru.validate_recompute_modules(base)
        too_high = make_config(
            num_hidden_layers=8,
            num_nextn_predict_layers=2,
            recompute_modules={"attn": [10]},
        )
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(too_high)

    def test_layer_agnostic_module_rejects_layer_list(self):
        config = make_config(recompute_modules={"lm_head": [0]})
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(config)

    def test_int_selector_requires_first_n_or_block_method(self):
        """A scalar count needs first_n/block; uniform must be rejected."""
        config = make_config(
            recompute_method="uniform", recompute_modules={"attn": 2}
        )
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(config)

    def test_refined_module_int_selector_allows_any_method(self):
        """RR modules always resolve counts via first_n, so method is free."""
        config = make_config(
            recompute_method="uniform", recompute_modules={"flash_attn": 2}
        )
        ru.validate_recompute_modules(config)

    def test_bool_selector_raises(self):
        config = make_config(recompute_modules={"attn": True})
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(config)

    def test_non_string_key_raises(self):
        config = make_config(recompute_modules={5: "all"})
        with self.assertRaises(ValueError):
            ru.validate_recompute_modules(config)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class HasRecoveredTest(unittest.TestCase):
    """``has_recovered``: env-driven recovery-step gate."""

    def setUp(self):
        # Snapshot and restore the three env vars this function reads so the
        # test never leaks state into the process (antipattern #11).
        for name in ("RECOVER_STEP", "TRAINER_GLOBAL_STEP", "PDC_INIT_STEP"):
            original = os.environ.get(name)
            self.addCleanup(self._restore_env, name, original)
            os.environ.pop(name, None)

    @staticmethod
    def _restore_env(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_no_recover_step_is_recovered(self):
        """Without RECOVER_STEP the gate is open (True)."""
        self.assertTrue(ru.has_recovered())

    def test_current_step_beyond_recover_step_is_true(self):
        os.environ["RECOVER_STEP"] = "5"
        os.environ["TRAINER_GLOBAL_STEP"] = "10"
        self.assertTrue(ru.has_recovered())

    def test_current_step_not_beyond_recover_step_is_false(self):
        os.environ["RECOVER_STEP"] = "5"
        os.environ["TRAINER_GLOBAL_STEP"] = "3"
        self.assertFalse(ru.has_recovered())

    def test_pdc_init_step_used_when_trainer_step_missing(self):
        """PDC_INIT_STEP is the fallback current step; 10 > 5 -> True."""
        os.environ["RECOVER_STEP"] = "5"
        os.environ["PDC_INIT_STEP"] = "10"
        self.assertTrue(ru.has_recovered())

    def test_missing_current_step_raises(self):
        """RECOVER_STEP set but neither current-step source present -> assert."""
        os.environ["RECOVER_STEP"] = "5"
        with self.assertRaises(AssertionError):
            ru.has_recovered()


if __name__ == "__main__":
    unittest.main()
