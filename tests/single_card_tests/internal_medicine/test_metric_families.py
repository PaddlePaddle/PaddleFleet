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

"""Metric family taxonomy: classification, selection parsing, drift guards.

The corpus fixture is the metric-name half of a real 8-GPU run's
``internal_medicine.jsonl`` (layer / expert / element indices normalised away,
values dropped). It is here so that "someone added a metric and forgot to put it
in a family" fails as a test rather than as a silently-missing curve months
later.
"""

import importlib
import unittest
from pathlib import Path

mf = importlib.import_module(
    "paddlefleet.internal_medicine.core.metric_families"
)

CORPUS = Path(__file__).resolve().parent / "fixtures" / "metric_key_corpus.txt"


def corpus_entries():
    for line in CORPUS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        monitor, _, sub = line.partition("/")
        # `layer_7/attn_h_post_mean` → the family is a property of the metric,
        # not of which layer it came from.
        if sub.startswith("layer_"):
            sub = sub.split("/", 1)[1]
        yield monitor, sub


class MetricFamilyTaxonomyTest(unittest.TestCase):
    def test_the_corpus_is_not_empty(self):
        """A silently-empty fixture would make every assertion below vacuous."""
        entries = list(corpus_entries())
        self.assertGreater(len(entries), 400)
        self.assertEqual(len({m for m, _ in entries}), 7)

    def test_every_real_metric_lands_in_a_declared_family(self):
        """Nothing may fall through to `other`.

        `other` is deliberately fail-open at collection time, so a metric landing
        there is not a crash — it is a metric that can never be switched off.
        Catch it here instead.
        """
        unmatched = {}
        for monitor, sub in corpus_entries():
            family, _component = mf.classify(monitor, sub)
            if family == mf.FAMILY_OTHER:
                unmatched.setdefault(monitor, []).append(sub)
        self.assertEqual(unmatched, {}, f"metrics with no family: {unmatched}")

    def test_every_declared_family_is_exercised_by_the_corpus(self):
        """A family no real metric matches is dead config — or a broken pattern."""
        seen = {monitor: set() for monitor in mf.METRIC_TAXONOMY}
        for monitor, sub in corpus_entries():
            family, _component = mf.classify(monitor, sub)
            if monitor in seen:
                seen[monitor].add(family)
        for monitor, declared in (
            (m, mf.families_of(m)) for m in mf.METRIC_TAXONOMY
        ):
            self.assertEqual(
                set(declared) - seen[monitor],
                set(),
                f"{monitor}: families matched by nothing in the corpus",
            )

    def test_components_are_stripped_before_family_matching(self):
        """The same quantity on two components is one family, not two."""
        self.assertEqual(
            mf.classify("mhc_health", "attn_h_post_mean"), ("gate", "attn")
        )
        self.assertEqual(
            mf.classify("mhc_health", "mlp_h_post_mean"), ("gate", "mlp")
        )
        self.assertEqual(
            mf.classify("vha_health", "main_postmix_u_norm")[0], "mix"
        )

    def test_attention_kind_and_global_prefixes_are_stripped(self):
        """`global_` and the attention kind are key layout, not metric identity."""
        self.assertEqual(
            mf.classify("mhc_health", "global_attn_amax_gain_fwd"),
            ("gain", "attn"),
        )
        self.assertEqual(
            mf.classify("qk_stats", "hca_entropy_mean")[0], "entropy"
        )
        self.assertEqual(
            mf.strip_prefixes("global_mla_sink_weight"), "sink_weight"
        )

    def test_moe_family_order_keeps_the_shared_catch_last(self):
        """`^shared_` would swallow these two if it were matched first."""
        self.assertEqual(
            mf.classify("moe_health", "shared_gate_stable_rank")[0], "spectrum"
        )
        self.assertEqual(
            mf.classify("moe_health", "shared_act_abs_max")[0], "act"
        )
        self.assertEqual(mf.classify("moe_health", "shared_frac")[0], "shared")

    def test_massive_act_channel_does_not_swallow_the_outlier_counts(self):
        self.assertEqual(
            mf.classify("massive_act", "channel_count_gt_100")[0], "outlier"
        )
        self.assertEqual(
            mf.classify("massive_act", "channel_median")[0], "channel"
        )

    def test_unknown_monitor_classifies_as_other(self):
        """A monitor with no taxonomy entry degrades to "cannot be narrowed"."""
        self.assertEqual(
            mf.classify("kda_health", "anything"), (mf.FAMILY_OTHER, "")
        )
        self.assertEqual(mf.families_of("kda_health"), ())


class FamilySelectionTest(unittest.TestCase):
    def test_no_exclusion_means_collect_everything(self):
        """The default has to be byte-for-byte today's behaviour."""
        for value in (None, "none", "", []):
            selection = mf.parse_exclude("mhc_health", value)
            self.assertEqual(
                selection.excluded, frozenset(), f"value={value!r}"
            )
            for monitor, sub in corpus_entries():
                if monitor == "mhc_health":
                    self.assertTrue(selection.allows(sub))

    def test_excluding_a_family_drops_exactly_that_family(self):
        selection = mf.parse_exclude("mhc_health", "mix")
        dropped, kept = set(), set()
        for monitor, sub in corpus_entries():
            if monitor != "mhc_health":
                continue
            (kept if selection.allows(sub) else dropped).add(
                mf.classify(monitor, sub)[0]
            )
        self.assertEqual(dropped, {"mix"})
        self.assertEqual(
            kept, {"gate", "stream", "cell", "gain", "param", "share"}
        )

    def test_excluding_several_families_at_once(self):
        selection = mf.parse_exclude("moe_health", "expert+act")
        self.assertFalse(selection.allows("expert_token_share_e0"))
        self.assertFalse(selection.allows("shared_act_abs_max"))
        self.assertTrue(selection.allows("router_entropy"))

    def test_exclusion_accepts_a_list_as_well_as_a_string(self):
        self.assertEqual(
            mf.parse_exclude("moe_health", ["expert", "act"]).excluded,
            frozenset({"expert", "act"}),
        )
        self.assertEqual(
            mf.parse_exclude("moe_health", "expert, act").excluded,
            frozenset({"expert", "act"}),
        )

    def test_an_unclassified_metric_is_collected_rather_than_dropped(self):
        """Fail open falls out of exclusion: a taxonomy hole is in no exclusion set."""
        selection = mf.parse_exclude("mhc_health", "mix")
        self.assertTrue(selection.allows("some_metric_added_after_this_table"))

    def test_a_typo_in_an_excluded_family_fails_at_startup(self):
        """Otherwise the family stays on and the only symptom is payload that did not shrink."""
        with self.assertRaises(mf.UnknownFamilyError) as caught:
            mf.parse_exclude("mhc_health", "mixx")
        self.assertIn("mixx", str(caught.exception))
        self.assertIn(
            "mix", str(caught.exception)
        )  # the message lists what is valid

    def test_excluding_from_a_monitor_without_a_taxonomy_fails(self):
        with self.assertRaises(mf.UnknownFamilyError):
            mf.parse_exclude("kda_health", "whatever")

    def test_parse_exclusions_reads_the_config_string(self):
        self.assertEqual(
            mf.parse_exclusions("mhc_health:mix, moe_health:expert+act"),
            {"mhc_health": ["mix"], "moe_health": ["expert", "act"]},
        )

    def test_parse_exclusions_of_an_absent_config_switches_nothing_off(self):
        for spec in (None, "", [], "moe_health", "moe_health:"):
            self.assertEqual(mf.parse_exclusions(spec), {}, f"spec={spec!r}")

    def test_parse_exclusions_tolerates_whitespace_and_repeats(self):
        self.assertEqual(
            mf.parse_exclusions(
                " mhc_health : mix + gain ,, mhc_health:param , "
            ),
            {"mhc_health": ["mix", "gain", "param"]},
        )

    def test_a_misspelled_monitor_name_raises_instead_of_switching_nothing_off(
        self,
    ):
        """The fail-open half: the backend loop would just never match "moe"."""
        with self.assertRaises(mf.UnknownMonitorError) as caught:
            mf.validate_exclusions(
                mf.parse_exclusions("moe:expert"),
                {"moe_health": None},
                backend="paddlefleet",
            )
        self.assertIn("moe", str(caught.exception))
        self.assertIn(
            "moe_health", str(caught.exception)
        )  # the message lists what is valid

    def test_validate_exclusions_also_checks_family_names(self):
        """So a family typo raises before the backends' per-monitor try/except."""
        with self.assertRaises(mf.UnknownFamilyError):
            mf.validate_exclusions(
                mf.parse_exclusions("moe_health:experts"), {"moe_health": None}
            )

    def test_a_known_but_unused_monitor_is_accepted(self):
        """One exclusion string may be shared by configs enabling different monitors."""
        mf.validate_exclusions(
            mf.parse_exclusions("moe_health:expert"),
            {"moe_health": None, "qk_stats": None},
        )


class DebugOnlyFamiliesTest(unittest.TestCase):
    def test_mhc_per_unit_slices_are_their_own_families(self):
        """The per-cell / per-stream curves must not fall into their aggregates.

        This split is the whole point of the production default: `h_res_cell{i}`
        and `h_{pre,post}_idx{i}` are one curve per structural unit, while
        `h_res_logits_max` / `h_post_mean` are one per layer and are what the
        diagnosis layer reads.
        """
        self.assertEqual(
            mf.classify("mhc_health", "attn_h_res_cell0"), ("cell", "attn")
        )
        self.assertEqual(
            mf.classify("mhc_health", "attn_h_res_logits_max"), ("mix", "attn")
        )
        self.assertEqual(
            mf.classify("mhc_health", "mlp_h_pre_idx3"), ("stream", "mlp")
        )
        self.assertEqual(
            mf.classify("mhc_health", "mlp_h_post_idx3"), ("stream", "mlp")
        )
        self.assertEqual(
            mf.classify("mhc_health", "mlp_h_post_mean"), ("gate", "mlp")
        )
        # `global_` aggregates have no index and so stay with the aggregates.
        self.assertEqual(
            mf.classify("mhc_health", "global_attn_h_post_token_std")[0], "gate"
        )

    def test_the_debug_only_set_is_pinned(self):
        """Changing what a production run stops collecting must be a visible diff."""
        self.assertEqual(
            {
                monitor: set(families)
                for monitor, families in mf.DEBUG_ONLY_FAMILIES.items()
            },
            {"moe_health": {"expert"}, "mhc_health": {"cell", "stream"}},
        )

    def test_every_debug_only_family_is_a_declared_family(self):
        """A stale name here would silently switch nothing off in production."""
        for monitor, families in mf.DEBUG_ONLY_FAMILIES.items():
            selection = mf.parse_exclude(monitor, list(families))
            self.assertEqual(selection.excluded, frozenset(families))

    def test_debug_mode_collects_everything(self):
        self.assertEqual(mf.exclusions_for(True), {})

    def test_production_drops_the_debug_only_families(self):
        self.assertEqual(
            mf.exclusions_for(False),
            {
                monitor: list(families)
                for monitor, families in mf.DEBUG_ONLY_FAMILIES.items()
            },
        )

    def test_the_returned_mapping_is_the_callers_to_mutate(self):
        """Backends merge per-monitor options into it; the constant must survive."""
        excluded = mf.exclusions_for(False)
        excluded["moe_health"].append("router")
        excluded["qk_stats"] = ["sink"]
        self.assertEqual(mf.DEBUG_ONLY_FAMILIES["moe_health"], ("expert",))
        self.assertNotIn("qk_stats", mf.DEBUG_ONLY_FAMILIES)

    def test_production_exclusions_pass_backend_validation(self):
        """The default must never be the thing that fails a run at startup."""
        known = dict.fromkeys(mf.METRIC_TAXONOMY)
        mf.validate_exclusions(
            mf.exclusions_for(False), known, backend="paddlefleet"
        )

    def test_the_debug_only_families_are_the_ones_that_grow_per_unit(self):
        """The selection criterion, checked against real keys rather than asserted.

        Indices are normalised out of the corpus, so this counts *distinct key
        shapes*, not the per-step key count — enough to show these families carry
        the per-expert / per-cell fan-out.
        """
        counts = {}
        for monitor, sub in corpus_entries():
            family, _component = mf.classify(monitor, sub)
            counts[(monitor, family)] = counts.get((monitor, family), 0) + 1
        for monitor, families in mf.DEBUG_ONLY_FAMILIES.items():
            for family in families:
                self.assertGreater(
                    counts.get((monitor, family), 0),
                    0,
                    f"{monitor}:{family} matches no real key",
                )


if __name__ == "__main__":
    unittest.main()
