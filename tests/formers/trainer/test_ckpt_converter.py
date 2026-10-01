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

"""Behavior tests for trainer/utils/ckpt_converter.py (CheckpointConverter).

Scope: the hybrid-parallel -> auto-parallel checkpoint conversion helpers that
run purely on CPU without a real process group: filename rank parsing, the
sharding greedy partition planner, optimizer<->model key mapping, the
optimizer-state renaming, and model/optimizer file pairing.

Independent oracle: every expected value below is derived BY HAND from the
documented semantics, NOT by calling the production routine a second time:
  * File rank parsing is checked against the integers a human reads out of the
    ``tpXX`` / ``ppXX`` / ``shardXX`` substrings (regex ``\\w+(\\d+)``), including
    the documented return ORDER ``(tp, pp, sharding)``.
  * ``partition_parameters`` is checked against a longhand replay of the
    "assign the next parameter to the currently-smallest bucket" greedy rule,
    with numel taken as the PRODUCT of the shape (so multi-dim shapes and the
    optional size-descending pre-sort are both exercised and distinguished).
  * key/suffix mappings are checked against literal expected strings.

The heavy distributed ``__init__`` (collectives, disk scan) is bypassed with
``object.__new__`` and ONLY the couple of plain attributes each method reads are
populated; the method under test is then executed for real and its computed
output compared to the hand-derived oracle. Nothing patches the routine under
test, and no expected value is produced by the production code.
"""

import unittest

try:
    from paddlefleet.trainer.utils.ckpt_converter import (
        MODEL_WEIGHT_SUFFIX,
        OPTIMIZER_STATE_NAME_SUFFIX,
        OPTIMIZER_WEIGHT_SUFFIX,
        CheckpointConverter,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this runner
    _IMPORT_ERROR = exc


def _new_converter():
    """Construct a CheckpointConverter without its distributed __init__.

    The methods exercised here read at most a couple of plain attributes; the
    real method bodies still run. This is not a mock of the routine under test.
    """
    return object.__new__(CheckpointConverter)


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestFileNameRankParsing(unittest.TestCase):
    """get_distribution_rank_from_file_name reads (tp, pp, sharding) integers."""

    def setUp(self):
        self.conv = _new_converter()

    def test_all_three_ranks_parsed_in_tp_pp_shard_order(self):
        # A human reads tp=2, pp=1, shard=3 out of the substrings; the contract
        # returns them in (tp, pp, sharding) order -- a swapped return would be
        # caught because the three values are deliberately distinct.
        got = self.conv.get_distribution_rank_from_file_name(
            "optimizer.tp02_pp01_shard03.pdopt"
        )
        self.assertEqual(got, (2, 1, 3))

    def test_missing_fields_default_to_zero(self):
        # Only tp present -> pp and sharding fall back to 0, not to tp's value.
        self.assertEqual(
            self.conv.get_distribution_rank_from_file_name(
                "model.tp07.pdparams"
            ),
            (7, 0, 0),
        )
        # No rank tokens at all -> all zero.
        self.assertEqual(
            self.conv.get_distribution_rank_from_file_name(
                "scheduler.pdparams"
            ),
            (0, 0, 0),
        )

    def test_multidigit_values_are_full_integers_not_first_digit(self):
        # "tp12" must parse as 12, not 1; guards against a single-digit regex.
        self.assertEqual(
            self.conv.get_distribution_rank_from_file_name(
                "model_state.tp12_pp05.pdparams"
            ),
            (12, 5, 0),
        )


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestPartitionParameters(unittest.TestCase):
    """partition_parameters greedily packs params into the smallest bucket.

    Oracle: replay the rule "assign the next parameter to the rank with the
    currently smallest accumulated numel (ties -> lowest rank index)" by hand,
    where numel is the PRODUCT of the shape.
    """

    def setUp(self):
        self.conv = _new_converter()

    def test_no_sort_preserves_input_order_and_packs_greedily(self):
        # numels: a=1, b=4, c=2, d=3 in this exact order (is_sort=False).
        #   start [0,0]  a->r0 [1,0]  b->r1 [1,4]  c->r0 [3,4]  d->r0 [6,4]
        params = [("a", [1]), ("b", [4]), ("c", [2]), ("d", [3])]
        mapping = self.conv.partition_parameters(params, False, 2)
        self.assertEqual(mapping[0], [("a", [1]), ("c", [2]), ("d", [3])])
        self.assertEqual(mapping[1], [("b", [4])])

    def test_sort_by_descending_numel_changes_assignment(self):
        # Same params, is_sort=True -> sorted b(4),d(3),c(2),a(1):
        #   [0,0] b->r0 [4,0]  d->r1 [4,3]  c->r1 [4,5]  a->r0 [5,5]
        # A no-op sort would leave the r0=[a,c,d] layout above and fail here.
        params = [("a", [1]), ("b", [4]), ("c", [2]), ("d", [3])]
        mapping = self.conv.partition_parameters(params, True, 2)
        self.assertEqual(mapping[0], [("b", [4]), ("a", [1])])
        self.assertEqual(mapping[1], [("d", [3]), ("c", [2])])

    def test_numel_uses_product_of_shape_not_leading_dim(self):
        # big=[2,3] has numel 6 and must dominate small=[4] (numel 4); a planner
        # that only looked at shape[0] would think small(4) > big(2) and swap.
        params = [("small", [4]), ("big", [2, 3])]
        mapping = self.conv.partition_parameters(params, True, 2)
        self.assertEqual(mapping[0], [("big", [2, 3])])
        self.assertEqual(mapping[1], [("small", [4])])

    def test_every_rank_bucket_exists_even_when_unused(self):
        params = [("only", [5])]
        mapping = self.conv.partition_parameters(params, False, 3)
        self.assertEqual(set(mapping.keys()), {0, 1, 2})
        self.assertEqual(mapping[0], [("only", [5])])
        self.assertEqual(mapping[1], [])
        self.assertEqual(mapping[2], [])


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestOptimizerKeyToModelStateKey(unittest.TestCase):
    """optimizer_key_to_model_state_key strips exactly one known opt suffix."""

    def setUp(self):
        self.conv = _new_converter()

    def test_each_known_suffix_is_removed(self):
        for suffix in OPTIMIZER_STATE_NAME_SUFFIX:
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    self.conv.optimizer_key_to_model_state_key(
                        "layer.w_0" + suffix
                    ),
                    "layer.w_0",
                )

    def test_key_without_known_suffix_is_returned_unchanged(self):
        # ".weight" is not in OPTIMIZER_STATE_NAME_SUFFIX -> left intact.
        self.assertEqual(
            self.conv.optimizer_key_to_model_state_key("layer.w_0.weight"),
            "layer.w_0.weight",
        )

    def test_only_trailing_suffix_stripped_not_interior_match(self):
        # ".moment1" appears in the middle but the key does not END with any
        # known suffix, so nothing is removed.
        self.assertEqual(
            self.conv.optimizer_key_to_model_state_key("a.moment1.extra"),
            "a.moment1.extra",
        )


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestGetModelStateFileFrom(unittest.TestCase):
    """get_model_state_file_from pairs an optimizer file to the model file
    sharing the same (tp, pp) rank, ignoring the sharding rank."""

    def setUp(self):
        self.conv = _new_converter()
        self.conv.global_model_state_file_names = [
            "model_state.tp00_pp00.pdparams",
            "model_state.tp00_pp01.pdparams",
            "model_state.tp01_pp00.pdparams",
        ]

    def test_matches_on_tp_and_pp_ignoring_shard(self):
        # optimizer at tp0/pp1/shard2 must pair with the tp0/pp1 model file,
        # not tp0/pp0; sharding rank in the name is irrelevant to the match.
        got = self.conv.get_model_state_file_from(
            "optimizer.tp00_pp01_shard02.pdopt"
        )
        self.assertEqual(got, "model_state.tp00_pp01.pdparams")

    def test_distinct_tp_selects_the_right_file(self):
        got = self.conv.get_model_state_file_from("optimizer.tp01_pp00.pdopt")
        self.assertEqual(got, "model_state.tp01_pp00.pdparams")

    def test_returns_none_when_no_rank_matches(self):
        # No model file at pp09 -> no pairing.
        self.assertIsNone(
            self.conv.get_model_state_file_from("optimizer.tp00_pp09.pdopt")
        )


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestRenameUsingStructuredNameMapping(unittest.TestCase):
    """rename_using_parameter_to_structured_name_mapping maps raw
    ``param+suffix`` optimizer keys onto ``structured_name + normalized_suffix``.

    Oracle: the structured names come from an explicit mapping and the
    normalized suffix is the literal ``.moment1`` / ``.moment2`` /
    ``.beta1_pow_acc`` / ``.beta2_pow_acc`` / ``.master_weight`` that the
    documented suffix-detection order dictates.
    """

    def setUp(self):
        self.conv = _new_converter()
        # Required by the paddle-only value filter inside the method.
        import paddle

        self.paddle = paddle
        self.p2s = {
            "embedding_0.w_0": "embed.tok_embeddings.weight",
            "linear_0.w_0": "layers.0.mlp.gate.weight",
        }

    def _t(self, *vals):
        return self.paddle.to_tensor(list(vals), dtype="float32")

    def test_optimizer_suffixes_map_to_structured_name(self):
        state = {
            "embedding_0.w_0.moment1": self._t(1.0),
            "linear_0.w_0.moment2": self._t(2.0),
            "linear_0.w_0.beta1_pow_acc": self._t(3.0),
            "linear_0.w_0.beta2_pow_acc": self._t(4.0),
        }
        out = self.conv.rename_using_parameter_to_structured_name_mapping(
            state, self.p2s
        )
        self.assertEqual(
            set(out),
            {
                "embed.tok_embeddings.weight.moment1",
                "layers.0.mlp.gate.weight.moment2",
                "layers.0.mlp.gate.weight.beta1_pow_acc",
                "layers.0.mlp.gate.weight.beta2_pow_acc",
            },
        )
        # Value identity must follow the key: moment1 tensor stays with the
        # embedding param, not silently swapped with another state.
        self.assertEqual(
            out["embed.tok_embeddings.weight.moment1"].tolist(), [1.0]
        )
        self.assertEqual(
            out["layers.0.mlp.gate.weight.beta2_pow_acc"].tolist(), [4.0]
        )

    def test_unrecognized_suffix_falls_back_to_master_weight(self):
        # A suffix matching none of moment/beta patterns is normalized to
        # ".master_weight" per the else-branch contract.
        state = {"embedding_0.w_0.master_weight": self._t(9.0)}
        out = self.conv.rename_using_parameter_to_structured_name_mapping(
            state, self.p2s
        )
        self.assertEqual(
            list(out), ["embed.tok_embeddings.weight.master_weight"]
        )

    def test_already_structured_key_is_kept_verbatim(self):
        # A key that is itself a structured target value is passed through
        # unchanged (no suffix appended).
        state = {"embed.tok_embeddings.weight": self._t(5.0, 6.0)}
        out = self.conv.rename_using_parameter_to_structured_name_mapping(
            state, self.p2s
        )
        self.assertEqual(list(out), ["embed.tok_embeddings.weight"])
        self.assertEqual(
            out["embed.tok_embeddings.weight"].tolist(), [5.0, 6.0]
        )

    def test_none_values_are_dropped(self):
        # None values are skipped entirely; initialized tensors survive.
        state = {
            "linear_0.w_0.moment1": self._t(1.0),
            "embedding_0.w_0.moment1": None,
        }
        out = self.conv.rename_using_parameter_to_structured_name_mapping(
            state, self.p2s
        )
        self.assertEqual(list(out), ["layers.0.mlp.gate.weight.moment1"])


@unittest.skipIf(
    _IMPORT_ERROR is not None, f"paddlefleet import failed: {_IMPORT_ERROR}"
)
class TestModuleSuffixContract(unittest.TestCase):
    """The file-suffix constants are the contract the pairing/parsing rely on."""

    def test_weight_and_optimizer_suffixes_are_distinct(self):
        self.assertEqual(MODEL_WEIGHT_SUFFIX, ".pdparams")
        self.assertEqual(OPTIMIZER_WEIGHT_SUFFIX, ".pdopt")
        self.assertNotEqual(MODEL_WEIGHT_SUFFIX, OPTIMIZER_WEIGHT_SUFFIX)


if __name__ == "__main__":
    unittest.main()
