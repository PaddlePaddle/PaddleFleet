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

"""The ``recompute_modules`` selector grammar.

Covers the shared entry points every selective-recompute submodule now goes
through -- ``module_needs_recompute`` and its refined-recompute (RR) sibling --
plus the startup validation in ``validate_recompute_modules``.
"""

import unittest

from paddlefleet.recompute_utils import (
    module_needs_recompute,
    module_needs_refined_recompute,
    validate_recompute_modules,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


def _config(**kwargs):
    kwargs.setdefault("num_hidden_layers", 8)
    kwargs.setdefault("recompute_granularity", "selective")
    return TransformerConfig(**kwargs)


def _hits(module_name, config, num_layers=8):
    return [
        layer
        for layer in range(num_layers)
        if module_needs_recompute(module_name, layer, config)
    ]


class TestListMode(unittest.TestCase):
    """List mode keeps its historical meaning."""

    def test_no_num_layers_covers_every_layer(self):
        config = _config(recompute_modules=["mlp", "core_attn"])
        self.assertEqual(_hits("mlp", config), list(range(8)))
        self.assertEqual(_hits("core_attn", config), list(range(8)))

    def test_absent_module_is_off(self):
        config = _config(recompute_modules=["mlp"])
        self.assertEqual(_hits("core_attn", config), [])

    def test_num_layers_is_shared_by_every_module(self):
        config = _config(
            recompute_modules=["mlp", "core_attn"],
            recompute_method="first_n",
            recompute_num_layers=3,
        )
        self.assertEqual(_hits("mlp", config), [0, 1, 2])
        self.assertEqual(_hits("core_attn", config), [0, 1, 2])

    def test_block_method(self):
        config = _config(
            recompute_modules=["mlp"],
            recompute_method="block",
            recompute_num_layers=3,
        )
        self.assertEqual(_hits("mlp", config), [0, 1, 2])

    def test_none_modules_is_off(self):
        config = TransformerConfig(num_hidden_layers=8)
        self.assertEqual(_hits("mlp", config), [])


class TestDictLayerCount(unittest.TestCase):
    """Dict mode with an int keeps the ``recompute_num_layers`` semantics."""

    def test_per_module_counts(self):
        config = _config(
            recompute_modules={"mlp": 2, "core_attn": 5},
            recompute_method="first_n",
        )
        self.assertEqual(_hits("mlp", config), [0, 1])
        self.assertEqual(_hits("core_attn", config), [0, 1, 2, 3, 4])

    def test_block_method(self):
        config = _config(
            recompute_modules={"mlp": 2},
            recompute_method="block",
        )
        self.assertEqual(_hits("mlp", config), [0, 1])

    def test_all_and_negative_mean_every_layer(self):
        for spec in ("all", -1, None):
            config = _config(recompute_modules={"mlp": spec})
            self.assertEqual(
                _hits("mlp", config), list(range(8)), f"spec={spec!r}"
            )


class TestDictLayerList(unittest.TestCase):
    """The new form: an explicit set of layers, per submodule."""

    def test_exact_layers(self):
        config = _config(recompute_modules={"mlp": [0, 3, 7]})
        self.assertEqual(_hits("mlp", config), [0, 3, 7])

    def test_independent_per_module(self):
        config = _config(
            recompute_modules={
                "core_attn": [0, 1],
                "mlp": [6, 7],
                "moe_gate_up": "all",
            }
        )
        self.assertEqual(_hits("core_attn", config), [0, 1])
        self.assertEqual(_hits("mlp", config), [6, 7])
        self.assertEqual(_hits("moe_gate_up", config), list(range(8)))

    def test_recompute_method_is_ignored(self):
        for method in (None, "first_n", "block"):
            config = _config(
                recompute_modules={"mlp": [2, 4]},
                recompute_method=method,
            )
            self.assertEqual(_hits("mlp", config), [2, 4], f"method={method}")

    def test_empty_list_disables_every_layer(self):
        config = _config(recompute_modules={"mlp": []})
        self.assertEqual(_hits("mlp", config), [])

    def test_tuple_and_set_are_accepted(self):
        for spec in ((1, 2), {1, 2}):
            config = _config(recompute_modules={"mlp": spec})
            self.assertEqual(_hits("mlp", config), [1, 2], f"spec={spec!r}")

    def test_layer_ids_skip_empty_head_layers(self):
        # Layer ids are logical: id 0 is the first real backbone layer, whatever
        # num_empty_layers_add_in_head is. Physical layer numbers are shifted.
        config = _config(
            num_hidden_layers=6,
            num_empty_layers_add_in_head=2,
            recompute_modules={"mlp": [0, 2]},
        )
        self.assertEqual(_hits("mlp", config, num_layers=8), [2, 4])

    def test_layer_ids_match_csa_compress_ratios_indexing(self):
        # The same index space per-layer model_config fields use, so
        # csa_compress_ratios[i] and a layer id of i mean the same layer.
        config = _config(num_hidden_layers=6, recompute_modules={"mlp": [0, 5]})
        self.assertEqual(_hits("mlp", config, num_layers=6), [0, 5])


class TestMTPLayers(unittest.TestCase):
    """MTP layers are addressed after the backbone, not aliased onto layer 0.

    An MTP layer is built with ``layer_number=i`` within the MTP block, so
    without the logical mapping it would collide with backbone layer ``i``.
    """

    def test_mtp_layer_id_follows_the_backbone(self):
        config = _config(
            num_hidden_layers=6,
            num_nextn_predict_layers=1,
            recompute_modules={"mlp": [6]},
        )
        self.assertTrue(
            module_needs_recompute("mlp", 0, config, is_mtp_layer=True)
        )
        self.assertFalse(module_needs_recompute("mlp", 0, config))

    def test_backbone_layer_zero_does_not_select_mtp(self):
        config = _config(
            num_hidden_layers=6,
            num_nextn_predict_layers=1,
            recompute_modules={"mlp": [0]},
        )
        self.assertTrue(module_needs_recompute("mlp", 0, config))
        self.assertFalse(
            module_needs_recompute("mlp", 0, config, is_mtp_layer=True)
        )

    def test_multiple_mtp_layers_are_addressed_independently(self):
        config = _config(
            num_hidden_layers=6,
            num_nextn_predict_layers=2,
            recompute_modules={"mlp": [7]},
        )
        self.assertFalse(
            module_needs_recompute("mlp", 0, config, is_mtp_layer=True)
        )
        self.assertTrue(
            module_needs_recompute("mlp", 1, config, is_mtp_layer=True)
        )

    def test_empty_head_layers_do_not_shift_mtp_ids(self):
        config = _config(
            num_hidden_layers=6,
            num_empty_layers_add_in_head=2,
            num_nextn_predict_layers=1,
            recompute_modules={"mlp": [6]},
        )
        self.assertTrue(
            module_needs_recompute("mlp", 0, config, is_mtp_layer=True)
        )

    def test_rr_list_inverts_on_the_logical_id(self):
        config = _config(
            num_hidden_layers=6,
            num_nextn_predict_layers=1,
            recompute_modules={"flash_attn": [6]},
        )
        self.assertFalse(
            module_needs_refined_recompute(
                "flash_attn", 0, config, is_mtp_layer=True
            )
        )
        self.assertTrue(module_needs_refined_recompute("flash_attn", 0, config))

    def test_mtp_id_is_valid_at_startup(self):
        _config(
            num_hidden_layers=6,
            num_nextn_predict_layers=1,
            recompute_modules={"mlp": [6]},
        )
        with self.assertRaises(ValueError):
            _config(
                num_hidden_layers=6,
                num_nextn_predict_layers=1,
                recompute_modules={"mlp": [7]},
            )


class TestRefinedRecompute(unittest.TestCase):
    """RR entries invert the selector: the spec picks the non-RR layers."""

    def test_list_mode_enables_rr_everywhere(self):
        config = _config(
            recompute_modules=["flash_attn"],
            recompute_method="first_n",
            recompute_num_layers=3,
        )
        for layer in range(8):
            self.assertTrue(
                module_needs_refined_recompute("flash_attn", layer, config)
            )

    def test_dict_count_keeps_first_n_on_plain_recompute(self):
        config = _config(
            recompute_modules={"flash_attn": 3},
            recompute_method="first_n",
        )
        rr = [
            layer
            for layer in range(8)
            if module_needs_refined_recompute("flash_attn", layer, config)
        ]
        self.assertEqual(rr, [3, 4, 5, 6, 7])

    def test_dict_layer_list_inverts(self):
        config = _config(recompute_modules={"flash_attn": [0, 1]})
        rr = [
            layer
            for layer in range(8)
            if module_needs_refined_recompute("flash_attn", layer, config)
        ]
        self.assertEqual(rr, [2, 3, 4, 5, 6, 7])

    def test_all_disables_rr(self):
        config = _config(recompute_modules={"flash_attn": "all"})
        for layer in range(8):
            self.assertFalse(
                module_needs_refined_recompute("flash_attn", layer, config)
            )

    def test_negative_count_disables_rr_like_all(self):
        """A negative count means every layer, so RR must be off."""
        for spec in (-1, -8):
            config = _config(recompute_modules={"flash_attn": spec})
            for layer in range(8):
                self.assertFalse(
                    module_needs_refined_recompute("flash_attn", layer, config),
                    msg=f"flash_attn={spec} must disable RR on layer {layer}",
                )

    def test_none_disables_rr(self):
        config = _config(recompute_modules={"flash_attn": None})
        for layer in range(8):
            self.assertFalse(
                module_needs_refined_recompute("flash_attn", layer, config)
            )

    def test_zero_and_negative_are_opposites(self):
        """``0`` selects nothing so RR covers everything; ``-1`` is the reverse."""
        zero = _config(
            recompute_modules={"flash_attn": 0}, recompute_method="first_n"
        )
        negative = _config(recompute_modules={"flash_attn": -1})
        for layer in range(8):
            self.assertTrue(
                module_needs_refined_recompute("flash_attn", layer, zero)
            )
            self.assertFalse(
                module_needs_refined_recompute("flash_attn", layer, negative)
            )

    def test_negative_count_is_topology_independent(self):
        """``need_recompute_in_first_n`` gives a negative count three different
        meanings across PP/VPP layouts; the shortcut must not."""
        for pp, vpp in ((1, None), (2, None), (2, 2), (4, None), (4, 2)):
            config = _config(
                recompute_modules={"flash_attn": -1},
                pipeline_model_parallel_size=pp,
                virtual_pipeline_model_parallel_size=vpp,
            )
            for layer in range(8):
                self.assertFalse(
                    module_needs_refined_recompute("flash_attn", layer, config),
                    msg=f"pp={pp} vpp={vpp} layer={layer}",
                )

    def test_negative_count_mirrors_plain_path(self):
        """Plain and RR paths must agree on what "every layer" means."""
        for spec in (-1, "all", None):
            plain = _config(recompute_modules={"core_attn": spec})
            rr = _config(recompute_modules={"flash_attn": spec})
            for layer in range(8):
                self.assertTrue(
                    module_needs_recompute("core_attn", layer, plain)
                )
                self.assertFalse(
                    module_needs_refined_recompute("flash_attn", layer, rr)
                )

    def test_absent_module_is_off(self):
        config = _config(
            recompute_modules={"mlp": 3}, recompute_method="first_n"
        )
        self.assertFalse(
            module_needs_refined_recompute("flash_attn", 0, config)
        )


class TestLayerAgnosticModules(unittest.TestCase):
    """``lm_head`` / ``loss_fn`` exist once per model, not once per layer."""

    def test_list_mode_ignores_num_layers(self):
        config = _config(
            recompute_modules=["lm_head", "loss_fn"],
            recompute_method="first_n",
            recompute_num_layers=3,
        )
        self.assertTrue(module_needs_recompute("lm_head", None, config))
        self.assertTrue(module_needs_recompute("loss_fn", None, config))

    def test_all_is_accepted(self):
        config = _config(recompute_modules={"lm_head": "all"})
        self.assertTrue(module_needs_recompute("lm_head", None, config))

    def test_layer_list_is_rejected(self):
        with self.assertRaises(ValueError):
            _config(recompute_modules={"lm_head": [0, 1]})


class TestDeferredLayerNumber(unittest.TestCase):
    """MoE submodules resolve their flags before the layer id is known."""

    def test_count_spec_defaults_to_on(self):
        config = _config(
            recompute_modules={"moe_gate_up": 3},
            recompute_method="first_n",
        )
        self.assertTrue(module_needs_recompute("moe_gate_up", None, config))

    def test_layer_list_needs_a_layer_number(self):
        config = _config(recompute_modules={"moe_gate_up": [1, 2]})
        with self.assertRaises(ValueError):
            module_needs_recompute("moe_gate_up", None, config)


class TestValidation(unittest.TestCase):
    def test_out_of_range_layer_id(self):
        with self.assertRaises(ValueError):
            _config(num_hidden_layers=4, recompute_modules={"mlp": [0, 4]})

    def test_empty_head_layers_do_not_extend_the_range(self):
        # Empty layers hold no recomputable module, so they are not addressable:
        # 4 backbone layers means ids 0..3 whatever the head layer count is.
        with self.assertRaises(ValueError):
            _config(
                num_hidden_layers=4,
                num_empty_layers_add_in_head=2,
                recompute_modules={"mlp": [0, 4]},
            )

    def test_mtp_layers_extend_the_range(self):
        # 4 backbone + 1 MTP layer -> id 4 is the MTP layer.
        _config(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            recompute_modules={"mlp": [0, 4]},
        )

    def test_negative_layer_id(self):
        with self.assertRaises(ValueError):
            _config(recompute_modules={"mlp": [-1]})

    def test_non_int_layer_id(self):
        with self.assertRaises(ValueError):
            _config(recompute_modules={"mlp": ["0"]})

    def test_bad_spec_type(self):
        with self.assertRaises(ValueError):
            _config(recompute_modules={"mlp": "every"})

    def test_count_spec_needs_a_method(self):
        with self.assertRaises(ValueError):
            _config(recompute_modules={"mlp": 3}, recompute_method=None)

    def test_bad_container_type(self):
        config = TransformerConfig(num_hidden_layers=8)
        config.recompute_modules = "mlp,core_attn"
        with self.assertRaises(ValueError):
            validate_recompute_modules(config)

    def test_list_mode_entries_must_be_str(self):
        config = TransformerConfig(num_hidden_layers=8)
        config.recompute_modules = [1]
        with self.assertRaises(ValueError):
            validate_recompute_modules(config)


if __name__ == "__main__":
    unittest.main()
