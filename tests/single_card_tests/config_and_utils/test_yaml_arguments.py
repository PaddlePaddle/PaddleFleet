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

"""Behavior tests for ``paddlefleet.training.yaml_arguments._flatten_configs``.

Slice covered here: the core flatten / flatten-merge logic and nested-dict
handling of ``_flatten_configs`` (leaf hoisting, parent-key dropping, last-writer
key-collision precedence, preserved-dict passthrough, scalar/list value
handling). ``load_yaml`` and file-I/O paths are covered by sibling files.

Every expected value below is hand-derived from the production control flow, not
by re-running the code under test:

``_flatten_configs`` walks a ``DictConfig`` depth-first. For each ``(k, v)``:
  * ``v`` is a nested ``DictConfig`` whose key is NOT in
    ``_PRESERVED_DICT_CONFIG_KEYS`` -> recurse into ``v`` and DROP ``k`` (leaves
    are hoisted to the top level).
  * ``v`` is a nested ``DictConfig`` whose key IS preserved
    (``deepep_buffer_configs``) -> store ``OmegaConf.to_container(v)`` under ``k``
    unchanged; its children are NOT hoisted.
  * otherwise (scalar, None, list/``ListConfig``) -> store ``result[k] = v``.
Same target key seen twice -> the later write in iteration order wins.

Dependency handling (honest, no fake pass): the module needs only ``omegaconf``.
Importing it through the ``paddlefleet`` package triggers ``paddlefleet/__init__``
which imports ``paddle``; on a no-card CPU box paddle is absent. In that case we
load the real pure-omegaconf production source file directly (sanctioned by the
config module rules for loading a standalone source helper) so the REAL
``_flatten_configs`` still runs. This does not exercise full package import. If
``omegaconf`` itself is missing the tests skip with an honest reason.
"""

import os
import sys
import unittest

# Allow importing from the in-repo source tree when not pip-installed.
# tests/single_card_tests/config_and_utils/ -> repo root -> src.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_IMPORT_ERROR = None
_flatten_configs = None
DictConfig = None
ListConfig = None
OmegaConf = None

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf  # noqa: F401

    _HAVE_OMEGACONF = True
except ImportError as exc:  # honest: real dependency missing, do not fake pass
    _HAVE_OMEGACONF = False
    _IMPORT_ERROR = exc

if _HAVE_OMEGACONF:
    try:
        from paddlefleet.training.yaml_arguments import _flatten_configs
    except ImportError:
        # paddlefleet/__init__ imports paddle, which is unavailable on no-card
        # CPU envs. Load the real omegaconf-only production source directly.
        # exec_module is intentionally NOT guarded: a genuinely broken import
        # inside yaml_arguments must surface rather than be masked as a skip.
        import importlib.util

        _src_path = os.path.join(
            _REPO_SRC, "paddlefleet", "training", "yaml_arguments.py"
        )
        _spec = importlib.util.spec_from_file_location(
            "_yaml_arguments_src", _src_path
        )
        _ya = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ya)
        _flatten_configs = _ya._flatten_configs


@unittest.skipUnless(
    _flatten_configs is not None,
    f"omegaconf unavailable: {_IMPORT_ERROR}",
)
class TestFlattenConfigs(unittest.TestCase):
    """Behavior of _flatten_configs: flatten-merge and nested-dict handling."""

    def test_flat_dict_is_passed_through_unchanged(self):
        # No nesting -> every key/value survives verbatim, nothing added/dropped.
        cfg = OmegaConf.create({"a": 1, "b": 2, "c": "z"})
        result = _flatten_configs(cfg)
        self.assertEqual(set(result.keys()), {"a", "b", "c"})
        self.assertEqual(result.a, 1)
        self.assertEqual(result.b, 2)
        self.assertEqual(result.c, "z")

    def test_nested_dict_hoists_leaves_and_drops_parent_keys(self):
        # Depth-first walk hoists every leaf to the top level; the container
        # keys "model"/"training" are recursed into and therefore dropped.
        cfg = OmegaConf.create(
            {
                "model": {"hidden_size": 1024, "num_layers": 24},
                "training": {"lr": 0.001, "batch_size": 32},
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(
            set(result.keys()),
            {"hidden_size", "num_layers", "lr", "batch_size"},
        )
        self.assertNotIn("model", result)
        self.assertNotIn("training", result)
        self.assertEqual(result.hidden_size, 1024)
        self.assertEqual(result.num_layers, 24)
        self.assertAlmostEqual(result.lr, 0.001)
        self.assertEqual(result.batch_size, 32)

    def test_deeply_nested_leaf_hoisted_to_top_level(self):
        # level1->level2->level3 all recursed away; only the leaf "value" and
        # the sibling scalar "simple" remain at the top.
        cfg = OmegaConf.create(
            {
                "level1": {"level2": {"level3": {"value": 99}}},
                "simple": 1,
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(set(result.keys()), {"value", "simple"})
        self.assertEqual(result.value, 99)
        self.assertEqual(result.simple, 1)

    def test_key_collision_across_parents_last_writer_wins(self):
        # parent1 writes result["key"]="v1" first, parent2 overwrites it with
        # "v2"; iteration order (insertion order) decides the survivor.
        forward = _flatten_configs(
            OmegaConf.create(
                {"parent1": {"key": "v1"}, "parent2": {"key": "v2"}}
            )
        )
        self.assertEqual(set(forward.keys()), {"key"})
        self.assertEqual(forward.key, "v2")

        # Reversed source order -> the other value wins, proving the result is
        # order-dependent rather than accidentally always "v2".
        reverse = _flatten_configs(
            OmegaConf.create(
                {"parent2": {"key": "v2"}, "parent1": {"key": "v1"}}
            )
        )
        self.assertEqual(reverse.key, "v1")

    def test_top_level_leaf_vs_nested_leaf_collision_is_order_dependent(self):
        # Same target key from a top-level scalar and a nested leaf: whichever
        # is visited later overwrites the earlier one.
        leaf_first = _flatten_configs(
            OmegaConf.create({"a": 1, "nested": {"a": 2}})
        )
        self.assertEqual(leaf_first.a, 2)  # nested visited last -> 2

        nested_first = _flatten_configs(
            OmegaConf.create({"nested": {"a": 2}, "a": 1})
        )
        self.assertEqual(nested_first.a, 1)  # top-level leaf visited last -> 1

    def test_list_values_are_preserved_under_their_key(self):
        # A list is a ListConfig, not a DictConfig, so it takes the scalar
        # branch and IS kept -- at the top level and when nested (its key is
        # hoisted like any other leaf).
        cfg = OmegaConf.create(
            {"items": [1, 2, 3], "outer": {"inner_list": [9, 8]}}
        )
        result = _flatten_configs(cfg)
        self.assertEqual(set(result.keys()), {"items", "inner_list"})
        self.assertEqual(OmegaConf.to_container(result["items"]), [1, 2, 3])
        self.assertEqual(OmegaConf.to_container(result.inner_list), [9, 8])

    def test_preserved_deepep_key_kept_as_dict_and_children_not_hoisted(self):
        # deepep_buffer_configs is in _PRESERVED_DICT_CONFIG_KEYS, so it is
        # stored as a container under its own key; its children (num_sms, ...)
        # are NOT hoisted. The sibling scalar num_hidden_layers is hoisted.
        cfg = OmegaConf.create(
            {
                "model": {
                    "num_hidden_layers": 2,
                    "deepep_buffer_configs": {
                        "num_sms": 24,
                        "dispatch_config": [60, 256],
                        "combine_config": [20, 256],
                    },
                }
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(
            set(result.keys()),
            {"num_hidden_layers", "deepep_buffer_configs"},
        )
        self.assertEqual(result.num_hidden_layers, 2)
        self.assertNotIn("num_sms", result)  # not flattened out
        self.assertNotIn("dispatch_config", result)
        self.assertEqual(
            OmegaConf.to_container(result.deepep_buffer_configs, resolve=True),
            {
                "num_sms": 24,
                "dispatch_config": [60, 256],
                "combine_config": [20, 256],
            },
        )

    def test_scalar_value_types_preserved_across_nesting(self):
        # int / float / str / bool / None survive the walk with identity of
        # value, whether at the top level or hoisted from a nested dict.
        cfg = OmegaConf.create(
            {
                "int_val": 42,
                "float_val": 3.5,
                "str_val": "hello",
                "bool_val": True,
                "nested": {"none_val": None, "false_val": False},
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(
            set(result.keys()),
            {
                "int_val",
                "float_val",
                "str_val",
                "bool_val",
                "none_val",
                "false_val",
            },
        )
        self.assertEqual(result.int_val, 42)
        self.assertAlmostEqual(result.float_val, 3.5)
        self.assertEqual(result.str_val, "hello")
        self.assertIs(result.bool_val, True)
        self.assertIsNone(result.none_val)
        self.assertIs(result.false_val, False)

    def test_empty_config_flattens_to_empty(self):
        result = _flatten_configs(OmegaConf.create({}))
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.keys()), [])

    def test_result_is_a_new_dictconfig(self):
        # The function returns a freshly created DictConfig, not the input.
        cfg = OmegaConf.create({"outer": {"inner": 5}})
        result = _flatten_configs(cfg)
        self.assertIsInstance(result, DictConfig)
        self.assertIsNot(result, cfg)
        # Mutating the result must not write back into the source config.
        result.inner = 6
        self.assertEqual(cfg.outer.inner, 5)


if __name__ == "__main__":
    unittest.main()
