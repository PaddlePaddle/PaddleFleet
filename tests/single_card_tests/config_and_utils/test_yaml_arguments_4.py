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

"""Behavior tests for paddlefleet.training.yaml_arguments (slice 4).

Covers ``_flatten_configs`` and ``load_yaml``: the config-flattening entry that
hoists every leaf of a nested OmegaConf tree to the top level, keeps the keys in
``_PRESERVED_DICT_CONFIG_KEYS`` (currently ``deepep_buffer_configs``) as intact
nested containers, and the on-disk ``load_yaml`` path that parses a YAML file
and applies the same flattening.

Every expected value below is hand-derived from the flattening contract (leaf
hoisting, last-writer-wins on key collision, preserved-key passthrough), not by
re-running the code under test.

Import strategy: ``paddlefleet/training/__init__.py`` imports ``initialize_fleet``
which pulls in ``paddle`` at package-import time, so the normal package import is
unavailable without paddle. ``yaml_arguments.py`` itself only depends on
``omegaconf``. We therefore try the real package import first and, if it fails
because paddle is absent, fall back to loading the same production source file
directly by path. This runs the genuine production functions on CPU; it does
NOT verify package-level import or Fleet startup. If omegaconf/the source cannot
be loaded, the tests skip with an honest reason rather than faking a pass.
"""

import importlib.util
import os
import sys
import tempfile
import unittest

# Allow ``import paddlefleet`` from the in-repo source tree when the package is
# not pip-installed. tests/single_card_tests/config_and_utils/ -> repo root.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_IMPORT_ERROR = None
load_yaml = None
_flatten_configs = None
_PRESERVED_DICT_CONFIG_KEYS = None
DictConfig = None
OmegaConf = None

try:
    from omegaconf import DictConfig, OmegaConf

    try:
        # Preferred: real package entry.
        from paddlefleet.training.yaml_arguments import (
            _PRESERVED_DICT_CONFIG_KEYS,
            _flatten_configs,
            load_yaml,
        )
    except ImportError:
        # Package __init__ pulls in paddle; load the same production source
        # file directly (it only needs omegaconf). Honest fallback, not a skip.
        _SRC = os.path.join(
            _REPO_SRC, "paddlefleet", "training", "yaml_arguments.py"
        )
        _spec = importlib.util.spec_from_file_location(
            "_yaml_arguments_under_test", _SRC
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        load_yaml = _mod.load_yaml
        _flatten_configs = _mod._flatten_configs
        _PRESERVED_DICT_CONFIG_KEYS = _mod._PRESERVED_DICT_CONFIG_KEYS
    _HAVE_MODULE = True
except Exception as exc:
    _IMPORT_ERROR = exc
    _HAVE_MODULE = False


@unittest.skipUnless(
    _HAVE_MODULE,
    f"omegaconf/yaml_arguments source unavailable: {_IMPORT_ERROR}",
)
class TestFlattenConfigs(unittest.TestCase):
    """_flatten_configs: leaf hoisting, collisions, preserved keys."""

    def test_flatten_configs_simple_keeps_scalars(self):
        """Flat input keeps every scalar under its own key, values intact."""
        cfg = OmegaConf.create(
            {"batch_size": 32, "learning_rate": 0.001, "name": "run"}
        )
        result = _flatten_configs(cfg)
        self.assertEqual(
            set(result.keys()), {"batch_size", "learning_rate", "name"}
        )
        self.assertEqual(result.batch_size, 32)
        self.assertAlmostEqual(result.learning_rate, 0.001)
        self.assertEqual(result.name, "run")

    def test_flatten_configs_nested_hoists_leaves(self):
        """Nested leaves are hoisted to top level; parent keys disappear."""
        cfg = OmegaConf.create(
            {
                "training": {"batch_size": 32, "lr": 0.001},
                "model": {"hidden_size": 768},
            }
        )
        result = _flatten_configs(cfg)
        # "training"/"model" are gone; their leaves are now top-level.
        self.assertEqual(
            set(result.keys()), {"batch_size", "lr", "hidden_size"}
        )
        self.assertEqual(result.batch_size, 32)
        self.assertAlmostEqual(result.lr, 0.001)
        self.assertEqual(result.hidden_size, 768)

    def test_flatten_configs_key_collision_last_writer_wins(self):
        """Duplicate leaf key across subtrees: later subtree overwrites."""
        cfg = OmegaConf.create(
            {
                "first": {"shared": 100, "only_a": 1},
                "second": {"shared": 200, "only_b": 2},
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(set(result.keys()), {"shared", "only_a", "only_b"})
        # "second" is iterated after "first", so its value wins.
        self.assertEqual(result.shared, 200)
        self.assertEqual(result.only_a, 1)
        self.assertEqual(result.only_b, 2)

    def test_flatten_configs_preserves_deepep_buffer_configs(self):
        """Preserved key is kept as an intact nested container, not flattened."""
        self.assertIn("deepep_buffer_configs", _PRESERVED_DICT_CONFIG_KEYS)
        cfg = OmegaConf.create(
            {
                "deepep_buffer_configs": {
                    "num_nvl_bytes": 1000,
                    "nested": {"deep_value": 5},
                },
                "training": {"batch_size": 16},
            }
        )
        result = _flatten_configs(cfg)
        self.assertEqual(
            set(result.keys()), {"deepep_buffer_configs", "batch_size"}
        )
        self.assertEqual(result.batch_size, 16)
        # The preserved subtree keeps its full nested structure.
        self.assertEqual(result.deepep_buffer_configs.num_nvl_bytes, 1000)
        self.assertEqual(result.deepep_buffer_configs.nested.deep_value, 5)
        # Its inner keys must NOT be hoisted to the top level.
        self.assertNotIn("num_nvl_bytes", result)
        self.assertNotIn("nested", result)
        self.assertNotIn("deep_value", result)

    def test_flatten_configs_list_leaf_preserved_under_key(self):
        """A list leaf is not a DictConfig, so it is hoisted verbatim."""
        cfg = OmegaConf.create({"outer": {"layers": [4, 8, 16]}})
        result = _flatten_configs(cfg)
        self.assertEqual(set(result.keys()), {"layers"})
        self.assertEqual(OmegaConf.to_container(result.layers), [4, 8, 16])


@unittest.skipUnless(
    _HAVE_MODULE,
    f"omegaconf/yaml_arguments source unavailable: {_IMPORT_ERROR}",
)
class TestLoadYaml(unittest.TestCase):
    """load_yaml: parse a real YAML file then apply the same flattening."""

    def _write_yaml(self, text):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_load_yaml_flattens_and_reads_values(self):
        """Nested YAML is loaded and flattened; leaf values read back exactly."""
        path = self._write_yaml(
            "training:\n"
            "  batch_size: 64\n"
            "  learning_rate: 0.01\n"
            "model:\n"
            "  hidden_size: 512\n"
            "  num_layers: 8\n"
        )
        config = load_yaml(path)
        self.assertEqual(
            set(config.keys()),
            {"batch_size", "learning_rate", "hidden_size", "num_layers"},
        )
        self.assertEqual(config.batch_size, 64)
        self.assertAlmostEqual(config.learning_rate, 0.01)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.num_layers, 8)

    def test_load_yaml_returns_dictconfig_with_value(self):
        """load_yaml returns an OmegaConf DictConfig carrying the parsed value."""
        path = self._write_yaml("key: value\n")
        config = load_yaml(path)
        self.assertIsInstance(config, DictConfig)
        self.assertEqual(config.key, "value")

    def test_load_yaml_preserves_deepep_buffer_configs(self):
        """Preserved key survives the on-disk load path as a nested container."""
        path = self._write_yaml(
            "deepep_buffer_configs:\n"
            "  num_nvl_bytes: 2048\n"
            "  num_rdma_bytes: 4096\n"
            "training:\n"
            "  batch_size: 8\n"
        )
        config = load_yaml(path)
        self.assertEqual(
            set(config.keys()), {"deepep_buffer_configs", "batch_size"}
        )
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.deepep_buffer_configs.num_nvl_bytes, 2048)
        self.assertEqual(config.deepep_buffer_configs.num_rdma_bytes, 4096)
        self.assertNotIn("num_nvl_bytes", config)


if __name__ == "__main__":
    unittest.main()
