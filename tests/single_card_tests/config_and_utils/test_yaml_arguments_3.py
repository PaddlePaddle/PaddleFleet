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

"""Behavior tests for ``paddlefleet.training.yaml_arguments`` (THIRD slice).

Module under test: ``src/paddlefleet/training/yaml_arguments.py`` -- a config
loader that reads a YAML file (``load_yaml``) and flattens its nested
``DictConfig`` into a single-level config (``_flatten_configs``), while keeping
any key in ``_PRESERVED_DICT_CONFIG_KEYS`` as a nested container.

Scope. Siblings cover the same module concurrently. This slice restricts itself
to the helpers exercised by its coverage source: ``_flatten_configs`` handling
of ``None`` leaves and empty nested dicts, and ``load_yaml`` on the more complex
YAML value forms (null, quoted strings, literal block scalars, merge-key
anchors, scientific notation). Every expected value is hand-derived from the
source recursion / YAML semantics, not read back from the function under test.

Loading boundary (no-card config test, per the repo unit-test rules). The
``paddlefleet`` package ``__init__`` imports ``paddle``, which is not installed
in this environment, so importing the module through the package is impossible
here. ``yaml_arguments.py`` itself depends only on ``omegaconf``. We therefore
load the production source directly by file path as an independent source
helper. This exercises the genuine production functions, but does NOT verify
package-level import or full startup -- that path needs ``paddle`` and belongs
to an environment where it is installed.
"""

import importlib.util
import os
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_UUT_PATH = os.path.join(
    _REPO_ROOT, "src", "paddlefleet", "training", "yaml_arguments.py"
)

try:
    # omegaconf is a genuine collaborator and is also required by the UUT.
    from omegaconf import OmegaConf

    _spec = importlib.util.spec_from_file_location(
        "paddlefleet_yaml_arguments_uut", _UUT_PATH
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _flatten_configs = _mod._flatten_configs
    load_yaml = _mod.load_yaml
    HAVE_DEPS = True
    _SKIP_REASON = ""
except ImportError as exc:  # honest: missing dependency only
    HAVE_DEPS = False
    _SKIP_REASON = f"omegaconf/UUT source not importable: {exc}"
    OmegaConf = None


@unittest.skipUnless(HAVE_DEPS, _SKIP_REASON)
class TestFlattenConfigsNoneAndEmpty(unittest.TestCase):
    """_flatten_configs behavior for None leaves and empty nested dicts."""

    def test_none_leaf_preserved_and_wrapper_flattened(self):
        # Top key "model" is a DictConfig -> recursion descends and its two
        # leaves are promoted to the top level. None is a non-DictConfig leaf,
        # so it is stored verbatim. The "model" wrapper key must disappear.
        cfg = OmegaConf.create({"model": {"name": None, "size": 10}})
        result = _flatten_configs(cfg)

        self.assertEqual(
            OmegaConf.to_container(result, resolve=True),
            {"name": None, "size": 10},
        )
        self.assertIsNone(result.name)
        self.assertEqual(result.size, 10)
        self.assertNotIn("model", result)

    def test_empty_nested_contributes_nothing_but_leaf_sibling_survives(self):
        # "model" is an empty DictConfig: recursion finds no items and adds no
        # keys. The sibling scalar "lr" is a leaf and is promoted. Exact-content
        # comparison (not just len) pins down that only "lr" survives.
        cfg = OmegaConf.create({"model": {}, "lr": 0.5})
        result = _flatten_configs(cfg)

        self.assertEqual(
            OmegaConf.to_container(result, resolve=True), {"lr": 0.5}
        )
        self.assertNotIn("model", result)
        self.assertEqual(result.lr, 0.5)

    def test_leaf_key_collision_across_branches_last_write_wins(self):
        # Two sibling DictConfigs both carry a leaf named "x". Flattening lifts
        # both to the same top-level key; iteration order is insertion order,
        # so branch "b" (later) overwrites branch "a". Result is {"x": 2}.
        cfg = OmegaConf.create({"a": {"x": 1}, "b": {"x": 2}})
        result = _flatten_configs(cfg)

        self.assertEqual(OmegaConf.to_container(result, resolve=True), {"x": 2})
        self.assertEqual(result.x, 2)


@unittest.skipUnless(HAVE_DEPS, _SKIP_REASON)
class TestLoadYamlComplexStructures(unittest.TestCase):
    """load_yaml over complex YAML value forms, with flattening applied."""

    def _load(self, yaml_content):
        # Real temp file -> real load_yaml -> real flatten. Cleaned up even on
        # assertion failure via addCleanup.
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w") as f:
            f.write(yaml_content)
        return load_yaml(path)

    def test_null_scalar_becomes_none(self):
        result = self._load("model:\n  name: null\n  size: 10\n")
        self.assertEqual(
            OmegaConf.to_container(result, resolve=True),
            {"name": None, "size": 10},
        )
        self.assertIsNone(result.name)
        self.assertEqual(result.size, 10)

    def test_quoted_strings_keep_exact_text(self):
        result = self._load(
            "model:\n  name: \"my-model-name\"\n  path: '/path/to/model'\n"
        )
        self.assertEqual(
            OmegaConf.to_container(result, resolve=True),
            {"name": "my-model-name", "path": "/path/to/model"},
        )

    def test_literal_block_scalar_keeps_newlines_and_trailing_newline(self):
        # YAML "|" is a literal block scalar: the common 4-space indent is
        # stripped, interior newlines are kept, and clip mode leaves exactly one
        # trailing newline. Hand-derived expected string below.
        result = self._load(
            "model:\n"
            "  description: |\n"
            "    This is a long\n"
            "    description that spans\n"
            "    multiple lines.\n"
        )
        self.assertEqual(
            result.description,
            "This is a long\ndescription that spans\nmultiple lines.\n",
        )

    def test_merge_key_anchor_expands_then_flattens(self):
        # &defaults defines batch_size/learning_rate. "training" pulls them in
        # via the "<<" merge key and adds epochs. Both "defaults" and "training"
        # are DictConfigs, so flattening lifts all leaves to the top level and
        # drops both wrapper keys.
        result = self._load(
            "defaults: &defaults\n"
            "  batch_size: 32\n"
            "  learning_rate: 0.001\n"
            "\n"
            "training:\n"
            "  <<: *defaults\n"
            "  epochs: 10\n"
        )
        self.assertEqual(
            set(result.keys()), {"batch_size", "learning_rate", "epochs"}
        )
        self.assertNotIn("defaults", result)
        self.assertNotIn("training", result)
        self.assertEqual(result.batch_size, 32)
        self.assertAlmostEqual(result.learning_rate, 0.001)
        self.assertEqual(result.epochs, 10)

    def test_scientific_notation_parsed_as_floats(self):
        # "1.0e-4" and "5.0e-2" are valid YAML 1.1 floats (dot + signed
        # exponent), so they must load as float, not str. Values hand-derived:
        # 1.0e-4 == 0.0001, 5.0e-2 == 0.05.
        result = self._load("training:\n  lr: 1.0e-4\n  weight_decay: 5.0e-2\n")
        self.assertIsInstance(result.lr, float)
        self.assertIsInstance(result.weight_decay, float)
        self.assertAlmostEqual(result.lr, 1.0e-4)
        self.assertAlmostEqual(result.weight_decay, 5.0e-2)


if __name__ == "__main__":
    unittest.main()
