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
"""Behavior tests for ``paddlefleet.training.yaml_arguments.load_yaml``.

Slice 5: the end-to-end ``load_yaml`` -> ``_flatten_configs`` pipeline over a
range of YAML shapes (empty, single scalar, deep nesting, multiple scalar
keys, list values, mixed numeric types). Every expected value below is
hand-derived from the flatten contract in the production source:

  * ``_flatten_configs`` walks the loaded ``DictConfig`` and, for every value
    that is itself a ``DictConfig``, RECURSES into it (dropping the parent
    key) unless the key is in ``_PRESERVED_DICT_CONFIG_KEYS``. Every non
    ``DictConfig`` leaf (scalars, lists) is written into a flat result under
    its OWN key. The final flat mapping is rebuilt with ``OmegaConf.create``.

So a nested tree collapses to only its leaf keys, list values survive intact,
and scalar Python types (str/int/float/bool) are preserved.

The module under test only depends on ``omegaconf``. Its parent package
``paddlefleet.training.__init__`` eagerly imports Fleet initialization (and
thus ``paddle``), which is not installed in the no-card CPU environment. To
exercise the REAL production function without that unrelated import side
effect, we load the actual source file directly via ``importlib`` (no mock of
the code under test). If ``omegaconf`` (a genuine collaborator) or the source
file is unavailable, we honestly skip.
"""

import importlib.util
import os
import tempfile
import unittest

_PROD_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
    "paddlefleet",
    "training",
    "yaml_arguments.py",
)

_LOAD_YAML = None
_OMEGACONF = None
_IMPORT_ERROR = None

try:
    import omegaconf as _OMEGACONF

    if not os.path.isfile(_PROD_PATH):
        raise ImportError(f"yaml_arguments.py source not found at {_PROD_PATH}")

    _spec = importlib.util.spec_from_file_location(
        "paddlefleet_yaml_arguments_under_test", _PROD_PATH
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _LOAD_YAML = _mod.load_yaml
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency skip only
    _IMPORT_ERROR = exc

_AVAILABLE = _LOAD_YAML is not None and _OMEGACONF is not None


@unittest.skipUnless(
    _AVAILABLE,
    f"omegaconf or yaml_arguments source unavailable: {_IMPORT_ERROR}",
)
class TestLoadYamlFlattenSlice5(unittest.TestCase):
    """load_yaml end-to-end flatten behavior (slice 5)."""

    def _load_yaml_from_text(self, text):
        """Write ``text`` to a temp .yaml, run the real ``load_yaml``, clean up.

        Returns the flattened config object produced by production code.
        """
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w") as f:
            f.write(text)
        return self._load_yaml(path)

    def _load_yaml(self, path):
        return _LOAD_YAML(path)

    def _as_container(self, cfg):
        return _OMEGACONF.OmegaConf.to_container(cfg, resolve=True)

    def test_load_yaml_empty_file(self):
        """An empty YAML file flattens to an empty mapping (no keys)."""
        cfg = self._load_yaml_from_text("")
        # Empty document -> empty DictConfig -> recurse over no items ->
        # OmegaConf.create({}) which is an empty mapping.
        self.assertIsInstance(cfg, _OMEGACONF.DictConfig)
        self.assertEqual(self._as_container(cfg), {})
        self.assertEqual(list(cfg.keys()), [])

    def test_load_yaml_single_value(self):
        """A single top-level scalar survives verbatim under its own key."""
        cfg = self._load_yaml_from_text("key: value\n")
        self.assertEqual(self._as_container(cfg), {"key": "value"})
        self.assertEqual(cfg["key"], "value")

    def test_load_yaml_deep_nesting_collapses_to_leaf(self):
        """Nested DictConfigs are recursed away; only the leaf key remains.

        For a:{b:{c:{d:42}}} the flatten drops parents a, b, c and keeps only
        the scalar leaf ``d``. This is the load-bearing behavior of
        _flatten_configs: intermediate mapping keys do not appear in the flat
        result.
        """
        cfg = self._load_yaml_from_text("a:\n  b:\n    c:\n      d: 42\n")
        self.assertEqual(self._as_container(cfg), {"d": 42})
        self.assertEqual(list(cfg.keys()), ["d"])
        self.assertEqual(cfg["d"], 42)
        for dropped in ("a", "b", "c"):
            self.assertNotIn(dropped, cfg)

    def test_load_yaml_sibling_leaf_name_collision_last_wins(self):
        """Distinct branches sharing a leaf name collide; the last write wins.

        a:{x:1} and b:{x:2} both flatten to key ``x``; because recursion writes
        result["x"]=1 then result["x"]=2, only the second value survives. This
        pins the flatten's key-collision semantics (dict iteration order, last
        assignment wins) rather than merely counting keys.
        """
        cfg = self._load_yaml_from_text("a:\n  x: 1\nb:\n  x: 2\n")
        self.assertEqual(self._as_container(cfg), {"x": 2})
        self.assertEqual(list(cfg.keys()), ["x"])
        self.assertEqual(cfg["x"], 2)

    def test_load_yaml_multiple_scalar_keys_and_types(self):
        """Multiple top-level scalars all survive with their parsed types."""
        cfg = self._load_yaml_from_text("key1: val1\nkey2: 42\nkey3: true\n")
        container = self._as_container(cfg)
        self.assertEqual(container, {"key1": "val1", "key2": 42, "key3": True})
        # Type identity is a separate contract from value equality
        # (1 == True in Python), so assert the concrete parsed types.
        self.assertIsInstance(container["key1"], str)
        self.assertIsInstance(container["key2"], int)
        self.assertIsInstance(container["key3"], bool)

    def test_load_yaml_list_value_preserved_in_order(self):
        """A list value is not a DictConfig, so it is kept intact and ordered.

        The three distinguishable, non-sorted-triggering elements a/b/c must
        appear under ``items`` in the original order, proving the list branch
        is preserved verbatim rather than flattened or reordered.
        """
        cfg = self._load_yaml_from_text("items:\n  - a\n  - b\n  - c\n")
        self.assertEqual(self._as_container(cfg), {"items": ["a", "b", "c"]})
        self.assertIsInstance(cfg["items"], _OMEGACONF.ListConfig)
        self.assertEqual(list(cfg["items"]), ["a", "b", "c"])

    def test_load_yaml_numeric_values_keep_int_and_float(self):
        """Numeric scalars keep distinct int vs float Python types."""
        cfg = self._load_yaml_from_text("int_val: 42\nfloat_val: 3.14\n")
        container = self._as_container(cfg)
        self.assertEqual(container, {"int_val": 42, "float_val": 3.14})
        self.assertIsInstance(container["int_val"], int)
        self.assertNotIsInstance(container["int_val"], bool)
        self.assertIsInstance(container["float_val"], float)


if __name__ == "__main__":
    unittest.main()
