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

"""Behavior tests for ``paddlefleet.training.yaml_arguments`` (config & run infra).

This file covers a slice that is *disjoint* from the flatten-core base file
(which drives plain scalar recursion / type-preservation of ``_flatten_configs``).
Here we exercise the behaviors that base does not:

* ``_flatten_configs`` -- the ``_PRESERVED_DICT_CONFIG_KEYS`` branch. A nested
  ``DictConfig`` whose *key* is ``"deepep_buffer_configs"`` must be kept as a
  nested sub-config (converted via ``OmegaConf.to_container(..., resolve=True)``)
  instead of being recursed into, so its inner keys are NOT hoisted to the top
  level. We check this both at the top level and when the preserved key sits
  under another section (the guard fires on the immediate dict key during
  recursion). We also pin the collision contract (later section overwrites an
  earlier leaf of the same name) and that ``list`` values are kept verbatim as
  leaves rather than being recursed into or dropped.
* ``load_yaml`` -- the full file-load pipeline: real temp-file write + read,
  section flattening, primitive-type preservation and the deepep preservation
  survive the ``OmegaConf.load`` -> ``_flatten_configs`` path end to end.

Every expected value is hand-derived from the branch logic in the module, not
read back from the config under test. The ``paddlefleet`` package imports
``paddle`` transitively at import time (via ``training.initialize``), so the
whole suite is skipped with an honest reason when Paddle / paddlefleet is
unavailable rather than reporting a hollow pass. All assertions concern locally
observable CPU behavior; no device numerics or multi-card semantics are claimed.
"""

import os
import sys
import tempfile
import unittest

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed (repo_root/src is 4 levels up from this file).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    # ``paddle`` is pulled in transitively by ``paddlefleet.__init__``; probing
    # it first yields a clean ImportError skip on hosts without Paddle, before
    # any other dependency is touched.
    import paddle  # noqa: F401
    from omegaconf import OmegaConf

    from paddlefleet.training.yaml_arguments import (
        _flatten_configs,
        load_yaml,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet / omegaconf not installed
    OmegaConf = None
    _flatten_configs = None
    load_yaml = None
    _IMPORT_ERROR = exc


_SKIP_REASON = f"paddle/paddlefleet/omegaconf unavailable: {_IMPORT_ERROR}"


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestFlattenConfigsPreservation(unittest.TestCase):
    """``_flatten_configs`` preserved-key / collision / list-leaf behavior."""

    def test_preserved_deepep_key_kept_nested_not_flattened(self):
        # "deepep_buffer_configs" is in _PRESERVED_DICT_CONFIG_KEYS, so its
        # sub-config is stored whole; "model" is a normal section and gets
        # flattened into its leaves.
        cfg = OmegaConf.create(
            {
                "deepep_buffer_configs": {
                    "num_max_dispatch_tokens_per_rank": 128,
                    "num_experts": 64,
                },
                "model": {"hidden_size": 1024, "num_layers": 12},
            }
        )
        result = _flatten_configs(cfg)

        self.assertEqual(
            set(result.keys()),
            {"deepep_buffer_configs", "hidden_size", "num_layers"},
        )
        # model section flattened to top level.
        self.assertEqual(result.hidden_size, 1024)
        self.assertEqual(result.num_layers, 12)
        # deepep sub-config preserved as a nested mapping.
        self.assertEqual(
            result.deepep_buffer_configs.num_max_dispatch_tokens_per_rank, 128
        )
        self.assertEqual(result.deepep_buffer_configs.num_experts, 64)
        # its inner keys are NOT hoisted (would be, if it had been recursed).
        self.assertNotIn("num_max_dispatch_tokens_per_rank", result)
        self.assertNotIn("num_experts", result)

    def test_preserved_key_detected_at_deeper_nesting(self):
        # The preserved-key guard fires on the immediate dict key encountered
        # during recursion, so it also protects a deepep block nested under
        # another (non-preserved) section.
        cfg = OmegaConf.create(
            {
                "runtime": {
                    "deepep_buffer_configs": {"num_sms": 20},
                    "seed": 42,
                },
            }
        )
        result = _flatten_configs(cfg)

        self.assertEqual(set(result.keys()), {"deepep_buffer_configs", "seed"})
        self.assertEqual(result.seed, 42)
        self.assertEqual(result.deepep_buffer_configs.num_sms, 20)
        # neither the wrapping section nor the preserved inner key leak out.
        self.assertNotIn("runtime", result)
        self.assertNotIn("num_sms", result)

    def test_duplicate_leaf_key_last_section_overwrites(self):
        # Flattening iterates sections in insertion order and writes each leaf
        # into one flat dict, so a later section's leaf overwrites an earlier
        # same-named leaf (observed, lossy-by-design contract).
        cfg = OmegaConf.create(
            {
                "first": {"shared": 1, "only_first": 10},
                "second": {"shared": 2, "only_second": 20},
            }
        )
        result = _flatten_configs(cfg)

        self.assertEqual(
            set(result.keys()), {"shared", "only_first", "only_second"}
        )
        self.assertEqual(result.shared, 2)  # "second" wins over "first"
        self.assertEqual(result.only_first, 10)
        self.assertEqual(result.only_second, 20)

    def test_list_value_kept_as_leaf(self):
        # A list value is not a DictConfig, so it is stored verbatim as a leaf
        # (not recursed into, not dropped).
        cfg = OmegaConf.create(
            {"optimizer": {"betas": [0.9, 0.999], "name": "adamw"}}
        )
        result = _flatten_configs(cfg)

        self.assertEqual(set(result.keys()), {"betas", "name"})
        betas = list(result.betas)
        self.assertEqual(len(betas), 2)
        self.assertAlmostEqual(betas[0], 0.9)
        self.assertAlmostEqual(betas[1], 0.999)
        self.assertEqual(result.name, "adamw")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestLoadYaml(unittest.TestCase):
    """``load_yaml`` file-load pipeline: flatten + type/deepep preservation."""

    def _write_yaml(self, text):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(path, "w") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_load_yaml_flattens_and_preserves_types(self):
        path = self._write_yaml(
            "model:\n"
            "  hidden_size: 768\n"
            "  name: gpt\n"
            "training:\n"
            "  lr: 0.001\n"
            "  use_amp: true\n"
            "  max_steps: 5000\n"
        )
        result = load_yaml(path)

        self.assertEqual(
            set(result.keys()),
            {"hidden_size", "name", "lr", "use_amp", "max_steps"},
        )
        # section wrappers are gone after flattening.
        self.assertNotIn("model", result)
        self.assertNotIn("training", result)

        # int stays int (and not misparsed as bool).
        self.assertEqual(result.hidden_size, 768)
        self.assertIsInstance(result.hidden_size, int)
        self.assertNotIsInstance(result.hidden_size, bool)
        self.assertEqual(result.max_steps, 5000)

        # str stays str.
        self.assertEqual(result.name, "gpt")

        # float stays float.
        self.assertAlmostEqual(result.lr, 0.001)
        self.assertIsInstance(result.lr, float)

        # YAML "true" becomes the bool True (not the string "true").
        self.assertIs(result.use_amp, True)

    def test_load_yaml_preserves_deepep_buffer_configs(self):
        path = self._write_yaml(
            "model:\n"
            "  hidden_size: 512\n"
            "deepep_buffer_configs:\n"
            "  num_max_dispatch_tokens_per_rank: 256\n"
            "  num_sms: 24\n"
        )
        result = load_yaml(path)

        self.assertEqual(
            set(result.keys()), {"hidden_size", "deepep_buffer_configs"}
        )
        self.assertEqual(result.hidden_size, 512)
        # deepep block survives the full load -> flatten pipeline intact.
        self.assertEqual(
            result.deepep_buffer_configs.num_max_dispatch_tokens_per_rank, 256
        )
        self.assertEqual(result.deepep_buffer_configs.num_sms, 24)
        self.assertNotIn("num_max_dispatch_tokens_per_rank", result)
        self.assertNotIn("num_sms", result)

    def test_load_yaml_duplicate_key_last_section_wins(self):
        path = self._write_yaml(
            "section_a:\n  dropout: 0.1\nsection_b:\n  dropout: 0.3\n"
        )
        result = load_yaml(path)

        self.assertEqual(set(result.keys()), {"dropout"})
        # section_b appears after section_a, so 0.3 overwrites 0.1.
        self.assertAlmostEqual(result.dropout, 0.3)


if __name__ == "__main__":
    unittest.main()
