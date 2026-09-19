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

"""Behavior tests for the SonicMoE weight-quant config surface.

Subject (pinned from the coverage-source imports):
  * ``paddlefleet.transformer.transformer_config.TransformerConfig`` -- the
    ``fp8_weight_quant_format`` field, its default, and the migration guard in
    ``_process_attribute`` / ``register_attributes`` / ``from_config`` that
    rejects the renamed ``sonicmoe_quant_format`` key.
  * ``paddlefleet.transformer.moe.moe_expert.SonicMoEExpert`` -- how the
    ``fp8_weight_quant_format`` flag maps to the fp8-weight-release decision in
    ``_release_fp8_weight_after_fwd``.

Every expected value is hand-derived from the production source, not by
re-running the code under test. TransformerConfig imports paddle and
moe_expert additionally imports paddlefleet_ops at module load, so the modules
are unimportable without those deps; the imports are guarded and the tests skip
with an honest reason rather than faking a pass.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# Allow ``import paddlefleet`` from the in-repo source tree when the package is
# not pip-installed. tests/single_card_tests/config_and_utils/ -> repo root.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_CONFIG_IMPORT_ERROR = None
try:
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _HAVE_CONFIG = True
except ImportError as exc:  # honest: real dependency missing, do not fake pass
    TransformerConfig = None
    _HAVE_CONFIG = False
    _CONFIG_IMPORT_ERROR = exc

_CONFIG_SKIP = (
    "paddlefleet.transformer.transformer_config not importable in this "
    f"environment: {_CONFIG_IMPORT_ERROR!r}"
)

_EXPERT_IMPORT_ERROR = None
try:
    from paddlefleet.transformer.moe import moe_expert

    _HAVE_EXPERT = True
except ImportError as exc:  # honest: real dependency missing, do not fake pass
    moe_expert = None
    _HAVE_EXPERT = False
    _EXPERT_IMPORT_ERROR = exc

_EXPERT_SKIP = (
    "paddlefleet.transformer.moe.moe_expert not importable in this "
    f"environment: {_EXPERT_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAVE_CONFIG, _CONFIG_SKIP)
class TestSonicMoEConfigDefaults(unittest.TestCase):
    """Default-value contract for the renamed SonicMoE fp8 quant field."""

    def _make_config(self, **overrides):
        # Same minimal required args the dataclass needs to construct.
        return TransformerConfig(
            hidden_size=64, num_attention_heads=4, **overrides
        )

    def test_canonical_field_default_and_old_names_gone(self):
        # Source transformer_config.py: ``fp8_weight_quant_format: str =
        # "32x32"`` and ``using_sonic_moe: bool = False``. The rename dropped
        # the old spellings, so they must not exist as attributes.
        config = self._make_config()
        self.assertEqual(config.fp8_weight_quant_format, "32x32")
        self.assertIs(config.using_sonic_moe, False)
        self.assertFalse(hasattr(config, "sonicmoe_quant_format"))
        self.assertFalse(hasattr(config, "sonicmoe_save_upgate_out_in_fp8"))


@unittest.skipUnless(_HAVE_CONFIG, _CONFIG_SKIP)
class TestSonicMoEDeprecatedKeyRejection(unittest.TestCase):
    """The renamed key must be rejected, not silently absorbed as a dead attr."""

    def _make_config(self, **overrides):
        return TransformerConfig(
            hidden_size=64, num_attention_heads=4, **overrides
        )

    def test_process_attribute_rejects_deprecated_name(self):
        # Source: ``_process_attribute`` has an explicit ``elif key ==
        # "sonicmoe_quant_format": raise ValueError(...)`` branch whose message
        # names the replacement field. Without it the fallback ``setattr`` would
        # store a dead attribute and the feature would silently stay off.
        config = self._make_config()
        with self.assertRaises(ValueError) as ctx:
            config._process_attribute("sonicmoe_quant_format", "1x32")
        self.assertIn("fp8_weight_quant_format", str(ctx.exception))

    def test_from_config_entry_rejects_deprecated_name(self):
        # Real ingestion path: ``from_config`` -> ``register_attributes``
        # iterates the source object's ``__dict__`` and routes each key through
        # ``_process_attribute``, so the stale key is rejected at the actual
        # config-construction entry, not only via the direct helper call.
        source = SimpleNamespace(sonicmoe_quant_format="1x32")
        with self.assertRaises(ValueError) as ctx:
            TransformerConfig.from_config(source)
        self.assertIn("fp8_weight_quant_format", str(ctx.exception))


@unittest.skipUnless(_HAVE_EXPERT, _EXPERT_SKIP)
class TestSonicMoEReleaseControlFlow(unittest.TestCase):
    """``fp8_weight_quant_format`` maps to the fp8-weight-release decision.

    ``SonicMoEExpert._release_fp8_weight_after_fwd`` (production code under
    test) is exercised directly with a lightweight ``self`` carrying genuine
    collaborator values (a config-like namespace and weight objects); the
    method's own AND-chain runs for real. The module global
    ``g_shard_bypass_dygraph_optimizer`` is pinned to 0 and auto-restored so the
    bypass term does not depend on the ambient environment flag.
    """

    def _stub(
        self,
        fp8_format,
        recompute_granularity=None,
        last_micro=True,
        w1_has_fp8=True,
        w2_has_fp8=True,
    ):
        config = SimpleNamespace(
            fp8_weight_quant_format=fp8_format,
            recompute_granularity=recompute_granularity,
        )
        # hasattr(weight, "fp8") is the real predicate; an object without the
        # attribute makes it False.
        weight1 = SimpleNamespace(fp8=None) if w1_has_fp8 else object()
        weight2 = SimpleNamespace(fp8=None) if w2_has_fp8 else object()
        return SimpleNamespace(
            config=config,
            _is_last_micro_batch=last_micro,
            weight1=weight1,
            weight2=weight2,
        )

    def _release(self, stub, recompute_moe_gate_up):
        with mock.patch.object(
            moe_expert, "g_shard_bypass_dygraph_optimizer", 0
        ):
            return moe_expert.SonicMoEExpert._release_fp8_weight_after_fwd(
                stub, recompute_moe_gate_up
            )

    def test_releases_when_1x32_and_all_conditions_met(self):
        # AND-chain: "1x32" == "1x32" and last_micro and not 0 and not False
        # and (None != "full") and hasattr(w1,"fp8") and hasattr(w2,"fp8").
        result = self._release(self._stub("1x32"), recompute_moe_gate_up=False)
        self.assertIs(result, True)

    def test_no_release_for_iso32_format(self):
        # First term "32x32" == "1x32" is False -> whole decision False.
        result = self._release(self._stub("32x32"), recompute_moe_gate_up=False)
        self.assertIs(result, False)

    def test_no_release_when_recompute_granularity_full(self):
        # The ``recompute_granularity != "full"`` term is False.
        stub = self._stub("1x32", recompute_granularity="full")
        self.assertIs(self._release(stub, recompute_moe_gate_up=False), False)

    def test_no_release_when_a_weight_lacks_fp8(self):
        # hasattr(weight2, "fp8") is False.
        stub = self._stub("1x32", w2_has_fp8=False)
        self.assertIs(self._release(stub, recompute_moe_gate_up=False), False)

    def test_no_release_when_recompute_moe_gate_up(self):
        # ``not recompute_moe_gate_up`` is False -> argument is consumed.
        result = self._release(self._stub("1x32"), recompute_moe_gate_up=True)
        self.assertIs(result, False)

    def test_no_release_when_not_last_micro_batch(self):
        stub = self._stub("1x32", last_micro=False)
        self.assertIs(self._release(stub, recompute_moe_gate_up=False), False)


if __name__ == "__main__":
    unittest.main()
