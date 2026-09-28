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

"""Behavior tests for the "hash" logic of ``TransformerConfig``.

IMPORTANT scoping note. There is no ``__hash__`` / fingerprint method on
``TransformerConfig`` and no config-hash contract (no equal-config -> equal-hash
relation exists to test). The only "hash" behavior in
``paddlefleet.transformer.transformer_config`` is *hash-based MoE routing*
validation performed in ``TransformerConfig.__post_init__`` and gated on
``moe_n_hash_layers > 0``. These tests therefore assert that validation
contract, exercised through the real constructor (which runs the real
``__post_init__``), not via ``__new__`` + manual attribute stuffing.

The contract, hand-derived from the production source, is:

* When ``moe_n_hash_layers == 0`` (default) the entire hash-routing validation
  block is skipped -- ``actual_vocab_size`` / ``scoring_func`` (which are
  validated nowhere else) and the routing-count relations are all ignored.
* When ``moe_n_hash_layers > 0`` the following each raise ``ValueError``:
    - ``first_k_dense_replace`` set and > 0 (mutually exclusive with hashing);
    - ``actual_vocab_size`` is None;
    - ``actual_vocab_size`` <= 0;
    - ``moe_n_hash_layers`` > ``num_hidden_layers``;
    - ``scoring_func`` not in {"softmax", "sigmoid", "sqrtsoftplus"};
    - ``num_experts_per_tok`` (top-k) is None or <= 0;
    - ``n_routed_experts`` is None or < ``num_experts_per_tok``.

Each expected relation below is derived from that source, never by running the
code and asserting its own output. ``assertRaisesRegex`` pins the *specific*
hash-routing message so an unrelated ``ValueError`` from another
``__post_init__`` branch cannot make a test pass for the wrong reason.

The production module does ``import paddle.nn.functional`` at import time, so it
is unimportable without paddle installed. The import is guarded and the tests
are skipped with an honest reason in that case rather than faking a pass.
"""

import os
import sys
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
try:
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _HAVE_MODULE = True
except ImportError as exc:  # honest: real dependency missing, do not fake pass
    TransformerConfig = None
    _HAVE_MODULE = False
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.transformer.transformer_config not importable in this "
    f"environment: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAVE_MODULE, _SKIP_REASON)
class TestHashRoutingValidationContract(unittest.TestCase):
    """Assert the ``__post_init__`` hash-routing validation contract."""

    # A minimal skeleton that is known to construct with hash routing
    # disabled; hash-specific fields are layered on top per test.
    _SKELETON = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "num_hidden_layers": 8,
    }

    def _valid_hash_kwargs(self, **overrides):
        """Kwargs for a config whose hash-routing checks all pass."""
        kwargs = dict(self._SKELETON)
        kwargs.update(
            moe_n_hash_layers=1,
            actual_vocab_size=128,
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
        )
        kwargs.update(overrides)
        return kwargs

    def _gate_off_kwargs(self, **overrides):
        """Skeleton with hash routing disabled (moe_n_hash_layers == 0)."""
        kwargs = dict(self._SKELETON)
        kwargs.update(moe_n_hash_layers=0)
        kwargs.update(overrides)
        return kwargs

    # -- positive branch -----------------------------------------------------

    def test_valid_hash_routing_config_constructs(self):
        """A fully consistent hash-routing config must not raise."""
        # No exception is the observable contract of the valid branch. We do
        # not read back the fields we set (that would be self-referential).
        TransformerConfig(**self._valid_hash_kwargs())

    # -- fields that are validated ONLY because hashing is enabled ----------
    # Each of these proves both directions: the invalid value raises when the
    # hash gate is on, and the *same* invalid value is accepted when the gate
    # is off (moe_n_hash_layers == 0), i.e. the field is inside the hash
    # contract and excluded from it otherwise.

    def test_missing_actual_vocab_size_gated_by_hash_layers(self):
        with self.assertRaisesRegex(
            ValueError, r"actual_vocab_size must be set when moe_n_hash_layers"
        ):
            TransformerConfig(**self._valid_hash_kwargs(actual_vocab_size=None))
        # Gate off: actual_vocab_size=None (the default) is fine.
        TransformerConfig(**self._gate_off_kwargs(actual_vocab_size=None))

    def test_nonpositive_actual_vocab_size_gated_by_hash_layers(self):
        with self.assertRaisesRegex(
            ValueError, r"actual_vocab_size must be positive"
        ):
            TransformerConfig(**self._valid_hash_kwargs(actual_vocab_size=-1))
        # Gate off: the same nonsensical value is not validated at all.
        TransformerConfig(**self._gate_off_kwargs(actual_vocab_size=-1))

    def test_invalid_scoring_func_gated_by_hash_layers(self):
        with self.assertRaisesRegex(
            ValueError, r"Hash routing requires scoring_func"
        ):
            TransformerConfig(**self._valid_hash_kwargs(scoring_func="relu"))
        # Gate off: scoring_func is validated nowhere else, so "relu" is OK.
        TransformerConfig(**self._gate_off_kwargs(scoring_func="relu"))

    def test_first_k_dense_replace_conflicts_only_under_hash_gate(self):
        with self.assertRaisesRegex(ValueError, r"mutually\s+exclusive"):
            TransformerConfig(
                **self._valid_hash_kwargs(first_k_dense_replace=2)
            )
        # Gate off: first_k_dense_replace just shapes moe_layer_freq, no error.
        TransformerConfig(**self._gate_off_kwargs(first_k_dense_replace=2))

    def test_allowed_scoring_funcs_do_not_raise(self):
        # sigmoid and sqrtsoftplus are the other two members of the allowed
        # set; softmax is already covered by the valid-config test.
        for func in ("sigmoid", "sqrtsoftplus"):
            with self.subTest(scoring_func=func):
                TransformerConfig(**self._valid_hash_kwargs(scoring_func=func))

    # -- routing-count relations (boundary-sensitive) -----------------------

    def test_hash_layers_cannot_exceed_num_hidden_layers(self):
        # Boundary: equal to num_hidden_layers is allowed (not '>').
        TransformerConfig(
            **self._valid_hash_kwargs(moe_n_hash_layers=8, num_hidden_layers=8)
        )
        # One past the boundary raises.
        with self.assertRaisesRegex(ValueError, r"cannot exceed"):
            TransformerConfig(
                **self._valid_hash_kwargs(
                    moe_n_hash_layers=9, num_hidden_layers=8
                )
            )

    def test_nonpositive_top_k_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"top-k\).*must be a positive integer"
        ):
            TransformerConfig(**self._valid_hash_kwargs(num_experts_per_tok=0))

    def test_routed_experts_below_top_k_relation(self):
        # Boundary: n_routed_experts == num_experts_per_tok is allowed.
        TransformerConfig(
            **self._valid_hash_kwargs(n_routed_experts=2, num_experts_per_tok=2)
        )
        # Strictly fewer routed experts than top-k raises.
        with self.assertRaisesRegex(ValueError, r"must be >="):
            TransformerConfig(
                **self._valid_hash_kwargs(
                    n_routed_experts=1, num_experts_per_tok=2
                )
            )

    # -- the gate itself -----------------------------------------------------

    def test_gate_off_skips_all_hash_validation(self):
        """moe_n_hash_layers == 0 skips the whole block, even when several
        hash-only fields are simultaneously invalid."""
        # actual_vocab_size stays at its None default AND scoring_func is
        # outside the allowed set AND first_k_dense_replace is set: all three
        # would raise under the gate, none do with the gate off.
        TransformerConfig(
            **self._gate_off_kwargs(
                scoring_func="relu",
                actual_vocab_size=None,
                first_k_dense_replace=3,
            )
        )


if __name__ == "__main__":
    unittest.main()
