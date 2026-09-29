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

"""Behavior tests for MLASelfAttention delayed weight-gradient dispatch and the
MLASelfAttentionSublayersSpec dataclass contract.

The delayed dw (``backward_dw``) path fans out to per-projection helpers whose
*only* job is to invoke ``backward_dw`` on the right child projections, in the
right order, and to skip children that do not exist in a given MLA variant
(split-absorption drops ``kv_b_proj``; the low-rank Q path swaps ``q_proj`` for
``q_a_proj``/``q_b_proj``). We drive the REAL production methods
(``multi_latent_attention.py`` lines 2504-2529) via a lightweight probe object
that borrows the unbound functions, so every branch decision is executed for
real; only the leaf projection layers -- genuine collaborators that are not
under test here -- are replaced by recording spies. We assert the exact call
*sequence* (count + order + which children) and prove branch exclusivity by
leaving the wrong-branch attribute unset so a mis-dispatch would raise
AttributeError rather than silently pass.

CPU-only: paddle is not required to exercise this pure dispatch logic, but the
production module imports paddle at module load, so the import is guarded and
the suite skips honestly when paddle is unavailable.
"""

import unittest
from dataclasses import fields

try:
    from paddlefleet.transformer.multi_latent_attention import (
        MLASelfAttention,
        MLASelfAttentionSublayersSpec,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    MLASelfAttention = None
    MLASelfAttentionSublayersSpec = None
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddlefleet.transformer.multi_latent_attention import failed: {_IMPORT_ERROR}"


class _RecordingProj:
    """Stand-in for a child projection layer.

    Records its own name into a shared log when ``backward_dw`` runs, so the
    order and identity of the children the production code touches is directly
    observable.
    """

    def __init__(self, name, log):
        self._name = name
        self._log = log

    def backward_dw(self):
        self._log.append(self._name)


def _make_probe(**attrs):
    """Build an object bound to the REAL MLASelfAttention dw dispatch methods.

    Borrowing the unbound production functions runs their genuine branch logic
    against a plain attribute holder, without invoking the heavyweight
    ``nn.Layer`` constructor. Only leaf collaborators are supplied by the test.
    """
    probe_cls = type(
        "_MLADwDispatchProbe",
        (),
        {
            "backward_dw": MLASelfAttention.backward_dw,
            "_backward_kv_proj": MLASelfAttention._backward_kv_proj,
            "_backward_q_proj": MLASelfAttention._backward_q_proj,
            "_backward_output_proj": MLASelfAttention._backward_output_proj,
        },
    )
    obj = probe_cls()
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


@unittest.skipUnless(MLASelfAttention is not None, _SKIP_REASON)
class TestMLASelfAttentionBackwardDwDispatch(unittest.TestCase):
    """End-to-end ordering of backward_dw through the real per-proj helpers."""

    def test_full_dispatch_low_rank_q_with_kv_b(self):
        # Low-rank Q (q_lora_rank set) and standard KV (kv_b_proj present).
        # Contract: kv projections first (kv_b then kv_a), then q_a/q_b, then
        # the output projection. q_proj must NOT be touched on this branch.
        log = []
        probe = _make_probe(
            kv_b_proj=_RecordingProj("kv_b", log),
            kv_a_proj_with_mqa=_RecordingProj("kv_a", log),
            q_lora_rank=768,
            q_a_proj=_RecordingProj("q_a", log),
            q_b_proj=_RecordingProj("q_b", log),
            q_proj=_RecordingProj("q_proj_should_not_run", log),
            o_proj=_RecordingProj("o", log),
        )

        probe.backward_dw()

        self.assertEqual(log, ["kv_b", "kv_a", "q_a", "q_b", "o"])

    def test_full_dispatch_plain_q_with_kv_b(self):
        # Plain Q (q_lora_rank None) selects q_proj; q_a/q_b must not run.
        log = []
        probe = _make_probe(
            kv_b_proj=_RecordingProj("kv_b", log),
            kv_a_proj_with_mqa=_RecordingProj("kv_a", log),
            q_lora_rank=None,
            q_proj=_RecordingProj("q_proj", log),
            q_a_proj=_RecordingProj("q_a_should_not_run", log),
            q_b_proj=_RecordingProj("q_b_should_not_run", log),
            o_proj=_RecordingProj("o", log),
        )

        probe.backward_dw()

        self.assertEqual(log, ["kv_b", "kv_a", "q_proj", "o"])


@unittest.skipUnless(MLASelfAttention is not None, _SKIP_REASON)
class TestMLASelfAttentionBackwardKvProj(unittest.TestCase):
    """_backward_kv_proj branch on kv_b_proj presence (lines 2511-2517)."""

    def test_calls_kv_b_then_kv_a_when_present(self):
        log = []
        probe = _make_probe(
            kv_b_proj=_RecordingProj("kv_b", log),
            kv_a_proj_with_mqa=_RecordingProj("kv_a", log),
        )

        probe._backward_kv_proj()

        # kv_b is computed before kv_a, and both exactly once.
        self.assertEqual(log, ["kv_b", "kv_a"])

    def test_skips_kv_b_in_split_absorption_mode(self):
        # Split-absorption drops kv_b_proj (set to None). kv_a_proj_with_mqa
        # must still run and no attribute error may leak. This is the branch
        # the coverage source never exercised.
        log = []
        probe = _make_probe(
            kv_b_proj=None,
            kv_a_proj_with_mqa=_RecordingProj("kv_a", log),
        )

        probe._backward_kv_proj()

        self.assertEqual(log, ["kv_a"])


@unittest.skipUnless(MLASelfAttention is not None, _SKIP_REASON)
class TestMLASelfAttentionBackwardQProj(unittest.TestCase):
    """_backward_q_proj branch on q_lora_rank (lines 2519-2525)."""

    def test_low_rank_calls_q_a_and_q_b_only(self):
        # q_proj is deliberately NOT set: if the low-rank branch wrongly touched
        # it, this would raise AttributeError instead of silently passing.
        log = []
        probe = _make_probe(
            q_lora_rank=16,
            q_a_proj=_RecordingProj("q_a", log),
            q_b_proj=_RecordingProj("q_b", log),
        )

        probe._backward_q_proj()

        self.assertEqual(log, ["q_a", "q_b"])

    def test_no_lora_calls_q_proj_only(self):
        # q_a_proj / q_b_proj deliberately unset: the plain branch must not
        # reach them.
        log = []
        probe = _make_probe(
            q_lora_rank=None,
            q_proj=_RecordingProj("q_proj", log),
        )

        probe._backward_q_proj()

        self.assertEqual(log, ["q_proj"])


@unittest.skipUnless(MLASelfAttention is not None, _SKIP_REASON)
class TestMLASelfAttentionBackwardOutputProj(unittest.TestCase):
    """_backward_output_proj forwards to o_proj (lines 2527-2529)."""

    def test_calls_o_proj_once(self):
        log = []
        probe = _make_probe(o_proj=_RecordingProj("o", log))

        probe._backward_output_proj()

        self.assertEqual(log, ["o"])


@unittest.skipUnless(MLASelfAttentionSublayersSpec is not None, _SKIP_REASON)
class TestMLASelfAttentionSublayersSpec(unittest.TestCase):
    """Dataclass default + name->attribute mapping contract (lines 293-307)."""

    _EXPECTED_FIELDS = (
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "core_attention",
        "o_proj",
        "gate_proj",
    )

    def test_field_set_matches_expected(self):
        actual = tuple(f.name for f in fields(MLASelfAttentionSublayersSpec))
        self.assertEqual(actual, self._EXPECTED_FIELDS)

    def test_all_fields_default_to_none(self):
        spec = MLASelfAttentionSublayersSpec()
        for name in self._EXPECTED_FIELDS:
            self.assertIsNone(
                getattr(spec, name), msg=f"{name} should default to None"
            )

    def test_keyword_arguments_map_to_matching_attribute(self):
        # Distinct sentinels per field catch any cross-wiring of names to slots.
        sentinels = {name: object() for name in self._EXPECTED_FIELDS}
        spec = MLASelfAttentionSublayersSpec(**sentinels)
        for name in self._EXPECTED_FIELDS:
            self.assertIs(getattr(spec, name), sentinels[name])


if __name__ == "__main__":
    unittest.main()
