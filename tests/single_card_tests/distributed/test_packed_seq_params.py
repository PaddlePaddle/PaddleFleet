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

"""Behavior tests for ``paddlefleet.packed_seq_params.PackedSeqParams``.

The production type is a plain (non-frozen) ``@dataclass`` describing the
parameters handed to DotProductAttention / fused RoPE kernels for the ``thd``
(packed) sequence format. It performs no runtime type validation, so the
observable contract is:

  * the generated ``__init__`` accepts the nine fields in a documented
    positional order and stores each argument in the matching attribute;
  * every field defaults to ``None`` (all parameters are optional);
  * distinct fields do not bleed into one another;
  * the generated ``__eq__`` compares all nine fields (value semantics);
  * instances are mutable (the dataclass is intentionally NOT frozen).

Expected values below are hand-authored from that contract, using nine
mutually distinct sentinels so that a swapped or dropped field is observable.
No production behavior is used to derive the expected values.

Environment note: this repository's ``paddlefleet`` package ``__init__`` imports
``paddle`` at import time, and ``paddle`` is not installed in the no-card CPU
environment. ``packed_seq_params`` itself only references ``paddle.Tensor`` in a
``TYPE_CHECKING`` block (annotations are strings via ``from __future__ import
annotations``), so the module has no runtime ``paddle`` dependency. We therefore
first try the normal package import; if that fails because the heavy package
``__init__`` cannot be satisfied, we load this single source file directly. That
fallback exercises the real production ``PackedSeqParams`` class but does NOT
verify the full package import chain -- honestly reported rather than skipped.
"""

import importlib.util
import os
import sys
import unittest

_REPO_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
_MODULE_SOURCE = os.path.join(_REPO_SRC, "paddlefleet", "packed_seq_params.py")

# Documented field order of the dataclass __init__ signature. This list is the
# hand-authored contract under test; it is NOT read back from the production
# class to build expected values.
_FIELD_ORDER = [
    "qkv_format",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "max_seqlen_q",
    "max_seqlen_kv",
    "total_seqlen_q",
    "total_seqlen_kv",
]

PackedSeqParams = None
_IMPORT_ERROR = None
_LOADED_STANDALONE = False

try:
    # Preferred path: real package import (verifies package import chain too).
    from paddlefleet.packed_seq_params import (  # type: ignore
        PackedSeqParams as PackedSeqParams,
    )
except ImportError as exc:  # narrow: only missing-dependency style failures
    _IMPORT_ERROR = exc
    # Fallback: load the standalone source file. It has no runtime paddle
    # dependency, so the real dataclass can be exercised without paddle.
    try:
        if os.path.isfile(_MODULE_SOURCE):
            _spec = importlib.util.spec_from_file_location(
                "pf_packed_seq_params_under_test", _MODULE_SOURCE
            )
            _mod = importlib.util.module_from_spec(_spec)
            # dataclass processing resolves the class module via sys.modules,
            # so register before executing the module body.
            sys.modules[_spec.name] = _mod
            _spec.loader.exec_module(_mod)
            PackedSeqParams = _mod.PackedSeqParams
            _LOADED_STANDALONE = True
    except ImportError as exc2:
        _IMPORT_ERROR = exc2


@unittest.skipUnless(
    PackedSeqParams is not None,
    f"PackedSeqParams unavailable (neither package import nor standalone "
    f"source load succeeded): {_IMPORT_ERROR!r}",
)
class TestPackedSeqParams(unittest.TestCase):
    """Behavior of the PackedSeqParams dataclass."""

    # Nine mutually distinct, order-revealing sentinels. Because the dataclass
    # does no runtime type checking, plain ints/strings are valid stand-ins and
    # keep the test independent of paddle. Distinctness is what makes a field
    # swap detectable.
    DISTINCT = {
        "qkv_format": "thd",
        "cu_seqlens_q": 11,
        "cu_seqlens_kv": 22,
        "cu_seqlens_q_padded": 33,
        "cu_seqlens_kv_padded": 44,
        "max_seqlen_q": 55,
        "max_seqlen_kv": 66,
        "total_seqlen_q": 77,
        "total_seqlen_kv": 88,
    }

    def test_field_order_positional_mapping(self):
        """Positional args map to fields in the documented order.

        Passing nine distinct values positionally and checking each named
        attribute would fail if any two fields were reordered or swapped.
        """
        positional = [self.DISTINCT[name] for name in _FIELD_ORDER]
        params = PackedSeqParams(*positional)
        for name in _FIELD_ORDER:
            self.assertEqual(
                getattr(params, name),
                self.DISTINCT[name],
                msg=f"field {name!r} did not receive its positional value",
            )

    def test_defaults_are_all_none(self):
        """A no-argument instance leaves every field at ``None``."""
        params = PackedSeqParams()
        for name in _FIELD_ORDER:
            self.assertIsNone(
                getattr(params, name),
                msg=f"field {name!r} should default to None",
            )

    def test_fields_are_independent(self):
        """Setting one field by keyword does not perturb the others.

        Only ``qkv_format`` and ``max_seqlen_q`` are provided; the remaining
        seven fields must stay at their ``None`` default, catching any
        cross-field assignment bug.
        """
        params = PackedSeqParams(qkv_format="hd", max_seqlen_q=128)
        self.assertEqual(params.qkv_format, "hd")
        self.assertEqual(params.max_seqlen_q, 128)
        for name in _FIELD_ORDER:
            if name in ("qkv_format", "max_seqlen_q"):
                continue
            self.assertIsNone(
                getattr(params, name),
                msg=f"untouched field {name!r} should remain None",
            )

    def test_equality_considers_every_field(self):
        """Value equality holds for identical instances and breaks on any
        single differing field.

        Two independently constructed instances with identical values must be
        equal (generated ``__eq__``). Then, mutating exactly one field to a
        value distinct from the baseline must make them unequal -- proving that
        field participates in comparison (i.e. it is not excluded via
        ``field(compare=False)``).
        """
        base_kwargs = dict(self.DISTINCT)
        a = PackedSeqParams(**base_kwargs)
        b = PackedSeqParams(**base_kwargs)
        self.assertEqual(a, b)

        for name in _FIELD_ORDER:
            differing = dict(base_kwargs)
            # 999 differs from every DISTINCT sentinel; for the string field
            # use a clearly different string.
            differing[name] = "OTHER" if name == "qkv_format" else 999
            self.assertNotEqual(
                PackedSeqParams(**differing),
                b,
                msg=f"changing {name!r} should break equality",
            )

    def test_instances_are_mutable(self):
        """The dataclass is intentionally not frozen: attributes can be
        reassigned after construction and read back."""
        params = PackedSeqParams()
        params.qkv_format = "thd"
        params.max_seqlen_kv = 256
        self.assertEqual(params.qkv_format, "thd")
        self.assertEqual(params.max_seqlen_kv, 256)

    def test_declared_field_set_and_order(self):
        """The dataclass declares exactly the nine expected fields, in order.

        This locks the public schema: an added, removed, or reordered field is
        rejected. Unlike a bare ``is_dataclass`` check, it pins the full
        ordered field list.
        """
        import dataclasses

        self.assertTrue(dataclasses.is_dataclass(PackedSeqParams))
        actual_order = [f.name for f in dataclasses.fields(PackedSeqParams)]
        self.assertEqual(actual_order, _FIELD_ORDER)


if __name__ == "__main__":
    unittest.main()
