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

"""Behavior tests for ``paddlefleet.process_groups_config.ProcessGroupCollection``.

``ProcessGroupCollection`` is the config object that holds the seven parallel
process groups (tp / pp / cp / ep / dp / cp_dp / expt_dp). The real logic under
test lives in two places:

* ``__init__`` accepts only the declared field names and stores each keyword to
  the matching attribute, raising ``ValueError`` (naming the offender) for any
  unknown key. Because the dataclass fields are ``init=False``, an attribute
  that was never supplied is genuinely absent rather than silently ``None``.
* ``use_mpu_process_groups`` maps each requested group name to a *specific*
  ``parallel_state`` getter, calls it with ``check_initialized=False`` (so the
  collection can be built before the groups are initialized) and feeds the
  result into the collection. Crucially ``dp`` and ``cp_dp`` share one getter
  (``get_data_parallel_group``) but differ by ``with_context_parallel``.

These tests drive the real routing/validation code and compare against a mapping
derived by hand from the module contract. The ``parallel_state`` getters are the
only mocked objects -- they are collaborators, not the code under test -- and are
each given a distinguishable sentinel so a swapped or dropped mapping (e.g. dp
<-> cp_dp, or forgetting ``with_context_parallel=True``) is rejected rather than
passing on shape alone. No real process group or collective is involved; this is
pure CPU config wiring, so it is not a stand-in for multi-card numerics.

Because the production module imports ``paddle`` (transitively via
``paddlefleet.parallel_state``) at import time, the whole suite is skipped with
an honest reason when Paddle / paddlefleet is unavailable on the host, rather
than reporting a hollow pass.
"""

import os
import sys
import unittest
from unittest import mock

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed into the environment (repo_root/src is 4 levels up from here).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle  # noqa: F401

    from paddlefleet import parallel_state
    from paddlefleet.process_groups_config import ProcessGroupCollection

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    parallel_state = None
    ProcessGroupCollection = None
    _IMPORT_ERROR = exc


# The contract this suite pins down: attribute name -> the parallel_state getter
# that must populate it. ``cp_dp`` deliberately shares ``get_data_parallel_group``
# with ``dp`` but is distinguished by ``with_context_parallel=True``.
_ALL_FIELDS = ("tp", "pp", "cp", "ep", "dp", "cp_dp", "expt_dp")


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class TestProcessGroupCollectionInit(unittest.TestCase):
    """The custom ``__init__`` stores known fields and rejects unknown ones."""

    def test_stores_supplied_fields_without_swapping(self):
        """Distinct sentinels land on the attribute named by their keyword."""
        tp_grp = object()
        pp_grp = object()
        dp_grp = object()
        pgc = ProcessGroupCollection(tp=tp_grp, pp=pp_grp, dp=dp_grp)
        self.assertIs(pgc.tp, tp_grp)
        self.assertIs(pgc.pp, pp_grp)
        self.assertIs(pgc.dp, dp_grp)

    def test_unsupplied_field_is_absent_not_none(self):
        """Fields are ``init=False``; an omitted group must not exist at all."""
        pgc = ProcessGroupCollection(tp=object())
        # Accessing an attribute that was never set raises, so downstream code
        # cannot mistake "not requested" for a real (possibly None) group.
        with self.assertRaises(AttributeError):
            _ = pgc.pp

    def test_unknown_field_raises_value_error_naming_it(self):
        """An unrecognised keyword is refused, not silently attached."""
        with self.assertRaises(ValueError) as ctx:
            ProcessGroupCollection(not_a_group=object())
        self.assertIn("not_a_group", str(ctx.exception))


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class TestUseMpuProcessGroups(unittest.TestCase):
    """``use_mpu_process_groups`` routes each name to the correct getter."""

    def _patch_getters(self):
        """Patch every parallel_state getter with a distinguishable sentinel.

        ``get_data_parallel_group`` returns a different marker depending on
        ``with_context_parallel`` so that the dp / cp_dp split is observable.
        Returns (markers, recorded_calls) where recorded_calls maps the getter
        name to its captured kwargs.
        """
        markers = {name: object() for name in _ALL_FIELDS}
        calls = {}

        def _dp_side_effect(
            check_initialized=True, with_context_parallel=False
        ):
            calls["get_data_parallel_group"] = {
                "check_initialized": check_initialized,
                "with_context_parallel": with_context_parallel,
            }
            return markers["cp_dp"] if with_context_parallel else markers["dp"]

        patchers = [
            mock.patch.object(
                parallel_state,
                "get_tensor_model_parallel_group",
                return_value=markers["tp"],
            ),
            mock.patch.object(
                parallel_state,
                "get_pipeline_model_parallel_group",
                return_value=markers["pp"],
            ),
            mock.patch.object(
                parallel_state,
                "get_context_parallel_group",
                return_value=markers["cp"],
            ),
            mock.patch.object(
                parallel_state,
                "get_expert_model_parallel_group",
                return_value=markers["ep"],
            ),
            mock.patch.object(
                parallel_state,
                "get_expert_data_parallel_group",
                return_value=markers["expt_dp"],
            ),
            mock.patch.object(
                parallel_state,
                "get_data_parallel_group",
                side_effect=_dp_side_effect,
            ),
        ]
        mocks = {}
        for p in patchers:
            m = p.start()
            self.addCleanup(p.stop)
            mocks[p.attribute] = m
        return markers, mocks, calls

    def test_default_requests_every_field_with_correct_source(self):
        """None -> all seven groups, each from its own getter (no swaps)."""
        markers, _mocks, calls = self._patch_getters()
        pgc = ProcessGroupCollection.use_mpu_process_groups()

        # Every declared field is populated, and by the getter the contract
        # assigns to it. Swapping any two mappings would surface here because
        # the sentinels are all distinct objects.
        for name in _ALL_FIELDS:
            self.assertIs(
                getattr(pgc, name),
                markers[name],
                msg=f"field {name!r} was populated from the wrong getter",
            )

        # dp and cp_dp share get_data_parallel_group but must differ by the
        # with_context_parallel flag; the recorded call proves cp_dp asked for
        # the context-parallel variant.
        self.assertIsNot(pgc.dp, pgc.cp_dp)

    def test_getters_called_with_check_initialized_false(self):
        """The collection is built pre-init, so it must not assert init."""
        _markers, mocks, calls = self._patch_getters()
        ProcessGroupCollection.use_mpu_process_groups()

        _kw = mocks["get_tensor_model_parallel_group"].call_args.kwargs
        self.assertEqual(_kw.get("check_initialized"), False)
        # cp_dp path additionally requests the context-parallel data group.
        self.assertEqual(
            calls["get_data_parallel_group"]["check_initialized"], False
        )
        self.assertEqual(
            calls["get_data_parallel_group"]["with_context_parallel"], True
        )

    def test_subset_builds_only_requested_fields(self):
        """Only names in ``required_pgs`` are populated; others stay absent."""
        markers, _mocks, _calls = self._patch_getters()
        pgc = ProcessGroupCollection.use_mpu_process_groups(["tp", "dp"])

        self.assertIs(pgc.tp, markers["tp"])
        self.assertIs(pgc.dp, markers["dp"])
        for absent in ("pp", "cp", "ep", "cp_dp", "expt_dp"):
            with self.assertRaises(AttributeError):
                getattr(pgc, absent)

    def test_invalid_name_raises_value_error_naming_it(self):
        """An unknown group name is rejected before any getter runs."""
        markers, mocks, _calls = self._patch_getters()
        with self.assertRaises(ValueError) as ctx:
            ProcessGroupCollection.use_mpu_process_groups(["tp", "bogus"])
        self.assertIn("bogus", str(ctx.exception))
        # Validation happens up front, so the valid getter is not invoked.
        mocks["get_tensor_model_parallel_group"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
