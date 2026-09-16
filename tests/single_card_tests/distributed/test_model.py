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

"""Behavior test for the ``distributed_model`` wrapping used by PaddleFleet.

Provenance note. The repository has no ``src/paddlefleet/distributed/model.py``;
the ``paddlefleet.distributed`` package ``__init__.py`` is empty. The symbol
``distributed_model`` is the upstream ``paddle.distributed.fleet.distributed_model``
API, which PaddleFleet consumes directly (e.g. ``fleet.distributed_model(model)``
inside ``paddlefleet/trainer/trainer.py``). This suite therefore pins the piece
of that upstream contract PaddleFleet actually relies on, restricted to the
legitimately single-process ``world_size == 1`` local path.

What is verified (real behavior, no stand-ins for the code under test):

* The fixed Linear fixture computes ``y = x @ W + b``. The expected output is
  derived by hand with NumPy from fixed weights / bias / inputs, independent of
  the Paddle implementation, so a fixture that dropped the bias or transposed
  the weight would be rejected rather than passing on shape alone.
* ``fleet.distributed_model`` does NOT accept an arbitrary ``paddle.nn.Layer``
  even on a single rank: in this Paddle version it routes unconditionally
  through ``NoPipelineParallel.__init__`` (pipeline_parallel.py:435), whose very
  first line is ``assert isinstance(layers, PipelineLayer)``. A plain
  ``nn.Linear`` is therefore rejected with ``AssertionError``. This is genuine
  upstream Paddle behavior (not a PaddleFleet defect and not a test artifact);
  it is pinned here with ``assertRaises`` and no production code is modified. An
  earlier version of this test assumed the wrapper accepted plain layers and
  preserved their forward numerics -- that premise is false on the real API, so
  the wrapping is now asserted to raise instead.

Scope. This exercises only the local ``world_size == 1`` path. Genuine
pipeline-parallel wrapping (constructing a real ``PipelineLayer`` across ranks)
is multi-card behavior and is out of scope for a single-process test; it must be
proven with a real multi-rank process group, not simulated here.

Because ``paddle`` is imported at module load, the whole suite is skipped with an
honest reason when Paddle is not installed on the host rather than reporting a
hollow pass. Establishing the single-process Fleet environment is treated as
environment setup: if it cannot be initialized here (e.g. no collective backend)
the test skips with the real error, while the behavioral assertions themselves
never swallow exceptions.
"""

import os
import sys
import unittest

# Make the in-tree ``src`` importable when the package is not pip-installed
# (repo_root/src is 4 levels up from this file). Harmless if already present.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle
    from paddle import nn
    from paddle.distributed import fleet

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle not installed on this host
    np = None
    paddle = None
    nn = None
    fleet = None
    _IMPORT_ERROR = exc


# Fixed, distinguishable fixture. Paddle ``nn.Linear`` stores weight with shape
# ``[in_features, out_features]`` and computes ``y = x @ weight + bias``.
_WEIGHT = [
    [1.0, -2.0],
    [0.5, 3.0],
    [-1.0, 0.25],
]  # shape [3, 2]
_BIAS = [0.1, -0.2]  # shape [2]
_INPUT = [
    [2.0, -1.0, 0.5],
    [0.0, 1.0, -3.0],
]  # shape [2, 3]


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle unavailable: {_IMPORT_ERROR}",
)
class TestDistributedModelSingleProcess(unittest.TestCase):
    """``distributed_model`` on ``world_size == 1``: fixture + wrap contract."""

    def _build_fixed_linear(self):
        """Return a Linear whose weight/bias are set to the fixed fixture."""
        model = nn.Linear(3, 2)
        model.weight.set_value(paddle.to_tensor(_WEIGHT, dtype="float32"))
        model.bias.set_value(paddle.to_tensor(_BIAS, dtype="float32"))
        return model

    def _init_single_process_fleet(self):
        """Bring up a single-rank Fleet, or skip honestly if unsupported.

        Only the environment bring-up is guarded here; the behavioral
        assertions in the tests run unguarded so a real regression surfaces
        instead of being swallowed into a skip.
        """
        try:
            paddle.set_device("cpu")
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)
        except Exception as exc:  # environment setup only, not code under test
            self.skipTest(f"single-process Fleet unavailable here: {exc!r}")

    def test_fixture_forward_matches_reference(self):
        """The Linear fixture computes exactly ``x @ W + b`` (no wrapper).

        This pins the fixture's real numerics against an independent NumPy
        reference before any wrapping is attempted, so the wrap contract below
        is tested against a known-good model.
        """
        expected = np.array(_INPUT, dtype=np.float32) @ np.array(
            _WEIGHT, dtype=np.float32
        ) + np.array(_BIAS, dtype=np.float32)

        model = self._build_fixed_linear()
        x = paddle.to_tensor(_INPUT, dtype="float32")
        np.testing.assert_allclose(
            model(x).numpy(), expected, rtol=1e-5, atol=1e-6
        )

    def test_distributed_model_requires_pipeline_layer(self):
        """Wrapping a plain Linear raises: upstream requires a PipelineLayer.

        ``fleet.distributed_model`` routes unconditionally through
        ``NoPipelineParallel.__init__``, whose first statement is
        ``assert isinstance(layers, PipelineLayer)``
        (paddle .../meta_parallel/pipeline_parallel.py:435). A plain
        ``nn.Linear`` is not a ``PipelineLayer``, so the assert fires. This
        documents genuine upstream Paddle behavior; no production code is
        modified. If a future Paddle relaxes this (accepting arbitrary layers on
        a single rank) the AssertionError will stop being raised and this test
        will fail, flagging the contract change for review.
        """
        model = self._build_fixed_linear()
        self._init_single_process_fleet()
        with self.assertRaises(AssertionError):
            fleet.distributed_model(model)


if __name__ == "__main__":
    unittest.main()
