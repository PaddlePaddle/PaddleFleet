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

* On a single rank, wrapping a plain (non-``PipelineLayer``) ``paddle.nn.Layer``
  is accepted and the wrapper preserves the model's forward numerics. The
  expected output is derived by hand with NumPy from fixed weights / bias /
  inputs (``y = x @ W + b``), so it is independent of the Paddle implementation:
  a wrapper that dropped the bias, transposed the weight, or reused the wrong
  sublayer would be rejected rather than passing on shape alone.
* The wrapped object still exposes the original parameter *values*, not merely
  parameters of the right shape.

Scope. This exercises only the local ``world_size == 1`` path. Pipeline-parallel
wrapping (the branch the original coverage test forced by mocking
``PipelineLayer.__instancecheck__`` and a fake ``world_size``) is genuinely
multi-card behavior and is out of scope for a single-process test; it must be
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
    """``distributed_model`` on ``world_size == 1`` preserves the model."""

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

    def test_forward_is_preserved_by_wrapping(self):
        """Wrapped forward equals the hand-derived ``x @ W + b``."""
        expected = np.array(_INPUT, dtype=np.float32) @ np.array(
            _WEIGHT, dtype=np.float32
        ) + np.array(_BIAS, dtype=np.float32)

        model = self._build_fixed_linear()
        x = paddle.to_tensor(_INPUT, dtype="float32")

        # Sanity-check the fixture against the independent reference before
        # touching the wrapper, so a mismatch below is attributable to it.
        np.testing.assert_allclose(
            model(x).numpy(), expected, rtol=1e-5, atol=1e-6
        )

        self._init_single_process_fleet()
        wrapped = fleet.distributed_model(model)

        # A non-PipelineLayer is accepted on a single rank, and the wrapper's
        # forward must reproduce the same numerics as the hand-derived output.
        out = wrapped(x)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_wrapped_parameters_keep_their_values(self):
        """The wrapper exposes the original weight/bias values, not just shapes."""
        model = self._build_fixed_linear()

        self._init_single_process_fleet()
        wrapped = fleet.distributed_model(model)

        params = {p.name: p for p in wrapped.parameters()}
        # Recover weight/bias by their known shapes and compare content against
        # the fixture; a wrapper that re-initialized or transposed them fails.
        weight = next(p for p in params.values() if list(p.shape) == [3, 2])
        bias = next(p for p in params.values() if list(p.shape) == [2])
        np.testing.assert_allclose(
            weight.numpy(),
            np.array(_WEIGHT, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            bias.numpy(),
            np.array(_BIAS, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
