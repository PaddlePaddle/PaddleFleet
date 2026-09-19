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

"""Exercise complete MLP.forward dispatch, activation, and output propagation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from paddlefleet.transformer import mlp


class TestAccuracyCompatibleProjectionSelection(unittest.TestCase):
    def test_each_projection_keeps_its_actual_parallel_layer(self):
        # Each projection's actual group must take precedence over a local config.
        for compatible, configured, global_tp, up_size, down_size, paths in [
            (True, 1, 2, 2, 2, ("native", "native")),
            (True, 1, 2, 1, 1, ("direct", "direct")),
            (True, 1, 2, 1, 2, ("direct", "native")),
            (True, 1, 2, 2, 1, ("native", "direct")),
            (True, 1, 2, None, None, ("native", "native")),
            (False, 1, 2, 1, 1, ("native", "native")),
            (True, 2, 2, 2, 2, ("native", "native")),
            (True, 1, 1, 1, 1, ("native", "native")),
        ]:
            with self.subTest(
                compatible=compatible,
                configured=configured,
                global_tp=global_tp,
                up_size=up_size,
                down_size=down_size,
            ):

                def projection(size):
                    return (
                        SimpleNamespace()
                        if size is None
                        else SimpleNamespace(world_size=size)
                    )

                up, down = projection(up_size), projection(down_size)
                activation = Mock(return_value=sentinel.activation)
                instance = SimpleNamespace(
                    config=SimpleNamespace(
                        tensor_model_parallel_size=configured,
                        use_accuracy_compatible=compatible,
                        bias_activation_fusion=False,
                        use_bias=False,
                        gated_linear_unit=False,
                        gpt_model_use_experimental_version=False,
                    ),
                    up_gate_proj=up,
                    down_proj=down,
                    _dw_up_gate_point="up",
                    _dw_down_point="down",
                    inspect_name="test_mlp",
                    hidden_act=activation,
                )
                calls = []

                def project(path, layer, value):
                    calls.append((path, layer, value))
                    if layer is up:
                        return sentinel.projected, None
                    self.assertIs(layer, down)
                    return sentinel.output, sentinel.bias

                with (
                    patch.object(
                        mlp,
                        "get_tensor_model_parallel_world_size",
                        return_value=global_tp,
                    ),
                    patch.object(
                        mlp,
                        "_accuracy_compatible_projection",
                        side_effect=lambda layer, value: project(
                            "direct", layer, value
                        ),
                    ),
                    patch.object(
                        mlp,
                        "deferrable_linear",
                        side_effect=lambda config, point, layer, value: project(
                            "native", layer, value
                        ),
                    ),
                    patch.object(mlp, "nvtx_range_push"),
                    patch.object(mlp, "nvtx_range_pop"),
                    patch.object(mlp, "get_current_layer", return_value=0),
                    patch.object(
                        mlp,
                        "inspect_tensor",
                        side_effect=lambda name, layer, value: value,
                    ),
                ):
                    result = mlp.MLP.forward(instance, sentinel.input)
                self.assertEqual(
                    calls,
                    [
                        (paths[0], up, sentinel.input),
                        (paths[1], down, sentinel.activation),
                    ],
                )
                activation.assert_called_once_with(sentinel.projected)
                self.assertEqual(result, (sentinel.output, sentinel.bias))


if __name__ == "__main__":
    unittest.main()
