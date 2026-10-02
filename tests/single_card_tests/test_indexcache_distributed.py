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

"""Native PP delegation and IndexCache tensor/state contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle
from paddle.distributed.fleet.meta_parallel import (
    PipelineLayer,
    PipelineParallel,
)
from paddle.distributed.fleet.meta_parallel.pp_utils import (
    utils as paddle_utils,
)

from paddlefleet.models.gpt.gpt_model import GPTModel, GPTSublayersSpec
from paddlefleet.transformer.csa_attention import CompressedSparseAttention
from paddlefleet.transformer.indexcache_state import apply_stop_gradient_mask


class TestIndexCacheNativePipeline(unittest.TestCase):
    def test_gpt_uses_native_pipeline_forward(self):
        self.assertIn(PipelineLayer, GPTModel.__bases__)
        self.assertIs(GPTModel.forward, PipelineLayer.forward)

    def test_native_codec_preserves_state_tensors_and_gradient_mask(self):
        for size in (3, 8):
            with self.subTest(size=size):
                state = apply_stop_gradient_mask(
                    tuple(paddle.ones([1]) for _ in range(size))
                )
                encoded = paddle_utils.dict_to_tuple_helper(
                    {"hidden": paddle.ones([1]), "indexcache_state": state}
                )
                decoded, is_dict = paddle_utils.tuple_to_dict_helper(encoded)
                self.assertTrue(is_dict)
                for i, tensor in enumerate(decoded["indexcache_state"]):
                    self.assertIs(tensor, state[i])
                    self.assertEqual(
                        tensor.stop_gradient, not (size == 8 and i == 5)
                    )

    def test_state_lifetime_stops_before_next_producer(self):
        # Output liveness is independent of the PP partition.
        for pattern, expected in (
            ("F", [False]),
            ("FF", [False, False]),
            ("FSF", [True, False, False]),
            ("FSSF", [True, True, False, False]),
            ("FSFS", [True, False, True, False]),
        ):
            with self.subTest(pattern=pattern):
                actual = [
                    CompressedSparseAttention._indexcache_has_future_served_layer(
                        pattern, i
                    )
                    for i in range(len(pattern))
                ]
                self.assertEqual(actual, expected)

    def _config(self, **overrides):
        values = {
            "indexcache_topk_pattern": "FS",
            "pipeline_model_parallel_size": 1,
            "virtual_pipeline_model_parallel_size": None,
            "tie_word_embeddings": False,
            "enable_mtp_magic_send": False,
            "gpt_model_use_experimental_version": False,
            "num_nextn_predict_layers": 0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _strategy(self, amp=False, **pp_overrides):
        pp_config = {
            "use_dualpipev": False,
            "forward_backward_overlap_scheduler": False,
        }
        pp_config.update(pp_overrides)
        return SimpleNamespace(
            amp=amp, hybrid_configs={"pp_configs": SimpleNamespace(**pp_config)}
        )

    def _build_model(self, config):
        spec = GPTSublayersSpec(
            embedding=paddle.nn.Identity,
            head_empty_layers=[],
            transformer_layers=[],
            tail_empty_layers=[],
            layer_norm=paddle.nn.Identity,
            lm_head=paddle.nn.Identity,
        )
        return GPTModel(
            spec, config=config, tie_word_embeddings=False, num_stages=1
        )

    def test_single_stage_build_and_backward_without_fleet_initialization(self):
        original = PipelineParallel._backward_step
        for vpp in (None, 1):
            with (
                self.subTest(vpp=vpp),
                patch(
                    "paddlefleet.models.gpt.gpt_model.fleet.fleet._user_defined_strategy",
                    None,
                    create=True,
                ),
            ):
                model = self._build_model(
                    self._config(virtual_pipeline_model_parallel_size=vpp)
                )
                self.assertIs(model.forward.__func__, PipelineLayer.forward)
                inputs = paddle.ones([2, 3])
                inputs.stop_gradient = False
                model(inputs).sum().backward()
                self.assertTrue(paddle.all(inputs.grad == 1).item())
        self.assertIs(PipelineParallel._backward_step, original)

    def test_unsupported_scheduling_rejected_before_layers_are_built(self):
        cases = [
            (
                {"virtual_pipeline_model_parallel_size": 2},
                self._strategy(),
                {},
                "VPP=1",
            ),
            ({}, self._strategy(), {"use_dualpipev": True}, "DualPipeV"),
            ({}, self._strategy(use_dualpipev=True), {}, "DualPipeV"),
            (
                {},
                self._strategy(forward_backward_overlap_scheduler=True),
                {},
                "compute-overlap",
            ),
            (
                {"pipeline_model_parallel_size": 2},
                self._strategy(amp=True),
                {},
                "Trainer-managed AMP",
            ),
        ]
        for config_updates, strategy, kwargs, message in cases:
            with (
                self.subTest(
                    config=config_updates, strategy=strategy, kwargs=kwargs
                ),
                patch(
                    "paddlefleet.models.gpt.gpt_model.fleet.fleet._user_defined_strategy",
                    strategy,
                    create=True,
                ),
                patch.object(GPTModel, "get_layer_desc_list") as build_layers,
            ):
                with self.assertRaisesRegex(NotImplementedError, message):
                    GPTModel(
                        None,
                        config=self._config(**config_updates),
                        tie_word_embeddings=False,
                        **kwargs,
                    )
                build_layers.assert_not_called()

    def test_strategy_amp_allowed_without_pipeline_parallel(self):
        with patch(
            "paddlefleet.models.gpt.gpt_model.fleet.fleet._user_defined_strategy",
            self._strategy(amp=True),
            create=True,
        ):
            self.assertIsInstance(self._build_model(self._config()), GPTModel)

    def test_supported_pipeline_and_disabled_indexcache_reach_layer_construction(
        self,
    ):
        for enabled in (True, False):
            config = self._config(
                indexcache_topk_pattern="FS" if enabled else None,
                virtual_pipeline_model_parallel_size=1 if enabled else 2,
                pipeline_model_parallel_size=2,
            )
            strategy = self._strategy(
                amp=not enabled,
                use_dualpipev=not enabled,
                forward_backward_overlap_scheduler=not enabled,
            )
            with (
                self.subTest(indexcache=enabled),
                patch(
                    "paddlefleet.models.gpt.gpt_model.fleet.fleet._user_defined_strategy",
                    strategy,
                    create=True,
                ),
                patch.object(
                    GPTModel,
                    "get_layer_desc_list",
                    side_effect=RuntimeError("build layers"),
                ) as build_layers,
            ):
                with self.assertRaisesRegex(RuntimeError, "build layers"):
                    GPTModel(
                        None,
                        config=config,
                        tie_word_embeddings=False,
                        use_dualpipev=not enabled,
                    )
                build_layers.assert_called_once()
