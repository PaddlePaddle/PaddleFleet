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

from types import SimpleNamespace
from unittest import TestCase

from paddlefleet.transformers.model_utils import (
    _legacy_autoregressive_mtp_export_config,
)


class TestLegacyAutoregressiveMtpExportConfig(TestCase):
    def test_missing_legacy_field_keeps_canonical_config(self):
        config = SimpleNamespace(num_nextn_predict_layers=2)

        with _legacy_autoregressive_mtp_export_config(config):
            self.assertFalse(hasattr(config, "mtp_num_layers"))
            self.assertEqual(config.num_nextn_predict_layers, 2)

    def test_zero_legacy_field_keeps_canonical_config(self):
        config = SimpleNamespace(mtp_num_layers=0, num_nextn_predict_layers=2)

        with _legacy_autoregressive_mtp_export_config(config):
            self.assertEqual(config.mtp_num_layers, 0)
            self.assertEqual(config.num_nextn_predict_layers, 2)

    def test_positive_legacy_field_is_swapped_and_restored(self):
        config = SimpleNamespace(mtp_num_layers=1, num_nextn_predict_layers=2)

        with _legacy_autoregressive_mtp_export_config(config):
            self.assertEqual(config.mtp_num_layers, 2)
            self.assertEqual(config.num_nextn_predict_layers, 1)

        self.assertEqual(config.mtp_num_layers, 1)
        self.assertEqual(config.num_nextn_predict_layers, 2)

    def test_export_failure_restores_the_config(self):
        config = SimpleNamespace(mtp_num_layers=1, num_nextn_predict_layers=2)

        with (
            self.assertRaisesRegex(RuntimeError, "export failed"),
            _legacy_autoregressive_mtp_export_config(config),
        ):
            raise RuntimeError("export failed")

        self.assertEqual(config.mtp_num_layers, 1)
        self.assertEqual(config.num_nextn_predict_layers, 2)

    def test_invalid_legacy_value_is_rejected(self):
        config = SimpleNamespace(mtp_num_layers="1", num_nextn_predict_layers=2)

        with (
            self.assertRaisesRegex(TypeError, "mtp_num_layers"),
            _legacy_autoregressive_mtp_export_config(config),
        ):
            pass


if __name__ == "__main__":
    import unittest

    unittest.main()
