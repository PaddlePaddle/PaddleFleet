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

import importlib.metadata
from types import SimpleNamespace
from unittest.mock import Mock

import paddle
import pytest

from paddlefleet.cli.train.sft.workflow import ModelReproObservationCallback


@pytest.mark.parametrize("package_version", ["2.28.3", None])
@pytest.mark.parametrize("device", ["gpu:0", "cpu"])
def test_environment_allows_bundled_nccl(monkeypatch, package_version, device):
    callback = ModelReproObservationCallback(weights_loaded=True)
    monkeypatch.setattr(callback, "_topology", lambda args: {})
    monkeypatch.setattr(paddle.device, "get_device", lambda: device)
    device_name = Mock(return_value="test GPU")
    monkeypatch.setattr(paddle.device.cuda, "get_device_name", device_name)

    def version(name):
        assert name == "nvidia-nccl-cu12"
        if package_version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return package_version

    monkeypatch.setattr(importlib.metadata, "version", version)
    payload = callback._environment_payload(SimpleNamespace(bf16=True))
    assert payload["nccl_package"] == package_version
    assert payload["weights_loaded"] is True
    assert payload["dtype"] == "bfloat16"
    if device == "cpu":
        assert payload["device"] == payload["device_name"] == "cpu"
        device_name.assert_not_called()
    else:
        assert payload["device"] == "cuda"
        assert payload["device_name"] == "test GPU"
        device_name.assert_called_once_with()
