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

from paddle import nn

from .pp_layers import PipelineLayer


# assume only the first stage and last stage need data, and data consumption is ordered
# to be replaced by real micro dataset from reader
class FakeMicroDataset:
    def __init__(
        self,
        data,
        is_first_stage,
        is_last_stage,
        acc_steps,
        micro_batch_size,
    ):
        self._data = data
        self._index = 0
        self._acc_steps = acc_steps
        self._is_first_stage = is_first_stage
        self._is_last_stage = is_last_stage
        self._micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._acc_steps:
            raise StopIteration
        assert self._is_first_stage or self._is_last_stage
        micro_batch_data = self._load_micro_batch(self._index)
        self._index += 1
        return micro_batch_data

    def _load_micro_batch(self, micro_step):
        inputs = self._data
        data = None
        label = None
        if self._is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            data = self._load_micro_batch_impl(inputs[0], micro_step)

        if self._is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            label = self._load_micro_batch_impl(inputs[1], micro_step)
        return (data, label)

    def _load_micro_batch_impl(self, inputs, micro_step):
        begin = micro_step * self._micro_batch_size
        end = begin + self._micro_batch_size

        if isinstance(inputs, tuple):
            output = []
            for data in inputs:
                if isinstance(data, list):
                    assert len(data) == self._acc_steps, (
                        f"length of data should be {self._acc_steps}, but it is {len(data)}"
                    )
                    output.append(
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_valid(data)
                    output.append(data[begin:end, :].detach())
                else:
                    output.append(None)
            return tuple(output)
        elif isinstance(inputs, dict):
            output_dict = {}
            for key, data in inputs.items():
                if isinstance(data, list):
                    assert len(data) == self._acc_steps, (
                        f"length of data should be {self._acc_steps}, but it is {len(data)}"
                    )
                    output_dict[key] = (
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_valid(data)
                    output_dict[key] = data[begin:end, :].detach()
                else:
                    output_dict[key] = None
            return output_dict
        elif isinstance(inputs, list):
            assert len(inputs) == self._acc_steps, (
                f"length of data should be {self._acc_steps}, but it is {len(inputs)}"
            )
            return inputs[micro_step].detach()
        elif inputs is not None:
            self._check_data_valid(inputs)
            return inputs[begin:end, :].detach()
        else:
            return None

    def _check_data_valid(self, data):
        batch_size = data.shape[0]
        assert self._micro_batch_size * self._acc_steps == batch_size, (
            "batch_size needs to be divisible by micro_batch_size. Currently, "
            f"batch_size = {batch_size}, micro_batch_size = {self._micro_batch_size}, accumulate_steps = {self._acc_steps}."
        )


class NoPipelineParallel(nn.Layer):
    def __init__(self, layers, strategy):
        assert isinstance(layers, PipelineLayer)
        super().__init__()
        self._layers = layers
        self._strategy = strategy
        self.micro_batch_size = self._strategy.pipeline_configs[
            "micro_batch_size"
        ]
        self.accumulate_steps = self._strategy.pipeline_configs[
            "accumulate_steps"
        ]

    def forward_backward_pipeline(
        self,
        data,
        scaler=None,
        return_micro_batch_loss=False,
    ):
        micro_dataset = FakeMicroDataset(
            data,
            True,
            True,
            self.accumulate_steps,
            self.micro_batch_size,
        )
        loss_list = []
        for _ in range(self.accumulate_steps):
            data_iter = next(micro_dataset)
            input_tensor = data_iter[0]
            label = data_iter[1]
            output_tensor = self._layers.forward(input_tensor)
            loss = self._layers._loss_fn[0](output_tensor, label)
            loss.backward()
            loss_list.append(loss)

        return sum(loss_list) / self.accumulate_steps


class PipelineParallel(nn.Layer):
    def __init__(self, layers, hcg, strategy):
        assert isinstance(layers, PipelineLayer)
        super().__init__()
        self._layers = layers
        self._strategy = strategy
        self._hcg = hcg

    def forward(self, data, scaler=None, return_micro_batch_loss=False):
        pass
