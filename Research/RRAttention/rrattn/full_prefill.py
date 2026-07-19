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

import math

import paddle
import paddle.nn.functional as F

enable_profile = False
attn_time_ms = 0.0
estimate_func_time_ms = 0.0


def set_profile(enable=True):
    global enable_profile
    enable_profile = enable


def is_enable_profile():
    global enable_profile
    return enable_profile


def set_attn_time(attn_time=0.0):
    global attn_time_ms
    attn_time_ms = attn_time


def get_attn_time():
    global attn_time_ms
    return attn_time_ms


def add_attn_time(attn_time):
    global attn_time_ms
    attn_time_ms += attn_time


def set_estimate_func_time(estimate_func_time=0.0):
    global estimate_func_time_ms
    estimate_func_time_ms = estimate_func_time


def get_estimate_func_time():
    global estimate_func_time_ms
    return estimate_func_time_ms


def add_estimate_func_time(estimate_func_time):
    global estimate_func_time_ms
    estimate_func_time_ms += estimate_func_time


def full_prefill(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    causal: bool = True,
    attention_mask=None,
):
    attn_weights = paddle.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(query_states.shape[-1])

    if causal:
        q_len = query_states.shape[-2]
        k_len = key_states.shape[-2]
        q_pos = paddle.arange(q_len, device=query_states.device) + k_len - q_len
        k_pos = paddle.arange(k_len, device=query_states.device)
        causal_mask = q_pos[:, None] < k_pos[None, :]
        attn_weights = paddle.where(
            causal_mask.unsqueeze(0).unsqueeze(0),
            paddle.full(
                attn_weights.shape,
                float("-inf"),
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            ),
            attn_weights,
        )

    if attention_mask is not None:
        if attention_mask.dtype != paddle.bool:
            attention_mask = paddle.where(attention_mask == 0, True, False)
        attn_weights = paddle.where(
            attention_mask,
            paddle.full(
                attn_weights.shape,
                float("-inf"),
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            ),
            attn_weights,
        )

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=paddle.float32).to(
        query_states.dtype
    )
    attn_output = paddle.matmul(attn_weights, value_states)

    return attn_output


def flash_full_prefill(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    causal: bool = True,
):
    if is_enable_profile():
        paddle.cuda.synchronize()
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()

    attn_output = F.flashmask_attention(
        query_states,
        key_states,
        value_states,
        startend_row_indices=None,
        causal=True,
    )
    if is_enable_profile():
        paddle.cuda.synchronize()
        end_event.record()
        paddle.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        add_attn_time(elapsed_time_ms)

    return attn_output
