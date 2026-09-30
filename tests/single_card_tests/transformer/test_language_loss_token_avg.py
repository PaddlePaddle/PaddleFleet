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

"""Unit coverage for the global token-averaged loss display (loss 口径统一).

``HB_LOSS_GLOBAL_TOKEN_AVG`` gates a logging-only convention: each rank
accumulates its raw ``(loss_sum, valid_tokens)`` into the class-level
``token_avg_tracker`` per micro-batch, and the trainer consumes it once per
logging interval to emit the global token-weighted loss ΣΣS/ΣΣn (matching
Megatron lm_loss). The backprop term is always ``loss_sum / local_valid``.

These are single-process unit tests: no fleet/dist init, so
``consume_token_avg_loss`` skips the all_reduce and just returns ΣS/Σn.
"""

from __future__ import annotations

import os

import paddle

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss


def _reduce(loss_sum, local_valid):
    """Call the (self-independent) instance method without building the layer."""
    return LanguageLoss._reduce_loss_by_tokens(object(), loss_sum, local_valid)


def _clear_tracker():
    LanguageLoss.token_avg_tracker.clear()


def test_disabled_returns_local_mean_and_leaves_tracker_empty():
    paddle.set_device("cpu")
    os.environ.pop("HB_LOSS_GLOBAL_TOKEN_AVG", None)
    _clear_tracker()

    loss_sum = paddle.to_tensor(20.0)
    valid = paddle.to_tensor(4.0)
    out = _reduce(loss_sum, valid)

    assert abs(float(out) - 5.0) < 1e-6, float(out)
    assert LanguageLoss.token_avg_tracker == {}, LanguageLoss.token_avg_tracker


def test_enabled_backprop_value_unchanged_and_accumulates():
    paddle.set_device("cpu")
    os.environ["HB_LOSS_GLOBAL_TOKEN_AVG"] = "1"
    _clear_tracker()
    try:
        loss_sum = paddle.to_tensor(20.0)
        valid = paddle.to_tensor(4.0)
        out = _reduce(loss_sum, valid)

        # Backprop term is still loss_sum / local_valid (bit-for-bit path).
        assert abs(float(out) - 5.0) < 1e-6, float(out)
        # The rank's raw (S, n) is stashed for the trainer.
        assert abs(float(LanguageLoss.token_avg_tracker["S"]) - 20.0) < 1e-6
        assert abs(float(LanguageLoss.token_avg_tracker["n"]) - 4.0) < 1e-6
    finally:
        os.environ.pop("HB_LOSS_GLOBAL_TOKEN_AVG", None)
        _clear_tracker()


def test_consume_is_token_weighted_across_microbatches():
    """Two uneven micro-batches: the global loss is ΣS/Σn, NOT the mean of the
    two per-batch means (that is the point of token-weighting).
    """
    paddle.set_device("cpu")
    os.environ["HB_LOSS_GLOBAL_TOKEN_AVG"] = "1"
    _clear_tracker()
    try:
        # batch A: mean 5 over 4 tokens ; batch B: mean 2 over 96 tokens.
        _reduce(paddle.to_tensor(20.0), paddle.to_tensor(4.0))
        _reduce(paddle.to_tensor(192.0), paddle.to_tensor(96.0))

        got = LanguageLoss.consume_token_avg_loss()
        expected = (20.0 + 192.0) / (4.0 + 96.0)  # = 2.12
        assert got is not None
        assert abs(got - expected) < 1e-6, (got, expected)
        # Simple average of the two means would be (5 + 2) / 2 = 3.5 -> differs.
        assert abs(got - 3.5) > 0.1
        # Tracker is cleared after consumption.
        assert LanguageLoss.token_avg_tracker == {}
    finally:
        os.environ.pop("HB_LOSS_GLOBAL_TOKEN_AVG", None)
        _clear_tracker()


def test_consume_returns_none_when_empty():
    paddle.set_device("cpu")
    _clear_tracker()
    assert LanguageLoss.consume_token_avg_loss() is None
    # Still empty / consumable again.
    assert LanguageLoss.consume_token_avg_loss() is None


if __name__ == "__main__":
    import sys

    try:
        test_disabled_returns_local_mean_and_leaves_tracker_empty()
        test_enabled_backprop_value_unchanged_and_accumulates()
        test_consume_is_token_weighted_across_microbatches()
        test_consume_returns_none_when_empty()
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("OK")
