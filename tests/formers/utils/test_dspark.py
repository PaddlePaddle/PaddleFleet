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

import numpy as np
import paddle
import pytest

from paddlefleet.utils.dspark import ReadOnlyProjection, readonly_embedding


@pytest.fixture(autouse=True, params=["cpu", "gpu:0"])
def device(request):
    if request.param.startswith("gpu") and (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() == 0
    ):
        pytest.skip("CUDA device unavailable")
    previous = paddle.device.get_device()
    paddle.set_device(request.param)
    try:
        yield
    finally:
        paddle.set_device(previous)


def _multimax_numpy(logits, ranges, scales):
    """Independent forward/derivative oracle away from the ReLU knots."""
    out = logits.copy()
    derivative = np.ones_like(logits)
    for i, (sign, power) in enumerate([(-1, 1), (1, 1), (-1, 2), (1, 2)]):
        distance = sign * (logits - ranges[i])
        active = distance > 0
        out += scales[i] * np.maximum(distance, 0) ** power
        derivative += (
            scales[i]
            * power
            * sign
            * np.where(active, distance ** (power - 1), 0)
        )
    return out, derivative


def test_readonly_embedding_uses_current_weight_without_weight_gradient():
    weight = paddle.arange(20, dtype="float32").reshape([5, 4])
    weight.stop_gradient = False
    token_ids = paddle.to_tensor([[0, 3], [4, 1]], dtype="int64")

    actual = readonly_embedding(token_ids, weight)
    expected = paddle.to_tensor(weight.numpy()[token_ids.numpy()])
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert actual.stop_gradient

    weight.set_value(weight + 1)
    updated = readonly_embedding(token_ids, weight)
    np.testing.assert_array_equal(updated.numpy(), expected.numpy() + 1)
    assert weight.stop_gradient is False
    assert weight.grad is None
    # The native target branch can still train these very same weights.
    paddle.nn.functional.embedding(token_ids, weight).sum().backward()
    expected_grad = np.zeros([5, 4], dtype="float32")
    expected_grad[[0, 1, 3, 4]] = 1
    np.testing.assert_array_equal(weight.grad.numpy(), expected_grad)


@pytest.mark.parametrize("transpose_y", [True, False])
@pytest.mark.parametrize("vocab_size", [4, 6])
def test_projection_supports_both_weight_layouts_and_preserves_hidden_gradient(
    transpose_y,
    vocab_size,
):
    paddle.seed(10)
    hidden = paddle.randn([2, 3, 4])
    hidden.stop_gradient = False
    weight = paddle.randn([vocab_size, 4] if transpose_y else [4, vocab_size])
    weight.stop_gradient = False
    projection = ReadOnlyProjection(weight, transpose_y=transpose_y)

    actual = projection(hidden)
    matrix = weight.numpy().T if transpose_y else weight.numpy()
    expected = hidden.numpy() @ matrix
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert projection.weight.stop_gradient
    assert projection.weight is not weight
    actual.sum().backward()
    expected_grad = np.broadcast_to(matrix.sum(axis=-1), hidden.shape)
    np.testing.assert_allclose(hidden.grad.numpy(), expected_grad, atol=1e-6)
    assert weight.grad is None
    assert not weight.stop_gradient


def test_projection_reads_updated_target_weight_each_call():
    weight = paddle.zeros([3, 2], dtype="float32")
    projection = ReadOnlyProjection(weight, transpose_y=True)
    hidden = paddle.ones([1, 2], dtype="float32")
    np.testing.assert_array_equal(projection(hidden).numpy(), np.zeros([1, 3]))

    weight.set_value(paddle.arange(6, dtype="float32").reshape([3, 2]))
    np.testing.assert_array_equal(
        projection(hidden).numpy(), np.array([[1, 5, 9]], dtype="float32")
    )


@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("with_multimax", [True, False])
def test_projection_applies_bias_and_multimax_without_target_gradients(
    with_bias,
    with_multimax,
):
    paddle.seed(11)
    hidden = paddle.randn([2, 3, 4])
    hidden.stop_gradient = False
    weight = paddle.randn([5, 4])
    bias = paddle.randn([5])
    ranges = paddle.to_tensor([-0.5, 0.1, 0.7, 1.2], dtype="float32")
    scales = paddle.to_tensor([0.2, -0.3, 0.05, 0.04], dtype="float32")
    for value in (weight, bias, ranges, scales):
        value.stop_gradient = False
    projection = ReadOnlyProjection(
        weight,
        transpose_y=True,
        bias=bias if with_bias else None,
        multimax_ranges=ranges if with_multimax else None,
        multimax_ts=scales if with_multimax else None,
    )

    linear = hidden.numpy() @ weight.numpy().T
    if with_bias:
        linear += bias.numpy()
    expected, derivative = (
        _multimax_numpy(linear, ranges.numpy(), scales.numpy())
        if with_multimax
        else (linear, np.ones_like(linear))
    )
    actual = projection(hidden)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
    actual.square().sum().backward()
    expected_grad = (2 * expected * derivative) @ weight.numpy()
    np.testing.assert_allclose(
        hidden.grad.numpy(), expected_grad, rtol=2e-6, atol=2e-6
    )
    for value in (weight, bias, ranges, scales):
        assert value.stop_gradient is False
        assert value.grad is None
    for name in ("weight", "bias", "multimax_ranges", "multimax_ts"):
        value = getattr(projection, name)
        if value is not None:
            assert value.stop_gradient


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("transpose_y", [True, False])
@pytest.mark.parametrize("transform_only", [True, False])
def test_projection_bias_matches_native_autocast_and_preserves_gradients(
    dtype, transpose_y, transform_only
):
    if paddle.device.get_device() == "cpu":
        pytest.skip("autocast projection parity test requires GPU")
    if dtype == "bfloat16" and not paddle.amp.is_bfloat16_supported():
        pytest.skip("BF16 autocast unavailable")

    paddle.seed(19)
    hidden = paddle.randn([2, 3, 16], dtype="float32")
    reference_hidden = hidden.detach().clone()
    weight = paddle.randn(
        [20, 16] if transpose_y else [16, 20], dtype="float32"
    )
    bias = paddle.randn([20], dtype="float32")
    for value in (hidden, reference_hidden, weight, bias):
        value.stop_gradient = False
    projection = ReadOnlyProjection(weight, transpose_y=transpose_y, bias=bias)

    with paddle.amp.auto_cast(dtype=dtype, level="O1"):
        native = paddle.nn.functional.linear(
            reference_hidden, weight.T if transpose_y else weight, bias
        )
        if transform_only:
            logits = paddle.matmul(
                hidden, projection.weight, transpose_y=transpose_y
            )
            actual = projection.transform_logits(logits)
        else:
            actual = projection(hidden)

    assert native.dtype == getattr(paddle, dtype)
    assert actual.dtype == native.dtype
    # Native linear can fuse bias into GEMM, while streamed callers add it
    # after the matmul has rounded. Allow the corresponding low-precision ULPs.
    tolerance = 2e-2 if dtype == "bfloat16" else 2e-3
    np.testing.assert_allclose(
        actual.astype("float32").numpy(),
        native.astype("float32").numpy(),
        rtol=tolerance,
        atol=tolerance,
    )
    actual.astype("float32").sum().backward()
    for value in (weight, bias):
        assert value.dtype == paddle.float32
        assert not value.stop_gradient
        assert value.grad is None
    native.astype("float32").sum().backward()
    np.testing.assert_allclose(
        hidden.grad.numpy(),
        reference_hidden.grad.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def test_native_loss_gradients_unchanged_with_readonly_draft_branch():
    head = paddle.nn.Linear(2, 3)
    head.weight.set_value(
        paddle.to_tensor([[0.2, -0.4, 0.6], [0.3, 0.5, -0.2]])
    )
    head.bias.set_value(paddle.to_tensor([0.1, 0.2, 0.3]))
    inputs = paddle.to_tensor([[1.0, -2.0]])
    head(inputs).square().sum().backward()
    native_grads = [p.grad.numpy().copy() for p in head.parameters()]
    head.clear_gradients()
    draft_hidden = paddle.to_tensor([[0.4, -0.7]], stop_gradient=False)
    projection = ReadOnlyProjection(head.weight, False, bias=head.bias)
    draft_loss = projection(draft_hidden).square().sum()
    (head(inputs).square().sum() + 0.5 * draft_loss).backward()
    for param, expected in zip(head.parameters(), native_grads):
        np.testing.assert_allclose(param.grad.numpy(), expected, atol=1e-6)
        assert not param.stop_gradient
    assert float(paddle.abs(draft_hidden.grad).sum()) > 0
    before = projection(inputs).numpy().copy()
    paddle.optimizer.SGD(learning_rate=0.1, parameters=head.parameters()).step()
    after = ReadOnlyProjection(head.weight, False, bias=head.bias)(inputs)
    np.testing.assert_allclose(after.numpy(), head(inputs).numpy(), atol=1e-6)
    assert not np.allclose(after.numpy(), before)


def test_transform_logits_does_not_mutate_input_and_zero_multimax_is_identity():
    logits = paddle.to_tensor([[-2.0, 0.5, 3.0]], stop_gradient=False)
    original = logits.numpy().copy()
    projection = ReadOnlyProjection(
        paddle.ones([3, 2]),
        True,
        multimax_ranges=paddle.zeros([4]),
        multimax_ts=paddle.zeros([4]),
    )
    result = projection.transform_logits(logits)
    np.testing.assert_array_equal(result.numpy(), original)
    result.sum().backward()
    np.testing.assert_array_equal(logits.grad.numpy(), np.ones_like(original))
    projection = ReadOnlyProjection(
        paddle.ones([3, 2]), True, bias=paddle.ones([3])
    )
    np.testing.assert_array_equal(
        projection.transform_logits(logits).numpy(), original + 1
    )
    np.testing.assert_array_equal(logits.numpy(), original)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ReadOnlyProjection(paddle.ones([3]), transpose_y=True),
        lambda: ReadOnlyProjection(paddle.ones([3, 2]), transpose_y=1),
        lambda: ReadOnlyProjection(
            paddle.ones([3, 2]), transpose_y=True, bias=paddle.ones([2])
        ),
        lambda: ReadOnlyProjection(
            paddle.ones([3, 2]),
            transpose_y=True,
            multimax_ranges=paddle.ones([4]),
        ),
        lambda: ReadOnlyProjection(
            paddle.ones([3, 2]),
            transpose_y=True,
            multimax_ranges=paddle.ones([3]),
            multimax_ts=paddle.ones([4]),
        ),
        lambda: ReadOnlyProjection(
            paddle.ones([3, 2]), transpose_y=False, bias=paddle.ones([3])
        ),
        lambda: ReadOnlyProjection(
            paddle.ones([3, 2]), transpose_y=True, multimax_ts=paddle.ones([4])
        ),
        lambda: ReadOnlyProjection(
            paddle.ones([3, 2]),
            transpose_y=True,
            multimax_ranges=paddle.ones([4]),
            multimax_ts=paddle.ones([1, 4]),
        ),
    ],
)
def test_projection_rejects_invalid_contracts(factory):
    with pytest.raises(ValueError):
        factory()


def test_embedding_rejects_non_matrix_weight():
    with pytest.raises(ValueError, match=r"\[vocab, hidden\]"):
        readonly_embedding(paddle.ones([2, 3, 4]), paddle.ones([2, 3, 4]))
