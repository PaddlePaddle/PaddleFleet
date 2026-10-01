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

"""Unit tests for the module-level helpers of ``multi_latent_attention``.

The accuracy-compatible projection helpers keep the forward value of a
plain linear while pinning *where* each gradient comes from, and the MLA
RoPE helper pins the Megatron ordering (interleave, then rotate-half) for
explicit position ids. Every test asserts the numeric result and, where a
helper exists only for a gradient, the gradient it routes.

Single card: the sequence-parallel and tensor-parallel arms are driven
with patched collectives and a patched ``get_pg_size``, so no process
group is needed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.tensor_parallel import mappings
from paddlefleet.transformer import multi_latent_attention as mla

TP_GROUP = "tp-group-sentinel"


def _tensor(array, stop_gradient=False):
    out = paddle.to_tensor(np.asarray(array, dtype="float32"))
    out.stop_gradient = stop_gradient
    return out


def _ramp(*shape):
    total = int(np.prod(shape))
    values = np.linspace(-1.5, 2.0, total, dtype="float32")
    return _tensor(values.reshape(shape))


def _projection(weight_rows, *, skip_bias_add, sequence_parallel=False):
    """Stand-in for the Linear variants the helpers are handed."""
    return SimpleNamespace(
        weight=_tensor(weight_rows),
        bias=_tensor([0.25] * np.asarray(weight_rows).shape[1]),
        skip_bias_add=skip_bias_add,
        sequence_parallel=sequence_parallel,
        tp_group=TP_GROUP,
    )


class LinearInputGradPyLayerTest(unittest.TestCase):
    """``_AccuracyCompatibleLinearInputGrad``: value path, dgrad only."""

    hidden_rows = [[1.0, 2.0, -1.0, 0.5], [0.0, -3.0, 2.0, 1.5]]
    weight_rows = [[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0], [0.25, -2.0]]

    def test_forward_is_plain_linear(self):
        hidden = _tensor(self.hidden_rows)
        weight = _tensor(self.weight_rows, stop_gradient=True)

        out = mla._AccuracyCompatibleLinearInputGrad.apply(hidden, weight)

        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, weight).numpy()
        )

    def test_backward_uses_materialized_transpose_and_drops_wgrad(self):
        hidden = _tensor(self.hidden_rows)
        weight = _tensor(self.weight_rows, stop_gradient=True)

        mla._AccuracyCompatibleLinearInputGrad.apply(
            hidden, weight
        ).sum().backward()

        weight_np = np.asarray(self.weight_rows, dtype="float32")
        grad_out = np.ones(
            (len(self.hidden_rows), weight_np.shape[1]), dtype="float32"
        )
        np.testing.assert_array_equal(
            hidden.grad.numpy(), grad_out @ weight_np.T
        )
        # The PyLayer returns None for the weight: wgrad belongs to the
        # parameter path of the caller, not to this dgrad-only branch.
        self.assertIsNone(weight.grad)


class QUpProjectionTest(unittest.TestCase):
    """``_accuracy_compatible_q_up_projection``: value + split grads."""

    hidden_rows = [[1.0, 2.0, -1.0, 0.5], [0.0, -3.0, 2.0, 1.5]]
    weight_rows = [[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0], [0.25, -2.0]]

    def test_replicated_value_is_a_plain_linear(self):
        projection = _projection(self.weight_rows, skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        with patch.object(mla, "get_pg_size", return_value=1):
            out, out_bias = mla._accuracy_compatible_q_up_projection(
                projection, hidden
            )

        # The input path is added as ``x - x.detach()``, i.e. numerically
        # zero: the value must stay bit-identical to F.linear.
        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, projection.weight).numpy()
        )
        self.assertIs(out_bias, projection.bias)

    def test_fused_bias_is_not_returned_to_the_caller(self):
        projection = _projection(self.weight_rows, skip_bias_add=False)
        hidden = _tensor(self.hidden_rows)

        with patch.object(mla, "get_pg_size", return_value=1):
            out, out_bias = mla._accuracy_compatible_q_up_projection(
                projection, hidden
            )

        self.assertIsNone(out_bias)
        # q-up never folds the bias in: the value is still the bias-free
        # linear even though skip_bias_add is off.
        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, projection.weight).numpy()
        )

    def test_wgrad_is_native_and_dgrad_uses_the_transpose(self):
        projection = _projection(self.weight_rows, skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        with patch.object(mla, "get_pg_size", return_value=1):
            out, _ = mla._accuracy_compatible_q_up_projection(
                projection, hidden
            )
        out.sum().backward()

        hidden_np = np.asarray(self.hidden_rows, dtype="float32")
        weight_np = np.asarray(self.weight_rows, dtype="float32")
        grad_out = np.ones((hidden_np.shape[0], weight_np.shape[1]), "float32")
        np.testing.assert_array_equal(
            projection.weight.grad.numpy(), hidden_np.T @ grad_out
        )
        np.testing.assert_array_equal(
            hidden.grad.numpy(), grad_out @ weight_np.T
        )

    def test_sequence_parallel_gathers_before_the_linear(self):
        projection = _projection(
            self.weight_rows, skip_bias_add=True, sequence_parallel=True
        )
        hidden = _tensor(self.hidden_rows)
        gathered = paddle.concat([hidden, hidden], axis=0)

        with (
            patch.object(mla, "get_pg_size", return_value=2),
            patch.object(
                mappings,
                "gather_from_sequence_parallel_region",
                side_effect=lambda value, group=None: paddle.concat(
                    [value, value], axis=0
                ),
            ) as gather,
        ):
            out, _ = mla._accuracy_compatible_q_up_projection(
                projection, hidden
            )

        gather.assert_called_once()
        self.assertEqual(gather.call_args.kwargs["group"], TP_GROUP)
        # ColumnParallelLinear must not gather twice: F.linear runs on the
        # gathered sequence and the local output shard is kept as is.
        self.assertEqual(list(out.shape), [4, 2])
        np.testing.assert_array_equal(
            out.numpy(), F.linear(gathered, projection.weight).numpy()
        )


class OutputProjectionTest(unittest.TestCase):
    """``_accuracy_compatible_projection``: local GEMM plus TP reduction."""

    hidden_rows = [[1.0, 2.0, -1.0, 0.5], [0.0, -3.0, 2.0, 1.5]]
    weight_rows = [[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0], [0.25, -2.0]]

    def test_single_rank_folds_the_bias_and_skips_the_collectives(self):
        projection = _projection(self.weight_rows, skip_bias_add=False)
        hidden = _tensor(self.hidden_rows)

        with (
            patch.object(mla, "get_pg_size", return_value=1),
            patch.object(
                mla, "reduce_from_tensor_model_parallel_region"
            ) as all_reduce,
            patch.object(
                mla, "reduce_scatter_to_sequence_parallel_region"
            ) as reduce_scatter,
        ):
            out, out_bias = mla._accuracy_compatible_projection(
                projection, hidden
            )

        self.assertIsNone(out_bias)
        all_reduce.assert_not_called()
        reduce_scatter.assert_not_called()
        np.testing.assert_array_equal(
            out.numpy(),
            F.linear(hidden, projection.weight, projection.bias).numpy(),
        )

    def test_skip_bias_add_defers_the_bias_and_drops_it_from_the_gemm(self):
        projection = _projection(self.weight_rows, skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        with patch.object(mla, "get_pg_size", return_value=1):
            out, out_bias = mla._accuracy_compatible_projection(
                projection, hidden
            )

        self.assertIs(out_bias, projection.bias)
        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, projection.weight).numpy()
        )

    def test_tensor_parallel_all_reduces_the_local_gemm(self):
        projection = _projection(self.weight_rows, skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        with (
            patch.object(mla, "get_pg_size", return_value=2),
            patch.object(
                mla,
                "reduce_from_tensor_model_parallel_region",
                side_effect=lambda value, group=None: value * 2.0,
            ) as all_reduce,
            patch.object(
                mla, "reduce_scatter_to_sequence_parallel_region"
            ) as reduce_scatter,
        ):
            out, _ = mla._accuracy_compatible_projection(projection, hidden)

        all_reduce.assert_called_once()
        self.assertEqual(all_reduce.call_args.kwargs["group"], TP_GROUP)
        reduce_scatter.assert_not_called()
        np.testing.assert_array_equal(
            out.numpy(), (F.linear(hidden, projection.weight) * 2.0).numpy()
        )

    def test_sequence_parallel_reduce_scatters_instead(self):
        projection = _projection(
            self.weight_rows, skip_bias_add=True, sequence_parallel=True
        )
        hidden = _tensor(self.hidden_rows)

        with (
            patch.object(mla, "get_pg_size", return_value=2),
            patch.object(
                mla,
                "reduce_from_tensor_model_parallel_region",
            ) as all_reduce,
            patch.object(
                mla,
                "reduce_scatter_to_sequence_parallel_region",
                side_effect=lambda value, group=None: value[:1],
            ) as reduce_scatter,
        ):
            out, _ = mla._accuracy_compatible_projection(projection, hidden)

        reduce_scatter.assert_called_once()
        self.assertEqual(reduce_scatter.call_args.kwargs["group"], TP_GROUP)
        all_reduce.assert_not_called()
        # The residual layout follows the reduce-scatter shard.
        self.assertEqual(list(out.shape), [1, 2])
        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, projection.weight).numpy()[:1]
        )


class QDownProjectionTest(unittest.TestCase):
    """``_accuracy_compatible_q_down_projection``: bias and SP gather."""

    hidden_rows = [[1.0, 2.0, -1.0, 0.5], [0.0, -3.0, 2.0, 1.5]]
    weight_rows = [[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0], [0.25, -2.0]]

    def test_replicated_linear_folds_the_bias(self):
        projection = _projection(self.weight_rows, skip_bias_add=False)
        hidden = _tensor(self.hidden_rows)

        with patch.object(mla, "get_pg_size", return_value=1):
            out, out_bias = mla._accuracy_compatible_q_down_projection(
                projection, hidden
            )

        self.assertIsNone(out_bias)
        np.testing.assert_array_equal(
            out.numpy(),
            F.linear(hidden, projection.weight, projection.bias).numpy(),
        )

    def test_skip_bias_add_hands_the_bias_back(self):
        projection = _projection(self.weight_rows, skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        with patch.object(mla, "get_pg_size", return_value=1):
            out, out_bias = mla._accuracy_compatible_q_down_projection(
                projection, hidden
            )

        self.assertIs(out_bias, projection.bias)
        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, projection.weight).numpy()
        )

    def test_sequence_parallel_gathers_the_sequence_first(self):
        projection = _projection(
            self.weight_rows, skip_bias_add=True, sequence_parallel=True
        )
        hidden = _tensor(self.hidden_rows)
        gathered = paddle.concat([hidden, hidden], axis=0)

        with (
            patch.object(mla, "get_pg_size", return_value=2),
            patch.object(
                mappings,
                "gather_from_sequence_parallel_region",
                side_effect=lambda value, group=None: paddle.concat(
                    [value, value], axis=0
                ),
            ) as gather,
        ):
            out, _ = mla._accuracy_compatible_q_down_projection(
                projection, hidden
            )

        gather.assert_called_once()
        self.assertEqual(gather.call_args.kwargs["group"], TP_GROUP)
        self.assertEqual(list(out.shape), [4, 2])
        np.testing.assert_array_equal(
            out.numpy(), F.linear(gathered, projection.weight).numpy()
        )

    def test_replicated_projection_never_takes_the_gather_branch(self):
        # A replicated Linear does not set sequence_parallel on itself, so
        # even with TP > 1 the sequence must stay local.
        projection = _projection(self.weight_rows, skip_bias_add=True)
        hidden = _tensor(self.hidden_rows)

        with (
            patch.object(mla, "get_pg_size", return_value=2),
            patch.object(
                mappings, "gather_from_sequence_parallel_region"
            ) as gather,
        ):
            out, _ = mla._accuracy_compatible_q_down_projection(
                projection, hidden
            )

        gather.assert_not_called()
        np.testing.assert_array_equal(
            out.numpy(), F.linear(hidden, projection.weight).numpy()
        )


def _rope_reference(tensor, positions, base, sequence_parallel):
    """Independent float64 reference for the Megatron MLA RoPE ordering."""
    array = np.asarray(tensor.numpy(), dtype=np.float64)
    head_dim = array.shape[-1]
    inv_freq = np.power(
        float(base),
        -np.arange(0, head_dim, 2, dtype=np.float64) / float(head_dim),
    )
    freqs = np.outer(np.asarray(positions, dtype=np.float64), inv_freq)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    if sequence_parallel:
        freqs = freqs[:, None, None, :]
    else:
        freqs = freqs[None, :, None, :]
    interleaved = np.concatenate([array[..., 0::2], array[..., 1::2]], axis=-1)
    half = head_dim // 2
    rotated = np.concatenate(
        [-interleaved[..., half:], interleaved[..., :half]], axis=-1
    )
    return interleaved * np.cos(freqs) + rotated * np.sin(freqs)


class MlaRopeApplyTest(unittest.TestCase):
    """``_accuracy_compatible_mla_rope_apply``: ordering and shard offsets."""

    base = 10000.0

    def test_batch_first_matches_the_reference_ordering(self):
        q_pe = _ramp(1, 3, 2, 4)
        k_pe = _ramp(1, 3, 1, 4)
        position_ids = paddle.to_tensor([0, 1, 2], dtype="int64")

        q_out, k_out = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, position_ids
        )

        self.assertEqual(list(q_out.shape), [1, 3, 2, 4])
        self.assertEqual(list(k_out.shape), [1, 3, 1, 4])
        np.testing.assert_allclose(
            q_out.numpy(),
            _rope_reference(q_pe, [0, 1, 2], self.base, False),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            k_out.numpy(),
            _rope_reference(k_pe, [0, 1, 2], self.base, False),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_position_zero_is_the_bare_interleave(self):
        q_pe = _ramp(1, 2, 1, 4)
        position_ids = paddle.to_tensor([0, 1], dtype="int64")

        q_out, _ = mla._accuracy_compatible_mla_rope_apply(
            q_pe, q_pe, self.base, position_ids
        )

        # cos(0) == 1 and sin(0) == 0, so row 0 must come out as the
        # even-then-odd interleave with no rotation mixed in.
        interleave = paddle.concat([q_pe[..., 0::2], q_pe[..., 1::2]], axis=-1)
        np.testing.assert_array_equal(
            q_out.numpy()[:, 0], interleave.numpy()[:, 0]
        )

    def test_sequence_parallel_layout_rotates_axis_zero(self):
        q_pe = _ramp(3, 1, 2, 4)
        k_pe = _ramp(3, 1, 1, 4)
        position_ids = paddle.to_tensor([0, 1, 2], dtype="int64")

        q_out, k_out = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, position_ids, sequence_parallel=True
        )

        np.testing.assert_allclose(
            q_out.numpy(),
            _rope_reference(q_pe, [0, 1, 2], self.base, True),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            k_out.numpy(),
            _rope_reference(k_pe, [0, 1, 2], self.base, True),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_two_dimensional_position_ids_are_flattened(self):
        q_pe = _ramp(1, 3, 2, 4)
        k_pe = _ramp(1, 3, 1, 4)
        flat = paddle.to_tensor([0, 1, 2], dtype="int64")
        batched = paddle.to_tensor([[0, 1, 2]], dtype="int64")

        q_flat, k_flat = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, flat
        )
        q_batched, k_batched = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, batched
        )

        np.testing.assert_array_equal(q_batched.numpy(), q_flat.numpy())
        np.testing.assert_array_equal(k_batched.numpy(), k_flat.numpy())

    def test_longer_position_ids_are_truncated_to_the_query_length(self):
        # MTP keeps one look-ahead token in the batch while the attention
        # tensors hold only the first seq_len positions.
        q_pe = _ramp(1, 3, 2, 4)
        k_pe = _ramp(1, 3, 1, 4)
        exact = paddle.to_tensor([0, 1, 2], dtype="int64")
        with_lookahead = paddle.to_tensor([0, 1, 2, 3], dtype="int64")

        q_exact, k_exact = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, exact
        )
        q_extra, k_extra = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, with_lookahead
        )

        np.testing.assert_array_equal(q_extra.numpy(), q_exact.numpy())
        np.testing.assert_array_equal(k_extra.numpy(), k_exact.numpy())

    def test_key_shard_offset_rotates_the_shard_positions(self):
        q_pe = _ramp(1, 4, 2, 4)
        k_pe = _ramp(1, 2, 1, 4)
        position_ids = paddle.to_tensor([0, 1, 2, 3], dtype="int64")

        _, k_shard = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, position_ids, k_seq_offset=2
        )
        _, k_rank0 = mla._accuracy_compatible_mla_rope_apply(
            q_pe, k_pe, self.base, position_ids, k_seq_offset=0
        )

        np.testing.assert_allclose(
            k_shard.numpy(),
            _rope_reference(k_pe, [2, 3], self.base, False),
            rtol=1e-6,
            atol=1e-6,
        )
        # The offset really moves the table: rank 1 must not reuse the
        # positions rank 0 rotated with.
        self.assertFalse(np.allclose(k_shard.numpy(), k_rank0.numpy()))

    def test_position_ids_shorter_than_the_query_are_rejected(self):
        q_pe = _ramp(1, 3, 2, 4)
        k_pe = _ramp(1, 3, 1, 4)
        position_ids = paddle.to_tensor([0, 1], dtype="int64")

        with self.assertRaisesRegex(
            ValueError, "shorter than the query sequence length"
        ):
            mla._accuracy_compatible_mla_rope_apply(
                q_pe, k_pe, self.base, position_ids
            )

    def test_position_ids_shorter_than_the_key_shard_end_are_rejected(self):
        q_pe = _ramp(1, 2, 2, 4)
        k_pe = _ramp(1, 3, 1, 4)
        position_ids = paddle.to_tensor([0, 1, 2], dtype="int64")

        with self.assertRaisesRegex(
            ValueError, "shorter than the key shard end"
        ):
            mla._accuracy_compatible_mla_rope_apply(
                q_pe, k_pe, self.base, position_ids, k_seq_offset=1
            )


if __name__ == "__main__":
    unittest.main()
