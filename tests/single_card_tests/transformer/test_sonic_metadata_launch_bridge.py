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

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle
import pytest

from paddlefleet.transformer.moe import fused_a2a, fusion_layer_utils


def _bytes_equal(lhs, rhs):
    return bool(paddle.all(lhs.view(paddle.uint8) == rhs.view(paddle.uint8)))


def test_scale_word_capability_detection_is_import_safe():
    class OpaqueQuantizer:
        @property
        def __signature__(self):
            raise ValueError("opaque")

        def __call__(self, value):
            return value

    assert not fused_a2a._supports_sonic_scale_word_packing(None)
    assert not fused_a2a._supports_sonic_scale_word_packing(OpaqueQuantizer())
    assert fused_a2a._supports_sonic_scale_word_packing(
        lambda value, pack_scale_words=False: value
    )


def test_scale_carrier_aligned_is_zero_copy_and_bit_exact():
    x_fp8 = paddle.empty([5, 256], dtype=paddle.uint8)
    raw = paddle.randint(0, 256, [5, 8], dtype=paddle.int32).cast(paddle.uint8)

    carrier = fused_a2a._pack_sonic_fp8_scale_for_deepep(x_fp8, raw)
    restored = fused_a2a._unpack_sonic_fp8_scale_from_deepep(x_fp8, carrier)

    assert carrier.dtype == paddle.int32
    assert tuple(carrier.shape) == (5, 2)
    assert carrier.data_ptr() == raw.data_ptr()
    assert restored.data_ptr() == raw.data_ptr()
    assert _bytes_equal(restored, raw)


def test_scale_carrier_non_aligned_uses_value_fallback():
    x_fp8 = paddle.empty([3, 160], dtype=paddle.uint8)
    raw = paddle.randint(0, 256, [3, 5], dtype=paddle.int32).cast(paddle.uint8)

    carrier = fused_a2a._pack_sonic_fp8_scale_for_deepep(x_fp8, raw)
    restored = fused_a2a._unpack_sonic_fp8_scale_from_deepep(x_fp8, carrier)

    assert carrier.dtype == paddle.int32
    assert tuple(carrier.shape) == (3, 5)
    assert bool(paddle.all(restored == raw))


def test_scale_carrier_rejects_invalid_shape_and_dtype():
    x_fp8 = paddle.empty([2, 256], dtype=paddle.uint8)
    with pytest.raises(RuntimeError, match="Invalid Sonic FP8 scale shape"):
        fused_a2a._pack_sonic_fp8_scale_for_deepep(
            x_fp8, paddle.empty([2, 7], dtype=paddle.uint8)
        )
    with pytest.raises(TypeError, match="expects raw uint8"):
        fused_a2a._pack_sonic_fp8_scale_for_deepep(
            x_fp8, paddle.empty([2, 8], dtype=paddle.float32)
        )
    with pytest.raises(RuntimeError, match="Invalid packed Sonic FP8 scale"):
        fused_a2a._unpack_sonic_fp8_scale_from_deepep(
            x_fp8, paddle.empty([2, 3], dtype=paddle.int32)
        )


def test_scale_carrier_normalizes_noncontiguous_inputs():
    x_fp8 = paddle.empty([5, 256], dtype=paddle.uint8)
    raw = (
        paddle.randint(0, 256, [8, 5], dtype=paddle.int32)
        .cast(paddle.uint8)
        .transpose([1, 0])
    )
    assert not raw.is_contiguous()

    carrier = fused_a2a._pack_sonic_fp8_scale_for_deepep(x_fp8, raw)
    noncontiguous_carrier = (
        carrier.transpose([1, 0]).contiguous().transpose([1, 0])
    )
    assert not noncontiguous_carrier.is_contiguous()

    repacked = fused_a2a._pack_sonic_fp8_scale_for_deepep(
        x_fp8, noncontiguous_carrier
    )
    restored = fused_a2a._unpack_sonic_fp8_scale_from_deepep(
        x_fp8, noncontiguous_carrier
    )
    assert tuple(repacked.shape) == (5, 2)
    assert _bytes_equal(restored, raw.contiguous())


def test_combine_helpers_preserve_sonic_scale_bytes():
    grad = paddle.randn([3, 256], dtype=paddle.float32)
    fp8 = paddle.empty([3, 256], dtype=paddle.uint8)
    raw = paddle.randint(0, 256, [3, 8], dtype=paddle.int32).cast(paddle.uint8)
    handle = {"using_sonic_moe": True}

    with (
        patch.object(fused_a2a, "_SONIC_PACK_SCALE_WORDS_AVAILABLE", True),
        patch.object(
            fused_a2a,
            "quantize_activation_blockscaled_fast",
            return_value=(fp8, raw),
        ) as quantize,
    ):
        packed = fused_a2a._quantize_combine_grad_for_deepep(grad, handle)
        restored = fused_a2a._record_fp8_combine_grad(packed, handle)

    assert quantize.call_args.kwargs == {"pack_scale_words": True}
    assert restored is fp8
    assert handle["data"] is fp8
    assert _bytes_equal(handle["scale"], raw)


def test_combine_helpers_keep_legacy_quantizer_contract():
    grad = paddle.randn([2, 160], dtype=paddle.float32)
    expected = (MagicMock(), MagicMock())
    with (
        patch.object(fused_a2a, "_SONIC_PACK_SCALE_WORDS_AVAILABLE", False),
        patch.object(
            fused_a2a,
            "quantize_activation_blockscaled_fast",
            return_value=expected,
        ) as quantize,
    ):
        actual = fused_a2a._quantize_combine_grad_for_deepep(
            grad, {"using_sonic_moe": True}
        )

    assert actual is expected
    assert quantize.call_args.kwargs == {"scale_dtype": paddle.int32}
    with pytest.raises(RuntimeError, match="combine_grad_handle"):
        fused_a2a._record_fp8_combine_grad(expected, None)


@pytest.mark.parametrize("packed_words", [True, False])
def test_dispatch_supports_new_and_legacy_sonic_quantizers(packed_words):
    ctx = MagicMock()
    x = paddle.randn([2, 256], dtype=paddle.float32)
    fp8 = paddle.empty([2, 256], dtype=paddle.uint8)
    raw = paddle.randint(0, 256, [2, 8], dtype=paddle.int32).cast(paddle.uint8)
    scale = raw if packed_words else raw.cast(paddle.int32)
    token_indices = MagicMock()
    token_probs = MagicMock()
    states = {"handle": MagicMock()}

    def dispatch(value, *_args, **_kwargs):
        return value, token_probs, states, None

    with (
        patch.object(
            fused_a2a, "_SONIC_PACK_SCALE_WORDS_AVAILABLE", packed_words
        ),
        patch.object(
            fused_a2a,
            "quantize_activation_blockscaled_fast",
            return_value=(fp8, scale),
        ) as quantize,
        patch.object(
            fused_a2a,
            "fused_dispatch_forward_func",
            side_effect=dispatch,
        ),
    ):
        recv_x, _, _, handle = fused_a2a.DeepEPDispatch.forward(
            ctx,
            x,
            token_indices,
            token_probs,
            2,
            MagicMock(),
            fp8_dispatch=True,
            using_sonic_moe=True,
        )

    assert recv_x is fp8
    assert handle["scale"].dtype == paddle.int32
    expected_kwargs = (
        {"pack_scale_words": True}
        if packed_words
        else {"scale_dtype": paddle.int32}
    )
    assert quantize.call_args.kwargs == expected_kwargs


@pytest.mark.parametrize(
    "combine_cls",
    [
        fused_a2a.DeepEPCombine,
        fused_a2a.DeepEPCombineAsync,
        fused_a2a.DeepEPCombineAsyncFunctor,
    ],
)
def test_all_combine_variants_use_shared_scale_bridge(combine_cls):
    grad_output = MagicMock()
    grad_data = MagicMock()
    ctx = SimpleNamespace(
        fp8_dispatch=True,
        combine_grad_handle={"using_sonic_moe": True},
        group=SimpleNamespace(id=1),
        handle=MagicMock(),
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        moe_ep_barrier=True,
        bwf=lambda *args: (),
    )
    with (
        patch.object(
            fused_a2a,
            "_quantize_combine_grad_for_deepep",
            return_value=MagicMock(),
        ) as quantize,
        patch.object(
            fused_a2a,
            "fused_combine_backward_func",
            return_value=MagicMock(),
        ),
        patch.object(
            fused_a2a,
            "_record_fp8_combine_grad",
            return_value=grad_data,
        ) as record,
        patch.object(fused_a2a, "wait_for_deepep"),
    ):
        result = combine_cls.backward(ctx, grad_output)

    quantize.assert_called_once_with(grad_output, ctx.combine_grad_handle)
    record.assert_called_once()
    if combine_cls is fused_a2a.DeepEPCombine:
        assert result is grad_data
    else:
        assert result == (grad_data,)


def _mock_sonic_metadata(with_gated_outputs, packed_scale=True):
    values = [MagicMock() for _ in range(6)]
    values.extend([256, 8, 4, MagicMock(), MagicMock()])
    if not packed_scale:
        values[10] = None
    if with_gated_outputs:
        values.extend(MagicMock() for _ in range(4))
    return tuple(values)


def _run_sonic_with_mocks(has_bridge, packed_scale=True):
    hidden_states = MagicMock(shape=(4, 256), dtype=paddle.float8_e4m3fn)
    topk_indices = MagicMock(dtype=paddle.int32)
    topk_scores = MagicMock()
    w1 = MagicMock(shape=(2, 512, 256))
    w2 = MagicMock()
    fp8_scale = MagicMock()
    fp8_config = SimpleNamespace(
        epilogue_quant=True,
        save_z_fp8=True,
        fused_gated=True,
        fuse_y1_quant=True,
    )
    metadata_result = _mock_sonic_metadata(has_bridge, packed_scale)
    up_projection = MagicMock()
    up_projection.apply.return_value = (MagicMock(), MagicMock())
    down_projection = MagicMock()
    down_projection.apply.return_value = MagicMock()

    patches = (
        patch.object(
            fusion_layer_utils,
            "_HAS_SONIC_METADATA_LAUNCH_BRIDGE",
            has_bridge,
        ),
        patch.object(
            fusion_layer_utils,
            "deepep_topk_to_sonic_metadata_with_scales",
            return_value=metadata_result,
        ),
        patch.object(fusion_layer_utils, "_UpProjection", up_projection),
        patch.object(fusion_layer_utils, "_DownProjection", down_projection),
        patch.object(
            fusion_layer_utils,
            "ActivationType",
            side_effect=lambda name: name,
        ),
        patch.object(
            fusion_layer_utils,
            "enable_fp8",
            side_effect=lambda enabled: contextlib.nullcontext(),
        ),
        patch.object(
            fusion_layer_utils.paddle.device,
            "current_stream",
            return_value=7,
        ),
        patch.object(
            fusion_layer_utils,
            "attach_preallocated_gated_outputs",
        ),
    )
    with contextlib.ExitStack() as stack:
        entered = [stack.enter_context(item) for item in patches]
        if not has_bridge:
            carrier = stack.enter_context(
                patch.object(
                    fusion_layer_utils._SonicRouterScoresFromMetadata,
                    "apply",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch.object(
                    fusion_layer_utils,
                    "_scatter_router_scores_i32",
                    MagicMock(),
                )
            )
        else:
            carrier = None
        fusion_layer_utils.run_sonic_moe(
            hidden_states,
            topk_indices,
            topk_scores,
            2,
            2,
            w1,
            w2,
            fp8=True,
            tokens_per_expert=[2, 2],
            fp8_scale=fp8_scale,
            fp8_config=fp8_config,
        )
    return metadata_result, down_projection, entered[-1], carrier


def test_run_sonic_moe_uses_preallocated_outputs_and_fused_router_edge():
    metadata, down_projection, attach_outputs, carrier = _run_sonic_with_mocks(
        True
    )

    attach_outputs.assert_called_once_with(metadata[10], metadata[11:])
    down_args = down_projection.apply.call_args.args
    assert down_args[-2] is not None
    assert down_args[-1] is metadata[9]
    assert carrier is None


def test_run_sonic_moe_preserves_legacy_router_carrier_fallback():
    _, down_projection, attach_outputs, carrier = _run_sonic_with_mocks(False)

    attach_outputs.assert_not_called()
    carrier.assert_called_once()
    assert down_projection.apply.call_args.kwargs.keys() == {"fp8_config"}


def test_run_sonic_moe_preserves_unpacked_scale_fallback():
    metadata, _, attach_outputs, _ = _run_sonic_with_mocks(
        False, packed_scale=False
    )

    assert metadata[10] is None
    attach_outputs.assert_not_called()


def test_resolve_sonic_config_bool_supports_values_and_resolvers():
    assert not fusion_layer_utils._resolve_sonic_config_bool(None, "enabled")
    assert fusion_layer_utils._resolve_sonic_config_bool(
        SimpleNamespace(enabled=True), "enabled"
    )
    assert fusion_layer_utils._resolve_sonic_config_bool(
        SimpleNamespace(resolve_enabled=lambda: True), "enabled"
    )
