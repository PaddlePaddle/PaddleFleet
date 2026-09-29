# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for
paddlefleet.quantization.unified_checkpoint_quantization.

These two functions orchestrate the optimizer-state checkpoint compression:

* ``quant_unified_optimizer`` walks an optimizer ``state_dict`` and, at stage
  O1, replaces each ``.../moment1_0`` tensor with its symmetric int8 quant and
  each ``.../moment2_0`` tensor with the asymmetric uint8 quant of the Adam
  update *ratio* ``1/(sqrt(m2)+eps)``, appending the per-key scale tensors
  (``@scales`` / ``@min_scales`` / ``@max_scales``) into the same dict.
* ``dequant_unified_optimizer`` consumes the quantized tensors plus a separate
  ``scale_dict`` and reconstructs moment1 (symmetric dequant) and moment2
  (asymmetric dequant of the ratio, then ``(1/ratio - eps)**2``).

Every expected value is hand-derived from the quantization protocol with plain
numpy on small, fully-known tensors -- the functions under test are never used
to build their own reference. The full quant -> dequant round-trip is checked
for *content* (not shape/existence) and, with two same-shape distinct-content
layers, for correct key -> tensor pairing so a swap cannot pass.

Environment: 无卡 (CPU). The save path in production stores numpy arrays via
``safetensors.numpy`` so ``quant_unified_optimizer`` runs on numpy arrays; the
numpy dequant path (``use_pd=False``) is likewise pure-CPU. Importing the module
still requires paddle (it imports ``paddle`` / ``fleet`` and calls
``paddle.distributed.get_world_size()``, which returns 1 without a launcher).
"""

import importlib
import os
import tempfile
import unittest

import numpy as np

from paddlefleet.quantization.unified_checkpoint_quantization import (
    dequant_unified_optimizer,
    quant_unified_optimizer,
)
from paddlefleet.utils.env import (
    ASYMMETRY_QUANT_SCALE_MAX,
    ASYMMETRY_QUANT_SCALE_MIN,
    MOMENT1_KEYNAME,
    MOMENT2_KEYNAME,
    SYMMETRY_QUANT_SCALE,
)

EPS = 1e-8

try:
    _st_numpy = importlib.import_module("safetensors.numpy")
    HAS_SAFETENSORS = True
except ModuleNotFoundError:
    _st_numpy = None
    HAS_SAFETENSORS = False


# --- Independent numpy references for the quantization protocol -------------
# Re-derived from the documented math, NOT from the functions under test.


def ref_sym_int8_quant(x):
    """Per-column (reduce over rows) abs-max symmetric int8 quant, bnt=127."""
    x = x.astype(np.float32)
    scales = np.max(np.abs(x), axis=0).astype(np.float32)
    scales = np.where(scales == 0, np.float32(EPS), scales)
    q = np.clip(np.round(x / scales * 127), -128, 127).astype(np.int8)
    return q, scales


def ref_sym_int8_dequant(q, scales):
    return (q / 127 * scales).astype(np.float32)


def ref_asym_uint8_quant(x):
    """Per-column asymmetric uint8 quant, bnt=255 (max/min are NOT abs)."""
    x = x.astype(np.float32)
    maxs = np.max(x, axis=0)
    mins = np.min(x, axis=0)
    maxs = np.where(maxs == 0, np.float32(EPS), maxs).astype(np.float32)
    mins = np.where(mins == 0, np.float32(EPS), mins).astype(np.float32)
    scales = maxs - mins
    q = np.clip(np.round((x - mins) / scales * 255), 0, 255).astype(np.uint8)
    return q, mins, maxs


def ref_asym_uint8_dequant(q, mins, maxs):
    scales = maxs - mins
    return ((q / 255 * scales) + mins).astype(np.float32)


def ref_ratio(m2):
    return (1.0 / (np.sqrt(m2.astype(np.float32)) + EPS)).astype(np.float32)


def ref_o1_roundtrip(m1, m2):
    """Independent expected reconstruction of (m1, m2) after an O1 round-trip."""
    m1_q, m1_scales = ref_sym_int8_quant(m1)
    ratio = ref_ratio(m2)
    ratio_q, r_mins, r_maxs = ref_asym_uint8_quant(ratio)

    m1_rec = ref_sym_int8_dequant(m1_q, m1_scales)
    ratio_rec = ref_asym_uint8_dequant(ratio_q, r_mins, r_maxs)
    m2_rec = np.square(1.0 / ratio_rec - EPS).astype(np.float32)
    return m1_rec, m2_rec


# Distinct, per-column-nonuniform fixtures (float32, matching the numpy save
# path). moment2 is strictly positive (it is a variance / second moment).
def _m1_layer_a():
    return np.array(
        [[100.0, -40.0, 30.0], [-50.0, 80.0, -60.0]], dtype=np.float32
    )


def _m2_layer_a():
    return np.array([[0.01, 0.04, 0.09], [0.16, 0.25, 0.36]], dtype=np.float32)


def _m1_layer_b():
    # Same shape as layer A but different content, to catch key/tensor swaps.
    return np.array([[-9.0, 12.0, -3.0], [6.0, -24.0, 15.0]], dtype=np.float32)


def _m2_layer_b():
    return np.array([[0.49, 0.64, 0.81], [1.00, 1.21, 1.44]], dtype=np.float32)


class TestStageGating(unittest.TestCase):
    """O0 and non-optimizer state dicts must pass through untouched."""

    def test_dequant_o0_returns_input_unchanged(self):
        m1 = _m1_layer_a()
        m2 = _m2_layer_a()
        state_dict = {"A/" + MOMENT1_KEYNAME: m1, "A/" + MOMENT2_KEYNAME: m2}
        result = dequant_unified_optimizer(state_dict, "O0", scale_dict={})
        # Same object, and byte-for-byte identical content (no accidental
        # quant/dequant on the O0 no-op path).
        self.assertIs(result, state_dict)
        np.testing.assert_array_equal(result["A/" + MOMENT1_KEYNAME], m1)
        np.testing.assert_array_equal(result["A/" + MOMENT2_KEYNAME], m2)

    def test_quant_o0_optimizer_weight_unchanged(self):
        m1 = _m1_layer_a()
        state_dict = {"A/" + MOMENT1_KEYNAME: m1.copy()}
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O0")
        self.assertEqual(list(result.keys()), ["A/" + MOMENT1_KEYNAME])
        self.assertEqual(result["A/" + MOMENT1_KEYNAME].dtype, np.float32)
        np.testing.assert_array_equal(result["A/" + MOMENT1_KEYNAME], m1)

    def test_quant_o1_non_optimizer_weight_unchanged(self):
        # Only "optimizer_weight" is quantized; model weights stay float32
        # even at O1. A regression that quantized model weights would change
        # dtype/content and fail here.
        m1 = _m1_layer_a()
        state_dict = {"A/" + MOMENT1_KEYNAME: m1.copy()}
        result = quant_unified_optimizer(state_dict, "model_weight", "O1")
        self.assertEqual(result["A/" + MOMENT1_KEYNAME].dtype, np.float32)
        np.testing.assert_array_equal(result["A/" + MOMENT1_KEYNAME], m1)


class TestO1QuantContract(unittest.TestCase):
    """O1 quant: key mapping, dtypes and scale tensors are all independently
    verified (not just 'a key exists')."""

    def test_quant_produces_expected_keys_dtypes_and_scales(self):
        m1 = _m1_layer_a()
        m2 = _m2_layer_a()
        m1_key = "A/" + MOMENT1_KEYNAME
        m2_key = "A/" + MOMENT2_KEYNAME
        state_dict = {m1_key: m1.copy(), m2_key: m2.copy()}

        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O1")

        # moment1 -> int8 symmetric quant; moment2 -> uint8 asymmetric quant
        # of the ratio. Values compared against the independent references.
        m1_q_ref, m1_scales_ref = ref_sym_int8_quant(m1)
        ratio_q_ref, r_mins_ref, r_maxs_ref = ref_asym_uint8_quant(
            ref_ratio(m2)
        )

        self.assertEqual(result[m1_key].dtype, np.int8)
        self.assertEqual(result[m2_key].dtype, np.uint8)
        np.testing.assert_array_equal(result[m1_key], m1_q_ref)
        np.testing.assert_array_equal(result[m2_key], ratio_q_ref)

        # Scale tensors are appended under the documented suffix keys and hold
        # the hand-derived per-column scales.
        np.testing.assert_allclose(
            result[m1_key + SYMMETRY_QUANT_SCALE], m1_scales_ref, atol=1e-6
        )
        np.testing.assert_allclose(
            result[m2_key + ASYMMETRY_QUANT_SCALE_MIN], r_mins_ref, atol=1e-6
        )
        np.testing.assert_allclose(
            result[m2_key + ASYMMETRY_QUANT_SCALE_MAX], r_maxs_ref, atol=1e-6
        )
        self.assertEqual(
            set(result),
            {
                m1_key,
                m2_key,
                m1_key + SYMMETRY_QUANT_SCALE,
                m2_key + ASYMMETRY_QUANT_SCALE_MIN,
                m2_key + ASYMMETRY_QUANT_SCALE_MAX,
            },
        )

    def test_quant_scale_anchor_values(self):
        # A tiny hand-computed anchor independent of any reference helper:
        # moment1 per-column abs-max is exactly [100, 80, 60].
        m1 = _m1_layer_a()
        m2 = _m2_layer_a()
        m1_key = "A/" + MOMENT1_KEYNAME
        m2_key = "A/" + MOMENT2_KEYNAME
        result = quant_unified_optimizer(
            {m1_key: m1.copy(), m2_key: m2.copy()},
            "optimizer_weight",
            "O1",
        )
        np.testing.assert_allclose(
            result[m1_key + SYMMETRY_QUANT_SCALE],
            [100.0, 80.0, 60.0],
            atol=1e-6,
        )


def _split_weights_and_scales(quant_dict):
    """Mirror the load path: scale tensors (keys carrying an '@' suffix) live
    in a separate scale_dict; the remaining tensors form the state_dict."""
    state_dict, scale_dict = {}, {}
    for key, value in quant_dict.items():
        if "@" in key:
            scale_dict[key] = value
        else:
            state_dict[key] = value
    return state_dict, scale_dict


class TestO1RoundTrip(unittest.TestCase):
    """Full quant -> dequant round-trip reconstructs moment1/moment2 to their
    independently derived post-quantization values, and never swaps tensors
    between two same-shape layers."""

    def test_roundtrip_reconstructs_hand_derived_values(self):
        m1 = _m1_layer_a()
        m2 = _m2_layer_a()
        m1_key = "A/" + MOMENT1_KEYNAME
        m2_key = "A/" + MOMENT2_KEYNAME

        quant_dict = quant_unified_optimizer(
            {m1_key: m1.copy(), m2_key: m2.copy()},
            "optimizer_weight",
            "O1",
        )
        state_dict, scale_dict = _split_weights_and_scales(quant_dict)

        # dequant mutates and returns state_dict in place.
        restored = dequant_unified_optimizer(
            state_dict, "O1", scale_dict, use_pd=False
        )

        m1_rec, m2_rec = ref_o1_roundtrip(m1, m2)
        np.testing.assert_allclose(
            restored[m1_key], m1_rec, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            restored[m2_key], m2_rec, rtol=1e-4, atol=1e-6
        )
        # Quantization is lossy but must stay in the neighbourhood of the
        # originals -- a mixed-up scale axis or dropped ratio step would not.
        np.testing.assert_allclose(restored[m1_key], m1, rtol=0.05, atol=1.0)
        np.testing.assert_allclose(restored[m2_key], m2, rtol=0.05, atol=1e-2)

    def test_roundtrip_does_not_swap_two_same_shape_layers(self):
        a1, a2 = _m1_layer_a(), _m2_layer_a()
        b1, b2 = _m1_layer_b(), _m2_layer_b()
        keys = {
            "A/" + MOMENT1_KEYNAME: a1.copy(),
            "A/" + MOMENT2_KEYNAME: a2.copy(),
            "B/" + MOMENT1_KEYNAME: b1.copy(),
            "B/" + MOMENT2_KEYNAME: b2.copy(),
        }
        quant_dict = quant_unified_optimizer(keys, "optimizer_weight", "O1")
        state_dict, scale_dict = _split_weights_and_scales(quant_dict)
        restored = dequant_unified_optimizer(
            state_dict, "O1", scale_dict, use_pd=False
        )

        a1_rec, a2_rec = ref_o1_roundtrip(a1, a2)
        b1_rec, b2_rec = ref_o1_roundtrip(b1, b2)
        np.testing.assert_allclose(
            restored["A/" + MOMENT1_KEYNAME], a1_rec, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            restored["A/" + MOMENT2_KEYNAME], a2_rec, rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            restored["B/" + MOMENT1_KEYNAME], b1_rec, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            restored["B/" + MOMENT2_KEYNAME], b2_rec, rtol=1e-4, atol=1e-6
        )
        # Layer A and layer B are distinct, so a swap would make these fail.
        self.assertFalse(
            np.allclose(a1_rec, b1_rec),
            "fixtures must differ for the swap check to have teeth",
        )


@unittest.skipUnless(
    HAS_SAFETENSORS, "safetensors.numpy not importable in this environment"
)
class TestO1RoundTripThroughSafetensorsFile(unittest.TestCase):
    """The real save path serialises the quantized dict to a .safetensors file
    (safetensors.numpy) and the load path reads it back and re-splits scales.
    This exercises a genuine file round-trip into NEW dict objects, checking
    reconstructed content -- not merely that a file exists."""

    def test_save_to_tempfile_then_load_and_dequant(self):
        m1, m2 = _m1_layer_a(), _m2_layer_a()
        m1_key = "A/" + MOMENT1_KEYNAME
        m2_key = "A/" + MOMENT2_KEYNAME
        quant_dict = quant_unified_optimizer(
            {m1_key: m1.copy(), m2_key: m2.copy()},
            "optimizer_weight",
            "O1",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "opt.safetensors")
            _st_numpy.save_file(quant_dict, path)
            # Fresh objects loaded from disk -- no reuse of quant_dict.
            loaded = _st_numpy.load_file(path)

        self.assertEqual(set(loaded), set(quant_dict))
        state_dict, scale_dict = _split_weights_and_scales(loaded)
        # int8 / uint8 dtypes must survive serialization.
        self.assertEqual(state_dict[m1_key].dtype, np.int8)
        self.assertEqual(state_dict[m2_key].dtype, np.uint8)

        restored = dequant_unified_optimizer(
            state_dict, "O1", scale_dict, use_pd=False
        )
        m1_rec, m2_rec = ref_o1_roundtrip(m1, m2)
        np.testing.assert_allclose(
            restored[m1_key], m1_rec, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            restored[m2_key], m2_rec, rtol=1e-4, atol=1e-6
        )


class TestO2DequantMutationBug(unittest.TestCase):
    """O2 dequant is expected (correctly) to reconstruct moment1/moment2 from
    the merged int4 payload. It currently cannot: ``dequant_unified_optimizer``
    calls ``state_dict.update(m1_state_dict)`` INSIDE ``for quant_key in
    state_dict.keys():`` (unified_checkpoint_quantization.py:161), which adds
    the moment1 key mid-iteration and raises
    ``RuntimeError: dictionary changed size during iteration``. The update
    belongs after the loop. This asserts the correct contract and is marked
    expectedFailure until the production bug is fixed."""

    @unittest.expectedFailure
    def test_o2_roundtrip_should_reconstruct_but_raises(self):
        rows, cols = 32, 2  # group_size=32 requires rows % 32 == 0
        base = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
        m1 = (base - 30.0).astype(np.float32)
        m2 = (base * 0.01 + 0.05).astype(np.float32)  # strictly positive
        m1_key = "L/" + MOMENT1_KEYNAME
        m2_key = "L/" + MOMENT2_KEYNAME

        quant_dict = quant_unified_optimizer(
            {m1_key: m1.copy(), m2_key: m2.copy()},
            "optimizer_weight",
            "O2",
        )
        state_dict, scale_dict = _split_weights_and_scales(quant_dict)

        # Correct behaviour: dequant reconstructs both moments (the moment1 key
        # is re-materialised from the packed payload). Production raises here.
        restored = dequant_unified_optimizer(
            state_dict, "O2", scale_dict, use_pd=False
        )
        self.assertIn(m1_key, restored)
        self.assertIn(m2_key, restored)
        self.assertEqual(list(restored[m2_key].shape), [rows, cols])
        # int4 group quant is lossy; a correct round-trip still stays close.
        np.testing.assert_allclose(restored[m2_key], m2, rtol=0.3, atol=0.2)


if __name__ == "__main__":
    unittest.main()
