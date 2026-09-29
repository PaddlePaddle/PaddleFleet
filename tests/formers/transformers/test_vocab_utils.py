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

"""Behavior tests for paddlefleet.transformers.vocab_utils.

Config / model-layer utility: pads a vocab size up to the next multiple of
``make_vocab_size_divisible_by * tensor_model_parallel_size`` so the embedding
table shards evenly across tensor-parallel ranks.

Every numeric expectation is derived independently from the padding definition
-- either a fully hand-written literal or an integer-only ceiling formula that
is structurally distinct from the production code's float ``math.ceil`` -- and
never by calling the function under test to build the reference.

CPU-only ("wu-ka" / no accelerator): the tested logic is pure Python integer
arithmetic. The module still does ``import paddle`` at import time and
``print_rank_0`` calls ``paddle.distributed.get_rank``, so running these tests
requires paddle installed even though none of the tested math touches a device.
"""

import unittest
from unittest.mock import patch

from paddlefleet.transformers.vocab_utils import (
    _calculate_padded_vocab_size_cached,
    calculate_padded_vocab_size,
    print_rank_0,
)

MODULE = "paddlefleet.transformers.vocab_utils"


def _independent_padded_size(vocab_size, divisible_by, tp_size):
    """Independent reference: round ``vocab_size`` up to the next multiple of
    ``divisible_by * tp_size`` using integer-only ceiling division.

    Deliberately avoids the float ``math.ceil(a / b)`` the production code uses,
    so a shared floating-point rounding mistake cannot pass on both sides.
    """
    multiple = divisible_by * tp_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


# Distinguishable, non-degenerate cases: already-divisible, needs-padding,
# one-below-boundary, one-above-boundary, tp > 1, and a large realistic vocab.
_CASES = [
    (128, 128, 1),
    (100, 128, 1),
    (129, 128, 1),
    (256, 128, 1),
    (100, 128, 2),
    (256, 128, 2),
    (257, 128, 2),
    (50000, 128, 8),
    (32000, 64, 4),
    (1, 8, 1),
]


class TestPaddedVocabSizeNumeric(unittest.TestCase):
    """Numeric behaviour of the padding computation itself."""

    def setUp(self):
        # lru_cache is process-global and shared with the wrapper; start clean.
        _calculate_padded_vocab_size_cached.cache_clear()

    def test_matches_independent_reference_and_invariants(self):
        for vocab, div, tp in _CASES:
            got = _calculate_padded_vocab_size_cached(vocab, div, tp)
            expected = _independent_padded_size(vocab, div, tp)
            multiple = div * tp
            self.assertEqual(
                got, expected, msg=f"vocab={vocab} div={div} tp={tp}"
            )
            # Padding contract: result is a multiple of div*tp, not smaller than
            # the input, and the padding added is strictly minimal (< multiple).
            self.assertEqual(got % multiple, 0)
            self.assertGreaterEqual(got, vocab)
            self.assertLess(got - vocab, multiple)

    def test_exact_hand_computed_literals(self):
        # Fully hand-derived, no formula: guards against a reference that merely
        # mirrors production. 50000 -> multiple 1024 -> 49*1024 = 50176.
        self.assertEqual(_calculate_padded_vocab_size_cached(128, 128, 1), 128)
        self.assertEqual(_calculate_padded_vocab_size_cached(100, 128, 1), 128)
        self.assertEqual(_calculate_padded_vocab_size_cached(129, 128, 1), 256)
        self.assertEqual(_calculate_padded_vocab_size_cached(100, 128, 2), 256)
        self.assertEqual(_calculate_padded_vocab_size_cached(256, 128, 2), 256)
        self.assertEqual(_calculate_padded_vocab_size_cached(257, 128, 2), 512)
        self.assertEqual(
            _calculate_padded_vocab_size_cached(50000, 128, 8), 50176
        )

    def test_tp_size_scales_the_divisor(self):
        # Same vocab, doubling tp doubles the alignment multiple, so a value
        # already aligned at tp=1 must pad further at tp=2. Catches ignoring tp.
        self.assertEqual(_calculate_padded_vocab_size_cached(128, 128, 1), 128)
        self.assertEqual(_calculate_padded_vocab_size_cached(128, 128, 2), 256)
        self.assertEqual(_calculate_padded_vocab_size_cached(128, 128, 3), 384)

    def test_wrapper_equals_core_and_reference(self):
        for vocab, div, tp in _CASES:
            got = calculate_padded_vocab_size(
                vocab, div, tp, logging_enabled=False
            )
            self.assertEqual(got, _independent_padded_size(vocab, div, tp))


class TestValidation(unittest.TestCase):
    """Positive-argument contract, checked via both entry points."""

    def setUp(self):
        _calculate_padded_vocab_size_cached.cache_clear()

    def test_non_positive_vocab_size_raises(self):
        for bad in (0, -1, -128):
            with self.assertRaisesRegex(
                ValueError, "vocab_size must be positive"
            ):
                _calculate_padded_vocab_size_cached(bad, 128, 1)
            with self.assertRaisesRegex(
                ValueError, "vocab_size must be positive"
            ):
                calculate_padded_vocab_size(bad, 128, 1, logging_enabled=False)

    def test_non_positive_divisible_by_raises(self):
        for bad in (0, -1, -64):
            with self.assertRaisesRegex(
                ValueError, "make_vocab_size_divisible_by must be positive"
            ):
                _calculate_padded_vocab_size_cached(100, bad, 1)

    def test_non_positive_tp_size_raises(self):
        for bad in (0, -1, -8):
            with self.assertRaisesRegex(
                ValueError, "tensor_model_parallel_size must be positive"
            ):
                _calculate_padded_vocab_size_cached(100, 128, bad)

    def test_error_message_reports_offending_value(self):
        with self.assertRaisesRegex(ValueError, r"got -7"):
            _calculate_padded_vocab_size_cached(-7, 128, 1)


class TestLruCache(unittest.TestCase):
    """The cached core must actually memoize on its argument tuple."""

    def setUp(self):
        _calculate_padded_vocab_size_cached.cache_clear()

    def test_repeated_call_hits_cache(self):
        first = _calculate_padded_vocab_size_cached(100, 128, 1)
        second = _calculate_padded_vocab_size_cached(100, 128, 1)
        info = _calculate_padded_vocab_size_cached.cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 1)
        self.assertEqual(first, second)
        self.assertEqual(first, 128)

    def test_distinct_args_each_miss(self):
        _calculate_padded_vocab_size_cached(100, 128, 1)
        _calculate_padded_vocab_size_cached(100, 128, 2)
        _calculate_padded_vocab_size_cached(200, 128, 1)
        info = _calculate_padded_vocab_size_cached.cache_info()
        self.assertEqual(info.misses, 3)
        self.assertEqual(info.hits, 0)


class TestWrapperLoggingOrchestration(unittest.TestCase):
    """calculate_padded_vocab_size forwards args to the core and logs the
    padding summary only when logging is enabled."""

    def setUp(self):
        _calculate_padded_vocab_size_cached.cache_clear()

    def test_forwards_all_args_and_returns_core_result(self):
        captured = {}

        def fake_core(vocab_size, divisible_by, tp_size):
            captured.update(
                vocab_size=vocab_size,
                divisible_by=divisible_by,
                tp_size=tp_size,
            )
            return 4242  # sentinel unrelated to any real padding value

        with patch(
            f"{MODULE}._calculate_padded_vocab_size_cached",
            side_effect=fake_core,
        ):
            result = calculate_padded_vocab_size(
                100, 64, 3, logging_enabled=False
            )

        self.assertEqual(
            captured, {"vocab_size": 100, "divisible_by": 64, "tp_size": 3}
        )
        self.assertEqual(result, 4242)

    def test_logging_enabled_emits_exact_summary(self):
        with patch(f"{MODULE}.print_rank_0") as mock_print:
            result = calculate_padded_vocab_size(
                100, 128, 1, logging_enabled=True
            )
        self.assertEqual(result, 128)
        mock_print.assert_called_once_with(
            " > padded vocab (size: 100) with 28 dummy tokens (new size: 128)"
        )

    def test_logging_enabled_summary_scales_with_padding(self):
        # tp=2 -> multiple 256 -> padded 256 -> 156 dummy tokens. Independent
        # hand computation guards the (dummy = padded - vocab) arithmetic.
        with patch(f"{MODULE}.print_rank_0") as mock_print:
            calculate_padded_vocab_size(100, 128, 2, logging_enabled=True)
        mock_print.assert_called_once_with(
            " > padded vocab (size: 100) with 156 dummy tokens (new size: 256)"
        )

    def test_logging_disabled_is_silent(self):
        with patch(f"{MODULE}.print_rank_0") as mock_print:
            result = calculate_padded_vocab_size(
                100, 128, 1, logging_enabled=False
            )
        mock_print.assert_not_called()
        self.assertEqual(result, 128)

    def test_logging_defaults_to_enabled(self):
        with patch(f"{MODULE}.print_rank_0") as mock_print:
            calculate_padded_vocab_size(100, 128, 1)
        mock_print.assert_called_once()


class TestPrintRank0(unittest.TestCase):
    """print_rank_0 must emit only on rank 0 (real branch on get_rank)."""

    def test_prints_on_rank0(self):
        with (
            patch(f"{MODULE}.paddle.distributed.get_rank", return_value=0),
            patch("builtins.print") as mock_print,
        ):
            print_rank_0("hello", 42, sep="-")
        mock_print.assert_called_once_with("hello", 42, sep="-")

    def test_silent_on_non_rank0(self):
        for rank in (1, 3, 7):
            with (
                patch(
                    f"{MODULE}.paddle.distributed.get_rank", return_value=rank
                ),
                patch("builtins.print") as mock_print,
            ):
                print_rank_0("should not appear")
            mock_print.assert_not_called()


if __name__ == "__main__":
    unittest.main()
