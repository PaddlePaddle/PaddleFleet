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

"""Behavior tests for paddlefleet.utils.type_validators.

Expected values below are derived by hand from the validator source, never by
calling the function under test. Each validator is checked for accept-valid and
reject-invalid (with the offending value surfaced in the raised error where the
production code includes it), and real numeric/type boundaries are exercised.
"""

import unittest

from paddlefleet.utils.type_validators import (
    device_validator,
    image_size_validator,
    padding_validator,
    positive_any_number,
    positive_int,
    resampling_validator,
    tensor_type_validator,
    truncation_validator,
    video_metadata_validator,
)


class TestPositiveAnyNumber(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(positive_any_number(None))

    def test_valid_numbers_accepted(self):
        for value in (5, 3.14, 1_000_000, 0.0):
            self.assertIsNone(positive_any_number(value))

    def test_zero_boundary_accepted(self):
        # 0 >= 0 is True, so exactly zero must pass.
        self.assertIsNone(positive_any_number(0))

    def test_negative_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            positive_any_number(-1)
        self.assertIn("-1", str(cm.exception))

    def test_non_numeric_string_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            positive_any_number("abc")
        self.assertIn("abc", str(cm.exception))

    def test_bool_is_accepted_as_int(self):
        # bool is a subclass of int; True==1>=0 and False==0>=0 both pass the
        # `isinstance(value, (int, float))` and `value >= 0` guards.
        self.assertIsNone(positive_any_number(True))
        self.assertIsNone(positive_any_number(False))


class TestPositiveInt(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(positive_int(None))

    def test_valid_int_accepted(self):
        self.assertIsNone(positive_int(5))

    def test_zero_boundary_accepted(self):
        self.assertIsNone(positive_int(0))

    def test_float_rejected_with_value(self):
        # A float fails `isinstance(value, int)` even though it is non-negative.
        with self.assertRaises(ValueError) as cm:
            positive_int(3.14)
        self.assertIn("3.14", str(cm.exception))

    def test_negative_int_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            positive_int(-7)
        self.assertIn("-7", str(cm.exception))

    def test_bool_is_accepted_as_int(self):
        # bool subclasses int, so True/False satisfy the integer guard.
        self.assertIsNone(positive_int(True))
        self.assertIsNone(positive_int(False))


class TestPaddingValidator(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(padding_validator(None))

    def test_valid_strings_accepted(self):
        for value in ("longest", "max_length", "do_not_pad"):
            self.assertIsNone(padding_validator(value))

    def test_bool_accepted_and_not_treated_as_string(self):
        # bool is checked before the string-membership branch, so True and
        # False are accepted and never compared against possible_names.
        self.assertIsNone(padding_validator(True))
        self.assertIsNone(padding_validator(False))

    def test_invalid_string_rejected(self):
        with self.assertRaises(ValueError) as cm:
            padding_validator("invalid")
        self.assertIn("one of", str(cm.exception))

    def test_non_bool_non_str_type_rejected(self):
        # An int (that is not a bool) fails the type guard entirely.
        with self.assertRaises(ValueError) as cm:
            padding_validator(42)
        self.assertIn("padding", str(cm.exception))


class TestTruncationValidator(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(truncation_validator(None))

    def test_valid_strings_accepted(self):
        for value in (
            "only_first",
            "only_second",
            "longest_first",
            "do_not_truncate",
        ):
            self.assertIsNone(truncation_validator(value))

    def test_bool_accepted_and_not_treated_as_string(self):
        self.assertIsNone(truncation_validator(True))
        self.assertIsNone(truncation_validator(False))

    def test_invalid_string_rejected(self):
        with self.assertRaises(ValueError) as cm:
            truncation_validator("invalid")
        self.assertIn("one of", str(cm.exception))

    def test_non_bool_non_str_type_rejected(self):
        with self.assertRaises(ValueError) as cm:
            truncation_validator(42)
        self.assertIn("truncation", str(cm.exception))


class TestImageSizeValidator(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(image_size_validator(None))

    def test_int_is_accepted(self):
        # Only dict inputs are inspected; a bare int falls through unchecked.
        self.assertIsNone(image_size_validator(224))

    def test_dict_with_all_valid_keys_accepted(self):
        value = {
            "height": 1,
            "width": 2,
            "longest_edge": 3,
            "shortest_edge": 4,
            "max_height": 5,
            "max_width": 6,
        }
        self.assertIsNone(image_size_validator(value))

    def test_empty_dict_accepted(self):
        # `any(...)` over an empty key set is False, so {} is accepted.
        self.assertIsNone(image_size_validator({}))

    def test_dict_with_invalid_key_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            image_size_validator({"invalid_key": 100})
        self.assertIn("invalid_key", str(cm.exception))

    def test_dict_with_one_bad_key_among_valid_rejected(self):
        # `any(k not in possible_keys ...)` rejects if a single key is invalid.
        with self.assertRaises(ValueError) as cm:
            image_size_validator({"height": 224, "depth": 3})
        self.assertIn("depth", str(cm.exception))


class TestDeviceValidator(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(device_validator(None))

    def test_valid_device_strings_accepted(self):
        # possible_names == ["cpu", "gpu"]; the prefix before ':' must match.
        for value in ("cpu", "gpu", "gpu:0", "gpu:7"):
            self.assertIsNone(device_validator(value))

    def test_non_negative_int_accepted(self):
        # 0 is the boundary: `value < 0` is False, so 0 and positive ids pass.
        self.assertIsNone(device_validator(0))
        self.assertIsNone(device_validator(3))

    def test_negative_int_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            device_validator(-1)
        self.assertIn("-1", str(cm.exception))

    def test_invalid_device_string_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            device_validator("tpu")
        self.assertIn("tpu", str(cm.exception))

    def test_float_rejected_with_value(self):
        # A float is neither int nor str, so it hits the final type guard.
        with self.assertRaises(ValueError) as cm:
            device_validator(3.14)
        self.assertIn("3.14", str(cm.exception))

    @unittest.expectedFailure
    def test_cuda_prefix_should_be_accepted(self):
        # BUG (type_validators.py:95-110): the final error message advertises
        # "'cuda:0'" as a valid device string example (line 109), but
        # possible_names only contains ["cpu", "gpu"] (line 96). A "cuda:0"
        # string therefore has prefix "cuda" not in possible_names and is
        # rejected at lines 103-106. Correct behavior: "cuda:0" should be
        # accepted. Asserting the correct behavior; marked expectedFailure
        # until the validator (not this test) is fixed.
        device_validator("cuda:0")


class TestResamplingValidator(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(resampling_validator(None))

    def test_valid_ints_accepted(self):
        # Valid range is list(range(6)) == [0, 1, 2, 3, 4, 5].
        for value in (0, 1, 3, 5):
            self.assertIsNone(resampling_validator(value))

    def test_upper_boundary_five_accepted_six_rejected(self):
        # 5 is the inclusive upper bound; 6 is the first rejected value.
        self.assertIsNone(resampling_validator(5))
        with self.assertRaises(ValueError) as cm:
            resampling_validator(6)
        self.assertIn("6", str(cm.exception))

    def test_negative_int_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            resampling_validator(-1)
        self.assertIn("-1", str(cm.exception))

    def test_non_int_type_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            resampling_validator("bilinear")
        self.assertIn("bilinear", str(cm.exception))


class TestTensorTypeValidator(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(tensor_type_validator(None))

    def test_valid_values_accepted(self):
        for value in ("pd", "np"):
            self.assertIsNone(tensor_type_validator(value))

    def test_invalid_string_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            tensor_type_validator("tf")
        self.assertIn("tf", str(cm.exception))

    def test_non_string_rejected_with_value(self):
        with self.assertRaises(ValueError) as cm:
            tensor_type_validator(5)
        self.assertIn("5", str(cm.exception))


class TestVideoMetadataValidator(unittest.TestCase):
    def test_none_is_accepted(self):
        self.assertIsNone(video_metadata_validator(None))

    def test_valid_dict_accepted(self):
        self.assertIsNone(
            video_metadata_validator({"fps": 30, "total_num_frames": 100})
        )

    def test_dict_with_invalid_key_rejected_with_key(self):
        with self.assertRaises(ValueError) as cm:
            video_metadata_validator({"bogus": 1})
        self.assertIn("bogus", str(cm.exception))

    def test_list_of_dict_valid_accepted(self):
        self.assertIsNone(video_metadata_validator([{"fps": 24}, {"width": 8}]))

    def test_list_of_dict_invalid_rejected_with_key(self):
        with self.assertRaises(ValueError) as cm:
            video_metadata_validator([{"fps": 24}, {"bogus": 2}])
        self.assertIn("bogus", str(cm.exception))

    def test_list_of_list_of_dict_invalid_rejected_with_key(self):
        with self.assertRaises(ValueError) as cm:
            video_metadata_validator([[{"height": 4}], [{"bogus": 3}]])
        self.assertIn("bogus", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
