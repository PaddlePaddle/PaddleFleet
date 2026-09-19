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

import importlib
import importlib.util

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
import unittest

import numpy as np
from parameterized import parameterized


def build_blending_indices_python(
    dataset_index, dataset_sample_index, weights, num_datasets, size, verbose
):
    """
    Given multiple datasets and a weighting array, build samples such that it follows those weights.

    Parameters:
    - dataset_index: NumPy array to store the dataset index for each sample.
    - dataset_sample_index: NumPy array to store the sample index within each dataset.
    - weights: NumPy array of weights for each dataset.
    - num_datasets: Integer, the number of datasets.
    - size: Integer, the total number of samples to generate.
    - verbose: Boolean, whether to print verbose output.
    """
    if verbose:
        print("> building indices for blendable datasets ...")

    # Initialize buffer for number of samples used for each dataset.
    current_samples = np.zeros(num_datasets, dtype=np.int64)

    # For each sample:
    for sample_idx in range(size):
        # Determine where the max error in sampling is happening.
        sample_idx_double = max(sample_idx, 1)
        max_error_index = 0
        max_error = weights[0] * sample_idx_double - current_samples[0]
        for dataset_idx in range(1, num_datasets):
            error = (
                weights[dataset_idx] * sample_idx_double
                - current_samples[dataset_idx]
            )
            if error > max_error:
                max_error = error
                max_error_index = dataset_idx

        # Populate the indices.
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index]

        # Update the total samples.
        current_samples[max_error_index] += 1

    # Print info
    if verbose:
        print(" > sample ratios:")
        for dataset_idx in range(num_datasets):
            ratio = current_samples[dataset_idx] / size
            print(
                f"   dataset {dataset_idx}, input: {weights[dataset_idx]}, achieved: {ratio}"
            )


def skip_if_version_not_equal(version="0.1.1", package_name="fast_dataindex"):
    try:
        importlib.import_module(package_name)
    except ImportError:
        return True, f"package<{package_name}> not found, so to skip this test"
    package_version = metadata.version(package_name)
    if package_version != version:
        return (
            True,
            f"{package_name} version must be equal to {version}, but got {package_version}!",
        )
    return False, f"{package_name} version is ok!"


class TestToolHelpers(unittest.TestCase):
    def _test_build_blending_indices(
        self,
        num_datasets=128,
        size=8192,
        dataset_index_dtype="uint8",
        verbose=False,
        seed=42,
        assert_true=True,
    ):
        if isinstance(dataset_index_dtype, str):
            dataset_index_dtype = np.dtype(dataset_index_dtype)
        assert dataset_index_dtype in [np.uint8, np.int16], (
            "dataset_index_dtype must be uint8 or int16!"
        )

        np.random.seed(seed)
        random_numbers = np.random.rand(num_datasets)
        random_numbers[0] = 200
        weights = random_numbers / random_numbers.sum()
        weights = weights.astype(np.float64)

        # for ground truth, so we use np.int32
        python_dataset_index = np.zeros(size, dtype=np.int32)
        python_dataset_sample_index = np.zeros(size, dtype=np.int64)
        build_blending_indices_python(
            python_dataset_index,
            python_dataset_sample_index,
            weights,
            num_datasets,
            size,
            verbose,
        )

        from fast_dataindex import helpers

        c_dataset_index = np.zeros(size, dtype=dataset_index_dtype)
        c_dataset_sample_index = np.zeros(size, dtype=np.int64)
        helpers.build_blending_indices(
            c_dataset_index,
            c_dataset_sample_index,
            weights,
            num_datasets,
            size,
            verbose,
        )

        assert_func = self.assertTrue if assert_true else self.assertFalse
        assert_func(
            np.all(
                python_dataset_index
                == c_dataset_index.astype(python_dataset_index.dtype)
            )
        )
        self.assertTrue(
            np.all(
                python_dataset_sample_index
                == c_dataset_sample_index.astype(
                    python_dataset_sample_index.dtype
                )
            )
        )

    @parameterized.expand(
        [
            (128, 8192, "uint8", False, 42, True),
            (1024, 8192, "uint8", False, 42, False),
            (128, 8192, "int16", False, 42, False),
            (1024, 8192, "int16", False, 42, False),
        ]
    )
    @unittest.skipIf(
        *skip_if_version_not_equal(
            version="0.1.1", package_name="fast_dataindex"
        )
    )
    def test_build_blending_indices_version_0_1_1(
        self,
        num_datasets=128,
        size=8192,
        dataset_index_dtype="uint8",
        verbose=False,
        seed=42,
        assert_true=True,
    ):
        self._test_build_blending_indices(
            num_datasets, size, dataset_index_dtype, verbose, seed, assert_true
        )

    @parameterized.expand(
        [
            (128, 8192, "uint8", False, 42, True),
            (1024, 8192, "uint8", False, 42, False),
            (128, 8192, "int16", False, 42, True),
            (1024, 8192, "int16", False, 42, True),
        ]
    )
    @unittest.skipIf(
        *skip_if_version_not_equal(
            version="0.1.2", package_name="fast_dataindex"
        )
    )
    def test_build_blending_indices_version_0_1_2(
        self,
        num_datasets=128,
        size=8192,
        dataset_index_dtype="uint8",
        verbose=False,
        seed=42,
        assert_true=True,
    ):
        self._test_build_blending_indices(
            num_datasets, size, dataset_index_dtype, verbose, seed, assert_true
        )


class _ListDataset:
    """Minimal real constituent dataset with content-distinguishable samples.

    Each sample carries a text tag and a numeric value derived from the
    dataset tag and the in-dataset sample index, so that a routing error
    (wrong constituent dataset or wrong in-dataset sample index) yields a
    distinguishable, independently predictable output.
    """

    def __init__(self, tag, num_samples):
        self.tag = tag
        self.desc = f"list-dataset-{tag}"
        self._samples = [
            {"text": f"{tag}-sample-{i}", "value": tag * 1000 + i}
            for i in range(num_samples)
        ]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return dict(self._samples[int(idx)])


_PADDLE_INSTALLED = importlib.util.find_spec("paddle") is not None


@unittest.skipUnless(
    _PADDLE_INSTALLED,
    "paddle is required to import "
    "paddlefleet.data.blendable_dataset.BlendableDataset",
)
class TestBlendableDatasetIndexing(unittest.TestCase):
    """Behavior tests for how BlendableDataset consumes indices in __getitem__.

    These exercise the real __getitem__ routing on CPU. __init__ is bypassed
    via __new__ because the production constructor needs the fast_dataindex
    extension plus file I/O to build the index arrays; here we install
    deterministic index arrays and observe how the real routing consumes them.
    Weight -> per-dataset sample proportions (index construction) is already
    covered by TestToolHelpers against the production fast_dataindex helper,
    so it is not duplicated here.
    """

    def _make_blendable(self, dataset_index, dataset_sample_index, datasets):
        from paddlefleet.data.blendable_dataset import BlendableDataset

        blend = BlendableDataset.__new__(BlendableDataset)
        blend.datasets = datasets
        blend.size = len(dataset_index)
        blend.dataset_index = np.asarray(dataset_index, dtype=np.int16)
        blend.dataset_sample_index = np.asarray(
            dataset_sample_index, dtype=np.int64
        )
        blend.desc = "test-blendable"
        return blend

    def test_getitem_routes_to_correct_dataset_and_sample(self):
        datasets = [
            _ListDataset(tag=0, num_samples=4),
            _ListDataset(tag=1, num_samples=4),
            _ListDataset(tag=2, num_samples=4),
        ]
        # Interleave constituent datasets and pick distinct in-dataset samples
        # so a swapped dataset or a mis-mapped sample index is observable.
        dataset_index = [0, 1, 2, 1, 0, 2]
        dataset_sample_index = [3, 0, 2, 1, 1, 3]
        blend = self._make_blendable(
            dataset_index, dataset_sample_index, datasets
        )

        # Independently computed expected output for each global index.
        expected = [
            {"dataset_idx": 0, "text": "0-sample-3", "value": 3},
            {"dataset_idx": 1, "text": "1-sample-0", "value": 1000},
            {"dataset_idx": 2, "text": "2-sample-2", "value": 2002},
            {"dataset_idx": 1, "text": "1-sample-1", "value": 1001},
            {"dataset_idx": 0, "text": "0-sample-1", "value": 1},
            {"dataset_idx": 2, "text": "2-sample-3", "value": 2003},
        ]
        self.assertEqual(len(expected), blend.size)
        for i, exp in enumerate(expected):
            item = blend[i]
            self.assertEqual(int(item["dataset_idx"]), exp["dataset_idx"])
            self.assertEqual(item["text"], exp["text"])
            self.assertEqual(item["value"], exp["value"])
            # The merged sample keys must accompany the injected dataset_idx.
            self.assertEqual(set(item), {"dataset_idx", "text", "value"})

    def test_getitem_preserves_sample_identity_per_dataset(self):
        # Two datasets with overlapping in-dataset indices but disjoint
        # content ranges; every global slot maps to exactly one (ds, sample).
        datasets = [
            _ListDataset(tag=0, num_samples=3),
            _ListDataset(tag=1, num_samples=3),
        ]
        dataset_index = [0, 0, 0, 1, 1, 1]
        dataset_sample_index = [0, 1, 2, 0, 1, 2]
        blend = self._make_blendable(
            dataset_index, dataset_sample_index, datasets
        )
        seen = [
            (blend[i]["text"], blend[i]["value"]) for i in range(blend.size)
        ]
        self.assertEqual(
            seen,
            [
                ("0-sample-0", 0),
                ("0-sample-1", 1),
                ("0-sample-2", 2),
                ("1-sample-0", 1000),
                ("1-sample-1", 1001),
                ("1-sample-2", 1002),
            ],
        )

    def test_len_returns_blend_size_not_constituent_totals(self):
        # Blend size (5) differs from any constituent length (4) and from
        # their total (8), so __len__ must return the requested blend size.
        datasets = [
            _ListDataset(tag=0, num_samples=4),
            _ListDataset(tag=1, num_samples=4),
        ]
        dataset_index = [0, 1, 0, 1, 0]
        dataset_sample_index = [0, 0, 1, 1, 2]
        blend = self._make_blendable(
            dataset_index, dataset_sample_index, datasets
        )
        self.assertEqual(len(blend), 5)

    def test_getitem_out_of_range_raises_index_error(self):
        # __init__ relies on __getitem__(size) raising IndexError as its
        # bound check; the index arrays hold exactly `size` entries.
        datasets = [
            _ListDataset(tag=0, num_samples=2),
            _ListDataset(tag=1, num_samples=2),
        ]
        blend = self._make_blendable([0, 1, 0], [0, 0, 1], datasets)
        # In-range access at the last valid slot succeeds.
        self.assertEqual(blend[blend.size - 1]["text"], "0-sample-1")
        with self.assertRaises(IndexError):
            _ = blend[blend.size]


if __name__ == "__main__":
    unittest.main()
