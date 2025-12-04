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

import argparse
import re

import numpy as np


def parse_ground_truth(file_path):
    """
    Parses the ground truth file.
    Reads lines in format: step loss
    """
    gt_loss_list = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                loss = float(parts[1])
                gt_loss_list.append(loss)
    return gt_loss_list


def parse_log_file(file_path):
    """
    Parses the log file to extract global_step and loss.
    """
    # Regex patterns to extract loss and global_step
    # Matches lines like: ... loss: 10.58292007 ... global_step: 1 ...
    # Using independent searches allows for flexible ordering in the log line
    loss_pattern = re.compile(r"loss:\s*([0-9\.]+)")
    step_pattern = re.compile(r"global_step:\s*(\d+)")

    loss_list = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Filter lines that contain both keywords to avoid false positives
            if "loss:" in line and "global_step:" in line:
                loss_match = loss_pattern.search(line)
                step_match = step_pattern.search(line)

                if loss_match and step_match:
                    loss_val = float(loss_match.group(1))
                    loss_list.append(loss_val)

    return loss_list


def main():
    parser = argparse.ArgumentParser(
        description="Check loss values in log against ground truth."
    )
    parser.add_argument("--log_file", type=str, help="Path to the log file.")
    parser.add_argument(
        "--gt_file", type=str, help="Path to the ground truth file."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Tolerance for loss comparison.",
    )
    args = parser.parse_args()

    print(f"Starting loss check with log file: {args.log_file}")
    print(f"Ground truth file: {args.gt_file}, Tolerance: {args.tolerance}")

    log_file = args.log_file
    gt_file = args.gt_file
    tolerance = args.tolerance

    loss_list = parse_log_file(log_file)
    gt_loss_list = parse_ground_truth(gt_file)

    print(f"\nExtracted {len(loss_list)} loss values from log:")
    print(
        "\n".join([f"{i + 1} {loss:.8f}" for i, loss in enumerate(loss_list)])
    )
    print(f"\nExtracted {len(gt_loss_list)} loss values from ground truth:")
    print(
        "\n".join(
            [f"{i + 1} {loss:.8f}" for i, loss in enumerate(gt_loss_list)]
        )
    )

    np.testing.assert_allclose(
        loss_list, gt_loss_list, rtol=tolerance, atol=tolerance
    )

    print("\033[92m\nAll loss checks passed!\033[92m")


if __name__ == "__main__":
    main()
