#!/usr/bin/env python3

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

"""
Remove diff hunks belonging to a specified path prefix from a unified diff file.

Usage:
    python ci/exclude_diff_file.py diff.txt "src/paddlefleet/triton_ops/"

This rewrites diff.txt in-place, stripping all diff blocks whose file path
starts with the given prefix. Useful for excluding directories from
diff-cover coverage checks.
"""

import sys


def exclude_path_from_diff(diff_file: str, exclude_path: str) -> None:
    """Remove all diff hunks for files matching *exclude_path* prefix."""
    with open(diff_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    output_lines: list[str] = []
    skip = False

    for line in lines:
        # Each file in unified diff starts with "diff --git a/... b/..."
        if line.startswith("diff --git "):
            # Extract b-side path: "diff --git a/foo b/foo"
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                file_path = parts[1].strip()
                skip = file_path.startswith(exclude_path)
            else:
                skip = False

        if not skip:
            output_lines.append(line)

    with open(diff_file, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    excluded_count = len(lines) - len(output_lines)
    print(
        f"[exclude_diff_file] Removed {excluded_count} lines "
        f"matching '{exclude_path}' from {diff_file}"
    )


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python ci/exclude_diff_file.py <diff_file> <exclude_path>"
        )
        sys.exit(1)

    diff_file = sys.argv[1]
    exclude_path = sys.argv[2]
    exclude_path_from_diff(diff_file, exclude_path)


if __name__ == "__main__":
    main()
