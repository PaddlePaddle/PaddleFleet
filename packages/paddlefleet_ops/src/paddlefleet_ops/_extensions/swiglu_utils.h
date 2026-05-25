// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <limits>

namespace paddlefleet {
namespace extensions {

constexpr int64_t kSwiGLUMaxRowGridSize = 65535;

inline int GetSwiGLURowGridSize(int64_t rows) {
  if (rows <= 0) {
    return 0;
  }
  return static_cast<int>(rows < kSwiGLUMaxRowGridSize ? rows
                                                       : kSwiGLUMaxRowGridSize);
}

inline bool ShouldUseInt64Index(int64_t rows, int64_t row_stride) {
  if (rows <= 0 || row_stride <= 0) {
    return false;
  }

  // The int32 path stores row_stride in IndexT, so row_stride itself must fit.
  // It is safe for offsets iff the largest linear offset is <= INT_MAX:
  //   rows * row_stride - 1 <= INT_MAX
  // Use division instead of multiplying to avoid int64 overflow.
  constexpr int64_t kIntMax =
      static_cast<int64_t>(std::numeric_limits<int>::max());
  constexpr int64_t kIntMaxPlusOne = kIntMax + 1;
  return row_stride > kIntMax || rows > kIntMaxPlusOne / row_stride;
}

}  // namespace extensions
}  // namespace paddlefleet
