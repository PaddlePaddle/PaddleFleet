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

// swiglu_kernel.cu
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>
#include <vector>
#include "paddle/extension.h"

// 128-bit memory alignment struct
struct __align__(16) Packed128 {
  int4 data;
};

template <typename T>
__device__ __forceinline__ float precise_sigmoid(T x) {
  return 1.0f / (1.0f + expf(-static_cast<float>(x)));
}

constexpr int kSwiGLUBackBlockSize = 256;

inline bool ShouldUseInt64Index(int64_t rows, int64_t row_stride) {
  // This predicate protects Y/DX offsets:
  //   row * row_stride + col
  // The G offset uses row * hidden_size + col. For fused_swiglu_bwd,
  // row_stride is 2 * hidden_size after shape checks, so any int32-safe Y/DX
  // offset range also bounds the G offset range.
  return rows >=
         static_cast<int64_t>(std::numeric_limits<int>::max()) / row_stride;
}

inline void CheckSwiGLUBackShape(const paddle::Tensor& g,
                                 const paddle::Tensor& y,
                                 int64_t input_dim,
                                 int64_t hidden_size) {
  auto g_shape = g.shape();
  auto y_shape = y.shape();
  PADDLE_ENFORCE_EQ(
      g_shape.size(),
      y_shape.size(),
      common::errors::InvalidArgument(
          "Input G and Y must have the same rank for fused_swiglu_bwd, "
          "but got G rank %d and Y rank %d.",
          g_shape.size(),
          y_shape.size()));

  PADDLE_ENFORCE_EQ(
      input_dim % 2,
      0,
      common::errors::InvalidArgument(
          "The last dimension of Input Y must be even for fused_swiglu_bwd, "
          "but got %d.",
          input_dim));

  PADDLE_ENFORCE_EQ(
      g_shape.back(),
      hidden_size,
      common::errors::InvalidArgument(
          "The last dimension of Input G must equal half of Input Y's last "
          "dimension for fused_swiglu_bwd, but got G last dimension %d and "
          "Y last dimension %d.",
          g_shape.back(),
          input_dim));

  for (size_t i = 0; i + 1 < y_shape.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        g_shape[i],
        y_shape[i],
        common::errors::InvalidArgument(
            "Input G and Y must have the same prefix shape for "
            "fused_swiglu_bwd, but got G.shape[%d] = %d and Y.shape[%d] = %d.",
            i,
            g_shape[i],
            i,
            y_shape[i]));
  }
}

inline void CheckSwiGLUBackPackedAccess(paddle::DataType dtype,
                                        int64_t hidden_size) {
  int vec_size = 0;
  if (dtype == paddle::DataType::BFLOAT16) {
    vec_size = 8;
  } else if (dtype == paddle::DataType::FLOAT32) {
    vec_size = 4;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "fused_swiglu_bwd only supports bfloat16 and float32, but got %s.",
        phi::DataTypeToString(dtype)));
  }

  PADDLE_ENFORCE_EQ(
      hidden_size % vec_size,
      0,
      common::errors::InvalidArgument(
          "The hidden size of fused_swiglu_bwd must be divisible by %d for "
          "128-bit vectorized access, but got %d.",
          vec_size,
          hidden_size));
}

// Custom Kernel Implementation
template <typename T, int VEC_SIZE, typename IndexT>
__global__ void VectorizedSwiGLUBackKernel(const T* __restrict__ g,
                                           const T* __restrict__ y,
                                           T* __restrict__ dx,
                                           IndexT hidden_size,
                                           IndexT input_stride) {
  IndexT row = static_cast<IndexT>(blockIdx.x);
  int tid = threadIdx.x;
  IndexT lane_idx = static_cast<IndexT>(tid) * VEC_SIZE;

  for (IndexT col = lane_idx; col < hidden_size;
       col += static_cast<IndexT>(blockDim.x) * VEC_SIZE) {
    IndexT g_offset = row * hidden_size + col;
    IndexT y1_offset = row * input_stride + col;
    IndexT y2_offset = y1_offset + hidden_size;

    Packed128 g_pack = *reinterpret_cast<const Packed128*>(&g[g_offset]);
    Packed128 y1_pack = *reinterpret_cast<const Packed128*>(&y[y1_offset]);
    Packed128 y2_pack = *reinterpret_cast<const Packed128*>(&y[y2_offset]);

    const T* g_ptr = reinterpret_cast<const T*>(&g_pack);
    const T* y1_ptr = reinterpret_cast<const T*>(&y1_pack);
    const T* y2_ptr = reinterpret_cast<const T*>(&y2_pack);

    T dx1_buffer[VEC_SIZE];
    T dx2_buffer[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float val_g = static_cast<float>(g_ptr[i]);
      float val_y1 = static_cast<float>(y1_ptr[i]);
      float val_y2 = static_cast<float>(y2_ptr[i]);

      float sig_y1 = precise_sigmoid(val_y1);
      float silu_y1 = val_y1 * sig_y1;

      dx2_buffer[i] = static_cast<T>(val_g * silu_y1);

      float d_silu = sig_y1 * (1.0f + val_y1 * (1.0f - sig_y1));
      float grad_y1 = val_g * val_y2 * d_silu;

      dx1_buffer[i] = static_cast<T>(grad_y1);
    }

    *reinterpret_cast<Packed128*>(&dx[y1_offset]) =
        *reinterpret_cast<Packed128*>(dx1_buffer);
    *reinterpret_cast<Packed128*>(&dx[y2_offset]) =
        *reinterpret_cast<Packed128*>(dx2_buffer);
  }
}

template <typename T, int VEC_SIZE, typename IndexT, typename StreamT>
void LaunchSwiGLUBackKernel(const T* g,
                            const T* y,
                            T* dx,
                            int grid_size,
                            int64_t hidden_size,
                            int64_t input_stride,
                            StreamT stream) {
  VectorizedSwiGLUBackKernel<T, VEC_SIZE, IndexT>
      <<<grid_size, kSwiGLUBackBlockSize, 0, stream>>>(
          g,
          y,
          dx,
          static_cast<IndexT>(hidden_size),
          static_cast<IndexT>(input_stride));
}

template <typename T, int VEC_SIZE, typename StreamT>
void DispatchSwiGLUBackKernel(const T* g,
                              const T* y,
                              T* dx,
                              int grid_size,
                              int64_t rows,
                              int64_t hidden_size,
                              int64_t input_stride,
                              StreamT stream) {
  if (ShouldUseInt64Index(rows, input_stride)) {
    LaunchSwiGLUBackKernel<T, VEC_SIZE, int64_t>(
        g, y, dx, grid_size, hidden_size, input_stride, stream);
  } else {
    LaunchSwiGLUBackKernel<T, VEC_SIZE, int>(
        g, y, dx, grid_size, hidden_size, input_stride, stream);
  }
}

std::vector<paddle::Tensor> SwiGLUBackward(const paddle::Tensor& g,
                                           const paddle::Tensor& y) {
  auto y_shape = y.shape();
  auto dx = paddle::empty_like(y);

  PADDLE_ENFORCE_GT(
      y_shape.size(),
      0,
      common::errors::InvalidArgument(
          "Input Y must have at least one dimension for fused_swiglu_bwd."));

  int64_t input_dim = y_shape.back();
  int64_t hidden_size = input_dim / 2;
  PADDLE_ENFORCE_EQ(
      g.dtype(),
      y.dtype(),
      common::errors::InvalidArgument(
          "Input G and Y must have the same dtype for fused_swiglu_bwd, "
          "but got G dtype %s and Y dtype %s.",
          phi::DataTypeToString(g.dtype()),
          phi::DataTypeToString(y.dtype())));
  CheckSwiGLUBackShape(g, y, input_dim, hidden_size);

  if (input_dim == 0) {
    return {dx};
  }

  int64_t rows = y.numel() / input_dim;
  if (rows == 0 || hidden_size == 0) {
    return {dx};
  }

  CheckSwiGLUBackPackedAccess(y.dtype(), hidden_size);

  PADDLE_ENFORCE_LE(
      rows,
      static_cast<int64_t>(std::numeric_limits<int>::max()),
      common::errors::InvalidArgument(
          "rows must be <= INT_MAX for fused_swiglu_bwd because one CUDA "
          "block is launched per row."));

  int grid_size = static_cast<int>(rows);
  auto stream = y.stream();

  if (y.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    DispatchSwiGLUBackKernel<cuda_bf16, 8>(
        reinterpret_cast<const cuda_bf16*>(g.data<paddle_bf16>()),
        reinterpret_cast<const cuda_bf16*>(y.data<paddle_bf16>()),
        reinterpret_cast<cuda_bf16*>(dx.data<paddle_bf16>()),
        grid_size,
        rows,
        hidden_size,
        input_dim,
        stream);
  } else if (y.dtype() == paddle::DataType::FLOAT32) {
    DispatchSwiGLUBackKernel<float, 4>(g.data<float>(),
                                       y.data<float>(),
                                       dx.data<float>(),
                                       grid_size,
                                       rows,
                                       hidden_size,
                                       input_dim,
                                       stream);
  }
  return {dx};
}

// Infer Functions
std::vector<std::vector<int64_t>> SwiGLUBackInferShape(std::vector<int64_t> g,
                                                       std::vector<int64_t> y) {
  return {y};
}
std::vector<paddle::DataType> SwiGLUBackInferDtype(paddle::DataType g,
                                                   paddle::DataType y) {
  return {y};
}

PD_BUILD_OP(fused_swiglu_bwd)
    .Inputs({"G", "Y"})
    .Outputs({"DX"})
    .SetKernelFn(PD_KERNEL(SwiGLUBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SwiGLUBackInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SwiGLUBackInferDtype));
