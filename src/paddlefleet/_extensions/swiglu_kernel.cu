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

// Custom Kernel Implementation
template <typename T, int VEC_SIZE>
__global__ void VectorizedSwiGLUBackKernel(const T* __restrict__ g,
                                           const T* __restrict__ y,
                                           T* __restrict__ dx,
                                           int hidden_size,
                                           int input_stride) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane_idx = tid * VEC_SIZE;

  for (int col = lane_idx; col < hidden_size; col += blockDim.x * VEC_SIZE) {
    int g_offset = row * hidden_size + col;
    int y1_offset = row * input_stride + col;
    int y2_offset = y1_offset + hidden_size;

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

std::vector<paddle::Tensor> SwiGLUBackward(const paddle::Tensor& g,
                                           const paddle::Tensor& y) {
  auto y_shape = y.shape();
  int rows = y.numel() / y_shape.back();
  int input_dim = y_shape.back();
  int hidden_size = input_dim / 2;
  auto dx = paddle::empty_like(y);

  int grid_size = rows;
  int block_size = 256;
  auto stream = y.stream();

  if (y.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    VectorizedSwiGLUBackKernel<cuda_bf16, 8>
        <<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const cuda_bf16*>(g.data<paddle_bf16>()),
            reinterpret_cast<const cuda_bf16*>(y.data<paddle_bf16>()),
            reinterpret_cast<cuda_bf16*>(dx.data<paddle_bf16>()),
            hidden_size,
            input_dim);
  } else if (y.dtype() == paddle::DataType::FLOAT32) {
    VectorizedSwiGLUBackKernel<float, 4>
        <<<grid_size, block_size, 0, stream>>>(g.data<float>(),
                                               y.data<float>(),
                                               dx.data<float>(),
                                               hidden_size,
                                               input_dim);
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
