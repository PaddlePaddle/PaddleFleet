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

#include <cuda_bf16.h>
#include <limits>
#include <vector>
#include "paddle/extension.h"

// ==========================================================================
// Utils: Packed Memory Access (128-bit Vectorization)
// ==========================================================================

struct __align__(16) Packed128 {
  int4 data;
};

// ------------------------------------------------------------------
// Sigmoid implementation
// ------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float precise_sigmoid(T x) {
  return 1.0f / (1.0f + expf(-static_cast<float>(x)));
}

// ==========================================================================
// Optimized Forward Kernel
// ==========================================================================
template <typename T, typename ScaleT, int VEC_SIZE>
__global__ void VectorizedFusedSwiGLUFwd(const T* __restrict__ x,
                                         const ScaleT* __restrict__ scale,
                                         T* __restrict__ out,
                                         int hidden_size,
                                         int row_stride) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane_idx = tid * VEC_SIZE;

  float s = static_cast<float>(scale[row]);

  for (int col = lane_idx; col < hidden_size; col += blockDim.x * VEC_SIZE) {
    int gate_offset = row * row_stride + col;
    int val_offset = gate_offset + hidden_size;
    int out_offset = row * hidden_size + col;

    Packed128 gate_pack = *reinterpret_cast<const Packed128*>(&x[gate_offset]);
    Packed128 val_pack = *reinterpret_cast<const Packed128*>(&x[val_offset]);

    T* gate_ptr = reinterpret_cast<T*>(&gate_pack);
    T* val_ptr = reinterpret_cast<T*>(&val_pack);

    T res_buffer[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float g = static_cast<float>(gate_ptr[i]);
      float v = static_cast<float>(val_ptr[i]);

      float swiglu = (g * precise_sigmoid(g)) * v;
      res_buffer[i] = static_cast<T>(swiglu * s);
    }

    *reinterpret_cast<Packed128*>(&out[out_offset]) =
        *reinterpret_cast<Packed128*>(res_buffer);
  }
}

// ==========================================================================
// Optimized Backward Kernel
// ==========================================================================
template <typename T, typename ScaleT, int VEC_SIZE>
__global__ void VectorizedFusedSwiGLUBwd(const T* __restrict__ x,
                                         const ScaleT* __restrict__ scale,
                                         const T* __restrict__ d_out,
                                         T* __restrict__ d_x,
                                         ScaleT* __restrict__ d_scale,
                                         int hidden_size,
                                         int row_stride) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane_idx = tid * VEC_SIZE;

  float local_d_scale_sum = 0.0f;
  float s = static_cast<float>(scale[row]);

  for (int col = lane_idx; col < hidden_size; col += blockDim.x * VEC_SIZE) {
    int gate_offset = row * row_stride + col;
    int val_offset = gate_offset + hidden_size;
    int out_offset = row * hidden_size + col;

    Packed128 gate_pack = *reinterpret_cast<const Packed128*>(&x[gate_offset]);
    Packed128 val_pack = *reinterpret_cast<const Packed128*>(&x[val_offset]);
    Packed128 dout_pack =
        *reinterpret_cast<const Packed128*>(&d_out[out_offset]);

    T* gate_ptr = reinterpret_cast<T*>(&gate_pack);
    T* val_ptr = reinterpret_cast<T*>(&val_pack);
    T* dout_ptr = reinterpret_cast<T*>(&dout_pack);

    T dg_buffer[VEC_SIZE];
    T dv_buffer[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float g = static_cast<float>(gate_ptr[i]);
      float v = static_cast<float>(val_ptr[i]);
      float dout = static_cast<float>(dout_ptr[i]);

      float sig_g = precise_sigmoid(g);
      float swiglu_val = (g * sig_g) * v;

      local_d_scale_sum += dout * swiglu_val;

      float d_u = dout * s;
      float silu_g = g * sig_g;

      dv_buffer[i] = static_cast<T>(d_u * silu_g);

      float d_g_val = d_u * v * sig_g * (1.0f + g * (1.0f - sig_g));
      dg_buffer[i] = static_cast<T>(d_g_val);
    }

    *reinterpret_cast<Packed128*>(&d_x[gate_offset]) =
        *reinterpret_cast<Packed128*>(dg_buffer);
    *reinterpret_cast<Packed128*>(&d_x[val_offset]) =
        *reinterpret_cast<Packed128*>(dv_buffer);
  }

  static __shared__ float shared_sum[256];
  if (tid < 256) shared_sum[tid] = local_d_scale_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride && (tid + stride) < 256) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_scale[row] = static_cast<ScaleT>(shared_sum[0]);
  }
}

// ==========================================================================
// Host Wrappers & Op Registration
// ==========================================================================

std::vector<paddle::Tensor> FusedSwiGLUScaleForward(
    const paddle::Tensor& x, const paddle::Tensor& scale) {
  auto rows = x.shape()[0];
  auto hidden2 = x.shape()[1];
  auto hidden_size = hidden2 / 2;
  auto out = paddle::empty({rows, hidden_size}, x.dtype(), x.place());

  if (rows == 0 || hidden_size == 0) {
    return {out};
  }

  PADDLE_ENFORCE_LE(
      rows * hidden2,
      static_cast<int64_t>(std::numeric_limits<int>::max()),
      common::errors::InvalidArgument(
          "rows * hidden2 must be <= INT_MAX for fused_swiglu_scale."));

  int grid_size = rows;
  int block_size = 256;
  auto stream = x.stream();

  if (x.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    if (scale.dtype() == paddle::DataType::FLOAT32) {
      VectorizedFusedSwiGLUFwd<cuda_bf16, float, 8>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              scale.data<float>(),
              reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
              hidden_size,
              hidden2);
    } else {
      VectorizedFusedSwiGLUFwd<cuda_bf16, cuda_bf16, 8>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(scale.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
              hidden_size,
              hidden2);
    }
  } else if (x.dtype() == paddle::DataType::FLOAT32) {
    VectorizedFusedSwiGLUFwd<float, float, 4>
        <<<grid_size, block_size, 0, stream>>>(x.data<float>(),
                                               scale.data<float>(),
                                               out.data<float>(),
                                               hidden_size,
                                               hidden2);
  }
  return {out};
}

std::vector<paddle::Tensor> FusedSwiGLUScaleBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& scale,
    const paddle::Tensor& d_out) {
  auto rows = x.shape()[0];
  auto hidden2 = x.shape()[1];
  auto hidden_size = hidden2 / 2;
  auto d_x = paddle::empty_like(x);
  auto d_scale = paddle::empty_like(scale);

  if (rows == 0 || hidden_size == 0) {
    return {d_x, d_scale};
  }

  PADDLE_ENFORCE_LE(
      rows * hidden2,
      static_cast<int64_t>(std::numeric_limits<int>::max()),
      common::errors::InvalidArgument(
          "rows * hidden2 must be <= INT_MAX for fused_swiglu_scale."));

  int grid_size = rows;
  int block_size = 256;
  auto stream = x.stream();

  if (x.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    if (scale.dtype() == paddle::DataType::FLOAT32) {
      VectorizedFusedSwiGLUBwd<cuda_bf16, float, 8>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              scale.data<float>(),
              reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
              d_scale.data<float>(),
              hidden_size,
              hidden2);
    } else {
      VectorizedFusedSwiGLUBwd<cuda_bf16, cuda_bf16, 8>
          <<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(scale.data<paddle_bf16>()),
              reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
              reinterpret_cast<cuda_bf16*>(d_scale.data<paddle_bf16>()),
              hidden_size,
              hidden2);
    }
  } else if (x.dtype() == paddle::DataType::FLOAT32) {
    VectorizedFusedSwiGLUBwd<float, float, 4>
        <<<grid_size, block_size, 0, stream>>>(x.data<float>(),
                                               scale.data<float>(),
                                               d_out.data<float>(),
                                               d_x.data<float>(),
                                               d_scale.data<float>(),
                                               hidden_size,
                                               hidden2);
  }
  return {d_x, d_scale};
}

// Registration
std::vector<std::vector<int64_t>> FusedGradInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> dout_shape) {
  return {x_shape, scale_shape};
}

std::vector<paddle::DataType> FusedGradInferDtype(paddle::DataType x_dtype,
                                                  paddle::DataType scale_dtype,
                                                  paddle::DataType dout_dtype) {
  return {x_dtype, scale_dtype};
}

PD_BUILD_OP(fused_swiglu_scale_bwd)
    .Inputs({"X", "Scale", "DOut"})
    .Outputs({"DX", "DScale"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedGradInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGradInferDtype));

PD_BUILD_OP(fused_swiglu_scale)
    .Inputs({"X", "Scale"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleForward))
    .SetInferShapeFn(
        PD_INFER_SHAPE(FusedGradInferShape))  // Reuse infer shape logic
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGradInferDtype));

PD_BUILD_GRAD_OP(fused_swiglu_scale)
    .Inputs({"X", "Scale", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Scale")})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleBackward));
