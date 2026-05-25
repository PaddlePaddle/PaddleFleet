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
#include <cstdint>
#include <vector>
#include "paddle/extension.h"
#include "swiglu_utils.h"

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

constexpr int kFusedSwiGLUScaleBlockSize = 256;

// ==========================================================================
// Optimized Forward Kernel
// ==========================================================================
template <typename T, typename ScaleT, int VEC_SIZE, typename IndexT>
__global__ void VectorizedFusedSwiGLUFwd(const T* __restrict__ x,
                                         const ScaleT* __restrict__ scale,
                                         T* __restrict__ out,
                                         int64_t rows,
                                         IndexT hidden_size,
                                         IndexT row_stride) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    int tid = threadIdx.x;
    IndexT row_index = static_cast<IndexT>(row);
    IndexT lane_idx = static_cast<IndexT>(tid) * VEC_SIZE;

    float s = static_cast<float>(scale[row]);

    for (IndexT col = lane_idx; col < hidden_size;
         col += static_cast<IndexT>(blockDim.x) * VEC_SIZE) {
      IndexT gate_offset = row_index * row_stride + col;
      IndexT val_offset = gate_offset + hidden_size;
      IndexT out_offset = row_index * hidden_size + col;

      Packed128 gate_pack =
          *reinterpret_cast<const Packed128*>(&x[gate_offset]);
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
}

// ==========================================================================
// Optimized Backward Kernel
// ==========================================================================
template <typename T, typename ScaleT, int VEC_SIZE, typename IndexT>
__global__ void VectorizedFusedSwiGLUBwd(const T* __restrict__ x,
                                         const ScaleT* __restrict__ scale,
                                         const T* __restrict__ d_out,
                                         T* __restrict__ d_x,
                                         ScaleT* __restrict__ d_scale,
                                         int64_t rows,
                                         IndexT hidden_size,
                                         IndexT row_stride) {
  int tid = threadIdx.x;
  __shared__ float shared_sum[kFusedSwiGLUScaleBlockSize];

  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    IndexT row_index = static_cast<IndexT>(row);
    IndexT lane_idx = static_cast<IndexT>(tid) * VEC_SIZE;

    float local_d_scale_sum = 0.0f;
    float s = static_cast<float>(scale[row]);

    for (IndexT col = lane_idx; col < hidden_size;
         col += static_cast<IndexT>(blockDim.x) * VEC_SIZE) {
      IndexT gate_offset = row_index * row_stride + col;
      IndexT val_offset = gate_offset + hidden_size;
      IndexT out_offset = row_index * hidden_size + col;

      Packed128 gate_pack =
          *reinterpret_cast<const Packed128*>(&x[gate_offset]);
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

    if (tid < kFusedSwiGLUScaleBlockSize) {
      shared_sum[tid] = local_d_scale_sum;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride && (tid + stride) < kFusedSwiGLUScaleBlockSize) {
        shared_sum[tid] += shared_sum[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      d_scale[row] = static_cast<ScaleT>(shared_sum[0]);
    }
    __syncthreads();
  }
}

template <typename T,
          typename ScaleT,
          int VEC_SIZE,
          typename IndexT,
          typename StreamT>
void LaunchFusedSwiGLUScaleFwd(const T* x,
                               const ScaleT* scale,
                               T* out,
                               int grid_size,
                               int64_t rows,
                               int64_t hidden_size,
                               int64_t row_stride,
                               StreamT stream) {
  VectorizedFusedSwiGLUFwd<T, ScaleT, VEC_SIZE, IndexT>
      <<<grid_size, kFusedSwiGLUScaleBlockSize, 0, stream>>>(
          x,
          scale,
          out,
          rows,
          static_cast<IndexT>(hidden_size),
          static_cast<IndexT>(row_stride));
}

template <typename T, typename ScaleT, int VEC_SIZE, typename StreamT>
void DispatchFusedSwiGLUScaleFwd(const T* x,
                                 const ScaleT* scale,
                                 T* out,
                                 int grid_size,
                                 int64_t rows,
                                 int64_t hidden_size,
                                 int64_t row_stride,
                                 StreamT stream) {
  if (paddlefleet::extensions::ShouldUseInt64Index(rows, row_stride)) {
    LaunchFusedSwiGLUScaleFwd<T, ScaleT, VEC_SIZE, int64_t>(
        x, scale, out, grid_size, rows, hidden_size, row_stride, stream);
  } else {
    LaunchFusedSwiGLUScaleFwd<T, ScaleT, VEC_SIZE, int>(
        x, scale, out, grid_size, rows, hidden_size, row_stride, stream);
  }
}

template <typename T,
          typename ScaleT,
          int VEC_SIZE,
          typename IndexT,
          typename StreamT>
void LaunchFusedSwiGLUScaleBwd(const T* x,
                               const ScaleT* scale,
                               const T* d_out,
                               T* d_x,
                               ScaleT* d_scale,
                               int grid_size,
                               int64_t rows,
                               int64_t hidden_size,
                               int64_t row_stride,
                               StreamT stream) {
  VectorizedFusedSwiGLUBwd<T, ScaleT, VEC_SIZE, IndexT>
      <<<grid_size, kFusedSwiGLUScaleBlockSize, 0, stream>>>(
          x,
          scale,
          d_out,
          d_x,
          d_scale,
          rows,
          static_cast<IndexT>(hidden_size),
          static_cast<IndexT>(row_stride));
}

template <typename T, typename ScaleT, int VEC_SIZE, typename StreamT>
void DispatchFusedSwiGLUScaleBwd(const T* x,
                                 const ScaleT* scale,
                                 const T* d_out,
                                 T* d_x,
                                 ScaleT* d_scale,
                                 int grid_size,
                                 int64_t rows,
                                 int64_t hidden_size,
                                 int64_t row_stride,
                                 StreamT stream) {
  if (paddlefleet::extensions::ShouldUseInt64Index(rows, row_stride)) {
    LaunchFusedSwiGLUScaleBwd<T, ScaleT, VEC_SIZE, int64_t>(x,
                                                            scale,
                                                            d_out,
                                                            d_x,
                                                            d_scale,
                                                            grid_size,
                                                            rows,
                                                            hidden_size,
                                                            row_stride,
                                                            stream);
  } else {
    LaunchFusedSwiGLUScaleBwd<T, ScaleT, VEC_SIZE, int>(x,
                                                        scale,
                                                        d_out,
                                                        d_x,
                                                        d_scale,
                                                        grid_size,
                                                        rows,
                                                        hidden_size,
                                                        row_stride,
                                                        stream);
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

  int grid_size = paddlefleet::extensions::GetSwiGLURowGridSize(rows);
  auto stream = x.stream();

  if (x.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    if (scale.dtype() == paddle::DataType::FLOAT32) {
      DispatchFusedSwiGLUScaleFwd<cuda_bf16, float, 8>(
          reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
          scale.data<float>(),
          reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
          grid_size,
          rows,
          hidden_size,
          hidden2,
          stream);
    } else {
      DispatchFusedSwiGLUScaleFwd<cuda_bf16, cuda_bf16, 8>(
          reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
          reinterpret_cast<const cuda_bf16*>(scale.data<paddle_bf16>()),
          reinterpret_cast<cuda_bf16*>(out.data<paddle_bf16>()),
          grid_size,
          rows,
          hidden_size,
          hidden2,
          stream);
    }
  } else if (x.dtype() == paddle::DataType::FLOAT32) {
    DispatchFusedSwiGLUScaleFwd<float, float, 4>(x.data<float>(),
                                                 scale.data<float>(),
                                                 out.data<float>(),
                                                 grid_size,
                                                 rows,
                                                 hidden_size,
                                                 hidden2,
                                                 stream);
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

  int grid_size = paddlefleet::extensions::GetSwiGLURowGridSize(rows);
  auto stream = x.stream();

  if (x.dtype() == paddle::DataType::BFLOAT16) {
    using paddle_bf16 = paddle::bfloat16;
    using cuda_bf16 = __nv_bfloat16;
    if (scale.dtype() == paddle::DataType::FLOAT32) {
      DispatchFusedSwiGLUScaleBwd<cuda_bf16, float, 8>(
          reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
          scale.data<float>(),
          reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
          reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
          d_scale.data<float>(),
          grid_size,
          rows,
          hidden_size,
          hidden2,
          stream);
    } else {
      DispatchFusedSwiGLUScaleBwd<cuda_bf16, cuda_bf16, 8>(
          reinterpret_cast<const cuda_bf16*>(x.data<paddle_bf16>()),
          reinterpret_cast<const cuda_bf16*>(scale.data<paddle_bf16>()),
          reinterpret_cast<const cuda_bf16*>(d_out.data<paddle_bf16>()),
          reinterpret_cast<cuda_bf16*>(d_x.data<paddle_bf16>()),
          reinterpret_cast<cuda_bf16*>(d_scale.data<paddle_bf16>()),
          grid_size,
          rows,
          hidden_size,
          hidden2,
          stream);
    }
  } else if (x.dtype() == paddle::DataType::FLOAT32) {
    DispatchFusedSwiGLUScaleBwd<float, float, 4>(x.data<float>(),
                                                 scale.data<float>(),
                                                 d_out.data<float>(),
                                                 d_x.data<float>(),
                                                 d_scale.data<float>(),
                                                 grid_size,
                                                 rows,
                                                 hidden_size,
                                                 hidden2,
                                                 stream);
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
