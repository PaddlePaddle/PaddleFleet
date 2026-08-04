// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// Fused 1x32 quantization primitives for SM100 W4A8 training.  The numerical
// contract intentionally matches paddlefleet.transformer.moe.fp8_utils:
//   * amax is clamped to 1e-4 before division;
//   * scale is rounded upward to an exact UE8M0 power of two;
//   * E2M1 uses strict midpoint comparisons and packs even K in the low nibble.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "paddle/extension.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float8_e4m3fn.h"

namespace {

constexpr int kQuantBlock = 32;
constexpr float kAmaxFloor = 1.0e-4f;
constexpr float kFp8Max = 448.0f;
constexpr float kFp4Max = 6.0f;
constexpr int kMaxGridX = 1024 * 1024;
constexpr int kGroupWarpsPerBlock = 8;
constexpr int kGroupThreadsPerBlock = kGroupWarpsPerBlock * 32;

enum class QuantKind : int64_t { kFp8 = 0, kFp4 = 1 };

template <typename T>
__device__ __forceinline__ float LoadAsFloat(const T* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <>
__device__ __forceinline__ float LoadAsFloat<__nv_bfloat16>(
    const __nv_bfloat16* ptr, int64_t index) {
  return __bfloat162float(ptr[index]);
}

__device__ __forceinline__ float WarpMax(float value) {
  constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kMask, value, offset));
  }
  return __shfl_sync(kMask, value, 0);
}

__device__ __forceinline__ float CeilUe8m0(float value) {
  const uint32_t bits = __float_as_uint(fabsf(value));
  uint32_t exponent = (bits >> 23) & 0xffu;
  exponent += (bits & 0x7fffffu) != 0u;
  exponent = max(1u, min(254u, exponent));
  return __uint_as_float(exponent << 23);
}

__device__ __forceinline__ uint8_t EncodeE2m1Strict(float value) {
  const float absolute = fabsf(value);
  uint8_t code = static_cast<uint8_t>((absolute > 0.25f) + (absolute > 0.75f) +
                                      (absolute > 1.25f) + (absolute > 1.75f) +
                                      (absolute > 2.5f) + (absolute > 3.5f) +
                                      (absolute > 5.0f));
  if (value < 0.0f && code != 0u) {
    code |= 0x8u;
  }
  return code;
}

inline int GridFor(int64_t work_items) {
  return static_cast<int>(std::min<int64_t>(work_items, kMaxGridX));
}

inline void CheckCudaLaunch(const char* op_name) {
  const cudaError_t error = cudaGetLastError();
  PD_CHECK(error == cudaSuccess,
           op_name,
           " CUDA launch failed: ",
           cudaGetErrorString(error));
}

template <typename T, QuantKind Kind>
__global__ void Quantize1x32Kernel(const T* __restrict__ input,
                                   void* __restrict__ output,
                                   float* __restrict__ scale,
                                   int64_t total_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * kGroupWarpsPerBlock + warp;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * kGroupWarpsPerBlock) {
    const int64_t input_base = group * kQuantBlock;
    float local_amax;

    if constexpr (Kind == QuantKind::kFp8) {
      const float value = LoadAsFloat(input, input_base + lane);
      local_amax = fabsf(value);
      const float amax = WarpMax(local_amax);
      const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp8Max);
      if (lane == 0) {
        scale[group] = group_scale;
      }
      auto* fp8_output = reinterpret_cast<__nv_fp8_e4m3*>(output);
      fp8_output[input_base + lane] =
          static_cast<__nv_fp8_e4m3>(value / group_scale);
    } else {
      float first = 0.0f;
      float second = 0.0f;
      if (lane < 16) {
        first = LoadAsFloat(input, input_base + lane * 2);
        second = LoadAsFloat(input, input_base + lane * 2 + 1);
      }
      local_amax = lane < 16 ? fmaxf(fabsf(first), fabsf(second)) : 0.0f;
      const float amax = WarpMax(local_amax);
      const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp4Max);
      if (lane == 0) {
        scale[group] = group_scale;
      }
      if (lane < 16) {
        const uint8_t low = EncodeE2m1Strict(first / group_scale);
        const uint8_t high = EncodeE2m1Strict(second / group_scale);
        reinterpret_cast<int8_t*>(output)[group * 16 + lane] =
            static_cast<int8_t>(low | (high << 4));
      }
    }
  }
}

template <typename T>
__global__ void StackTransposeFp4Quantize1x32Kernel(const T* __restrict__ input,
                                                    int8_t* __restrict__ output,
                                                    float* __restrict__ scale,
                                                    int64_t rows,
                                                    int64_t cols) {
  // input [E, rows, cols], output [E, cols, rows/2].  Padding one shared
  // column avoids bank conflicts while each column computes its own scale.
  __shared__ float tile[kQuantBlock][kQuantBlock + 1];
  __shared__ float tile_scale[kQuantBlock];

  const int64_t expert = blockIdx.z;
  const int64_t row_group = blockIdx.y;
  const int64_t col_group = blockIdx.x;
  const int thread = threadIdx.x;
  const int64_t input_expert_base = expert * rows * cols;

#pragma unroll
  for (int linear = thread; linear < kQuantBlock * kQuantBlock;
       linear += blockDim.x) {
    const int local_row = linear / kQuantBlock;
    const int local_col = linear % kQuantBlock;
    const int64_t global_row = row_group * kQuantBlock + local_row;
    const int64_t global_col = col_group * kQuantBlock + local_col;
    tile[local_row][local_col] =
        LoadAsFloat(input, input_expert_base + global_row * cols + global_col);
  }
  __syncthreads();

  if (thread < kQuantBlock) {
    float amax = 0.0f;
#pragma unroll
    for (int local_row = 0; local_row < kQuantBlock; ++local_row) {
      amax = fmaxf(amax, fabsf(tile[local_row][thread]));
    }
    const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp4Max);
    tile_scale[thread] = group_scale;
    const int64_t global_col = col_group * kQuantBlock + thread;
    scale[(expert * cols + global_col) * (rows / kQuantBlock) + row_group] =
        group_scale;
  }
  __syncthreads();

  // A 32x32 input tile becomes a 32x16 packed output tile.  The flattened
  // mapping gives each warp two adjacent 16-byte row segments.
#pragma unroll
  for (int linear = thread; linear < kQuantBlock * (kQuantBlock / 2);
       linear += blockDim.x) {
    const int local_col = linear / (kQuantBlock / 2);
    const int local_pair = linear % (kQuantBlock / 2);
    const float group_scale = tile_scale[local_col];
    const uint8_t low =
        EncodeE2m1Strict(tile[local_pair * 2][local_col] / group_scale);
    const uint8_t high =
        EncodeE2m1Strict(tile[local_pair * 2 + 1][local_col] / group_scale);
    const int64_t global_col = col_group * kQuantBlock + local_col;
    const int64_t global_pair = row_group * (kQuantBlock / 2) + local_pair;
    output[(expert * cols + global_col) * (rows / 2) + global_pair] =
        static_cast<int8_t>(low | (high << 4));
  }
}

template <bool HasClamp>
__global__ void WeightedSwigluFp8Quantize1x32Kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ probs,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int64_t rows,
    int64_t hidden,
    float clamp_value,
    int64_t total_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int64_t group_index =
           static_cast<int64_t>(blockIdx.x) * kGroupWarpsPerBlock + warp;
       group_index < total_groups;
       group_index += static_cast<int64_t>(gridDim.x) * kGroupWarpsPerBlock) {
    const int64_t groups_per_row = hidden / kQuantBlock;
    const int64_t row = group_index / groups_per_row;
    const int64_t group = group_index - row * groups_per_row;
    const int64_t col = group * kQuantBlock + lane;
    const int64_t input_base = row * hidden * 2;
    float gate = __bfloat162float(input[input_base + col]);
    float up = __bfloat162float(input[input_base + hidden + col]);
    if constexpr (HasClamp) {
      gate = fminf(gate, clamp_value);
      up = fmaxf(fminf(up, clamp_value), -clamp_value);
    }
    const float sigmoid = HasClamp ? 1.0f / (1.0f + __expf(-gate))
                                   : __frcp_rn(1.0f + __expf(-gate));
    const float value = gate * sigmoid * up * probs[row];
    const float amax = WarpMax(fabsf(value));
    const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp8Max);
    if (lane == 0) {
      scale[group_index] = group_scale;
    }
    output[row * hidden + col] =
        static_cast<__nv_fp8_e4m3>(value / group_scale);
  }
}

__global__ void DequantizeFp8OneBy32Kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    int64_t numel) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float value =
        static_cast<float>(input[index]) * scale[index / kQuantBlock];
    output[index] = __float2bfloat16_rn(value);
  }
}

void CheckGpuContiguous(const paddle::Tensor& input, const char* op_name) {
  PD_CHECK(input.is_gpu(), op_name, " requires a GPU tensor");
  PD_CHECK(input.is_contiguous(), op_name, " requires a contiguous tensor");
}

template <typename Fn>
void DispatchFloatInput(const paddle::Tensor& input, Fn&& fn) {
  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16:
      fn(reinterpret_cast<const __nv_bfloat16*>(
          mutable_input.data<phi::bfloat16>()));
      break;
    case paddle::DataType::FLOAT32:
      fn(mutable_input.data<float>());
      break;
    default:
      PD_THROW("W4A8 quantization supports only bfloat16 and float32 input");
  }
}

}  // namespace

std::vector<paddle::Tensor> W4A8Quantize1x32(const paddle::Tensor& input,
                                             int64_t quant_kind) {
  constexpr const char* kOpName = "w4a8_quantize_1x32";
  CheckGpuContiguous(input, kOpName);
  PD_CHECK(input.shape().size() == 2, kOpName, " expects rank-2 input");
  PD_CHECK(input.shape()[1] % kQuantBlock == 0,
           kOpName,
           " last dimension must be divisible by 32");
  PD_CHECK(quant_kind == static_cast<int64_t>(QuantKind::kFp8) ||
               quant_kind == static_cast<int64_t>(QuantKind::kFp4),
           kOpName,
           " quant_kind must be 0 (FP8) or 1 (FP4)");

  const int64_t rows = input.shape()[0];
  const int64_t cols = input.shape()[1];
  const bool is_fp8 = quant_kind == static_cast<int64_t>(QuantKind::kFp8);
  auto output = paddle::empty(
      {rows, is_fp8 ? cols : cols / 2},
      is_fp8 ? paddle::DataType::FLOAT8_E4M3FN : paddle::DataType::INT8,
      input.place());
  auto scale = paddle::empty(
      {rows, cols / kQuantBlock}, paddle::DataType::FLOAT32, input.place());
  const int64_t groups = rows * (cols / kQuantBlock);
  if (groups == 0) {
    return {output, scale};
  }

  DispatchFloatInput(input, [&](auto* input_ptr) {
    using InputT = std::remove_cv_t<std::remove_pointer_t<decltype(input_ptr)>>;
    if (is_fp8) {
      Quantize1x32Kernel<InputT, QuantKind::kFp8>
          <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
             kGroupThreadsPerBlock,
             0,
             input.stream()>>>(
              input_ptr, output.data(), scale.data<float>(), groups);
    } else {
      Quantize1x32Kernel<InputT, QuantKind::kFp4>
          <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
             kGroupThreadsPerBlock,
             0,
             input.stream()>>>(
              input_ptr, output.data(), scale.data<float>(), groups);
    }
  });
  CheckCudaLaunch(kOpName);
  return {output, scale};
}

std::vector<paddle::Tensor> W4A8StackQuantize1x32(const paddle::Tensor& input,
                                                  bool transpose) {
  constexpr const char* kOpName = "w4a8_stack_quantize_1x32";
  CheckGpuContiguous(input, kOpName);
  PD_CHECK(input.shape().size() == 3, kOpName, " expects [E, R, C]");
  const int64_t experts = input.shape()[0];
  const int64_t rows = input.shape()[1];
  const int64_t cols = input.shape()[2];
  const int64_t quant_dim = transpose ? rows : cols;
  PD_CHECK(quant_dim % kQuantBlock == 0,
           kOpName,
           " quantized dimension must be divisible by 32");
  if (transpose) {
    PD_CHECK(cols % kQuantBlock == 0,
             kOpName,
             " transpose output rows must be divisible by 32");
  }

  const std::vector<int64_t> output_shape =
      transpose ? std::vector<int64_t>{experts, cols, rows / 2}
                : std::vector<int64_t>{experts, rows, cols / 2};
  const std::vector<int64_t> scale_shape =
      transpose ? std::vector<int64_t>{experts, cols, rows / kQuantBlock}
                : std::vector<int64_t>{experts, rows, cols / kQuantBlock};
  auto output =
      paddle::empty(output_shape, paddle::DataType::INT8, input.place());
  auto scale =
      paddle::empty(scale_shape, paddle::DataType::FLOAT32, input.place());
  if (input.numel() == 0) {
    return {output, scale};
  }

  DispatchFloatInput(input, [&](auto* input_ptr) {
    using InputT = std::remove_cv_t<std::remove_pointer_t<decltype(input_ptr)>>;
    if (transpose) {
      const dim3 grid(cols / kQuantBlock,
                      rows / kQuantBlock,
                      static_cast<unsigned int>(experts));
      StackTransposeFp4Quantize1x32Kernel<InputT>
          <<<grid, 256, 0, input.stream()>>>(input_ptr,
                                             output.data<int8_t>(),
                                             scale.data<float>(),
                                             rows,
                                             cols);
    } else {
      const int64_t groups = experts * rows * (cols / kQuantBlock);
      Quantize1x32Kernel<InputT, QuantKind::kFp4>
          <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
             kGroupThreadsPerBlock,
             0,
             input.stream()>>>(
              input_ptr, output.data(), scale.data<float>(), groups);
    }
  });
  CheckCudaLaunch(kOpName);
  return {output, scale};
}

std::vector<paddle::Tensor> W4A8WeightedSwigluQuantize1x32(
    const paddle::Tensor& input,
    const paddle::Tensor& probs,
    double clamp_value) {
  constexpr const char* kOpName = "w4a8_weighted_swiglu_quantize_1x32";
  CheckGpuContiguous(input, kOpName);
  CheckGpuContiguous(probs, kOpName);
  PD_CHECK(input.dtype() == paddle::DataType::BFLOAT16,
           kOpName,
           " input must be bfloat16");
  PD_CHECK(probs.dtype() == paddle::DataType::FLOAT32,
           kOpName,
           " probs must be float32");
  PD_CHECK(input.shape().size() == 2 && input.shape()[1] % 2 == 0,
           kOpName,
           " input must have shape [M, 2H]");
  const int64_t rows = input.shape()[0];
  const int64_t hidden = input.shape()[1] / 2;
  PD_CHECK(hidden % kQuantBlock == 0, kOpName, " H must be divisible by 32");
  PD_CHECK(
      probs.numel() == rows, kOpName, " probs must contain one float per row");

  auto output = paddle::empty(
      {rows, hidden}, paddle::DataType::FLOAT8_E4M3FN, input.place());
  auto scale = paddle::empty(
      {rows, hidden / kQuantBlock}, paddle::DataType::FLOAT32, input.place());
  const int64_t groups = rows * (hidden / kQuantBlock);
  if (groups == 0) {
    return {output, scale};
  }

  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  auto& mutable_probs = const_cast<paddle::Tensor&>(probs);
  const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(
      mutable_input.data<phi::bfloat16>());
  auto* output_ptr =
      reinterpret_cast<__nv_fp8_e4m3*>(output.data<phi::float8_e4m3fn>());
  if (clamp_value > 0.0) {
    WeightedSwigluFp8Quantize1x32Kernel<true>
        <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
           kGroupThreadsPerBlock,
           0,
           input.stream()>>>(input_ptr,
                             mutable_probs.data<float>(),
                             output_ptr,
                             scale.data<float>(),
                             rows,
                             hidden,
                             static_cast<float>(clamp_value),
                             groups);
  } else {
    WeightedSwigluFp8Quantize1x32Kernel<false>
        <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
           kGroupThreadsPerBlock,
           0,
           input.stream()>>>(input_ptr,
                             mutable_probs.data<float>(),
                             output_ptr,
                             scale.data<float>(),
                             rows,
                             hidden,
                             0.0f,
                             groups);
  }
  CheckCudaLaunch(kOpName);
  return {output, scale};
}

std::vector<paddle::Tensor> W4A8Dequantize1x32Impl(
    const paddle::Tensor& input, const paddle::Tensor& scale) {
  constexpr const char* kOpName = "w4a8_dequantize_1x32";
  CheckGpuContiguous(input, kOpName);
  CheckGpuContiguous(scale, kOpName);
  PD_CHECK(input.dtype() == paddle::DataType::FLOAT8_E4M3FN,
           kOpName,
           " input must be float8_e4m3fn");
  PD_CHECK(scale.dtype() == paddle::DataType::FLOAT32,
           kOpName,
           " scale must be float32");
  PD_CHECK(input.shape().size() == 2 && input.shape()[1] % kQuantBlock == 0,
           kOpName,
           " input must be [M, K] with K divisible by 32");
  const std::vector<int64_t> expected_scale_shape = {
      input.shape()[0], input.shape()[1] / kQuantBlock};
  PD_CHECK(scale.shape() == expected_scale_shape,
           kOpName,
           " scale shape must be [M, K/32]");

  auto output =
      paddle::empty(input.shape(), paddle::DataType::BFLOAT16, input.place());
  if (input.numel() == 0) {
    return {output};
  }
  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  auto& mutable_scale = const_cast<paddle::Tensor&>(scale);
  const int threads = 256;
  const int blocks = GridFor((input.numel() + threads - 1) / threads);
  DequantizeFp8OneBy32Kernel<<<blocks, threads, 0, input.stream()>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(
          mutable_input.data<phi::float8_e4m3fn>()),
      mutable_scale.data<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data<phi::bfloat16>()),
      input.numel());
  CheckCudaLaunch(kOpName);
  return {output};
}

std::vector<std::vector<int64_t>> QuantInferShape(
    std::vector<int64_t> input_shape, int64_t quant_kind) {
  const bool fp8 = quant_kind == static_cast<int64_t>(QuantKind::kFp8);
  return {{input_shape[0], fp8 ? input_shape[1] : input_shape[1] / 2},
          {input_shape[0], input_shape[1] / kQuantBlock}};
}

std::vector<paddle::DataType> QuantInferDtype(paddle::DataType input_dtype,
                                              int64_t quant_kind) {
  return {quant_kind == static_cast<int64_t>(QuantKind::kFp8)
              ? paddle::DataType::FLOAT8_E4M3FN
              : paddle::DataType::INT8,
          paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> StackInferShape(
    std::vector<int64_t> input_shape, bool transpose) {
  const int64_t e = input_shape[0];
  const int64_t r = input_shape[1];
  const int64_t c = input_shape[2];
  return transpose ? std::vector<std::vector<int64_t>>{{e, c, r / 2},
                                                       {e, c, r / kQuantBlock}}
                   : std::vector<std::vector<int64_t>>{{e, r, c / 2},
                                                       {e, r, c / kQuantBlock}};
}

std::vector<paddle::DataType> StackInferDtype(paddle::DataType input_dtype,
                                              bool transpose) {
  return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> WeightedSwigluInferShape(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> probs_shape,
    double clamp_value) {
  const int64_t hidden = input_shape[1] / 2;
  return {{input_shape[0], hidden}, {input_shape[0], hidden / kQuantBlock}};
}

std::vector<paddle::DataType> WeightedSwigluInferDtype(
    paddle::DataType input_dtype, paddle::DataType probs_dtype) {
  return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> DequantInferShape(
    std::vector<int64_t> input_shape, std::vector<int64_t> scale_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> DequantInferDtype(paddle::DataType input_dtype,
                                                paddle::DataType scale_dtype) {
  return {paddle::DataType::BFLOAT16};
}

PD_BUILD_OP(w4a8_quantize_1x32)
    .Inputs({"input"})
    .Outputs({"output", "scale"})
    .Attrs({"quant_kind: int64_t"})
    .SetKernelFn(PD_KERNEL(W4A8Quantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(QuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(QuantInferDtype));

PD_BUILD_OP(w4a8_stack_quantize_1x32)
    .Inputs({"input"})
    .Outputs({"output", "scale"})
    .Attrs({"transpose: bool"})
    .SetKernelFn(PD_KERNEL(W4A8StackQuantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(StackInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(StackInferDtype));

PD_BUILD_OP(w4a8_weighted_swiglu_quantize_1x32)
    .Inputs({"input", "probs"})
    .Outputs({"output", "scale"})
    .Attrs({"clamp_value: double"})
    .SetKernelFn(PD_KERNEL(W4A8WeightedSwigluQuantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightedSwigluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightedSwigluInferDtype));

PD_BUILD_OP(w4a8_dequantize_1x32)
    .Inputs({"input", "scale"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(W4A8Dequantize1x32Impl))
    .SetInferShapeFn(PD_INFER_SHAPE(DequantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DequantInferDtype));
