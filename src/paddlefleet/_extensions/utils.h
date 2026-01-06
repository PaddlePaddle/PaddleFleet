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
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/core/utils/data_type.h"
#ifdef __CUDACC__
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#endif

template <paddle::DataType DType>
struct TypeMap;
template <>
struct TypeMap<paddle::DataType::BFLOAT16> {
  using type = phi::bfloat16;
};
template <>
struct TypeMap<paddle::DataType::FLOAT16> {
  using type = phi::float16;
};
template <>
struct TypeMap<paddle::DataType::FLOAT32> {
  using type = float;
};
template <>
struct TypeMap<paddle::DataType::INT32> {
  using type = int;
};
template <>
struct TypeMap<paddle::DataType::INT64> {
  using type = int64_t;
};

inline paddle::DataType TransToDataType(int64_t dtype) {
  return phi::TransToPhiDataType(dtype);
}

inline int LimitGridDim(int64_t n) {
  return static_cast<int>(std::min<int64_t>(n, 1024 * 1024));
}

#ifdef __CUDACC__
template <typename T>
T** GetTensorDevicePtrs(const std::vector<paddle::Tensor>& tensors,
                        paddle::Tensor* ptr_tensor,
                        cudaStream_t stream,
                        phi::Place place) {
  auto nbytes = tensors.size() * sizeof(T*);
  std::vector<const T*> cpu_ptrs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    cpu_ptrs[i] = tensors[i].data<T>();
  }
  *ptr_tensor = paddle::empty(
      {static_cast<int64_t>(nbytes)}, paddle::DataType::UINT8, place);
  auto* device_ptrs = reinterpret_cast<T**>(ptr_tensor->data());
  auto err = cudaMemcpyAsync(
      device_ptrs, cpu_ptrs.data(), nbytes, cudaMemcpyHostToDevice, stream);
  PD_CHECK(
      err == cudaSuccess, "cudaMemcpyAsync error", cudaGetErrorString(err));
  err = cudaStreamSynchronize(stream);
  PD_CHECK(err == cudaSuccess,
           "cudaStreamSynchronize error",
           cudaGetErrorString(err));
  return device_ptrs;
}
#endif

template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

#ifdef __CUDACC__
template <>
struct alignas(16) VectorType<float, 4> {
  float4 data;  // Built-in CUDA vector type
};

template <>
struct alignas(16) VectorType<__nv_bfloat16, 8> {
  __nv_bfloat16 data[8];
};

template <>
struct alignas(16) VectorType<__nv_fp8_e4m3, 16> {
  __nv_fp8_e4m3 data[16];
};
#endif

template <>
struct alignas(16) VectorType<uint8_t, 16> {
  uint8_t data[16];
};

#ifdef __CUDACC__
template <typename T>
__device__ __forceinline__ void unrolled_memcpy(const T* src,
                                                T* dst,
                                                const int num_elements) {
#pragma unroll
  for (int idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    dst[idx] = src[idx];
  }
}

// Helper function to perform vectorized memory copy
template <typename T>
__device__ __forceinline__ void vectorized_memcpy(const T* src,
                                                  T* dst,
                                                  const int num_elements) {
  constexpr int vector_size_in_bytes = 16;
  const int elements_per_vector = vector_size_in_bytes / sizeof(T);

  int num_vectors = num_elements / elements_per_vector;
  int remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  const VecType* src_vec = reinterpret_cast<const VecType*>(src);
  VecType* dst_vec = reinterpret_cast<VecType*>(dst);

#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = src_vec[idx];
  }

  if (remaining_elements > 0) {
    int offset = num_vectors * elements_per_vector;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] = src[offset + i];
    }
  }
}

template <typename T>
__device__ __forceinline__ void try_vectorized_memcpy(const T* src,
                                                      T* dst,
                                                      const int num_elements) {
  bool is_aligned_128bit =
      ((uintptr_t)src & 0xF) == 0 && ((uintptr_t)dst & 0xF) == 0;
  if (is_aligned_128bit) {
    vectorized_memcpy(src, dst, num_elements);
  } else {
    unrolled_memcpy(src, dst, num_elements);
  }
}
#endif

#define PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, __max_num_experts, ...) \
  if (__num_expert <= __max_num_experts) {                               \
    constexpr auto MAX_NUM_EXPERTS_C = __max_num_experts;                \
    do {                                                                 \
      __VA_ARGS__();                                                     \
    } while (0);                                                         \
    break;                                                               \
  }

#define PD_SWITCH_NUM_EXPERTS(__num_experts_expr, ...)                        \
  do {                                                                        \
    auto __num_expert = (__num_experts_expr);                                 \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 8, __VA_ARGS__);                 \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 16, __VA_ARGS__);                \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 32, __VA_ARGS__);                \
    PD_SWITCH_NUM_EXPERTS_IMPL(__num_expert, 64, __VA_ARGS__);                \
    PD_THROW("Unsupported expert number %d", static_cast<int>(__num_expert)); \
  } while (0)

#define DISPATCH_BOOL(condition, ConstName, ...) \
  {                                              \
    if (condition) {                             \
      constexpr bool ConstName = true;           \
      {                                          \
        __VA_ARGS__                              \
      }                                          \
    } else {                                     \
      constexpr bool ConstName = false;          \
      {                                          \
        __VA_ARGS__                              \
      }                                          \
    }                                            \
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define BF16_MAX(a, b) __hmax(a, b)
#define BF16_ABS(x) __habs(x)
#else
#define BF16_MAX(a, b) \
  __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)))
#define BF16_ABS(x) __float2bfloat16(fabsf(__bfloat162float(x)))
#endif

template <typename T>
struct FastDivMod {
  T d_;

  __host__ __device__ FastDivMod(T d) : d_(d) {}

  __host__ __device__ T Div(T n) const { return n / d_; }
};

template <typename T>
struct F8LimitsTrait;

template <>
struct F8LimitsTrait<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
};
// paddle::DataType::FLOAT8_E4M3FN maps to phi::float8_e4m3fn which is usually
// mapped to __nv_fp8_e4m3 on CUDA

template <typename T, bool ForcePow2>
struct HighPrecisionFloatScaleLimitsTrait;

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, false> {
  static constexpr float max = 3.402823466e+38f;  // FLT_MAX
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, true> {
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<__nv_bfloat16, false> {
  static constexpr float max = 0x1.FEp127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<__nv_bfloat16, true> {
  static constexpr float max = 0x1.0p127;
};

template <typename IType, typename OType, bool Power2Scaling = false>
__device__ __forceinline__ float ComputeScaleImpl(const float amax,
                                                  const float eps) {
  constexpr float fp8_max = F8LimitsTrait<OType>::max;
  float amax_mod = fmaxf(amax, eps);
  if (amax_mod == 0.f) {
    return 1.f;
  }
  float scale = fp8_max / amax_mod;

  if (isinf(scale)) {
    return HighPrecisionFloatScaleLimitsTrait<IType, Power2Scaling>::max;
  }
  if (scale == 0.0) {
    return scale;
  }
  if constexpr (Power2Scaling) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&scale);
    uint8_t exp = scale_bits >> 23;
    int32_t normal_biased_exp = static_cast<int32_t>(exp) - 127;
    // __builtin_assume(exp != 0);
    scale = ldexpf(1.0f, normal_biased_exp);
  }
  return scale;
}

template <bool Power2Scaling>
__device__ __forceinline__ float RoundPower2Scale(float scale) {
#ifdef __CUDA_ARCH__
  return __CUDA_ARCH__ != 900 && Power2Scaling &&
                 (scale == static_cast<float>(0x1.0p127))
             ? static_cast<float>(1.0f)
             : scale;
#else
  return scale;
#endif
}

template <typename IType, typename OType, bool Power2Scaling = false>
__device__ __forceinline__ float ComputeScale(const float amax,
                                              const float eps) {
  return RoundPower2Scale<Power2Scaling>(
      ComputeScaleImpl<IType, OType, Power2Scaling>(amax, eps));
}
