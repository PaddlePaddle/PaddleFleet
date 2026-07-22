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

#include <paddle/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kFp4GroupSize = 32;
constexpr float kFp4Max = 6.0f;

enum class Mxfp4FakeQuantMode : int {
  kIndexerLegacy = 0,
  kIndexerOfficial = 1,
  kExpertWeight = 2,
};

template <typename T>
struct PaddleType;

template <>
struct PaddleType<float> {
  using type = float;
};

template <>
struct PaddleType<half> {
  using type = paddle::float16;
};

template <>
struct PaddleType<__nv_bfloat16> {
  using type = paddle::bfloat16;
};

__device__ __forceinline__ float CeilPow2Ue8m0(float value) {
  int bits = __float_as_int(value);
  int exponent = (bits >> 23) & 0xFF;
  exponent += ((bits & 0x7FFFFF) != 0) ? 1 : 0;
  exponent = max(1, min(254, exponent));
  return __int_as_float(exponent << 23);
}
__device__ __forceinline__ float QuantizeE2m1(float value) {
  const float absolute = fabsf(value);
  const int index = (absolute > 0.25f) + (absolute > 0.75f) +
                    (absolute > 1.25f) + (absolute > 1.75f) +
                    (absolute > 2.5f) + (absolute > 3.5f) + (absolute > 5.0f);
  constexpr float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return copysignf(grid[index], value);
}

__device__ __forceinline__ float GroupScale(float amax,
                                            Mxfp4FakeQuantMode mode) {
  float raw_scale;
  if (mode == Mxfp4FakeQuantMode::kIndexerLegacy) {
    raw_scale = fmaxf(amax / kFp4Max, 1e-4f);
  } else if (mode == Mxfp4FakeQuantMode::kIndexerOfficial) {
    raw_scale = fmaxf(amax / kFp4Max, 0x1p-126f);
  } else {
    raw_scale = fmaxf(amax / kFp4Max, 0x1p-126f);
  }
  return CeilPow2Ue8m0(raw_scale);
}

template <typename T>
__device__ __forceinline__ float LoadValue(const T* input, int64_t offset) {
  return static_cast<float>(input[offset]);
}

template <typename T>
__global__ void Mxfp4LastAxisKernel(const T* input,
                                    T* output,
                                    int64_t group_count,
                                    Mxfp4FakeQuantMode mode) {
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < group_count;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t base = group * kFp4GroupSize;
    float amax = 0.0f;
#pragma unroll
    for (int index = 0; index < kFp4GroupSize; ++index) {
      amax = fmaxf(amax, fabsf(LoadValue(input, base + index)));
    }
    const float scale = GroupScale(amax, mode);
#pragma unroll
    for (int index = 0; index < kFp4GroupSize; ++index) {
      const int64_t offset = base + index;
      const float value = LoadValue(input, offset);
      output[offset] = static_cast<T>(QuantizeE2m1(value / scale) * scale);
    }
  }
}

template <typename T>
__global__ void Mxfp4ExpertAxisKernel(const T* input,
                                      T* output,
                                      int64_t input_features,
                                      int64_t output_features,
                                      int64_t group_count) {
  const int64_t groups_per_expert = input_features / kFp4GroupSize;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < group_count;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t output_feature = group % output_features;
    const int64_t remaining = group / output_features;
    const int64_t input_group = remaining % groups_per_expert;
    const int64_t expert = remaining / groups_per_expert;
    const int64_t base = expert * input_features * output_features +
                         input_group * kFp4GroupSize * output_features +
                         output_feature;
    float amax = 0.0f;
#pragma unroll
    for (int index = 0; index < kFp4GroupSize; ++index) {
      amax = fmaxf(
          amax,
          fabsf(LoadValue(
              input, base + static_cast<int64_t>(index) * output_features)));
    }
    const float scale = GroupScale(amax, Mxfp4FakeQuantMode::kExpertWeight);
#pragma unroll
    for (int index = 0; index < kFp4GroupSize; ++index) {
      const int64_t offset =
          base + static_cast<int64_t>(index) * output_features;
      const float value = LoadValue(input, offset);
      output[offset] = static_cast<T>(QuantizeE2m1(value / scale) * scale);
    }
  }
}

template <typename T>
void LaunchMxfp4FakeQuant(const paddle::Tensor& input,
                          paddle::Tensor* output,
                          Mxfp4FakeQuantMode mode) {
  if (input.numel() == 0) {
    return;
  }

  constexpr int threads = 256;
  int64_t group_count;
  if (mode == Mxfp4FakeQuantMode::kExpertWeight) {
    const auto shape = input.shape();
    const int64_t experts = shape[0];
    const int64_t input_features = shape[1];
    const int64_t output_features = shape[2];
    group_count = experts * (input_features / kFp4GroupSize) * output_features;
    const int blocks = static_cast<int>(
        std::min<int64_t>((group_count + threads - 1) / threads, 65535));
    Mxfp4ExpertAxisKernel<T><<<blocks, threads, 0, input.stream()>>>(
        reinterpret_cast<const T*>(input.data<typename PaddleType<T>::type>()),
        reinterpret_cast<T*>(output->data<typename PaddleType<T>::type>()),
        input_features,
        output_features,
        group_count);
  } else {
    group_count = input.numel() / kFp4GroupSize;
    const int blocks = static_cast<int>(
        std::min<int64_t>((group_count + threads - 1) / threads, 65535));
    Mxfp4LastAxisKernel<T><<<blocks, threads, 0, input.stream()>>>(
        reinterpret_cast<const T*>(input.data<typename PaddleType<T>::type>()),
        reinterpret_cast<T*>(output->data<typename PaddleType<T>::type>()),
        group_count,
        mode);
  }
}

}  // namespace

std::vector<paddle::Tensor> FusedMxfp4FakeQuant(const paddle::Tensor& input,
                                                int mode) {
  PD_CHECK(input.is_gpu(), "fused_mxfp4_fake_quant requires a GPU tensor");
  PD_CHECK(mode >= static_cast<int>(Mxfp4FakeQuantMode::kIndexerLegacy) &&
               mode <= static_cast<int>(Mxfp4FakeQuantMode::kExpertWeight),
           "fused_mxfp4_fake_quant mode must be 0, 1, or 2");

  const auto shape = input.shape();
  const auto quant_mode = static_cast<Mxfp4FakeQuantMode>(mode);
  if (quant_mode == Mxfp4FakeQuantMode::kExpertWeight) {
    PD_CHECK(shape.size() == 3 && shape[1] % kFp4GroupSize == 0,
             "expert input must be rank 3 with axis 1 divisible by 32");
  } else {
    PD_CHECK(!shape.empty() && shape.back() % kFp4GroupSize == 0,
             "indexer input last dimension must be divisible by 32");
  }

  auto output = paddle::empty(input.shape(), input.dtype(), input.place());
  if (input.numel() == 0) {
    return {output};
  }
  switch (input.dtype()) {
    case paddle::DataType::FLOAT32:
      LaunchMxfp4FakeQuant<float>(input, &output, quant_mode);
      break;
    case paddle::DataType::FLOAT16:
      LaunchMxfp4FakeQuant<half>(input, &output, quant_mode);
      break;
    case paddle::DataType::BFLOAT16:
      LaunchMxfp4FakeQuant<__nv_bfloat16>(input, &output, quant_mode);
      break;
    default:
      PD_CHECK(
          false,
          "fused_mxfp4_fake_quant supports float32, float16, and bfloat16");
  }
  return {output};
}

std::vector<std::vector<int64_t>> FusedMxfp4FakeQuantInferShape(
    const std::vector<int64_t>& input_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> FusedMxfp4FakeQuantInferDtype(
    paddle::DataType input_dtype) {
  return {input_dtype};
}

PD_BUILD_OP(fused_mxfp4_fake_quant)
    .Inputs({"input"})
    .Outputs({"output"})
    .Attrs({"mode: int"})
    .SetKernelFn(PD_KERNEL(FusedMxfp4FakeQuant))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMxfp4FakeQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMxfp4FakeQuantInferDtype));
