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

/* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved. */

#include <cassert>
#include <vector>

#include <cuda_bf16.h>
#include <cuda.h>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/common/float8_e4m3fn.h"

#include <thrust/sort.h>

#include "fleety_utils.h"
#include "fused_moe_op.h"
#include "fused_moe_bwd_op.h"
#include "moe_kernel_impl.h"

#include <Python.h>

static PyMethodDef module_methods[] = {{NULL, NULL, 0, NULL}};

static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                       "moe_ops_fp8",
                                       NULL,
                                       -1,
                                       module_methods};

PyMODINIT_FUNC PyInit_moe_ops_fp8(void) {
  return PyModule_Create(&module_def);
}

constexpr int64_t TileSize = 128;
constexpr int64_t WarpSize = 32;

#define CHECK_CUDA(x) PD_CHECK(!x.is_cpu(), #x " must be a CUDA tensor")
#define DEFAULT_THROW(NAME, TYPE)                           \
  default:                                                  \
    do                                                      \
    {                                                       \
      PD_THROW(#NAME, " not implemented for '", TYPE, "'"); \
    } while (0);                                            \
    break


#define LAUNCH_KERNEL(ELEMENTS_PER_THREAD, THREADS, POWER2SCALING) \
  initialize_moe_routing_kernel<ELEMENTS_PER_THREAD, THREADS, POWER2SCALING> \
      <<<blocks, THREADS, 0, stream>>>( \
          unpermuted_input, \
          permuted_output, \
          scale_out, \
          expanded_dest_row_to_expanded_source_row, \
          expanded_source_row_to_expanded_dest_row, \
          permuted_experts, \
          expert_offset, \
          combine_weights, \
          num_rows, \
          cols, \
          k, \
          capacity, \
          num_experts, \
          use_pad \
      )

static inline size_t AlignTo16(const size_t &input)
{
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

template <typename KeyT>
size_t getWorkspaceSize(const int num_rows,
                        const int hidden_size,
                        const int inter_size,
                        const int num_experts,
                        const int k,
                        phi::CubKeyValueSorter &sorter)
{

  const int num_moe_inputs = AlignTo16(k * num_rows);
  int num_softmax_outs = 0;
  size_t total_ws_bytes = 4 * num_moe_inputs * sizeof(int); // source_rows_, permuted_rows_, permuted_experts_

  const int sorter_ws_size_bytes = AlignTo16(sorter.getWorkspaceSize(k * num_rows));
  total_ws_bytes += sorter_ws_size_bytes; // intermediate (fc1) output + cub sorting workspace
  return total_ws_bytes;
}

template <bool Power2Scaling>
__device__ __forceinline__ float ScaleWrapper(const float amax,
                                              const float eps=0.f) {
  constexpr float fp8_max = 448.0f;
  float amax_mod = fmaxf(amax, eps);
  if (amax_mod == 0.f) {
    return 1.f;
  }
  float scale = fp8_max / amax_mod;

  if (isinf(scale)) {
    if constexpr (Power2Scaling) {
      return 0x1.0p127;
    } else {
      return 0x1.FEp127;
    }
  }

  if (scale == 0.0) {
    return scale;
  }
  if constexpr (Power2Scaling) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
    uint8_t exp = scale_bits >> 23;
    int32_t normal_biased_exp = static_cast<int32_t>(exp) - 127;
    __builtin_assume(exp != 0);
    scale = ldexpf(1.0f, normal_biased_exp);
  }
  return scale;
}
template<int VecSize, bool Power2Scaling>
__device__ void ComputeScaleAndWrite(
  phi::dtype::bfloat16 *data_ptr,
  float *scale,
  float *scale_out,
  int64_t local_scale_id,
  int64_t dest_scale_row,
  int64_t dest_scale_col,
  int64_t scale_row_num,
  int64_t scale_col_num
) {
  __nv_bfloat16 *data = reinterpret_cast<__nv_bfloat16*>(data_ptr);
  // step1: local_max
  __nv_bfloat16 local_max = __float2bfloat16(-INFINITY); // Initialize to -inf
  for (int i = 0; i < VecSize; ++i) {
    __nv_bfloat16 val = __habs(data[i]);
    if (__hgt(val, local_max)) local_max = val;
  }
 
  // step2: reduce per TileSize
  // 目的是: 每个线程持有VecSize个元素, warp内部, 在连续的TileSize个元素中求max
  // 做法是: 通过分组shfl进行规约
  static_assert(VecSize >= 4);  // 因为不想跨warp规约, 所以每个tile只能出现在一个warp内。但一个warp不一定只处理128个元素。
  __nv_bfloat16 global_max = local_max;
  static_assert(TileSize >= VecSize && TileSize % VecSize == 0);
  constexpr int group_size = TileSize / VecSize;  // 每个group处理一个tile
  int lane_id = threadIdx.x % WarpSize;
  int group_id = lane_id / group_size;
  int group_lane = lane_id % group_size;
  unsigned mask = (1u << ((group_id + 1) * group_size)) - (1u << (group_id * group_size));

  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    __nv_bfloat16 other = __shfl_down_sync(mask, global_max, stride);
    if (__hgt(other, global_max)) global_max = other;
  }

  // step3: write scale to smem
  if (group_lane == 0){
    float scale_value = ScaleWrapper<Power2Scaling>(__bfloat162float(global_max));
    scale[local_scale_id] = scale_value;  // to smem
    // CUBLAS需要在这里做了转置, CUTLASS不需要
    // scale_out[dest_scale_col * scale_row_num + dest_scale_row] = 1.0f / scale_value; // to gmem
    scale_out[dest_scale_row * scale_col_num + dest_scale_col] = 1.0f / scale_value; // to gmem
  }
  __syncthreads();
}

template<int VecSize>
__device__ void ApplyScale(const phi::dtype::bfloat16 *src_ptr, phi::dtype::float8_e4m3fn *dest_ptr, const float *scale, int64_t scale_id) {
  const __nv_bfloat16 *data = reinterpret_cast<const __nv_bfloat16*>(src_ptr);
  float scale_value = scale[scale_id];
  for (int64_t i = 0; i < VecSize; i++) {
    dest_ptr[i] = static_cast<__nv_fp8_e4m3>(static_cast<float>(data[i]) * scale_value);
  }
}

template<int VecSize, int ThreadNum, bool Power2Scaling>
__global__ void initialize_moe_routing_kernel(const phi::dtype::bfloat16*   unpermuted_input,
                                              phi::dtype::float8_e4m3fn*    permuted_output,
                                              float* scale_out,
                                              const int* expanded_dest_row_to_expanded_source_row,
                                              int*       expanded_source_row_to_expanded_dest_row,
                                              const int* permuted_experts,
                                              const int64_t* expert_offset,
                                              float* combine_weights, //output
                                              const int  num_rows,
                                              const int  cols,
                                              const int  k,
                                              const int64_t capacity,
                                              const int64_t num_experts,
                                              bool use_pad
                                              )
{
    static_assert(VecSize * ThreadNum % TileSize == 0);
    static_assert(VecSize <= TileSize);

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    using LoadT = phi::AlignedVector<phi::dtype::bfloat16, VecSize>;
    LoadT src_vec;    
    using StoreT = phi::AlignedVector<phi::dtype::float8_e4m3fn, VecSize>;
    StoreT dest_vec;
  
    const int64_t expanded_dest_row   = blockIdx.x;
    const int64_t expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    const int64_t iexpert = permuted_experts[expanded_dest_row];
    const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
    const int64_t row_in_expert = expanded_dest_row - offset;
    if (row_in_expert >= capacity){
        if (threadIdx.x == 0) {
            expanded_source_row_to_expanded_dest_row[expanded_source_row] = 0; // unset scatter-idx
            auto ik = expanded_source_row / num_rows;
            auto isent = expanded_source_row % num_rows; // transpose
            combine_weights[isent * k + ik] = 0.f; //unset combine-weight            
        }
        return;
    }
    int64_t num_padded = 0;
    if (threadIdx.x == 0) {
        if (use_pad)
            num_padded = iexpert * capacity - offset;
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row + num_padded;
    }
    // Duplicate and permute rows
    const int64_t source_row = expanded_source_row % num_rows;

    const phi::dtype::bfloat16* source_row_ptr = unpermuted_input + source_row * cols;
    phi::dtype::float8_e4m3fn* dest_row_ptr;
    float *scale_out_prt;
    int64_t dest_row;
    int64_t dest_row_num;
    if (use_pad){
      dest_row = iexpert * capacity + row_in_expert;
      dest_row_num = num_experts * capacity;
    }else{
      dest_row = expanded_dest_row;
      dest_row_num = num_rows * k;
    }
    dest_row_ptr = permuted_output + dest_row * cols;

    __shared__ float scale[ThreadNum * VecSize / TileSize];

    for (int64_t element_id = threadIdx.x * VecSize; element_id < cols; element_id += blockDim.x* VecSize) {
        // 每个线程读VecSize个元素, 一共读取了ThreadNum*VecSize个元素
        // 注意: 一个线程最多只能计算一个scale, 即一个线程最多处理TileSize个元素
        phi::Load<phi::dtype::bfloat16, VecSize>(&source_row_ptr[element_id], &src_vec);

        int64_t local_scale_id = VecSize * threadIdx.x / TileSize;

        ComputeScaleAndWrite<VecSize, Power2Scaling>(
          src_vec.val, scale, scale_out, local_scale_id, dest_row, element_id / TileSize, dest_row_num, cols / TileSize
        );
        ApplyScale<VecSize>(src_vec.val, dest_vec.val, scale, local_scale_id);

        phi::Store<phi::dtype::float8_e4m3fn, VecSize>(dest_vec, &dest_row_ptr[element_id]);
    }
}

void initialize_moe_routing_kernelLauncher(const phi::dtype::bfloat16*  unpermuted_input,
                                           phi::dtype::float8_e4m3fn*  permuted_output,
                                           float* scale_out,
                                           const int*   expanded_dest_row_to_expanded_source_row,
                                           int*         expanded_source_row_to_expanded_dest_row,
                                           const int*   permuted_experts,
                                           const int64_t* expert_offset,
                                           float* combine_weights, //output
                                           const int    num_rows,
                                           const int    cols,
                                           const int    k,
                                           const int64_t  capacity,
                                           const int64_t num_experts,
                                           bool use_pad,
                                           bool use_pow2_scale,
                                           cudaStream_t stream)
{   
    const int blocks  = num_rows * k;
    if (use_pow2_scale) {
      if (cols % 2048 == 0) {
        constexpr int threads = 256;
        LAUNCH_KERNEL(8, threads, true);
      } else if (cols % 1024 == 0) {
        constexpr int threads = 256;
        LAUNCH_KERNEL(4, threads, true);
      } else if (cols % 512 == 0) {
        constexpr int threads = 128;
        LAUNCH_KERNEL(4, threads, true);
      } else if (cols % 256 == 0) {
        constexpr int threads = 64;
        LAUNCH_KERNEL(4, threads, true);
      } else if (cols % 128 == 0) {
        constexpr int threads = 32;
        LAUNCH_KERNEL(4, threads, true);
      } else {
        assert(0);
      }
    } else {
      if (cols % 2048 == 0) {
        constexpr int threads = 256;
        LAUNCH_KERNEL(8, threads, false);
      } else if (cols % 1024 == 0) {
        constexpr int threads = 256;
        LAUNCH_KERNEL(4, threads, false);
      } else if (cols % 512 == 0) {
        constexpr int threads = 128;
        LAUNCH_KERNEL(4, threads, false);
      } else if (cols % 256 == 0) {
        constexpr int threads = 64;
        LAUNCH_KERNEL(4, threads, false);
      } else if (cols % 128 == 0) {
        constexpr int threads = 32;
        LAUNCH_KERNEL(4, threads, false);
      } else {
        assert(0);
      }
    }
}

void apply_moe_dispatch_fwd(
    const phi::dtype::bfloat16 *x,
    const float *gate_logits,
    const float *corr_bias,
    int64_t num_rows,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t capacity,
    int64_t k,
    phi::dtype::float8_e4m3fn *out_fp8,
    float *out_scale,
    float *combine_weights,
    int *scatter_index,
    int64_t *expert_offset,
    int *expert_id,
    bool use_pad,
    bool use_all2all_permute,
    int64_t world_size,
    int64_t num_local_experts,
    bool use_pow2_scale,
    cudaStream_t stream,
    const phi::Place &place)
{
  phi::CubKeyValueSorter sorter(stream);

  paddle::Tensor expanded_source_row_to_expanded_dest_row_tensor =
      paddle::empty({num_rows, k}, paddle::DataType::INT32, place);

  paddle::Tensor active_cnt_tensor = paddle::empty({1}, paddle::DataType::INT32, place);

  int64_t bytes = getWorkspaceSize<phi::dtype::bfloat16>(num_rows,
                                      hidden_size, // hidden-size=0
                                      0,           // inter-size=0
                                      num_experts,
                                      k,
                                      sorter);

  paddle::Tensor ws_ptr_tensor = paddle::empty({bytes}, paddle::DataType::INT8, place);
  int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();

  // Pointers
  int *source_rows_;
  int *permuted_rows_;
  int *permuted_experts_;
  int *expert_id_;

  float *softmax_out_;
  phi::dtype::bfloat16 *fc1_result_;

  const int sorter_ws_size_bytes = AlignTo16(sorter.getWorkspaceSize(k * num_rows));

  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(k * num_rows);

  source_rows_ = reinterpret_cast<int *>(ws_ptr);
  permuted_rows_ = source_rows_ + num_moe_inputs;
  permuted_experts_ = permuted_rows_ + num_moe_inputs;
  expert_id_ = permuted_experts_ + num_moe_inputs;

  fc1_result_ = reinterpret_cast<phi::dtype::bfloat16 *>(expert_id_ + num_moe_inputs);
  softmax_out_ = nullptr;

  topk_gating_softmax_kernelLauncher<float>(gate_logits, 
                                            corr_bias,
                                            combine_weights, // output
                                            softmax_out_,    // no use
                                            expert_id,       // output
                                            source_rows_,    // output
                                            num_rows,
                                            num_experts,
                                            k,
                                            stream);

  // modifiy expert-id according to k
  if (use_pad) // 为了区分 k=1 选择和 k=2 选择，修改 expert-id
    modify_expert_id_launcher(expert_id, expert_id_, k, num_rows, num_experts, stream);

  sorter.run(fc1_result_,
             sorter_ws_size_bytes,
             use_pad ? expert_id_ : expert_id,         // key in
             permuted_experts_, // key out // [num_row, k]: expert-id
             source_rows_,      // value in
             permuted_rows_,    // value out //[num_row, k]: id在原 activation 中的位置
             k * num_rows,      // num_rows
             false,
             stream);
    
  if (use_pad)  
    unmodify_expert_id_launcher(permuted_experts_, permuted_experts_, k, num_rows, num_experts, stream);

  compute_total_rows_before_expert(
      permuted_experts_,
      k * num_rows,
      num_experts,
      expert_offset,
      stream);

  
  initialize_moe_routing_kernelLauncher(x,
                                        out_fp8,
                                        out_scale,
                                        permuted_rows_,
                                        scatter_index,
                                        permuted_experts_,
                                        expert_offset,
                                        combine_weights,                                        
                                        static_cast<int>(num_rows),
                                        static_cast<int>(hidden_size),
                                        static_cast<int>(k),                                        
                                        capacity,
                                        num_experts,
                                        use_pad,
                                        use_pow2_scale,
                                        stream);

  return;
}

std::vector<paddle::Tensor> MoEDispatchFwd(const paddle::Tensor &x,
                                           const paddle::Tensor &gate_logits,
                                           const paddle::optional<paddle::Tensor> &corr_bias,
                                           int64_t k,
                                           int64_t capacity,
                                           bool use_pad,
                                           bool use_pow2_scale)
{

  const auto &x_shape = x.shape();
  const auto &gate_logits_shape = gate_logits.shape();

  PD_CHECK(x_shape.size() == 2);
  PD_CHECK(gate_logits_shape.size() == 2);

  int64_t num_rows = x_shape[0];
  int64_t hidden_size = x_shape[1];
  int64_t num_experts = gate_logits_shape[1];
  PD_CHECK(num_rows == gate_logits_shape[0]);
  CHECK_CUDA(x);
  PD_CHECK(num_experts >= k);

  PD_CHECK(gate_logits.type() == paddle::DataType::FLOAT32);
  if (corr_bias){
    PD_CHECK(corr_bias.get().dtype() == paddle::DataType::FLOAT32);
    PD_CHECK(corr_bias.get().shape().size() == 1);
    PD_CHECK(corr_bias.get().shape()[0] == num_experts);
  }

  std::vector<int64_t> out_shape;
  std::vector<int64_t> scale_shape;
  PD_CHECK(x_shape[1] >= TileSize && x_shape[1] % TileSize == 0);
  int64_t numel = 1;
  if (use_pad) {
    out_shape = {num_experts * capacity, x_shape[1]} ;
    numel = num_experts * capacity * x_shape[1];
    scale_shape = {num_experts * capacity, x_shape[1] / TileSize} ;
  } else {
    out_shape = {num_rows * k, x_shape[1]};
    numel = num_rows * k * x_shape[1];
    scale_shape = {num_rows * k, x_shape[1] / TileSize} ;
  }

  auto place = x.place();
  paddle::Tensor out_fp8 = paddle::empty(out_shape, paddle::DataType::FLOAT8_E4M3FN, place);
  cudaMemsetAsync(reinterpret_cast<void *>(out_fp8.data<phi::dtype::float8_e4m3fn>()),
                  0,
                  sizeof(phi::dtype::float8_e4m3fn) * numel,
                  x.stream());
  paddle::Tensor scale = paddle::ones(scale_shape, paddle::DataType::FLOAT32, place);
  paddle::Tensor combine_weights = paddle::empty({num_rows, k}, paddle::DataType::FLOAT32, place);
  paddle::Tensor scatter_index = paddle::empty({k, num_rows}, paddle::DataType::INT32, place);
  paddle::Tensor expert_offset = paddle::empty({num_experts}, paddle::DataType::INT64, place);
  paddle::Tensor expert_id = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);

  apply_moe_dispatch_fwd(
          x.data<phi::dtype::bfloat16>(),
          gate_logits.data<float>(),
          corr_bias? corr_bias.get().data<float>() : nullptr,
          num_rows,
          num_experts,
          hidden_size,
          capacity,
          k,
          const_cast<phi::dtype::float8_e4m3fn *>(out_fp8.data<phi::dtype::float8_e4m3fn>()),
          const_cast<float *>(scale.data<float>()),
          const_cast<float *>(combine_weights.data<float>()),
          const_cast<int *>(scatter_index.data<int>()),
          const_cast<int64_t *>(expert_offset.data<int64_t>()),
          const_cast<int *>(expert_id.data<int>()),
          use_pad,
          false,    // use_all2all_permute
          -1,       // world_size
          -1,       // num_local_experts
          use_pow2_scale,
          x.stream(),
          x.place());

  return {out_fp8, scale, combine_weights, scatter_index, expert_offset, expert_id};
}

PD_BUILD_OP(moe_gate_dispatch_and_quant1)
    .Inputs({"x", "gate_logtis", paddle::Optional("corr_bias")})
    .Outputs({"out_fp8", "scale", "combine_weights", "scatter_index", "expert_offset", "expert_id"})
    .Attrs({"k: int64_t", "capacity: int64_t", "use_pad: bool", "use_pow2_scale: bool"})
    .SetKernelFn(PD_KERNEL(MoEDispatchFwd));
