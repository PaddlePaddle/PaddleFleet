// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef _FUSED_MOE_OP_H_
#define _FUSED_MOE_OP_H_


#include "paddle/phi/common/memory_utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h> // 包含常用的 thrust 算法
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/common/enforce.h"

#include "paddle/extension.h"
#include "./moe_kernel_impl.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>
#define WARP_SIZE 32
// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "paddle/phi/backends/gpu/gpu_info.h"
#pragma GCC diagnostic pop

// namespace paddle {
// namespace operators {

template <class Container>
std::string join_strings(const Container& strs, char delim) {
    std::string str;
    size_t i = 0;
    for (auto& elem : strs) {
        if (i > 0) {
          str += delim;
        }

        std::stringstream ss;
        ss << elem;
        str += ss.str();
        ++i;
    }

    return str;
}


template<typename T>
void  print_to_screen(const T* result, const int size)
{
    if (result == nullptr) {
        return;
    }
    T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * size));
    cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        printf("%d, %f\n", i, static_cast<float>(tmp[i]));
    }
    free(tmp);
}


template void  print_to_screen(const float* result, const int size);
template void  print_to_screen(const half* result, const int size);
template void  print_to_screen(const int* result, const int size);
template void  print_to_screen(const int64_t* result, const int size);
template void  print_to_screen(const uint8_t* result, const int size);
template void  print_to_screen(const int8_t* result, const int size);

template<typename T>
void  print_to_screen1(const T* result, const int size, const int len, std::string msg)
{
    if (result == nullptr) {
        return;
    }
    T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * len));
    cudaMemcpy(tmp, result, sizeof(T) * len, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        std::cerr << "[" << msg << "](" << i << "), " << tmp[i] << std::endl;
    }

    // for (int i = len / 2 - size; i < len/2; ++i) {
    //     std::cerr << "[" << msg << "](" << i << "), " << static_cast<float>(tmp[i]) << std::endl;
    // }

    for (int i = len - size; i < len; ++i) {
        std::cerr << "[" << msg << "](" << i << "), " << tmp[i] << std::endl;
    }
    free(tmp);
}

// template<typename T>
// void  print_to_screen_int(const T* result, const int size, const int len)
// {
//     if (result == nullptr) {
//         return;
//     }
//     T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * len));
//     cudaMemcpy(tmp, result, sizeof(T) * len, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < size; ++i) {
//         printf("%d, %d\n", i, static_cast<float>(tmp[i]));
//     }

//     for (int i = len / 2 - size; i < len/2; ++i) {
//         printf("%d, %d\n", i, static_cast<float>(tmp[i]));
//     }

//     for (int i = len - size; i < len; ++i) {
//         printf("%d, %d\n", i, static_cast<float>(tmp[i]));
//     }
//     free(tmp);
// }


template void  print_to_screen1(const float* result, const int size, const int len, std::string);
// template void  print_to_screen1(const half* result, const int size, const int len, std::string);
template void  print_to_screen1(const int* result, const int size, const int len, std::string);
template void  print_to_screen1(const int64_t* result, const int size, const int len, std::string);
// template void  print_to_screen1(const __nv_bfloat16* result, const int size, const int len, std::string);

// template void  print_to_screen_int(const int64_t* result, const int size, const int len);
// template void  print_to_screen_int(const int* result, const int size, const int len);

template<typename T>
__global__ void cal_expert_size_and_filter(
    T* expert_id,
    const int64_t* expert_offset,
    int64_t len,
    int64_t num_experts,
    int64_t capcity,
    int64_t expert_start_index,
    int64_t expert_end_index,
    bool reverse){
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len)
        return;
    int64_t off = reverse? expert_offset[expert_end_index-1] : 0;
    if (reverse){
        for (int64_t i = expert_end_index - 1; i >= expert_start_index; --i){
            if (idx >= expert_offset[i])
                break;
            off = expert_offset[i];
        }
    }else{
        for (int64_t i = expert_start_index; i != expert_end_index; ++i){
            if (idx < expert_offset[i])
                break;
            off = expert_offset[i];
        }
    }
    if (reverse){
        if(((off-1) - idx) >= capcity){
            expert_id[idx] = num_experts;
        }
    }else{
        if ((idx - off) >= capcity){
            expert_id[idx] = num_experts;
        }
    }
}

template<typename T>
void cal_expert_size_and_filter_launcher(T* expert_id,
        const int64_t* expert_offset,
        int64_t len,
        int64_t num_experts,
        int64_t capcity,
        int64_t expert_start_index,
        int64_t expert_end_index,
        bool reverse,
        const cudaStream_t& stream){
    if (len <= 0)
        return;
    const int64_t threads = std::min(static_cast<int64_t>(1024), len);
    const int64_t blocks = (len + threads - 1) / threads;
    cal_expert_size_and_filter<T><<<blocks, threads, 0, stream>>>(
        expert_id,
        expert_offset,
        len,
        num_experts,
        capcity,
        expert_start_index,
        expert_end_index,
        reverse
    );
}


template<typename T>
__global__ void modify_expert_id(const T*  expert_id,
                                T*         expert_id_out,
                                const int k,
                                const int num_rows,
                                const int64_t num_experts){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * num_rows)
        return;
    int ik = idx % k;
    int irow = idx / k;
    // const T mask = (~0) >> (8*sizeof(T)-ik); // 最后 ik 位为 1 其他位为 0
    int mask = ik; // k => 2(11)
    // printf("before: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id[idx], ik);
    int offset = log2(k) + 1;
    expert_id_out[idx] = (expert_id[idx]<<offset) | mask;
    // printf("after: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id_out[idx], ik);
}

template<typename T>
void modify_expert_id_launcher(const T* expert_id,
        T* expert_id_out,
        const int k,
        const int num_rows,
        const int64_t num_experts,
        const cudaStream_t& stream){
    int max = 1024;
    const int threads = std::min(max, num_rows * k);
    const int blocks = (num_rows * k + threads - 1) / threads;

    modify_expert_id<T><<<blocks, threads, 0, stream>>>(
        expert_id,
        expert_id_out,
        k,
        num_rows,
        num_experts
    );
}

template<typename T>
__global__ void modify_and_mask_expert_id(const T*  expert_id,
                                T*         expert_id_out,
                                const int k,
                                const int num_rows,
                                const int num_experts,
                                const int expert_start_index,
                                const int expert_end_index
                            ){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * num_rows)
        return;
    int ik = idx % k;
    int irow = idx / k;
    // const T mask = (~0) >> (8*sizeof(T)-ik); // 最后 ik 位为 1 其他位为 0
    int mask = ik; // k => 2(11)
    // printf("before: idx=%d, expert-id:%d, ik=%d, s=%d, e=%d\n", idx, expert_id[idx], ik, expert_start_index, expert_end_index);
    int offset = log2(k) + 1;
    if (expert_id[idx] < expert_start_index || expert_id[idx] >= expert_end_index){
        expert_id_out[idx] = (num_experts << offset) ; // -1 means
    }else{
        expert_id_out[idx] = (expert_id[idx]<<offset) | mask;
    }
    // printf("after: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id_out[idx], ik);
}

template<typename T>
void modify_and_mask_expert_id_launcher(const T* expert_id,
        T* expert_id_out,
        const int k,
        const int num_rows,
        const int num_experts,
        const int expert_start_index,
        const int expert_end_index,
        const cudaStream_t& stream){
    int max = 1024;
    const int threads = std::min(max, num_rows * k);
    const int blocks = (num_rows * k + threads - 1) / threads;

    modify_and_mask_expert_id<T><<<blocks, threads, 0, stream>>>(
        expert_id,
        expert_id_out,
        k,
        num_rows,
        num_experts,
        expert_start_index,
        expert_end_index
    );
}

template<typename T>
__global__ void
unmodify_expert_id(const T*  expert_id,
                                T*         expert_id_out,
                                const int k,
                                const int num_rows,
                                const int64_t num_experts){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * num_rows)
        return;
    int ik = idx % k;
    int irow = idx / k;
    int offset = log2(k) + 1;
    expert_id_out[idx] = (expert_id[idx]>>offset);
}

template<typename T>
void unmodify_expert_id_launcher(const T* expert_id,
        T* expert_id_out,
        const int k,
        const int num_rows,
        const int64_t num_experts,
        const cudaStream_t& stream){
    int max = 1024;
    const int threads = std::min(max, num_rows * k);
    const int blocks = (num_rows * k + threads - 1) / threads;

    unmodify_expert_id<T><<<blocks, threads, 0, stream>>>(
        expert_id,
        expert_id_out,
        k,
        num_rows,
        num_experts
    );
}

template<typename T>
__global__ void
build_src_row(T* output,
            const int64_t k,
            const int64_t num_rows,
            const int64_t num_experts){
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * num_rows)
        return;
    int64_t ik = idx / num_rows;
    int64_t irow = idx % num_rows;
    output[idx] = static_cast<int>(ik * num_rows + irow);
}

template<typename T>
void build_src_row_launcher(T* output,
        const int64_t k,
        const int64_t num_rows,
        const int64_t num_experts,
        const cudaStream_t& stream){
    int64_t max = 1024;
    const int64_t threads = std::min(max, num_rows * k);
    const int64_t blocks = (num_rows * k + threads - 1) / threads;

    build_src_row<T><<<blocks, threads, 0, stream>>>(
        output,
        k,
        num_rows,
        num_experts
    );
}


template<typename T>
__global__ void pad_data_to_capcity(T* permuted_data,
                      T* padded_out, //output
                      int* expanded_dest_row_to_expanded_source_row,
                      int* expanded_source_row_to_expanded_dest_row, //output, aka scatter-idx
                      float* combine_weights, //output
                      int64_t* expert_offset,
                      const int* permuted_experts,
                      int64_t num_rows,
                      int64_t k,
                      int64_t num_experts,
                      int64_t capacity,
                      int64_t hidden_size,
                      int64_t expert_start_index, // for partial expert-out
                      int64_t num_active
                    ){
    const int64_t irow = blockIdx.x;
    int64_t src_idx = expanded_dest_row_to_expanded_source_row[irow];
    const int64_t iexpert = permuted_experts[irow];

    int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
    T* src_ptr = permuted_data + offset * hidden_size;
    T* tgt_ptr = padded_out + (iexpert - expert_start_index) * capacity * hidden_size;
    int64_t irow_in_expert = irow - offset;

    if (irow_in_expert >= capacity || irow >= num_active){ // out of capacity
        if (threadIdx.x == 0) {
            expanded_source_row_to_expanded_dest_row[src_idx] = 0; // unset scatter-idx
            auto ik = src_idx / num_rows;
            auto isent = src_idx % num_rows; // transpose
            combine_weights[isent * k + ik] = 0.f; //unset combine-weight
        }
        return;
    }
    auto num_padded = (iexpert - expert_start_index) * capacity - offset;
    src_ptr += irow_in_expert * hidden_size;
    tgt_ptr += irow_in_expert * hidden_size;

    for (int tid = threadIdx.x; tid < hidden_size; tid += blockDim.x) {
        tgt_ptr[tid] = src_ptr[tid]; // copy
    }

    if (threadIdx.x == 0) {
        // auto expanded_source_row = expanded_dest_row_to_expanded_source_row[irow];
        expanded_source_row_to_expanded_dest_row[src_idx] = irow + num_padded;
    }
}

template<typename T>
void pad_data_to_capcity_launcher(T* permuted_data,
                      T* y,
                      int* expanded_dest_row_to_expanded_source_row,
                      int* expanded_source_row_to_expanded_dest_row,
                      float* combine_weights,
                      int64_t* expert_offset,
                      int* permuted_exerts,
                      int64_t num_rows,
                      int64_t k,
                      int64_t num_experts,
                      int64_t capacity,
                      int64_t hidden_size,
                      int64_t expert_start_index,
                      int64_t num_active,
                      const cudaStream_t& stream){

    const int blocks  = num_rows * k;
    int64_t max = 1024;
    const int threads = std::min(hidden_size, max);
    pad_data_to_capcity<T><<<blocks, threads, 0, stream>>>(permuted_data,
        y,
        expanded_dest_row_to_expanded_source_row,
        expanded_source_row_to_expanded_dest_row,
        combine_weights,
        expert_offset,
        permuted_exerts,
        num_rows,
        k,
        num_experts,
        capacity,
        hidden_size,
        expert_start_index,
        num_active > 0 ? num_active : k * num_rows
    );
}

template<typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_top_k(const T*    inputs_after_softmax,
                                                 const T*    bias, //bias could be nullptr if not used
                                                 T*          output,
                                                 int*        indices,
                                                 int*        source_rows,
                                                 const int   num_experts,
                                                 const int   k)
{

    using cub_kvp     = cub::KeyValuePair<int, T>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp     thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows  = gridDim.x;
    const int block_row = blockIdx.x;
    const int  thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key   = 0;
        thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key   = expert;
            inp_kvp.value = bias ? inputs_after_softmax[idx] + bias[expert] : inputs_after_softmax[idx] ;

            for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            const int idx    = k * block_row + k_idx;
            output[idx]      = bias ? inputs_after_softmax[thread_read_offset + result_kvp.key]: result_kvp.value;
            indices[idx]     = result_kvp.key;
            source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/

template<typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topk_gating_softmax(
    const T* input,
    const bool* finished,
    T* output,
    const int num_rows,
    int* indices,
    int* source_rows,
    const int k)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG    = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW    = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD  = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA  = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row         = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
        return;
    const bool should_process_row = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx         = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const T*  thread_read_ptr          = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    cutlass::Array<T, VPT> row_chunk_input;
    AccessType*            row_chunk_vec_ptr   = reinterpret_cast<AccessType*>(&row_chunk_input);
    const AccessType*      vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    using ComputeType = float;
    using Converter   = cutlass::NumericArrayConverter<ComputeType, T, VPT>;
    Converter                        compute_type_converter;
    cutlass::Array<ComputeType, VPT> row_chunk = compute_type_converter(row_chunk_input);

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    ComputeType thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.​
    int                  start_col          = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    // T denorm = T(0.0);
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int   expert  = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val) {
                    max_val = val;
                    expert  = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int   other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert  = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0) {
            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            // denorm = denorm + T(max_val);
            const int idx    = k * thread_row + k_idx;
            output[idx]      = T(max_val);
            indices[idx]     = should_process_row ? expert : NUM_EXPERTS;
            source_rows[idx] = k_idx * num_rows + thread_row;
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k) {
            const int ldg_group_for_expert     = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group) {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = ComputeType(-10000.f);
            }
        }
    }
    // for (int k_idx = 0; k_idx < k; ++k_idx) {
    //     const int idx    = k * thread_row + k_idx;
    //     output[idx]      = output[idx] / denorm;
    // }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at compile time.
template<typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT             = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP   = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template<typename T, int EXPERTS, int WARPS_PER_TB>
void topk_gating_softmax_launcher_helper(const T*     input,
                                         const bool*  finished,
                                         T*           output,
                                         int*         indices,
                                         int*         source_row,
                                         const int    num_rows,
                                         const int    num_experts,
                                         const int    k,
                                         cudaStream_t stream)
{
    static constexpr unsigned long MAX_BYTES_PER_LDG = 16;
    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
    using Constants                    = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT           = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int            num_warps     = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int            num_blocks    = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topk_gating_softmax<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
        <<<num_blocks, block_dim, 0, stream>>>(input, finished, output, num_rows, indices, source_row, k);
}




template<typename T>
void topk_gating_softmax_kernelLauncher(const T*     input,
                                        const T*     bias,
                                        T*           output,
                                        T*           softmax, //no use
                                        int*         indices,
                                        int*         source_row,
                                        const int    num_rows,
                                        const int    num_experts,
                                        const int    k,
                                        cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4;
    static constexpr int TPB = 256;
    moe_top_k<T, TPB><<<num_rows, TPB, 0, stream>>>(
        input, bias, output, indices, source_row, num_experts, k);
}

template<typename T, int VecSize>
__global__ void initialize_moe_routing_old_kernel(const T*   unpermuted_input,
                                              T*         permuted_output,
                                              const int* expanded_dest_row_to_expanded_source_row,
                                              int*       expanded_source_row_to_expanded_dest_row,
                                              const int  num_rows,
                                              const int  cols,
                                              const int  num_active
                                              )
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    using LoadT = phi::AlignedVector<T, VecSize>;
    LoadT src_vec;
    const int expanded_dest_row   = blockIdx.x;
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0) {
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row;
    }
    if (blockIdx.x >= num_active){
        return;
    }

    // Duplicate and permute rows
    const int source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T*       dest_row_ptr   = permuted_output + expanded_dest_row * cols;

    for (int tid = threadIdx.x * VecSize; tid < cols; tid += blockDim.x* VecSize) {
        phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
        phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
    }
}


// only used in hard gate moe
template<typename T>
void initialize_moe_routing_old_kernelLauncher(const T*     unpermuted_input,
                                           T*           permuted_output,
                                           const int*   expanded_dest_row_to_expanded_source_row,
                                           int*         expanded_source_row_to_expanded_dest_row,
                                           const int    num_rows,
                                           const int    cols,
                                           const int    k,
                                           cudaStream_t stream)
{
    const int blocks  = num_rows * k;
    const int threads = std::min(cols, 1024);
    constexpr int max_pack_size = 16 / sizeof(T);
    if (cols % max_pack_size == 0) {
        initialize_moe_routing_old_kernel<T, max_pack_size><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    num_rows,
                                                                    cols,
                                                                    k * num_rows
                                                                    );
    } else {
        initialize_moe_routing_old_kernel<T, 1><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    num_rows,
                                                                    cols,
                                                                    k * num_rows
                                                                    );

    }
}

// ========================== Permutation things =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

template<typename T, int VecSize>
__global__ void initialize_moe_routing_kernel(const T*   unpermuted_input,
                                              T*         permuted_output,
                                              const int* expanded_dest_row_to_expanded_source_row,
                                              int*       expanded_source_row_to_expanded_dest_row,
                                              const int* permuted_experts,
                                              const int64_t* expert_offset,
                                              float* combine_weights, //output
                                              const int  num_rows,
                                              const int  cols,
                                              const int  k,
                                              const int64_t capacity,
                                              bool use_pad
                                              )
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    using LoadT = phi::AlignedVector<T, VecSize>;
    LoadT src_vec;
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
        // printf("going through: capacity=%lld, num_active=%lld, row=[%d->%d], row-in-expert %lld\n",
        //     capacity,
        //     num_active,
        //     expanded_dest_row, expanded_source_row,
        //     row_in_expert
        // );
        if (use_pad)
            num_padded = iexpert * capacity - offset;
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row + num_padded;
    }
    // Duplicate and permute rows
    const int64_t source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T* dest_row_ptr;
    if (use_pad){
        dest_row_ptr = permuted_output +
                       iexpert * capacity * cols +
                       row_in_expert * cols;
    }else{
        dest_row_ptr = permuted_output + expanded_dest_row * cols;
    }

    for (int64_t tid = threadIdx.x * VecSize; tid < cols; tid += blockDim.x* VecSize) {
        phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
        phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
    }
}

/**
 * 原逻辑的output:
 * R0E0
 * R0E1
 * R1E0
 * R1E1
 *
 * 我们想对all2all和专家gemm做overlap, 所以需要将all2all拆成流水线, 为了便于后续计算, 此kernel的output:
 * R0E0
 * R1E0
 * R0E1
 * R1E1
*/
template<typename T, int VecSize, int LoopSize>
__global__ void initialize_moe_routing_permute_kernel(const T*   unpermuted_input,
                                                        T*         permuted_output,
                                                        const int* expanded_dest_row_to_expanded_source_row,
                                                        int*       expanded_source_row_to_expanded_dest_row,
                                                        const int* permuted_experts,
                                                        const int64_t* expert_offset,
                                                        float* combine_weights, //output
                                                        const int  num_rows,
                                                        const int  cols,
                                                        const int  k,
                                                        const int64_t capacity,
                                                        const int64_t world_size,
                                                        const int64_t num_local_experts
                                              )
{
    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
#pragma unroll
    for (int i = 0; i < LoopSize; i++) {
        using LoadT = phi::AlignedVector<T, VecSize>;
        LoadT src_vec;
        const int expanded_dest_row   = blockIdx.x + i * gridDim.x;
        const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
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
            continue;
        }
        int64_t num_padded = 0;
        if (threadIdx.x == 0) {
            num_padded = iexpert * capacity - offset;
            expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row + num_padded;
        }
        // Duplicate and permute rows
        const int source_row = expanded_source_row % num_rows;

        const T* source_row_ptr = unpermuted_input + source_row * cols;
        T* dest_row_ptr;

        const int64_t irank = iexpert / num_local_experts;
        const int64_t local_iexpert = iexpert % num_local_experts;
        dest_row_ptr = permuted_output + local_iexpert * world_size * capacity * cols + irank * capacity * cols + row_in_expert * cols;

        for (int tid = threadIdx.x * VecSize; tid < cols; tid += blockDim.x * VecSize) {
            phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
            phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
        }
    }
}

template<typename T>
void initialize_moe_routing_permute_kernelLauncher(const T*     unpermuted_input,
                                                    T*           permuted_output,
                                                    const int*   expanded_dest_row_to_expanded_source_row,
                                                    int*         expanded_source_row_to_expanded_dest_row,
                                                    const int*   permuted_experts,
                                                    const int64_t* expert_offset,
                                                    float* combine_weights, //output
                                                    const int    num_rows,
                                                    const int    cols,
                                                    const int    k,
                                                    const int64_t  capacity,
                                                    const int64_t world_size,
                                                    const int64_t  num_local_experts,
                                                    cudaStream_t stream)
{
    const int loop_size = 2;
    const int blocks  = (num_rows * k) / loop_size;
    assert((num_rows * k) % loop_size == 0);
    const int threads = std::min(cols, 1024);
    constexpr int max_pack_size = 16 / sizeof(T);
    if (cols % max_pack_size == 0) {
        initialize_moe_routing_permute_kernel<T, max_pack_size, loop_size><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    world_size,
                                                                    num_local_experts
                                                                    );
    } else {
        initialize_moe_routing_permute_kernel<T, 1, loop_size><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    world_size,
                                                                    num_local_experts
                                                                );
    }
}

template<typename T>
void initialize_moe_routing_kernelLauncher(const T*     unpermuted_input,
                                           T*           permuted_output,
                                           const int*   expanded_dest_row_to_expanded_source_row,
                                           int*         expanded_source_row_to_expanded_dest_row,
                                           const int*   permuted_experts,
                                           const int64_t* expert_offset,
                                           float* combine_weights, //output
                                           const int    num_rows,
                                           const int    cols,
                                           const int    k,
                                           const int64_t  capacity,
                                           bool use_pad,
                                           cudaStream_t stream)
{
    const int blocks  = num_rows * k;
    const int threads = std::min(cols, 1024);
    constexpr int max_pack_size = 16 / sizeof(T);
    if (cols % max_pack_size == 0) {
        initialize_moe_routing_kernel<T, max_pack_size><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    use_pad
                                                                    );
    } else {
        initialize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                    permuted_output,
                                                                    expanded_dest_row_to_expanded_source_row,
                                                                    expanded_source_row_to_expanded_dest_row,
                                                                    permuted_experts,
                                                                    expert_offset,
                                                                    combine_weights,
                                                                    num_rows,
                                                                    cols,
                                                                    k,
                                                                    capacity,
                                                                    use_pad
                                                                );
    }
}


template<typename T, int VecSize>
__global__ void copy_unpermuted_to_permuted_kernel(const T*   unpermuted_input,
                                              T*         permuted_output,
                                              const int*  padded_out_to_unpermuted_input,
                                              const int*  padded_out_to_expanded_input,
                                              int* expanded_input_to_padded_out,
                                              const int64_t padded_len,
                                              const int64_t num_rows,
                                              const int64_t k,
                                              const int64_t cols)
{
    using LoadT = phi::AlignedVector<T, VecSize>;
    LoadT src_vec;
    const int padded_dest_row = blockIdx.x;
    if (padded_out_to_unpermuted_input[padded_dest_row] == num_rows){
        //     padded_out_to_unpermuted_input[padded_dest_row] = -1;
        return; // padded place
    }
    const int source_row = padded_out_to_unpermuted_input[padded_dest_row];
    const int source_row_expanded = padded_out_to_expanded_input[padded_dest_row];
    if (threadIdx.x == 0){
        expanded_input_to_padded_out[source_row_expanded] = padded_dest_row;
    }

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T* padded_dest_row_ptr = permuted_output + padded_dest_row * cols;

    for (int tid = threadIdx.x * VecSize; tid < cols; tid += blockDim.x* VecSize) {
        phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
        phi::Store<T, VecSize>(src_vec, &padded_dest_row_ptr[tid]);
    }
    PADDLE_ENFORCE((padded_dest_row < padded_len)&&(source_row_expanded < num_rows * k),
    "The index is out of bounds, "
    "origin_input[%d] -> distributed_input:[%d], should < [%ld],[%ld] \n",
    source_row_expanded, padded_dest_row,  num_rows*k, padded_len);

    // for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
    //     padded_dest_row_ptr[tid] = source_row_ptr[tid]; // copy
    // }
}

template<typename T>
void copy_unpermuted_to_permuted_kernelLauncher(const T*   unpermuted_input,
                                              T*         permuted_output,
                                              const int* padded_out_to_unpermuted_input,
                                              const int* padded_out_to_expanded_input,
                                              int* expanded_input_to_padded_out,
                                              const int64_t padded_len,
                                              const int64_t num_rows, //unpermuted_input_len
                                              const int64_t k,
                                              const int64_t num_cols,
                                              cudaStream_t stream)
{
    auto blocks  = padded_len;
    auto threads = std::min(num_cols, static_cast<int64_t>(1024));
    constexpr int64_t max_pack_size = 16 / sizeof(T);
    if (num_cols % max_pack_size == 0) {
        copy_unpermuted_to_permuted_kernel<T, max_pack_size><<<blocks, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            padded_out_to_unpermuted_input,
            padded_out_to_expanded_input,
            expanded_input_to_padded_out,
            padded_len,
            num_rows,
            k,
            num_cols);
    }else{
        copy_unpermuted_to_permuted_kernel<T, 1><<<blocks, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            padded_out_to_unpermuted_input,
            padded_out_to_expanded_input,
            expanded_input_to_padded_out,
            padded_len,
            num_rows,
            k,
            num_cols);
    }
}

template<typename T>
__global__ void build_seqsort_kv_pairs_kernel( T*  seqsort_key,
                                    T*           seqsort_value,
                                    const int* expanded_dest_row_to_expanded_source_row,
                                    // int*       expanded_source_row_to_expanded_dest_row,
                                    const int* permuted_experts,
                                    const int64_t* expert_offset,
                                    float* combine_weights, //output
                                    const int  num_rows,
                                    const int  k,
                                    const int64_t  num_active,
                                    const int64_t  capacity,
                                    int64_t expert_start_index,
                                    bool use_pad)
{
    const int expanded_dest_row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (expanded_dest_row >= num_rows * k){
        return;
    }
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    const int64_t iexpert = permuted_experts[expanded_dest_row];
    const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
    const int64_t row_in_expert = expanded_dest_row - offset;
    // printf("DEBUG %d=>%d, num_active=%lld, offset=%lld, cap=%lld \n", expanded_dest_row,  expanded_source_row, num_active, row_in_expert, capacity);
    // 从此以后不会发生截断，后续的 seqsort 也不会截断。
    // printf("expanded_dest_row:%d row_in_expert:%lld capacity:%lld num_active:%lld\n", expanded_dest_row, row_in_expert, capacity, num_active);
    if ((use_pad && row_in_expert >= capacity) || expanded_dest_row >= num_active){
        // expanded_source_row_to_expanded_dest_row[expanded_source_row] = 0; // unset scatter-idx
        auto ik = expanded_source_row / num_rows;
        auto isent = expanded_source_row % num_rows; // transpose
        combine_weights[isent * k + ik] = 0.f; //unset combine-weight
        return;
    }

    // auto num_padded = use_pad ? (iexpert - expert_start_index) * capacity - offset : 0;
    // expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row + num_padded;

    // Duplicate and permute rows
    T source_row = expanded_source_row % num_rows;

    if (use_pad){
        // printf("inner print: k=%d num_row=%d before minus %d\n", k, num_rows, source_row);
        seqsort_key  [(iexpert - expert_start_index) * capacity + row_in_expert] = source_row; // 为保证 padding 位置(0)在最后, 所以对 pos-id 取减去其最大值
        seqsort_value[(iexpert - expert_start_index) * capacity + row_in_expert] = expanded_source_row;
    }else{
        seqsort_key[expanded_dest_row] = source_row;
        seqsort_value[expanded_dest_row] = expanded_source_row;
    }
}



template<typename T>
void build_seqsort_kv_pairs_kernel_launcher(T*           seqsort_key, // 实现初始化为 num-rows，保证 sort 到最后
                                           T*           seqsort_value,
                                           const int*   expanded_dest_row_to_expanded_source_row,
                                        //    int*         expanded_source_row_to_expanded_dest_row,
                                           const int*   permuted_experts,
                                           const int64_t* expert_offset,
                                           float* combine_weights, //output
                                           const int    num_rows,
                                           const int    k,
                                           const int64_t  num_active, // -1 expert pos
                                           const int64_t  capacity,
                                           const int64_t expert_start_index,
                                           bool use_pad,
                                           cudaStream_t stream)
{
    int max = 1024;
    const int threads = std::min(max, num_rows * k);
    const int blocks = (num_rows * k + threads - 1) / threads;
    build_seqsort_kv_pairs_kernel<<<blocks, threads, 0, stream>>>(seqsort_key,
                                                                seqsort_value,
                                                                expanded_dest_row_to_expanded_source_row,
                                                                // expanded_source_row_to_expanded_dest_row,
                                                                permuted_experts,
                                                                expert_offset,
                                                                combine_weights,
                                                                num_rows,
                                                                k,
                                                                num_active,
                                                                capacity,
                                                                expert_start_index,
                                                                use_pad
                                                            );

}

template<typename T>
__global__ void combine_moe_kernel(const T*   x,
                                   const T* combine_weights,
                                   const int*  scatter_index,
                                   T* y,
                                   const int64_t k,
                                   const int64_t seqlen,
                                   const int64_t hidden_size,
                                   const int64_t n)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int64_t row_i = i / hidden_size;
        int64_t slice_i = i - row_i * hidden_size;
        const int * scatter_index_start = scatter_index + row_i * k;
        T* dest_ptr = y + i;
        for (int ki = 0; ki < k; ki ++) {
            // get combine_weights i
            const T* w_ptr = combine_weights + row_i * k + ki;
            const T* x_ptr = x +  static_cast<int64_t>(*(scatter_index_start + ki)) * hidden_size + slice_i;
            *(dest_ptr) += (*w_ptr) * (*x_ptr);
        }
    }
}


template<typename T>
__global__ void combine_moe_bwd_kernel(const T*   x,
                                   const T* combine_weights,
                                   const int*  scatter_index,
                                   const T* grad_y,
                                   T* grad_x,
                                   T* grad_combine_weights_helper,
                                   const int64_t k,
                                   const int64_t seqlen,
                                   const int64_t hidden_size,
                                   const int64_t n)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int64_t row_i = i / hidden_size;
        int64_t slice_i = i - row_i * hidden_size;
        const int* scatter_index_start = scatter_index + row_i * k;
        const T grad_y_i = * (grad_y + i);
        // y [ row_i, slice_i]
        // combine [row_i, k, slice_i]
        int64_t weight_base = row_i * k * hidden_size + slice_i;

        T* grad_cw_ptr = grad_combine_weights_helper + weight_base; // stride hidden_size
        for (int64_t ki = 0; ki < k; ki ++) {
            // get combine_weights i
            int64_t ele_index = static_cast<int64_t>(*(scatter_index_start + ki)) * hidden_size + slice_i;
            const T* w_ptr = combine_weights + row_i * k + ki;
            const T* x_ptr = x + ele_index;
            if ( (* w_ptr) != T(0)){
                *(grad_x + ele_index) = grad_y_i * (* w_ptr);
            }
            *(grad_cw_ptr + ki * hidden_size) = grad_y_i * (*x_ptr);
        }
    }
}


template<typename T>
void combine_moe_kernelLauncher(const T*         x,
                                const T*         combine_weights,
                                const int*   scatter_index,
                                T*               y,
                                const int64_t    k,
                                const int64_t    seqlen,
                                const int64_t    hidden_size,
                                cudaStream_t     stream)
{
    // y is [seqlen, hidden_size]
    // for kk in k:
    //     y[i][j] += x[scatter_index[i][kk]][j] * combine_weights[i][kk]
    const int64_t n = hidden_size * seqlen;

    const int64_t threads = 1024;
    const int64_t blocks =  (n + threads - 1) / threads;

    combine_moe_kernel<T><<<blocks, threads, 0, stream>>>(x,
                      combine_weights,
                      scatter_index,
                      y,
                      k,
                      seqlen,
                      hidden_size,
                      n);
}




template<typename T>
void combine_moe_bwd_kernelLauncher(const T*         x,
                                const T*         combine_weights,
                                const int*   scatter_index,
                                const T*         grad_y,
                                      T*         grad_x,
                                      T*         grad_combine_weights_helper,
                                const int64_t    k,
                                const int64_t    seqlen,
                                const int64_t    hidden_size,
                                cudaStream_t     stream)
{
    // y is [seqlen, hidden_size]
    // for kk in k:
    //     y[i][j] += x[scatter_index[i][kk]][j] * combine_weights[i][kk]

    const int64_t n = hidden_size * seqlen;

    const int64_t threads = 1024;
    const int64_t blocks =  (n + threads - 1) / threads;

    combine_moe_bwd_kernel<T><<<blocks, threads, 0, stream>>>(x,
                      combine_weights,
                      scatter_index,
                      grad_y,
                      grad_x,
                      grad_combine_weights_helper,
                      k,
                      seqlen,
                      hidden_size,
                      n);
}





// ============================== Infer GEMM sizes =================================
template<typename T>
__device__ inline int find_total_elts_leq_target(const T* sorted_indices, const int arr_length, const int target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target) {
            high = mid - 1;
        }
        else {
            low             = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

template<typename T>
__global__ void compute_total_rows_before_expert_kernel(const T*    sorted_experts,
                                                        const int     sorted_experts_len,
                                                        const int64_t num_experts,
                                                        int64_t*      total_rows_before_expert)
{

    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
        return;


    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = find_total_elts_leq_target<T>(sorted_experts, sorted_experts_len, expert);
    // total_rows_before_expert[0] = 0;
    // total_rows_before_expert[1] = 1;
    // if (sorted_experts_len > 3) {
    //     for (int i=0; i<35;i++){
    //         total_rows_before_expert[i] = i;
    //     }
    // }


}

// template<typename T>
// __global__  void compute_expert_offset_kernel(const T* sorted_experts,
//     int64_t* expert_offset,
//     const int64_t len,
//     const int64_t num_experts
// ){
//     auto x = blockIdx.x * blockDim.x + threadIdx.x;
//     if (x >= len)
//         return;
//     auto this_expert = sorted_experts[x];
//     if (this_expert == num_experts)
//         return;

//     auto diff = x == (len - 1)? 1: this_expert - static_cast<T>(sorted_experts[x+1]);
//     if (diff != 0){
//         expert_offset[this_expert] = x+1;
//     }
// }

template<typename T>
void compute_global_expert_offset(const T* expert_id, //[len]
                           T* sort_buffer,  //[len]
                           int64_t* expert_offset,//[num_experts]
                           const int64_t len,
                           const int64_t num_experts,
                           const int64_t capacity,
                           const cudaStream_t& stream,
                           const phi::memory_utils::ThrustAllocator<cudaStream_t>& allocator){
    auto ptr = thrust::device_pointer_cast(expert_id);
    auto outptr = thrust::device_pointer_cast(sort_buffer);
    auto offsetptr = thrust::device_pointer_cast(expert_offset);
    const auto& exec_policy = thrust::cuda::par(allocator).on(stream);
    thrust::copy(exec_policy, ptr, ptr + len, outptr);
    thrust::sort(exec_policy, outptr, outptr + len);
    const int threads = std::min(static_cast<int64_t>(1024), num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;

    compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
        sort_buffer, len, num_experts, expert_offset);
    thrust::adjacent_difference(exec_policy, offsetptr, offsetptr + num_experts, offsetptr);
    // thrust::transform(offsetptr,
    //     offsetptr + num_experts,
    //     thrust::constant_iterator<int64_t>(capacity),
    //     offsetptr,
    //     thrust::minimum<int64_t>()
    // );
}


template<typename T>
void compute_local_expert_offset(const T* sorted_expert_id, //[len]
                           int64_t* expert_offset,//[num_experts]
                           int64_t* expert_num,
                           const int64_t len,
                           const int64_t num_experts,
                           const int64_t capacity,
                           const cudaStream_t& stream,
                           const phi::memory_utils::ThrustAllocator<cudaStream_t>& allocator){
    auto offset_ptr = thrust::device_pointer_cast(expert_offset);
    auto expert_num_ptr = thrust::device_pointer_cast(expert_num);
    const auto& exec_policy = thrust::cuda::par(allocator).on(stream);
    thrust::fill(exec_policy, offset_ptr, offset_ptr + num_experts, static_cast<T>(0));

    const int threads = std::min(static_cast<int64_t>(1024), num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;

    compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
        sorted_expert_id, len, num_experts, expert_offset);
    // 不考虑 capcity 影响
    thrust::adjacent_difference(exec_policy, offset_ptr, offset_ptr + num_experts, expert_num_ptr);

}


template<typename T>
void compute_total_rows_before_expert(const T*   sorted_indices,
                                      const int    total_indices,
                                      const int64_t num_experts,
                                      int64_t*     total_rows_before_expert,
                                      const cudaStream_t& stream)
{
    const int threads = std::min(static_cast<int64_t>(1024), num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;


    compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template<typename T, int RESIDUAL_NUM>
__global__ void finalize_moe_routing_kernel(const T*   expanded_permuted_rows,
                                            T*         reduced_unpermuted_output,
                                            const T*   skip_1,
                                            const T*   skip_2,
                                            const T*   bias,
                                            const T*   scales,
                                            const int* expanded_source_row_to_expanded_dest_row,
                                            const int* expert_for_source_row,
                                            const int  cols,
                                            const int  k)
{

    const int original_row    = blockIdx.x;
    const int num_rows        = gridDim.x;
    T*        reduced_row_ptr = reduced_unpermuted_output + original_row * cols;

    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
        T thread_output = T(0.0);
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int expanded_original_row = original_row + k_idx * num_rows;
            const int expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            const int64_t k_offset                       = original_row * k + k_idx;
            const T       row_scale                      = scales[k_offset];
            const T*      expanded_permuted_rows_row_ptr = expanded_permuted_rows + expanded_permuted_row * cols;

            const int expert_idx = expert_for_source_row[k_offset];
            const T*  bias_ptr   = bias + expert_idx * cols;

            thread_output = thread_output + row_scale * (expanded_permuted_rows_row_ptr[tid] + bias_ptr[tid]);
        }
        reduced_row_ptr[tid] = thread_output;
    }
}



template<typename T>
void finalize_moe_routing_kernelLauncher(const T*     expanded_permuted_rows,
                                         T*           reduced_unpermuted_output,
                                         const T*     skip,
                                         const T*     bias,
                                         const T*     scales,
                                         const int*   expanded_source_row_to_expanded_dest_row,
                                         const int*   expert_for_source_row,
                                         const int    num_rows,
                                         const int    cols,
                                         const int    k,
                                         cudaStream_t stream)
{
    const int blocks  = num_rows;
    const int threads = std::min(cols, 1024);

    finalize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                                                      reduced_unpermuted_output,
                                                                      nullptr,
                                                                      nullptr,
                                                                      bias,
                                                                      scales,
                                                                      expanded_source_row_to_expanded_dest_row,
                                                                      expert_for_source_row,
                                                                      cols,
                                                                      k);

}



// ========================= TopK Softmax specializations ===========================
template void topk_gating_softmax_kernelLauncher(
    const float*, const float*, float*, float*, int*, int*, const int, const int, const int, cudaStream_t);
template void topk_gating_softmax_kernelLauncher(
    const half*, const half*, half*, half*, int*, int*, const int, const int, const int, cudaStream_t);
template void topk_gating_softmax_kernelLauncher(const __nv_bfloat16*,
                                                 const __nv_bfloat16*,
                                                 __nv_bfloat16*,
                                                 __nv_bfloat16*,
                                                 int*,
                                                 int*,
                                                 const int,
                                                 const int,
                                                 const int,
                                                 cudaStream_t);
// ===================== Specializations for init routing =========================
template void initialize_moe_routing_kernelLauncher(
    const float*, float*,
    const int*,
    int*,
    const int*,
    const int64_t*,
    float*,
    const int,
    const int,
    const int,
    const int64_t,
    bool,
    cudaStream_t);
template void initialize_moe_routing_kernelLauncher(
    const half*, half*,
    const int*,
    int*,
    const int*,
    const int64_t*,
    float*,
    const int,
    const int,
    const int,
    const int64_t,
    bool,
    cudaStream_t);
template void initialize_moe_routing_kernelLauncher(
    const __nv_bfloat16*, __nv_bfloat16*,
    const int*,
    int*,
    const int*,
    const int64_t*,
    float*,
    const int,
    const int,
    const int,
    const int64_t,
    bool,
    cudaStream_t);
// ==================== Specializations for final routing ===================================
template void finalize_moe_routing_kernelLauncher(const float*,
                                                  float*,
                                                  const float*,
                                                  const float*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const half*,
                                                  half*,
                                                  const half*,
                                                  const half*,
                                                  const half*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                  __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);

// ===================== Specializations for copy unpermuted =========================
template void copy_unpermuted_to_permuted_kernelLauncher(
    const float*, float* ,
    const int*, const int*, int*,
    const int64_t, const int64_t, const int64_t, const int64_t,
    cudaStream_t);

template void copy_unpermuted_to_permuted_kernelLauncher(
    const half*, half* ,
    const int*, const int*, int*,
    const int64_t, const int64_t, const int64_t, const int64_t,
    cudaStream_t);

template void copy_unpermuted_to_permuted_kernelLauncher(
    const __nv_bfloat16*, __nv_bfloat16* ,
    const int*, const int*, int*,
    const int64_t, const int64_t, const int64_t, const int64_t,

    cudaStream_t);
// ===================== Specializations for init routing =========================
template void initialize_moe_routing_old_kernelLauncher(
    const float*, float*, const int*, int*, const int, const int, const int,  cudaStream_t);
template void initialize_moe_routing_old_kernelLauncher(
    const half*, half*, const int*, int*, const int, const int, const int, cudaStream_t);
template void initialize_moe_routing_old_kernelLauncher(
    const __nv_bfloat16*, __nv_bfloat16*, const int*, int*, const int, const int, const int, cudaStream_t);

// template void compute_total_rows_before_expert(int*,
//                                       half*,
//                                       const int,
//                                       const int,
//                                       int64_t*,
//                                       cudaStream_t stream);


// }  // namespace operators
// }  // namespace paddle

#endif
