/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#pragma GCC diagnostic pop

#include "src/fastertransformer/kernels/moe_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/fetcher.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

namespace fastertransformer {

static constexpr int WARP_SIZE = 32;

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template<typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_softmax(const T* input, const bool* finished, T* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float    threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x]) {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData    = max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0) {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0) {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int   idx = thread_row_offset + ii;
        const float val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx]     = T(val);
    }
}

template<typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_top_k(const T*    inputs_after_softmax,
                                                 const bool* finished,
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

    const bool should_process_row = finished ? !finished[block_row] : true;
    const int  thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key   = 0;
        thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key   = expert;
            inp_kvp.value = inputs_after_softmax[idx];

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
            output[idx]      = result_kvp.value;
            indices[idx]     = should_process_row ? result_kvp.key : num_experts;
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
    const T* input, const bool* finished, T* output, const int num_rows, int* indices, int* source_rows, const int k)
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
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr unsigned long MAX_BYTES_PER_LDG = 16;

    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
    using Constants                    = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT           = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int            num_warps     = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int            num_blocks    = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    // print the args
    FT_LOG_TRACE("topk_gating_softmax_launcher_helper kernel args: input: %x, \
         finished: %x, output: %x, num_rows: %d, indices: %x, source_row: %x, k: %d",
                 input,
                 finished,
                 output,
                 num_rows,
                 indices,
                 source_row,
                 k);
    topk_gating_softmax<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
        <<<num_blocks, block_dim, 0, stream>>>(input, finished, output, num_rows, indices, source_row, k);
    FT_LOG_TRACE("kernel finished");
}

template<typename T>
void topk_gating_softmax_kernelLauncher(const T*     input,
                                        const bool*  finished,
                                        T*           output,
                                        T*           softmax_temp_output,
                                        int*         indices,
                                        int*         source_row,
                                        const int    num_rows,
                                        const int    num_experts,
                                        const int    k,
                                        cudaStream_t stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr int WARPS_PER_TB = 4;

    switch (num_experts) {
        case 2: {
            topk_gating_softmax_launcher_helper<T, 2, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 4: {
            topk_gating_softmax_launcher_helper<T, 4, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 8: {
            topk_gating_softmax_launcher_helper<T, 8, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 16: {
            topk_gating_softmax_launcher_helper<T, 16, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 32: {
            topk_gating_softmax_launcher_helper<T, 32, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 64: {
            topk_gating_softmax_launcher_helper<T, 64, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 128: {
            topk_gating_softmax_launcher_helper<T, 128, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 256: {
            topk_gating_softmax_launcher_helper<T, 256, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        default: {
            static constexpr int TPB = 256;
            FT_CHECK(softmax_temp_output != nullptr);
            moe_softmax<T, TPB><<<num_rows, TPB, 0, stream>>>(input, finished, softmax_temp_output, num_experts);
            moe_top_k<T, TPB><<<num_rows, TPB, 0, stream>>>(
                softmax_temp_output, finished, output, indices, source_row, num_experts, k);
        }
    }
}

// ========================== CUB Sorting things ====================================
CubKeyValueSorter::CubKeyValueSorter(): num_experts_(0), num_bits_(sizeof(int) * 8)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

CubKeyValueSorter::CubKeyValueSorter(const int num_experts):
    num_experts_(num_experts), num_bits_((int)log2(num_experts) + 1)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

void CubKeyValueSorter::update_num_experts(const int num_experts)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    num_experts_ = num_experts;
    num_bits_    = (int)log2(num_experts) + 1;
}

size_t CubKeyValueSorter::getWorkspaceSize(const size_t num_key_value_pairs)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    num_key_value_pairs_    = num_key_value_pairs;
    size_t required_storage = 0;
    int*   null_int         = nullptr;
    cub::DeviceRadixSort::SortPairs(
        NULL, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits_);
    return required_storage;
}

void CubKeyValueSorter::run(void*        workspace,
                            const size_t workspace_size,
                            const int*   keys_in,
                            int*         keys_out,
                            const int*   values_in,
                            int*         values_out,
                            const size_t num_key_value_pairs,
                            cudaStream_t stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs);
    size_t actual_ws_size   = workspace_size;

    if (expected_ws_size > workspace_size) {
        std::stringstream err_ss;
        err_ss << "[FT Error][CubKeyValueSorter::run]\n";
        err_ss << "Error. The allocated workspace is too small to run this problem.\n";
        err_ss << "Expected workspace size of at least " << expected_ws_size << " but got problem size "
               << workspace_size << "\n";
        throw std::runtime_error(err_ss.str());
    }
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ============================== Infer GEMM sizes =================================
__device__ inline int find_total_elts_leq_target(const int* sorted_indices, const int arr_length, const int target)
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

// Sets up the gemm assuming the inputs, experts and outputs are stored in row major order.
// Assumes we want to perform output = matmul(inputs, experts) + bias
__global__ void compute_total_rows_before_expert_kernel(const int*    sorted_experts,
                                                        const int     sorted_experts_len,
                                                        const int64_t num_experts,
                                                        int64_t*      total_rows_before_expert)
{

    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
        return;

    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = find_total_elts_leq_target(sorted_experts, sorted_experts_len, expert);
}

template<typename T, typename WeightType, typename Enable>
CutlassMoeFCRunner<T, WeightType, Enable>::CutlassMoeFCRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T, typename WeightType, typename Enable>
size_t CutlassMoeFCRunner<T, WeightType, Enable>::getWorkspaceSize(
    const int num_rows, const int hidden_size, const int inter_size, const int num_experts, const int k)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int buf_size         = pad_to_multiple_of_16(k * num_rows * hidden_size);
    const int interbuf_size    = pad_to_multiple_of_16(k * num_rows * inter_size);
    const int padded_experts   = pad_to_multiple_of_16(num_experts);
    const int num_moe_inputs   = (128 + pad_to_multiple_of_16(k * num_rows));
    int       num_softmax_outs = 0;

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
        num_softmax_outs = pad_to_multiple_of_16(num_rows * num_experts);
    }

    // softmax output, permuted_rows and permuted_experts have moved to outside of moe kernel, allocate them
    // in Encoder or Decoder before invoking FfnLayer forward.
    size_t total_ws_bytes = 4 * num_moe_inputs * sizeof(int);  // source_rows_, permuted_rows_, permuted_experts_
                        // + expert_for_source_row_backup_
    total_ws_bytes += buf_size * sizeof(T);                    // permuted_data
    total_ws_bytes += padded_experts * sizeof(int64_t);        // Hold total_rows_before_expert_
    total_ws_bytes += num_softmax_outs * sizeof(T);
    const int bytes_for_fc1_result = interbuf_size * sizeof(T);
    const int sorter_ws_size_bytes = pad_to_multiple_of_16(sorter_.getWorkspaceSize(num_rows));
    sorter_.update_num_experts(num_experts);

    int bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
    if (sorter_ws_size_bytes > bytes_for_fc1_result) {
        int remaining_bytes = pad_to_multiple_of_16(sorter_ws_size_bytes - bytes_for_fc1_result);
        bytes_for_intermediate_and_sorting += remaining_bytes;
    }

    total_ws_bytes += bytes_for_intermediate_and_sorting;  // intermediate (fc1) output + cub sorting workspace
    return total_ws_bytes;
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::configure_ws_ptrs(
    char* ws_ptr, const int num_rows, const int hidden_size, const int inter_size, const int num_experts, const int k)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int buf_size       = pad_to_multiple_of_16(k * num_rows * hidden_size);
    const int interbuf_size  = pad_to_multiple_of_16(k * num_rows * inter_size);
    const int padded_experts = pad_to_multiple_of_16(num_experts);
    const int num_moe_inputs = 128 + pad_to_multiple_of_16(k * num_rows);
    // const int num_softmax_outs = pad_to_multiple_of_16(num_rows * num_experts);

    source_rows_      = (int*)ws_ptr;
    permuted_rows_    = source_rows_ + num_moe_inputs;
    expert_for_source_row_backup_ = permuted_rows_ + num_moe_inputs;
    permuted_experts_ = expert_for_source_row_backup_ + num_moe_inputs;
    permuted_data_    = (T*)(permuted_experts_ + num_moe_inputs);

    total_rows_before_expert_ = (int64_t*)(permuted_data_ + buf_size);

    fc1_result_ = (T*)(total_rows_before_expert_ + padded_experts);

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
        softmax_out_ = (T*)(fc1_result_ + interbuf_size);
    }
    else {
        softmax_out_ = nullptr;
    }

    // align expert_for_source_row_backup_ to 128 bytes
    expert_for_source_row_backup_ = (int*)(((uintptr_t)expert_for_source_row_backup_ + 127) & ~127);
}


__global__ void do_mapping(int *expert_for_source_row, int len, int *map) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        expert_for_source_row[idx] = map[expert_for_source_row[idx]];
    }
}

void map_on(int *expert_for_source_row, int len, int *map, cudaStream_t stream) {
    // expert_for_source_row and map is on GPU
    int block_size = 256;
    int grid_size = (len + block_size - 1) / block_size;
    do_mapping<<<grid_size, block_size, 0, stream>>>(expert_for_source_row, len, map);
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::setFetcherContext(
    FetcherContext<WeightType, T> *fetcher_ctx) {

    this->fetcher_context = fetcher_ctx;
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::run_moe_fc(const T*          input_activations,     // [num_rows, hidden_size]
                                                           const T*          gating_output,         // [num_rows, expert_nums]
                                                           const WeightType* fc1_expert_weights,    // [num_experts, hidden_size, inter_size]
                                                           const T*          fc1_scales,            
                                                           const T*          fc1_expert_biases,     // [num_experts, inter_size]
                                                           ActivationType    fc1_activation_type,   
                                                           const WeightType* fc2_expert_weights,    // [num_experts, inter_size, hidden_size]
                                                           const T*          fc2_scales,
                                                           const int         num_rows,              // h_token_num
                                                           const int         hidden_size,
                                                           const int         inter_size,
                                                                 int         num_experts,
                                                           const int         k,
                                                           char*             workspace_ptr,
                                                           T*                fc2_result,            // [num_rows, hidden_size]
                                                           const bool*       finished,
                                                           const int         active_rows,           // num_rows
                                                           T*                expert_scales,
                                                           int*              expanded_source_row_to_expanded_dest_row, // h_token_num, moe_k_
                                                           int*              expert_for_source_row, // h_token_num, moe_k_
                                                           cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr bool scales_required =
        std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
    FT_LOG_TRACE("scales_required: %d", scales_required);
    if (scales_required) {
        if (fc1_scales == nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales expected but scale for first matmul is a null pointer");
        }
        else if (fc2_scales == nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales expected but scale for second matmul is a null pointer");
        }
    }
    else {
        if (fc1_scales != nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales are ignored for fp32/fp16/bf16 but received scale for FC1");
        }
        else if (fc2_scales != nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales are ignored for fp32/fp16/bf16 but received scale for FC2");
        }
    }

    FT_LOG_TRACE("=== milestone run_moe_fc 0 ===");

    // print all the parameters in the list
    FT_LOG_TRACE("=== configure_ws_ptrs ===");
    FT_LOG_TRACE("num_rows: %d", num_rows);
    FT_LOG_TRACE("hidden_size: %d", hidden_size);
    FT_LOG_TRACE("inter_size: %d", inter_size);
    FT_LOG_TRACE("num_experts: %d", num_experts);
    FT_LOG_TRACE("k: %d", k);

    configure_ws_ptrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts, k);

    // print all the parameters in the list
    FT_LOG_TRACE("=== topk_gating_softmax_kernelLauncher ===");
    FT_LOG_TRACE("finished: %x", finished);
    FT_LOG_TRACE("expert_scales: %x", expert_scales);
    FT_LOG_TRACE("num_rows: %d", num_rows);
    FT_LOG_TRACE("num_experts: %d", num_experts);
    FT_LOG_TRACE("k: %d", k);
    FT_LOG_TRACE("expert_for_source_row: %x", expert_for_source_row);
    FT_LOG_TRACE("source_rows_: %x", source_rows_);
    FT_LOG_TRACE("gating_output:");
    // printMatrix((T*)gating_output, min(10, num_rows), num_experts, num_experts, true);
    topk_gating_softmax_kernelLauncher<T>(gating_output,
                                          finished,
                                          expert_scales,  // [num_experts]
                                          softmax_out_,   // k * num_rows * inter_size OR NULL
                                          expert_for_source_row, // [num_rows * k]
                                          source_rows_, // [k * num_rows]
                                          num_rows,
                                          num_experts,
                                          k,
                                          stream);
    
    // sync to get expert_for_source_row
    check_cuda_error(cudaStreamSynchronize(stream));

    // for prefetch
    bool prefetch_enable = false;
    if (fetcher_context != nullptr) {
        FT_LOG_TRACE("begin do fetching");
        if (fetcher_context->mode == PREFETCH) {
            // supopse each time we run topk_gating, we get the same source_rows_.
            prefetch_enable = true;

            if (fetcher_context->first_time == true) {
                // for layer 1, we use the current routing, rather than the prvious one.
                // load layer-1 weights.
                const WeightType *layer_2_inter = fetcher_context->intermediate_w_src;
                const WeightType *layer_2_output = fetcher_context->output_w_src;
                const T *layer_2_bias = fetcher_context->intermediate_bias_src;

                fetcher_context->intermediate_w_src = fc1_expert_weights;
                fetcher_context->output_w_src = fc2_expert_weights;
                fetcher_context->intermediate_bias_src = fc1_expert_biases;
                
                FT_LOG_DEBUG("=== start prefetch layer 1 ===");
                // get time
                auto start = std::chrono::high_resolution_clock::now();
                fetcher_context->fetch(expert_for_source_row, num_rows * k);
                fetcher_context->sync();
                auto end = std::chrono::high_resolution_clock::now();
                extern int64_t layer_1_fetch_time;
                layer_1_fetch_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                FT_LOG_DEBUG("=== prefetch layer 1 end ===");

                // restore the source to layer 2, for next fetch();
                fetcher_context->intermediate_w_src = layer_2_inter;
                fetcher_context->output_w_src = layer_2_output;
                fetcher_context->intermediate_bias_src = layer_2_bias;
            } else {
                fetcher_context->sync();
            }

            
            num_experts = fetcher_context->active_experts_count;
            fc1_expert_weights = fetcher_context->intermediate_dst;
            fc2_expert_weights = fetcher_context->output_dst;
            fc1_expert_biases = fetcher_context->intermediate_bias_dst;
            

            // set the expert_for_source_row points to the last layer's output backup.
            // print all args for debug
            FT_LOG_TRACE("expert_for_source_row_backup_ %x", expert_for_source_row_backup_);
            FT_LOG_TRACE("expert_for_source_row_fetching %x", fetcher_context->expert_for_source_row_fetching);
            FT_LOG_TRACE("sizeof(int) * num_rows * k = %d", sizeof(int) * num_rows * k);

            // get start time
            
            auto start = std::chrono::high_resolution_clock::now();
            check_cuda_error(cudaMemcpyAsync(expert_for_source_row_backup_, 
                fetcher_context->expert_for_source_row_fetching,
                 sizeof(int) * num_rows * k, cudaMemcpyDeviceToDevice, stream));
            //sync
            check_cuda_error(cudaStreamSynchronize(stream));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            extern int64_t expert_for_row_backup_time;
            expert_for_row_backup_time += duration.count();

            fetcher_context->fetch(expert_for_source_row, num_rows * k);
            // now fetcher_context->expert_for_source_row_fetching is new.


            expert_for_source_row = expert_for_source_row_backup_;
            // the original expert_for_source_row is preserved for final_routing.
        } else if (fetcher_context->mode == FETCH_ON_DEMAND){
            // FT_LOG_ERROR("FETCH ON DEMAND is not implemented."); exit(0);
            fetcher_context->fetch(expert_for_source_row, num_rows * k);
            fetcher_context->sync();
            num_experts = fetcher_context->active_experts_count;
            fc1_expert_weights = fetcher_context->intermediate_dst;
            fc2_expert_weights = fetcher_context->output_dst;
            fc1_expert_biases = fetcher_context->intermediate_bias_dst;
        }
        FT_LOG_TRACE("fetching finished");
    }

    // static int cnt = 0;
    // if ((++cnt) % 5 == 0 ) {
    //     FT_LOG_INFO("active_experts_count: %d", num_experts);
    //     FT_LOG_INFO("num_rows: %d", num_rows * k);
    // }

    FT_LOG_TRACE("expert_scales: %x", expert_scales);
    // printMatrix(expert_scales, 1, num_experts, num_experts, true);

    FT_LOG_TRACE("softmax_out_: %x", softmax_out_);
    FT_LOG_TRACE("expert_for_source_row: %x", expert_for_source_row);
    // printMatrix(expert_for_source_row, 1, num_rows * k, num_rows * k, true);

    FT_LOG_TRACE("source_rows_: %x", source_rows_);
    // printMatrix(source_rows_, 1, num_rows * k, num_rows * k, true);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    const int sorter_ws_size_bytes = pad_to_multiple_of_16(sorter_.getWorkspaceSize(k * num_rows));
    // print all the parameters in the list
    FT_LOG_TRACE("=== sorter_.run ===");
    FT_LOG_TRACE("sorter_ws_size_bytes: %d", sorter_ws_size_bytes);
    FT_LOG_TRACE("k * num_rows: %d", k * num_rows);

    sorter_.run((void*)fc1_result_,
                sorter_ws_size_bytes,
                expert_for_source_row,  // [num_rows, k]  input
                permuted_experts_,      // [num_rows * k] output
                source_rows_,           // [k * num_rows] input 
                permuted_rows_,         // [k * num_rows] output
                k * num_rows,
                stream);
    FT_LOG_TRACE("permuted_experts_: %x", permuted_experts_);
    // printMatrix(permuted_experts_, 1, k * num_rows, k * num_rows, true);
    FT_LOG_TRACE("permuted_rows_ / expanded_dest_row_to_expanded_source_row: %x", permuted_rows_);
    // printMatrix(permuted_rows_, 1, k * num_rows, k * num_rows, true);
    
#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    // print all the parameters in the list
    FT_LOG_TRACE("=== initialize_moe_routing_kernelLauncher ===");
    FT_LOG_TRACE("num_rows: %d", num_rows);
    FT_LOG_TRACE("active_rows: %d", active_rows);
    FT_LOG_TRACE("hidden_size: %d", hidden_size);
    FT_LOG_TRACE("k: %d", k);

    initialize_moe_routing_kernelLauncher(input_activations,            // [num_rows, hidden_size] input
                                          permuted_data_,               // [k * num_rows, hidden_size] output
                                          permuted_rows_,               // [k * num_rows]  input expanded_dest_row_to_expanded_source_row
                                          expanded_source_row_to_expanded_dest_row, // [k * num_rows] output
                                          num_rows,
                                          active_rows,
                                          hidden_size,
                                          k,
                                          stream);
    FT_LOG_TRACE("expanded_source_row_to_expanded_dest_row: %x", expanded_source_row_to_expanded_dest_row);
    // printMatrix(expanded_source_row_to_expanded_dest_row, 1, k * num_rows, k * num_rows, true);
    FT_LOG_TRACE("=== milestone run_moe_fc 5 ===");

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    if (prefetch_enable) {
        check_cuda_error(cudaMemcpyAsync(expert_for_source_row_backup_, expert_for_source_row, 
            sizeof(int) * num_rows * k, cudaMemcpyDeviceToDevice, stream));
        // sync
        check_cuda_error(cudaStreamSynchronize(stream));
        
        expert_for_source_row = expert_for_source_row_backup_;

        map_on(expert_for_source_row, num_rows * k, fetcher_context->expert_sparse_idx, stream);
        map_on(permuted_experts_, num_rows * k, fetcher_context->expert_sparse_idx, stream);
    }

    const int expanded_active_expert_rows = k * active_rows;    
    FT_LOG_TRACE("=== compute_total_rows_before_expert ===");
    FT_LOG_TRACE("expanded_active_expert_rows: %d", expanded_active_expert_rows);
    compute_total_rows_before_expert(
        permuted_experts_, expanded_active_expert_rows, num_experts, total_rows_before_expert_, stream);
    FT_LOG_TRACE("total_rows_before_expert_: %x", total_rows_before_expert_);
    // printMatrix((size_t*)total_rows_before_expert_, 1, num_experts, num_experts, true);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    FT_LOG_TRACE("=== milestone run_moe_fc 7 ===");
    moe_gemm_runner_.moe_gemm_bias_act(permuted_data_,              // [k * num_rows, hidden_size] input
                                       fc1_expert_weights,          // [num_experts, hidden_size, inter_size] input
                                       fc1_scales,                  // NULL
                                       fc1_expert_biases,           // [num_experts, inter_size] input
                                       fc1_result_,                 // [k * num_rows, inter_size] output
                                       total_rows_before_expert_,   // [num_experts] input
                                       expanded_active_expert_rows, // = k * num_rows
                                       inter_size,
                                       hidden_size,
                                       num_experts,
                                       fc1_activation_type,
                                       stream);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    FT_LOG_TRACE("=== milestone run_moe_fc 8 ===");
    moe_gemm_runner_.moe_gemm(fc1_result_,                      // [k * num_rows, inter_size] input
                              fc2_expert_weights,               // [num_experts, inter_size, hidden_size] input
                              fc2_scales,                       // NULL
                              fc2_result,                       // [k * num_rows, hidden_size] output
                              total_rows_before_expert_,        // [num_experts] input
                              expanded_active_expert_rows,      // = k * num_rows
                              hidden_size,
                              inter_size,
                              num_experts,
                              stream);
    FT_LOG_TRACE("=== milestone run_moe_fc 9 ===");

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::run_moe_fc(const T*          input_activations,
                                                           const T*          gating_output,
                                                           const WeightType* fc1_expert_weights,
                                                           const T*          fc1_scales,
                                                           const T*          fc1_expert_biases,
                                                           ActivationType    fc1_activation_type,
                                                           const WeightType* fc2_expert_weights,
                                                           const T*          fc2_scales,
                                                           const int         num_rows,
                                                           const int         hidden_size,
                                                           const int         inter_size,
                                                                 int         num_experts,
                                                           const int         k,
                                                           char*             workspace_ptr,
                                                           T*                fc2_result,
                                                           T*                expert_scales,
                                                           int*              expanded_source_row_to_expanded_dest_row,
                                                           int*              expert_for_source_row,
                                                           cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    run_moe_fc(input_activations,
               gating_output,
               fc1_expert_weights,
               fc1_scales,
               fc1_expert_biases,
               fc1_activation_type,
               fc2_expert_weights,
               fc2_scales,
               num_rows,
               hidden_size,
               inter_size,
               num_experts,
               k,
               workspace_ptr,
               fc2_result,
               nullptr,
               num_rows,
               expert_scales,
               expanded_source_row_to_expanded_dest_row,
               expert_for_source_row,
               stream);
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::compute_total_rows_before_expert(const int*   sorted_indices,
                                                                                 const int    total_indices,
                                                                                 const int    num_experts,
                                                                                 int64_t*     total_rows_before_expert,
                                                                                 cudaStream_t stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int threads = std::min(1024, num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;

    compute_total_rows_before_expert_kernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
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

template<typename T>
__global__ void initialize_moe_routing_kernel(const T*   unpermuted_input,
                                              T*         permuted_output,
                                              const int* expanded_dest_row_to_expanded_source_row,
                                              int*       expanded_source_row_to_expanded_dest_row,
                                              const int  num_rows,
                                              const int  active_rows,
                                              const int  cols)
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    const int expanded_dest_row   = blockIdx.x;
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0) {
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row;
    }

    if (blockIdx.x < active_rows) {
        // Duplicate and permute rows
        const int source_row = expanded_source_row % num_rows;

        const T* source_row_ptr = unpermuted_input + source_row * cols;
        T*       dest_row_ptr   = permuted_output + expanded_dest_row * cols;

        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
            dest_row_ptr[tid] = source_row_ptr[tid];
        }
    }
}

template<typename T>
void initialize_moe_routing_kernelLauncher(const T*     unpermuted_input,
                                           T*           permuted_output,
                                           const int*   expanded_dest_row_to_expanded_source_row,
                                           int*         expanded_source_row_to_expanded_dest_row,
                                           const int    num_rows,
                                           const int    active_rows,
                                           const int    cols,
                                           const int    k,
                                           cudaStream_t stream)
{
    const int blocks  = num_rows * k;
    const int threads = std::min(cols, 1024);
    initialize_moe_routing_kernel<T><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                     permuted_output,
                                                                     expanded_dest_row_to_expanded_source_row,
                                                                     expanded_source_row_to_expanded_dest_row,
                                                                     num_rows,
                                                                     k * active_rows,
                                                                     cols);
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
    const T*  skip_1_row_ptr  = skip_1 + original_row * cols;
    const T*  skip_2_row_ptr;
    if (RESIDUAL_NUM == 2) {
        skip_2_row_ptr = skip_2 + original_row * cols;
    }

    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
        T thread_output;
        if (RESIDUAL_NUM == 1) {
            thread_output = skip_1_row_ptr[tid];
        }
        else if (RESIDUAL_NUM == 2) {
            thread_output = skip_1_row_ptr[tid] + skip_2_row_ptr[tid];
        }
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
                                                                      skip,
                                                                      nullptr,
                                                                      bias,
                                                                      scales,
                                                                      expanded_source_row_to_expanded_dest_row,
                                                                      expert_for_source_row,
                                                                      cols,
                                                                      k);
}

template<typename T>
void finalize_moe_routing_kernelLauncher(const T*     expanded_permuted_rows,
                                         T*           reduced_unpermuted_output,
                                         const T*     skip_1,
                                         const T*     skip_2,
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
    if (skip_2 == nullptr) {
        finalize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                                                          reduced_unpermuted_output,
                                                                          skip_1,
                                                                          skip_2,
                                                                          bias,
                                                                          scales,
                                                                          expanded_source_row_to_expanded_dest_row,
                                                                          expert_for_source_row,
                                                                          cols,
                                                                          k);
    }
    else {
        finalize_moe_routing_kernel<T, 2><<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                                                          reduced_unpermuted_output,
                                                                          skip_1,
                                                                          skip_2,
                                                                          bias,
                                                                          scales,
                                                                          expanded_source_row_to_expanded_dest_row,
                                                                          expert_for_source_row,
                                                                          cols,
                                                                          k);
    }
}

// ========================= TopK Softmax specializations ===========================
template void topk_gating_softmax_kernelLauncher(
    const float*, const bool*, float*, float*, int*, int*, const int, const int, const int, cudaStream_t);
template void topk_gating_softmax_kernelLauncher(
    const half*, const bool*, half*, half*, int*, int*, const int, const int, const int, cudaStream_t);

#ifdef ENABLE_BF16
template void topk_gating_softmax_kernelLauncher(const __nv_bfloat16*,
                                                 const bool*,
                                                 __nv_bfloat16*,
                                                 __nv_bfloat16*,
                                                 int*,
                                                 int*,
                                                 const int,
                                                 const int,
                                                 const int,
                                                 cudaStream_t);
#endif

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;

// ===================== Specializations for init routing =========================
template void initialize_moe_routing_kernelLauncher(
    const float*, float*, const int*, int*, const int, const int, const int, const int, cudaStream_t);
template void initialize_moe_routing_kernelLauncher(
    const half*, half*, const int*, int*, const int, const int, const int, const int, cudaStream_t);
#ifdef ENABLE_BF16
template void initialize_moe_routing_kernelLauncher(
    const __nv_bfloat16*, __nv_bfloat16*, const int*, int*, const int, const int, const int, const int, cudaStream_t);
#endif

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
template void finalize_moe_routing_kernelLauncher(const float*,
                                                  float*,
                                                  const float*,
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
                                                  const half*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
#ifdef ENABLE_BF16
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
template void finalize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                  __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
#endif


// __global__ void tag_all_actiavted_experts(
//     int *expert_sparse_idx,
//     const int *expert_for_source_row,
//     const int num_rows) {

//     int row = blockIdx.x;
//     if (row < num_rows) {
//         expert_sparse_idx[expert_for_source_row[row]] = 1;
//     }
// }

// __global__ void prefix_sum_to_get_sparse_index(
//     int *expert_sparse_idx,
//     const int num_experts) {

//     int tid = threadIdx.x;
//     if (0 < tid && tid < num_experts) {
//         expert_sparse_idx[tid] += expert_sparse_idx[tid - 1];
//     }
// }

// void get_expert_sparse_idx_kernelLauncher(
//     int *expert_sparse_idx,
//     const int *expert_for_source_row,
//     const int num_rows,
//     const int num_experts,
//     int *active_expert_count // cpu
//     ) {
    
//     check_cuda_error(cudaMemset(expert_sparse_idx, 0, sizeof(int) * num_experts));
        
//     tag_all_actiavted_experts<<<1, num_rows>>>(
//         expert_sparse_idx, 
//         expert_for_source_row, 
//         num_rows);
//     prefix_sum_to_get_sparse_index<<<1, num_experts>>>
//         (expert_sparse_idx, num_experts);
// }


}  // namespace fastertransformer
