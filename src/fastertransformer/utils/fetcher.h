#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <future>
#include "cuda_utils.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/utils/allocator.h"
#include "arena.h"
#include "src/fastertransformer/utils/config.h"

namespace fastertransformer {


// there are two things we need to fetch: intermediate weights and output weights
// the workflow:
// ffn_layer calls set_source(layer_2)
// (specially replaced it with set_source(layer_1) for the first time, and do fetching)
// ffn_layer calls sync()             sycn and swap workingspace and dst space
//                                    we start to use dst space now

// ffn_layer calls fetch()            start a new fetching to the working space
// ffn_layer calls set_source(layer_x)
// ffn_layer calls sync()             
// ffn_layer calls fetch()
// ...
template<class ActT, class WeightT = ActT, class BiasT = ActT>
class FetcherContext {
    // TODO: Refactor naming
private:
    WeightT *intermediate_working;          // on-GPU
    WeightT *output_working;                // on-GPU
    BiasT *intermediate_bias_working;       // on-GPU
    BiasT *intermediate_scale_working;
    BiasT *output_scale_working;
    int *expert_sparse_idx_working;         // on-GPU

    int *row_expert_sorting_buffer;         // on-CPU
    int *expert_sparse_idx_cpu;             // on-CPU

    size_t intermediate_w_size_per_expert_;
    size_t output_w_size_per_expert_;
    size_t intermediate_b_size_per_expert_;
    size_t intermediate_scale_size_per_expert_;
    size_t output_scale_size_per_expert_;
    size_t weight_size_per_expert_;

    size_t num_rows;
    size_t num_experts;

    std::vector<void*> big_buffer_on_device;

    bool buffer_allocated = false;
    IAllocator* allocator;

    std::string next_layer_name_;
    std::string current_layer_name_;
    std::vector<std::future<void>> futures_;

    const char* next_weight_src_;
    const char* current_weight_src_;

public:
    cudaStream_t stream;
    FetchType mode; // 1: FETCH_ON_DEMAND
              // 2: PREFETCH
              // it doesn't affect the functionality, just a signal.

    bool first_time;  // first layer should fetch itself and prefetch next layer
    bool last_time;  //last layer should not prefetch

    // working and dst for the next layer use
    int *expert_for_source_row_fetching; // on-GPU buffer

    // dst space (things we need to use after sync)
    int *expert_sparse_idx;          // on-GPU
    int active_experts_count;        // on CPU
    
    WeightT *intermediate_dst;        // on-GPU buffer
    WeightT *output_dst;              // on-GPU buffer
    BiasT *intermediate_bias_dst;   // on-GPU buffer
    BiasT *intermediate_scale_dst;
    BiasT *output_scale_dst;

    // 1. copy to expert_for_source_row_fetching
    // 2. calc expert_sparse_idx_working
    // 3. launch fetch on the stream, from source to working
    void fetch(int *next_expert_for_source_row, size_t num_rows, bool prefetch);

    // finish previous job
    // drop all previous dst space things and update them.
    void sync(); 

    // called in FfnLayer.cc
    void set_source(const char* next_weight_src, const char* current_weight_src);

    void set_layer(const std::string& next_layer_name, const std::string& current_layer_name, bool is_first_moe, bool is_last_moe);

    FetcherContext(
        FetchType mode, int num_experts, 
        size_t intermediate_w_size_per_expert, size_t output_w_size_per_expert, 
        size_t intermediate_b_size_per_expert,
        size_t intermediate_scale_size_per_expert, size_t output_scale_size_per_expert,
        size_t arena_size);
    ~FetcherContext();

    void allocateBuffer(IAllocator* allocator, size_t num_rows);
    void freeBuffer();
    
    void* mallocOnDeviceAligned(size_t size);

    using tag_t = typename GroupedMemoryArena::tag_t;
};



} // namespace fastertransformer