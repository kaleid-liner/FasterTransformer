#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/utils/allocator.h"

namespace fastertransformer {

# define GPU_ONLY 0
# define FETCH_ON_DEMAND 1
# define PREFETCH 2


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
template<class WeightT, class BiasT = WeightT>
class FetcherContext {
private:
    WeightT *intermediate_working;          // on-GPU
    WeightT *output_working;                // on-GPU
    BiasT *intermediate_bias_working;       // on-GPU
    int *expert_sparse_idx_working;         // on-GPU

    int *row_expert_sorting_buffer;         // on-CPU
    int *expert_sparse_idx_cpu;             // on-CPU

    size_t intermediate_w_size_per_expert;
    size_t output_w_size_per_expert;
    size_t intermediate_b_size_per_expert;
    size_t num_rows;
    size_t num_experts;

    std::vector<void*> big_buffer_on_device;

    bool has_source = false;
    bool buffer_allocated = false;
    IAllocator* allocator;
public:
    cudaStream_t stream;
    int mode; // 1: FETCH_ON_DEMAND
              // 2: PREFETCH
              // it doesn't affect the functionality, just a signal.

    bool first_time = true; // for prefetch mode

    // the source we use to launch next fetch
    const WeightT *intermediate_w_src;       // on cpu
    const WeightT *output_w_src;             // on cpu
    const BiasT *intermediate_bias_src;      // on GPU

    // working and dst for the next layer use
    int *expert_for_source_row_fetching; // on-GPU buffer

    // dst space (things we need to use after sync)
    int *expert_sparse_idx;          // on-GPU
    int active_experts_count;        // on CPU
    
    WeightT *intermediate_dst;        // on-GPU buffer
    WeightT *output_dst;              // on-GPU buffer
    BiasT *intermediate_bias_dst;   // on-GPU buffer

    // 1. copy to expert_for_source_row_fetching
    // 2. calc expert_sparse_idx_working
    // 3. launch fetch on the stream, from source to working
    void fetch(int *next_expert_for_source_row, size_t num_rows);

    // finish previous job
    // drop all previous dst space things and update them.
    void sync(); 

    // called in FfnLayer.cc
    void set_source(const FfnWeight<WeightT, BiasT> *w_of_the_layer_to_load);

    FetcherContext(int mode, int num_experts, 
        size_t intermediate_w_size_per_expert, size_t output_w_size_per_expert, size_t intermediate_b_size_per_expert);
    ~FetcherContext();

    void allocateBuffer(IAllocator* allocator, size_t num_rows);
    void freeBuffer();
    
    void* mallocOnDeviceAligned(size_t size);
};



} // namespace fastertransformer