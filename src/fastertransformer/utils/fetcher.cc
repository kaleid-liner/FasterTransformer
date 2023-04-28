#include "fetcher.h"


#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/utils/cuda_utils.h"


namespace fastertransformer {

// the linker asks me to do so

template class FetcherContext<float, float>;
template class FetcherContext<float, half>;

template class FetcherContext<half, float>;
template class FetcherContext<half, half>;

template class FetcherContext<cutlass::uint4b_t, float>;
template class FetcherContext<cutlass::uint4b_t, half>;
template class FetcherContext<uint8_t, float>;
template class FetcherContext<uint8_t, half>;

#ifdef ENABLE_BF16
template class FetcherContext<float, __nv_bfloat16>;
template class FetcherContext<half, __nv_bfloat16>;

template class FetcherContext<__nv_bfloat16, float>;
template class FetcherContext<__nv_bfloat16, half>;
template class FetcherContext<__nv_bfloat16, __nv_bfloat16>;


template class FetcherContext<cutlass::uint4b_t, __nv_bfloat16>;
template class FetcherContext<uint8_t, __nv_bfloat16>;
#endif


// 1. copy to expert_for_source_row_fetching
// 2. calc expert_sparse_idx_working
// 3. launch fetch on the stream, from source to working
template<class WeightT, class BiasT> 
void FetcherContext<WeightT, BiasT>::fetch(int *next_expert_for_source_row, size_t num_rows) {
    if (this->has_source == false) return;
    if (this->num_rows != num_rows) {
        FT_LOG_ERROR("num_rows mismatch: %d vs %d", this->num_rows, num_rows);
        exit(0);
    }
    FT_LOG_TRACE("fetching");
    // print all src
    FT_LOG_TRACE("intermediate_w_src: %p", this->intermediate_w_src);
    FT_LOG_TRACE("output_w_src: %p", this->output_w_src);
    FT_LOG_TRACE("intermediate_bias_src: %p", this->intermediate_bias_src);
    FT_LOG_TRACE("intermediate_w_size_per_expert: %d", this->intermediate_w_size_per_expert);
    FT_LOG_TRACE("output_w_size_per_expert: %d", this->output_w_size_per_expert);
    FT_LOG_TRACE("intermediate_b_size_per_expert: %d", this->intermediate_b_size_per_expert);
    FT_LOG_TRACE("num_experts: %d", this->num_experts);
    FT_LOG_TRACE("num_rows: %d", this->num_rows);
    // print next_expert_for_source_row
    // printMatrix(next_expert_for_source_row, 1, this->num_rows, this->num_rows, true);

    this->first_time = false;
    {   // sparse indexing (CPU version)
        // copy all experts to the CPU buffer
        check_cuda_error(cudaMemcpy(this->row_expert_sorting_buffer, 
            next_expert_for_source_row, sizeof(int) * this->num_rows, cudaMemcpyDeviceToHost));

        FT_LOG_DEBUG("fetch milestone 0");
        std::sort(this->row_expert_sorting_buffer, this->row_expert_sorting_buffer + this->num_rows);
        // remove duplicates
        int *end = std::unique(this->row_expert_sorting_buffer, this->row_expert_sorting_buffer + this->num_rows);

        this->active_experts_count = end - this->row_expert_sorting_buffer;

        memset(this->expert_sparse_idx_cpu, 0, sizeof(int) * this->num_rows);
        for (int i = 0; i < this->active_experts_count; i++) {
            this->expert_sparse_idx_cpu[this->row_expert_sorting_buffer[i]] = i;
        }

        FT_LOG_DEBUG("fetch milestone 1");
        // copy to GPU
        check_cuda_error(cudaMemcpy(this->expert_sparse_idx_working, 
            this->expert_sparse_idx_cpu, sizeof(int) * this->num_experts, cudaMemcpyHostToDevice));
            
        FT_LOG_DEBUG("fetch milestone 2");
    }
 

    // copy to expert_for_source_row_fetching, for next layer to get
    check_cuda_error(cudaMemcpyAsync(this->expert_for_source_row_fetching, 
        next_expert_for_source_row, sizeof(int) * this->num_rows, cudaMemcpyDeviceToDevice, this->stream));

    #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 0");
    #endif


    // launch fetch on the stream, from source to working
    for(int i = 0; i < this->active_experts_count; i++) {
        int expert = this->row_expert_sorting_buffer[i];

        // copy three things
        // 1. output_w
        // 2. intermediate_w
        // 3. intermediate_bias
        check_cuda_error(cudaMemcpyAsync(this->output_working + i * this->output_w_size_per_expert, 
            this->output_w_src + expert * this->output_w_size_per_expert, 
            sizeof(WeightT) * this->output_w_size_per_expert, cudaMemcpyHostToDevice, this->stream));

        #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 1");
        #endif

        check_cuda_error(cudaMemcpyAsync(this->intermediate_working + i * this->intermediate_w_size_per_expert, 
            this->intermediate_w_src + expert * this->intermediate_w_size_per_expert, 
            sizeof(WeightT) * this->intermediate_w_size_per_expert, cudaMemcpyHostToDevice, this->stream));
        
        #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 2");
        #endif
        
        check_cuda_error(cudaMemcpyAsync(this->intermediate_bias_working + i * this->intermediate_b_size_per_expert,
            this->intermediate_bias_src + expert * this->intermediate_b_size_per_expert, 
            sizeof(BiasT) * this->intermediate_b_size_per_expert, cudaMemcpyDeviceToDevice, this->stream));
        
        #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 3");
        #endif
    }
}

// finish previous job
// update dst from working (swap them)
// update expert_sparse_idx from expert_sparse_idx_working (swap them)
template<class WeightT, class BiasT> 
void FetcherContext<WeightT, BiasT>::sync() {
    // sync the steam
    FT_LOG_TRACE("sync stream\n");
    check_cuda_error(cudaStreamSynchronize(this->stream));
    FT_LOG_TRACE("sync end");

    // update dst from working (swap them)
    std::swap(this->output_dst, this->output_working);
    std::swap(this->intermediate_dst, this->intermediate_working);
    std::swap(this->intermediate_bias_dst, this->intermediate_bias_working);
}

// called in FfnLayer.cc
template<class WeightT, class BiasT> 
void FetcherContext<WeightT, BiasT>::set_source(const FfnWeight<WeightT, BiasT> *w_of_the_layer_to_load) {
    if (w_of_the_layer_to_load == nullptr) {
        this->output_w_src = nullptr;
        this->intermediate_w_src = nullptr;
        this->intermediate_bias_src = nullptr;
        this->has_source = false;
        return;
    }
    this->has_source = true;
    this->output_w_src = w_of_the_layer_to_load->intermediate_weight.kernel;
    this->intermediate_w_src = w_of_the_layer_to_load->output_weight.kernel;
    this->intermediate_bias_src = w_of_the_layer_to_load->intermediate_weight.bias;
}

template<class WeightT, class BiasT> 
FetcherContext<WeightT, BiasT>::~FetcherContext() {
    this->freeBuffer();
    //destroy the stream
    check_cuda_error(cudaStreamDestroy(this->stream));
}

template<class WeightT, class BiasT> 
FetcherContext<WeightT, BiasT>::FetcherContext(int mode, int num_experts, 
    size_t intermediate_w_size_per_expert, size_t output_w_size_per_expert, size_t intermediate_b_size_per_expert) 
    :   mode (mode),
        first_time (true),
        num_experts (num_experts),
        intermediate_w_size_per_expert (intermediate_w_size_per_expert),
        output_w_size_per_expert (output_w_size_per_expert),
        intermediate_b_size_per_expert (intermediate_b_size_per_expert) {
    
    // create cuda stream
    check_cuda_error(cudaStreamCreate(&this->stream));
}


template<class WeightT, class BiasT> 
void FetcherContext<WeightT, BiasT>::allocateBuffer(IAllocator* allocator, size_t num_rows) {
    this->num_rows = num_rows;
    // allocated all buffer
    this->expert_for_source_row_fetching = (int*)
        allocator->reMalloc(&this->expert_for_source_row_fetching, sizeof(int) * num_rows);
    this->expert_sparse_idx = (int*)allocator->reMalloc(this->expert_sparse_idx, sizeof(int) * num_experts);
    this->expert_sparse_idx_working = (int*)allocator->reMalloc(this->expert_sparse_idx_working, sizeof(int) * num_experts);
    this->output_dst = (WeightT*)allocator->reMalloc(this->output_dst, sizeof(WeightT) * output_w_size_per_expert * num_experts);
    this->intermediate_dst = (WeightT*)allocator->reMalloc(this->intermediate_dst, sizeof(WeightT) * intermediate_w_size_per_expert * num_experts);
    this->intermediate_bias_dst = (BiasT*)allocator->reMalloc(this->intermediate_bias_dst, sizeof(BiasT) * intermediate_b_size_per_expert * num_experts);
    this->output_working = (WeightT*)allocator->reMalloc(this->output_working, sizeof(WeightT) * output_w_size_per_expert * num_experts);
    this->intermediate_working = (WeightT*)allocator->reMalloc(this->intermediate_working, sizeof(WeightT) * intermediate_w_size_per_expert * num_experts);
    this->intermediate_bias_working = (BiasT*)allocator->reMalloc(this->intermediate_bias_working, sizeof(BiasT) * intermediate_b_size_per_expert * num_experts);
    this->row_expert_sorting_buffer = (int*)allocator->reMalloc(this->row_expert_sorting_buffer, sizeof(int) * num_rows, false, true);
    this->expert_sparse_idx_cpu = (int*)allocator->reMalloc(this->expert_sparse_idx_cpu, sizeof(int) * num_experts, false, true);
    this->allocator = allocator;
}


template<class WeightT, class BiasT> 
void FetcherContext<WeightT, BiasT>::freeBuffer() {
    FT_LOG_DEBUG("fetcher context free buffer");
    // free all buffer
    // print all buffers to be freed
    FT_LOG_DEBUG("free expert_for_source_row_fetching: %p", this->expert_for_source_row_fetching);
    FT_LOG_DEBUG("free expert_sparse_idx: %p", this->expert_sparse_idx);
    FT_LOG_DEBUG("free expert_sparse_idx_working: %p", this->expert_sparse_idx_working);
    FT_LOG_DEBUG("free output_dst: %p", this->output_dst);
    FT_LOG_DEBUG("free intermediate_dst: %p", this->intermediate_dst);
    FT_LOG_DEBUG("free intermediate_bias_dst: %p", this->intermediate_bias_dst);
    FT_LOG_DEBUG("free output_working: %p", this->output_working);
    FT_LOG_DEBUG("free intermediate_working: %p", this->intermediate_working);
    FT_LOG_DEBUG("free intermediate_bias_working: %p", this->intermediate_bias_working);
    FT_LOG_DEBUG("free row_expert_sorting_buffer: %p", this->row_expert_sorting_buffer);
    FT_LOG_DEBUG("free expert_sparse_idx_cpu: %p", this->expert_sparse_idx_cpu);

    allocator->free((void**)&this->expert_for_source_row_fetching);
    allocator->free((void**)&this->expert_sparse_idx);
    allocator->free((void**)&this->expert_sparse_idx_working);
    allocator->free((void**)&this->output_dst);
    allocator->free((void**)&this->intermediate_dst);
    allocator->free((void**)&this->intermediate_bias_dst);
    allocator->free((void**)&this->output_working);
    allocator->free((void**)&this->intermediate_working);
    allocator->free((void**)&this->intermediate_bias_working);
    allocator->free((void**)&this->row_expert_sorting_buffer, true);
    allocator->free((void**)&this->expert_sparse_idx_cpu, true);
    FT_LOG_DEBUG("fetcher context free buffer finished");
}

} // namespace fastertransformer