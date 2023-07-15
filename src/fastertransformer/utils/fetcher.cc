#include "fetcher.h"


#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/profiling.h"
#include "src/fastertransformer/utils/config.h"
#include "src/fastertransformer/utils/random.h"
#include <chrono>


namespace fastertransformer {

// namespace {
// MemoryArena::tag_t getTagForExpert(size_t layer_index, size_t expert_index, size_t expert_num, size_t type, size_t type_num, size_t base)
// {
//     return (layer_index * expert_num + expert_index) * type_num + type + base;
// }
// } // namespace

// the linker asks me to do so

template class FetcherContext<half, cutlass::fp4_t>;

template class FetcherContext<float, cutlass:uint4b_t>;
template class FetcherContext<half, cutlass::uint4b_t>;
template class FetcherContext<float, uint8_t>;
template class FetcherContext<half, uint8_t>;

#ifdef ENABLE_BF16
template class FetcherContext<float, __nv_bfloat16>;
template class FetcherContext<half, __nv_bfloat16>;

template class FetcherContext<__nv_bfloat16, float>;
template class FetcherContext<__nv_bfloat16, half>;
template class FetcherContext<__nv_bfloat16, __nv_bfloat16>;


template class FetcherContext<__nv_bfloat16, cutlass::uint4b_t>;
template class FetcherContext<__nv_bfloat16, uint8_t>;
#endif


int64_t calc_sparse_time = 0; // microseconds
int64_t cpy_expert_array_to_cpu_time = 0;
int64_t total_row_cpy = 0;
int64_t layer_1_fetch_time = 0;

// 1. copy to expert_for_source_row_fetching
// 2. calc expert_sparse_idx_working
// 3. launch fetch on the stream, from source to working
template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::fetch(int *next_expert_for_source_row, size_t num_rows) {
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
    FT_LOG_TRACE("intermediate_w_size_per_expert: %d", this->intermediate_w_size_per_expert_);
    FT_LOG_TRACE("output_w_size_per_expert: %d", this->output_w_size_per_expert_);
    FT_LOG_TRACE("intermediate_b_size_per_expert: %d", this->intermediate_b_size_per_expert_);
    FT_LOG_TRACE("num_experts: %d", this->num_experts);
    FT_LOG_TRACE("num_rows: %d", this->num_rows);
    // print next_expert_for_source_row
    // printMatrix(next_expert_for_source_row, 1, this->num_rows, this->num_rows, true);

    this->first_time = false;
    // first copy it to the aligned buffer, and then send it back to the CPU
    check_cuda_error(cudaMemcpyAsync(this->expert_for_source_row_fetching, 
        next_expert_for_source_row, sizeof(int) * this->num_rows, cudaMemcpyDeviceToDevice, this->stream));
    {   
        // sparse indexing (CPU version)
        // copy all experts to the CPU buffer
        auto start = std::chrono::high_resolution_clock::now();

        // printf("two pointer should be aligned %x %x\n", 
        //     (size_t)(this->expert_for_source_row_fetching) % 128, 
        //     (size_t)(this->row_expert_sorting_buffer) % 128);
        // exit(0);
        check_cuda_error(cudaMemcpyAsync(this->row_expert_sorting_buffer, 
            this->expert_for_source_row_fetching, sizeof(int) * this->num_rows, cudaMemcpyDeviceToHost, this->stream));
        // sync
        check_cuda_error(cudaStreamSynchronize(this->stream));

        total_row_cpy += this->num_rows;
        auto end_time = std::chrono::high_resolution_clock::now();
        cpy_expert_array_to_cpu_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start).count();

        start = std::chrono::high_resolution_clock::now();
        FT_LOG_DEBUG("fetch milestone 0");
        std::sort(this->row_expert_sorting_buffer, this->row_expert_sorting_buffer + this->num_rows);
        // remove duplicates
        int *end = std::unique(this->row_expert_sorting_buffer, this->row_expert_sorting_buffer + this->num_rows);

        this->active_experts_count = end - this->row_expert_sorting_buffer;

        // memset(this->expert_sparse_idx_cpu, 0, sizeof(int) * this->num_experts);
        for (int i = 0; i < this->active_experts_count; i++) {
            this->expert_sparse_idx_cpu[this->row_expert_sorting_buffer[i]] = i;
        }
        end_time = std::chrono::high_resolution_clock::now();
        calc_sparse_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start).count();

        FT_LOG_DEBUG("fetch milestone 1");
        // copy to GPU
        check_cuda_error(cudaMemcpyAsync(this->expert_sparse_idx_working, 
            this->expert_sparse_idx_cpu, sizeof(int) * this->num_experts, 
                cudaMemcpyHostToDevice, this->stream));
            
        FT_LOG_DEBUG("fetch milestone 2");
    }

    
    // {   // sparse indexing (next_expert_for_source_row -> expert_sparse_index) gpu version
    //     // call into moe_kernels.cu

    //     //measure the time
    //     auto start = std::chrono::high_resolution_clock::now();
    //     get_expert_sparse_idx_kernelLauncher(this->expert_sparse_idx_working,
    //          next_expert_for_source_row, 
    //          this->num_rows, this->num_experts,
    //          &this->active_expert_count);
    //     auto end_time = std::chrono::high_resolution_clock::now();
    //     // active_expert_count
        
    //     calc_sparse_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start).count();
    // }
    
    // expert_for_source_row 

    // copy to expert_for_source_row_fetching, for next layer to get

    #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 0");
    #endif


    // launch fetch on the stream, from source to working
    for(int i = 0; i < this->active_experts_count; i++) {
        int expert = this->row_expert_sorting_buffer[i];
        expert = stdex::randint(0, (int)num_experts - 1);
        std::cout << "Expert:" << expert << std::endl;

        // copy 4 things
        // 1. intermediate_w
        // 2. output_w
        // 3. intermediate_scale
        // 4. output_scale
        if (GlobalConfig::instance().profiling) {
            Profiling::instance().insert(stream, EventType::MEM_START);
        }
        
        #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 1");
        #endif

        // Use the same tag for output and intermediate
        futures_.push_back(GroupedMemoryArena::instance().allocate(layer_name_, {
            reinterpret_cast<char*>(intermediate_working) + i * intermediate_w_size_per_expert_,
            reinterpret_cast<char*>(output_working) + i * output_w_size_per_expert_,
            reinterpret_cast<char*>(intermediate_scale_working) + i * intermediate_scale_size_per_expert_,
            reinterpret_cast<char*>(output_scale_working) + i * output_scale_size_per_expert_
        }, weight_src + expert * weight_size_per_expert_));
        
        #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 2");
        #endif
    }
}

// finish previous job
// update dst from working (swap them)
// update expert_sparse_idx from expert_sparse_idx_working (swap them)

int64_t fetcher_sync_wait_time = 0; // microseconds

template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::sync() {
    // sync the steam
    FT_LOG_TRACE("sync stream\n");
    // get the millisec
    for (auto& future : futures_) {
        future.wait();
    }
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_END);
    }
    futures_.clear();
    check_cuda_error(cudaStreamSynchronize(stream));

    // update dst from working (swap them)
    std::swap(this->intermediate_dst, this->intermediate_working);
    std::swap(this->output_dst, this->output_working);
    std::swap(this->intermediate_bias_dst, this->intermediate_bias_working);
    std::swap(this->intermediate_scale_dst, this->intermediate_scale_working);
    std::swap(this->output_scale_dst, this->output_scale_working);
}

// called in FfnLayer.cc
// 
template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::set_source(const char *w_of_the_layer_to_load, const std::string &layer_name) {
    if (w_of_the_layer_to_load == nullptr) {
        this->weight_src = nullptr;
        this->has_source = false;
        return;
    }
    this->has_source = true;
    this->layer_name_ = layer_name;
}

int64_t expert_for_row_backup_time = 0; // microseconds

template<class ActT, class WeightT, class BiasT> 
FetcherContext<ActT, WeightT, BiasT>::~FetcherContext() {
    this->freeBuffer();
    //destroy the stream
    Profiling::instance().report();
    Profiling::instance().reset();
    check_cuda_error(cudaStreamDestroy(this->stream));
}

template<class ActT, class WeightT, class BiasT> 
FetcherContext<ActT, WeightT, BiasT>::FetcherContext(
    int mode, int num_experts, 
    size_t intermediate_w_size_per_expert, size_t output_w_size_per_expert,
    size_t intermediate_b_size_per_expert,
    size_t intermediate_scale_size_per_expert, size_t output_scale_size_per_expert,
    size_t arena_size) 
    :   mode(mode),
        first_time(true),
        num_experts(num_experts),
        intermediate_w_size_per_expert_(get_real_size<WeightT>(intermediate_w_size_per_expert)),
        output_w_size_per_expert_(get_real_size<WeightT>(output_w_size_per_expert)),
        intermediate_b_size_per_expert_(get_real_size<BiasT>(intermediate_b_size_per_expert)),
        intermeiate_scale_size_per_expert_(get_real_size<BiasT>(intermediate_scale_size_per_expert)),
        output_scale_size_per_expert_(get_real_size<BiasT>(output_scale_size_per_expert)) {
    
    // create cuda stream
    check_cuda_error(cudaStreamCreate(&this->stream));
    // TODO: set the size elsewhere (after refactoring with DI)
    FT_LOG_ERROR("%d", stream);
    weight_size_per_expert_ = intermediate_w_size_per_expert_ + output_w_size_per_expert_ + intermediate_scale_size_per_expert_ + output_scale_size_per_expert_;
    GroupedMemoryArena::instance().initIfUninit(arena_size, {
        intermediate_w_size_per_expert_,
        output_w_size_per_expert_,
        intermediate_scale_size_per_expert_,
        output_scale_size_per_expert_}, stream);
    Profiling::instance().reset();
}


template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::allocateBuffer(IAllocator* allocator, size_t num_rows) {
    if (this->buffer_allocated == true) 
        return;
    this->buffer_allocated = true; 
    this->num_rows = num_rows;
    // allocated all buffer
    this->expert_for_source_row_fetching = (int*)mallocOnDeviceAligned(sizeof(int) * num_rows);
    this->expert_sparse_idx = (int*)mallocOnDeviceAligned(sizeof(int) * num_experts);
    this->expert_sparse_idx_working = (int*)mallocOnDeviceAligned(sizeof(int) * num_experts);

    // TODO: refactor
    this->intermediate_dst = (WeightT*)mallocOnDeviceAligned(intermediate_w_size_per_expert_ * num_experts);
    this->output_dst = (WeightT*)mallocOnDeviceAligned(output_w_size_per_expert_ * num_experts);
    this->intermediate_bias_dst = (BiasT*)mallocOnDeviceAligned(intermediate_b_size_per_expert_ * num_experts);
    this->intermediate_scale_dst = (BiasT*)mallocOnDeviceAligned(intermediate_scale_size_per_expert_ * num_experts);
    this->output_scale_dst = (BiasT*)mallocOnDeviceAligned(output_scale_size_per_expert_ * num_experts);
    this->intermediate_working = (WeightT*)mallocOnDeviceAligned(intermediate_w_size_per_expert_ * num_experts);
    this->output_working = (WeightT*)mallocOnDeviceAligned(output_w_size_per_expert_ * num_experts);
    this->intermediate_bias_working = (BiasT*)mallocOnDeviceAligned(intermediate_b_size_per_expert_ * num_experts);
    this->intermediate_scale_working = (BiasT*)mallocOnDeviceAligned(intermediate_scale_size_per_expert_ * num_experts);
    this->output_scale_working = (BiasT*)mallocOnDeviceAligned(output_scale_size_per_expert_ * num_experts);

    // this->row_expert_sorting_buffer = (int*)allocator->reMalloc(this->row_expert_sorting_buffer, sizeof(int) * num_rows, false, true);
    // this->expert_sparse_idx_cpu = (int*)allocator->reMalloc(this->expert_sparse_idx_cpu, sizeof(int) * num_experts, false, true);
    check_cuda_error(cudaMallocHost(&this->row_expert_sorting_buffer, sizeof(int) * num_rows + 128));
    row_expert_sorting_buffer = (int*)((void*)(((size_t)row_expert_sorting_buffer + 128 - 1) & ~(128 - 1)));
    check_cuda_error(cudaMallocHost(&this->expert_sparse_idx_cpu, sizeof(int) * num_experts + 128));
    expert_sparse_idx_cpu = (int*)((void*)(((size_t)expert_sparse_idx_cpu + 128 - 1) & ~(128 - 1)));
    this->allocator = allocator;
}


template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::freeBuffer() {
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


    // free all big buffer, ignore small buffers
    for (auto ptr : this->big_buffer_on_device) {
        check_cuda_error(cudaFree(ptr));
    }
    FT_LOG_DEBUG("fetcher context free buffer finished");
    FT_LOG_ERROR("fetcher_sync_wait_time %lld us", fetcher_sync_wait_time);
    FT_LOG_ERROR("calc_sparse_time %lld us", calc_sparse_time);
    FT_LOG_ERROR("expert_for_row_backup_time %lld us", expert_for_row_backup_time);
    FT_LOG_ERROR("expert_for_row_time %lld us", cpy_expert_array_to_cpu_time);
    FT_LOG_ERROR("total_row_cpy %lld", total_row_cpy);
    FT_LOG_ERROR("layer_1_fetch_time %lld us", layer_1_fetch_time);
}

// TODO: refactor
template<class ActT, class WeightT, class BiasT> 
void* FetcherContext<ActT, WeightT, BiasT>::mallocOnDeviceAligned(size_t size) {
    size_t alignment = 128;
    void* ptr;
    check_cuda_error(cudaMalloc(&ptr, size + alignment));
    void* aligned_ptr = (void*)(((size_t)ptr + alignment - 1) & ~(alignment - 1));
    big_buffer_on_device.push_back(ptr);
    return aligned_ptr;
}
} // namespace fastertransformer