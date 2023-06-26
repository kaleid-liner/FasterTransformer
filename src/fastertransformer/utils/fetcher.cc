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


int64_t calc_sparse_time = 0; // microseconds
int64_t cpy_expert_array_to_cpu_time = 0;
int64_t total_row_cpy = 0;
int64_t layer_1_fetch_time = 0;

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

        // copy three things
        // 1. output_w
        // 2. intermediate_w
        // 3. intermediate_bias
        // check_cuda_error(cudaMemcpyAsync(this->output_working + i * this->output_w_size_per_expert, 
        //     this->output_w_src + expert * this->output_w_size_per_expert, 
        //     sizeof(WeightT) * this->output_w_size_per_expert, cudaMemcpyHostToDevice, this->stream));
        auto dst = this->output_working + i * this->output_w_size_per_expert;
        auto src = this->output_w_src + expert * this->output_w_size_per_expert;
        // Currently use cpu address of the memory as tag
        // need a better hash func to better manage the cache for experts in the same layer
        auto tag = reinterpret_cast<tag_t>(src);
        if (GlobalConfig<WeightT>::instance().profiling) {
            Profiling::instance().insert(stream, EventType::MEM_START);
        }
        
        futures_.push_back(GroupedMemoryArena<WeightT>::allocate(prefix_ + "::output", tag, dst, src));

        #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 1");
        #endif

        // check_cuda_error(cudaMemcpyAsync(this->intermediate_working + i * this->intermediate_w_size_per_expert, 
        //     this->intermediate_w_src + expert * this->intermediate_w_size_per_expert, 
        //     sizeof(WeightT) * this->intermediate_w_size_per_expert, cudaMemcpyHostToDevice, this->stream));
        dst = this->intermediate_working + i * this->intermediate_w_size_per_expert;
        src = this->intermediate_w_src + expert * this->intermediate_w_size_per_expert;
        // Use the same tag for output and intermediate
        futures_.push_back(GroupedMemoryArena<WeightT>::allocate(prefix_ + "::intermediate", tag, dst, src));
        
        #ifdef FETCHER_DEBUG
        // sync the cuda stream for debug
        cudaStreamSynchronize(this->stream);
        FT_LOG_DEBUG("sych 2");
        #endif

        // This is not used currently. TODO: check if used
        // check_cuda_error(cudaMemcpyAsync(this->intermediate_bias_working + i * this->intermediate_b_size_per_expert,
        //     this->intermediate_bias_src + expert * this->intermediate_b_size_per_expert, 
        //     sizeof(BiasT) * this->intermediate_b_size_per_expert, cudaMemcpyDeviceToDevice, this->stream));
        
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

int64_t fetcher_sync_wait_time = 0; // microseconds

template<class WeightT, class BiasT> 
void FetcherContext<WeightT, BiasT>::sync() {
    // sync the steam
    FT_LOG_TRACE("sync stream\n");
    // get the millisec
    for (auto& future : futures_) {
        future.wait();
    }
    if (GlobalConfig<WeightT>::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_END);
    }
    futures_.clear();
    check_cuda_error(cudaStreamSynchronize(stream));

    // update dst from working (swap them)
    std::swap(this->output_dst, this->output_working);
    std::swap(this->intermediate_dst, this->intermediate_working);
    std::swap(this->intermediate_bias_dst, this->intermediate_bias_working);
}

// called in FfnLayer.cc
// 
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

int64_t expert_for_row_backup_time = 0; // microseconds

template<class WeightT, class BiasT> 
FetcherContext<WeightT, BiasT>::~FetcherContext() {
    this->freeBuffer();
    //destroy the stream
    Profiling::instance().report();
    Profiling::instance().reset();
    check_cuda_error(cudaStreamDestroy(this->stream));
}

template<class WeightT, class BiasT> 
FetcherContext<WeightT, BiasT>::FetcherContext(
    int mode, int num_experts, 
    size_t intermediate_w_size_per_expert, size_t output_w_size_per_expert,
    size_t intermediate_b_size_per_expert, size_t arena_size,
    std::string prefix) 
    :   mode(mode),
        first_time(true),
        num_experts(num_experts),
        intermediate_w_size_per_expert(intermediate_w_size_per_expert),
        output_w_size_per_expert(output_w_size_per_expert),
        intermediate_b_size_per_expert(intermediate_b_size_per_expert),
        prefix_(prefix) {
    
    // create cuda stream
    check_cuda_error(cudaStreamCreate(&this->stream));
    // TODO: set the size elsewhere (after refactoring with DI)
    FT_LOG_ERROR("%d", stream);
    GroupedMemoryArena<WeightT>::createMemoryArena(prefix_ + "::intermediate", arena_size, intermediate_w_size_per_expert, stream);
    GroupedMemoryArena<WeightT>::createMemoryArena(prefix_ + "::output", arena_size, output_w_size_per_expert, stream);
    Profiling::instance().reset();
}


template<class WeightT, class BiasT> 
void FetcherContext<WeightT, BiasT>::allocateBuffer(IAllocator* allocator, size_t num_rows) {
    if (this->buffer_allocated == true) 
        return;
    this->buffer_allocated = true; 
    this->num_rows = num_rows;
    // allocated all buffer
    this->expert_for_source_row_fetching = (int*)this->mallocOnDeviceAligned(sizeof(int) * num_rows);
    this->expert_sparse_idx = (int*)this->mallocOnDeviceAligned(sizeof(int) * num_experts);
    this->expert_sparse_idx_working = (int*)this->mallocOnDeviceAligned(sizeof(int) * num_experts);
    this->intermediate_bias_dst = (BiasT*)this->mallocOnDeviceAligned(sizeof(BiasT) * intermediate_b_size_per_expert * num_experts);
    this->intermediate_bias_working = (BiasT*)this->mallocOnDeviceAligned(sizeof(BiasT) * intermediate_b_size_per_expert * num_experts);

    // TODO: refactor
    this->output_dst = GroupedMemoryArena<WeightT>::mallocBuffer(prefix_ + "::output", output_w_size_per_expert * num_experts, 1);
    this->intermediate_dst = GroupedMemoryArena<WeightT>::mallocBuffer(prefix_ + "::intermediate", intermediate_w_size_per_expert * num_experts, 1);
    this->output_working = GroupedMemoryArena<WeightT>::mallocBuffer(prefix_ + "::output", output_w_size_per_expert * num_experts, 1);
    this->intermediate_working = GroupedMemoryArena<WeightT>::mallocBuffer(prefix_ + "::intermediate", intermediate_w_size_per_expert * num_experts, 1);
    big_buffer_on_device.push_back(output_dst);
    big_buffer_on_device.push_back(intermediate_dst);
    big_buffer_on_device.push_back(output_working);
    big_buffer_on_device.push_back(intermediate_working);

    // this->row_expert_sorting_buffer = (int*)allocator->reMalloc(this->row_expert_sorting_buffer, sizeof(int) * num_rows, false, true);
    // this->expert_sparse_idx_cpu = (int*)allocator->reMalloc(this->expert_sparse_idx_cpu, sizeof(int) * num_experts, false, true);
    check_cuda_error(cudaMallocHost(&this->row_expert_sorting_buffer, sizeof(int) * num_rows + 128));
    row_expert_sorting_buffer = (int*)((void*)(((size_t)row_expert_sorting_buffer + 128 - 1) & ~(128 - 1)));
    check_cuda_error(cudaMallocHost(&this->expert_sparse_idx_cpu, sizeof(int) * num_experts + 128));
    expert_sparse_idx_cpu = (int*)((void*)(((size_t)expert_sparse_idx_cpu + 128 - 1) & ~(128 - 1)));
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
template<class WeightT, class BiasT> 
void* FetcherContext<WeightT, BiasT>::mallocOnDeviceAligned(size_t size) {
    size_t alignment = 128;
    void* ptr;
    check_cuda_error(cudaMalloc(&ptr, size + alignment));
    void* aligned_ptr = (void*)(((size_t)ptr + alignment - 1) & ~(alignment - 1));
    big_buffer_on_device.push_back(ptr);
    return aligned_ptr;
}
} // namespace fastertransformer