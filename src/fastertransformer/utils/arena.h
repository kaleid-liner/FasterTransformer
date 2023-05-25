#pragma once

#include <cuda.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include "src/fastertransformer/utils/cuda_utils.h"
#include "cache.h"
#include "cache_policy.h"
#include "lru_cache_policy.h"

namespace fastertransformer {

template<typename T>
class MemoryArena {
public:
    using tag_t = int64_t;

    MemoryArena(size_t size, size_t chunk_size, cudaStream_t stream) 
        : size_(size), chunk_size_(chunk_size), 
          ptr_(nullptr), stream_(stream)
    {
        chunk_num_ = size_ / chunk_size_;
        cache_ = std::make_shared<cache_t>(chunk_num_);

        // Ensure every experts is aligned
        FT_LOG_ERROR("%ld", (size_t)chunk_size_ * sizeof(T));
        check_cuda_error(cudaMallocPitch(&ptr_, &pitch_size_, (size_t)chunk_size_ * sizeof(T), chunk_num_));
        // Pre-cache
        // This is a workaround, currently this process is necessary
        for (tag_t t = 0; t < chunk_num_; t++) {
            cache_->Put(t, (T*)((char*)ptr_ + pitch_size_ * t));
        }
    }

    ~MemoryArena()
    {
        cudaFree(ptr_);
    }

    // Allocate a chunk
    // note: tag 0 ~ chunk_num is reserved
    T* allocate(tag_t tag, T* dst = nullptr, const T* src = nullptr)
    {
        auto repl = cache_->PutKey(tag, nullptr);
        if (repl.first != nullptr && (!repl.second || true) && src != nullptr) {
            //FT_LOG_ERROR("Cache miss");
            check_cuda_error(
                cudaMemcpyAsync(
                    repl.first, src, chunk_size_ * sizeof(T),
                    cudaMemcpyHostToDevice, stream_));
        }
        else {
            //FT_LOG_ERROR("Cache hit");
        }
        // not needed for batch size = 1
        if (dst != nullptr) {
            check_cuda_error(
                cudaMemcpyAsync(
                    dst, repl.first, chunk_size_ * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream_));
        }
        return repl.first;
    }

    // Wait until all previous work is done
    void synchronize()
    {
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    //
    size_t getCapacity()
    {
        return chunk_num_;
    }

private:
    size_t chunk_size_;
    size_t size_;
    size_t pitch_size_;
    size_t chunk_num_;
    T* ptr_;
    using cache_t = caches::fixed_sized_cache<tag_t, T*, caches::LRUCachePolicy>;
    std::shared_ptr<cache_t> cache_;
    cudaStream_t stream_;
};

}  // namespace fastertransformer
