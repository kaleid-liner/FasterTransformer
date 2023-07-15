#pragma once

#include <cuda.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <future>
#include <fstream>
#include <algorithm>
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/config.h"
#include "src/fastertransformer/utils/profiling.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "cache.h"
#include "cache_policy.h"
#include "lru_cache_policy.h"

namespace fastertransformer {

template <typename T>
class MemoryArena {
public:
    using tag_t = std::string;

    MemoryArena(size_t size, size_t chunk_size, cudaStream_t stream) 
        : chunk_size_(chunk_size), 
          size_(size),
          chunk_num_(0),
          ptr_(nullptr),
          cache_(nullptr),
          stream_(stream),
          offload_buffer_(nullptr),
          pitch_sizes_()
    {
        chunk_num_ = size_ / chunk_size_;
        cache_ = std::make_shared<cache_t>(chunk_num_);

        // Ensure every experts is aligned
        ptr_ = mallocBuffer(chunk_size_, chunk_num_);
        // Pre-cache
        // This is a workaround, currently this process is necessary
        for (int t = 0; t < chunk_num_; t++) {
            cache_->Put(std::to_string(t), (T*)((char*)ptr_ + pitch_sizes_[chunk_size_] * t));
        }

        if (GlobalConfig::instance().disk_offload) {
            offload_buffer_ = new char[chunk_size_ * sizeof(T)];
        }
    }

    MemoryArena(const MemoryArena& o) = delete;

    MemoryArena(MemoryArena&& o)
        : chunk_size_(o.chunk_size_),
          size_(o.size_),
          chunk_num_(o.chunk_num_),
          ptr_(o.ptr_),
          cache_(std::move(o.cache_)),
          stream_(o.stream_),
          offload_buffer_(o.offload_buffer_),
          pitch_sizes_(std::move(o.pitch_sizes_))
    {
        o.ptr_ = nullptr;
        o.offload_buffer_ = nullptr;
    }

    ~MemoryArena()
    {
        if (ptr_) {
            cudaFree(ptr_);
        }
        if (offload_buffer_) {
            delete[] offload_buffer_;
        }
    }

    // Allocate a chunk
    // note: tag < 0 is reserved
    // post_callback is used to do further operations on cached data
    std::future<void> allocate(const tag_t& tag, T* dst = nullptr, const T* src = nullptr, 
                               std::function<void(const T*, cudaStream_t)> post_callback = nullptr);

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

    // Malloc a buffer
    // The buffer is unmanaged and uncached
    T* mallocBuffer(size_t width, size_t height)
    {
        size_t pitch_size;
        T* ptr;
        check_cuda_error(cudaMallocPitch(&ptr, &pitch_size, width * sizeof(T), height));
        pitch_sizes_[width] = pitch_size;
        return ptr;
    }

    size_t getPitchSize(size_t width)
    {
        return pitch_sizes_[width];
    }

private:
    size_t chunk_size_;
    size_t size_;
    size_t chunk_num_;
    T* ptr_;
    using cache_t = caches::fixed_sized_cache<tag_t, T*, caches::LRUCachePolicy>;
    std::shared_ptr<cache_t> cache_;
    cudaStream_t stream_;
    char* offload_buffer_;
    
    std::unordered_map<size_t, size_t> pitch_sizes_;
};

class GroupedMemoryArena {
public:
    using tag_t = typename MemoryArena<char>::tag_t;

    void init(size_t size, const std::vector<size_t>& tensor_sizes, cudaStream_t stream)
    {
        tensor_sizes_ = tensor_sizes;
        size_t chunk_size = std::accumulate(tensor_sizes_.begin(), tensor_sizes_.end(), 0);
        arena_ = std::make_unique<MemoryArena<char>>(size, chunk_size, stream);
    }

    void initIfUninit(size_t size, const std::vector<size_t>& tensor_sizes, cudaStream_t stream)
    {
        if (arena_ != nullptr) {
            init(size, tensor_sizes, stream);
        }
    }

    std::future<void> allocate(const tag_t& tag, const std::vector<char*>& dsts, const char* src = nullptr)
    {
        FT_CHECK_WITH_INFO(arena_ != nullptr, "Memory arena uninitialized.");
        auto post_callback = [](const char* cached_ptr, cudaStream_t stream) {
            FT_CHECK(dsts.size() == tensor_sizes_.size());
            const char* ptr = cached_ptr;
            for (int i = 0; i < dsts.size(); ++i)
                invokeCudaD2DcpyConvert<char, char>(dsts[i], ptr, tensor_sizes_[i], stream_);
                ptr += tensor_sizes_[i];
            }
        }
        return arena_->allocate(tag, nullptr, src, post_callback);
    }

    char* mallocBuffer(size_t width, size_t height)
    {
        FT_CHECK_WITH_INFO(arena_ != nullptr, "Memory arena uninitialized.");
        return arena_->mallocBuffer(width, height);
    }

    static GroupedMemoryArena& instance()
    {
        static GroupedMemoryArena instance;
        return instance;
    }

private:
    GroupedMemoryArena() {}

    std::unique_ptr<MemoryArena<char>> arena_;

    std::vector<size_t> tensor_sizes_;
};

}  // namespace fastertransformer
