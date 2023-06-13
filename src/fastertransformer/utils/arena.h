#pragma once

#include <cuda.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <future>
#include <fstream>
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/config.h"
#include "src/fastertransformer/utils/profiling.h"
#include "cache.h"
#include "cache_policy.h"
#include "lru_cache_policy.h"

namespace fastertransformer {

template <typename T>
class MemoryArena {
public:
    using tag_t = int64_t;

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
        for (tag_t t = 0; t < chunk_num_; t++) {
            cache_->Put(-(t + 1), (T*)((char*)ptr_ + pitch_sizes_[chunk_size_] * t));
        }

        if (GlobalConfig<T>::instance().disk_offload) {
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
    std::future<void> allocate(tag_t tag, T* dst = nullptr, const T* src = nullptr)
    {
        auto repl = cache_->PutKey(tag, nullptr);
        auto future = std::async(std::launch::async, [&]() {
            if (repl.first != nullptr && !repl.second && src != nullptr) {
                // FT_LOG_ERROR("Cache miss");
                // FT_LOG_ERROR("%p %p %p %d", repl.first, src, ptr_, stream_);
                if (GlobalConfig<T>::instance().disk_offload) {
                    std::ifstream ifs(GlobalConfig<T>::instance().offload_path, std::ifstream::binary);
                    ifs.read(offload_buffer_, chunk_size_ * sizeof(T));
                }
                check_cuda_error(
                    cudaMemcpyAsync(
                        repl.first, src, chunk_size_ * sizeof(T),
                        cudaMemcpyHostToDevice, stream_));
            } else {
                //FT_LOG_ERROR("Cache hit");
            }
            // not needed for batch size = 1
            if (dst != nullptr) {
                // try {
                //     check_cuda_error(
                //         cudaMemcpyAsync(
                //             dst, repl.first, chunk_size_ * sizeof(T),
                //             cudaMemcpyDeviceToDevice, stream_));
                // } catch (const std::exception &exc) {
                //     std::cerr << exc.what() << std::endl;
                // }
            }
        });
        return future;
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

template <typename T>
class GroupedMemoryArena {
public:
    using tag_t = typename MemoryArena<T>::tag_t;

    static void createMemoryArena(const std::string& group, size_t size, size_t chunk_size, cudaStream_t stream)
    {
        if (arenas_ == nullptr) {
            arenas_ = std::make_unique<std::unordered_map<std::string, MemoryArena<T>>>();
        }
        auto iter = arenas_->find(group);
        if (iter != arenas_->end()) {
            arenas_->erase(iter);
        }
        arenas_->emplace(group, MemoryArena<T>(size, chunk_size, stream));
    }

    static std::future<void> allocate(const std::string& group, tag_t tag, T* dst = nullptr, const T* src = nullptr)
    {
        return arenas_->at(group).allocate(tag, dst, src);
    }

    static T* mallocBuffer(const std::string& group, size_t width, size_t height)
    {
        return arenas_->at(group).mallocBuffer(width, height);
    }
private:
    static std::unique_ptr<std::unordered_map<std::string, MemoryArena<T>>> arenas_;
};

template <typename T>
std::unique_ptr<std::unordered_map<std::string, MemoryArena<T>>> GroupedMemoryArena<T>::arenas_ = nullptr;

}  // namespace fastertransformer
