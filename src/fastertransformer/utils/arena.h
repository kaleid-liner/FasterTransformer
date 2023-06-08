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
        : size_(size), chunk_size_(chunk_size), 
          ptr_(nullptr), stream_(stream)
    {
        chunk_num_ = size_ / chunk_size_;
        cache_ = std::make_shared<cache_t>(chunk_num_);

        // Ensure every experts is aligned
        check_cuda_error(cudaMallocPitch(&ptr_, &pitch_size_, (size_t)chunk_size_ * sizeof(T), chunk_num_));
        // Pre-cache
        // This is a workaround, currently this process is necessary
        for (tag_t t = 0; t < chunk_num_; t++) {
            cache_->Put(-(t + 1), (T*)((char*)ptr_ + pitch_size_ * t));
        }

        if (GlobalConfig<T>::instance().disk_offload) {
            offload_buffer_ = new char [chunk_size_ * sizeof(T)];
        }
    }

    MemoryArena(const MemoryArena& o) = delete;

    MemoryArena(MemoryArena&& o)
        : size_(o.size_), chunk_size_(o.chunk_size_),
          ptr_(o.ptr_), stream_(o.stream_),
          chunk_num_(o.chunk_num_), pitch_size_(o.pitch_size_),
          cache_(std::move(o.cache_))
    {
        o.ptr_ = nullptr;
    }

    ~MemoryArena()
    {
        cudaFree(ptr_);
        if (GlobalConfig<T>::instance().disk_offload) {
            delete[] offload_buffer_;
        }
    }

    // Allocate a chunk
    // note: tag < 0 is reserved
    std::future<void> allocate(tag_t tag, T* dst = nullptr, const T* src = nullptr)
    {
        if (GlobalConfig<T>::instance().profiling) {
            Profiling::instance().insert(stream_, EventType::MEM_START);
        }
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
            if (GlobalConfig<T>::instance().profiling) {
                Profiling::instance().insert(stream_, EventType::MEM_END);
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

private:
    size_t chunk_size_;
    size_t size_;
    size_t pitch_size_;
    size_t chunk_num_;
    T* ptr_;
    using cache_t = caches::fixed_sized_cache<tag_t, T*, caches::LRUCachePolicy>;
    std::shared_ptr<cache_t> cache_;
    cudaStream_t stream_;
    char* offload_buffer_;
};

template <typename T>
class GroupedMemoryArena {
public:
    using tag_t = typename MemoryArena<T>::tag_t;

    static void createMemoryArena(std::string group, size_t size, size_t chunk_size, cudaStream_t stream)
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

    static std::future<void> allocate(std::string group, tag_t tag, T* dst = nullptr, const T* src = nullptr)
    {
        return arenas_->at(group).allocate(tag, dst, src);
    }
private:
    static std::unique_ptr<std::unordered_map<std::string, MemoryArena<T>>> arenas_;
};

template <typename T>
std::unique_ptr<std::unordered_map<std::string, MemoryArena<T>>> GroupedMemoryArena<T>::arenas_ = nullptr;

}  // namespace fastertransformer
