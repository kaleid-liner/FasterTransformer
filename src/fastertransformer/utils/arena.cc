#include "src/fastertransformer/utils/arena.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"

namespace fastertransformer {

template class MemoryArena<char>;

template <typename T>
std::future<void> MemoryArena<T>::allocate(const tag_t& tag, T* dst, const T* src,
                                           std::function<void(const T*, cudaStream_t)> post_callback)
{
    auto repl = cache_->PutKey(tag, nullptr);
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().cacheHit(repl.second);
    }
    auto future = pool_->push([=](int) {
        if (!GlobalConfig::instance().use_cache  // if not use_cache, do this anyway
            || (repl.first != nullptr && !repl.second && src != nullptr)) {
            const T* cpy_src = src;
            if (GlobalConfig::instance().disk_offload) {
                std::string filename = GlobalConfig::instance().offload_path + tag + ".bin";
                std::ifstream ifs(filename, std::ifstream::binary);
                ifs.read(offload_buffer_, chunk_size_ * sizeof(T));
                FT_CHECK_WITH_INFO(ifs, "Read from " + filename + " failed");
                cpy_src = offload_buffer_;
            }
            check_cuda_error(
                cudaMemcpyAsync(
                    repl.first, cpy_src, chunk_size_ * sizeof(T),
                    cudaMemcpyHostToDevice, stream_));
        }
        if (post_callback == nullptr && dst != nullptr) {
            check_cuda_error(cudaMemcpyAsync(dst, repl.first, chunk_size_, cudaMemcpyDeviceToDevice, stream_));
        } else {
            post_callback(repl.first, stream_);
        }
    });
    return future;
}

} // namespace fastertransformer
