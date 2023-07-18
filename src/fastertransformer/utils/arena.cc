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
        if (repl.first != nullptr && !repl.second && src != nullptr) {
            FT_LOG_INFO(tag);
            const T* cpy_src = src;
            if (!GlobalConfig::instance().offload_path.empty()) {
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
            invokeCudaD2DcpyConvert<char, char>((char *)dst, (const char *)repl.first, chunk_size_ * sizeof(T), stream_);
        } else {
            post_callback(repl.first, stream_);
        }
    });
    return future;
}

} // namespace fastertransformer
