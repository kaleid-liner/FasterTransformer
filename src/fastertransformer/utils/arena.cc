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
    auto future = std::async(std::launch::async, [=, this]() {
        if (repl.first != nullptr && !repl.second && src != nullptr) {
            // FT_LOG_ERROR("Cache miss");
            // FT_LOG_ERROR("%p %p %p %d", repl.first, src, ptr_, stream_);
            if (GlobalConfig::instance().disk_offload) {
                std::ifstream ifs(GlobalConfig::instance().offload_path + tag, std::ifstream::binary);
                ifs.read(offload_buffer_, chunk_size_ * sizeof(T));
            }
            check_cuda_error(
                cudaMemcpyAsync(
                    repl.first, src, chunk_size_ * sizeof(T),
                    cudaMemcpyHostToDevice, stream_));
        } else {
            //FT_LOG_ERROR("Cache hit");
        }
        // Use specified type as workaround of undefined symbol
        if (post_callback == nullptr && dst != nullptr) {
            invokeCudaD2DcpyConvert<char, char>((char *)dst, (const char *)repl.first, chunk_size_ * sizeof(T), stream_);
        } else {
            post_callback(repl.first, stream_);
        }
    });
    return future;
}

} // namespace fastertransformer
