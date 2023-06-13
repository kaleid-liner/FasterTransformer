#include "src/fastertransformer/utils/arena.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template class MemoryArena<float>;
template class MemoryArena<half>;
template class MemoryArena<cutlass::uint4b_t>;
template class MemoryArena<uint8_t>;

#ifdef ENABLE_BF16
template class MemoryArena<__nv_bfloat16>;
#endif

template <typename T>
std::future<void> MemoryArena<T>::allocate(tag_t tag, T* dst, const T* src)
{
    auto repl = cache_->PutKey(tag, nullptr);
    auto future = std::async(std::launch::async, [=, this]() {
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
            // check_cuda_error(
            //     cudaMemcpyAsync(
            //         dst, repl.first, chunk_size_ * sizeof(T),
            //         cudaMemcpyDeviceToDevice, stream_));
            // Use specified type as workaround of undefined symbol
            invokeCudaD2DcpyConvert<float, float>((float *)dst, (const float *)repl.first, chunk_size_ * sizeof(T) / sizeof(float), stream_);
        }
    });
    return future;
}

} // namespace fastertransformer
