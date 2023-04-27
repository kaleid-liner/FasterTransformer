#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace fastertransformer {

template<class T>
class Fetcher {
private:
    cudaStream_t stream;
    size_t buffer_size;
    T* buffer;
    bool fetched = false;
public:
    Fetcher(size_t size_per_expert, int max_expert_fetched);
    ~Fetcher();

    void fetch(T* src, std::vector<int> experts);
    std::vector<T*> get_last_result();
};





} // namespace fastertransformer