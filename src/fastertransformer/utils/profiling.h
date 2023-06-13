#pragma once

#include <vector>
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

enum class EventType
{
    COMP_START,
    COMP_END,
    MEM_START,
    MEM_END
};

class Profiling
{
public:
    void reset();

    void insert(cudaStream_t stream, EventType type);

    ~Profiling();

    void report() const;

    static Profiling& instance()
    {
        static Profiling instance;
        return instance;
    }

private:
    std::vector<cudaEvent_t> comp_start_events_;
    std::vector<cudaEvent_t> comp_end_events_;
    std::vector<cudaEvent_t> mem_start_events_;
    std::vector<cudaEvent_t> mem_end_events_;
};

} // namespace fastertransformer
