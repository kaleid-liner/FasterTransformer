#pragma once

#include <vector>
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/meter.h"

namespace fastertransformer {

enum class EventType
{
    COMP_START,
    COMP_END,
    MEM_START,
    MEM_END,
    BLOCK_START,
    BLOCK_END
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

    void cacheHit(bool hit)
    {
        cache_hit_rate_.update(hit);
    }

private:
    std::vector<cudaEvent_t> comp_start_events_;
    std::vector<cudaEvent_t> comp_end_events_;
    std::vector<cudaEvent_t> mem_start_events_;
    std::vector<cudaEvent_t> mem_end_events_;
    std::vector<cudaEvent_t> block_start_events_;
    std::vector<cudaEvent_t> block_end_events_;
    
    AverageMeter<double> cache_hit_rate_;
};

} // namespace fastertransformer
