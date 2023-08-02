#pragma once

#include <vector>
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/meter.h"

namespace fastertransformer {

enum class EventType
{
    NON_MOE_BLOCK_START,
    NON_MOE_BLOCK_END,
    BLOCK_START,
    BLOCK_END,
    MEM_START,
    MEM_END,
    COMP_START,
    COMP_END
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
    static constexpr int NUM_EVENT_TYPE = (int)EventType::COMP_END + 1;

    std::vector<std::vector<cudaEvent_t>> events_;
    static const std::vector<std::string> event_names_;
    
    AverageMeter<double> cache_hit_rate_;

    Profiling() {
        events_.resize(NUM_EVENT_TYPE);
    }
};

} // namespace fastertransformer
