#include "profiling.h"
#include "meter.h"
#include "src/fastertransformer/utils/logger.h"


namespace {

void clearEvents(std::vector<cudaEvent_t>& events)
{
    for (cudaEvent_t event : events)
    {
        cudaEventDestroy(event);
    }
    events.clear();
}

} // namespace

namespace fastertransformer {

void Profiling::insert(cudaStream_t stream, EventType type)
{
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    switch (type)
    {
        case EventType::COMP_START:
            comp_start_events_.push_back(event);
            break;
        case EventType::COMP_END:
            comp_end_events_.push_back(event);
            break;
        case EventType::MEM_START:
            mem_start_events_.push_back(event);
            break;
        case EventType::MEM_END:
            mem_end_events_.push_back(event);
            break;
    }
}

Profiling::~Profiling()
{
    reset();
}

void Profiling::reset()
{
    clearEvents(comp_start_events_);
    clearEvents(comp_end_events_);
    clearEvents(mem_start_events_);
    clearEvents(mem_end_events_);
    cache_hit_rate_.reset();
}

void Profiling::report() const
{
    FT_CHECK(comp_start_events_.size() == comp_end_events_.size());
    FT_CHECK(mem_start_events_.size() == mem_end_events_.size());

    float ms;
    AverageMeter<float> comp_lats, mem_lats;

    std::cout << "Total events num:"
              << " (comp)" << comp_start_events_.size()
              << " (mem)" << mem_start_events_.size()
              << std::endl;

    for (int i = 0; i < comp_start_events_.size(); i++) {
        cudaEventElapsedTime(&ms, comp_start_events_[i], comp_end_events_[i]);
        comp_lats.update(ms);
    }
    std::cout << "Comp avg lats: " << comp_lats.getAvg() << " ms" << std::endl;

    for (int i = 0; i < mem_start_events_.size(); i++) {
        cudaEventElapsedTime(&ms, mem_start_events_[i], mem_end_events_[i]);
        mem_lats.update(ms);
    }
    std::cout << "Mem avg lats: " << mem_lats.getAvg() << " ms" << std::endl;

    std::cout << "Average cache hit rate: " << cache_hit_rate_.getAvg() << std::endl;
}

} // namespace fastertransformer
