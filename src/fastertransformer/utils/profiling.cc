#include "profiling.h"


namespace {

void clearEvents(std::vector<cudaEvent_t>& events)
{
    for (cudaEvent_t event : events)
    {
        cudaEventDestroy(event);
    }
    events.clear();
}

}

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
}

void Profiling::report() const
{
    float ms;
    for (int i = 0; i < comp_start_events_.size(); i++) {
        cudaEventElapsedTime(&ms, comp_start_events_[i], comp_end_events_[i]);
        std::cout << "Comp kernel " << i << ": " << ms << " ms" << std::endl;
    }
    for (int i = 0; i < mem_start_events_.size(); i++) {
        cudaEventElapsedTime(&ms, mem_start_events_[i], mem_end_events_[i]);
        std::cout << "Mem kernel " << i << ": " << ms << " ms" << std::endl;
    }
}

}