#pragma once

#include <memory>
#include <iostream>

namespace fastertransformer {

# define GPU_ONLY 0
# define FETCH_ON_DEMAND 1
# define PREFETCH 2

template <typename T>
class GlobalConfig {
public:
    static GlobalConfig& instance()
    {
        static GlobalConfig instance;
        return instance;
    }

    void setDefault()
    {
        encoder_arena_size = (size_t)5 * 1024 * 1024 * 1024 / sizeof(T);
        decoder_arena_size = (size_t)15 * 1024 * 1024 * 1024 / sizeof(T);

        encoder_fetcher_mode = FETCH_ON_DEMAND;
        decoder_fetcher_mode = FETCH_ON_DEMAND;

        profiling = true;
        disk_offload = true;
        offload_path = "/workspace/weights.dat";
    }

    void print() const
    {
        // TODO: replace with FT_LOG
        std::cout << "encoder_arena_size: " << encoder_arena_size << std::endl
                  << "decoder_arena_size: " << decoder_arena_size << std::endl
                  << "encoder_fetcher_mode: " << encoder_fetcher_mode << std::endl
                  << "decoder_fetcher_mode: " << decoder_fetcher_mode << std::endl;
    }

    size_t encoder_arena_size;
    size_t decoder_arena_size;

    int encoder_fetcher_mode;
    int decoder_fetcher_mode;
    bool profiling;
    bool disk_offload;
    std::string offload_path;
private:
    GlobalConfig() {}
};

}