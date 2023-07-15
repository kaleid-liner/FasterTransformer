#pragma once

#include <memory>
#include <iostream>

namespace fastertransformer {

enum class FetchType {
    GPU_ONLY,
    FETCH_ON_DEMAND,
    PREFETCH
};

class GlobalConfig {
public:
    static GlobalConfig& instance()
    {
        static GlobalConfig instance;
        return instance;
    }

    void setDefault();

    void print() const
    {
        // TODO: replace with FT_LOG
        std::cout << "encoder_arena_size: " << encoder_arena_size << std::endl
                  << "decoder_arena_size: " << decoder_arena_size << std::endl
                  << "encoder_fetcher_mode: " << int(encoder_fetcher_mode) << std::endl
                  << "decoder_fetcher_mode: " << int(decoder_fetcher_mode) << std::endl;
    }

    size_t arena_size;

    FetchType encoder_fetcher_mode;
    FetchType decoder_fetcher_mode;
    bool profiling;
    bool disk_offload;
    std::string offload_path;
    bool use_cache;
    // if saved_dir != "", load from saved_dir
    std::string saved_dir;
private:
    GlobalConfig() {}
};

}