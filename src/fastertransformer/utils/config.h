#pragma once

#include <memory>
#include <iostream>
#include "cutlass/numeric_types.h"

namespace fastertransformer {

enum class FetchType {
    GPU_ONLY,
    FETCH_ON_DEMAND,
    PREFETCH
};

enum class QuantType {
    NO_QUANT,
    WEIGHT_ONLY,
    SMOOTH_QUANT
};

class GlobalConfig {
public:
    using quant_t = cutlass::uint4b_t;

    using weight_t = float;

    using act_t = float;

    static GlobalConfig& instance()
    {
        static GlobalConfig instance;
        return instance;
    }

    void setDefault()
    {
        arena_size = (size_t)20 * 1024 * 1024 * 1024;

        encoder_fetcher_mode = FetchType::FETCH_ON_DEMAND;
        decoder_fetcher_mode = FetchType::PREFETCH;

        profiling = true;

        offload_path = "/data/ft-switch-base-8/1-gpu/";
        disk_offload = false;

        load_from_cpp = true;

        use_cache = true;

        quant_mode = QuantType::NO_QUANT;

        vocab_size = 32128;
    }

    void print() const
    {
        // TODO: replace with FT_LOG
        std::cout << "arena_size: " << arena_size << std::endl
                  << "encoder_fetcher_mode: " << int(encoder_fetcher_mode) << std::endl
                  << "decoder_fetcher_mode: " << int(decoder_fetcher_mode) << std::endl;
    }


    size_t arena_size;

    FetchType encoder_fetcher_mode;
    FetchType decoder_fetcher_mode;

    bool profiling;

    std::string offload_path;
    bool disk_offload;

    bool load_from_cpp;

    bool use_cache;

    QuantType quant_mode;

    int64_t vocab_size;  // workaround for missing vocab_size arg in encoder
private:
    GlobalConfig() {}
};

}