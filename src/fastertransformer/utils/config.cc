#include "src/fastertransformer/utils/config.h"

namespace fastertransformer {

void GlobalConfig::setDefault()
{
    arena_size = (size_t)5 * 1024 * 1024 * 1024;
    arena_size = (size_t)15 * 1024 * 1024 * 1024;

    encoder_fetcher_mode = FetchType::FETCH_ON_DEMAND;
    decoder_fetcher_mode = FetchType::FETCH_ON_DEMAND;

    profiling = true;
    disk_offload = true;

    offload_path = "/workspace/weights.dat";
    //saved_dir = "/data/ft-switch-base-8/1-gpu/";
    saved_dir = "";
}

}
