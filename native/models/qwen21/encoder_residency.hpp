#pragma once
#include "../../core/common.hpp"
#include "../../backends/ane_memory.hpp"
#include <sys/stat.h>

namespace tc::qwen21 {
struct EncoderSourceGeneration {
    std::string identity;
    uint64_t bytes = 0;
};

// Local read-only checkpoint generation, NOT a full payload signature or
// immutable lease. ctime catches ordinary same-size/mtime-preserving writes.
inline EncoderSourceGeneration encoder_source_generation(const std::filesystem::path &path) {
    const auto canonical=std::filesystem::canonical(path);
    struct stat info{};
    require(::stat(canonical.c_str(),&info)==0 && S_ISREG(info.st_mode) && info.st_size>0,
        "Qwen encoder source must be a nonempty regular file");
    std::string identity=canonical.string();
    for(auto value:{int64_t(info.st_dev),int64_t(info.st_ino),int64_t(info.st_size),
                   int64_t(info.st_mtimespec.tv_sec),int64_t(info.st_mtimespec.tv_nsec),
                   int64_t(info.st_ctimespec.tv_sec),int64_t(info.st_ctimespec.tv_nsec)})
        identity+=":"+std::to_string(value);
    return {std::move(identity),uint64_t(info.st_size)};
}

inline ane::MemoryDecision admit_encoder_weights(const ane::MemoryObservation &observation,
        uint64_t source_bytes,uint64_t upcoming_model_bytes) {
    constexpr uint64_t gib=uint64_t(1)<<30;
    if(!observation.available)return {ane::MemoryDenial::Unavailable,0};
    if(observation.physical_bytes<48*gib)return {ane::MemoryDenial::SystemReserve,0};
    if(!source_bytes || source_bytes>20*gib)return {ane::MemoryDenial::OptionalLimit,0};
    // The arrays already contribute to process/MLX/free observations. Charge
    // only future model payload and a separate 4GiB workspace allowance here;
    // the retained source has its own 20GiB logical ceiling. Not a RAM cap.
    if(upcoming_model_bytes>UINT64_MAX-4*gib)return {ane::MemoryDenial::GrowthLimit,0};
    return ane::admit_memory(observation,{4*gib,observation.physical_bytes},0,
        upcoming_model_bytes+4*gib);
}
} // namespace tc::qwen21
