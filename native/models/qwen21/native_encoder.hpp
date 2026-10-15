#pragma once
#include "vision.hpp"
#include "text_encoder.hpp"
#include <memory>

namespace tc::qwen21 {
inline bool native_encoder_requested() {
    const auto value=[] {const char* raw=std::getenv("TURBOCIDER_QWEN21_ENCODER_BACKEND");return std::string(raw?raw:"mlx");}();
    require(value=="mlx" || value=="metal","encoder backend requires mlx or metal");
    static const auto recipe=value;require(value==recipe,"encoder backend changed within a cached process");return value=="metal";
}
inline bool native_encoder_vision_bf16() {
    const auto value=[] {const char* raw=std::getenv("TURBOCIDER_QWEN21_NATIVE_VISION_PRECISION");return std::string(raw?raw:"fp32");}();
    require(value=="fp32" || value=="bf16","native vision precision requires fp32 or bf16");
    static const auto recipe=value;require(value==recipe,"native vision precision changed within a cached process");return value=="bf16";
}

inline std::filesystem::path native_encoder_ane_cache() {
    const char* raw=std::getenv("TURBOCIDER_QWEN21_NATIVE_ENCODER_ANE_CACHE");
    const std::string value=raw?raw:"";static const auto recipe=value;
    require(recipe==value,"native encoder ANE recipe changed within a cached process");
    require(value.empty() || std::filesystem::path(value).is_absolute(),"native encoder ANE requires explicit absolute owned cache");
    return value;
}
struct NativeEncoderStats {
    uint64_t vision_layers=0, language_layers=0, graph_submissions=0, dispatches=0;
    uint64_t future_weight_windows=0, skipped_future_windows=0;
    double preparation_seconds=0, source_ready_seconds=0, gpu_wait_seconds=0;
    uint64_t ane_ffns=0, ane_chunks=0, ane_padded_rows=0;
    double ane_producer_seconds=0, ane_branch_seconds=0, ane_consumer_seconds=0;
    bool physically_complete=false;
};
// Model arithmetic is recorded into owned Metal graphs, including vision.
// MLX is used only for checkpoint readers, preprocessing and produced-buffer
// boundaries. No MLX attention/norm/projection/FFN is called by this executor.
class NativeEncoder {
public:
    NativeEncoder();
    ~NativeEncoder();
    NativeEncoder(const NativeEncoder&)=delete;
    NativeEncoder& operator=(const NativeEncoder&)=delete;
    VisionFeatures vision(Weights&, const Tensor& patches, int grid_height, int grid_width,
        const Event&, std::atomic<bool>&, VisionConfig = {});
    Tensor language(Weights&, const Tensor& embeddings, const Tensor& positions,
        int valid_tokens, const std::vector<Tensor>& deepstack,
        const Event&, std::atomic<bool>&, TextConfig = {});
    NativeEncoderStats statistics() const;
    bool healthy() const;
    void reset_statistics();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tc::qwen21
