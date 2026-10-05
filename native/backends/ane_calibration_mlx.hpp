#pragma once

#include "ane_calibration.hpp"
#include "ane_ffn.hpp"
#include "ane_smoothquant.hpp"
#include <set>
#include <cstring>
#include <iostream>

namespace tc::ane::calibration {

// Diagnostic owner-thread bridge. A request owns all statistics and samples;
// no process-wide tensor/model cache survives its scope. The ordinary path
// neither creates a sampler nor materializes any additional MLX arrays.
class ScopedCapture {
    inline static thread_local ScopedCapture *current_ = nullptr;
    ScopedCapture *previous_ = nullptr;
    std::unique_ptr<Sampler> sampler_;
    std::set<int> weight_layers_;
    std::filesystem::path parent_;
    std::string request_id_, source_identity_;
    int step_ = 0;
    bool finished_ = false;
  public:
    ScopedCapture(const Request &request, const std::filesystem::path &checkpoint,
                  const std::string &adapter_identity) {
        const char *raw = std::getenv("TURBOCIDER_ANE_CALIBRATION_DIR");
        if (!raw || !*raw) return;
        require((request.model == "z-image-turbo" || request.model == "qwen-image-2.1") &&
                request.hybrid_mlp_mode == "runtime" && request.width == 512 && request.height == 512,
                "bounded calibration requires an explicit 512px Runtime FFN request");
        const char *chunks = std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS");
        const char *profile = std::getenv("TURBOCIDER_RUNTIME_ANE_PROFILE");
        require(chunks && (std::string(chunks) == "0" || std::string(chunks) == "1") &&
                profile && std::string(profile) == "0",
                "bounded calibration requires fixed chunks=0 or 1 and profile=0; sampling must not train scheduler timing");
        const char *s1 = std::getenv("TURBOCIDER_PRIVATE_ANE_S1_PROFILE");
        require(!s1 || !*s1, "capture and an experimental S1 profile must use separate requests");
        parent_ = raw;
        require(parent_.is_absolute() && parent_ == parent_.lexically_normal(),
                "bounded calibration directory must be absolute and normalized");
        source_identity_ = smoothquant::checkpoint_fingerprint(checkpoint);
        request_id_ = std::to_string(::getpid()) + "-" + std::to_string(Clock::now().time_since_epoch().count());
        Metadata metadata;
        metadata.request_id = request_id_; metadata.model_id = request.model;
        metadata.model_fingerprint = source_identity_; metadata.identity_kind = "canonical-stat-identity";
        metadata.recipe = capture_recipe; metadata.seed = request.seed;
        metadata.width = request.width; metadata.height = request.height; metadata.total_steps = request.steps;
        metadata.reference_count = int(request.inputs.size());
        metadata.reference_size = request.inputs.empty() ? 0 : request.qwen21_reference_size;
        metadata.execution_route = "requested-runtime-ffn";
        const auto configured = [](const char *key, const char *fallback) {
            const char *value = std::getenv(key);
            return std::string(value && *value ? value : fallback);
        };
        metadata.configured_backend = configured("TURBOCIDER_ANE_BACKEND","public");
        metadata.runtime_recipe = configured("TURBOCIDER_PRIVATE_ANE_DATA_PATH","fp16") +
            ":channels=" + configured("TURBOCIDER_PRIVATE_ANE_CHANNELS","auto");
        // Zero chunks selects the family's complete GPU FFN. A split capture
        // can contain GPU fallbacks or earlier quantized blocks; preserve that
        // distinction rather than qualifying either trajectory as a baseline.
        metadata.actual_execution = std::string(chunks) == "0"
            ? "forced-gpu-runtime-boundary" : "unknown-candidate-trajectory";
        for (size_t i = 0; i < request.loras.size(); ++i)
            metadata.loras.push_back({std::filesystem::path(request.loras[i].path).filename().string(),
                smoothquant::fingerprint_adapter(adapter_identity,i), request.loras[i].strength});
        Config config; config.enabled = true;
        config.layers.clear();
        for (int layer = 0; layer < 32; ++layer) config.layers.push_back(layer);
        config.rows_per_point = 8;
        config.steps = {0,request.steps / 2,request.steps - 1};
        std::sort(config.steps.begin(), config.steps.end());
        config.steps.erase(std::unique(config.steps.begin(),config.steps.end()),config.steps.end());
        sampler_ = std::make_unique<Sampler>(std::move(config),std::move(metadata));
        previous_ = current_; current_ = this;
    }
    ScopedCapture(const ScopedCapture &) = delete;
    ScopedCapture &operator=(const ScopedCapture &) = delete;
    ~ScopedCapture() { if (sampler_) current_ = previous_; }
    bool enabled() const { return bool(sampler_); }
    bool wants(int layer) const { return sampler_ && sampler_->wants(layer,step_); }
    void step(int value) { step_ = value; }
    static ScopedCapture *current() { return current_; }
    void observe(int layer, const Tensor &input, const Weights &weights,
                 const std::vector<std::string> &gate_up, Point point) {
        if (!wants(layer)) return;
        tc::require(input.ndim() == 3 && input.shape(0) == 1,
                    "bounded calibration needs a batch-one FFN input");
        const size_t rows = input.shape(1), hidden = input.shape(2);
        if (!weight_layers_.contains(layer)) {
            std::optional<Tensor> maximum;
            for (const auto &prefix : gate_up) {
                tc::require(!weights.quantized(prefix) && !weights.convrot(prefix) && !weights.nvfp4(prefix),
                            "bounded calibration supports dense local checkpoint projections");
                const auto &weight = weights.at(prefix + ".weight");
                tc::require(weight.ndim() == 2 && size_t(weight.shape(1)) == hidden,
                            "bounded calibration weight/input geometry mismatch");
                // Do not materialize abs(W) for an entire checkpoint matrix.
                // Reduce at most 128 output rows, evaluating each H-vector
                // before the next batch so the lazy graph cannot retain them.
                for (int first = 0; first < weight.shape(0); first += 128) {
                    auto batch = slice_axis(weight,0,first,std::min(first+128,weight.shape(0)));
                    auto current = mx::astype(mx::max(mx::abs(batch),0),mx::float32);
                    maximum = maximum ? mx::maximum(*maximum,current) : current;
                    mx::eval(*maximum);
                }
            }
            tc::require(bool(maximum), "bounded calibration requires gate/up weights");
            mx::eval(*maximum);
            sampler_->set_weight_channel_max(layer,
                std::span<const float>(maximum->data<float>(),hidden),source_identity_);
            weight_layers_.insert(layer);
        }
        point.layer = layer; point.step = step_;
        sampler_->observe_rows(std::move(point),rows,hidden,
            [&](size_t first, size_t count, std::span<uint16_t> destination) {
                auto batch = mx::contiguous(mx::astype(slice_axis(input,1,int(first),int(first+count)),mx::float16));
                mx::eval(batch);
                std::memcpy(destination.data(),batch.data<mx::float16_t>(),destination.size_bytes());
            });
    }
    void finish() {
        if (!sampler_ || finished_) return;
        tc::require(!sampler_->records().empty(), "bounded calibration did not observe any Runtime FFN inputs");
        std::filesystem::create_directories(parent_);
        const auto directory = parent_ / request_id_;
        sampler_->write(directory);
        finished_ = true;
        std::cerr << "{\"ane_calibration_capture\":" << quote(directory.string())
                  << ",\"complete\":" << (sampler_->complete() ? "true" : "false")
                  << ",\"not_performance\":true,\"quantization_pending\":true}\n";
    }
};
} // namespace tc::ane::calibration
