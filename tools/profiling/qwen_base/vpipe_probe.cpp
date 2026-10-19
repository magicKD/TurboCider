// Standalone diagnostic consumer of the unmodified vpipe checkout.
// Real base weights, synthetic conditioning/latents, no sampler or VAE.
// This is a DiT component comparison, never a complete-request benchmark.
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Model = vpipe::genai::MetalQwenImage21Transformer;
using Buffer = vpipe::metal_compute::SharedBuffer;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

void require(bool value, const std::string& error) {
    if (!value) throw std::runtime_error(error);
}

Buffer random_bf16(vpipe::metal_compute::MetalCompute* mc, size_t n, unsigned seed) {
    auto result = mc->make_shared_buffer(n * 2);
    require(!result.empty(), "synthetic input allocation failed");
    auto* dst = static_cast<uint16_t*>(result.contents());
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float value = (float((seed >> 8) & 0xffff) / 32768.f - 1.f) * .5f;
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        dst[i] = uint16_t(bits >> 16);
    }
    return result;
}
} // namespace

extern "C" __attribute__((visibility("default")))
int tc_vpipe_qwen_probe(const char* checkpoint, int edit, int arm, int reps,
                        const char* output, void (*scope)(int)) {
    try {
        require(checkpoint && output && edit >= 0 && edit <= 1 && arm >= 0 && arm <= 2 &&
                reps >= 1 && reps <= 5, "invalid probe arguments");
        const fs::path root(output);
        require(fs::is_directory(root) && fs::is_empty(root), "fresh empty trial directory required");
        vpipe::Session session;
        auto* mc = session.metal_compute();
        require(mc != nullptr, "Metal backend unavailable");
        Model::Config config;
        require(Model::Config::read_dims(checkpoint, &config), "base checkpoint not recognized");
        require(config.n_layers == 32 && config.hidden == 4096 && config.mlp_ratio == 3 &&
                config.in_channels == 64 && config.out_channels == 64, "Qwen2.1 base shape required");
        config.ane_ffn = arm != 0;
        config.ane_qkv = arm == 2;
        const auto load_start = Clock::now();
        auto model = Model::load(checkpoint, mc, config, false);
        require(model != nullptr && model->quant_bits() == 0 && model->lora_slots() == 0,
                "original BF16 resident base without adapters required");
        const double load = std::chrono::duration<double>(Clock::now() - load_start).count();
        std::vector<uint8_t> slots(64, 0);
        std::vector<vpipe::genai::qi21::ImgBlock> blocks;
        if (edit) {
            slots.insert(slots.end(), 1024, 1); // 1024px reference ->4096 latent rows
            blocks.push_back({1, 64, 64});
        }
        slots.insert(slots.end(), 256, 1); // 512px target ->1024 latent rows
        blocks.push_back({1, 32, 32});
        vpipe::genai::qi21::Layout layout;
        std::string error;
        require(vpipe::genai::qi21::build_layout(slots, {}, blocks, &layout, &error), error);
        auto latents = random_bf16(mc, size_t(layout.image_len) * 64, 61);
        auto text = random_bf16(mc, slots.size() * 4096, 62);
        Model::KvCache cache;
        Model::Request request;
        request.latents = &latents;
        request.txt = &text;
        request.layout = &layout;
        request.kv = &cache;
        std::ofstream report(root / "report.json", std::ios::out);
        require(bool(report), "cannot create report");
        report << std::setprecision(17)
               << "{\"schema\":\"tc-vpipe-qwen-dit-v1\",\"complete_request\":false,"
               << "\"synthetic_inputs\":true,\"arm\":" << arm << ",\"edit\":" << edit
               << ",\"joint_rows\":" << layout.joint_len << ",\"decode_rows\":" << layout.target_len
               << ",\"reference_rows\":" << (edit ? 4096 : 0)
               << ",\"load_seconds\":" << load << ",\"samples\":[";
        for (int rep = 0; rep <= reps; ++rep) {
            for (int phase = 0; phase != 2; ++phase) {
                const int index = rep * 2 + phase;
                if (scope) scope(index);
                if (!phase) cache.clear();
                request.kv_mode = phase ? Model::KvMode::kCached : Model::KvMode::kFill;
                request.timestep = phase ? .5f : .75f;
                auto gpu0 = vpipe::perf_flops::totals(vpipe::perf_flops::kGpu);
                auto ane0 = vpipe::perf_flops::totals(vpipe::perf_flops::kAne);
                const auto start = Clock::now();
                auto result = model->forward(request, &error);
                const double wall = std::chrono::duration<double>(Clock::now() - start).count();
                auto gpu1 = vpipe::perf_flops::totals(vpipe::perf_flops::kGpu);
                auto ane1 = vpipe::perf_flops::totals(vpipe::perf_flops::kAne);
                require(!result.empty() && cache.populated() && result.byte_size() >= size_t(1024 * 64 * 2),
                        "forward/cache failed: " + error);
                const auto* raw = static_cast<const uint16_t*>(result.contents());
                std::vector<float> values(1024 * 64);
                for (size_t i = 0; i < values.size(); ++i) {
                    uint32_t bits = uint32_t(raw[i]) << 16;
                    std::memcpy(&values[i], &bits, sizeof(bits));
                    require(std::isfinite(values[i]), "nonfinite DiT output");
                }
                std::ofstream data(root / ("sample-" + std::to_string(index) + ".f32"), std::ios::binary);
                data.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
                require(bool(data), "output array write failed");
                if (index) report << ',';
                report << "{\"index\":" << index << ",\"rep\":" << rep
                       << ",\"warmup\":" << (rep == 0 ? "true" : "false")
                       << ",\"phase\":\"" << (phase ? "decode" : "prefill") << "\""
                       << ",\"seconds\":" << wall
                       << ",\"gpu_flops\":" << (gpu1.flops - gpu0.flops)
                       << ",\"ane_flops\":" << (ane1.flops - ane0.flops)
                       << ",\"ane_predict_records\":" << (ane1.records - ane0.records)
                       << ",\"ffn_armed\":" << (model->ane_armed() ? "true" : "false")
                       << ",\"qkv_armed\":" << (model->ane_qkv_armed() ? "true" : "false")
                       << ",\"ffn_failed\":" << (model->ane_gpu_only() ? "true" : "false")
                       << ",\"qkv_failed\":" << (model->ane_qkv_gpu_only() ? "true" : "false") << '}';
                report.flush();
            }
        }
        report << "],\"status\":\"complete\"}\n";
        require(bool(report), "report write failed");
        if (scope) scope(-3);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vpipe base probe failed: %s\n", error.what());
        return 1;
    }
}
