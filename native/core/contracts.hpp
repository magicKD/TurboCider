#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tc {
struct InputAsset {
    std::string kind, role, path, text, audio_path;
    bool include_audio = true;
    float strength = 0.75f;
};
struct LoRAAsset {
    std::string path;
    float strength = 1.0f;
    std::string role = "transformer";
};
struct Request {
    std::string model = "flux2-klein-4b";
    std::string operation = "image.generate";
    std::string prompt, output, execution = "gpu", dump, ane_manifest;
    std::string profile, residency = "resident";
    std::string profile_identity;
    std::string model_variant = "auto";
    uint64_t memory_budget_bytes = 0, allocator_cache_bytes = 512ull << 20;
    int warmup_iterations = 0;
    std::vector<InputAsset> inputs;
    std::vector<LoRAAsset> loras;
    int width = 512, height = 512, steps = 4, frames = 1, fps = 24;
    uint64_t seed = 42;
    /* LTX native currently emits video-only output.  Other model families
     * keep audio enabled by default for schema-2 compatibility.  The JSON
     * parser applies the model-specific default before validation. */
    bool dynamic_text = true, allow_approximation = false, audio = true;
};
struct Stage {
    std::string id;
    std::vector<std::string> dependencies;
    int iterations = 1;
};
struct Recipe {
    std::string model;
    std::vector<Stage> stages;
    bool executable = false;
};
}
