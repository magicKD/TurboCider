#pragma once
#include "common.hpp"
#include <optional>
namespace tc {
struct ExecutionPlan {
    Request request;
    Recipe recipe;
    std::optional<uint64_t> memory_estimate_bytes;
};
ExecutionPlan make_plan(const Request &);
struct HybridMetrics {
    double load_seconds = 0, prediction_seconds = 0;
    uint64_t calls = 0, copied_bytes = 0;
    int bucket = 0;
};
struct LoadResult {
    uint64_t weight_bytes = 0, active_bytes = 0;
};
struct Timings {
    double wall = 0, text = 0, image = 0, hybrid = 0, denoise = 0, decode = 0;
};
struct RunResult {
    bool prepared = false, warmup = false, prompt_cache_hit = false;
    std::string selection;
    Request request;
    ExecutionPlan plan;
    int text_tokens = 0, valid_text_tokens = 0, total_tokens = 0, reference_tokens = 0,
        actual_steps = 0;
    Timings timings;
    uint64_t active_bytes = 0, peak_bytes = 0;
    std::optional<HybridMetrics> hybrid;
};
class ModelSession {
  public:
    virtual ~ModelSession() = default;
    virtual RunResult generate(const Request &, const Event &, std::atomic<bool> &) = 0;
    virtual LoadResult load(const Event &, std::atomic<bool> &) {
        throw std::runtime_error("explicit loading unavailable");
    }
    virtual void unload() = 0;
    virtual RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) {
        throw std::runtime_error("preparation unavailable");
    }
};
struct ModelDescriptor {
    std::string id, name;
    bool executable = false;
    std::vector<std::string> operations, inputs, roles;
    int max_images = 0;
    std::string output;
    int steps = 4, frames = 1, width = 512, height = 512;
    bool weight_validation_pending = false;
};
struct ModelModule {
    std::string id;
    std::function<Recipe()> recipe;
    std::function<void(const Request &)> validate;
    std::function<std::unique_ptr<ModelSession>(const std::filesystem::path &)> create;
    std::function<ModelDescriptor()> describe;
};
const ModelModule &module_for(const std::string &);
std::vector<ModelDescriptor> describe_modules();
Recipe model_recipe(const std::string &);
void validate_recipe(const Recipe &);
std::vector<float> flux_sigmas(int, int);
} // namespace tc
