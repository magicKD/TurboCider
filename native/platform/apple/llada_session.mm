#include "bridge.hpp"

namespace tc {
std::unique_ptr<ModelSession> create_llada_image_native(const std::filesystem::path &);

namespace {
// Preserve lazy model loading. Python reference execution is development-only,
// never a process environment switch in the shipping adapter.
class LLaDASession final : public ModelSession {
    std::filesystem::path root_;
    std::unique_ptr<ModelSession> native_;

    ModelSession &native() {
        if (!native_) native_ = create_llada_image_native(root_);
        return *native_;
    }

  public:
    explicit LLaDASession(const std::filesystem::path &root)
        : root_(std::filesystem::absolute(root)) {
        require(std::filesystem::is_directory(root_), "LLaDA model directory is missing");
        for (const char *file : {"model_index.json",
                                "transformer/diffusion_pytorch_model.safetensors.index.json",
                                "text_encoder/model.safetensors.index.json"})
            require(std::filesystem::is_regular_file(root_ / file),
                    "LLaDA model asset is missing: " + std::string(file));
        for (const char *component : {"queryformer", "text_projection", "sigvq", "vae",
                                     "tokenizer", "scheduler"})
            require(std::filesystem::is_directory(root_ / component),
                    "LLaDA component is missing: " + std::string(component));
    }

    bool uses_parent_mlx() const override { return true; }

    void unload() override {
        if (native_) native_->unload();
        native_.reset();
    }

    RunResult generate(const Request &requested, const Event &event,
                       std::atomic<bool> &cancelled) override {
        checkpoint(cancelled);
        auto request = requested;
        if (request.execution == "auto") request.execution = "gpu";
        return native().generate(request, event, cancelled);
    }

    RunResult prepare(const Request &requested, bool warmup, const Event &event,
                      std::atomic<bool> &cancelled) override {
        checkpoint(cancelled);
        auto request = requested;
        if (request.execution == "auto") request.execution = "gpu";
        return native().prepare(request, warmup, event, cancelled);
    }
};
} // namespace

std::unique_ptr<ModelSession> create_llada_image(const std::filesystem::path &root) {
    return std::make_unique<LLaDASession>(root);
}
} // namespace tc
