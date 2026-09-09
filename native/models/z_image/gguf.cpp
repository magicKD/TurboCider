#include "../../runtime/session.hpp"
#include "z_image.hpp"
#include "../../core/gguf.hpp"
#include <algorithm>
#include <cctype>

namespace tc {
namespace {
namespace fs = std::filesystem;

class NativeGGUF final : public ModelSession {
    fs::path root_;
    std::vector<fs::path> checkpoints_;
    fs::path selected_;
    std::unique_ptr<ZImage> session_;

    ZImage &select(const std::string &variant) {
        std::vector<fs::path> matches;
        auto lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });
            return s;
        };
        for (const auto &path : checkpoints_) {
            if (variant.empty() || variant == "auto" || variant == "z-image-turbo-gguf" ||
                lower(path.filename().string()).find(lower(variant)) != std::string::npos)
                matches.push_back(path);
        }
        require(matches.size() == 1,
                matches.empty() ? "no matching GGUF transformer found" :
                "multiple GGUF transformers found; select an explicit model_variant or checkpoint file");
        if (!session_ || selected_ != matches.front()) {
            unload();
            selected_ = matches.front();
            validate_native_gguf(selected_);
            session_ = std::make_unique<ZImage>(root_, "z-image-turbo-gguf", selected_);
        }
        return *session_;
    }

  public:
    explicit NativeGGUF(const fs::path &path) {
        // Resolve components relative to the user's selected location, even
        // when the checkpoint itself is a symlink to a shared weight store.
        auto absolute = fs::absolute(path).lexically_normal();
        require(fs::exists(absolute), "GGUF model path does not exist: " + absolute.string());
        root_ = fs::is_regular_file(absolute) ? absolute.parent_path() : absolute;
        if (fs::is_regular_file(absolute)) {
            require(absolute.extension() == ".gguf", "expected a GGUF checkpoint");
            checkpoints_.push_back(absolute);
        } else {
            for (const auto &entry : fs::recursive_directory_iterator(root_))
                if (entry.is_regular_file() && entry.path().extension() == ".gguf" &&
                    entry.path().filename().string().find("lora") == std::string::npos)
                    checkpoints_.push_back(entry.path());
        }
        require(!checkpoints_.empty(), "no GGUF transformer found in model root");
        // Components must belong to the explicitly selected model root.
        // Never discover sibling projects or depend on shell environment.
        require(fs::is_directory(root_ / "tokenizer"),
                "native GGUF requires tokenizer/ in the selected model root");
    }
    LoadResult load(const Event &event, std::atomic<bool> &cancelled) override {
        return select("auto").load(event, cancelled);
    }
    void unload() override {
        if (session_) session_->unload();
        session_.reset();
        selected_.clear();
    }
    RunResult prepare(const Request &r, bool warmup, const Event &event,
                      std::atomic<bool> &cancelled) override {
        return select(r.model_variant).prepare(r, warmup, event, cancelled);
    }
    RunResult generate(const Request &r, const Event &event,
                       std::atomic<bool> &cancelled) override {
        return select(r.model_variant).generate(r, event, cancelled);
    }
};
} // namespace

std::unique_ptr<ModelSession> create_z_image_gguf(const std::filesystem::path &root) {
    return std::make_unique<NativeGGUF>(root);
}
} // namespace tc
