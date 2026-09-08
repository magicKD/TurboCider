#include "bridge.hpp"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace tc {
std::unique_ptr<ModelSession> create_llada_image_native(const std::filesystem::path &);

namespace {

static bool use_reference_worker(const Request &request) {
    const char *forced = std::getenv("TURBOCIDER_LLADA_REFERENCE");
    return forced && *forced && std::string(forced) != "0";
}

static std::filesystem::path environment_path(const char *name) {
    const char *value = std::getenv(name);
    return value && *value ? std::filesystem::path(value) : std::filesystem::path{};
}

static void require_regular(const std::filesystem::path &path, const std::string &label) {
    require(!path.empty() && std::filesystem::is_regular_file(path),
            "LLaDA " + label + " is missing: " + path.string());
}

static void require_directory(const std::filesystem::path &path, const std::string &label) {
    require(!path.empty() && std::filesystem::is_directory(path),
            "LLaDA " + label + " is missing: " + path.string());
}

static std::filesystem::path library_directory() {
    Dl_info info{};
    require(dladdr(reinterpret_cast<void *>(&library_directory), &info) != 0 &&
                info.dli_fname,
            "cannot locate TurboCider library");
    return std::filesystem::weakly_canonical(info.dli_fname).parent_path();
}

struct LLaDAConfig {
    std::filesystem::path python;
    std::filesystem::path worker;
    std::filesystem::path reference_root;
};

static LLaDAConfig load_config() {
    LLaDAConfig config;
    config.python = environment_path("TURBOCIDER_LLADA_PYTHON");
    config.worker = environment_path("TURBOCIDER_LLADA_WORKER");
    config.reference_root = environment_path("TURBOCIDER_LLADA_SOURCE");
    require(!config.worker.empty(),
            "LLaDA diagnostic worker requires TURBOCIDER_LLADA_WORKER");
    require(!config.python.empty(),
            "LLaDA diagnostic worker requires TURBOCIDER_LLADA_PYTHON");
    require(!config.reference_root.empty(),
            "LLaDA diagnostic worker requires TURBOCIDER_LLADA_SOURCE");
    require_regular(config.python, "Python executable");
    require_regular(config.worker, "persistent worker");
    require_regular(config.reference_root / "src/__init__.py", "reference pipeline");
    return config;
}

static std::string line_tail(const std::string &log) {
    constexpr size_t limit = 16 * 1024;
    return log.size() <= limit ? log : log.substr(log.size() - limit);
}

static void write_all(int fd, const std::string &value) {
    size_t offset = 0;
    while (offset < value.size()) {
        auto written = ::write(fd, value.data() + offset, value.size() - offset);
        if (written < 0 && errno == EINTR)
            continue;
        require(written > 0, "LLaDA worker pipe write failed");
        offset += static_cast<size_t>(written);
    }
}

static NSDictionary *decode_worker_message(const std::string &line) {
    if (line.empty() || line.front() != '{')
        return nil;
    NSData *data = [NSData dataWithBytes:line.data() length:line.size()];
    id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [value isKindOfClass:NSDictionary.class] ? value : nil;
}

class LLaDAWorker {
    LLaDAConfig config_;
    std::filesystem::path root_;
    std::string execution_;
    std::filesystem::path manifest_;
    pid_t pid_ = -1;
    int input_ = -1;
    int output_ = -1;
    std::string pending_;
    std::string log_;

    void stop_process() noexcept {
        if (pid_ >= 0) {
            ::kill(pid_, SIGTERM);
            int status = 0;
            for (int i = 0; i < 50; ++i) {
                if (::waitpid(pid_, &status, WNOHANG) == pid_)
                    break;
                usleep(10000);
            }
            if (::waitpid(pid_, &status, WNOHANG) == 0) {
                ::kill(pid_, SIGKILL);
                ::waitpid(pid_, &status, 0);
            }
        }
        if (input_ >= 0)
            ::close(input_);
        if (output_ >= 0)
            ::close(output_);
        pid_ = -1;
        input_ = output_ = -1;
        pending_.clear();
    }

    NSDictionary *next_message(std::atomic<bool> *cancelled) {
        for (;;) {
            if (cancelled && cancelled->load()) {
                stop_process();
                throw Cancelled();
            }
            auto newline = pending_.find('\n');
            if (newline != std::string::npos) {
                auto line = pending_.substr(0, newline);
                pending_.erase(0, newline + 1);
                if (auto message = decode_worker_message(line))
                    return message;
                if (!line.empty())
                    log_.append(line).append("\n");
                continue;
            }
            pollfd descriptor{output_, POLLIN | POLLHUP, 0};
            int ready = ::poll(&descriptor, 1, 100);
            if (ready < 0 && errno == EINTR)
                continue;
            require(ready >= 0, "LLaDA worker pipe poll failed");
            if (!ready)
                continue;
            char buffer[8192];
            auto count = ::read(output_, buffer, sizeof(buffer));
            if (count > 0) {
                pending_.append(buffer, static_cast<size_t>(count));
                continue;
            }
            int status = 0;
            ::waitpid(pid_, &status, 0);
            pid_ = -1;
            throw std::runtime_error("LLaDA worker exited: " + line_tail(log_));
        }
    }

    void start() {
        int input_pipe[2] = {-1, -1};
        int output_pipe[2] = {-1, -1};
        require(::pipe(input_pipe) == 0 && ::pipe(output_pipe) == 0,
                "cannot create LLaDA worker pipes");
        std::vector<std::string> args = {
            config_.python.string(), config_.worker.string(),
            "--model-root", root_.string(),
            "--reference-root", config_.reference_root.string(),
            "--device", "mps", "--dtype", "bf16",
        };
        if (execution_ == "gpu_ane") {
            require_regular(manifest_, "compiled Core ML manifest");
            auto library = library_directory() / "libturbocider.dylib";
            require_regular(library, "native Core ML bridge library");
            args.insert(args.end(), {"--ane-manifest", manifest_.string(),
                                     "--native-library", library.string()});
        }
        std::vector<char *> argv;
        for (auto &arg : args)
            argv.push_back(arg.data());
        argv.push_back(nullptr);
        std::vector<std::string> environment;
        for (char **item = environ; item && *item; ++item) {
            std::string value(*item);
            if (!value.starts_with("PYTHONPATH=") &&
                !value.starts_with("TOKENIZERS_PARALLELISM="))
                environment.push_back(std::move(value));
        }
        environment.push_back("TOKENIZERS_PARALLELISM=false");
        std::vector<char *> envp;
        for (auto &value : environment)
            envp.push_back(value.data());
        envp.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        require(posix_spawn_file_actions_init(&actions) == 0,
                "cannot initialize LLaDA spawn actions");
        posix_spawn_file_actions_adddup2(&actions, input_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDERR_FILENO);
        posix_spawn_file_actions_addclose(&actions, input_pipe[1]);
        posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
        pid_t child = -1;
        auto status = posix_spawn(&child, config_.python.c_str(), &actions, nullptr,
                                  argv.data(), envp.data());
        posix_spawn_file_actions_destroy(&actions);
        require(status == 0 && child > 0,
                "cannot spawn LLaDA worker: " + std::string(std::strerror(status)));
        ::close(input_pipe[0]);
        ::close(output_pipe[1]);
        input_ = input_pipe[1];
        output_ = output_pipe[0];
        pid_ = child;
#ifdef F_SETNOSIGPIPE
        ::fcntl(input_, F_SETNOSIGPIPE, 1);
#endif
        auto ready = next_message(nullptr);
        auto type = string_value(ready, @"type");
        if (type == "error" || type == "fatal")
            throw std::runtime_error("LLaDA worker startup failed: " +
                                     string_value(ready, @"error", line_tail(log_)));
        require(type == "ready", "LLaDA worker did not become ready: " + line_tail(log_));
    }

  public:
    LLaDAWorker(LLaDAConfig config, std::filesystem::path root, std::string execution,
                std::filesystem::path manifest)
        : config_(std::move(config)), root_(std::move(root)), execution_(std::move(execution)),
          manifest_(std::move(manifest)) {
        start();
    }
    ~LLaDAWorker() { stop_process(); }
    LLaDAWorker(const LLaDAWorker &) = delete;
    LLaDAWorker &operator=(const LLaDAWorker &) = delete;

    NSDictionary *generate(const Request &request, const std::filesystem::path &output,
                           const Event &event, std::atomic<bool> &cancelled) {
        NSMutableDictionary *payload = [@{
            @"action" : @"generate",
            @"prompt" : @(request.prompt.c_str()),
            @"output" : @(output.string().c_str()),
            @"generation_mode" : request.operation == "image.edit" ? @"editing" : @"text",
            @"width" : @(request.width),
            @"height" : @(request.height),
            @"steps" : @(request.steps),
            @"guidance_scale" : @1.0,
            @"seed" : @(request.seed),
        } mutableCopy];
        if (request.operation == "image.edit")
            payload[@"input_image"] = @(request.inputs.front().path.c_str());
        write_all(input_, json(payload) + "\n");
        for (;;) {
            auto message = next_message(&cancelled);
            auto type = string_value(message, @"type");
            if (type == "progress") {
                event(string_value(message, @"phase", "denoise"),
                      [message[@"completed"] intValue], [message[@"total"] intValue]);
            } else if (type == "result") {
                id result = message[@"result"];
                require([result isKindOfClass:NSDictionary.class],
                        "LLaDA worker returned an invalid result");
                return result;
            } else if (type == "error" || type == "fatal") {
                throw std::runtime_error("LLaDA worker error: " +
                                         string_value(message, @"error", line_tail(log_)));
            }
        }
    }
};

class LLaDASession final : public ModelSession {
    std::filesystem::path root_;
    std::optional<LLaDAConfig> config_;
    std::unique_ptr<ModelSession> native_;
    std::unique_ptr<LLaDAWorker> worker_;
    std::string worker_execution_;
    std::filesystem::path worker_manifest_;
    uint64_t sequence_ = 0;

  public:
    explicit LLaDASession(const std::filesystem::path &root)
        : root_(std::filesystem::absolute(root)) {
        require_directory(root_, "model directory");
        require_regular(root_ / "model_index.json", "model index");
        require_regular(root_ / "transformer/diffusion_pytorch_model.safetensors.index.json",
                        "transformer shard index");
        require_regular(root_ / "text_encoder/model.safetensors.index.json",
                        "text encoder shard index");
        for (auto component : {"queryformer", "text_projection", "sigvq", "vae",
                               "tokenizer", "scheduler"})
            require_directory(root_ / component, std::string(component));
    }

    bool uses_parent_mlx() const override { return true; }
    bool uses_parent_mlx(const Request &request) const override {
        return !use_reference_worker(request);
    }
    void unload() override {
        if (native_)
            native_->unload();
        native_.reset();
        worker_.reset();
    }

    RunResult generate(const Request &requested, const Event &event,
                       std::atomic<bool> &cancelled) override {
        auto request = requested;
        if (request.execution == "auto")
            request.execution = "gpu";
        if (!use_reference_worker(request)) {
            worker_.reset();
            if (!native_)
                native_ = create_llada_image_native(root_);
            return native_->generate(request, event, cancelled);
        }
        if (!config_)
            config_ = load_config();
        if (native_) {
            native_->unload();
            native_.reset();
        }
        auto plan = make_plan(request);
        require(request.execution == "gpu" || request.execution == "gpu_ane",
                "LLaDA execution must be gpu or gpu_ane");
        require(!request.output.empty() &&
                    std::filesystem::path(request.output).extension() == ".png",
                "LLaDA output must be a .png file");
        auto requested_manifest = request.execution == "gpu_ane"
                                      ? std::filesystem::absolute(request.ane_manifest)
                                      : std::filesystem::path{};
        if (worker_ && (worker_execution_ != request.execution ||
                        worker_manifest_ != requested_manifest))
            worker_.reset();
        if (!worker_) {
            event("model_load", 0, 1);
            worker_ = std::make_unique<LLaDAWorker>(*config_, root_, request.execution,
                                                    requested_manifest);
            worker_execution_ = request.execution;
            worker_manifest_ = requested_manifest;
            event("model_load", 1, 1);
        }
        auto output = std::filesystem::absolute(request.output);
        std::error_code error;
        std::filesystem::create_directories(output.parent_path(), error);
        require(!error, "cannot create LLaDA output directory");
        auto temporary = output;
        temporary += ".tmp." + std::to_string(static_cast<long long>(::getpid())) + "." +
                     std::to_string(++sequence_) + ".png";
        std::filesystem::remove(temporary, error);
        NSDictionary *metrics = nil;
        try {
            metrics = worker_->generate(request, temporary, event, cancelled);
        } catch (...) {
            std::filesystem::remove(temporary, error);
            if (cancelled.load())
                worker_.reset();
            throw;
        }
        require(std::filesystem::is_regular_file(temporary),
                "LLaDA worker completed without producing an image");
        std::filesystem::rename(temporary, output, error);
        require(!error, "cannot atomically publish LLaDA image: " + error.message());
        event("export", 1, 1);
        NSMutableDictionary *result = [metrics mutableCopy];
        result[@"model"] = @"llada-image-turbo";
        result[@"output"] = @(output.string().c_str());
        result[@"execution"] = @(request.execution.c_str());
        result[@"acceleration_selection"] = request.execution == "gpu_ane"
            ? @"gpu_ane: explicit LLaDA Core ML FFN prefix candidate"
            : @"gpu: persistent official Diffusers MPS path";
        return native_run_result(result, request, plan);
    }

    RunResult prepare(const Request &requested, bool warmup, const Event &event,
                      std::atomic<bool> &cancelled) override {
        auto request = requested;
        if (request.execution == "auto")
            request.execution = "gpu";
        require(!use_reference_worker(request),
                "LLaDA compatibility-worker preparation is not part of the native executor");
        worker_.reset();
        if (!native_)
            native_ = create_llada_image_native(root_);
        return native_->prepare(request, warmup, event, cancelled);
    }
};

} // namespace

std::unique_ptr<ModelSession> create_llada_image(const std::filesystem::path &root) {
    return std::make_unique<LLaDASession>(root);
}

} // namespace tc
