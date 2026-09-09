#include "../../runtime/session.hpp"
#include "../../models/z_image/z_image.hpp"
#include "bridge.hpp"

#import <Foundation/Foundation.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <netinet/in.h>
#include <libproc.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>

namespace tc {
namespace {

namespace fs = std::filesystem;

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return char(std::tolower(c));
    });
    return value;
}

bool executable(const fs::path &path) {
    return fs::is_regular_file(path) && access(path.c_str(), X_OK) == 0;
}

std::vector<fs::path> split_path(const char *value) {
    std::vector<fs::path> result;
    if (!value)
        return result;
    std::string paths(value);
    size_t begin = 0;
    while (begin <= paths.size()) {
        auto end = paths.find(':', begin);
        auto item = paths.substr(begin, end == std::string::npos ? end : end - begin);
        if (!item.empty())
            result.emplace_back(item);
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return result;
}

fs::path executable_directory() {
    uint32_t size = 1024;
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        buffer.resize(size);
        require(_NSGetExecutablePath(buffer.data(), &size) == 0,
                "cannot resolve TurboCider executable path");
    }
    std::error_code error;
    auto resolved = fs::canonical(buffer.data(), error);
    return (error ? fs::absolute(buffer.data()) : resolved).parent_path();
}

fs::path find_sd_server(const fs::path &root) {
    std::vector<fs::path> candidates;
    if (const char *configured = std::getenv("TURBOCIDER_SD_CPP_BIN")) {
        fs::path value(configured);
        candidates.push_back(fs::is_directory(value) ? value / "sd-server" : value);
    }
    for (const auto &base : {root, executable_directory()}) {
        candidates.push_back(base / "sd-server");
        candidates.push_back(base / "bin" / "sd-server");
        candidates.push_back(base / "stable-diffusion.cpp" / "build" / "bin" / "sd-server");
    }
    for (const auto &directory : split_path(std::getenv("PATH")))
        candidates.push_back(directory / "sd-server");
    for (const auto &candidate : candidates)
        if (executable(candidate)) {
            std::error_code error;
            auto resolved = fs::canonical(candidate, error);
            return error ? fs::absolute(candidate) : resolved;
        }
    require(false,
            "compatible sd-server not found; set TURBOCIDER_SD_CPP_BIN to the managed "
            "stable-diffusion.cpp server binary");
    return {};
}

fs::path configured_component(const char *name) {
    if (const char *value = std::getenv(name)) {
        fs::path path(value);
        require(fs::is_regular_file(path), std::string(name) + " is not a regular file");
        return fs::canonical(path);
    }
    return {};
}

fs::path first_existing(const fs::path &root, const std::vector<fs::path> &relative) {
    for (const auto &name : relative) {
        auto path = root / name;
        if (fs::is_regular_file(path))
            return fs::canonical(path);
    }
    return {};
}

std::vector<fs::path> find_ggufs(const fs::path &root) {
    std::vector<fs::path> result;
    if (fs::is_regular_file(root)) {
        if (lower(root.extension().string()) == ".gguf")
            result.push_back(fs::canonical(root));
        return result;
    }
    require(fs::is_directory(root), "Z-Image GGUF model root is missing: " + root.string());
    for (const auto &entry : fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file() || lower(entry.path().extension().string()) != ".gguf")
            continue;
        auto name = lower(entry.path().filename().string());
        if (name.find("lora") == std::string::npos)
            result.push_back(fs::canonical(entry.path()));
    }
    std::sort(result.begin(), result.end());
    return result;
}

void validate_gguf_header(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot open GGUF checkpoint: " + path.string());
    char magic[4] = {};
    uint32_t version = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char *>(&version), sizeof(version));
    require(input.gcount() == sizeof(version) && std::string(magic, sizeof(magic)) == "GGUF",
            "invalid GGUF header: " + path.string());
    require(version >= 2 && version <= 3,
            "unsupported GGUF version " + std::to_string(version));
}

fs::path auto_gguf(const std::vector<fs::path> &values) {
    require(!values.empty(), "no GGUF transformer found in the Z-Image model root");
    if (values.size() == 1)
        return values.front();
    const std::vector<std::string> preference = {
        "q4_k_m", "q4_k_s", "iq4_nl", "iq4_xs", "q4_0", "q4_1",
        "q5_k_m", "q5_k_s", "q5_0", "q5_1", "q6_k", "q8_0",
        "q3_k_m", "q3_k_s", "q3_k_l", "q2_k"};
    for (const auto &token : preference)
        for (const auto &path : values)
            if (lower(path.filename().string()).find(token) != std::string::npos)
                return path;
    require(false,
            "multiple GGUF transformers found; set model_variant to a filename or quantization tag");
    return {};
}

fs::path select_gguf(const std::vector<fs::path> &values, const std::string &variant) {
    if (variant.empty() || variant == "auto" || variant == "z-image-turbo-gguf")
        return auto_gguf(values);
    auto wanted = lower(variant);
    std::vector<fs::path> matches;
    for (const auto &path : values) {
        auto filename = lower(path.filename().string());
        auto stem = lower(path.stem().string());
        if (filename == wanted || stem == wanted || filename.find(wanted) != std::string::npos)
            matches.push_back(path);
    }
    require(matches.size() == 1,
            matches.empty() ? "model_variant did not match a GGUF checkpoint: " + variant
                            : "model_variant matched multiple GGUF checkpoints: " + variant);
    return matches.front();
}

std::string quantization_label(const fs::path &path) {
    auto name = lower(path.filename().string());
    for (const auto *token : {"iq4_nl", "iq4_xs", "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s",
                              "q3_k_m", "q3_k_s", "q3_k_l", "q2_k", "q6_k", "q8_0",
                              "q4_0", "q4_1", "q5_0", "q5_1", "f16", "bf16", "f32"})
        if (name.find(token) != std::string::npos)
            return token;
    return "mixed_or_unspecified";
}

bool native_mlx_gguf_supported(const fs::path &path) {
    const auto label = quantization_label(path);
    return label == "q8_0" || label == "q4_0" || label == "q4_1" ||
           label == "f16" || label == "bf16" || label == "f32";
}

bool enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] && std::string(value) != "0";
}

bool native_component_layout(const fs::path &root) {
    const bool tokenizer = fs::is_directory(root / "tokenizer");
    const bool comfy = fs::is_regular_file(
                           root / "split_files/text_encoders/qwen_3_4b.safetensors") &&
                       fs::is_regular_file(root / "split_files/vae/ae.safetensors");
    const bool diffusers = fs::is_directory(root / "text_encoder") &&
                           fs::is_directory(root / "vae");
    return tokenizer && (comfy || diffusers);
}

fs::path find_native_component_root(const fs::path &gguf_root) {
    std::vector<fs::path> candidates;
    if (const char *configured = std::getenv("TURBOCIDER_Z_IMAGE_NATIVE_ROOT"))
        candidates.emplace_back(configured);
    auto directory = fs::is_directory(gguf_root) ? gguf_root : gguf_root.parent_path();
    candidates.push_back(directory);
    candidates.push_back(directory.parent_path() / "Comfy-Org-z_image_turbo");
    candidates.push_back(directory.parent_path() / "Tongyi-MAI-Z-Image-Turbo");
    for (const auto &candidate : candidates)
        if (native_component_layout(candidate))
            return fs::canonical(candidate);
    require(false,
            "native GGUF execution needs the Z-Image tokenizer, Qwen3 and VAE; set "
            "TURBOCIDER_Z_IMAGE_NATIVE_ROOT to a Comfy-Org or Tongyi component root");
    return {};
}

int free_loopback_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "cannot allocate sd-server socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    int status = bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address));
    socklen_t size = sizeof(address);
    if (status == 0)
        status = getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size);
    close(fd);
    require(status == 0, "cannot select a loopback port for sd-server");
    return ntohs(address.sin_port);
}

fs::path make_temporary_directory() {
    std::string value = "/private/tmp/turbocider-sdcpp-XXXXXX";
    std::vector<char> bytes(value.begin(), value.end());
    bytes.push_back('\0');
    auto result = mkdtemp(bytes.data());
    require(result != nullptr, "cannot create sd-server scratch directory");
    return result;
}

std::string safe_alias(std::string value) {
    value = fs::path(value).stem().string();
    for (char &c : value)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-')
            c = '_';
    while (!value.empty() && (value.front() == '_' || value.front() == '-'))
        value.erase(value.begin());
    while (!value.empty() && (value.back() == '_' || value.back() == '-'))
        value.pop_back();
    return value.empty() ? "lora" : value;
}

struct HTTPResult {
    NSInteger status = 0;
    NSData *__strong data = nil;
    NSError *__strong error = nil;
};

HTTPResult http(NSURLSession *session, NSString *method, NSURL *url, NSDictionary *body,
                std::atomic<bool> &cancelled, NSTimeInterval timeout = 30) {
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = method;
    request.timeoutInterval = timeout;
    if (body) {
        NSError *serialization_error = nil;
        request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0
                                                            error:&serialization_error];
        require(request.HTTPBody != nil,
                serialization_error ? serialization_error.localizedDescription.UTF8String
                                    : "cannot serialize sd-server request");
        [request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    }
    __block HTTPResult result;
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    NSURLSessionDataTask *task = [session
        dataTaskWithRequest:request
          completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
            result.data = data;
            result.error = error;
            if ([response isKindOfClass:NSHTTPURLResponse.class])
                result.status = ((NSHTTPURLResponse *)response).statusCode;
            dispatch_semaphore_signal(finished);
          }];
    [task resume];
    while (dispatch_semaphore_wait(finished,
                                   dispatch_time(DISPATCH_TIME_NOW, 200 * NSEC_PER_MSEC)) != 0) {
        if (cancelled.load()) {
            [task cancel];
            throw Cancelled();
        }
    }
    return result;
}

NSDictionary *json_object(const HTTPResult &response, const std::string &context) {
    require(response.error == nil,
            context + ": " + (response.error ? response.error.localizedDescription.UTF8String
                                               : "transport failure"));
    require(response.data != nil, context + " returned no data");
    NSError *error = nil;
    id value = [NSJSONSerialization JSONObjectWithData:response.data options:0 error:&error];
    require([value isKindOfClass:NSDictionary.class],
            context + " returned invalid JSON" +
                (error ? ": " + std::string(error.localizedDescription.UTF8String) : ""));
    return value;
}

std::string log_tail(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};
    input.seekg(0, std::ios::end);
    auto size = input.tellg();
    auto start = std::max<std::streamoff>(0, std::streamoff(size) - 8192);
    input.seekg(start);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

struct ProcessMemory {
    uint64_t resident = 0;
    uint64_t physical_footprint = 0;
    uint64_t peak_physical_footprint = 0;
};

ProcessMemory process_memory(pid_t process) {
    struct rusage_info_v4 usage {};
    if (process <= 0 ||
        proc_pid_rusage(process, RUSAGE_INFO_V4,
                        reinterpret_cast<rusage_info_t *>(&usage)) != 0)
        return {};
    return {usage.ri_resident_size, usage.ri_phys_footprint,
            usage.ri_lifetime_max_phys_footprint};
}

class ZImageGGUF final : public ModelSession {
    fs::path root_;
    std::vector<fs::path> ggufs_;
    fs::path vae_, llm_, active_gguf_, binary_, scratch_, log_path_;
    NSTask *__strong server_ = nil;
    NSFileHandle *__strong log_handle_ = nil;
    NSURLSession *__strong session_ = nil;
    int port_ = 0;
    uint64_t generation_ = 0;
    std::unique_ptr<ZImage> native_session_;
    fs::path native_checkpoint_;
    std::string server_options_key_;

    NSURL *url(NSString *path) const {
        return [NSURL URLWithString:[NSString stringWithFormat:@"http://127.0.0.1:%d%@", port_, path]];
    }

    void stop_server() {
        if (server_ && server_.running) {
            [server_ terminate];
            auto deadline = Clock::now() + std::chrono::seconds(5);
            while (server_.running && Clock::now() < deadline)
                usleep(50000);
            if (server_.running)
                kill(server_.processIdentifier, SIGKILL);
            [server_ waitUntilExit];
        }
        [session_ invalidateAndCancel];
        [log_handle_ closeFile];
        server_ = nil;
        session_ = nil;
        log_handle_ = nil;
        port_ = 0;
        active_gguf_.clear();
        server_options_key_.clear();
        if (!scratch_.empty()) {
            std::error_code error;
            fs::remove_all(scratch_, error);
            scratch_.clear();
        }
        log_path_.clear();
    }

    void stop_native() {
        if (native_session_)
            native_session_->unload();
        native_session_.reset();
        native_checkpoint_.clear();
        mx::clear_cache();
    }

    bool use_native(const Request &request, const fs::path &checkpoint) const {
        if (request.streaming_offload || request.residency == "streaming")
            return false;
        if (!request.loras.empty()) {
            const auto strategy = effective_lora_strategy(request);
            if (strategy == "inference_time")
                return enabled("TURBOCIDER_Z_GGUF_NATIVE_GPU") &&
                       native_mlx_gguf_supported(checkpoint);
            if (strategy == "in_memory_merge")
                return true;
        }
        if (request.execution == "gpu_ane")
            return true;
        return enabled("TURBOCIDER_Z_GGUF_NATIVE_GPU") &&
               native_mlx_gguf_supported(checkpoint);
    }

    ZImage &ensure_native(const fs::path &checkpoint) {
        require(native_mlx_gguf_supported(checkpoint),
                "native MLX GGUF currently supports Q8_0, Q4_0, Q4_1, F16, BF16 or F32; "
                "the selected mixed K-quant remains on sd.cpp Metal");
        stop_server();
        if (!native_session_ || native_checkpoint_ != checkpoint) {
            stop_native();
            native_session_ = std::make_unique<ZImage>(
                find_native_component_root(root_), "z-image-turbo-gguf", checkpoint);
            native_checkpoint_ = checkpoint;
        }
        return *native_session_;
    }

    std::string server_options_key(const Request &request) const {
        const bool streaming = request.streaming_offload || request.residency == "streaming";
        return streaming ? "streaming:" + std::to_string(request.memory_budget_bytes)
                         : "resident";
    }

    void ensure_server(const fs::path &model_path, const Request &request,
                       const Event &event, std::atomic<bool> &cancelled) {
        const auto options_key = server_options_key(request);
        if (server_ && server_.running && active_gguf_ == model_path &&
            server_options_key_ == options_key)
            return;
        stop_native();
        stop_server();
        checkpoint(cancelled);
        validate_gguf_header(model_path);
        binary_ = find_sd_server(root_);
        scratch_ = make_temporary_directory();
        log_path_ = scratch_ / "sd-server.log";
        require([[NSFileManager defaultManager] createFileAtPath:@(log_path_.c_str())
                                                        contents:nil attributes:nil],
                "cannot create sd-server log");
        log_handle_ = [NSFileHandle fileHandleForWritingAtPath:@(log_path_.c_str())];
        require(log_handle_ != nil, "cannot open sd-server log");
        port_ = free_loopback_port();
        server_ = [NSTask new];
        server_.executableURL = [NSURL fileURLWithPath:@(binary_.c_str())];
        server_.currentDirectoryURL = [NSURL fileURLWithPath:@(binary_.parent_path().c_str())];
        NSMutableArray<NSString *> *arguments = [NSMutableArray arrayWithArray:@[
            @"--diffusion-model", @(model_path.c_str()), @"--vae", @(vae_.c_str()),
            @"--llm", @(llm_.c_str()), @"--listen-ip", @"127.0.0.1", @"--listen-port",
            [NSString stringWithFormat:@"%d", port_], @"--lora-model-dir", @(scratch_.c_str()),
            @"--hires-upscalers-dir", @(scratch_.c_str()), @"--embd-dir", @(scratch_.c_str()),
            @"--diffusion-fa", @"--diffusion-conv-direct", @"--clip-on-cpu",
            @"--cfg-scale", @"1.0", @"-v"
        ]];
        if (request.streaming_offload || request.residency == "streaming") {
            require(request.memory_budget_bytes > 0,
                    "Z-Image GGUF streaming offload requires memory_budget_bytes");
            const double gib = double(request.memory_budget_bytes) / double(1ull << 30);
            [arguments addObjectsFromArray:@[
                // Layer streaming is only active when diffusion parameters use
                // the CPU backend. Keep the text encoder/VAE disk-backed, but
                // retain diffusion weights in host memory so sd.cpp can
                // prefetch/evict transformer blocks instead of re-reading each
                // block from disk and silently ignoring --stream-layers.
                @"--params-backend", @"diffusion=cpu,te=disk,vae=disk", @"--mmap", @"--stream-layers",
                @"--max-vram", [NSString stringWithFormat:@"%.3f", std::max(1.0, gib)],
                @"--vae-tiling"
            ]];
        }
        server_.arguments = arguments;
        NSMutableDictionary *environment = [NSProcessInfo.processInfo.environment mutableCopy];
        NSString *directory = @(binary_.parent_path().c_str());
        NSString *existing = environment[@"DYLD_LIBRARY_PATH"];
        environment[@"DYLD_LIBRARY_PATH"] = existing.length
            ? [NSString stringWithFormat:@"%@:%@", directory, existing]
            : directory;
        server_.environment = environment;
        server_.standardOutput = log_handle_;
        server_.standardError = log_handle_;
        event("native_server_load", 0, 1);
        NSError *launch_error = nil;
        require([server_ launchAndReturnError:&launch_error],
                "cannot launch sd-server: " +
                    std::string(launch_error ? launch_error.localizedDescription.UTF8String
                                             : "unknown launch failure"));
        NSURLSessionConfiguration *configuration =
            NSURLSessionConfiguration.ephemeralSessionConfiguration;
        configuration.connectionProxyDictionary = @{};
        configuration.timeoutIntervalForRequest = 3;
        session_ = [NSURLSession sessionWithConfiguration:configuration];
        auto deadline = Clock::now() + std::chrono::minutes(10);
        while (Clock::now() < deadline) {
            checkpoint(cancelled);
            require(server_.running,
                    "sd-server exited during GGUF load:\n" + log_tail(log_path_));
            auto response = http(session_, @"GET", url(@"/v1/models"), nil, cancelled, 2);
            if (!response.error && response.status == 200) {
                active_gguf_ = model_path;
                server_options_key_ = options_key;
                event("native_server_load", 1, 1);
                return;
            }
            usleep(250000);
        }
        auto tail = log_tail(log_path_);
        stop_server();
        require(false, "sd-server timed out while loading GGUF:\n" + tail);
    }

    struct StagedLoRA {
        std::vector<fs::path> files;
        NSArray *__strong payload = nil;
        StagedLoRA() = default;
        StagedLoRA(const StagedLoRA &) = delete;
        StagedLoRA &operator=(const StagedLoRA &) = delete;
        StagedLoRA(StagedLoRA &&other) noexcept
            : files(std::move(other.files)), payload(other.payload) {
            other.files.clear();
            other.payload = nil;
        }
        ~StagedLoRA() {
            for (const auto &file : files) {
                std::error_code error;
                fs::remove(file, error);
            }
        }
    };

    StagedLoRA stage_loras(const std::vector<LoRAAsset> &loras) {
        StagedLoRA staged;
        if (loras.empty()) {
            staged.payload = @[];
            return staged;
        }
        auto generation = "gen_" + std::to_string(++generation_) + "_";
        NSMutableArray *payload = [NSMutableArray array];
        std::set<std::string> used;
        for (const auto &lora : loras) {
            require(fs::is_regular_file(lora.path), "LoRA file missing: " + lora.path);
            fs::path source = fs::canonical(lora.path);
            auto extension = lower(source.extension().string());
            require(extension == ".safetensors" || extension == ".gguf",
                    "sd.cpp LoRA must be safetensors or GGUF");
            auto alias = safe_alias(source.filename().string());
            auto base = alias;
            for (int suffix = 2; used.count(alias); ++suffix)
                alias = base + "_" + std::to_string(suffix);
            used.insert(alias);
            auto target = scratch_ / (generation + alias + extension);
            std::error_code error;
            fs::create_symlink(source, target, error);
            if (error) {
                error.clear();
                fs::copy_file(source, target, fs::copy_options::overwrite_existing, error);
                require(!error, "cannot stage LoRA for sd-server: " + error.message());
            }
            staged.files.push_back(target);
            [payload addObject:@{@"path" : @(target.filename().c_str()),
                                 @"multiplier" : @(lora.strength)}];
        }
        staged.payload = payload;
        return staged;
    }

    RunResult run(const Request &request, const Event &event,
                  std::atomic<bool> &cancelled) {
        auto plan = make_plan(request);
        auto checkpoint_path = select_gguf(ggufs_, request.model_variant);
        if (use_native(request, checkpoint_path)) {
            auto result = ensure_native(checkpoint_path).generate(request, event, cancelled);
            result.selection = request.execution == "gpu_ane"
                                   ? "gpu_ane_native_mlx_gguf"
                                   : "gpu_native_mlx_gguf";
            result.backend = request.execution == "gpu_ane"
                                 ? "mlx_cpp_metal_gguf+coreml"
                                 : "mlx_cpp_metal_gguf";
            result.precision = "gguf:" + quantization_label(checkpoint_path);
            result.checkpoint = checkpoint_path.filename().string();
            return result;
        }
        ensure_server(checkpoint_path, request, event, cancelled);
        auto staged = stage_loras(request.loras);
        uint64_t peak_resident = 0;
        ProcessMemory last_memory;
        auto sample_memory = [&] {
            if (!server_)
                return;
            auto sampled = process_memory(server_.processIdentifier);
            if (sampled.resident) {
                last_memory = sampled;
                peak_resident = std::max(peak_resident, sampled.resident);
            }
        };
        sample_memory();
        NSMutableDictionary *sample = [@{@"sample_steps" : @(request.steps)} mutableCopy];
        NSMutableDictionary *payload = [@{
            @"prompt" : @(request.prompt.c_str()), @"negative_prompt" : @"",
            @"width" : @(request.width), @"height" : @(request.height),
            @"batch_count" : @1, @"output_format" : @"png", @"seed" : @(request.seed),
            @"sample_params" : sample
        } mutableCopy];
        if (staged.payload.count)
            payload[@"lora"] = staged.payload;
        auto begin = Clock::now();
        event("denoise", 0, request.steps);
        auto submit = http(session_, @"POST", url(@"/sdcpp/v1/img_gen"), payload,
                           cancelled, 60);
        std::string submit_body;
        if (submit.data)
            submit_body.assign(reinterpret_cast<const char *>(submit.data.bytes), submit.data.length);
        require(submit.status == 200 || submit.status == 202,
                "sd-server rejected generation with HTTP " + std::to_string(submit.status) +
                    (submit_body.empty() ? "" : ": " + submit_body.substr(0, 1000)));
        auto submitted = json_object(submit, "sd-server generation submit");
        NSString *job = [submitted[@"id"] isKindOfClass:NSString.class] ? submitted[@"id"] : nil;
        require(job.length > 0, "sd-server generation returned no job id");
        NSDictionary *completed = nil;
        auto deadline = Clock::now() + std::chrono::hours(6);
        while (Clock::now() < deadline) {
            sample_memory();
            if (cancelled.load()) {
                stop_server();
                throw Cancelled();
            }
            require(server_ && server_.running,
                    "sd-server exited during generation:\n" + log_tail(log_path_));
            auto status = http(session_, @"GET",
                               url([NSString stringWithFormat:@"/sdcpp/v1/jobs/%@", job]),
                               nil, cancelled, 10);
            if (!status.error && status.status == 200) {
                auto value = json_object(status, "sd-server job status");
                NSString *state = [value[@"status"] isKindOfClass:NSString.class]
                    ? value[@"status"] : @"";
                if ([state isEqual:@"completed"]) {
                    completed = value;
                    break;
                }
                if ([state isEqual:@"failed"] || [state isEqual:@"cancelled"])
                    require(false, "sd-server generation " +
                                       std::string(state.UTF8String));
            }
            // Keep completion-detection jitter below the control-plane
            // overhead being measured against a direct resident sd-server.
            // The benchmark reference uses the same 100 ms cadence.
            usleep(100000);
        }
        require(completed != nil, "sd-server generation timed out");
        sample_memory();
        NSDictionary *result_value = [completed[@"result"] isKindOfClass:NSDictionary.class]
            ? completed[@"result"] : nil;
        NSArray *images = [result_value[@"images"] isKindOfClass:NSArray.class]
            ? result_value[@"images"] : nil;
        require(images.count > 0 && [images[0] isKindOfClass:NSDictionary.class],
                "sd-server completed without an image");
        NSString *encoded = [images[0][@"b64_json"] isKindOfClass:NSString.class]
            ? images[0][@"b64_json"] : nil;
        NSData *image = [[NSData alloc] initWithBase64EncodedString:encoded options:0];
        require(image.length > 0, "sd-server returned invalid image data");
        require(!request.output.empty(), "Z-Image GGUF output path is required");
        auto output = fs::absolute(request.output);
        if (!output.parent_path().empty())
            fs::create_directories(output.parent_path());
        NSError *write_error = nil;
        require([image writeToFile:@(output.c_str()) options:NSDataWritingAtomic error:&write_error],
                "cannot write generated image: " +
                    std::string(write_error ? write_error.localizedDescription.UTF8String
                                            : "unknown write failure"));
        event("denoise", request.steps, request.steps);
        event("vae_decode", 1, 1);
        event("export", 1, 1);
        RunResult result;
        result.request = request;
        result.plan = plan;
        const bool streaming = request.streaming_offload || request.residency == "streaming";
        result.selection = streaming ? "gpu_sd_cpp_gguf_streaming" : "gpu_sd_cpp_gguf";
        result.backend = streaming ? "stable-diffusion-cpp-metal-streaming"
                                   : "stable-diffusion-cpp-metal";
        result.precision = "gguf:" + quantization_label(checkpoint_path);
        result.checkpoint = checkpoint_path.filename().string();
        result.actual_steps = request.steps;
        result.timings.wall = std::chrono::duration<double>(Clock::now() - begin).count();
        result.external_resident_bytes = last_memory.resident;
        result.external_peak_resident_bytes = peak_resident;
        result.external_physical_footprint_bytes = last_memory.physical_footprint;
        result.external_peak_physical_footprint_bytes =
            last_memory.peak_physical_footprint;
        return result;
    }

  public:
    explicit ZImageGGUF(const fs::path &root)
        : root_(fs::absolute(root)), ggufs_(find_ggufs(root_)) {
        auto component_root = fs::is_regular_file(root_) ? root_.parent_path() : root_;
        vae_ = configured_component("TURBOCIDER_Z_IMAGE_VAE");
        if (vae_.empty())
            vae_ = first_existing(component_root, {
                "split_files/vae/ae.safetensors", "vae/ae.safetensors", "ae.safetensors",
                "vae/diffusion_pytorch_model.safetensors"});
        llm_ = configured_component("TURBOCIDER_Z_IMAGE_LLM");
        if (llm_.empty())
            llm_ = first_existing(component_root, {
                "split_files/text_encoders/qwen_3_4b.safetensors",
                "text_encoders/qwen_3_4b.safetensors", "qwen_3_4b.safetensors"});
        require(!vae_.empty(),
                "Z-Image VAE not found; place ae.safetensors under split_files/vae or set "
                "TURBOCIDER_Z_IMAGE_VAE");
        require(!llm_.empty(),
                "Z-Image Qwen3 encoder not found; place qwen_3_4b.safetensors under "
                "split_files/text_encoders or set TURBOCIDER_Z_IMAGE_LLM");
    }

    ~ZImageGGUF() override {
        stop_native();
        stop_server();
    }
    bool uses_parent_mlx() const override {
        if (native_session_)
            return true;
        if (!enabled("TURBOCIDER_Z_GGUF_NATIVE_GPU"))
            return false;
        return native_mlx_gguf_supported(auto_gguf(ggufs_));
    }

    bool uses_parent_mlx(const Request &request) const override {
        return use_native(request, select_gguf(ggufs_, request.model_variant));
    }

    LoadResult load(const Event &event, std::atomic<bool> &cancelled) override {
        try {
            auto checkpoint_path = auto_gguf(ggufs_);
            if (enabled("TURBOCIDER_Z_GGUF_NATIVE_GPU") &&
                native_mlx_gguf_supported(checkpoint_path))
                return ensure_native(checkpoint_path).load(event, cancelled);
            Request resident;
            resident.model = "z-image-turbo-gguf";
            resident.operation = "image.generate";
            resident.execution = "gpu";
            resident.residency = "resident";
            ensure_server(checkpoint_path, resident, event, cancelled);
            return {fs::file_size(checkpoint_path) + fs::file_size(vae_) + fs::file_size(llm_), 0};
        } catch (const Cancelled &) {
            stop_native();
            stop_server();
            throw;
        }
    }

    void unload() override {
        stop_native();
        stop_server();
    }

    RunResult prepare(const Request &request, bool warmup, const Event &event,
                      std::atomic<bool> &cancelled) override {
        try {
            auto checkpoint_path = select_gguf(ggufs_, request.model_variant);
            if (use_native(request, checkpoint_path)) {
                auto result = ensure_native(checkpoint_path).prepare(
                    request, warmup, event, cancelled);
                result.selection = request.execution == "gpu_ane"
                                       ? "gpu_ane_native_mlx_gguf"
                                       : "gpu_native_mlx_gguf";
                result.backend = request.execution == "gpu_ane"
                                     ? "mlx_cpp_metal_gguf+coreml"
                                     : "mlx_cpp_metal_gguf";
                result.precision = "gguf:" + quantization_label(checkpoint_path);
                result.checkpoint = checkpoint_path.filename().string();
                return result;
            }
            if (warmup) {
                auto warm = request;
                if (scratch_.empty()) {
                    ensure_server(checkpoint_path, request, event, cancelled);
                }
                warm.output = (scratch_ / "warmup.png").string();
                auto result = run(warm, event, cancelled);
                std::error_code error;
                fs::remove(warm.output, error);
                result.request = request;
                result.plan = make_plan(request);
                result.prepared = true;
                result.warmup = true;
                return result;
            }
            auto plan = make_plan(request);
            ensure_server(checkpoint_path, request, event, cancelled);
            RunResult result;
            result.request = request;
            result.plan = plan;
            const bool streaming = request.streaming_offload || request.residency == "streaming";
            result.selection = streaming ? "gpu_sd_cpp_gguf_streaming" : "gpu_sd_cpp_gguf";
            result.backend = streaming ? "stable-diffusion-cpp-metal-streaming"
                                       : "stable-diffusion-cpp-metal";
            result.precision = "gguf:" + quantization_label(active_gguf_);
            result.checkpoint = active_gguf_.filename().string();
            result.prepared = true;
            return result;
        } catch (const Cancelled &) {
            stop_server();
            throw;
        }
    }

    RunResult generate(const Request &request, const Event &event,
                       std::atomic<bool> &cancelled) override {
        try {
            return run(request, event, cancelled);
        } catch (const Cancelled &) {
            stop_server();
            throw;
        }
    }
};

} // namespace

std::unique_ptr<ModelSession> create_z_image_gguf(const std::filesystem::path &root) {
    return std::make_unique<ZImageGGUF>(root);
}

} // namespace tc
