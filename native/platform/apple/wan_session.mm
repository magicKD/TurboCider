#include "bridge.hpp"
#include "../../models/wan/pipeline.hpp"
#include "../../core/unigram_tokenizer.hpp"
#include "../../media/video.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>

namespace tc {
namespace {

constexpr const char* kWanSchema = "turbocider-wan-ane-mlp-v1";
constexpr const char* kWanLoRASchema =
    "turbocider-wan-premerged-lora-v1";
// Actual upstream checkpoint identity, not a runtime name or compatibility alias.
constexpr const char* kWanRepository = "FastVideo/FastMetal-1.3B-QAD";
constexpr const char* kWanRevision =
    "2dac0154b217adabf8895d6cde7d6d93e68b7bec";
constexpr const char* kWanCheckpointSha =
    "a48f7370cab9664ebc71afefa6cbc2ea2ab1970b06aa9ad77712179cf52213ff";
constexpr const char* kWanConfigSha =
    "db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691";

static bool valid_sha256(const std::string& value) {
    if (value.size() != CC_SHA256_DIGEST_LENGTH * 2) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

static std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) return {};
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) return {};
    std::array<char, 1 << 20> buffer{};
    while (stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0 && CC_SHA256_Update(&context, buffer.data(),
                                          static_cast<CC_LONG>(count)) != 1)
            return {};
    }
    if (!stream.eof()) return {};
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (CC_SHA256_Final(digest, &context) != 1) return {};
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(CC_SHA256_DIGEST_LENGTH * 2, '0');
    for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        result[i * 2] = hex[digest[i] >> 4];
        result[i * 2 + 1] = hex[digest[i] & 15];
    }
    return result;
}

struct HashCache {
    std::filesystem::path path;
    std::filesystem::file_time_type mtime{};
    uintmax_t bytes = 0;
    std::string sha256;
};

struct WanHashCaches {
    HashCache base_weights;
    HashCache base_config;
    HashCache lora;
    HashCache output_weights;
    HashCache output_config;
};

static std::string cached_sha256(const std::filesystem::path& path,
                                 HashCache& cache) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    require(!error, "cannot resolve Wan checkpoint path: " + path.string());
    auto bytes = std::filesystem::file_size(absolute, error);
    require(!error, "cannot inspect Wan checkpoint size: " + absolute.string());
    auto mtime = std::filesystem::last_write_time(absolute, error);
    require(!error, "cannot inspect Wan checkpoint timestamp: " + absolute.string());
    if (absolute != cache.path || bytes != cache.bytes || mtime != cache.mtime) {
        cache.sha256 = sha256_file(absolute);
        require(!cache.sha256.empty(),
                "cannot hash Wan checkpoint: " + absolute.string());
        auto final_bytes = std::filesystem::file_size(absolute, error);
        require(!error && final_bytes == bytes,
                "Wan checkpoint changed while it was being hashed: " +
                    absolute.string());
        auto final_mtime = std::filesystem::last_write_time(absolute, error);
        require(!error && final_mtime == mtime,
                "Wan checkpoint changed while it was being hashed: " +
                    absolute.string());
        cache.path = absolute;
        cache.bytes = bytes;
        cache.mtime = mtime;
    }
    return cache.sha256;
}

static void require_regular(const std::filesystem::path& path,
                            const std::string& label) {
    require(!path.empty() && std::filesystem::is_regular_file(path),
            "Wan " + label + " is missing: " + path.string());
}

static void require_directory(const std::filesystem::path& path,
                              const std::string& label) {
    require(!path.empty() && std::filesystem::is_directory(path),
            "Wan " + label + " is missing: " + path.string());
}

static bool safe_filename(const std::string& value) {
    if (value.empty()) return false;
    const std::filesystem::path path(value);
    return !path.is_absolute() && path == path.filename() &&
        value != "." && value != "..";
}

static uintmax_t manifest_bytes(NSDictionary* dictionary, NSString* key,
                                const std::string& context) {
    id value = dictionary[key];
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            context + " has an invalid byte count");
    auto bytes = [value unsignedLongLongValue];
    require(bytes > 0, context + " has a zero byte count");
    return static_cast<uintmax_t>(bytes);
}

static double manifest_number(NSDictionary* dictionary, NSString* key,
                              const std::string& context) {
    id value = dictionary[key];
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            context + " has an invalid numeric field");
    const double result = [value doubleValue];
    require(std::isfinite(result), context + " has a non-finite numeric field");
    return result;
}

struct WanLoRASelection {
    std::filesystem::path checkpoint_dir;
    std::filesystem::path manifest;
    std::string checkpoint_sha256;
    std::string checkpoint_json_sha256;
    std::string lora_sha256;
    double strength = 0.0;
};

static void require_wan_shape(NSDictionary* shape) {
    require([shape isKindOfClass:NSDictionary.class],
            "Wan LoRA manifest shape is missing");
    auto positive = [&](NSString* key) {
        id value = shape[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] > 0,
                "Wan LoRA manifest has an invalid shape field");
        return [value unsignedLongLongValue];
    };
    require(positive(@"rows") == 32760 && positive(@"hidden") == 1536 &&
                positive(@"intermediate") == 8960 &&
                positive(@"ane_intermediate") == 4096 &&
                positive(@"gpu_intermediate") == 4864 &&
                positive(@"blocks") == 30 &&
                positive(@"attention_heads") == 12 &&
                positive(@"attention_head_dim") == 128,
            "Wan LoRA manifest shape does not match Wan 1.3B");
}

static void require_wan_config(NSDictionary* config,
                                      const std::string& checkpoint_json_sha) {
    require([config isKindOfClass:NSDictionary.class],
            "Wan LoRA manifest model config is missing");
    require(string_value(config, @"format") == "mlx_dit-v1" &&
                string_value(config, @"config_sha256") == checkpoint_json_sha,
            "Wan LoRA manifest model config identity is not trusted");
    auto number = [&](NSString* key, uint64_t expected) {
        id value = config[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] == expected,
                "Wan LoRA manifest model config does not match Wan 1.3B");
    };
    number(@"format_version", 1);
    number(@"num_blocks", 30);
    number(@"in_channels", 16);
    number(@"out_channels", 16);
    number(@"attention_head_dim", 128);
    number(@"num_attention_heads", 12);
    number(@"ffn_dim", 8960);
    number(@"text_dim", 4096);
    number(@"quantization_bits", 8);
    number(@"quantization_group_size", 64);
    require(string_value(config, @"quantization_mode") == "affine",
            "Wan LoRA manifest quantization mode is unsupported");
    NSArray* patch_size = [config[@"patch_size"] isKindOfClass:NSArray.class]
        ? config[@"patch_size"] : nil;
    require(patch_size.count == 3 && [patch_size[0] unsignedIntValue] == 1 &&
                [patch_size[1] unsignedIntValue] == 2 &&
                [patch_size[2] unsignedIntValue] == 2,
            "Wan LoRA manifest patch size does not match Wan 1.3B");
}

static void require_wan_checkpoint_config(NSDictionary* checkpoint) {
    id format_version = checkpoint[@"format_version"];
    id num_blocks = checkpoint[@"num_blocks"];
    NSDictionary* config = [checkpoint[@"config"] isKindOfClass:NSDictionary.class]
        ? checkpoint[@"config"] : nil;
    NSDictionary* quantization =
        [checkpoint[@"quantization"] isKindOfClass:NSDictionary.class]
            ? checkpoint[@"quantization"] : nil;
    require([format_version isKindOfClass:NSNumber.class] &&
                [format_version unsignedIntValue] == 1 &&
                [num_blocks isKindOfClass:NSNumber.class] &&
                [num_blocks unsignedIntValue] == 30 && config && quantization,
            "Wan merged mlx_dit.json has an unsupported format");
    auto config_number = [&](NSString* key, uint64_t expected) {
        id value = config[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] == expected,
                "Wan merged mlx_dit.json does not match Wan 1.3B");
    };
    config_number(@"in_channels", 16);
    config_number(@"out_channels", 16);
    config_number(@"attention_head_dim", 128);
    config_number(@"num_attention_heads", 12);
    config_number(@"num_layers", 30);
    config_number(@"ffn_dim", 8960);
    config_number(@"text_dim", 4096);
    NSArray* patch_size = [config[@"patch_size"] isKindOfClass:NSArray.class]
        ? config[@"patch_size"] : nil;
    require(patch_size.count == 3 && [patch_size[0] unsignedIntValue] == 1 &&
                [patch_size[1] unsignedIntValue] == 2 &&
                [patch_size[2] unsignedIntValue] == 2,
            "Wan merged mlx_dit.json patch size does not match Wan 1.3B");
    id bits = quantization[@"bits"];
    id group_size = quantization[@"group_size"];
    require(string_value(quantization, @"mode") == "affine" &&
                [bits isKindOfClass:NSNumber.class] &&
                [bits unsignedIntValue] == 8 &&
                [group_size isKindOfClass:NSNumber.class] &&
                [group_size unsignedIntValue] == 64,
            "Wan merged mlx_dit.json quantization is unsupported");
}

static std::filesystem::path manifest_file(NSDictionary* record,
                                            NSString* key,
                                            const std::string& context) {
    auto name = string_value(record, key);
    require(safe_filename(name), context + " must be a plain filename");
    return std::filesystem::path(name);
}

static void verify_file_record(const std::filesystem::path& path,
                               NSDictionary* record,
                               const std::string& context,
                               HashCache& cache,
                               NSString* bytes_key = @"bytes",
                               NSString* sha_key = @"sha256",
                               std::string* digest_out = nullptr) {
    require_regular(path, context);
    auto bytes = manifest_bytes(record, bytes_key, context);
    require(bytes == std::filesystem::file_size(path),
            context + " size does not match its manifest");
    auto expected = string_value(record, sha_key);
    require(valid_sha256(expected), context + " contains a malformed SHA-256");
    auto actual = cached_sha256(path, cache);
    require(actual == expected, context + " SHA-256 does not match its manifest");
    if (digest_out) *digest_out = std::move(actual);
}

static bool inspect_wan_lora_manifest(
        const std::filesystem::path& manifest_path,
        const std::filesystem::path& model_root,
        const LoRAAsset& requested,
        WanHashCaches& caches,
        WanLoRASelection& selection,
        std::string& failure) {
    try {
        auto manifest = read_json(manifest_path);
        require(string_value(manifest, @"schema") == kWanLoRASchema,
                "unsupported Wan LoRA manifest schema");
        require(string_value(manifest, @"algorithm") ==
                    "fastvideo-mlx-runtime-equivalent-transformer-lora-premerge-v1",
                "unsupported Wan LoRA merge algorithm");
        require(string_value(manifest, @"repository") == kWanRepository &&
                    string_value(manifest, @"revision") == kWanRevision,
                "Wan LoRA manifest repository or revision is not trusted");
        NSDictionary* base = [manifest[@"base"] isKindOfClass:NSDictionary.class]
            ? manifest[@"base"] : nil;
        NSDictionary* lora = [manifest[@"lora"] isKindOfClass:NSDictionary.class]
            ? manifest[@"lora"] : nil;
        NSDictionary* output = [manifest[@"output"] isKindOfClass:NSDictionary.class]
            ? manifest[@"output"] : nil;
        require(base && lora && output,
                "Wan LoRA manifest is missing base/lora/output records");
        auto base_weights = manifest_file(base, @"weights_filename",
                                         "Wan LoRA base weights");
        auto base_config = manifest_file(base, @"config_filename",
                                         "Wan LoRA base config");
        require(base_weights == "mlx_dit.safetensors" &&
                    base_config == "mlx_dit.json",
                "Wan LoRA base filenames are not the packed MLX checkpoint");
        auto base_sha = string_value(base, @"weights_sha256");
        auto base_json_sha = string_value(base, @"config_sha256");
        require(base_sha == kWanCheckpointSha &&
                    base_json_sha == kWanConfigSha,
                "Wan LoRA base checkpoint identity is not trusted");
        require_regular(model_root / base_config, "Wan LoRA base config");
        require(manifest_bytes(base, @"config_bytes", "Wan LoRA base config") ==
                    std::filesystem::file_size(model_root / base_config),
                "Wan LoRA base config size does not match its manifest");

        auto lora_filename = manifest_file(lora, @"filename",
                                           "Wan LoRA record");
        require(lora_filename == std::filesystem::path(requested.path).filename(),
                "Wan LoRA manifest does not describe the requested file");
        require(lora_filename.extension() == ".safetensors",
                "Wan LoRA file must use the .safetensors extension");
        require(string_value(lora, @"role") == "transformer",
                "Wan LoRA role must be transformer");
        require(string_value(lora, @"adapter_format") == "safetensors",
                "Wan LoRA adapter format is unsupported");
        auto expected_strength = manifest_number(lora, @"strength",
                                                 "Wan LoRA record");
        require(expected_strength > 0.0 && expected_strength <= 4.0 &&
                    std::abs(requested.strength - static_cast<float>(expected_strength)) <= 1e-6f,
                "Wan requested LoRA strength does not match the manifest");
        verify_file_record(requested.path, lora, "Wan LoRA record",
                           caches.lora);

        auto output_directory = manifest_file(output, @"directory",
                                              "Wan LoRA output directory");
        auto output_weights = manifest_file(output, @"weights_filename",
                                            "Wan LoRA output weights");
        auto output_config = manifest_file(output, @"config_filename",
                                           "Wan LoRA output config");
        require(output_weights == "mlx_dit.safetensors" &&
                    output_config == "mlx_dit.json",
                "Wan LoRA output filenames are not the packed MLX checkpoint");
        auto output_dir = manifest_path.parent_path() / output_directory;
        require_directory(output_dir, "Wan LoRA merged checkpoint directory");
        auto output_weights_path = output_dir / output_weights;
        auto output_config_path = output_dir / output_config;
        std::string output_sha;
        verify_file_record(output_weights_path, output,
                           "Wan LoRA merged weights", caches.output_weights,
                           @"weights_bytes",
                           @"weights_sha256", &output_sha);
        require_regular(output_config_path, "Wan LoRA output config");
        require(manifest_bytes(output, @"config_bytes", "Wan LoRA output config") ==
                    std::filesystem::file_size(output_config_path),
                "Wan LoRA output config size does not match its manifest");
        auto output_json_sha = string_value(output, @"config_sha256");
        require(valid_sha256(output_json_sha) &&
                    cached_sha256(output_config_path, caches.output_config) ==
                        output_json_sha,
                "Wan LoRA output config SHA-256 does not match its manifest");
        require_wan_checkpoint_config(read_json(output_config_path));
        require(output_sha != kWanCheckpointSha,
                "Wan LoRA output is identical to the unmerged base checkpoint");
        NSDictionary* mapping = [manifest[@"mapping"] isKindOfClass:NSDictionary.class]
            ? manifest[@"mapping"] : nil;
        require(mapping, "Wan LoRA manifest mapping is missing");
        auto mapping_total = manifest_bytes(mapping, @"total", "Wan LoRA mapping");
        auto mapping_matched = manifest_bytes(mapping, @"matched", "Wan LoRA mapping");
        id missing_value = mapping[@"missing"];
        require([missing_value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)missing_value) != CFBooleanGetTypeID() &&
                    [missing_value unsignedLongLongValue] == 0,
                "Wan LoRA mapping is incomplete");
        require(mapping_total == mapping_matched,
                "Wan LoRA mapping matched count is incomplete");
        require_wan_shape(manifest[@"shape"]);
        require_wan_config(manifest[@"model_config"], output_json_sha);
        verify_file_record(model_root / base_weights, base,
                           "Wan LoRA base weights", caches.base_weights,
                           @"weights_bytes", @"weights_sha256");
        require(cached_sha256(model_root / base_config, caches.base_config) ==
                    base_json_sha,
                "Wan LoRA base config SHA-256 does not match its manifest");

        selection.checkpoint_dir = std::filesystem::absolute(output_dir);
        selection.manifest = std::filesystem::absolute(manifest_path);
        selection.checkpoint_sha256 = std::move(output_sha);
        selection.checkpoint_json_sha256 = std::move(output_json_sha);
        selection.lora_sha256 = caches.lora.sha256;
        selection.strength = expected_strength;
        return true;
    } catch (const std::exception& exception) {
        failure = exception.what();
        return false;
    }
}

static void add_wan_manifest_candidate(
        std::vector<std::filesystem::path>& candidates,
        const std::filesystem::path& path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    if (error || !std::filesystem::is_regular_file(absolute)) return;
    for (const auto& existing : candidates)
        if (existing == absolute) return;
    candidates.push_back(std::move(absolute));
}

static WanLoRASelection resolve_wan_lora(
        const std::filesystem::path& model_root, const LoRAAsset& requested,
        WanHashCaches& caches) {
    const std::filesystem::path lora_path(requested.path);
    require(std::filesystem::is_regular_file(lora_path),
            "requested Wan LoRA file is missing: " + lora_path.string());
    std::vector<std::filesystem::path> candidates;
    add_wan_manifest_candidate(candidates,
                                     lora_path.string() + ".manifest.json");
    add_wan_manifest_candidate(candidates,
                                     lora_path.parent_path() /
                                         "wan-lora.manifest.json");
    add_wan_manifest_candidate(candidates,
                                     model_root / "wan-lora.manifest.json");
    std::string failure = "no candidate manifest found";
    WanLoRASelection selection;
    for (const auto& candidate : candidates) {
        if (inspect_wan_lora_manifest(candidate, model_root, requested,
                                            caches, selection, failure))
            return selection;
    }
    require(false, "Wan LoRA has no provenance-verified premerged checkpoint: " +
                      failure);
    return selection;
}


class WanSession final : public ModelSession {
    std::filesystem::path root_;
    std::unique_ptr<wan::Pipeline> pipeline_;
    std::unique_ptr<UnigramTokenizer> tokenizer_;
    std::string pipeline_key_;
    WanHashCaches lora_hash_caches_;

    void ensure_pipeline(const WanLoRASelection &selection, const Event &event,
                         std::atomic<bool> &cancelled) {
        const auto key = selection.checkpoint_dir.string() + "\n" +
            selection.checkpoint_sha256 + "\n" + selection.checkpoint_json_sha256;
        if (pipeline_ && pipeline_key_ == key) return;
        unload();
        checkpoint(cancelled);
        event("model_load", 0, 1);
        // The converted TAEHV decoder belongs to the model package. Never
        // discover a Python toolchain, sibling repository or current directory.
        auto tokenizer = std::make_unique<UnigramTokenizer>(root_ / "tokenizer");
        auto pipeline = std::make_unique<wan::Pipeline>(
            selection.checkpoint_dir, root_ / "text_encoder",
            root_ / "vae/taew2_1.safetensors", event, cancelled);
        checkpoint(cancelled);
        tokenizer_ = std::move(tokenizer);
        pipeline_ = std::move(pipeline);
        pipeline_key_ = key;
        event("model_load", 1, 1);
    }

public:
    explicit WanSession(const std::filesystem::path &root)
        : root_(std::filesystem::absolute(root)) {
        require_directory(root_, "model directory");
        require_regular(root_ / "mlx_dit.json", "mlx_dit.json");
        require_regular(root_ / "mlx_dit.safetensors", "mlx_dit.safetensors");
        require_directory(root_ / "tokenizer", "tokenizer directory");
        require_directory(root_ / "text_encoder", "text_encoder directory");
        require_regular(root_ / "vae/taew2_1.safetensors",
                        "native TAEHV decoder; import the offline-converted model package");
        require(sha256_file(root_ / "mlx_dit.safetensors") == kWanCheckpointSha,
                "Wan checkpoint SHA-256 does not match the validated QAD release");
        require(sha256_file(root_ / "mlx_dit.json") == kWanConfigSha,
                "Wan mlx_dit.json SHA-256 does not match the validated QAD release");
    }

    bool uses_parent_mlx() const override { return true; }
    void unload() override {
        pipeline_.reset();
        tokenizer_.reset();
        pipeline_key_.clear();
    }

    RunResult generate(const Request &request, const Event &event,
                       std::atomic<bool> &cancelled) override {
        require(request.model == "wan2.1-1.3b-qad", "request model differs from Wan session");
        module_for(request.model).validate(request);
        const auto plan = make_plan(request);
        require(!request.output.empty() &&
                    std::filesystem::path(request.output).extension() == ".mp4",
                "Wan output must be an .mp4 file");
        checkpoint(cancelled);
        const auto start = Clock::now();
        WanLoRASelection selection;
        selection.checkpoint_dir = root_;
        selection.checkpoint_sha256 = kWanCheckpointSha;
        selection.checkpoint_json_sha256 = kWanConfigSha;
        if (!request.loras.empty())
            selection = resolve_wan_lora(root_, request.loras.front(), lora_hash_caches_);
        ensure_pipeline(selection, event, cancelled);
        auto tokens = tokenizer_->prompt(request.prompt);
        wan::GenerateOptions options;
        options.width = request.width; options.height = request.height;
        options.frames = request.frames; options.fps = request.fps;
        options.seed = request.seed;
        options.compile_dit = request.compile_gpu;
        if (request.execution == "gpu_ane") options.hybrid_manifest = request.ane_manifest;
        bool prompt_cache_hit = false;
        Event progress = [&](const std::string &phase, int done, int total) {
            if (phase == "umt5_prompt_cache") prompt_cache_hit = true;
            event(phase, done, total);
        };
        const auto output = std::filesystem::absolute(request.output);
        std::error_code error;
        std::filesystem::create_directories(output.parent_path(), error);
        require(!error, "cannot create Wan output directory");
        const auto temporary = output.parent_path() /
            (output.filename().string() + ".tmp." + NSUUID.UUID.UUIDString.UTF8String + ".mp4");
        try {
            auto pixels = pipeline_->generate(tokens, options, progress, cancelled);
            require(pixels.shape() == mx::Shape{1, request.frames, 3, request.height, request.width},
                    "Wan generated incorrect video dimensions");
            require(mx::all(mx::isfinite(pixels)).item<bool>(), "Wan decoder produced nonfinite pixels");
            auto rgb = mx::astype(mx::contiguous(mx::transpose(pixels, {0, 1, 3, 4, 2})) *
                                      Tensor(255.f), mx::uint8);
            mx::eval(rgb);
            checkpoint(cancelled);
            event("export", 0, 1);
            write_video_rgb24(temporary, rgb.data<uint8_t>(), request.frames, request.width,
                             request.height, request.fps);
            const auto info = probe_video(temporary);
            require(info.frames == request.frames && info.width == request.width &&
                        info.height == request.height && info.fps == request.fps,
                    "Wan encoded video failed validation");
            checkpoint(cancelled);
            std::filesystem::rename(temporary, output, error);
            require(!error, "cannot atomically publish Wan output: " + error.message());
        } catch (...) {
            std::filesystem::remove(temporary, error);
            // Retire graphs and partially used Core ML resources after failure.
            unload();
            throw;
        }
        event("export", 1, 1);
        NSMutableDictionary *result = [@{
            @"schema_version": @1, @"model": @(request.model.c_str()),
            @"operation": @"video.generate", @"output": @(output.c_str()),
            @"width": @(request.width), @"height": @(request.height),
            @"frames": @(request.frames), @"fps": @(request.fps),
            @"seed": @(request.seed), @"steps": @(request.steps),
            @"actual_denoise_steps": @(request.steps),
            @"execution": @(request.execution.c_str()),
            @"runtime_backend": request.execution == "gpu_ane" ? @"wan-mlx+coreml" : @"wan-mlx",
            @"runtime_dependency": @"native",
            @"runtime_precision": @"fp16-int8-affine-dit+bf16-umt5+fp32-taehv",
            @"persistent_model": @YES, @"prompt_cache_hit": @(prompt_cache_hit),
            @"text_tokens": @(tokens.ids.size()), @"valid_text_tokens": @(tokens.valid),
            @"seconds": @(std::chrono::duration<double>(Clock::now() - start).count()),
            @"mlx_active_bytes": @(mx::get_active_memory()),
            @"mlx_peak_bytes": @(mx::get_peak_memory()),
            @"mlx_checkpoint_sha256": @(selection.checkpoint_sha256.c_str()),
            @"mlx_dit_json_sha256": @(selection.checkpoint_json_sha256.c_str()),
            @"lora_fusion": request.loras.empty() ? @"none" : @"premerged_manifest_verified"
        } mutableCopy];
        if (!request.loras.empty()) {
            result[@"lora_manifest"] = @(selection.manifest.c_str());
            result[@"lora_sha256"] = @(selection.lora_sha256.c_str());
            result[@"lora_strength"] = @(selection.strength);
        }
        return native_run_result(result, request, plan);
    }
};

} // namespace

std::string validate_wan_ane_manifest(const std::filesystem::path& path) {
    require(std::filesystem::is_regular_file(path),
            "Wan ANE manifest is missing: " + path.string());
    auto manifest = read_json(path);
    require(string_value(manifest, @"schema") == kWanSchema,
            "unsupported Wan ANE manifest schema");
    NSDictionary* shape = [manifest[@"shape"] isKindOfClass:NSDictionary.class]
        ? manifest[@"shape"] : nil;
    require([shape isKindOfClass:NSDictionary.class], "Wan ANE manifest shape is missing");
    auto positive = [&](NSString* key) {
        id value = shape[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] > 0,
                "Wan ANE manifest has an invalid shape field");
        return [value unsignedLongLongValue];
    };
    require(positive(@"rows") == 32760 && positive(@"hidden") == 1536 &&
                positive(@"intermediate") == 8960 && positive(@"ane_intermediate") == 4096 &&
                positive(@"gpu_intermediate") == 4864,
            "Wan ANE manifest shape does not match Wan 1.3B");
    const auto variant = string_value(manifest, @"variant");
    require(variant == "int8_pc" || variant == "fp16",
            "unsupported Wan ANE manifest variant");
    NSArray* blocks = [manifest[@"blocks"] isKindOfClass:NSArray.class]
        ? manifest[@"blocks"] : nil;
    require([blocks isKindOfClass:NSArray.class] && blocks.count == 30,
            "Wan ANE manifest must cover exactly 30 blocks");
    for (NSUInteger index = 0; index < blocks.count; ++index) {
        id value = blocks[index];
        require([value isKindOfClass:NSNumber.class] &&
                    [value integerValue] == static_cast<NSInteger>(index),
                "Wan ANE blocks must be sorted 0...29");
    }
    NSDictionary* artifacts = [manifest[@"artifacts"] isKindOfClass:NSDictionary.class]
        ? manifest[@"artifacts"] : nil;
    require([artifacts isKindOfClass:NSDictionary.class] && artifacts.count == 30,
            "Wan ANE manifest must contain 30 artifacts");
    auto parent = std::filesystem::absolute(path).parent_path();
    for (int block = 0; block < 30; ++block) {
        NSString* key = [NSString stringWithFormat:@"%d", block];
        id value = artifacts[key];
        require([value isKindOfClass:NSString.class],
                "Wan ANE artifact path must be a string");
        auto relative = std::filesystem::path([value UTF8String]);
        require(!relative.is_absolute() && relative == relative.filename(),
                "Wan ANE artifact path must be a file name");
        require_directory(parent / relative, "ANE block artifact");
    }
    auto checkpoint = string_value(manifest, @"checkpoint_sha256");
    auto config = string_value(manifest, @"mlx_dit_json_sha256");
    require(checkpoint == kWanCheckpointSha &&
                config == kWanConfigSha,
            "Wan ANE checkpoint identity is not trusted");
    // The selected session verifies its actual checkpoint, not a historical
    // machine-specific source path embedded in a copied manifest.
    return std::filesystem::absolute(path).string();
}

NSDictionary* preflight_wan_lora(const std::filesystem::path& root,
                                       const LoRAAsset& lora) {
    require(std::filesystem::is_directory(root),
            "Wan model directory is missing: " + root.string());
    WanHashCaches caches;
    auto selection = resolve_wan_lora(
        std::filesystem::absolute(root), lora, caches);
    return @{ @"validation": @"premerged_manifest_verified",
              @"checkpoint_directory": @(selection.checkpoint_dir.string().c_str()),
              @"checkpoint_sha256": @(selection.checkpoint_sha256.c_str()),
              @"mlx_dit_json_sha256":
                  @(selection.checkpoint_json_sha256.c_str()),
              @"lora_manifest": @(selection.manifest.string().c_str()),
              @"lora_sha256": @(selection.lora_sha256.c_str()),
              @"lora_strength": @(selection.strength),
              @"execution": @"gpu",
              @"runtime_lora": @NO };
}

std::unique_ptr<ModelSession> create_wan(const std::filesystem::path& root) {
    return std::make_unique<WanSession>(root);
}
}
