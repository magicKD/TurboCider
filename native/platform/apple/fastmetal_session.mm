#include "bridge.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace tc {
namespace {

constexpr const char* kFastMetalSchema = "turbocider-fastmetal-ane-mlp-v1";
constexpr const char* kFastMetalLoRASchema =
    "turbocider-fastmetal-premerged-lora-v1";
constexpr const char* kFastMetalRepository = "FastVideo/FastMetal-1.3B-QAD";
constexpr const char* kFastMetalRevision =
    "2dac0154b217adabf8895d6cde7d6d93e68b7bec";
constexpr const char* kFastMetalCheckpointSha =
    "a48f7370cab9664ebc71afefa6cbc2ea2ab1970b06aa9ad77712179cf52213ff";
constexpr const char* kFastMetalConfigSha =
    "db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691";
constexpr const char* kFastMetalConfigName = "turbocider-fastmetal.json";

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

struct FastMetalHashCaches {
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
    require(!error, "cannot resolve FastMetal checkpoint path: " + path.string());
    auto bytes = std::filesystem::file_size(absolute, error);
    require(!error, "cannot inspect FastMetal checkpoint size: " + absolute.string());
    auto mtime = std::filesystem::last_write_time(absolute, error);
    require(!error, "cannot inspect FastMetal checkpoint timestamp: " + absolute.string());
    if (absolute != cache.path || bytes != cache.bytes || mtime != cache.mtime) {
        cache.sha256 = sha256_file(absolute);
        require(!cache.sha256.empty(),
                "cannot hash FastMetal checkpoint: " + absolute.string());
        auto final_bytes = std::filesystem::file_size(absolute, error);
        require(!error && final_bytes == bytes,
                "FastMetal checkpoint changed while it was being hashed: " +
                    absolute.string());
        auto final_mtime = std::filesystem::last_write_time(absolute, error);
        require(!error && final_mtime == mtime,
                "FastMetal checkpoint changed while it was being hashed: " +
                    absolute.string());
        cache.path = absolute;
        cache.bytes = bytes;
        cache.mtime = mtime;
    }
    return cache.sha256;
}

static std::filesystem::path environment_path(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? std::filesystem::path(value) : std::filesystem::path{};
}

static void require_regular(const std::filesystem::path& path,
                            const std::string& label) {
    require(!path.empty() && std::filesystem::is_regular_file(path),
            "FastMetal " + label + " is missing: " + path.string());
}

static void require_directory(const std::filesystem::path& path,
                              const std::string& label) {
    require(!path.empty() && std::filesystem::is_directory(path),
            "FastMetal " + label + " is missing: " + path.string());
}

static std::filesystem::path resolve_relative(const std::filesystem::path& base,
                                              const std::string& value) {
    require(!value.empty(), "FastMetal profile contains an empty path");
    const std::filesystem::path candidate(value);
    return candidate.is_absolute() ? candidate : (base / candidate).lexically_normal();
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

struct FastMetalConfig {
    std::filesystem::path python;
    std::filesystem::path engine_root;
    std::filesystem::path script;
    std::filesystem::path worker;
    std::filesystem::path bridge_dir;
    std::string decode_backend = "taehv";
    std::string text_encoder_dtype = "bf16";
    std::string vae_decode_dtype = "bf16";
    std::string mlx_dtype = "fp16";
    std::string mlx_quantization = "int8";
    bool mlx_compile = true;
    bool prompt_cache = true;
    bool taehv_parallel = false;
    std::string identity;
};

struct FastMetalLoRASelection {
    std::filesystem::path checkpoint_dir;
    std::filesystem::path manifest;
    std::string checkpoint_sha256;
    std::string checkpoint_json_sha256;
    std::string lora_sha256;
    double strength = 0.0;
};

static void require_fastmetal_shape(NSDictionary* shape) {
    require([shape isKindOfClass:NSDictionary.class],
            "FastMetal LoRA manifest shape is missing");
    auto positive = [&](NSString* key) {
        id value = shape[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] > 0,
                "FastMetal LoRA manifest has an invalid shape field");
        return [value unsignedLongLongValue];
    };
    require(positive(@"rows") == 32760 && positive(@"hidden") == 1536 &&
                positive(@"intermediate") == 8960 &&
                positive(@"ane_intermediate") == 4096 &&
                positive(@"gpu_intermediate") == 4864 &&
                positive(@"blocks") == 30 &&
                positive(@"attention_heads") == 12 &&
                positive(@"attention_head_dim") == 128,
            "FastMetal LoRA manifest shape does not match FastMetal 1.3B");
}

static void require_fastmetal_config(NSDictionary* config,
                                      const std::string& checkpoint_json_sha) {
    require([config isKindOfClass:NSDictionary.class],
            "FastMetal LoRA manifest model config is missing");
    require(string_value(config, @"format") == "mlx_dit-v1" &&
                string_value(config, @"config_sha256") == checkpoint_json_sha,
            "FastMetal LoRA manifest model config identity is not trusted");
    auto number = [&](NSString* key, uint64_t expected) {
        id value = config[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] == expected,
                "FastMetal LoRA manifest model config does not match FastMetal 1.3B");
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
            "FastMetal LoRA manifest quantization mode is unsupported");
    NSArray* patch_size = [config[@"patch_size"] isKindOfClass:NSArray.class]
        ? config[@"patch_size"] : nil;
    require(patch_size.count == 3 && [patch_size[0] unsignedIntValue] == 1 &&
                [patch_size[1] unsignedIntValue] == 2 &&
                [patch_size[2] unsignedIntValue] == 2,
            "FastMetal LoRA manifest patch size does not match FastMetal 1.3B");
}

static void require_fastmetal_checkpoint_config(NSDictionary* checkpoint) {
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
            "FastMetal merged mlx_dit.json has an unsupported format");
    auto config_number = [&](NSString* key, uint64_t expected) {
        id value = config[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] == expected,
                "FastMetal merged mlx_dit.json does not match FastMetal 1.3B");
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
            "FastMetal merged mlx_dit.json patch size does not match FastMetal 1.3B");
    id bits = quantization[@"bits"];
    id group_size = quantization[@"group_size"];
    require(string_value(quantization, @"mode") == "affine" &&
                [bits isKindOfClass:NSNumber.class] &&
                [bits unsignedIntValue] == 8 &&
                [group_size isKindOfClass:NSNumber.class] &&
                [group_size unsignedIntValue] == 64,
            "FastMetal merged mlx_dit.json quantization is unsupported");
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

static bool inspect_fastmetal_lora_manifest(
        const std::filesystem::path& manifest_path,
        const std::filesystem::path& model_root,
        const LoRAAsset& requested,
        FastMetalHashCaches& caches,
        FastMetalLoRASelection& selection,
        std::string& failure) {
    try {
        auto manifest = read_json(manifest_path);
        require(string_value(manifest, @"schema") == kFastMetalLoRASchema,
                "unsupported FastMetal LoRA manifest schema");
        require(string_value(manifest, @"algorithm") ==
                    "fastvideo-mlx-runtime-equivalent-transformer-lora-premerge-v1",
                "unsupported FastMetal LoRA merge algorithm");
        require(string_value(manifest, @"repository") == kFastMetalRepository &&
                    string_value(manifest, @"revision") == kFastMetalRevision,
                "FastMetal LoRA manifest repository or revision is not trusted");
        NSDictionary* base = [manifest[@"base"] isKindOfClass:NSDictionary.class]
            ? manifest[@"base"] : nil;
        NSDictionary* lora = [manifest[@"lora"] isKindOfClass:NSDictionary.class]
            ? manifest[@"lora"] : nil;
        NSDictionary* output = [manifest[@"output"] isKindOfClass:NSDictionary.class]
            ? manifest[@"output"] : nil;
        require(base && lora && output,
                "FastMetal LoRA manifest is missing base/lora/output records");
        auto base_weights = manifest_file(base, @"weights_filename",
                                         "FastMetal LoRA base weights");
        auto base_config = manifest_file(base, @"config_filename",
                                         "FastMetal LoRA base config");
        require(base_weights == "mlx_dit.safetensors" &&
                    base_config == "mlx_dit.json",
                "FastMetal LoRA base filenames are not the packed MLX checkpoint");
        auto base_sha = string_value(base, @"weights_sha256");
        auto base_json_sha = string_value(base, @"config_sha256");
        require(base_sha == kFastMetalCheckpointSha &&
                    base_json_sha == kFastMetalConfigSha,
                "FastMetal LoRA base checkpoint identity is not trusted");
        require_regular(model_root / base_config, "FastMetal LoRA base config");
        require(manifest_bytes(base, @"config_bytes", "FastMetal LoRA base config") ==
                    std::filesystem::file_size(model_root / base_config),
                "FastMetal LoRA base config size does not match its manifest");

        auto lora_filename = manifest_file(lora, @"filename",
                                           "FastMetal LoRA record");
        require(lora_filename == std::filesystem::path(requested.path).filename(),
                "FastMetal LoRA manifest does not describe the requested file");
        require(lora_filename.extension() == ".safetensors",
                "FastMetal LoRA file must use the .safetensors extension");
        require(string_value(lora, @"role") == "transformer",
                "FastMetal LoRA role must be transformer");
        require(string_value(lora, @"adapter_format") == "safetensors",
                "FastMetal LoRA adapter format is unsupported");
        auto expected_strength = manifest_number(lora, @"strength",
                                                 "FastMetal LoRA record");
        require(expected_strength > 0.0 && expected_strength <= 4.0 &&
                    std::abs(requested.strength - static_cast<float>(expected_strength)) <= 1e-6f,
                "FastMetal requested LoRA strength does not match the manifest");
        verify_file_record(requested.path, lora, "FastMetal LoRA record",
                           caches.lora);

        auto output_directory = manifest_file(output, @"directory",
                                              "FastMetal LoRA output directory");
        auto output_weights = manifest_file(output, @"weights_filename",
                                            "FastMetal LoRA output weights");
        auto output_config = manifest_file(output, @"config_filename",
                                           "FastMetal LoRA output config");
        require(output_weights == "mlx_dit.safetensors" &&
                    output_config == "mlx_dit.json",
                "FastMetal LoRA output filenames are not the packed MLX checkpoint");
        auto output_dir = manifest_path.parent_path() / output_directory;
        require_directory(output_dir, "FastMetal LoRA merged checkpoint directory");
        auto output_weights_path = output_dir / output_weights;
        auto output_config_path = output_dir / output_config;
        std::string output_sha;
        verify_file_record(output_weights_path, output,
                           "FastMetal LoRA merged weights", caches.output_weights,
                           @"weights_bytes",
                           @"weights_sha256", &output_sha);
        require_regular(output_config_path, "FastMetal LoRA output config");
        require(manifest_bytes(output, @"config_bytes", "FastMetal LoRA output config") ==
                    std::filesystem::file_size(output_config_path),
                "FastMetal LoRA output config size does not match its manifest");
        auto output_json_sha = string_value(output, @"config_sha256");
        require(valid_sha256(output_json_sha) &&
                    cached_sha256(output_config_path, caches.output_config) ==
                        output_json_sha,
                "FastMetal LoRA output config SHA-256 does not match its manifest");
        require_fastmetal_checkpoint_config(read_json(output_config_path));
        require(output_sha != kFastMetalCheckpointSha,
                "FastMetal LoRA output is identical to the unmerged base checkpoint");
        NSDictionary* mapping = [manifest[@"mapping"] isKindOfClass:NSDictionary.class]
            ? manifest[@"mapping"] : nil;
        require(mapping, "FastMetal LoRA manifest mapping is missing");
        auto mapping_total = manifest_bytes(mapping, @"total", "FastMetal LoRA mapping");
        auto mapping_matched = manifest_bytes(mapping, @"matched", "FastMetal LoRA mapping");
        id missing_value = mapping[@"missing"];
        require([missing_value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)missing_value) != CFBooleanGetTypeID() &&
                    [missing_value unsignedLongLongValue] == 0,
                "FastMetal LoRA mapping is incomplete");
        require(mapping_total == mapping_matched,
                "FastMetal LoRA mapping matched count is incomplete");
        require_fastmetal_shape(manifest[@"shape"]);
        require_fastmetal_config(manifest[@"model_config"], output_json_sha);
        verify_file_record(model_root / base_weights, base,
                           "FastMetal LoRA base weights", caches.base_weights,
                           @"weights_bytes", @"weights_sha256");
        require(cached_sha256(model_root / base_config, caches.base_config) ==
                    base_json_sha,
                "FastMetal LoRA base config SHA-256 does not match its manifest");

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

static void add_fastmetal_manifest_candidate(
        std::vector<std::filesystem::path>& candidates,
        const std::filesystem::path& path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    if (error || !std::filesystem::is_regular_file(absolute)) return;
    for (const auto& existing : candidates)
        if (existing == absolute) return;
    candidates.push_back(std::move(absolute));
}

static FastMetalLoRASelection resolve_fastmetal_lora(
        const std::filesystem::path& model_root, const LoRAAsset& requested,
        FastMetalHashCaches& caches) {
    const std::filesystem::path lora_path(requested.path);
    require(std::filesystem::is_regular_file(lora_path),
            "requested FastMetal LoRA file is missing: " + lora_path.string());
    std::vector<std::filesystem::path> candidates;
    add_fastmetal_manifest_candidate(candidates,
                                     lora_path.string() + ".manifest.json");
    add_fastmetal_manifest_candidate(candidates,
                                     lora_path.parent_path() /
                                         "fastmetal-lora.manifest.json");
    add_fastmetal_manifest_candidate(candidates,
                                     model_root / "fastmetal-lora.manifest.json");
    std::string failure = "no candidate manifest found";
    FastMetalLoRASelection selection;
    for (const auto& candidate : candidates) {
        if (inspect_fastmetal_lora_manifest(candidate, model_root, requested,
                                            caches, selection, failure))
            return selection;
    }
    require(false, "FastMetal LoRA has no provenance-verified premerged checkpoint: " +
                      failure);
    return selection;
}

static void validate_fastmetal_request(const Request& request) {
    require(request.model == "fastmetal-1.3b-qad",
            "request model differs from FastMetal session");
    require(request.operation == "video.generate",
            "FastMetal supports video.generate only");
    require(request.width % 16 == 0 && request.height % 16 == 0,
            "FastMetal dimensions must be multiples of 16");
    require(request.frames >= 1 && request.frames <= 161 &&
                request.frames % 4 == 1,
            "FastMetal frames must be 4n+1 in 1...161");
    require(request.fps == 16, "FastMetal requires 16 fps");
    require(request.steps == 3,
            "FastMetal QAD requires three denoising steps");
    require(request.inputs.empty(),
            "FastMetal text-to-video does not accept media inputs");
    require(request.loras.size() <= 1,
            "FastMetal supports at most one premerged LoRA");
    if (!request.loras.empty()) {
        require(request.loras.front().role == "transformer",
                "FastMetal LoRA role must be transformer");
        require(std::isfinite(request.loras.front().strength) &&
                    request.loras.front().strength > 0.0f &&
                    request.loras.front().strength <= 4.0f,
                "FastMetal LoRA strength must be in (0, 4]");
        require(request.execution != "gpu_ane",
                "FastMetal premerged LoRA currently supports GPU execution only");
    }
    require(request.residency == "resident" ||
                request.residency == "component_staged",
            "unsupported FastMetal residency");
}

static void profile_keys(NSDictionary* profile, NSArray* allowed) {
    NSSet* names = [NSSet setWithArray:allowed];
    for (NSString* key in profile)
        require([names containsObject:key], "unknown FastMetal profile field: " +
                std::string(key.UTF8String));
}

static bool is_bool(id value) {
    return value && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

static FastMetalConfig load_config(const std::filesystem::path& root) {
    FastMetalConfig config;
    std::filesystem::path profile = environment_path("TURBOCIDER_FASTMETAL_CONFIG");
    if (profile.empty() && std::filesystem::is_regular_file(root / kFastMetalConfigName))
        profile = root / kFastMetalConfigName;
    if (!profile.empty()) {
        profile = std::filesystem::absolute(profile);
        auto dictionary = read_json(profile);
        profile_keys(dictionary, @[@"schema", @"python", @"engine_root", @"script", @"worker",
                                   @"bridge_dir", @"decode_backend", @"text_encoder_dtype", @"vae_decode_dtype",
                                   @"mlx_dtype", @"mlx_quantization", @"mlx_compile",
                                   @"prompt_cache", @"taehv_parallel"]);
        require(string_value(dictionary, @"schema") == "turbocider-fastmetal-v1",
                "unsupported FastMetal profile schema");
        auto base = profile.parent_path();
        config.python = resolve_relative(base, string_value(dictionary, @"python"));
        config.engine_root = resolve_relative(base, string_value(dictionary, @"engine_root"));
        config.script = resolve_relative(base, string_value(dictionary, @"script"));
        config.worker = resolve_relative(base, string_value(dictionary, @"worker"));
        auto bridge = string_value(dictionary, @"bridge_dir");
        if (!bridge.empty()) config.bridge_dir = resolve_relative(base, bridge);
        config.decode_backend = string_value(dictionary, @"decode_backend", config.decode_backend);
        config.text_encoder_dtype = string_value(dictionary, @"text_encoder_dtype", config.text_encoder_dtype);
        config.vae_decode_dtype = string_value(dictionary, @"vae_decode_dtype", config.vae_decode_dtype);
        config.mlx_dtype = string_value(dictionary, @"mlx_dtype", config.mlx_dtype);
        config.mlx_quantization = string_value(dictionary, @"mlx_quantization", config.mlx_quantization);
        if (dictionary[@"mlx_compile"]) {
            require(is_bool(dictionary[@"mlx_compile"]), "FastMetal mlx_compile must be boolean");
            config.mlx_compile = [dictionary[@"mlx_compile"] boolValue];
        }
        if (dictionary[@"prompt_cache"]) {
            require(is_bool(dictionary[@"prompt_cache"]), "FastMetal prompt_cache must be boolean");
            config.prompt_cache = [dictionary[@"prompt_cache"] boolValue];
        }
        if (dictionary[@"taehv_parallel"]) {
            require(is_bool(dictionary[@"taehv_parallel"]), "FastMetal taehv_parallel must be boolean");
            config.taehv_parallel = [dictionary[@"taehv_parallel"] boolValue];
        }
        config.identity = profile.lexically_normal().string();
    } else {
        config.python = environment_path("TURBOCIDER_FASTMETAL_PYTHON");
        config.engine_root = environment_path("TURBOCIDER_FASTMETAL_ENGINE_ROOT");
        config.script = environment_path("TURBOCIDER_FASTMETAL_SCRIPT");
        config.worker = environment_path("TURBOCIDER_FASTMETAL_WORKER");
        config.bridge_dir = environment_path("TURBOCIDER_FASTMETAL_BRIDGE_DIR");
        config.identity = "environment";
    }
    require_regular(config.python, "Python executable");
    require_directory(config.engine_root, "engine root");
    require_regular(config.script, "inference script");
    require_regular(config.worker, "persistent worker");
    require(config.decode_backend == "taehv" || config.decode_backend == "wan-vae",
            "FastMetal decode_backend must be taehv or wan-vae");
    require(config.text_encoder_dtype == "bf16" || config.text_encoder_dtype == "fp16" ||
                config.text_encoder_dtype == "fp32",
            "FastMetal text_encoder_dtype must be bf16, fp16, or fp32");
    require(config.vae_decode_dtype == "bf16" || config.vae_decode_dtype == "fp16" ||
                config.vae_decode_dtype == "fp32",
            "FastMetal vae_decode_dtype must be bf16, fp16, or fp32");
    require(config.mlx_dtype == "fp16" || config.mlx_dtype == "bf16" || config.mlx_dtype == "fp32",
            "FastMetal mlx_dtype must be fp16, bf16, or fp32");
    require(config.mlx_quantization == "int8" || config.mlx_quantization == "none" ||
                config.mlx_quantization == "int4" || config.mlx_quantization == "mxfp8" ||
                config.mlx_quantization == "mxfp4" || config.mlx_quantization == "nvfp4",
            "FastMetal mlx_quantization is unsupported");
    return config;
}

static std::string line_tail(const std::string& log) {
    constexpr size_t limit = 16 * 1024;
    return log.size() <= limit ? log : log.substr(log.size() - limit);
}

static void write_all(int fd, const std::string& value) {
    size_t offset = 0;
    while (offset < value.size()) {
        const ssize_t written = ::write(fd, value.data() + offset,
                                        value.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        require(written > 0, "FastMetal worker pipe write failed");
        offset += static_cast<size_t>(written);
    }
}

static NSDictionary* decode_worker_message(const std::string& line) {
    if (line.empty() || line.front() != '{') return nil;
    NSData* data = [NSData dataWithBytes:line.data() length:line.size()];
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [object isKindOfClass:NSDictionary.class] ? object : nil;
}

class PersistentWorker {
    const FastMetalConfig& config_;
    std::filesystem::path root_;
    std::filesystem::path checkpoint_;
    std::string execution_;
    std::string manifest_;
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
                if (::waitpid(pid_, &status, WNOHANG) == pid_) break;
                usleep(10000);
            }
            if (::waitpid(pid_, &status, WNOHANG) == 0) {
                ::kill(pid_, SIGKILL);
                ::waitpid(pid_, &status, 0);
            }
        }
        if (input_ >= 0) ::close(input_);
        if (output_ >= 0) ::close(output_);
        pid_ = -1;
        input_ = output_ = -1;
        pending_.clear();
    }

    NSDictionary* next_message(std::atomic<bool>* cancelled) {
        for (;;) {
            if (cancelled && cancelled->load()) {
                stop_process();
                throw Cancelled();
            }
            auto newline = pending_.find('\n');
            if (newline != std::string::npos) {
                auto line = pending_.substr(0, newline);
                pending_.erase(0, newline + 1);
                if (auto message = decode_worker_message(line)) return message;
                if (!line.empty()) log_.append(line).append("\n");
                continue;
            }
            pollfd descriptor{output_, POLLIN | POLLHUP, 0};
            const int ready = ::poll(&descriptor, 1, 100);
            if (ready < 0 && errno == EINTR) continue;
            require(ready >= 0, "FastMetal worker pipe poll failed");
            if (ready == 0) continue;
            char buffer[8192];
            const ssize_t count = ::read(output_, buffer, sizeof(buffer));
            if (count > 0) {
                pending_.append(buffer, static_cast<size_t>(count));
                continue;
            }
            int status = 0;
            ::waitpid(pid_, &status, 0);
            pid_ = -1;
            throw std::runtime_error("FastMetal persistent worker exited: " + line_tail(log_));
        }
    }

    void start() {
        int input_pipe[2] = {-1, -1};
        int output_pipe[2] = {-1, -1};
        require(::pipe(input_pipe) == 0 && ::pipe(output_pipe) == 0,
                "cannot create FastMetal persistent worker pipes");
        std::vector<std::string> args = {
            config_.python.string(), config_.worker.string(),
            "--model-root", root_.string(), "--entrypoint", config_.script.string(),
            "--mlx-checkpoint", checkpoint_.string(),
            "--decode-backend", config_.decode_backend,
            "--text-encoder-dtype", config_.text_encoder_dtype,
            "--vae-decode-dtype", config_.vae_decode_dtype,
            "--mlx-dtype", config_.mlx_dtype,
            config_.mlx_compile ? "--mlx-compile" : "--no-mlx-compile",
            config_.prompt_cache ? "--prompt-cache" : "--no-prompt-cache",
        };
        if (config_.taehv_parallel) args.emplace_back("--taehv-parallel");
        if (!manifest_.empty()) {
            args.emplace_back("--ane-manifest"); args.emplace_back(manifest_);
            args.emplace_back("--ane-bridge-dir"); args.emplace_back(config_.bridge_dir.string());
        }
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        std::vector<std::string> environment;
        for (char** item = ::environ; item && *item; ++item) {
            std::string value(*item);
            if (!value.starts_with("PYTHONPATH=") &&
                !value.starts_with("TOKENIZERS_PARALLELISM="))
                environment.push_back(std::move(value));
        }
        std::string pythonpath = config_.engine_root.string();
        if (const char* current = std::getenv("PYTHONPATH"); current && *current)
            pythonpath += ":" + std::string(current);
        environment.push_back("PYTHONPATH=" + pythonpath);
        environment.push_back("TOKENIZERS_PARALLELISM=false");
        std::vector<char*> envp;
        envp.reserve(environment.size() + 1);
        for (auto& value : environment) envp.push_back(value.data());
        envp.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        require(posix_spawn_file_actions_init(&actions) == 0,
                "cannot initialize FastMetal spawn actions");
        posix_spawn_file_actions_adddup2(&actions, input_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDERR_FILENO);
        posix_spawn_file_actions_addclose(&actions, input_pipe[1]);
        posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
        pid_t child = -1;
        const int spawn_status = posix_spawn(
            &child, config_.python.c_str(), &actions, nullptr, argv.data(), envp.data());
        posix_spawn_file_actions_destroy(&actions);
        require(spawn_status == 0 && child > 0,
                "cannot spawn FastMetal persistent worker: " +
                    std::string(std::strerror(spawn_status)));
        ::close(input_pipe[0]); ::close(output_pipe[1]);
        input_ = input_pipe[1]; output_ = output_pipe[0]; pid_ = child;
#ifdef F_SETNOSIGPIPE
        ::fcntl(input_, F_SETNOSIGPIPE, 1);
#endif
        auto ready = next_message(nullptr);
        auto type = string_value(ready, @"type");
        if (type == "fatal" || type == "error")
            throw std::runtime_error("FastMetal worker startup failed: " +
                                     string_value(ready, @"error", line_tail(log_)));
        require(type == "ready", "FastMetal worker did not become ready: " + line_tail(log_));
    }

public:
    PersistentWorker(const FastMetalConfig& config, const std::filesystem::path& root,
                     const std::filesystem::path& checkpoint,
                     std::string execution, std::string manifest)
        : config_(config), root_(root), checkpoint_(checkpoint),
          execution_(std::move(execution)),
          manifest_(std::move(manifest)) { start(); }
    ~PersistentWorker() { stop_process(); }
    PersistentWorker(const PersistentWorker&) = delete;
    PersistentWorker& operator=(const PersistentWorker&) = delete;

    NSDictionary* generate(const Request& request, const std::filesystem::path& output,
                           const Event& event, std::atomic<bool>& cancelled) {
        NSMutableDictionary* payload = [@{ @"action": @"generate",
                                           @"request": @{ @"prompt": @(request.prompt.c_str()),
                                                          @"output": @(output.string().c_str()),
                                                          @"width": @(request.width),
                                                          @"height": @(request.height),
                                                          @"frames": @(request.frames),
                                                          @"fps": @(request.fps),
                                                          @"seed": @(request.seed),
                                                          @"dump": request.dump.empty() ? @"" : @(request.dump.c_str()) }} mutableCopy];
        write_all(input_, json(payload) + "\n");
        for (;;) {
            auto message = next_message(&cancelled);
            auto type = string_value(message, @"type");
            if (type == "progress") {
                auto phase = string_value(message, @"phase", "inference");
                int completed = [message[@"completed"] intValue];
                int total = [message[@"total"] intValue];
                if (phase == "dit") phase = "denoise";
                event(phase, completed, total);
            } else if (type == "result") {
                id value = message[@"result"];
                require([value isKindOfClass:NSDictionary.class], "FastMetal worker returned an invalid result");
                return value;
            } else if (type == "error" || type == "fatal") {
                throw std::runtime_error("FastMetal worker error: " + string_value(message, @"error", line_tail(log_)));
            }
        }
    }
};

class FastMetalSession final : public ModelSession {
    std::filesystem::path root_;
    FastMetalConfig config_;
    std::unique_ptr<PersistentWorker> worker_;
    std::string worker_key_;
    uint64_t request_sequence_ = 0;
    FastMetalHashCaches lora_hash_caches_;

    void ensure_worker(const Request& request,
                       const FastMetalLoRASelection& selection,
                       const Event& event) {
        std::string manifest;
        if (request.execution == "gpu_ane")
            manifest = std::filesystem::absolute(request.ane_manifest).string();
        auto key = request.execution + "\n" + manifest + "\n" +
            selection.checkpoint_dir.string() + "\n" +
            selection.checkpoint_sha256 + "\n" +
            selection.checkpoint_json_sha256 + "\n" + config_.identity;
        if (worker_ && key == worker_key_) return;
        worker_.reset();
        event("model_load", 0, 1);
        worker_ = std::make_unique<PersistentWorker>(
            config_, root_, selection.checkpoint_dir, request.execution, manifest);
        worker_key_ = std::move(key);
        event("model_load", 1, 1);
    }

public:
    explicit FastMetalSession(const std::filesystem::path& root)
        : root_(std::filesystem::absolute(root)), config_(load_config(root_)) {
        require_directory(root_, "model directory");
        require_regular(root_ / "mlx_dit.json", "mlx_dit.json");
        require_regular(root_ / "mlx_dit.safetensors", "mlx_dit.safetensors");
        require_directory(root_ / "tokenizer", "tokenizer directory");
        require_directory(root_ / "text_encoder", "text_encoder directory");
        require_directory(root_ / "vae", "VAE directory");
        require(sha256_file(root_ / "mlx_dit.safetensors") == kFastMetalCheckpointSha,
                "FastMetal checkpoint SHA-256 does not match the validated QAD release");
        require(sha256_file(root_ / "mlx_dit.json") ==
                    "db5603223f17a03051d36e4a3a218477dd48a7723117743d5d916d1db3f87691",
                "FastMetal mlx_dit.json SHA-256 does not match the validated QAD release");
    }

    bool uses_parent_mlx() const override { return false; }
    void unload() override { worker_.reset(); worker_key_.clear(); }

    RunResult generate(const Request& request, const Event& event,
                           std::atomic<bool>& cancelled) override {
        validate_fastmetal_request(request);
        FastMetalLoRASelection selection;
        selection.checkpoint_dir = root_;
        selection.checkpoint_sha256 = kFastMetalCheckpointSha;
        selection.checkpoint_json_sha256 = kFastMetalConfigSha;
        if (!request.loras.empty())
            selection = resolve_fastmetal_lora(
                root_, request.loras.front(), lora_hash_caches_);
        require(!request.output.empty() &&
                    std::filesystem::path(request.output).extension() == ".mp4",
                "FastMetal output must be an .mp4 file");
        if (request.execution == "gpu_ane") {
            require(request.loras.empty(),
                    "FastMetal premerged LoRA currently supports GPU execution only; "
                    "matching ANE artifacts are required before gpu_ane can be enabled");
            require(!request.ane_manifest.empty(),
                    "FastMetal gpu_ane requires an explicit ANE manifest");
            require(!config_.bridge_dir.empty(),
                    "FastMetal gpu_ane requires bridge_dir in the profile or environment");
            require_directory(config_.bridge_dir, "ANE bridge directory");
            validate_fastmetal_ane_manifest(request.ane_manifest);
        }
        checkpoint(cancelled);
        ensure_worker(request, selection, event);
        auto output = std::filesystem::absolute(request.output);
        std::error_code error;
        std::filesystem::create_directories(output.parent_path(), error);
        require(!error, "cannot create FastMetal output directory");
        const auto nonce = std::to_string(static_cast<long long>(::getpid())) + "." +
            std::to_string(++request_sequence_);
        auto temporary_output = output;
        temporary_output += ".tmp." + nonce + ".mp4";
        std::filesystem::remove(temporary_output, error);
        NSDictionary* metrics = nil;
        try {
            metrics = worker_->generate(request, temporary_output, event, cancelled);
        } catch (...) {
            std::filesystem::remove(temporary_output, error);
            if (cancelled.load()) {
                worker_.reset();
                worker_key_.clear();
            }
            throw;
        }
        require(std::filesystem::is_regular_file(temporary_output),
                "FastMetal worker completed without producing an output video");
        std::filesystem::rename(temporary_output, output, error);
        require(!error, "cannot atomically publish FastMetal output: " + error.message());
        event("export", 1, 1);
        NSMutableDictionary* result = [metrics mutableCopy];
        result[@"schema_version"] = @1;
        result[@"model"] = @(request.model.c_str());
        result[@"output"] = @(output.string().c_str());
        result[@"execution"] = @(request.execution.c_str());
        result[@"profile_identity"] = @(config_.identity.c_str());
        result[@"persistent_model"] = @YES;
        result[@"worker"] = @"fastmetal-python-persistent";
        result[@"mlx_checkpoint_sha256"] =
            @(selection.checkpoint_sha256.c_str());
        result[@"mlx_dit_json_sha256"] =
            @(selection.checkpoint_json_sha256.c_str());
        result[@"lora_fusion"] = request.loras.empty()
            ? @"none" : @"premerged_manifest_verified";
        if (!request.loras.empty()) {
            result[@"lora_manifest"] = @(selection.manifest.string().c_str());
            result[@"lora_sha256"] = @(selection.lora_sha256.c_str());
            result[@"lora_strength"] = @(selection.strength);
        }
        return native_run_result(result, request, make_plan(request));
    }
};

} // namespace

std::string validate_fastmetal_ane_manifest(const std::filesystem::path& path) {
    require(std::filesystem::is_regular_file(path),
            "FastMetal ANE manifest is missing: " + path.string());
    auto manifest = read_json(path);
    require(string_value(manifest, @"schema") == kFastMetalSchema,
            "unsupported FastMetal ANE manifest schema");
    NSDictionary* shape = [manifest[@"shape"] isKindOfClass:NSDictionary.class]
        ? manifest[@"shape"] : nil;
    require([shape isKindOfClass:NSDictionary.class], "FastMetal ANE manifest shape is missing");
    auto positive = [&](NSString* key) {
        id value = shape[key];
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                    [value unsignedLongLongValue] > 0,
                "FastMetal ANE manifest has an invalid shape field");
        return [value unsignedLongLongValue];
    };
    require(positive(@"rows") == 32760 && positive(@"hidden") == 1536 &&
                positive(@"intermediate") == 8960 && positive(@"ane_intermediate") == 4096 &&
                positive(@"gpu_intermediate") == 4864,
            "FastMetal ANE manifest shape does not match FastMetal 1.3B");
    const auto variant = string_value(manifest, @"variant");
    require(variant == "int8_pc" || variant == "fp16",
            "unsupported FastMetal ANE manifest variant");
    NSArray* blocks = [manifest[@"blocks"] isKindOfClass:NSArray.class]
        ? manifest[@"blocks"] : nil;
    require([blocks isKindOfClass:NSArray.class] && blocks.count == 30,
            "FastMetal ANE manifest must cover exactly 30 blocks");
    for (NSUInteger index = 0; index < blocks.count; ++index) {
        id value = blocks[index];
        require([value isKindOfClass:NSNumber.class] &&
                    [value integerValue] == static_cast<NSInteger>(index),
                "FastMetal ANE blocks must be sorted 0...29");
    }
    NSDictionary* artifacts = [manifest[@"artifacts"] isKindOfClass:NSDictionary.class]
        ? manifest[@"artifacts"] : nil;
    require([artifacts isKindOfClass:NSDictionary.class] && artifacts.count == 30,
            "FastMetal ANE manifest must contain 30 artifacts");
    auto parent = std::filesystem::absolute(path).parent_path();
    for (int block = 0; block < 30; ++block) {
        NSString* key = [NSString stringWithFormat:@"%d", block];
        id value = artifacts[key];
        require([value isKindOfClass:NSString.class],
                "FastMetal ANE artifact path must be a string");
        auto relative = std::filesystem::path([value UTF8String]);
        require(!relative.is_absolute() && relative == relative.filename(),
                "FastMetal ANE artifact path must be a file name");
        require_directory(parent / relative, "ANE block artifact");
    }
    auto checkpoint = string_value(manifest, @"checkpoint_sha256");
    auto config = string_value(manifest, @"mlx_dit_json_sha256");
    require(checkpoint == kFastMetalCheckpointSha &&
                config == kFastMetalConfigSha,
            "FastMetal ANE checkpoint identity is not trusted");
    auto checkpoint_path = string_value(manifest, @"checkpoint");
    if (!checkpoint_path.empty() && std::filesystem::is_regular_file(checkpoint_path))
        require(sha256_file(checkpoint_path) == checkpoint,
                "FastMetal ANE checkpoint SHA-256 does not match the manifest");
    return std::filesystem::absolute(path).string();
}

NSDictionary* preflight_fastmetal_lora(const std::filesystem::path& root,
                                       const LoRAAsset& lora) {
    require(std::filesystem::is_directory(root),
            "FastMetal model directory is missing: " + root.string());
    FastMetalHashCaches caches;
    auto selection = resolve_fastmetal_lora(
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

std::unique_ptr<ModelSession> create_fastmetal(const std::filesystem::path& root) {
    return std::make_unique<FastMetalSession>(root);
}
}
