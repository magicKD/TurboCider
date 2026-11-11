#pragma once

// Private session component, included after the native JSON, exact I/O and
// SHA-256 helpers. Shared with the CPU-only conditioning-cache tests.

struct Conditioning {
    std::filesystem::path directory;
    uint32_t rows = 0;
    uint32_t raw_rows = 0;
    bool connected = false;
    bool cache_hit = false;
    std::vector<uint16_t> video;
    std::vector<uint16_t> audio;
    std::vector<uint16_t> mask;
};

struct ConditioningCacheLocation {
    std::filesystem::path root;
    std::filesystem::path directory;
    std::string key;
};

static std::string ltx_cache_file_identity(
        const std::filesystem::path& path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error).lexically_normal();
    if (error) return path.string() + ":unresolved";
    auto bytes = std::filesystem::file_size(absolute, error);
    if (error) return absolute.string() + ":missing";
    auto mtime = std::filesystem::last_write_time(absolute, error);
    if (error) return absolute.string() + ":" + std::to_string(bytes) +
        ":unknown-mtime";
    return absolute.string() + ":" + std::to_string(bytes) + ":" +
        std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

static std::filesystem::path conditioning_directory(
        const std::filesystem::path& root) {
    if (std::filesystem::is_regular_file(root / "conditioning" /
                                         "conditioning.json"))
        return root / "conditioning";
    return {};
}

static std::filesystem::path raw_conditioning_directory(
        const std::filesystem::path& root) {
    if (std::filesystem::is_regular_file(root / "conditioning-raw" /
                                         "raw_conditioning.json"))
        return root / "conditioning-raw";
    return {};
}

static ConditioningCacheLocation ltx_conditioning_cache_location(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::string& prompt,
        const std::string& source_binding = {}) {
    ConditioningCacheLocation result;
    const char* configured = std::getenv(
        "TURBOCIDER_LTX_CONDITIONING_CACHE_DIR");
    if (!configured || !configured[0]) return result;
    std::error_code error;
    result.root = std::filesystem::absolute(configured, error).lexically_normal();
    if (error) return {};
    if (!source_binding.empty()) {
        // A derived cache entry belongs to the held request sources, rather
        // than to model paths reopened after the public SourceLease capture.
        result.key = sha256_text("ltx-connected-lease-cache-v1\n" + source_binding +
            "\n" + tc::catalog_runtime_identity() + "\n" +
            ltx_convrot_compute_recipe() + "\n" + prompt);
        result.directory = result.root / result.key / "conditioning";
        return result;
    }
    const auto absolute_root = std::filesystem::absolute(
        root, error).lexically_normal();
    if (error) return {};
    const auto gemma = root / "text_encoders" /
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";
    const auto tokenizer = root / "gemma4-12b-ltx-v1" / "tokenizer.json";
    const std::string material = absolute_root.string() + "\n" +
        ltx_cache_file_identity(checkpoint) + "\n" +
        ltx_cache_file_identity(gemma) + "\n" +
        ltx_cache_file_identity(tokenizer) + "\n" +
        tc::catalog_runtime_identity() + "\n" + ltx_convrot_compute_recipe() + "\n" + prompt;
    result.key = sha256_text(material);
    result.directory = result.root / result.key / "conditioning";
    return result;
}

static std::optional<std::string> conditioning_prompt(
        const std::filesystem::path& root,
        const std::filesystem::path& directory,
        NSDictionary* metadata, bool connected) {
    id direct = metadata[@"prompt"];
    if (direct) {
        require([direct isKindOfClass:NSString.class],
                "LTX conditioning prompt must be a string");
        return string_value(metadata, @"prompt");
    }
    if (!connected) return std::nullopt;
    const std::filesystem::path candidates[] = {
        directory.parent_path() / "raw" / "raw_conditioning.json",
        root / "conditioning-raw" / "raw_conditioning.json",
    };
    for (const auto& candidate : candidates) {
        if (!std::filesystem::is_regular_file(candidate)) continue;
        auto raw = read_json(candidate);
        require(string_value(raw, @"format") ==
                    "turbocider-ltx-raw-conditioning-v1",
                "unsupported LTX raw conditioning format");
        id prompt = raw[@"prompt"];
        require([prompt isKindOfClass:NSString.class],
                "LTX raw conditioning is missing its prompt identity");
        return string_value(raw, @"prompt");
    }
    return std::nullopt;
}

static Conditioning load_conditioning_directory(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::filesystem::path& directory,
        const std::string& requested_prompt,
        bool cache_hit,
        const std::string& source_binding = {}) {
    Conditioning result;
    result.directory = directory;
    result.cache_hit = cache_hit;
    if (result.directory.empty()) return result;
    result.connected = std::filesystem::is_regular_file(
        result.directory / "conditioning.json");
    auto metadata = read_json(result.directory /
        (result.connected ? "conditioning.json" : "raw_conditioning.json"));
    auto format = string_value(metadata, @"format");
    require((result.connected && (format == "turbocider-ltx-conditioning-v1" ||
                                  format == "ltx-mac-conditioning-v1")) ||
            (!result.connected && (format == "turbocider-ltx-raw-conditioning-v1" ||
                                   format == "ltx-mac-raw-conditioning-v1")),
            "unsupported LTX conditioning format");
    if (cache_hit) {
        require(result.connected,
                "LTX conditioning cache must contain connected tensors");
        auto cached = ltx_conditioning_cache_location(
            root, checkpoint, requested_prompt, source_binding);
        require(string_value(metadata, @"cache_identity") == cached.key,
                "LTX conditioning cache identity mismatch");
    }
    auto bound_prompt = conditioning_prompt(
        root, result.directory, metadata, result.connected);
    /* Unbound or stale fixtures are diagnostics, not a license to ignore the
     * request prompt.  Fall back to the dynamic Gemma path instead. */
    if (!bound_prompt || *bound_prompt != requested_prompt) return {};
    id rows = metadata[@"output_rows"];
    id raw_rows = metadata[@"input_rows"];
    id video_dim = metadata[@"video_dim"];
    id audio_dim = metadata[@"audio_dim"];
    if (!result.connected) {
        NSArray *video_shape = metadata[@"video_shape"];
        NSArray *audio_shape = metadata[@"audio_shape"];
        raw_rows = video_shape.count == 2 ? video_shape[0] : nil;
        video_dim = video_shape.count == 2 ? video_shape[1] : nil;
        audio_dim = audio_shape.count == 2 ? audio_shape[1] : nil;
        require(audio_shape.count == 2 &&
                [audio_shape[0] unsignedIntValue] == [raw_rows unsignedIntValue],
                "LTX raw conditioning row counts differ");
    }
    require((result.connected ? [rows isKindOfClass:NSNumber.class] :
             [metadata[@"expected_processed_rows"] isKindOfClass:NSNumber.class]) &&
            [raw_rows isKindOfClass:NSNumber.class] &&
            [video_dim isKindOfClass:NSNumber.class] &&
            [audio_dim isKindOfClass:NSNumber.class],
            "LTX conditioning metadata is incomplete");
    require([video_dim unsignedIntValue] == kLtxVideoDim &&
            [audio_dim unsignedIntValue] == kLtxAudioDim,
            "LTX conditioning dimensions do not match the native transformer");
    result.raw_rows = [raw_rows unsignedIntValue];
    result.rows = result.connected ? [rows unsignedIntValue] :
        [metadata[@"expected_processed_rows"] unsignedIntValue];
    require(result.raw_rows > 0 && result.raw_rows <= 4096 &&
            result.rows >= result.raw_rows && result.rows <= 4096,
            "LTX conditioning row count is out of range");
    result.video.resize(static_cast<size_t>(result.connected ? result.rows : result.raw_rows) * kLtxVideoDim);
    result.audio.resize(static_cast<size_t>(result.connected ? result.rows : result.raw_rows) * kLtxAudioDim);
    result.mask.resize(result.connected ? result.rows : result.raw_rows);
    const char* video_name = result.connected ? "video_context.bf16" : "raw_video_context.bf16";
    const char* audio_name = result.connected ? "audio_context.bf16" : "raw_audio_context.bf16";
    const char* mask_name = result.connected ? "text_mask.bf16" : "raw_text_mask.bf16";
    read_exact(result.directory / video_name, result.video.data(), result.video.size() * sizeof(uint16_t));
    read_exact(result.directory / audio_name, result.audio.data(), result.audio.size() * sizeof(uint16_t));
    read_exact(result.directory / mask_name, result.mask.data(), result.mask.size() * sizeof(uint16_t));
    if (cache_hit) {
        const auto video_hash = string_value(metadata, @"video_sha256");
        const auto audio_hash = string_value(metadata, @"audio_sha256");
        const auto mask_hash = string_value(metadata, @"mask_sha256");
        require(valid_sha256(video_hash) && valid_sha256(audio_hash) &&
                    valid_sha256(mask_hash),
                "LTX conditioning cache hashes are invalid");
        require(sha256_file(result.directory / video_name) == video_hash &&
                    sha256_file(result.directory / audio_name) == audio_hash &&
                    sha256_file(result.directory / mask_name) == mask_hash,
                "LTX conditioning cache content hash mismatch");
    }
    return result;
}

static Conditioning load_conditioning(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::string& requested_prompt,
        const std::string& source_binding = {}) {
    auto directory = source_binding.empty() ? conditioning_directory(root) :
        std::filesystem::path{};
    if (!directory.empty())
        return load_conditioning_directory(
            root, checkpoint, directory, requested_prompt, false);
    auto cached = ltx_conditioning_cache_location(
        root, checkpoint, requested_prompt, source_binding);
    if (!cached.directory.empty() && std::filesystem::is_regular_file(
            cached.directory / "conditioning.json")) {
        try {
            return load_conditioning_directory(
                root, checkpoint, cached.directory, requested_prompt, true, source_binding);
        } catch (...) {
            /* A cache entry is an optimization, never a reason to reject a
             * valid generation request. Recompute it from the bound Gemma
             * and connector instead of consuming stale/corrupt tensors. */
            std::error_code ignored;
            std::filesystem::remove_all(
                cached.directory.parent_path(), ignored);
        }
    }
    if (!source_binding.empty()) return {};
    directory = raw_conditioning_directory(root);
    return load_conditioning_directory(
        root, checkpoint, directory, requested_prompt, false);
}

static void write_ltx_conditioning_cache(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::string& prompt,
        const Conditioning& conditioning,
        const std::string& source_binding = {}) {
    if (!conditioning.connected || conditioning.rows == 0 ||
        conditioning.video.empty() || conditioning.audio.empty() ||
        conditioning.mask.empty()) return;
    auto location = ltx_conditioning_cache_location(root, checkpoint, prompt, source_binding);
    if (location.root.empty() || location.directory.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(location.root, error);
    require(!error, "cannot create LTX conditioning cache root: " +
                location.root.string());
    const auto final_directory = location.directory.parent_path();
    const auto final_parent = final_directory.parent_path();
    if (std::filesystem::is_regular_file(
            location.directory / "conditioning.json")) return;

    std::string template_text = (location.root / ".tmp-conditioning-XXXXXX").string();
    std::vector<char> mutable_template(template_text.begin(),
                                        template_text.end());
    mutable_template.push_back('\0');
    char* created = ::mkdtemp(mutable_template.data());
    require(created != nullptr,
            "cannot create LTX conditioning cache staging directory: " +
                std::string(std::strerror(errno)));
    const std::filesystem::path staging(created);
    auto cleanup = [&] {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
    };
    try {
        const auto staging_directory = staging / "conditioning";
        std::filesystem::create_directories(staging_directory, error);
        require(!error, "cannot create LTX conditioning cache directory");
        const auto video_path = staging_directory / "video_context.bf16";
        const auto audio_path = staging_directory / "audio_context.bf16";
        const auto mask_path = staging_directory / "text_mask.bf16";
        write_exact(video_path,
                    conditioning.video.data(),
                    conditioning.video.size() * sizeof(uint16_t),
                    "LTX cached video conditioning");
        write_exact(audio_path,
                    conditioning.audio.data(),
                    conditioning.audio.size() * sizeof(uint16_t),
                    "LTX cached audio conditioning");
        write_exact(mask_path,
                    conditioning.mask.data(),
                    conditioning.mask.size() * sizeof(uint16_t),
                    "LTX cached text mask");
        const auto video_hash = sha256_file(video_path);
        const auto audio_hash = sha256_file(audio_path);
        const auto mask_hash = sha256_file(mask_path);
        require(valid_sha256(video_hash) && valid_sha256(audio_hash) &&
                    valid_sha256(mask_hash),
                "cannot hash LTX conditioning cache tensors");
        auto metadata = json(@{
            @"format": @"turbocider-ltx-conditioning-v1",
            @"prompt": @(prompt.c_str()),
            @"checkpoint": @(checkpoint.string().c_str()),
            @"cache_identity": @(location.key.c_str()),
            @"input_rows": @(conditioning.rows),
            @"output_rows": @(conditioning.rows),
            @"register_rows": @128,
            @"video_dim": @4096,
            @"audio_dim": @2048,
            @"warmup": @0,
            @"cache": @YES,
            @"video_sha256": @(video_hash.c_str()),
            @"audio_sha256": @(audio_hash.c_str()),
            @"mask_sha256": @(mask_hash.c_str()),
        });
        std::ofstream metadata_stream(staging_directory / "conditioning.json");
        require(metadata_stream.good(),
                "cannot create LTX conditioning cache metadata");
        metadata_stream << metadata;
        metadata_stream.close();
        require(metadata_stream.good(),
                "cannot write LTX conditioning cache metadata");
        std::filesystem::create_directories(final_parent, error);
        require(!error, "cannot create LTX conditioning cache parent");
        std::filesystem::rename(staging, final_directory, error);
        if (error && !std::filesystem::is_directory(final_directory))
            require(false, "cannot publish LTX conditioning cache: " +
                    error.message());
        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

static Conditioning load_conditioning_with_sources(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::string& prompt,
        const streaming::SourceLease* sources) {
    if (sources) sources->revalidate_after_drain();
    auto result = load_conditioning(root, checkpoint, prompt,
        sources ? std::string(sources->digest()) : std::string{});
    if (sources) sources->revalidate_after_drain();
    return result;
}
