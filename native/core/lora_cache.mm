#include "runtime.hpp"

#include <mach-o/dyld.h>
#include <limits.h>
#include <sys/wait.h>
#include <vector>

namespace tc {
namespace {

static std::filesystem::path cache_script() {
    if (const char* configured = std::getenv("TURBOCIDER_LORA_CACHE_SCRIPT")) {
        if (*configured && std::filesystem::is_regular_file(configured))
            return std::filesystem::absolute(configured);
    }
    if (const char* workspace = std::getenv("TURBOCIDER_WORKSPACE")) {
        auto path = std::filesystem::path(workspace) /
            "TurboCider/tools/native/lora_runtime_cache.py";
        if (std::filesystem::is_regular_file(path)) return std::filesystem::absolute(path);
    }
    uint32_t size = PATH_MAX;
    std::vector<char> executable(size);
    if (_NSGetExecutablePath(executable.data(), &size) != 0) {
        executable.resize(size);
        _NSGetExecutablePath(executable.data(), &size);
    }
    std::error_code error;
    auto executable_path = std::filesystem::canonical(executable.data(), error);
    if (!error) {
        auto directory = executable_path.parent_path();
        const std::filesystem::path candidates[] = {
            directory / "lora_runtime_cache.py",
            directory.parent_path().parent_path() /
                "tools/native/lora_runtime_cache.py",
            directory.parent_path() / "Resources/TurboCider/scripts/lora_runtime_cache.py",
        };
        for (const auto& candidate : candidates)
            if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    auto cwd = std::filesystem::current_path() /
        "tools/native/lora_runtime_cache.py";
    if (std::filesystem::is_regular_file(cwd)) return std::filesystem::absolute(cwd);
    return {};
}

static std::filesystem::path cache_python() {
    if (const char* configured = std::getenv("TURBOCIDER_PREPARE_PYTHON")) {
        if (*configured && std::filesystem::is_regular_file(configured))
            return std::filesystem::absolute(configured);
    }
    auto script = cache_script();
    if (!script.empty()) {
        auto bundled = script.parent_path().parent_path().parent_path() /
            "Python/bin/python";
        if (std::filesystem::is_regular_file(bundled)) return bundled;
    }
    return "/usr/bin/python3";
}

static std::string read_pipe(NSPipe* pipe) {
    NSData* data = [[pipe fileHandleForReading] readDataToEndOfFile];
    if (!data || !data.length) return {};
    return std::string(static_cast<const char*>(data.bytes), data.length);
}

} // namespace

RuntimeLoRACache ensure_runtime_lora_cache(
        const std::string& model,
        const std::filesystem::path& base,
        const LoRAAsset& adapter,
        const std::string& profile) {
    auto script = cache_script();
    require(!script.empty(),
            "TurboCider runtime LoRA cache helper is unavailable; set "
            "TURBOCIDER_LORA_CACHE_SCRIPT or TURBOCIDER_WORKSPACE");
    require(std::filesystem::exists(base),
            "runtime LoRA cache base checkpoint is missing: " + base.string());
    require(model == "h3" || model == "ltx",
            "runtime LoRA cache supports H3 or LTX only");
    require(std::filesystem::is_regular_file(adapter.path),
            "runtime LoRA adapter is missing: " + adapter.path);
    NSTask* task = [NSTask new];
    task.launchPath = @(cache_python().c_str());
    NSMutableArray<NSString*>* arguments = [NSMutableArray arrayWithObjects:
        @(script.c_str()), model == "h3" ? @"h3" : @"ltx",
        @(std::filesystem::absolute(base).c_str()),
        @(std::filesystem::absolute(adapter.path).c_str()),
        @"--strength", @(std::to_string(adapter.strength).c_str()),
        @"--role", @(adapter.role.c_str()),
        @"--profile", @(profile.c_str()), nil];
    const char* configured_cache = std::getenv("TURBOCIDER_LORA_CACHE_DIR");
    if (configured_cache && *configured_cache) {
        [arguments addObject:@"--cache-dir"];
        [arguments addObject:@(configured_cache)];
    }
    task.arguments = arguments;
    NSPipe* output = [NSPipe pipe];
    NSPipe* errors = [NSPipe pipe];
    task.standardOutput = output;
    task.standardError = errors;
    @try {
        [task launch];
        [task waitUntilExit];
    } @catch (NSException* exception) {
        throw std::runtime_error(
            "cannot launch runtime LoRA cache helper: " +
            std::string(exception.reason.UTF8String));
    }
    const std::string stdout_text = read_pipe(output);
    const std::string stderr_text = read_pipe(errors);
    require(task.terminationStatus == 0,
            "runtime LoRA cache helper failed: " +
            (stderr_text.empty() ? stdout_text : stderr_text));
    auto value = parse_json(stdout_text.c_str());
    require(value != nil && [value isKindOfClass:NSDictionary.class],
            "runtime LoRA cache helper returned invalid JSON");
    RuntimeLoRACache result;
    result.artifact = string_value(value, @"artifact");
    result.manifest = string_value(value, @"manifest");
    result.cache_key = string_value(value, @"cache_key");
    result.cache_hit = [value[@"cache_hit"] boolValue];
    require(!result.artifact.empty() && !result.manifest.empty() &&
                std::filesystem::exists(result.artifact) &&
                std::filesystem::is_regular_file(result.manifest),
            "runtime LoRA cache helper returned missing artifact");
    return result;
}

} // namespace tc
