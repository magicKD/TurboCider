// Test-only service executable. Never add this translation unit to a product
// target: its four engine entry points intentionally replace inference with
// observable CPU lifecycle records. Planning still uses libturbocider.
#import <Foundation/Foundation.h>
#include "turbocider/turbocider.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unistd.h>

int tc_service_main(const char *, const char *, const char *);

struct tc_engine {
    unsigned creation_id;
    std::string model;
    std::string path;
    std::atomic<bool> cancelled{false};
};

namespace {
std::atomic<unsigned> next_creation_id{0};
std::mutex journal_mutex;

bool record(const char *event, unsigned creation_id) {
    std::lock_guard<std::mutex> lock(journal_mutex);
    const char *path = std::getenv("TC_SERVICE_REUSE_JOURNAL");
    if (!path || !*path) return false;
    std::ofstream journal(path, std::ios::app);
    journal << event << ' ' << creation_id << '\n';
    journal.flush();
    return journal.good();
}

int failure(char **error, const char *message) {
    if (error) *error = ::strdup(message);
    return 1;
}

NSDictionary *read_request(const char *text) {
    if (!text) return nil;
    NSData *data = [NSData dataWithBytes:text length:std::strlen(text)];
    id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [value isKindOfClass:NSDictionary.class] ? value : nil;
}
}

extern "C" int tc_engine_create_model(const char *model, const char *path,
                                        tc_engine **engine, char **error) {
    if (engine) *engine = nullptr;
    if (error) *error = nullptr;
    if (!engine || !model || !path) return failure(error, "missing fixture creation input");
    try {
        if (std::filesystem::path(path).filename() == "absent-create-fails") {
            if (!record("create_failed", 0))
                return failure(error, "cannot record fixture creation failure");
            return failure(error, "controlled CPU fixture creation failure");
        }
        auto *created = new tc_engine{++next_creation_id, model, path};
        if (!record("create", created->creation_id)) {
            delete created;
            return failure(error, "cannot record fixture creation");
        }
        *engine = created;
        return 0;
    } catch (...) {
        return failure(error, "cannot create CPU fixture engine");
    }
}

extern "C" void tc_engine_free(tc_engine *engine) {
    if (!engine) return;
    if (!record("free", engine->creation_id))
        std::cerr << "cannot record fixture engine release\n";
    delete engine;
}

extern "C" void tc_engine_cancel(tc_engine *engine) {
    if (engine) engine->cancelled.store(true);
}

extern "C" int tc_engine_generate(tc_engine *engine, const char *request,
                                    tc_event_callback callback, void *context,
                                    char **result, char **error) {
    if (result) *result = nullptr;
    if (error) *error = nullptr;
    if (!engine || !result) return failure(error, "missing fixture generation input");
    @autoreleasepool {
        NSDictionary *parsed = read_request(request);
        if (!parsed || ![parsed[@"model"] isEqual:@(engine->model.c_str())])
            return failure(error, "fixture engine/request model mismatch");
        if (!record("generate", engine->creation_id))
            return failure(error, "cannot record fixture generation");
        engine->cancelled.store(false);
        if (callback)
            callback("{\"schema_version\":1,\"phase\":\"fixture_cpu\",\"completed\":1,\"total\":1}", context);
        if ([parsed[@"prompt"] isEqual:@"CPU lifecycle fixture hold"]) {
            // Keep the real worker occupied so queued cancellation and queue
            // capacity are observable without weights, GPU work or timing luck.
            const char *release_path = std::getenv("TC_SERVICE_REUSE_HOLD_RELEASE");
            if (!release_path || !record("holding", engine->creation_id))
                return failure(error, "cannot initialize CPU fixture hold");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (!engine->cancelled.load() && !std::filesystem::exists(release_path)) {
                if (std::chrono::steady_clock::now() >= deadline)
                    return failure(error, "CPU fixture hold deadline exceeded");
                usleep(10000);
            }
        }
        if (engine->cancelled.load()) {
            if (error) *error = ::strdup("CPU fixture cancelled");
            return 2;
        }
        NSDictionary *receipt = @{
            @"schema_version": @1,
            @"creation_id": @(engine->creation_id),
            @"fixture_engine_model": @(engine->model.c_str()),
            @"fixture_model_path": @(engine->path.c_str()),
            @"fixture_cpu_no_inference": @YES,
        };
        NSData *data = [NSJSONSerialization dataWithJSONObject:receipt options:0 error:nil];
        if (!data) return failure(error, "cannot encode CPU fixture receipt");
        std::string text(static_cast<const char *>(data.bytes), data.length);
        *result = ::strdup(text.c_str());
        return *result ? 0 : failure(error, "cannot allocate CPU fixture receipt");
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc == 4 && std::strcmp(argv[1], "serve") == 0) {
            const auto executable = std::filesystem::absolute(argv[0]).string();
            return tc_service_main(argv[2], argv[3], executable.c_str());
        }
        if (argc == 4 && std::strcmp(argv[1], "ltx-worker") == 0) {
            // Exercise the real service-owned disposable child path without
            // delegating to the production video executor or touching weights.
            NSData *data = [NSData dataWithContentsOfFile:@(argv[3])];
            NSDictionary *request = data ?
                [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
            if (![request isKindOfClass:NSDictionary.class] ||
                ![request[@"model"] isEqual:@"ltx-2.5-distilled"] ||
                !record("external", 0)) return 2;
            NSDictionary *receipt = @{
                @"schema_version": @1,
                @"fixture_external_worker": @YES,
                @"fixture_cpu_no_inference": @YES,
                @"fixture_worker_pid": @(getpid()),
            };
            NSData *encoded = [NSJSONSerialization dataWithJSONObject:receipt options:0 error:nil];
            if (!encoded) return 2;
            std::cout.write(static_cast<const char *>(encoded.bytes), encoded.length);
            std::cout << '\n';
            return 0;
        }
        std::cerr << "test fixture expects serve SOCKET STATE or ltx-worker MODEL REQUEST\n";
        return 2;
    }
}
