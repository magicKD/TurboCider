#pragma once

#include "../../backends/mlx.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace tc {

// Opt-in host wall-clock diagnostics. "overlap" retains concurrent MLP
// submission; "serial" isolates the two branches; "gpu_split" measures the
// same unfused pre/post path with a compiled full-width GPU MLP; "gpu_detail"
// inserts per-operation completion boundaries in the dense GPU block. Extra
// evals affect scheduling, so always compare against an uninstrumented control.
struct ZProfileRequest;
inline thread_local ZProfileRequest *z_active_profile = nullptr;

struct ZProfileRequest {
    FILE *file = nullptr;
    std::string mode;
    unsigned request = 0, block = 0;
    ZProfileRequest *previous = z_active_profile;
    int exceptions = std::uncaught_exceptions();
    Clock::time_point start = Clock::now();

    ZProfileRequest(const Request &r) {
        const char *path = std::getenv("TURBOCIDER_Z_PROFILE");
        if (!path || !*path) return;
        require(std::filesystem::path(path).is_absolute(), "Z-Image profile path must be absolute");
        const char *configured = std::getenv("TURBOCIDER_Z_PROFILE_MODE");
        mode = configured ? configured : "overlap";
        require(mode == "overlap" || mode == "serial" || mode == "gpu_split" ||
                    mode == "gpu_detail",
                "Z-Image profile mode must be overlap, serial, gpu_split or gpu_detail");
        require((r.execution == "gpu" || r.execution == "gpu_ane") &&
                    ((mode != "gpu_split" && mode != "gpu_detail") ||
                     r.execution == "gpu"),
                "Z-Image gpu_split/gpu_detail profiling requires explicit GPU execution");
        file = std::fopen(path, "a");
        require(file != nullptr, "cannot open Z-Image profile output");
        static thread_local unsigned sequence = 0;
        request = ++sequence;
        z_active_profile = this;
        std::fprintf(file,
            "{\"type\":\"request\",\"request\":%u,\"mode\":\"%s\",\"execution\":\"%s\","
            "\"width\":%d,\"height\":%d,\"steps\":%d}\n",
            request, mode.c_str(), r.execution.c_str(), r.width, r.height, r.steps);
    }
    ~ZProfileRequest() {
        if (!file) return;
        std::fprintf(file,
            "{\"type\":\"request_end\",\"request\":%u,\"blocks\":%u,\"completed\":%s}\n",
            request, block, std::uncaught_exceptions() == exceptions ? "true" : "false");
        std::fclose(file);
        z_active_profile = previous;
    }
    void phase(const char *name) {
        if (!file) return;
        std::fprintf(file,
            "{\"type\":\"phase\",\"request\":%u,\"phase\":\"%s\",\"seconds\":%.9f,"
            "\"mlx_active_bytes\":%zu,\"mlx_peak_bytes\":%zu}\n",
            request, name, std::chrono::duration<double>(Clock::now() - start).count(),
            mx::get_active_memory(), mx::get_peak_memory());
    }
};

struct ZBlockProfile {
    ZProfileRequest *request = z_active_profile;
    Clock::time_point start{}, last{}, branch_start{}, gpu_start{}, coreml_start{};
    double pre = 0, pack = 0, submit = 0, gpu = 0, coreml = 0, tail = 0, window = 0;
    const std::string &name;
    bool hybrid;

    ZBlockProfile(const std::string &prefix, bool is_hybrid) : name(prefix), hybrid(is_hybrid) {
        if (request) start = last = Clock::now();
    }
    explicit operator bool() const { return request != nullptr; }
    bool split_gpu() const { return request && request->mode == "gpu_split"; }
    bool detail_gpu() const { return request && request->mode == "gpu_detail"; }
    bool serial() const { return request && request->mode == "serial"; }
    bool separate_pack() const { return serial() || split_gpu(); }
    double since(Clock::time_point time) const {
        return std::chrono::duration<double>(Clock::now() - time).count();
    }
    void pre_done(const std::vector<Tensor> &values) {
        if (!separate_pack()) return;
        mx::eval(values);
        pre = since(last);
        last = Clock::now();
    }
    void packed() {
        if (!request) return;
        if (separate_pack()) pack = since(last);
        else pre = since(last); // Includes input casting/padding in overlap mode.
        branch_start = gpu_start = Clock::now();
    }
    void submitted(const Tensor &value) {
        if (!request) return;
        submit = since(gpu_start);
        if (serial()) {
            mx::eval(value);
            gpu = since(gpu_start);
        }
        coreml_start = Clock::now();
    }
    void predicted(const Tensor &value) {
        if (!request) return;
        coreml = since(coreml_start);
        auto join = Clock::now();
        mx::eval(value);
        tail = since(join);
        window = since(branch_start);
        last = Clock::now();
    }
    void full_mlp(const Tensor &value) {
        if (!request) return;
        mx::eval(value);
        gpu = since(last);
        last = Clock::now();
    }
    void detail(const char *phase, const std::vector<Tensor> &values) {
        if (!detail_gpu()) return;
        mx::eval(values);
        const double seconds = since(last);
        last = Clock::now();
        std::fprintf(request->file,
            "{\"type\":\"detail\",\"request\":%u,\"index\":%u,\"name\":\"%s\","
            "\"phase\":\"%s\",\"seconds\":%.9f}\n",
            request->request, request->block, name.c_str(), phase, seconds);
    }
    void finish(const Tensor &value, bool fused = false) {
        if (!request) return;
        mx::eval(value);
        const double total = since(start), merge = fused ? 0 : since(last);
        std::fprintf(request->file,
            "{\"type\":\"block\",\"request\":%u,\"index\":%u,\"name\":\"%s\","
            "\"mode\":\"%s\",\"hybrid\":%s,\"fused\":%s,\"rows\":%d,"
            "\"pre_includes_pack\":%s,\"pre_mlp_seconds\":%.9f,\"pack_seconds\":%.9f,"
            "\"gpu_submit_seconds\":%.9f,\"gpu_isolated_seconds\":%.9f,"
            "\"coreml_seconds\":%.9f,\"gpu_tail_wait_seconds\":%.9f,"
            "\"branch_window_seconds\":%.9f,\"merge_seconds\":%.9f,\"total_seconds\":%.9f}\n",
            request->request, request->block++, name.c_str(), request->mode.c_str(),
            hybrid ? "true" : "false", fused ? "true" : "false", value.shape(1),
            separate_pack() ? "false" : "true", pre, pack, submit, gpu, coreml, tail,
            window, merge, total);
    }
};
} // namespace tc
