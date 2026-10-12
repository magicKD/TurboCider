#include "../../native/backends/ane_public_w8_availability.hpp"
// Parsing/fallback only: no private build, graph compilation, model load or GPU work.
#include "../../native/backends/ane_backend.hpp"
#include <cstdlib>
#include <iostream>

using namespace tc::ane;
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    try {
        if (public_w8_array_type(false) != MLMultiArrayDataTypeFloat16) return 1;
#if __MAC_OS_X_VERSION_MAX_ALLOWED < 260000
        try { public_w8_array_type(true); return 1; }
        catch (const CapabilityError &) {}
#endif
        const std::string mode = argv[2];
        if (mode == "reject") {
            try { runtime_template_descriptor(argv[1]); return 1; }
            catch (const std::runtime_error &) { return 0; }
        }
        const auto descriptor = runtime_template_descriptor(argv[1]);
        const auto &shape = descriptor.shape;
        if (shape.kind != Kind::SwiGLU || shape.rows != 33 || shape.hidden != 128 ||
            shape.width != 512 || !shape.lora_inputs) return 1;
        if (mode == "public") return descriptor.has_public_artifact() ? 0 : 1;
        if (mode != "private" || descriptor.has_public_artifact()) return 1;
        unsetenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
        unsetenv("TURBOCIDER_PRIVATE_ANE_CACHE_DIR");
        const GraphGeometry geometry{Kind::SwiGLU, 128, 512, true};
        try { build_runtime_executor(argv[1], 256u << 20, geometry, {BackendPreference::Public, false}); return 1; }
        catch (const std::runtime_error &error) {
            if (std::string(error.what()).find("no Public Core ML artifact") == std::string::npos) return 1;
        }
        try { RuntimeGraph::prepare(argv[1], 256u << 20, true, geometry); return 1; }
        catch (const std::runtime_error &error) {
            if (std::string(error.what()).find("no Public Core ML artifact") == std::string::npos) return 1;
        }
        // An omitted/unavailable Private backend must return the existing GPU
        // fallback contract, never attempt loading a nonexistent public model.
        for (bool allow : {false, true}) {
            const auto fallback = build_runtime_executor(argv[1], 256u << 20, geometry,
                                                         {BackendPreference::Auto, allow});
            if (fallback.executor || fallback.fallback_reason.find("using GPU") == std::string::npos) return 1;
            if (allow && fallback.fallback_reason.find("omitted") == std::string::npos) return 1;
        }
        try { build_runtime_executor(argv[1], 256u << 20, geometry, {BackendPreference::Private, true}); return 1; }
        catch (const CapabilityError &) {}
        try { build_runtime_executor(argv[1], 256u << 20, {Kind::SwiGLU, 128, 513, true},
                                     {BackendPreference::Auto, false}); return 1; }
        catch (const std::runtime_error &) {}
        for (const char *bad : {"", "relative/cache"}) {
            setenv("TURBOCIDER_PRIVATE_ANE_CACHE_DIR", bad, 1);
            try { build_runtime_executor(argv[1], 256u << 20, geometry,
                                         {BackendPreference::Auto, true}); return 1; }
            catch (const std::runtime_error &error) {
                if (std::string(error.what()).find("CACHE_DIR") == std::string::npos) return 1;
            }
        }
        unsetenv("TURBOCIDER_PRIVATE_ANE_CACHE_DIR");
        std::cout << "PASS private shape ABI, Public rejection, unavailable Private GPU fallback and full geometry validation\n";
    } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
}
