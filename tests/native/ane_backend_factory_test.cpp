#include "../../native/backends/ane_backend.hpp"
#include <iostream>

using namespace tc::ane;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        const GraphGeometry geometry{Kind::SwiGLU, 64, 96, true};
        auto public_graph = build_runtime_executor(argv[1], 256u << 20, geometry, {BackendPreference::Public, false});
        if (!public_graph.executor || public_graph.executor->backend() != BackendKind::PublicCoreML) return 1;
        std::string error;
        if (!public_graph.executor->self_test(error)) throw std::runtime_error(error);
        auto allowed = build_runtime_executor(argv[1], 256u << 20, geometry, {BackendPreference::Auto, true});
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
        if (!allowed.executor || allowed.executor->backend() != BackendKind::PrivateANE || !allowed.self_test_passed ||
            allowed.self_test_seconds <= 0 || !allowed.fallback_reason.empty()) return 1;
        std::cout << "PASS private-enabled factory selects native private executor after numerical self-test\n";
#else
        if (!allowed.executor || allowed.executor->backend() != BackendKind::PublicCoreML ||
            allowed.fallback_reason.find("omitted") == std::string::npos) return 1;
        try { build_runtime_executor(argv[1], 256u << 20, geometry, {BackendPreference::Private, true}); return 1; }
        catch (const CapabilityError &) {}
        std::cout << "PASS public-only factory falls back to Core ML; explicit unavailable private reports capability failure\n";
#endif
        auto denied = build_runtime_executor(argv[1], 256u << 20, geometry, {BackendPreference::Auto, false});
        if (!denied.executor || denied.executor->backend() != BackendKind::PublicCoreML) return 1;
        auto off = build_runtime_executor("nonexistent", 0, geometry, {BackendPreference::Off, false});
        if (off.executor || off.fallback_reason.empty()) return 1;
        try { build_runtime_executor(argv[1], 256u << 20, geometry, {BackendPreference::Private, false}); return 1; }
        catch (const std::runtime_error &) {}
        try { build_runtime_executor(argv[1], 256u << 20, {Kind::SwiGLU, 64, 97, true}, {BackendPreference::Auto, true}); return 1; }
        catch (const std::runtime_error &) {}
        std::cout << "PASS common geometry, public default, private authorization, off policy and LoRA-input contract\n";
    } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
}
