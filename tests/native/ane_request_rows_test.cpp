#include "../../native/backends/ane_backend.hpp"
#include <fstream>
#include <iostream>

// Explicit opt-in component check; no checkpoint or full-model inference.
int main(int argc, char **argv) {
    try {
        if (argc != 2) return 2;
        const auto root = std::filesystem::absolute(argv[1]);
        if (!std::filesystem::create_directory(root)) return 2;
        const auto manifest = root / "shape.json";
        std::ofstream(manifest) << R"({"schema_version":1,"descriptor_version":1,
            "backend":"private_runtime_shape","kind":"swiglu","rows":1056,
            "hidden":128,"width":1024,"tile_k":128,"tile_n":512,
            "layout":"out_in","biases":false,"lora_inputs":true})";
        setenv("TURBOCIDER_PRIVATE_ANE_ROWS", "request", 1);
        setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS", "512", 1);
        setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH", "w8a8", 1);
        setenv("TURBOCIDER_PRIVATE_ANE_GPU_IO", "1", 1);
        setenv("TURBOCIDER_PRIVATE_ANE_CACHE_DIR", (root / "programs").c_str(), 1);
        using namespace tc::ane;
        for (int rows : {1024, 1088}) {
            auto built = build_runtime_executor(manifest, 512ull << 20,
                GraphGeometry{Kind::SwiGLU, 128, 1024, true, rows},
                {BackendPreference::Private, true});
            if (!built.self_test_passed || !built.executor ||
                built.executor->shape().rows != private_request_bucket(rows) ||
                !built.executor->shape().lora_inputs) return 1;
            std::cout << "{\"request_rows\":" << rows << ",\"bucket\":"
                << built.executor->shape().rows << ",\"private_self_test\":true}\n";
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
