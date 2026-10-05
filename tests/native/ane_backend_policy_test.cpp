#include "../../native/backends/ane_backend.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <cstdlib>
#include <iostream>
#include <type_traits>

using namespace tc::ane;
int main() {
    static_assert(std::is_base_of_v<Executor, RuntimeGraph>);
    unsetenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
    if (private_channel_count(10240) != 0) return 1;
    for (const auto &value : {"0","512","3072"}) { setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS",value,1); private_channel_count(10240); }
    for (const auto &value : {"","-512","513","10240","9999999999999"," 512"}) {
        setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS",value,1);
        try { private_channel_count(10240); return 1; } catch (const std::runtime_error &) {}
    }
    unsetenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
    unsetenv("TURBOCIDER_ANE_BACKEND"); unsetenv("TURBOCIDER_ALLOW_PRIVATE_ANE");
    if (configured_backend().preferred != BackendPreference::Public) return 1;
    for (const auto &mode : {"auto", "private", "public", "off"}) {
        setenv("TURBOCIDER_ANE_BACKEND", mode, 1); setenv("TURBOCIDER_ALLOW_PRIVATE_ANE", "1", 1);
        configured_backend();
    }
    setenv("TURBOCIDER_ANE_BACKEND", "private", 1); setenv("TURBOCIDER_ALLOW_PRIVATE_ANE", "0", 1);
    try { configured_backend(); return 1; } catch (const std::runtime_error &) {}
    for (const auto &value : {"", "PRIVATE", "gpu", " private", "1"}) {
        setenv("TURBOCIDER_ANE_BACKEND", value, 1);
        try { configured_backend(); return 1; } catch (const std::runtime_error &) {}
    }
    setenv("TURBOCIDER_ANE_BACKEND", "auto", 1); setenv("TURBOCIDER_ALLOW_PRIVATE_ANE", "yes", 1);
    try { configured_backend(); return 1; } catch (const std::runtime_error &) {}
    GraphShape shape{Kind::SwiGLU, 33, 64, 96, 32, 48, true};
    const auto mil = private_api::fp16_program(shape);
    for (const auto &text : {"tensor_buffer<fp16", "strides=[4096, 4096, 64, 1]", "slice_by_size", "concat", "exp(x = neg)", "real_div", " dg", " du", "packed_yh", "-> (y)"})
        if (mil.find(text) == std::string::npos) { std::cerr << text; return 1; }
    if (mil.find("sigmoid(") != std::string::npos || mil.find("BLOBFILE") != std::string::npos) return 1;
    for (const auto &bad : {
        GraphShape{Kind::GELU, 33, 64, 96, 32, 48, false},
        GraphShape{Kind::Matmul, 33, 64, 96, 32, 48, true},
        GraphShape{Kind::Matmul, 33, 64, 96, 0, 48, false},
        GraphShape{Kind::Matmul, 32769, 64, 96, 32, 48, false},
        GraphShape{Kind::Matmul, 33, 32768, 32768, 1, 1, false},
        GraphShape{Kind::SwiGLU, 1, 32768, 1, 32768, 1, false}}) {
        try { private_api::fp16_program(bad); return 1; } catch (const CapabilityError &) {}
    }
    const auto w8 = private_api::w8_swiglu_program({Kind::SwiGLU, 33, 384, 512, 256, 512, true});
    const auto large = private_api::w8_swiglu_program({Kind::SwiGLU, 2112, 384, 512, 256, 512, false});
    if (large.mil.find("[1, 1, 384, 2112]")==std::string::npos || large.packed_rows!=385) return 1;
    const auto full = private_api::w8_swiglu_program({Kind::SwiGLU,4224,384,512,256,512,false});
    if (full.mil.find("[1, 1, 384, 4224]")==std::string::npos || full.packed_rows!=385) return 1;
    try { private_api::w8_swiglu_program({Kind::SwiGLU,4225,384,512,256,512,false});return 1; }
    catch (const CapabilityError&) {}
    for (const auto &text : {"gw0q = slice_by_size(x = wg_t", "gw0 = dequantize(input = gw0q",
                            "gx256q = slice_by_size(x = x_t", "gx256 = dequantize(input = gx256q",
                            "dx256q = slice_by_size(x = hq", "dx256 = dequantize(input = dx256q"})
        if (w8.mil.find(text) == std::string::npos) { std::cerr << text; return 1; }
    if (w8.mil.find("wg_d = dequantize") != std::string::npos || w8.mil.find("xd = dequantize") != std::string::npos)
        return 1;
    std::cout << "PASS backend public default/private authorization and native MIL bounds/LoRA/exp lowering\n";
}
