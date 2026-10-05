#include "../../native/backends/ane_backend.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <cstdlib>
#include <bit>
#include <cstring>
#include <iostream>
#include <type_traits>

using namespace tc::ane;
int main() {
    static_assert(std::is_base_of_v<Executor, RuntimeGraph>);
    unsetenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
    unsetenv("TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES");
    if(configured_convrot_bf16_boundaries())return 1;
    for(const char *value:{"0","1"}) {setenv("TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES",value,1);configured_convrot_bf16_boundaries();}
    for(const char *value:{"","true","2"," 1"}) {
        setenv("TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES",value,1);
        try {configured_convrot_bf16_boundaries();return 1;}catch(const std::runtime_error&) {}
    }
    unsetenv("TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES");
    if (private_channel_count(10240) != 0) return 1;
    setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","auto",1);
    if(private_channel_count(10240)!=-1 || resolved_private_channel_count(10240,4096)!=4096 ||
        resolved_private_channel_count(10240,0)!=0)return 1;
    try {resolved_private_channel_count(10240);return 1;}catch(const std::runtime_error&) {}
    for(int bad:{-1,513,10240}) {
        try {resolved_private_channel_count(10240,bad);return 1;}catch(const std::runtime_error&) {}
    }
    setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","4096",1);
    if(resolved_private_channel_count(10240)!=4096)return 1;
    try {resolved_private_channel_count(10240,5120);return 1;}catch(const std::runtime_error&) {}
    unsetenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
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
    const auto comfy=private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,true},0,1,W8Basis::ComfyH256);
    const auto rounded=private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,false},0,1,W8Basis::ComfyH256,0,0,true);
    for(const char *name:{"gate_bf16_value_out","up_bf16_value_out","silu_bf16_value_out","hidden_bf16_value_out","rotated_hidden_bf16_value_out"})
        if(rounded.mil.find(name)==std::string::npos)return 1;
    if(rounded.constants!=comfy.constants || rounded.mil.find("round(")!=std::string::npos ||
        rounded.mil.find("fp16(0x1p-23)")!=std::string::npos || rounded.mil.find("gate_bf16_value_even_lower = add(")==std::string::npos)return 1;
    if(rounded.mil.find("rotated_hidden_bf16_value_guard_token")==std::string::npos ||
       rounded.mil.find("select(cond = carrier_ok, a = hscale_positive, b = fp16(-1))")==std::string::npos)return 1;
    for(const auto &bad:{GraphShape{Kind::SwiGLU,33,512,512,256,512,true},GraphShape{Kind::SwiGLU,33,384,512,256,512,false}}) {
        try {private_api::w8_swiglu_program(bad,0,1,W8Basis::ComfyH256,0,0,true);return 1;}catch(const CapabilityError&) {}
    }
    try {private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,false},0,4,W8Basis::ComfyH256,0,0,true);return 1;}
    catch(const CapabilityError&) {}
    const auto grouped=private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,true},0,1,W8Basis::ComfyH256,256);
    if(grouped.packed_rows!=comfy.packed_rows || grouped.constants!=comfy.constants ||
        grouped.mil.find("[1, 2, 256, 33]")==std::string::npos ||
        grouped.mil.find("gscaled256")==std::string::npos || grouped.mil.find("hratio_group")==std::string::npos)return 1;
    const auto input_only=private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,true},0,1,W8Basis::ComfyH256,256,0);
    const auto hidden_only=private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,true},0,1,W8Basis::ComfyH256,0,256);
    if(input_only.mil.find("tx_ratio")==std::string::npos || input_only.mil.find("hratio_group")!=std::string::npos ||
        hidden_only.mil.find("tx_ratio")!=std::string::npos || hidden_only.mil.find("hratio_group")==std::string::npos)return 1;
    try {private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,true},20260930,1,W8Basis::SylvesterDH,256);return 1;}
    catch(const CapabilityError&) {}
    if(comfy.constants.size()!=128+512*256*2 || comfy.mil.find("groups = int32(2)")==std::string::npos ||
        comfy.mil.find("[512, 256, 1, 1]")==std::string::npos) return 1;
    constexpr int h4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
    for(int out=0;out<512;++out)for(int in=0;in<256;++in) {
        int sign=1;
        for(int shift=0;shift<8;shift+=2)sign*=h4[(out>>shift)&3][(in>>shift)&3];
        uint16_t bits;std::memcpy(&bits,comfy.constants.data()+128+(out*256+in)*2,2);
        if(bits!=std::bit_cast<uint16_t>(_Float16(sign/16.f)))return 1;
    }
    try {private_api::w8_swiglu_program({Kind::SwiGLU,33,512,512,256,512,true},1,1,W8Basis::ComfyH256);return 1;}
    catch(const CapabilityError&) {}
    std::cout << "PASS backend public default/private authorization and native MIL bounds/LoRA/exp lowering\n";
}
