#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace tc::ane {
enum class W8Activation { ExpDivide, Sigmoid, SiLU, SigmoidComplement };
inline W8Activation parse_w8_activation(const char *raw) {
    if(!raw || std::string_view(raw)=="exp")return W8Activation::ExpDivide;
    if(std::string_view(raw)=="sigmoid")return W8Activation::Sigmoid;
    if(std::string_view(raw)=="silu")return W8Activation::SiLU;
    if(std::string_view(raw)=="sigmoid-complement")return W8Activation::SigmoidComplement;
    throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION requires exp, sigmoid, silu or sigmoid-complement");
}
inline W8Activation configured_w8_activation() {
    return parse_w8_activation(std::getenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"));
}
inline std::string_view w8_activation_name(W8Activation value) {
    switch(value){case W8Activation::ExpDivide:return "exp";case W8Activation::Sigmoid:return "sigmoid";case W8Activation::SiLU:return "silu";
        case W8Activation::SigmoidComplement:return "sigmoid-complement";}
    throw std::runtime_error("invalid W8 activation policy");
}
} // namespace tc::ane
