#pragma once
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace tc::ane {
enum class W8Projection { Descriptor, K4096, Single, UpSingle, DownSingle };
inline W8Projection parse_w8_projection(const char *raw) {
    if(!raw || std::string_view(raw)=="descriptor")return W8Projection::Descriptor;
    if(std::string_view(raw)=="k4096")return W8Projection::K4096;
    if(std::string_view(raw)=="single")return W8Projection::Single;
    if(std::string_view(raw)=="up-single")return W8Projection::UpSingle;
    if(std::string_view(raw)=="down-single")return W8Projection::DownSingle;
    throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_W8_PROJECTION requires descriptor, k4096, single, up-single or down-single");
}
inline W8Projection configured_w8_projection() {
    return parse_w8_projection(std::getenv("TURBOCIDER_PRIVATE_ANE_W8_PROJECTION"));
}
inline std::string_view w8_projection_name(W8Projection value) {
    switch(value){case W8Projection::Descriptor:return "descriptor";case W8Projection::K4096:return "k4096";case W8Projection::Single:return "single";
        case W8Projection::UpSingle:return "up-single";case W8Projection::DownSingle:return "down-single";}
    throw std::runtime_error("invalid W8 projection policy");
}
inline W8Projection w8_projection_leaf(W8Projection value,bool down) {
    switch(value){
        case W8Projection::Descriptor:case W8Projection::K4096:case W8Projection::Single:return value;
        case W8Projection::UpSingle:return down?W8Projection::Descriptor:W8Projection::Single;
        case W8Projection::DownSingle:return down?W8Projection::Single:W8Projection::Descriptor;
    }
    throw std::runtime_error("invalid W8 projection policy");
}
inline int w8_projection_k_tile(W8Projection value,int descriptor,int reduction,bool grouped) {
    if(reduction<1)throw std::runtime_error("W8 projection needs positive reduction dimension");
    if(grouped){if(value!=W8Projection::Descriptor)throw std::runtime_error("grouped A8 keeps its original K recipe");return 256;}
    switch(value){
        case W8Projection::Descriptor:return descriptor>0?std::min(descriptor,2048):2048;
        case W8Projection::K4096:return 4096;
        case W8Projection::Single:return reduction;
        case W8Projection::UpSingle:case W8Projection::DownSingle:break;
    }
    throw std::runtime_error("resolve mixed W8 projection to an operand policy before choosing its K tile");
}
} // namespace tc::ane
