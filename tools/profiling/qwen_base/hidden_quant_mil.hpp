#pragma once
// Diagnostic single-output consumer of the production hidden normalization.
// Uses the unchanged H512 constants; never embeds checkpoint weights.
#include "rotation_mil.hpp"

namespace tc::qwen_hidden_quant_lab {
using Program=qwen_rotation_lab::Program;
inline std::string normalization(const std::string&hr,int channels,int rows,int policy) {
    if((channels!=512&&channels!=6144)||(rows!=32&&rows!=1024)||policy<0||policy>2)
        throw std::invalid_argument("bounded hidden normalization policy/shape required");
    auto shape=[&](int c){return qwen_rotation_lab::shape(1,1,c,rows);};
    std::string body;
    auto f=[&](const std::string&name,int c,const std::string&op){body+="        tensor<fp16, "+shape(c)+"> "+name+" = "+op+";\n";};
    f("habs",channels,"abs(x = "+hr+")");
    f("peak",1,"reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
    f("floor",1,"maximum(x = peak, y = fp16(0x1p-12))");
    if(policy==0)f("ratio",channels,"real_div(x = "+hr+", y = floor)");
    else {
        f("inv_floor",1,policy==1?"real_div(x = fp16(1), y = floor)":"inverse(x = floor, epsilon = fp16(0))");
        f("ratio",channels,"mul(x = "+hr+", y = inv_floor)");
    }
    f("a8",channels,"mul(x = ratio, y = fp16(127))");
    body+="        tensor<int8, "+shape(channels)+"> hq = quantize(input = a8, scale = fp16(1), output_dtype = string(\"int8\"));\n";
    // Same dequantize scale consumed by production down MatMul. Return q/128
    // plus the existing scale as ONE output, not additional diagnostic taps.
    f("normalized",channels,"dequantize(input = hq, scale = fp16(0x1p-7))");
    f("hscale",1,"mul(x = floor, y = fp16(0x1.0204081020408p+0))");
    f("packed",channels+1,"concat(values = (normalized, hscale), axis = int32(2), interleave = bool(false))");
    const auto plane=std::to_string(uint64_t(channels+1)*rows),pitch=std::to_string(rows);
    const auto buffer="tensor_buffer<fp16, shape="+shape(channels+1)+", strides=["+plane+", "+plane+", "+pitch+", 1], interleave_factors=[1, 1, 1, 1]>";
    body+="        "+buffer+" y = tensor_to_tensor_buffer<ios17>(input = packed, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "+
          "strides = tensor<int64, [4]>(["+plane+", "+plane+", "+pitch+", 1]));\n";
    return body;
}
inline Program program(int channels,int rows,bool rotate,int policy) {
    auto p=qwen_rotation_lab::program(channels,rows,rotate?512:0);
    const auto begin=p.mil.find("        tensor_buffer<fp16");
    if(begin==std::string::npos||p.mil.find("        tensor_buffer<fp16",begin+1)!=std::string::npos)
        throw std::invalid_argument("original isolated H512 output contract changed");
    const auto end=p.mil.find('\n',begin);
    p.mil.replace(begin,end-begin+1,normalization(rotate?"value":"identity",channels,rows,policy));
    return p;
}
}
