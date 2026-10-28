#pragma once
// Isolated runtime-weight + runtime-factor prefix FFN. No base/adapter
// constants and no LoRA merge. The caller still joins the GPU complement
// and applies ONE complete down-LoRA to the returned corrected hidden.
#include "../../native/backends/private/ane_mil.hpp"
#include <cmath>
#include <sstream>

namespace tc::qwen21::runtime_factor_lab {
inline void replace_once(std::string &text,const std::string &from,const std::string &to){
    const auto at=text.find(from);
    if(at==std::string::npos||text.find(from,at+from.size())!=std::string::npos)
        throw std::invalid_argument("runtime factor emitter source contract changed");
    text.replace(at,from.size(),to);
}
inline std::string shape(int rows,int cols){return "[1, 1, "+std::to_string(rows)+", "+std::to_string(cols)+"]";}
inline std::string tensor(int rows,int cols){return "tensor<fp16, "+shape(rows,cols)+">";}
inline std::string buffer(int rows,int cols){
    const int pitch=(cols+31)/32*32;const auto plane=std::to_string(uint64_t(rows)*pitch);
    return "tensor_buffer<fp16, shape="+shape(rows,cols)+", strides=["+plane+", "+plane+", "+std::to_string(pitch)+", 1], interleave_factors=[1, 1, 1, 1]>";
}
inline std::string program(const ane::GraphShape &s,int rank,float gs=1.f,float us=1.f){
    if(s.kind!=ane::Kind::SwiGLU||!s.lora_inputs||s.rows<=0||s.rows>1024||s.hidden<=0||s.hidden>4096||s.width<=0||s.width>2048||
       s.hidden%32||s.width%32||(rank!=32&&rank!=128)||!std::isfinite(gs)||!std::isfinite(us)||std::abs(gs)>8||std::abs(us)>8)
        throw std::invalid_argument("runtime factor pilot shape/rank/scale unsupported");
    auto mil=ane::private_api::fp16_program(s);
    const auto scalar=[](float value){std::ostringstream out;out<<std::hexfloat<<value;return "fp16("+out.str()+")";};
    for(int kind=0;kind<2;++kind){
        const std::string suffix=kind==0?"g":"u",delta="d"+suffix,a="a"+suffix,b="b"+suffix;
        replace_once(mil,buffer(s.width,s.rows)+" "+delta,
            buffer(rank,s.hidden)+" "+a+", "+buffer(s.width,rank)+" "+b);
        const auto convert=[&](const std::string &name,int rows,int cols){return tensor(rows,cols)+" "+name+"_t = tensor_buffer_to_tensor<ios17>(input = "+name+");\n        ";};
        const auto low="rank_"+suffix,correction="correction_"+suffix;
        replace_once(mil,tensor(s.width,s.rows)+" "+delta+"_t = tensor_buffer_to_tensor<ios17>(input = "+delta+")",
            convert(a,rank,s.hidden)+convert(b,s.width,rank)+
            tensor(rank,s.rows)+" "+low+" = matmul(transpose_x = bool(false), transpose_y = bool(false), x = "+a+"_t, y = x_t);\n        "+
            tensor(s.width,s.rows)+" "+correction+" = matmul(transpose_x = bool(false), transpose_y = bool(false), x = "+b+"_t, y = "+low+");\n        "+
            tensor(s.width,s.rows)+" "+delta+"_t = mul(x = "+correction+", y = "+scalar(kind==0?gs:us)+")");
    }
    // Stable exp lowering, matching the existing Private prefix experiment.
    const auto t=tensor(s.width,s.rows)+" ";
    replace_once(mil,t+"neg = mul(x = gc, y = fp16(-1))",t+"abs_gate = abs(x = gc);\n        "+t+"neg = mul(x = abs_gate, y = fp16(-1))");
    replace_once(mil,t+"silu = real_div(x = gc, y = denom)",t+"negative_gate = mul(x = gc, y = eg);\n        tensor<bool, "+shape(s.width,s.rows)+
        "> positive_gate = greater_equal(x = gc, y = fp16(0));\n        "+t+"numerator = select(cond = positive_gate, a = gc, b = negative_gate);\n        "+t+"silu = real_div(x = numerator, y = denom)");
    return mil;
}
}
