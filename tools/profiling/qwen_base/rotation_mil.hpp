#pragma once
// Diagnostic-only H512 factorization. No model/checkpoint constants.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc::qwen_rotation_lab {
struct Program {std::string mil;std::vector<uint8_t> constants;};
inline int sign(uint64_t seed,uint32_t i) {
    uint64_t x=seed+uint64_t(i)*0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;
    return ((x^(x>>31))&1)?-1:1;
}
inline std::string shape(int a,int b,int c,int d) {
    return "["+std::to_string(a)+", "+std::to_string(b)+", "+std::to_string(c)+", "+std::to_string(d)+"]";
}
inline Program program(int channels,int rows,int block) {
    if((channels!=512&&channels!=6144)||(rows!=32&&rows!=1024)||
       (block!=0&&block!=512&&block!=16&&block!=32&&block!=64&&block!=128&&block!=-32&&block!=-64&&block!=10032&&block!=10064))
        throw std::invalid_argument("bounded synthetic H512 diagnostic only");
    const bool flat=block>10000;if(flat)block-=10000;
    const bool batched=block<0||flat;block=std::abs(block);
    Program result;std::string body;
    auto line=[&](const std::string&s){body+="        "+s+";\n";};
    auto type=[](const std::string&s){return "tensor<fp16, "+s+">";};
    const auto standard=shape(1,1,channels,rows),conv_shape=shape(1,channels,1,rows);
    const auto plane=std::to_string(uint64_t(channels)*rows),pitch=std::to_string(rows);
    const auto buffer="tensor_buffer<fp16, shape="+standard+", strides=["+plane+", "+plane+", "+pitch+", 1], interleave_factors=[1, 1, 1, 1]>";
    line(type(standard)+" xt = tensor_buffer_to_tensor<ios17>(input = x)");
    std::string value="xt";
    if(block) {
        result.constants.resize(64);
        auto put=[&](size_t p,auto v){std::memcpy(result.constants.data()+p,&v,sizeof(v));};
        put(0,uint32_t(block==512?1:2));put(4,uint32_t(2));
        auto weights=[&](int width,bool first,const std::string&name) {
            const int out_rows=flat&&!first?channels/block:channels;
            const uint64_t record=result.constants.size(),data=record+64,bytes=uint64_t(out_rows)*width*2;
            result.constants.resize(data+bytes);put(record,uint32_t(0xdeadbeef));put(record+4,uint32_t(1));
            put(record+8,bytes);put(record+16,data);
            const float norm=1.f/std::sqrt(float(width));
            for(int out=0;out<out_rows;++out)for(int in=0;in<width;++in) {
                const int lane=out%width,source=(out%512/width)*width+in;
                const int s=((std::popcount(unsigned(lane&in))&1)?-1:1)*(first?sign(20260930,source):1);
                put(data+(uint64_t(out)*width+in)*2,std::bit_cast<uint16_t>(_Float16(norm*s)));
            }
            const auto ws=batched?(first?shape(channels/512,512/block,block,block):flat?shape(1,channels/512,512/block,512/block):shape(channels/512,block,512/block,512/block)):
                shape(channels,width,1,1);
            line(type(ws)+" "+name+" = const()[name = string(\""+name+"\"), val = "+type(ws)+
                 "(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64("+std::to_string(record)+")))]");
        };
        auto conv=[&](const std::string&name,const std::string&input,const std::string&w,int width) {
            line(type(conv_shape)+" "+name+" = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32("+
                 std::to_string(channels/width)+"), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), "+
                 "strides = tensor<int32, [2]>([1, 1]), weight = "+w+", x = "+input+")");
        };
        auto reshape=[&](const std::string&name,const std::string&input,const std::string&s) {
            line(type(s)+" "+name+" = reshape(x = "+input+", shape = tensor<int32, [4]>("+s+"))");
        };
        weights(block,true,"rotation_first");
        if(batched) {
            const auto ab=shape(channels/512,512/block,block,rows),ba=shape(channels/512,block,512/block,rows);
            reshape("a4","xt",ab);
            line(type(ab)+" stage_first = matmul(transpose_x = bool(false), transpose_y = bool(false), x = rotation_first, y = a4)");
            weights(512/block,false,"rotation_second");
            if(flat) {
                const auto flatten=shape(1,channels/512,512/block,block*rows);
                reshape("at4","stage_first",flatten);
                line(type(flatten)+" stage_second = matmul(transpose_x = bool(false), transpose_y = bool(false), x = rotation_second, y = at4)");
                value="stage_second";
            } else {
                line(type(ba)+" at4 = transpose(x = stage_first, perm = tensor<int32, [4]>([0, 2, 1, 3]))");
                line(type(ba)+" stage_second = matmul(transpose_x = bool(false), transpose_y = bool(false), x = rotation_second, y = at4)");
                line(type(ab)+" back4 = transpose(x = stage_second, perm = tensor<int32, [4]>([0, 2, 1, 3]))");
                value="back4";
            }
        } else {
          reshape("h4","xt",conv_shape);conv("stage_first","h4","rotation_first",block);value="stage_first";
          if(block!=512) {
            const int groups=channels/512,outer=512/block;
            weights(outer,false,"rotation_second");
            const auto ab=shape(groups,outer,block,rows),ba=shape(groups,block,outer,rows);
            reshape("a4","stage_first",ab);
            line(type(ba)+" at4 = transpose(x = a4, perm = tensor<int32, [4]>([0, 2, 1, 3]))");
            reshape("b4","at4",conv_shape);conv("stage_second","b4","rotation_second",outer);
            reshape("bt4","stage_second",ba);
            line(type(ab)+" back4 = transpose(x = bt4, perm = tensor<int32, [4]>([0, 2, 1, 3]))");
            value="back4";
          }
        }
        reshape("value",value,standard);value="value";
    } else {line(type(standard)+" identity = mul(x = xt, y = fp16(1))");value="identity";}
    line(buffer+" y = tensor_to_tensor_buffer<ios17>(input = "+value+", interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "+
         "strides = tensor<int64, [4]>(["+plane+", "+plane+", "+pitch+", 1]))");
    result.mil="program(1.3)\n{\n    func main_ane<ios18>("+buffer+" x) {\n"+body+"    } -> (y);\n}\n";
    return result;
}
}
