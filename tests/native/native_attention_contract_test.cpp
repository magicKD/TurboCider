#include "../../native/backends/metal_graph/kernels/dense_attention.hpp"

#include <iostream>
#include <limits>

using namespace tc::metal_graph;
using namespace tc::metal_graph::dense_attention;
namespace {
size_t checks=0;
void check(bool value,const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F operation) {
    bool rejected=false;
    try { operation(); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid native attention contract accepted");
}
Layout head(uint32_t heads,uint32_t tokens,size_t row_stride=272,size_t extra_head=16,size_t offset=16) {
    const auto pitch=size_t(tokens)*row_stride+extra_head;
    return {heads,tokens,128,row_stride,pitch,size_t(heads-1)*pitch+size_t(tokens-1)*row_stride+256,
            offset,ScalarType::BF16};
}
}

int main() {
    try {
        for (uint32_t heads : {1U,30U,32U}) {
            validate_geometry({heads,128,1});
            check(validate_layout(head(heads,33))>0,"valid head-major geometry");
        }
        for (auto geometry : {AttentionGeometry{0,128,1},{33,128,1},{32,64,1},{32,128,2}})
            rejects([&] { validate_geometry(geometry); });

        for (uint32_t tokens : {1U,15U,16U,17U,31U,32U,33U,uint32_t(std::numeric_limits<int>::max())}) {
            const auto layout=head(1,tokens);
            check(validate_layout(layout)==size_t(tokens-1)*272+256,"dynamic token extent/tails");
        }
        auto valid=head(2,33),bad=valid;
        bad.view_size_bytes-=2; rejects([&] { validate_layout(bad); });
        for (auto type : {ScalarType::FP16,ScalarType::FP32,ScalarType::UInt32,ScalarType::Int8,ScalarType::Bytes}) {
            bad=valid; bad.type=type;
            rejects([&] { validate_layout(bad); });
        }
        bad=valid; bad.heads=0; rejects([&] { validate_layout(bad); });
        bad=valid; bad.heads=33; rejects([&] { validate_layout(bad); });
        bad=valid; bad.tokens=0; rejects([&] { validate_layout(bad); });
        bad=valid; bad.tokens=uint32_t(std::numeric_limits<int>::max())+1;
        rejects([&] { validate_layout(bad); });
        bad=valid; bad.dimension=64; rejects([&] { validate_layout(bad); });
        bad=valid; bad.token_stride_bytes=0; rejects([&] { validate_layout(bad); });
        bad=valid; bad.token_stride_bytes=254; rejects([&] { validate_layout(bad); });
        bad=valid; bad.token_stride_bytes=273; rejects([&] { validate_layout(bad); });
        bad=valid; bad.token_stride_bytes=(size_t(std::numeric_limits<int>::max()/32)+1)*2;
        rejects([&] { validate_layout(bad); });
        bad=valid; bad.head_stride_bytes=0; rejects([&] { validate_layout(bad); });
        bad=valid; bad.head_stride_bytes=1; rejects([&] { validate_layout(bad); });
        bad=valid; bad.head_stride_bytes=32*272+254; rejects([&] { validate_layout(bad); });
        bad=valid; bad.offset_bytes=1; rejects([&] { validate_layout(bad); });
        bad=valid; bad.offset_bytes=std::numeric_limits<size_t>::max()-1; bad.view_size_bytes=4;
        rejects([&] { validate_layout(bad); });
        bad=valid; bad.head_stride_bytes=std::numeric_limits<size_t>::max()-1;
        bad.offset_bytes=0; bad.view_size_bytes=std::numeric_limits<size_t>::max();
        rejects([&] { validate_layout(bad); });

        // A suffix keeps local Q/O data and the original physical head pitch.
        auto q=head(2,2,256,1536,512),k=head(2,5,256),v=k,o=head(2,2,256);
        const auto q_before=q;
        AttentionOptions suffix{true,3};
        auto checked=validate_bindings({2,128,1},q,k,v,o,suffix);
        check(checked.parameters.query_offset==3 && checked.parameters.query_length==2 &&
              checked.parameters.key_length==5,"explicit suffix logical offset retained");
        check(checked.parameters.query_stride==128 && checked.parameters.query_head_stride==1024 &&
              q.offset_bytes==q_before.offset_bytes,"logical offset must not shift physical Q");
        check(checked.parameters.reserved0==0 && checked.parameters.reserved1==0 &&
              checked.parameters.reserved2==0,"copied parameter padding initialized");
        q=head(32,33); k=head(32,65); v=k; o=head(32,33);
        checked=validate_bindings({32,128,1},q,k,v,o,{true,32});
        check(checked.parameters.query_length==33 && checked.parameters.key_length==65 &&
              checked.parameters.query_offset==32,"Q/K tail plus causal suffix geometry");
        rejects([&] { validate_bindings({32,128,1},q,k,v,o,{true,33}); });
        rejects([&] { validate_bindings({32,128,1},q,k,v,o,{true,std::numeric_limits<uint32_t>::max()}); });
        checked=validate_bindings({32,128,1},q,k,v,o,{false,std::numeric_limits<uint32_t>::max()});
        check(checked.parameters.query_offset==std::numeric_limits<uint32_t>::max(),"noncausal offset is mask-irrelevant");
        for (float scale : {0.0f,-1.0f,std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::max()})
            rejects([&] { validate_bindings({32,128,1},q,k,v,o,{false,0,scale}); });
        auto gqa=head(8,65);
        rejects([&] { validate_bindings({32,128,1},q,gqa,gqa,o,{}); });
        checked=validate_bindings({32,128,1,8,ScalarType::BF16},q,gqa,gqa,o,{true,32});
        check(checked.parameters.reserved0==4 && checked.key_bytes==validate_layout(gqa),"explicit GQA preserves original KV extent");
        rejects([&] { validate_geometry({32,128,1,7,ScalarType::BF16}); });
        rejects([&] { validate_bindings({32,128,1,8,ScalarType::BF16},q,k,v,o,{}); });
        Layout vision{16,33,80,320,33*320,16*33*320,0,ScalarType::FP32};
        checked=validate_bindings({16,80,1,0,ScalarType::FP32},vision,vision,vision,vision,{false,0,1.f/std::sqrt(72.f)});
        check(checked.parameters.query_stride==80 && checked.query_bytes==16*33*320 && checked.parameters.scale==1.f/std::sqrt(72.f),"D72 padded D80 keeps FP32 extents and original scale");
        auto short_vision=vision;--short_vision.view_size_bytes;
        rejects([&] { validate_layout(short_vision); });
        rejects([&] { validate_geometry({16,80,1,0,ScalarType::BF16}); });
        auto wrong_v=head(32,64);
        rejects([&] { validate_bindings({32,128,1},q,k,wrong_v,o,{}); });
        auto wrong_o=head(32,32);
        rejects([&] { validate_bindings({32,128,1},q,k,v,wrong_o,{}); });

        check(ranges_overlap(16,256,16,256),"equal intervals alias");
        check(ranges_overlap(16,256,270,2),"partial intervals alias");
        check(ranges_overlap(270,2,16,256),"interval alias symmetric");
        check(!ranges_overlap(16,256,272,2),"touching intervals disjoint");
        check(!ranges_overlap(16,0,16,256),"empty interval disjoint");
        check(NativeAttention::temporary_bytes()==0 && threadgroup_bytes==14848,"no N-squared score allocation");
        const auto shader=source();
        check(shader.find("#include \"mlx/")==std::string::npos && shader.find("/Users/")==std::string::npos,
              "vendored shader must not depend on external filesystem include paths");
        check(shader.find("tc_native_sdpa_d128_causal")!=std::string::npos &&
              shader.find("row_reduce<TCMax>")!=std::string::npos,"prepared tiled online-softmax shader present");
        std::cout<<"native_attention_cpu=PASS checks="<<checks<<" shader_bytes="<<shader.size()
                 <<" device_created=0 model_loads=0 score_buffer_bytes=0\n";
    } catch (const std::exception& e) {
        std::cerr<<"native_attention_cpu=FAIL "<<e.what()<<'\n';
        return 1;
    }
}
