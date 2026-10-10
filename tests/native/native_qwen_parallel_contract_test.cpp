// CPU-only shared production contracts. No Backend/device is constructed.
#include "../../native/models/qwen21/native_block.hpp"

#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace qwen = tc::qwen21::native_graph;
namespace ffn = tc::metal_graph::dense_ffn;
using tc::metal_graph::ScalarType;
namespace {
void check(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function) {
    bool rejected=false;
    try { function(); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid native parallel contract accepted");
}
uint16_t bf16(float value) {
    auto bits=std::bit_cast<uint32_t>(value);
    return uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
float expand(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits)<<16); }
ffn::Layout layout(uint32_t rows,uint32_t columns,size_t extra,ScalarType type,size_t offset=0) {
    const auto item=tc::metal_graph::scalar_bytes(type),pitch=(columns+extra)*item;
    return {rows,columns,pitch,size_t(rows-1)*pitch+columns*item,offset,type};
}
}

int main() {
    using Phase=qwen::BlockPhase;
    const qwen::BlockGeometry production{},tiny{128,1,128,1024};
    const ffn::Geometry half{4096,12288,6144},tiny_half{128,1024,512};
    qwen::NativeBlock::validate_parallel_geometry(production,half,1024,34,Phase::SteadyDecode);
    qwen::NativeBlock::validate_parallel_geometry(tiny,tiny_half,32,3,Phase::SteadyDecode);
    for (auto wrong:{ffn::Geometry{4096,12288,12288},{4096,12288,4096},
                    {4096,12288,6143},{128,12288,6144},{4096,10240,5120}})
        rejects([&]{qwen::NativeBlock::validate_parallel_geometry(production,wrong,1024,34,Phase::SteadyDecode);});
    rejects([&]{qwen::NativeBlock::validate_parallel_geometry(production,half,1058,34,Phase::PrefixBuild);});
    rejects([&]{qwen::NativeBlock::validate_parallel_geometry(production,half,0,34,Phase::SteadyDecode);});
    rejects([&]{qwen::NativeBlock::validate_parallel_geometry(production,half,65536,1,Phase::SteadyDecode);});
    rejects([&]{qwen::NativeBlock::validate_parallel_geometry(production,half,UINT32_MAX,UINT32_MAX,Phase::SteadyDecode);});
    // The existing arena is reused; the two separately owned FP32 partials
    // must be included by the controller's own allocation plan.
    check(qwen::NativeBlock::workspace_bytes(production,1024,0,Phase::SteadyDecode)==134217728,
          "parallel plan must not silently grow the ordinary block arena");

    auto gpu=layout(3,128,8,ScalarType::FP32,16);
    auto ane=layout(3,128,4,ScalarType::FP32,8);
    auto output=layout(3,128,4,ScalarType::BF16,2);
    const auto checked=ffn::validate_join(128,gpu,ane,output);
    check(checked.parameters.rows==3 && checked.parameters.gpu_stride==136 &&
          checked.parameters.ane_stride==132 && checked.parameters.output_stride==132,
          "join must preserve physical pitches and dynamic logical rows");
    check(checked.gpu_bytes==gpu.view_size_bytes && checked.ane_bytes==ane.view_size_bytes &&
          checked.output_bytes==output.view_size_bytes,"join retained extents mismatch");
    auto bad=gpu; bad.type=ScalarType::BF16;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=ane; bad.type=ScalarType::FP16;
    rejects([&]{ffn::validate_join(128,gpu,bad,output);});
    bad=output; bad.type=ScalarType::FP32;
    rejects([&]{ffn::validate_join(128,gpu,ane,bad);});
    bad=gpu; bad.rows=2;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=gpu; bad.columns=127;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=gpu; --bad.view_size_bytes;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=gpu; ++bad.offset_bytes;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=gpu; bad.row_stride_bytes=127*4;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=gpu; bad.row_stride_bytes=std::numeric_limits<size_t>::max()-3;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=gpu; bad.offset_bytes=std::numeric_limits<size_t>::max()-3;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    bad=gpu; bad.rows=UINT32_MAX;
    rejects([&]{ffn::validate_join(128,bad,ane,output);});
    rejects([&]{ffn::validate_join(0,gpu,ane,output);});
    check(!ffn::ranges_overlap(16,64,80,32) && ffn::ranges_overlap(16,64,79,32) &&
          ffn::ranges_overlap(79,32,16,64),"join range separation must be symmetric");
    const auto maximum=std::numeric_limits<size_t>::max();
    check(!ffn::ranges_overlap(maximum-64,32,maximum-32,32) &&
          ffn::ranges_overlap(maximum-64,33,maximum-32,32),"join range checks must not add overflowing offsets");

    // These fixtures distinguish the new single-narrow join from rounding
    // each partial first. They document CPU arithmetic only, not Metal proof.
    const float g=1.00390625f,a=-1.f;
    check(bf16(g+a)==0x3b80 && bf16(expand(bf16(g))+expand(bf16(a)))==0,
          "early branch narrowing erases a representable cancellation result");
    const float largest=std::numeric_limits<float>::max();
    check(bf16(largest-largest)==0 && std::isnan(expand(bf16(largest))+expand(bf16(-largest))),
          "single FP32 join must precede the sole BF16 cast");
    check(bf16(1.f+0.00390625f)==0x3f80 && bf16(1.f+0.01171875f)==0x3f82,
          "BF16 ties-to-even fixture changed");
    check(ffn::source(half).find("tc_native_ffn_join_partials")==std::string::npos &&
          ffn::source(half,true).find("tc_native_ffn_join_partials")!=std::string::npos,
          "full GPU source must not add the optional parallel join kernel");
    check(ffn::precision_recipe.ends_with("-v3") && ffn::parallel_recipe.ends_with("-v1"),
          "full and parallel precision recipes must remain separate");
    std::cout<<"native_qwen_parallel_cpu_contract=PASS device_created=0 allocation_bytes=0\n";
}
