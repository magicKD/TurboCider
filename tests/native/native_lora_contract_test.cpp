#include "../../native/backends/metal_graph/kernels/native_lora.hpp"

#include <array>
#include <iostream>
#include <limits>

using tc::metal_graph::ScalarType;
using tc::metal_graph::scalar_bytes;
using namespace tc::metal_graph::native_lora;

namespace {
size_t checks=0;
void check(bool value,const char* message) {
    ++checks;
    if(!value)throw std::runtime_error(message);
}
template<class F> void rejects(F operation) {
    bool rejected=false;
    try { operation(); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid native LoRA contract accepted");
}
Layout matrix(uint32_t rows,uint32_t columns,uintptr_t storage,
              ScalarType type=ScalarType::BF16,size_t padding=4,size_t offset=32) {
    const size_t item=scalar_bytes(type),pitch=(size_t(columns)+padding)*item;
    return {rows,columns,pitch,size_t(rows-1)*pitch+size_t(columns)*item,offset,type,storage};
}
struct Bindings {
    Geometry geometry{2,5,128};
    Layout input=matrix(3,2,1),base=matrix(3,5,2),output=matrix(3,5,3);
    Layout low=matrix(3,128,4,ScalarType::FP32);
    std::array<TermLayout,1> terms{{{matrix(3,2,5),matrix(3,3,6),1.0F,2,3}}};
    CheckedBindings validate(std::optional<Layout> bias={}) const {
        return validate_bindings(geometry,input,terms,base,output,low,bias);
    }
};
CheckedTerm checked_sg_term(uint32_t rows,uint32_t columns,uint32_t count=128,
                           uint32_t first=1,ScalarType a_type=ScalarType::BF16,
                           ScalarType b_type=ScalarType::BF16) {
    const Geometry geometry{12288,32768,128};
    const auto input=matrix(rows,columns,1),base=matrix(rows,count+first,2);
    const auto output=matrix(rows,count+first,3),low=matrix(rows,128,4,ScalarType::FP32);
    const std::array<TermLayout,1> terms{{{matrix(128,columns,5,a_type),
        matrix(count,128,6,b_type),1,first,count}}};
    return validate_bindings(geometry,input,terms,base,output,low).terms.front();
}
void sg_contract() {
    check(scalar_threadgroup_bytes==5120 && sg_mma_threadgroup_bytes==6144,
          "separate scalar/candidate shared-memory footprints");
    validate_variant(Geometry{2,5,3},KernelVariant::Scalar);
    validate_variant(Geometry{},KernelVariant::SgMmaR128);
    for(const auto rank:{1U,16U,127U})
        rejects([&] { validate_variant(Geometry{32,128,rank},KernelVariant::SgMmaR128); });
    rejects([&] { validate_variant(Geometry{0,128,128},KernelVariant::Scalar); });
    rejects([&] { validate_variant(Geometry{32,0,128},KernelVariant::SgMmaR128); });
    const auto unknown=static_cast<KernelVariant>(255);
    rejects([&] { validate_variant(Geometry{},unknown); });
    rejects([&] { (void)recipe_for_variant(unknown); });
    rejects([&] { validate_capability(unknown,6144,32,128); });
    check(recipe_for_variant(KernelVariant::Scalar)==precision_recipe,
          "default scalar precision identity remains v1");
    check(recipe_for_variant(KernelVariant::SgMmaR128)==
          "runtime-lora-fp32-sgmma8x8-aligned-r128-scalar-tail-ordered-bf16-bias-last-v2" &&
          recipe_for_variant(KernelVariant::SgMmaR128)!=recipe_for_variant(KernelVariant::Scalar),
          "SG candidate has a distinct recipe and cannot inherit scalar qualification");
    for(const auto variant:{KernelVariant::Scalar,KernelVariant::SgMmaR128}) {
        const size_t needed=variant==KernelVariant::Scalar?5120:6144;
        validate_capability(variant,needed,32,128);
        validate_capability(variant,needed+1,32,1024);
        check(true,"exact and larger checked capabilities accepted");
        rejects([&] { validate_capability(variant,needed-1,32,128); });
        for(const auto width:{0U,31U,64U})
            rejects([&] { validate_capability(variant,needed,width,128); });
        for(const auto threads:{size_t{0},size_t{127}})
            rejects([&] { validate_capability(variant,needed,32,threads); });
    }

    for(const auto rows:{8U,16U,32U,8192U})
        for(const auto columns:{32U,4096U,12288U})
            check(sg_mma_eligible(checked_sg_term(rows,columns)),
                  "aligned rank128 terms retain padded pitches and selected nonzero output offset");
    for(const auto a_type:{ScalarType::BF16,ScalarType::FP32})
        for(const auto b_type:{ScalarType::BF16,ScalarType::FP32})
            check(sg_mma_eligible(checked_sg_term(8,32,32,0,a_type,b_type)),
                  "SG routing permits every BF16/FP32 factor combination after exact float decoding");
    const auto valid=checked_sg_term(8,4096);
    static_assert(noexcept(sg_mma_eligible(valid)));
    auto invalid=valid;
    auto ineligible=[&](const CheckedTerm& term) {
        check(!sg_mma_eligible(term),"unsafe or scalar-tail term routed to the SG candidate");
    };
    for(const auto rank:{0U,3U,16U,127U,129U}) {
        invalid=valid;invalid.low.rank=rank;invalid.apply.rank=rank;ineligible(invalid);
    }
    invalid=valid;invalid.apply.rank=127;ineligible(invalid);
    for(const auto rows:{0U,3U,7U,9U,33U,8200U}) {
        invalid=valid;invalid.low.rows=rows;invalid.apply.rows=rows;ineligible(invalid);
    }
    invalid=valid;invalid.apply.rows=16;ineligible(invalid);
    for(const auto columns:{0U,31U,33U,12320U}) {
        invalid=valid;invalid.low.input_columns=columns;ineligible(invalid);
    }
    for(const auto count:{0U,31U,33U,32800U}) {
        invalid=valid;invalid.apply.output_count=count;ineligible(invalid);
    }
    invalid=valid;invalid.low.a_type=2;ineligible(invalid);
    invalid=valid;invalid.apply.b_type=2;ineligible(invalid);
    invalid=valid;invalid.apply.low_pitch+=4;ineligible(invalid);
    for(const auto pitch:{0U,508U,514U}) {
        invalid=valid;invalid.low.low_pitch=pitch;invalid.apply.low_pitch=pitch;ineligible(invalid);
    }
    for(const auto pitch:{0U,8190U,8193U}) {
        invalid=valid;invalid.low.input_pitch=pitch;ineligible(invalid);
    }
    for(const auto pitch:{8190U,8193U}) {
        invalid=valid;invalid.low.a_pitch=pitch;ineligible(invalid);
    }
    invalid=valid;invalid.low.a_type=1;invalid.low.a_pitch=16386;ineligible(invalid);
    for(const auto pitch:{254U,257U}) {
        invalid=valid;invalid.apply.b_pitch=pitch;ineligible(invalid);
    }
    invalid=valid;invalid.apply.b_type=1;invalid.apply.b_pitch=514;ineligible(invalid);
    for(const auto pitch:{256U,259U}) {
        invalid=valid;invalid.apply.output_pitch=pitch;ineligible(invalid);
    }
    invalid=valid;invalid.apply.first_output=UINT32_MAX;ineligible(invalid);
    for(const auto scale:{std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        invalid=valid;invalid.apply.effective_scale=scale;ineligible(invalid);
    }
    for(const auto scale:{-2.0F,0.0F,-0.0F}) {
        auto scaled=valid;scaled.apply.effective_scale=scale;
        check(sg_mma_eligible(scaled),"finite negative and signed-zero scales remain eligible");
    }
    // Scalar geometry validation remains independent of candidate eligibility.
    check(!sg_mma_eligible(checked_sg_term(3,31,33,1)),
          "otherwise valid M/K/N tails retain the scalar route");
}
}

int main() {
    try {
        sg_contract();
        for(const auto geometry:{Geometry{1,1,1},{2,5,3},{4096,12288,16},
                                 {12288,32768,128}})validate_geometry(geometry);
        for(const auto geometry:{Geometry{0,1,1},{1,0,1},{1,1,0},
                                 {12289,1,1},{1,32769,1},{1,1,129}})
            rejects([&] { validate_geometry(geometry); });

        const auto padded=matrix(3,5,1);
        check(extent(padded)==46,"padded row extent retains physical pitch");
        auto bad=padded;
        bad.view_size_bytes-=2;rejects([&] { (void)extent(bad); });
        bad=padded;bad.storage=0;rejects([&] { (void)extent(bad); });
        for(const auto type:{ScalarType::Bytes,ScalarType::FP16,ScalarType::UInt32,
                             ScalarType::Int8,static_cast<ScalarType>(255)}) {
            bad=padded;bad.type=type;rejects([&] { (void)extent(bad); });
        }
        bad=padded;bad.rows=0;rejects([&] { (void)extent(bad); });
        bad=padded;bad.rows=32769;rejects([&] { (void)extent(bad); });
        bad=padded;bad.columns=0;rejects([&] { (void)extent(bad); });
        bad=padded;bad.columns=32769;rejects([&] { (void)extent(bad); });
        bad=padded;bad.row_stride_bytes=8;rejects([&] { (void)extent(bad); });
        bad=padded;bad.row_stride_bytes=19;rejects([&] { (void)extent(bad); });
        bad=padded;bad.offset_bytes=1;rejects([&] { (void)extent(bad); });
        bad=padded;bad.row_stride_bytes=size_t(UINT32_MAX)+1;
        rejects([&] { (void)extent(bad); });
        bad=padded;bad.offset_bytes=std::numeric_limits<size_t>::max()-1;
        rejects([&] { (void)extent(bad); });
        auto fp32=matrix(3,5,2,ScalarType::FP32);
        check(extent(fp32)==92,"FP32 uses four-byte elements");
        fp32.offset_bytes=2;rejects([&] { (void)extent(fp32); });
        auto large_pitch=matrix(32768,1,1);
        large_pitch.row_stride_bytes=size_t(UINT32_MAX)-1;
        large_pitch.view_size_bytes=size_t(32767)*large_pitch.row_stride_bytes+2;
        check(extent(large_pitch)==large_pitch.view_size_bytes,
              "bounded multiplication uses size_t before computing extent");

        auto touching=padded;touching.offset_bytes=padded.offset_bytes+extent(padded);
        check(!overlaps(padded,touching),"touching same-backing intervals are disjoint");
        touching.offset_bytes-=2;
        check(overlaps(padded,touching)&&overlaps(touching,padded),"partial alias is symmetric");
        touching.storage=2;check(!overlaps(padded,touching),"different backings are independent");
        auto same=padded;same.view_size_bytes+=16;
        check(exact_matrix(padded,same),"exact matrix ignores extra retained view capacity");
        same.offset_bytes+=2;check(!exact_matrix(padded,same),"offset is part of exact in-place identity");
        auto high=matrix(1,1,1,ScalarType::BF16,0,std::numeric_limits<size_t>::max()-5);
        auto high_touch=high;high_touch.offset_bytes+=2;
        check(!overlaps(high,high_touch),"alias subtraction handles near-size_t-max offsets");

        Bindings bindings;
        auto checked=bindings.validate();
        check(validate_term(bindings.geometry,2,5,bindings.terms[0])==3,
              "shared term helper returns the compact rank");
        for(const auto columns:{0U,3U})
            rejects([&] { (void)validate_term(bindings.geometry,columns,5,bindings.terms[0]); });
        for(const auto columns:{0U,6U})
            rejects([&] { (void)validate_term(bindings.geometry,2,columns,bindings.terms[0]); });
        check(checked.copy_base && checked.terms.size()==1,"out-of-place base copy is explicit");
        const auto& term=checked.terms.front();
        check(term.low.rows==3 && term.low.input_columns==2 && term.low.rank==3 &&
              term.low.input_pitch==12 && term.low.a_pitch==12 && term.low.low_pitch==528 &&
              term.low.a_type==0 && term.low.padding==0,"low parameters retain padded byte pitches");
        check(term.apply.rows==3 && term.apply.rank==3 && term.apply.first_output==2 &&
              term.apply.output_count==3 && term.apply.low_pitch==528 && term.apply.b_pitch==14 &&
              term.apply.output_pitch==18 && term.apply.b_type==0 && term.apply.effective_scale==1,
              "rank-three tail range is resolved without rank/output alignment assumptions");
        check(checked.copy.rows==3 && checked.copy.columns==5 && checked.copy.first_pitch==18 &&
              checked.copy.second_pitch==0 && checked.copy.output_pitch==18,"base-copy parameters");
        const auto bias=matrix(1,5,7);
        checked=bindings.validate(bias);
        check(checked.bias.rows==3 && checked.bias.columns==5 && checked.bias.first_pitch==18 &&
              checked.bias.second_pitch==18 && checked.bias.output_pitch==18,"bias uses full output width once");
        for(const auto rank:{16U,128U}) {
            auto selected=bindings;selected.terms[0].a=matrix(rank,2,5);
            selected.terms[0].b=matrix(3,rank,6);
            check(selected.validate().terms.front().low.rank==rank,"rank16/rank128 accepted");
        }
        for(const auto a_type:{ScalarType::BF16,ScalarType::FP32})
            for(const auto b_type:{ScalarType::BF16,ScalarType::FP32}) {
                auto mixed=bindings;mixed.terms[0].a=matrix(3,2,5,a_type);
                mixed.terms[0].b=matrix(3,3,6,b_type);
                checked=mixed.validate();
                check(checked.terms[0].low.a_type==uint32_t(a_type==ScalarType::FP32) &&
                      checked.terms[0].apply.b_type==uint32_t(b_type==ScalarType::FP32),
                      "BF16 and raw FP32 factor boundaries are explicit");
            }
        for(const auto scale:{0.0F,-2.0F}) {
            auto scaled=bindings;scaled.terms[0].effective_scale=scale;
            check(scaled.validate().terms[0].apply.effective_scale==scale,"zero/negative scales preserved");
        }
        auto in_place=bindings;in_place.output=in_place.base;
        check(!in_place.validate().copy_base,"exact base/output in-place is accepted");
        auto read_alias=bindings;read_alias.terms[0].a.storage=read_alias.input.storage;
        (void)read_alias.validate();
        check(true,"immutable read-only source alias is accepted");
        auto disjoint=bindings;disjoint.output.storage=disjoint.input.storage;
        disjoint.output.offset_bytes=disjoint.input.offset_bytes+extent(disjoint.input);
        check(disjoint.validate().copy_base,"nonoverlapping output and source slices may share backing");

        // Real projection B rows are N, distinct from the dynamic M bound.
        const Geometry maximum{12288,32768,128};
        auto x=matrix(8192,12288,1),base=matrix(8192,32768,2),out=matrix(8192,32768,3);
        auto low=matrix(8192,128,4,ScalarType::FP32);
        const std::array<TermLayout,1> full{{{matrix(128,12288,5),matrix(32768,128,6),1,0,32768}}};
        checked=validate_bindings(maximum,x,full,base,out,low);
        check(checked.terms[0].apply.output_count==32768 && checked.terms[0].low.rows==8192,
              "maximum B output rows and M bound have separate contracts");

        auto invalid=bindings;
        invalid.input=matrix(8193,2,1);invalid.base=matrix(8193,5,2);
        invalid.output=matrix(8193,5,3);invalid.low=matrix(8193,128,4,ScalarType::FP32);
        rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.low=matrix(8193,128,4,ScalarType::FP32);
        rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.input.type=ScalarType::FP32;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.base.type=ScalarType::FP32;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.output.type=ScalarType::FP32;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.geometry.input_columns=1;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.geometry.output_columns=4;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.output.rows=2;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.output.columns=4;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.base.rows=2;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.low.type=ScalarType::BF16;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.low.rows=2;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.low.columns=129;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.low.columns=2;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].a.columns=1;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].a=matrix(129,2,5);rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].b.rows=2;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].b.columns=2;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].output_count=0;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].first_output=5;rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].first_output=UINT32_MAX;
        rejects([&] { (void)invalid.validate(); });
        invalid=bindings;invalid.terms[0].output_count=UINT32_MAX;
        rejects([&] { (void)invalid.validate(); });
        for(const auto scale:{std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
            invalid=bindings;invalid.terms[0].effective_scale=scale;
            rejects([&] { (void)invalid.validate(); });
        }
        for(const auto source:{bindings.input,bindings.base,bindings.terms[0].a,bindings.terms[0].b}) {
            invalid=bindings;invalid.output.storage=source.storage;invalid.output.offset_bytes=source.offset_bytes+2;
            rejects([&] { (void)invalid.validate(); });
        }
        for(const auto source:{bindings.input,bindings.base,bindings.output,
                               bindings.terms[0].a,bindings.terms[0].b,bias}) {
            invalid=bindings;invalid.low.storage=source.storage;invalid.low.offset_bytes=4096;
            rejects([&] { (void)invalid.validate(bias); });
        }
        auto wrong_bias=bias;wrong_bias.rows=2;rejects([&] { (void)bindings.validate(wrong_bias); });
        wrong_bias=bias;wrong_bias.columns=4;rejects([&] { (void)bindings.validate(wrong_bias); });
        wrong_bias=bias;wrong_bias.type=ScalarType::FP32;
        rejects([&] { (void)bindings.validate(wrong_bias); });
        wrong_bias=bias;wrong_bias.storage=bindings.output.storage;
        rejects([&] { (void)bindings.validate(wrong_bias); });

        std::vector<TermLayout> ordered{bindings.terms[0],bindings.terms[0]};
        ordered[1].a.storage=8;ordered[1].first_output=0;ordered[1].effective_scale=-2;
        checked=validate_bindings(bindings.geometry,bindings.input,ordered,bindings.base,
                                  bindings.output,bindings.low);
        check(checked.terms[0].apply.first_output==2 && checked.terms[1].apply.first_output==0 &&
              checked.terms[0].apply.effective_scale==1 && checked.terms[1].apply.effective_scale==-2,
              "caller order is retained for overlapping adapter ranges and distinct A");
        ordered.resize(64,bindings.terms[0]);
        check(validate_bindings(bindings.geometry,bindings.input,ordered,bindings.base,
                                bindings.output,bindings.low).terms.size()==64,"bounded term count accepted");
        ordered.push_back(bindings.terms[0]);
        rejects([&] { (void)validate_bindings(bindings.geometry,bindings.input,ordered,
                                             bindings.base,bindings.output,bindings.low); });
        const std::span<const TermLayout> empty;
        checked=validate_bindings(bindings.geometry,bindings.input,empty,bindings.base,bindings.base,{});
        check(!checked.copy_base && checked.terms.empty(),"empty in-place path needs no low workspace");
        rejects([&] { (void)validate_bindings(bindings.geometry,bindings.input,empty,bindings.base,
                                             bindings.output,{}); });
        rejects([&] { (void)validate_bindings(bindings.geometry,bindings.input,empty,bindings.base,
                                             bindings.base,{},bias); });

        auto gate=matrix(3,5,1),up=matrix(3,5,2),hidden=matrix(3,5,3);
        const auto silu=validate_silu_up(bindings.geometry,gate,up,hidden);
        check(silu.rows==3 && silu.columns==5 && silu.first_pitch==18 &&
              silu.second_pitch==18 && silu.output_pitch==18,"pre-SiLU gate/up uses physical byte pitches");
        auto invalid_hidden=hidden;invalid_hidden.storage=gate.storage;
        rejects([&] { (void)validate_silu_up(bindings.geometry,gate,up,invalid_hidden); });
        invalid_hidden=hidden;invalid_hidden.storage=up.storage;
        rejects([&] { (void)validate_silu_up(bindings.geometry,gate,up,invalid_hidden); });
        invalid_hidden=hidden;invalid_hidden.type=ScalarType::FP32;
        rejects([&] { (void)validate_silu_up(bindings.geometry,gate,up,invalid_hidden); });
        invalid_hidden=hidden;invalid_hidden.columns=4;
        rejects([&] { (void)validate_silu_up(bindings.geometry,gate,up,invalid_hidden); });
        invalid_hidden=hidden;invalid_hidden.rows=2;
        rejects([&] { (void)validate_silu_up(bindings.geometry,gate,up,invalid_hidden); });
        rejects([&] { (void)validate_silu_up(Geometry{2,4,128},gate,up,hidden); });
        gate=matrix(8193,5,1);up=matrix(8193,5,2);hidden=matrix(8193,5,3);
        rejects([&] { (void)validate_silu_up(bindings.geometry,gate,up,hidden); });

        std::cout<<"native_lora_contract_cpu=PASS checks="<<checks
                 <<" device_created=0 gpu_submissions=0 model_loads=0\n";
    } catch(const std::exception& error) {
        std::cerr<<"native_lora_contract_cpu=FAIL "<<error.what()<<'\n';
        return 1;
    }
}
