#include "../../native/backends/metal_graph/native_lora.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

using namespace tc::metal_graph;
using namespace tc::metal_graph::native_lora;
namespace {
size_t checks=0;
void check(bool condition,const char* message) {
    ++checks;
    if(!condition)throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected=false;
    try {action();} catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"invalid split LoRA descriptor metadata accepted");
}
Layout matrix(uint32_t rows,uint32_t columns,uintptr_t storage,
              ScalarType type=ScalarType::BF16,size_t padding=6,size_t offset=32) {
    const size_t pitch=(size_t(columns)+padding)*scalar_bytes(type);
    return {rows,columns,pitch,size_t(rows-1)*pitch+size_t(columns)*scalar_bytes(type),offset,type,storage};
}
struct Fixture {
    Geometry geometry;
    Layout input,output,low;
    TermLayout term;
    Fixture(uint32_t rows=1024,uint32_t columns=4096,uint32_t outputs=12288,
            uint32_t count=6144,uint32_t first=0,uint32_t rank=128,
            ScalarType a_type=ScalarType::BF16,ScalarType b_type=ScalarType::BF16)
        :geometry{},input(matrix(rows,columns,1)),output(matrix(rows,outputs,2)),
         low(matrix(rows,128,3,ScalarType::FP32)),
         term{matrix(rank,columns,4,a_type),matrix(count,rank,5,b_type),0.75F,first,count} {}
    CheckedProjection validate(KernelVariant variant=KernelVariant::SgMmaR128) const {
        return native_lora::validate_projection(geometry,variant,input,term,output,low);
    }
};

void unchanged_abi_and_sources() {
    static_assert(!std::is_default_constructible_v<ValidatedLoRAProjection>);
    static_assert(!std::is_aggregate_v<ValidatedLoRAProjection>);
    static_assert(std::is_copy_constructible_v<ValidatedLoRAProjection> &&
                  std::is_move_constructible_v<ValidatedLoRAProjection>);
    static_assert(std::is_same_v<decltype(std::declval<const ValidatedLoRAProjection&>().parameters()),
                                 const CheckedTerm&>);
    static_assert(std::is_same_v<decltype(std::declval<const ValidatedLoRAProjection&>().term()),
                                 const OrderedLoRATerm&>);
    struct PreviousInstanceLayout {
        Geometry geometry;
        KernelVariant variant;
        void* device;
        Pipeline low,apply,copy,bias,silu,low_sg,apply_sg;
    };
    check(sizeof(NativeLoRA)==sizeof(PreviousInstanceLayout),"split API must not grow native PSO bundle state");
    check(sizeof(LowParameters)==32 && sizeof(ApplyParameters)==36,
          "low/apply POD ABI remains unchanged");
    auto fingerprint=[](std::string_view text) {
        uint64_t value=14695981039346656037ULL;
        for(const auto character:text) {
            value^=static_cast<unsigned char>(character);value*=1099511628211ULL;
        }
        return value;
    };
    check(source_body.size()==5931 && fingerprint(source_body)==0xd6c67c3ecb9e7553ULL,
          "legacy scalar source body must remain byte-identical");
    check(sg_mma_source_body.size()==4953 && fingerprint(sg_mma_source_body)==0xfbfbfb125c907be1ULL,
          "legacy SGv2 source body must remain byte-identical");
}

void compare_original(const Fixture& fixture,KernelVariant variant) {
    const auto split=fixture.validate(variant);
    const auto previous=validate_bindings(fixture.geometry,fixture.input,
        std::span<const TermLayout>(&fixture.term,1),fixture.output,fixture.output,fixture.low);
    const bool original_sg=variant==KernelVariant::SgMmaR128 && sg_mma_eligible(previous.terms.front());
    check(split.selected_variant==(original_sg?KernelVariant::SgMmaR128:KernelVariant::Scalar),
          "both split phases must share original complete-term selection");
    check(std::memcmp(&split.term.low,&previous.terms.front().low,sizeof(LowParameters))==0 &&
          std::memcmp(&split.term.apply,&previous.terms.front().apply,sizeof(ApplyParameters))==0 &&
          !previous.copy_base,"split phases must retain original in-place PODs without copy/bias");
}

void selection_contract() {
    for(const auto a_type:{ScalarType::BF16,ScalarType::FP32}) {
        for(const auto b_type:{ScalarType::BF16,ScalarType::FP32}) {
            for(const auto first:{0U,6144U}) {
                const Fixture fixture(1024,4096,12288,6144,first,128,a_type,b_type);
                const auto checked=fixture.validate();
                check(checked.selected_variant==KernelVariant::SgMmaR128,
                      "aligned r128 channel halves select the same SG low/apply pair");
                check(checked.term.low.a_type==uint32_t(a_type==ScalarType::FP32) &&
                      checked.term.apply.b_type==uint32_t(b_type==ScalarType::FP32),
                      "raw FP32/BF16 factor types are preserved");
                check(checked.term.low.input_pitch==8204 && checked.term.apply.output_pitch==24588 &&
                      checked.term.low.low_pitch==536 && checked.term.apply.first_output==first,
                      "source/output byte pitches, offset and FP32 low pitch are preserved");
                compare_original(fixture,KernelVariant::Scalar);
                compare_original(fixture,KernelVariant::SgMmaR128);
            }
        }
    }
    for(const auto rows:{1U,7U,9U,33U}) {
        const Fixture fixture(rows);
        check(fixture.validate().selected_variant==KernelVariant::Scalar,"M tail remains scalar for both phases");
        compare_original(fixture,KernelVariant::SgMmaR128);
    }
    for(const auto columns:{31U,33U,4095U}) {
        const Fixture fixture(8,columns);
        check(fixture.validate().selected_variant==KernelVariant::Scalar,"K tail remains scalar for both phases");
        compare_original(fixture,KernelVariant::SgMmaR128);
    }
    for(const auto count:{1U,31U,33U,6143U}) {
        const Fixture fixture(8,4096,12288,count);
        const auto checked=fixture.validate();
        check(checked.selected_variant==KernelVariant::Scalar,"B/output tail must keep the aligned low scalar");
        auto misleading=checked.term;misleading.apply.output_count=32;
        check(sg_mma_eligible(misleading),"low alone is eligible: full B range must determine routing");
        compare_original(fixture,KernelVariant::SgMmaR128);
    }
    for(const auto rank:{3U,16U,127U}) {
        const Fixture fixture(8,4096,12288,6144,0,rank);
        check(fixture.validate().selected_variant==KernelVariant::Scalar,"rank tail remains scalar for both phases");
        compare_original(fixture,KernelVariant::SgMmaR128);
    }
    Fixture nonaligned_first(8,4096,128,32,5);
    check(nonaligned_first.validate().selected_variant==KernelVariant::SgMmaR128,
          "existing SG policy allows a resolved unaligned first-output offset");
    compare_original(nonaligned_first,KernelVariant::SgMmaR128);
    for(const auto scale:{-2.F,0.F,-0.F,1.F}) {
        Fixture fixture;fixture.term.effective_scale=scale;
        check(fixture.validate().term.apply.effective_scale==scale,"finite scales are not reordered or filtered");
        compare_original(fixture,KernelVariant::SgMmaR128);
    }
}

void invalid_and_alias_contract() {
    const Fixture valid;
    auto bad=valid;bad.geometry.max_rank=16;rejects([&]{(void)bad.validate();});
    rejects([&]{(void)valid.validate(static_cast<KernelVariant>(255));});
    for(const auto scale:{std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        bad=valid;bad.term.effective_scale=scale;rejects([&]{(void)bad.validate();});
    }
    bad=valid;bad.term.output_count=0;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.first_output=valid.output.columns;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.first_output=UINT32_MAX;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.output_count=valid.output.columns+1;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.a.columns--;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.b.columns--;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.b.rows--;rejects([&]{(void)bad.validate();});
    bad=valid;bad.low.rows--;rejects([&]{(void)bad.validate();});
    bad=valid;bad.low.columns=127;rejects([&]{(void)bad.validate();});
    bad=valid;bad.low.columns=129;rejects([&]{(void)bad.validate();});
    bad=valid;bad.output.rows--;rejects([&]{(void)bad.validate();});
    bad=valid;bad.output.columns=6143;rejects([&]{(void)bad.validate();});
    bad=valid;bad.input.type=ScalarType::FP32;rejects([&]{(void)bad.validate();});
    bad=valid;bad.output.type=ScalarType::FP32;rejects([&]{(void)bad.validate();});
    bad=valid;bad.low.type=ScalarType::BF16;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.a.type=ScalarType::FP16;rejects([&]{(void)bad.validate();});
    bad=valid;bad.term.b.type=ScalarType::FP16;rejects([&]{(void)bad.validate();});
    for(unsigned operand=0;operand<5;++operand) {
        auto get=[](Fixture& f,unsigned index)->Layout& {
            switch(index) {case 0:return f.input;case 1:return f.term.a;
            case 2:return f.term.b;case 3:return f.output;default:return f.low;}
        };
        bad=valid;get(bad,operand).view_size_bytes--;
        rejects([&]{(void)bad.validate();});
        bad=valid;get(bad,operand).offset_bytes++;
        rejects([&]{(void)bad.validate();});
        bad=valid;get(bad,operand).row_stride_bytes++;
        rejects([&]{(void)bad.validate();});
        bad=valid;get(bad,operand).storage=0;
        rejects([&]{(void)bad.validate();});
    }
    for(const auto source:{valid.input,valid.term.a,valid.term.b}) {
        bad=valid;bad.output.storage=source.storage;bad.output.offset_bytes=source.offset_bytes;
        rejects([&]{(void)bad.validate();});
    }
    for(const auto source:{valid.input,valid.term.a,valid.term.b,valid.output}) {
        bad=valid;bad.low.storage=source.storage;
        bad.low.offset_bytes=source.offset_bytes+extent(source)+32; // Disjoint bytes still share forbidden scratch backing.
        rejects([&]{(void)bad.validate();});
    }
    bad=valid;bad.output.storage=valid.term.b.storage;
    bad.output.offset_bytes=valid.term.b.offset_bytes+extent(valid.term.b);
    check(bad.validate().selected_variant==KernelVariant::SgMmaR128,
          "disjoint immutable/output views may share a retained allocation");
    bad=valid;bad.low.rows+=8;bad.low.view_size_bytes+=8*bad.low.row_stride_bytes;
    check(bad.validate().term.low.rows==valid.input.rows,"larger low capacity never changes actual written rows");
}
}

int main() {
    try {
        unchanged_abi_and_sources();selection_contract();invalid_and_alias_contract();
        std::cout<<"native_lora_split_contract=PASS checks="<<checks
                 <<" device_created=0 model_loads=0\n";
    } catch(const std::exception& error) {
        std::cerr<<"native_lora_split_contract=FAIL "<<error.what()<<'\n';
        return 1;
    }
}
