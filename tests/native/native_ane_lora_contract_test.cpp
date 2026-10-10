// Host-only tests of the production ABI descriptor. No real allocations,
// model bytes, Metal/private ANE calls, completion permits or device math.
#include "../../native/backends/metal_graph/native_ane_lora_contract.hpp"

#include <algorithm>
#include <iostream>

namespace c=tc::metal_graph::native_ane_lora_contract;
namespace s=tc::metal_graph::native_ane_stage_contract;
using tc::metal_graph::ScalarType;
namespace {
size_t assertions=0,rejections=0;
void check(bool value,const char* message) {
    ++assertions;
    if(!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& fn,const char* message) {
    bool rejected=false;
    try {fn();} catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,message);++rejections;
}
c::Config config(c::Variant variant,bool tiny=false,uint32_t rank=128) {
    c::Config out;
    out.variant=variant;
    if(tiny) out.geometry={32,128,1024,512};
    if(variant!=c::Variant::Base) {out.rank=rank;out.adapter_count=1;out.source_strength=1.f;}
    return out;
}
c::Layout surface_layout(const c::SurfaceSpec& spec,uintptr_t handle) {
    return {spec.rows,spec.columns,size_t(spec.pitch_bytes),0,size_t(spec.payload_bytes),spec.type,handle};
}
std::vector<c::NamedLayout> layouts(const std::vector<c::SurfaceSpec>& specs) {
    std::vector<c::NamedLayout> out;
    for(const auto& spec:specs) out.push_back({spec.name,surface_layout(spec,100+out.size())});
    return out;
}
c::SourceView source(uint32_t rows,uint32_t columns,uintptr_t handle,size_t offset=0,size_t padding=0) {
    const auto pitch=size_t(columns)*2+padding;
    return {std::make_shared<const int>(1),{rows,columns,pitch,offset,size_t(rows-1)*pitch+size_t(columns)*2,ScalarType::BF16,handle}};
}
c::AdapterBinding adapter(const c::Spec& spec) {
    c::AdapterBinding out;
    out.source_generation=3;out.source_strength=1.f;
    out.verified_artifact_digest=std::string(64,'a');
    out.adapter_sha256=std::string(64,'b');out.adapter_revision=std::string(64,'c');
    const auto g=spec.geometry();
    for(size_t i=0;i<out.projections.size();++i) {
        auto& p=out.projections[i];
        const auto k=i==2?g.width:g.hidden,n=i==2?g.hidden:g.width;
        p.a=source(spec.rank(),k,10+i*2,8,64);
        p.b=source(n,spec.rank(),11+i*2,32,64);
        p.output_count=n;p.effective_scale=1.f;
    }
    return out;
}
c::InvocationIdentity invocation(const c::Spec& spec) {
    c::InvocationIdentity id;
    id.domain=1;id.epoch=2;id.request=3;id.slot_generation=4;id.producer_generation=5;
    id.program_generation=6;id.base_generation=7;
    id.program_key="actual-provider-key";id.abi_fingerprint=spec.abi_fingerprint();id.base_revision="held-base-source-revision";
    if(spec.variant()!=c::Variant::Base) {id.adapter_generation=3;id.adapter_revision=std::string(64,'c');}
    return id;
}
void base_and_budgets() {
    const auto base=c::Spec::make(),hidden=c::Spec::make(config(c::Variant::HiddenReference)),
        rank=c::Spec::make(config(c::Variant::Rank128Candidate));
    check(base.inputs().size()==7 && base.surfaces().size()==9 && base.packed_rows()==4097,"Base ABI changed");
    check(base.writer_masks().producer==s::producer_mask && base.writer_masks().restore==s::restore_mask,"Base masks changed");
    check(base.surface("sd").role==c::SurfaceRole::GpuOnly && base.surface("y").role==c::SurfaceRole::ProgramOutput,"non-input roles lost");
    const std::array<std::string_view,7> expected{"x","tx","wg","sg","wu","su","wd"};
    for(size_t i=0;i<expected.size();++i) check(base.inputs()[i].name==expected[i],"Base MIL argument order changed");
    const auto b=base.budget();
    check(b.surface_payload_bytes==89133056 && b.surface_capacity_upper_bytes==89161728,"Base payload/upper differ from existing phase formula");
    check(double(b.surface_payload_bytes)/1048576.==85.00390625 && double(b.surface_capacity_upper_bytes)/1048576.==85.03125,"Base MiB accounting regressed");
    check(!b.additional_surface_payload_bytes && !b.additional_surface_capacity_upper_bytes && !b.additional_low_payload_bytes,"Base acquired LoRA storage");
    check(hidden.inputs().size()==9 && hidden.surfaces().size()==11 && hidden.packed_rows()==10241,"Hidden ABI shape mismatch");
    check(rank.inputs().size()==10 && rank.surfaces().size()==12 && rank.packed_rows()==4225,"Rank ABI shape mismatch");
    check(hidden.surface("dg").payload_bytes==12582912 && hidden.surface("du").payload_bytes==12582912,"correction surfaces underbudgeted");
    check(rank.surface("ad").payload_bytes==1572864 && rank.surface("ad").rows==128,"Adown orientation/bytes incorrect");
    check(rank.gpu_channels().first==0 && rank.gpu_channels().count==6144 &&
          rank.ane_channels().first==6144 && rank.ane_channels().count==6144,"channel split/range mismatch");
    check(hidden.budget().additional_surface_payload_bytes==37748736 && hidden.budget().additional_surface_capacity_upper_bytes==37748736,"Hidden must add 36 MiB");
    check(rank.budget().additional_surface_payload_bytes==27000832 && rank.budget().additional_surface_capacity_upper_bytes==27000832,"Rank must add 25.75 MiB");
    check(hidden.budget().surface_payload_bytes-rank.budget().surface_payload_bytes==10747904,"rank net saving ignores Adown");
    for(const auto* p:{&hidden,&rank}) {
        check(p->budget().base_surface_payload_bytes==b.surface_payload_bytes,"LoRA baseline changed");
        check(p->budget().additional_low_payload_bytes==524288 && p->budget().additional_low_capacity_upper_bytes==524288,"one extra shared low must be 0.5 MiB");
    }
    const auto tiny=c::Spec::make(config(c::Variant::Base,true));
    const auto tiny16=c::Spec::make(config(c::Variant::HiddenReference,true,16));
    const auto tiny128=c::Spec::make(config(c::Variant::Rank128Candidate,true));
    check(tiny.budget().surface_payload_bytes==286848 && tiny.budget().surface_capacity_upper_bytes==327680,"tiny Base page accounting mismatch");
    check(tiny16.budget().additional_surface_payload_bytes==98304 && tiny16.budget().additional_low_payload_bytes==2048 && tiny16.budget().additional_low_capacity_upper_bytes==16384,"tiny r16 accounting mismatch");
    check(tiny128.budget().additional_surface_payload_bytes==204800 && tiny128.budget().additional_low_payload_bytes==16384,"tiny Rank128 accounting mismatch");
    check(hidden.writer_masks().producer==111 && hidden.writer_masks().restore==272,"Hidden writer set mismatch");
    check(rank.writer_masks().producer==239 && rank.writer_masks().restore==528,"Rank writer set mismatch");
}
void invalid_config_and_fingerprint() {
    const auto original=config(c::Variant::HiddenReference);
    const auto invalid=[&](const auto& change) {auto x=original;change(x);rejects([&]{(void)c::Spec::make(x);},"invalid config accepted");};
    invalid([](auto& x){x.variant=static_cast<c::Variant>(255);});
    invalid([](auto& x){x.version=0;});invalid([](auto& x){x.bias_projections=1;});
    invalid([](auto& x){x.adapter_count=0;});invalid([](auto& x){x.adapter_count=2;});
    invalid([](auto& x){x.rank=16;});invalid([](auto& x){x.rank=129;});
    invalid([](auto& x){x.source_strength=-1;});invalid([](auto& x){x.source_strength=0;});
    invalid([](auto& x){x.source_strength=std::numeric_limits<float>::quiet_NaN();});
    invalid([](auto& x){x.headroom=2;});invalid([](auto& x){x.headroom=std::numeric_limits<float>::infinity();});
    invalid([](auto& x){x.geometry.rows=0;});invalid([](auto& x){x.geometry.rows=1088;});
    invalid([](auto& x){x.geometry.hidden=256;});invalid([](auto& x){x.geometry.width=UINT32_MAX;});
    invalid([](auto& x){x.geometry.ane_channels=0;});invalid([](auto& x){x.geometry.ane_channels=12289;});
    invalid([](auto& x){x.geometry.ane_channels=512;});
    auto bad_base=config(c::Variant::Base);bad_base.rank=128;
    rejects([&]{(void)c::Spec::make(bad_base);},"Base accepted rank");
    bad_base=config(c::Variant::Base);bad_base.source_strength=-0.f;
    rejects([&]{(void)c::Spec::make(bad_base);},"Base coerced noncanonical strength bits");
    rejects([&]{(void)c::Spec::make(config(c::Variant::Rank128Candidate,true,16));},"Rank recipe accepted r16");
    const auto h=c::Spec::make(original),r=c::Spec::make(config(c::Variant::Rank128Candidate));
    check(h.abi_fingerprint()==c::Spec::make(original).abi_fingerprint(),"same ABI fingerprint unstable");
    check(h.abi_fingerprint()!=r.abi_fingerprint() && h.precision_recipe()!=r.precision_recipe(),"hidden and rank recipe collapsed");
    check(c::Spec::make(config(c::Variant::HiddenReference,true,16)).abi_fingerprint()!=
          c::Spec::make(config(c::Variant::HiddenReference,true,128)).abi_fingerprint(),"rank change retained ABI fingerprint");
    check(h.abi_fingerprint()!=c::Spec::make(config(c::Variant::HiddenReference,true)).abi_fingerprint(),"geometry change retained ABI fingerprint");
    rejects([]{(void)c::detail::add(UINT64_MAX,1);},"overflow addition accepted");
    rejects([]{(void)c::detail::multiply(UINT64_MAX,2);},"overflow multiplication accepted");
    rejects([]{(void)c::detail::align(UINT64_MAX,64);},"overflow alignment accepted");
    rejects([]{(void)c::detail::align(64,3);},"nonpower alignment accepted");
}
void layouts_and_masks() {
    for(const auto variant:{c::Variant::Base,c::Variant::HiddenReference,c::Variant::Rank128Candidate}) {
        const auto spec=c::Spec::make(config(variant));
        auto inputs=layouts(spec.inputs()),all=layouts(spec.surfaces());
        c::validate_program_inputs(spec,inputs);c::validate_surface_bindings(spec,all);
        std::reverse(inputs.begin(),inputs.end());c::validate_program_inputs(spec,inputs);
        auto wrong=inputs;wrong.back().name=wrong.front().name;
        rejects([&]{c::validate_program_inputs(spec,wrong);},"duplicate input accepted");
        wrong=inputs;wrong[0].name="foreign";
        rejects([&]{c::validate_program_inputs(spec,wrong);},"foreign input accepted");
        wrong=inputs;wrong.pop_back();
        rejects([&]{c::validate_program_inputs(spec,wrong);},"missing input accepted");
        wrong=inputs;wrong[0].layout.storage=wrong[1].layout.storage;
        rejects([&]{c::validate_program_inputs(spec,wrong);},"aliased inputs accepted");
        wrong=inputs;wrong[0].layout.type=ScalarType::BF16;
        rejects([&]{c::validate_program_inputs(spec,wrong);},"BF16 silently coerced to FP16 surface");
        wrong=inputs;++wrong[0].layout.offset_bytes;
        rejects([&]{c::validate_program_inputs(spec,wrong);},"subview accepted as full input surface");
        wrong=inputs;wrong[0].layout.row_stride_bytes+=64;
        rejects([&]{c::validate_program_inputs(spec,wrong);},"wrong surface pitch accepted");
        wrong=inputs;--wrong[0].layout.view_size_bytes;
        rejects([&]{c::validate_program_inputs(spec,wrong);},"truncated physical surface accepted");
        auto alias=all;alias.back().layout.storage=alias.front().layout.storage;
        rejects([&]{c::validate_surface_bindings(spec,alias);},"output aliases input surface");
        alias=all;alias[0].name="y";
        rejects([&]{c::validate_surface_bindings(spec,alias);},"duplicate physical output accepted");
        const auto masks=spec.writer_masks();
        c::validate_writer_mask(spec,s::StatusRole::Producer,masks.producer,0);
        c::validate_writer_mask(spec,s::StatusRole::Restore,masks.restore,0);
        rejects([&]{c::validate_writer_mask(spec,s::StatusRole::Producer,masks.producer^1U,0);},"missing producer writer accepted");
        rejects([&]{c::validate_writer_mask(spec,s::StatusRole::Restore,masks.restore|0x80000000U,0);},"foreign restore writer accepted");
        rejects([&]{c::validate_writer_mask(spec,s::StatusRole::Producer,masks.producer,1);},"finite failure ignored");
        rejects([&]{c::validate_writer_mask(spec,static_cast<s::StatusRole>(255),masks.producer,0);},"invalid status enum accepted");
    }
    const auto spec=c::Spec::make(config(c::Variant::HiddenReference));
    rejects([&]{c::validate_writer_mask(spec,s::StatusRole::Producer,s::producer_mask,0);},"Base mask blessed LoRA producer");
    rejects([&]{s::validate_status_mask(s::StatusRole::Producer,spec.writer_masks().producer,0);},"new masks accidentally accepted by old stage ABI");
}
void source_identity() {
    const auto spec=c::Spec::make(config(c::Variant::HiddenReference));
    const auto bound=adapter(spec);
    c::validate_adapter_binding(spec,bound);c::validate_adapter_binding_match(spec,bound,bound);
    const auto invalid=[&](const auto& change) {auto x=bound;change(x);rejects([&]{c::validate_adapter_binding(spec,x);},"invalid factor binding accepted");};
    invalid([](auto& x){x.source_generation=0;});invalid([](auto& x){x.adapter_revision.clear();});
    invalid([](auto& x){x.adapter_sha256=std::string(64,'G');});invalid([](auto& x){x.verified_artifact_digest+='a';});
    invalid([](auto& x){x.source_strength=0.5f;});
    for(size_t i=0;i<3;++i) {
        invalid([i](auto& x){x.projections[i].a.allocation.reset();});
        invalid([i](auto& x){x.projections[i].a.layout.type=ScalarType::FP32;});
        invalid([i](auto& x){x.projections[i].b.layout.type=ScalarType::FP16;});
        invalid([i](auto& x){++x.projections[i].a.layout.columns;});
        invalid([i](auto& x){--x.projections[i].b.layout.view_size_bytes;});
        invalid([i](auto& x){x.projections[i].a.layout.offset_bytes=SIZE_MAX-1;});
        invalid([i](auto& x){x.projections[i].effective_scale=2.f;});
        invalid([i](auto& x){x.projections[i].source_adapter_ordinal=1;});
        invalid([i](auto& x){x.projections[i].first_output=1;});
        invalid([i](auto& x){--x.projections[i].output_count;});
    }
    const auto changed=[&](const auto& change) {auto x=bound;change(x);c::validate_adapter_binding(spec,x);
        rejects([&]{c::validate_adapter_binding_match(spec,bound,x);},"source dependency changed without rebinding");};
    changed([](auto& x){++x.source_generation;});changed([](auto& x){x.adapter_revision[0]='d';});
    changed([](auto& x){x.adapter_sha256[0]='e';});changed([](auto& x){x.verified_artifact_digest[0]='f';});
    changed([](auto& x){x.projections[0].a.layout.offset_bytes+=2;});
    changed([](auto& x){x.projections[1].b.layout.view_size_bytes+=2;});
    changed([](auto& x){auto& l=x.projections[2].a.layout;l.row_stride_bytes+=64;l.view_size_bytes+=size_t(l.rows-1)*64;});
    changed([](auto& x){++x.projections[2].b.layout.storage;});
    auto x=bound;
    const auto& real=bound.projections[0].a.allocation;
    x.projections[0].a.allocation=std::shared_ptr<const void>(real.get(),[](const void*){});
    check(real.get()==x.projections[0].a.allocation.get(),"same-address fixture failed");
    rejects([&]{c::validate_adapter_binding_match(spec,bound,x);},"foreign same-address control block accepted");
    // A null get() alias still owns the actual allocation; no dereference occurs.
    x=bound;x.projections[0].a.allocation=std::shared_ptr<const void>(real,nullptr);
    c::validate_adapter_binding_match(spec,bound,x);
    const auto& full=bound.projections[2].a;
    for(auto range:{spec.gpu_channels(),spec.ane_channels()}) {
        auto part=full;part.layout.columns=range.count;
        part.layout.offset_bytes+=size_t(range.first)*2;
        part.layout.view_size_bytes=size_t(spec.rank()-1)*part.layout.row_stride_bytes+size_t(range.count)*2;
        c::validate_down_a_window(spec,full,part,range);
        auto bad=part;bad.layout.offset_bytes+=2;
        rejects([&]{c::validate_down_a_window(spec,full,bad,range);},"shifted Adown channel slice accepted");
        bad=part;bad.layout.row_stride_bytes=size_t(range.count)*2;
        rejects([&]{c::validate_down_a_window(spec,full,bad,range);},"Adown silently compacted original F pitch");
        bad=part;bad.layout.view_size_bytes+=2;
        rejects([&]{c::validate_down_a_window(spec,full,bad,range);},"Adown slice spans extra channel");
        rejects([&]{c::validate_down_a_window(spec,full,part,{1,range.count});},"foreign channel range accepted");
    }
    rejects([&]{c::validate_adapter_binding(c::Spec::make(),bound);},"Base accepted adapter factors");
}
void generation_and_windows() {
    const auto spec=c::Spec::make(config(c::Variant::Rank128Candidate));
    const auto id=invocation(spec);const auto a=adapter(spec);
    c::validate_identity_match(spec,id,id);c::validate_binding_identity(spec,id,a);
    const auto changed=[&](const auto& change) {auto x=id;change(x);rejects([&]{c::validate_identity_match(spec,id,x);},"foreign/stale identity accepted");};
    changed([](auto& x){++x.domain;});changed([](auto& x){++x.epoch;});changed([](auto& x){++x.request;});
    changed([](auto& x){++x.slot_generation;});changed([](auto& x){++x.producer_generation;});
    changed([](auto& x){++x.program_generation;});changed([](auto& x){++x.base_generation;});
    changed([](auto& x){++x.adapter_generation;});changed([](auto& x){x.function_index=1;});
    changed([](auto& x){x.program_key+='x';});changed([](auto& x){x.base_revision+='x';});
    changed([](auto& x){x.adapter_revision[0]='d';});changed([](auto& x){x.abi_fingerprint+='x';});
    changed([](auto& x){x.domain=0;});changed([](auto& x){x.adapter_generation=0;});
    auto stale=a;++stale.source_generation;
    rejects([&]{c::validate_binding_identity(spec,id,stale);},"old invocation accepted new factors");
    const auto base=c::Spec::make();const auto base_id=invocation(base);c::validate_identity(base,base_id);
    rejects([&]{c::validate_identity(base,id);},"Base accepted rank invocation");
    for(const auto variant:{c::Variant::Base,c::Variant::HiddenReference,c::Variant::Rank128Candidate}) {
        const auto v=c::Spec::make(config(variant));
        c::SourceView whole{std::make_shared<const int>(1),surface_layout(v.surface("y"),500)};
        for(const auto& section:v.sections()) {
            auto window=whole;
            window.layout.rows=section.rows;window.layout.offset_bytes=size_t(section.first_row)*whole.layout.row_stride_bytes;
            window.layout.view_size_bytes=size_t(section.rows)*whole.layout.row_stride_bytes;
            c::validate_output_window(v,section.part,whole,window);
            auto bad=window;bad.layout.offset_bytes+=whole.layout.row_stride_bytes;
            rejects([&]{c::validate_output_window(v,section.part,whole,bad);},"shifted output section accepted");
            bad=window;bad.layout.view_size_bytes+=whole.layout.row_stride_bytes;
            rejects([&]{c::validate_output_window(v,section.part,whole,bad);},"output read crosses section boundary");
            bad=window;bad.layout.type=ScalarType::BF16;
            rejects([&]{c::validate_output_window(v,section.part,whole,bad);},"output dtype coerced");
            bad=window;bad.allocation=std::shared_ptr<const void>(whole.allocation.get(),[](const void*){});
            rejects([&]{c::validate_output_window(v,section.part,whole,bad);},"output backing owner changed");
        }
        const auto absent=variant==c::Variant::Rank128Candidate?c::OutputPart::CorrectedHidden:c::OutputPart::DownRank;
        rejects([&]{(void)v.section(absent);},"foreign output part exists");
        rejects([&]{(void)v.section(static_cast<c::OutputPart>(255));},"unknown output part accepted");
    }
}
void finite_boundaries() {
    const auto hidden=c::Spec::make(config(c::Variant::HiddenReference));
    const auto rank=c::Spec::make(config(c::Variant::Rank128Candidate));
    using B=c::FiniteBoundary;
    for(auto boundary:{B::ProducerLow,B::CorrectionNarrow,B::DownFactorNarrow,B::CorrectedGateUp,
        B::CorrectedHidden,B::RotatedHidden,B::DownNormalized,B::HiddenScale,B::RankPartial,B::RankSum,B::JoinedLow,B::PublishedOutput}) {
        c::validate_boundary_value(rank,boundary,1.f);
        rejects([&]{c::validate_boundary_value(rank,boundary,std::numeric_limits<float>::infinity());},"infinite boundary accepted");
        rejects([&]{c::validate_boundary_value(rank,boundary,std::numeric_limits<float>::quiet_NaN());},"NaN boundary accepted");
    }
    c::validate_boundary_value(rank,B::CorrectionNarrow,65504.f);
    rejects([&]{c::validate_boundary_value(rank,B::CorrectionNarrow,65520.f);},"FP16 overflow silently narrowed");
    rejects([&]{c::validate_boundary_value(rank,B::DownFactorNarrow,-65536.f);},"negative FP16 overflow silently narrowed");
    rejects([&]{c::validate_boundary_value(rank,B::HiddenScale,-1.f);},"MIL failure sentinel accepted");
    rejects([&]{c::validate_boundary_value(rank,B::HiddenScale,0.f);},"zero hidden scale accepted");
    rejects([&]{c::validate_boundary_value(hidden,B::RankPartial,1.f);},"Hidden accepted rank-only boundary");
    rejects([&]{c::validate_boundary_value(c::Spec::make(),B::CorrectionNarrow,1.f);},"Base accepted correction boundary");
    rejects([&]{c::validate_boundary_value(rank,static_cast<B>(255),1.f);},"unknown finite boundary accepted");
}
} // namespace
int main() {
    try {
        base_and_budgets();invalid_config_and_fingerprint();layouts_and_masks();
        source_identity();generation_and_windows();finite_boundaries();
        std::cout<<"native_ane_lora_contract=PASS assertions="<<assertions<<" rejections="<<rejections
                 <<" devices=0 models=0 physical_permits=0\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
