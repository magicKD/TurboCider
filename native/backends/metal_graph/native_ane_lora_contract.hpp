#pragma once

#include "native_ane_stage_contract.hpp"

#include <bit>
#include <optional>

namespace tc::metal_graph::native_ane_lora_contract {

// CPU metadata for the native provider ABI. This file neither creates a device
// nor mints a physical read permit, source-verification proof or memory lease.
// The existing Base controller/stages do not accept the new writer masks yet.
using Geometry = native_ane_stage_contract::Geometry;
using Layout = native_ane_stage_contract::Layout;
enum class Variant : uint8_t { Base, HiddenReference, Rank128Candidate };
enum class SurfaceRole : uint8_t { ProgramInput, ProgramOutput, GpuOnly };
enum class OutputPart : uint8_t { DownNormalized, HiddenScale, CorrectedHidden, DownRank };

struct Config {
    Variant variant = Variant::Base;
    Geometry geometry{};
    uint32_t rank = 0, adapter_count = 0, bias_projections = 0;
    float source_strength = 0.f, headroom = 1.f;
    uint32_t version = 1;
};
struct SurfaceSpec {
    std::string_view name;
    uint32_t rows = 0, columns = 0;
    ScalarType type = ScalarType::Bytes;
    uint64_t pitch_bytes = 0, payload_bytes = 0, capacity_upper_bytes = 0;
    SurfaceRole role = SurfaceRole::ProgramInput;
};
struct OutputSection {
    OutputPart part{};
    uint32_t first_row = 0, rows = 0;
    ScalarType type = ScalarType::FP16;
};
struct WriterMasks { uint32_t producer = 0, restore = 0; };
struct ChannelRange { uint32_t first = 0, count = 0; };
// The original five bits retain their existing meanings. Additional bits are
// proposed writers for this ABI, NOT accepted by today's AneStageStatus API.
inline constexpr uint32_t correction_gate_mask = 1U << 5;
inline constexpr uint32_t correction_up_mask = 1U << 6;
inline constexpr uint32_t down_a_mask = 1U << 7;
inline constexpr uint32_t hidden_restore_mask = 1U << 8;
inline constexpr uint32_t rank_restore_join_mask = 1U << 9;
static_assert(native_ane_stage_contract::producer_mask == 15 &&
              native_ane_stage_contract::restore_mask == 16);

namespace detail {
inline void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
inline uint64_t add(uint64_t a, uint64_t b) {
    require(a <= UINT64_MAX-b, "native ANE LoRA byte addition overflows");
    return a+b;
}
inline uint64_t multiply(uint64_t a, uint64_t b) {
    require(!b || a <= UINT64_MAX/b, "native ANE LoRA byte product overflows");
    return a*b;
}
inline uint64_t align(uint64_t value, uint64_t alignment) {
    require(alignment && !(alignment & (alignment-1)), "native ANE LoRA invalid byte alignment");
    return add(value, alignment-1) & ~(alignment-1);
}
inline SurfaceSpec surface(std::string_view name, uint32_t rows, uint32_t columns,
                           ScalarType type, SurfaceRole role) {
    require(rows && columns && (type==ScalarType::Int8 || type==ScalarType::FP16),
            "native ANE LoRA surface dtype/shape unsupported");
    const auto pitch=align(multiply(columns,scalar_bytes(type)),64);
    const auto payload=multiply(rows,pitch);
    return {name,rows,columns,type,pitch,payload,align(payload,16384),role};
}
inline bool digest(std::string_view value) {
    if(value.size()!=64) return false;
    for(char c:value) if(!((c>='0' && c<='9') || (c>='a' && c<='f'))) return false;
    return true;
}
inline bool same_owner(const std::shared_ptr<const void>& a, const std::shared_ptr<const void>& b) {
    // Aliasing shared_ptrs may have null/different get() while owning the same
    // allocation. Compare live control blocks, never the wrapper/get address.
    return a.use_count()>0 && b.use_count()>0 && !a.owner_before(b) && !b.owner_before(a);
}
} // namespace detail

struct Budget {
    uint64_t base_surface_payload_bytes = 0, base_surface_capacity_upper_bytes = 0;
    uint64_t surface_payload_bytes = 0, surface_capacity_upper_bytes = 0;
    uint64_t additional_surface_payload_bytes = 0, additional_surface_capacity_upper_bytes = 0;
    uint64_t additional_low_payload_bytes = 0, additional_low_capacity_upper_bytes = 0;
};

class Spec {
public:
    static Spec make(Config config = {}) {
        native_ane_stage_contract::validate_geometry(config.geometry);
        detail::require(config.version==1 && config.bias_projections==0 &&
            std::bit_cast<uint32_t>(config.headroom)==std::bit_cast<uint32_t>(1.f),
            "native ANE LoRA requires v1/headroom1/no bias");
        const bool tiny=config.geometry.rows==32;
        switch(config.variant) {
        case Variant::Base:
            detail::require(!config.rank && !config.adapter_count &&
                std::bit_cast<uint32_t>(config.source_strength)==std::bit_cast<uint32_t>(0.f),
                "native Base ABI cannot carry adapter/rank/strength fields");
            break;
        case Variant::HiddenReference:
        case Variant::Rank128Candidate:
            detail::require(config.adapter_count==1 &&
                std::bit_cast<uint32_t>(config.source_strength)==std::bit_cast<uint32_t>(1.f) &&
                (config.rank==128 || (tiny && config.variant==Variant::HiddenReference && config.rank==16)),
                "native ANE LoRA requires one strength-one r128 adapter (tiny Hidden also r16)");
            break;
        default: throw std::invalid_argument("native ANE LoRA unknown variant");
        }
        Spec out(config);
        const auto g=config.geometry;
        const auto add=[&](std::string_view name,uint32_t rows,uint32_t columns,ScalarType type,SurfaceRole role) {
            const auto s=detail::surface(name,rows,columns,type,role);
            out.surfaces_.push_back(s);
            if(role==SurfaceRole::ProgramInput) out.inputs_.push_back(s);
        };
        constexpr auto input=SurfaceRole::ProgramInput;
        // Preserve the legacy MIL argument order. Binding vectors may be
        // unordered, but exact names/layouts and uniqueness remain required.
        add("x",g.hidden,g.rows,ScalarType::Int8,input);
        add("tx",1,g.rows,ScalarType::FP16,input);
        add("wg",g.ane_channels,g.hidden,ScalarType::Int8,input);
        add("sg",g.ane_channels,1,ScalarType::FP16,input);
        add("wu",g.ane_channels,g.hidden,ScalarType::Int8,input);
        add("su",g.ane_channels,1,ScalarType::FP16,input);
        add("wd",g.hidden,g.ane_channels,ScalarType::Int8,input);
        add("sd",g.hidden,1,ScalarType::FP16,SurfaceRole::GpuOnly);
        out.sections_={{OutputPart::DownNormalized,0,g.hidden,ScalarType::FP16},
                       {OutputPart::HiddenScale,g.hidden,1,ScalarType::FP16}};
        uint32_t packed=g.hidden+1; // Geometry was bounded above before adding.
        out.masks_={native_ane_stage_contract::producer_mask,native_ane_stage_contract::restore_mask};
        if(config.variant!=Variant::Base) {
            add("dg",g.ane_channels,g.rows,ScalarType::FP16,input);
            add("du",g.ane_channels,g.rows,ScalarType::FP16,input);
            out.masks_.producer |= correction_gate_mask|correction_up_mask;
            if(config.variant==Variant::HiddenReference) {
                out.sections_.push_back({OutputPart::CorrectedHidden,packed,g.ane_channels,ScalarType::FP16});
                packed+=g.ane_channels;
                out.masks_.restore |= hidden_restore_mask;
            } else {
                add("ad",config.rank,g.ane_channels,ScalarType::FP16,input);
                out.sections_.push_back({OutputPart::DownRank,packed,config.rank,ScalarType::FP16});
                packed+=config.rank;
                out.masks_.producer |= down_a_mask;
                out.masks_.restore |= rank_restore_join_mask;
            }
        }
        add("y",packed,g.rows,ScalarType::FP16,SurfaceRole::ProgramOutput);
        for(const auto& s:out.surfaces_) {
            out.budget_.surface_payload_bytes=detail::add(out.budget_.surface_payload_bytes,s.payload_bytes);
            out.budget_.surface_capacity_upper_bytes=detail::add(out.budget_.surface_capacity_upper_bytes,s.capacity_upper_bytes);
            if(s.name=="dg" || s.name=="du" || s.name=="ad") continue;
            const auto b=s.name=="y" ? detail::surface("y",g.hidden+1,g.rows,ScalarType::FP16,SurfaceRole::ProgramOutput) : s;
            out.budget_.base_surface_payload_bytes=detail::add(out.budget_.base_surface_payload_bytes,b.payload_bytes);
            out.budget_.base_surface_capacity_upper_bytes=detail::add(out.budget_.base_surface_capacity_upper_bytes,b.capacity_upper_bytes);
        }
        out.budget_.additional_surface_payload_bytes=out.budget_.surface_payload_bytes-out.budget_.base_surface_payload_bytes;
        out.budget_.additional_surface_capacity_upper_bytes=out.budget_.surface_capacity_upper_bytes-out.budget_.base_surface_capacity_upper_bytes;
        if(config.variant!=Variant::Base) {
            out.budget_.additional_low_payload_bytes=detail::multiply(detail::multiply(g.rows,config.rank),4);
            out.budget_.additional_low_capacity_upper_bytes=detail::align(out.budget_.additional_low_payload_bytes,16384);
        }
        return out;
    }
    Variant variant() const noexcept { return config_.variant; }
    const Geometry& geometry() const noexcept { return config_.geometry; }
    uint32_t rank() const noexcept { return config_.rank; }
    float headroom() const noexcept { return config_.headroom; }
    ChannelRange gpu_channels() const noexcept { return {0,geometry().first_channel()}; }
    ChannelRange ane_channels() const noexcept { return {geometry().first_channel(),geometry().ane_channels}; }
    uint32_t packed_rows() const noexcept { return surfaces_.back().rows; }
    WriterMasks writer_masks() const noexcept { return masks_; }
    const Budget& budget() const noexcept { return budget_; }
    const std::vector<SurfaceSpec>& inputs() const noexcept { return inputs_; }
    const std::vector<SurfaceSpec>& surfaces() const noexcept { return surfaces_; }
    const std::vector<OutputSection>& sections() const noexcept { return sections_; }
    const SurfaceSpec& surface(std::string_view name) const {
        for(const auto& s:surfaces_) if(s.name==name) return s;
        throw std::invalid_argument("native ANE LoRA unknown surface name");
    }
    const OutputSection& section(OutputPart part) const {
        for(const auto& s:sections_) if(s.part==part) return s;
        throw std::invalid_argument("native ANE LoRA output part is absent from this recipe");
    }
    std::string_view precision_recipe() const noexcept {
        switch(config_.variant) {
        case Variant::Base: return "native-hybrid-base-sylvester-w8a8-headroom1-v1";
        case Variant::HiddenReference: return "native-hybrid-lora-hidden-fp16-correction-bf16-restore-v1";
        case Variant::Rank128Candidate: return "native-hybrid-lora-rank128-fp16-correction-fp16-low-v1";
        }
        return {}; // Impossible for a successfully constructed Spec.
    }
    // Canonical, lossless ABI identity text, NOT a cryptographic content hash.
    // Program cache hashes this with the existing MIL/constants/environment key.
    // Dynamic source/factor identity belongs to the binding, not this template.
    std::string abi_fingerprint() const {
        const auto g=geometry();
        std::string s="tc-native-ane-lora-abi-v1;recipe="+std::string(precision_recipe());
        const auto append=[&](uint64_t v) { s+=';';s+=std::to_string(v); };
        append(uint8_t(variant()));append(g.rows);append(g.hidden);append(g.width);append(g.ane_channels);
        append(g.first_channel());append(rank());append(std::bit_cast<uint32_t>(headroom()));
        append(config_.adapter_count);append(std::bit_cast<uint32_t>(config_.source_strength));
        append(config_.bias_projections);append(masks_.producer);append(masks_.restore);
        for(const auto& v:surfaces_) {
            s+=';';s+=v.name;append(v.rows);append(v.columns);append(uint8_t(v.type));
            append(uint8_t(v.role));append(v.pitch_bytes);append(v.payload_bytes);append(v.capacity_upper_bytes);
        }
        for(const auto& v:sections_) {append(uint8_t(v.part));append(v.first_row);append(v.rows);append(uint8_t(v.type));}
        return s;
    }
private:
    explicit Spec(Config config):config_(config) {}
    Config config_;
    std::vector<SurfaceSpec> inputs_,surfaces_;
    std::vector<OutputSection> sections_;
    WriterMasks masks_;
    Budget budget_;
};

inline void validate_writer_mask(const Spec& spec,native_ane_stage_contract::StatusRole role,
                                 uint32_t encoded,uint32_t observed_flags) {
    detail::require(role==native_ane_stage_contract::StatusRole::Producer ||
                    role==native_ane_stage_contract::StatusRole::Restore,
                    "native ANE LoRA unknown status role");
    const auto masks=spec.writer_masks();
    detail::require(encoded==(role==native_ane_stage_contract::StatusRole::Producer?masks.producer:masks.restore),
                    "native ANE LoRA incomplete or foreign writers");
    detail::require(observed_flags==0,"native ANE LoRA observed GPU flags are nonzero");
    // A zero value supplied here is only metadata. The controller must read its
    // actual generation's flags after the captured physical producer/consumer.
}

struct NamedLayout { std::string_view name; Layout layout; };
inline void validate_surface_layout(const SurfaceSpec& spec,const Layout& layout) {
    native_ane_stage_contract::surface(layout,spec.rows,spec.columns,spec.type);
    detail::require(layout.row_stride_bytes==spec.pitch_bytes && layout.view_size_bytes>=spec.payload_bytes,
                    "native ANE LoRA surface pitch/extent mismatch");
}
inline void validate_program_inputs(const Spec& spec,std::span<const NamedLayout> values) {
    detail::require(values.size()==spec.inputs().size(),"native ANE LoRA program input count mismatch");
    for(const auto& required:spec.inputs()) {
        size_t count=0;
        for(const auto& v:values) if(v.name==required.name) {validate_surface_layout(required,v.layout);++count;}
        detail::require(count==1,"native ANE LoRA program input missing/duplicated");
    }
    for(size_t i=0;i<values.size();++i) for(size_t j=i+1;j<values.size();++j)
        native_ane_stage_contract::disjoint_storage(values[i].layout,values[j].layout);
}
inline void validate_surface_bindings(const Spec& spec,std::span<const NamedLayout> values) {
    detail::require(values.size()==spec.surfaces().size(),"native ANE LoRA physical surface count mismatch");
    for(const auto& required:spec.surfaces()) {
        size_t count=0;
        for(const auto& v:values) if(v.name==required.name) {validate_surface_layout(required,v.layout);++count;}
        detail::require(count==1,"native ANE LoRA physical surface missing/duplicated");
    }
    for(size_t i=0;i<values.size();++i) for(size_t j=i+1;j<values.size();++j)
        native_ane_stage_contract::disjoint_storage(values[i].layout,values[j].layout);
}

struct SourceView {
    std::shared_ptr<const void> allocation; // Actual Data/Buffer owner, not a view wrapper.
    Layout layout;
};
inline void validate_source(const SourceView& source,uint32_t rows,uint32_t columns) {
    detail::require(source.allocation.use_count()>0,"native ANE LoRA source has no allocation owner");
    native_ane_stage_contract::matrix(source.layout,rows,columns,ScalarType::BF16);
}
inline bool same_source(const SourceView& a,const SourceView& b) {
    const auto& x=a.layout;const auto& y=b.layout;
    return detail::same_owner(a.allocation,b.allocation) && x.storage==y.storage &&
        x.rows==y.rows && x.columns==y.columns && x.row_stride_bytes==y.row_stride_bytes &&
        x.offset_bytes==y.offset_bytes && x.view_size_bytes==y.view_size_bytes && x.type==y.type;
}
// Validate a channel column slice of the actual full Adown source. Preserve
// its F pitch and ownership; converting/staging to FP16 is a separate writer.
inline void validate_down_a_window(const Spec& spec,const SourceView& full,
                                   const SourceView& window,ChannelRange range) {
    detail::require(spec.variant()!=Variant::Base,"native Base ABI has no down adapter window");
    const auto g=spec.geometry();
    validate_source(full,spec.rank(),g.width);
    const auto gpu=spec.gpu_channels(),ane=spec.ane_channels();
    detail::require((range.first==gpu.first && range.count==gpu.count) ||
        (range.first==ane.first && range.count==ane.count),"native ANE LoRA foreign channel window");
    validate_source(window,spec.rank(),range.count);
    const auto offset=detail::add(full.layout.offset_bytes,detail::multiply(range.first,2));
    const auto extent=detail::add(detail::multiply(spec.rank()-1,full.layout.row_stride_bytes),detail::multiply(range.count,2));
    detail::require(detail::same_owner(full.allocation,window.allocation) &&
        full.layout.storage==window.layout.storage && window.layout.row_stride_bytes==full.layout.row_stride_bytes &&
        window.layout.offset_bytes==offset && window.layout.view_size_bytes==extent,
        "native ANE LoRA down factor slice changed owner, pitch or column extent");
}
struct ProjectionBinding {
    SourceView a,b;
    uint32_t first_output=0,output_count=0,source_adapter_ordinal=0;
    float effective_scale=0;
};
struct AdapterBinding {
    uint64_t source_generation=0;
    std::string verified_artifact_digest,adapter_sha256,adapter_revision;
    float source_strength=0;
    std::array<ProjectionBinding,3> projections; // gate, up, down; exactly one term each.
};
inline void validate_adapter_binding(const Spec& spec,const AdapterBinding& binding) {
    detail::require(spec.variant()!=Variant::Base && binding.source_generation &&
        detail::digest(binding.verified_artifact_digest) && detail::digest(binding.adapter_sha256) &&
        detail::digest(binding.adapter_revision) &&
        std::bit_cast<uint32_t>(binding.source_strength)==std::bit_cast<uint32_t>(1.f),
        "native ANE LoRA binding lacks its exact source metadata");
    const auto g=spec.geometry();
    for(size_t i=0;i<binding.projections.size();++i) {
        const auto& p=binding.projections[i];
        const uint32_t input=i==2?g.width:g.hidden,output=i==2?g.hidden:g.width;
        validate_source(p.a,spec.rank(),input);validate_source(p.b,output,spec.rank());
        detail::require(!p.first_output && p.output_count==output && !p.source_adapter_ordinal &&
            std::bit_cast<uint32_t>(p.effective_scale)==std::bit_cast<uint32_t>(1.f),
            "native ANE LoRA first package requires one full-range unit-scale term per projection");
    }
    // Syntax/view validation is not held-fd verification. Session must obtain
    // these fields from its actual verified SourceLease/snapshot, retain it,
    // compare the pinned student at route admission and revalidate after drain.
}
inline void validate_adapter_binding_match(const Spec& spec,const AdapterBinding& expected,const AdapterBinding& observed) {
    validate_adapter_binding(spec,expected);validate_adapter_binding(spec,observed);
    detail::require(expected.source_generation==observed.source_generation &&
        expected.verified_artifact_digest==observed.verified_artifact_digest &&
        expected.adapter_sha256==observed.adapter_sha256 && expected.adapter_revision==observed.adapter_revision,
        "native ANE LoRA adapter source metadata changed after binding");
    for(size_t i=0;i<expected.projections.size();++i)
        detail::require(same_source(expected.projections[i].a,observed.projections[i].a) &&
            same_source(expected.projections[i].b,observed.projections[i].b),
            "native ANE LoRA produced factor allocation/window changed after binding");
}

// Compare metadata derived from the actual sealed handoff against the retained
// invocation. It does not assert that the handoff completed or create a permit.
struct InvocationIdentity {
    uint64_t domain=0,epoch=0,request=0,slot_generation=0,producer_generation=0;
    uint64_t program_generation=0,base_generation=0,adapter_generation=0;
    uint32_t function_index=0;
    std::string program_key,abi_fingerprint,base_revision,adapter_revision;
    bool operator==(const InvocationIdentity&) const = default;
};
inline void validate_identity(const Spec& spec,const InvocationIdentity& value) {
    detail::require(value.domain && value.epoch && value.request && value.slot_generation &&
        value.producer_generation && value.program_generation && value.base_generation &&
        !value.function_index && !value.program_key.empty() && !value.base_revision.empty() &&
        value.abi_fingerprint==spec.abi_fingerprint(),"native ANE LoRA invocation metadata incomplete/foreign");
    if(spec.variant()==Variant::Base)
        detail::require(!value.adapter_generation && value.adapter_revision.empty(),
                        "native Base invocation cannot carry an adapter identity");
    else detail::require(value.adapter_generation && detail::digest(value.adapter_revision),
                         "native ANE LoRA invocation needs its held adapter epoch/revision");
}
inline void validate_identity_match(const Spec& spec,const InvocationIdentity& expected,const InvocationIdentity& observed) {
    validate_identity(spec,expected);validate_identity(spec,observed);
    detail::require(expected==observed,"native ANE LoRA stale/foreign invocation metadata");
}
inline void validate_binding_identity(const Spec& spec,const InvocationIdentity& identity,const AdapterBinding& binding) {
    validate_identity(spec,identity);validate_adapter_binding(spec,binding);
    detail::require(identity.adapter_generation==binding.source_generation && identity.adapter_revision==binding.adapter_revision,
                    "native ANE LoRA factor binding differs from this invocation");
}

// A permitted *layout* section, not permission to read its bytes. The caller
// must separately retain a successful NativeAneReadPermit for THIS backing/job.
inline void validate_output_window(const Spec& spec,OutputPart part,const SourceView& whole,const SourceView& window) {
    validate_surface_layout(spec.surface("y"),whole.layout);
    detail::require(detail::same_owner(whole.allocation,window.allocation) &&
        whole.layout.storage==window.layout.storage,"native ANE LoRA output slice changed allocation owner");
    const auto& section=spec.section(part);
    native_ane_stage_contract::matrix(window.layout,section.rows,spec.geometry().rows,section.type);
    const auto offset=detail::multiply(section.first_row,whole.layout.row_stride_bytes);
    const auto extent=detail::multiply(section.rows,whole.layout.row_stride_bytes);
    detail::require(window.layout.row_stride_bytes==whole.layout.row_stride_bytes &&
        window.layout.offset_bytes==offset && window.layout.view_size_bytes==extent,
        "native ANE LoRA output slice overlaps another packed section or padding extent");
}

// Arithmetic boundary policy only. Real flags must be produced by GPU/MIL at
// these boundaries; CPU values or masks here cannot certify submitted work.
enum class FiniteBoundary : uint8_t {
    ProducerLow,CorrectionNarrow,DownFactorNarrow,CorrectedGateUp,CorrectedHidden,
    RotatedHidden,DownNormalized,HiddenScale,RankPartial,RankSum,JoinedLow,PublishedOutput
};
inline void validate_boundary_value(const Spec& spec,FiniteBoundary boundary,float value) {
    bool half=false,positive=false,lora=false,rank=false;
    switch(boundary) {
    case FiniteBoundary::ProducerLow:lora=true;break;
    case FiniteBoundary::CorrectionNarrow:half=true;lora=true;break;
    case FiniteBoundary::DownFactorNarrow:half=true;rank=true;break;
    case FiniteBoundary::CorrectedGateUp:
    case FiniteBoundary::CorrectedHidden:half=true;lora=true;break;
    case FiniteBoundary::RotatedHidden:
    case FiniteBoundary::DownNormalized:half=true;break;
    case FiniteBoundary::HiddenScale:half=true;positive=true;break;
    case FiniteBoundary::RankPartial:
    case FiniteBoundary::RankSum:half=true;rank=true;break;
    case FiniteBoundary::JoinedLow:rank=true;break;
    case FiniteBoundary::PublishedOutput:break;
    default:throw std::invalid_argument("native ANE LoRA unknown finite boundary");
    }
    detail::require((!lora || spec.variant()!=Variant::Base) &&
        (!rank || spec.variant()==Variant::Rank128Candidate),"native ANE LoRA foreign finite boundary");
    detail::require(std::isfinite(value) && (!half || std::abs(value)<=65504.f) && (!positive || value>0.f),
                    "native ANE LoRA invalid/non-finite boundary value");
}

} // namespace tc::metal_graph::native_ane_lora_contract
