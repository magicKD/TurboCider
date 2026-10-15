#include "native_ane_stages.hpp"
#include "kernels/native_ane_stages.hpp"
#include "../ane_w8a8_math.hpp"

#import <Metal/Metal.h>

namespace tc::metal_graph {
namespace {
namespace contract = native_ane_stage_contract;
using tc::ane::CapabilityError;
using tc::ane::private_api::AneRequestProvenance;

void require(bool value,const char* error) {
    if (!value) throw CapabilityError(error);
}
void require_identity(const NativeAneJobIdentity& id) {
    require(id.domain && id.epoch && id.request && id.slot_generation && id.producer_generation,
        "native ANE stage identity must be nonzero");
}
bool same_identity(const NativeAneJobIdentity& id,const AneRequestProvenance& p) {
    return id.domain==p.domain && id.epoch==p.epoch && id.request==p.request &&
        id.slot_generation==p.slot_generation && id.producer_generation==p.producer_generation;
}
void require_device(const BufferView& view,void* device) {
    require(bool(view) && device,"native ANE stage has an empty buffer/device");
    const id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)view.buffer().native_handle();
    require(buffer && (__bridge void*)buffer.device==device,
        "native ANE stage buffer belongs to another device");
}
contract::Layout layout(const MatrixView& matrix,void* device) {
    require_device(matrix.storage,device);
    contract::Layout result{matrix.rows,matrix.columns,matrix.row_stride_bytes,
        matrix.storage.offset_bytes(),matrix.storage.size_bytes(),matrix.storage.scalar_type(),
        reinterpret_cast<uintptr_t>(matrix.storage.buffer().native_handle())};
    (void)contract::extent(result);
    return result;
}
contract::Layout layout(const ImportedAneSurface& surface,void* device) {
    surface.lease.validate(device);
    require_device(surface.view,device);
    const auto& lease=surface.lease;
    const auto type=lease.element==tc::ane::gpu::Element::FP16?ScalarType::FP16:ScalarType::Int8;
    require(surface.buffer.native_handle()==lease.native_buffer &&
        surface.view.buffer().native_handle()==lease.native_buffer &&
        surface.buffer.size_bytes()==lease.backing_bytes &&
        surface.view.offset_bytes()==lease.offset_bytes && surface.view.size_bytes()==lease.size_bytes &&
        surface.view.scalar_type()==type,"native ANE stage imported surface metadata changed");
    contract::Layout result{lease.rows,lease.cols,lease.row_stride_bytes,
        lease.offset_bytes,lease.size_bytes,type,reinterpret_cast<uintptr_t>(lease.native_buffer)};
    (void)contract::extent(result);
    return result;
}
contract::Layout compact_layout(const BufferView& view,uint32_t rows,void* device) {
    require_device(view,device);
    contract::Layout result{rows,1,sizeof(uint16_t),view.offset_bytes(),view.size_bytes(),
        view.scalar_type(),reinterpret_cast<uintptr_t>(view.buffer().native_handle())};
    contract::matrix(result,rows,1,ScalarType::FP16);
    require(view.size_bytes()==size_t(rows)*sizeof(uint16_t),
        "native ANE compact scale view must contain exactly rows FP16 values");
    return result;
}
void disjoint_compact_ranges(const std::array<contract::Layout,3>& values) {
    for(size_t i=0;i<values.size();++i)for(size_t j=i+1;j<values.size();++j) {
        const auto& a=values[i];const auto& b=values[j];
        require(a.storage!=b.storage || a.offset_bytes+a.view_size_bytes<=b.offset_bytes ||
            b.offset_bytes+b.view_size_bytes<=a.offset_bytes,
            "native ANE compact scale slices overlap");
    }
}
void disjoint(const contract::Layout& other,const BufferView& fixed) {
    require(other.storage!=reinterpret_cast<uintptr_t>(fixed.buffer().native_handle()),
        "native ANE stage operand aliases status/sign/dummy backing");
}
BufferView fresh_status(Backend& backend,std::string_view label) {
    auto storage=backend.allocate(sizeof(uint32_t),label,StorageMode::Shared);
    auto view=storage.view(0,sizeof(uint32_t),ScalarType::UInt32);
    require(view.contents()!=nullptr,"native ANE stage Shared status is not CPU accessible");
    // Fresh job-owned allocation, before any graph is recorded/submitted.
    // Never reset this word after submission or use an event as a read fence.
    *static_cast<uint32_t*>(view.contents())=0;
    return view;
}
uint32_t read_flags(const BufferView& view,void* device) {
    require_device(view,device);
    require(view.scalar_type()==ScalarType::UInt32 && view.size_bytes()==sizeof(uint32_t) &&
        view.contents()!=nullptr,"native ANE stage invalid Shared validation word");
    return *static_cast<const uint32_t*>(view.contents());
}
void check_pipeline(const Pipeline& pipeline,size_t threads) {
    require(pipeline.thread_execution_width()==32 && pipeline.max_threads_per_threadgroup()>=threads,
        "native ANE stage unsupported SIMD/threadgroup capability");
}
} // namespace

AneStageStatus AneStageStatus::producer(Backend& backend,NativeAneJobIdentity identity) {
    require_identity(identity);
    AneStageStatus result;
    result.role_=contract::StatusRole::Producer;
    result.identity_=identity;result.device_=backend.native_device();
    result.status_=fresh_status(backend,"tc-native-ane-stage-producer-status");
    return result;
}
AneStageStatus AneStageStatus::restore(Backend& backend,const NativeAneReadPermit& permit,
    const AneStageStatus& producer) {
    require(bool(permit),"native ANE restore status requires an actual read permit");
    const auto& p=permit.provenance(); // also rejects a retired/failed state
    require(producer.role_==contract::StatusRole::Producer && producer.validated_ &&
        producer.encoded_==contract::producer_mask && producer.provenance_ && producer.down_scales_ && producer.geometry_ &&
        producer.device_==backend.native_device() && detail::same_ane_request(*producer.provenance_,p),
        "native ANE restore status belongs to a different/unvalidated producer");
    AneStageStatus result;
    result.role_=contract::StatusRole::Restore;
    result.identity_=producer.identity_;result.device_=backend.native_device();
    result.geometry_=producer.geometry_;
    result.down_scales_=producer.down_scales_;result.provenance_=p;
    result.status_=fresh_status(backend,"tc-native-ane-stage-restore-status");
    return result;
}
void AneStageStatus::validate_after(const NativeAneProducerPermit& permit) {
    require(role_==contract::StatusRole::Producer && encoded_==contract::producer_mask &&
        !validated_ && bool(permit),
        "native ANE stage requires a fresh successful captured producer permit");
    const auto& p=permit.provenance();
    require(same_identity(identity_,p) && !p.program_key.empty() && p.program_generation &&
        p.function_index==0 && down_scales_ && geometry_,"native ANE stage producer provenance mismatch");
    contract::validate_status_mask(role_,encoded_,read_flags(status_,device_));
    provenance_=p;
    validated_=true;
}
void AneStageStatus::validate_after(const Ticket& actual_consumer) {
    require(role_==contract::StatusRole::Restore && encoded_==contract::restore_mask &&
        !validated_ && provenance_ && down_scales_,
        "native ANE restore status has no matching producer snapshot");
    const auto completed=actual_consumer.status();
    require(completed.ok && completed.physically_complete && !completed.timed_out,
        "native ANE restore flags cannot be read before successful physical consumer completion");
    contract::validate_status_mask(role_,encoded_,read_flags(status_,device_));
    validated_=true;
}

PreparedAneStages PreparedAneStages::prepare(Backend& backend,AneStageGeometry geometry,AneWeightStaging staging,
    bool scale_reuse) {
    contract::validate_geometry(geometry);
    require(staging==AneWeightStaging::Generic || staging==AneWeightStaging::SylvesterRegister,
        "native ANE stage unknown weight staging recipe");
    PreparedAneStages result;result.geometry_=geometry;result.device_=backend.native_device();
    result.weight_staging_=staging;
    auto library=backend.compile_library(native_ane_stage_kernels::source(),"tc-native-ane-base-stages-v1");
    result.scales_=backend.prepare_pipeline(library,"tc_ane_w8_scales");
    result.codes_=backend.prepare_pipeline(library,"tc_ane_w8_codes");
    result.restore_=backend.prepare_pipeline(library,"tc_ane_restore");
    check_pipeline(result.scales_,512);check_pipeline(result.codes_,512);check_pipeline(result.restore_,256);
    if(scale_reuse) {
        result.scale_copy_=backend.prepare_pipeline(library,"tc_ane_w8_scale_copy");
        check_pipeline(result.scale_copy_,64);
    }
    if (staging==AneWeightStaging::SylvesterRegister) {
        // Dedicated BF16 H128/H512 libraries are prepared only on request.
        // The register kernels use their compile-time block, never p.block.
        const auto prepare_register=[&](uint32_t block,Pipeline& scales,Pipeline& codes) {
            const auto specialized=backend.compile_library(native_ane_stage_kernels::register_source(block),
                "tc-native-ane-register-weight-bf16-h"+std::to_string(block)+"-v1");
            scales=backend.prepare_pipeline(specialized,"tc_ane_sylvester_register_scales");
            codes=backend.prepare_pipeline(specialized,"tc_ane_sylvester_register_codes");
            check_pipeline(scales,32);check_pipeline(codes,32);
        };
        prepare_register(128,result.register_scales128_,result.register_codes128_);
        prepare_register(512,result.register_scales512_,result.register_codes512_);
    }
    auto signs=backend.allocate(512*sizeof(float),"tc-native-ane-stage-signs",StorageMode::Shared);
    result.signs_=signs.view(0,512*sizeof(float),ScalarType::FP32);
    auto* values=static_cast<float*>(result.signs_.contents());
    require(values!=nullptr,"native ANE stage Shared signs are not CPU accessible");
    for(uint32_t i=0;i<512;++i)values[i]=float(tc::ane::rotation_sign(20260930,i));
    result.zero_=fresh_status(backend,"tc-native-ane-stage-zero");
    return result;
}
void PreparedAneStages::encode_quant(CommandGraph& graph,const MatrixView& source,
    const ImportedAneSurface& codes,const ImportedAneSurface& scales,const contract::W8Parameters& p,
    const BufferView& status,bool skip_scales) const {
    require(!skip_scales || !p.transpose,"native ANE A8 scales cannot be reused");
    if (!p.transpose && weight_staging_==AneWeightStaging::SylvesterRegister) {
        require(source.storage.scalar_type()==ScalarType::BF16 && p.encoding==0 && p.dtype==1 &&
            (p.block==128 || p.block==512) && p.basis==0 && !p.activation_group &&
            !p.has_column_scale && !p.inverse_column_scale,
            "native ANE register W staging requires dense BF16 Sylvester H128/H512 without groups/S1");
        const auto& scale_pipeline=p.block==128?register_scales128_:register_scales512_;
        const auto& code_pipeline=p.block==128?register_codes128_:register_codes512_;
        require(bool(scale_pipeline) && bool(code_pipeline),"native ANE register W pipelines were not cold prepared");
        if(!skip_scales)graph.dispatch(scale_pipeline,{source.storage,zero_,zero_,scales.view,status,signs_},p,
            {p.rows,1,1},{32,1,1});
        graph.dispatch(code_pipeline,{source.storage,zero_,zero_,scales.view,codes.view,status,signs_},p,
            {p.rows,p.columns/p.block,1},{32,1,1});
        return;
    }
    // Parameters are copied last. Unused dense metadata/column-scale bindings
    // use a separate immutable zero, never the asynchronously written status.
    if(!skip_scales)graph.dispatch(scales_,{source.storage,zero_,zero_,scales.view,status,signs_,zero_},p,
        {p.rows,1,1},{p.block,1,1});
    graph.dispatch(codes_,{source.storage,zero_,zero_,scales.view,codes.view,status,signs_,zero_},p,
        {p.rows,p.columns/p.block,1},{p.block,1,1});
}
void PreparedAneStages::encode_scale_copy(CommandGraph& graph,const BufferView& source,
    const BufferView& destination,uint32_t rows,uint32_t source_half_pitch,uint32_t destination_half_pitch) const {
    // Legacy MSL uint3 occupies 16 bytes. Strides are ushort elements, not bytes.
    struct alignas(16) CopyParameters { uint32_t rows, source_pitch, destination_pitch, padding; };
    static_assert(sizeof(CopyParameters)==16 && alignof(CopyParameters)==16);
    require(bool(scale_copy_),"native ANE compact scale copy was not cold prepared");
    const CopyParameters parameters{rows,source_half_pitch,destination_half_pitch,0};
    graph.dispatch(scale_copy_,{source,destination},parameters,{(size_t(rows)+63)/64,1,1},{64,1,1});
}
void PreparedAneStages::encode_weight_bank(CommandGraph& graph,const MatrixView& gate,
    const MatrixView& up,const MatrixView& down,const AneWeightBankViews& bank,AneStageStatus& status,
    const AneCompactWeightScales* reuse,const AneCompactWeightScales* fill) const {
    require(!reuse || !fill,"native ANE compact scales cannot be filled and reused together");
    require(status.device_==device_ && status.role_==contract::StatusRole::Producer &&
        !status.validated_ && !(status.encoded_&contract::weight_mask) &&
        (!status.geometry_ || *status.geometry_==geometry_),
        "native ANE weight stage has foreign/reused status");
    require_device(status.status_,device_);
    const std::array<contract::Layout,3> sources{layout(gate,device_),layout(up,device_),layout(down,device_)};
    const std::array<contract::Layout,6> outputs{layout(bank.g,device_),layout(bank.sg,device_),
        layout(bank.u,device_),layout(bank.su,device_),layout(bank.d,device_),layout(bank.sd,device_)};
    const auto p=contract::weight_parameters(geometry_,sources,outputs);
    for(const auto& value:sources) {disjoint(value,status.status_);disjoint(value,signs_);disjoint(value,zero_);}
    for(const auto& value:outputs) {disjoint(value,status.status_);disjoint(value,signs_);disjoint(value,zero_);}
    if(status.encoded_&contract::activation_mask)contract::cross_stage_aliases(sources,outputs,status.activation_);
    const auto* selected=reuse?reuse:fill;
    std::optional<std::array<contract::Layout,3>> compact;
    if(selected) {
        require(bool(scale_copy_),"native ANE compact scale reuse was not cold prepared");
        compact=std::array<contract::Layout,3>{compact_layout(selected->gate,geometry_.ane_channels,device_),
            compact_layout(selected->up,geometry_.ane_channels,device_),
            compact_layout(selected->down,geometry_.hidden,device_)};
        disjoint_compact_ranges(*compact);
        for(const auto& value:*compact) {
            disjoint(value,status.status_);disjoint(value,signs_);disjoint(value,zero_);
            for(const auto& source:sources)contract::disjoint_storage(value,source);
            for(const auto& output:outputs)contract::disjoint_storage(value,output);
            if(status.encoded_&contract::activation_mask)
                for(const auto& activation:status.activation_)contract::disjoint_storage(value,activation);
        }
    }
    // Prepare this retained snapshot before appending; any allocation/metadata
    // exception leaves the caller graph and status writer mask unchanged.
    auto captured_sd=bank.sd;
    CommandGraph draft;
    const auto quant=[&](const MatrixView& source,const ImportedAneSurface& codes,
        const ImportedAneSurface& scales,const contract::W8Parameters& parameters,const BufferView* cache) {
        if(reuse)encode_scale_copy(draft,*cache,scales.view,parameters.rows,1,parameters.scale_pitch/2);
        encode_quant(draft,source,codes,scales,parameters,status.status_,reuse!=nullptr);
        if(fill)encode_scale_copy(draft,scales.view,*cache,parameters.rows,parameters.scale_pitch/2,1);
    };
    quant(gate,bank.g,bank.sg,p[0],selected?&selected->gate:nullptr);
    quant(up,bank.u,bank.su,p[1],selected?&selected->up:nullptr);
    quant(down,bank.d,bank.sd,p[2],selected?&selected->down:nullptr);
    graph.append_graph(std::move(draft));
    status.weight_sources_=sources;status.weight_outputs_=outputs;
    status.compact_scales_=compact;
    status.down_scales_=std::move(captured_sd);status.geometry_=geometry_;status.encoded_|=contract::weight_mask;
}
void PreparedAneStages::encode_activation(CommandGraph& graph,const MatrixView& input,
    const AneActivationViews& activation,AneStageStatus& status) const {
    require(status.device_==device_ && status.role_==contract::StatusRole::Producer &&
        !status.validated_ && !(status.encoded_&contract::activation_mask) &&
        (!status.geometry_ || *status.geometry_==geometry_),
        "native ANE activation stage has foreign/reused status");
    require_device(status.status_,device_);
    const std::array<contract::Layout,3> operands{layout(input,device_),layout(activation.x,device_),
        layout(activation.tx,device_)};
    const auto p=contract::activation_parameters(geometry_,operands[0],operands[1],operands[2]);
    for(const auto& value:operands) {disjoint(value,status.status_);disjoint(value,signs_);disjoint(value,zero_);}
    if(status.encoded_&contract::weight_mask)contract::cross_stage_aliases(status.weight_sources_,status.weight_outputs_,operands);
    if(status.compact_scales_)for(const auto& compact:*status.compact_scales_)
        for(const auto& operand:operands)contract::disjoint_storage(compact,operand);
    CommandGraph draft;
    encode_quant(draft,input,activation.x,activation.tx,p,status.status_);
    graph.append_graph(std::move(draft));
    status.activation_=operands;status.geometry_=geometry_;status.encoded_|=contract::activation_mask;
}
void PreparedAneStages::encode_restore(CommandGraph& graph,const NativeAneReadPermit& permit,
    const AneStageStatus& producer,const MatrixView& partial,AneStageStatus& status) const {
    require(bool(permit),"native ANE restore requires a successful read permit");
    const auto& p=permit.provenance();
    require(producer.role_==contract::StatusRole::Producer && producer.validated_ && producer.provenance_ &&
        producer.down_scales_ && producer.device_==device_ && producer.geometry_ && *producer.geometry_==geometry_ &&
        detail::same_ane_request(*producer.provenance_,p) && status.device_==device_ &&
        status.role_==contract::StatusRole::Restore && !status.validated_ && !status.encoded_ &&
        status.provenance_ && status.down_scales_ && status.geometry_ && *status.geometry_==geometry_ &&
        detail::same_ane_request(*status.provenance_,p),
        "native ANE restore has a foreign/unvalidated producer or reused status");
    const auto& y=permit.output("y");
    const auto output=layout(partial,device_);
    const auto source=layout(y,device_);
    const auto scales=layout(*status.down_scales_,device_);
    require(status.down_scales_->lease.native_buffer==producer.down_scales_->lease.native_buffer,
        "native ANE restore down-scale snapshot changed");
    const auto parameters=contract::restore_parameters(geometry_,source,scales,output);
    require_device(status.status_,device_);
    for(const auto& value:{source,scales,output}) {
        disjoint(value,status.status_);disjoint(value,signs_);disjoint(value,zero_);
    }
    for (const auto& value:producer.weight_sources_)contract::disjoint_storage(value,output);
    for (const auto& value:producer.weight_outputs_)contract::disjoint_storage(value,output);
    for (const auto& value:producer.activation_)contract::disjoint_storage(value,output);
    if(producer.compact_scales_)for(const auto& value:*producer.compact_scales_)contract::disjoint_storage(value,output);
    // The full program output is [H+1,M]: first H rows are normalized
    // channels; the last row is the positive hidden headroom per token.
    const auto norm=y.buffer.view(0,size_t(geometry_.hidden)*source.row_stride_bytes,ScalarType::FP16);
    const auto headroom=y.buffer.view(size_t(geometry_.hidden)*source.row_stride_bytes,
        source.row_stride_bytes,ScalarType::FP16);
    CommandGraph draft;
    draft.dispatch(restore_,{norm,partial.storage,status.status_,zero_,status.down_scales_->view,headroom,zero_},
        parameters,{(size_t(geometry_.rows)+31)/32,(size_t(geometry_.hidden)+31)/32,1},{32,8,1});
    graph.append_graph(std::move(draft));
    status.encoded_=contract::restore_mask;
}
} // namespace tc::metal_graph
