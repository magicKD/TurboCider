// One existing LoRA trajectory input, one layer, six native FFNs at most.
// Only the base gate/up schedule differs. Both arms share one SGMMA LoRA
// instance, one down projection, the same ordered corrections and six buffers.
// Default CPU contract never creates a device or opens a model artifact.
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/backends/mlx.hpp"
#include "../../native/backends/mlx_fd_reader.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"
#include "../../native/backends/metal_graph/native_linear.hpp"
#include "../../native/backends/metal_graph/native_gate_up.hpp"
#include "../../native/backends/metal_graph/native_lora.hpp"
#include "../../native/backends/metal_graph/memory_bridge.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../native/models/qwen21/viggle_adapter.hpp"
#include "../../native/runtime/memory_execution.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/runtime/streaming/canonical_encoding.hpp"
#include "turbocider/turbocider.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

#ifndef TC_PAIRED_REPLAY_SOURCE_SHA
#define TC_PAIRED_REPLAY_SOURCE_SHA "unrecorded-host-only"
#endif
#ifndef TC_PAIRED_REPLAY_EXPECTED_BUILD_ID
#define TC_PAIRED_REPLAY_EXPECTED_BUILD_ID "unbound-host-only"
#endif
using namespace tc;
namespace mg=tc::metal_graph;
namespace {
constexpr uint32_t rows=1024,hidden=4096,width=12288,rank=128;
constexpr double rrms_limit=.005;
constexpr uint64_t source_bytes=(3ULL*hidden*width+3ULL*rank*(hidden+width)+rows*hidden)*2;
constexpr const char* capture_id="75991-1226054446574458";
constexpr const char* prefix="transformer_blocks.0.img_mlp.";
constexpr const char* prompt="A cobalt blue ceramic teapot on a pale wooden table, soft window light, realistic product photography.";
constexpr const char* split_schedule="split-base-gate-then-up-before-ordered-corrections-v1";
constexpr const char* paired_schedule="paired-base-gate-up-before-ordered-corrections-v1";
enum class Stage { GateBase, UpBase, PairedBase, GateCorrection, UpCorrection, SiluUp, DownBase, DownCorrection };
constexpr std::array split_stages{Stage::GateBase,Stage::UpBase,Stage::GateCorrection,Stage::UpCorrection,
    Stage::SiluUp,Stage::DownBase,Stage::DownCorrection};
constexpr std::array paired_stages{Stage::PairedBase,Stage::GateCorrection,Stage::UpCorrection,
    Stage::SiluUp,Stage::DownBase,Stage::DownCorrection};
constexpr std::span<const Stage> stages(bool paired) {
    return paired?std::span<const Stage>(paired_stages):std::span<const Stage>(split_stages);
}
constexpr size_t stage_dispatches(Stage stage) {
    return stage==Stage::GateCorrection || stage==Stage::UpCorrection || stage==Stage::DownCorrection?2:1;
}
constexpr size_t full_dispatches(bool paired) {
    size_t count=0;for(auto stage:stages(paired))count+=stage_dispatches(stage);return count;
}
using Time=std::chrono::steady_clock;
double seconds(Time::time_point t) {return std::chrono::duration<double>(Time::now()-t).count();}
NSString* ns(std::string_view s) {return [[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding];}
std::string fingerprint(const std::string& value) {
    streaming::CanonicalEncoder encoder("tc-ane-calibration-source-v1");
    encoder.string_field("identity",value);return encoder.sha256();
}
std::string held_fingerprint(const streaming::SourceLease& lease) {
    const auto& f=lease.file("checkpoint");auto fd=lease.duplicate_fd("checkpoint");struct stat st{};
    require(::fstat(fd.get(),&st)==0 && S_ISREG(st.st_mode),"held checkpoint stat unavailable");
    const auto path=f.canonical_target_path.string();
    return fingerprint(":"+std::to_string(path.size())+":"+path+":"+std::to_string(st.st_dev)+":"+
        std::to_string(st.st_ino)+":"+std::to_string(st.st_size)+":"+std::to_string(st.st_mtimespec.tv_sec)+":"+
        std::to_string(st.st_mtimespec.tv_nsec)+":"+std::to_string(st.st_ctimespec.tv_sec)+":"+std::to_string(st.st_ctimespec.tv_nsec));
}
std::string read_small(int fd,uint64_t bytes) {
    require(bytes && bytes<2ULL<<20,"capture completion exceeds metadata bound");
    std::string out(size_t(bytes),'\0');size_t done=0;
    while(done<out.size()) {auto n=::pread(fd,out.data()+done,out.size()-done,off_t(done));
        if(n<0 && errno==EINTR)continue;
        require(n>0,"held capture metadata short read");done+=size_t(n);}
    require(out.find('\0')==std::string::npos,"metadata contains NUL");return out;
}
uint64_t integer(NSDictionary* d,NSString* key) {
    id v=d[key];require([v isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)v)!=CFBooleanGetTypeID(),"invalid capture integer");
    const double n=[v doubleValue];require(std::isfinite(n) && n>=0 && n<1e12 && n==std::floor(n),"capture integer outside bound");return uint64_t(n);
}
void write_exclusive(const std::filesystem::path& path,NSDictionary* receipt) {
    const auto text=json(receipt);streaming::OwnedSourceFd fd(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600));
    require(bool(fd),"receipt must be new in an existing directory");size_t done=0;
    while(done<text.size()) {const auto n=::write(fd.get(),text.data()+done,text.size()-done);
        if(n<0 && errno==EINTR)continue;
        require(n>0,"receipt write failed");done+=size_t(n);}
    require(::fsync(fd.get())==0,"receipt fsync failed");
}
struct CpuView {const void* data;size_t rows,columns,pitch,extent;};
struct Comparison {double rrms=0,max_abs=0;size_t differences=0,elements=0;};
Comparison compare(CpuView actual,CpuView expected) {
    for(const auto& v:{actual,expected})require(v.data && v.rows && v.columns && v.columns<=SIZE_MAX/2 &&
        v.pitch>=v.columns*2 && v.rows<=SIZE_MAX/v.pitch && v.rows*v.pitch<=v.extent,"invalid BF16 CPU view");
    require(actual.rows==expected.rows && actual.columns==expected.columns,"comparison geometry differs");
    Comparison result;double energy=0,error=0;
    for(size_t r=0;r<actual.rows;++r)for(size_t c=0;c<actual.columns;++c) {
        uint16_t av,bv;std::memcpy(&av,static_cast<const char*>(actual.data)+r*actual.pitch+c*2,2);
        std::memcpy(&bv,static_cast<const char*>(expected.data)+r*expected.pitch+c*2,2);
        const double a=std::bit_cast<float>(uint32_t(av)<<16),b=std::bit_cast<float>(uint32_t(bv)<<16);
        require(std::isfinite(a) && std::isfinite(b),"nonfinite FFN output");
        error+=(a-b)*(a-b);energy+=b*b;result.max_abs=std::max(result.max_abs,std::abs(a-b));result.differences+=av!=bv;
    }
    require(energy>0,"degenerate FFN reference");result.rrms=std::sqrt(error/energy);result.elements=actual.rows*actual.columns;return result;
}
CpuView cpu(const mg::MatrixView& v) {return {v.storage.contents(),v.rows,v.columns,v.row_stride_bytes,v.storage.size_bytes()};}
NSDictionary* dictionary(const Comparison& c) {return @{@"rrms":@(c.rrms),@"max_abs":@(c.max_abs),@"bf16_differences":@(c.differences),@"finite_elements":@(c.elements),@"passes_rrms_limit":@(c.rrms<=rrms_limit)};}
struct Scratch {const char* label;uint32_t columns;mg::ScalarType type;};
constexpr std::array<Scratch,6> specs{{{"lora-replay-gate",width,mg::ScalarType::BF16},
    {"lora-replay-up",width,mg::ScalarType::BF16},{"lora-replay-hidden",width,mg::ScalarType::BF16},
    {"lora-replay-output",hidden,mg::ScalarType::BF16},{"lora-replay-low",rank,mg::ScalarType::FP32},
    {"lora-replay-pair-copy",hidden,mg::ScalarType::BF16}}};
size_t pitch(const Scratch& s) {return (size_t(s.columns)+8)*mg::scalar_bytes(s.type);}
size_t backing(const Scratch& s) {return (rows+2)*pitch(s);}
size_t upper(const Scratch& s) {return (backing(s)+16383)&~size_t(16383);}
std::vector<mg::AllocationSiteBinding> site_bindings() {
    std::vector<mg::AllocationSiteBinding> out;for(const auto& s:specs)out.push_back({s.label,s.label,MemoryClass::Activation});return out;
}
std::unique_ptr<MemoryExecutionContext> context(const ProcessMemoryObservation& baseline,const std::string& source,bool cpu_only=false) {
    require(baseline.available && baseline.process_footprint_bytes && baseline.physical_memory_bytes,"missing process baseline");
    MemoryManifest m;m.candidate_id="native-lora-paired-base-layer0-component";m.checkpoint_digest=memory_sha256_hex(source);
    m.backend_revision="shared-six-scratch-v1";m.runtime_revision=memory_runtime_revision();
    uint64_t native_upper=0;for(const auto& s:specs) {
        m.sites.push_back({s.label,"native_lora_ffn","replay",MemoryClass::Activation,AllocationLifetime::Request,UpperProvenance::ExactShapeFormula});
        m.instances.push_back({s.label,0,upper(s),0,2,0,"physical_drain"});native_upper+=upper(s);
    }
    constexpr uint64_t framework=512ULL<<20,system=4ULL<<30;
    require(baseline.physical_memory_bytes>system,"insufficient physical memory guard");
    const auto budget=baseline.physical_memory_bytes-system;
    require(baseline.process_footprint_bytes<budget && native_upper+framework<=budget-baseline.process_footprint_bytes &&
        baseline.system_available_bytes>=system+native_upper+framework,"insufficient available memory for bounded replay");
    auto compiled=compile_memory_plan(m,baseline.process_footprint_bytes,framework,budget,true);
    require(compiled.complete && compiled.fits_budget,"replay phase plan is not executable");
    EffectiveMemoryPolicy p;p.enabled=true;p.effective_budget_bytes=budget;p.framework_upper_bytes=framework;
    p.planned_upper_bytes=compiled.planned_peak_bytes;p.system_reserve_bytes=system;p.denoiser_budget_bytes=native_upper;
    p.capability_level=MemoryCapabilityLevel::AllocationGuarded;p.certification_state=MemoryCertificationState::ExperimentalGuarded;
    p.execution_supported=true;p.route_available=true;p.estimate_fits=true;
    p.enforcement_scope="owned six native scratch backings only; produced MLX sources remain in retained baseline";
    ProcessMemoryObserver observer;if(cpu_only)observer=[baseline]{return baseline;};
    auto c=std::make_unique<MemoryExecutionContext>(p,std::move(compiled),baseline,std::move(observer));c->begin_running();return c;
}
mg::MatrixView imported(mg::Backend& b,const Tensor& a) {
    auto v=mg::import_mlx_matrix(b,a);
    const auto owner=v.storage.buffer().allocation_identity();const std::shared_ptr<const void> actual=a.data_shared_ptr();
    require(owner && actual && !owner.owner_before(actual) && !actual.owner_before(owner),"import lost actual MLX Data owner");
    return {v.storage,v.rows,v.columns,v.row_stride_bytes};
}
mg::MatrixView allocate(mg::Backend& b,const Scratch& s) {
    auto a=b.allocate(backing(s),s.label,mg::StorageMode::Shared);require(a.contents(),"shared scratch unavailable");
    std::memset(a.contents(),0x5a,a.size_bytes());return {a.view(pitch(s),rows*pitch(s),s.type),rows,s.columns,pitch(s)};
}
void padding(const mg::MatrixView& v) {
    const auto* data=static_cast<const unsigned char*>(v.storage.buffer().contents());require(data,"padding storage missing");
    const size_t item=mg::scalar_bytes(v.storage.scalar_type()),start=v.storage.offset_bytes();
    const auto unchanged=[&](size_t begin,size_t end) {for(size_t i=begin;i<end;++i)
        require(data[i]==0x5a,"native scratch guard/padding changed");};
    unchanged(0,start);
    for(size_t r=0;r<v.rows;++r)unchanged(start+r*v.row_stride_bytes+v.columns*item,start+(r+1)*v.row_stride_bytes);
    unchanged(start+v.rows*v.row_stride_bytes,v.storage.buffer().size_bytes());
}
std::string matrix_hash(const mg::MatrixView& v) {
    require(v.storage.contents() && v.row_stride_bytes==size_t(v.columns)*mg::scalar_bytes(v.storage.scalar_type()),"source must be compact and shared");
    return memory_sha256_hex({static_cast<const char*>(v.storage.contents()),size_t(v.rows)*v.row_stride_bytes});
}
mg::native_lora::Layout layout(const mg::MatrixView& v) {
    return {v.rows,v.columns,v.row_stride_bytes,v.storage.size_bytes(),v.storage.offset_bytes(),v.storage.scalar_type(),
        reinterpret_cast<uintptr_t>(v.storage.buffer().native_handle())};
}
bool eligible(const mg::MatrixView& x,const mg::OrderedLoRATerm& term,const mg::MatrixView& out,const mg::MatrixView& low) {
    mg::native_lora::TermLayout t{layout(term.a),layout(term.b),term.effective_scale,term.first_output,term.output_count};
    auto checked=mg::native_lora::validate_bindings({width,width,rank},layout(x),std::span(&t,1),layout(out),layout(out),layout(low));
    return checked.terms.size()==1 && mg::native_lora::sg_mma_eligible(checked.terms.front());
}
struct Trial {
    std::shared_ptr<const streaming::SourceLease> base_lease,adapter_lease;
    Weights weights;std::vector<Tensor> tensors;
    std::shared_ptr<MlxLeaseFdReader> input_reader;
    std::unique_ptr<MemoryExecutionContext> memory;
    std::unique_ptr<mg::MemoryBridge> bridge;
    std::unique_ptr<mg::Backend> backend;
    std::vector<mg::MatrixView> sources,scratch;
    std::array<mg::OrderedLoRATerm,3> terms;
    std::optional<mg::NativeLinear> gate_up,down;
    std::optional<mg::NativeGateUp> paired_base;
    std::optional<mg::NativeLoRA> corrections;
    std::optional<mg::Ticket> pending;
    bool uncertain=false;
    void release_native() {
        pending.reset();terms={};sources.clear();scratch.clear();corrections.reset();paired_base.reset();gate_up.reset();down.reset();backend.reset();
    }
};
std::string build_identity() {
    char* value=tc_runtime_build_identity();require(value,"runtime identity missing");const std::string s(value);tc_string_free(value);
    require(s==TC_PAIRED_REPLAY_EXPECTED_BUILD_ID,"loaded runtime differs from explicit linked build");return s;
}
void run(const std::filesystem::path& checkpoint,const std::filesystem::path& adapter,
         const std::filesystem::path& capture,const std::filesystem::path& output) {
    const auto build=build_identity();
    for(const auto& p:{checkpoint,adapter,capture,output})require(p.is_absolute() && p==p.lexically_normal(),"explicit normalized absolute paths required");
    require(capture.filename()==std::string(capture_id)+"-replay" && !std::filesystem::exists(output) && std::filesystem::is_directory(output.parent_path()),"wrong capture or output already exists");
    const auto* pin=qwen21::viggle_v021_adapter(adapter.filename().string());require(pin && pin->rank=="r128","only pinned Viggle r128 accepted");
    auto trial=std::make_unique<Trial>();NSMutableArray* records=[NSMutableArray array];
    NSMutableArray* pairs=[NSMutableArray array];
    try {
        const auto source_start=Time::now();std::atomic<bool> cancelled{false};
        const auto input_path=capture/"layer-0-step-5.safetensors";
        require(!std::filesystem::is_symlink(input_path),"capture symlink is not accepted");
        trial->base_lease=streaming::SourceLease::capture({{.logical_id="checkpoint",.path=checkpoint},
            {.logical_id="complete",.path=capture/"complete.json"},{.logical_id="input",.path=input_path}});
        const auto identity=held_fingerprint(*trial->base_lease);
        const auto hash_start=Time::now();
        trial->adapter_lease=streaming::SourceLease::capture_verified({{.logical_id="adapter",.path=adapter,.content_digest=std::string(pin->sha256)}},&cancelled);
        const double adapter_verify_seconds=seconds(hash_start);
        require(trial->adapter_lease->has_verified_content() && trial->adapter_lease->file("adapter").content_digest==pin->sha256,"held adapter content pin mismatch");
        const auto& af=trial->adapter_lease->file("adapter");
        const auto adapter_identity=af.canonical_target_path.string()+":"+std::to_string(af.bytes)+":"+
            std::to_string(af.mtime_ns)+":"+std::to_string(std::bit_cast<uint32_t>(1.f))+":"+af.content_digest;
        auto complete_fd=trial->base_lease->duplicate_fd("complete");
        NSDictionary* complete=parse_json(read_small(complete_fd.get(),trial->base_lease->file("complete").bytes).c_str());
        NSDictionary* binding=complete[@"binding"];
        require([complete[@"complete"] isEqual:@YES] && integer(complete,@"schema_version")==1 &&
            string_value(complete,@"artifact_type")=="bounded_ffn_input_calibration" &&
            string_value(complete,@"capture_recipe")=="ffn-input-fp16-regions-outliers-v1" && [binding isKindOfClass:NSDictionary.class],"invalid capture completion");
        require(string_value(binding,@"request_id")==capture_id && string_value(binding,@"model_id")=="qwen-image-2.1" &&
            string_value(binding,@"identity_kind")=="canonical-stat-identity" && string_value(binding,@"model_fingerprint")==identity &&
            string_value(binding,@"actual_execution")=="forced-gpu-runtime-boundary" && integer(binding,@"width")==512 &&
            integer(binding,@"height")==512 && integer(binding,@"total_steps")==6 && integer(binding,@"seed")==42 &&
            integer(binding,@"reference_count")==0 && integer(binding,@"reference_size")==0,"capture provenance mismatch");
        NSArray* loras=binding[@"loras"];require([loras isKindOfClass:NSArray.class] && loras.count==1,"capture adapter count mismatch");
        NSDictionary* lora=loras[0];require([lora isKindOfClass:NSDictionary.class] && integer(lora,@"strength")==1 &&
            string_value(lora,@"id")==adapter.filename().string() && string_value(lora,@"fingerprint")==fingerprint(adapter_identity+":0"),"capture adapter identity mismatch");
        require(trial->base_lease->file("input").bytes<9ULL<<20,"input exceeds replay bound");
        auto fd=trial->base_lease->duplicate_fd("input");trial->input_reader=std::make_shared<MlxLeaseFdReader>(MlxOwnedFd(fd.release()),"input");
        auto [arrays,metadata]=mx::load_safetensors(trial->input_reader);
        require(metadata.at("capture_recipe")=="qwen21-real-ffn-bf16-v1" && metadata.at("trajectory")=="forced-gpu" &&
            metadata.at("model_identity")==identity && metadata.at("adapter_identity")==adapter_identity &&
            metadata.at("prompt")==prompt && metadata.at("seed")=="42" && metadata.at("total_steps")=="6" &&
            metadata.at("layer")=="0" && metadata.at("step")=="5" && arrays.size()==1 &&
            arrays.at("input").shape()==mx::Shape{1,int(rows),int(hidden)} && arrays.at("input").dtype()==mx::bfloat16,"full BF16 LoRA capture mismatch");
        const auto file_digest=[&](const char* id) {auto f=trial->base_lease->duplicate_fd(id);return memory_sha256_fd(f.get(),trial->base_lease->file(id).bytes);};
        const auto complete_sha=file_digest("complete"),input_sha=file_digest("input");
        configure_streams();auto event=[](const std::string&,int,int){};
        trial->weights.load_lease(trial->base_lease,{"checkpoint"},event,cancelled);trial->weights.set_runtime_lora_fp16(false);
        require(trial->weights.apply_loras_lease(trial->adapter_lease,{"adapter"},{{adapter.string(),1.f,"transformer"}},"transformer",event,cancelled,true,true)==227,"expected exactly 227 resolved adapter projections");
        const auto snapshot=trial->weights.runtime_lora_snapshot();
        require(snapshot.native_metadata_compatible && !snapshot.fp16_rank_math && snapshot.bindings.size()==195,"unexpected adapter snapshot");
        const auto fused=trial->weights.at(std::string(prefix)+"gate_up.weight");const auto down=trial->weights.at(std::string(prefix)+"out.weight");
        require(fused.shape()==mx::Shape{int(2*width),int(hidden)} && down.shape()==mx::Shape{int(hidden),int(width)} &&
            fused.dtype()==mx::bfloat16 && down.dtype()==mx::bfloat16,"dense layer0 shape mismatch");
        trial->tensors={slice_axis(fused,0,0,width),slice_axis(fused,0,width,2*width),down,arrays.at("input")};
        std::array<std::optional<Weights::ResolvedLoRATerm>,3> selected;
        for(const auto& b:snapshot.bindings)for(const auto& t:b.ordered_terms) {
            int which=-1;if(b.prefix==std::string(prefix)+"gate_up")which=t.output_start==0?0:1;
            else if(b.prefix==std::string(prefix)+"out")which=2;
            if(which<0)continue;
            require(!selected[size_t(which)] && t.source_lease==trial->adapter_lease && t.lease_backed &&
                t.source_logical_id=="adapter" && t.source_role=="transformer" && t.source_adapter_ordinal==0 &&
                t.source_strength==1.f && t.scale==1.f && t.down_dtype==mx::bfloat16 && t.up_dtype==mx::bfloat16 &&
                t.output_start==(which==1?int(width):0) && t.output_end==int(which==2?hidden:(which==1?2*width:width)) &&
                t.down.shape()==mx::Shape{int(rank),int(which==2?width:hidden)} &&
                t.up.shape()==mx::Shape{int(which==2?hidden:width),int(rank)},"unexpected resolved layer0 factor/range");
            selected[size_t(which)]=t;
        }
        for(const auto& t:selected) {require(bool(t),"missing layer0 LoRA term");trial->tensors.push_back(t->down);trial->tensors.push_back(t->up);}
        mx::eval(trial->tensors);mx::synchronize();mx::clear_cache();
        require(mx::get_active_memory()<=source_bytes+(32ULL<<20),"unexpected full-model/factor materialization");
        trial->base_lease->revalidate_after_drain();trial->adapter_lease->revalidate_after_drain();
        const double source_prepare_seconds=seconds(source_start);
        // Hold the actual source Data owners before the oracle can donate storage.
        std::vector<std::shared_ptr<const void>> data_owners;for(const auto& t:trial->tensors)data_owners.push_back(t.data_shared_ptr());
        const auto oracle_start=Time::now();
        {auto oracle=qwen21::runtime_ffn::full(trial->weights,prefix);auto values=oracle({trial->tensors[3]});
         require(values.size()==1,"oracle output count differs");trial->tensors.push_back(mx::contiguous(values[0]));mx::eval(trial->tensors.back());mx::synchronize();}
        mx::clear_cache();const double oracle_seconds=seconds(oracle_start);const auto source_active=mx::get_active_memory();
        require(source_active<=source_bytes+(48ULL<<20),"oracle retained unexpected model allocations");
        const auto baseline=observe_process_memory();trial->memory=context(baseline,std::string(trial->base_lease->digest())+std::string(trial->adapter_lease->artifact_digest()));
        trial->bridge=std::make_unique<mg::MemoryBridge>(*trial->memory,0x514c4f524146464eULL,trial->base_lease->generation(),site_bindings());
        mg::BackendOptions options;options.allocation=trial->bridge->hooks();trial->backend=std::make_unique<mg::Backend>(std::move(options));auto& backend=*trial->backend;
        for(const auto& t:trial->tensors)trial->sources.push_back(imported(backend,t));
        for(const auto& s:specs)trial->scratch.push_back(allocate(backend,s));
        for(size_t i=0;i<3;++i)trial->terms[i]={trial->sources[4+i*2],trial->sources[5+i*2],1.f,0,i==2?hidden:width,0};
        auto& v=trial->scratch;for(size_t i=0;i<3;++i)require(eligible(i==2?v[2]:trial->sources[3],trial->terms[i],i==2?v[3]:v[i],v[4]),"real r128 term cannot select SGMMA");
        auto start=Time::now();trial->gate_up=mg::NativeLinear::prepare(backend,{hidden,width});trial->down=mg::NativeLinear::prepare(backend,{width,hidden});const auto linear_prepare=seconds(start);
        start=Time::now();trial->paired_base=mg::NativeGateUp::prepare(backend,{hidden,width,width});const auto paired_prepare=seconds(start);
        start=Time::now();trial->corrections=mg::NativeLoRA::prepare(backend,{width,width,rank},mg::native_lora::KernelVariant::SgMmaR128);const auto correction_prepare=seconds(start);
        const std::string correction_recipe(trial->corrections->actual_precision_recipe());
        const std::string split_base_recipe(mg::NativeLinear::precision_recipe());
        const std::string paired_base_recipe(mg::NativeGateUp::precision_recipe());
        require(correction_recipe==mg::native_lora::sg_mma_precision_recipe,"paired replay requires the frozen SGMMA correction recipe");
        std::vector<std::string> source_hashes;for(const auto& s:trial->sources)source_hashes.push_back(matrix_hash(s));
        const auto verify=[&](bool content) {trial->base_lease->revalidate_after_drain();trial->adapter_lease->revalidate_after_drain();
            require(held_fingerprint(*trial->base_lease)==identity,"held model generation changed");
            if(content) {
                require(file_digest("complete")==complete_sha && file_digest("input")==input_sha,"held capture source changed");
                for(size_t i=0;i<trial->sources.size();++i)require(matrix_hash(trial->sources[i])==source_hashes[i],"immutable source/oracle tensor changed");
            }
            for(const auto& s:v)padding(s);};
        bool numeric_pass=true;size_t calls=0;
        for(int round=0;round<3 && numeric_pass;++round) {
            const bool ab=round<2;NSMutableArray* pair=[NSMutableArray array];
            for(int order=0;order<2;++order) {
                const bool candidate=ab?(order==1):(order==0);const auto& op=*trial->corrections;
                start=Time::now();mg::CommandGraph graph;
                for(const auto stage:stages(candidate)) {
                    const auto before=graph.steps().size();
                    switch(stage) {
                        case Stage::GateBase:trial->gate_up->encode(graph,{trial->sources[3],trial->sources[0],v[0],{}});break;
                        case Stage::UpBase:trial->gate_up->encode(graph,{trial->sources[3],trial->sources[1],v[1],{}});break;
                        case Stage::PairedBase:trial->paired_base->encode(graph,trial->sources[3],trial->sources[0],trial->sources[1],v[0],v[1]);break;
                        case Stage::GateCorrection:op.encode_ordered_corrections(graph,trial->sources[3],std::span(&trial->terms[0],1),v[0],v[0],v[4]);break;
                        case Stage::UpCorrection:op.encode_ordered_corrections(graph,trial->sources[3],std::span(&trial->terms[1],1),v[1],v[1],v[4]);break;
                        case Stage::SiluUp:op.encode_silu_up(graph,v[0],v[1],v[2]);break;
                        case Stage::DownBase:trial->down->encode(graph,{v[2],trial->sources[2],v[3],{}});break;
                        case Stage::DownCorrection:op.encode_ordered_corrections(graph,v[2],std::span(&trial->terms[2],1),v[3],v[3],v[4]);break;
                    }
                    require(graph.steps().size()-before==stage_dispatches(stage),"FFN stage dispatch count differs from isolated schedule");
                }
                require(graph.steps().size()==full_dispatches(candidate),"unexpected complete FFN dispatch count");
                trial->uncertain=true;trial->pending.emplace(backend.submit(std::move(graph)));
                const auto done=trial->pending->wait(std::chrono::seconds(30));const double wall=seconds(start);
                if(done.physically_complete)trial->uncertain=false;
                require(done.ok && done.physically_complete && !done.timed_out && done.submitted_buffers && done.completed_buffers==done.submitted_buffers,"FFN physical completion failed");
                ++calls;
                NSMutableDictionary* record=[@{@"arm":candidate?@"paired-base":@"split-base",@"round":@(round),@"warmup":@(round==0),
                    @"order":@(order),@"wall_seconds":@(wall),@"gpu_span_seconds":@(done.gpu_seconds),@"gpu_span_available":@(done.gpu_seconds>0),
                    @"dispatches":@(full_dispatches(candidate)),@"base_gate_up_dispatches":@(candidate?1:2),
                    @"base_gate_up_kernel":candidate?@"paired-native-gate-up":@"two-native-linear",
                    @"base_gate_up_recipe":ns(candidate?paired_base_recipe:split_base_recipe),
                    @"schedule_id":@(candidate?paired_schedule:split_schedule),
                    @"actual_sgmma_low_rank_dispatches":@6,@"eligible_terms":@3,
                    @"lora_recipe":ns(correction_recipe),@"down_recipe":ns(split_base_recipe),@"physical_complete":@YES,
                    @"submitted_buffers":@(done.submitted_buffers),@"completed_buffers":@(done.completed_buffers)} mutableCopy];
                const auto snapshot_start=Time::now();
                if(order==0)for(size_t r=0;r<rows;++r)std::memcpy(static_cast<char*>(v[5].storage.contents())+r*v[5].row_stride_bytes,
                    static_cast<const char*>(v[3].storage.contents())+r*v[3].row_stride_bytes,hidden*2);
                record[@"output_snapshot_seconds"]=@(order==0?seconds(snapshot_start):0.);
                record[@"output_snapshot_bytes"]=@(order==0?size_t(rows)*hidden*2:0);
                [pair addObject:record];[records addObject:record];trial->pending.reset();
            }
            // No full CPU comparison between A and B. Only the first arm's
            // bounded 8-MiB snapshot separates their submissions; its time is
            // explicit. Keep the second output in place until pair validation.
            const auto validation_start=Time::now();
            const auto first_error=compare(cpu(v[5]),cpu(trial->sources.back()));
            const auto second_error=compare(cpu(v[3]),cpu(trial->sources.back()));
            NSMutableDictionary* first=pair[0];NSMutableDictionary* second=pair[1];
            first[@"oracle_error"]=dictionary(first_error);second[@"oracle_error"]=dictionary(second_error);
            second[@"previous_arm_error"]=dictionary(compare(cpu(v[3]),cpu(v[5])));
            numeric_pass=first_error.rrms<=rrms_limit && second_error.rrms<=rrms_limit;
            verify(false);trial->memory->checkpoint("lora_component_round_complete");
            second[@"pair_validation_after_both_arms_seconds"]=@(seconds(validation_start));
            NSDictionary* split=ab?first:second;NSDictionary* paired=ab?second:first;
            const double split_wall=[split[@"wall_seconds"] doubleValue];
            const double paired_wall=[paired[@"wall_seconds"] doubleValue];
            require(split_wall>0 && paired_wall>0,"invalid complete FFN wall timing");
            [pairs addObject:@{@"round":@(round),@"warmup":@(round==0),@"order":ab?@"AB":@"BA",
                @"split_full_ffn_wall_seconds":@(split_wall),@"paired_full_ffn_wall_seconds":@(paired_wall),
                @"paired_over_split_wall":@(paired_wall/split_wall),@"time_saved_fraction":@(1.-paired_wall/split_wall),
                @"numeric_screening_pass":@(numeric_pass),
                @"scope":@"Each arm is the complete base+ordered-LoRA+SiLU+down FFN through physical completion; excludes snapshot/validation and cold preparation"}];
        }
        verify(true);const auto observation=observe_process_memory();const auto before_release=trial->memory->admission().snapshot();
        const auto device=backend.device_name();const auto manifest=trial->memory->compiled_plan().manifest_digest;
        auto memory=std::move(trial->memory);trial->release_native();trial->bridge->close();trial->bridge.reset();
        const auto released=memory->admission().snapshot();require(!released.storage_count && !released.reservation_count &&
            !released.pending_release_count && !released.unknown_bytes,"same native phase ledger did not return to zero");
        memory->begin_draining();memory->finish_success();const auto report=memory->take_report();
        require(!report.metrics.tainted && !report.metrics.worker_quarantined,"memory phase failed or quarantined");
        NSMutableArray* retained_source_digests=[NSMutableArray array];for(const auto& digest:source_hashes)[retained_source_digests addObject:ns(digest)];
        write_exclusive(output,@{@"schema":@"native-qwen-lora-paired-ffn-replay-v1",@"status":numeric_pass?@"screening-pass":@"screening-rejected",
            @"scope":@"one historical LoRA layer0/step5 full FFN; not DiT/E2E, quality qualification or ANE",@"library_build_identity":ns(build),
            @"tool_source_sha256":@(TC_PAIRED_REPLAY_SOURCE_SHA),@"device":ns(device),@"capture_id":@(capture_id),@"step":@5,
            @"rows":@(rows),@"hidden":@(hidden),@"width":@(width),@"rank":@(rank),@"strength":@1,@"native_ffn_calls":@(calls),
            @"rrms_limit":@(rrms_limit),@"model_stat_fingerprint":ns(identity),@"capture_complete_sha256":ns(complete_sha),@"input_file_sha256":ns(input_sha),
            @"base_source_proof":@"same held descriptor and canonical stat generation; no full-checkpoint hash",@"adapter_source_proof":@"verified-held-fd",
            @"adapter_sha256":ns(af.content_digest),@"adapter_artifact_digest":ns(trial->adapter_lease->artifact_digest()),
            @"adapter_source_generation":@(trial->adapter_lease->generation()),@"adapter_verification_bytes_read":@(trial->adapter_lease->verification_bytes_read()),
            @"adapter_verification_cache_hits":@(trial->adapter_lease->verification_cache_hits()),
            @"actual_resolved_projections":@227,@"actual_resolved_bindings":@(snapshot.bindings.size()),
            @"produced_source_pre_post_sha256":retained_source_digests,
            @"produced_source_digest_order":@[@"gate",@"up",@"down",@"input",@"gate.A",@"gate.B",@"up.A",@"up.B",@"down.A",@"down.B",@"mlx_oracle"],
            @"prepare_seconds":@{@"source_including_adapter_verification":@(source_prepare_seconds),@"adapter_verification_included":@(adapter_verify_seconds),
                @"one_mlx_oracle":@(oracle_seconds),@"split_and_down_linear_pso":@(linear_prepare),
                @"paired_base_pso":@(paired_prepare),@"shared_sgmma_pso":@(correction_prepare)},
            @"memory":@{@"source_payload_bytes":@(source_bytes),@"mlx_active_after_oracle":@(source_active),@"native_scratch_backings":@6,
                @"same_phase_peak_committed_bytes":@(before_release.peak_committed_bytes),@"native_owned_bytes_before_release":@(before_release.storage_bytes),
                @"process_baseline_bytes":@(baseline.process_footprint_bytes),@"process_observed_after_bytes":@(observation.process_footprint_bytes),
                @"manifest_digest":ns(manifest),@"ledger_storage_count_after_release":@(released.storage_count),@"ledger_storage_bytes_after_release":@(released.storage_bytes),
                @"ledger_reservation_count_after_release":@(released.reservation_count),@"ledger_pending_release_count_after_release":@(released.pending_release_count),
                @"ledger_unknown_bytes_after_release":@(released.unknown_bytes),@"whole_request_certified":@NO,@"framework_guard_certified":@NO},
            @"source_immutable":@YES,@"actual_mlx_data_owner_verified":@YES,@"padding_intact":@YES,@"all_work_physically_complete":@YES,@"same_ledger_released":@YES,
            @"timing_scope":@"encode/submit/physical completion; no allocation, PSO, oracle, CPU comparison or provenance hashing in arm timings",
            @"pair_validation_scope":@"finite/RRMS comparisons, stat revalidation, padding and memory checkpoint after both arms; only an explicitly timed 8-MiB output snapshot between arms",
            @"gpu_span_scope":@"ticket first GPUStartTime to last GPUEndTime; zero unavailable, not kernel busy time",
            @"base_gate_up_kernel":@{@"A":@"two-native-linear",@"B":@"paired-native-gate-up"},
            @"base_gate_up_recipe":@{@"A":ns(split_base_recipe),@"B":ns(paired_base_recipe)},
            @"schedule_id":@{@"A":@(split_schedule),@"B":@(paired_schedule)},
            @"shared_lora_recipe":ns(correction_recipe),@"shared_down_recipe":ns(split_base_recipe),
            @"full_ffn_dispatches":@{@"A":@(full_dispatches(false)),@"B":@(full_dispatches(true))},
            @"same_sources_same_instances":@"Both arms reuse identical input/base/factor views, one SGMMA NativeLoRA, one down NativeLinear, one FP32 low view and all six scratch backings",
            @"shared_tail_order":@[@"gate-low",@"gate-apply",@"up-low",@"up-apply",@"silu-up",@"down-base",@"down-low",@"down-apply"],
            @"baseline_schedule_scope":@"A encodes gate-base then up-base before either correction. Historical SGMMA replay interleaved gate-base/gate-correction/up-base/up-correction; these timings do not reproduce that historical schedule or qualify E2E",
            @"oracle_scope":@"One actual compiled runtime_ffn::full MLX projection graph on the same retained Weights; oracle preparation includes first execution and is not a warmed MLX timing arm",
            @"cold_scope":@"split/down Linear, then paired base, then shared SGMMA prepared on one Backend; OS/driver compiler cache shared and not reset",
            @"sample_limit":@"one input; warm AB, record AB, record BA; six complete native FFNs at most; one MLX oracle outside timings",
            @"pairs":pairs,@"records":records});
        std::cout<<output<<'\n';require(numeric_pass,"RRMS screening rejected; receipt preserved");
    } catch(const std::exception& e) {
        const bool complete=trial && !trial->uncertain;
        if(trial && complete) {
            trial->release_native();if(trial->bridge)trial->bridge->close();
            if(trial->memory) {trial->memory->finalize_failure(e.what());trial->memory->complete_failure_cleanup(true);}
        }
        // Quarantine before receipt I/O: a disk/encoding error must not unwind
        // an uncertain owner whose imported sources may still be consumed.
        if(!complete)(void)trial.release();
        if(!std::filesystem::exists(output))write_exclusive(output,@{@"schema":@"native-qwen-lora-paired-ffn-replay-v1",@"status":@"failed",
            @"error":ns(e.what()),@"library_build_identity":ns(build),@"all_work_physically_complete":@(complete),
            @"owner_quarantined":@(!complete),@"process_exit_is_not_physical_proof":@YES,@"pairs":pairs,@"records":records});
        throw;
    }
}
void cpu_contract() {
    static_assert(source_bytes==308ULL<<20);
    static_assert(full_dispatches(false)==10 && full_dispatches(true)==9);
    require(split_stages[0]==Stage::GateBase && split_stages[1]==Stage::UpBase &&
        paired_stages[0]==Stage::PairedBase &&
        std::equal(split_stages.begin()+2,split_stages.end(),paired_stages.begin()+1,paired_stages.end()),
        "paired and split full FFNs must have exactly the same ordered correction/SiLU/down tail");
    const std::array<uint16_t,4> a{0x3f80,0xbf80,0,0x8000};auto b=a;
    const auto view=[](const auto& v){return CpuView{v.data(),2,2,4,v.size()*2};};
    require(compare(view(a),view(a)).rrms==0,"CPU exact comparison failed");b[3]=0;
    require(compare(view(a),view(b)).differences==1 && compare(view(a),view(b)).rrms==0,"signed zero diagnostic changed");
    b[0]=0x4000;require(compare(view(b),view(a)).rrms>rrms_limit,"CPU numeric rejection failed");
    size_t rejected=0;try {auto bad=view(a);bad.extent=1;(void)compare(bad,view(a));}catch(const std::exception&){++rejected;}
    b[0]=0x7f80;try {(void)compare(view(b),view(a));}catch(const std::exception&){++rejected;}
    using namespace mg::native_lora;
    for(int which=0;which<3;++which) {
        const uint32_t k=which==2?width:hidden,n=which==2?hidden:width;
        const Layout x{rows,k,size_t(k)*2,size_t(rows)*k*2,0,mg::ScalarType::BF16,1};
        const Layout out{rows,n,size_t(n)*2,size_t(rows)*n*2,0,mg::ScalarType::BF16,2};
        const Layout low{rows,rank,rank*4,size_t(rows)*rank*4,0,mg::ScalarType::FP32,3};
        const TermLayout term{{rank,k,size_t(k)*2,size_t(rank)*k*2,0,mg::ScalarType::BF16,4},
            {n,rank,rank*2,size_t(n)*rank*2,0,mg::ScalarType::BF16,5},1.f,0,n};
        auto checked=validate_bindings({width,width,rank},x,std::span(&term,1),out,out,low);
        require(sg_mma_eligible(checked.terms.front()),"real FFN projection should select SGMMA");
        auto tail=checked.terms.front();tail.low.rank=127;require(!sg_mma_eligible(tail),"rank tail must remain scalar");
    }
    {
        using GateLayout=mg::dense_gate_up::Layout;
        // The two read views can share the original fused checkpoint backing;
        // padded writes remain disjoint and preserve their byte offsets.
        const GateLayout x{rows,hidden,hidden*2,size_t(rows)*hidden*2,0,mg::ScalarType::BF16,10};
        const GateLayout gate{width,hidden,hidden*2,size_t(width)*hidden*2,0,mg::ScalarType::BF16,11};
        auto up=gate;up.offset_bytes=size_t(width)*hidden*2;
        const GateLayout g{rows,width,pitch(specs[0]),rows*pitch(specs[0]),pitch(specs[0]),mg::ScalarType::BF16,12};
        auto u=g;u.native_identity=13;
        const auto checked=mg::dense_gate_up::validate_bindings({hidden,width,width},x,gate,up,g,u,0);
        require(checked.parameters.rows==rows && checked.parameters.gate_output_stride==pitch(specs[0])/2 &&
            checked.parameters.up_output_stride==pitch(specs[1])/2,"paired FFN padded output contract changed");
        try {(void)mg::dense_gate_up::validate_bindings({hidden,width,width},x,gate,up,g,g,0);}catch(const std::exception&){++rejected;}
        auto aliases_source=g;aliases_source.native_identity=gate.native_identity;aliases_source.offset_bytes=0;
        try {(void)mg::dense_gate_up::validate_bindings({hidden,width,width},x,gate,up,aliases_source,u,0);}catch(const std::exception&){++rejected;}
        try {(void)mg::dense_gate_up::validate_bindings({hidden,width,width},x,gate,up,g,u,1);}catch(const std::exception&){++rejected;}
    }
    const ProcessMemoryObservation baseline{true,400ULL<<20,16ULL<<30,12ULL<<30,"cpu-fixture"};
    auto c=context(baseline,"cpu-only",true);mg::MemoryBridge bridge(*c,55,1,site_bindings());auto hooks=bridge.hooks();
    std::array<int,6> handles{};std::vector<std::shared_ptr<void>> claims;
    for(size_t i=0;i<specs.size();++i) {auto claim=hooks.reserve(upper(specs[i]),specs[i].label);require(bool(claim),"CPU site admission failed");
        hooks.allocated(claim,&handles[i],backing(specs[i]));claims.push_back(std::move(claim));}
    require(c->admission().snapshot().storage_count==6,"six backings must have six claims");
    try {(void)hooks.reserve(1,specs[0].label);}catch(const std::exception&){++rejected;}
    claims.clear();require(c->admission().snapshot().storage_count==0,"CPU ledger cleanup failed");bridge.close();c->begin_draining();c->finish_success();
    require(rejected==6,"CPU negative checks did not all reject");
    std::cout<<"lora_paired_ffn_replay_cpu=PASS split_dispatches=10 paired_dispatches=9 shared_tail_dispatches=8 real_geometry_eligible_terms=3 rejected=6 device_execution=0 model_loads=0\n";
}
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            if(argc==1 || (argc==2 && std::string_view(argv[1])=="--cpu-contract")) {cpu_contract();return 0;}
            require(argc==6 && std::string_view(argv[1])=="--device","usage: replay [--cpu-contract | --device checkpoint adapter capture-directory new-receipt.json]");
            run(argv[2],argv[3],argv[4],argv[5]);return 0;
        } catch(const std::exception& e) {std::cerr<<"native paired LoRA FFN replay failed: "<<e.what()<<'\n';return 1;}
    }
}
