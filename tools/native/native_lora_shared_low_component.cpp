// One generated SGv2 low/apply correctness fixture: two native submissions,
// five dispatches, no timing loop/model/ANE/dense-base work. Default execution
// is metadata only. CPU-only compilation excludes all MLX/device code.
#include "../../native/backends/metal_graph/native_lora.hpp"

#include <array>
#include <cstring>
#include <iostream>

namespace mg=tc::metal_graph;
namespace nl=mg::native_lora;
namespace {
void require(bool value,const std::string& message) {
    if(!value)throw std::runtime_error(message);
}
void host_contract() {
    const nl::Geometry geometry{128,128,128};
    const nl::Layout input{8,128,256,2048,16,mg::ScalarType::BF16,1};
    const nl::Layout output{8,128,272,2160,272,mg::ScalarType::BF16,2};
    const nl::Layout low{8,128,544,4320,544,mg::ScalarType::FP32,3};
    const nl::Layout a{128,128,256,32768,32,mg::ScalarType::BF16,4};
    const nl::Layout b{128,128,256,32768,48,mg::ScalarType::BF16,5};
    const nl::TermLayout term{a,b,1.F,0,128};
    const auto full=nl::validate_projection(geometry,nl::KernelVariant::SgMmaR128,input,term,output,low);
    require(full.selected_variant==nl::KernelVariant::SgMmaR128,"full-term SG selection failed");
    for(uint32_t first:{0U,64U}) {
        auto half=term;half.b.rows=64;half.b.offset_bytes+=first*256;
        half.b.view_size_bytes=64*256;half.first_output=first;half.output_count=64;
        const auto selected=nl::validate_projection(geometry,nl::KernelVariant::SgMmaR128,input,half,output,low);
        require(selected.selected_variant==full.selected_variant &&
                std::memcmp(&selected.term.low,&full.term.low,sizeof(nl::LowParameters))==0,
                "half-term changed shared-low precision/shape contract");
    }
    auto rejected=[&](auto action) {bool bad=false;try {action();}catch(const std::invalid_argument&){bad=true;}
        require(bad,"bad shared-low CPU metadata accepted");};
    auto bad=term;bad.b.rows=63;
    rejected([&]{(void)nl::validate_projection(geometry,nl::KernelVariant::SgMmaR128,input,bad,output,low);});
    auto aliased=low;aliased.storage=a.storage;
    rejected([&]{(void)nl::validate_projection(geometry,nl::KernelVariant::SgMmaR128,input,term,output,aliased);});
    rejected([&]{(void)nl::validate_projection({128,128,16},nl::KernelVariant::SgMmaR128,input,term,output,low);});
    std::cout<<"shared_low_cpu_contract=PASS device_created=0 model_loads=0\n";
}
} // namespace

#ifndef TC_SHARED_LOW_CPU_ONLY
#include "../../native/backends/mlx.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"
#include "../../native/backends/metal_graph/memory_bridge.hpp"
#include "../../native/runtime/memory_execution.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/platform/apple/platform.hpp"
#include <turbocider/turbocider.h>

#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>

#ifndef TC_SHARED_LOW_SOURCE_SHA
#define TC_SHARED_LOW_SOURCE_SHA "host-only-unbound"
#endif
#ifndef TC_SHARED_LOW_EXPECTED_BUILD_ID
#define TC_SHARED_LOW_EXPECTED_BUILD_ID "host-only-unbound"
#endif
namespace mx=tc::mx;
using tc::Tensor;
namespace {
constexpr uint32_t rows=8,columns=128,rank=128;
std::string quote(std::string_view text) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:text) {
        if(c=='"'||c=='\\')out<<'\\'<<char(c);
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else out<<char(c);
    }
    out<<'"';return out.str();
}
struct Receipt {
    std::ofstream file;
    explicit Receipt(const std::filesystem::path& path):file(path) {require(bool(file),"cannot open receipt");}
    void emit(const std::string& line) {
        file<<line<<'\n';file.flush();require(bool(file),"receipt write failed");std::cout<<line<<std::endl;
    }
};
Tensor tensor(const std::vector<float>& data,int m,int n) {
    return mx::astype(Tensor(data.data(),{m,n},mx::float32),mx::bfloat16);
}
mg::MatrixView imported(mg::Backend& backend,const Tensor& value) {
    const auto matrix=mg::import_mlx_matrix(backend,value);
    const auto owner=matrix.storage.buffer().allocation_identity();
    const std::shared_ptr<const void> actual=value.data_shared_ptr();
    require(owner && actual && !owner.owner_before(actual) && !actual.owner_before(owner),
            "import does not retain the actual MLX Data owner");
    return {matrix.storage,matrix.rows,matrix.columns,matrix.row_stride_bytes};
}
std::string digest(const mg::MatrixView& matrix) {
    const auto item=mg::scalar_bytes(matrix.storage.scalar_type());
    require(matrix.storage.contents() && matrix.row_stride_bytes==size_t(matrix.columns)*item,
            "source guard requires compact completed Shared storage");
    return tc::memory_sha256_hex({static_cast<const char*>(matrix.storage.contents()),
                                  size_t(matrix.rows)*matrix.row_stride_bytes});
}
struct Guard {mg::MatrixView source;std::string hash;};
size_t pitch(mg::ScalarType type) {return(size_t(columns)+8)*mg::scalar_bytes(type);}
size_t backing_bytes(mg::ScalarType type) {return(size_t(rows)+2)*pitch(type);}
size_t upper(mg::ScalarType type) {return(backing_bytes(type)+16383)&~size_t(16383);}
struct Fixture {
    tc::Weights weights;
    std::shared_ptr<const tc::streaming::SourceLease> lease;
    std::vector<Tensor> tensors;
    std::vector<Guard> guards;
    std::unique_ptr<tc::MemoryExecutionContext> memory;
    std::unique_ptr<mg::MemoryBridge> bridge;
    std::unique_ptr<mg::Backend> backend;
    std::optional<mg::NativeLoRA> operation;
    std::array<mg::MatrixView,2> output,low;
    mg::MatrixView input,base,oracle,low_oracle;
    mg::OrderedLoRATerm term;
    std::optional<mg::Ticket> pending;
    bool uncertain=false;
    size_t submissions=0,dispatches=0,generated_file_bytes=0;
    uint64_t baseline_bytes=0,planned_bytes=0;

    void prepare(const std::filesystem::path& directory) {
        const std::string prefix="transformer_blocks.0.attn.to_q";
        std::vector<float> xv(rows*columns),yv(rows*columns),av(rank*columns),bv(columns*rank);
        // Dyadic integer numerators keep every K/rank partial sum exactly
        // representable in FP32, allowing a strict bit oracle for SG/MLX.
        for(size_t i=0;i<xv.size();++i) {
            xv[i]=float(int(i%7)-3)/16.F;yv[i]=float(int(i%11)-5)/16.F;
        }
        for(size_t i=0;i<av.size();++i)av[i]=float(int((i/columns*3+i%columns*5)%9)-4)/64.F;
        for(size_t i=0;i<bv.size();++i)bv[i]=float(int((i/rank*7+i%rank*11)%11)-5)/64.F;
        Tensor x=tensor(xv,rows,columns),y=tensor(yv,rows,columns);
        Tensor a=tensor(av,rank,columns),b=tensor(bv,columns,rank);
        // Shape-only metadata. The supplied-base Weights oracle does not
        // materialize this dense weight or compute a base GEMM.
        weights.bind_arrays({prefix+".weight"},{mx::zeros({int(columns),int(columns)},mx::bfloat16)});
        const auto artifact=directory/"generated-r128.safetensors";
        mx::save_safetensors(artifact.string(),{{prefix+".lora_A.weight",a},{prefix+".lora_B.weight",b}});
        generated_file_bytes=std::filesystem::file_size(artifact);
        require(generated_file_bytes<=70*1024,"generated artifact exceeds fixed 70 KiB bound");
        tc::streaming::SourceFileIdentity file;file.logical_id="generated-r128";file.path=artifact;
        lease=tc::streaming::SourceLease::capture_verified({file});
        std::atomic<bool> cancelled=false;
        require(weights.apply_loras_lease(lease,{file.logical_id},{{artifact.string(),1.F,"transformer"}},
            "transformer",[](const std::string&,int,int){},cancelled,true,true)==1,
            "actual held-fd Weights loader did not resolve one adapter");
        const auto snapshot=weights.runtime_lora_snapshot();
        require(snapshot.native_metadata_compatible && !snapshot.fp16_rank_math && snapshot.bindings.size()==1 &&
            snapshot.bindings[0].prefix==prefix && snapshot.bindings[0].ordered_terms.size()==1,
            "unexpected actual Weights snapshot");
        const auto& resolved=snapshot.bindings[0].ordered_terms[0];
        require(resolved.lease_backed && resolved.down_dtype==mx::bfloat16 && resolved.up_dtype==mx::bfloat16 &&
            resolved.down.shape()==mx::Shape{int(rank),int(columns)} &&
            resolved.up.shape()==mx::Shape{int(columns),int(rank)} && resolved.output_start==0 &&
            resolved.output_end==int(columns),"resolved factor source/shape/dtype changed");
        tensors={x,y,resolved.down,resolved.up};mx::eval(tensors);mx::synchronize();
        const auto baseline=tc::observe_process_memory();
        require(baseline.available && baseline.physical_memory_bytes>4ULL<<30,"memory observation unavailable");
        baseline_bytes=baseline.process_footprint_bytes;
        tc::MemoryManifest manifest;manifest.candidate_id="generated-native-lora-shared-low-v1";
        manifest.checkpoint_digest=std::string(lease->artifact_digest());
        manifest.backend_revision="same-sgv2-low-two-channel-apply";manifest.runtime_revision=tc::memory_runtime_revision();
        std::vector<mg::AllocationSiteBinding> bindings;
        for(const auto& label:{std::string("shared-low-output"),std::string("shared-low-rank")}) {
            const auto type=label.ends_with("rank")?mg::ScalarType::FP32:mg::ScalarType::BF16;
            manifest.sites.push_back({label,"generated-component","shared-low",tc::MemoryClass::Activation,
                tc::AllocationLifetime::Request,tc::UpperProvenance::ExactShapeFormula});
            bindings.push_back({label,label,tc::MemoryClass::Activation});
            for(uint32_t i=0;i<2;++i)manifest.instances.push_back({label,i,upper(type),0,1,0,"physical-completion-and-release"});
        }
        constexpr uint64_t framework=512ULL<<20,system=4ULL<<30;
        const auto budget=baseline.physical_memory_bytes-system;
        auto compiled=tc::compile_memory_plan(manifest,baseline_bytes,framework,budget,true);
        require(compiled.complete && compiled.fits_budget,"generated native memory plan not admitted");
        planned_bytes=compiled.planned_peak_bytes;
        tc::EffectiveMemoryPolicy policy;policy.enabled=true;policy.effective_budget_bytes=budget;
        policy.planned_upper_bytes=planned_bytes;policy.framework_upper_bytes=framework;policy.system_reserve_bytes=system;
        memory=std::make_unique<tc::MemoryExecutionContext>(policy,std::move(compiled),baseline);memory->begin_running();
        bridge=std::make_unique<mg::MemoryBridge>(*memory,0x53484152454c4f57ULL,lease->generation(),std::move(bindings));
        mg::BackendOptions options;options.allocation=bridge->hooks();backend=std::make_unique<mg::Backend>(std::move(options));
        auto keep=[&](const Tensor& value) {auto matrix=imported(*backend,value);guards.push_back({matrix,digest(matrix)});return matrix;};
        input=keep(x);base=keep(y);
        term={keep(resolved.down),keep(resolved.up),resolved.scale,0,columns,resolved.source_adapter_ordinal};
        // Imports retain all original Data owners before the oracle can donate.
        Tensor expected=mx::contiguous(weights.apply_runtime_lora_slice(y,x,prefix,0,columns,0,columns,false));
        Tensor expected_low=mx::contiguous(mx::matmul(mx::astype(x,mx::float32),
            mx::transpose(mx::astype(resolved.down,mx::float32))));
        mx::eval(expected,expected_low);mx::synchronize();
        tensors.push_back(expected);tensors.push_back(expected_low);oracle=keep(expected);low_oracle=keep(expected_low);
        operation=mg::NativeLoRA::prepare(*backend,{columns,columns,rank},nl::KernelVariant::SgMmaR128);
        for(size_t arm=0;arm<2;++arm) {
            output[arm]=allocate(mg::ScalarType::BF16,"shared-low-output");
            low[arm]=allocate(mg::ScalarType::FP32,"shared-low-rank");
            for(uint32_t r=0;r<rows;++r)std::memcpy(static_cast<std::byte*>(output[arm].storage.contents())+
                size_t(r)*output[arm].row_stride_bytes,static_cast<const std::byte*>(base.storage.contents())+
                size_t(r)*base.row_stride_bytes,columns*2);
        }
        verify_sources();memory->checkpoint("shared_low_prepared");
    }
    mg::MatrixView allocate(mg::ScalarType type,const char* label) {
        auto buffer=backend->allocate(backing_bytes(type),label);
        require(buffer.contents(),"scratch is not Shared storage");std::memset(buffer.contents(),0x5a,buffer.size_bytes());
        return {buffer.view(pitch(type),size_t(rows)*pitch(type),type),rows,columns,pitch(type)};
    }
    mg::OrderedLoRATerm half(uint32_t first) const {
        auto selected=term;const auto begin=term.b.storage.offset_bytes()+size_t(first)*term.b.row_stride_bytes;
        const auto bytes=63*term.b.row_stride_bytes+size_t(rank)*2;
        selected.b={term.b.storage.buffer().view(begin,bytes,mg::ScalarType::BF16),64,rank,term.b.row_stride_bytes};
        selected.first_output=first;selected.output_count=64;return selected;
    }
    mg::Completion execute(mg::CommandGraph graph,size_t count) {
        require(graph.steps().size()==count && submissions<2,"fixed two-submission dispatch budget exceeded");
        dispatches+=count;++submissions;uncertain=true;
        pending.emplace(backend->submit(std::move(graph)));
        const auto result=pending->wait(std::chrono::seconds(10));
        if(result.physically_complete)uncertain=false;
        require(result.ok && result.physically_complete && !result.timed_out &&
            result.submitted_buffers==1 && result.completed_buffers==1,
            "generated submission did not physically succeed: "+result.error);
        return result;
    }
    void run(Receipt& receipt) {
        const auto complete=operation->validate_projection(input,term,output[1],low[1]);
        const auto first=operation->validate_projection(input,half(0),output[1],low[1]);
        const auto second=operation->validate_projection(input,half(64),output[1],low[1]);
        require(complete.selected_variant()==nl::KernelVariant::SgMmaR128 &&
            first.selected_variant()==complete.selected_variant() && second.selected_variant()==complete.selected_variant() &&
            std::memcmp(&complete.parameters().low,&first.parameters().low,sizeof(nl::LowParameters))==0 &&
            std::memcmp(&complete.parameters().low,&second.parameters().low,sizeof(nl::LowParameters))==0,
            "shared low descriptors changed selected pair/low arithmetic");
        mg::CommandGraph full_graph;operation->encode_ordered_corrections(full_graph,input,
            std::span(&term,1),output[0],output[0],low[0]);
        const auto full_result=execute(std::move(full_graph),2);
        mg::CommandGraph split_graph;operation->encode_low(split_graph,complete);
        operation->encode_apply_inplace(split_graph,first);operation->encode_apply_inplace(split_graph,second);
        const auto split_result=execute(std::move(split_graph),3);
        compare(low[0],low[1],"old-vs-shared-low");compare(low[1],low_oracle,"shared-low-vs-actual-mlx-low");
        compare(output[0],output[1],"old-vs-shared-output");
        compare(output[0],oracle,"old-output-vs-actual-weights");compare(output[1],oracle,"shared-output-vs-actual-weights");
        for(const auto& view:{output[0],output[1],low[0],low[1]})padding(view);
        verify_sources();memory->checkpoint("shared_low_physically_complete");
        receipt.emit("{\"event\":\"math_complete\",\"actual_pair\":\"SgMmaR128\",\"M\":8,\"K\":128,\"N\":128,\"rank\":128,\"old_dispatches\":2,\"shared_dispatches\":3,\"shared_low_dispatches\":1,\"channel_apply_ranges\":[[0,64],[64,128]],\"low_bit_differences\":0,\"output_bit_differences\":0,\"finite\":true,\"source_padding_unchanged\":true,\"actual_weights_gpu_oracle\":true,\"actual_mlx_data_owners_verified\":true,\"physical_native_tickets_completed\":2,\"native_low_readiness_scope\":\"same graph order and successful physical ticket; not a cross-graph certificate\",\"submitted_buffers\":"+
            std::to_string(full_result.submitted_buffers+split_result.submitted_buffers)+",\"completed_buffers\":"+
            std::to_string(full_result.completed_buffers+split_result.completed_buffers)+"}");
    }
    static void compare(const mg::MatrixView& actual,const mg::MatrixView& expected,const char* name) {
        require(actual.storage.contents() && expected.storage.contents() && actual.rows==expected.rows &&
            actual.columns==expected.columns && actual.storage.scalar_type()==expected.storage.scalar_type(),"comparison shape/storage mismatch");
        const auto item=mg::scalar_bytes(actual.storage.scalar_type());size_t differences=0;
        for(uint32_t r=0;r<actual.rows;++r)for(uint32_t c=0;c<actual.columns;++c) {
            uint32_t a=0,b=0;
            std::memcpy(&a,static_cast<const std::byte*>(actual.storage.contents())+size_t(r)*actual.row_stride_bytes+c*item,item);
            std::memcpy(&b,static_cast<const std::byte*>(expected.storage.contents())+size_t(r)*expected.row_stride_bytes+c*item,item);
            require(std::isfinite(std::bit_cast<float>(item==2?a<<16:a)) &&
                std::isfinite(std::bit_cast<float>(item==2?b<<16:b)),"nonfinite result");differences+=a!=b;
        }
        std::cout<<"comparison="<<name<<" bit_differences="<<differences<<'\n';
        require(differences==0,"generated dyadic comparison differs: "+std::string(name));
    }
    static void padding(const mg::MatrixView& matrix) {
        const auto* data=static_cast<const unsigned char*>(matrix.storage.buffer().contents());
        const auto item=mg::scalar_bytes(matrix.storage.scalar_type());require(data,"padding read requires physical completion");
        for(size_t i=0;i<matrix.storage.buffer().size_bytes();++i) {
            const bool logical=i>=matrix.storage.offset_bytes() &&
                i<matrix.storage.offset_bytes()+size_t(matrix.rows)*matrix.row_stride_bytes &&
                (i-matrix.storage.offset_bytes())%matrix.row_stride_bytes<size_t(matrix.columns)*item;
            if(!logical)require(data[i]==0x5a,"scratch prefix/tail/padding changed");
        }
    }
    void verify_sources() const {
        for(const auto& guard:guards)require(digest(guard.source)==guard.hash,"immutable produced source/oracle changed");
        lease->revalidate_after_drain();
    }
    void release_native() {
        pending.reset();term={};output={};low={};input={};base={};oracle={};low_oracle={};guards.clear();
        operation.reset();backend.reset();
    }
    void finish() {
        require(!uncertain && submissions==2 && dispatches==5,"native fixture not physically complete");
        release_native();bridge->close();memory->checkpoint("shared_low_released");
        memory->enter_epoch(1,"shared_low_physical_release");memory->begin_draining();memory->finish_success();
        const auto& m=memory->take_report().metrics;
        require(m.execution_state=="succeeded" && !m.tainted && !m.worker_quarantined &&
            !m.ledger_storage_count && !m.ledger_storage_bytes && !m.ledger_reserved_bytes &&
            !m.ledger_pending_release_bytes && !m.ledger_unknown_bytes && !m.ledger_cached_bytes,
            "owned native ledger did not return to zero");
    }
    void failure_cleanup() noexcept {
        // Uncertain submission keeps the complete fixture/owners in quarantine
        // until process exit. There is no retry or guessed event completion.
        if(uncertain)return;
        try {
            release_native();if(bridge)bridge->close();
            if(memory) {const auto d=memory->finalize_failure("shared-low generated fixture failed");
                if(d==tc::MemoryFailureDisposition::NeedsGpuDrain)(void)memory->complete_failure_cleanup(true,{});}
        }catch(...) {}
    }
};
void tiny_device(const std::filesystem::path& directory,const char* binary) {
    char* raw=tc_runtime_build_identity();require(raw,"native library identity missing");
    const std::string identity(raw);tc_string_free(raw);
    require(identity==TC_SHARED_LOW_EXPECTED_BUILD_ID,"loaded library differs from fresh pinned build");
    require(directory.is_absolute() && directory==directory.lexically_normal() && directory!=directory.root_path() &&
        std::filesystem::is_directory(directory.parent_path()) && !std::filesystem::exists(directory) &&
        !std::filesystem::is_symlink(std::filesystem::symlink_status(directory)),"output must be fresh normalized absolute directory");
    require(std::filesystem::create_directory(directory),"cannot create output directory");
    Receipt receipt(directory/"receipt.jsonl");
    receipt.emit("{\"event\":\"identity\",\"library_build_id\":"+quote(identity)+",\"tool_source_sha256\":"+
        quote(TC_SHARED_LOW_SOURCE_SHA)+",\"binary_sha256\":"+quote(tc::sha256_file(std::filesystem::canonical(binary)))+
        ",\"scope\":\"generated shared low and split apply only; no model/dense base/ANE/full FFN/E2E\",\"planned_native_submissions\":2,\"planned_native_dispatches\":5,\"performance_qualified\":false}");
    tc::configure_streams();auto fixture=std::make_unique<Fixture>();
    try {
        fixture->prepare(directory);receipt.emit("{\"event\":\"prepared\",\"device\":"+quote(fixture->backend->device_name())+
            ",\"recipe\":"+quote(fixture->operation->actual_precision_recipe())+",\"generated_artifact_sha256\":"+
            quote(fixture->lease->artifact_digest())+",\"generated_artifact_bytes\":"+std::to_string(fixture->generated_file_bytes)+
            ",\"source_generation\":"+std::to_string(fixture->lease->generation())+",\"native_scratch_backings\":4,\"native_capacity_upper_bytes\":65536,\"process_baseline_bytes\":"+
            std::to_string(fixture->baseline_bytes)+",\"planned_peak_bytes\":"+std::to_string(fixture->planned_bytes)+
            ",\"framework_guard_certified\":false,\"whole_request_certified\":false}");
        fixture->run(receipt);fixture->finish();
        receipt.emit("{\"event\":\"complete\",\"accepted\":true,\"native_submissions\":2,\"native_dispatches\":5,\"all_work_physically_complete\":true,\"native_ledger_zero\":true,\"models_loaded\":0,\"ane_calls\":0,\"performance_qualified\":false}");
    }catch(...) {
        if(fixture->uncertain) {
            if(fixture->bridge)fixture->bridge->close();(void)fixture.release();
            receipt.emit("{\"event\":\"quarantined\",\"unknown_physical_completion\":true,\"fixture_owners_retained\":true,\"retry\":false}");
        }else fixture->failure_cleanup();
        throw;
    }
}
} // namespace
#endif

int main(int argc,char** argv) {
    try {
        if(argc==1 || (argc==2 && std::string_view(argv[1])=="--cpu-contract")) {host_contract();return 0;}
#ifndef TC_SHARED_LOW_CPU_ONLY
        require(argc==3 && std::string_view(argv[1])=="--tiny-device",
            "usage: native-lora-shared-low-component [--cpu-contract|--tiny-device /absolute/fresh-output]");
        tiny_device(argv[2],argv[0]);return 0;
#else
        throw std::invalid_argument("CPU-only binary cannot execute device work");
#endif
    }catch(const std::exception& error) {
        std::cerr<<"shared_low_component=FAIL "<<error.what()<<'\n';return 1;
    }
}
