// Explicit generated correction-path probe. It never loads a model, computes
// a dense base projection, runs ANE or claims a complete FFN/E2E speedup.
#include "../../native/backends/mlx.hpp"
#include "../../native/backends/metal_graph/native_lora.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"
#include "../../native/backends/metal_graph/memory_bridge.hpp"
#include "../../native/runtime/memory_execution.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "native_lora_sgmma_probe_contract.hpp"
#include <turbocider/turbocider.h>

#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>

#ifndef TC_LORA_PROBE_SOURCE_SHA
#define TC_LORA_PROBE_SOURCE_SHA "host-only-unbound"
#endif
#ifndef TC_LORA_PROBE_CONTRACT_SHA
#define TC_LORA_PROBE_CONTRACT_SHA "host-only-unbound"
#endif
#ifndef TC_LORA_PROBE_EXPECTED_BUILD_ID
#define TC_LORA_PROBE_EXPECTED_BUILD_ID "host-only-unbound"
#endif
namespace mg=tc::metal_graph;
namespace mx=tc::mx;
namespace lp=tc::lora_probe;
using tc::Tensor;
using Clock=std::chrono::steady_clock;
namespace {
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
std::string quote(std::string_view s){std::ostringstream out;out<<'"';for(unsigned char c:s){
    if(c=='"'||c=='\\')out<<'\\'<<char(c);else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
    else out<<char(c);}out<<'"';return out.str();}
struct Log { std::ofstream file; explicit Log(const std::filesystem::path& path):file(path){require(bool(file),"cannot open receipt");}
    void emit(const std::string& row){std::cout<<row<<std::endl;file<<row<<'\n';file.flush();require(bool(file),"receipt write failed");} };
mg::MatrixView imported(mg::Backend& b,const Tensor& x){const auto v=mg::import_mlx_matrix(b,x);return{v.storage,v.rows,v.columns,v.row_stride_bytes};}
uint64_t digest(const mg::MatrixView& x){const auto* p=static_cast<const unsigned char*>(x.storage.contents());require(p,"fixture source needs Shared storage");
    uint64_t h=14695981039346656037ULL;for(size_t i=0;i<x.storage.size_bytes();++i){h^=p[i];h*=1099511628211ULL;}return h;}
lp::View cpu(const mg::MatrixView& x){const auto t=x.storage.scalar_type();require(t==mg::ScalarType::BF16||t==mg::ScalarType::FP32,"bad comparison dtype");
    return{static_cast<const std::byte*>(x.storage.contents()),x.storage.size_bytes(),x.rows,x.columns,x.row_stride_bytes,
        t==mg::ScalarType::BF16?lp::Type::BF16:lp::Type::FP32};}
uint32_t mix(uint32_t v){v^=v>>16;v*=0x7feb352dU;v^=v>>15;v*=0x846ca68bU;return v^(v>>16);}
Tensor generated(uint32_t rows,uint32_t cols,uint32_t salt,float amplitude,mx::Dtype dtype){
    std::vector<float> values(size_t(rows)*cols);for(size_t i=0;i<values.size();++i)values[i]=float(int(mix(uint32_t(i)+salt)&2047U)-1024)*amplitude/1024.f;
    return mx::astype(Tensor(values.data(),{int(rows),int(cols)},mx::float32),dtype);
}
struct Guard{mg::MatrixView view;uint64_t hash;};
struct Sources {
    mg::Backend importer;
    tc::Weights weights;
    std::optional<Tensor> input,base,oracle,low_oracle;
    std::vector<Tensor> retained;
    std::vector<Guard> guards;
    std::shared_ptr<const tc::streaming::SourceLease> lease;
    std::vector<mg::OrderedLoRATerm> terms;
    mg::MatrixView x,y,expected,expected_low;
    uint32_t rows,k,n;bool trap;
    std::string prefix="transformer_blocks.0.attn.to_q";
    Sources(const std::filesystem::path& directory,uint32_t m,uint32_t in,uint32_t out,bool exact)
        :rows(m),k(in),n(out),trap(exact) {
        require(std::filesystem::create_directory(directory),"fixture case directory already exists");
        input=generated(m,k,31,0.5f,mx::bfloat16);base=generated(m,n,71,0.25f,mx::bfloat16);
        if(trap){std::vector<float> a(size_t(m)*k,0.f),b(size_t(m)*n,-8192.f);for(uint32_t r=0;r<m;++r)a[size_t(r)*k]=1.f;
            input=mx::astype(Tensor(a.data(),{int(m),int(k)},mx::float32),mx::bfloat16);
            base=mx::astype(Tensor(b.data(),{int(m),int(n)},mx::float32),mx::bfloat16);}
        // Shape-only lazy metadata for the real Weights supplied-base API.
        // Never materialize this [N,K] tensor or call project_base_slice.
        weights.bind_arrays({prefix+".weight"},{mx::zeros({int(n),int(k)},mx::bfloat16)});
        std::vector<tc::LoRAAsset> assets;std::vector<std::string> ids;std::vector<tc::streaming::SourceFileIdentity> files;
        for(uint32_t ordinal=0;ordinal<(trap?2U:1U);++ordinal){
            const auto type=ordinal?mx::float32:mx::bfloat16;
            Tensor a=generated(128,k,101+ordinal,0.2f/std::sqrt(float(k)),type);
            Tensor b=generated(n,128,307+ordinal,0.3f/std::sqrt(128.f),type);
            if(trap){std::vector<float> av(size_t(128)*k),bv(size_t(n)*128);
                av[0]=ordinal?1.0001220703125f:1.f;
                for(uint32_t row=0;row<n;++row)bv[size_t(row)*128]=ordinal?8193.f:34.f;
                a=mx::astype(Tensor(av.data(),{128,int(k)},mx::float32),type);
                b=mx::astype(Tensor(bv.data(),{int(n),128},mx::float32),type);}
            const auto path=directory/("generated-"+std::to_string(ordinal)+".safetensors");
            mx::save_safetensors(path.string(),{{prefix+".lora_A.weight",a},{prefix+".lora_B.weight",b}});
            tc::streaming::SourceFileIdentity file;file.logical_id="generated-"+std::to_string(ordinal);file.path=path;
            ids.push_back(file.logical_id);files.push_back(std::move(file));assets.push_back({path.string(),trap&&!ordinal?0.5f:1.f,"transformer"});
        }
        lease=tc::streaming::SourceLease::capture_verified(std::move(files));
        std::atomic<bool> cancelled=false;
        require(weights.apply_loras_lease(lease,ids,assets,"transformer",[](const std::string&,int,int){},cancelled,true,true)==assets.size(),"actual Weights resolver did not bind all terms");
        const auto snapshot=weights.runtime_lora_snapshot();require(snapshot.bindings.size()==1&&!snapshot.fp16_rank_math&&snapshot.native_metadata_compatible,"unsupported generated adapter snapshot");
        std::vector<Tensor> ready{*input,*base};for(const auto& term:snapshot.bindings[0].ordered_terms){ready.push_back(term.down);ready.push_back(term.up);}
        mx::eval(ready);mx::synchronize();
        auto keep=[&](const Tensor& value){retained.push_back(value);auto view=imported(importer,value);guards.push_back({view,digest(view)});return view;};
        x=keep(*input);y=keep(*base); // Retain before oracle construction prevents donation.
        for(const auto& t:snapshot.bindings[0].ordered_terms){const auto expected_type=trap&&t.source_adapter_ordinal?mx::float32:mx::bfloat16;require(t.down_dtype==expected_type&&t.up_dtype==expected_type,"factor dtype was narrowed");require(t.down.shape()==mx::Shape{128,int(k)}&&t.up.shape()==mx::Shape{int(n),128},"factor shape changed");
            terms.push_back({keep(t.down),keep(t.up),t.scale,0,n,t.source_adapter_ordinal});}
        oracle=mx::contiguous(weights.apply_runtime_lora_slice(*base,*input,prefix,0,int(n),0,int(k),false));
        const auto& last=snapshot.bindings[0].ordered_terms.back();
        low_oracle=mx::contiguous(mx::matmul(mx::astype(*input,mx::float32),mx::transpose(mx::astype(last.down,mx::float32))));
        mx::eval(*oracle,*low_oracle);mx::synchronize();expected=keep(*oracle);expected_low=keep(*low_oracle);
        if(trap){const auto* actual=static_cast<const uint16_t*>(expected.storage.contents());
            for(size_t i=0;i<size_t(m)*n;++i)require(actual[i]==uint16_t(0x4208),"actual Weights oracle lost FP32 factor bits or ordered BF16 boundary");
            const auto* rank=static_cast<const float*>(expected_low.storage.contents());
            for(uint32_t r=0;r<m;++r)require(rank[size_t(r)*128]==1.0001220703125f,"FP32 rank oracle narrowed the precision trap");}
        verify();mx::clear_cache();
    }
    void verify() const{for(const auto& g:guards)require(digest(g.view)==g.hash,"immutable generated source/oracle changed");lease->revalidate_after_drain();}
};
size_t allocation_bytes(uint32_t rows,uint32_t columns,size_t item){return(size_t(rows)+2)*(size_t(columns)+8)*item;}
size_t capacity(size_t bytes){return(bytes+16383)&~size_t(16383);}
struct Trial {
    std::shared_ptr<Sources> source;
    std::unique_ptr<tc::MemoryExecutionContext> context;
    std::unique_ptr<mg::MemoryBridge> bridge;
    std::unique_ptr<mg::Backend> backend;
    std::array<std::optional<mg::NativeLoRA>,2> ops;
    std::array<mg::MatrixView,2> output,low;
    std::optional<mg::Ticket> pending;
    bool attempted=false;
    uint64_t baseline_bytes=0,planned_upper_bytes=0,native_capacity_upper_bytes=0;
    std::array<double,2> prepare_seconds{};
    explicit Trial(std::shared_ptr<Sources> s,uint64_t generation):source(std::move(s)) {
        const auto baseline=tc::observe_process_memory();require(baseline.available,"process memory baseline unavailable");baseline_bytes=baseline.process_footprint_bytes;
        tc::MemoryManifest manifest;manifest.candidate_id="generated-lora-sgmma-probe-v1";
        manifest.checkpoint_digest=std::string(source->lease->artifact_digest());manifest.backend_revision="native-lora-correction-probe-v1";manifest.runtime_revision=tc::memory_runtime_revision();
        std::vector<mg::AllocationSiteBinding> bindings;
        for(const auto& label:{std::string("lora-probe-output"),std::string("lora-probe-low")}){
            const bool rank=label.ends_with("low");const size_t bytes=allocation_bytes(source->rows,rank?128:source->n,rank?4:2);
            manifest.sites.push_back({label,"generated-component","probe",tc::MemoryClass::Activation,tc::AllocationLifetime::Request,tc::UpperProvenance::ExactShapeFormula});
            bindings.push_back({label,label,tc::MemoryClass::Activation});
            native_capacity_upper_bytes+=2*capacity(bytes);
            for(uint32_t i=0;i<2;++i)manifest.instances.push_back({label,i,capacity(bytes),0,1,0,"physical-drain-and-release"});
        }
        constexpr uint64_t framework=1ULL<<30,reserve=4ULL<<30;
        require(baseline.physical_memory_bytes>reserve,"insufficient physical memory");
        const auto budget=baseline.physical_memory_bytes-reserve;
        auto compiled=tc::compile_memory_plan(manifest,baseline.process_footprint_bytes,framework,budget);
        planned_upper_bytes=compiled.planned_peak_bytes;
        tc::EffectiveMemoryPolicy policy;policy.enabled=true;policy.effective_budget_bytes=budget;
        policy.planned_upper_bytes=compiled.planned_peak_bytes;policy.framework_upper_bytes=framework;policy.system_reserve_bytes=reserve;
        context=std::make_unique<tc::MemoryExecutionContext>(policy,std::move(compiled),baseline);context->begin_running();
        bridge=std::make_unique<mg::MemoryBridge>(*context,0x4c4f524153474d4dULL,generation,std::move(bindings));
        mg::BackendOptions options;options.allocation=bridge->hooks();backend=std::make_unique<mg::Backend>(std::move(options));
        for(size_t i=0;i<2;++i){auto start=Clock::now();ops[i]=mg::NativeLoRA::prepare(*backend,{source->k,source->n,128},i?mg::native_lora::KernelVariant::SgMmaR128:mg::native_lora::KernelVariant::Scalar);prepare_seconds[i]=seconds(start);
            output[i]=allocate(source->n,mg::ScalarType::BF16,"lora-probe-output");low[i]=allocate(128,mg::ScalarType::FP32,"lora-probe-low");}
        context->checkpoint("probe_prepared");
    }
    mg::MatrixView allocate(uint32_t cols,mg::ScalarType type,const char* label){const auto item=mg::scalar_bytes(type),pitch=(size_t(cols)+8)*item;
        auto backing=backend->allocate(allocation_bytes(source->rows,cols,item),label);require(backing.contents(),"probe requires Shared allocations");
        std::memset(backing.contents(),0x5a,backing.size_bytes());return{backing.view(pitch,size_t(source->rows)*pitch,type),source->rows,cols,pitch};}
    struct Timing {double graph_wall=0,submit_wall=0,gpu_span=0;size_t dispatches=0;};
    Timing execute(size_t arm){const auto start=Clock::now();mg::CommandGraph graph;
        ops[arm]->encode_ordered_corrections(graph,source->x,source->terms,source->y,output[arm],low[arm]);
        Timing t;t.dispatches=graph.steps().size();const auto submit_start=Clock::now();attempted=true;
        pending.emplace(backend->submit(std::move(graph)));const auto status=pending->wait(std::chrono::seconds(30));
        t.submit_wall=seconds(submit_start);t.graph_wall=seconds(start);t.gpu_span=status.gpu_seconds;
        require(status.ok&&status.physically_complete&&!status.timed_out&&status.submitted_buffers==status.completed_buffers,"native correction failed physical completion: "+status.error);
        attempted=false;return t;}
    static void padding(const mg::MatrixView& v){const auto item=mg::scalar_bytes(v.storage.scalar_type());const auto* p=static_cast<const unsigned char*>(v.storage.buffer().contents());require(p,"padding needs completed Shared buffer");
        for(size_t i=0;i<v.storage.buffer().size_bytes();++i){const bool logical=i>=v.storage.offset_bytes()&&i<v.storage.offset_bytes()+size_t(v.rows)*v.row_stride_bytes&&(i-v.storage.offset_bytes())%v.row_stride_bytes<size_t(v.columns)*item;
            if(!logical)require(p[i]==0x5a,"native correction changed padding/canary");}}
    void release(){pending.reset();output={};low={};ops={};backend.reset();}
    void finish(){require(!attempted,"cannot finish uncertain physical work");release();bridge->close();context->checkpoint("probe_released");context->enter_epoch(1,"probe_physical_drain_and_release");context->begin_draining();context->finish_success();
        const auto report=context->take_report();const auto& m=report.metrics;require(m.execution_state=="succeeded"&&!m.tainted&&!m.worker_quarantined&&m.ledger_storage_count==0&&m.ledger_storage_bytes==0&&m.ledger_reserved_bytes==0&&m.ledger_pending_release_bytes==0&&m.ledger_unknown_bytes==0&&m.ledger_cached_bytes==0,"probe phase did not finish with zero ledger");}
};
void run_case(Log& log,const std::filesystem::path& directory,uint32_t rows,uint32_t k,uint32_t n,bool trap,uint64_t generation){
    const auto setup=Clock::now();auto sources=std::make_shared<Sources>(directory,rows,k,n,trap);const double source_seconds=seconds(setup);
    auto trial=std::make_unique<Trial>(sources,generation);
    try {
        std::ostringstream prep;prep<<std::setprecision(12)<<"{\"event\":\"prepare\",\"case\":"<<quote(directory.filename().string())<<",\"M\":"<<rows<<",\"K\":"<<k<<",\"N\":"<<n
            <<",\"source_and_oracle_seconds\":"<<source_seconds<<",\"scalar_pso_seconds\":"<<trial->prepare_seconds[0]<<",\"sgmma_pso_seconds\":"<<trial->prepare_seconds[1]
            <<",\"scalar_recipe\":"<<quote(trial->ops[0]->actual_precision_recipe())<<",\"sgmma_recipe\":"<<quote(trial->ops[1]->actual_precision_recipe())
            <<",\"baseline_process_bytes\":"<<trial->baseline_bytes<<",\"planned_upper_bytes\":"<<trial->planned_upper_bytes<<",\"native_capacity_upper_bytes\":"<<trial->native_capacity_upper_bytes
            <<",\"artifact_sha256\":"<<quote(sources->lease->artifact_digest())<<",\"baseline_includes_immutable_sources\":true,\"anonymous_import_claims\":false,\"native_backings\":4,\"framework_guard_bytes\":1073741824}";log.emit(prep.str());
        for(int round=0;round<(trap?1:3);++round){const bool ab=round<2;std::array<Trial::Timing,2> times;
            for(size_t arm:ab?std::array<size_t,2>{0,1}:std::array<size_t,2>{1,0})times[arm]=trial->execute(arm);
            // Both independent retained outputs survive the pair. No CPU
            // validation/source scan is inserted between A and B submissions.
            const auto comparison_start=Clock::now();std::array<lp::Comparison,2> lows,outputs;bool pass=true;
            for(size_t arm=0;arm<2;++arm){lows[arm]=lp::compare(cpu(trial->low[arm]),cpu(sources->expected_low));outputs[arm]=lp::compare(cpu(trial->output[arm]),cpu(sources->expected));pass=pass&&lp::accepted(lows[arm],outputs[arm],trap);}
            const double comparison_seconds=seconds(comparison_start);const auto integrity=Clock::now();sources->verify();
            for(size_t arm=0;arm<2;++arm){Trial::padding(trial->low[arm]);Trial::padding(trial->output[arm]);}
            const double integrity_seconds=seconds(integrity);trial->context->checkpoint("probe_pair_complete");
            for(size_t arm=0;arm<2;++arm){std::ostringstream row;row<<std::setprecision(12)<<"{\"event\":\"sample\",\"case\":"<<quote(directory.filename().string())<<",\"route\":"<<quote(arm?"sgmma":"scalar")
                <<",\"round\":"<<round<<",\"order\":"<<quote(ab?"AB":"BA")<<",\"scope\":"<<quote(trap?"correctness-only":round?"record-single-bf16-r128-correction":"warm-single-bf16-r128-correction")
                <<",\"graph_encode_submit_physical_seconds\":"<<times[arm].graph_wall<<",\"submit_to_physical_seconds\":"<<times[arm].submit_wall<<",\"ticket_gpu_span_seconds\":"<<times[arm].gpu_span
                <<",\"dispatches\":"<<times[arm].dispatches<<",\"low_rrms\":"<<lows[arm].rrms<<",\"low_max_abs\":"<<lows[arm].max_abs<<",\"low_bit_differences\":"<<lows[arm].bit_differences
                <<",\"output_rrms\":"<<outputs[arm].rrms<<",\"output_max_abs\":"<<outputs[arm].max_abs<<",\"output_bit_differences\":"<<outputs[arm].bit_differences
                <<",\"pair_comparison_cpu_seconds\":"<<comparison_seconds<<",\"pair_integrity_cpu_seconds\":"<<integrity_seconds<<",\"accepted\":"<<(pass?"true":"false")<<'}';log.emit(row.str());}
            require(pass,"fixed low/output numerical gate failed; do not continue or relax tolerance");
        }
        sources->verify();trial->finish();log.emit("{\"event\":\"phase_complete\",\"case\":"+quote(directory.filename().string())+",\"zero_ledger\":true,\"all_work_physically_complete\":true,\"source_padding_unchanged\":true}");
    }catch(...){bool drained=!trial->attempted;
        if(!drained&&trial->pending){try{drained=trial->pending->wait(std::chrono::seconds(30)).physically_complete;}catch(...){}}
        if(drained){trial->release();trial->bridge->close();const auto d=trial->context->finalize_failure("generated correction probe failed");
            if(d==tc::MemoryFailureDisposition::NeedsGpuDrain)(void)trial->context->complete_failure_cleanup(true,{});
        }else{trial->bridge->close();(void)trial.release();log.emit("{\"event\":\"quarantined\",\"pending_physical_work\":true,\"retry\":false}");}
        throw;
    }
}
} // namespace
int main(int argc,char** argv){std::unique_ptr<Log> log;try{
    require(argc==3||argc==5,"usage: native-lora-sgmma-component --device /absolute/fresh-output [--rows 1..1024]");
    require(std::string_view(argv[1])=="--device","device execution requires explicit --device");uint32_t rows=1024;
    if(argc==5){require(std::string_view(argv[3])=="--rows","only one explicit row count is supported");size_t used=0;const auto parsed=std::stoul(argv[4],&used);require(used==std::strlen(argv[4])&&parsed>0&&parsed<=1024,"rows outside bounded single-shape probe");rows=uint32_t(parsed);}
    char* raw=tc_runtime_build_identity();require(raw,"runtime build identity missing");const std::string build(raw);tc_string_free(raw);
    require(build==TC_LORA_PROBE_EXPECTED_BUILD_ID,"loaded native build differs from explicit link identity");
    const auto dir=std::filesystem::path(argv[2]);require(dir.is_absolute()&&dir!=dir.root_path()&&!std::filesystem::exists(dir)&&!std::filesystem::is_symlink(std::filesystem::symlink_status(dir)),"probe output must be fresh, absolute and owned");
    require(std::filesystem::create_directory(dir),"cannot create probe output directory");log=std::make_unique<Log>(dir/"receipt.jsonl");
    const auto binary=std::filesystem::canonical(argv[0]);
    log->emit("{\"event\":\"identity\",\"library_build_id\":"+quote(build)+",\"tool_source_sha256\":"+quote(TC_LORA_PROBE_SOURCE_SHA)+",\"cpu_contract_source_sha256\":"+quote(TC_LORA_PROBE_CONTRACT_SHA)+",\"binary_sha256\":"+quote(tc::sha256_file(binary))+",\"scope\":\"generated LoRA correction only; supplied shared base; no dense GEMM/full FFN\",\"performance_qualified\":false,\"planned_native_submissions\":14,\"gpu_span_scope\":\"first GPU start to last end, not busy time or overlap proof\"}");
    tc::configure_streams();
    run_case(*log,dir/"gate-or-up",rows,4096,12288,false,1);
    run_case(*log,dir/"down",rows,12288,4096,false,2);
    run_case(*log,dir/"fp32-ordered-trap",8,128,64,true,3);
    log->emit("{\"event\":\"complete\",\"accepted\":true,\"native_submissions\":14,\"model_loads\":0,\"ane_calls\":0,\"e2e_images\":0,\"performance_qualified\":false}");return 0;
}catch(const std::exception& e){if(log)log->emit("{\"event\":\"failure\",\"accepted\":false,\"reason\":"+quote(e.what())+'}');std::cerr<<e.what()<<'\n';return 1;}}
