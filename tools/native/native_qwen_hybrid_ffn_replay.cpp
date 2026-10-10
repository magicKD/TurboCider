// Bounded, read-only layer-0 replay; compile as Objective-C++ for strict JSON.
// Native dispatches only after explicit device mode. --cpu-only creates no device.
// No complete-model eval/export; historical receipts/binaries are never replaced.
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/backends/mlx.hpp"
#include "../../native/backends/mlx_fd_reader.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"
#include "../../native/backends/metal_graph/native_hybrid_ffn.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/runtime/streaming/canonical_encoding.hpp"
#include "../../native/runtime/memory_manifest.hpp"
#include "turbocider/turbocider.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <sys/stat.h>
#include <unistd.h>

#ifndef TC_NATIVE_REPLAY_SOURCE_SHA
#define TC_NATIVE_REPLAY_SOURCE_SHA "unrecorded-host-only"
#endif
#ifndef TC_NATIVE_REPLAY_EXPECTED_BUILD_ID
#define TC_NATIVE_REPLAY_EXPECTED_BUILD_ID "unbound-host-only"
#endif
using namespace tc;
namespace mg = tc::metal_graph;
namespace ag = tc::ane::gpu;
namespace {
constexpr uint32_t rows=1024, hidden=4096, width=12288, channels=6144;
constexpr uint64_t native_limit=512ULL<<20, source_weight_bytes=3ULL*hidden*width*2;
constexpr double rrms_limit=0.08;
constexpr const char* capture_id="75956-1226003966809291";
using Time = std::chrono::steady_clock;
bool same_owner(const std::shared_ptr<const void>& a,const std::shared_ptr<const void>& b) {
    return a && b && !a.owner_before(b) && !b.owner_before(a);
}
constexpr bool crossover_ab(int round) { return round<2; } // warm AB, record AB, record BA
std::string verified_runtime_build() {
    char* raw=tc_runtime_build_identity();require(raw,"runtime build identity unavailable");
    const std::string actual(raw);tc_string_free(raw);
    require(actual==TC_NATIVE_REPLAY_EXPECTED_BUILD_ID,"loaded library differs from the replay's explicit link build");
    return actual;
}
double elapsed(Time::time_point start) { return std::chrono::duration<double>(Time::now()-start).count(); }
NSString* ns(std::string_view s) { return [[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding]; }

std::string read_fd(int fd,uint64_t bytes,uint64_t limit) {
    require(bytes && bytes<=limit,"replay metadata file exceeds bound");
    std::string result(size_t(bytes),'\0');
    size_t done=0;
    while (done<result.size()) {
        const auto n=::pread(fd,result.data()+done,result.size()-done,off_t(done));
        if (n<0 && errno==EINTR) continue;
        require(n>0,"replay held-file short read"); done+=size_t(n);
    }
    require(result.find('\0')==std::string::npos,"replay JSON contains NUL");
    return result;
}
std::string held_fingerprint(const streaming::SourceLease& lease) {
    const auto& file=lease.file("checkpoint");
    auto fd=lease.duplicate_fd("checkpoint"); struct stat info{};
    require(::fstat(fd.get(),&info)==0 && S_ISREG(info.st_mode),"checkpoint held stat unavailable");
    const auto name=file.canonical_target_path.string();
    const auto identity=":"+std::to_string(name.size())+":"+name+":"+std::to_string(info.st_dev)+":"+
        std::to_string(info.st_ino)+":"+std::to_string(info.st_size)+":"+
        std::to_string(info.st_mtimespec.tv_sec)+":"+std::to_string(info.st_mtimespec.tv_nsec)+":"+
        std::to_string(info.st_ctimespec.tv_sec)+":"+std::to_string(info.st_ctimespec.tv_nsec);
    streaming::CanonicalEncoder encoder("tc-ane-calibration-source-v1");
    encoder.string_field("identity",identity); return encoder.sha256();
}
uint64_t integer(NSDictionary* object,NSString* key) {
    id value=object[key];
    require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),
            "capture receipt requires integer field");
    const double number=[value doubleValue];
    require(std::isfinite(number) && number>=0 && number==std::floor(number) && number<1e12,
            "invalid capture receipt integer");
    return uint64_t(number);
}
struct Sample {
    int step=0;
    std::string logical_id,digest;
    Tensor input;
    std::shared_ptr<MlxLeaseFdReader> reader;
};
struct Claim {
    std::optional<MemoryReservation> reservation;
    std::optional<StorageLease> lease;
    uint64_t generation=0;
    size_t upper=0;
    bool surface=false;
};
// The existing ledger, shared by every newly allocated Metal/IOSurface backing.
// Produced immutable MLX source imports remain explicitly in the baseline;
// their anonymous allocator owners are NOT fabricated accounting claims.
struct Accounting {
    MemoryLedger ledger{native_limit};
    uint64_t next=1;
    size_t surface_bytes=0,surfaces=0,native_allocations=0;
    std::map<void*,std::weak_ptr<Claim>> active;
    std::shared_ptr<void> reserve(size_t upper,std::string_view label,bool surface) {
        auto reservation=ledger.try_reserve(MemoryClass::Activation,upper,std::string(label));
        require(bool(reservation),"native replay backings exceed 512 MiB ledger");
        auto claim=std::make_shared<Claim>(); claim->reservation.emplace(std::move(*reservation));
        claim->generation=next++; claim->upper=upper; claim->surface=surface; return claim;
    }
    void commit(const std::shared_ptr<void>& token,void* handle,size_t bytes,bool surface) {
        auto claim=std::static_pointer_cast<Claim>(token);
        require(claim && claim->surface==surface && claim->reservation && !claim->lease && handle &&
            bytes && bytes<=claim->upper,"invalid replay backing accounting commit");
        require(!active.count(handle) || active.at(handle).expired(),"duplicate replay backing charge");
        claim->lease.emplace(claim->reservation->commit(
            {0x5148573146464eULL,uint64_t(reinterpret_cast<uintptr_t>(handle)),bytes,claim->generation}));
        claim->reservation.reset(); active[handle]=claim;
        if (surface) {++surfaces;surface_bytes+=bytes;} else ++native_allocations;
    }
};
mg::AllocationHooks hooks(const std::shared_ptr<Accounting>& account,bool surface) {
    return {[account,surface](size_t upper,std::string_view label){return account->reserve(upper,label,surface);},
        [account,surface](const std::shared_ptr<void>& token,void* handle,size_t bytes){account->commit(token,handle,bytes,surface);}};
}
mg::MatrixView import(mg::Backend& backend,const Tensor& tensor) {
    const auto view=mg::import_mlx_matrix(backend,tensor);
    return {view.storage,view.rows,view.columns,view.row_stride_bytes};
}
mg::MatrixView allocate(mg::Backend& backend,uint32_t m,uint32_t n,mg::ScalarType type,const char* label) {
    const size_t pitch=size_t(n)*mg::scalar_bytes(type);
    auto buffer=backend.allocate(size_t(m)*pitch,label,mg::StorageMode::Shared);
    return {buffer.view(0,size_t(m)*pitch,type),m,n,pitch};
}
std::string hash_matrix(const mg::MatrixView& value) {
    require(value.storage.contents() && value.row_stride_bytes==size_t(value.columns)*mg::scalar_bytes(value.storage.scalar_type()),
            "replay source must be CPU-visible compact rows");
    return memory_sha256_hex({static_cast<const char*>(value.storage.contents()),size_t(value.rows)*value.row_stride_bytes});
}
// Pure CPU view contract. No Metal access, tensor evaluation or per-element
// diagnostic string construction in the comparison loop. The caller has
// already collected successful physical completion before constructing views.
struct CpuBf16View {
    const void* data=nullptr;
    size_t rows=0,columns=0,row_stride_bytes=0,extent_bytes=0;
    mg::ScalarType type=mg::ScalarType::Bytes;
};
struct Comparison {
    double rrms=0,maximum=0,peak_relative=0,cosine=0;
    size_t finite_elements=0;
    bool bitwise_equal=true;
};
size_t validate_cpu_view(const CpuBf16View& value) {
    require(value.type==mg::ScalarType::BF16 && value.data &&
        reinterpret_cast<uintptr_t>(value.data)%alignof(uint16_t)==0,"replay output unavailable or unaligned");
    require(value.rows && value.columns && value.columns<=SIZE_MAX/sizeof(uint16_t) &&
        value.row_stride_bytes==value.columns*sizeof(uint16_t),"replay comparison requires compact BF16 rows");
    require(value.rows<=SIZE_MAX/value.row_stride_bytes &&
        value.rows*value.row_stride_bytes<=value.extent_bytes,"replay comparison escapes its view extent");
    return value.rows*value.columns;
}
Comparison compare_cpu(const CpuBf16View& expected,const CpuBf16View& actual) {
    const auto count=validate_cpu_view(expected);(void)validate_cpu_view(actual);
    require(expected.rows==actual.rows && expected.columns==actual.columns,"replay comparison shape mismatch");
    const auto* reference=static_cast<const uint16_t*>(expected.data);
    const auto* candidate=static_cast<const uint16_t*>(actual.data);
    Comparison result;result.finite_elements=count;
    double error=0,energy=0,actual_energy=0,dot=0,max_abs=0,peak=0;
    for (size_t i=0;i<count;++i) {
        const auto a_bits=reference[i],b_bits=candidate[i];
        const double a=std::bit_cast<float>(uint32_t(a_bits)<<16),b=std::bit_cast<float>(uint32_t(b_bits)<<16);
        if(!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite dense/hybrid FFN output");
        result.bitwise_equal&=a_bits==b_bits; // Includes the +0/-0 distinction.
        error+=(a-b)*(a-b);energy+=a*a;actual_energy+=b*b;dot+=a*b;
        max_abs=std::max(max_abs,std::abs(a-b));peak=std::max(peak,std::abs(a));
    }
    require(energy>0 && actual_energy>0,"degenerate FFN output energy");
    result.rrms=std::sqrt(error/energy);result.maximum=max_abs;result.peak_relative=max_abs/peak;
    result.cosine=dot/std::sqrt(energy*actual_energy);
    return result;
}
CpuBf16View cpu_view(const mg::MatrixView& value) {
    return {value.storage.contents(),value.rows,value.columns,value.row_stride_bytes,
        value.storage.size_bytes(),value.storage.scalar_type()};
}
NSDictionary* comparison_dictionary(const Comparison& value) {
    return @{@"rrms":@(value.rrms),@"rrms_limit":@(rrms_limit),@"screening_pass":@(value.rrms<=rrms_limit),
        @"max_abs":@(value.maximum),@"max_abs_over_reference_peak":@(value.peak_relative),
        @"cosine":@(value.cosine),@"finite_elements":@(value.finite_elements),
        @"max_and_cosine_scope":@"diagnostic only; no per-element 8% acceptance rule"};
}
void physical(const mg::NativeHybridFfnStatus& status) {
    const auto& e=status.evidence;
    require(status.ok && !status.retired && !status.running && status.generation && status.all_work_physically_complete &&
        e.handoff.ok && !e.handoff.retired && e.handoff.producer_submitted && e.handoff.producer_physical_complete &&
        e.handoff.ane_submission_attempted && e.handoff.ane_physical_complete && e.handoff.all_work_physically_complete,
        "replay hybrid producer/ANE physical proof unavailable");
    require(e.gpu_attempted && e.consumer_attempted && e.gpu && e.consumer &&
        e.gpu->ok && e.gpu->physically_complete && !e.gpu->timed_out && e.gpu->submitted_buffers &&
        e.gpu->completed_buffers==e.gpu->submitted_buffers && e.consumer->ok &&
        e.consumer->physically_complete && !e.consumer->timed_out && e.consumer->submitted_buffers &&
        e.consumer->completed_buffers==e.consumer->submitted_buffers,
        "replay GPU complement/restore consumer physical proof unavailable");
}
// Retain this entire owner on uncertain completion, including baseline sources.
struct Trial {
    std::shared_ptr<Accounting> account=std::make_shared<Accounting>();
    std::shared_ptr<const streaming::SourceLease> lease;
    Weights weights;
    std::vector<Sample> samples;
    std::unique_ptr<mg::Backend> backend;
    std::optional<mg::NativeHybridFfn> hybrid,cache_off;
    std::optional<mg::Ticket> dense_ticket;
    std::vector<mg::MatrixView> reimported_weights;
    std::vector<mg::MatrixView> sources,owned;
};
void write_exclusive(const std::filesystem::path& path,NSString* value) {
    NSData* data=[value dataUsingEncoding:NSUTF8StringEncoding];
    require(data,"receipt encoding failed");
    streaming::OwnedSourceFd fd(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600));
    require(bool(fd),"refusing to overwrite/unable to create replay receipt");
    size_t done=0; while (done<data.length) {
        const auto n=::write(fd.get(),static_cast<const char*>(data.bytes)+done,data.length-done);
        if (n<0 && errno==EINTR) continue;
        require(n>0,"receipt write failed");done+=size_t(n);
    }
    require(::fsync(fd.get())==0,"receipt fsync failed");
}

void run(const std::filesystem::path& checkpoint,const std::filesystem::path& capture,
         const std::filesystem::path& cache,const std::filesystem::path& output,
         bool register_weight_staging,bool scale_cache_comparison) {
    const auto runtime_build=verified_runtime_build(); // Before model/header reads or any native preparation.
    require(!scale_cache_comparison || register_weight_staging,"scale-cache comparison requires register staging");
    require(checkpoint.is_absolute() && capture.is_absolute() && cache.is_absolute() && output.is_absolute(),
            "replay paths must be explicit absolute paths");
    require(!std::filesystem::exists(output) && std::filesystem::is_directory(output.parent_path()),"receipt must be new in an existing directory");
    require(!std::filesystem::exists(cache) && std::filesystem::is_directory(cache.parent_path()),"ANE cache must be fresh in an existing owned parent");
    require(capture.filename()==std::string(capture_id)+"-replay","unexpected capture trajectory directory");
    // New register-staging measurement isolates order on ONE held-out input.
    // Keep the original --device recipe intact for explicit baseline replay.
    const std::vector<int> capture_steps=register_weight_staging ? std::vector<int>{19} : std::vector<int>{1,19};
    auto trial=std::make_unique<Trial>(); const auto account=trial->account;
    try {
        std::vector<streaming::SourceFileIdentity> files{{.logical_id="checkpoint",.path=checkpoint},
            {.logical_id="complete",.path=capture/"complete.json"}};
        for (int step:capture_steps) {
            const auto path=capture/("layer-0-step-"+std::to_string(step)+".safetensors");
            require(!std::filesystem::is_symlink(path),"capture input symlink is not accepted");
            files.push_back({.logical_id="step"+std::to_string(step),.path=path});
        }
        trial->lease=streaming::SourceLease::capture(std::move(files));
        const auto identity=held_fingerprint(*trial->lease);
        std::map<std::string,std::string> file_digests;
        std::vector<std::string> capture_ids{"complete"};
        for (int step:capture_steps) capture_ids.push_back("step"+std::to_string(step));
        for (const auto& id:capture_ids) {
            const uint64_t limit=id=="complete" ? 2ULL<<20 : 9ULL<<20;
            require(trial->lease->file(id).bytes<=limit,"capture file exceeds replay bound");
            auto fd=trial->lease->duplicate_fd(id);
            file_digests[id]=memory_sha256_fd(fd.get(),trial->lease->file(id).bytes);
        }
        auto fd=trial->lease->duplicate_fd("complete");
        NSDictionary* receipt=parse_json(read_fd(fd.get(),trial->lease->file("complete").bytes,2ULL<<20).c_str());
        require([receipt[@"complete"] isEqual:@YES] && integer(receipt,@"schema_version")==1 &&
            string_value(receipt,@"artifact_type")=="bounded_ffn_input_calibration" &&
            string_value(receipt,@"capture_recipe")=="ffn-input-fp16-regions-outliers-v1","invalid original capture completion");
        NSDictionary* binding=receipt[@"binding"];
        require([binding isKindOfClass:NSDictionary.class] &&
            string_value(binding,@"request_id")==capture_id && string_value(binding,@"model_id")=="qwen-image-2.1" &&
            string_value(binding,@"model_fingerprint")==identity && string_value(binding,@"identity_kind")=="canonical-stat-identity" &&
            string_value(binding,@"actual_execution")=="forced-gpu-runtime-boundary" &&
            integer(binding,@"width")==512 && integer(binding,@"height")==512 && integer(binding,@"total_steps")==20 &&
            integer(binding,@"seed")==42 && integer(binding,@"reference_count")==0 && integer(binding,@"reference_size")==0 &&
            [binding[@"loras"] isKindOfClass:NSArray.class] && [binding[@"loras"] count]==0,
            "capture completion/model provenance mismatch");
        std::string prompt;
        for (int step:capture_steps) {
            const auto id="step"+std::to_string(step);auto input_fd=trial->lease->duplicate_fd(id);
            auto reader=std::make_shared<MlxLeaseFdReader>(MlxOwnedFd(input_fd.release()),id);
            auto [arrays,m]=mx::load_safetensors(reader);
            require(m.at("capture_recipe")=="qwen21-real-ffn-bf16-v1" && m.at("trajectory")=="forced-gpu" &&
                m.at("adapter_identity").empty() && m.at("model_identity")==identity && m.at("total_steps")=="20" &&
                m.at("layer")=="0" && m.at("step")==std::to_string(step) && m.at("seed")=="42",
                "full BF16 capture metadata mismatch");
            if (prompt.empty()) prompt=m.at("prompt");
            require(!prompt.empty() && m.at("prompt")==prompt && arrays.size()==1 &&
                arrays.at("input").shape()==mx::Shape{1,int(rows),int(hidden)} && arrays.at("input").dtype()==mx::bfloat16,
                "capture shape/dtype/trajectory mismatch");
            trial->samples.push_back({step,id,file_digests.at(id),arrays.at("input"),reader});
        }
        trial->lease->revalidate_after_drain();
        const auto load_start=Time::now();
        configure_streams();std::atomic<bool> cancelled{false};
        trial->weights.load_lease(trial->lease,{"checkpoint"},[](const std::string&,int,int){},cancelled);
        const std::string prefix="transformer_blocks.0.img_mlp.";
        auto fused=trial->weights.at(prefix+"gate_up.weight"),down=trial->weights.at(prefix+"out.weight");
        require(fused.shape()==mx::Shape{int(width*2),int(hidden)} && down.shape()==mx::Shape{int(hidden),int(width)} &&
            fused.dtype()==mx::bfloat16 && down.dtype()==mx::bfloat16,"source is not dense BF16 Qwen layer0");
        auto gate=slice_axis(fused,0,0,width),up=slice_axis(fused,0,width,2*width);
        mx::eval({gate,up,down});for (auto& sample:trial->samples) mx::eval(sample.input);
        mx::synchronize();mx::clear_cache();trial->lease->revalidate_after_drain();
        const double source_prepare_seconds=elapsed(load_start);
        const auto baseline=observe_process_memory();const auto mlx_source_active=mx::get_active_memory();
        require(baseline.available,"process baseline observation unavailable");
        // Do not accidentally evaluate the checkpoint's other 31 layers.
        require(mlx_source_active<=source_weight_bytes+(64ULL<<20),"unexpected MLX source allocation growth");
        mg::BackendOptions backend_options;backend_options.allocation=hooks(account,false);
        trial->backend=std::make_unique<mg::Backend>(std::move(backend_options));auto& backend=*trial->backend;
        trial->sources={import(backend,gate),import(backend,up),import(backend,down)};
        for (auto& sample:trial->samples) trial->sources.push_back(import(backend,sample.input));
        const std::array<Tensor,3> weight_arrays{gate,up,down};
        NSDictionary* import_identity=@{@"verified":@NO};
        const auto refresh_imports=[&] {
            // Distinct array/view wrappers must resolve to the ACTUAL shared
            // MLX Data allocation, not to per-import wrapper control blocks.
            trial->reimported_weights.clear();trial->reimported_weights.reserve(3);
            NSMutableArray* weights=[NSMutableArray array];
            for(size_t i=0;i<weight_arrays.size();++i) {
              Tensor again=weight_arrays[i];
              trial->reimported_weights.push_back(import(backend,again));
              const auto& original=trial->sources[i];const auto& repeated=trial->reimported_weights.back();
              const std::shared_ptr<const void> data_owner=weight_arrays[i].data_shared_ptr();
              require(&again!=&weight_arrays[i] && &original!=&repeated &&
                same_owner(original.storage.buffer().allocation_identity(),data_owner) &&
                same_owner(repeated.storage.buffer().allocation_identity(),data_owner) &&
                original.storage.buffer().native_handle()==repeated.storage.buffer().native_handle() &&
                original.storage.offset_bytes()==size_t(weight_arrays[i].offset()) &&
                repeated.storage.offset_bytes()==size_t(again.offset()) &&
                original.storage.offset_bytes()==repeated.storage.offset_bytes() &&
                original.storage.size_bytes()==repeated.storage.size_bytes() &&
                original.row_stride_bytes==repeated.row_stride_bytes,
                "re-import did not preserve actual MLX Data ownership and view offsets");
              [weights addObject:@{@"weight":i==0?@"gate":(i==1?@"up":@"down"),
                @"owner_equals_actual_mlx_data":@YES,@"same_native_buffer":@YES,
                @"original_offset_bytes":@(original.storage.offset_bytes()),
                @"reimported_offset_bytes":@(repeated.storage.offset_bytes()),
                @"extent_bytes":@(original.storage.size_bytes())}];
            }
            return @{@"verified":@YES,@"fresh_imports_per_record_round":@3,@"distinct_array_and_view_wrappers":@YES,
                @"weights":weights,@"gate_up_share_actual_mlx_data":@(same_owner(gate.data_shared_ptr(),up.data_shared_ptr())),
                @"gate_offset_bytes":@(trial->sources[0].storage.offset_bytes()),
                @"up_offset_bytes":@(trial->sources[1].storage.offset_bytes()),
                @"scope":@"allocation lifetime identity only; immutable source separately held and checked"};
        };
        if(scale_cache_comparison) import_identity=refresh_imports();
        std::vector<std::string> source_hashes;
        for (const auto& source:trial->sources) source_hashes.push_back(hash_matrix(source));
        auto verify_sources=[&] {
            trial->lease->revalidate_after_drain();
            require(held_fingerprint(*trial->lease)==identity,"held model generation changed");
            for (size_t i=0;i<trial->sources.size();++i)
                require(hash_matrix(trial->sources[i])==source_hashes[i],"native execution mutated immutable source");
            for (const auto& [id,digest]:file_digests) {
                auto file=trial->lease->duplicate_fd(id);
                require(memory_sha256_fd(file.get(),trial->lease->file(id).bytes)==digest,"capture bytes changed");
            }
            trial->lease->revalidate_after_drain();
        };
        auto& v=trial->owned;
        v={allocate(backend,rows,width,mg::ScalarType::BF16,"replay-dense-hidden"),
           allocate(backend,rows,hidden,mg::ScalarType::BF16,"replay-dense-output"),
           allocate(backend,rows,hidden,mg::ScalarType::BF16,"replay-hybrid-output"),
           allocate(backend,rows,hidden,mg::ScalarType::FP32,"replay-gpu-partial"),
           allocate(backend,rows,hidden,mg::ScalarType::FP32,"replay-ane-partial")};
        auto start=Time::now();std::optional<mg::NativeFfn> dense;
        if(!scale_cache_comparison) dense.emplace(mg::NativeFfn::prepare(backend,{hidden,width,width}));
        const double dense_prepare_seconds=scale_cache_comparison?0:elapsed(start);
        const auto sh=hooks(account,true);start=Time::now();
        mg::NativeHybridFfnOptions options;
        options.rows=rows;options.hidden=hidden;options.width=width;options.ane_channels=channels;
        options.program_cache=scale_cache_comparison?cache/"on":cache;options.surface_allocation={sh.reserve,sh.allocated};
        options.register_weight_staging=register_weight_staging;
        double cache_off_prepare_seconds=0;
        if(scale_cache_comparison) {
            auto off=options;off.program_cache=cache/"off";
            trial->cache_off.emplace(mg::NativeHybridFfn::prepare(backend,std::move(off)));
            cache_off_prepare_seconds=elapsed(start);
            require(trial->cache_off->weight_scale_cache_capacity()==0 && trial->cache_off->weight_scale_cache_payload_bytes()==0,
                "OFF controller unexpectedly enabled scale cache");
            options.weight_scale_cache_entries=1;
            options.weight_scale_source_revision=std::string(trial->lease->digest());
            options.weight_scale_source_generation=trial->lease->generation();
            start=Time::now();
        }
        trial->hybrid.emplace(mg::NativeHybridFfn::prepare(backend,std::move(options)));
        const double hybrid_prepare_seconds=elapsed(start);auto& hybrid=*trial->hybrid;
        if(scale_cache_comparison) require(hybrid.weight_scale_cache_capacity()==1 &&
            hybrid.weight_scale_cache_payload_bytes()==size_t(2*channels+hidden)*2,"ON scale-cache payload/capacity mismatch");
        require(account->surfaces==(scale_cache_comparison?18U:9U) &&
            hybrid.surface_bytes()+(trial->cache_off?trial->cache_off->surface_bytes():0)==account->surface_bytes,
            "Surface accounting coverage mismatch");
        NSMutableArray* records=[NSMutableArray array];bool numeric_pass=true,all_bitwise_equal=true;
        uint32_t cache_hits=0,cache_misses=0;
        for (size_t index=0;index<trial->samples.size();++index) {
            const auto input=trial->sources[3+index];
            const auto dense_arm=[&] {
                const auto began=Time::now();mg::CommandGraph graph;
                require(bool(dense),"dense arm is forbidden in scale-cache-only comparison");
                dense->encode(graph,{input,trial->sources[0],trial->sources[1],trial->sources[2],v[0],v[1],0});
                trial->dense_ticket.emplace(backend.submit(std::move(graph)));
                const auto done=trial->dense_ticket->wait(std::chrono::seconds(30));
                require(done.ok && done.physically_complete && !done.timed_out && done.submitted_buffers &&
                    done.completed_buffers==done.submitted_buffers,"dense FFN did not physically complete");
                return @{@"wall_seconds":@(elapsed(began)),@"gpu_span_seconds":@(done.gpu_seconds),
                    @"submitted_buffers":@(done.submitted_buffers),@"physical_complete":@YES};
            };
            const auto hybrid_arm=[&](mg::NativeHybridFfn& controller,bool cache_enabled,int round) {
                const auto began=Time::now();mg::NativeHybridFfnPlan p;
                const auto& weights=cache_enabled && round>0?trial->reimported_weights:trial->sources;
                p.input=input;p.gate=weights[0];p.up=weights[1];p.down=weights[2];
                p.gpu_partial=v[3];p.ane_partial=v[4];p.published_output=scale_cache_comparison && !cache_enabled?v[1]:v[2];
                if(cache_enabled) p.weight_scale_cache_slot=0;
                auto half_hidden=v[0];half_hidden.columns=channels;
                controller.gpu_complement().encode(p.gpu_ffn,{p.input,p.gate,p.up,p.down,half_hidden,p.gpu_partial,0});
                controller.gpu_complement().encode_join_partials(p.after_join,p.gpu_partial,p.ane_partial,p.published_output);
                const auto result=controller.execute(std::move(p),std::chrono::seconds(30));
                physical(result.completion);physical(controller.status());
                require(result.weight_scale_cache_hit==(cache_enabled && round>0) &&
                    result.weight_scale_cache_miss==(cache_enabled && round==0),"unexpected actual scale-cache hit/miss");
                if(cache_enabled) {cache_hits+=result.weight_scale_cache_hit;cache_misses+=result.weight_scale_cache_miss;}
                require(std::isfinite(result.producer_gpu_span_seconds) && result.producer_gpu_span_seconds>=0 &&
                    std::isfinite(result.gpu_complement_span_seconds) && result.gpu_complement_span_seconds>=0,
                    "invalid device timing diagnostic");
                return @{@"wall_seconds":@(elapsed(began)),@"controller_seconds":@(result.total_seconds),
                    @"producer_seconds":@(result.producer_seconds),@"submit_ane_seconds":@(result.submit_ane_seconds),
                    @"branch_join_seconds":@(result.branch_join_seconds),@"consumer_seconds":@(result.consumer_seconds),
                    @"producer_gpu_span_seconds":@(result.producer_gpu_span_seconds),
                    @"gpu_complement_span_seconds":@(result.gpu_complement_span_seconds),
                    @"weight_scale_cache_hit":@(result.weight_scale_cache_hit),
                    @"weight_scale_cache_miss":@(result.weight_scale_cache_miss),
                    @"device_spans_available":@(result.producer_gpu_span_seconds>0 && result.gpu_complement_span_seconds>0),
                    @"physical_complete":@YES,@"generation":@(result.completion.generation)};
            };
            const int rounds=register_weight_staging ? 3 : 2;
            for (int round=0;round<rounds;++round) {
                if(scale_cache_comparison && round>0) import_identity=refresh_imports();
                NSDictionary *a=nil,*b=nil;
                const bool ab=register_weight_staging ? crossover_ab(round) : index==0;
                const auto arm_a=[&] {return scale_cache_comparison?hybrid_arm(*trial->cache_off,false,round):dense_arm();};
                const auto arm_b=[&] {return hybrid_arm(hybrid,scale_cache_comparison,round);};
                if (ab) {a=arm_a();b=arm_b();} else {b=arm_b();a=arm_a();}
                const auto comparison_start=Time::now();
                const auto comparison=compare_cpu(cpu_view(v[1]),cpu_view(v[2]));
                const double comparison_cpu_seconds=elapsed(comparison_start);
                NSDictionary* error=comparison_dictionary(comparison);numeric_pass&=comparison.rrms<=rrms_limit;
                const bool bitwise=comparison.bitwise_equal;
                all_bitwise_equal&=bitwise;
                if(scale_cache_comparison) numeric_pass&=bitwise;
                [records addObject:@{@"step":@(trial->samples[index].step),@"layer":@0,
                    @"kind":round==0 ? @"warmup" : @"record",@"round":@(round),@"order":ab ? @"AB" : @"BA",
                    @"input_sha256":ns(trial->samples[index].digest),
                    (scale_cache_comparison?@"cache_off_w1s":@"dense_gpu"):a,
                    (scale_cache_comparison?@"cache_on_w1s":@"hybrid_w1s"):b,
                    @"bf16_bitwise_equal":@(bitwise),@"comparison_cpu_seconds":@(comparison_cpu_seconds),@"error":error}];
                if (!numeric_pass) break; // Preserve failure evidence, do not add device runs.
            }
            if (!numeric_pass) break;
        }
        physical(hybrid.drain(std::chrono::milliseconds(0)));
        if(trial->cache_off) physical(trial->cache_off->drain(std::chrono::milliseconds(0)));
        if(scale_cache_comparison && numeric_pass) require(cache_hits==2 && cache_misses==1,"cache crossover did not complete two actual hits after one miss");
        verify_sources();
        const auto peak=observe_process_memory();
        const auto snapshot=account->ledger.snapshot();
        require(!snapshot.reservation_count && !snapshot.unknown_bytes && snapshot.storage_bytes<=native_limit,
                "native replay ledger is not fully committed/bounded");
        const auto device=backend.device_name(),program=hybrid.program_key();
        if(trial->cache_off) require(trial->cache_off->program_key()==program,"OFF/ON ANE program identity changed");
        const auto source_digest=std::string(trial->lease->digest());
        const auto source_generation=trial->lease->generation();
        const auto cache_payload=hybrid.weight_scale_cache_payload_bytes();
        NSMutableArray* input_digests=[NSMutableArray array];
        NSMutableDictionary* capture_digests=[NSMutableDictionary dictionary];
        for (size_t i=3;i<source_hashes.size();++i) [input_digests addObject:ns(source_hashes[i])];
        capture_digests[@"complete.json"]=ns(file_digests.at("complete"));
        for (int step:capture_steps)
            capture_digests[ns("layer-0-step-"+std::to_string(step)+".safetensors")]=ns(file_digests.at("step"+std::to_string(step)));
        // All operations are physically complete. Release every graph/view
        // owner before checking the same ledger, then release baseline sources.
        trial->dense_ticket.reset();trial.reset();
        const auto released=account->ledger.snapshot();
        require(!released.storage_count && !released.reservation_count && !released.pending_release_count && !released.unknown_bytes,
                "native replay ledger did not return to zero");
        NSDictionary* report=@{@"schema":@"native-qwen-hybrid-ffn-replay-v3",@"status":numeric_pass?@"screening-pass":@"screening-rejected",
            @"experiment":scale_cache_comparison?@"register-w1s-scale-cache-off-on-step19-crossover":
                (register_weight_staging ? @"register-weight-staging-step19-crossover" : @"baseline-step1-step19"),
            @"arm_a":scale_cache_comparison?@"register_w1s_cache_off":@"dense_native_gpu",
            @"arm_b":scale_cache_comparison?@"register_w1s_cache_on":@"hybrid_w1s",
            @"scale_cache_comparison":@(scale_cache_comparison),@"cache_on_hits":@(cache_hits),@"cache_on_misses":@(cache_misses),
            @"cache_on_payload_bytes":@(cache_payload),@"all_pairs_bf16_bitwise_equal":@(all_bitwise_equal),
            @"cache_configuration_extra_copy_pso":@(scale_cache_comparison?1:0),
            @"cache_configuration_scope":@"one additional prepared scale-copy pipeline and compact FP16 scales; not a measured compiler/driver memory bound",
            @"mlx_reimport_identity":import_identity,@"source_lease_generation":@(source_generation),
            @"register_weight_staging":@(register_weight_staging),
            @"scope":@"BaseOnly layer0 complete FFN component; not DiT/E2E/quality or performance qualification",
            @"control_plane":@"synchronous_host_admission",@"rows":@(rows),@"hidden":@(hidden),@"width":@(width),@"ane_channels":@(channels),
            @"capture_id":@(capture_id),@"model_stat_fingerprint":ns(identity),@"source_lease_digest":ns(source_digest),
            @"source_proof_scope":@"held descriptors + canonical stat generation; not whole-checkpoint content certification",
            @"source_weight_bytes":@(source_weight_bytes),@"source_pre_and_post_sha256":source_hashes.empty()?@[]:@[ns(source_hashes[0]),ns(source_hashes[1]),ns(source_hashes[2])],
            @"source_digest_order":@[@"layer0.gate BF16",@"layer0.up BF16",@"layer0.down BF16"],
            @"input_tensor_pre_and_post_sha256":input_digests,
            @"capture_file_pre_and_post_sha256":capture_digests,
            @"source_immutable":@YES,@"all_work_physical_complete":@YES,@"ledger_released":@YES,
            @"library_build_identity":ns(runtime_build),@"tool_source_sha256":@(TC_NATIVE_REPLAY_SOURCE_SHA),
            @"expected_library_build_identity":@(TC_NATIVE_REPLAY_EXPECTED_BUILD_ID),
            @"device":ns(device),@"program_key":ns(program),@"owned_cache":ns(cache.string()),
            @"precision":ns(mg::NativeFfn::parallel_recipe()),@"rrms_limit":@(rrms_limit),
            @"prepare_seconds":@{@"source_materialization":@(source_prepare_seconds),@"dense_pso":@(dense_prepare_seconds),
                @"hybrid_program_pso":@(hybrid_prepare_seconds),@"cache_off_controller":@(cache_off_prepare_seconds),
                @"cache_on_controller":@(scale_cache_comparison?hybrid_prepare_seconds:0)},
            @"memory":@{@"native_owned_limit":@(native_limit),@"native_owned_peak":@(snapshot.peak_committed_bytes),
                @"surfaces":@(account->surfaces),@"surface_bytes":@(account->surface_bytes),
                @"process_baseline_bytes":@(baseline.process_footprint_bytes),@"process_observed_after_bytes":@(peak.process_footprint_bytes),
                @"mlx_source_active_bytes":@(mlx_source_active),@"imports_coverage":@"retained immutable MLX baseline; no synthetic ledger claims",
                @"measurement_scope":scale_cache_comparison?
                    @"combined simultaneous OFF and ON controllers plus shared workspaces; not the incremental memory cost of scale cache" : @"single hybrid controller plus oracle workspaces",
                @"compiler_driver_budget_certified":@NO,@"whole_process_budget_certified":@NO},
            @"timing_scope":@"wall includes encode/submit/physical completion; allocation and PSO/program prepare separate; host stage spans are not additive GPU/ANE busy time",
            @"comparison_cpu_scope":@"post-pair validation outside both arm timings; part of the interval with no work submitted by this tool before the next pair, not device execution time",
            @"device_span_scope":@"first command buffer GPUStartTime to last GPUEndTime for that ticket; elapsed device span, not kernel busy time or physical overlap proof; zero means unavailable",
            @"comparison_acceptance":scale_cache_comparison?@"finite and BF16 bitwise equality required; RRMS diagnostic only":@"finite and RRMS <= 0.08",
            @"sample_limit":scale_cache_comparison ?
                @"only step19: one warmup OFF/ON, two record rounds OFF/ON and ON/OFF; at most six complete hybrid FFNs, no dense GPU arm; same-input crossover, not performance qualification" : (register_weight_staging ?
                @"only step19: one warmup AB, two recorded rounds AB/BA; at most six complete FFNs; single same-input crossover, not performance qualification" :
                @"per input one warmup and one recorded sample per arm; AB step1 and BA step19 are different inputs, not repeated same-input statistics"),
            @"cold_scope":scale_cache_comparison?
                @"distinct fresh owned MIL directories; OFF prepared before ON on one Backend; driver/compiler caches shared and coldness unknown; prepare timings are not independent cold-start comparisons; source loading common" :
                @"fresh owned MIL cache; system/driver compiler cache coldness unknown; first invocation retained as warmup",
            @"physical_overlap_verified":@NO,@"hardware_int8_arithmetic_verified":@NO,@"records":records};
        write_exclusive(output,ns(json(report)));std::cout<<output<<'\n';
        require(numeric_pass,scale_cache_comparison?
            "scale-cache OFF/ON output differs bitwise; rejection receipt preserved":
            "complete native hybrid FFN exceeds RRMS screening; rejection receipt preserved");
    } catch (...) {
        if (trial) {
            bool ended=false;
            try { ended=(!trial->hybrid || trial->hybrid->status().all_work_physically_complete) &&
                (!trial->cache_off || trial->cache_off->status().all_work_physically_complete) &&
                (!trial->dense_ticket || trial->dense_ticket->status().physically_complete); } catch (...) {}
            if (!ended) (void)trial.release(); // No uncertain backing reuse or source release.
        }
        throw;
    }
}
void cpu_only() {
    require(crossover_ab(0) && crossover_ab(1) && !crossover_ab(2),"bounded crossover plan changed");
    auto owner=std::make_shared<int>(1);std::shared_ptr<const void> a=owner,b=owner,c=std::make_shared<int>(1);
    require(same_owner(a,b) && !same_owner(a,c) && !same_owner(a,{}),"ownership comparator contract failed");
    mg::NativeHybridFfnOptions options;options.program_cache="/private/tmp/unopened-replay-cpu-cache";
    options.register_weight_staging=true;mg::NativeHybridFfn::validate_options(options);
    options.weight_scale_cache_entries=1;options.weight_scale_source_revision="cpu-metadata-only";
    options.weight_scale_source_generation=1;mg::NativeHybridFfn::validate_options(options);
    require(mg::NativeHybridFfnPlan{}.weight_scale_cache_slot==UINT32_MAX,"OFF cache slot default changed");
    // Known BF16 values +1,-1,+0,-0,+2,-2; no array/device allocator involved.
    const std::array<uint16_t,6> values{0x3f80,0xbf80,0x0000,0x8000,0x4000,0xc000};
    const auto view=[](const auto& data) {return CpuBf16View{data.data(),2,3,6,data.size()*2,mg::ScalarType::BF16};};
    const auto original=view(values);const auto exact=compare_cpu(original,original);
    require(exact.rrms==0 && exact.maximum==0 && exact.cosine==1 && exact.bitwise_equal && exact.finite_elements==6,
        "CPU exact comparison contract failed");
    auto changed=values;changed[3]=0x0000;
    const auto zero=compare_cpu(original,view(changed));
    require(zero.rrms==0 && zero.maximum==0 && !zero.bitwise_equal,"CPU signed-zero bitwise distinction lost");
    changed=values;changed[0]=0x4000;
    const auto different=compare_cpu(original,view(changed));
    require(std::abs(different.rrms-std::sqrt(0.1))<1e-12 && different.maximum==1 &&
        different.peak_relative==0.5 && std::abs(different.cosine-11/std::sqrt(130.0))<1e-12 &&
        !different.bitwise_equal,"CPU nonzero error normalization changed");
    size_t rejected=0;
    const auto rejects=[&](const CpuBf16View& a,const CpuBf16View& b) {
        bool failed=false;try {(void)compare_cpu(a,b);} catch(const std::exception&) {failed=true;}
        require(failed,"CPU invalid BF16 comparison accepted");++rejected;
    };
    for(uint16_t bits:{uint16_t(0x7fc1),uint16_t(0x7f80),uint16_t(0xff80)}) {
        changed=values;changed[1]=bits;rejects(original,view(changed));rejects(view(changed),original);
    }
    auto invalid=original;invalid.row_stride_bytes+=2;rejects(original,invalid);
    invalid=original;--invalid.extent_bytes;rejects(original,invalid);
    invalid=original;invalid.type=mg::ScalarType::FP16;rejects(original,invalid);
    invalid=original;invalid.data=nullptr;rejects(original,invalid);
    invalid=original;invalid.data=static_cast<const std::byte*>(original.data)+1;rejects(original,invalid);
    invalid=original;invalid.rows=1;rejects(original,invalid);
    invalid=original;invalid.rows=SIZE_MAX;rejects(original,invalid);
    changed.fill(0);rejects(view(changed),view(changed));
    std::cout<<"native_qwen_hybrid_ffn_replay_cpu=PASS comparison_cases=17 rejected_cases="<<rejected
        <<" device_created=0 files_opened=0 model_loads=0\n";
}
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            if(argc==2 && std::string_view(argv[1])=="--cpu-only") {(void)verified_runtime_build();cpu_only();return 0;}
            require(argc==6 && (std::string_view(argv[1])=="--device" || std::string_view(argv[1])=="--register-device" ||
                    std::string_view(argv[1])=="--scale-cache-device"),
                "usage: native-qwen-hybrid-ffn-replay --cpu-only | --device|--register-device|--scale-cache-device checkpoint capture-dir fresh-cache new-receipt.json");
            run(argv[2],argv[3],argv[4],argv[5],std::string_view(argv[1])!="--device",
                std::string_view(argv[1])=="--scale-cache-device");return 0;
        } catch (const std::exception& error) {
            std::cerr<<"native_qwen_hybrid_ffn_replay=FAIL "<<error.what()<<'\n';return 1;
        }
    }
}
