// One generated full dense FFN and one complete native W1 FFN. No MLX/model.
#include "../../native/backends/metal_graph/native_hybrid_ffn.hpp"
#include "../../native/backends/metal_graph/native_weight_scale_cache_contract.hpp"
#include "../../native/runtime/memory_accounting.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>

// Standalone component link of the existing ledger; same throwing contract as
// core/common.cpp, without linking the complete model/request runtime.
namespace tc {
void require(bool value,const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
}

using namespace tc::metal_graph;
namespace ag=tc::ane::gpu;
namespace {
constexpr uint32_t rows=32,hidden=128,width=1024,channels=512;
constexpr uint64_t seed=2026100801ULL;
constexpr double screening_rrms=0.08,screening_peak_relative=0.08;
constexpr size_t guard=64;
constexpr unsigned char canary=0x6d;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function function,const char* message) {
    bool rejected=false;
    try { function(); } catch (const std::exception&) { rejected=true; }
    check(rejected,message);
}
uint16_t narrow(float value) {
    auto bits=std::bit_cast<uint32_t>(value);
    return uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
float expand(uint16_t value) { return std::bit_cast<float>(uint32_t(value)<<16); }
struct Generator {
    uint64_t state=seed;
    float centered() {
        state^=state>>12; state^=state<<25; state^=state>>27;
        const uint64_t value=state*2685821657736338717ULL;
        return 2.f*(float(uint32_t(value>>40))*(1.f/16777216.f))-1.f;
    }
};

// Both Metal allocations and original IOSurface backings use this ONE ledger.
// Counters observe actual storage hooks; imports may never charge a live native
// handle again. This fixture does not certify compiler/driver/process memory.
struct Claim {
    std::optional<tc::MemoryReservation> reservation;
    std::optional<tc::StorageLease> lease;
    uint64_t generation=0;
    size_t upper=0;
    bool surface=false;
};
struct Accounting {
    tc::MemoryLedger ledger{64ULL<<20};
    uint64_t next_generation=1;
    size_t native_allocations=0,surface_allocations=0,surface_bytes=0;
    std::map<void*,std::weak_ptr<Claim>> active;
    std::shared_ptr<void> reserve(size_t upper,std::string_view label,bool surface) {
        auto reservation=ledger.try_reserve(tc::MemoryClass::Activation,upper,std::string(label));
        check(bool(reservation),"fixture-owned backing exceeds 64 MiB ledger");
        auto claim=std::make_shared<Claim>();
        claim->reservation.emplace(std::move(*reservation));
        claim->generation=next_generation++; claim->upper=upper; claim->surface=surface;
        return claim;
    }
    void allocated(const std::shared_ptr<void>& token,void* handle,size_t bytes,bool surface) {
        auto claim=std::static_pointer_cast<Claim>(token);
        check(claim && claim->surface==surface && claim->reservation && !claim->lease &&
              handle && bytes && bytes<=claim->upper,"invalid fixture backing commit");
        check(!active.count(handle) || active.at(handle).expired(),
              "an imported native/IOSurface backing was charged twice");
        claim->lease.emplace(claim->reservation->commit(
            {0x574146464e544553ULL,uint64_t(reinterpret_cast<uintptr_t>(handle)),bytes,claim->generation}));
        claim->reservation.reset(); active[handle]=claim;
        if (surface) { ++surface_allocations; surface_bytes+=bytes; }
        else ++native_allocations;
    }
};
AllocationHooks native_hooks(const std::shared_ptr<Accounting>& a) {
    return {[a](size_t upper,std::string_view label){return a->reserve(upper,label,false);},
        [a](const std::shared_ptr<void>& token,void* handle,size_t bytes){a->allocated(token,handle,bytes,false);}};
}
ag::SurfaceAllocationHooks surface_hooks(const std::shared_ptr<Accounting>& a) {
    return {[a](size_t upper,std::string_view label){return a->reserve(upper,label,true);},
        [a](const std::shared_ptr<void>& token,void* handle,size_t bytes){a->allocated(token,handle,bytes,true);}};
}

struct Matrix {
    Buffer backing;
    MatrixView view;
};
Matrix matrix(Backend& backend,uint32_t m,uint32_t n,ScalarType type,std::string_view label,uint32_t padding=8) {
    const size_t item=scalar_bytes(type),pitch=size_t(n+padding)*item;
    auto buffer=backend.allocate(guard+size_t(m)*pitch+guard,label,StorageMode::Shared);
    std::memset(buffer.contents(),canary,buffer.size_bytes());
    return {buffer,{buffer.view(guard,size_t(m-1)*pitch+size_t(n)*item,type),m,n,pitch}};
}
void fill(const MatrixView& m,Generator& random,float sigma) {
    check(m.storage.scalar_type()==ScalarType::BF16 && m.storage.contents(),"fixture BF16 source missing");
    for (uint32_t row=0;row<m.rows;++row) {
        auto* target=reinterpret_cast<uint16_t*>(static_cast<std::byte*>(m.storage.contents())+row*m.row_stride_bytes);
        for (uint32_t col=0;col<m.columns;++col) target[col]=narrow(random.centered()*sigma*std::sqrt(3.f));
    }
}
uint64_t hash(const Buffer& buffer) {
    uint64_t result=1469598103934665603ULL;
    const auto* data=static_cast<const unsigned char*>(buffer.contents());
    for (size_t i=0;i<buffer.size_bytes();++i) { result^=data[i]; result*=1099511628211ULL; }
    return result;
}
void check_padding(const Matrix& m) {
    const auto* data=static_cast<const unsigned char*>(m.backing.contents());
    const size_t first=m.view.storage.offset_bytes(),line=size_t(m.view.columns)*scalar_bytes(m.view.storage.scalar_type());
    for (size_t i=0;i<m.backing.size_bytes();++i) {
        const bool valid=i>=first && (i-first)/m.view.row_stride_bytes<m.view.rows &&
                         (i-first)%m.view.row_stride_bytes<line;
        if (!valid) check(data[i]==canary,"fixture row padding or allocation guard overwritten");
    }
}
float element(const MatrixView& m,uint32_t row,uint32_t col) {
    const auto* address=static_cast<const std::byte*>(m.storage.contents())+row*m.row_stride_bytes;
    if (m.storage.scalar_type()==ScalarType::BF16) return expand(reinterpret_cast<const uint16_t*>(address)[col]);
    check(m.storage.scalar_type()==ScalarType::FP32,"fixture output dtype unsupported");
    return reinterpret_cast<const float*>(address)[col];
}
struct Error { double rrms=0,maximum=0,reference_peak=0,peak_relative=0,cosine=0; size_t finite=0; };
Error compare(const MatrixView& expected,const MatrixView& actual) {
    Error result; double squared=0,reference_squared=0,actual_squared=0,dot=0;
    for (uint32_t row=0;row<expected.rows;++row) for (uint32_t col=0;col<expected.columns;++col) {
        const double a=element(expected,row,col),b=element(actual,row,col);
        check(std::isfinite(a) && std::isfinite(b),"nonfinite dense/hybrid FFN output");
        ++result.finite; squared+=(a-b)*(a-b); reference_squared+=a*a; actual_squared+=b*b; dot+=a*b;
        result.maximum=std::max(result.maximum,std::abs(a-b));
        result.reference_peak=std::max(result.reference_peak,std::abs(a));
    }
    check(reference_squared>0 && actual_squared>0,"degenerate generated FFN fixture");
    result.rrms=std::sqrt(squared/reference_squared);
    result.peak_relative=result.maximum/result.reference_peak;
    result.cosine=dot/std::sqrt(reference_squared*actual_squared);
    return result;
}
void physical_evidence(const NativeHybridFfnStatus& s) {
    const auto& e=s.evidence;
    check(s.ok && !s.retired && !s.running && s.all_work_physically_complete && s.generation,
          "hybrid controller did not complete successfully");
    check(e.handoff.ok && !e.handoff.retired && e.handoff.producer_submitted &&
          e.handoff.producer_physical_complete && e.handoff.ane_submission_attempted &&
          e.handoff.ane_physical_complete && e.handoff.all_work_physically_complete,
          "captured producer or ANE physical receipt missing");
    check(e.gpu_attempted && e.consumer_attempted && e.gpu && e.consumer &&
          e.gpu->ok && e.gpu->physically_complete && !e.gpu->timed_out &&
          e.consumer->ok && e.consumer->physically_complete && !e.consumer->timed_out &&
          e.gpu->submitted_buffers && e.gpu->completed_buffers==e.gpu->submitted_buffers &&
          e.consumer->submitted_buffers && e.consumer->completed_buffers==e.consumer->submitted_buffers,
          "complement or restore/join consumer physical receipt missing");
}
NativeHybridFfnOptions fixture_options(const std::filesystem::path& cache,
                                      ag::SurfaceAllocationHooks allocation={}) {
    NativeHybridFfnOptions options;
    options.rows=rows;options.hidden=hidden;options.width=width;options.ane_channels=channels;
    options.program_cache=cache;options.surface_allocation=std::move(allocation);
    return options;
}

void cpu_only() {
    auto options=fixture_options("/private/tmp/native-hybrid-unused-cache");
    NativeHybridFfn::validate_options(options);
    auto invalid=options; invalid.ane_channels=513;
    rejects([&]{NativeHybridFfn::validate_options(invalid);},"invalid stage width accepted");
    Generator a,b;
    for (int i=0;i<128;++i) check(a.centered()==b.centered(),"fixture generator is not deterministic");
    check(narrow(1.00390625f-1.f)==0x3b80,"single-cast BF16 join fixture invalid");
    auto cached=options;
    cached.register_weight_staging=true; cached.weight_scale_cache_entries=2;
    cached.weight_scale_source_revision="generated-immutable-scale-cache-ab-a-v1";
    cached.weight_scale_source_generation=seed;
    NativeHybridFfn::validate_options(cached);
    check(2*compact_weight_scale_bytes(hidden,channels)==4608,"two-entry scale payload changed");
    invalid=cached; invalid.weight_scale_source_revision.clear();
    rejects([&]{NativeHybridFfn::validate_options(invalid);},"cache without source revision accepted");
    invalid=cached; invalid.weight_scale_source_generation=0;
    rejects([&]{NativeHybridFfn::validate_options(invalid);},"cache without source generation accepted");
    invalid=cached; invalid.weight_scale_cache_entries=33;
    rejects([&]{NativeHybridFfn::validate_options(invalid);},"unbounded scale cache accepted");
    invalid=cached; invalid.weight_scale_cache_entries=0;
    rejects([&]{NativeHybridFfn::validate_options(invalid);},"disabled cache with source epoch accepted");
    std::cout<<"native_hybrid_ffn_cpu_contract=PASS device_created=0 model_loads=0\n";
}

// Keeping all visible backings under one owner makes failure retention explicit
// if the controller cannot prove physical completion. Such a fixture exits
// unsuccessfully; it never reads partial output or retries device work.
struct Trial {
    std::shared_ptr<Accounting> accounting=std::make_shared<Accounting>();
    std::unique_ptr<Backend> backend;
    std::optional<NativeHybridFfn> hybrid;
    std::optional<Ticket> oracle_ticket;
    std::vector<Matrix> matrices;
};

void run_device(const std::filesystem::path& cache) {
    check(cache.is_absolute() && !std::filesystem::exists(cache),
          "component program cache must be a fresh, absolute, independently owned path");
    auto trial=std::make_unique<Trial>();
    const auto accounting=trial->accounting;
    try {
      {
        BackendOptions backend_options; backend_options.allocation=native_hooks(accounting);
        trial->backend=std::make_unique<Backend>(std::move(backend_options));
        auto& backend=*trial->backend;
        auto& values=trial->matrices; values.reserve(8);
        values.push_back(matrix(backend,rows,hidden,ScalarType::BF16,"fixture-input"));
        values.push_back(matrix(backend,2*width,hidden,ScalarType::BF16,"fixture-fused-gate-up",16));
        values.push_back(matrix(backend,hidden,width,ScalarType::BF16,"fixture-down",16));
        values.push_back(matrix(backend,rows,width,ScalarType::BF16,"fixture-hidden",16));
        values.push_back(matrix(backend,rows,hidden,ScalarType::BF16,"fixture-dense-output"));
        values.push_back(matrix(backend,rows,hidden,ScalarType::BF16,"fixture-hybrid-output"));
        values.push_back(matrix(backend,rows,hidden,ScalarType::FP32,"fixture-gpu-partial"));
        values.push_back(matrix(backend,rows,hidden,ScalarType::FP32,"fixture-ane-partial"));
        const auto input=values[0].view,down=values[2].view;
        const auto fused=values[1].view;
        const size_t weight_extent=size_t(width-1)*fused.row_stride_bytes+hidden*2;
        const MatrixView gate_view{fused.storage.buffer().view(fused.storage.offset_bytes(),weight_extent,ScalarType::BF16),
            width,hidden,fused.row_stride_bytes};
        const MatrixView up_view{fused.storage.buffer().view(fused.storage.offset_bytes()+size_t(width)*fused.row_stride_bytes,
            weight_extent,ScalarType::BF16),width,hidden,fused.row_stride_bytes};
        auto gpu_hidden=values[3].view; gpu_hidden.columns=channels;
        Generator random;
        fill(input,random,0.1f); fill(fused,random,1.f/std::sqrt(float(hidden)));
        fill(down,random,1.f/std::sqrt(float(width)));
        const std::array<uint64_t,3> source_hashes{hash(values[0].backing),hash(values[1].backing),hash(values[2].backing)};
        auto verify_sources=[&] {
            for (size_t i=0;i<source_hashes.size();++i)
                check(hash(values[i].backing)==source_hashes[i],"generated read-only source or padding mutated");
        };

        const auto prepare_start=std::chrono::steady_clock::now();
        auto dense=NativeFfn::prepare(backend,{hidden,width,width});
        auto options=fixture_options(cache,surface_hooks(accounting));
        trial->hybrid.emplace(NativeHybridFfn::prepare(backend,std::move(options)));
        const double prepare_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prepare_start).count();
        auto& hybrid=*trial->hybrid;
        check(accounting->surface_allocations>0 && hybrid.surface_bytes()==accounting->surface_bytes,
              "surface hooks did not cover original controller backings exactly once");
        const auto before=hybrid.status();
        check(!before.running && !before.retired && before.all_work_physically_complete,
              "fresh hybrid controller has pending work");

        CommandGraph dense_graph;
        dense.encode(dense_graph,{input,gate_view,up_view,down,values[3].view,values[4].view,0});
        const auto dense_start=std::chrono::steady_clock::now();
        trial->oracle_ticket.emplace(backend.submit(std::move(dense_graph)));
        const auto dense_done=trial->oracle_ticket->wait(std::chrono::seconds(10));
        const double dense_wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-dense_start).count();
        check(dense_done.ok && dense_done.physically_complete,"dense FFN oracle did not physically complete");
        verify_sources(); check_padding(values[3]); check_padding(values[4]);

        auto plan=[&] {
            NativeHybridFfnPlan p;
            p.input=input; p.gate=gate_view; p.up=up_view; p.down=down;
            p.gpu_partial=values[6].view; p.ane_partial=values[7].view; p.published_output=values[5].view;
            hybrid.gpu_complement().encode(p.gpu_ffn,{input,gate_view,up_view,down,gpu_hidden,p.gpu_partial,0});
            hybrid.gpu_complement().encode_join_partials(p.after_join,p.gpu_partial,p.ane_partial,p.published_output);
            return p;
        };
        const auto result=hybrid.execute(plan(),std::chrono::seconds(30));
        physical_evidence(result.completion); physical_evidence(hybrid.status());
        physical_evidence(hybrid.drain(std::chrono::milliseconds(0)));
        physical_evidence(hybrid.drain(std::chrono::milliseconds(1)));
        rejects([&]{(void)hybrid.drain(std::chrono::milliseconds(-1));},"negative controller drain accepted");
        verify_sources();
        for (size_t i=3;i<values.size();++i) check_padding(values[i]);
        for (uint32_t r=0;r<rows;++r) for (uint32_t c=0;c<hidden;++c)
            check(std::isfinite(element(values[6].view,r,c)) && std::isfinite(element(values[7].view,r,c)),
                  "nonfinite complete GPU/ANE partial");
        const auto difference=compare(values[4].view,values[5].view);
        auto invalid=plan(); invalid.ane_partial=invalid.gpu_partial;
        const auto generation=hybrid.status().generation;
        rejects([&]{(void)hybrid.execute(std::move(invalid));},"aliased complete-path partials accepted");
        const auto after_invalid=hybrid.status();
        check(after_invalid.generation==generation,"invalid plan began a new submitted generation");
        physical_evidence(after_invalid);
        verify_sources();
        const auto snapshot=accounting->ledger.snapshot();
        const auto active_backings=size_t(std::count_if(accounting->active.begin(),accounting->active.end(),
            [](const auto& entry){return !entry.second.expired();}));
        check(!snapshot.reservation_count && !snapshot.unknown_bytes &&
              snapshot.storage_count==active_backings,
              "fixture ledger has uncommitted/unknown/duplicate backing ownership");
        std::cout<<std::setprecision(10)
            <<"native_hybrid_ffn_numeric rrms="<<difference.rrms<<" max_abs="<<difference.maximum
            <<" reference_absmax="<<difference.reference_peak<<" max_abs_over_reference_peak="<<difference.peak_relative
            <<" cosine="<<difference.cosine<<" finite="<<difference.finite
            <<" rrms_limit="<<screening_rrms<<" peak_relative_limit="<<screening_peak_relative<<'\n'
            <<"native_hybrid_ffn_receipt rows="<<rows<<" hidden="<<hidden<<" width="<<width
            <<" ane_channels="<<channels<<" seed="<<seed<<" program_key="<<hybrid.program_key()
            <<" prepare_seconds="<<prepare_seconds<<" dense_wall_seconds="<<dense_wall
            <<" hybrid_wall_seconds="<<result.total_seconds<<" producer_seconds="<<result.producer_seconds
            <<" submit_ane_seconds="<<result.submit_ane_seconds<<" branch_join_seconds="<<result.branch_join_seconds
            <<" consumer_seconds="<<result.consumer_seconds<<" original_surface_allocations="<<accounting->surface_allocations
            <<" surface_bytes="<<accounting->surface_bytes<<" native_allocations="<<accounting->native_allocations
            <<" ledger_storage_bytes="<<snapshot.storage_bytes<<" ledger_peak_bytes="<<snapshot.peak_committed_bytes
            <<" duplicate_import_charges=0 producer_ane_gpu_consumer_physical=1 source_immutable=1 padding_preserved=1\n";
        check(difference.rrms<=screening_rrms && difference.peak_relative<=screening_peak_relative,
              "generated complete W1 FFN exceeds declared screening threshold");
        // Drop copied graph/ticket owners before auditing storage reclamation.
        trial->oracle_ticket.reset();
      }
        trial.reset();
        const auto released=accounting->ledger.snapshot();
        check(!released.storage_count && !released.reservation_count && !released.pending_release_count &&
              !released.unknown_bytes,"completed component retained fixture ledger backings");
        std::cout<<"native_hybrid_ffn_component=PASS dense_requests=1 hybrid_requests=1 invalid_requests_submitted=0"
                    " ledger_released=1 models=0 full_model=0 e2e=0 performance_qualified=0 quality_scope=generated_component_only\n";
    } catch (...) {
        if (trial) {
            bool physically_complete=false;
            try {
                physically_complete=(!trial->hybrid || trial->hybrid->status().all_work_physically_complete) &&
                    (!trial->oracle_ticket || trial->oracle_ticket->status().physically_complete);
            } catch (...) {}
            if (!physically_complete) (void)trial.release();
        }
        throw;
    }
}

// Exactly one dense A oracle and three complete cached hybrid FFNs. No source
// mutation, synthetic completion, prepacked bank reuse, model or file export.
// The controller itself must repack A after B has overwritten its one W8 bank.
void run_scale_cache_device(const std::filesystem::path& cache) {
    check(cache.is_absolute() && !std::filesystem::exists(cache),
          "scale-cache component requires a fresh absolute owned program cache");
    auto trial=std::make_unique<Trial>();
    const auto accounting=trial->accounting;
    try {
      {
        BackendOptions backend_options;backend_options.allocation=native_hooks(accounting);
        trial->backend=std::make_unique<Backend>(std::move(backend_options));
        auto& backend=*trial->backend;auto& v=trial->matrices;v.reserve(12);
        v.push_back(matrix(backend,rows,hidden,ScalarType::BF16,"scale-fixture-input"));
        for (const auto* name:{"scale-fixture-a-gate","scale-fixture-a-up","scale-fixture-a-down",
                              "scale-fixture-b-gate","scale-fixture-b-up","scale-fixture-b-down"}) {
            const bool down=v.size()==3 || v.size()==6;
            v.push_back(matrix(backend,down?hidden:width,down?width:hidden,ScalarType::BF16,name,16));
        }
        v.push_back(matrix(backend,rows,width,ScalarType::BF16,"scale-fixture-hidden",16));
        v.push_back(matrix(backend,rows,hidden,ScalarType::BF16,"scale-fixture-dense-output"));
        v.push_back(matrix(backend,rows,hidden,ScalarType::BF16,"scale-fixture-hybrid-output"));
        v.push_back(matrix(backend,rows,hidden,ScalarType::FP32,"scale-fixture-gpu-partial"));
        v.push_back(matrix(backend,rows,hidden,ScalarType::FP32,"scale-fixture-ane-partial"));
        Generator random;fill(v[0].view,random,0.1f);
        for(size_t i=1;i<7;++i) fill(v[i].view,random,1.f/std::sqrt(float(i%3==0?width:hidden)));
        std::array<uint64_t,7> source_hashes{};
        for(size_t i=0;i<source_hashes.size();++i) source_hashes[i]=hash(v[i].backing);
        check(source_hashes[1]!=source_hashes[4] && source_hashes[2]!=source_hashes[5] &&
              source_hashes[3]!=source_hashes[6],"generated A and B weights are not distinct");
        auto verify=[&] {
            for(size_t i=0;i<source_hashes.size();++i)
                check(hash(v[i].backing)==source_hashes[i],"scale-cache generated source/padding mutated");
            for(const auto& item:v) check_padding(item);
        };
        const auto prepared=std::chrono::steady_clock::now();
        auto dense=NativeFfn::prepare(backend,{hidden,width,width});
        auto options=fixture_options(cache,surface_hooks(accounting));
        options.register_weight_staging=true;options.weight_scale_cache_entries=2;
        options.weight_scale_source_revision="generated-immutable-scale-cache-ab-a-v1";
        options.weight_scale_source_generation=seed;
        trial->hybrid.emplace(NativeHybridFfn::prepare(backend,std::move(options)));
        const double prepare_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prepared).count();
        auto& hybrid=*trial->hybrid;
        check(hybrid.weight_scale_cache_capacity()==2 && hybrid.weight_scale_cache_payload_bytes()==4608,
              "controller compact scale-cache capacity/payload mismatch");
        check(accounting->surface_allocations==9 && hybrid.surface_bytes()==accounting->surface_bytes,
              "scale-cache original Surface accounting coverage mismatch");
        CommandGraph oracle;
        dense.encode(oracle,{v[0].view,v[1].view,v[2].view,v[3].view,v[7].view,v[8].view,0});
        trial->oracle_ticket.emplace(backend.submit(std::move(oracle)));
        const auto done=trial->oracle_ticket->wait(std::chrono::seconds(10));
        check(done.ok && done.physically_complete && !done.timed_out && done.submitted_buffers &&
              done.completed_buffers==done.submitted_buffers,"scale-cache dense A physical receipt missing");
        verify();
        auto gpu_hidden=v[7].view;gpu_hidden.columns=channels;
        auto plan=[&](uint32_t slot) {
            const size_t first=slot==0?1:4;
            NativeHybridFfnPlan p;
            p.input=v[0].view;p.gate=v[first].view;p.up=v[first+1].view;p.down=v[first+2].view;
            p.gpu_partial=v[10].view;p.ane_partial=v[11].view;p.published_output=v[9].view;
            p.weight_scale_cache_slot=slot;
            hybrid.gpu_complement().encode(p.gpu_ffn,{p.input,p.gate,p.up,p.down,gpu_hidden,p.gpu_partial,0});
            hybrid.gpu_complement().encode_join_partials(p.after_join,p.gpu_partial,p.ane_partial,p.published_output);
            return p;
        };
        std::array<uint16_t,size_t(rows)*hidden> first_output{};
        uint32_t hits=0,misses=0;Error first_error,last_error;
        for(uint32_t invocation=0;invocation<3;++invocation) {
            const uint32_t slot=invocation==1?1:0;
            const auto result=hybrid.execute(plan(slot),std::chrono::seconds(30));
            physical_evidence(result.completion);physical_evidence(hybrid.status());
            check(result.completion.generation==invocation+1,"scale-cache physical generation mismatch");
            check(result.weight_scale_cache_hit==(invocation==2) && result.weight_scale_cache_miss==(invocation!=2),
                  "scale-cache cold A / cold B / hit A transitions differ from expectation");
            hits+=result.weight_scale_cache_hit;misses+=result.weight_scale_cache_miss;
            verify();size_t changed=0;
            for(uint32_t row=0;row<rows;++row) for(uint32_t col=0;col<hidden;++col) {
                for(size_t index:{9U,10U,11U})
                    check(std::isfinite(element(v[index].view,row,col)),"nonfinite cached FFN output/partial");
                const auto* line=reinterpret_cast<const uint16_t*>(
                    static_cast<const std::byte*>(v[9].view.storage.contents())+row*v[9].view.row_stride_bytes);
                const size_t index=size_t(row)*hidden+col;
                if(invocation==0) first_output[index]=line[col];
                else changed+=first_output[index]!=line[col];
            }
            if(invocation==1) check(changed>0,"B did not change the published output after overwriting the bank");
            if(invocation==2) check(changed==0,"cache-hit A differs bitwise from cold A after B overwrote the bank");
            if(slot==0) {
                auto error=compare(v[8].view,v[9].view);
                check(error.rrms<=screening_rrms,"cached complete A FFN exceeds declared RRMS screening");
                (invocation==0?first_error:last_error)=error;
            }
            std::cout<<std::setprecision(10)<<"scale_cache_invocation="<<invocation+1<<" source="<<(slot==0?"A":"B")
                <<" slot="<<slot<<" hit="<<result.weight_scale_cache_hit<<" miss="<<result.weight_scale_cache_miss
                <<" generation="<<result.completion.generation<<" physical_complete=1 finite=1"
                <<" wall_seconds="<<result.total_seconds<<" producer_seconds="<<result.producer_seconds
                <<" producer_gpu_span_seconds="<<result.producer_gpu_span_seconds
                <<" gpu_complement_span_seconds="<<result.gpu_complement_span_seconds<<'\n';
        }
        physical_evidence(hybrid.drain(std::chrono::milliseconds(0)));verify();
        check(hits==1 && misses==2,"scale-cache invocation counters do not match A/B/A");
        const auto snapshot=accounting->ledger.snapshot();
        const auto active=size_t(std::count_if(accounting->active.begin(),accounting->active.end(),
            [](const auto& entry){return !entry.second.expired();}));
        check(!snapshot.reservation_count && !snapshot.unknown_bytes && snapshot.storage_count==active,
              "scale-cache ledger has unknown/uncommitted/duplicate backing ownership");
        std::cout<<std::setprecision(10)<<"native_scale_cache_receipt H="<<hidden<<" F="<<width<<" M="<<rows
            <<" A="<<channels<<" seed="<<seed<<" cache_capacity="<<hybrid.weight_scale_cache_capacity()
            <<" scale_payload_bytes="<<hybrid.weight_scale_cache_payload_bytes()<<" hits="<<hits<<" misses="<<misses
            <<" prepare_seconds="<<prepare_seconds<<" rrms_cold="<<first_error.rrms<<" rrms_hit="<<last_error.rrms
            <<" rrms_limit="<<screening_rrms<<" max_abs="<<last_error.maximum<<" cosine="<<last_error.cosine
            <<" max_role=diagnostic_only bitwise_A_after_B=1 source_hash_algorithm=fixture_rolling64 source_hashes=";
        for(size_t i=0;i<source_hashes.size();++i) std::cout<<(i?",":"")<<source_hashes[i];
        std::cout<<" source_immutable=1 padding_preserved=1 physical_complete=1 ledger_peak_bytes="
            <<snapshot.peak_committed_bytes<<" scale_payload_excludes_allocation_page_rounding=1\n";
        trial->oracle_ticket.reset();
      }
        trial.reset();
        const auto released=accounting->ledger.snapshot();
        check(!released.storage_count && !released.reservation_count && !released.pending_release_count &&
              !released.unknown_bytes,"completed scale-cache fixture retained ledger backings");
        std::cout<<"native_scale_cache_controller=PASS dense_requests=1 hybrid_requests=3 hits=1 misses=2"
            " ledger_released=1 models=0 e2e=0 performance_qualified=0 quality_scope=generated_component_only\n";
    } catch(...) {
        if(trial) {
            bool complete=false;
            try { complete=(!trial->hybrid || trial->hybrid->status().all_work_physically_complete) &&
                (!trial->oracle_ticket || trial->oracle_ticket->status().physically_complete); } catch(...) {}
            if(!complete) (void)trial.release();
        }
        throw;
    }
}
}

int main(int argc,char** argv) {
    try {
        if (argc==2 && std::string_view(argv[1])=="--cpu-only") { cpu_only(); return 0; }
        if (argc==3 && std::string_view(argv[1])=="--tiny-device") {
            run_device(std::filesystem::absolute(argv[2]).lexically_normal()); return 0;
        }
        if (argc==3 && std::string_view(argv[1])=="--tiny-scale-cache-device") {
            run_scale_cache_device(std::filesystem::absolute(argv[2]).lexically_normal());return 0;
        }
        std::cout<<"usage: native-hybrid-ffn-component --cpu-only | --tiny-device|--tiny-scale-cache-device <fresh-owned-cache>\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr<<"native_hybrid_ffn_component=FAIL "<<error.what()<<'\n'; return 1;
    }
}
