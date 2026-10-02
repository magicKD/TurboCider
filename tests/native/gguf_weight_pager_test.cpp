#include "runtime/streaming/gguf_weight_pager.hpp"
#include "runtime/streaming/context.hpp"
#include <iostream>
#include <fcntl.h>

namespace {
using namespace tc;
using namespace tc::streaming;
void ensure(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
class Adapter final : public ModelSlotAdapter {
    struct Job { Adapter *owner; const Group *group = nullptr; };
    GgufWeightPager &pager_;
    std::array<Job,3> jobs_;
    Weights bound_;
    Tensor input_;
    uint32_t pool_ = 0;
    uint64_t sequence_ = 0;
  public:
    uint64_t computes = 0;
    std::optional<Tensor> escaped;
    explicit Adapter(GgufWeightPager &p) : pager_(p), jobs_{{{this},{this},{this}}}, input_(mx::ones({1,64},mx::float32)) {}
    void create_pool(const PoolLayout &pool) override { pool_ = pool.id; pager_.create_pool(pool); }
    FillJob make_fill_job(const Group &group, const tc_stream_slot_ticket_v1 &ticket) override {
        auto &job = jobs_.at(ticket.slot); job.group = &group;
        return {ticket,&job,[](void *pointer,const tc_stream_slot_ticket_v1 *ticket,const std::atomic<bool> *cancel,uint64_t *bytes)->int {
            auto &j = *static_cast<Job *>(pointer);
            try { *bytes=j.owner->pager_.fill(*j.group,*ticket,cancel); return 0; } catch (...) { return -1; }
        }};
    }
    void encode_prefix(uint32_t) override {}
    void prepare_group(const Group &group, const tc_stream_slot_ticket_v1 &ticket) override {
        bound_ = pager_.bind(group,ticket);
        auto stale = ticket; ++stale.content_generation;
        bool rejected=false; try { (void)pager_.bind(group,stale); } catch (...) { rejected=true; }
        ensure(rejected,"stale content ticket accepted");
    }
    bool overlap_next_fill_after_claim() const noexcept override { return true; }
    ReaderSet encode_group(const Group &group,const tc_stream_slot_ticket_v1 &,CompletionMailbox &) override {
        const auto name="layers."+std::to_string(group.blocks.front());
        auto output=linear(input_,bound_,name);
        mx::eval(output);
        ensure(mx::all(output == Tensor(float(64*(group.blocks.front()+1)))).item<bool>(),"decoded GPU matrix result mismatch");
        if (!escaped) escaped=bound_.at(name+".weight");
        ++computes; bound_.clear();
        ReaderSet readers; readers.count=1; readers.fences[0]={1,++sequence_}; readers.already_complete=true; return readers;
    }
    bool drain() noexcept override { try { mx::synchronize(); return true; } catch (...) { return false; } }
    void destroy_pool() noexcept override { bound_.clear(); pager_.destroy_pool(pool_); }
};
}

int main(int argc,char **argv) {
    try {
        ensure(argc==2 || (argc==3 && std::string(argv[2])=="--mutate-owned-fixture"),"fixture required"); configure_streams(); mx::set_cache_limit(0);
        SourceFileIdentity file; file.logical_id="transformer"; file.path=argv[1];
        auto lease=SourceLease::capture_verified({file});
        auto fd=lease->duplicate_fd("transformer");
        auto directory=gguf::read_directory(fd.get(),lease->file("transformer").bytes);
        Descriptor d; d.model="test-gguf"; d.checkpoint_identity=std::string(lease->artifact_digest());
        d.backend_revision="gguf-pager-test-v1"; d.workload={{"shape","4x64"}};
        d.artifacts.push_back({"transformer",lease->file("transformer").content_digest,lease->file("transformer").bytes,SourceIdentityKind::content_sha256});
        StageDescriptor stage; stage.id="denoiser"; stage.adapter_revision="gguf-pager-test-v1";
        stage.max_slots=3; stage.pass_count=2; stage.passes={{0,"test",{512,64}},{1,"test",{512,64}}};
        auto field=[&](const gguf::TensorDescriptor &t,bool alias) {
            Materialization m; m.format=t.type==0 ? "F32" : "BF16"; m.storage_mode="mlx-metal-shared";
            m.conversion=alias?"gguf-native-alias-v1":"gguf-cpu-rne-v1"; m.shape=t.logical_shape();
            m.reads.push_back({0,t.file_offset,t.bytes,t.name,gguf::type_info(t.type).name,t.logical_shape()});
            return FieldSpec{"weight",t.name,t.elements*(t.type==0?4:2),GgufWeightPager::buffer_alignment,m};
        };
        for(uint32_t i=0;i<4;++i) stage.blocks.push_back({i,"same-layout",{field(directory.tensor("layers."+std::to_string(i)+".weight"),false)}});
        stage.resident_fields.push_back(field(directory.tensor("fixed"),true));
        auto packed = field(directory.tensor("embedding.weight"), false);
        packed.name = "embedding.packed"; packed.bytes = directory.tensor("embedding.weight").bytes;
        packed.materialization->conversion = "gguf-packed-gather-source-v1";
        packed.materialization->format = "U8"; packed.materialization->shape = {packed.bytes};
        stage.resident_fields.push_back(std::move(packed)); d.stages.push_back(stage);
        for (bool streamed : {false, true}) for(uint32_t slots=1;slots<=3;++slots) {
            d.workload["source_residency"] = streamed ? "packed_streamed" : "packed_resident";
            d.workload["packed_read_buffer_bytes"] = "16384";
            StreamingConfig c; c.enabled=true; c.schema_version=1; c.selection="manual"; c.retention="request";
            c.stages["denoiser"]={"streamed",1,slots,0,slots-1,1};
            const auto layout=compile_layout(c,d); MemoryLedger ledger(1ull<<20); Weights fixed;
            auto pager=std::make_unique<GgufWeightPager>(lease,d,d.stages.front(),layout.stages.front(),ledger);
            pager->load_packed(); pager->load_resident_aliases(fixed);
            ensure(fixed.at("fixed").item<float>()==3.5f,"float source alias mismatch");
            ensure(!fixed.has("embedding.weight"), "packed embedding pretended to be a full dense matrix");
            const uint64_t rows[] = {3, 1, 1, 0};
            auto gathered = pager->gather_rows("embedding.weight", rows);
            ensure(gathered.shape() == mx::Shape{4, 64} && gathered.dtype() == mx::bfloat16, "gather output contract mismatch");
            const float values[] = {4, 2, 2, 1};
            auto expected = mx::broadcast_to(mx::reshape(Tensor(values, {4}, mx::bfloat16), {4, 1}), {4, 64});
            ensure(mx::all(gathered == expected).item<bool>(), "gather row order/duplicates changed");
            const uint64_t invalid[] = {4}; bool bad_row = false;
            try { (void)pager->gather_rows("embedding.weight", invalid); } catch (...) { bad_row = true; }
            ensure(bad_row, "invalid embedding token accepted");
            auto adapter=std::make_shared<Adapter>(*pager); std::atomic<bool> cancel{false};
            {
                StageExecutor executor(0,100+slots,adapter); const auto counts=executor.run(layout.stages.front(),cancel);
                ensure(counts.fills==8 && counts.groups_submitted==8 && adapter->computes==8,"wrong execution totals");
                ensure(counts.slot_bundles==slots,"extra dense slots allocated");
            }
            const auto metrics=pager->metrics();
            ensure(metrics.fill_count==8 && metrics.dense_pool_capacity_bytes==0 &&
                metrics.maximum_dense_pool_capacity_bytes==slots*65536,"capacity accounting mismatch");
            if (streamed) {
                ensure(metrics.packed_source_bytes == 4 && metrics.packed_capacity_bytes >= 4 && metrics.packed_capacity_bytes <= 16384 &&
                       metrics.read_buffer_capacity_bytes == 16384, "streamed source retained full packed layers/embedding");
                ensure(metrics.source_read_bytes == 4 + 8 * 34816 + 4 * 68,
                       "streamed I/O bytes mismatch or hidden reread");
                // Failed/cancelled contents never become bindable; the same
                // read buffer must be usable after the synchronous failure.
                const auto &g = layout.stages.front().groups.front();
                const auto &p = layout.stages.front().pools.front();
                pager->create_pool(p);
                tc_stream_slot_ticket_v1 ticket{};
                ticket.struct_size = sizeof(ticket); ticket.version = 1;
                ticket.pool = g.pool; ticket.slot = g.slot; ticket.request_generation = 123;
                ticket.content_generation = 1; ticket.item = {0, 0, 0, g.id};
                std::atomic<bool> stopped{true}; bool cancelled_fill = false;
                try { (void)pager->fill(g, ticket, &stopped); } catch (...) { cancelled_fill = true; }
                ensure(cancelled_fill, "streamed pre-cancel fill succeeded");
                bool partial_ready = false; try { (void)pager->bind(g, ticket); partial_ready = true; } catch (...) {}
                ensure(!partial_ready, "cancelled partial source was publishable");
                stopped.store(false);
                ensure(pager->fill(g, ticket, &stopped) == g.bytes, "streamed refill recovery failed");
                auto rebound = pager->bind(g, ticket); rebound.clear();
                pager->destroy_pool(p.id);
                auto bad = d; bad.workload["packed_read_buffer_bytes"] = "16384junk";
                bool malformed = false;
                try { GgufWeightPager wrong(lease, bad, bad.stages.front(), layout.stages.front(), ledger); } catch (...) { malformed = true; }
                ensure(malformed, "malformed streamed buffer plan accepted");
            }
            pager.reset(); fixed.clear();
            ensure(ledger.snapshot().storage_bytes>0,"escaped MLX view lost physical storage claim");
            adapter->escaped.reset(); adapter.reset();
            gathered = mx::zeros({1}, mx::bfloat16); expected = mx::zeros({1}, mx::bfloat16);
            mx::synchronize(); mx::clear_cache();
            ensure(ledger.snapshot().storage_bytes==0 && ledger.snapshot().reserved_bytes==0,"buffer/ledger release leak");
        }
        {
            auto bf16 = d;
            bf16.workload["source_residency"] = "packed_streamed";
            bf16.workload["precision"] = "z-dense-bf16-v1";
            bf16.workload["packed_read_buffer_bytes"] = "16384";
            auto &fixed_fields = bf16.stages.front().resident_fields;
            fixed_fields.clear();
            for (uint32_t type : {0u, 1u, 30u}) {
                auto converted = field(directory.tensor("fixed" + std::to_string(type)), true);
                converted.name = "fixed" + std::to_string(type);
                converted.bytes = 257 * 64 * 2;
                converted.materialization->format = "BF16";
                converted.materialization->conversion = "gguf-bf16-fixed-rne-v1";
                fixed_fields.push_back(std::move(converted));
            }
            StreamingConfig c; c.enabled=true; c.schema_version=1; c.selection="manual"; c.retention="request";
            c.stages["denoiser"]={"streamed",1,1,0,0,1};
            auto layout = compile_layout(c, bf16);
            MemoryLedger ledger(16384 + 3 * 49152);
            {
                GgufWeightPager pager(lease, bf16, bf16.stages.front(), layout.stages.front(), ledger);
                pager.load_packed(); Weights fixed; pager.load_resident_aliases(fixed);
                for (uint32_t type : {0u, 1u, 30u}) {
                    const auto name="fixed" + std::to_string(type);
                    const auto &tensor=directory.tensor(name);
                    std::vector<std::byte> raw(tensor.bytes), expected(tensor.elements * 2);
                    ensure(::pread(fd.get(),raw.data(),raw.size(),off_t(tensor.file_offset))==ssize_t(raw.size()),"oracle read failed");
                    gguf::decode_cpu_into({raw,type,257,64},{0,257,0,64},
                        {expected,gguf::DecodeDType::bf16,128,2},nullptr,{false});
                    ensure(fixed.at(name).dtype()==mx::bfloat16 &&
                        std::memcmp(fixed.at(name).data<std::byte>(),expected.data(),expected.size())==0,
                        "fixed BF16 conversion differs from scalar RNE oracle");
                }
                const auto metrics=pager.metrics();
                ensure(metrics.source_read_bytes == 257*64*8 && metrics.packed_source_bytes == 3*257*64*2,
                       "fixed BF16 read/output bytes mismatch");
                ensure(ledger.snapshot().peak_committed_bytes<=16384+3*49152,"fixed BF16 conversion retained raw duplicate");
                fixed.clear();
            }
            ensure(ledger.snapshot().storage_bytes==0,"fixed BF16 target escaped release");
            for (const auto *fault : {"profile", "resident", "dtype", "alias"}) {
                auto bad=bf16;
                if (std::string(fault)=="profile")bad.workload.erase("precision");
                if (std::string(fault)=="resident")bad.workload["source_residency"]="packed_resident";
                if (std::string(fault)=="dtype")bad.stages.front().resident_fields.front().materialization->format="F16";
                if (std::string(fault)=="alias")bad.stages.front().resident_fields.front().materialization->conversion="gguf-native-alias-v1";
                bool rejected=false;
                try { GgufWeightPager wrong(lease,bad,bad.stages.front(),layout.stages.front(),ledger); } catch (...) { rejected=true; }
                ensure(rejected,"undeclared fixed BF16 conversion accepted");
            }
        }
        if (argc == 3) {
            const auto path = std::filesystem::canonical(argv[1]);
            ensure(path.filename() == "fixture.gguf" && path.parent_path().filename().string().starts_with("tc-gguf-pager-"),
                   "refusing to mutate a non-owned test fixture");
            d.workload["source_residency"] = "packed_streamed";
            StreamingConfig c; c.enabled=true; c.schema_version=1; c.selection="manual"; c.retention="request";
            c.stages["denoiser"]={"streamed",1,1,0,0,1};
            const auto layout = compile_layout(c, d); MemoryLedger ledger(1ull << 20);
            GgufWeightPager pager(lease,d,d.stages.front(),layout.stages.front(),ledger);
            pager.load_packed(); const auto &pool=layout.stages.front().pools.front(); pager.create_pool(pool);
            const auto &group=layout.stages.front().groups.front();
            tc_stream_slot_ticket_v1 ticket{};ticket.struct_size=sizeof(ticket);ticket.version=1;
            ticket.pool=group.pool;ticket.slot=group.slot;ticket.request_generation=999;ticket.content_generation=1;
            ticket.item={0,0,0,group.id};std::atomic<bool> cancel{false};
            ensure(pager.fill(group,ticket,&cancel)==group.bytes,"owned mutation fixture fill failed");
            const int writable=::open(path.c_str(),O_WRONLY|O_CLOEXEC);
            ensure(writable>=0,"cannot open owned fault fixture");
            const int truncated=::ftruncate(writable,0);::close(writable);
            ensure(truncated==0,"cannot truncate owned fault fixture");
            bool changed=false;try { (void)pager.fill(group,ticket,&cancel); }catch (...) {changed=true;}
            ensure(changed,"changed/short source was accepted");
            bool stale_ready=false;try { (void)pager.bind(group,ticket);stale_ready=true; }catch (...) {}
            ensure(!stale_ready,"changed source retained old Ready contents");pager.destroy_pool(pool.id);
        }
        std::cout<<"PASS GGUF pager Metal: K1/K2/K3 resident/streamed, chunked I/O, two passes, GPU numerics, stale tickets, escaped-view ledger, release\n";
        return 0;
    } catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
}
