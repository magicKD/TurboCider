#include "runtime/streaming/gguf_weight_pager.hpp"
#include "runtime/streaming/context.hpp"
#include <iostream>

namespace {
using namespace tc;
using namespace tc::streaming;
void ensure(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
class Adapter final : public ModelSlotAdapter {
    struct Job { Adapter *owner; const Group *group = nullptr; };
    GgufWeightPager &pager_;
    std::array<Job,2> jobs_;
    Weights bound_;
    Tensor input_;
    uint32_t pool_ = 0;
    uint64_t sequence_ = 0;
  public:
    uint64_t computes = 0;
    std::optional<Tensor> escaped;
    explicit Adapter(GgufWeightPager &p) : pager_(p), jobs_{{{this},{this}}}, input_(mx::ones({1,64},mx::float32)) {}
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
        ensure(argc==2,"fixture required"); configure_streams(); mx::set_cache_limit(0);
        SourceFileIdentity file; file.logical_id="transformer"; file.path=argv[1];
        auto lease=SourceLease::capture_verified({file});
        auto fd=lease->duplicate_fd("transformer");
        auto directory=gguf::read_directory(fd.get(),lease->file("transformer").bytes);
        Descriptor d; d.model="test-gguf"; d.checkpoint_identity=std::string(lease->artifact_digest());
        d.backend_revision="gguf-pager-test-v1"; d.workload={{"shape","4x64"}};
        d.artifacts.push_back({"transformer",lease->file("transformer").content_digest,lease->file("transformer").bytes,SourceIdentityKind::content_sha256});
        StageDescriptor stage; stage.id="denoiser"; stage.adapter_revision="gguf-pager-test-v1";
        stage.max_slots=2; stage.pass_count=2; stage.passes={{0,"test",{4,64}},{1,"test",{4,64}}};
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
        for(uint32_t slots=1;slots<=2;++slots) {
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
                metrics.maximum_dense_pool_capacity_bytes==slots*GgufWeightPager::buffer_alignment,"capacity accounting mismatch");
            pager.reset(); fixed.clear();
            ensure(ledger.snapshot().storage_bytes>0,"escaped MLX view lost physical storage claim");
            adapter->escaped.reset(); adapter.reset();
            gathered = mx::zeros({1}, mx::bfloat16); expected = mx::zeros({1}, mx::bfloat16);
            mx::synchronize(); mx::clear_cache();
            ensure(ledger.snapshot().storage_bytes==0 && ledger.snapshot().reserved_bytes==0,"buffer/ledger release leak");
        }
        std::cout<<"PASS GGUF pager Metal: K1/K2, two passes, GPU numerics, stale tickets, escaped-view ledger, release\n";
        return 0;
    } catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
}
