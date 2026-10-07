#include "runtime/streaming/gguf_packed_bank.hpp"
#include <iostream>
#include <fcntl.h>
#include <thread>
#include <cstring>

namespace {
using namespace tc;
using namespace tc::streaming;
void ensure(bool value,const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
    bool failed=false;try { function(); } catch (const std::exception &) { failed=true; }
    ensure(failed,"invalid packed bank operation accepted");
}
std::shared_ptr<const SourceLease> lease(const char *path) {
    SourceFileIdentity file;file.logical_id="weights";file.path=path;
    return SourceLease::capture_verified({std::move(file)});
}
}
int main(int argc,char **argv) {
    try {
        ensure(argc==6,"five owned fixtures required");configure_streams();mx::set_cache_limit(0);
        auto source=lease(argv[1]);
        const auto content=source->file("weights").content_digest;
        Weights golden;golden.load_gguf_file(argv[1]);golden.materialize();
        // Reference preparation changes ctime on this fresh macOS fixture.
        // Rebind AFTER reference preparation and verify unchanged content;
        // do not weaken generation checks or inherit the old native proof.
        source=lease(argv[1]);ensure(source->file("weights").content_digest==content,"reference changed source bytes");
        source->revalidate_after_drain();
        const auto keys=golden.sorted_keys();ensure(keys.size()==12,"wrong fixture field count");
        MemoryLedger ledger(32ull<<20);Weights weights;
        auto bank=std::make_unique<GgufPackedBank>(source,"weights",ledger,16384);
        ensure(!ledger.snapshot().storage_bytes,"metadata-only bank allocated payload backing");
        const auto upper=bank->metrics().planned_packed_capacity_bytes;
        {
            MemoryLedger old_budget(32ull<<20);Weights legacy;
            GgufPackedBank old(source,"weights",old_budget,16384,false);old.load(legacy);
            ensure(old.metrics().plan_digest!=bank->metrics().plan_digest,"packing policy absent from plan digest");
            ensure(old.metrics().affine_packing_recipe=="legacy-affine-three-pass-v1","wrong legacy packing receipt");
            for(const auto &key:keys) {
                const auto &a=legacy.at(key),&b=golden.at(key);
                ensure(a.shape()==b.shape() && a.dtype()==b.dtype() &&
                    std::memcmp(a.data<std::byte>(),b.data<std::byte>(),a.nbytes())==0,"legacy bank field bits changed");
            }
            legacy.clear();mx::synchronize();
            ensure(!old_budget.snapshot().storage_bytes,"legacy control retained model payload");
        }
        {
            MemoryLedger too_small(upper+16384-1);Weights untouched;
            GgufPackedBank floor(source,"weights",too_small,16384);
            rejects([&]{floor.load(untouched);});
            ensure(!too_small.snapshot().storage_bytes && untouched.sorted_keys().empty(),"floor allocated or published partial bank");
        }
        bank->load(weights);
        ensure(weights.sorted_keys()==keys,"native packed binding keys changed");
        for (const auto &key : keys) {
            const auto &actual=weights.at(key), &expected=golden.at(key);
            ensure(actual.dtype()==expected.dtype() && actual.shape()==expected.shape(),"native packed field geometry/dtype changed");
            ensure(std::memcmp(actual.data<std::byte>(),expected.data<std::byte>(),actual.nbytes())==0,"native packed field bits differ");
        }
        const auto metrics=bank->metrics();
        ensure(metrics.affine_packing_recipe=="fused-affine-one-pass-v1","wrong fused packing receipt");
        ensure(metrics.tensor_count==6 && metrics.field_count==12 && metrics.plan_digest.size()==64,"wrong bank identity/counts");
        ensure(metrics.source_read_bytes==metrics.logical_source_bytes,"source reread for affine parts or hidden packed cache");
        ensure(metrics.read_buffer_capacity_bytes==16384 && metrics.managed_peak_bytes<=upper+16384,"unplanned import backing");
        ensure(ledger.snapshot().storage_bytes==metrics.packed_capacity_bytes,"read buffer remained resident after import");
        rejects([&]{bank->load(weights);});
        std::thread intruder([&]{rejects([&]{bank->check_unchanged();});});intruder.join();
        // A lazy GPU projection must retain all packed field claims even after
        // the producer and dictionary are destroyed. No early uncharged free.
        auto input=mx::ones({2,64},mx::float32);
        auto expected=linear(input,golden,"q8");mx::eval(expected);
        auto escaped=linear(input,weights,"q8");bank.reset();weights.clear();
        ensure(ledger.snapshot().storage_bytes>0,"lazy reader lost packed storage claims");
        mx::eval(escaped);ensure(mx::all(escaped==expected).item<bool>(),"escaped lazy GPU projection changed");
        escaped=mx::zeros({1});mx::synchronize();mx::clear_cache();
        ensure(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"escaped bank release leaked ledger claims");
        for (bool final_cancel : {false,true}) {
            std::atomic<bool> cancel{false};Weights partial;MemoryLedger budget(32ull<<20);
            GgufPackedBank cancelled_bank(source,"weights",budget,16384);
            rejects([&]{cancelled_bank.load(partial,&cancel,[&](const std::string &,int current,int total){
                if (current==(final_cancel ? total : 1)) cancel.store(true);
            });});
            ensure(partial.sorted_keys().empty() && !budget.snapshot().storage_bytes && !budget.snapshot().reserved_bytes,
                   "cancel published partial/final bank or leaked backing");
            cancel.store(false);rejects([&]{cancelled_bank.load(partial,&cancel);});
            GgufPackedBank retry(source,"weights",budget,16384);retry.load(partial,&cancel);partial.clear();
            ensure(!budget.snapshot().storage_bytes,"new-bank retry leaked backing");
        }
        // Nonfinite float conversion fails transactionally; output alias
        // collisions, unsupported K and oversized source rows fail metadata.
        for (int i=2;i<argc;++i) {
            MemoryLedger budget(32ull<<20);Weights partial;
            rejects([&]{GgufPackedBank invalid(lease(argv[i]),"weights",budget,16384);invalid.load(partial);});
            ensure(partial.sorted_keys().empty() && !budget.snapshot().storage_bytes,"invalid source published output or allocated leaked backing");
        }
        {
            auto metadata=SourceLease::capture({source->file("weights")});MemoryLedger budget(32ull<<20);
            rejects([&]{GgufPackedBank unverified(metadata,"weights",budget,16384);});
        }
        // Raw source window is separate from affine GPU masters. One cache
        // slot makes eviction deterministic; escaped readers stay charged.
        {
            MemoryLedger budget(32ull<<20);Weights affine;
            auto raw_bank=std::make_unique<GgufPackedBank>(source,"weights",budget,16384,true,98304,1);
            raw_bank->load(affine);
            auto q8=raw_bank->raw_matrix("q8.weight");
            ensure(q8.type==8 && q8.columns==64 && q8.values.dtype()==mx::uint8 &&
                q8.values.shape()==mx::Shape{512,68},"raw window returned affine/logical dense layout");
            const auto id=q8.values.data_shared_ptr();
            const auto content_id=q8.logical_content_identity;ensure(bool(content_id),"verified raw source has no logical identity");
            ensure(raw_bank->raw_matrix("q8.weight").values.data_shared_ptr()==id,"raw cache hit changed generation");
            {auto q4=raw_bank->raw_matrix("q4.weight");ensure(q4.type==2 && q4.values.shape()==mx::Shape{512,36},"raw Q4 geometry changed");}
            auto m=raw_bank->metrics();
            ensure(m.raw_window_entries==1 && m.raw_window_hits==1 && m.raw_window_misses==2 && m.raw_window_evictions==1 &&
                m.raw_window_source_read_bytes==512*(68+36) && m.raw_window_live_bytes==81920 &&
                m.raw_window_peak_bytes<=98304,"raw source traffic/cache/escaped accounting mismatch");
            rejects([&]{raw_bank->raw_matrix("q41.weight");});
            rejects([&]{raw_bank->raw_matrix("f32");});
            std::atomic<bool> cancel{true};rejects([&]{raw_bank->raw_matrix("q4.weight",&cancel);});
            std::thread intruder([&]{rejects([&]{raw_bank->raw_matrix("q8.weight");});});intruder.join();
            raw_bank->clear_raw_window();
            ensure(raw_bank->metrics().raw_window_live_bytes==49152,"raw eviction lost escaped owner's claim");
            {
                auto refill=raw_bank->raw_matrix("q8.weight");
                ensure(refill.values.data_shared_ptr()!=id && refill.logical_content_identity==content_id,
                    "raw refill changed immutable content identity or reused an escaped physical backing");
            }
            raw_bank->clear_raw_window();
            auto *data=q8.values.data<uint8_t>();ensure(data[0]==0 && data[1]==0x30,"raw source payload changed");
            raw_bank.reset();affine.clear();
            ensure(data[0]==0 && data[1]==0x30,"raw reader died with producer");
            // Allocation admission includes a live evicted reader, even when
            // the cache is empty. Releasing it permits a clean refill.
            GgufPackedBank floor(source,"weights",budget,16384,true,49152,1);floor.load(affine);
            auto held=floor.raw_matrix("q8.weight");
            ensure(held.logical_content_identity!=content_id,"new bank/source proof reused an old logical content identity");
            rejects([&]{floor.raw_matrix("q4.weight");});
            ensure(floor.metrics().raw_window_entries==0 && floor.metrics().raw_window_live_bytes==49152,
                "raw floor published partial destination or forgot escaped claim");
            held.values=mx::zeros({1},mx::uint8);mx::synchronize();
            ensure(!floor.metrics().raw_window_live_bytes,"content identity retained packed payload after reader release");
            ensure(floor.raw_matrix("q4.weight").type==2,"raw source clean refill failed");
            floor.clear_raw_window();ensure(!floor.metrics().raw_window_live_bytes,"raw refill leaked storage");
        }
        // Only the test-owned file is changed; model fixtures are never writable.
        const auto path=std::filesystem::canonical(argv[1]);
        ensure(path.filename()=="fixture.gguf" && path.parent_path().filename().string().starts_with("tc-gguf-bank-"),
               "refusing to mutate a non-owned source");
        {
            MemoryLedger budget(32ull<<20);Weights partial;GgufPackedBank changed(source,"weights",budget,16384);
            MemoryLedger raw_budget(32ull<<20);Weights raw_affine;
            GgufPackedBank raw_changed(source,"weights",raw_budget,16384,true,98304,1);
            raw_changed.load(raw_affine);raw_changed.raw_matrix("q8.weight");
            rejects([&]{changed.load(partial,nullptr,[&](const std::string &,int current,int){
                if (current!=1) return;
                const int fd=::open(path.c_str(),O_WRONLY|O_CLOEXEC);ensure(fd>=0,"cannot open owned fault fixture");
                const int status=::ftruncate(fd,0);::close(fd);ensure(!status,"cannot truncate owned fixture");
            });});
            ensure(partial.sorted_keys().empty() && !budget.snapshot().storage_bytes,"source change published partial bank");
            rejects([&]{changed.check_unchanged();});
            rejects([&]{raw_changed.raw_matrix("q8.weight");}); // even a cached hit revalidates generation
        }
        std::cout<<"PASS GGUF packed bank Metal: native field/projection exact, bounded chunks/claims, cancellation/floors/source change\n";
        return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
