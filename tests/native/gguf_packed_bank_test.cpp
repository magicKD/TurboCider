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
        // Only the test-owned file is changed; model fixtures are never writable.
        const auto path=std::filesystem::canonical(argv[1]);
        ensure(path.filename()=="fixture.gguf" && path.parent_path().filename().string().starts_with("tc-gguf-bank-"),
               "refusing to mutate a non-owned source");
        {
            MemoryLedger budget(32ull<<20);Weights partial;GgufPackedBank changed(source,"weights",budget,16384);
            rejects([&]{changed.load(partial,nullptr,[&](const std::string &,int current,int){
                if (current!=1) return;
                const int fd=::open(path.c_str(),O_WRONLY|O_CLOEXEC);ensure(fd>=0,"cannot open owned fault fixture");
                const int status=::ftruncate(fd,0);::close(fd);ensure(!status,"cannot truncate owned fixture");
            });});
            ensure(partial.sorted_keys().empty() && !budget.snapshot().storage_bytes,"source change published partial bank");
            rejects([&]{changed.check_unchanged();});
        }
        std::cout<<"PASS GGUF packed bank Metal: native field/projection exact, bounded chunks/claims, cancellation/floors/source change\n";
        return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
