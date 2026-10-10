#include "models/qwen21/gguf_weights.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
using namespace tc::streaming;

int main(int argc,char **argv) {try {
    require(argc>=4 && argc<=7,"usage: qwen21-gguf-encoder-import-probe original-text-gguf workers packed|dense [--verify-overlap|--registered [1|4|8 [--read-ahead]]]");
    const bool registered=argc>=5 && std::string_view(argv[4])=="--registered";
    const bool overlap=argc>=5 && !registered;
    require(!overlap || (argc==5 && std::string_view(argv[4])=="--verify-overlap"),"invalid verification option");
    const auto read_mib=argc>=6 ? std::string_view(argv[5]) : std::string_view("1");
    const bool ahead=argc==7;
    require(!ahead || std::string_view(argv[6])=="--read-ahead","invalid read schedule");
    require(registered || argc<6,"read buffer screen requires registered source");
    require(read_mib=="1" || read_mib=="4" || read_mib=="8","read buffer must be 1,4 or8 MiB");
    const uint64_t read_bytes=uint64_t(read_mib[0]-'0')<<20;
    const auto worker=std::string_view(argv[2]),mode=std::string_view(argv[3]);
    require(worker.size()==1 && worker[0]>='1' && worker[0]<='8',"workers must be 1..8");
    require(mode=="packed" || mode=="dense","embedding mode must be packed or dense");
    configure_streams();mx::set_cache_limit(0);
    auto elapsed=[](auto start){return std::chrono::duration<double>(Clock::now()-start).count();};
    const auto start=Clock::now();SourceFileIdentity source;source.logical_id="text";source.path=argv[1];
    MemoryLedger ledger(uint64_t(16)<<30);Weights weights;
    GgufKImportOptions options;options.enabled=true;options.floating_dtype=mx::float16;
    options.decode_workers=uint32_t(worker[0]-'0');
    options.read_ahead=ahead;
    options.include_tensor=[](std::string_view key){return !key.starts_with("output.");};
    std::shared_ptr<const SourceLease> lease;std::unique_ptr<GgufPackedBank> bank;double verify=0;
    if(overlap) {
        PendingSourceVerification pending({source});
        bank=GgufPackedBank::load_verifying(pending,"text",ledger,weights,uint64_t(1)<<20,options,
            [](const GgufPackedBankMetrics &m) {
                const auto observed=ane::observe_runtime_memory(mx::get_active_memory());
                const auto admission=ane::admit_memory(observed,{uint64_t(4)<<30,observed.physical_bytes},0,
                    m.planned_packed_capacity_bytes+(uint64_t(2)<<20));
                require(admission.allowed(),"probe concurrent import memory admission declined");
            });
        lease=pending.finish();verify=pending.verification_seconds();
    } else {
        lease=registered ? SourceLease::capture_registered({source}) : SourceLease::capture_verified({source});verify=elapsed(start);
        bank=std::make_unique<GgufPackedBank>(lease,"text",ledger,read_bytes,true,0,6,options);
        bank->load(weights);
    }
    require(registered ? (lease->verification_bytes_read()==0 && lease->registration_cache_hits()==1) :
        (lease->verification_bytes_read()==lease->file("text").bytes && lease->verification_cache_hits()==0),
        "actual initial full-SHA or registered generation-bound source proof required");
    weights.materialize();const auto metrics=bank->metrics();
    const auto before_embedding=Clock::now();
    if(mode=="dense")weights.dequantize({"token_embd"});
    const double embedding_setup=elapsed(before_embedding);
    const int vocabulary=weights.at("token_embd.weight").shape(0);
    std::vector<int> token_ids{151644,8948,198,151645,151644,872,198,40,151645,151644,77091,198,40};
    for(int id:token_ids)require(id>=0 && id<vocabulary,"test token out of original vocabulary");
    Tensor ids(token_ids.data(),{int(token_ids.size())},mx::int32);
    const auto gather_start=Clock::now();auto rows=weights.embedding_rows(ids,"token_embd");mx::eval(rows);
    const double gather=elapsed(gather_start);
    require(rows.shape()==mx::Shape{int(token_ids.size()),4096} && rows.dtype()==mx::float16 &&
        mx::all(mx::isfinite(rows)).item<bool>(),"gathered original embeddings shape/dtype/finite failed");
    bank->check_unchanged();
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen21-encoder-import-screen-v1\",\"qualification_passed\":false"
        <<",\"scope\":\"complete native full-SHA and mixed-K encoder import plus 13 original token rows; no encoder-layer/DiT timing\""
        <<",\"source_sha256\":\""<<metrics.source_sha256<<"\",\"plan_digest\":\""<<metrics.plan_digest
        <<"\",\"workers\":"<<metrics.decode_workers<<",\"embedding_mode\":\""<<mode<<"\""
        <<",\"verification_bytes\":"<<lease->verification_bytes_read()<<",\"source_read_bytes\":"<<metrics.source_read_bytes
        <<",\"registration_cache_hits\":"<<lease->registration_cache_hits()
        <<",\"packed_capacity_bytes\":"<<metrics.packed_capacity_bytes<<",\"read_buffer_capacity_bytes\":"<<metrics.read_buffer_capacity_bytes
        <<",\"read_buffer_slots\":"<<metrics.read_buffer_slots<<",\"read_ahead_enabled\":"<<(metrics.read_ahead_enabled?"true":"false")
        <<",\"read_ahead_drained\":"<<(metrics.read_ahead_drained?"true":"false")
        <<",\"read_ahead_jobs\":"<<metrics.read_ahead_jobs<<",\"read_ahead_bytes\":"<<metrics.read_ahead_bytes
        <<",\"read_wait_seconds\":"<<metrics.read_wait_seconds
        <<",\"managed_peak_bytes\":"<<metrics.managed_peak_bytes<<",\"final_weight_logical_bytes\":"<<weights.bytes()
        <<",\"native_verification_seconds\":"<<verify<<",\"bank_load_seconds\":"<<metrics.load_seconds
        <<",\"verification_overlap\":"<<(overlap ? "true" : "false")
        <<",\"source_verified_before_publication\":true,\"verified_import_seconds\":"<<metrics.verified_import_seconds
        <<",\"bank_read_seconds\":"<<metrics.read_seconds<<",\"bank_decode_seconds\":"<<metrics.decode_seconds
        <<",\"embedding_setup_seconds\":"<<embedding_setup<<",\"token_gather_seconds\":"<<gather
        <<",\"total_observed_seconds\":"<<elapsed(start)<<",\"mlx_logical_peak_bytes\":"<<mx::get_peak_memory()<<"}\n";
    rows=mx::zeros({1});weights.clear();mx::synchronize();mx::clear_cache();
    require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"import probe leaked affine source backing");
    return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
