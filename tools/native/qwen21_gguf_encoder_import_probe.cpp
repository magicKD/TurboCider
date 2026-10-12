#include "models/qwen21/gguf_weights.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
using namespace tc::streaming;

int main(int argc,char **argv) {try {
    require(argc==4,"usage: qwen21-gguf-encoder-import-probe original-text-gguf workers packed|dense");
    const auto worker=std::string_view(argv[2]),mode=std::string_view(argv[3]);
    require(worker.size()==1 && worker[0]>='1' && worker[0]<='8',"workers must be 1..8");
    require(mode=="packed" || mode=="dense","embedding mode must be packed or dense");
    configure_streams();mx::set_cache_limit(0);
    auto elapsed=[](auto start){return std::chrono::duration<double>(Clock::now()-start).count();};
    const auto start=Clock::now();SourceFileIdentity source;source.logical_id="text";source.path=argv[1];
    auto lease=SourceLease::capture_verified({source});const double verify=elapsed(start);
    require(lease->verification_bytes_read()==lease->file("text").bytes && lease->verification_cache_hits()==0,
        "fresh native whole-file proof required, not a serialized/warm verification cache");
    MemoryLedger ledger(uint64_t(16)<<30);Weights weights;
    GgufKImportOptions options;options.enabled=true;options.floating_dtype=mx::float16;
    options.decode_workers=uint32_t(worker[0]-'0');
    options.include_tensor=[](std::string_view key){return !key.starts_with("output.");};
    GgufPackedBank bank(lease,"text",ledger,uint64_t(1)<<20,true,0,6,options);
    bank.load(weights);weights.materialize();const auto metrics=bank.metrics();
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
    bank.check_unchanged();
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen21-encoder-import-screen-v1\",\"qualification_passed\":false"
        <<",\"scope\":\"complete native full-SHA and mixed-K encoder import plus 13 original token rows; no encoder-layer/DiT timing\""
        <<",\"source_sha256\":\""<<metrics.source_sha256<<"\",\"plan_digest\":\""<<metrics.plan_digest
        <<"\",\"workers\":"<<metrics.decode_workers<<",\"embedding_mode\":\""<<mode<<"\""
        <<",\"verification_bytes\":"<<lease->verification_bytes_read()<<",\"source_read_bytes\":"<<metrics.source_read_bytes
        <<",\"packed_capacity_bytes\":"<<metrics.packed_capacity_bytes<<",\"read_buffer_capacity_bytes\":"<<metrics.read_buffer_capacity_bytes
        <<",\"managed_peak_bytes\":"<<metrics.managed_peak_bytes<<",\"final_weight_logical_bytes\":"<<weights.bytes()
        <<",\"native_verification_seconds\":"<<verify<<",\"bank_load_seconds\":"<<metrics.load_seconds
        <<",\"bank_read_seconds\":"<<metrics.read_seconds<<",\"bank_decode_seconds\":"<<metrics.decode_seconds
        <<",\"embedding_setup_seconds\":"<<embedding_setup<<",\"token_gather_seconds\":"<<gather
        <<",\"total_observed_seconds\":"<<elapsed(start)<<",\"mlx_logical_peak_bytes\":"<<mx::get_peak_memory()<<"}\n";
    rows=mx::zeros({1});weights.clear();mx::synchronize();mx::clear_cache();
    require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"import probe leaked affine source backing");
    return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
