#include "components/text/qwen3_gguf.hpp"
#include <fstream>
#include <iostream>

using namespace tc;
void insist(bool value,const char *message) {if (!value) throw std::runtime_error(message);}
template<class F> void rejects(F action,const char *message) {
    bool rejected=false;try {action();} catch (const std::exception &) {rejected=true;}
    insist(rejected,message);
}
int main(int argc,char **argv) {
    try {
        insist(argc==5,"GGUF/config/tokenizer and owned temporary fixture directory required");
        configure_streams();mx::set_cache_limit(0);const auto initial=mx::get_active_memory();
        const std::filesystem::path gguf(argv[1]),cfg(argv[2]),tok(argv[3]),folder(argv[4]);
        const auto config=folder/"config.json",tokenizer=folder/"tokenizer.json",link=folder/"config-link.json";
        std::filesystem::copy_file(cfg,config);std::filesystem::create_symlink(std::filesystem::absolute(tok),tokenizer);
        std::filesystem::create_symlink(config,link);
        std::atomic<bool> cancel{false};
        auto source=components::Qwen3GgufPreparedSource::prepare(gguf,link,tokenizer,cancel);
        insist(source->lease()->has_verified_content(),"unverified metadata published");
        insist(source->reusable_for(gguf,link,tokenizer),"unchanged source metadata not reusable");
        const auto identity=source->execution_identity(2,1ull<<30,{},"packed_streamed");
        auto a=source->tokenize("An adult ceramic artist holding a blue cup.",true);
        auto b=source->tokenize("A clear glass pitcher beside a red apple.",true);
        auto again=source->tokenize("An adult ceramic artist holding a blue cup.",true);
        insist(a.ids==again.ids && a.valid==again.valid && a.ids!=b.ids,"metadata tokenizer A/B/A changed");
        {
            components::Qwen3GgufEncoder encoder(source,2,1ull<<30,{},cancel,{},"packed_streamed");
            insist(encoder.identity()==identity,"prepared source changed execution identity");
        }
        insist(mx::get_active_memory()==initial,"metadata/cache constructed GPU weight or execution buffers");
        rejects([&]{source->execution_identity(3,1ull<<30,{},"packed_streamed");},"p3 bypassed cached metadata policy");
        rejects([&]{source->execution_identity(2,1ull<<30,{},"unknown");},"bad residency bypassed cached metadata policy");
        setenv("TURBOCIDER_QWEN3_EVAL_INTERVAL","4",1);
        rejects([&]{source->execution_identity(2,1ull<<30,{},"packed_streamed");},"lazy policy bypassed metadata cache hit");
        unsetenv("TURBOCIDER_QWEN3_EVAL_INTERVAL");
        insist(source->execution_identity(2,1ull<<30,{},"packed_streamed")==identity,"rejected policy corrupted metadata source");
        cancel.store(true);
        rejects([&]{components::Qwen3GgufPreparedSource::prepare(gguf,link,tokenizer,cancel);},"cancelled metadata prepared");
        rejects([&]{components::Qwen3GgufEncoder encoder(source,2,1ull<<30,{},cancel,{},"packed_streamed");},"cancelled encoder started from cache");
        cancel.store(false);
        {std::ofstream append(config,std::ios::app);append<<' ';}
        insist(!source->reusable_for(gguf,link,tokenizer),"same-path changed config reused old proof");
        rejects([&]{source->tokenize("stale source",true);},"stale source tokenizer accepted");
        auto changed=components::Qwen3GgufPreparedSource::prepare(gguf,link,tokenizer,cancel);
        insist(changed->execution_identity(2,1ull<<30,{},"packed_streamed")!=identity,"new config content retained old identity");
        const auto other=folder/"other-config.json";std::filesystem::copy_file(config,other);
        std::filesystem::remove(link);std::filesystem::create_symlink(other,link);
        insist(!changed->reusable_for(gguf,link,tokenizer),"retargeted symlink retained old proof");
        const auto wrong=folder/"wrong-config.json";
        {std::ofstream out(wrong);out<<"{}";}
        rejects([&]{components::Qwen3GgufPreparedSource::prepare(gguf,wrong,tokenizer,cancel);},"malformed replacement config was trusted");
        source.reset();changed.reset();
        insist(mx::get_active_memory()==initial,"metadata-only retention leaked GPU resources");
        std::cout<<"PASS Qwen3 metadata: no GPU allocations, verified reuse, A/B/A, policy/cancel gates, file mutation and symlink rebind\n";
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
