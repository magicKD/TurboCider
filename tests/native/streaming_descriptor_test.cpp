#include "streaming/layout.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace tc;
using namespace tc::streaming;

static StreamingConfig config() {
    StreamingConfig c;
    c.enabled=true; c.schema_version=1; c.selection="manual"; c.retention="request";
    c.stages["denoiser"]={"streamed",1,1,1,0,1};
    return c;
}
static Descriptor metadata() {
    Descriptor d{"fixture", "checkpoint-snapshot", "fake-reader-v2", {}};
    d.artifacts.push_back({"weights", "snapshot-size-and-index", 1024});
    d.workload={{"shape","2x4"},{"operation","fixture"}};
    StageDescriptor s;
    s.id="denoiser"; s.adapter_revision="adapter-v2"; s.pass_count=2;
    s.passes={{7,"stage1",{2,4}},{9,"stage2",{4,4}}};
    for (uint32_t b=0; b<3; ++b) {
        BlockSpec block; block.id=b; block.layout_class="table";
        // Two independent BF16 destinations read the SAME F32 source twice.
        Materialization base{"bf16", "cpu", "f32-to-bf16-rne", {4,4},
            {{0, uint64_t(b)*64, 64, "table-"+std::to_string(b), "f32", {4,4}}}, "", 0};
        Materialization row{"bf16", "metal-shared", "copy", {4}, {}, "base", 8};
        block.fields={{"base", "base-"+std::to_string(b),32,64,base},
                      {"copy", "copy-"+std::to_string(b),32,64,base},
                      {"row", "row-"+std::to_string(b),8,64,row}};
        s.blocks.push_back(std::move(block));
    }
    d.stages.push_back(std::move(s)); return d;
}
template<class F> static void rejects(F fn, const char *part) {
    try { fn(); } catch (const std::invalid_argument &e) {
        if (std::string(e.what()).find(part)!=std::string::npos) return;
        std::cerr<<e.what()<<" expected "<<part<<'\n'; std::abort();
    }
    std::cerr<<"expected rejection "<<part<<'\n'; std::abort();
}
int main() {
    const auto d=metadata(); const auto c=config(); const auto p=compile_layout(c,d);
    const auto &s=p.stages[0];
    assert(p.materializations_complete);
    assert(s.prefix_bytes==192 && s.peak_pool_bytes==192);
    assert(s.groups[0].bytes==72 && s.suffix_content_bytes_per_pass==144);
    assert(s.prefix_source_read_bytes==128 && s.source_read_bytes_per_pass==256);
    assert(compile_layout(c,d).canonical==p.canonical);
    auto x=d; x.stages[0].blocks[2].fields[0].materialization->reads[0].offset++;
    assert(compile_layout(c,x).digest!=p.digest);
    x=d; x.workload["shape"]="4x4"; assert(compile_layout(c,x).digest!=p.digest);
    x=d; x.stages[0].passes[0].step++; assert(compile_layout(c,x).digest!=p.digest);
    x=d; x.stages[0].passes[0].shape[0]++; assert(compile_layout(c,x).digest!=p.digest);
    x=d; x.artifacts[0].identity="another-snapshot"; assert(compile_layout(c,x).digest!=p.digest);
    x=d; x.stages[0].passes.clear(); assert(!compile_layout(c,x).materializations_complete);
    x=d; x.workload.clear(); assert(!compile_layout(c,x).materializations_complete);
    x=d; x.stages[0].passes.pop_back(); rejects([&]{compile_layout(c,x);},"pass template");
    x=d; x.stages[0].blocks[0].fields[0].materialization->reads[0].offset=UINT64_MAX;
    rejects([&]{compile_layout(c,x);},"source range");
    x=d; x.stages[0].blocks[0].fields[0].materialization->reads[0].artifact=1;
    rejects([&]{compile_layout(c,x);},"unknown source");
    x=d; x.stages[0].blocks[0].fields[0].materialization->reads[0].bytes=0;
    rejects([&]{compile_layout(c,x);},"source range");
    x=d; x.stages[0].blocks[0].fields[2].materialization->derived_from="row";
    rejects([&]{compile_layout(c,x);},"earlier materialized");
    x=d; x.stages[0].blocks[0].fields[2].materialization->derived_offset=32;
    rejects([&]{compile_layout(c,x);},"derived range");
    x=d; x.stages[0].blocks[0].fields[2].materialization->reads=d.stages[0].blocks[0].fields[0].materialization->reads;
    rejects([&]{compile_layout(c,x);},"reads OR derivation");
    x=d; x.stages[0].blocks[0].fields[0].materialization->shape={UINT64_MAX,2};
    rejects([&]{compile_layout(c,x);},"shape overflow");
    x=d; x.stages[0].blocks[1].fields[0].materialization->shape={2,8};
    rejects([&]{compile_layout(c,x);},"class materialization");
    x=d; x.artifacts[0].identity_kind=SourceIdentityKind::content_sha256;
    rejects([&]{compile_layout(c,x);},"SHA256");
    x.artifacts[0].identity=std::string(64,'a'); assert(compile_layout(c,x).digest!=p.digest);
    x=d;
    for (auto &b : x.stages[0].blocks) {
        b.fields.resize(1); b.fields[0].materialization.reset();
    }
    auto unknown=compile_layout(c,x);
    assert(!unknown.materializations_complete && !unknown.stages[0].source_read_bytes_per_pass);
    assert(!unknown.stages[0].prefix_source_read_bytes);
    // Prefix aliases share allocation and reads only when construction matches.
    x=d; x.stages[0].blocks[0].fields[1].storage_id=x.stages[0].blocks[0].fields[0].storage_id;
    auto alias=compile_layout(c,x);
    assert(alias.stages[0].prefix_bytes==128 && alias.stages[0].prefix_source_read_bytes==64);
    x.stages[0].blocks[0].fields[1].materialization->reads[0].offset++;
    rejects([&]{compile_layout(c,x);},"alias materialization");
    // A huge logical source read repeated across passes must overflow safely.
    x=d; x.artifacts[0].bytes=UINT64_MAX;
    for (auto &b : x.stages[0].blocks) {
        b.fields.resize(1);
        auto &r=b.fields[0].materialization->reads[0];r.offset=0;r.bytes=UINT64_MAX/3;
    }
    rejects([&]{compile_layout(c,x);},"source pass byte count overflow");
    std::cout<<"PASS streaming descriptor: source/destination/derived/padding accounting, identity, ranges, aliases and passes\n";
}
