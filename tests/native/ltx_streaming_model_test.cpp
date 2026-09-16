// Real-checkpoint integration smoke, not a release performance campaign.
#include "streaming/layout.hpp"
#include "ltx_streaming_descriptor.hpp"
extern "C" {
#include "ltx_native.h"
#include "ltx_streaming_layout.h"
#include "ltx.h"
}
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
static uint16_t bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return uint16_t(bits >> 16);
}
static void checked(int ok, const char *error) { if (!ok) throw std::runtime_error(error); }
static void finite(const std::vector<uint16_t> &v) {
    for (auto value : v) { uint32_t bits=uint32_t(value)<<16; float f; std::memcpy(&f,&bits,4); assert(std::isfinite(f)); }
}
struct Output {
    std::vector<uint16_t> connected_video, connected_audio, connected_mask;
    std::vector<uint16_t> video1, audio1, upsampled, video2, audio2;
};
static int stop_at_block(const char *phase, int current, int, void *user) {
    return std::strcmp(phase,"ltx_block")==0 && current>=*static_cast<int *>(user);
}

static int create_exact(const ltx_native_options *options,
                        const tc_stream_stage_plan_v1 &plan, uint32_t prefix,
                        const tc::ltx::StreamingMetadata *metadata, ltx_native_denoiser **out,
                        char *error, size_t size) {
    const ltx_native_streaming_options_v1 v1{sizeof(v1), 1, prefix, &plan};
    if (!metadata) return ltx_native_create_streamed_v1(options, &v1, out, nullptr, nullptr, error, size);
    metadata->check_unchanged();
    const ltx_native_streaming_options_v2 v2{sizeof(v2), 2, v1, &metadata->header(), &metadata->mapping()};
    return ltx_native_create_streamed_v2(options, &v2, out, nullptr, nullptr, error, size);
}
static void snapshot_alive(const tc::ltx::StreamingMetadata *metadata) {
    if (metadata) {
        metadata->check_unchanged();
        assert(fcntl(metadata->mapping().descriptor, F_GETFD) >= 0);
    }
}

static void cancellation_case(const char *checkpoint, const char *shader,
                              const tc_stream_stage_plan_v1 &plan, uint32_t prefix,
                              const tc::ltx::StreamingMetadata *metadata) {
    char error[1024]={}; ltx_native_options options{};
    options.checkpoint=checkpoint; options.shader_source=shader;
    options.width=64; options.height=64; options.frames=9; options.fps=24;
    options.parallel_av=1; options.batch_audio_commands=1;
    ltx_native_denoiser *ctx=nullptr;
    // Refuse incompatible intent before loading any weights.
    options.memory_budget_bytes=1;
    assert(!create_exact(&options,plan,prefix,metadata,&ctx,error,sizeof(error)) && !ctx);
    options.memory_budget_bytes=0;
    auto wrong=plan; wrong.pass_count=1;
    assert(!create_exact(&options,wrong,prefix,metadata,&ctx,error,sizeof(error)) && !ctx);
    // Shader failure exercises native construction cleanup after metadata/map
    // validation; the borrowed snapshot must remain usable for a later create.
    options.shader_source="missing-ltx-snapshot-test-shader.metal";
    assert(!create_exact(&options,plan,prefix,metadata,&ctx,error,sizeof(error)) && !ctx);
    snapshot_alive(metadata);
    options.shader_source=shader;
    checked(create_exact(&options,plan,prefix,metadata,&ctx,error,sizeof(error)),error);
    std::thread foreign([&]{
        auto *copy=ctx;char message[1024]={};
        assert(!ltx_native_streaming_destroy(&copy,message,sizeof(message)) && copy==ctx);
        assert(std::strstr(message,"owner thread"));
    });foreign.join();
    ltx_workload w{};checked(ltx_workload_init(&w,64,64,9,24,error,sizeof(error)),error);
    std::vector<uint16_t> video(w.stage1_video_tokens*128,bf16(0.125f));
    std::vector<uint16_t> audio(size_t(w.audio_tokens)*128,bf16(0.25f));
    const auto original_video=video,original_audio=audio;
    std::vector<uint16_t> textv(16*4096),texta(16*2048),mask(16);
    int stop=int(prefix)+2;
    assert(!ltx_native_run(ctx,1,42,video.data(),video.size(),audio.data(),audio.size(),textv.data(),texta.data(),mask.data(),16,
                          nullptr,1,stop_at_block,&stop,error,sizeof(error)));
    assert(video==original_video && audio==original_audio);
    assert(!ltx_native_run(ctx,1,42,video.data(),video.size(),audio.data(),audio.size(),textv.data(),texta.data(),mask.data(),16,
                          nullptr,1,nullptr,nullptr,error,sizeof(error)));
    checked(ltx_native_streaming_destroy(&ctx,error,sizeof(error)),error);assert(!ctx);
    checked(ltx_native_streaming_destroy(&ctx,error,sizeof(error)),error);
    snapshot_alive(metadata);
    std::cout<<"PASS real LTX cancel during streamed suffix: unchanged caller latents, sticky failure, joined cleanup; invalid intent rejected\n";
}

static Output run(bool exact, const char *checkpoint, const char *shader, const char *up, const char *vae,
                  const tc_stream_stage_plan_v1 &plan, uint32_t prefix, uint64_t block_bytes,
                  const tc::ltx::StreamingMetadata *metadata, bool use_connector) {
    char error[1024] = {};
    ltx_native_options options{};
    options.checkpoint=checkpoint; options.shader_source=shader;
    options.width=64; options.height=64; options.frames=9; options.fps=24;
    options.parallel_av=1; options.batch_audio_commands=1;
    options.stream_blocks=1;
    if (!exact) {
        options.max_refill_slots=plan.slot_count;
        options.memory_budget_bytes=(4ull<<30)+uint64_t(options.width)*options.height*options.frames*128+
            (prefix+plan.slot_count)*block_bytes;
    }
    ltx_workload workload{};
    checked(ltx_workload_init(&workload,64,64,9,24,error,sizeof(error)),error);
    ltx_native_denoiser *ctx=nullptr;
    auto total=Clock::now(), phase=total;
    try {
        if (exact) {
            checked(create_exact(&options,plan,prefix,metadata,&ctx,error,sizeof(error)),error);
        } else { ctx=ltx_native_create(&options,nullptr,nullptr,error,sizeof(error)); checked(ctx!=nullptr,error); }
        ltx_native_streaming_info info{};
        checked(ltx_native_get_streaming_info(ctx,&info),error);
        assert(info.pinned_blocks==prefix && info.refill_slots==plan.slot_count);
        std::cout<<(exact?"exact":"legacy")<<" load_seconds="<<seconds(phase)<<std::endl;
        Output out;
        out.video1.resize(workload.stage1_video_tokens*128);
        out.audio1.resize(size_t(workload.audio_tokens)*128);
        for(size_t i=0;i<out.video1.size();++i) out.video1[i]=bf16(std::sin(float(i)*0.17f)*0.5f);
        for(size_t i=0;i<out.audio1.size();++i) out.audio1[i]=bf16(std::cos(float(i)*0.13f)*0.5f);
        const uint32_t rows=use_connector?1024u:16u;
        std::vector<uint16_t> video_text(rows*4096), audio_text(rows*2048), mask(rows,0);
        if (use_connector) {
            std::vector<uint16_t> raw_video(16*4096),raw_audio(16*2048),raw_mask(16,0);
            for(size_t i=0;i<raw_video.size();++i) raw_video[i]=bf16(std::sin(float(i%4096)*0.01f)*0.125f);
            for(size_t i=0;i<raw_audio.size();++i) raw_audio[i]=bf16(std::cos(float(i%2048)*0.01f)*0.125f);
            phase=Clock::now();
            checked(ltx_native_connect_conditioning(ctx,video_text.data(),video_text.size(),
                audio_text.data(),audio_text.size(),mask.data(),mask.size(),rows,
                raw_video.data(),raw_video.size(),raw_audio.data(),raw_audio.size(),
                raw_mask.data(),raw_mask.size(),16,error,sizeof(error)),error);
            std::cout<<(exact?"exact":"legacy")<<" connector_seconds="<<seconds(phase)<<std::endl;
            out.connected_video=video_text;out.connected_audio=audio_text;out.connected_mask=mask;
            finite(video_text);finite(audio_text);snapshot_alive(metadata);
        } else {
            for(size_t i=0;i<video_text.size();++i) video_text[i]=bf16(std::sin(float(i%4096)*0.01f)*0.125f);
            for(size_t i=0;i<audio_text.size();++i) audio_text[i]=bf16(std::cos(float(i%2048)*0.01f)*0.125f);
        }
        phase=Clock::now();
        checked(ltx_native_run(ctx,1,42,out.video1.data(),out.video1.size(),out.audio1.data(),out.audio1.size(),
            video_text.data(),audio_text.data(),mask.data(),rows,nullptr,1,nullptr,nullptr,error,sizeof(error)),error);
        std::cout<<(exact?"exact":"legacy")<<" stage1_seconds="<<seconds(phase)<<std::endl;
        if(exact) {
            tc_stream_counters_v1 c{}; checked(ltx_native_streaming_counters(ctx,&c,error,sizeof(error)),error);
            assert(c.pool_creates==1 && c.slot_bundles==plan.slot_count && c.fills==8ull*(48-prefix));
        }
        out.upsampled.resize(workload.stage2_video_tokens*128);
        phase=Clock::now();
        checked(ltx_native_upsample_stage2(ctx,up,vae,out.upsampled.data(),out.upsampled.size(),out.video1.data(),out.video1.size(),error,sizeof(error)),error);
        std::cout<<(exact?"exact":"legacy")<<" upsample_seconds="<<seconds(phase)<<std::endl;
        out.video2=out.upsampled; out.audio2=out.audio1;
        phase=Clock::now();
        checked(ltx_native_run(ctx,2,42,out.video2.data(),out.video2.size(),out.audio2.data(),out.audio2.size(),
            video_text.data(),audio_text.data(),mask.data(),rows,nullptr,1,nullptr,nullptr,error,sizeof(error)),error);
        std::cout<<(exact?"exact":"legacy")<<" stage2_seconds="<<seconds(phase)<<std::endl;
        if(exact) {
            tc_stream_counters_v1 c{}; checked(ltx_native_streaming_counters(ctx,&c,error,sizeof(error)),error);
            assert(c.pool_creates==1 && c.slot_bundles==plan.slot_count && c.fills==11ull*(48-prefix));
            assert(c.groups_submitted==c.fills && c.content_bytes_loaded==c.fills*block_bytes);
            checked(ltx_native_streaming_destroy(&ctx,error,sizeof(error)),error);
            snapshot_alive(metadata);
        } else { ltx_native_free(ctx); ctx=nullptr; }
        std::cout<<(exact?"exact":"legacy")<<" total_seconds="<<seconds(total)<<std::endl;
        finite(out.video1); finite(out.audio1); finite(out.upsampled); finite(out.video2); finite(out.audio2);
        return out;
    } catch (...) {
        if(exact && ctx) {
            if(!ltx_native_streaming_destroy(&ctx,error,sizeof(error)))
                std::cerr<<"quarantined context retained until process exit: "<<error<<'\n';
        } else if(ctx) ltx_native_free(ctx);
        throw;
    }
}

int main(int argc,char **argv) {
    if(argc<6 || argc>8) {std::cerr<<"checkpoint shader upsampler video-vae slots [exact-api [conditioning]] required\n";return 2;}
    try {
        const uint32_t slots=uint32_t(std::stoul(argv[5]));
        if(slots<1 || slots>3) throw std::invalid_argument("slots must be 1..3");
        tc::ltx::StreamingMetadata metadata(argv[1]);
        const unsigned exact_api=argc>=7?unsigned(std::stoul(argv[6])):2u;
        if (exact_api!=1 && exact_api!=2) throw std::invalid_argument("exact-api must be 1 or 2");
        const auto *borrowed=exact_api==2?&metadata:nullptr;
        const std::string conditioning=argc==8?argv[7]:"synthetic";
        if (conditioning!="synthetic" && conditioning!="connector") throw std::invalid_argument("invalid conditioning");
        const bool use_connector=conditioning=="connector";
        const auto descriptor=metadata.describe({64,64,9,24,use_connector?1024u:16u,
            true,true,false,use_connector?"connected":"synthetic"});
        const auto &first=metadata.block(0);
        const auto block_bytes=first.gpu_bytes+first.cpu_bytes;
        tc::StreamingConfig config;
        config.enabled=true;config.schema_version=1;config.selection="manual";config.retention="request";
        config.stages["denoiser"]={"streamed",1,slots,1,slots-1,slots};
        auto layout=tc::streaming::compile_layout(config,descriptor);
        assert(layout.materializations_complete);
        const auto &s=layout.stages[0];
        std::vector<uint64_t> capacities;
        std::vector<tc_stream_group_v1> groups;
        for(const auto &slot:s.pools[0].slots) capacities.push_back(slot.capacity_bytes);
        for(const auto &g:s.groups) groups.push_back({g.id,g.slot,uint32_t(g.blocks.size()),g.blocks.data(),g.bytes});
        tc_stream_stage_plan_v1 plan{sizeof(plan),TC_STREAM_SLOT_ABI_V1,0,0,s.slot_count,s.distance,s.workers,
            s.pass_count,1,capacities.data(),uint32_t(groups.size()),groups.data()};
        assert(s.source_read_bytes_per_pass && s.prefix_source_read_bytes);
        uint64_t expected_read=0;
        for (uint32_t b=s.prefix;b<48;++b) expected_read+=metadata.block(b).source_read_bytes;
        assert(*s.source_read_bytes_per_pass==expected_read);
        metadata.check_unchanged();
        std::cout<<"TEST ONLY: 64x64x9, fabricated 16-row projections, conditioning="<<conditioning
                 <<", actual 48-block checkpoint; no Gemma/VAE output, not end-to-end certification or ABBA\n";
        std::cout<<"exact_api="<<exact_api<<" borrowed_snapshot="<<(borrowed!=nullptr)<<std::endl;
        auto a=run(false,argv[1],argv[2],argv[3],argv[4],plan,s.prefix,block_bytes,nullptr,use_connector);
        auto b=run(true,argv[1],argv[2],argv[3],argv[4],plan,s.prefix,block_bytes,borrowed,use_connector);
        assert(a.connected_video==b.connected_video && a.connected_audio==b.connected_audio && a.connected_mask==b.connected_mask);
        assert(a.video1==b.video1 && a.audio1==b.audio1 && a.upsampled==b.upsampled && a.video2==b.video2 && a.audio2==b.audio2);
        std::cout<<"PASS real LTX exact adapter: stage1/upsample/stage2 video+audio byte-exact, one pool/"<<slots<<" slots/517 fills\n";
        cancellation_case(argv[1],argv[2],plan,s.prefix,borrowed);
        metadata.check_unchanged();
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
