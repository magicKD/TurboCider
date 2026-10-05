#include "../../native/backends/ane_channel_selection.hpp"
#include <iostream>

using namespace tc::ane;
namespace {
void check(bool value,const char *why) {if(!value)throw std::runtime_error(why);}
ChannelCalibrationIdentity key() {
    return {std::string(64,'a'),"","bf16-dense","bf16","private_ane","sylvester",
        "M4 Max","25G1","build-1","metal-1","graph-1-b1056","sources-1",1056,3840,10240,1024,512,false};
}
ChannelTrialEvidence evidence() {
    return {{.04,.042,.041},{.031,.032,.033},12,0,0,.01,.9998,true};
}
}
int main() {
    try {
        const auto identity=key();const auto trial=evidence();
        check(identity.valid() && trial.accepts(),"valid identity/trial rejected");
        auto owner=std::make_shared<int>(1);
        const std::vector<std::weak_ptr<void>> sources{owner};
        ChannelSelectionCache cache;
        check(!cache.find(identity),"empty cache hit");
        cache.admit(identity,{4096,false,true,"accepted"},trial,sources);
        auto found=cache.find(identity);
        check(found && found->channels==4096 && found->cache_hit && found->trial_passed,"accepted candidate not cached");
        for(int field=0;field<19;++field) {
            auto changed=identity;
            switch(field) {
            case 0:changed.model_sha256[0]='b';break;
            case 1:changed.adapter="adapter-1";break;
            case 2:changed.encoding="q4";break;
            case 3:changed.precision="fp16";break;
            case 4:changed.backend="public_coreml";break;
            case 5:changed.recipe="comfy";break;
            case 6:changed.soc="M5";break;
            case 7:changed.os_build="25G2";break;
            case 8:changed.runtime_build="build-2";break;
            case 9:changed.metal_abi="metal-2";break;
            case 10:changed.graph_abi="graph-1-b4224";break;
            case 11:changed.source_generation="sources-2";break;
            case 12:changed.rows=1024;break;
            case 13:changed.hidden=4096;break;
            case 14:changed.width=12288;break;
            case 15:changed.tile_k=512;break;
            case 16:changed.tile_n=256;break;
            case 17:changed.prefetch=true;break;
            case 18:changed.adapter="negative-strength";break;
            }
            check(!cache.find(changed),"calibration identity field did not isolate cache");
        }
        for(int failure=0;failure<11;++failure) {
            auto bad=trial;
            switch(failure) {
            case 0:bad.completed=false;break;
            case 1:bad.calls=0;break;
            case 2:bad.fallbacks=1;break;
            case 3:bad.retries=1;break;
            case 4:bad.relative_l2=.031;break;
            case 5:bad.cosine=.998;break;
            case 6:bad.candidate_seconds[0]=NAN;break;
            case 7:bad.candidate_seconds={.040,.042,.043};break;
            case 8:bad.gpu_seconds.pop_back();break;
            case 9:bad.relative_l2=INFINITY;break;
            case 10:bad.cosine=1.001;break;
            }
            check(!bad.accepts(),"failed/slow/partial trial accepted");
            bool rejected=false;
            try{cache.admit(identity,{4096,false,true,"bad"},bad,sources);}catch(const std::invalid_argument&){rejected=true;}
            check(rejected,"cache admitted an unverified trial");
        }
        for(int channels:{0,-1,513,10240}) {
            bool rejected=false;
            try{cache.admit(identity,{channels,false,true,"bad"},trial,sources);}catch(const std::invalid_argument&){rejected=true;}
            check(rejected,"invalid/unmeasured selected width admitted");
        }
        bool rejected=false;
        try{cache.admit(identity,{4096,false,false,"proposal only"},trial,sources);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"proposal flag bypassed actual trial gate");
        owner.reset();
        check(!cache.find(identity),"expired source generation hit cache");
        check(sources[0].expired(),"cache pinned model weights outside the optional budget");
        rejected=false;
        try{cache.admit(identity,{4096,false,true,"expired"},trial,sources);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"expired source owner admitted");
        owner=std::make_shared<int>(2);
        ChannelSelectionCache bounded(2);
        auto second=identity,third=identity;second.adapter="b";third.adapter="c";
        bounded.admit(identity,{4096,false,true,"one"},trial,{owner});
        bounded.admit(second,{4096,false,true,"two"},trial,{owner});
        check(bool(bounded.find(identity)),"LRU touch failed");
        bounded.admit(third,{4096,false,true,"three"},trial,{owner});
        check(!bounded.find(second) && bounded.find(identity) && bounded.find(third),"bounded LRU evicted wrong generation");
        std::cout<<"PASS native channel selection: 19 identity branches, complete-runtime/quality/gain gates, "
            "source-lifetime invalidation without pinning and bounded LRU\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
