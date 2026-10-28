#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/viggle_adapter.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

using namespace tc;
#ifndef TC_ASYNC_CANCEL_CPP_SHA
#define TC_ASYNC_CANCEL_CPP_SHA "unbound"
#endif
namespace {
NSString *ns(const std::string &s){return [NSString stringWithUTF8String:s.c_str()];}
void set_mode(bool async,bool edit){
    for(const char *flag:{"TURBOCIDER_QWEN21_LORA_BF16_AB","TURBOCIDER_QWEN21_LORA_SWIGLU","TURBOCIDER_QWEN21_CORRECTED_QKV","TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","TURBOCIDER_QWEN21_STUDENT_FIVE_STEP"})setenv(flag,"1",1);
    setenv("TURBOCIDER_QWEN21_ASYNC_STEP_PAIR",async?"1":"0",1);setenv("TURBOCIDER_QWEN21_PREFIX_SNAPSHOT",edit?"1":"0",1);
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX",edit?"1":"0",1);
}
void checked(const RunResult &r){
    require(r.actual_steps==5&&r.qwen_async_step_pair.has_value(),"retry did not execute a complete async request");const auto &a=*r.qwen_async_step_pair;
    require(a.submitted_steps==5&&a.checked_steps==5&&a.accepted_steps==5&&a.completion_batches==3&&a.evaluation_calls==10&&
        a.peak_pending_steps==2&&a.final_drain_complete&&!a.diagnostic_submit_observer,"retry async completion evidence differs");
}
}
int main(int argc,const char **argv){@autoreleasepool{
    if(argc==2&&std::string_view(argv[1])=="--describe"){
        char *id=tc_runtime_build_identity();if(!id)return 1;auto info=@{@"recipe":@"bounded-async-cancel-v1",@"cpp_sha256":@TC_ASYNC_CANCEL_CPP_SHA,@"loaded_runtime_id":ns(id)};
        tc_string_free(id);std::cout<<json(info)<<'\n';return 0;
    }
    if(argc!=5){std::cerr<<"usage: cancel-probe MODEL PINNED_R128 REF512 FRESH_OUTPUT\n";return 2;}
    const auto out=std::filesystem::absolute(argv[4]);bool created=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-async-cancel-v1",@"status":@"starting",@"qualification_passed":@NO,
        @"scope":@"same-process submit-hook cancellation at one/two retained steps, real model retry/restore; not performance or arbitrary GPU fault qualification"} mutableCopy];
    try{
        require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned output required");created=true;
        require(std::filesystem::space(out).available>(3ull<<30),"3GiB disk reserve required");
        const auto adapter=std::filesystem::canonical(argv[2]);const auto *pin=qwen21::viggle_v021_adapter(adapter.filename().string());
        require(pin&&pin->rank=="r128"&&sha256_file(adapter)==pin->sha256,"original pinned r128 required");configure_streams();
        std::atomic<bool> cancelled{false};qwen21::Session session(argv[1]);NSMutableArray *cases=[NSMutableArray array];report[@"cases"]=cases;
        const auto reference=std::filesystem::canonical(argv[3]);report[@"adapter_sha256"]=ns(sha256_file(adapter));report[@"reference_sha256"]=ns(sha256_file(reference));
        int successful=0;
        for(bool edit:{false,true}){
            const std::string task=edit?"edit":"generate";Request r;r.model="qwen-image-2.1";r.operation="image."+task;r.width=r.height=512;r.steps=5;
            r.frames=1;r.audio=false;r.seed=73;r.execution="gpu";r.residency="resident";r.allow_approximation=true;r.lora_strategy="inference_time";
            r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;r.loras.push_back({adapter.string(),1.f,"transformer"});
            r.prompt=edit?"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.":"A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
            if(edit){InputAsset input;input.kind="image";input.role="reference";input.path=reference.string();r.inputs.push_back(input);r.qwen21_reference_size=512;}
            NSMutableDictionary *c=[@{@"case":ns(task)} mutableCopy];NSMutableArray *outputs=[NSMutableArray array];c[@"successful"]=outputs;
            auto run=[&](const std::string &label,bool async){set_mode(async,edit);cancelled.store(false);r.output=(out/(task+"-"+label+".png")).string();
                auto result=session.generate(r,[](const auto&,int,int){},cancelled);++successful;[outputs addObject:@{@"label":ns(label),@"filename":ns(task+"-"+label+".png"),@"metrics":to_dictionary(result)}];return result;};
            auto baseline=run("sync",false);require(!baseline.qwen_async_step_pair,"sync control claimed async work");
            if(edit){auto hit=run("sync-hit",false);require(hit.qwen_corrected_qkv&&hit.qwen_corrected_qkv->prefix_snapshot_hit,"sync strong baseline did not hit its prefix");}
            const int target=edit?2:1;size_t pending=0;bool not_ready=false;int max_completed=0,observed=0;
            set_mode(true,edit);cancelled.store(false);r.output=(out/(task+"-must-not-export.png")).string();
            session.set_diagnostic_async_step_observer([&](int submitted,size_t count,bool unready){++observed;if(submitted==target){pending=count;not_ready=unready;cancelled.store(true);}});
            bool stopped=false;try{(void)session.generate(r,[&](const std::string &phase,int n,int){if(phase=="denoise")max_completed=std::max(max_completed,n);},cancelled);}
            catch(const std::exception &e){stopped=std::string(e.what()).find("cancel")!=std::string::npos;}
            session.set_diagnostic_async_step_observer({});
            require(stopped&&pending==size_t(target)&&observed==target&&max_completed==0&&!std::filesystem::exists(r.output),"cancel failed to drain/withhold completed progress and PNG");
            c[@"cancel_after_submit"]=@(target);c[@"retained_steps_at_cancel"]=@(pending);c[@"some_result_not_ready_at_cancel"]=@(not_ready);
            c[@"completed_progress_before_cancel"]=@(max_completed);c[@"cancelled_png_absent"]=@YES;c[@"cancelled"]=@YES;
            auto retry=run("retry",true);checked(retry);
            if(edit){require(retry.qwen_corrected_qkv&&!retry.qwen_corrected_qkv->prefix_snapshot_hit,"retry reused a cancelled prefix bank");
                auto hit=run("retry-hit",true);checked(hit);require(hit.qwen_corrected_qkv&&hit.qwen_corrected_qkv->prefix_snapshot_hit,"retry did not repopulate its own prefix");}
            auto restored=run("restore",false);require(!restored.qwen_async_step_pair,"sync restoration retained async state");
            [cases addObject:c];
        }
        session.unload();mx::synchronize();report[@"successful_full_requests"]=@(successful);report[@"cancelled_requests"]=@2;
        char *identity=tc_runtime_build_identity();require(identity,"loaded runtime identity missing");report[@"loaded_runtime_id"]=ns(identity);tc_string_free(identity);
        report[@"status"]=@"complete";
    }catch(const std::exception &e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"complete"]?0:1;
}}
