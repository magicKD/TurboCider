#include "models/z_image/hybrid_stream.hpp"
#include <fstream>
#include <iostream>
#include <thread>
using namespace tc;
namespace fs=std::filesystem;
StreamingConfig config() {
    StreamingConfig c;c.enabled=true;c.schema_version=1;c.selection="manual";c.retention="request";
    c.stages["denoiser"]={"streamed",1,2,1,1,2};return c;
}
template<class F> void rejects(F f) {bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}require(rejected,"expected preflight rejection");}
int main(int argc,char **argv) {
  try {
    require(argc==6 || argc==7,"expected checkpoint, manifest, managed, output, mode, optional inputs directory");
    const std::string mode=argv[5];require(mode=="complete"||mode=="cancel"||mode=="unsafe","unknown mode");
    fs::path output=argv[4];fs::create_directories(output);
    mx::set_default_device(mx::Device(mx::Device::gpu,0));
    streaming::SourceFileIdentity file;file.logical_id="transformer";file.path=argv[1];
    auto parent=streaming::SourceLease::capture_verified({file});const auto source_generation=parent->generation();
    fs::path manifest=argv[2];
    auto generation=z_image::CoreMLGeneration::import_tree(manifest.parent_path(),argv[3]);const auto root=generation->root();
    auto bundle=z_image::VerifiedCoreMLBundleLease::bind(generation,manifest.filename(),parent,"transformer");
    auto signal=std::make_shared<std::atomic<bool>>(false);
    Event event=[mode,signal](const std::string &phase,int block,int) {
        if(phase=="z_image_denoise_block" && block==1) {
            if(mode=="cancel")signal->store(true);
            if(mode=="unsafe")throw std::runtime_error("injected main event failure");
        }
    };
    auto owner=std::make_unique<ZImageHybridStream>(parent,bundle,config(),z_image::StreamingWorkload{512,512,64,9},
        8ull<<30,0,event,signal,42);
    parent.reset();bundle.reset();generation.reset();signal.reset();event={};
    require(fs::exists(root),"owner lost generation");
    auto latent=mx::reshape(mx::sin(mx::arange(0,65536,mx::float32)*Tensor(.01f)),{16,1,64,64});
    auto caption=mx::zeros({64,2560},mx::bfloat16);
    if(argc==7) {
        require(mode=="complete", "external inputs require complete mode");
        latent=mx::load((fs::path(argv[6])/"initial.npy").string());
        caption=mx::astype(mx::load((fs::path(argv[6])/"caption.npy").string()),mx::bfloat16);
        require(latent.dtype()==mx::float32 && latent.shape()==mx::Shape({16,1,64,64}),"invalid fixture latent");
        require(caption.ndim()==2 && caption.shape(0)>32 && caption.shape(0)<=64 && caption.shape(1)==2560,"invalid fixture caption");
    }
    mx::eval({latent,caption});
    auto schedule=owner->sigmas();require(schedule.size()==10 && schedule.back()==0,"wrong sigma schedule");
    rejects([&]{owner->transform(latent,caption,schedule[0],1);});
    rejects([&]{owner->transform(mx::zeros({16,1,32,32}),caption,schedule[0],0);});
    rejects([&]{owner->receipt();});require(!owner->failed(),"preflight rejection poisoned owner");
    std::atomic<bool> wrong_thread{false};
    std::thread foreign([&]{try{owner->transform(latent,caption,schedule[0],0);}catch(const std::invalid_argument&){wrong_thread=true;}});
    foreign.join();require(wrong_thread && !owner->failed(),"owner thread check failed");
    if(mode!="complete") {
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
        if(mode=="unsafe")owner->test_set_drain_failure(true);
#endif
        bool primary=false;
        try{owner->transform(latent,caption,schedule[0],0);}
        catch(const Cancelled&){primary=mode=="cancel";}
        catch(const std::runtime_error&e){primary=mode=="unsafe" && std::string(e.what())=="injected main event failure";}
        require(primary && owner->failed(),"primary failure classification lost");
        const auto completed=owner->branches().size();require(completed>0 && completed<32,"unexpected partial branch count");
        rejects([&]{owner->transform(latent,caption,schedule[0],0);});
        if(mode=="unsafe") {
            require(!owner->drain_safely() && fs::exists(root),"unsafe owner freed generation");
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
            owner->test_set_drain_failure(false);
#endif
        }
        require(owner->drain_safely() && owner->failed(),"safe teardown cleared failed state");
        owner.reset();require(!fs::exists(root),"failed owner cleanup leaked generation");
        std::ofstream report(output/"result.json");
        report<<"{\"mode\":\""<<mode<<"\",\"passed\":true,\"completed_branches\":"<<completed<<",\"generation_cleanup\":true}\n";
        std::cout<<mode<<" failure/drain/retention checks PASS\n";return 0;
    }
    auto initial=latent;
    mx::save((output/"initial.npy").string(),initial);
    mx::save((output/"caption.npy").string(),mx::astype(caption,mx::float32));
    for(unsigned step=0;step<9;++step) {
        auto velocity=owner->transform(latent,caption,schedule[step],step);
        require(velocity.dtype()==mx::float32 && velocity.shape()==latent.shape() &&
                mx::all(mx::isfinite(velocity)).item<bool>(),"invalid transformer output");
        require(mx::max(mx::abs(velocity)).item<float>()>0,"transformer output unexpectedly zero");
        mx::save((output/("velocity-"+std::to_string(step)+".npy")).string(),velocity);
        latent=euler_step(latent,velocity,schedule[step+1]-schedule[step]);mx::eval(latent);
        require(latent.dtype()==mx::float32 && mx::all(mx::isfinite(latent)).item<bool>(),"invalid Euler state");
        require(owner->branches().size()==(step+1)*32,"missing branch completions");
        std::cout<<"step="<<step<<" completed_branches="<<owner->branches().size()<<std::endl;
    }
    owner->finish();auto receipt=owner->receipt();auto metrics=owner->hybrid_metrics();auto counts=owner->counters();
    streaming::verify_actual_stage_receipt(owner->layout().stages[0],0,42,
        {owner->layout().digest,"z-image-verified-hybrid-stage-v1",source_generation},*receipt);
    require(receipt->completed_passes==9 && receipt->completed_groups==261 && receipt->reader_fences_completed==261 &&
            counts.fills==261 && counts.groups_submitted==261 && metrics.runtime_calls==288 && metrics.runtime_failures==0,
            "stage/call totals mismatch");
    require(!mx::array_equal(initial,latent).item<bool>(),"Euler trajectory unchanged");
    mx::save((output/"final.npy").string(),latent);
    std::ofstream branches(output/"branches.csv");branches<<"step,branch,rows,coreml_sequence\n";
    size_t index=0;for(const auto &b:owner->branches()) {
        require(b.step==index/32 && b.branch==index%32 && b.rows==(b.branch<2?1024u:1088u) && b.coreml_sequence==index+1,"branch receipt mismatch");++index;
        branches<<b.step<<','<<b.branch<<','<<b.rows<<','<<b.coreml_sequence<<'\n';
    }
    std::ofstream groups(output/"groups.csv");
    groups<<"pass,step,group,pool,slot,fill_count,request_generation,content_generation,source_generation,expected_bytes,actual_bytes,fill_completed,group_submitted,reader_count,queue,sequence,reader_completed\n";
    for(const auto &g:receipt->groups) {
        require(g.reader_count==1,"expected one final GPU reader");const auto &r=g.readers[0];
        groups<<g.pass<<','<<g.step<<','<<g.group<<','<<g.pool<<','<<g.slot<<','<<g.fill_count<<','<<g.request_generation<<','<<g.content_generation<<','<<g.source_generation<<','<<g.expected_bytes<<','<<g.actual_bytes<<','<<g.fill_completed<<','<<g.group_submitted<<','<<g.reader_count<<','<<r.fence.queue<<','<<r.fence.sequence<<','<<r.completed<<'\n';
    }
    const auto layout_digest=owner->layout().digest;
    owner.reset();require(!fs::exists(root),"completed owner leaked generation");
    std::ofstream report(output/"result.json");
    report<<"{\"mode\":\"complete\",\"passed\":true,\"completed_passes\":9,\"coreml_calls\":288,\"stage_groups\":261,\"stage_read_bytes\":"<<receipt->logical_read_bytes
          <<",\"layout_digest\":\""<<layout_digest<<"\",\"stage_event_digest\":\""<<receipt->event_digest<<"\",\"stage_canonical_digest\":\""<<receipt->canonical_digest<<"\"}\n";
    report.close();branches.close();groups.close();
    std::cout<<"full transformer stage + branch receipts + owner cleanup PASS\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
