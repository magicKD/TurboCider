#include "backends/coreml.hpp"
#include "models/z_image/coreml_bundle.hpp"
#include <iostream>
namespace fs = std::filesystem;
using namespace tc;
int main(int argc, char **argv) {
    require(argc == 4, "expected checkpoint, staged manifest, private parent");
    mx::set_default_device(mx::Device(mx::Device::gpu,0));
    streaming::SourceFileIdentity file; file.logical_id="denoiser"; file.path=argv[1];
    auto parent=streaming::SourceLease::capture_verified({file});
    fs::path manifest(argv[2]);
    auto generation=z_image::CoreMLGeneration::import_tree(manifest.parent_path(),argv[3]);
    auto bundle=z_image::VerifiedCoreMLBundleLease::bind(generation,manifest.filename(),parent,"denoiser");
    auto root=generation->root();
    std::atomic<bool> stop{false};
    // Cancel after the first load callback, exercising partial model cleanup.
    bool cancelled=false;
    try { HybridSession partial(bundle,[&](const std::string &,int i,int){if(i==0)stop=true;},stop); }
    catch(const Cancelled &) {cancelled=true;}
    require(cancelled,"constructor lost cancellation type"); bundle->revalidate();
    stop=false;
    bool event_failed=false;
    try { HybridSession partial(bundle,[&](const std::string &,int i,int){if(i==1)throw std::runtime_error("injected load event");},stop); }
    catch(const std::runtime_error &e) {event_failed=std::string(e.what())=="injected load event";}
    require(event_failed,"load event failure not propagated"); bundle->revalidate();
    auto session=std::make_unique<HybridSession>(bundle,[](const std::string &,int,int){},stop);
    require(session->metrics().weight_variant == bundle->partition().precision_revision,
            "typed precision reporting lost verified export variant");
    generation.reset(); parent.reset(); bundle.reset();
    require(fs::exists(root),"session did not retain generation");
    require(session->rows==1088 && session->hidden==3840 && session->block_count==32 &&
            session->ane_mlp_end==5120 && session->output_scale==32 && session->checkpoint_sha_verified,
            "typed partition fields changed");
    auto input=mx::contiguous(mx::zeros({1,1088,3840},mx::float16)); mx::eval(input);
    for (auto invalid : {mx::zeros({1,1024,3840},mx::float16), mx::zeros({1,1088,3840},mx::float32),
                         mx::transpose(mx::zeros({1,3840,1088},mx::float16),{0,2,1})}) {
        bool rejected=false;
        try { session->predict(0,invalid); } catch(const std::invalid_argument &) {rejected=true;}
        require(rejected,"invalid input reached Core ML");
    }
    bool bad_block=false;
    try {session->predict(32,input);} catch(const std::invalid_argument &) {bad_block=true;}
    require(bad_block && session->runtime_available(),"input validation incorrectly latched runtime failure");
    const void *backing=nullptr;
    for(int block=0;block<32;++block) {
        auto output=session->predict(block,input);
        require(output.shape()==mx::Shape{1,1088,3840},"output shape mismatch");
        // GPU consumer completes before the shared backing is reused.
        require(mx::all(mx::equal(output,mx::array(0,mx::float16))).item<bool>(),"nonzero/nonfinite zero-input result");
        const void *address=output.data<mx::float16_t>();
        if(backing)require(backing==address,"output backing not shared"); else backing=address;
    }
    session->revalidate_source();
    auto metrics=session->metrics();
    require(metrics.runtime_calls==32 && metrics.runtime_failures==0 && metrics.warmup_calls==0,"unexpected prediction counters");
    session.reset(); require(!fs::exists(root),"generation leaked after model/session release");
    std::cout << "typed_session predictions=32 failures=0 warmups=0 zero_oracle=PASS shared_backing=PASS partial_cancel=PASS event_failure=PASS owner_cleanup=PASS\n";
}
