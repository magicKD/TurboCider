#include "models/z_image/hybrid_layout.hpp"
#include <iostream>
namespace fs=std::filesystem;
using namespace tc;
using namespace tc::z_image;
StreamingConfig config(unsigned prefix,unsigned slots) {
    StreamingConfig c;c.enabled=true;c.schema_version=1;c.selection="manual";c.retention="request";
    c.stages["denoiser"]={"streamed",1,slots,prefix,slots==2?1u:0u,slots};return c;
}
template<class F> void rejects(F f) {bool failed=false;try{f();}catch(const std::invalid_argument&){failed=true;}require(failed,"expected layout rejection");}
int main(int argc,char **argv) {
  try {
    require(argc==9,"expected checkpoint, other parent, managed and five banks");
    streaming::SourceFileIdentity file;file.logical_id="transformer";file.path=argv[1];
    auto parent=streaming::SourceLease::capture_verified({file});StreamingMetadata metadata(parent);
    auto unverified=streaming::SourceLease::capture({file});StreamingMetadata plain(unverified);
    const StreamingWorkload work{512,512,64,9};
    auto suffix=metadata.describe_gpu_suffix(work,5120);
    auto original=streaming::compile_layout(config(1,1),suffix.descriptor);
    const auto original_exact=streaming::compile_layout(config(1,1),metadata.describe_verified(work)).digest;
    std::string baseline;
    for(int i=4;i<9;++i) {
        auto generation=CoreMLGeneration::import_tree(argv[i],argv[3]);
        auto bound_parent=parent;
        if(i==8){auto other=file;other.path=argv[2];bound_parent=streaming::SourceLease::capture_verified({other});}
        auto bundle=VerifiedCoreMLBundleLease::bind(generation,"manifest.json",bound_parent,"transformer");
        if(i==8){rejects([&]{describe_hybrid_streaming(metadata,*bundle,config(1,1),work);});continue;}
        auto plan=describe_hybrid_streaming(metadata,*bundle,config(1,1),work);
        require(plan.gpu.recipe_digest==suffix.recipe_digest && plan.gpu.packing.size()==32,"GPU recipe changed by bundle binding");
        require(plan.gpu.descriptor.artifacts.size()==suffix.descriptor.artifacts.size(),"source count changed");
        for(size_t j=0;j<suffix.descriptor.artifacts.size();++j) {
            const auto &a=plan.gpu.descriptor.artifacts[j];const auto &b=suffix.descriptor.artifacts[j];
            require(a.id==b.id && a.identity==b.identity && a.bytes==b.bytes && a.identity_kind==b.identity_kind,"source identity relabeled");
        }
        require(plan.gpu.descriptor.stages[0].resident_fields==suffix.descriptor.stages[0].resident_fields,"fixed GPU fields changed");
        for(unsigned b=0;b<30;++b)require(plan.gpu.descriptor.stages[0].blocks[b].fields==suffix.descriptor.stages[0].blocks[b].fields,"slot GPU fields changed");
        auto &a=plan.layout.stages[0];auto &b=original.stages[0];
        require(a.resident_bytes==b.resident_bytes && a.prefix_bytes==b.prefix_bytes && a.peak_pool_bytes==b.peak_pool_bytes && a.source_read_bytes_per_pass==b.source_read_bytes_per_pass,"binding changed GPU capacity/I/O");
        require(plan.layout.digest!=original.digest && plan.layout.digest!=original_exact,"hybrid reused GPU identity");
        if(i==4)baseline=plan.layout.digest;
        else if(i==5)require(plan.layout.digest==baseline,"relocated identity changed");
        else require(plan.layout.digest!=baseline,"bundle/precision change did not change layout");
        auto overlap=describe_hybrid_streaming(metadata,*bundle,config(7,2),work);
        require(overlap.layout.digest!=plan.layout.digest && overlap.layout.stages[0].prefix==7 && overlap.layout.stages[0].slot_count==2 && overlap.layout.stages[0].pass_count==9,"P/K/pass policy not preserved");
        rejects([&]{describe_hybrid_streaming(plain,*bundle,config(1,1),work);});
        rejects([&]{describe_hybrid_streaming(metadata,*bundle,config(1,1),{256,512,64,9});});
        rejects([&]{describe_hybrid_streaming(metadata,*bundle,config(1,1),{512,512,32,9});});
        rejects([&]{describe_hybrid_streaming(metadata,*bundle,config(1,1),{512,512,64,8});});
        require(streaming::compile_layout(config(1,1),metadata.describe_verified(work)).digest==original_exact,"exact descriptor mutated");
        std::cout<<fs::path(argv[i]).filename()<<" layout="<<plan.layout.digest<<" PASS\n";
    }
    require(fs::is_empty(argv[3]),"planning leaked generation");
    std::cout<<"verified metadata binding: unchanged GPU fields/recipe/capacity, portable identities, changed bundle/precision, parent/scope rejection PASS\n";
  } catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
