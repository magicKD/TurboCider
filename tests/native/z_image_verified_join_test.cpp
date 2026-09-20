#include "backends/coreml.hpp"
#include "models/z_image/coreml_bundle.hpp"
#include "models/z_image/hybrid_math.hpp"
#include "models/z_image/streaming_descriptor.hpp"
#include "models/z_image/weight_stream.hpp"
#include <iostream>
using namespace tc;
namespace fs=std::filesystem;
int main(int argc,char **argv) {
  try {
    require(argc==4,"expected checkpoint, staged manifest, private parent");
    mx::set_default_device(mx::Device(mx::Device::gpu,0));
    streaming::SourceFileIdentity file;file.logical_id="transformer";file.path=argv[1];
    auto parent=streaming::SourceLease::capture_verified({file});
    fs::path manifest(argv[2]);
    auto generation=z_image::CoreMLGeneration::import_tree(manifest.parent_path(),argv[3]);
    auto root=generation->root();
    auto bundle=z_image::VerifiedCoreMLBundleLease::bind(generation,manifest.filename(),parent,"transformer");
    std::atomic<bool> stop{false};Event event=[](const std::string &,int,int){};
    z_image::StreamingMetadata metadata(parent);
    std::shared_ptr<const z_image::GpuSuffixSource> source=metadata.materialize_gpu_suffix({512,512,64,9},bundle->partition().ane_end,stop);
    const auto derived_digest=source->content_digest();
    require(derived_digest=="59b03803a9dac69d0fe89a79ef318056260b77d48cdc588744aef6bee89fcb47", "derived bytes differ from recorded real-model materialization");
    {
        Weights fixed;
        ZImageWeightStream reader(source,1,1,8ull<<30,0,fixed,event,stop);
        reader.create_exact_pool(1,reader.metrics().block_bytes);
        HybridSession session(bundle,event,stop);
        auto graph=z_image::make_hybrid_gpu_graph(3840,10240,5120);
        uint64_t fill_bytes=0; unsigned fills=0,joins=0;
        for(unsigned branch=0;branch<32;++branch) {
            std::string prefix; Weights weights;
            if(branch<2) {prefix="noise_refiner."+std::to_string(branch);weights=fixed;}
            else {
                auto layer=branch-2;prefix="layers."+std::to_string(layer);
                if(layer==0)weights=reader.prefix_weights(0);
                else {fill_bytes+=reader.fill_exact(0,layer,&stop);++fills;weights=reader.bind_exact(0,layer);}
            }
            const int rows=branch<2?1024:1088;
            auto input=mx::contiguous(mx::zeros({1,rows,3840},mx::bfloat16));
            auto packed=mx::contiguous(mx::zeros({1,1088,3840},mx::float16));mx::eval({input,packed});
            const auto ffn=prefix+".feed_forward";
            auto gpu=graph({input,weights.at(ffn+".w1.weight"),weights.at(ffn+".w3.weight"),weights.at(ffn+".w2.weight")})[0];
            mx::async_eval({gpu});
            auto ane=slice_axis(session.predict(int(branch),packed),1,0,rows);
            auto joined=z_image::join_hybrid_ffn(gpu,ane,Tensor(session.output_scale,mx::bfloat16));
            require(joined.shape()==mx::Shape{1,rows,3840} && joined.dtype()==mx::bfloat16,"join ABI changed");
            // This GPU consumer drains the lazy join before either backing is reused.
            require(mx::all(mx::equal(joined,Tensor(0,mx::bfloat16))).item<bool>(),"nonzero/nonfinite joined zero oracle");
            ++joins;
        }
        reader.check_unchanged();session.revalidate_source();source->check_unchanged();
        require(fills==29 && joins==32 && session.metrics().runtime_calls==32 && session.metrics().runtime_failures==0,"incomplete split branch coverage");
        require(reader.metrics().pinned_blocks==1 && reader.metrics().refill_slots==1 && reader.metrics().request_pack_read_bytes==0 && reader.metrics().request_pack_write_bytes==0,"reader repartitioned or repacked");
        reader.destroy_exact_pool();
        std::cout<<"joined_zero_branches="<<joins<<" fills="<<fills<<" fill_bytes="<<fill_bytes
                 <<" derived_sha256="<<derived_digest<<" source_parent="<<source->parent_file().content_digest<<"\n";
    }
    bundle.reset();generation.reset();require(!fs::exists(root),"Core ML generation leaked");
    std::cout<<"verified GPU suffix + Core ML prefix join PASS; no full model or nonzero quality claim\n";
  } catch(const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
}
