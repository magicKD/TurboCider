#include "../../native/models/qwen21/convrot_weights.hpp"
#include "turbocider/turbocider.h"
#include <cmath>
#include <iostream>

int main(int argc,char **argv){try{
    using namespace tc;char*id=tc_runtime_build_identity();require(id,"native identity missing");const std::string loaded(id);tc_string_free(id);
    require(loaded==TC_TEST_BOUND_NATIVE_ID,"ConvRot test/new Weights ABI requires the matching library");configure_streams();
    constexpr int rows=69,pitch=512,m=33,rb=5,re=67,cb=256,ce=512;
    auto ids=mx::arange(rows*pitch,mx::int32);auto q=mx::reshape(mx::astype((ids*37+13)%256-128,mx::int8),{rows,pitch});
    std::vector<float>scales;for(int i=0;i<rows;++i)scales.push_back(float(i%17-8)*.00012345f);
    Tensor s(scales.data(),{rows,1},mx::float32);auto tag=mx::zeros({1},mx::uint8);
    Weights w;w.bind_arrays({"proj.weight","proj.weight_scale","proj.comfy_quant"},{q,s,tag});
    require(w.pack_convrot_q8(128,mx::float32)==1,"exact F32 coefficient packing missing");
    require(w.at("proj.weight").dtype()==mx::uint32&&w.at("proj.scales").dtype()==mx::float32&&w.at("proj.biases").dtype()==mx::float32,"source coefficients silently narrowed");
    auto x=mx::reshape(mx::astype(mx::arange(m*(ce-cb),mx::int32)%29-14,mx::bfloat16)/Tensor(32.f,mx::bfloat16),{1,m,ce-cb});
    const float h4_values[]={.5f,.5f,.5f,-.5f,.5f,.5f,-.5f,.5f,.5f,-.5f,.5f,.5f,-.5f,.5f,.5f,.5f};
    auto h4=Tensor(h4_values,{4,4},mx::bfloat16),h16=mx::kron(h4,h4),h256=mx::kron(h16,h16);
    auto xr=mx::reshape(mx::matmul(mx::reshape(x,{1,m,1,256}),h256),x.shape());
    auto logical=mx::astype(slice_axis(slice_axis(q,0,rb,re),1,cb,ce),mx::float32)*slice_axis(s,0,rb,re);
    auto reference=mx::matmul(mx::astype(xr,mx::float32),mx::transpose(logical));
    auto partial=w.project_base_slice_fp32(x,"proj",rb,re,cb,ce);mx::eval(reference,partial);
    require(partial.dtype()==mx::float32&&partial.shape()==mx::Shape{1,m,re-rb},"actual partial was rounded before join");
    auto rel=mx::sqrt(mx::sum(mx::square(reference-partial))/mx::sum(mx::square(reference)));mx::eval(rel);
    require(rel.item<float>()<1e-4f,"F32 original coefficient output differs from independent raw-code projection");
    for(int which=0;which<4;++which){bool denied=false;try{(void)w.project_base_slice_fp32(x,"proj",which==0?-1:rb,which==1?rows+1:re,
            which==2?128:cb,which==3?pitch+256:ce);}catch(const std::exception&){denied=true;}require(denied,"invalid physical ConvRot range accepted");}
    // Metadata-only full-Qwen fixture: no eval/dense/model allocation.
    Weights whole;std::vector<std::string>names;std::vector<Tensor>values;
    for(const auto& spec:{std::tuple{"block.gate_up",24576,4096},std::tuple{"block.out",4096,12288}}){const auto&[name,n,k]=spec;
        names.insert(names.end(),{std::string(name)+".weight",std::string(name)+".scales",std::string(name)+".biases",std::string(name)+".comfy_quant"});
        values.insert(values.end(),{mx::zeros({n,k/4},mx::uint32),mx::ones({n,k/128},mx::float32),mx::full({n,k/128},-128.f,mx::float32),tag});}
    whole.bind_arrays(names,values);auto source=qwen21::convrot_ffn_sources(whole,"block.");require(source.size()==3,"three complete source views required");
    for(const auto&v:source)require(v.bits==8&&v.group_size==128&&v.transform==ane::FfnWeight::Transform::ComfyH256Inverse&&v.scales->dtype()==mx::float32&&v.offsets->dtype()==mx::float32,"source basis/coefficient metadata lost");
    require(source[0].values.shape()==mx::Shape{12288,1024}&&source[1].values.shape()==source[0].values.shape()&&source[2].values.shape()==mx::Shape{4096,3072},"original fused source halves/physical pitch changed");
    if(argc==2){
        // Real broadcast metadata, lazy gate/up halves and actual private
        // staging. Metadata-only shape tests cannot catch layout changes
        // which occur when MLX evaluates a split of a repeated row scale.
        Weights staged;names.clear();values.clear();
        for(const auto& spec:{std::tuple{"block.gate_up",24576,4096},std::tuple{"block.out",4096,12288}}){const auto&[name,n,k]=spec;
            auto scale=mx::repeat(mx::full({n,1},.125f,mx::float32),k/128,1);
            auto bias=scale*Tensor(-128.f,mx::float32);auto codes=mx::full({n,k/4},uint32_t(0x80808080),mx::uint32);
            mx::eval(codes,scale,bias);
            names.insert(names.end(),{std::string(name)+".weight",std::string(name)+".scales",std::string(name)+".biases",std::string(name)+".comfy_quant"});
            values.insert(values.end(),{codes,scale,bias,tag});
        }
        staged.bind_arrays(names,values);
        Weights replaced;auto replacements=values;
        for(size_t i=0;i<names.size();++i)if(names[i].ends_with(".scales")||names[i].ends_with(".biases"))
            replacements[i]=replacements[i]*Tensor(2.f,mx::float32);
        replaced.bind_arrays(names,replacements);std::atomic<bool>cancelled{false};
        ane::HybridFfn runtime(argv[1],4096,12288,size_t(4)<<30,cancelled);
        require(runtime.available(),"actual ConvRot private executor capability required");runtime.begin_request();
        auto input=mx::zeros({1,1024,4096},mx::bfloat16);mx::eval(input);
        auto gpu=[](const Tensor&a){return mx::zeros(a.shape(),a.dtype());};
        auto channels=[](const Tensor&a,int begin,int end){return std::pair{mx::zeros(a.shape(),mx::float32),mx::zeros({1,a.shape(1),end-begin},a.dtype())};};
        for(int layer=0;layer<3;++layer){
            runtime.stage_weights(layer,1024,qwen21::convrot_ffn_sources(layer==2?replaced:staged,"block."));
            auto output=runtime.run(layer,input,gpu,cancelled,nullptr,channels,
                [&](int next){return next<=2?qwen21::convrot_ffn_sources(staged,"block."):std::vector<ane::FfnWeight>{};});
            mx::eval(output);require(runtime.available(),"real ConvRot staging/launch fell back: "+runtime.reason());
            auto maximum=mx::astype(mx::max(mx::abs(output)),mx::float32);mx::eval(maximum);
            require(maximum.item<float>()==0,"zero logical ConvRot source produced nonzero output: "+std::to_string(maximum.item<float>()));
        }
        runtime.drain();const auto metrics=runtime.metrics();
        require(metrics.runtime_weight_prefetch_submissions==2&&metrics.runtime_weight_prefetch_hits==1&&metrics.runtime_weight_prefetch_discards==1,
            "actual ConvRot future source hit/changed-metadata generation rejection differs");
        std::cout<<"PASS actual ConvRot broadcast-metadata stage, ANE launch, future-bank reuse and stale-metadata rejection\n";
    }
    std::cout<<"PASS ConvRot F32 coefficient partial, physical ranges and explicit H256-aware source views\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
