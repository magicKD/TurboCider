#include "../../native/backends/ane_ffn.hpp"
#include <iostream>

using namespace tc;
namespace {
void check(bool value,const char *why){if(!value)throw std::runtime_error(why);}
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    try {
        constexpr int h=128,f=2560,rows=17;
        std::atomic<bool> cancelled{false};
        setenv("TURBOCIDER_ANE_BACKEND","private",1);setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","auto",1);setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);
        setenv("TURBOCIDER_PRIVATE_ANE_GPU_IO","1",1);setenv("TURBOCIDER_PRIVATE_ANE_PREFETCH","0",1);
        setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
        unsetenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN");unsetenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN");
        unsetenv("TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES");unsetenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE");
        bool rejected=false;
        try{ane::HybridFfn incomplete(argv[1],h,f,512u<<20,cancelled);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"automatic channels silently selected an unresolved graph width");
        std::vector<std::array<Tensor,3>> sources;
        ane::HybridFfn::CalibrationWorkload workload;
        workload.model_sha256=std::string(64,'a');workload.encoding="synthetic-bf16";
        workload.rows=rows;workload.layers=5;
        for(int layer=0;layer<5;++layer) {
            std::vector<float> gate(size_t(f)*h),up(gate.size()),down(size_t(h)*f);
            for(int r=0;r<f;++r) {
                gate[size_t(r)*h+r%h]=.125f+layer*.0125f;
                up[size_t(r)*h+r%h]=-.25f+layer*.01f;
                down[size_t(r%h)*f+r]=.03125f;
            }
            sources.push_back({mx::astype(Tensor(gate.data(),{f,h},mx::float32),mx::bfloat16),
                mx::astype(Tensor(up.data(),{f,h},mx::float32),mx::bfloat16),
                mx::astype(Tensor(down.data(),{h,f},mx::float32),mx::bfloat16)});
            for(const auto &value:sources.back()) {
                mx::eval(value);workload.source_generation+=':'+std::to_string(value.id());
                workload.source_owners.push_back(value.data_shared_ptr());
            }
        }
        int supplied=0,full=0,heads=0;
        workload.weights=[&](int layer) {
            ++supplied;const auto &source=sources.at(layer);
            return std::vector<ane::FfnWeight>{{source[0],std::nullopt,std::nullopt},
                {source[1],std::nullopt,std::nullopt},{source[2],std::nullopt,std::nullopt}};
        };
        auto gpu=mx::compile([](const std::vector<Tensor> &a) {
            auto gate=mx::matmul(a[0],mx::transpose(a[1])),up=mx::matmul(a[0],mx::transpose(a[2]));
            return std::vector<Tensor>{mx::matmul(silu(gate)*up,mx::transpose(a[3]))};
        });
        workload.gpu=[&](int layer,const Tensor &input) {
            ++full;const auto &source=sources.at(layer);return gpu({input,source[0],source[1],source[2]})[0];
        };
        workload.channel_gpu=[&](int layer,const Tensor &input,int first,int count) {
            ++heads;const auto &source=sources.at(layer);
            auto gate=mx::matmul(input,mx::transpose(slice_axis(source[0],0,first,first+count)));
            auto up=mx::matmul(input,mx::transpose(slice_axis(source[1],0,first,first+count)));
            auto hidden=silu(gate)*up;
            return std::make_pair(mx::matmul(hidden,mx::transpose(slice_axis(source[2],1,first,first+count))),hidden);
        };
        ane::HybridFfn automatic(argv[1],h,f,512u<<20,cancelled,false,&workload);
        std::cout<<"native auto callback counts: sources="<<supplied<<" full="<<full<<" heads="<<heads
            <<" decision="<<automatic.selection_label()<<std::endl;
        check(supplied==5 && full>0 && heads>0,"native constructor bypassed the model-supplied sampling callbacks");
        check(automatic.usable_configuration() && !automatic.metrics().runtime_failed,"native auto returned a failed configuration");
        check(automatic.selection_label().find("native channel auto")!=std::string::npos,"native calibration decision missing from actual selection");
        if(automatic.available()) {
            check(automatic.channel_split() && automatic.ane_channels()>0 && automatic.ane_channels()%512==0 &&
                automatic.ane_channels()<f,"automatic accepted graph has invalid selected geometry");
            check(automatic.selection_label().find("accepted complete-runtime")!=std::string::npos,
                "graph adopted without the complete-runtime candidate trial");
        } else {
            check(automatic.plan_block(0,rows).mode==ane::RowScheduler::Mode::Gpu,"declined auto route still scheduled hybrid");
            check(automatic.metrics().runtime_weight_fallback_blocks==0,"voluntary GPU-only counted as failed scratch fallback");
        }
        workload.adapter_identity="actual-adapter";
        const int old_full=full,old_heads=heads;
        ane::HybridFfn adapter(argv[1],h,f,512u<<20,cancelled,true,&workload);
        check(!adapter.available() && adapter.usable_configuration() && !adapter.metrics().runtime_failed &&
            full==old_full && heads==old_heads,"adapter reused a base-only cost model or failed the GPU-only route");
        workload.adapter_identity.clear();workload.rows=34;
        ane::HybridFfn multichunk(argv[1],h,f,512u<<20,cancelled,false,&workload);
        check(!multichunk.available() && multichunk.usable_configuration() && !multichunk.metrics().runtime_failed,
            "unmeasured multiple chunks silently reused a single-chunk model");
        std::cout<<"PASS native automatic channel constructor: actual one/four W8 sampling, padded rows, "
            "accepted-or-voluntary-GPU selection and adapter/multi-chunk isolation\n";
        std::cout<<automatic.selection_label()<<"; synthetic fixture, no model/E2E performance qualification\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
