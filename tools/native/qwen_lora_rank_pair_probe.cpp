#include "lora_rank_pair_candidate.hpp"
#include "backends/dense_gpu_projection.hpp"
#include "backends/dense_gpu_lora_b.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
namespace {
double rel(const Tensor &a,const Tensor &b) {
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    return mx::sqrt(mx::sum(mx::square(af-bf))/mx::maximum(mx::sum(mx::square(bf)),Tensor(1e-20f))).item<float>();
}
void self_test() {
    int cases=0,rejected=0;
    for(auto dtype:{mx::float16,mx::bfloat16})for(int m:{1,33,145})for(int rank:{64,256})for(bool strided:{false,true}) {
        const int k=512;
        auto w0=mx::astype(mx::reshape(mx::sin(mx::arange((rank+2)*768,mx::float32)*.003f)*.04f,{rank+2,768}),dtype);
        auto w1=mx::astype(mx::reshape(mx::cos(mx::arange((rank+2)*896,mx::float32)*.009f)*.02f,{rank+2,896}),dtype);
        auto a0=mx::slice(w0,{1,0},{rank+1,768}),a1=mx::slice(w1,{1,0},{rank+1,896});
        auto source=mx::astype(mx::reshape(mx::sin(mx::arange(m*k,mx::float32)*.017f)*.2f,strided?mx::Shape{k,m}:mx::Shape{m,k}),dtype);
        auto x=mx::expand_dims(strided?mx::transpose(source):source,0);mx::eval({x,a0,a1});
        auto oracle0=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(mx::slice(a0,{0,128},{rank,640}),mx::float32)));
        auto oracle1=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(mx::slice(a1,{0,128},{rank,640}),mx::float32)));
        mx::eval({oracle0,oracle1});
        for(int bm:{16,32,64})for(int bn:{64,128})for(bool same_group:{false,true}) {
            if(rank%bn)continue;
            auto output=research::lora_pair::ranks(x,a0,a1,128,640,bm,bn,same_group);mx::eval(output);
            require(output.size()==2 && output[0].dtype()==mx::float32 && output[1].dtype()==mx::float32 &&
                output[0].flags().row_contiguous && output[1].flags().row_contiguous &&
                mx::all(mx::isfinite(output[0])).item<bool>() && mx::all(mx::isfinite(output[1])).item<bool>() &&
                rel(output[0],oracle0)<=1e-5 && rel(output[1],oracle1)<=1e-5,"paired rank source/pitch/tail/oracle mismatch");++cases;
        }
        for(int invalid=0;invalid<8;++invalid) {
            auto other=a1;int begin=128,end=640,bm=16,bn=64;
            if(invalid==0)begin=-1;if(invalid==1)end=128;if(invalid==2)end=900;
            if(invalid==3)bm=24;if(invalid==4)bn=96;if(invalid==5)other=mx::astype(other,mx::float32);
            if(invalid==6)other=mx::slice(other,{0,0},{rank-1,896});if(invalid==7)other=mx::transpose(other);
            bool denied=false;try{(void)research::lora_pair::ranks(x,a0,other,begin,end,bm,bn);}catch(const std::exception &){denied=true;}
            require(denied,"malformed pair ranks accepted");++rejected;
        }
    }
    std::cout<<"PASS "<<cases<<" paired rank Metal numeric cases and "<<rejected<<" malformed contracts: two dtypes, two original sources, offsets/pitches, tails/strided X, independent contiguous F32 outputs\n";
}
}
int main(int argc,char **argv){try {
    configure_streams();if(argc==1){self_test();return 0;}
    require(argc==2,"usage: qwen-lora-rank-pair-probe original-local-adapter");
    Weights w;w.load_file(argv[1],"transformer.transformer_blocks.0.img_mlp.");w.materialize();
    const auto &a0=w.at("gate_layer.lora_A.weight"),&a1=w.at("proj.lora_A.weight");
    const auto &b0=w.at("gate_layer.lora_B.weight"),&b1=w.at("proj.lora_B.weight");
    require(a0.dtype()==mx::bfloat16 && a1.dtype()==mx::bfloat16 && a0.shape()==mx::Shape{256,4096} && a1.shape()==a0.shape() &&
        b0.dtype()==mx::bfloat16 && b1.dtype()==mx::bfloat16 && b0.shape()==mx::Shape{12288,256} && b1.shape()==b0.shape(),"original paired LoRA fixture mismatch");
    struct Recipe{std::string name;int bm,bn;bool same_group;int packed;};
    std::vector<Recipe> recipes{{"current_joint_two_a",16,64,false,0}};
    for(const auto &tile:std::vector<std::pair<int,int>>{{16,64},{16,128},{32,64},{32,128}})for(bool same:{false,true})
        recipes.push_back({std::string(same?"same_tg":"paired_dispatch")+"_m"+std::to_string(tile.first)+"n"+std::to_string(tile.second),tile.first,tile.second,same,0});
    recipes.push_back({"packed_a_f32_rank_views",16,64,false,1});
    auto packed=mx::contiguous(mx::concatenate({a0,a1},0));mx::eval(packed);
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-paired-rank-screen-v1\",\"scope\":\"original layer0 independent BF16 A/B, synthetic input; online two A ranks and complete B/scale/delta, serial host spans; not model/visual/physical single-X-read\",\"packed_candidate_bytes\":"<<packed.nbytes()<<",\"cases\":[";
    bool comma=false;
    for(int m:{1024,2096,3144})for(int portion:{0,1}) {
        const int first=portion?7168:0,n=portion?5120:7168;
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(m*4096,mx::float32)*.001f)*.125f,{1,m,4096}),mx::bfloat16);mx::eval(x);
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;std::vector<Fn> rank_functions,delta_functions;
        for(size_t i=0;i<recipes.size();++i) {
            const auto r=recipes[i];
            auto ranks=mx::compile([&,r,i](const std::vector<Tensor> &input)->std::vector<Tensor>{
                if(i==0)return {dense_gpu::projection_range(input[0],a0,0,256,0,4096,16,true,64),dense_gpu::projection_range(input[0],a1,0,256,0,4096,16,true,64)};
                if(r.packed)return mx::split(dense_gpu::projection_range(input[0],packed,0,512,0,4096,16,true,64),2,-1);
                return research::lora_pair::ranks(input[0],a0,a1,0,4096,r.bm,r.bn,r.same_group);
            });
            rank_functions.push_back(ranks);
            delta_functions.push_back(mx::compile([&,ranks,first,n](const std::vector<Tensor> &input) {
                auto low=ranks(input);return std::vector<Tensor>{dense_gpu::lora_b_epilogue(low[0],b0,first,first+n,.75f,mx::bfloat16),
                    dense_gpu::lora_b_epilogue(low[1],b1,first,first+n,.75f,mx::bfloat16)};
            }));
        }
        const auto expected=delta_functions[0]({x});mx::eval(expected);std::vector<double> errors;
        std::vector<std::vector<double>> rank_ms(recipes.size()),delta_ms(recipes.size());
        for(size_t i=0;i<recipes.size();++i) {
            const auto y=delta_functions[i]({x});mx::eval(y);const double error=std::max(rel(y[0],expected[0]),rel(y[1],expected[1]));
            require(std::isfinite(error) && error<=.05 && mx::all(mx::isfinite(y[0])).item<bool>() && mx::all(mx::isfinite(y[1])).item<bool>(),"paired online delta numerical gate failed");errors.push_back(error);
            for(int warm=0;warm<3;++warm){mx::eval(rank_functions[i]({x}));mx::eval(delta_functions[i]({x}));}
        }
        for(int sample=0;sample<15;++sample)for(size_t visit=0;visit<recipes.size();++visit) {
            const size_t i=(visit+size_t(sample))%recipes.size();auto start=Clock::now();mx::eval(rank_functions[i]({x}));
            rank_ms[i].push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());start=Clock::now();mx::eval(delta_functions[i]({x}));
            delta_ms[i].push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;std::cout<<"{\"M\":"<<m<<",\"rank_per_source\":256,\"K\":4096,\"first\":"<<first<<",\"N_per_delta\":"<<n<<",\"recipes\":[";
        for(size_t i=0;i<recipes.size();++i) {
            std::cout<<(i?",":"")<<"{\"name\":\""<<recipes[i].name<<"\",\"delta_rel_l2_vs_current_joint\":"<<errors[i]<<",\"rank_samples_ms\":[";
            for(size_t j=0;j<rank_ms[i].size();++j){std::cout<<(j?",":"")<<rank_ms[i][j];}
            std::cout<<"],\"online_delta_samples_ms\":[";
            for(size_t j=0;j<delta_ms[i].size();++j){std::cout<<(j?",":"")<<delta_ms[i][j];}
            std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"warmups_per_arm\":3,\"samples_per_arm\":15,\"qualification_passed\":false,\"default_route_changed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
