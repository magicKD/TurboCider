// Replay actual original-BF16 dense FFN inputs. No performance/model gate.
#include "backends/mlx.hpp"
#include "backends/private/ane_executor.hpp"
#include "models/z_image/metal/projection.hpp"
#include "models/z_image/metal/swiglu_gemm.hpp"
#include "platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

using namespace tc;
namespace {
constexpr int hidden=3840,width=10240,bucket=352;
Tensor round_to(const Tensor &a,mx::Dtype dtype) {
    auto out=mx::astype(a,dtype);mx::eval(out);return out;
}
struct Ffn { Tensor gate,up,activation,output; };
Ffn gpu_ffn(const Tensor &input,const std::vector<Tensor> &w,bool mpp,bool fused) {
    auto project=[&](const Tensor &x,const Tensor &weight) {
        return mpp?z_metal::projection(x,weight):mx::matmul(x,mx::transpose(weight));
    };
    auto gate=project(input,w[0]),up=project(input,w[1]);
    auto h=fused?z_metal::swiglu_gemm(input,w[0],up):(gate*mx::sigmoid(gate))*up;
    auto y=project(h,w[2]);mx::eval({gate,up,h,y});return {gate,up,h,y};
}
Tensor fp16_project(const Tensor &x,const Tensor &w,int tile) {
    std::optional<Tensor> total;
    for(int begin=0;begin<x.shape(-1);begin+=tile) {
        const int end=std::min(begin+tile,x.shape(-1));
        auto p=round_to(mx::matmul(slice_axis(x,-1,begin,end),
            mx::transpose(slice_axis(w,1,begin,end))),mx::float16);
        total=total?round_to(*total+p,mx::float16):p;
    }
    return *total;
}
Ffn emulate_fp16(const Tensor &input,const std::vector<Tensor> &w,int tile,float scale) {
    auto x=round_to(input,mx::float16);
    auto g=fp16_project(x,round_to(w[0],mx::float16),tile);
    auto u=fp16_project(x,round_to(mx::astype(w[1],mx::float32)/Tensor(scale),mx::float16),tile);
    auto neg=round_to(g*Tensor(-1.f,mx::float16),mx::float16);
    auto eg=round_to(mx::exp(neg),mx::float16);
    auto denom=round_to(eg+Tensor(1.f,mx::float16),mx::float16);
    auto s=round_to(g/denom,mx::float16);
    auto h=round_to(s*u,mx::float16);
    auto y=fp16_project(h,round_to(w[2],mx::float16),tile);
    auto restore=[&](const Tensor &v) {return round_to(mx::astype(v,mx::float32)*Tensor(scale),mx::bfloat16);};
    return {round_to(g,mx::bfloat16),restore(u),restore(h),restore(y)};
}
void metric(const Tensor &candidate,const Tensor &reference) {
    require(candidate.shape()==reference.shape(),"replay comparison shape mismatch");
    auto a=mx::contiguous(mx::astype(candidate,mx::float32));
    auto b=mx::contiguous(mx::astype(reference,mx::float32));mx::eval({a,b});
    double diff=0,aa=0,bb=0,dot=0,maximum=0;size_t wrong=0;
    for(size_t i=0;i<a.size();++i) {
        const double x=a.data<float>()[i],y=b.data<float>()[i];
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite replay comparison");
        diff+=(x-y)*(x-y);aa+=x*x;bb+=y*y;dot+=x*y;
        maximum=std::max(maximum,std::abs(x-y));wrong+=x!=y;
    }
    require(aa>0 && bb>0,"zero-energy replay comparison");
    std::cout<<"{\"rel_l2\":"<<std::sqrt(diff/bb)<<",\"cosine\":"<<dot/std::sqrt(aa*bb)
        <<",\"max_abs\":"<<maximum<<",\"unequal_values\":"<<wrong<<",\"elements\":"<<a.size()<<"}";
}
void stats(const Tensor &input) {
    auto a=mx::contiguous(mx::astype(input,mx::float32));mx::eval(a);
    size_t negative_exp=0,positive=0;double maximum=0,energy=0;
    for(size_t i=0;i<a.size();++i) {
        const float v=a.data<float>()[i];require(std::isfinite(v),"nonfinite replay statistics");
        maximum=std::max(maximum,std::abs(double(v)));energy+=double(v)*v;
        negative_exp+=v<-11.089866f;positive+=v>11.089866f;
    }
    std::cout<<"{\"max_abs\":"<<maximum<<",\"rms\":"<<std::sqrt(energy/a.size())
        <<",\"negative_fp16_exp_overflow_region\":"<<negative_exp
        <<",\"positive_gt_11\":"<<positive<<",\"elements\":"<<a.size()<<"}";
}
ane::MatrixView view(const Tensor &a) {
    require(a.dtype()==mx::bfloat16 && a.flags().row_contiguous,"replay weight must be contiguous original BF16");
    return {a.data<mx::bfloat16_t>(),a.nbytes(),a.shape(0),a.shape(1),size_t(a.shape(1))*2,ane::DType::BF16};
}
ane::DeviceMatrixView device_view(const Tensor &a) {
    require(a.dtype()==mx::bfloat16 && a.flags().row_contiguous && a.buffer().ptr() && a.offset()>=0,
            "replay device view is not ready original BF16");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),int(a.size()/a.shape(-1)),a.shape(-1),
        size_t(a.shape(-1))*2,ane::DType::BF16,std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
}
int main(int argc,char **argv) {
    if(argc!=6) {std::cerr<<"usage: checkpoint captures block fresh-cache tile-k\n";return 2;}
    try {
        const int block=std::stoi(argv[3]),tile=std::stoi(argv[5]);
        require(block>=0&&block<32&&tile>0&&tile<=32768,"invalid replay block/tile");
        configure_streams();
        const auto checkpoint_sha=sha256_file(argv[1]);
        char *identity=tc_runtime_build_identity();require(identity,"runtime build identity absent");
        const std::string build(identity);tc_string_free(identity);
        auto all=mx::load_safetensors(argv[1]).first;
        const auto prefix=(block<2?"noise_refiner.":"layers.")+std::to_string(block<2?block:block-2)+".feed_forward";
        std::vector<Tensor> weights;
        for(const auto *name:{".w1.weight",".w3.weight",".w2.weight"}) {
            const auto key=prefix+name;
            if(all.contains(key))weights.push_back(all.at(key));
            else if(all.contains("diffusion_model."+key))weights.push_back(all.at("diffusion_model."+key));
            else weights.push_back(all.at("transformer."+key));
        }
        require(weights[0].shape()==mx::Shape{width,hidden}&&weights[1].shape()==weights[0].shape()&&
                weights[2].shape()==mx::Shape{hidden,width},"dense replay checkpoint FFN geometry mismatch");
        for(const auto &w:weights)require(w.dtype()==mx::bfloat16,"dense replay never recasts source weights");
        mx::eval(weights);
        std::vector<std::filesystem::path> files;
        for(const auto &entry:std::filesystem::directory_iterator(argv[2])) {
            require(entry.is_regular_file()&&!entry.is_symlink(),"capture directory contains nonregular data");
            require(entry.path().extension()==".safetensors" && entry.path().filename().string().find("partial")==std::string::npos,
                    "capture directory contains incomplete data");files.push_back(entry.path());
        }
        std::sort(files.begin(),files.end());require(!files.empty()&&files.size()<=32,"replay capture count must be 1..32");
        setenv("TURBOCIDER_PRIVATE_ANE_GPU_IO","1",1);
        ane::PrivateGraph graph({ane::Kind::SwiGLU,bucket,hidden,width,tile,512,true},1ull<<30,argv[4]);
        std::string error;require(graph.self_test(error),error);
        graph.stage_weights({view(weights[0]),view(weights[1]),view(weights[2])});
        require(graph.wait_stage().ok,"replay original BF16 weight staging failed");
        auto out=mx::zeros({1,bucket,hidden},mx::bfloat16),h=mx::zeros({1,bucket,width},mx::bfloat16);
        auto zero=mx::zeros({1,bucket,width},mx::bfloat16);mx::eval({out,h,zero});
        const bool fused=z_image_mpp_swiglu_default();
        for(size_t index=0;index<files.size();++index) {
            const auto capture_sha=sha256_file(files[index]);
            auto captured=mx::load_safetensors(files[index].string());
            const auto &metadata=captured.second;
            require(metadata.at("capture_recipe")=="z-dense-runtime-original-bf16-v1" && metadata.at("block")==std::to_string(block) &&
                    metadata.at("sample")==std::to_string(index) && metadata.at("runtime_build")==build,
                    "capture recipe/source sequence/build identity mismatch");
            require(captured.first.size()==1,"capture must contain exactly original input");
            auto input=captured.first.at("tensor");
            require(input.ndim()==3 && input.shape(0)==1 && input.shape(1)>=bucket && input.shape(1)<=4224 &&
                    input.shape(2)==hidden && input.dtype()==mx::bfloat16,"capture dtype/geometry mismatch");
            mx::eval(input);
            const bool mpp=z_image_small_shape_metal_default()&&input.shape(1)<=1056;
            auto full=gpu_ffn(input,weights,mpp,fused);
            auto tail=mx::contiguous(slice_axis(input,1,input.shape(1)-bucket,input.shape(1)));mx::eval(tail);
            auto reference=slice_axis(full.output,1,input.shape(1)-bucket,input.shape(1));
            auto reference_h=slice_axis(full.activation,1,input.shape(1)-bucket,input.shape(1));
            auto reference_g=slice_axis(full.gate,1,input.shape(1)-bucket,input.shape(1));
            auto reference_u=slice_axis(full.up,1,input.shape(1)-bucket,input.shape(1));
            auto tail_gpu=gpu_ffn(tail,weights,mpp,fused);
            auto manual=gpu_ffn(tail,weights,mpp,false);
            ane::DeviceAdapterInput adapter{device_view(zero),device_view(zero),device_view(h)};
            graph.launch_device(device_view(tail),device_view(out),adapter);
            const auto result=graph.finish();require(result.ok,result.error);
            auto emulated=emulate_fp16(tail,weights,tile,result.headroom_scale);
            auto bf16_from_projections=(emulated.gate*mx::sigmoid(emulated.gate))*emulated.up;
            auto gpu_down=[&](const Tensor &v) {return mpp?z_metal::projection(v,weights[2]):mx::matmul(v,mx::transpose(weights[2]));};
            auto bf16_projection_down=gpu_down(bf16_from_projections);
            auto emulated_hidden_down=gpu_down(emulated.activation);
            auto native_hidden_down=gpu_down(h);mx::eval({bf16_projection_down,emulated_hidden_down,native_hidden_down});
            std::cout<<std::setprecision(17)<<"{\"schema\":\"tc-z-runtime-ffn-error-replay-v1\",\"scope\":\"real frozen BF16 FFN input; explicit software FP16 model and actual Private execution, NOT E2E/performance qualification\",\"block\":"<<block
                <<",\"sample\":"<<index<<",\"rows\":"<<input.shape(1)<<",\"tail_rows\":"<<bucket
                <<",\"tile_k\":"<<tile<<",\"checkpoint_sha256\":\""<<checkpoint_sha<<"\",\"capture_sha256\":\""<<capture_sha
                <<"\",\"runtime_build\":\""<<build<<"\",\"calls\":"<<result.calls<<",\"retries\":"<<result.overflow_retries
                <<",\"headroom\":"<<result.headroom_scale<<",\"input_stats\":";stats(tail);
            std::cout<<",\"gate_stats\":";stats(reference_g);std::cout<<",\"up_stats\":";stats(reference_u);
            std::cout<<",\"source_gpu_tail_vs_full\":";metric(tail_gpu.output,reference);
            std::cout<<",\"manual_bf16_hidden\":";metric(manual.activation,reference_h);
            std::cout<<",\"fp16_gate_vs_bf16\":";metric(emulated.gate,reference_g);
            std::cout<<",\"fp16_up_vs_bf16\":";metric(emulated.up,reference_u);
            std::cout<<",\"fp16_emulated_hidden\":";metric(emulated.activation,reference_h);
            std::cout<<",\"native_hidden\":";metric(h,reference_h);
            std::cout<<",\"native_hidden_vs_emulated\":";metric(h,emulated.activation);
            std::cout<<",\"bf16_activation_gpu_down\":";metric(bf16_projection_down,reference);
            std::cout<<",\"fp16_activation_gpu_down\":";metric(emulated_hidden_down,reference);
            std::cout<<",\"native_activation_gpu_down\":";metric(native_hidden_down,reference);
            std::cout<<",\"fp16_emulated_output\":";metric(emulated.output,reference);
            std::cout<<",\"native_output\":";metric(out,reference);
            std::cout<<",\"native_output_vs_emulated\":";metric(out,emulated.output);std::cout<<"}"<<std::endl;
            require(sha256_file(files[index])==capture_sha,"capture changed during replay");
        }
        require(sha256_file(argv[1])==checkpoint_sha,"checkpoint changed during replay");
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
