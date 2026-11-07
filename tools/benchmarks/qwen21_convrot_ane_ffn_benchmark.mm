#include "../../native/backends/private/ane_w8_executor.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "turbocider/turbocider.h"
#include <mlx/allocator.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
using namespace tc;
namespace ffn=qwen21::runtime_ffn;
using Clock=std::chrono::steady_clock;
constexpr int hidden=4096, width=12288;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}

ane::DeviceMatrixView matrix(const Tensor &value, int rows, int cols, ane::DType dtype) {
    require(value.flags().row_contiguous && value.buffer().ptr() && value.offset()>=0 &&
            value.size()==size_t(rows)*cols,"ANE benchmark expects a ready contiguous matrix");
    return {const_cast<void*>(value.buffer().ptr()),value.buffer_size(),size_t(value.offset()),rows,cols,
            size_t(cols)*value.itemsize(),dtype,std::make_shared<Tensor>(value),value.data_shared_ptr()};
}

ane::DeviceWeightView weights(const Tensor &codes, const Tensor &scales, int first, int rows, int cols) {
    require(codes.dtype()==mx::int8 && scales.dtype()==mx::float32 && codes.ndim()==2 &&
            codes.shape(1)==cols && scales.shape()==mx::Shape{codes.shape(0),1} && first>=0 &&
            rows>0 && first<=codes.shape(0)-rows && codes.flags().row_contiguous && codes.buffer().ptr(),
            "original ConvRot signed codes/F32 row coefficients required");
    auto scale=matrix(scales,scales.shape(0),1,ane::DType::FP32);
    scale.rows=rows;
    scale.offset_bytes+=size_t(first)*scale.row_stride_bytes;
    return {const_cast<void*>(codes.buffer().ptr()),codes.buffer_size(),size_t(codes.offset())+size_t(first)*cols,
            size_t(cols),rows,cols,ane::DeviceWeightEncoding::ConvrotQ8Signed,ane::DType::FP32,32,scale,{},
            std::make_shared<Tensor>(codes),codes.data_shared_ptr(),true};
}

double relative_l2(const Tensor &base, const Tensor &candidate) {
    auto a=mx::contiguous(mx::astype(base,mx::float32));
    auto b=mx::contiguous(mx::astype(candidate,mx::float32));
    mx::eval(a,b);
    double diff=0,norm=0;
    for(size_t i=0;i<a.size();++i) {
        const double x=a.data<float>()[i],y=b.data<float>()[i];
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite FFN result");
        diff+=(x-y)*(x-y);
        norm+=x*x;
    }
    require(norm>0,"empty FFN result");
    return std::sqrt(diff/norm);
}

void print_samples(const std::vector<double> &values) {
    std::cout<<'[';
    for(size_t i=0;i<values.size();++i)std::cout<<(i?",":"")<<values[i];
    std::cout<<']';
}
}

int main(int argc,char **argv) {
    @autoreleasepool {
        try {
            require(argc==7,"usage: benchmark gate-up.safetensors down.safetensors private-cache rows ane-channels repeats");
            const int rows=std::stoi(argv[4]),ane_channels=std::stoi(argv[5]),repeats=std::stoi(argv[6]);
            require((rows==256 || rows==1024) &&
                    (ane_channels==3072 || ane_channels==4096 || ane_channels==6144) &&
                    repeats>=3 && repeats<=15,"benchmark geometry outside the bounded matrix");
            char *raw_identity=tc_runtime_build_identity();
            require(raw_identity,"missing native library build identity");
            const std::string identity(raw_identity);tc_string_free(raw_identity);
            require(identity==TC_TEST_BOUND_NATIVE_ID,"benchmark requires matching native library build");
            require(convrot_w8a8_available(),"M5 Metal TensorOps unavailable");
            configure_streams();
            mx::set_cache_limit(size_t(64)<<20);
            mx::reset_peak_memory();
            auto gate_up=mx::load_safetensors(argv[1]).first;
            auto down=mx::load_safetensors(argv[2]).first;
            require(!gate_up.contains("p0.bias") && !down.contains("p0.bias"),"probe supports bias-free fused FFN");
            const auto &gu=gate_up.at("p0.weight"),&gus=gate_up.at("p0.weight_scale");
            const auto &d=down.at("p0.weight"),&ds=down.at("p0.weight_scale");
            require(gu.shape()==mx::Shape{2*width,hidden} && d.shape()==mx::Shape{hidden,width},"unexpected real Qwen FFN shape");
            Weights source;
            source.bind_arrays({"gate_up.weight","gate_up.weight_scale","gate_up.comfy_quant",
                                "out.weight","out.weight_scale","out.comfy_quant"},
                {gu,gus,gate_up.at("p0.comfy_quant"),d,ds,down.at("p0.comfy_quant")});
            source.set_convrot_w8a8(true);
            source.materialize();
            const int gpu_channels=width-ane_channels;
            const auto basis=ane::W8Basis::ComfyH256;
            const std::vector<ane::DeviceWeightRegion> regions{
                {weights(gu,gus,0,width,hidden),{gpu_channels,ane_channels,0,hidden,256,0,false,basis}},
                {weights(gu,gus,width,width,hidden),{gpu_channels,ane_channels,0,hidden,256,0,false,basis}},
                {weights(d,ds,0,hidden,width),{0,hidden,gpu_channels,ane_channels,256,0,false,basis}}};
            // Compile exactly one fixed Comfy W8 graph with only one layer's
            // resident codes. The capability probe is outside warm timing.
            ane::PrivateW8Graph graph({ane::Kind::SwiGLU,rows,hidden,ane_channels,2048,1024,false},
                                     size_t(640)<<20,argv[3],basis);
            std::string error;
            if (!graph.self_test(error)) throw std::runtime_error("Private ANE self-test failed: "+error);
            const auto capability_calls=graph.capability_calls();
            require(graph.supports_fp32_device_output() && capability_calls>0,"missing ANE F32 partial capability");
            auto full=ffn::full(source,"");
            auto complement=ffn::channels(source,"",0,gpu_channels,hidden,width,true);
            std::mt19937 random(20261107);
            std::normal_distribution<float> normal;
            std::vector<Tensor> inputs;
            for(int input=0;input<2;++input) {
                std::vector<mx::bfloat16_t> values(size_t(rows)*hidden);
                for(auto &value:values)value=mx::bfloat16_t(normal(random));
                inputs.emplace_back(values.data(),mx::Shape{1,rows,hidden},mx::bfloat16);
            }
            mx::eval(inputs);
            auto output=Tensor(mx::allocator::malloc(size_t(rows)*hidden*sizeof(float)),{1,rows,hidden},mx::float32);
            const auto device_output=matrix(output,rows,hidden,ane::DType::FP32);
            std::vector<double> gpu,hybrid,hybrid_with_stage,staging,ane_predictions;
            uint64_t calls=0;
            double max_error=0;
            auto run_gpu=[&](const Tensor &input, bool timed) {
                const auto start=Clock::now();
                auto value=full({input})[0];
                mx::eval(value);
                if(timed)gpu.push_back(elapsed(start));
                return value;
            };
            auto run_hybrid=[&](const Tensor &input, bool timed) {
                const auto with_stage=Clock::now();
                graph.stage_device_weight_regions(regions);
                const auto stage=graph.wait_stage();
                require(stage.ok,"ANE source staging failed: "+stage.error);
                const double stage_time=elapsed(with_stage);
                const auto ready=Clock::now();
                graph.launch_device(matrix(input,rows,hidden,ane::DType::BF16),device_output);
                auto gpu_partial=complement({input})[0];
                mx::async_eval(gpu_partial);
                const auto result=graph.finish();
                require(result.ok && result.calls==1 && result.overflow_retries==0,"ANE prediction failed: "+result.error);
                auto value=mx::astype(gpu_partial+output,input.dtype());
                mx::eval(value);
                const auto completed=Clock::now();
                if(timed) {
                    calls+=result.calls;
                    staging.push_back(stage_time);
                    ane_predictions.push_back(result.prediction_seconds);
                    hybrid.push_back(std::chrono::duration<double>(completed-ready).count());
                    hybrid_with_stage.push_back(std::chrono::duration<double>(completed-with_stage).count());
                }
                return value;
            };
            for(int i=-2;i<repeats;++i) {
                const auto &input=inputs[(i+2)%2];
                auto gpu_result=input,hybrid_result=input;
                if((i+2)%2==0) {
                    gpu_result=run_gpu(input,i>=0);
                    hybrid_result=run_hybrid(input,i>=0);
                } else {
                    hybrid_result=run_hybrid(input,i>=0);
                    gpu_result=run_gpu(input,i>=0);
                }
                if(i<0)max_error=std::max(max_error,relative_l2(gpu_result,hybrid_result));
            }
            mx::synchronize();
            const auto policy=graph.program_policy_stats();
            require(policy.fixed && policy.instances==1 && !policy.blocked_rebuilds && !policy.disabled,
                    "fixed ANE program did not remain healthy");
            std::cout<<std::setprecision(12)<<"{\"rows\":"<<rows<<",\"ane_channels\":"<<ane_channels
                <<",\"runtime_id\":"<<std::quoted(identity)<<",\"data_path\":"<<std::quoted(graph.data_path())
                <<",\"capability_predictions\":"<<capability_calls<<",\"timed_predictions\":"<<calls
                <<",\"load_seconds\":"<<graph.load_seconds()<<",\"max_warmup_relative_l2\":"<<max_error
                <<",\"slot_bytes\":"<<graph.slot_bytes()<<",\"graph_estimated_bytes\":"<<graph.estimated_bytes()
                <<",\"peak_mlx_bytes\":"<<mx::get_peak_memory()<<",\"gpu_seconds\":";print_samples(gpu);
            std::cout<<",\"hybrid_ready_seconds\":";print_samples(hybrid);
            std::cout<<",\"hybrid_with_stage_seconds\":";print_samples(hybrid_with_stage);
            std::cout<<",\"stage_seconds\":";print_samples(staging);
            std::cout<<",\"ane_prediction_seconds\":";print_samples(ane_predictions);
            std::cout<<"}\n";
        } catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
