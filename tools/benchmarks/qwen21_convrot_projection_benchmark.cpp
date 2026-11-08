// Production Weights routes, original checkpoint bytes, synthetic BF16 input.
// The Python runner extracts selected tensors and keeps provenance beside JSON.
#include "../../native/backends/mlx.hpp"
#include "turbocider/turbocider.h"
#include <mlx/backend/metal/metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using mx::array;
using Clock=std::chrono::steady_clock;

static double quantile(std::vector<double> values,double p) {
    std::sort(values.begin(),values.end());
    const auto index=p*(values.size()-1);
    const auto lo=size_t(index),hi=std::min(lo+1,values.size()-1);
    return values[lo]+(values[hi]-values[lo])*(index-lo);
}
static std::vector<float> floats(const std::vector<array>& arrays) {
    std::vector<float> result;
    for(const auto& value:arrays) {
        auto data=mx::contiguous(mx::astype(value,mx::float32));mx::eval(data);
        result.insert(result.end(),data.data<float>(),data.data<float>()+data.size());
    }
    return result;
}
int main(int argc,char** argv) {try {
    if(argc!=6 && argc!=7)throw std::invalid_argument("expected weights, projection count, rows, warmup, repeats, optional capture path");
    const int projections=std::stoi(argv[2]),rows=std::stoi(argv[3]),warmup=std::stoi(argv[4]),repeats=std::stoi(argv[5]);
    if(projections<1 || projections>2 || rows<1 || warmup<1 || repeats<3)
        throw std::invalid_argument("invalid benchmark geometry or repetition count");
    char* raw_identity=tc_runtime_build_identity();
    if(!raw_identity)throw std::runtime_error("native identity missing");
    const std::string identity(raw_identity);tc_string_free(raw_identity);
    if(identity!=TC_TEST_BOUND_NATIVE_ID)throw std::runtime_error("ConvRot Weights ABI requires the matching library");
    if(!tc::convrot_w8a8_available())throw std::runtime_error("M5 TensorOps unavailable");
    tc::configure_streams();
    tc::Weights reference,candidate;reference.load_file(argv[1]);candidate.load_file(argv[1]);
    reference.pack_convrot_q8(128,mx::float32);candidate.set_convrot_w8a8(true);
    reference.materialize();candidate.materialize();
    const int k=candidate.at("p0.weight").shape(1);
    std::vector<int> n;
    for(int i=0;i<projections;++i)n.push_back(candidate.at("p"+std::to_string(i)+".weight").shape(0));
    std::mt19937 generator(20261106);std::normal_distribution<float> normal(0.f,1.f);
    std::vector<array> inputs;
    for(int alternate=0;alternate<2;++alternate) {
        std::vector<float> source(size_t(rows)*k);
        for(auto& v:source)v=normal(generator);
        inputs.push_back(mx::astype(array(source.data(),{1,rows,k},mx::float32),mx::bfloat16));
    }
    mx::eval(inputs);
    const auto make=[&](const tc::Weights& weights) {
        return mx::compile([&weights,projections](const std::vector<array>& input) {
            std::vector<array> output;
            for(int i=0;i<projections;++i)output.push_back(weights.project(input[0],"p"+std::to_string(i)));
            return output;
        });
    };
    auto base=make(reference),a8=make(candidate);
    if(argc==7) {
        // Keep initial pipeline creation inside the capture so its function
        // descriptors are recorded alongside the dispatches. Never time it.
        mx::synchronize();
        const auto before_capture=tc::convrot_w8a8_stats();
        mx::metal::start_capture(argv[6]);
        try {auto output=a8({inputs[1]});mx::eval(output);mx::synchronize();}
        catch(...) {mx::metal::stop_capture();throw;}
        mx::metal::stop_capture();
        const auto after_capture=tc::convrot_w8a8_stats();
        if(after_capture.projections-before_capture.projections!=uint64_t(projections))
            throw std::runtime_error("capture did not encode the expected W8A8 projections");
        std::cout<<"{\"capture_only\":true,\"rows\":"<<rows
            <<",\"gpu_encoded_projections\":"<<after_capture.projections-before_capture.projections
            <<",\"gpu_quantized_rows\":"<<after_capture.activation_rows-before_capture.activation_rows<<"}\n";
        return 0;
    }
    const auto before=tc::convrot_w8a8_stats();
    auto original=base({inputs[0]}),changed=a8({inputs[0]});mx::eval(original);mx::eval(changed);
    auto a=floats(original),b=floats(changed);
    double error=0,norma=0,normb=0,dot=0,maxabs=0;
    for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i]) || !std::isfinite(b[i]))throw std::runtime_error("nonfinite projection");
        const double difference=double(b[i])-a[i];error+=difference*difference;
        norma+=double(a[i])*a[i];normb+=double(b[i])*b[i];dot+=double(a[i])*b[i];
        maxabs=std::max(maxabs,std::abs(difference));
    }
    a.clear();a.shrink_to_fit();b.clear();b.shrink_to_fit();original.clear();changed.clear();
    std::vector<double> baseline_ms,w8a8_ms;
    const auto call=[&](auto& function,const array& input) {
        const auto begin=Clock::now();auto output=function({input});mx::eval(output);
        return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    };
    for(int i=0;i<warmup+repeats;++i) {
        double bt,at;
        if(i%2==0){bt=call(base,inputs[i%2]);at=call(a8,inputs[i%2]);}
        else {at=call(a8,inputs[i%2]);bt=call(base,inputs[i%2]);}
        if(i>=warmup){baseline_ms.push_back(bt);w8a8_ms.push_back(at);}
    }
    const auto after=tc::convrot_w8a8_stats();
    const auto expected=uint64_t(projections)*uint64_t(1+warmup+repeats);
    if(after.projections-before.projections!=expected || after.activation_rows-before.activation_rows!=expected*rows)
        throw std::runtime_error("compiled benchmark did not encode the expected actual W8A8 projections");
    const double bm=quantile(baseline_ms,.5),am=quantile(w8a8_ms,.5);
    std::cout<<std::setprecision(12)<<"{\"rows\":"<<rows<<",\"k\":"<<k<<",\"n\":[";
    for(size_t i=0;i<n.size();++i)std::cout<<(i?",":"")<<n[i];
    std::cout<<"],\"warmup\":"<<warmup<<",\"repeats\":"<<repeats
        <<",\"reference_median_ms\":"<<bm<<",\"reference_p90_ms\":"<<quantile(baseline_ms,.9)
        <<",\"w8a8_median_ms\":"<<am<<",\"w8a8_p90_ms\":"<<quantile(w8a8_ms,.9)
        <<",\"speedup\":"<<bm/am<<",\"relative_l2\":"<<std::sqrt(error/std::max(norma,1e-30))
        <<",\"cosine\":"<<dot/std::sqrt(std::max(norma*normb,1e-30))<<",\"max_absolute\":"<<maxabs
        <<",\"gpu_encoded_projections\":"<<after.projections-before.projections
        <<",\"gpu_quantized_rows\":"<<after.activation_rows-before.activation_rows
        <<",\"reference_ms\":[";
    for(size_t i=0;i<baseline_ms.size();++i)std::cout<<(i?",":"")<<baseline_ms[i];
    std::cout<<"],\"w8a8_ms\":[";
    for(size_t i=0;i<w8a8_ms.size();++i)std::cout<<(i?",":"")<<w8a8_ms[i];
    std::cout<<"]}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
