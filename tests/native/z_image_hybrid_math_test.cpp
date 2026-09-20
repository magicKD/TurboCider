#include "models/z_image/hybrid_math.hpp"
#include <cmath>
#include <iostream>
using namespace tc;
using namespace tc::z_image;
Tensor weight(int projection, bool compact) {
    const int h=3840, width=compact?5120:10240, start=compact?0:5120;
    std::vector<float> data(size_t(h)*width,0);
    const int channel[3]={start,start+7,start+5119};
    if(projection==2) {
        for(int i=0;i<3;++i)data[size_t(i==0?0:i==1?2000:3839)*width+channel[i]]=i==0?1:i==1?2:-1;
    } else {
        const float gate[3]={4,-2,-1},up[3]={4,-4,2};
        for(int i=0;i<3;++i)data[size_t(channel[i])*h+i]=projection==1?gate[i]:up[i];
    }
    auto out=mx::astype(Tensor(data.data(),projection==2?mx::Shape{h,width}:mx::Shape{width,h},mx::float32),mx::bfloat16);
    mx::eval(out); return out;
}
template<class F> void rejects(F f) {
    bool rejected=false;try{f();}catch(const std::exception &){rejected=true;}
    require(rejected,"expected hybrid math rejection");
}
int main() {
    mx::set_default_device(mx::Device(mx::Device::gpu,0));
    auto graph=make_hybrid_gpu_graph(3840,10240,5120);
    std::vector<float> values(2*3840,0); values[0]=.25f;values[1]=-.5f;values[2]=1;
    auto input=mx::astype(Tensor(values.data(),{1,2,3840},mx::float32),mx::bfloat16); mx::eval(input);
    auto full=graph({input,weight(1,false),weight(3,false),weight(2,false)})[0]; mx::eval(full);
    auto suffix=graph({input,weight(1,true),weight(3,true),weight(2,true)})[0]; mx::eval(suffix);
    require(mx::array_equal(full,suffix).item<bool>(),"full and compact suffix differ");
    std::vector<float> expected(2*3840,0);
    expected[0]=1.f/(1.f+std::exp(-1.f));expected[2000]=4.f/(1.f+std::exp(-1.f));expected[3839]=2.f/(1.f+std::exp(1.f));
    auto oracle=Tensor(expected.data(),{1,2,3840},mx::float32);
    require(mx::max(mx::abs(mx::astype(suffix,mx::float32)-oracle)).item<float>()<.03f,"independent sparse MLP oracle failed");
    rejects([&]{make_hybrid_gpu_graph(3840,10240,10240);});
    rejects([&]{graph({input,input,input});});
    rejects([&]{graph({mx::zeros({1,2,3839},mx::bfloat16),weight(1,true),weight(3,true),weight(2,true)});});
    auto gpu=mx::contiguous(mx::ones({1,2,3840},mx::bfloat16));
    auto ane=mx::contiguous(mx::full({1,2,3840},Tensor(32752.f),mx::float16));
    auto scale=Tensor(32.f,mx::bfloat16);
    auto joined=join_hybrid_ffn(gpu,ane,scale); mx::eval(joined);
    require(mx::all(mx::equal(joined,Tensor(1048576.f,mx::bfloat16))).item<bool>(),"BF16 cast-before-scale order failed");
    auto compiled=mx::compile([](const std::vector<Tensor> &a){return std::vector<Tensor>{join_hybrid_ffn(a[0],a[1],a[2])};});
    auto fused=compiled({gpu,ane,scale})[0]; mx::eval(fused);
    require(mx::array_equal(joined,fused).item<bool>(),"compiled/eager join differ");
    // Reuse only after both consumers finish; completed outputs own their data.
    ane.data<mx::float16_t>()[0]=mx::float16_t(0.f);
    require(joined.data<mx::bfloat16_t>()[0]==mx::bfloat16_t(1048576.f),"completed join aliases ANE input");
    rejects([&]{join_hybrid_ffn(gpu,mx::zeros({1,1,3840},mx::float16),scale);});
    rejects([&]{join_hybrid_ffn(gpu,mx::astype(ane,mx::float32),scale);});
    rejects([&]{join_hybrid_ffn(gpu,ane,Tensor(32.f,mx::float32));});
    std::cout<<"Metal full/compact suffix, sparse oracle, eager/compiled cast-scale join, completed-output independence and invalid geometry passed\n";
}
