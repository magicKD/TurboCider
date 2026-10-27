#include "../../native/backends/mlx.hpp"
#include "../../native/models/qwen21/metal/vae_norm_silu.hpp"
#include <iostream>

using namespace tc;
int main() {try {
    configure_streams();int cases=0,rejected=0,lazy_metadata_changes=0;
    double worst=0;
    for(int c:{1,7,32,64,96,144,192,288,384,512,576,768,1152})for(bool channels_last:{false,true})
        for(auto gamma_type:{mx::bfloat16,mx::float32})for(bool activation:{false,true})for(bool ready:{false,true}) {
            const int h=3,w=5;
            auto shape=channels_last?mx::Shape{1,h+2,w+2,c+2}:mx::Shape{1,c+2,h+2,w+2};
            auto storage=mx::reshape(mx::sin(mx::arange((h+2)*(w+2)*(c+2),mx::float32)*.071f)*.3f,shape);
            auto view=channels_last?mx::slice(storage,{0,1,1,1},{1,h+1,w+1,c+1}):
                                    mx::slice(storage,{0,1,1,1},{1,c+1,h+1,w+1});
            auto x=channels_last?mx::transpose(view,{0,3,1,2}):view;
            auto gs=mx::astype(mx::sin(mx::arange(c+2,mx::float32)*.13f)*1.7f,gamma_type);
            auto gamma=mx::slice(gs,{1},{c+1});
            if(channels_last)gamma=mx::reshape(gamma,{c,1,1,1});
            if(ready) {
                mx::eval(x,gamma);
                require(x.offset()>0 && (channels_last?x.strides()[1]==1:x.strides()[3]==1),
                        "offset/physical-stride fixture was compacted away");
            }
            const auto initial_strides=x.strides();
            auto reference=(x/mx::maximum(mx::sqrt(mx::sum(x*x,1,true)),Tensor(1e-12f)))*
                std::sqrt(float(c))*mx::reshape(gamma,{1,c,1,1});
            if(activation)reference=silu(reference);
            auto actual=qwen21::metal::vae_norm_silu(x,gamma,activation);mx::eval(reference,actual);
            if(!ready && initial_strides!=x.strides())++lazy_metadata_changes;
            require(actual.dtype()==mx::float32 && actual.shape()==x.shape() &&
                actual.strides()[1]==1 && actual.strides()[3]==c,"fused norm NHWC backing contract failed");
            auto af=mx::contiguous(actual),rf=mx::contiguous(reference),xf=mx::contiguous(x),gf=mx::contiguous(mx::astype(gamma,mx::float32));
            mx::eval(af,rf,xf,gf);
            double error=0,energy=0,cpu_error=0,cpu_energy=0;
            for(int y=0;y<h;++y)for(int z=0;z<w;++z) {
                double sum=0;
                for(int ch=0;ch<c;++ch) {const auto v=xf.data<float>()[size_t(ch)*h*w+y*w+z];sum+=double(v)*v;}
                const double denominator=std::max(std::sqrt(sum),1e-12);
                for(int ch=0;ch<c;++ch) {
                    const size_t i=size_t(ch)*h*w+y*w+z;
                    auto source=xf.data<float>()[i];
                    double cpu=source/denominator*std::sqrt(double(c))*gf.data<float>()[ch];
                    if(activation)cpu=cpu/(1+std::exp(-cpu));
                    const double a=af.data<float>()[i],r=rf.data<float>()[i];
                    require(std::isfinite(a),"fused norm returned nonfinite output");
                    error+=(a-r)*(a-r);energy+=r*r;cpu_error+=(a-cpu)*(a-cpu);cpu_energy+=cpu*cpu;
                }
            }
            const double rel=std::sqrt(error/std::max(energy,1e-30)),cpu_rel=std::sqrt(cpu_error/std::max(cpu_energy,1e-30));
            if(rel>=2e-6 || cpu_rel>=2e-6)std::cerr<<"C="<<c<<" channels_last="<<channels_last
                <<" gamma="<<gamma_type<<" activation="<<activation<<" rel="<<rel<<" cpu_rel="<<cpu_rel
                <<" actual0="<<af.data<float>()[0]<<" reference0="<<rf.data<float>()[0]<<" offset="<<x.offset()<<'\n';
            require(rel<2e-6 && cpu_rel<2e-6,"fused norm exceeded FP32 full-output/oracle gate");
            worst=std::max(worst,std::max(rel,cpu_rel));++cases;
        }
    for(float scale:{0.f,1e-15f,1e-10f})for(bool activation:{false,true}) {
        auto x=mx::full({1,32,2,3},scale,mx::float32),g=mx::ones({32},mx::bfloat16);
        auto out=qwen21::metal::vae_norm_silu(x,g,activation);mx::eval(out);
        const double denom=std::max(std::sqrt(32.*scale*scale),1e-12);
        double expected=scale/denom*std::sqrt(32.);if(activation)expected/=1+std::exp(-expected);
        require(mx::max(mx::abs(out-Tensor(float(expected)))).item<float>()<1e-6,"fused norm zero/tiny epsilon contract failed");++cases;
    }
    auto x=mx::ones({1,32,2,3},mx::float32),g=mx::ones({32},mx::bfloat16);
    for(int i=0;i<5;++i) {
        auto a=x,b=g;
        if(i==0)a=mx::astype(a,mx::float16);if(i==1)a=mx::reshape(a,{1,32,6});
        if(i==2)b=mx::ones({33},mx::float32);if(i==3)b=mx::astype(b,mx::int32);
        if(i==4)a=mx::ones({2,32,2,3},mx::float32);
        bool denied=false;try{(void)qwen21::metal::vae_norm_silu(a,b,true);}catch(const std::exception&){denied=true;}
        require(denied,"fused norm malformed contract accepted");++rejected;
    }
    require(lazy_metadata_changes>0,"lazy stride regression never exercised changed execution metadata");
    std::cout<<"{\"cases\":"<<cases<<",\"rejected\":"<<rejected<<",\"lazy_metadata_changes\":"<<lazy_metadata_changes
             <<",\"worst_relative_l2\":"<<worst<<"}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
