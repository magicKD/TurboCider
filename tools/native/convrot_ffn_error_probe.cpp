// Frozen real-layer replay: separate source rotation, channel rounding, A8,
// normalized FP16 arithmetic and actual Private execution. NOT a benchmark.
#include "backends/mlx.hpp"
#include "backends/convrot_rotation.hpp"
#include "backends/affine_gpu_fp32.hpp"
#include "backends/private/ane_w8_executor.hpp"
#include "backends/ane_w8a8_math.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

using namespace tc;
namespace {
constexpr int hidden=3840,width=10240,share=4096,first=width-share,bucket=1056;
Tensor rounded(const Tensor &a,mx::Dtype dtype) {
    auto value=mx::astype(a,dtype);mx::eval(value);return value;
}
Tensor h256(const Tensor &x) {
    static const Tensor h=[] {
        std::vector<float> data(256*256);
        for(int r=0;r<256;++r)for(int c=0;c<256;++c)data[r*256+c]=ane::comfy_h256_sign(r,c)/16.f;
        return Tensor(data.data(),{256,256},mx::float32);
    }();
    auto shape=x.shape();shape.back()/=256;shape.push_back(256);
    return mx::reshape(mx::matmul(mx::reshape(x,shape),mx::astype(h,x.dtype())),x.shape());
}
struct Quantized {Tensor restored,codes,normalized;};
Quantized quantized(const Tensor &x,int group) {
    const int k=x.shape(-1),m=int(x.size()/k);
    require(group>0 && k%group==0,"A8 diagnostic group geometry mismatch");
    auto a=mx::reshape(mx::astype(x,mx::float32),{m,k/group,group});
    auto peak=mx::max(mx::abs(a),-1,true);
    auto normalized=rounded(mx::where(peak==Tensor(0.f),Tensor(128.f),
        mx::maximum((peak/Tensor(127.f))*Tensor(128.f),Tensor(0x1p-24f))),mx::float16);
    require(mx::all(mx::isfinite(normalized)).item<bool>(),"A8 normalized scale overflow");
    auto q=mx::round(mx::clip((a/mx::astype(normalized,mx::float32))*Tensor(128.f),Tensor(-127.f),Tensor(127.f)));
    auto restore=mx::astype(q*mx::astype(normalized,mx::float32)/Tensor(128.f),x.dtype());
    return {mx::reshape(restore,x.shape()),mx::reshape(mx::astype(q,mx::int8),x.shape()),normalized};
}
Tensor packed_project(const Tensor &rotated,const Weights &w,const std::string &prefix,
                      int rb,int re,int cb,int ce) {
    auto q=slice_axis(slice_axis(w.at(prefix+".weight"),0,rb,re),1,cb/4,ce/4);
    auto s=slice_axis(slice_axis(w.at(prefix+".scales"),0,rb,re),1,cb/32,ce/32);
    auto b=slice_axis(slice_axis(w.at(prefix+".biases"),0,rb,re),1,cb/32,ce/32);
    // Widen the ACTUAL stored BF16 metadata exactly, not the original F32
    // checkpoint scale. Source-vs-F32-full control below qualifies this step.
    if(rotated.dtype()==mx::float32) {s=mx::astype(s,mx::float32);b=mx::astype(b,mx::float32);}
    return mx::astype(mx::quantized_matmul(rotated,q,s,b,true,32,8,"affine"),rotated.dtype());
}
Tensor merge(const Tensor &a,const Tensor &b) {
    return mx::astype(mx::astype(a,mx::float32)+mx::astype(b,mx::float32),mx::bfloat16);
}
void metric(const Tensor &candidate,const Tensor &reference) {
    require(candidate.shape()==reference.shape(),"diagnostic comparison shape mismatch");
    auto a=mx::contiguous(mx::astype(candidate,mx::float32)),b=mx::contiguous(mx::astype(reference,mx::float32));
    mx::eval({a,b});
    double diff=0,an=0,bn=0,dot=0,maximum=0;
    for(size_t i=0;i<a.size();++i) {
        const double x=a.data<float>()[i],y=b.data<float>()[i];
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite diagnostic tensor");
        diff+=(x-y)*(x-y);an+=x*x;bn+=y*y;dot+=x*y;maximum=std::max(maximum,std::abs(x-y));
    }
    std::cout<<"{\"rel_l2\":"<<std::sqrt(diff/std::max(bn,1e-24))
        <<",\"cosine\":"<<(an+bn?dot/std::max(std::sqrt(an*bn),1e-24):1)
        <<",\"max_abs\":"<<maximum<<"}";
}
ane::DeviceMatrixView device_view(const Tensor &a) {
    require(a.flags().row_contiguous && a.buffer().ptr() && a.offset()>=0,"unready diagnostic device view");
    const auto dtype=a.dtype()==mx::bfloat16?ane::DType::BF16:a.dtype()==mx::float16?ane::DType::FP16:ane::DType::FP32;
    require(a.dtype()==mx::bfloat16 || a.dtype()==mx::float16 || a.dtype()==mx::float32,"diagnostic device dtype unsupported");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),int(a.size()/a.shape(-1)),a.shape(-1),
        size_t(a.shape(-1))*a.itemsize(),dtype,std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
ane::DeviceWeightView weight_view(const Weights &w,const std::string &prefix) {
    const auto &q=w.at(prefix+".weight");
    auto s=mx::contiguous(w.at(prefix+".scales")),b=mx::contiguous(w.at(prefix+".biases"));mx::eval({q,s,b});
    return {const_cast<void*>(q.buffer().ptr()),q.buffer_size(),size_t(q.offset()),size_t(q.shape(1))*4,q.shape(0),q.shape(1)*4,
        ane::DeviceWeightEncoding::ConvrotQ8Packed,ane::DType::BF16,32,device_view(s),device_view(b),std::make_shared<Tensor>(q)};
}
Tensor normalized_weights(const Weights &w,const std::string &prefix,int rb,int re,int cb,int ce) {
    auto bytes=mx::view(w.at(prefix+".weight"),mx::uint8);
    auto codes=mx::astype(slice_axis(slice_axis(bytes,0,rb,re),1,cb,ce),mx::int16)-Tensor(128,mx::int16);
    return rounded(mx::astype(codes,mx::float16)*Tensor(1.f/128,mx::float16),mx::float16);
}
Tensor normalized_rows(const Weights &w,const std::string &prefix,int rb,int re) {
    auto s=slice_axis(slice_axis(w.at(prefix+".scales"),0,rb,re),1,0,1);
    return mx::reshape(rounded(mx::astype(s,mx::float32)*Tensor(128.f),mx::float16),{1,1,re-rb});
}
Tensor normalized_projection(const Tensor &x,const Tensor &w) {
    std::optional<Tensor> total;
    for(int begin=0;begin<x.shape(-1);begin+=1024) {
        const int end=std::min(begin+1024,x.shape(-1));
        auto p=rounded(mx::matmul(slice_axis(x,-1,begin,end),mx::transpose(slice_axis(w,1,begin,end))),mx::float16);
        total=total?rounded(*total+p,mx::float16):p;
    }
    return *total;
}
struct Emulation {Tensor output,hidden,output_fp32;};
Emulation emulate(const Quantized &a,const Tensor &gn,const Tensor &un,const Tensor &dn,
                  const Tensor &sg,const Tensor &su,const Tensor &sd) {
    auto x=rounded(mx::astype(a.codes,mx::float16)*Tensor(1.f/128,mx::float16),mx::float16);
    auto tx=mx::reshape(a.normalized,{1,x.shape(1),1});
    auto gate=rounded(rounded(normalized_projection(x,gn)*sg,mx::float16)*tx,mx::float16);
    auto up=rounded(rounded(normalized_projection(x,un)*su,mx::float16)*tx,mx::float16);
    auto neg=rounded(gate*Tensor(-1.f,mx::float16),mx::float16);
    auto exponential=rounded(mx::exp(neg),mx::float16);
    auto denom=rounded(exponential+Tensor(1.f,mx::float16),mx::float16);
    auto h=rounded(rounded(gate/denom,mx::float16)*up,mx::float16);
    auto hr=rounded(h256(h),mx::float16);
    auto floor=rounded(mx::maximum(mx::max(mx::abs(hr),-1,true),Tensor(0x1p-12f,mx::float16)),mx::float16);
    auto ratio=rounded(hr/floor,mx::float16);
    auto a8=rounded(ratio*Tensor(127.f,mx::float16),mx::float16);
    auto hq=rounded(mx::astype(mx::round(a8),mx::int8),mx::int8);
    auto hs=rounded(floor*Tensor(float(_Float16(128.f/127)),mx::float16),mx::float16);
    auto y=normalized_projection(rounded(mx::astype(hq,mx::float16)*Tensor(1.f/128,mx::float16),mx::float16),dn);
    auto restored=(mx::astype(y,mx::float32)*mx::astype(sd,mx::float32))*mx::astype(hs,mx::float32);
    return {rounded(restored,mx::bfloat16),rounded(h,mx::bfloat16),restored};
}
}
int main(int argc,char **argv) {
    if(argc!=5) {std::cerr<<"usage: checkpoint captures cache comma-separated-blocks\n";return 2;}
    try {
        configure_streams();
        auto all=mx::load_safetensors(argv[1]).first;
        std::vector<int> blocks;
        std::string selected(argv[4]);size_t at=0;
        while(at<selected.size()) {size_t end=selected.find(',',at);if(end==std::string::npos)end=selected.size();
            blocks.push_back(std::stoi(selected.substr(at,end-at)));at=end+1;}
        setenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD","0",1);
        ane::PrivateW8Graph base({ane::Kind::SwiGLU,bucket,hidden,share,1024,512,false},1u<<30,
            std::filesystem::path(argv[3])/"base",ane::W8Basis::ComfyH256);
        ane::PrivateW8Graph diagnostic({ane::Kind::SwiGLU,bucket,hidden,share,1024,512,true},1u<<30,
            std::filesystem::path(argv[3])/"hidden",ane::W8Basis::ComfyH256);
        std::string error;require(base.self_test(error),error);require(diagnostic.self_test(error),error);
        bool comma=false;
        std::cout<<std::setprecision(17)<<"{\"schema\":\"tc-convrot-ffn-error-replay-v1\",\"scope\":\"frozen real FFN inputs including physical padding; not model/performance qualification\",\"cases\":[";
        for(int block:blocks) {
            require(block>=0&&block<32,"invalid capture block");
            const auto stem=block<2?"noise_refiner."+std::to_string(block):"layers."+std::to_string(block-2);
            const auto ffn=stem+".feed_forward";
            std::vector<std::string> keys;std::vector<Tensor> arrays;
            for(const auto &suffix:{".w1",".w3",".w2"})for(const auto &part:{".weight",".weight_scale",".comfy_quant"}) {
                auto key=ffn+suffix+part;keys.push_back(key);arrays.push_back(all.at(key));
            }
            Weights w;w.bind_arrays(keys,arrays);w.pack_convrot_q8(32,mx::bfloat16);
            std::vector<ane::DeviceWeightRegion> regions{{weight_view(w,ffn+".w1"),{first,share,0,hidden,256,0,false,ane::W8Basis::ComfyH256}},
                {weight_view(w,ffn+".w3"),{first,share,0,hidden,256,0,false,ane::W8Basis::ComfyH256}},
                {weight_view(w,ffn+".w2"),{0,hidden,first,share,256,0,false,ane::W8Basis::ComfyH256}}};
            base.stage_device_weight_regions(regions);require(base.wait_stage().ok,"base weight staging failed");
            diagnostic.stage_device_weight_regions(regions);require(diagnostic.wait_stage().ok,"hidden weight staging failed");
            auto gn=normalized_weights(w,ffn+".w1",first,width,0,hidden),un=normalized_weights(w,ffn+".w3",first,width,0,hidden),
                dn=normalized_weights(w,ffn+".w2",0,hidden,first,width);
            auto sg=normalized_rows(w,ffn+".w1",first,width),su=normalized_rows(w,ffn+".w3",first,width),sd=normalized_rows(w,ffn+".w2",0,hidden);
            std::vector<std::filesystem::path> samples;
            for(const auto &entry:std::filesystem::directory_iterator(std::filesystem::path(argv[2])/("block"+std::to_string(block))))
                if(entry.path().extension()==".npy")samples.push_back(entry.path());
            std::sort(samples.begin(),samples.end());require(samples.size()==4,"replay requires exactly four captured steps per block");
            for(size_t step=0;step<samples.size();++step) {
                auto original=mx::load(samples[step].string());
                require(original.ndim()==2 && original.shape(1)==hidden && original.dtype()==mx::float16,"capture geometry/dtype mismatch");
                auto x=mx::reshape(mx::astype(original,mx::bfloat16),{1,original.shape(0),hidden});
                require(mx::all(mx::astype(mx::reshape(x,original.shape()),mx::float16)==original).item<bool>(),"capture not losslessly representable in BF16");
                const int rows=x.shape(1),padded=(rows+bucket-1)/bucket*bucket;
                auto g=w.project(x,ffn+".w1"),u=w.project(x,ffn+".w3"),h=silu(g)*u;
                auto source=w.project(h,ffn+".w2");mx::eval({g,u,h,source});
                auto gh=w.project_range(x,ffn+".w1",0,first,0,hidden),gt=w.project_range(x,ffn+".w1",first,width,0,hidden);
                auto uh=w.project_range(x,ffn+".w3",0,first,0,hidden),ut=w.project_range(x,ffn+".w3",first,width,0,hidden);
                auto hh=silu(gh)*uh,ht=silu(gt)*ut;
                auto head=w.project_range(hh,ffn+".w2",0,hidden,0,first);
                auto head32=packed_project(mx::astype(h256(hh),mx::float32),w,ffn+".w2",0,hidden,0,first);
                auto decoded_partial=[&](const Tensor &rotated,int cb,int ce) {
                    return affine_gpu::projection_fp32(rotated,w.at(ffn+".w2.weight"),w.at(ffn+".w2.scales"),
                        w.at(ffn+".w2.biases"),8,0,hidden,cb,ce);
                };
                auto head_decoded=decoded_partial(h256(hh),0,first);
                std::map<std::string,Tensor> full;
                full.emplace("gpu_split_no_a8",merge(head,w.project_range(ht,ffn+".w2",0,hidden,first,width)));
                full.emplace("gpu_full_down_fp32_control",mx::astype(packed_project(mx::astype(h256(h),mx::float32),w,ffn+".w2",0,hidden,0,width),mx::bfloat16));
                full.emplace("gpu_split_fp32_join_no_a8",merge(head32,packed_project(mx::astype(h256(ht),mx::float32),w,ffn+".w2",0,hidden,first,width)));
                full.emplace("gpu_full_decoded_t_fp32_control",mx::astype(decoded_partial(h256(h),0,width),mx::bfloat16));
                full.emplace("gpu_split_decoded_t_fp32_join_no_a8",merge(head_decoded,decoded_partial(h256(ht),first,width)));
                const auto xr=h256(x),xm=convrot_kernel::rotate(x,convrot_kernel::Rotation::Shared);
                auto gm=packed_project(xm,w,ffn+".w1",first,width,0,hidden),um=packed_project(xm,w,ffn+".w3",first,width,0,hidden);
                auto hm=silu(gm)*um;
                full.emplace("gpu_radix4_no_a8",merge(head,packed_project(convrot_kernel::rotate(hm,convrot_kernel::Rotation::Shared),w,ffn+".w2",0,hidden,first,width)));
                for(int group:{hidden,256}) {
                    auto qx=quantized(xr,group);
                    auto ga=packed_project(qx.restored,w,ffn+".w1",first,width,0,hidden),ua=packed_project(qx.restored,w,ffn+".w3",first,width,0,hidden);
                    auto ha=silu(ga)*ua;
                    const auto tag=group==hidden?"row":"group256";
                    full.emplace(std::string("gpu_input_a8_")+tag,merge(head,w.project_range(ha,ffn+".w2",0,hidden,first,width)));
                    const int hg=group==hidden?share:256;
                    auto qh=quantized(h256(ht),hg),qb=quantized(h256(ha),hg);
                    full.emplace(std::string("gpu_hidden_a8_")+tag,merge(head,packed_project(qh.restored,w,ffn+".w2",0,hidden,first,width)));
                    full.emplace(std::string("gpu_both_a8_")+tag,merge(head,packed_project(qb.restored,w,ffn+".w2",0,hidden,first,width)));
                    full.emplace(std::string("gpu_both_a8_fp32_join_")+tag,merge(head32,packed_project(mx::astype(qb.restored,mx::float32),w,ffn+".w2",0,hidden,first,width)));
                    full.emplace(std::string("gpu_both_a8_decoded_t_fp32_join_")+tag,merge(head_decoded,decoded_partial(qb.restored,first,width)));
                }
                auto emulated=emulate(quantized(xm,hidden),gn,un,dn,sg,su,sd);
                full.emplace("gpu_normalized_fp16_emulation",merge(head,emulated.output));
                full.emplace("gpu_normalized_fp16_fp32_join_emulation",merge(head32,emulated.output_fp32));
                full.emplace("gpu_normalized_fp16_decoded_t_fp32_join_emulation",merge(head_decoded,emulated.output_fp32));
                auto packed=mx::contiguous(padded==rows?x:mx::concatenate({x,mx::zeros({1,padded-rows,hidden},mx::bfloat16)},1));
                auto out=mx::zeros({1,padded,hidden},mx::bfloat16),hout=mx::zeros({1,padded,share},mx::bfloat16),
                    zero=mx::zeros({1,padded,share},mx::float32);mx::eval({packed,out,hout,zero});
                base.launch_device(device_view(packed),device_view(out));auto br=base.finish();require(br.ok,br.error);
                auto actual=mx::copy(slice_axis(out,1,0,rows));mx::eval(actual);
                full.emplace("private_base",merge(head,actual));
                auto out32=mx::zeros({1,padded,hidden},mx::float32);mx::eval(out32);
                base.launch_device(device_view(packed),device_view(out32));const auto f32_run=base.finish();require(f32_run.ok,f32_run.error);
                auto actual32=mx::copy(slice_axis(out32,1,0,rows));mx::eval(actual32);
                full.emplace("private_base_fp32_decoded_t_join",merge(head_decoded,actual32));
                diagnostic.launch_device(device_view(packed),device_view(out),ane::DeviceAdapterInput{device_view(zero),device_view(zero),device_view(hout)});
                auto dr=diagnostic.finish();require(dr.ok,dr.error);
                full.emplace("private_zero_correction_graph",merge(head,slice_axis(out,1,0,rows)));
                if(comma)std::cout<<',';comma=true;
                std::cout<<"{\"block\":"<<block<<",\"step\":"<<step+1<<",\"rows\":"<<rows<<",\"padded\":"<<padded
                    <<",\"base_calls\":"<<br.calls<<",\"hidden_calls\":"<<dr.calls<<",\"output_errors\":{";
                bool next=false;for(const auto &[name,value]:full) {if(next)std::cout<<',';next=true;std::cout<<'"'<<name<<"\":";metric(value,source);}
                std::cout<<"},\"split_gate_error\":";metric(mx::concatenate({gh,gt},-1),g);
                std::cout<<",\"split_up_error\":";metric(mx::concatenate({uh,ut},-1),u);
                std::cout<<",\"split_hidden_error\":";metric(mx::concatenate({hh,ht},-1),h);
                std::cout<<",\"private_hidden_source_error\":";metric(slice_axis(hout,1,0,rows),ht);
                std::cout<<",\"emulated_hidden_source_error\":";metric(emulated.hidden,ht);
                std::cout<<",\"private_hidden_emulation_error\":";metric(slice_axis(hout,1,0,rows),emulated.hidden);
                std::cout<<",\"private_down_emulation_error\":";metric(actual,emulated.output);
                std::cout<<",\"private_f32_final_bf16_control\":";metric(mx::astype(actual32,mx::bfloat16),actual);
                std::cout<<",\"headroom\":"<<br.headroom_scale<<",\"scope\":\"frozen source input, includes physical padding, no timing acceptance\"}"<<std::flush;
            }
            mx::clear_cache();
        }
        std::cout<<"]}\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
