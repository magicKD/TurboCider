#include "convrot_factorized.hpp"
#include "../../../native/backends/mlx.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

namespace {
using namespace tc;
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
NSDictionary*quality(const Tensor&a,const Tensor&b){auto rel=mx::sqrt(mx::sum(mx::square(a-b))/mx::maximum(mx::sum(mx::square(a)),Tensor(1e-30f)));
    auto maximum=mx::max(mx::abs(a-b));mx::eval(rel,maximum);return @{@"relative_l2":@(rel.item<float>()),@"max_abs":@(maximum.item<float>()),@"elements":@(a.size())};}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"row-scale probe report write failed");}
}
extern "C" __attribute__((visibility("default")))
int tc_convrot_factorized_probe(const char*model,const char*capture,const char*output,int repeats,int bm,int bn,void(*scope)(int)){@autoreleasepool{
    const bool real=capture&&std::string_view(capture)!="-";const std::filesystem::path out(output?output:"");
    NSMutableDictionary*r=[@{@"schema":@"tc-convrot-row-scale-dot-v1",@"status":@"starting",@"real_weights":@(real),@"complete_request":@NO,@"image_export":@NO,
        @"cpp_sha256":@TC_CONVROT_FACTOR_CPP_SHA,@"header_sha256":@TC_CONVROT_FACTOR_HEADER_SHA,@"compiled_native_id":@TC_CONVROT_FACTOR_NATIVE_ID,
        @"repeats":@(repeats),@"warmups":@2,@"bm":@(bm),@"bn":@(bn),@"bk":@64,@"ane_predictions":@0,@"candidate_owned_dense_weight_bytes":@0}mutableCopy];
    try{require(model&&capture&&output&&(repeats==16||repeats==32)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh bounded row-scale fixture required");
        char*raw=tc_runtime_build_identity();require(raw,"native identity missing");const std::string id(raw);tc_string_free(raw);r[@"loaded_runtime_id"]=ns(id);
        require(id==TC_CONVROT_FACTOR_NATIVE_ID,"row-scale helper/native C++ ABI mismatch");configure_streams();
        int m=33,k=128,n=257,wp=320,rb=5,cb=32;std::vector<Tensor>inputs,qs,ss;
        if(real){m=1024;k=6144;n=4096;wp=12288;rb=cb=0;auto c=mx::load_safetensors(capture);inputs={c.first.at("hidden_original_a"),c.first.at("hidden_original_b")};
            auto w=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_int8_convrot.safetensors");
            for(int layer=0;layer<2;++layer){const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.out";qs.push_back(w.first.at(p+".weight"));ss.push_back(w.first.at(p+".weight_scale"));}}
        else{auto ids=mx::arange(m*k,mx::int32);auto x=mx::reshape(mx::astype(ids%29-14,mx::bfloat16)/Tensor(32.f,mx::bfloat16),{1,m,k});inputs={x,x*Tensor(.75f,mx::bfloat16)+Tensor(.015625f,mx::bfloat16)};
            const int rows=n+rb+7;auto wi=mx::arange((rows+2)*wp,mx::int32);auto q=mx::reshape(mx::astype((wi*37+13)%256-128,mx::int8),{rows+2,wp});qs={slice_axis(q,0,1,rows+1)};
            qs.push_back(mx::astype(((mx::astype(qs[0],mx::int32)+Tensor(128,mx::int32))*7+11)%256-128,mx::int8));
            std::vector<float>s;for(int i=0;i<rows;++i)s.push_back(float(i%17-8)*.00012345f);ss={Tensor(s.data(),{rows,1},mx::float32)};ss.push_back(ss[0]*Tensor(.875f));}
        std::vector<Tensor>words,logical,packed,scales,biases;NSMutableDictionary*source_hashes=[NSMutableDictionary dictionary];
        for(int layer=0;layer<2;++layer){require(qs[layer].dtype()==mx::int8&&qs[layer].ndim()==2&&qs[layer].shape(1)==wp&&ss[layer].dtype()==mx::float32&&ss[layer].shape()==mx::Shape{qs[layer].shape(0),1},"original signed I8 plus exact F32 row scale required");
            mx::eval(qs[layer],ss[layer]);require(mx::all(mx::isfinite(ss[layer])).item<bool>(),"nonfinite source scale");
            source_hashes[ns(std::to_string(layer))]=@{@"codes":ns(hash(qs[layer])),@"row_scale":ns(hash(ss[layer]))};
            words.push_back(mx::view(mx::astype(mx::astype(qs[layer],mx::int32)+Tensor(128,mx::int32),mx::uint8),mx::uint32));
            auto selected=slice_axis(slice_axis(qs[layer],0,rb,rb+n),1,cb,cb+k);auto scale=slice_axis(ss[layer],0,rb,rb+n);
            logical.push_back(mx::astype(selected,mx::float32)*scale);
            packed.push_back(mx::contiguous(slice_axis(slice_axis(words.back(),0,rb,rb+n),1,cb/4,(cb+k)/4)));
            scales.push_back(mx::repeat(scale,k/128,1));biases.push_back(scales.back()*Tensor(-128.f));
            mx::eval({words.back(),logical.back(),packed.back(),scales.back(),biases.back()});}
        for(auto&x:inputs){mx::eval(x);require(x.dtype()==mx::bfloat16&&x.shape()==mx::Shape{1,m,k},"complete ready BF16 dot-domain inputs required");}
        const std::array<std::string,2>input_hash{hash(inputs[0]),hash(inputs[1])};const char*names[]{"raw-f32-ready","q8-f32-ready","factored"};
        auto run=[&](int arm,int variant,int layer){if(arm==0)return mx::matmul(mx::astype(inputs[variant],mx::float32),mx::transpose(logical[layer]));
            if(arm==1)return mx::quantized_matmul(inputs[variant],packed[layer],scales[layer],biases[layer],true,128,8,"affine");
            return convrot_factorized_lab::projection(inputs[variant],words[layer],ss[layer],rb,rb+n,cb,cb+k,bm,bn,64);};
        std::unordered_map<std::string,Tensor>payload{{"input_a",inputs[0]},{"input_b",inputs[1]}};std::array<std::array<std::string,3>,3>expected;
        NSMutableDictionary*checks=[NSMutableDictionary dictionary];const char*tags[]{"a","b","w"};
        for(int c=0;c<3;++c){std::array<Tensor,3>values{Tensor(0.f),Tensor(0.f),Tensor(0.f)};for(int arm=0;arm<3;++arm){if(scope)scope(1+c*3+arm);values[arm]=run(arm,c==1,c==2);mx::eval(values[arm]);
                require(values[arm].dtype()==mx::float32&&mx::all(mx::isfinite(values[arm])).item<bool>(),"full finite F32 projection required");expected[c][arm]=hash(values[arm]);payload.emplace(std::string(names[arm])+"_"+tags[c],values[arm]);}
            checks[@(tags[c])]=@{@"packed":quality(values[0],values[1]),@"factored":quality(values[0],values[2])};}
        require(expected[0][0]!=expected[1][0]&&expected[0][0]!=expected[2][0],"changed ready X/W returned stale output");
        uint64_t rejected=0;for(int bad=0;bad<8;++bad){bool denied=false;try{(void)convrot_factorized_lab::projection(bad==0?mx::astype(inputs[0],mx::float32):inputs[0],
                bad==1?mx::astype(words[0],mx::int32):words[0],bad==2?mx::astype(ss[0],mx::bfloat16):ss[0],bad==3?-1:rb,bad==4?qs[0].shape(0)+1:rb+n,
                bad==5?cb+1:cb,bad==6?cb+k-32:cb+k,bm,bad==7?32:bn,64);}catch(const std::invalid_argument&){denied=true;}require(denied,"invalid original row-scale/range/dtype admitted");++rejected;}
        NSMutableArray*samples=[NSMutableArray array];r[@"samples"]=samples;std::array<std::optional<Tensor>,3>last;
        for(int rep=-2;rep<repeats;++rep)for(int j=0;j<3;++j){const int arm=(j+rep+3)%3,variant=bool(rep&1),layer=(rep+2)%4==2;if(scope)scope(100+(rep+2)*3+j);
            const auto start=Clock::now();auto value=run(arm,variant,layer);require(!value.is_available(),"row-scale dot reused completed output");mx::eval(value);
            const auto wall=std::chrono::duration<double>(Clock::now()-start).count();last[arm]=value;
            [samples addObject:@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(variant)),@"layer":@(layer),@"seconds":@(wall)}];}
        if(scope)scope(-3);for(int arm=0;arm<3;++arm){require(last[arm]&&hash(*last[arm])==expected[1][arm],"last dot output not independently owned/restored");auto restored=run(arm,0,0);mx::eval(restored);require(hash(restored)==expected[0][arm],"row-scale X/W A/B/A restore failed");}
        for(int layer=0;layer<2;++layer)require(hash(qs[layer])==std::string([source_hashes[ns(std::to_string(layer))][@"codes"]UTF8String])&&hash(ss[layer])==std::string([source_hashes[ns(std::to_string(layer))][@"row_scale"]UTF8String]),"original I8/F32 coefficients mutated");
        require(hash(inputs[0])==input_hash[0]&&hash(inputs[1])==input_hash[1],"ready input mutated");mx::save_safetensors((out/"arrays.safetensors").string(),payload);
        r[@"m"]=@(m);r[@"k"]=@(k);r[@"n"]=@(n);r[@"physical_pitch"]=@(wp);r[@"row_begin"]=@(rb);r[@"column_begin"]=@(cb);r[@"checks"]=checks;r[@"source_hashes"]=source_hashes;
        r[@"source_unchanged"]=@YES;r[@"rejected"]=@(rejected);r[@"strong_f32_control_weight_bytes"]=@(2ull*n*k*4);r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
