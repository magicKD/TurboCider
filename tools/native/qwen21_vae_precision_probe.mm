// Same original Qwen VAE weights/latents; constructor + complete RGBA decode.
// FP16 convolution and/or fused FP32 norm, never an encoder/state dtype swap.
#include "../../native/models/qwen21/vae.hpp"
#include "../../native/media/image.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>
#include <regex>

using namespace tc;
namespace {
double seconds(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
NSString *ns(const std::string &s) {return [NSString stringWithUTF8String:s.c_str()];}
std::string legacy_key(std::string result) {
    if(result.starts_with("conv1."))result.replace(0,5,"quant_conv");
    else if(result.starts_with("conv2."))result.replace(0,5,"post_quant_conv");
    // Literal old implementation, intentionally recompiles each pattern.
    const std::pair<const char *,const char *> rules[] = {
        {"^(encoder|decoder)\\.conv1\\.","$1.conv_in."}, {"^(encoder|decoder)\\.head\\.0\\.","$1.norm_out."},
        {"^(encoder|decoder)\\.head\\.2\\.","$1.conv_out."}, {"\\.middle\\.0\\.",".mid_block.resnets.0."},
        {"\\.middle\\.1\\.",".mid_block.attentions.0."}, {"\\.middle\\.2\\.",".mid_block.resnets.1."},
        {"decoder\\.upsamples\\.([0-4])\\.upsamples\\.3\\.","decoder.up_blocks.$1.upsampler."},
        {"decoder\\.upsamples\\.([0-4])\\.upsamples\\.([0-2])\\.","decoder.up_blocks.$1.resnets.$2."},
        {"encoder\\.downsamples\\.([0-4])\\.downsamples\\.2\\.","encoder.down_blocks.$1.downsampler."},
        {"encoder\\.downsamples\\.([0-4])\\.downsamples\\.([0-1])\\.","encoder.down_blocks.$1.resnets.$2."},
        {"\\.residual\\.0\\.",".norm1."}, {"\\.residual\\.2\\.",".conv1."},
        {"\\.residual\\.3\\.",".norm2."}, {"\\.residual\\.6\\.",".conv2."}, {"\\.shortcut\\.",".conv_shortcut."}};
    for(const auto &[pattern,replacement]:rules)result=std::regex_replace(result,std::regex(pattern),replacement);
    return result;
}
size_t key_contract() {
    const std::pair<const char *,const char *> examples[] = {
        {"conv1.weight","quant_conv.weight"},{"conv2.bias","post_quant_conv.bias"},
        {"encoder.conv1.weight","encoder.conv_in.weight"}, {"decoder.head.2.bias","decoder.conv_out.bias"},
        {"decoder.upsamples.3.upsamples.2.residual.2.weight","decoder.up_blocks.3.resnets.2.conv1.weight"},
        {"encoder.downsamples.4.downsamples.2.resample.1.weight","encoder.down_blocks.4.downsampler.resample.1.weight"}};
    size_t cases=0;
    for(const auto &[input,expected]:examples) {
        require(qwen21::VAE::canonical_key(input)==expected,"canonical known mapping differs");++cases;
    }
    for(const char *stage:{"encoder","decoder","other"})for(int b=0;b<7;++b)for(int layer=0;layer<8;++layer)
        for(const char *tail:{"weight","bias","gamma"}) {
            const auto key=std::string(stage)+".upsamples."+std::to_string(b)+".upsamples."+std::to_string(layer)+
                ".residual."+std::to_string(layer)+"."+tail;
            require(qwen21::VAE::canonical_key(key)==legacy_key(key),"cached regex changed legacy mapping");++cases;
        }
    return cases;
}
std::string hash(const Tensor &a) {
    auto bytes=mx::view(mx::contiguous(a),mx::uint8);mx::eval(bytes);
    return memory_sha256_hex({reinterpret_cast<const char *>(bytes.data<uint8_t>()),bytes.nbytes()});
}
NSDictionary *difference(const Tensor &reference,const Tensor &candidate) {
    require(reference.shape()==candidate.shape(),"VAE comparison shape differs");
    auto a=mx::contiguous(mx::astype(reference,mx::float32)),b=mx::contiguous(mx::astype(candidate,mx::float32));mx::eval(a,b);
    double error=0,energy=0,maximum=0;size_t changed=0;
    for(size_t i=0;i<a.size();++i) {
        const double x=a.data<float>()[i],y=b.data<float>()[i],d=x-y;
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite VAE output");
        error+=d*d;energy+=x*x;maximum=std::max(maximum,std::abs(d));changed+=x!=y;
    }
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-30))),@"max_abs":@(maximum),@"changed":@(changed)};
}
} // namespace

int main(int argc,const char **argv) {@autoreleasepool {
    if(argc==2 && std::string_view(argv[1])=="--key-contract") {
        try {std::cout<<"key_contract_cases="<<key_contract()<<" device_calls=0\n";return 0;}
        catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc!=5 && argc!=6){std::cerr<<"usage: probe WEIGHTS LATENTS_OR_IMAGE_INPUT FRESH_OUTPUT REFERENCE_OR_DASH [fp16|fused|combined]\n";return 2;}
    const auto out=std::filesystem::absolute(argv[3]);bool created=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-vae-precision-v1",@"status":@"starting",@"qualification_passed":@NO,
        @"scope":@"same actual latent and original VAE; constructor+full RGBA decode+finite; not full request; checkpoint load/readback/export outside timer"} mutableCopy];
    try {
        require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned VAE output required");created=true;
        require(std::filesystem::space(out).available>(1ull<<30),"VAE probe requires 1GiB free reserve");
        const std::string recipe=argc==6?argv[5]:"fp16";
        require(recipe=="fp16" || recipe=="fused" || recipe=="combined","unknown VAE candidate recipe");
        const bool use_half=recipe!="fused",use_fused=recipe!="fp16";
        report[@"candidate_recipe"]=ns(recipe);
        report[@"key_contract_cases"]=@(key_contract());
        configure_streams();const auto checkpoint=std::filesystem::canonical(argv[1]);
        report[@"weights_sha256"]=ns(sha256_file(checkpoint));report[@"input_sha256"]=ns(sha256_file(argv[2]));
        Weights weights;auto load_start=Clock::now();weights.load_file(checkpoint);weights.materialize();
        report[@"weight_load_seconds"]=@(seconds(load_start));
        const auto keys=weights.sorted_keys();
        for(const auto &key:keys)require(qwen21::VAE::canonical_key(key)==legacy_key(key),"actual VAE key mapping differs");
        report[@"actual_keys_checked"]=@(keys.size());
        NSMutableArray *key_samples=[NSMutableArray array];size_t key_sink=0;
        for(int visit=0;visit<4;++visit)for(int j=0;j<2;++j) {
            const bool cached=(visit+j)%2;const auto start=Clock::now();
            for(const auto &key:keys)key_sink+=(cached?qwen21::VAE::canonical_key(key):legacy_key(key)).size();
            [key_samples addObject:@{@"cached":@(cached),@"seconds":@(seconds(start))}];
        }
        report[@"key_mapping_samples"]=key_samples;report[@"key_sink"]=@(key_sink);
        auto [arrays,metadata]=mx::load_safetensors(argv[2]);
        Tensor latents(0.f);
        if(arrays.count("image")) {
            qwen21::VAE encoder(weights);std::atomic<bool> encode_cancelled{false};
            latents=encoder.encode(arrays.at("image"),{},encode_cancelled);mx::eval(latents);
            mx::save_safetensors((out/"encoded_input_latents.safetensors").string(),{{"latents",latents}});
            report[@"input_kind"]=@"original VAE encoding of explicit RGBA image; outside decode timing";
        } else {
            latents=arrays.at(arrays.count("latents")?"latents":"tensor");
            report[@"input_kind"]=@"normalized actual denoised latent";
        }
        if(latents.shape()==mx::Shape{1,1024,64})latents=mx::transpose(mx::reshape(latents,{1,32,32,64}),{0,3,1,2});
        require(latents.shape()==mx::Shape{1,64,32,32},"actual normalized 512px latent required");mx::eval(latents);
        const auto latent_hash=hash(latents);std::atomic<bool> cancelled{false};
        std::optional<Tensor> reference,last_candidate;
        NSMutableArray *samples=[NSMutableArray array];report[@"samples"]=samples;
        for(int i=-1;i<4;++i)for(int visit=0;visit<2;++visit) {
            const bool candidate=(i<0?visit:(i+visit)%2)!=0,half=candidate&&use_half;
            const auto start=Clock::now();qwen21::VAE vae(weights,half?qwen21::VAE::DecodeConvolution::FP16WithFP32State:
                                                                           qwen21::VAE::DecodeConvolution::Original,candidate&&use_fused);
            const double construction=seconds(start);
            std::vector<double> stage_seconds;auto stage_start=Clock::now();
            const Event phases=[&](const std::string &phase,int completed,int total) {
                require((phase=="qwen21_vae_decode" || phase=="qwen21_vae_decode_fp32_fallback") &&
                        total==5 && completed==int(stage_seconds.size()%5)+1,
                        "VAE decode phase boundary differs");
                stage_seconds.push_back(seconds(stage_start));stage_start=Clock::now();
            };
            auto decoded=vae.decode_checked(latents,phases,cancelled);auto y=decoded.pixels;mx::eval(y);
            require(y.dtype()==mx::float32 && y.shape()==mx::Shape{1,4,512,512} && mx::all(mx::isfinite(y)).item<bool>(),
                    "VAE FP32 RGBA state/output contract failed");
            const double wall=seconds(start);
            const double final_head_seconds=seconds(stage_start);
            require(stage_seconds.size()==(decoded.fp16_nonfinite_fallback?10:5),"VAE decode phase evidence incomplete");
            require(!decoded.fp16_nonfinite_fallback || half,"unexpected fallback without optional FP16 convolution");
            if(!reference) {require(!candidate,"original reference must be first");reference=y;}
            auto quality=difference(*reference,y);
            if(!candidate)require([quality[@"changed"] unsignedLongLongValue]==0,"original VAE replay changed");
            else last_candidate=y;
            NSMutableArray *stages=[NSMutableArray array];for(double value:stage_seconds)[stages addObject:@(value)];
            [samples addObject:@{@"iteration":@(i),@"half":@(half),@"candidate":@(candidate),@"constructor_seconds":@(construction),
                @"seconds":@(wall),@"quality":quality,@"existing_stage_boundaries_seconds":stages,
                @"fp16_nonfinite_fallback":@(decoded.fp16_nonfinite_fallback),@"final_head_and_finite_seconds":@(final_head_seconds)}];
        }
        mx::save_safetensors((out/"decoded.safetensors").string(),{{"original",*reference},{"candidate",*last_candidate}});
        save_rgba_png(mx::transpose(*reference,{0,2,3,1}),out/"original.png");
        save_rgba_png(mx::transpose(*last_candidate,{0,2,3,1}),out/"candidate.png");
        if(recipe=="fp16")save_rgba_png(mx::transpose(*last_candidate,{0,2,3,1}),out/"half.png");
        if(std::string_view(argv[4])!="-") {
            auto image=mx::transpose(load_rgba_image_tensor(argv[4])*2.f-1.f,{0,3,1,2});
            qwen21::VAE ordinary(weights),island(weights,use_half?qwen21::VAE::DecodeConvolution::FP16WithFP32State:
                                                               qwen21::VAE::DecodeConvolution::Original,use_fused);
            auto a=ordinary.encode(image,{},cancelled),b=island.encode(image,{},cancelled);mx::eval(a,b);
            report[@"reference_encoder_quality"]=difference(a,b);
            require(hash(a)==hash(b),"decoder policy changed reference encoding");
            report[@"reference_encoder_exact"]=@YES;
        }
        require(hash(latents)==latent_hash && sha256_file(checkpoint)==std::string([report[@"weights_sha256"] UTF8String]),"VAE source or input changed");
        char *identity=tc_runtime_build_identity();require(identity,"runtime identity missing");
        report[@"loaded_runtime_id"]=ns(identity);tc_string_free(identity);
        report[@"status"]=@"complete";
    }catch(const std::exception &error){report[@"status"]=@"failed";report[@"error"]=ns(error.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}
    std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"complete"]?0:1;
}}
