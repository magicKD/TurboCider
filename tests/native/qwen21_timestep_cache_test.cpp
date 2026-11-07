#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/models/qwen21/checkpoint_recipe.hpp"
#include "turbocider/turbocider.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

using namespace tc;
using namespace tc::qwen21;
using Frame=Transformer::PreparedTimestepCondition;
static_assert(!std::is_default_constructible_v<Frame>);
static_assert(std::is_copy_constructible_v<Frame>);
constexpr const char *source_sha="0000000000000000000000000000000000000000000000000000000000000000";
// The digest above labels a synthetic component source namespace only. No
// original model payload is loaded by this fixture; the runner states this.
template<typename Function> void rejects(Function fn,const char *reason) {
    try { fn(); }catch(const std::exception &error) {
        require(std::string(error.what()).find(reason)!=std::string::npos,
            std::string("wrong timestep rejection: ")+error.what());return;
    }
    throw std::runtime_error("invalid prepared timestep accepted");
}
Tensor matrix(int out,int in,int seed) {
    return mx::astype(mx::random::normal({out,in},mx::float32,mx::random::key(seed))/std::sqrt(float(in)),mx::bfloat16);
}
void bind_preserving(Weights &w,const std::vector<std::string> &added,const std::vector<Tensor> &values) {
    // Production bind_arrays replaces the complete bank. Fixture updates
    // must carry every other binding, including the four time matrices.
    auto names=w.sorted_keys();std::vector<Tensor> arrays;
    for(const auto &name:names)arrays.push_back(w.at(name));
    for(size_t i=0;i<added.size();++i) {
        const auto found=std::find(names.begin(),names.end(),added[i]);
        if(found==names.end()) {names.push_back(added[i]);arrays.push_back(values.at(i));}
        else arrays.at(size_t(found-names.begin()))=values.at(i);
    }
    w.bind_arrays(names,arrays);
}
void bind_time(Weights &w,int h) {
    bind_preserving(w,Transformer::timestep_condition_weight_names(),
        {matrix(h,256,171),matrix(h,h,172),matrix(4*h,h,173),matrix(h,h,174)});
    w.materialize();mx::synchronize();
}
bool same_bits(const Tensor &a,const Tensor &b) {
    require(a.shape()==b.shape()&&a.dtype()==b.dtype(),"condition geometry changed");
    auto x=mx::view(mx::contiguous(a),mx::uint8),y=mx::view(mx::contiguous(b),mx::uint8);
    mx::eval(x,y);mx::synchronize();return std::memcmp(x.data<uint8_t>(),y.data<uint8_t>(),x.nbytes())==0;
}
std::string sha(const Tensor &value) {
    auto bytes=mx::view(mx::contiguous(value),mx::uint8);mx::eval(bytes);mx::synchronize();
    require(bytes.nbytes()<=UINT32_MAX,"bounded condition hash overflow");
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data<uint8_t>(),CC_LONG(bytes.nbytes()),hash);
    std::ostringstream out;for(unsigned char v:hash)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v);return out.str();
}
std::array<Tensor,2> original_expressions(const Weights &w,float sigma) {
    // Faithful original private embedding expression, full M=2 throughout.
    // This oracle is separate from prepare_timestep_conditions and retains
    // the original production linear/silu operators and independent calls.
    std::vector<float> values(128);
    for(int i=0;i<128;++i)values[i]=float(std::exp(-std::log(10000.0)*double(i)/128.0));
    auto frequencies=Tensor(values.data(),{128},mx::float32);
    const float times_data[]{sigma,0.f};auto times=Tensor(times_data,{2,1},mx::float32);
    auto angles=(times*1000.f)*frequencies;
    auto input=mx::astype(mx::concatenate({mx::cos(angles),mx::sin(angles)},-1),mx::bfloat16);
    auto temb=linear(silu(linear(input,w,"time_text_embed.timestep_embedder.linear_1")),w,"time_text_embed.timestep_embedder.linear_2");
    auto modulation=linear(silu(temb),w,"modulation.1");
    auto raw_output_scale=linear(silu(temb),w,"norm_out.linear");
    return {modulation,raw_output_scale};
}
bool leaf(const Tensor &x) {return x.is_available()&&!x.has_primitive()&&x.inputs().empty()&&x.siblings().empty();}
std::vector<float> sigmas() {return {turbo_sigma_nodes.begin(),turbo_sigma_nodes.end()-1};}
void bind_tiny(Weights &w) {
    constexpr int h=128;bind_time(w,h);
    std::vector<std::string> names;std::vector<Tensor> arrays;int seed=1000;
    auto mat=[&](const std::string &name,int out,int in) {names.push_back(name+".weight");arrays.push_back(matrix(out,in,seed++));};
    auto vec=[&](const std::string &name,int n,float x) {names.push_back(name+".weight");arrays.push_back(mx::full({n},x,mx::bfloat16));};
    mat("img_in",h,64);vec("txt_in.text_norm",h,0.f);mat("txt_in.in_layer",h,h);mat("txt_in.out_layer",h,h);mat("proj_out",64,h);
    for(const char *name:{"to_q","to_k","to_v","to_out.0"})mat(std::string("transformer_blocks.0.attn.")+name,h,h);
    vec("transformer_blocks.0.attn.norm_q",128,1.f);vec("transformer_blocks.0.attn.norm_k",128,1.f);
    mat("transformer_blocks.0.img_mlp.gate_up",512,h);mat("transformer_blocks.0.img_mlp.out",h,256);
    bind_preserving(w,names,arrays);w.materialize();mx::synchronize();
}
TransformerConfig tiny_config() {TransformerConfig c;c.layers=1;c.heads=1;c.context_dim=128;return c;}
int main() {try {
    char *identity=tc_runtime_build_identity();require(identity,"native runtime identity missing");
    const std::string actual_identity(identity);tc_string_free(identity);
    require(actual_identity==TC_TEST_BOUND_NATIVE_ID,"timestep fixture/native ABI/source mismatch");
    require(std::getenv("MLX_ENABLE_TF32")&&std::string(std::getenv("MLX_ENABLE_TF32"))=="0","TF32 disabled control required");
    mx::set_default_device(mx::Device::gpu);mx::set_cache_limit(0);mx::set_memory_limit(3ull<<30);
    Weights full;bind_time(full,4096);Transformer owner(full),foreign(full),invalid(full);
    auto schedule=sigmas();
    rejects([&]{invalid.prepare_timestep_conditions(schedule,mx::float32,source_sha);},"BF16");
    rejects([&]{invalid.prepare_timestep_conditions(schedule,mx::bfloat16,"not-a-sha");},"SHA256");
    auto bad_schedule=schedule;bad_schedule.pop_back();
    rejects([&]{invalid.prepare_timestep_conditions(bad_schedule,mx::bfloat16,source_sha);},"eight");
    bad_schedule=schedule;bad_schedule[0]=std::nextafter(1.f,0.f);
    rejects([&]{invalid.prepare_timestep_conditions(bad_schedule,mx::bfloat16,source_sha);},"exact authored");
    auto frames=owner.prepare_timestep_conditions(schedule,mx::bfloat16,source_sha);
    uint64_t logical=0,capacity=0;std::vector<std::string> hashes;
    for(size_t step=0;step<frames.size();++step) {
        const auto &f=frames[step];auto expected=original_expressions(full,schedule[step]);
        require(f.source_sha256()==source_sha&&f.sigma_bits()==std::bit_cast<uint32_t>(schedule[step])&&f.dtype()==mx::bfloat16,
            "published condition provenance changed");
        require(f.modulation().shape()==mx::Shape{2,16384}&&f.raw_output_scale().shape()==mx::Shape{2,4096},"full original M2 output rows missing");
        require(leaf(f.modulation())&&leaf(f.raw_output_scale()),"prepared frame kept unfinished graph/siblings");
        require(same_bits(f.modulation(),expected[0])&&same_bits(f.raw_output_scale(),expected[1]),"full original M2 BF16 condition bytes changed");
        logical+=f.logical_bytes();capacity+=f.capacity_bytes();hashes.push_back(sha(f.modulation()));hashes.push_back(sha(f.raw_output_scale()));
    }
    require(frames.size()==8&&logical==655360&&capacity>=logical&&capacity<=8*(65536+16384),"actual shape condition capacity changed");
    auto x=mx::zeros({1,1,64},mx::bfloat16),text=mx::zeros({1,1,4096},mx::bfloat16);
    rejects([&]{foreign.forward(x,text,schedule[0],1,1,true,nullptr,{},&frames[0]);},"owner mismatch");
    rejects([&]{owner.forward(x,text,std::nextafter(1.f,0.f),1,1,true,nullptr,{},&frames[0]);},"sigma bits mismatch");
    rejects([&]{owner.forward(mx::astype(x,mx::float32),mx::astype(text,mx::float32),schedule[0],1,1,true,nullptr,{},&frames[0]);},"dtype");
    {
        const auto key=Transformer::timestep_condition_weight_names()[0];auto original=full.at(key);
        bind_preserving(full,{key},{mx::zeros(original.shape(),mx::bfloat16)});full.materialize();
        rejects([&]{owner.forward(x,text,schedule[0],1,1,true,nullptr,{},&frames[0]);},"source binding changed");
        bind_preserving(full,{key},{original});
        full.erase(key);
        rejects([&]{owner.forward(x,text,schedule[0],1,1,true,nullptr,{},&frames[0]);},"retirement is incomplete");
        bind_preserving(full,{key},{original});
    }
    auto source_owners=std::array{
        std::weak_ptr(full.at(Transformer::timestep_condition_weight_names()[0]).data_shared_ptr()),
        std::weak_ptr(full.at(Transformer::timestep_condition_weight_names()[1]).data_shared_ptr()),
        std::weak_ptr(full.at(Transformer::timestep_condition_weight_names()[2]).data_shared_ptr()),
        std::weak_ptr(full.at(Transformer::timestep_condition_weight_names()[3]).data_shared_ptr())};
    for(const auto &name:Transformer::timestep_condition_weight_names())full.erase(name);
    mx::synchronize();mx::clear_cache();
    for(const auto &weak:source_owners)require(weak.expired(),"condition or compiled function retained original time source owner");
    owner.reset();rejects([&]{owner.prepare_timestep_conditions(schedule,mx::bfloat16,source_sha);},"fresh unprepared");
    for(const auto &f:frames)require(leaf(f.modulation())&&leaf(f.raw_output_scale()),"reset invalidated external frame values");

    // Tiny complete transformer: geometry initialization/change, prefix reuse,
    // reset/re-entry and actual original forward arithmetic after source erase.
    Weights base_weights,cache_weights;bind_tiny(base_weights);bind_tiny(cache_weights);
    Transformer base(base_weights,tiny_config()),cached(cache_weights,tiny_config());
    auto tiny_frames=cached.prepare_timestep_conditions(schedule,mx::bfloat16,source_sha);
    for(const auto &name:Transformer::timestep_condition_weight_names())cache_weights.erase(name);
    mx::synchronize();mx::clear_cache();cached.reset();
    auto latent_a=matrix(4,64,1122);latent_a=mx::reshape(latent_a,{1,4,64});auto latent_b=latent_a;
    auto context=mx::reshape(matrix(3,128,1123),{1,3,128});
    uint32_t full_forwards=0;
    for(size_t step=0;step<schedule.size();++step) {
        auto a=base.forward(latent_a,context,schedule[step],2,2,true);
        auto b=cached.forward(latent_b,context,schedule[step],2,2,true,nullptr,{},&tiny_frames[step]);
        require(same_bits(a,b),"prepared original forward velocity bytes changed");
        const float dt=turbo_sigma_nodes[step+1]-schedule[step];
        latent_a=latent_a+a*Tensor(dt,mx::bfloat16);latent_b=latent_b+b*Tensor(dt,mx::bfloat16);
        require(same_bits(latent_a,latent_b),"prepared original Euler latent bytes changed");++full_forwards;
        require(base.cached_layers()==1&&cached.cached_layers()==1,"prepared condition broke prefix reuse");
    }
    base.reset();cached.reset();
    auto new_context=mx::reshape(matrix(4,128,1188),{1,4,128});
    auto a=base.forward(latent_a,new_context,schedule[0],1,4,true);
    auto b=cached.forward(latent_b,new_context,schedule[0],1,4,true,nullptr,{},&tiny_frames[0]);
    require(same_bits(a,b),"geometry/reset cleared the independent prepared condition");++full_forwards;
    rejects([&]{cached.forward(latent_b,new_context,schedule[0],1,4,true);},"missing weight");
    b=cached.forward(latent_b,new_context,schedule[0],1,4,true,nullptr,{},&tiny_frames[0]);
    a=base.forward(latent_a,new_context,schedule[0],1,4,false);
    require(same_bits(a,b),"failed unprepared re-entry invalidated valid external frames");++full_forwards;
    std::vector<Frame> dangling;
    {Transformer temporary(base_weights,tiny_config());dangling=temporary.prepare_timestep_conditions(schedule,mx::bfloat16,source_sha);}
    rejects([&]{cached.forward(latent_b,new_context,schedule[0],1,4,true,nullptr,{},&dangling[0]);},"owner mismatch");
    std::cout<<"{\"synthetic_source_only\":true,\"full_m2_sigma_cases\":8,\"logical_condition_bytes\":"<<logical
        <<",\"actual_condition_capacity_bytes\":"<<capacity<<",\"original_source_owners_expired\":true,\"tiny_exact_forward_cases\":"<<full_forwards
        <<",\"owner_sigma_dtype_binding_retirement_reentry_guards_passed\":true,\"output_sha256\":[";
    for(size_t i=0;i<hashes.size();++i)std::cout<<(i?",":"")<<'"'<<hashes[i]<<'"';
    std::cout<<"]}\n";return 0;
}catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}}
