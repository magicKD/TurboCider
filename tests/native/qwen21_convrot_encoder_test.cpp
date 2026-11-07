#include "../../native/models/qwen21/encoder_convrot.hpp"
#include "../../native/models/qwen21/text_encoder.hpp"
#include "../../native/models/qwen21/conditioning.hpp"
#include "turbocider/turbocider.h"
#include <cmath>
#include <bit>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace tc;
namespace {
const std::string marker = "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256}";
Tensor tag() { return Tensor(reinterpret_cast<const uint8_t *>(marker.data()), {int(marker.size())}, mx::uint8); }
Tensor cpu_h256_inverse(const Tensor &input,bool bf16=false) {
    // Independent CPU-double regular-H4 Kronecker oracle. A GPU F32 matmul
    // may use TF32 operands on M5 and is NOT a reference for F32 inverse math.
    const int h4[] = {1,1,1,-1, 1,1,-1,1, 1,-1,1,1, -1,1,1,1};
    std::array<int,256*256> h{};
    for(int r=0;r<256;++r)for(int c=0;c<256;++c) {
        int sign=1;for(int shift=0;shift<8;shift+=2)sign*=h4[((r>>shift)&3)*4+((c>>shift)&3)];
        h[size_t(r)*256+c]=sign;
    }
    auto source=mx::astype(input,mx::float32);mx::eval(source);
    require(source.size()%256==0,"invalid CPU H256 oracle geometry");
    std::vector<float> values(source.size());const auto *data=source.data<float>();
    for(size_t base=0;base<values.size();base+=256)for(int column=0;column<256;++column) {
        double sum=0;for(int k=0;k<256;++k)sum+=double(data[base+k])*h[size_t(k)*256+column];
        float value=float(sum/16.0);
        if(bf16) {
            auto bits=std::bit_cast<uint32_t>(value);
            bits=(bits+0x7fffu+((bits>>16)&1u))&0xffff0000u;
            value=std::bit_cast<float>(bits);
        }
        values[base+column]=value;
    }
    return Tensor(values.data(),source.shape(),mx::float32);
}
float relative(const Tensor &a, const Tensor &b) {
    auto x=mx::astype(a,mx::float32),y=mx::astype(b,mx::float32);
    auto result=mx::sqrt(mx::sum(mx::square(x-y))/mx::maximum(mx::sum(mx::square(y)),Tensor(1e-30f)));
    mx::eval(result);return result.item<float>();
}
float max_absolute(const Tensor &a,const Tensor &b) {
    auto error=mx::max(mx::abs(mx::astype(a,mx::float32)-mx::astype(b,mx::float32)));
    mx::eval(error);return error.item<float>();
}
template<class F> void rejected(F body,const char *message) {
    bool denied=false;try { body(); } catch(const std::exception &) { denied=true; }
    require(denied,message);
}
void embeddings() {
    constexpr int rows=10000,width=512;
    auto q=mx::reshape(mx::astype((mx::arange(rows*width,mx::int32)*37+13)%256-128,mx::int8),{rows,width});
    auto scales=mx::reshape(mx::astype(mx::arange(rows,mx::int32)%17+1,mx::float32)*.00012345f,{rows,1});
    Weights raw;raw.bind_arrays({"model.embed_tokens.weight","model.embed_tokens.weight_scale","model.embed_tokens.comfy_quant"},{q,scales,tag()});
    const int values[]={7,2,7,0,rows-1};const Tensor ids(values,{5},mx::int32);
    auto selected=mx::astype(mx::take(q,ids,0),mx::float32)*mx::take(scales,ids,0);
    auto reference=cpu_h256_inverse(selected,true);
    auto inverse_fp32=convrot_kernel::rotate(selected,convrot_kernel::Rotation::SimdRegister);
    auto reference_fp32=cpu_h256_inverse(selected);
    require(relative(inverse_fp32,reference_fp32)<2e-6f&&max_absolute(inverse_fp32,reference_fp32)<1e-6f,
            "pre-BF16 ConvRot inverse differs from independent CPU double oracle");
    auto decoded=qwen21::encoder_embedding_rows(raw,ids,"model.embed_tokens");
    mx::eval(reference,decoded);
    require(decoded.shape()==mx::Shape{5,width}&&decoded.dtype()==mx::bfloat16,"ConvRot token rows lost shape/BF16 dtype");
    require(relative(decoded,reference)<1e-6f&&max_absolute(decoded,reference)<1e-6f,
            "raw ConvRot embedding differs from independent CPU double oracle");
    // Nonzero BF16 values must match bit-for-bit. Different F32 accumulation
    // orders may leave tiny residues in mathematically cancelling channels.
    auto decoded_f32=mx::astype(decoded,mx::float32);mx::eval(decoded_f32,reference);
    for(size_t i=0;i<reference.size();++i)if(std::abs(reference.data<float>()[i])>1e-5f)
        require(std::bit_cast<uint32_t>(decoded_f32.data<float>()[i])==std::bit_cast<uint32_t>(reference.data<float>()[i]),
                "nonzero BF16 embedding differs from the CPU oracle");
    // Exact powers of two exercise forward/inverse signs and group ordering.
    auto source=mx::reshape(mx::astype(mx::arange(1024,mx::int32)%31-15,mx::float32)/16.f,{2,512});
    auto forward=cpu_h256_inverse(source);
    auto inverse=convrot_kernel::rotate(forward,convrot_kernel::Rotation::SimdRegister);
    mx::eval(source,inverse);require(relative(inverse,source)==0.f,"regular H256 inverse changed signs or permutation");
    require(raw.pack_convrot_q8(128,mx::float32)==1,"embedding baseline packing failed");
    auto packed=qwen21::encoder_embedding_rows(raw,ids,"model.embed_tokens");mx::eval(packed);
    require(relative(packed,reference)<1e-6f&&max_absolute(packed,reference)<1e-6f,
            "packed ConvRot token rows lost F32 coefficients or inverse rotation");
    auto binary_scales=mx::reshape(mx::astype(mx::arange(rows,mx::int32)%17+1,mx::float32)/1024.f,{rows,1});
    Weights binary;binary.bind_arrays({"model.embed_tokens.weight","model.embed_tokens.weight_scale","model.embed_tokens.comfy_quant"},{q,binary_scales,tag()});
    auto exact_input=mx::astype(mx::take(q,ids,0),mx::float32)*mx::take(binary_scales,ids,0);
    auto exact_output=qwen21::encoder_embedding_rows(binary,ids,"model.embed_tokens");
    require(max_absolute(exact_output,cpu_h256_inverse(exact_input,true))==0.f,
            "binary-exact ConvRot row scales are not bit-exact after inverse/BF16 cast");
    std::atomic<bool> cancelled{false};qwen21::TextEncoder encoder(raw);
    for(int invalid:{-1,rows}) {
        Tokens tokens{{invalid},1};
        rejected([&]{(void)encoder.encode(tokens,{},cancelled);},"text encoder accepted out-of-vocabulary ConvRot token");
        rejected([&]{(void)qwen21::assemble_prompt(tokens,raw,{});},"prompt assembly accepted out-of-vocabulary ConvRot token");
    }
    Weights dense;dense.bind_arrays({"table.weight"},{mx::astype(selected,mx::bfloat16)});
    const int dense_ids[]={3,1};auto di=Tensor(dense_ids,{2},mx::int32);
    auto ordinary=qwen21::encoder_embedding_rows(dense,di,"table");
    require(relative(ordinary,mx::take(dense.at("table.weight"),di,0))==0.f,"dense embedding route changed");
    std::cout<<"PASS raw/packed ConvRot embedding: selected rows, independent regular-H256 inverse, BF16 output, token bounds\n";
}
void source_selection() {
    auto root=std::filesystem::temp_directory_path()/ ("tc-convrot-encoder-source-"+std::to_string(getpid()));
    std::filesystem::create_directories(root/"text_encoders");
    struct Cleanup { std::filesystem::path path;~Cleanup(){std::filesystem::remove_all(path);} } cleanup{root};
    rejected([&]{(void)qwen21::encoder_checkpoint_source(root);},"missing encoder source accepted");
    for(const auto *name:{"Qwen3-VL-8B-Instruct-Q4_K_M.gguf","qwen3vl_8b_int8_convrot.safetensors","qwen3vl_8b_bf16.safetensors"}) {
        std::ofstream(root/"text_encoders"/name).put('x');
        require(qwen21::encoder_checkpoint_source(root).filename()==name,"encoder source precedence changed");
    }
    require(qwen21::convrot_encoder_checkpoint("qwen3vl_8b_int8_convrot.safetensors")&&
            !qwen21::convrot_encoder_checkpoint("qwen3vl_8b_bf16.safetensors"),"encoder variant classification failed");
}
void metadata() {
    // Lazy zero sources establish the complete published geometry without
    // evaluating or allocating the multi-gigabyte model.
    Weights weights;std::vector<std::string> keys;std::vector<Tensor> values;
    const auto marker_tensor=tag();
    auto matrix=[&](const std::string &name,int n,int k) {
        keys.insert(keys.end(),{name+".weight",name+".weight_scale",name+".comfy_quant"});
        values.insert(values.end(),{mx::zeros({n,k},mx::int8),mx::ones({n,1},mx::float32),marker_tensor});
    };
    matrix("model.embed_tokens",151936,4096);matrix("lm_head",151936,4096);
    for(int i=0;i<36;++i) {
        const auto prefix="model.layers."+std::to_string(i);
        for(const auto *name:{"q_proj","o_proj"})matrix(prefix+".self_attn."+name,4096,4096);
        for(const auto *name:{"k_proj","v_proj"})matrix(prefix+".self_attn."+name,1024,4096);
        for(const auto *name:{"gate_proj","up_proj"})matrix(prefix+".mlp."+name,12288,4096);
        matrix(prefix+".mlp.down_proj",4096,12288);
    }
    weights.bind_arrays(keys,values);
    qwen21::prepare_convrot_encoder(weights,true);
    require(weights.convrot_w8a8()&&!weights.has("lm_head.weight")&&weights.at("model.embed_tokens.weight").dtype()==mx::int8,
            "W8A8 preparation widened/repacked codes or retained unused logits");
    auto malformed=values;std::string bad=marker;bad.replace(bad.find("256"),3,"064");
    malformed[2]=Tensor(reinterpret_cast<const uint8_t *>(bad.data()),{int(bad.size())},mx::uint8);
    Weights wrong;wrong.bind_arrays(keys,malformed);
    rejected([&]{qwen21::prepare_convrot_encoder(wrong,true);},"wrong ConvRot group marker accepted");
    malformed=values;malformed[1]=mx::ones({151936,2},mx::float32);wrong.bind_arrays(keys,malformed);
    rejected([&]{qwen21::prepare_convrot_encoder(wrong,true);},"wrong per-row scale geometry accepted");
}
void compiled_routes() {
    constexpr int h=256,f=512,vocab=32;
    std::vector<std::string> keys;std::vector<Tensor> values;
    auto matrix=[&](const std::string &name,int n,int k) {
        auto q=mx::reshape(mx::astype((mx::arange(n*k,mx::int32)*13+7)%127-63,mx::int8),{n,k});
        keys.insert(keys.end(),{name+".weight",name+".weight_scale",name+".comfy_quant"});
        values.insert(values.end(),{q,mx::full({n,1},.0004f,mx::float32),tag()});
    };
    auto norm=[&](const std::string &name,int size){keys.push_back(name+".weight");values.push_back(mx::ones({size},mx::bfloat16));};
    matrix("model.embed_tokens",vocab,h);
    for(const auto *name:{"q_proj","o_proj"})matrix(std::string("model.layers.0.self_attn.")+name,h,h);
    for(const auto *name:{"k_proj","v_proj"})matrix(std::string("model.layers.0.self_attn.")+name,128,h);
    for(const auto *name:{"gate_proj","up_proj"})matrix(std::string("model.layers.0.mlp.")+name,f,h);
    matrix("model.layers.0.mlp.down_proj",h,f);
    for(const auto *name:{"input_layernorm","post_attention_layernorm"})norm(std::string("model.layers.0.")+name,h);
    for(const auto *name:{"q_norm","k_norm"})norm(std::string("model.layers.0.self_attn.")+name,128);
    Weights weights;weights.bind_arrays(keys,values);weights.set_convrot_w8a8(true);
    qwen21::TextConfig config;config.layers=1;config.heads=2;config.kv_heads=1;config.head_dim=128;
    config.final_norm=false;config.compiled_gpu_blocks=true;
    std::atomic<bool> cancelled{false};Tokens tokens{{1,2,3,4,5,6,7,8},8};
    const auto before=convrot_w8a8_stats();
    qwen21::TextEncoder encoder(weights,config);
    auto result=encoder.encode(tokens,{},cancelled);mx::eval(result);
    require(result.shape()==mx::Shape{1,8,h}&&result.dtype()==mx::bfloat16,"compiled ConvRot encoder shape/dtype changed");
    require(convrot_w8a8_stats().projections>=before.projections+7,"compiled block lost its W8A8 Weights mode");
    const int token_ids[]={1,2,3,4,5,6,7,8};auto ids=Tensor(token_ids,{8},mx::int32);
    auto embedded=mx::expand_dims(qwen21::encoder_embedding_rows(weights,ids,"model.embed_tokens"),0);
    auto positions=mx::broadcast_to(mx::reshape(mx::arange(8,mx::int32),{1,8}),{3,8});
    qwen21::TextFfnHooks hooks;
    hooks.prepare=[](int,const std::string &,int){};
    hooks.execute=[](int,const Tensor &x,const qwen21::TextFfnHooks::Gpu &gpu){return gpu(x);};
    const auto split_before=convrot_w8a8_stats();
    auto split=encoder.encode_embeddings(embedded,positions,8,{},cancelled,{}, {},&hooks);mx::eval(split);
    require(convrot_w8a8_stats().projections>=split_before.projections+7,"compiled split FFN lost its W8A8 Weights mode");
    require(relative(split,result)<1e-5f,"compiled full/split ConvRot encoder disagree");
    std::cout<<"PASS compiled encoder block + split FFN preserve W8A8 mode, 7 projections each\n";
}
} // namespace
int main() { try {
    char *identity=tc_runtime_build_identity();require(identity,"native identity missing");
    const std::string loaded(identity);tc_string_free(identity);
    require(loaded==TC_TEST_BOUND_NATIVE_ID,"ConvRot encoder test requires the matching native ABI");
    configure_streams();embeddings();source_selection();
    if(convrot_w8a8_available()){metadata();compiled_routes();}
    else std::cout<<"SKIP W8A8 execution: Metal TensorOps device unavailable\n";
    std::cout<<"PASS Qwen3-VL ConvRot encoder\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;} }
