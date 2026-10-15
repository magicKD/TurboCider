// Optional product client: bounded QKV replay from actual ordinary GPU block
// arguments. Extra graph outputs are NOT claimed to be an internal capture of
// the unchanged compiled kernel. Full-block/prefix controls qualify the replay.
#include "models/qwen21/pipeline.hpp"
#include "turbocider/turbocider.h"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace tc;
namespace {
constexpr uint64_t cap = 128ull * 1024 * 1024;
std::string quote(std::string_view value) {
    std::ostringstream out; out << '"';
    for (unsigned char ch : value) {
        if (ch=='"' || ch=='\\') out << '\\' << char(ch);
        else { require(ch>=32,"control character in capture metadata"); out << char(ch); }
    }
    return out.str()+'"';
}
template<class T> void numbers(std::ostream& out,const T& values) {
    out << '['; bool first=true;
    for (const auto& value:values) { if(!first)out << ','; first=false; out << value; }
    out << ']';
}
std::string hash(const Tensor& value) {
    auto bytes=mx::view(mx::contiguous(value),mx::uint8); mx::eval(bytes);
    require(bytes.size()>0 && bytes.size()<=UINT32_MAX,"bounded logical tensor digest required");
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
    require(CC_SHA256(bytes.data<uint8_t>(),CC_LONG(bytes.size()),digest.data())!=nullptr,"tensor SHA failed");
    static constexpr char hex[]="0123456789abcdef";
    std::string result(digest.size()*2,'0');
    for(size_t i=0;i<digest.size();++i) { result[2*i]=hex[digest[i]>>4]; result[2*i+1]=hex[digest[i]&15]; }
    return result;
}
void finite(const Tensor& value) {
    auto f=mx::contiguous(mx::astype(value,mx::float32));mx::eval(f);
    const float* data=f.data<float>();
    for(size_t i=0;i<f.size();++i) require(std::isfinite(data[i]),"nonfinite captured value");
}
std::array<double,2> difference(const Tensor& value,const Tensor& reference) {
    require(value.shape()==reference.shape() && value.dtype()==mx::bfloat16 && reference.dtype()==mx::bfloat16,
            "original BF16 block output ABI differs");
    auto a=mx::contiguous(mx::astype(value,mx::float32));
    auto b=mx::contiguous(mx::astype(reference,mx::float32));mx::eval(a,b);
    const auto *x=a.data<float>(),*y=b.data<float>();
    double error=0,energy=0,maximum=0;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(x[i]) && std::isfinite(y[i]),"nonfinite block comparison");
        const double d=double(x[i])-y[i];error+=d*d;energy+=double(y[i])*y[i];maximum=std::max(maximum,std::abs(d));
    }
    return {std::sqrt(error/std::max(energy,1e-30)),maximum};
}
Tensor rotate(const Tensor& x,const Tensor& cosine,const Tensor& sine) {
    auto shape=x.shape(),paired_shape=shape;paired_shape.back()/=2;paired_shape.push_back(2);
    auto paired=mx::reshape(mx::astype(x,mx::float32),paired_shape);
    auto a=mx::squeeze(slice_axis(paired,-1,0,1),-1),b=mx::squeeze(slice_axis(paired,-1,1,2),-1);
    auto c=mx::reshape(cosine,{1,1,shape[2],shape[3]/2});
    auto s=mx::reshape(sine,{1,1,shape[2],shape[3]/2});
    return mx::astype(mx::reshape(mx::stack({a*c-b*s,a*s+b*c},-1),shape),x.dtype());
}
std::vector<Tensor> replay(const qwen21::DiagnosticBlockObservation& observed) {
    const auto& w=observed.weights;const auto segments=observed.segments;
    const bool steady=observed.steady;
    auto function=mx::compile([&w,segments,steady](const std::vector<Tensor>& a) {
        const std::string p="transformer_blocks.0.";
        auto input=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
        auto q=linear(input,w,p+"attn.to_q"),k=linear(input,w,p+"attn.to_k");
        auto v=heads(linear(input,w,p+"attn.to_v"),32,128);
        q=rotate(mx::fast::rms_norm(heads(q,32,128),w.at(p+"attn.norm_q.weight"),1e-6f),a[5],a[6]);
        k=rotate(mx::fast::rms_norm(heads(k,32,128),w.at(p+"attn.norm_k.weight"),1e-6f),a[5],a[6]);
        Tensor attention=a[0];
        if(steady) {
            k=mx::concatenate({a[7],k},2);v=mx::concatenate({a[8],v},2);attention=attend(q,k,v);
        } else {
            std::vector<Tensor> parts;
            for(const auto& segment:segments)
                parts.push_back(attend(slice_axis(q,2,segment.start,segment.end),
                    slice_axis(k,2,0,segment.end),slice_axis(v,2,0,segment.end),false,{},false,segment.causal?"causal":""));
            attention=mx::concatenate(parts,1);
        }
        auto hidden=a[0]+mx::tanh(a[2])*linear(attention,w,p+"attn.to_out.0");
        input=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);
        auto gu=linear(input,w,p+"img_mlp.gate_up");const int half=gu.shape(-1)/2;
        auto ff=silu(slice_axis(gu,-1,0,half))*slice_axis(gu,-1,half,gu.shape(-1));
        hidden=hidden+mx::tanh(a[4])*linear(ff,w,p+"img_mlp.out");
        auto head_output=mx::transpose(mx::reshape(attention,{1,attention.shape(1),32,128}),{0,2,1,3});
        return std::vector<Tensor>{hidden,q,k,v,head_output};
    });
    auto result=function(observed.arguments);mx::eval(result);return result;
}
}

int main(int argc,char** argv) {
    try {
        require(argc==4,"usage: capture-qwen-attention MODEL OUTPUT_DIRECTORY SEED");
        const auto root=std::filesystem::canonical(argv[1]);
        const auto output=std::filesystem::canonical(argv[2]);
        require(std::filesystem::is_directory(output) && !std::filesystem::exists(output/"driver.json"),"fresh capture directory required");
        size_t used=0;const auto seed=std::stoull(argv[3],&used);
        require(used==std::string(argv[3]).size() && seed<=UINT32_MAX,"bounded seed required");
        configure_streams();std::atomic<bool> cancelled{false};qwen21::Session session(root);
        std::vector<std::string> records;uint64_t written=0;bool completed=false;
        session.set_diagnostic_block_observer([&](const qwen21::DiagnosticBlockObservation& o) {
            require(o.layer==0 && records.size()<2 && o.steady==(records.size()==1) && o.prefix_tokens>0 && o.prefix_tokens<=512 &&
                o.arguments.size()==(o.steady?9u:7u) && o.outputs.size()==(o.steady?1u:3u) && !o.weights.has_runtime_loras(),
                "unexpected ordinary base block boundary");
            const auto& args=o.arguments;const int rows=o.steady?1024:1024+o.prefix_tokens;
            require(args[0].shape()==mx::Shape{1,rows,4096} && args[0].dtype()==mx::bfloat16,"actual 512px BF16 input required");
            std::vector<std::string> before;for(const auto& value:args)before.push_back(hash(value));
            const auto reference_sha=hash(o.outputs[0]);
            auto values=replay(o);const auto error=difference(values[0],o.outputs[0]);
            require(error[0]<=.001,"exposed-output graph failed fixed whole-block relative L2 gate");
            const auto actual_k=o.steady?args[7]:o.outputs[1],actual_v=o.steady?args[8]:o.outputs[2];
            require(hash(slice_axis(values[2],2,0,o.prefix_tokens))==hash(actual_k) &&
                    hash(slice_axis(values[3],2,0,o.prefix_tokens))==hash(actual_v),"replayed prefix differs from actual original GPU cache");
            const std::string phase=o.steady?"steady":"prefill",filename=phase+".safetensors";
            std::unordered_map<std::string,Tensor> arrays;
            std::ostringstream record;record << std::setprecision(17) << "{\"phase\":" << quote(phase)
                << ",\"file\":" << quote(filename) << ",\"query_rows\":" << rows << ",\"key_rows\":" << values[2].shape(2)
                << ",\"prefix_rows\":" << o.prefix_tokens << ",\"source_layer\":0,\"source_step\":" << (o.steady?1:0)
                << ",\"full_block_relative_l2\":" << error[0] << ",\"full_block_max_abs\":" << error[1]
                << ",\"actual_prefix_exact\":true,\"input_sha256\":" << quote(before[0])
                << ",\"ordinary_block_output_sha256\":" << quote(reference_sha) << ",\"replayed_block_output_sha256\":" << quote(hash(values[0]))
                << ",\"segments\":[";
            if(!o.steady)for(size_t i=0;i<o.segments.size();++i) {
                if(i)record << ',';const auto& s=o.segments[i];
                record << "{\"start\":" << s.start << ",\"end\":" << s.end << ",\"causal\":" << (s.causal?"true":"false") << '}';
            }
            record << "],\"tensors\":[";
            uint64_t bytes=0;const std::array<const char*,4> keys{"q","k","v","reference"};
            for(size_t i=0;i<keys.size();++i) {
                const auto& value=values[i+1];require(value.ndim()==4 && value.dtype()==mx::bfloat16 && value.shape(0)==1 &&
                    value.shape(1)==32 && value.shape(3)==128,"captured attention ABI differs");
                finite(value);bytes+=value.nbytes();require(bytes<=cap,"single-phase tensor cap exceeded");
                auto copy=mx::copy(mx::contiguous(value));mx::eval(copy);arrays.emplace(keys[i],copy);
                require(hash(copy)==hash(value),"serialization copy changed logical values");
                if(i)record << ',';record << "{\"name\":" << quote(keys[i]) << ",\"shape\":";numbers(record,value.shape());
                record << ",\"replay_element_strides\":";numbers(record,value.strides());
                record << ",\"replay_offset_bytes\":" << value.offset() << ",\"sha256\":" << quote(hash(copy)) << '}';
            }
            require(written+bytes+(1<<20)<=cap && std::filesystem::space(output).available>bytes+512ull*1024*1024,
                    "bounded local tensor/disk admission failed");
            require(!std::filesystem::exists(output/filename),"capture file already exists");
            mx::save_safetensors((output/filename).string(),arrays,{{"recipe","qwen21-observer-attention-replay-v1"},{"phase",phase},
                {"serialization","logical contiguous BF16 copy; original physical strides separately recorded"}});
            written+=std::filesystem::file_size(output/filename);require(written<=cap,"actual capture cap exceeded");
            for(size_t i=0;i<args.size();++i)require(hash(args[i])==before[i],"replay mutated an original input/modulation/RoPE/cache allocation");
            require(hash(o.outputs[0])==reference_sha,"replay mutated ordinary block output");
            record << "],\"logical_tensor_bytes\":" << bytes << ",\"file_bytes\":" << std::filesystem::file_size(output/filename)
                << ",\"source_inputs_unchanged\":true}";records.push_back(record.str());
            std::cerr << "attention_replay phase=" << phase << " block_relative_l2=" << error[0] << " logical_bytes=" << bytes << '\n';
            if(o.steady)throw qwen21::DiagnosticBlockReplayComplete();
        });
        Request request;request.model="qwen-image-2.1";request.width=request.height=512;request.steps=20;request.seed=uint32_t(seed);
        request.prompt="A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
        request.audio=false;request.execution="gpu";request.residency="component_staged";request.prompt_enhance=false;
        request.qwen21_dit_cache="off";request.qwen21_dit_cache_explicit=true;request.output=(output/"forbidden-image.png").string();
        const Event events=[](const std::string& phase,int step,int total) {
            require((phase.find("vae")==std::string::npos || phase=="load_qwen21_vae") && phase!="export","capture entered VAE decode/export");
            std::cerr << phase << ' ' << step << '/' << total << '\n';
        };
        try { (void)session.generate(request,events,cancelled); }
        catch(const qwen21::DiagnosticBlockReplayComplete&) { completed=true; }
        session.set_diagnostic_block_observer({});session.unload();mx::synchronize();
        require(completed && records.size()==2 && !std::filesystem::exists(request.output),"bounded attention trajectory did not stop before image decode");
        char* identity=tc_runtime_build_identity();require(identity,"loaded native identity missing");
        std::ofstream out(output/"driver.json");out << "{\"schema\":\"tc-qwen-attention-observer-replay-v1\",\"loaded_runtime_id\":"
            << quote(identity) << ",\"seed\":" << seed << ",\"total_schedule_steps\":20,\"completed_full_dit_steps\":1,\"stopped_step\":1,\"stopped_layer\":0,"
            << "\"raw_capture_bytes\":" << written << ",\"production_qualified\":false,\"not_performance\":true,\"vae_decode\":false,\"image_export\":false,\"records\":["
            << records[0] << ',' << records[1] << "]}\n";tc_string_free(identity);require(bool(out),"capture receipt write failed");
        std::cout << "bounded original-input attention replay captured\n";return 0;
    } catch(const std::exception& error) { std::cerr << "attention_capture_failed: " << error.what() << '\n';return 3; }
}
