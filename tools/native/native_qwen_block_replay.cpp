// Opt-in, bounded same-process replay of actual Qwen base block inputs.
// Step zero completes only to obtain the real Euler input for step one; that
// step stops after block zero. No VAE inference, image export or tensor capture
// files; the ordinary route may preload the existing VAE checkpoint.
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/diagnostic_block_observer.hpp"
#include "../../native/models/qwen21/memory_policy.hpp"
#include "../../native/models/qwen21/native_block.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"
#include "../../native/backends/ane_smoothquant.hpp"
#include "native_qwen_block_replay_math.hpp"
#include "turbocider/turbocider.h"

#include <CommonCrypto/CommonDigest.h>
#include <crt_externs.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

using namespace tc;
namespace mg = tc::metal_graph;
namespace ng = tc::qwen21::native_graph;
namespace math = tc::qwen21::block_replay_math;

namespace {
std::string quote(std::string_view s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                             << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    }
    out << '"';
    return out.str();
}
double ms(Clock::time_point begin) {
    return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
}
mg::MatrixView matrix(mg::Backend& backend,const Tensor& tensor) {
    const auto imported=mg::import_mlx_matrix(backend,tensor);
    return {imported.storage,imported.rows,imported.columns,imported.row_stride_bytes};
}
size_t prefix_stride_bytes(size_t elements) {
    require(elements<=std::numeric_limits<size_t>::max()/2,"prefix pitch overflow");
    return elements*2;
}
struct PrefixSource {
    mg::Buffer backing;
    math::View view;
    std::array<int,4> shape;
    std::array<size_t,4> strides_bytes;
    size_t offset_bytes;
};
PrefixSource prefix_source(mg::Backend& backend,const Tensor& tensor,uint32_t tokens) {
    require(tensor.is_available() && tensor.ndim()==4 && tensor.shape(0)==1 &&
        tensor.shape(1)==32 && tensor.shape(2)==int(tokens) && tensor.shape(3)==128 &&
        tensor.dtype()==mx::bfloat16 && tensor.strides().back()==1 && tensor.offset()>=0,
        "replay requires completed BF16 [1,32,tokens,128] actual prefix storage");
    PrefixSource result;
    for(size_t i=0;i<4;++i) {
        result.shape[i]=tensor.shape(int(i));
        result.strides_bytes[i]=prefix_stride_bytes(tensor.strides()[i]);
    }
    result.offset_bytes=size_t(tensor.offset());
    result.backing=backend.import_buffer(const_cast<void*>(tensor.buffer().ptr()),
                                       std::make_shared<Tensor>(tensor));
    require(result.backing.contents() && tensor.buffer_size()<=result.backing.size_bytes() &&
        result.offset_bytes<=tensor.buffer_size(),"actual prefix offset/backing is invalid");
    result.view={static_cast<const std::byte*>(result.backing.contents())+result.offset_bytes,
        tensor.buffer_size()-result.offset_bytes,32,tokens,128,
        result.strides_bytes[1],result.strides_bytes[2]};
    math::validate(result.view); // CPU only: also accepts separated interleaved heads.
    return result;
}
mg::HeadView heads(mg::Backend& backend,const Tensor& tensor,uint32_t tokens) {
    require(tensor.is_available() && tensor.ndim()==4 && tensor.shape(0)==1 &&
        tensor.shape(1)==32 && tensor.shape(2)==int(tokens) && tensor.shape(3)==128 &&
        tensor.dtype()==mx::bfloat16 && tensor.strides().back()==1 && tensor.offset()>=0,
        "replay requires completed BF16 [1,32,tokens,128] prefix storage");
    const size_t token_stride=prefix_stride_bytes(tensor.strides()[2]);
    const size_t head_stride=prefix_stride_bytes(tensor.strides()[1]);
    require(size_t(tensor.offset())<=tensor.buffer_size(),"prefix offset exceeds backing");
    const size_t extent=mg::qwen_ops::head_extent(32,tokens,128,token_stride,head_stride,
        2,size_t(tensor.offset()),tensor.buffer_size()-size_t(tensor.offset()));
    auto backing=backend.import_buffer(const_cast<void*>(tensor.buffer().ptr()),
                                      std::make_shared<Tensor>(tensor));
    return {backing.view(size_t(tensor.offset()),extent,mg::ScalarType::BF16),
            32,tokens,128,token_stride,head_stride};
}
mg::HeadView allocate_heads(mg::Backend& backend,uint32_t tokens,const char* label) {
    require(tokens && tokens<=512,"bounded base text prefix required");
    const size_t bytes=size_t(tokens)*4096*2;
    auto backing=backend.allocate(bytes,label,mg::StorageMode::Shared);
    return {backing.view(0,bytes,mg::ScalarType::BF16),32,tokens,128,256,size_t(tokens)*256};
}
math::View cpu(const mg::MatrixView& m) {
    require(m.storage.scalar_type()==mg::ScalarType::BF16 && m.storage.contents(),
            "CPU inspection requires completed Shared BF16 matrix");
    return {static_cast<const std::byte*>(m.storage.contents()),m.storage.size_bytes(),
            1,m.rows,m.columns,m.storage.size_bytes(),m.row_stride_bytes};
}
math::View cpu(const mg::HeadView& h) {
    require(h.storage.scalar_type()==mg::ScalarType::BF16 && h.storage.contents(),
            "CPU inspection requires completed Shared BF16 head storage");
    return {static_cast<const std::byte*>(h.storage.contents()),h.storage.size_bytes(),
            h.heads,h.tokens,h.dimension,h.head_stride_bytes,h.token_stride_bytes};
}
math::View raw_cpu(const mg::MatrixView& m) {
    const auto item=mg::scalar_bytes(m.storage.scalar_type());
    require((m.storage.scalar_type()==mg::ScalarType::BF16 ||
             m.storage.scalar_type()==mg::ScalarType::FP32) && m.storage.contents(),
            "raw CPU digest requires completed Shared BF16/FP32 matrix");
    // for_each_row only hashes bytes; FP32 rows are represented as twice as
    // many two-byte units without conversion, evaluation, or numerical scan.
    return {static_cast<const std::byte*>(m.storage.contents()),m.storage.size_bytes(),
            1,m.rows,size_t(m.columns)*(item/2),m.storage.size_bytes(),m.row_stride_bytes};
}
std::string hash(math::View view) {
    CC_SHA256_CTX context;
    require(CC_SHA256_Init(&context)==1,"cannot initialize replay SHA");
    math::for_each_row(view,[&](const std::byte* data,size_t bytes) {
        require(bytes<=UINT32_MAX && CC_SHA256_Update(&context,data,CC_LONG(bytes))==1,
                "cannot update replay SHA");
    });
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
    require(CC_SHA256_Final(digest.data(),&context)==1,"cannot finalize replay SHA");
    static constexpr char hex[]="0123456789abcdef";
    std::string value(digest.size()*2,'0');
    for(size_t i=0;i<digest.size();++i) {
        value[2*i]=hex[digest[i]>>4]; value[2*i+1]=hex[digest[i]&15];
    }
    return value;
}
ng::BlockWeights block_weights(mg::Backend& backend,const Weights& weights) {
    require(!weights.has_runtime_loras(),"block replay is BaseOnly");
    const std::string base="transformer_blocks.0.";
    for(const auto& suffix:{"img_mlp.gate_up.bias","img_mlp.out.bias"})
        require(!weights.has(base+suffix),"native FFN replay cannot omit checkpoint bias");
    auto gu=weights.at(base+"img_mlp.gate_up.weight");
    require(gu.shape()==mx::Shape{24576,4096} && gu.dtype()==mx::bfloat16,
            "replay requires local dense BF16 Qwen fused gate/up");
    auto gate=slice_axis(gu,0,0,12288),up=slice_axis(gu,0,12288,24576);
    auto down=weights.at(base+"img_mlp.out.weight");
    auto qnorm=mx::reshape(weights.at(base+"attn.norm_q.weight"),{1,128});
    auto knorm=mx::reshape(weights.at(base+"attn.norm_k.weight"),{1,128});
    // Replay itself only evaluates the observed layer's source tensors; the
    // ordinary Session may already have materialized the full DiT. This driver
    // performs no materialize() or second checkpoint load.
    std::vector<Tensor> selected{gate,up,down,qnorm,knorm};
    std::array<std::string,4> names{base+"attn.to_q",base+"attn.to_k",
        base+"attn.to_v",base+"attn.to_out.0"};
    for(const auto& name:names) {
        selected.push_back(weights.at(name+".weight"));
        if(weights.has(name+".bias"))selected.push_back(weights.at(name+".bias"));
    }
    mx::eval(selected);
    ng::BlockWeights result;
    std::array<ng::Projection*,4> projections{&result.query,&result.key,&result.value,
                                            &result.attention_output};
    for(size_t i=0;i<names.size();++i) {
        projections[i]->weight=matrix(backend,weights.at(names[i]+".weight"));
        if(weights.has(names[i]+".bias")) {
            auto bias=mx::reshape(weights.at(names[i]+".bias"),{1,4096});mx::eval(bias);
            projections[i]->bias=matrix(backend,bias);
        }
    }
    result.query_norm=matrix(backend,qnorm);result.key_norm=matrix(backend,knorm);
    result.gate=matrix(backend,gate);result.up=matrix(backend,up);result.down=matrix(backend,down);
    return result;
}
struct Result {
    bool steady;
    uint32_t rows,prefix;
    size_t graph_steps,workspace_bytes;
    std::string input_sha,reference_sha,output_sha,prefix_key_sha,prefix_value_sha;
    std::array<std::string,7> argument_sha{};
    std::array<int,4> prefix_shape{};
    std::array<size_t,4> prefix_key_strides_bytes{},prefix_value_strides_bytes{};
    size_t prefix_key_offset_bytes=0,prefix_value_offset_bytes=0,prefix_copy_bytes=0;
    double prefix_copy_ms=0;
    math::Difference difference{},prefix_key_difference{},prefix_value_difference{};
    std::array<double,2> gpu_ms{},native_wall_ms{},native_device_ms{};
};
void array(std::ostream& out,const std::array<double,2>& value) {
    out << '[' << value[0] << ',' << value[1] << ']';
}
template<class T,size_t N>
void integers(std::ostream& out,const std::array<T,N>& values) {
    out << '[';
    for(size_t i=0;i<N;++i) { if(i)out << ',';out << values[i]; }
    out << ']';
}
void difference(std::ostream& out,const math::Difference& value) {
    out << "{\"rrms\":" << value.relative_rms << ",\"max_abs\":" << value.max_abs
        << ",\"finite_values\":" << value.values << '}';
}
void clear_experiments() {
    std::vector<std::string> remove;
    for(char** item=*::_NSGetEnviron();item && *item;++item) {
        std::string entry=*item;
        if(entry.starts_with("TURBOCIDER_"))remove.push_back(entry.substr(0,entry.find('=')));
    }
    for(const auto& key:remove)require(::unsetenv(key.c_str())==0,"cannot clear inherited experiment");
}
}

int main(int argc,char** argv) {
    try {
        require(argc==5,"usage: native-qwen-block-replay MODEL_DIRECTORY OUTPUT_JSON PROMPT SEED");
        const auto root=std::filesystem::canonical(argv[1]);
        const auto output=std::filesystem::absolute(argv[2]);
        require(!std::filesystem::exists(output) && std::filesystem::is_directory(output.parent_path()),
                "replay receipt requires a fresh file in an existing directory");
        const auto checkpoint=root/"diffusion_models/qwen_image_2.1_bf16.safetensors";
        const auto identity=ane::smoothquant::checkpoint_fingerprint(checkpoint);
        const auto encoder=root/"text_encoders/qwen3vl_8b_bf16.safetensors";
        const auto tokenizer=root/"processor/tokenizer.json";
        const auto encoder_identity=ane::smoothquant::checkpoint_fingerprint(encoder);
        const auto tokenizer_identity=ane::smoothquant::checkpoint_fingerprint(tokenizer);
        const std::string prompt=argv[3];
        require(!prompt.empty() && prompt.size()<=1024,"replay prompt must be bounded and nonempty");
        size_t used=0;const auto seed=std::stoull(argv[4],&used);
        require(used==std::string(argv[4]).size() && seed<=UINT32_MAX,"invalid replay seed");
        clear_experiments();configure_streams();
        mg::Backend backend;
        const auto begin_prepare=Clock::now();
        auto native=ng::NativeBlock::prepare(backend);
        const double prepare_ms=ms(begin_prepare);
        std::vector<Result> results;
        bool completed=false;
        std::atomic<bool> cancelled{false};
        qwen21::Session session(root);
        session.set_diagnostic_block_observer([&](const qwen21::DiagnosticBlockObservation& o) {
            if(o.layer!=0)return;
            require(results.size()<2 && o.steady==(results.size()==1) && o.prefix_tokens>0 &&
                    o.prefix_tokens<=512 && o.arguments.size()==(o.steady?9u:7u) &&
                    o.outputs.size()==(o.steady?1u:3u) && bool(o.gpu_block),
                    "unexpected block/step/route in actual trajectory");
            auto arguments=o.arguments; // Retain inputs across all compiled/native submissions.
            const auto reference=o.outputs[0];
            auto ready=arguments;ready.push_back(reference);
            if(!o.steady) { ready.push_back(o.outputs[1]);ready.push_back(o.outputs[2]); }
            mx::eval(ready); // Sole cross-queue readiness boundary, outside paired timings.
            const uint32_t rows=uint32_t(arguments[0].shape(1));
            require(arguments[0].shape()==mx::Shape{1,int(o.steady?1024:1024+o.prefix_tokens),4096} &&
                    arguments[0].dtype()==mx::bfloat16 && reference.shape()==arguments[0].shape() &&
                    reference.dtype()==mx::bfloat16,"capture is not an actual 512px base block");
            const auto actual_key=o.steady?arguments[7]:o.outputs[1];
            const auto actual_value=o.steady?arguments[8]:o.outputs[2];
            const auto key_source=prefix_source(backend,actual_key,uint32_t(o.prefix_tokens));
            const auto value_source=prefix_source(backend,actual_value,uint32_t(o.prefix_tokens));
            const auto key_sha=hash(key_source.view),value_sha=hash(value_source.view);
            const auto copy_started=Clock::now();
            // Preserve the ordinary callable's original interleaved/sliced
            // cache. This bounded copy follows the existing prefix snapshot
            // idiom and occurs entirely outside all paired block timings.
            auto compact_key=mx::copy(mx::contiguous(mx::astype(actual_key,actual_key.dtype(),
                                                      std::optional<bool>{true})));
            auto compact_value=mx::copy(mx::contiguous(mx::astype(actual_value,actual_value.dtype(),
                                                        std::optional<bool>{true})));
            mx::eval(compact_key,compact_value);
            const double copy_ms=ms(copy_started);
            require(compact_key.flags().row_contiguous && compact_value.flags().row_contiguous &&
                compact_key.buffer().ptr()!=actual_key.buffer().ptr() &&
                compact_value.buffer().ptr()!=actual_value.buffer().ptr(),
                "prefix compact copy must have independent row-contiguous backing");
            const auto compact_k=heads(backend,compact_key,uint32_t(o.prefix_tokens));
            const auto compact_v=heads(backend,compact_value,uint32_t(o.prefix_tokens));
            require(hash(cpu(compact_k))==key_sha && hash(cpu(compact_v))==value_sha,
                    "prefix compaction changed its actual logical values");
            auto candidate=mg::allocate_mlx_matrix(rows,4096);
            ng::BlockBindings bindings;
            bindings.input=matrix(backend,arguments[0]);bindings.output=matrix(backend,candidate);
            for(size_t i=0;i<4;++i)bindings.modulation[i]=matrix(backend,arguments[i+1]);
            bindings.cosine=matrix(backend,arguments[5]);bindings.sine=matrix(backend,arguments[6]);
            bindings.weights=block_weights(backend,o.weights);
            bindings.phase=o.steady?ng::BlockPhase::SteadyDecode:ng::BlockPhase::PrefixBuild;
            bindings.prefix_tokens=uint32_t(o.prefix_tokens);
            if(o.steady)
                bindings.prefix=ng::PrefixKv{compact_k,compact_v};
            else {
                bindings.prefix=ng::PrefixKv{allocate_heads(backend,bindings.prefix_tokens,"replay-prefix-k"),
                                             allocate_heads(backend,bindings.prefix_tokens,"replay-prefix-v")};
                bindings.segments=o.segments;
            }
            auto workspace=native.allocate_workspace(backend,rows,bindings.prefix_tokens,bindings.phase);
            const auto reference_view=matrix(backend,reference);
            Result result{};
            result.steady=o.steady;result.rows=rows;result.prefix=bindings.prefix_tokens;
            result.workspace_bytes=workspace.reserved_bytes;
            result.prefix_key_sha=key_sha;result.prefix_value_sha=value_sha;
            result.prefix_shape=key_source.shape;
            result.prefix_key_strides_bytes=key_source.strides_bytes;
            result.prefix_value_strides_bytes=value_source.strides_bytes;
            result.prefix_key_offset_bytes=key_source.offset_bytes;
            result.prefix_value_offset_bytes=value_source.offset_bytes;
            result.prefix_copy_bytes=compact_key.nbytes()+compact_value.nbytes();
            result.prefix_copy_ms=copy_ms;
            require(result.prefix_copy_bytes==size_t(o.prefix_tokens)*4096*2*2,
                    "prefix copy exceeds its bounded BF16 cache payload");
            std::cerr << "block_replay_prefix phase=" << (o.steady?"steady":"prefill")
                      << " compact_bytes=" << result.prefix_copy_bytes
                      << " copy_ms=" << result.prefix_copy_ms << '\n';
            result.input_sha=hash(cpu(bindings.input));result.reference_sha=hash(cpu(reference_view));
            for(size_t i=0;i<result.argument_sha.size();++i)
                result.argument_sha[i]=hash(raw_cpu(matrix(backend,arguments[i])));
            const auto execute_native=[&] {
                const auto started=Clock::now();
                mg::CommandGraph graph;native.encode(graph,bindings,workspace);
                result.graph_steps=graph.steps().size();
                auto ticket=backend.submit(std::move(graph));
                auto completion=ticket.wait(std::chrono::seconds(30));
                require(completion.ok && completion.physically_complete,
                        "native block did not physically complete: "+completion.error);
                return std::array<double,2>{ms(started),completion.gpu_seconds*1000.};
            };
            const auto execute_gpu=[&] {
                const auto started=Clock::now();auto values=o.gpu_block(arguments);mx::eval(values);
                const double elapsed=ms(started);
                require(values[0].shape()==reference.shape() && values[0].dtype()==mx::bfloat16 &&
                        hash(cpu(matrix(backend,values[0])))==result.reference_sha,
                        "ordinary compiled block replay changed its actual reference");
                return elapsed;
            };
            // One warmup of each prepared recipe, then precisely two paired
            // samples in opposite order. CPU hash/error work is never timed.
            (void)execute_gpu();(void)execute_native();
            result.difference=math::compare(cpu(bindings.output),cpu(reference_view));
            std::cerr << "block_replay_screen phase=" << (o.steady?"steady":"prefill")
                      << " rows=" << rows << " rrms=" << result.difference.relative_rms
                      << " max_abs=" << result.difference.max_abs << '\n';
            require(result.difference.relative_rms<=0.08,"native block screening RRMS exceeds 0.08");
            const auto screened_native_sha=hash(cpu(bindings.output));
            if(!o.steady) {
                result.prefix_key_difference=math::compare(cpu(bindings.prefix->key),cpu(compact_k));
                result.prefix_value_difference=math::compare(cpu(bindings.prefix->value),cpu(compact_v));
                require(result.prefix_key_difference.relative_rms<=0.08 &&
                        result.prefix_value_difference.relative_rms<=0.08,
                        "native prefill prefix screening RRMS exceeds 0.08");
            } else {
                require(result.prefix_key_sha==results[0].prefix_key_sha &&
                        result.prefix_value_sha==results[0].prefix_value_sha,
                        "step one cache is not the captured actual prefill cache");
            }
            for(size_t round=0;round<2;++round) {
                const auto run_native=[&] { const auto spans=execute_native();
                    result.native_wall_ms[round]=spans[0];result.native_device_ms[round]=spans[1];
                    // As with the ordinary callable, verify reproducibility
                    // only after the timed physical-completion boundary.
                    require(hash(cpu(bindings.output))==screened_native_sha,
                            "timed native block changed its screened output"); };
                if(round==0) { result.gpu_ms[round]=execute_gpu();run_native(); }
                else { run_native();result.gpu_ms[round]=execute_gpu(); }
            }
            result.output_sha=hash(cpu(bindings.output));
            require(hash(cpu(bindings.input))==result.input_sha,"replay mutated its real input");
            for(size_t i=0;i<result.argument_sha.size();++i)
                require(hash(raw_cpu(matrix(backend,arguments[i])))==result.argument_sha[i],
                        "replay mutated actual modulation/RoPE/input storage");
            if(o.steady)require(hash(cpu(bindings.prefix->key))==result.prefix_key_sha &&
                hash(cpu(bindings.prefix->value))==result.prefix_value_sha,"replay mutated its real prefix");
            require(hash(key_source.view)==key_sha && hash(value_source.view)==value_sha,
                    "replay mutated the ordinary GPU's original prefix storage");
            std::cerr << "block_replay phase=" << (o.steady?"steady":"prefill")
                      << " rows=" << rows << " rrms=" << result.difference.relative_rms << '\n';
            results.push_back(std::move(result));
            if(o.steady)throw qwen21::DiagnosticBlockReplayComplete();
        });
        Request request;
        request.model="qwen-image-2.1";request.width=request.height=512;request.steps=20;
        request.seed=uint32_t(seed);request.prompt=prompt;request.audio=false;
        request.execution="gpu";request.residency="component_staged";request.prompt_enhance=false;
        request.qwen21_dit_cache="off";request.qwen21_dit_cache_explicit=true;
        const bool actual_component_staged=request.residency=="component_staged";
        const bool actual_layer_staged=qwen21::layer_staged_t2i(request,device_info());
        std::cerr << "block_replay_route component_staged=" << actual_component_staged
                  << " layer_staged=" << actual_layer_staged << '\n';
        request.output=(output.parent_path()/"forbidden-block-replay-image.png").string();
        require(!std::filesystem::exists(request.output),"stop sentinel already exists");
        bool actual_vae_weights_loaded=false;
        const Event events=[&](const std::string& phase,int step,int total) {
            // The ordinary component-staged route may preload the existing
            // VAE checkpoint alongside the DiT. Loading is not VAE inference.
            if(phase=="load_qwen21_vae" && step==total)actual_vae_weights_loaded=true;
            require((phase.find("vae")==std::string::npos || phase=="load_qwen21_vae") && phase!="export",
                    "bounded replay unexpectedly entered VAE/export");
            std::cerr << phase << ' ' << step << '/' << total << '\n';
        };
        try { (void)session.generate(request,events,cancelled); }
        catch(const qwen21::DiagnosticBlockReplayComplete&) { completed=true; }
        session.set_diagnostic_block_observer({});session.unload();
        require(completed && results.size()==2 && !std::filesystem::exists(request.output),
                "bounded trajectory did not stop at step one block zero");
        require(ane::smoothquant::checkpoint_fingerprint(checkpoint)==identity,
                "checkpoint source generation changed during replay");
        require(ane::smoothquant::checkpoint_fingerprint(encoder)==encoder_identity &&
                ane::smoothquant::checkpoint_fingerprint(tokenizer)==tokenizer_identity,
                "ordinary encoder/tokenizer source generation changed during replay");
        char* build=tc_runtime_build_identity();require(build,"native runtime identity unavailable");
        const std::string runtime_identity(build);tc_string_free(build);
        std::ofstream out(output,std::ios::out|std::ios::binary);
        out << std::setprecision(12) << "{\"schema\":\"native-qwen-block-replay-v1\","
            "\"scope\":\"BaseOnly layer0 full block; actual GPU step0 and Euler step1 inputs; not E2E or quality qualification\","
            "\"capture_mode\":\"transient_same_process_actual_compiled_callable\","
            "\"steady_cache_source\":\"actual_GPU_prefill_cache_compact_copy\",\"native_prefill_cache_reused\":false,"
            "\"native_timed_output_repeatability_checked\":true,"
            "\"text_source\":\"ordinary_Qwen_encoder_in_new_session\",\"tensor_files_bytes\":0,"
            "\"actual_component_staged\":" << (actual_component_staged?"true":"false")
            << ",\"actual_layer_staged\":" << (actual_layer_staged?"true":"false")
            << ",\"actual_vae_weights_loaded\":" << (actual_vae_weights_loaded?"true":"false")
            << ",\"completed_full_dit_steps\":1,\"stopped_step\":1,\"stopped_layer\":0,\"vae_decode\":false,"
            "\"image_export\":false,\"source_identity_kind\":\"canonical-stat-identity\",\"source_identity\":"
            << quote(identity) << ",\"runtime_build\":" << quote(runtime_identity)
            << ",\"encoder_source_identity\":" << quote(encoder_identity)
            << ",\"tokenizer_source_identity\":" << quote(tokenizer_identity)
            << ",\"device\":" << quote(backend.device_name()) << ",\"prompt\":" << quote(prompt)
            << ",\"seed\":" << seed << ",\"total_schedule_steps\":20,\"prepare_ms\":" << prepare_ms
            << ",\"screening_rrms_limit\":0.08,\"warmups_each_phase\":1,\"paired_samples_each_phase\":2,"
            "\"native_wall_scope\":\"encode/submit/physical completion; preparation/import/allocation excluded\","
            "\"prefix_compact_copy_scope\":\"same BF16 values; creation/evaluation outside paired block timings; logical bytes\","
            "\"gpu_wall_scope\":\"actual compiled callable/evaluation; hash excluded\",\"samples\":[";
        for(size_t i=0;i<results.size();++i) {
            if(i)out << ',';const auto& r=results[i];
            out << "{\"step\":" << i << ",\"layer\":0,\"phase\":" << quote(r.steady?"steady":"prefill")
                << ",\"rows\":" << r.rows << ",\"prefix_tokens\":" << r.prefix
                << ",\"workspace_bytes\":" << r.workspace_bytes << ",\"graph_steps\":" << r.graph_steps
                << ",\"input_sha256\":" << quote(r.input_sha) << ",\"reference_sha256\":" << quote(r.reference_sha)
                << ",\"native_sha256\":" << quote(r.output_sha) << ",\"prefix_key_sha256\":" << quote(r.prefix_key_sha)
                << ",\"prefix_value_sha256\":" << quote(r.prefix_value_sha)
                << ",\"prefix_original_shape\":";integers(out,r.prefix_shape);
            out << ",\"prefix_key_original_strides_bytes\":";integers(out,r.prefix_key_strides_bytes);
            out << ",\"prefix_value_original_strides_bytes\":";integers(out,r.prefix_value_strides_bytes);
            out << ",\"prefix_key_original_offset_bytes\":" << r.prefix_key_offset_bytes
                << ",\"prefix_value_original_offset_bytes\":" << r.prefix_value_offset_bytes
                << ",\"prefix_compact_copy_bytes\":" << r.prefix_copy_bytes
                << ",\"prefix_compact_copy_wall_ms\":" << r.prefix_copy_ms << ",\"output_error\":";
            difference(out,r.difference);out << ",\"native_prefix_key_error\":";
            if(r.steady)out << "null";else difference(out,r.prefix_key_difference);
            out << ",\"native_prefix_value_error\":";
            if(r.steady)out << "null";else difference(out,r.prefix_value_difference);
            out << ",\"actual_argument_names\":[\"input\",\"mod0\",\"mod1\",\"mod2\",\"mod3\",\"cosine\",\"sine\"],"
                   "\"actual_argument_sha256\":[";
            for(size_t j=0;j<r.argument_sha.size();++j) {
                if(j)out << ',';out << quote(r.argument_sha[j]);
            }
            out << ']';
            out << ",\"gpu_ms\":";array(out,r.gpu_ms);out << ",\"native_wall_ms\":";array(out,r.native_wall_ms);
            out << ",\"native_device_ms\":";array(out,r.native_device_ms);out << '}';
        }
        out << "]}\n";out.close();require(bool(out),"failed writing replay receipt");
        require(std::filesystem::file_size(output)<1024*1024,"unbounded replay receipt");
        std::cout << output << '\n';return 0;
    } catch(const std::exception& error) {
        std::cerr << "native Qwen block replay failed: " << error.what() << '\n';return 1;
    }
}
