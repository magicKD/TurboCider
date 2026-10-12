#include "native_denoiser.hpp"
#include "native_denoiser_contract.hpp"

#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
#include "../../backends/metal_graph/native_hybrid_ffn.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>

namespace tc::qwen21::native_graph {
namespace {
using namespace metal_graph;
using Clock=std::chrono::steady_clock;
size_t bytes(uint32_t rows,uint32_t columns,ScalarType type=ScalarType::BF16) {
    if (!rows || !columns || !scalar_bytes(type) ||
        size_t(rows)>std::numeric_limits<size_t>::max()/columns/scalar_bytes(type))
        throw std::invalid_argument("native denoiser matrix extent overflow");
    return size_t(rows)*columns*scalar_bytes(type);
}
void shape(const MatrixView& value,uint32_t rows,uint32_t columns,
           ScalarType type=ScalarType::BF16) {
    dense_linear::validate_layout({value.rows,value.columns,value.row_stride_bytes,
        value.storage.size_bytes(),value.storage.offset_bytes(),value.storage.scalar_type()});
    if (!value.storage || value.rows!=rows || value.columns!=columns || value.storage.scalar_type()!=type)
        throw std::invalid_argument("native denoiser matrix shape/dtype mismatch");
}
void projection(const Projection& value,uint32_t input,uint32_t output) {
    shape(value.weight,output,input);
    if (value.bias) shape(*value.bias,1,output);
}
struct LoRARequirements { uint32_t max_rank=0; bool ffn=false; };
native_lora::Layout lora_layout(const MatrixView& v) {
    if (!v.storage) throw std::invalid_argument("native adapter factor requires a retained produced backing");
    return {v.rows,v.columns,v.row_stride_bytes,v.storage.size_bytes(),v.storage.offset_bytes(),
        v.storage.scalar_type(),reinterpret_cast<uintptr_t>(v.storage.buffer().native_handle())};
}
void adapter_terms(LoRARequirements& requirements,const std::vector<OrderedLoRATerm>& terms,
                   uint32_t input,uint32_t output) {
    if (terms.size()>64) throw std::invalid_argument("native projection has too many ordered adapter terms");
    for (const auto& term:terms) {
        const auto rank=native_lora::validate_term({},input,output,
            {lora_layout(term.a),lora_layout(term.b),term.effective_scale,term.first_output,term.output_count});
        requirements.max_rank=std::max(requirements.max_rank,rank);
    }
}
LoRARequirements weights(const DenoiserWeights& w,DenoiserGeometry g) {
    const auto h=g.block.hidden, f=g.block.ffn_width;
    LoRARequirements requirements;
    auto adapted_projection=[&](const Projection& p,uint32_t input,uint32_t output) {
        projection(p,input,output); adapter_terms(requirements,p.lora,input,output);
    };
    adapted_projection(w.image_input,64,h); adapted_projection(w.time_input,256,h);
    adapted_projection(w.time_output,h,h); adapted_projection(w.text_input,h,h); adapted_projection(w.text_output,h,h);
    adapted_projection(w.modulation,h,4*h); adapted_projection(w.output_scale,h,h); adapted_projection(w.output,h,64);
    shape(w.text_norm,1,h);
    if (w.blocks.size()!=g.layers) throw std::invalid_argument("native denoiser requires every layer's source weights");
    for (const auto& b:w.blocks) {
        for (const auto* p:{&b.query,&b.key,&b.value,&b.attention_output}) adapted_projection(*p,h,h);
        shape(b.query_norm,1,g.block.head_dim); shape(b.key_norm,1,g.block.head_dim);
        shape(b.gate,f,h); shape(b.up,f,h); shape(b.down,h,f);
        adapter_terms(requirements,b.gate_lora,h,f); adapter_terms(requirements,b.up_lora,h,f);
        adapter_terms(requirements,b.down_lora,f,h);
        requirements.ffn=requirements.ffn || !b.gate_lora.empty() || !b.up_lora.empty() || !b.down_lora.empty();
    }
    return requirements;
}
MatrixView allocate_matrix(Backend& backend,DenoiserMemory& memory,uint32_t rows,uint32_t columns,
    std::string_view label,StorageMode mode=StorageMode::Private,ScalarType type=ScalarType::BF16) {
    const auto size=bytes(rows,columns,type);
    if (size>std::numeric_limits<size_t>::max()-memory.request_bytes)
        throw std::invalid_argument("native denoiser request byte overflow");
    auto buffer=backend.allocate(size,label,mode);
    memory.request_bytes+=size;
    return {buffer.view(0,size,type),rows,columns,size_t(columns)*scalar_bytes(type)};
}
void add_memory(DenoiserMemory& memory,size_t count) {
    if (count>std::numeric_limits<size_t>::max()-memory.request_bytes)
        throw std::invalid_argument("native denoiser request byte overflow");
    memory.request_bytes+=count;
}
void linear(CommandGraph& graph,const NativeLinear& op,const MatrixView& input,
            const Projection& p,const MatrixView& output,const std::optional<NativeLoRA>& lora,
            const MatrixView& low) {
    if (p.lora.empty()) { op.encode(graph,{input,p.weight,output,p.bias}); return; }
    if (!lora) throw std::invalid_argument("native projection adapter was not prepared");
    op.encode(graph,{input,p.weight,output,{}});
    lora->encode_ordered_corrections(graph,input,p.lora,output,output,matrix_rows(low,0,input.rows),p.bias);
}
MatrixView flatten(const HeadView& h) {
    // Only the executor's tightly packed cache banks use this finite-check view.
    if (h.token_stride_bytes!=size_t(h.dimension)*2 ||
        h.head_stride_bytes!=size_t(h.tokens)*h.token_stride_bytes ||
        uint64_t(h.heads)*h.tokens>UINT32_MAX)
        throw std::invalid_argument("native denoiser prefix bank is not tightly packed");
    return {h.storage,h.heads*h.tokens,h.dimension,h.token_stride_bytes};
}
void schedule(const std::vector<float>& sigmas) {
    if (sigmas.size()<2 || sigmas.size()>257 || sigmas.back()!=0.f)
        throw std::invalid_argument("native denoiser needs an exact bounded schedule ending at zero");
    for (size_t i=0;i<sigmas.size();++i) {
        if (!std::isfinite(sigmas[i]) || sigmas[i]<0.f || sigmas[i]>1.f ||
            (i && sigmas[i]>=sigmas[i-1]))
            throw std::invalid_argument("native denoiser sigmas must strictly decrease in [0,1]");
    }
}
} // namespace

void validate_geometry(DenoiserGeometry g) {
    validate_geometry(g.block);
    if (!g.layers || g.layers>32 || !g.maximum_rows || g.maximum_rows>65536)
        throw std::invalid_argument("native Qwen denoiser bounded layer/row contract");
}
MatrixView matrix_rows(const MatrixView& source,uint32_t first,uint32_t count) {
    dense_linear::validate_layout({source.rows,source.columns,source.row_stride_bytes,
        source.storage.size_bytes(),source.storage.offset_bytes(),source.storage.scalar_type()});
    if (!source.storage || !count || first>=source.rows || count>source.rows-first)
        throw std::invalid_argument("native denoiser row subview outside retained source");
    const size_t offset=size_t(first)*source.row_stride_bytes;
    const size_t extent=size_t(count-1)*source.row_stride_bytes+
        size_t(source.columns)*scalar_bytes(source.storage.scalar_type());
    return {source.storage.buffer().view(source.storage.offset_bytes()+offset,extent,
        source.storage.scalar_type()),count,source.columns,source.row_stride_bytes};
}
MatrixView matrix_columns(const MatrixView& source,uint32_t first,uint32_t count) {
    dense_linear::validate_layout({source.rows,source.columns,source.row_stride_bytes,
        source.storage.size_bytes(),source.storage.offset_bytes(),source.storage.scalar_type()});
    if (!source.storage || !count || first>=source.columns || count>source.columns-first)
        throw std::invalid_argument("native denoiser column subview outside retained source");
    const auto item=scalar_bytes(source.storage.scalar_type());
    const size_t offset=size_t(first)*item;
    const size_t extent=size_t(source.rows-1)*source.row_stride_bytes+size_t(count)*item;
    return {source.storage.buffer().view(source.storage.offset_bytes()+offset,extent,
        source.storage.scalar_type()),source.rows,count,source.row_stride_bytes};
}

struct NativeDenoiser::Impl {
    Backend* backend;
    DenoiserGeometry geometry;
    DenoiserWeights weights;
    std::string source_revision;
    NativeLinear square, image_input, time_input, modulation, output;
    NativeQwenOps ops;
    NativeStepOps step_ops;
    NativeBlock block;
    NativeDenoiserExecutionOptions execution;
    std::optional<NativeLoRA> lora;
    LoRARequirements lora_requirements;
};
NativeDenoiser::NativeDenoiser(std::shared_ptr<Impl> impl):impl_(std::move(impl)) {}
NativeDenoiser NativeDenoiser::prepare(Backend& backend,DenoiserWeights w,
                                     std::string source,DenoiserGeometry g,
                                     NativeDenoiserExecutionOptions execution) {
    validate_geometry(g);
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
    detail::validate_execution_options(g,execution,true);
#else
    detail::validate_execution_options(g,execution,false);
#endif
    const auto requirements=weights(w,g);
    execution.schedule_variant=detail::validate_lora_recipe(g,execution,requirements.max_rank!=0);
    if (requirements.max_rank) native_lora::validate_variant(
        {std::max({g.block.hidden,g.block.ffn_width,256U}),
         std::max(4*g.block.hidden,g.block.ffn_width),requirements.max_rank}, execution.lora_kernel);
    if (source.empty() || !backend.healthy())
        throw std::invalid_argument("native denoiser needs a healthy backend and verified source revision");
    const auto h=g.block.hidden;
    auto square=NativeLinear::prepare(backend,{h,h});
    auto image_input=NativeLinear::prepare(backend,{64,h});
    auto time_input=NativeLinear::prepare(backend,{256,h});
    auto modulation=NativeLinear::prepare(backend,{h,4*h});
    auto output=NativeLinear::prepare(backend,{h,64});
    auto ops=NativeQwenOps::prepare(backend,{h,g.block.heads,g.block.head_dim});
    auto step_ops=NativeStepOps::prepare(backend);
    std::optional<NativeLoRA> lora;
    if (requirements.max_rank) lora=NativeLoRA::prepare(backend,
        {std::max({h,g.block.ffn_width,256U}),std::max(4*h,g.block.ffn_width),requirements.max_rank},
        execution.lora_kernel);
    auto block=NativeBlock::prepare_with_shared_ops(backend,g.block,square,ops,lora,requirements.ffn);
    return NativeDenoiser(std::make_shared<Impl>(Impl{&backend,g,std::move(w),std::move(source),
        std::move(square),std::move(image_input),std::move(time_input),std::move(modulation),
        std::move(output),std::move(ops),std::move(step_ops),std::move(block),std::move(execution),
        std::move(lora),requirements}));
}
DenoiserGeometry NativeDenoiser::geometry() const noexcept { return impl_->geometry; }
std::string_view NativeDenoiser::lora_precision_recipe() const noexcept {
    return impl_->lora ? impl_->lora->actual_precision_recipe() : std::string_view{};
}

struct DenoiserRun::Impl {
    std::shared_ptr<NativeDenoiser::Impl> model;
    DenoiserInput input;
    SequenceGeometry sequence;
    uint32_t target=0, prefix=0, full=0;
    DenoiserMemory memory;
    DenoiserState state=DenoiserState::Prepared;
    std::string retirement_error;
    // A failed step must not discard the only observable completion receipt.
    // Subsequent steps replace it only after the previous submission finished.
    std::optional<metal_graph::Ticket> last_submission;
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
    // One weight/activation bank and one controller slot per run, shared by all
    // steady layers. A slot is reused only after its complete physical join.
    std::optional<NativeHybridFfn> hybrid;
#endif
    ParallelFfnWorkspace parallel;
    DenoiserHybridMetrics hybrid_metrics;
    size_t completed=0;
    bool prefix_valid=false;
    MatrixView current;
    std::array<MatrixView,2> hidden, latent;
    std::array<MatrixView,4> expanded_modulation;
    MatrixView time_constants,time_first,time_silu,time_embedding,embedding_silu;
    MatrixView modulation,output_scale,normalized_output,noise;
    MatrixView normalized_text,text_first,text_gelu,text_context,cosine,sine;
    MatrixView lora_low,lora_gate,lora_up;
    std::vector<PrefixKv> bank;
    BlockWorkspace prefill_workspace,decode_workspace;
    BufferView invalid;
    mutable std::mutex mutex;

    Completion physical_status() const {
        auto result=last_submission ? last_submission->status() : Completion{true,true,false,0,0,0,{}};
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
        if (hybrid) {
            const auto status=hybrid->status();
            const bool active=status.generation || status.retired || status.running;
            const bool timed_out=status.evidence.handoff.timed_out ||
                (status.evidence.gpu && status.evidence.gpu->timed_out) ||
                (status.evidence.consumer && status.evidence.consumer->timed_out);
            result=detail::join_physical_domain(std::move(result),active,status.ok,
                status.all_work_physically_complete,timed_out,status.error);
            // Counts describe only the directly retained Metal ticket above.
            // ANE/producer coverage is in the physical conjunction, not fake
            // command-buffer counts or summed overlap timings.
            if (active) result.gpu_seconds=0;
        }
#endif
        if (state==DenoiserState::Retired) {
            result.ok=false;
            if (result.error.empty()) result.error=retirement_error;
        }
        return result;
    }
};

std::unique_ptr<DenoiserRun> NativeDenoiser::begin(DenoiserInput input) const {
    auto& model=*impl_; auto& backend=*model.backend;
    if (!backend.healthy() || input.identity.source_revision!=model.source_revision ||
        input.identity.conditioning_revision.empty() || !input.identity.generation)
        throw std::invalid_argument("native denoiser source/conditioning identity mismatch");
    detail::validate_adapter_identity(model.execution.adapter_revision,input.identity.adapter_revision);
    if (input.height<=0 || input.width<=0 ||
        uint64_t(input.height)*uint64_t(input.width)>model.geometry.maximum_rows ||
        input.text.rows>model.geometry.maximum_rows || input.references.size()>2)
        throw std::invalid_argument("native denoiser target/text/reference geometry outside bounded recipe");
    schedule(input.sigmas);
    detail::validate_schedule_variant(model.execution.schedule_variant,input.sigmas.size()-1);
    const auto h=model.geometry.block.hidden;
    const auto target=uint32_t(uint64_t(input.height)*uint64_t(input.width));
    if (model.execution.hybrid_ffn)
        detail::validate_hybrid_request(model.geometry,target,input.references.size(),input.sigmas.size()-1);
    shape(input.latents,target,64); shape(input.text,input.text.rows,h);
    std::vector<ReferenceGeometry> references;
    uint64_t total=uint64_t(target)+input.text.rows;
    for (const auto& reference:input.references) {
        const auto& g=reference.geometry;
        if (g.height<=0 || g.width<=0 || uint64_t(g.height)*uint64_t(g.width)>model.geometry.maximum_rows)
            throw std::invalid_argument("native denoiser reference extent outside row budget");
        const auto count=uint32_t(uint64_t(g.height)*uint64_t(g.width));
        shape(reference.latents,count,64); references.push_back(g); total+=count;
    }
    if (total>model.geometry.maximum_rows)
        throw std::invalid_argument("native denoiser combined conditioning exceeds row budget");
    if (model.lora && total>8192)
        throw std::invalid_argument("native LoRA request exceeds its bounded row geometry");
    auto sequence=make_sequence_geometry(int(input.text.rows),input.height,input.width,references);
    const auto full=uint32_t(sequence.positions.size()), prefix=uint32_t(sequence.prefix_length);
    auto rope=make_rope_tables(sequence);
    auto run=std::make_unique<DenoiserRun::Impl>();
    run->model=impl_; run->input=std::move(input); run->sequence=std::move(sequence);
    run->target=target; run->prefix=prefix; run->full=full; run->current=run->input.latents;
    auto allocate=[&](uint32_t rows,uint32_t columns,std::string_view label,
                      StorageMode mode=StorageMode::Private,ScalarType type=ScalarType::BF16) {
        return allocate_matrix(backend,run->memory,rows,columns,label,mode,type);
    };
    for (auto& value:run->hidden) value=allocate(full,h,"qwen-native-hidden");
    for (auto& value:run->latent) value=allocate(target,64,"qwen-native-latents",StorageMode::Shared);
    for (auto& value:run->expanded_modulation) value=allocate(full,h,"qwen-native-prefix-modulation");
    run->time_constants=allocate(2,256,"qwen-native-time-constants",StorageMode::Shared);
    run->time_first=allocate(2,h,"qwen-native-time-first");
    run->time_silu=allocate(2,h,"qwen-native-time-silu");
    run->time_embedding=allocate(2,h,"qwen-native-time-embedding");
    run->embedding_silu=allocate(2,h,"qwen-native-embedding-silu");
    run->modulation=allocate(2,4*h,"qwen-native-modulation");
    run->output_scale=allocate(1,h,"qwen-native-output-scale");
    run->normalized_output=allocate(target,h,"qwen-native-output-normalized");
    run->noise=allocate(target,64,"qwen-native-noise",StorageMode::Shared);
    const auto text_rows=run->input.text.rows;
    run->normalized_text=allocate(text_rows,h,"qwen-native-text-normalized");
    run->text_first=allocate(text_rows,h,"qwen-native-text-first");
    run->text_gelu=allocate(text_rows,h,"qwen-native-text-gelu");
    run->text_context=allocate(text_rows,h,"qwen-native-text-context");
    run->cosine=allocate(full,64,"qwen-native-rope-cosine",StorageMode::Shared,ScalarType::FP32);
    run->sine=allocate(full,64,"qwen-native-rope-sine",StorageMode::Shared,ScalarType::FP32);
    std::memcpy(run->cosine.storage.contents(),rope.cosine.data(),bytes(full,64,ScalarType::FP32));
    std::memcpy(run->sine.storage.contents(),rope.sine.data(),bytes(full,64,ScalarType::FP32));
    run->memory.block_arena_bytes=std::max(
        NativeBlock::workspace_bytes(model.geometry.block,full,prefix,BlockPhase::PrefixBuild),
        NativeBlock::workspace_bytes(model.geometry.block,target,prefix,BlockPhase::SteadyDecode));
    auto arena=backend.allocate(run->memory.block_arena_bytes,"qwen-native-shared-block-arena",StorageMode::Private);
    add_memory(run->memory,run->memory.block_arena_bytes);
    run->prefill_workspace=model.block.bind_workspace(arena,full,prefix,BlockPhase::PrefixBuild);
    run->decode_workspace=model.block.bind_workspace(arena,target,prefix,BlockPhase::SteadyDecode);
    if (model.lora) {
        const auto before=run->memory.request_bytes;
        run->lora_low=allocate(std::max(full,2U),model.lora_requirements.max_rank,
            "qwen-native-lora-low",StorageMode::Private,ScalarType::FP32);
        if (model.lora_requirements.ffn) {
            run->lora_gate=allocate(full,model.geometry.block.ffn_width,"qwen-native-lora-gate");
            run->lora_up=allocate(full,model.geometry.block.ffn_width,"qwen-native-lora-up");
        }
        for (auto* workspace:{&run->prefill_workspace,&run->decode_workspace}) {
            workspace->lora_low=matrix_rows(run->lora_low,0,workspace->rows);
            if (model.lora_requirements.ffn) {
                workspace->lora_gate=matrix_rows(run->lora_gate,0,workspace->rows);
                workspace->lora_up=matrix_rows(run->lora_up,0,workspace->rows);
            }
        }
        run->memory.lora_scratch_bytes=run->memory.request_bytes-before;
    }
    run->bank.reserve(model.geometry.layers);
    for (uint32_t layer=0;layer<model.geometry.layers;++layer) {
        auto key=allocate(prefix,h,"qwen-native-prefix-key");
        auto value=allocate(prefix,h,"qwen-native-prefix-value");
        const auto heads=model.geometry.block.heads, dimension=model.geometry.block.head_dim;
        auto head=[&](const MatrixView& m) {
            return HeadView{m.storage,heads,prefix,dimension,size_t(dimension)*2,size_t(prefix)*dimension*2};
        };
        run->bank.push_back({head(key),head(value)});
        run->memory.prefix_bytes+=2*bytes(prefix,h);
    }
    auto flag=backend.allocate(sizeof(uint32_t),"qwen-native-invalid",StorageMode::Shared);
    run->invalid=flag.view(0,sizeof(uint32_t),ScalarType::UInt32); add_memory(run->memory,sizeof(uint32_t));
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
    if (model.execution.hybrid_ffn) {
        run->parallel.gpu_hidden=matrix_columns(run->decode_workspace.ffn_hidden,0,model.geometry.block.ffn_width/2);
        run->parallel.gpu_partial=allocate(target,h,"qwen-native-hybrid-gpu-partial",StorageMode::Private,ScalarType::FP32);
        run->parallel.ane_partial=allocate(target,h,"qwen-native-hybrid-ane-partial",StorageMode::Private,ScalarType::FP32);
        NativeHybridFfnOptions options;
        options.rows=target; options.hidden=h; options.width=model.geometry.block.ffn_width;
        options.ane_channels=options.width/2; options.program_cache=model.execution.ane_program_cache;
        options.surface_allocation={model.execution.allocation.reserve,model.execution.allocation.allocated};
        options.register_weight_staging=model.execution.register_weight_staging;
        if (model.execution.cache_weight_scales) {
            options.weight_scale_cache_entries=model.geometry.layers;
            options.weight_scale_source_revision=model.source_revision;
            options.weight_scale_source_generation=run->input.identity.generation;
        }
        run->hybrid.emplace(NativeHybridFfn::prepare(backend,std::move(options)));
        run->memory.hybrid_partial_bytes=2*bytes(target,h,ScalarType::FP32);
        run->memory.hybrid_surface_bytes=run->hybrid->surface_bytes();
        run->hybrid_metrics.enabled=true;
        run->hybrid_metrics.weight_staging=model.execution.register_weight_staging ? "register" : "generic";
        run->hybrid_metrics.weight_scale_cache_entries=run->hybrid->weight_scale_cache_capacity();
        run->hybrid_metrics.weight_scale_cache_enabled=run->hybrid_metrics.weight_scale_cache_entries!=0;
        run->hybrid_metrics.weight_scale_cache_payload_bytes=run->hybrid->weight_scale_cache_payload_bytes();
        const auto expected_cache_entries=model.execution.cache_weight_scales ? model.geometry.layers : 0U;
        const size_t expected_cache_payload=size_t(expected_cache_entries)*
            (size_t(model.geometry.block.ffn_width)+h)*sizeof(uint16_t);
        if (run->hybrid_metrics.weight_scale_cache_entries!=expected_cache_entries ||
            run->hybrid_metrics.weight_scale_cache_payload_bytes!=expected_cache_payload)
            throw std::runtime_error("native denoiser scale-cache allocation receipt differs from its phase plan");
        run->hybrid_metrics.partial_bytes=run->memory.hybrid_partial_bytes;
        run->hybrid_metrics.surface_bytes=run->memory.hybrid_surface_bytes;
    }
#endif
    return std::unique_ptr<DenoiserRun>(new DenoiserRun(std::move(run)));
}
DenoiserRun::DenoiserRun(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
DenoiserRun::~DenoiserRun() {
    // A caller normally drains before releasing the run/backend. On an unknown
    // or still-live completion retain the entire source/workspace transaction,
    // not merely the latest GPU ticket while ANE may still read its inputs.
    try {
        if (impl_ && !impl_->physical_status().physically_complete) (void)impl_.release();
    } catch (...) { (void)impl_.release(); }
}

metal_graph::Completion DenoiserRun::step(std::chrono::milliseconds timeout) {
    std::lock_guard lock(impl_->mutex);
    auto& r=*impl_; auto& m=*r.model;
    if (r.state==DenoiserState::Complete || r.state==DenoiserState::Retired ||
        r.completed+1>=r.input.sigmas.size() || timeout.count()<=0)
        throw std::logic_error("native denoiser step is unavailable or deadline invalid");
    const auto end=detail::step_deadline(timeout,Clock::now());
    const auto remaining=[&] {
        const auto value=detail::step_remaining(end,Clock::now());
        if (!value.count()) throw std::runtime_error("native denoiser step deadline exceeded");
        return value;
    };
    r.state=DenoiserState::Running;
    try {
        if (!m.backend->healthy() || (r.completed && !r.prefix_valid) ||
            !r.physical_status().physically_complete)
            throw std::runtime_error("native denoiser backend retired or prefix transaction incomplete");
        const auto time=timestep_input(r.input.sigmas[r.completed]);
        std::memcpy(r.time_constants.storage.contents(),time.data(),sizeof(time));
        const bool prefill=r.completed==0;
        const auto rows=prefill?r.full:r.target;
        std::array<MatrixView,2> hidden{matrix_rows(r.hidden[0],0,rows),matrix_rows(r.hidden[1],0,rows)};
        CommandGraph graph;
        m.step_ops.encode_reset_invalid_flag(graph,r.invalid);
        linear(graph,m.time_input,r.time_constants,m.weights.time_input,r.time_first,m.lora,r.lora_low);
        m.ops.encode_silu(graph,r.time_first,r.time_silu);
        linear(graph,m.square,r.time_silu,m.weights.time_output,r.time_embedding,m.lora,r.lora_low);
        m.ops.encode_silu(graph,r.time_embedding,r.embedding_silu);
        linear(graph,m.modulation,r.embedding_silu,m.weights.modulation,r.modulation,m.lora,r.lora_low);
        std::array<MatrixView,4> mods;
        if (prefill) {
            m.ops.encode_expand_modulation(graph,r.modulation,r.expanded_modulation,r.prefix);
            mods=r.expanded_modulation;
            m.ops.encode_condition_norm(graph,r.input.text,m.weights.text_norm,r.normalized_text);
            linear(graph,m.square,r.normalized_text,m.weights.text_input,r.text_first,m.lora,r.lora_low);
            m.ops.encode_gelu(graph,r.text_first,r.text_gelu);
            linear(graph,m.square,r.text_gelu,m.weights.text_output,r.text_context,m.lora,r.lora_low);
            for (const auto& segment:r.sequence.segments) {
                const auto count=uint32_t(segment.end-segment.start);
                auto destination=matrix_rows(hidden[0],uint32_t(segment.start),count);
                if (segment.causal) {
                    m.step_ops.encode_copy(graph,matrix_rows(r.text_context,uint32_t(segment.text_start),count),destination);
                } else {
                    const auto& source=segment.image_index<int(r.input.references.size())?
                        r.input.references[size_t(segment.image_index)].latents:r.current;
                    linear(graph,m.image_input,source,m.weights.image_input,destination,m.lora,r.lora_low);
                }
            }
        } else {
            const auto target_modulation=matrix_rows(r.modulation,0,1);
            for (uint32_t i=0;i<4;++i) mods[i]=matrix_columns(target_modulation,i*m.geometry.block.hidden,m.geometry.block.hidden);
            linear(graph,m.image_input,r.current,m.weights.image_input,hidden[0],m.lora,r.lora_low);
        }
        const auto cosine=prefill?r.cosine:matrix_rows(r.cosine,r.prefix,r.target);
        const auto sine=prefill?r.sine:matrix_rows(r.sine,r.prefix,r.target);
        uint32_t next_hidden=0;
        for (uint32_t layer=0;layer<m.geometry.layers;++layer) {
            BlockBindings binding;
            binding.input=hidden[next_hidden]; binding.output=hidden[1-next_hidden];
            binding.modulation=mods; binding.cosine=cosine; binding.sine=sine;
            binding.weights=m.weights.blocks[layer]; binding.prefix_tokens=r.prefix;
            binding.prefix=r.bank[layer]; binding.phase=prefill?BlockPhase::PrefixBuild:BlockPhase::SteadyDecode;
            if (prefill) binding.segments=r.sequence.segments;
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
            if (r.hybrid && !prefill) {
                auto block=m.block.plan_parallel(binding,r.decode_workspace,r.hybrid->gpu_complement(),r.parallel);
                // The first producer also owns this step's time/modulation and
                // image projection. Later layers start from the preceding
                // consumer's physically completed residual output.
                graph.append_graph(std::move(block.before_ffn));
                NativeHybridFfnPlan plan;
                plan.before_ffn=std::move(graph);
                plan.gpu_ffn=std::move(block.gpu_ffn);
                plan.after_join=std::move(block.after_join);
                plan.input=block.ane_input;
                plan.gate=binding.weights.gate; plan.up=binding.weights.up; plan.down=binding.weights.down;
                plan.gpu_partial=r.parallel.gpu_partial; plan.ane_partial=r.parallel.ane_partial;
                plan.published_output=binding.output;
                if (m.execution.cache_weight_scales) plan.weight_scale_cache_slot=layer;
                (void)remaining();
                const auto result=r.hybrid->execute_until(std::move(plan),end);
                if (!result.completion.ok || !result.completion.all_work_physically_complete ||
                    result.completion.retired)
                    throw std::runtime_error("native denoiser hybrid FFN failed: "+result.completion.error);
                auto& metrics=r.hybrid_metrics;
                if (m.execution.cache_weight_scales ?
                    (result.weight_scale_cache_hit==result.weight_scale_cache_miss) :
                    (result.weight_scale_cache_hit || result.weight_scale_cache_miss))
                    throw std::runtime_error("native denoiser scale-cache receipt does not match the executed route");
                ++metrics.successful_ffns;
                metrics.weight_scale_cache_hits+=result.weight_scale_cache_hit;
                metrics.weight_scale_cache_misses+=result.weight_scale_cache_miss;
                metrics.producer_seconds+=result.producer_seconds;
                metrics.submit_ane_seconds+=result.submit_ane_seconds;
                metrics.branch_join_seconds+=result.branch_join_seconds;
                metrics.consumer_seconds+=result.consumer_seconds;
                metrics.total_seconds+=result.total_seconds;
                if (result.producer_gpu_span_seconds>0 && result.gpu_complement_span_seconds>0) {
                    metrics.producer_gpu_span_seconds+=result.producer_gpu_span_seconds;
                    metrics.gpu_complement_span_seconds+=result.gpu_complement_span_seconds;
                    ++metrics.device_timed_ffns;
                }
                (void)remaining();
                graph=CommandGraph{};
            } else
#endif
                m.block.encode(graph,binding,prefill?r.prefill_workspace:r.decode_workspace);
            next_hidden=1-next_hidden;
            if (prefill) {
                m.step_ops.encode_finite(graph,flatten(r.bank[layer].key),r.invalid);
                m.step_ops.encode_finite(graph,flatten(r.bank[layer].value),r.invalid);
            }
        }
        const auto target_hidden=prefill?matrix_rows(hidden[next_hidden],r.prefix,r.target):hidden[next_hidden];
        linear(graph,m.square,matrix_rows(r.embedding_silu,0,1),m.weights.output_scale,r.output_scale,m.lora,r.lora_low);
        m.ops.encode_norm_modulation(graph,target_hidden,r.output_scale,r.normalized_output);
        linear(graph,m.output,r.normalized_output,m.weights.output,r.noise,m.lora,r.lora_low);
        const auto destination=r.latent[r.completed%2];
        m.step_ops.encode_euler_bf16(graph,r.current,r.noise,
            r.input.sigmas[r.completed+1]-r.input.sigmas[r.completed],destination);
        m.step_ops.encode_finite(graph,r.noise,r.invalid);
        m.step_ops.encode_finite(graph,destination,r.invalid);
        (void)remaining();
        r.last_submission.emplace(m.backend->submit(std::move(graph)));
        auto& ticket=*r.last_submission;
        const auto result=ticket.wait(remaining());
        if (!result.ok || !result.physically_complete || result.timed_out || Clock::now()>=end)
            throw std::runtime_error("native denoiser submission failed or retired: "+result.error);
        NativeStepOps::require_finite(ticket,r.invalid);
        const auto complete=r.physical_status();
        if (!complete.ok || !complete.physically_complete || complete.timed_out)
            throw std::runtime_error("native denoiser complete physical domain rejected publication: "+complete.error);
        if (r.hybrid_metrics.enabled &&
            (r.hybrid_metrics.successful_ffns!=uint64_t(m.geometry.layers)*r.completed ||
             r.hybrid_metrics.completed_hybrid_steps!=(r.completed ? r.completed-1 : 0)))
            throw std::runtime_error("native denoiser hybrid layer/step receipt mismatch");
        // Atomic host publication after the entire graph, not per-block.
        r.current=destination; r.prefix_valid=true; ++r.completed;
        if (!prefill && r.hybrid_metrics.enabled) ++r.hybrid_metrics.completed_hybrid_steps;
        r.state=r.completed+1==r.input.sigmas.size()?DenoiserState::Complete:DenoiserState::Prepared;
        return complete;
    } catch (const std::exception& error) {
        r.retirement_error=error.what();
        r.prefix_valid=false; r.state=DenoiserState::Retired;
        // Submitted graphs/tickets retain every view until physical completion.
        // No timeout signalling or backing reuse is attempted here.
        throw;
    } catch (...) {
        r.retirement_error="native denoiser unknown execution failure";
        r.prefix_valid=false; r.state=DenoiserState::Retired;
        throw;
    }
}
metal_graph::Completion DenoiserRun::physical_status() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->physical_status();
}
metal_graph::Completion DenoiserRun::drain(std::chrono::milliseconds timeout) {
    if (timeout.count()<0)
        throw std::invalid_argument("native denoiser drain timeout must be nonnegative");
    std::lock_guard lock(impl_->mutex);
    auto& r=*impl_;
    // Ticket::wait(0) is a real deadline and would poison an unfinished ticket.
    // A zero-time drain is deliberately observational instead.
    if (!timeout.count()) return r.physical_status();
    const auto end=detail::step_deadline(timeout,Clock::now());
    if (r.last_submission && !r.last_submission->status().physically_complete) {
        const auto left=detail::step_remaining(end,Clock::now());
        if (left.count()) (void)r.last_submission->wait(left);
    }
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
    if (r.hybrid) (void)r.hybrid->drain(detail::step_remaining(end,Clock::now()));
#endif
    auto completion=r.physical_status();
    if (!completion.ok || !completion.physically_complete || completion.timed_out) {
        r.prefix_valid=false;
        r.state=DenoiserState::Retired;
        if (r.retirement_error.empty()) r.retirement_error=completion.error.empty() ?
            "native denoiser physical drain incomplete" : completion.error;
    }
    // Do not run finite checks, publish current/banks or change Retired back to
    // Prepared/Complete. A physically drained failure is still a failed run.
    return completion;
}
MatrixView DenoiserRun::final_latents() const {
    std::lock_guard lock(impl_->mutex);
    if (impl_->state!=DenoiserState::Complete)
        throw std::logic_error("native denoiser result requires every scheduled step to finish");
    return impl_->current;
}
DenoiserState DenoiserRun::state() const { std::lock_guard lock(impl_->mutex); return impl_->state; }
size_t DenoiserRun::completed_steps() const { std::lock_guard lock(impl_->mutex); return impl_->completed; }
DenoiserMemory DenoiserRun::memory() const noexcept { return impl_->memory; }
DenoiserHybridMetrics DenoiserRun::hybrid_metrics() const {
    std::lock_guard lock(impl_->mutex); return impl_->hybrid_metrics;
}

} // namespace tc::qwen21::native_graph
