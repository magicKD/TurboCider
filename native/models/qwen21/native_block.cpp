#include "native_block.hpp"

#include <limits>

namespace tc::qwen21::native_graph {
namespace {
using namespace metal_graph;
constexpr size_t alignment = 256;

size_t aligned(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max()-(alignment-1))
        throw std::invalid_argument("native block workspace byte overflow");
    return (bytes+alignment-1)&~(alignment-1);
}
size_t matrix_bytes(uint32_t rows, uint32_t columns) {
    if (!rows || !columns || size_t(rows) > std::numeric_limits<size_t>::max()/columns/2)
        throw std::invalid_argument("native block empty matrix or byte overflow");
    return size_t(rows)*columns*2;
}
void validate_rows(uint32_t rows, uint32_t prefix, BlockPhase phase) {
    if (phase != BlockPhase::PrefixBuild && phase != BlockPhase::SteadyDecode)
        throw std::invalid_argument("native block unsupported phase");
    if (!rows || rows > 65536 || prefix > 65536 ||
        (phase == BlockPhase::PrefixBuild ? prefix >= rows : prefix > 65536-rows))
        throw std::invalid_argument("native block bounded row/prefix contract");
}
void same_shape(const MatrixView& value, uint32_t rows, uint32_t columns,
                ScalarType type = ScalarType::BF16) {
    dense_linear::validate_layout({value.rows,value.columns,value.row_stride_bytes,
        value.storage.size_bytes(),value.storage.offset_bytes(),value.storage.scalar_type()});
    if (!value.storage || value.rows != rows || value.columns != columns ||
        value.storage.scalar_type() != type)
        throw std::invalid_argument("native block logical matrix shape/dtype mismatch");
}
void same_head_shape(const HeadView& value, BlockGeometry g, uint32_t rows) {
    if (!value.storage || value.storage.scalar_type()!=ScalarType::BF16 ||
        value.heads!=g.heads || value.tokens!=rows || value.dimension!=g.head_dim)
        throw std::invalid_argument("native block head shape/dtype mismatch");
    (void)token_range(value,0,rows); // Includes physical stride and retained extent.
}
void distinct(const BufferView& a, const BufferView& b) {
    if (!a || !b || !a.buffer().native_handle() || !b.buffer().native_handle())
        throw std::invalid_argument("native block empty source/output view");
    if (a.buffer().native_handle()==b.buffer().native_handle() &&
        dense_linear::ranges_overlap(a.offset_bytes(),a.size_bytes(),b.offset_bytes(),b.size_bytes()))
        throw std::invalid_argument("native block cross-step writable storage alias");
}
void check_cross_step_aliases(const BlockBindings& b, const BlockWorkspace& w,
                             const ParallelFfnWorkspace* parallel = nullptr) {
    std::vector<BufferView> read{b.input.storage,b.cosine.storage,b.sine.storage,
        b.weights.query.weight.storage,b.weights.key.weight.storage,b.weights.value.weight.storage,
        b.weights.attention_output.weight.storage,b.weights.query_norm.storage,b.weights.key_norm.storage,
        b.weights.gate.storage,b.weights.up.storage,b.weights.down.storage};
    for (const auto& m:b.modulation) read.push_back(m.storage);
    auto terms=[&](const std::vector<OrderedLoRATerm>& values) {
        for (const auto& term:values) { read.push_back(term.a.storage); read.push_back(term.b.storage); }
    };
    for (const auto* p:{&b.weights.query,&b.weights.key,&b.weights.value,&b.weights.attention_output}) {
        if (p->bias) read.push_back(p->bias->storage);
        terms(p->lora);
    }
    terms(b.weights.gate_lora); terms(b.weights.up_lora); terms(b.weights.down_lora);
    std::vector<BufferView> write{b.output.storage,w.normalized_attention.storage,
        w.raw_query.storage,w.raw_key.storage,w.raw_value.storage,w.query.storage,w.key.storage,
        w.value.storage,w.attended.storage,w.row_attention.storage,w.projected_attention.storage,
        w.attention_residual.storage,w.normalized_ffn.storage,
        parallel ? parallel->gpu_hidden.storage : w.ffn_hidden.storage,w.ffn_output.storage};
    for (const auto* scratch:{&w.lora_low,&w.lora_gate,&w.lora_up})
        if (scratch->storage) write.push_back(scratch->storage);
    if (parallel) {
        // Treat the ANE restore destination as a write even though this class
        // does not encode it. It may be written while other branch resources
        // remain live; no cross-graph source/scratch alias is permitted.
        write.push_back(parallel->gpu_partial.storage);
        write.push_back(parallel->ane_partial.storage);
    }
    if (b.phase==BlockPhase::SteadyDecode && b.prefix_tokens) {
        read.push_back(b.prefix->key.storage); read.push_back(b.prefix->value.storage);
        write.push_back(w.joined_key.storage); write.push_back(w.joined_value.storage);
    } else if (b.phase==BlockPhase::PrefixBuild && b.prefix_tokens) {
        write.push_back(b.prefix->key.storage); write.push_back(b.prefix->value.storage);
    }
    // Initial schedule deliberately does not reuse scratch lifetimes. This
    // conservative check also catches a late weight/input overwritten early.
    for (size_t i=0;i<write.size();++i) {
        for (const auto& r:read) distinct(write[i],r);
        for (size_t j=0;j<i;++j) distinct(write[i],write[j]);
    }
}
bool ffn_adapted(const BlockWeights& w) {
    return !w.gate_lora.empty() || !w.up_lora.empty() || !w.down_lora.empty();
}
bool adapted(const BlockWeights& w) {
    return ffn_adapted(w) || !w.query.lora.empty() || !w.key.lora.empty() ||
        !w.value.lora.empty() || !w.attention_output.lora.empty();
}
void adapted_projection(CommandGraph& graph,const NativeLinear& base,
    const std::optional<NativeLoRA>& lora,const MatrixView& input,const Projection& p,
    const MatrixView& output,const MatrixView& low) {
    if (p.lora.empty()) { base.encode(graph,{input,p.weight,output,p.bias}); return; }
    if (!lora) throw std::invalid_argument("native block adapter was not prepared");
    base.encode(graph,{input,p.weight,output,{}});
    lora->encode_ordered_corrections(graph,input,p.lora,output,output,low,p.bias);
}
void validate_segments(const BlockBindings& b) {
    if (b.phase==BlockPhase::SteadyDecode) {
        if (!b.segments.empty()) throw std::invalid_argument("native steady block cannot bind prefix-build segments");
        return;
    }
    uint32_t next=0;
    bool boundary=b.prefix_tokens==0;
    for (const auto& s:b.segments) {
        if (s.start<0 || s.end<=s.start || uint32_t(s.start)!=next || uint32_t(s.end)>b.input.rows)
            throw std::invalid_argument("native prefix segments must partition all rows");
        next=uint32_t(s.end);
        if (next==b.prefix_tokens) boundary=true;
    }
    if (!boundary || next!=b.input.rows || b.segments.empty() ||
        uint32_t(b.segments.back().start)!=b.prefix_tokens || b.segments.back().causal)
        throw std::invalid_argument("native prefix-build requires last noncausal target segment");
}
void validate_block_bindings(BlockGeometry g,const BlockBindings& b,
                             const BlockWorkspace& w,const ParallelFfnWorkspace* parallel = nullptr) {
    validate_rows(b.input.rows,b.prefix_tokens,b.phase);
    if (w.rows!=b.input.rows || w.prefix_tokens!=b.prefix_tokens || w.phase!=b.phase)
        throw std::invalid_argument("native block workspace belongs to another row/phase plan");
    same_shape(b.input,w.rows,g.hidden); same_shape(b.output,w.rows,g.hidden);
    if (b.prefix_tokens) {
        if (!b.prefix) throw std::invalid_argument("native block prefix K/V missing");
        same_head_shape(b.prefix->key,g,b.prefix_tokens); same_head_shape(b.prefix->value,g,b.prefix_tokens);
        if (b.phase==BlockPhase::SteadyDecode) {
            same_head_shape(w.joined_key,g,w.rows+b.prefix_tokens);
            same_head_shape(w.joined_value,g,w.rows+b.prefix_tokens);
        }
    } else if (b.prefix) throw std::invalid_argument("native block zero prefix cannot bind a cache");
    validate_segments(b);
    check_cross_step_aliases(b,w,parallel);
}
} // namespace

void validate_geometry(BlockGeometry g) {
    if (!g.heads || g.heads>32 || g.head_dim!=128 || g.hidden!=g.heads*g.head_dim)
        throw std::invalid_argument("native Qwen block requires batch-one D128 equal Q/KV heads");
    metal_graph::dense_ffn::validate_geometry({g.hidden,g.ffn_width,g.ffn_width});
}

HeadView token_range(const HeadView& source,uint32_t start,uint32_t count) {
    const auto maximum=std::numeric_limits<size_t>::max();
    if (!source.storage || !source.heads || source.heads>32 || !source.tokens || source.dimension!=128 || !count ||
        start>=source.tokens || count>source.tokens-start ||
        source.storage.scalar_type()!=ScalarType::BF16 || source.storage.offset_bytes()%2 ||
        source.token_stride_bytes%2 || source.head_stride_bytes%2 ||
        size_t(source.dimension)>maximum/2 || source.token_stride_bytes<size_t(source.dimension)*2 ||
        size_t(source.tokens-1)>(maximum-size_t(source.dimension)*2)/source.token_stride_bytes)
        throw std::invalid_argument("native head token range has invalid extent/stride");
    const size_t head_extent=size_t(source.tokens-1)*source.token_stride_bytes+source.dimension*2;
    if (source.head_stride_bytes<head_extent ||
        size_t(source.heads-1)>(maximum-head_extent)/source.head_stride_bytes)
        throw std::invalid_argument("native head range overlapping heads or byte overflow");
    const size_t full_extent=size_t(source.heads-1)*source.head_stride_bytes+head_extent;
    if (full_extent>source.storage.size_bytes())
        throw std::invalid_argument("native head range exceeds retained source view");
    const size_t offset=size_t(start)*source.token_stride_bytes;
    const size_t extent=size_t(source.heads-1)*source.head_stride_bytes+
        size_t(count-1)*source.token_stride_bytes+source.dimension*2;
    return {source.storage.buffer().view(source.storage.offset_bytes()+offset,extent,ScalarType::BF16),
            source.heads,count,source.dimension,source.token_stride_bytes,source.head_stride_bytes};
}

NativeBlock::NativeBlock(BlockGeometry g,metal_graph::NativeLinear p,
    metal_graph::NativeAttention a,metal_graph::NativeQwenOps o,metal_graph::NativeFfn f,
    std::optional<NativeLoRA> lora,std::optional<NativeLinear> gate_up,std::optional<NativeLinear> down)
    :geometry_(g),projection_(std::move(p)),attention_(std::move(a)),ops_(std::move(o)),ffn_(std::move(f)),
     lora_(std::move(lora)),lora_gate_up_(std::move(gate_up)),lora_down_(std::move(down)) {}
NativeBlock NativeBlock::prepare(metal_graph::Backend& backend,BlockGeometry g) {
    validate_geometry(g);
    auto projection=metal_graph::NativeLinear::prepare(backend,{g.hidden,g.hidden});
    auto ops=metal_graph::NativeQwenOps::prepare(backend,{g.hidden,g.heads,g.head_dim});
    return prepare_with_shared_ops(backend,g,std::move(projection),std::move(ops));
}
NativeBlock NativeBlock::prepare_with_shared_ops(metal_graph::Backend& backend,BlockGeometry g,
    metal_graph::NativeLinear projection,metal_graph::NativeQwenOps ops,
    std::optional<NativeLoRA> lora,bool lora_ffn) {
    validate_geometry(g);
    const auto p=projection.geometry(); const auto o=ops.geometry();
    if (p.input_columns!=g.hidden || p.output_columns!=g.hidden ||
        o.hidden!=g.hidden || o.heads!=g.heads || o.head_dim!=g.head_dim)
        throw std::invalid_argument("native block shared operators have different geometry");
    auto attention=metal_graph::NativeAttention::prepare(backend,{g.heads,g.head_dim,1});
    auto ffn=metal_graph::NativeFfn::prepare(backend,{g.hidden,g.ffn_width,g.ffn_width});
    std::optional<NativeLinear> gate_up,down;
    if (lora_ffn) {
        if (!lora) throw std::invalid_argument("native FFN adapters require prepared LoRA operators");
        gate_up=NativeLinear::prepare(backend,{g.hidden,g.ffn_width});
        down=NativeLinear::prepare(backend,{g.ffn_width,g.hidden});
    }
    return NativeBlock(g,std::move(projection),std::move(attention),std::move(ops),std::move(ffn),
        std::move(lora),std::move(gate_up),std::move(down));
}

size_t NativeBlock::workspace_bytes(BlockGeometry g,uint32_t rows,uint32_t prefix,BlockPhase phase) {
    validate_geometry(g); validate_rows(rows,prefix,phase);
    const size_t ordinary=aligned(matrix_bytes(rows,g.hidden));
    const size_t hidden=aligned(matrix_bytes(rows,g.ffn_width));
    const size_t joined=phase==BlockPhase::SteadyDecode && prefix ?
        aligned(matrix_bytes(rows+prefix,g.hidden)):0;
    if (ordinary>(std::numeric_limits<size_t>::max()-hidden)/13 ||
        joined>(std::numeric_limits<size_t>::max()-13*ordinary-hidden)/2)
        throw std::invalid_argument("native block combined workspace overflow");
    return 13*ordinary+hidden+2*joined;
}
BlockWorkspace NativeBlock::allocate_workspace(metal_graph::Backend& backend,uint32_t rows,
    uint32_t prefix,BlockPhase phase,metal_graph::StorageMode storage) const {
    const size_t total=workspace_bytes(geometry_,rows,prefix,phase);
    auto arena=backend.allocate(total,"qwen-native-block-workspace",storage);
    return bind_workspace(arena,rows,prefix,phase);
}
BlockWorkspace NativeBlock::bind_workspace(const metal_graph::Buffer& arena,uint32_t rows,
    uint32_t prefix,BlockPhase phase) const {
    const size_t total=workspace_bytes(geometry_,rows,prefix,phase);
    if (!arena || arena.size_bytes()<total)
        throw std::invalid_argument("native block request arena is too small");
    size_t offset=0;
    auto matrix=[&](uint32_t columns) {
        const size_t bytes=matrix_bytes(rows,columns);
        MatrixView m{arena.view(offset,bytes,ScalarType::BF16),rows,columns,size_t(columns)*2};
        offset+=aligned(bytes); return m;
    };
    auto heads=[&](uint32_t tokens) {
        const size_t bytes=matrix_bytes(tokens,geometry_.hidden);
        HeadView h{arena.view(offset,bytes,ScalarType::BF16),geometry_.heads,tokens,geometry_.head_dim,
            size_t(geometry_.head_dim)*2,size_t(tokens)*geometry_.head_dim*2};
        offset+=aligned(bytes); return h;
    };
    BlockWorkspace w;
    w.rows=rows; w.prefix_tokens=prefix; w.phase=phase; w.reserved_bytes=total;
    w.normalized_attention=matrix(geometry_.hidden); w.raw_query=matrix(geometry_.hidden);
    w.raw_key=matrix(geometry_.hidden); w.raw_value=matrix(geometry_.hidden);
    w.query=heads(rows); w.key=heads(rows); w.value=heads(rows); w.attended=heads(rows);
    w.row_attention=matrix(geometry_.hidden); w.projected_attention=matrix(geometry_.hidden);
    w.attention_residual=matrix(geometry_.hidden); w.normalized_ffn=matrix(geometry_.hidden);
    w.ffn_output=matrix(geometry_.hidden); w.ffn_hidden=matrix(geometry_.ffn_width);
    if (phase==BlockPhase::SteadyDecode && prefix) {
        w.joined_key=heads(rows+prefix); w.joined_value=heads(rows+prefix);
    }
    if (offset!=total) throw std::logic_error("native block workspace layout accounting mismatch");
    return w;
}

void NativeBlock::encode(metal_graph::CommandGraph& graph,const BlockBindings& b,
                         const BlockWorkspace& w) const {
    validate_block_bindings(geometry_,b,w);
    metal_graph::CommandGraph draft;
    encode_before_ffn(draft,b,w);
    if (ffn_adapted(b.weights)) {
        if (!lora_ || !lora_gate_up_ || !lora_down_)
            throw std::invalid_argument("native FFN adapter geometry was not prepared");
        same_shape(w.lora_gate,w.rows,geometry_.ffn_width);
        same_shape(w.lora_up,w.rows,geometry_.ffn_width);
        lora_gate_up_->encode(draft,{w.normalized_ffn,b.weights.gate,w.lora_gate,{}});
        lora_gate_up_->encode(draft,{w.normalized_ffn,b.weights.up,w.lora_up,{}});
        if (!b.weights.gate_lora.empty()) lora_->encode_ordered_corrections(draft,w.normalized_ffn,
            b.weights.gate_lora,w.lora_gate,w.lora_gate,w.lora_low);
        if (!b.weights.up_lora.empty()) lora_->encode_ordered_corrections(draft,w.normalized_ffn,
            b.weights.up_lora,w.lora_up,w.lora_up,w.lora_low);
        lora_->encode_silu_up(draft,w.lora_gate,w.lora_up,w.ffn_hidden);
        lora_down_->encode(draft,{w.ffn_hidden,b.weights.down,w.ffn_output,{}});
        if (!b.weights.down_lora.empty()) lora_->encode_ordered_corrections(draft,w.ffn_hidden,
            b.weights.down_lora,w.ffn_output,w.ffn_output,w.lora_low);
    } else {
        ffn_.encode(draft,{w.normalized_ffn,b.weights.gate,b.weights.up,b.weights.down,w.ffn_hidden,w.ffn_output,0});
    }
    ops_.encode_gated_residual(draft,w.attention_residual,w.ffn_output,b.modulation[3],b.output);
    graph.append_graph(std::move(draft));
}

void NativeBlock::encode_before_ffn(metal_graph::CommandGraph& draft,const BlockBindings& b,
                                   const BlockWorkspace& w) const {
    ops_.encode_norm_modulation(draft,b.input,b.modulation[0],w.normalized_attention);
    adapted_projection(draft,projection_,lora_,w.normalized_attention,b.weights.query,w.raw_query,w.lora_low);
    adapted_projection(draft,projection_,lora_,w.normalized_attention,b.weights.key,w.raw_key,w.lora_low);
    adapted_projection(draft,projection_,lora_,w.normalized_attention,b.weights.value,w.raw_value,w.lora_low);
    ops_.encode_qk_norm_rope(draft,w.raw_query,w.raw_key,b.weights.query_norm,b.weights.key_norm,
        b.cosine,b.sine,w.query,w.key);
    ops_.encode_pack_heads(draft,w.raw_value,w.value);
    if (b.phase==BlockPhase::PrefixBuild) {
        if (b.prefix_tokens) {
            ops_.encode_copy_heads(draft,token_range(w.key,0,b.prefix_tokens),b.prefix->key);
            ops_.encode_copy_heads(draft,token_range(w.value,0,b.prefix_tokens),b.prefix->value);
        }
        for (const auto& s:b.segments) {
            const auto start=uint32_t(s.start),count=uint32_t(s.end-s.start),end=uint32_t(s.end);
            attention_.encode(draft,{token_range(w.query,start,count),token_range(w.key,0,end),
                token_range(w.value,0,end),token_range(w.attended,start,count),
                {s.causal,start,0.08838834764831844f}});
        }
    } else {
        HeadView k=w.key,v=w.value;
        if (b.prefix_tokens) {
            ops_.encode_copy_heads(draft,b.prefix->key,token_range(w.joined_key,0,b.prefix_tokens));
            ops_.encode_copy_heads(draft,b.prefix->value,token_range(w.joined_value,0,b.prefix_tokens));
            ops_.encode_copy_heads(draft,w.key,token_range(w.joined_key,b.prefix_tokens,w.rows));
            ops_.encode_copy_heads(draft,w.value,token_range(w.joined_value,b.prefix_tokens,w.rows));
            k=w.joined_key; v=w.joined_value;
        }
        attention_.encode(draft,{w.query,k,v,w.attended,{false,0,0.08838834764831844f}});
    }
    ops_.encode_unpack_heads(draft,w.attended,w.row_attention);
    adapted_projection(draft,projection_,lora_,w.row_attention,b.weights.attention_output,
        w.projected_attention,w.lora_low);
    ops_.encode_gated_residual(draft,b.input,w.projected_attention,b.modulation[1],w.attention_residual);
    ops_.encode_norm_modulation(draft,w.attention_residual,b.modulation[2],w.normalized_ffn);
}

void NativeBlock::validate_parallel_geometry(BlockGeometry geometry,dense_ffn::Geometry gpu,
    uint32_t rows,uint32_t prefix,BlockPhase phase) {
    validate_geometry(geometry); dense_ffn::validate_geometry(gpu);
    validate_rows(rows,prefix,phase);
    if (phase!=BlockPhase::SteadyDecode || gpu.hidden!=geometry.hidden ||
        gpu.width!=geometry.ffn_width || gpu.channels!=geometry.ffn_width/2 ||
        gpu.channels==geometry.ffn_width)
        throw std::invalid_argument("native parallel block requires steady decode and a prepared equal channel split");
}

ParallelBlockPlan NativeBlock::plan_parallel(const BlockBindings& b,const BlockWorkspace& w,
    const NativeFfn& gpu,const ParallelFfnWorkspace& partials) const {
    const auto g=geometry_;
    if (adapted(b.weights))
        throw std::invalid_argument("native hybrid block does not support adapter terms");
    validate_parallel_geometry(g,gpu.geometry(),b.input.rows,b.prefix_tokens,b.phase);
    if (!gpu.has_parallel_join())
        throw std::invalid_argument("native parallel block requires a cold-prepared FP32 join pipeline");
    same_shape(partials.gpu_hidden,b.input.rows,gpu.geometry().channels);
    same_shape(partials.gpu_partial,b.input.rows,g.hidden,ScalarType::FP32);
    same_shape(partials.ane_partial,b.input.rows,g.hidden,ScalarType::FP32);
    validate_block_bindings(g,b,w,&partials);
    ParallelBlockPlan plan;
    // Recording only: all three remain private until every operator, source,
    // partial and final residual binding has passed validation. No submission,
    // activation allocation, source copy or readiness token is manufactured.
    encode_before_ffn(plan.before_ffn,b,w);
    gpu.encode(plan.gpu_ffn,{w.normalized_ffn,b.weights.gate,b.weights.up,b.weights.down,
        partials.gpu_hidden,partials.gpu_partial,0});
    gpu.encode_join_partials(plan.after_join,partials.gpu_partial,partials.ane_partial,w.ffn_output);
    ops_.encode_gated_residual(plan.after_join,w.attention_residual,w.ffn_output,b.modulation[3],b.output);
    plan.ane_input=w.normalized_ffn;
    return plan;
}
} // namespace tc::qwen21::native_graph
