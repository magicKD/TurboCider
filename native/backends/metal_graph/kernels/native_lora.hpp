#pragma once

#include "../backend.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <string>

namespace tc::metal_graph::native_lora {

inline constexpr std::string_view precision_recipe =
    "runtime-lora-tiled8x32-fp32-fma-low-fp32-scaled-ordered-bf16-bias-last-v1";
inline constexpr std::string_view sg_mma_precision_recipe =
    "runtime-lora-fp32-sgmma8x8-aligned-r128-scalar-tail-ordered-bf16-bias-last-v2";
enum class KernelVariant : uint8_t { Scalar, SgMmaR128 };
inline constexpr size_t scalar_threadgroup_bytes = 5*1024;
inline constexpr size_t sg_mma_threadgroup_bytes = 6*1024;

// This identifies the complete selected bundle, including its scalar tail
// policy; it does not certify that any recorded graph was submitted or passed.
inline std::string_view recipe_for_variant(KernelVariant variant) {
    switch(variant) {
    case KernelVariant::Scalar:return precision_recipe;
    case KernelVariant::SgMmaR128:return sg_mma_precision_recipe;
    }
    throw std::invalid_argument("native LoRA unknown kernel variant");
}

struct Geometry {
    uint32_t input_columns = 12288, output_columns = 32768, max_rank = 128;
};
struct LowParameters {
    uint32_t rows, input_columns, rank, input_pitch, a_pitch, low_pitch, a_type, padding;
};
struct ApplyParameters {
    uint32_t rows, rank, output_count, first_output, low_pitch, b_pitch, output_pitch, b_type;
    float effective_scale;
};
struct ElementParameters {
    uint32_t rows, columns, first_pitch, second_pitch, output_pitch;
};
static_assert(sizeof(LowParameters)==32 && sizeof(ApplyParameters)==36 && sizeof(ElementParameters)==20);

// Shared CPU metadata contract. Storage is a backing identity, not ownership
// or a forged readiness permit; the .mm wrapper checks actual Metal devices.
struct Layout {
    uint32_t rows=0, columns=0;
    size_t row_stride_bytes=0, view_size_bytes=0, offset_bytes=0;
    ScalarType type=ScalarType::Bytes;
    uintptr_t storage=0;
};
struct TermLayout {
    Layout a, b;
    float effective_scale=0;
    uint32_t first_output=0, output_count=0;
};
struct CheckedTerm { LowParameters low; ApplyParameters apply; };
struct CheckedBindings {
    std::vector<CheckedTerm> terms;
    bool copy_base=false;
    ElementParameters copy{}, bias{};
};

inline void validate_geometry(Geometry g) {
    if(!g.input_columns || g.input_columns>12288 || !g.output_columns ||
       g.output_columns>32768 || !g.max_rank || g.max_rank>128)
        throw std::invalid_argument("native LoRA requires bounded K<=12288, N<=32768, rank<=128");
}
inline void validate_variant(Geometry g,KernelVariant variant) {
    validate_geometry(g);(void)recipe_for_variant(variant);
    if(variant==KernelVariant::SgMmaR128 && g.max_rank!=128)
        throw std::invalid_argument("native LoRA SgMmaR128 bundle requires rank128 capacity");
}
// Applied to each actual low/apply PSO after cold preparation. The independent
// copy/bias/SiLU PSOs still require their existing 256-thread capacity checks.
inline void validate_capability(KernelVariant variant,size_t threadgroup_bytes,
    uint32_t simd_width,size_t maximum_threads) {
    (void)recipe_for_variant(variant);
    const auto needed=variant==KernelVariant::SgMmaR128?sg_mma_threadgroup_bytes:scalar_threadgroup_bytes;
    if(threadgroup_bytes<needed || simd_width!=32 || maximum_threads<128)
        throw std::invalid_argument("native LoRA FP32 tile/SIMD capability unavailable");
}
// Production passes only fully validated bindings. Both phases of a term use
// the candidate together; every other bounded shape retains the scalar path.
inline bool sg_mma_eligible(const CheckedTerm& term) noexcept {
    const auto& l=term.low;const auto& a=term.apply;
    if(l.rank!=128 || a.rank!=128 || l.rows<8 || l.rows>8192 || l.rows%8 || a.rows!=l.rows ||
       !l.input_columns || l.input_columns>12288 || l.input_columns%32 ||
       !a.output_count || a.output_count>32768 || a.output_count%32 ||
       l.a_type>1 || a.b_type>1 || l.low_pitch!=a.low_pitch || l.low_pitch%4 ||
       l.low_pitch<128*4 || l.input_pitch%2 || size_t(l.input_pitch)<size_t(l.input_columns)*2 ||
       l.a_pitch%(l.a_type?4:2) || size_t(l.a_pitch)<size_t(l.input_columns)*(l.a_type?4:2) ||
       a.b_pitch%(a.b_type?4:2) || a.b_pitch<128*(a.b_type?4:2) ||
       a.output_pitch%2 || size_t(a.output_pitch)<(size_t(a.first_output)+a.output_count)*2 ||
       !std::isfinite(a.effective_scale))return false;
    return true;
}
inline size_t extent(Layout m) {
    if(!m.storage || (m.type!=ScalarType::BF16 && m.type!=ScalarType::FP32) ||
       !m.rows || !m.columns || m.rows>32768 || m.columns>32768)
        throw std::invalid_argument("native LoRA invalid storage/type/matrix bounds");
    const auto item=scalar_bytes(m.type), maximum=std::numeric_limits<size_t>::max();
    if(m.row_stride_bytes>UINT32_MAX || m.row_stride_bytes%item || m.offset_bytes%item ||
       m.view_size_bytes>maximum-m.offset_bytes || size_t(m.columns)>maximum/item)
        throw std::invalid_argument("native LoRA alignment/view/stride overflow");
    const auto tail=size_t(m.columns)*item;
    if(m.row_stride_bytes<tail || size_t(m.rows-1)>(maximum-tail)/m.row_stride_bytes)
        throw std::invalid_argument("native LoRA matrix pitch/extent overflow");
    const auto used=size_t(m.rows-1)*m.row_stride_bytes+tail;
    if(used>m.view_size_bytes)throw std::invalid_argument("native LoRA matrix exceeds retained view");
    return used;
}
inline bool overlaps(Layout a,Layout b) {
    const auto as=extent(a), bs=extent(b);
    return a.storage==b.storage && (a.offset_bytes<=b.offset_bytes ?
        b.offset_bytes-a.offset_bytes<as : a.offset_bytes-b.offset_bytes<bs);
}
inline bool exact_matrix(Layout a,Layout b) {
    return a.storage==b.storage && a.offset_bytes==b.offset_bytes && a.rows==b.rows &&
        a.columns==b.columns && a.row_stride_bytes==b.row_stride_bytes && a.type==b.type;
}
inline void reject_overlap(Layout output,Layout source) {
    if(overlaps(output,source))throw std::invalid_argument("native LoRA output overlaps an immutable source");
}
inline void reject_scratch_alias(Layout scratch,Layout other) {
    if(scratch.storage==other.storage)
        throw std::invalid_argument("native LoRA low workspace aliases source/output backing");
}
inline uint32_t validate_term(Geometry g,uint32_t input_columns,uint32_t output_columns,const TermLayout& term) {
    validate_geometry(g);(void)extent(term.a);(void)extent(term.b);
    const uint32_t rank=term.a.rows;
    if(!input_columns || input_columns>g.input_columns || !output_columns || output_columns>g.output_columns ||
       rank>g.max_rank || term.a.columns!=input_columns || !term.output_count ||
       term.first_output>=output_columns || term.output_count>output_columns-term.first_output ||
       term.b.rows!=term.output_count || term.b.columns!=rank || !std::isfinite(term.effective_scale))
        throw std::invalid_argument("native LoRA rank/range/factor/effective-scale mismatch");
    return rank;
}
inline CheckedBindings validate_bindings(Geometry g,Layout input,
    std::span<const TermLayout> terms,Layout base,Layout output,Layout low,
    std::optional<Layout> bias={}) {
    validate_geometry(g);(void)extent(input);(void)extent(base);(void)extent(output);
    if(input.type!=ScalarType::BF16 || base.type!=ScalarType::BF16 || output.type!=ScalarType::BF16 ||
       input.rows>8192 || input.columns>g.input_columns || base.columns>g.output_columns ||
       output.rows!=input.rows || base.rows!=input.rows || output.columns!=base.columns)
        throw std::invalid_argument("native LoRA input/base/output shape or BF16 boundary mismatch");
    CheckedBindings result;
    result.copy_base=!exact_matrix(base,output);
    if(terms.empty()) {
        if(result.copy_base || bias)
            throw std::invalid_argument("native LoRA empty terms must use the original no-bias in-place path");
        return result;
    }
    if(terms.size()>64)throw std::invalid_argument("native LoRA ordered term count exceeds bounded graph capacity");
    (void)extent(low);
    if(low.type!=ScalarType::FP32 || low.rows<input.rows || low.rows>8192 || low.columns>g.max_rank)
        throw std::invalid_argument("native LoRA low must be sufficient FP32 rank workspace");
    reject_overlap(output,input);
    if(result.copy_base)reject_overlap(output,base);
    for(const auto& m:{input,base,output})reject_scratch_alias(low,m);
    if(bias) {
        (void)extent(*bias);
        if(bias->type!=ScalarType::BF16 || bias->rows!=1 || bias->columns!=output.columns)
            throw std::invalid_argument("native LoRA bias must be BF16 [1,N]");
        reject_overlap(output,*bias);reject_scratch_alias(low,*bias);
    }
    result.copy={input.rows,output.columns,uint32_t(base.row_stride_bytes),0,uint32_t(output.row_stride_bytes)};
    if(bias)result.bias={input.rows,output.columns,uint32_t(output.row_stride_bytes),
        uint32_t(bias->row_stride_bytes),uint32_t(output.row_stride_bytes)};
    result.terms.reserve(terms.size());
    for(const auto& term:terms) {
        const uint32_t rank=validate_term(g,input.columns,output.columns,term);
        if(rank>low.columns)throw std::invalid_argument("native LoRA low workspace has too few rank columns");
        reject_overlap(output,term.a);reject_overlap(output,term.b);
        reject_scratch_alias(low,term.a);reject_scratch_alias(low,term.b);
        result.terms.push_back({{input.rows,input.columns,rank,uint32_t(input.row_stride_bytes),
            uint32_t(term.a.row_stride_bytes),uint32_t(low.row_stride_bytes),
            uint32_t(term.a.type==ScalarType::FP32),0},
            {input.rows,rank,term.output_count,term.first_output,uint32_t(low.row_stride_bytes),
            uint32_t(term.b.row_stride_bytes),uint32_t(output.row_stride_bytes),
            uint32_t(term.b.type==ScalarType::FP32),term.effective_scale}});
    }
    return result;
}

// A split low/apply pair keeps the full term's original eligibility decision.
// In particular an aligned low must not become SG when its B/output range
// would have selected a scalar tail in encode_ordered_corrections. Metadata
// alone carries neither retained device storage nor low-buffer readiness.
struct CheckedProjection {
    CheckedTerm term;
    KernelVariant selected_variant = KernelVariant::Scalar;
};
inline CheckedProjection validate_projection(Geometry geometry,KernelVariant variant,
    Layout input,const TermLayout& term,Layout output_inplace,Layout low) {
    validate_variant(geometry,variant);
    const auto checked=validate_bindings(geometry,input,std::span<const TermLayout>(&term,1),
        output_inplace,output_inplace,low);
    if(checked.copy_base || checked.terms.size()!=1)
        throw std::logic_error("native LoRA split projection must be one in-place term");
    const auto selected=variant==KernelVariant::SgMmaR128 && sg_mma_eligible(checked.terms.front())?
        KernelVariant::SgMmaR128:KernelVariant::Scalar;
    return {checked.terms.front(),selected};
}

inline ElementParameters validate_silu_up(Geometry g,Layout gate,Layout up,Layout hidden) {
    validate_geometry(g);(void)extent(gate);(void)extent(up);(void)extent(hidden);
    if(gate.type!=ScalarType::BF16 || up.type!=ScalarType::BF16 || hidden.type!=ScalarType::BF16 ||
       gate.rows!=up.rows || gate.rows!=hidden.rows || gate.columns!=up.columns ||
       gate.columns!=hidden.columns || gate.rows>8192 || gate.columns>g.output_columns)
        throw std::invalid_argument("native LoRA SiLU/up requires matching BF16 matrices");
    reject_overlap(hidden,gate);reject_overlap(hidden,up);
    return {gate.rows,gate.columns,uint32_t(gate.row_stride_bytes),
        uint32_t(up.row_stride_bytes),uint32_t(hidden.row_stride_bytes)};
}

// Compact-rank GEMMs use 8x32 output tiles and a shared 32-column K tile.
// Factors are decoded into FP32 shared storage; every accumulator and the low
// backing is FP32. No FP16 matrix multiply or BF16 low intermediate exists.
inline constexpr std::string_view source_body=R"metal(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct TCLoRALowParams {
    uint rows, input_columns, rank, input_pitch, a_pitch, low_pitch, a_type, padding;
};
struct TCLoRAApplyParams {
    uint rows, rank, output_count, first_output, low_pitch, b_pitch, output_pitch, b_type;
    float effective_scale;
};
struct TCLoRAElementParams { uint rows, columns, first_pitch, second_pitch, output_pitch; };
inline float tc_lora_factor(device const uchar* source,uint row,uint column,uint pitch,uint fp32) {
    device const uchar* p=source+ulong(row)*pitch;
    if(fp32)return ((device const float*)p)[column];
    return as_type<float>(uint(((device const ushort*)p)[column])<<16);
}

kernel void tc_native_lora_low(device const bfloat* input [[buffer(0)]],
    device const uchar* a [[buffer(1)]],device float* low [[buffer(2)]],
    constant TCLoRALowParams& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float x_tile[8*32], a_tile[32*32];
    const uint row_first=group.y*8, rank_first=group.x*32;
    const uint lane=tid&31, local_row=tid/32;
    float acc0=0.f, acc1=0.f;
    for(uint k=0;k<p.input_columns;k+=32) {
        for(uint i=tid;i<8*32;i+=128) {
            const uint row=row_first+i/32, column=k+i%32;
            x_tile[i]=(row<p.rows && column<p.input_columns)?
                float(input[ulong(row)*(p.input_pitch/2)+column]):0.f;
        }
        for(uint i=tid;i<32*32;i+=128) {
            const uint rank=rank_first+i/32, column=k+i%32;
            a_tile[i]=(rank<p.rank && column<p.input_columns)?
                tc_lora_factor(a,rank,column,p.a_pitch,p.a_type):0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<32;++i) {
            const float factor=a_tile[lane*32+i];
            acc0=metal::fma(x_tile[local_row*32+i],factor,acc0);
            acc1=metal::fma(x_tile[(local_row+4)*32+i],factor,acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint rank=rank_first+lane, row=row_first+local_row;
    if(rank<p.rank && row<p.rows)low[ulong(row)*(p.low_pitch/4)+rank]=acc0;
    if(rank<p.rank && row+4<p.rows)low[ulong(row+4)*(p.low_pitch/4)+rank]=acc1;
}

kernel void tc_native_lora_apply(device const float* low [[buffer(0)]],
    device const uchar* b [[buffer(1)]],device bfloat* output [[buffer(2)]],
    constant TCLoRAApplyParams& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float low_tile[8*32], b_tile[32*32];
    const uint row_first=group.y*8, output_first=group.x*32;
    const uint lane=tid&31, local_row=tid/32;
    float acc0=0.f, acc1=0.f;
    for(uint k=0;k<p.rank;k+=32) {
        for(uint i=tid;i<8*32;i+=128) {
            const uint row=row_first+i/32, rank=k+i%32;
            low_tile[i]=(row<p.rows && rank<p.rank)?low[ulong(row)*(p.low_pitch/4)+rank]:0.f;
        }
        for(uint i=tid;i<32*32;i+=128) {
            const uint column=output_first+i/32, rank=k+i%32;
            b_tile[i]=(column<p.output_count && rank<p.rank)?
                tc_lora_factor(b,column,rank,p.b_pitch,p.b_type):0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<32;++i) {
            const float factor=b_tile[lane*32+i];
            acc0=metal::fma(low_tile[local_row*32+i],factor,acc0);
            acc1=metal::fma(low_tile[(local_row+4)*32+i],factor,acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint column=output_first+lane, row=row_first+local_row;
    // Each thread owns distinct output elements. It reads the previous BF16
    // value only at those exact elements; successive terms are graph-ordered.
    if(column<p.output_count && row<p.rows) {
        const ulong index=ulong(row)*(p.output_pitch/2)+p.first_output+column;
        const float delta=acc0*p.effective_scale;
        output[index]=bfloat(float(output[index])+delta);
    }
    if(column<p.output_count && row+4<p.rows) {
        const ulong index=ulong(row+4)*(p.output_pitch/2)+p.first_output+column;
        const float delta=acc1*p.effective_scale;
        output[index]=bfloat(float(output[index])+delta);
    }
}

kernel void tc_native_lora_copy(device const ushort* base [[buffer(0)]],
    device ushort* output [[buffer(1)]],constant TCLoRAElementParams& p [[buffer(2)]],
    uint2 index [[thread_position_in_grid]]) {
    if(index.x<p.columns && index.y<p.rows)
        output[ulong(index.y)*(p.output_pitch/2)+index.x]=base[ulong(index.y)*(p.first_pitch/2)+index.x];
}
kernel void tc_native_lora_bias(device bfloat* output [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],constant TCLoRAElementParams& p [[buffer(2)]],
    uint2 index [[thread_position_in_grid]]) {
    if(index.x<p.columns && index.y<p.rows) {
        const ulong at=ulong(index.y)*(p.output_pitch/2)+index.x;
        output[at]=bfloat(float(output[at])+float(bias[index.x]));
    }
}
kernel void tc_native_lora_silu_up(device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],device bfloat* hidden [[buffer(2)]],
    constant TCLoRAElementParams& p [[buffer(3)]],uint2 index [[thread_position_in_grid]]) {
    if(index.x>=p.columns || index.y>=p.rows)return;
    const bfloat g=gate[ulong(index.y)*(p.first_pitch/2)+index.x];
    const bfloat u=up[ulong(index.y)*(p.second_pitch/2)+index.x];
    // Verbatim BF16 boundaries from the existing NativeFfn gate/up kernel.
    bfloat exponential=bfloat(metal::fast::exp(metal::abs(float(g))));
    bfloat denominator=bfloat(1.0f+float(exponential));
    bfloat y=bfloat(1.0f/float(denominator));
    bfloat sigmoid=g<0?y:bfloat(1.0f-float(y));
    bfloat activated=bfloat(float(g)*float(sigmoid));
    hidden[ulong(index.y)*(p.output_pitch/2)+index.x]=bfloat(float(activated)*float(u));
}
)metal";
inline std::string source() { return std::string(source_body); }

// Opt-in library only. The scalar library above is deliberately unchanged.
// Public 8x8 load/store primitives avoid assumptions about matrix lane layout.
// The extra 1 KiB result tile has no global allocation or factor/source copy.
inline constexpr std::string_view sg_mma_source_body=R"metal(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#pragma clang fp contract(off)

struct TCLoRALowParams {
    uint rows, input_columns, rank, input_pitch, a_pitch, low_pitch, a_type, padding;
};
struct TCLoRAApplyParams {
    uint rows, rank, output_count, first_output, low_pitch, b_pitch, output_pitch, b_type;
    float effective_scale;
};
inline float tc_lora_sg_factor(device const uchar* source,uint row,uint column,uint pitch,uint fp32) {
    device const uchar* p=source+ulong(row)*pitch;
    if(fp32)return ((device const float*)p)[column];
    return as_type<float>(uint(((device const ushort*)p)[column])<<16);
}

kernel void tc_native_lora_low_sgmma(device const bfloat* input [[buffer(0)]],
    device const uchar* a [[buffer(1)]],device float* low [[buffer(2)]],
    constant TCLoRALowParams& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float x_tile[8*32], a_tile[32*32], result_tile[8*32];
    const uint row_first=group.y*8, rank_first=group.x*32;
    const uint lane=tid&31, local_row=tid/32;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.f);
    for(uint k=0;k<p.input_columns;k+=32) {
        for(uint i=tid;i<8*32;i+=128) {
            const uint row=row_first+i/32, column=k+i%32;
            x_tile[i]=(row<p.rows && column<p.input_columns)?
                float(input[ulong(row)*(p.input_pitch/2)+column]):0.f;
        }
        for(uint i=tid;i<32*32;i+=128) {
            const uint rank=rank_first+i/32, column=k+i%32;
            a_tile[i]=(rank<p.rank && column<p.input_columns)?
                tc_lora_sg_factor(a,rank,column,p.a_pitch,p.a_type):0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint sub_k=0;sub_k<32;sub_k+=8) {
            simdgroup_float8x8 left,right;
            simdgroup_load(left,x_tile+sub_k,32,ulong2(0),false);
            // A is [rank,K]. Its shared tile is transposed logically, while
            // its produced backing and physical pitch remain untouched.
            simdgroup_load(right,a_tile+sg*8*32+sub_k,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc,left,right,acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,result_tile+sg*8,32,ulong2(0),false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rank=rank_first+lane, row=row_first+local_row;
    if(rank<p.rank && row<p.rows)low[ulong(row)*(p.low_pitch/4)+rank]=result_tile[local_row*32+lane];
    if(rank<p.rank && row+4<p.rows)low[ulong(row+4)*(p.low_pitch/4)+rank]=result_tile[(local_row+4)*32+lane];
}

kernel void tc_native_lora_apply_sgmma(device const float* low [[buffer(0)]],
    device const uchar* b [[buffer(1)]],device bfloat* output [[buffer(2)]],
    constant TCLoRAApplyParams& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float low_tile[8*32], b_tile[32*32], result_tile[8*32];
    const uint row_first=group.y*8, output_first=group.x*32;
    const uint lane=tid&31, local_row=tid/32;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.f);
    for(uint k=0;k<p.rank;k+=32) {
        for(uint i=tid;i<8*32;i+=128) {
            const uint row=row_first+i/32, rank=k+i%32;
            low_tile[i]=(row<p.rows && rank<p.rank)?low[ulong(row)*(p.low_pitch/4)+rank]:0.f;
        }
        for(uint i=tid;i<32*32;i+=128) {
            const uint column=output_first+i/32, rank=k+i%32;
            b_tile[i]=(column<p.output_count && rank<p.rank)?
                tc_lora_sg_factor(b,column,rank,p.b_pitch,p.b_type):0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint sub_k=0;sub_k<32;sub_k+=8) {
            simdgroup_float8x8 left,right;
            simdgroup_load(left,low_tile+sub_k,32,ulong2(0),false);
            simdgroup_load(right,b_tile+sg*8*32+sub_k,32,ulong2(0),true);
            simdgroup_multiply_accumulate(acc,left,right,acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,result_tile+sg*8,32,ulong2(0),false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint column=output_first+lane, row=row_first+local_row;
    if(column<p.output_count && row<p.rows) {
        const ulong index=ulong(row)*(p.output_pitch/2)+p.first_output+column;
        const float delta=result_tile[local_row*32+lane]*p.effective_scale;
        output[index]=bfloat(float(output[index])+delta);
    }
    if(column<p.output_count && row+4<p.rows) {
        const ulong index=ulong(row+4)*(p.output_pitch/2)+p.first_output+column;
        const float delta=result_tile[(local_row+4)*32+lane]*p.effective_scale;
        output[index]=bfloat(float(output[index])+delta);
    }
}
)metal";
inline std::string sg_mma_source() { return std::string(sg_mma_source_body); }

} // namespace tc::metal_graph::native_lora
