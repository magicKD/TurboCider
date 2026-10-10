#pragma once

#include "../native_attention.hpp"
#include "vendor/steel_attention_support.hpp"

#include <cmath>
#include <limits>

namespace tc::metal_graph::dense_attention {

inline constexpr size_t threadgroup_bytes = (32*136+128*24)*sizeof(uint16_t);
static_assert(threadgroup_bytes == 14848);

struct Parameters {
    uint32_t query_length = 0, key_length = 0, query_offset = 0, heads = 0;
    uint32_t query_stride = 0, key_stride = 0, value_stride = 0, output_stride = 0;
    uint64_t query_head_stride = 0, key_head_stride = 0, value_head_stride = 0, output_head_stride = 0;
    float scale = 0;
    uint32_t reserved0 = 0, reserved1 = 0, reserved2 = 0;
};
static_assert(sizeof(Parameters) == 80 && offsetof(Parameters,scale) == 64);

struct Layout {
    uint32_t heads = 0, tokens = 0, dimension = 128;
    size_t token_stride_bytes = 0, head_stride_bytes = 0;
    size_t view_size_bytes = 0, offset_bytes = 0;
    ScalarType type = ScalarType::Bytes;
};
struct CheckedBindings {
    Parameters parameters;
    size_t query_bytes = 0, key_bytes = 0, value_bytes = 0, output_bytes = 0;
};

inline void validate_geometry(AttentionGeometry geometry) {
    if (geometry.batch != 1 || geometry.heads == 0 || geometry.heads > 32 || ((geometry.dimension != 128 || geometry.type != ScalarType::BF16) &&
        (geometry.dimension != 80 || geometry.type != ScalarType::FP32)))
        throw std::invalid_argument("native SDPA requires batch1 heads1..32 and BF16 D128 or FP32 D80");
    if (geometry.kv_heads && (geometry.kv_heads>geometry.heads || geometry.heads%geometry.kv_heads))
        throw std::invalid_argument("native SDPA GQA needs a positive query/kv head divisor");
}

inline size_t validate_layout(Layout layout) {
    const auto maximum = std::numeric_limits<size_t>::max();
    const auto element=scalar_bytes(layout.type);
    const auto tail=size_t(layout.dimension)*element;
    // Steel BlockLoaderT and MMATile store use int row strides internally.
    // A uniform /32 ceiling keeps every tile-local product safely signed.
    if (((layout.type != ScalarType::BF16 || layout.dimension != 128) &&
        (layout.type != ScalarType::FP32 || layout.dimension != 80)) || layout.heads == 0 || layout.heads > 32 ||
        layout.tokens == 0 ||
        layout.tokens > uint32_t(std::numeric_limits<int>::max()) ||
        layout.token_stride_bytes < tail || layout.token_stride_bytes % element ||
        layout.token_stride_bytes/element > size_t(std::numeric_limits<int>::max()/32) ||
        layout.head_stride_bytes % element || layout.offset_bytes % element ||
        layout.view_size_bytes > maximum-layout.offset_bytes)
        throw std::invalid_argument("native SDPA invalid BF16 head-major layout, alignment or stride");
    if (size_t(layout.tokens-1) > (maximum-tail)/layout.token_stride_bytes)
        throw std::invalid_argument("native SDPA token extent overflows");
    const auto token_span = size_t(layout.tokens-1)*layout.token_stride_bytes+tail;
    if (layout.head_stride_bytes < token_span ||
        size_t(layout.heads-1) > (maximum-token_span)/layout.head_stride_bytes)
        throw std::invalid_argument("native SDPA overlapping head layout or head extent overflow");
    const auto extent = size_t(layout.heads-1)*layout.head_stride_bytes+token_span;
    if (extent > layout.view_size_bytes)
        throw std::invalid_argument("native SDPA head-major extent exceeds retained buffer view");
    return extent;
}

inline CheckedBindings validate_bindings(AttentionGeometry geometry,Layout q,Layout k,Layout v,Layout o,
                                         AttentionOptions options) {
    validate_geometry(geometry);
    if (!std::isfinite(options.scale) || options.scale <= 0 ||
        !std::isfinite(options.scale*1.44269504088896340736f))
        throw std::invalid_argument("native SDPA requires a finite positive attention scale");
    CheckedBindings checked;
    checked.query_bytes = validate_layout(q);
    checked.key_bytes = validate_layout(k);
    checked.value_bytes = validate_layout(v);
    checked.output_bytes = validate_layout(o);
    if (q.heads != geometry.heads || k.heads != (geometry.kv_heads?geometry.kv_heads:geometry.heads) ||
        v.heads != k.heads || o.heads != geometry.heads ||
        k.tokens != v.tokens || o.tokens != q.tokens ||
        q.dimension!=geometry.dimension || k.dimension!=geometry.dimension || v.dimension!=geometry.dimension || o.dimension!=geometry.dimension ||
        q.type!=geometry.type || k.type!=geometry.type || v.type!=geometry.type || o.type!=geometry.type)
        throw std::invalid_argument("native SDPA Q/K/V/output mismatch; GQA is not supported");
    // This first path serves self-attention chunks within concatenated keys.
    // The physical views do not shift with the logical causal query offset.
    if (options.causal && uint64_t(options.query_position_offset)+q.tokens > k.tokens)
        throw std::invalid_argument("native SDPA causal queries exceed the available logical key interval");
    auto& p = checked.parameters;
    p.query_length=q.tokens; p.key_length=k.tokens;
    p.query_offset=options.query_position_offset; p.heads=geometry.heads;
    const auto e=scalar_bytes(geometry.type);
    p.query_stride=uint32_t(q.token_stride_bytes/e); p.key_stride=uint32_t(k.token_stride_bytes/e);
    p.value_stride=uint32_t(v.token_stride_bytes/e); p.output_stride=uint32_t(o.token_stride_bytes/e);
    p.query_head_stride=q.head_stride_bytes/e; p.key_head_stride=k.head_stride_bytes/e;
    p.value_head_stride=v.head_stride_bytes/e; p.output_head_stride=o.head_stride_bytes/e;
    p.reserved0=geometry.kv_heads?geometry.heads/geometry.kv_heads:0;
    p.scale=options.scale;
    return checked;
}

inline bool ranges_overlap(size_t offset_a,size_t bytes_a,size_t offset_b,size_t bytes_b) {
    if (!bytes_a || !bytes_b) return false;
    return offset_a <= offset_b ? offset_b-offset_a < bytes_a : offset_a-offset_b < bytes_b;
}

// Copyright © 2024-25 Apple Inc. Derived from MLX Steel attention (MIT).
// TurboCider changes: native ABI, BF16 D128 only, dynamic safe tails, two
// explicit causal/noncausal entry points, no mask/sinks/spans/GQA and uint64
// logical endpoints. Register tiles, online-softmax rescaling and MMA retain
// the Steel algorithm. See THIRD_PARTY_NOTICES/mlx-steel-attention.md.
inline constexpr std::string_view source_body = R"TCSDPA(
using namespace mlx::steel;
struct TCNativeAttentionParams {
    uint query_length, key_length, query_offset, heads;
    uint query_stride, key_stride, value_stride, output_stride;
    ulong query_head_stride, key_head_stride, value_head_stride, output_head_stride;
    float scale;
    uint reserved0, reserved1, reserved2;
};
struct TCMax { template<typename T> METAL_FUNC static constexpr T apply(T x,T y) { return metal::max(x,y); } };
struct TCSum { template<typename T> METAL_FUNC static constexpr T apply(T x,T y) { return x+y; } };
struct TCMul { template<typename T> METAL_FUNC static constexpr T apply(T x,T y) { return x*y; } };
struct TCExpSub { template<typename T> METAL_FUNC static constexpr T apply(T x,T y) { return fast::exp2(x-y); } };
struct TCDiv { template<typename T> METAL_FUNC static constexpr T apply(T x,T y) { return x/y; } };

template<typename T, short BD, bool Causal>
METAL_FUNC void tc_native_sdpa_body(const device T* Q,const device T* K,
    const device T* V,device T* O,constant TCNativeAttentionParams& p,
    threadgroup T* Qs,threadgroup T* KV_smem,
    uint simd_lane_id,uint simd_group_id,uint2 tid) {
    constexpr short BQ=32,BK=16,FS=8,LDQ=BD+8,LDK=24,LDV=BD+8;
    constexpr int TQ=1,TK=2,TD=BD/8;
    const uint q_row=tid.x*BQ;
    Q += ulong(tid.y)*p.query_head_stride + ulong(q_row)*p.query_stride;
    K += ulong(tid.y/(p.reserved0?p.reserved0:1))*p.key_head_stride;
    V += ulong(tid.y/(p.reserved0?p.reserved0:1))*p.value_head_stride;
    O += ulong(tid.y)*p.output_head_stride + ulong(q_row)*p.output_stride;
    threadgroup T* Ks=KV_smem;
    threadgroup T* Vs=KV_smem;
    using QLoader=BlockLoaderT<T,BQ,BD,LDQ,1,1,128>;
    using KLoader=BlockLoaderT<T,BK,BD,1,LDK,0,128>;
    using VLoader=BlockLoaderT<T,BK,BD,LDV,1,0,128>;
    QLoader loader_q(Q,int(p.query_stride),Qs,ushort(simd_group_id),ushort(simd_lane_id));
    KLoader loader_k(K,int(p.key_stride),Ks,ushort(simd_group_id),ushort(simd_lane_id));
    VLoader loader_v(V,int(p.value_stride),Vs,ushort(simd_group_id),ushort(simd_lane_id));

    using Frag=BaseMMAFrag<float,FS,FS>;
    MMATile<float,TQ,1,Frag> Qtile;
    MMATile<float,1,TK,Frag> Ktile;
    MMATile<float,TQ,TK,Frag> Stile;
    MMATile<float,1,1,Frag> Vtile;
    MMATile<float,TQ,TD,Frag> Otile;
    Otile.clear();
    const short2 coord=Frag::get_coord(ushort(simd_lane_id));
    const short sm=coord.y,sn=coord.x,tm=FS*TQ*short(simd_group_id);
    const short qs_offset=(tm+sm)*LDQ+sn;
    const short ks_offset=sm*LDK+sn;
    const short vs_offset=sm*LDV+sn;
    const float scale=p.scale*1.44269504088896340736f;
    constexpr short row_count=decltype(Stile)::kRowsPerThread;
    constexpr float negative= -metal::numeric_limits<float>::max();
    float max_score[row_count],sum_score[row_count]={0};
    STEEL_PRAGMA_UNROLL
    for (short i=0;i<row_count;++i) max_score[i]=negative;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.query_length-q_row < BQ)
        loader_q.load_safe(short2(BD,short(p.query_length-q_row)));
    else loader_q.load_unsafe();

    const uint nk=uint((ulong(p.key_length)+BK-1)/BK);
    uint limit=nk,first_masked=nk;
    if constexpr (Causal) {
        const ulong qmax=ulong(q_row)+p.query_offset+BQ;
        limit=uint(metal::min(ulong(nk),(qmax+BK-1)/BK));
        first_masked=uint(metal::min(ulong(nk),(ulong(q_row)+p.query_offset)/BK));
    }
    for (uint kb=0;kb<limit;++kb) {
        const uint remaining=p.key_length-kb*BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (remaining < BK) loader_k.load_safe(short2(BD,short(remaining)));
        else loader_k.load_unsafe();
        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short dd=0;dd<TD;++dd) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.template load<T,1,1,LDQ,1>(&Qs[qs_offset+dd*FS]);
            Ktile.template load<T,1,1,LDK,1>(&Ks[ks_offset+dd*FS*LDK]);
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(Stile,Qtile,Ktile,Stile);
        }
        STEEL_PRAGMA_UNROLL
        for (short ii=0;ii<decltype(Stile)::kElemsPerTile;++ii) Stile.elems()[ii]*=scale;
        using ST=decltype(Stile);
        if (remaining < BK) {
            STEEL_PRAGMA_UNROLL
            for (short i=0;i<ST::kTileRows;++i) {
                STEEL_PRAGMA_UNROLL
                for (short j=0;j<ST::kTileCols;++j) {
                    const short col_pos=sn+j*ST::kFragCols;
                    STEEL_PRAGMA_UNROLL
                    for (short jj=0;jj<ST::MMAFrag_t::kElemCols;++jj)
                        if (uint(col_pos+jj)>=remaining) Stile.frag_at(i,j)[jj]=negative;
                }
            }
        }
        if constexpr (Causal) {
            if (kb>=first_masked) {
                STEEL_PRAGMA_UNROLL
                for (short i=0;i<ST::kTileRows;++i) {
                    const ulong row_pos=ulong(q_row)+p.query_offset+tm+sm+i*ST::kFragRows;
                    STEEL_PRAGMA_UNROLL
                    for (short j=0;j<ST::kTileCols;++j) {
                        const ulong col_pos=ulong(kb)*BK+sn+j*ST::kFragCols;
                        STEEL_PRAGMA_UNROLL
                        for (short jj=0;jj<ST::MMAFrag_t::kElemCols;++jj)
                            if (row_pos<col_pos+jj) Stile.frag_at(i,j)[jj]=negative;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (remaining < BK) loader_v.load_safe(short2(BD,short(remaining)));
        else loader_v.load_unsafe();
        float new_max[row_count],factor[row_count];
        STEEL_PRAGMA_UNROLL
        for (short i=0;i<row_count;++i) new_max[i]=max_score[i];
        Stile.template row_reduce<TCMax>(new_max);
        Stile.template row_bin_op<TCExpSub>(new_max);
        STEEL_PRAGMA_UNROLL
        for (short i=0;i<row_count;++i) {
            factor[i]=fast::exp2(max_score[i]-new_max[i]);
            max_score[i]=new_max[i];
        }
        float tile_sum[row_count]={0};
        Stile.template row_reduce<TCSum>(tile_sum);
        STEEL_PRAGMA_UNROLL
        for (short i=0;i<row_count;++i) sum_score[i]=sum_score[i]*factor[i]+tile_sum[i];
        Otile.template row_bin_op<TCMul>(factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short iq=0;iq<TQ;++iq) {
            STEEL_PRAGMA_UNROLL
            for (short id=0;id<TD;++id) {
                STEEL_PRAGMA_UNROLL
                for (short ik=0;ik<TK;++ik) {
                    simdgroup_barrier(mem_flags::mem_none);
                    Vtile.template load<T,1,1,LDV,1>(&Vs[vs_offset+ik*FS*LDV+id*FS]);
                    simdgroup_barrier(mem_flags::mem_none);
                    Frag::mma(Otile.frag_at(iq,id),Stile.frag_at(iq,ik),Vtile.frag_at(0,0),Otile.frag_at(iq,id));
                }
            }
        }
        loader_k.next(); loader_v.next();
    }
    Otile.template row_bin_op<TCDiv>(sum_score);
    threadgroup_barrier(mem_flags::mem_none);
    O += (tm+sm)*int(p.output_stride)+sn;
    if (p.query_length-q_row < BQ) {
        const auto dst_dims=short2(BD-sn,short(p.query_length-q_row)-(tm+sm));
        if (dst_dims.x<=0 || dst_dims.y<=0) return;
        Otile.template store_safe<T,1,1>(O,int(p.output_stride),dst_dims);
    } else Otile.template store<T,1,1>(O,int(p.output_stride));
}

[[kernel,max_total_threads_per_threadgroup(128)]] void tc_native_sdpa_d128(const device bfloat* q [[buffer(0)]],
    const device bfloat* k [[buffer(1)]],const device bfloat* v [[buffer(2)]],device bfloat* o [[buffer(3)]],
    constant TCNativeAttentionParams& p [[buffer(4)]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat qs[32*136],kv[128*24];
    tc_native_sdpa_body<bfloat,128,false>(q,k,v,o,p,qs,kv,lane,simd,group);
}
[[kernel,max_total_threads_per_threadgroup(128)]] void tc_native_sdpa_d128_causal(const device bfloat* q [[buffer(0)]],
    const device bfloat* k [[buffer(1)]],const device bfloat* v [[buffer(2)]],device bfloat* o [[buffer(3)]],
    constant TCNativeAttentionParams& p [[buffer(4)]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat qs[32*136],kv[128*24];
    tc_native_sdpa_body<bfloat,128,true>(q,k,v,o,p,qs,kv,lane,simd,group);
}

[[kernel,max_total_threads_per_threadgroup(128)]] void tc_native_sdpa_d80_f32(const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],device float* o [[buffer(3)]],
    constant TCNativeAttentionParams& p [[buffer(4)]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float qs[32*88],kv[80*24];
    tc_native_sdpa_body<float,80,false>(q,k,v,o,p,qs,kv,lane,simd,group);
}
[[kernel,max_total_threads_per_threadgroup(128)]] void tc_native_sdpa_d80_f32_causal(const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],device float* o [[buffer(3)]],
    constant TCNativeAttentionParams& p [[buffer(4)]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float qs[32*88],kv[80*24];
    tc_native_sdpa_body<float,80,true>(q,k,v,o,p,qs,kv,lane,simd,group);
}
)TCSDPA";

inline std::string source() {
    return "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
           "#define METAL_FUNC inline\nusing namespace metal;\n"+
           std::string(steel_vendor::support_source)+std::string(source_body);
}

} // namespace tc::metal_graph::dense_attention
