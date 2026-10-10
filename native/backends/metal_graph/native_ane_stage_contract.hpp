#pragma once

#include "backend.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace tc::metal_graph::native_ane_stage_contract {

struct Geometry {
    uint32_t rows = 1024, hidden = 4096, width = 12288, ane_channels = 6144;
    uint32_t first_channel() const { return width-ane_channels; }
    bool operator==(const Geometry&) const = default;
};
inline void validate_geometry(Geometry g) {
    // Fixed ANE buckets, not a variable-shape MIL promise. The row adapter
    // zero pads tails; the emitter has an independent 4224-row ceiling.
    const bool real = g.rows>=128 && g.rows<=4096 && g.rows%128==0 &&
        g.hidden==4096 && g.width==12288 &&
        (g.ane_channels==3072 || g.ane_channels==6144 || g.ane_channels==8192);
    const bool tiny = g.rows==32 && g.hidden==128 && g.width==1024 && g.ane_channels==512;
    if (!real && !tiny)
        throw std::invalid_argument("native ANE stages require a bounded aligned Qwen FFN bucket or generated tiny geometry");
}

// CPU-checkable metadata. The .mm binding adapter also checks the actual
// retained MTLBuffer device; storage is its native backing identity.
struct Layout {
    uint32_t rows = 0, columns = 0;
    size_t row_stride_bytes = 0, offset_bytes = 0, view_size_bytes = 0;
    ScalarType type = ScalarType::Bytes;
    uintptr_t storage = 0;
};
inline size_t extent(const Layout& m) {
    const auto item = scalar_bytes(m.type);
    if (!m.storage || !item || !m.rows || !m.columns ||
        m.row_stride_bytes > UINT32_MAX || m.offset_bytes % item ||
        m.row_stride_bytes % item || size_t(m.columns)>SIZE_MAX/item)
        throw std::invalid_argument("native ANE stage invalid type/extent/stride");
    const auto row = size_t(m.columns)*item;
    if (m.row_stride_bytes<row || size_t(m.rows-1)>(SIZE_MAX-row)/m.row_stride_bytes)
        throw std::invalid_argument("native ANE stage matrix pitch/extent overflow");
    const auto used = size_t(m.rows-1)*m.row_stride_bytes+row;
    if (used>m.view_size_bytes || m.offset_bytes>SIZE_MAX-m.view_size_bytes)
        throw std::invalid_argument("native ANE stage matrix exceeds retained view");
    return used;
}
inline void matrix(const Layout& m,uint32_t rows,uint32_t columns,ScalarType type) {
    (void)extent(m);
    if (m.rows!=rows || m.columns!=columns || m.type!=type)
        throw std::invalid_argument("native ANE stage source logical shape/dtype mismatch");
}
inline void surface(const Layout& m,uint32_t rows,uint32_t columns,ScalarType type) {
    matrix(m,rows,columns,type);
    const size_t pitch=(size_t(columns)*scalar_bytes(type)+63)/64*64;
    if (m.offset_bytes || m.row_stride_bytes!=pitch ||
        size_t(rows)>SIZE_MAX/pitch || size_t(rows)*pitch>m.view_size_bytes)
        throw std::invalid_argument("native ANE stage requires a full ABI-matching IOSurface");
}
inline void disjoint_storage(const Layout& read,const Layout& write) {
    if (read.storage==write.storage)
        throw std::invalid_argument("native ANE stage write aliases immutable source/backing");
}
template<size_t N> inline void distinct(const std::array<Layout,N>& outputs) {
    for (size_t i=0;i<N;++i)for(size_t j=i+1;j<N;++j)
        disjoint_storage(outputs[i],outputs[j]);
}
// Checks both recording orders: bank writes cannot destroy the A8 input;
// A8 writes cannot destroy checkpoint weights or any bank binding.
inline void cross_stage_aliases(const std::array<Layout,3>& weights,
    const std::array<Layout,6>& bank,const std::array<Layout,3>& activation) {
    for (const auto& output:bank) {
        disjoint_storage(output,activation[0]);
        disjoint_storage(output,activation[1]);
        disjoint_storage(output,activation[2]);
    }
    for (const auto& source:weights) {
        disjoint_storage(source,activation[1]);
        disjoint_storage(source,activation[2]);
    }
}

// Exact legacy POD layout; source assembler moves only MSL buffer attributes.
struct W8Parameters {
    uint32_t source_pitch, physical_cols, encoding, dtype, group_size;
    uint32_t meta_pitch, meta_dtype, offset_pitch, offset_dtype, has_offset;
    uint32_t row_begin, rows, column_begin, columns, block, code_pitch, scale_pitch, transpose;
    uint32_t source_aligned, seed_low, seed_high, basis, activation_group;
    uint32_t has_column_scale, inverse_column_scale;
    float norm;
};
struct RestoreParameters {
    uint32_t rows, cols, source_pitch, target_pitch, dtype, validate_only;
    float scale;
    uint32_t row_scale_pitch, scaled, second_scaled, signed_row_scale;
};
static_assert(sizeof(W8Parameters)==104 && offsetof(W8Parameters,norm)==100);
static_assert(sizeof(RestoreParameters)==44 && offsetof(RestoreParameters,scale)==24);

inline W8Parameters parameters(const Layout& source,const Layout& codes,
    const Layout& scales,uint32_t first_row,uint32_t rows,uint32_t first_column,
    uint32_t columns,uint32_t block,bool transpose) {
    W8Parameters p{};
    p.source_pitch=uint32_t(source.row_stride_bytes);p.physical_cols=source.columns;
    p.encoding=0;p.dtype=1;p.group_size=32;
    p.row_begin=first_row;p.rows=rows;p.column_begin=first_column;p.columns=columns;
    p.block=block;p.code_pitch=uint32_t(codes.row_stride_bytes);
    // tx is [1,M], but scales are indexed per logical token, not per surface row.
    p.scale_pitch=transpose?2U:uint32_t(scales.row_stride_bytes);
    p.transpose=uint32_t(transpose);p.source_aligned=1;
    p.seed_low=20260930;p.seed_high=0;p.basis=0;
    p.norm=1.f/std::sqrt(float(block));
    return p;
}
inline std::array<W8Parameters,3> weight_parameters(Geometry g,
    const std::array<Layout,3>& source,const std::array<Layout,6>& target) {
    validate_geometry(g);
    matrix(source[0],g.width,g.hidden,ScalarType::BF16);
    matrix(source[1],g.width,g.hidden,ScalarType::BF16);
    matrix(source[2],g.hidden,g.width,ScalarType::BF16);
    surface(target[0],g.ane_channels,g.hidden,ScalarType::Int8);
    surface(target[1],g.ane_channels,1,ScalarType::FP16);
    surface(target[2],g.ane_channels,g.hidden,ScalarType::Int8);
    surface(target[3],g.ane_channels,1,ScalarType::FP16);
    surface(target[4],g.hidden,g.ane_channels,ScalarType::Int8);
    surface(target[5],g.hidden,1,ScalarType::FP16);
    distinct(target);
    for(const auto& s:source)for(const auto& t:target)disjoint_storage(s,t);
    return {parameters(source[0],target[0],target[1],g.first_channel(),g.ane_channels,0,g.hidden,128,false),
        parameters(source[1],target[2],target[3],g.first_channel(),g.ane_channels,0,g.hidden,128,false),
        parameters(source[2],target[4],target[5],0,g.hidden,g.first_channel(),g.ane_channels,512,false)};
}
inline W8Parameters activation_parameters(Geometry g,const Layout& input,
    const Layout& codes,const Layout& scales) {
    validate_geometry(g);matrix(input,g.rows,g.hidden,ScalarType::BF16);
    surface(codes,g.hidden,g.rows,ScalarType::Int8);
    surface(scales,1,g.rows,ScalarType::FP16);
    disjoint_storage(input,codes);disjoint_storage(input,scales);disjoint_storage(codes,scales);
    return parameters(input,codes,scales,0,g.rows,0,g.hidden,128,true);
}
inline RestoreParameters restore_parameters(Geometry g,const Layout& y,
    const Layout& down_scales,const Layout& output) {
    validate_geometry(g);surface(y,g.hidden+1,g.rows,ScalarType::FP16);
    surface(down_scales,g.hidden,1,ScalarType::FP16);
    matrix(output,g.rows,g.hidden,ScalarType::FP32);
    disjoint_storage(y,output);disjoint_storage(down_scales,output);
    disjoint_storage(y,down_scales);
    return {g.rows,g.hidden,uint32_t(y.row_stride_bytes),uint32_t(output.row_stride_bytes),2,0,
        1.f,uint32_t(down_scales.row_stride_bytes),1,0,0};
}

enum class StatusRole : uint8_t { Producer, Restore };
inline constexpr uint32_t weight_mask=7, activation_mask=8, producer_mask=15, restore_mask=16;
inline void validate_status_mask(StatusRole role,uint32_t encoded,uint32_t flags) {
    if (role!=StatusRole::Producer && role!=StatusRole::Restore)
        throw std::invalid_argument("native ANE stage status has an unknown role");
    if (encoded!=(role==StatusRole::Producer?producer_mask:restore_mask))
        throw std::invalid_argument("native ANE stage status has incomplete/foreign writers");
    if (flags) throw std::runtime_error("native ANE stage GPU validation flags are nonzero: "+std::to_string(flags));
}

} // namespace tc::metal_graph::native_ane_stage_contract
