#pragma once

#include "../../ane_transfer_kernels.hpp"
#include "../../ane_w8_kernels.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tc::metal_graph::native_ane_stage_kernels {
namespace detail {
inline size_t find_once(std::string_view source, std::string_view expected) {
    const auto position = source.find(expected);
    if (expected.empty() || position == std::string_view::npos ||
        source.find(expected, position + 1) != std::string_view::npos)
        throw std::invalid_argument("native ANE stage source contract changed");
    return position;
}

inline void replace_once(std::string& source, std::string_view expected,
                         std::string_view replacement) {
    const auto position = find_once(source, expected);
    source.replace(position, expected.size(), replacement.data(), replacement.size());
}
} // namespace detail

// Reuse the legacy arithmetic verbatim. Only the three generic entry signatures
// put copied parameters after their native buffer table; other kernels remain
// unchanged and are not dispatched by this stage bundle.
inline std::string source() {
    std::string shader = tc::ane::gpu::transfer_source;
    shader += tc::ane::gpu::w8_source;
    for (const auto entry : {"kernel void tc_ane_restore(",
                             "kernel void tc_ane_w8_scales(",
                             "kernel void tc_ane_w8_codes("})
        (void)detail::find_once(shader, entry);

    detail::replace_once(shader,
R"signature(kernel void tc_ane_restore(device const ushort *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                           device atomic_uint *status [[buffer(2)]], constant Params &p [[buffer(3)]],
                           device const uint *failed [[buffer(4)]], uint2 tile [[threadgroup_position_in_grid]],
                           device const ushort *row_scales [[buffer(5)]], device const ushort *token_scales [[buffer(6)]],
                           device const ushort *second_scales [[buffer(7)]],
                           uint2 t [[thread_position_in_threadgroup]]) {)signature",
R"signature(kernel void tc_ane_restore(device const ushort *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                           device atomic_uint *status [[buffer(2)]], constant Params &p [[buffer(7)]],
                           device const uint *failed [[buffer(3)]], uint2 tile [[threadgroup_position_in_grid]],
                           device const ushort *row_scales [[buffer(4)]], device const ushort *token_scales [[buffer(5)]],
                           device const ushort *second_scales [[buffer(6)]],
                           uint2 t [[thread_position_in_threadgroup]]) {)signature");

    detail::replace_once(shader,
R"signature(kernel void tc_ane_w8_scales(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device ushort *dst [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(5)]], constant float *signs [[buffer(6)]], device const float *column_scale [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {)signature",
R"signature(kernel void tc_ane_w8_scales(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device ushort *dst [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(7)]], constant float *signs [[buffer(5)]], device const float *column_scale [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {)signature");

    detail::replace_once(shader,
R"signature(kernel void tc_ane_w8_codes(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device const ushort *row_scales [[buffer(3)]], device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]], constant W8Params &p [[buffer(6)]], constant float *signs [[buffer(7)]], device const float *column_scale [[buffer(8)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {)signature",
R"signature(kernel void tc_ane_w8_codes(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device const ushort *row_scales [[buffer(3)]], device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]], constant W8Params &p [[buffer(8)]], constant float *signs [[buffer(6)]], device const float *column_scale [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {)signature");

    // The native bundle selects only generic kernels. These ordinary constants
    // also leave the unused register entry points compilable without supplying
    // Metal function constants or changing any helper/kernel body.
    detail::replace_once(shader,
        "constant bool tc_w8_specialize [[function_constant(0)]];",
        "constant bool tc_w8_specialize = false;");
    detail::replace_once(shader,
        "constant uint tc_w8_encoding [[function_constant(1)]];",
        "constant uint tc_w8_encoding = 0;");
    detail::replace_once(shader,
        "constant uint tc_w8_dtype [[function_constant(2)]];",
        "constant uint tc_w8_dtype = 1;");
    detail::replace_once(shader,
        "constant uint tc_w8_block [[function_constant(3)]];",
        "constant uint tc_w8_block = 128;");
    if (shader.find("[[function_constant(") != std::string::npos)
        throw std::invalid_argument("native ANE stage has an unknown function constant");
    return shader;
}

// Compile the two register entries for one fixed BF16 rotation block. Dispatch
// generic W8/A8 from source(), whose ordinary constants remain unspecialized.
inline std::string register_source(uint32_t block) {
    if (block != 128 && block != 512)
        throw std::invalid_argument("native ANE register stage requires H128 or H512");
    std::string shader = source();
    for (const auto entry : {"kernel void tc_ane_sylvester_register_scales(",
                             "kernel void tc_ane_sylvester_register_codes("})
        (void)detail::find_once(shader, entry);
    (void)detail::find_once(shader, "constant uint tc_w8_encoding = 0;");
    (void)detail::find_once(shader, "constant uint tc_w8_dtype = 1;");
    detail::replace_once(shader,
        "constant bool tc_w8_specialize = false;",
        "constant bool tc_w8_specialize = true;");
    detail::replace_once(shader,
        "constant uint tc_w8_block = 128;",
        block == 128 ? "constant uint tc_w8_block = 128;" : "constant uint tc_w8_block = 512;");

    detail::replace_once(shader,
R"signature(kernel void tc_ane_sylvester_register_scales(device const uchar *src [[buffer(0)]],device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]],device ushort *dst [[buffer(3)]],device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(5)]],constant float *signs [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {)signature",
R"signature(kernel void tc_ane_sylvester_register_scales(device const uchar *src [[buffer(0)]],device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]],device ushort *dst [[buffer(3)]],device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(6)]],constant float *signs [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {)signature");
    detail::replace_once(shader,
R"signature(kernel void tc_ane_sylvester_register_codes(device const uchar *src [[buffer(0)]],device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]],device const ushort *row_scales [[buffer(3)]],device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]],constant W8Params &p [[buffer(6)]],constant float *signs [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {)signature",
R"signature(kernel void tc_ane_sylvester_register_codes(device const uchar *src [[buffer(0)]],device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]],device const ushort *row_scales [[buffer(3)]],device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]],constant W8Params &p [[buffer(7)]],constant float *signs [[buffer(6)]],
    uint2 tile [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {)signature");
    return shader;
}

} // namespace tc::metal_graph::native_ane_stage_kernels
