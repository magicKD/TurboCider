#include "ltx_gpu.h"
#include "ltx_w8a8.h"
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <stdexcept>
#include <vector>

static char error[2048];
static void check(bool ok) { if (!ok) throw std::runtime_error(error[0] ? error : "check failed"); }
using Buffer = std::unique_ptr<ltx_gpu_buffer, decltype(&ltx_gpu_buffer_free)>;
static Buffer tensor(ltx_gpu *gpu, size_t bytes, const void *data = nullptr) {
    auto *buffer = data ? ltx_gpu_buffer_new_copy(gpu, data, bytes, error, sizeof(error)) :
        ltx_gpu_buffer_new(gpu, bytes, error, sizeof(error));
    check(buffer != nullptr);
    return Buffer(buffer, ltx_gpu_buffer_free);
}
static float f32(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16; float out; std::memcpy(&out, &bits, 4); return out;
}
static uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) == 0x7f800000u) return uint16_t((bits >> 16) | ((bits & 0x7fffffu) ? 64 : 0));
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static std::vector<uint16_t> values(size_t count, float scale = 1.f) {
    std::vector<uint16_t> data(count);
    for (size_t i = 0; i < count; ++i) data[i] = bf16(scale * std::sin(float(i * 7 + 11) / 19.f));
    return data;
}
static std::vector<int8_t> weights(size_t count) {
    std::vector<int8_t> data(count);
    for (size_t i = 0; i < count; ++i) data[i] = int8_t(int((i * 73 + i / 13) % 256) - 128);
    return data;
}
// Real Gemma gates exceed the safe range of Metal's tanh implementation.
// Exercise the fused product directly so a NaN cannot be hidden by quantization.
static void gated_product_extremes() {
    const float cases[] = {-1e10f, -1024.f, -50.f, -10.f, -3.f, -1.f,
                            0.f, 1.f, 3.f, 10.f, 50.f, 1024.f, 1e10f};
    constexpr uint32_t count = 513;
    std::vector<uint16_t> gate(count), up(count), actual(count);
    for (uint32_t i = 0; i < count; ++i) {
        gate[i] = bf16(cases[i % (sizeof(cases) / sizeof(cases[0]))]);
        up[i] = bf16(i % 2 ? -0.75f : 1.25f);
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    void *context = ltx_w8a8_create((__bridge void *)device, error, sizeof(error));
    check(context != nullptr);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> g = [device newBufferWithBytes:gate.data() length:count * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> u = [device newBufferWithBytes:up.data() length:count * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:count * 2 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    ltx_w8a8_gated_product(context, (__bridge void *)encoder, (__bridge void *)g,
        (__bridge void *)u, (__bridge void *)out, count);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted);
    std::memcpy(actual.data(), out.contents, count * 2);
    ltx_w8a8_free(context);
    for (uint32_t i = 0; i < count; ++i) {
        float x = f32(gate[i]);
        float inner = std::clamp((x + ((x * x) * x * 0.044715f)) * 0.7978845608028654f, -10.f, 10.f);
        float activated = f32(bf16((x * 0.5f) * (1.f + std::tanh(inner))));
        float expected = f32(bf16(activated * f32(up[i])));
        check(std::isfinite(f32(actual[i])));
        check(std::abs(f32(actual[i]) - expected) <= std::max(0.008f * std::abs(expected), 1e-5f));
    }
    std::puts("gated GELU/product large finite values and partial grid PASS");
}
static void simd_rotation_exact(ltx_gpu *gpu, const char *shader) {
    check(ltx_gpu_convrot_simd_enabled(gpu));
    setenv("TURBOCIDER_LTX_CONVROT_SIMD", "0", 1);
    std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)> reference(
        ltx_gpu_create(shader, error, sizeof(error)), ltx_gpu_free);
    setenv("TURBOCIDER_LTX_CONVROT_SIMD", "1", 1);
    check(bool(reference) && !ltx_gpu_convrot_simd_enabled(reference.get()));
    uint32_t random = 73;
    for (auto [rows, columns] : {std::pair{1u, 256u}, std::pair{33u, 768u},
                                std::pair{65u, 4096u}, std::pair{65u, 16384u}}) {
        size_t count = size_t(rows) * columns;
        std::vector<uint16_t> input(count), actual(count), expected(count);
        for (size_t i = 0; i < count; ++i) {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            input[i] = uint16_t((random & 0x807fu) | ((8u + (random >> 8) % 235u) << 7));
        }
        input[0] = 0; input[1] = 0x8000u;
        auto in = tensor(gpu, count * 2, input.data()), out = tensor(gpu, count * 2);
        auto ref_in = tensor(reference.get(), count * 2, input.data());
        auto ref_out = tensor(reference.get(), count * 2);
        check(ltx_gpu_convrot_bf16(reference.get(), ref_out.get(), ref_in.get(), rows,
                                  columns, 256, error, sizeof(error)));
        check(ltx_gpu_convrot_bf16(gpu, out.get(), in.get(), rows, columns, 256, error, sizeof(error)));
        check(ltx_gpu_buffer_read(ref_out.get(), expected.data(), count * 2, error, sizeof(error)));
        check(ltx_gpu_buffer_read(out.get(), actual.data(), count * 2, error, sizeof(error)));
        check(actual == expected);
        for (auto value : actual) check(std::isfinite(f32(value)));
    }
    std::puts("SIMD ConvRot byte-exact vs original GPU: wide BF16 exponents, signed zero and row tails PASS");
}

static void fused_gelu_rotation_exact(ltx_gpu *gpu) {
    // Exhaust all BF16 signs/mantissas/exponents through 240, including
    // subnormals and signed zero. The bound keeps H4 FP32 intermediates finite.
    for (auto [rows, columns] : {std::pair{241u, 256u}, std::pair{33u, 768u},
                                std::pair{65u, 4096u}}) {
        size_t count = size_t(rows) * columns;
        std::vector<uint16_t> input(count), expected(count), actual(count);
        for (size_t i = 0; i < count; ++i) {
            uint32_t code = uint32_t(i % (241u * 256u));
            input[i] = uint16_t((code % (241u * 128u)) |
                               (code >= 241u * 128u ? 0x8000u : 0u));
        }
        auto in = tensor(gpu, count * 2, input.data());
        auto activated = tensor(gpu, count * 2);
        auto reference = tensor(gpu, count * 2), fused = tensor(gpu, count * 2);
        check(ltx_gpu_gelu_tanh_bf16(gpu, activated.get(), in.get(), uint32_t(count),
                                    error, sizeof(error)));
        check(ltx_gpu_convrot_bf16(gpu, reference.get(), activated.get(), rows,
                                  columns, 256, error, sizeof(error)));
        check(ltx_gpu_gelu_convrot_bf16(gpu, fused.get(), in.get(), rows, columns,
                                       256, error, sizeof(error)));
        check(ltx_gpu_buffer_read(reference.get(), expected.data(), count * 2, error, sizeof(error)));
        check(ltx_gpu_buffer_read(fused.get(), actual.data(), count * 2, error, sizeof(error)));
        check(actual == expected);
        for (auto value : actual) check(std::isfinite(f32(value)));
    }
    std::puts("Fused GELU/BF16-round/SIMD ConvRot byte-exact: 61696 BF16 patterns and row tails PASS");
}

// Independent dense H4-kron oracle, not the GPU butterfly implementation.
static std::vector<float> rotate(const std::vector<uint16_t>& x, int m, int k) {
    std::vector<float> out(x.size());
    for (int row = 0; row < m; ++row) for (int block = 0; block < k; block += 256)
        for (int n = 0; n < 256; ++n) {
            double sum = 0;
            for (int j = 0; j < 256; ++j) {
                int sign = 1;
                for (int shift = 0; shift < 8; shift += 2) {
                    int a = (n >> shift) & 3, b = (j >> shift) & 3;
                    if (a + b == 3) sign = -sign;
                }
                sum += sign * double(f32(x[size_t(row) * k + block + j]));
            }
            out[size_t(row) * k + block + n] = f32(bf16(float(sum / 16.)));
        }
    return out;
}
static void linear_oracle(ltx_gpu *gpu, int m, int n, int k, bool bias) {
    auto x = values(size_t(m) * k); auto w = weights(size_t(n) * k);
    for (int i = 0; i < k; ++i) x[i] = 0;
    if (m > 1) for (int i = 0; i < k; ++i) x[k + i] = bf16(f32(x[k + i]) * 100000.f);
    auto b = values(n, 0.1f);
    std::vector<float> scales(n);
    for (int i = 0; i < n; ++i) scales[i] = i % 11 == 0 ? 0.f : (i % 2 ? -1.f : 1.f) / 256.f;
    auto xb = tensor(gpu, x.size() * 2, x.data()), wb = tensor(gpu, w.size(), w.data());
    auto sb = tensor(gpu, scales.size() * 4, scales.data()), bb = tensor(gpu, b.size() * 2, b.data());
    auto out = tensor(gpu, size_t(m) * n * 2);
    check(ltx_gpu_linear_int8_convrot_mps_bf16(gpu, out.get(), xb.get(), wb.get(), sb.get(),
        bias ? bb.get() : nullptr, m, k, n, 256, error, sizeof(error)));
    std::vector<uint16_t> actual(size_t(m) * n);
    check(ltx_gpu_buffer_read(out.get(), actual.data(), actual.size() * 2, error, sizeof(error)));
    auto rotated = rotate(x, m, k);
    double max_error = 0;
    for (int row = 0; row < m; ++row) {
        float peak = 0.f;
        for (int i = 0; i < k; ++i) peak = std::max(peak, std::abs(rotated[size_t(row) * k + i]));
        float a_scale = peak == 0.f ? 1.f : peak / 127.f;
        for (int col = 0; col < n; ++col) {
            int32_t dot = 0;
            for (int i = 0; i < k; ++i) {
                int q = int(std::nearbyint(rotated[size_t(row) * k + i] / a_scale));
                dot += std::clamp(q, -127, 127) * int(w[size_t(col) * k + i]);
            }
            float expected = f32(bf16((float(dot) * a_scale) * scales[col]));
            if (bias) expected = f32(bf16(expected + f32(b[col])));
            double delta = std::abs(double(f32(actual[size_t(row) * n + col])) - expected);
            max_error = std::max(max_error, delta / std::max(std::abs(double(expected)), 1.));
            if (delta > std::max(0.008 * std::abs(double(expected)), 0.004)) {
                std::fprintf(stderr, "oracle mismatch m=%d n=%d k=%d row=%d col=%d expected=%g actual=%g\n",
                    m, n, k, row, col, expected, f32(actual[size_t(row) * n + col]));
                throw std::runtime_error("integer oracle mismatch");
            }
        }
    }
    std::printf("oracle M=%d N=%d K=%d bias=%d max_relative=%g PASS\n", m, n, k, bias, max_error);
}

static void varying_projections(ltx_gpu *gpu) {
    constexpr uint32_t rows = 33, k = 256;
    uint32_t dims[] = {129, 256, 33};
    auto input = values(rows * k);
    auto xb = tensor(gpu, input.size() * 2, input.data());
    std::vector<Buffer> output_storage, weight_storage, scale_storage, bias_storage;
    ltx_gpu_buffer *outputs[3];
    const ltx_gpu_buffer *weights_ptr[3], *scales_ptr[3], *biases_ptr[3];
    for (uint32_t i = 0; i < 3; ++i) {
        auto w = weights(size_t(dims[i]) * k);
        auto b = values(dims[i], 0.1f);
        std::vector<float> scales(dims[i]);
        for (uint32_t j = 0; j < dims[i]; ++j) scales[j] = (j % 2 ? -1.f : 1.f) / 256.f;
        output_storage.push_back(tensor(gpu, size_t(rows) * dims[i] * 2));
        weight_storage.push_back(tensor(gpu, w.size(), w.data()));
        scale_storage.push_back(tensor(gpu, scales.size() * 4, scales.data()));
        bias_storage.push_back(tensor(gpu, b.size() * 2, b.data()));
        outputs[i] = output_storage.back().get(); weights_ptr[i] = weight_storage.back().get();
        scales_ptr[i] = scale_storage.back().get(); biases_ptr[i] = i == 1 ? bias_storage.back().get() : nullptr;
    }
    const bool w8a8 = ltx_gpu_convrot_w8a8_info(gpu, nullptr, nullptr);
    for (uint32_t count : {3u, 2u}) {
        uint64_t p0, a0, p1, a1;
        ltx_gpu_convrot_w8a8_info(gpu, &p0, &a0);
        check(ltx_gpu_linears_int8_convrot_bf16(gpu, outputs, xb.get(), weights_ptr, scales_ptr,
            biases_ptr, dims, count, rows, k, error, sizeof(error)));
        ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1);
        check(p1 - p0 == (w8a8 ? count : 0u) && a1 - a0 == (w8a8 ? 1u : 0u));
        for (uint32_t i = 0; i < count; ++i) {
            auto reference = tensor(gpu, size_t(rows) * dims[i] * 2);
            check(ltx_gpu_linear_int8_convrot_mps_bf16(gpu, reference.get(), xb.get(), weights_ptr[i],
                scales_ptr[i], biases_ptr[i], rows, k, dims[i], 256, error, sizeof(error)));
            std::vector<uint16_t> a(size_t(rows) * dims[i]), b(a.size());
            check(ltx_gpu_buffer_read(outputs[i], a.data(), a.size() * 2, error, sizeof(error)));
            check(ltx_gpu_buffer_read(reference.get(), b.data(), b.size() * 2, error, sizeof(error)));
            check(a == b);
        }
    }
    uint32_t bad_dims[] = {130, 256, 33};
    check(!ltx_gpu_linears_int8_convrot_bf16(gpu, outputs, xb.get(), weights_ptr, scales_ptr,
        biases_ptr, bad_dims, 3, rows, k, error, sizeof(error)));
    std::printf("varying-width GQA shared projections mode=%s count=2/3 byte_exact=PASS padding/bias/bounds=PASS\n", w8a8 ? "w8a8" : "bf16");
}
static void composites(ltx_gpu *gpu) {
    constexpr int m = 7, k = 256, n = 256, h = 512;
    auto x = values(m * k), b = values(n, 0.01f), norm = std::vector<uint16_t>(n, bf16(1.f));
    auto w = weights(n * k), wh = weights(h * k), wd = weights(n * h);
    std::vector<float> scale(h, 1.f / 256.f);
    auto xb = tensor(gpu, x.size()*2, x.data()), wb = tensor(gpu, w.size(), w.data());
    auto sh = tensor(gpu, scale.size()*4, scale.data()), bb = tensor(gpu, b.size()*2, b.data());
    auto nb = tensor(gpu, norm.size()*2, norm.data()), whb = tensor(gpu, wh.size(), wh.data());
    auto wdb = tensor(gpu, wd.size(), wd.data());
    auto q = tensor(gpu, m*n*2), key = tensor(gpu, m*n*2), v = tensor(gpu, m*n*2), out = tensor(gpu, m*n*2);
    uint64_t p0, a0, p1, a1;
    check(ltx_gpu_convrot_w8a8_info(gpu, &p0, &a0));
    check(ltx_gpu_qkv_int8_convrot_mps_bf16(gpu, q.get(), key.get(), v.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), m, k, n, 256, 1e-6f, error, sizeof(error)));
    check(ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1));
    check(p1 - p0 == 3 && a1 - a0 == 1);
    p0 = p1; a0 = a1;
    check(ltx_gpu_gated_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), whb.get(), sh.get(), wdb.get(), sh.get(), m, k, h, n, 256, error, sizeof(error)));
    check(ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1));
    check(p1 - p0 == 3 && a1 - a0 == 2);
    const uint64_t fused_before = ltx_gpu_gelu_convrot_dispatches(gpu);
    const uint64_t quant_before = ltx_gpu_rot_quant_dispatches(gpu);
    const uint64_t output_before = ltx_gpu_gelu_output_dispatches(gpu);
    check(ltx_gpu_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), nullptr, wdb.get(), sh.get(), bb.get(), m, k, h, n, 256, error, sizeof(error)));
    const char *fusion = getenv("TURBOCIDER_LTX_W8A8_FUSED_GELU_CONVROT");
    const char *output_fusion = getenv("TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT");
    const bool expected_output = output_fusion && strcmp(output_fusion, "1") == 0 && ltx_w8a8_fused_output_enabled();
    const bool expected_fusion = !expected_output && fusion && strcmp(fusion, "1") == 0 && ltx_gpu_convrot_simd_enabled(gpu);
    check(ltx_gpu_gelu_output_dispatches(gpu) - output_before == uint64_t(expected_output));
    check(ltx_gpu_gelu_convrot_dispatches(gpu) - fused_before == uint64_t(expected_fusion));
    const char *quant_fusion = getenv("TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT");
    const bool expected_quant = (!quant_fusion || strcmp(quant_fusion, "1") == 0) && ltx_gpu_convrot_simd_enabled(gpu);
    check(ltx_gpu_rot_quant_dispatches(gpu) - quant_before == (expected_quant ? 2u : 0u));
    auto cos = std::vector<uint16_t>(m*n/2, bf16(1.f)), sin = std::vector<uint16_t>(m*n/2, 0);
    auto cb = tensor(gpu, cos.size()*2, cos.data()), sb = tensor(gpu, sin.size()*2, sin.data());
    auto gatew = values(2*k, 0.001f); auto gw = tensor(gpu, gatew.size()*2, gatew.data());
    p0 = p1; a0 = a1;
    check(ltx_gpu_self_attention_int8_mps_bf16(gpu, out.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), nullptr, wb.get(), sh.get(), bb.get(), cb.get(), sb.get(),
        m, 2, 128, 256, 1e-6f, error, sizeof(error)));
    check(ltx_gpu_convrot_w8a8_info(gpu, &p1, &a1));
    // Includes the preceding two-projection FFN.
    check(p1 - p0 == 6 && a1 - a0 == 4);
    check(ltx_gpu_cross_attention_int8_mps_bf16(gpu, out.get(), xb.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), nullptr, wb.get(), sh.get(), bb.get(), nullptr, nullptr, nullptr, nullptr,
        m, m, k, k, 2, 128, n, 256, 1e-6f, error, sizeof(error)));
    std::vector<uint16_t> result(m*n);
    check(ltx_gpu_buffer_read(out.get(), result.data(), result.size()*2, error, sizeof(error)));
    for (auto value : result) check(std::isfinite(f32(value)));
    std::puts("composites QKV/shared-quantization/gated-MLP/MLP/self-attention/cross-attention PASS");
}

static void adaln_mlp_exact(ltx_gpu *gpu, const char *shader) {
    constexpr uint32_t hidden = 256, outdim = 129;
    auto other_gpu = std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)>(
        ltx_gpu_create(shader, error, sizeof(error)), ltx_gpu_free);
    check(other_gpu != nullptr);
    for (auto [rows, dim] : {std::pair{7u, 4096u}, std::pair{33u, 4096u},
                            std::pair{65u, 4096u}, std::pair{7u, 2048u}}) {
        auto input = values(size_t(rows) * dim, .7f), ns = values(dim, .13f), nt = values(dim, .07f);
        auto first = weights(size_t(hidden) * dim), second = weights(size_t(outdim) * hidden);
        std::vector<float> s1(hidden, .0013f), s2(outdim, .0021f);
        auto b1 = values(hidden, .03f), b2 = values(outdim, .02f);
        auto x = tensor(gpu, input.size() * 2, input.data()), scale = tensor(gpu, ns.size() * 2, ns.data());
        auto shift = tensor(gpu, nt.size() * 2, nt.data()), normalized = tensor(gpu, input.size() * 2);
        auto w1 = tensor(gpu, first.size(), first.data()), w2 = tensor(gpu, second.size(), second.data());
        auto ws1 = tensor(gpu, s1.size() * 4, s1.data()), ws2 = tensor(gpu, s2.size() * 4, s2.data());
        auto bias1 = tensor(gpu, b1.size() * 2, b1.data()), bias2 = tensor(gpu, b2.size() * 2, b2.data());
        auto reference = tensor(gpu, size_t(rows) * outdim * 2), actual = tensor(gpu, size_t(rows) * outdim * 2);
        check(ltx_gpu_adaln_bf16(gpu, normalized.get(), x.get(), scale.get(), shift.get(), rows, dim, 1u,
            1e-6f, error, sizeof(error)));
        check(ltx_gpu_mlp_int8_convrot_mps_bf16(gpu, reference.get(), normalized.get(),
            w1.get(), ws1.get(), bias1.get(), w2.get(), ws2.get(), bias2.get(),
            rows, dim, hidden, outdim, 256, error, sizeof(error)));
        const uint64_t before = ltx_gpu_adaln_rot_quant_dispatches(gpu);
        check(ltx_gpu_mlp_adaln_int8_convrot_mps_bf16(gpu, actual.get(), x.get(),
            scale.get(), shift.get(), normalized.get(), 1e-6f,
            w1.get(), ws1.get(), bias1.get(), w2.get(), ws2.get(), bias2.get(),
            rows, dim, hidden, outdim, 256, error, sizeof(error)));
        const char *prepare = getenv("TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT");
        const bool fused = dim == 4096 && ltx_gpu_convrot_simd_enabled(gpu) &&
            (!prepare || strcmp(prepare, "1") == 0);
        check(ltx_gpu_adaln_rot_quant_dispatches(gpu) == before + unsigned(fused));
        check(!memcmp(ltx_gpu_buffer_contents(reference.get()), ltx_gpu_buffer_contents(actual.get()),
            size_t(rows) * outdim * 2));
        auto short_scale = tensor(gpu, 2);
        check(!ltx_gpu_mlp_adaln_int8_convrot_mps_bf16(gpu, actual.get(), x.get(),
            short_scale.get(), shift.get(), normalized.get(), 1e-6f,
            w1.get(), ws1.get(), bias1.get(), w2.get(), ws2.get(), bias2.get(),
            rows, dim, hidden, outdim, 256, error, sizeof(error)));
        check(ltx_gpu_adaln_rot_quant_dispatches(gpu) == before + unsigned(fused));
        error[0] = 0;
        auto hidden_scratch = tensor(gpu, size_t(rows) * hidden * 2);
        const uint64_t reused_before = ltx_gpu_mlp_hidden_scratch_reuses(gpu);
        check(ltx_gpu_mlp_int8_convrot_mps_bf16_with_scratch(gpu, actual.get(), normalized.get(),
            hidden_scratch.get(), w1.get(), ws1.get(), bias1.get(), w2.get(), ws2.get(), bias2.get(),
            rows, dim, hidden, outdim, 256, error, sizeof(error)));
        check(!memcmp(ltx_gpu_buffer_contents(reference.get()), ltx_gpu_buffer_contents(actual.get()),
            size_t(rows) * outdim * 2));
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            check(ltx_gpu_batch_begin(gpu, error, sizeof(error)));
            check(ltx_gpu_mlp_adaln_int8_convrot_mps_bf16_with_scratch(gpu, actual.get(), x.get(),
                hidden_scratch.get(), scale.get(), shift.get(), normalized.get(), 1e-6f,
                w1.get(), ws1.get(), bias1.get(), w2.get(), ws2.get(), bias2.get(),
                rows, dim, hidden, outdim, 256, error, sizeof(error)));
            check(ltx_gpu_batch_end(gpu, error, sizeof(error)));
            check(!memcmp(ltx_gpu_buffer_contents(reference.get()), ltx_gpu_buffer_contents(actual.get()),
                size_t(rows) * outdim * 2));
        }
        check(ltx_gpu_mlp_hidden_scratch_reuses(gpu) == reused_before + 3);
        auto short_hidden = tensor(gpu, 2);
        auto foreign_hidden = tensor(other_gpu.get(), size_t(rows) * hidden * 2);
        for (auto *invalid_hidden : {short_hidden.get(), x.get(), normalized.get(), w1.get(), foreign_hidden.get()}) {
            check(!ltx_gpu_mlp_adaln_int8_convrot_mps_bf16_with_scratch(gpu, actual.get(), x.get(),
                invalid_hidden, scale.get(), shift.get(), normalized.get(), 1e-6f,
                w1.get(), ws1.get(), bias1.get(), w2.get(), ws2.get(), bias2.get(),
                rows, dim, hidden, outdim, 256, error, sizeof(error)));
            check(ltx_gpu_mlp_hidden_scratch_reuses(gpu) == reused_before + 3);
            check(ltx_gpu_streaming_boundary_completed(gpu));
            error[0] = 0;
        }
    }
    std::puts("AdaLN/rotation/quantization MLP output byte-exact; bias, output tails, SIMD/quantizer fallback and short-buffer rejection PASS");
    std::puts("Caller-owned FFN hidden workspace: ordinary/AdaLN, repeated outer batches, exact output and alias/short-buffer/foreign-owner rejection PASS");
}

static void gelu_output_rounding_exact(ltx_gpu *gpu) {
    constexpr uint32_t m = 65280, n = 17, k = 256;
    std::vector<float> scales;
    for (uint32_t bits = 0; bits < 65536u; ++bits)
        if ((bits & 0x7f80u) != 0x7f80u) scales.push_back(f32(uint16_t(bits)));
    check(scales.size() == m);
    std::vector<int8_t> codes(size_t(m) * k), weight(128u * k);
    for (uint32_t row = 0; row < m; ++row) codes[size_t(row) * k] = 1;
    for (uint32_t col = 0; col < n; ++col) weight[size_t(col) * k] = 1;
    std::vector<float> channel_scales(n, 1.f);
    std::vector<uint16_t> bias(n), expected(size_t(m) * n);
    const float dequant_scales[] = {1.f, 1.0009765625f, .75f, .73125f};
    for (uint32_t col = 0; col < n; ++col) {
        channel_scales[col] = dequant_scales[col % 4];
        bias[col] = bf16((int(col) - 8) / 1024.f);
    }
    // For x=1/128 in column 9, the original pre-bias BF16 round cancels
    // this bias to zero. Removing that round leaves a nonzero GELU input.
    bias[9] = bf16(-1.f / 128.f);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    std::unique_ptr<void, decltype(&ltx_w8a8_free)> context(
        ltx_w8a8_create((__bridge void *)device, error, sizeof(error)), ltx_w8a8_free);
    check(context != nullptr);
    check(ltx_w8a8_prepare_gelu_output(context.get(), error, sizeof(error)));
    id<MTLCommandQueue> queue = [device newCommandQueue];
    auto buffer = [&](const void *data, size_t bytes) -> id<MTLBuffer> {
        return data ? [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared] :
                      [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> a = buffer(codes.data(), codes.size()), w = buffer(weight.data(), weight.size());
    id<MTLBuffer> sa = buffer(scales.data(), scales.size() * 4), sw = buffer(channel_scales.data(), n * 4);
    id<MTLBuffer> bb = buffer(bias.data(), n * 2), original = buffer(nullptr, expected.size() * 2);
    id<MTLBuffer> actual = buffer(nullptr, expected.size() * 2);
    for (bool has_bias : {false, true}) {
        auto encode = [&](bool activated, id<MTLBuffer> destination) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            ltx_w8a8_project_activation(context.get(), (__bridge void *)encoder, (__bridge void *)a,
                (__bridge void *)w, (__bridge void *)sw, (__bridge void *)sa, nullptr,
                (__bridge void *)destination, has_bias ? (__bridge void *)bb : nullptr,
                m, k, n, 1, activated);
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            check(command.status == MTLCommandBufferStatusCompleted);
        };
        encode(false, original);
        auto reference = tensor(gpu, expected.size() * 2, original.contents);
        check(ltx_gpu_gelu_tanh_bf16(gpu, reference.get(), reference.get(), uint32_t(expected.size()), error, sizeof(error)));
        check(ltx_gpu_buffer_read(reference.get(), expected.data(), expected.size() * 2, error, sizeof(error)));
        if (has_bias) check(expected[size_t(0x3c00) * n + 9] == 0u);
        encode(true, actual);
        const auto *values = static_cast<const uint16_t *>(actual.contents);
        for (size_t index = 0; index < expected.size(); ++index) {
            if (values[index] != expected[index]) {
                std::fprintf(stderr, "GELU output rounding mismatch bias=%d row=%zu col=%zu actual=%04x expected=%04x\n",
                    has_bias, index / n, index % n, values[index], expected[index]);
                check(false);
            }
        }
    }
    std::puts("GELU output: all 65280 finite BF16 patterns, F32 dequant/bias rounding and output tails byte-exact PASS");
}

using Captures = std::vector<std::pair<std::string, std::vector<uint16_t>>>;
static Captures capture_operators(ltx_gpu *gpu) {
    constexpr int m = 7, kv = 5, k = 256, h = 512, n = 256;
    auto x = values(m * k), z = values(kv * k, 0.7f);
    auto b = values(h, 0.01f), norm = std::vector<uint16_t>(n, bf16(1.f));
    auto w = weights(n * k), wh = weights(h * k), wd = weights(n * h);
    std::vector<float> scale(h, 1.f/256.f);
    auto xb = tensor(gpu, x.size()*2, x.data()), zb = tensor(gpu, z.size()*2, z.data());
    auto wb = tensor(gpu, w.size(), w.data()), sh = tensor(gpu, scale.size()*4, scale.data());
    auto bb = tensor(gpu, b.size()*2, b.data()), nb = tensor(gpu, norm.size()*2, norm.data());
    auto whb = tensor(gpu, wh.size(), wh.data()), wdb = tensor(gpu, wd.size(), wd.data());
    auto q = tensor(gpu, m*n*2), key = tensor(gpu, m*n*2), v = tensor(gpu, m*n*2), out = tensor(gpu, m*n*2);
    Captures captures;
    auto save = [&](const char* name, ltx_gpu_buffer* buffer) {
        std::vector<uint16_t> result(m*n);
        check(ltx_gpu_buffer_read(buffer, result.data(), result.size()*2, error, sizeof(error)));
        captures.emplace_back(name, std::move(result));
    };
    check(ltx_gpu_qkv_int8_convrot_mps_bf16(gpu, q.get(), key.get(), v.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), m, k, n, 256, 1e-6f, error, sizeof(error)));
    save("QKV/query", q.get()); save("QKV/key", key.get()); save("QKV/value", v.get());
    check(ltx_gpu_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), bb.get(), wdb.get(), sh.get(), bb.get(), m, k, h, n, 256, error, sizeof(error)));
    save("MLP", out.get());
    check(ltx_gpu_gated_mlp_int8_convrot_mps_bf16(gpu, out.get(), xb.get(),
        whb.get(), sh.get(), whb.get(), sh.get(), wdb.get(), sh.get(), m, k, h, n, 256, error, sizeof(error)));
    save("gated_MLP", out.get());
    auto cos = std::vector<uint16_t>(m*n/2, bf16(0.8f)), sin = std::vector<uint16_t>(m*n/2, bf16(0.6f));
    auto cb = tensor(gpu, cos.size()*2, cos.data()), sb = tensor(gpu, sin.size()*2, sin.data());
    auto gatew = values(2*k, 0.001f); auto gw = tensor(gpu, gatew.size()*2, gatew.data());
    check(ltx_gpu_self_attention_int8_mps_bf16(gpu, out.get(), xb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), bb.get(), wb.get(), sh.get(), bb.get(), cb.get(), sb.get(),
        m, 2, 128, 256, 1e-6f, error, sizeof(error)));
    save("self_attention_rope", out.get());
    auto mask = std::vector<uint16_t>(m*kv, 0);
    for (int i=0; i<m; ++i) mask[i*kv + i%kv] = bf16(-10000.f);
    auto mb = tensor(gpu, mask.size()*2, mask.data());
    check(ltx_gpu_cross_attention_int8_mps_bf16_masked(gpu, out.get(), xb.get(), zb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        nb.get(), nb.get(), gw.get(), bb.get(), wb.get(), sh.get(), bb.get(), cb.get(), sb.get(), cb.get(), sb.get(),
        mb.get(), m, m, kv, k, k, 2, 128, n, 256, 1e-6f, error, sizeof(error)));
    save("cross_attention_rope_mask", out.get());
    auto kc = tensor(gpu, kv*n*2), vc = tensor(gpu, kv*n*2);
    check(ltx_gpu_cross_attention_kv_int8_mps_bf16(gpu, kc.get(), vc.get(), zb.get(),
        wb.get(), sh.get(), bb.get(), wb.get(), sh.get(), bb.get(), nb.get(), kv, k, 2, 128,
        256, 1e-6f, error, sizeof(error)));
    check(ltx_gpu_cross_attention_query_int8_mps_bf16_masked(gpu, out.get(), xb.get(), kc.get(), vc.get(),
        wb.get(), sh.get(), bb.get(), nb.get(), gw.get(), bb.get(), wb.get(), sh.get(), bb.get(),
        mb.get(), m, m, kv, k, 2, 128, n, 256, 1e-6f, error, sizeof(error)));
    save("cross_attention_cached_KV_mask", out.get());
    auto pw = weights(3*n*k); std::vector<float> ps(3*n, 1.f/256.f); auto pb = values(3*n, 0.01f);
    auto pwb = tensor(gpu, pw.size(), pw.data()), psb = tensor(gpu, ps.size()*4, ps.data());
    auto pbb = tensor(gpu, pb.size()*2, pb.data());
    check(ltx_gpu_qkv_packed_int8_convrot_mps_bf16(gpu, q.get(), key.get(), v.get(), xb.get(),
        pwb.get(), psb.get(), pbb.get(), nb.get(), nb.get(), m, k, n, 256, 1e-6f, error, sizeof(error)));
    save("packed_QKV/query", q.get()); save("packed_QKV/key", key.get()); save("packed_QKV/value", v.get());
    return captures;
}
static void compare_operators(const Captures& actual, const Captures& baseline) {
    check(actual.size() == baseline.size());
    for (size_t c=0; c<actual.size(); ++c) {
        double delta=0, norm=0, anorm=0, dot=0;
        for (size_t i=0; i<actual[c].second.size(); ++i) {
            double a = f32(actual[c].second[i]), b = f32(baseline[c].second[i]);
            check(std::isfinite(a) && std::isfinite(b));
            delta += (a-b)*(a-b); norm += b*b; anorm += a*a; dot += a*b;
        }
        double relative = std::sqrt(delta/std::max(norm, 1e-30));
        double cosine = dot/std::sqrt(std::max(anorm*norm, 1e-30));
        std::printf("quality %s relative_l2=%.7f cosine=%.7f\n", actual[c].first.c_str(), relative, cosine);
        check(relative < 0.07 && cosine > 0.998);
    }
}
int main(int argc, char **argv) {
    try {
        check(argc == 2);
        unsetenv("TURBOCIDER_LTX_CONVROT_W8A8");
        unsetenv("TURBOCIDER_LTX_RAW_CONVROT");
        std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)> gpu(ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        check(bool(gpu));
        const char *simd = getenv("TURBOCIDER_LTX_CONVROT_SIMD");
        const bool expected_simd = simd ? strcmp(simd, "1") == 0 :
            ltx_gpu_convrot_w8a8_info(gpu.get(), nullptr, nullptr);
        check(bool(ltx_gpu_convrot_simd_enabled(gpu.get())) == expected_simd);
        if (!ltx_gpu_convrot_w8a8_info(gpu.get(), nullptr, nullptr)) {
            std::puts("SKIP native W8A8 unavailable; BF16 context creation PASS");
            return 0;
        }
        auto stage = std::unique_ptr<ltx_gpu_stage_workspace, decltype(&ltx_gpu_stage_workspace_free)>(
            nullptr, ltx_gpu_stage_workspace_free);
        if (getenv("TURBOCIDER_TEST_STAGE_WORKSPACE")) {
            stage.reset(ltx_gpu_stage_workspace_create(gpu.get(), 4004u, error, sizeof(error)));
            check(stage != nullptr);
        }
        gated_product_extremes();
        if (ltx_gpu_convrot_simd_enabled(gpu.get())) {
            simd_rotation_exact(gpu.get(), argv[1]);
            fused_gelu_rotation_exact(gpu.get());
        }
        linear_oracle(gpu.get(), 1, 33, 256, false);
        linear_oracle(gpu.get(), 33, 129, 512, true);
        linear_oracle(gpu.get(), 65, 131, 768, false);
        const uint64_t before_large = ltx_gpu_rot_quant_dispatches(gpu.get());
        linear_oracle(gpu.get(), 2, 33, 16640, false);
        check(ltx_gpu_rot_quant_dispatches(gpu.get()) == before_large);
        varying_projections(gpu.get());
        composites(gpu.get());
        adaln_mlp_exact(gpu.get(), argv[1]);
        const char *gelu_output = getenv("TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT");
        if (gelu_output && strcmp(gelu_output, "1") == 0)
            gelu_output_rounding_exact(gpu.get());
        auto actual = capture_operators(gpu.get());
        setenv("TURBOCIDER_LTX_CONVROT_W8A8", "0", 1);
        std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)> baseline(ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        check(bool(baseline));
        check(!ltx_gpu_convrot_w8a8_info(baseline.get(), nullptr, nullptr));
        varying_projections(baseline.get());
        compare_operators(actual, capture_operators(baseline.get()));
        if (stage) {
            check(ltx_gpu_stage_workspace_reuses(gpu.get()) > 0);
            stage.reset();
            check(ltx_gpu_streaming_boundary_completed(gpu.get()));
            std::puts("Stage projection workspace: integer oracles, tails, shared quantization and teardown PASS");
        }
        std::puts("PASS native LTX ConvRot W8A8");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
