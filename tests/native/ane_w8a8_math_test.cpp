#include "../../native/backends/ane_w8a8_math.hpp"
#include "../../native/backends/ane_w8_stage.hpp"
#include <iostream>
#include <limits>

namespace {
using namespace tc::ane;
void require(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }
template<class F> void rejected(F fn) {
    try { fn(); } catch (const CapabilityError &) { return; }
    throw std::runtime_error("invalid S1 contract was accepted");
}
std::vector<float> scaled_h128(const std::vector<float> &input, const std::vector<float> &full_scale, size_t first, bool inverse) {
    require(input.size() == 128 && first <= full_scale.size() && input.size() <= full_scale.size() - first, "S1 oracle geometry");
    std::vector<float> output = input;
    for (size_t i = 0; i < output.size(); ++i) {
        const float scale = full_scale[first + i];
        if (!std::isfinite(scale) || scale < 1.f / 16 || scale > 16) throw CapabilityError("invalid S1 oracle value");
        output[i] = inverse ? input[i] / scale : input[i] * scale;
        if (!std::isfinite(output[i])) throw CapabilityError("S1 oracle scaled overflow");
    }
    rotate_block(output, 20260930);
    return output;
}
void test_s1_math() {
    // Select the second physical H128 block; the first contains different
    // scales to catch accidental slice-relative S1 addressing.
    std::vector<float> scales(256), x(128), w(128);
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = std::ldexp(1.f, int(i % 9) - 4);
    for (size_t i = 0; i < x.size(); ++i) { x[i] = float(int(i % 7) - 3) / 8; w[i] = float(int(i % 11) - 5) / 8; }
    const auto xr = scaled_h128(x, scales, 128, true), wr = scaled_h128(w, scales, 128, false);
    double before = 0, after = 0;
    for (size_t i = 0; i < x.size(); ++i) { before += double(x[i]) * w[i]; after += double(xr[i]) * wr[i]; }
    require(std::abs(after - before) < 3e-5, "paired S1 before H128 failed dot-product identity");
    // Independent dense-H oracle confirms S1 precedes signs/rotation, with
    // the global column offset used by both scale and code Metal passes.
    const double norm = 1 / std::sqrt(128.);
    for (unsigned out = 0; out < 128; ++out) {
        double expected_x = 0, expected_w = 0;
        for (unsigned in = 0; in < 128; ++in) {
            const double h = ((std::popcount(out & in) & 1) ? -1 : 1) * rotation_sign(20260930, in) * norm;
            expected_x += double(x[in]) / scales[128 + in] * h;
            expected_w += double(w[in]) * scales[128 + in] * h;
        }
        require(std::abs(xr[out] - expected_x) < 4e-6 && std::abs(wr[out] - expected_w) < 4e-6, "S1 dense-H oracle mismatch");
    }
    auto plain = x; rotate_block(plain, 20260930);
    const std::vector<float> ones(256, 1);
    const auto scaled = scaled_h128(x, ones, 128, false), inverse = scaled_h128(x, ones, 128, true);
    require(plain == scaled && plain == inverse, "identity S1 changed plain rotation bits");
    const float peak = *std::max_element(plain.begin(), plain.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
    const auto plain_scale = normalized_scale(std::abs(peak));
    for (size_t i = 0; i < plain.size(); ++i)
        require(quantize_rotated(plain[i], plain_scale) == quantize_rotated(scaled[i], plain_scale) &&
                quantize_rotated(plain[i], plain_scale) == quantize_rotated(inverse[i], plain_scale), "identity S1 changed signed RNE codes");
    for (float bad : {0.f, -1.f, 1.f / 32, 32.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        auto invalid = scales; invalid[128] = bad;
        rejected([&] { scaled_h128(x, invalid, 128, false); });
        rejected([&] { scaled_h128(x, invalid, 128, true); });
    }
    auto overflow = x; overflow[0] = std::numeric_limits<float>::max();
    auto extremes = scales; extremes[128] = 16;
    rejected([&] { scaled_h128(overflow, extremes, 128, false); });
    extremes[128] = 1.f / 16;
    rejected([&] { scaled_h128(overflow, extremes, 128, true); });
}
void test_s1_keys_and_leases() {
    auto weight_owner = std::make_shared<int>(1), scale_owner = std::make_shared<int>(2);
    DeviceWeightView source{weight_owner.get(), 4096, 0, 1024, 4, 256, DeviceWeightEncoding::Dense, DType::FP32, 32,
                            {}, {}, weight_owner, weight_owner, true};
    DeviceMatrixView scale{scale_owner.get(), 1040, 16, 1, 256, 1040, DType::FP32, scale_owner, scale_owner};
    W8StageSpec spec{0, 4, 128, 128, 128}; spec.column_scale = scale;
    w8_stage::validate_column_scale(source, spec);
    DeviceWeightRegion region{source, spec};
    const auto key = w8_stage::weak_key(source, spec);
    require(w8_stage::same_region(region, key) && w8_stage::live_key(key), "weak S1 key no longer matches live staging request");
    require(!key.source.owner && !key.selection.column_scale->owner, "cache retained full source/S1 GPU allocation");
    auto changed = region;
    auto next_generation = std::make_shared<int>(3);
    changed.selection.column_scale->allocation_identity = next_generation;
    require(!w8_stage::same_region(region, changed), "S1 reused address matched another generation");
    changed = region; changed.selection.inverse_column_scale = true;
    require(!w8_stage::same_region(region, changed), "inverse S1 omitted from cache/prefetch key");
    changed = region; changed.selection.column_scale.reset();
    require(!w8_stage::same_region(region, changed), "plain path matched S1 key");
    changed = region; changed.source.allocation_identity = next_generation;
    require(!w8_stage::same_region(region, changed), "source generation omitted from prefetch key");
    changed = region; changed.selection.column_scale->offset_bytes = 0;
    require(!w8_stage::same_region(region, changed), "S1 view offset omitted from cache/prefetch key");
    changed = region; changed.selection.basis = W8Basis::ComfyH256;
    require(!w8_stage::same_region(region, changed), "rotation basis omitted from cache/prefetch key");
    changed = region; changed.selection.activation_group_size = 256;
    require(!w8_stage::same_region(region, changed), "activation grouping omitted from cache/prefetch key");
    std::weak_ptr<void> generation = scale_owner;
    auto ticket_spec = spec;
    region.selection.column_scale.reset(); changed.selection.column_scale.reset(); spec.column_scale.reset();
    scale.owner.reset(); scale_owner.reset();
    require(!generation.expired() && w8_stage::live_key(key), "ticket failed to keep S1 allocation until completion");
    ticket_spec.column_scale.reset();
    require(generation.expired() && !w8_stage::live_key(key), "weak cache retained S1 after ticket completion");
}
void test_s1_contract() {
    auto g_owner = std::make_shared<int>(1), u_owner = std::make_shared<int>(2), d_owner = std::make_shared<int>(3), s_owner = std::make_shared<int>(4);
    DeviceMatrixView s{s_owner.get(), 1024, 0, 1, 256, 0, DType::FP32, s_owner, s_owner};
    DeviceWeightView g{g_owner.get(), 512 * 256 * 4, 0, 256 * 4, 512, 256, DeviceWeightEncoding::Dense, DType::FP32, 32, {}, {}, g_owner, g_owner, true};
    DeviceWeightView u = g; u.buffer = u_owner.get(); u.owner = u_owner; u.allocation_identity = u_owner;
    DeviceWeightView d{d_owner.get(), 256 * 512 * 4, 0, 512 * 4, 256, 512, DeviceWeightEncoding::Dense, DType::FP32, 32, {}, {}, d_owner, d_owner, true};
    std::vector<DeviceWeightRegion> weights{{g,{0,512,0,256,128}}, {u,{0,512,0,256,128}}, {d,{0,256,0,512,512}}};
    w8_stage::validate_swiglu_column_scales(weights, 256); // old plain ABI
    weights[0].selection.column_scale = weights[1].selection.column_scale = s;
    w8_stage::validate_swiglu_column_scales(weights, 256);
    auto a8 = weights[0].selection; a8.transpose = true; a8.inverse_column_scale = true;
    w8_stage::validate_column_scale(g, a8);
    auto bad = weights;
    bad[1].selection.column_scale.reset(); rejected([&] { w8_stage::validate_swiglu_column_scales(bad, 256); });
    bad = weights; bad[1].selection.column_scale->allocation_identity = d_owner;
    rejected([&] { w8_stage::validate_swiglu_column_scales(bad, 256); });
    bad = weights; bad[2].selection.column_scale = s;
    rejected([&] { w8_stage::validate_swiglu_column_scales(bad, 256); });
    bad = weights; bad[2].selection.inverse_column_scale = true;
    rejected([&] { w8_stage::validate_swiglu_column_scales(bad, 256); });
    bad = weights; bad[0].selection.inverse_column_scale = true;
    rejected([&] { w8_stage::validate_swiglu_column_scales(bad, 256); });
    auto invalid = a8; invalid.inverse_column_scale = false;
    rejected([&] { w8_stage::validate_column_scale(g, invalid); });
    invalid = a8; invalid.column_scale.reset();
    rejected([&] { w8_stage::validate_column_scale(g, invalid); });
    const auto reject_scale = [&](auto mutate) {
        auto selected = weights[0].selection; mutate(*selected.column_scale);
        rejected([&] { w8_stage::validate_column_scale(g, selected); });
    };
    reject_scale([](auto &v) { v.rows = 2; });
    reject_scale([](auto &v) { v.cols = 128; });
    reject_scale([](auto &v) { v.dtype = DType::BF16; });
    reject_scale([](auto &v) { v.owner.reset(); });
    reject_scale([](auto &v) { v.allocation_identity.reset(); });
    reject_scale([](auto &v) { v.offset_bytes = 2; });
    reject_scale([](auto &v) { v.offset_bytes = SIZE_MAX; });
    reject_scale([](auto &v) { v.buffer_bytes = 1023; });
    reject_scale([](auto &v) { v.row_stride_bytes = 1020; });
    reject_scale([](auto &v) { v.row_stride_bytes = 1025; });
    reject_scale([](auto &v) { v.row_stride_bytes = size_t(UINT32_MAX) + 1; });
    reject_scale([&](auto &v) { v.buffer = g.buffer; });
    auto metadata_alias = g; metadata_alias.scales = s;
    rejected([&] { w8_stage::validate_column_scale(metadata_alias, weights[0].selection); });
    metadata_alias = g; metadata_alias.offsets = s;
    rejected([&] { w8_stage::validate_column_scale(metadata_alias, weights[0].selection); });
    auto h512 = weights[0].selection; h512.rotation_block = 512;
    rejected([&] { w8_stage::validate_column_scale(g, h512); });
    bad = weights; bad[0].selection.column_begin = bad[1].selection.column_begin = 128;
    rejected([&] { w8_stage::validate_swiglu_column_scales(bad, 256); });
}
}

int main() {
    using namespace tc::ane;
    for (int size : {128, 512}) {
        std::vector<float> a(size), b(size);
        for (int i = 0; i < size; ++i) { a[i] = float(i % 7 - 3) / 8; b[i] = float(i % 11 - 5) / 8; }
        double before = 0; for (int i = 0; i < size; ++i) before += double(a[i]) * b[i];
        auto rotated = a; rotate_block(rotated, 20260930);
        // Independent dense Hadamard matrix, not the butterfly under test.
        const double norm = 1 / std::sqrt(double(size));
        for (int out = 0; out < size; ++out) {
            double expected = 0; for (int in = 0; in < size; ++in) expected += ((std::popcount(unsigned(out & in)) & 1) ? -1 : 1) * double(a[in]) * rotation_sign(20260930, in);
            expected *= norm;
            if (std::abs(rotated[out] - expected) > 2e-6) return 1;
        }
        rotate_block(a, 20260930); rotate_block(b, 20260930);
        double after = 0; for (int i = 0; i < size; ++i) after += double(a[i]) * b[i];
        if (std::abs(after - before) > 2e-5) return 1;
    }
    if (normalized_scale(0) != 0x5800 || normalized_scale(1e-30f) != 1) return 1;
    const auto scale = tc::gguf::float_to_fp16_rne(128.f);
    for (const auto &[value, expected] : {std::pair{.5f, 0}, {1.5f, 2}, {2.5f, 2}, {-.5f, 0}, {-1.5f, -2}, {-2.5f, -2}, {1000.f, 127}, {-1000.f, -127}})
        if (quantize_rotated(value, scale) != expected) return 1;
    test_s1_math(); test_s1_keys_and_leases(); test_s1_contract();
    std::cout << "PASS independent H128/H512 orthogonality/dense oracle, S1 pre-H128 paired/global-column oracle, identity codes, bounded/finite scales, generation keys/weak leases, gate/up/A8/down contracts, scale tiny/zero, signed RNE ties/clipping\n";
    // All independent Comfy H4^4 basis vectors distinguish ordering/signs
    // from Sylvester; every coefficient is exactly representable in FP32.
    for (int basis=0;basis<256;++basis) {
        std::vector<float> x(256);x[basis]=1;rotate_comfy_block(x,DType::FP32);
        constexpr int h4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
        for (int out=0;out<256;++out) {
            int sign=1;
            for (int shift=0;shift<8;shift+=2) sign*=h4[(out>>shift)&3][(basis>>shift)&3];
            if (x[out]!=sign/16.f || comfy_h256_sign(out,basis)!=sign) return 1;
        }
    }
    if (std::string(convrot_w8a8_recipe)==w8a8_recipe) return 1;
    std::cout << "PASS independent H128/H512 orthogonality/dense oracle, deterministic signs, scale tiny/zero, signed RNE ties/clipping\n";
    std::cout << "PASS independent 256 Comfy H4^4 basis rows and distinct direct-Q8 recipe\n";
}
