// CPU-only geometry/range contract checks. No Backend/device is constructed.
#include "../../native/backends/metal_graph/kernels/qwen_ops.hpp"

#include <bit>
#include <iostream>
#include <vector>

namespace ops = tc::metal_graph::qwen_ops;
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid native Qwen contract was accepted");
}
float bf16(float value) {
    auto bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}
float qwen_gelu(float input) {
    const float x=bf16(input);
    const float half_x=bf16(0.5f*x);
    const float cube=bf16(std::pow(x,3.f));
    const float inner=bf16(x+bf16(bf16(0.044715f)*cube));
    const float argument=bf16(bf16(0.7978845608028654f)*inner);
    return bf16(half_x*bf16(1.f+bf16(std::tanh(argument))));
}
float qwen_silu(float input) {
    // Host exp tests the BF16 stage boundaries only. It is not an exact oracle
    // for Metal fast::exp near rounding midpoints; actual tc::silu comparison
    // in --device-silu is the separate GPU intrinsic acceptance test.
    const float x=bf16(input);
    const float exponential=bf16(std::exp(std::abs(x)));
    const float denominator=bf16(1.f+exponential);
    const float small=bf16(1.f/denominator);
    const float sigmoid=bf16(x<0 ? small : 1.f-small);
    return bf16(x*sigmoid);
}
void conditioning_fixtures() {
    check(bf16(0.7978845608028654f)==0.796875f &&
          bf16(0.044715f)==0.044677734375f,
          "Qwen text-in GELU scalars must be BF16");
    for (auto pair : {std::pair{-3.f,-0.005859375f}, {-1.f,-0.158203125f},
                      {0.f,0.f}, {1.f,0.84375f}, {1.5f,1.390625f}, {3.f,3.f}})
        check(qwen_gelu(pair.first)==pair.second,"Qwen BF16 text-in GELU fixture changed");
    const float fp32_gelu=bf16(-0.5f*(1.f+std::tanh(0.7978845608028654f*(-1.f-0.044715f))));
    check(fp32_gelu!=qwen_gelu(-1.f),"GELU fixture must distinguish FP32-scalar vision recipe");
    check(qwen_silu(-4.5f)==-0.04931640625f &&
          bf16(-4.5f/(1.f+std::exp(4.5f)))==-0.049560546875f,
          "SiLU must round sigmoid before BF16 product");
    check(qwen_silu(-2.f)==-0.2392578125f && qwen_silu(0.5f)==0.3125f,
          "MLX BF16 sigmoid exp/denominator/reciprocal boundaries changed");
    const float prior_negative=bf16(-2.f*bf16(1.f/(1.f+std::exp(2.f))));
    const float prior_positive=bf16(0.5f*bf16(1.f-1.f/(1.f+std::exp(0.5f))));
    check(prior_negative==-0.23828125f && prior_positive==0.310546875f &&
          prior_negative!=qwen_silu(-2.f) && prior_positive!=qwen_silu(0.5f),
          "SiLU fixtures must reject the previous FP32-interior recipe");
    check(qwen_silu(0.f)==0.f && qwen_silu(100.f)==100.f &&
          std::isfinite(qwen_silu(-100.f)),"stable SiLU finite-input tails");

    // The condition norm retains FP32 normalization and shifted weight until
    // one final cast; copying the Q/K pre-weight BF16 rule is incorrect here.
    const float normalized=1.00390625f, shifted_weight=bf16(0.5f)+1.f;
    check(bf16(normalized*shifted_weight)==1.5078125f &&
          bf16(bf16(normalized)*shifted_weight)==1.5f,
          "condition norm final-only BF16 fixture changed");
    check(bf16(normalized*(bf16(-1.f)+1.f))==0.f,
          "condition norm weight is shifted by FP32 one");
    for (uint32_t prefix : {0U,1U,3U}) ops::validate_modulation_rows(3,prefix);
    rejects([] { ops::validate_modulation_rows(0,0); });
    rejects([] { ops::validate_modulation_rows(3,4); });
    rejects([] { ops::validate_modulation_rows(3,UINT32_MAX); });
    check(ops::conditioning_recipe.ends_with("-v3") &&
          ops::conditioning_recipe.find("fast-exp")!=std::string_view::npos,
          "compiled MLX exp variant needs independent conditioning v3");
}
void math_fixtures() {
    // Every input is exactly BF16. One sparse 8-unit deviation at a high mean
    // must survive normalization; the raw-second-moment shortcut loses it.
    std::vector<float> values(4096, 1024.f);
    values.back() = 1032.f;
    float sum = 0.f, raw_square = 0.f;
    for (float x : values) { sum += x; raw_square += x*x; }
    const float mean = sum / 4096.f;
    float centered_square = 0.f;
    for (float x : values) centered_square += (x-mean)*(x-mean);
    const float variance = centered_square / 4096.f;
    check(mean == 1024.001953125f && variance > 0.015f && variance < 0.016f,
          "high-mean LayerNorm fixture must preserve centered variance");
    check(raw_square/4096.f - mean*mean <= 0.f,
          "high-mean fixture no longer exposes raw-moment cancellation");
    const float inv = 1.f/std::sqrt(variance+1e-6f);
    check(bf16((values.back()-mean)*inv) == 64.f &&
          bf16((values.front()-mean)*inv) == -0.015625f,
          "high-mean BF16 LayerNorm fixture output changed");

    // Both Q and K need normalization -> BF16 -> weight -> BF16. A non-unit
    // weight makes the omitted pre-weight rounding observable on a tie.
    for (float sign : {1.f,-1.f}) {
        const float normalized = sign*1.00390625f;
        const float weighted = bf16(bf16(normalized)*1.5f);
        const float late_round = bf16(normalized*1.5f);
        check(weighted == sign*1.5f && late_round == sign*1.5078125f,
              "RMSNorm pre-weight BF16 fixture no longer distinguishes recipes");
    }
    check(ops::precision_recipe.ends_with("-v2"), "changed rounding requires v2 recipe identity");
}
}
int main() {
    math_fixtures();
    conditioning_fixtures();
    for (auto geometry : {ops::Geometry{}, ops::Geometry{128,1,128}, {3840,30,128}})
        ops::validate_geometry(geometry);
    for (auto geometry : {ops::Geometry{0,0,128}, {4096,32,64}, {128,2,128}, {4224,33,128}})
        rejects([&] { ops::validate_geometry(geometry); });
    ops::validate_epsilon(1e-6f);
    for (float epsilon : {0.f, -1.f, 1e-5f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()})
        rejects([&] { ops::validate_epsilon(epsilon); });

    // Physical row padding and a sliced buffer offset are supported. Exclude
    // final-row padding from the used span, but retain all inter-row gaps.
    check(ops::matrix_extent(3,128,272,2,16,800) == 800, "padded matrix extent");
    rejects([] { ops::matrix_extent(3,128,272,2,16,799); });
    rejects([] { ops::matrix_extent(3,128,254,2,0,1024); });
    rejects([] { ops::matrix_extent(3,128,257,2,0,1024); });
    rejects([] { ops::matrix_extent(3,128,256,2,1,1024); });
    rejects([] { ops::matrix_extent(0,128,256,2,0,1024); });
    rejects([] { ops::matrix_extent(1,0,256,2,0,1024); });
    rejects([] { ops::matrix_extent(1,128,256,0,0,1024); });
    rejects([] { ops::matrix_extent(1,128,(size_t(UINT32_MAX)+1)*2,2,0,1024); });
    rejects([] { ops::matrix_extent(UINT32_MAX,128,size_t(UINT32_MAX)*4,4,0,SIZE_MAX); });

    // A prefix token range keeps a larger source head stride. Distinct heads
    // may have gaps, but overlapping head storage and truncated tails reject.
    check(ops::head_extent(2,3,128,272,2048,2,16,2848) == 2848, "head subview extent");
    rejects([] { ops::head_extent(2,3,128,272,2048,2,16,2847); });
    rejects([] { ops::head_extent(2,3,128,272,798,2,0,4096); });
    rejects([] { ops::head_extent(2,3,128,272,2049,2,0,4096); });
    rejects([] { ops::head_extent(0,3,128,272,2048,2,0,4096); });
    check(!ops::ranges_overlap(16,128,144,64), "adjacent views must not alias");
    check(ops::ranges_overlap(16,128,143,64), "overlapping last byte must reject");
    check(ops::ranges_overlap(32,16,16,128), "contained view must reject");
    check(!ops::ranges_overlap(SIZE_MAX-64,16,0,32), "range comparison must not overflow");
    std::cout << "qwen_native_ops_cpu_contract=PASS device_created=0\n";
}
