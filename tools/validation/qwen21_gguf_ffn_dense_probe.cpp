#include "runtime/streaming/gguf_packed_bank.hpp"
#include "backends/ane_memory.hpp"
#include "qwen21_gguf_u4_mpp.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace tc;
using namespace tc::streaming;

namespace {
using Function = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
double relative(const Tensor &a, const Tensor &b) {
    auto af = mx::astype(a, mx::float32), bf = mx::astype(b, mx::float32);
    return mx::sqrt(mx::sum(mx::square(af - bf)) /
        mx::maximum(mx::sum(mx::square(bf)), Tensor(1e-20f))).item<float>();
}
Tensor ffn(const Tensor &input, const Tensor &gu, const Tensor &down) {
    auto pair = mx::split(mx::matmul(input, mx::transpose(gu)), 2, -1);
    return mx::matmul(silu(pair[0]) * pair[1], mx::transpose(down));
}
}

int main(int argc, char **argv) try {
    require(argc == 3 || argc == 4, "usage: qwen21-gguf-ffn-dense-probe original-source dit|encoder [--direct-u4]");
    const bool direct = argc == 4;
    require(!direct || std::string_view(argv[3]) == "--direct-u4", "invalid direct U4 option");
    const std::string component = argv[2];
    require(component == "dit" || component == "encoder", "invalid component");
    configure_streams(); mx::set_cache_limit(0);
    SourceFileIdentity source; source.logical_id = "component"; source.path = argv[1];
    auto lease = SourceLease::capture_registered({source});
    const std::string stem = component == "dit" ?
        "model.diffusion_model.transformer_blocks.0.img_mlp." : "blk.0.ffn_";
    const std::string gate = stem + (component == "dit" ? "gate_up" : "gate");
    const std::string up = stem + "up";
    const std::string down = stem + (component == "dit" ? "out" : "down");
    const std::string fused = component == "dit" ? gate : "probe.gate_up";
    GgufKImportOptions options; options.enabled = true; options.floating_dtype = mx::float16;
    options.decode_workers = 8;
    options.include_tensor = [&](std::string_view key) {
        return key == gate + ".weight" || key == down + ".weight" ||
            (component == "encoder" && key == up + ".weight");
    };
    MemoryLedger ledger(uint64_t(1) << 30); Weights weights;
    GgufPackedBank bank(lease, "component", ledger, uint64_t(1) << 20, true, 0, 6, options);
    bank.load(weights);
    if (component == "encoder") {
        for (const auto *plane : {".weight", ".scales", ".biases"})
            weights.fuse_keys(fused + plane, {gate + plane, up + plane}, 0);
    }
    weights.materialize(); bank.check_unchanged();
    require(weights.quantized(fused) && weights.quantized(down), "original affine FFN matrices required");
    const auto &gc = weights.at(fused + ".weight"), &gs = weights.at(fused + ".scales"),
               &gb = weights.at(fused + ".biases");
    const auto &dc = weights.at(down + ".weight"), &ds = weights.at(down + ".scales"),
               &db = weights.at(down + ".biases");
    const int hidden = dc.shape(0), intermediate = ds.shape(1) * 32;
    const int gate_bits = gc.shape(1) * 32 / hidden, down_bits = dc.shape(1) * 32 / intermediate;
    require(hidden == 4096 && gc.shape(0) == 2 * intermediate && gs.shape(1) * 32 == hidden &&
        (gate_bits == 4 || gate_bits == 8) && (down_bits == 4 || down_bits == 8), "unexpected source FFN geometry");
    const uint64_t dense_bytes = uint64_t(2) * (uint64_t(gc.shape(0)) * hidden + uint64_t(hidden) * intermediate);
    auto observed = ane::observe_runtime_memory(mx::get_active_memory());
    // At most two dense versions of this ONE block plus bounded activations.
    auto admission = ane::admit_memory(observed, {uint64_t(4) << 30, observed.physical_bytes}, 0,
        dense_bytes * 2 + (uint64_t(1) << 30));
    require(admission.allowed(), "bounded FFN probe physical admission declined");
    auto decode = [](const Tensor &codes, const Tensor &scales, const Tensor &biases, int bits) {
        return mx::dequantize(codes, scales, biases, 32, bits, "affine", std::nullopt, mx::float16);
    };
    const auto decode_start = Clock::now();
    auto dense_gu = decode(gc, gs, gb, gate_bits), dense_down = decode(dc, ds, db, down_bits);
    mx::eval({dense_gu, dense_down}); mx::synchronize();
    const auto decode_seconds = std::chrono::duration<double>(Clock::now() - decode_start).count();
    const auto dense_gu_shape = dense_gu.shape(), dense_down_shape = dense_down.shape();
    require(dense_gu.dtype() == mx::float16 && dense_down.dtype() == mx::float16 &&
        dense_gu.shape() == mx::Shape{2 * intermediate, hidden} &&
        dense_down.shape() == mx::Shape{hidden, intermediate}, "dequantized typed geometry mismatch");
    Function packed = mx::compile([&weights, fused, down](const std::vector<Tensor> &a) {
        auto pair = mx::split(linear(a[0], weights, fused), 2, -1);
        return std::vector<Tensor>{linear(silu(pair[0]) * pair[1], weights, down)};
    });
    Function cached = mx::compile([](const std::vector<Tensor> &a) {
        return std::vector<Tensor>{ffn(a[0], a[1], a[2])};
    });
    Function fresh = mx::compile([decode, gate_bits, down_bits](const std::vector<Tensor> &a) {
        // Packed inputs are DYNAMIC arguments, not captured constants. A
        // compiler-folded/all-model dense cache must not masquerade as fresh decode.
        return std::vector<Tensor>{ffn(a[0], decode(a[1], a[2], a[3], gate_bits),
                                           decode(a[4], a[5], a[6], down_bits))};
    });
    require(!direct || (gate_bits == 4 && down_bits == 4), "direct U4 probe does not support Q8 planes");
    Function direct_fn;
    if (direct) direct_fn = mx::compile([](const std::vector<Tensor> &a) {
        auto pair = mx::split(qwen21_probe::u4_projection(a[0], a[1], a[2], a[3]), 2, -1);
        return std::vector<Tensor>{qwen21_probe::u4_projection(silu(pair[0]) * pair[1], a[4], a[5], a[6])};
    });
    std::cout << std::setprecision(12) << "{\"schema\":\"tc-qwen21-real-ffn-dense-screen-v1\",\"qualification_passed\":false"
        << ",\"scope\":\"one complete original layer0 FFN with synthetic FP16 inputs; no whole-model or ahead-decode qualification\""
        << ",\"component\":\"" << component << "\",\"source_sha256\":\"" << lease->file("component").content_digest
        << "\",\"verification_bytes\":" << lease->verification_bytes_read()
        << ",\"registration_cache_hits\":" << lease->registration_cache_hits()
        << ",\"source_read_bytes\":" << bank.metrics().source_read_bytes
        << ",\"packed_capacity_bytes\":" << bank.metrics().packed_capacity_bytes
        << ",\"selected_tensor_count\":" << bank.metrics().tensor_count
        << ",\"dense_block_bytes\":" << dense_bytes << ",\"dense_blocks_retained\":1"
        << ",\"fresh_packed_arguments_dynamic\":true,\"group_size\":32"
        << ",\"direct_u4_mpp\":" << (direct ? "true" : "false")
        << ",\"gate_bits\":" << gate_bits << ",\"down_bits\":" << down_bits
        << ",\"K\":" << hidden << ",\"intermediate\":" << intermediate
        << ",\"initial_dequantize_seconds\":" << decode_seconds << ",\"cases\":[";
    bool comma = false;
    for (int rows : {32, 38, 128, 512, 1024, 1048, 2112, 4096}) {
        auto input = mx::astype(mx::reshape(mx::sin(mx::arange(rows * hidden, mx::float32) * .0017f) * .25f,
                                           {1, rows, hidden}), mx::float16);
        mx::eval(input); mx::synchronize();
        const std::vector<Tensor> ca{input, dense_gu, dense_down}, fa{input, gc, gs, gb, dc, ds, db};
        auto invoke = [&](int mode) { return mode == 0 ? packed({input}) : mode == 1 ? fresh(fa) :
                                            mode == 2 ? cached(ca) : direct_fn(fa); };
        auto expected = invoke(0)[0]; mx::eval(expected);
        auto actual = invoke(1)[0], held = invoke(2)[0]; mx::eval({actual, held});
        const double fresh_error = relative(actual, expected), cached_error = relative(held, expected);
        const double variants_error = relative(actual, held);
        double direct_error = 0;
        if (direct) {
            auto candidate = invoke(3)[0]; mx::eval(candidate); direct_error = relative(candidate, expected);
            require(candidate.shape() == expected.shape() && candidate.dtype() == mx::float16 &&
                mx::all(mx::isfinite(candidate)).item<bool>() && std::isfinite(direct_error) && direct_error <= .05,
                "direct U4 complete FFN numerical budget failed");
        }
        require(actual.shape() == expected.shape() && actual.dtype() == mx::float16 &&
            mx::all(mx::isfinite(actual)).item<bool>() && mx::all(mx::isfinite(held)).item<bool>() &&
            std::isfinite(fresh_error) && std::isfinite(cached_error) &&
            fresh_error <= .05 && cached_error <= .05 && variants_error == 0,
            "complete FFN dense projection numerical budget or fresh/cached identity failed");
        const int modes = direct ? 4 : 3;
        for (int warm = 0; warm < 3; ++warm) for (int mode = 0; mode < modes; ++mode) mx::eval(invoke(mode));
        std::vector<double> samples[4];
        for (int sample = 0; sample < 9; ++sample) for (int visit = 0; visit < modes; ++visit) {
            const int mode = (sample + visit) % modes; mx::synchronize();
            const auto started = Clock::now(); mx::eval(invoke(mode)); mx::synchronize();
            samples[mode].push_back(std::chrono::duration<double>(Clock::now() - started).count());
        }
        if (comma) std::cout << ','; comma = true;
        std::cout << "{\"M\":" << rows << ",\"complete_output_elements\":" << expected.size()
            << ",\"fresh_relative_l2\":" << fresh_error << ",\"cached_relative_l2\":" << cached_error
            << ",\"fresh_cached_relative_l2\":" << variants_error;
        if (direct) std::cout << ",\"direct_u4_relative_l2\":" << direct_error;
        for (int mode = 0; mode < modes; ++mode) {
            const char *name = mode == 0 ? "packed_seconds" : mode == 1 ? "fresh_dequant_dense_seconds" :
                                         mode == 2 ? "cached_dense_seconds" : "direct_u4_mpp_seconds";
            std::cout << ",\"" << name << "\":[";
            for (size_t i = 0; i < samples[mode].size(); ++i) std::cout << (i ? "," : "") << samples[mode][i];
            std::cout << ']';
        }
        std::cout << '}'; bank.check_unchanged();
    }
    bank.check_unchanged(); lease->revalidate_after_drain();
    require(dense_gu.shape() == dense_gu_shape && dense_down.shape() == dense_down_shape, "probe changed dense cache geometry");
    std::cout << "],\"warmups\":3,\"hot_samples\":9,\"mlx_logical_peak_bytes\":" << mx::get_peak_memory() << "}\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
