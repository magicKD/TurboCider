#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
struct Buffer {
    id<MTLBuffer> value;
    std::shared_ptr<void> owner;
    Buffer(id<MTLDevice> d, size_t bytes) {
        value = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!value) throw std::runtime_error("test buffer allocation failed");
        owner = {(__bridge_retained void *)value, [](void *p) { CFRelease(p); }}; std::memset(value.contents, 0, bytes);
    }
    DeviceMatrixView matrix(int rows, int cols, DType dtype = DType::FP32) {
        return {(__bridge void *)value, value.length, 0, rows, cols, size_t(cols) * (dtype == DType::FP32 ? 4 : 2), dtype, owner};
    }
    DeviceWeightView weight(int rows, int cols) {
        return {(__bridge void *)value, value.length, 0, size_t(cols) * 4, rows, cols, DeviceWeightEncoding::Dense, DType::FP32, 32, {}, {}, owner};
    }
};
float half_value(Surface &s, int row, int col) { return tc::gguf::fp16_to_float(reinterpret_cast<const uint16_t *>(static_cast<const char *>(s.data()) + size_t(row) * s.pitch())[col]); }
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    @autoreleasepool {
      try {
        constexpr int m = 33, h = 128, f = 512;
        auto gpu = MTLCreateSystemDefaultDevice(); Device device;
        Buffer xbuf(gpu, m * h * 4), wg(gpu, f * h * 4), wu(gpu, f * h * 4), wd(gpu, h * f * 4),
            dg(gpu, m * f * 4), du(gpu, m * f * 4), out(gpu, m * h * 2), hout(gpu, m * f * 2);
        for (int r = 0; r < m; ++r) for (int c = 0; c < h; ++c)
            static_cast<float *>(xbuf.value.contents)[r * h + c] = r == 0 ? 0.f : r == 1 ? 1e-20f : ((r * 3 + c * 7) % 17 - 8) / 8.f;
        for (int r = 0; r < f; ++r) {
            static_cast<float *>(wg.value.contents)[r * h + r % h] = .125f;
            static_cast<float *>(wu.value.contents)[r * h + r % h] = -.25f;
        }
        for (int r = 0; r < h; ++r) static_cast<float *>(wd.value.contents)[r * f + r] = .25f;
        for (int r = 0; r < m; ++r) for (int c = 0; c < f; ++c) {
            static_cast<float *>(dg.value.contents)[r * f + c] = .02f * ((r + c) % 3 - 1);
            static_cast<float *>(du.value.contents)[r * f + c] = .03f * ((r * 2 + c) % 3 - 1);
        }
        Surface x(device, h, m, Element::I8), tx(device, 1, m, Element::FP16),
            g(device, f, h, Element::I8), sg(device, f, 1, Element::FP16),
            u(device, f, h, Element::I8), su(device, f, 1, Element::FP16),
            d(device, h, f, Element::I8), sd(device, h, 1, Element::FP16),
            delta_g(device, f, m, Element::FP16), delta_u(device, f, m, Element::FP16), ones(device, f, 1, Element::FP16);
        for (int r = 0; r < f; ++r) *reinterpret_cast<uint16_t *>(static_cast<char *>(ones.data()) + r * ones.pitch()) = 0x3c00;
        auto xs = device.stage_w8(xbuf.weight(m, h), {0, m, 0, h, 128, 20260930, true}, x, tx);
        auto gs = device.stage_w8(wg.weight(f, h), {0, f, 0, h, 128}, g, sg);
        auto us = device.stage_w8(wu.weight(f, h), {0, f, 0, h, 128}, u, su);
        auto ds = device.stage_w8(wd.weight(h, f), {0, h, 0, f, 512}, d, sd);
        if (!xs.finish().ok || !gs.finish().ok || !us.finish().ok || !ds.finish().ok) throw std::runtime_error("FFN GPU staging failed");
        uint64_t timeline = 0;
        for (bool adapter : {false, true}) {
            const auto emitted = w8_swiglu_program({Kind::SwiGLU, m, h, f, 512, 512, adapter}, 20260930, 1.f);
            Program program(device, emitted.mil, emitted.constants, std::filesystem::path(argv[1]) / (adapter ? "lora" : "base"));
            Surface y(device, emitted.packed_rows, m, Element::FP16);
            std::vector<std::pair<std::string, Surface>> inputs{{"x", x}, {"tx", tx}, {"wg", g}, {"sg", sg}, {"wu", u}, {"su", su}, {"wd", d}};
            std::vector<Upload> upload;
            if (adapter) { inputs.emplace_back("dg", delta_g); inputs.emplace_back("du", delta_u);
                upload.push_back({dg.matrix(m, f), delta_g}); upload.push_back({du.matrix(m, f), delta_u}); }
            auto down = y.slice_rows(0, h), hs = y.slice_rows(h, 1);
            std::vector<Download> download{{down, out.matrix(m, h, DType::BF16), 0, DType::BF16, emitted.headroom, sd, hs}};
            if (adapter) download.push_back({y.slice_rows(h + 1, f), hout.matrix(m, f, DType::BF16), 0, DType::BF16, emitted.headroom});
            const uint64_t ready = ++timeline, done = ++timeline;
            auto io = device.prepare_transfer(std::move(upload), std::move(download), ready, done);
            std::pair<std::string, Surface> outputs[]{{"y", y}};
            auto request = program.enqueue(inputs, outputs, ready, done, io.failure_callback()); io.submit();
            if (!request.finish().ok || !io.finish().ok || io.validation_flags()) throw std::runtime_error("W8A8 SwiGLU evaluation/epilogue failed");
            double error = 0, norm = 0, hidden_error = 0, hidden_norm = 0;
            // Independent dense source oracle. Extra A8/FP16/conv order is a
            // model approximation, not an integer-MAC or exact FFN assertion.
            for (int r = 0; r < m; ++r) for (int c = 0; c < h; ++c) {
                const float xv = static_cast<const float *>(xbuf.value.contents)[r * h + c];
                const float gv = xv * .125f + (adapter ? static_cast<const float *>(dg.value.contents)[r * f + c] : 0);
                const float uv = xv * -.25f + (adapter ? static_cast<const float *>(du.value.contents)[r * f + c] : 0);
                const float hidden = gv / (1 + std::exp(-gv)) * uv, expected = hidden * .25f;
                const float actual = std::bit_cast<float>(uint32_t(static_cast<const uint16_t *>(out.value.contents)[r * h + c]) << 16);
                if (!std::isfinite(actual)) throw std::runtime_error("nonfinite full FFN output");
                error += double(actual - expected) * (actual - expected); norm += double(expected) * expected;
                if (adapter) {
                    const float hv = std::bit_cast<float>(uint32_t(static_cast<const uint16_t *>(hout.value.contents)[r * f + c]) << 16);
                    int64_t gi = 0, ui = 0;
                    for (int kk = 0; kk < h; ++kk) {
                        const int qx = static_cast<const int8_t *>(x.data())[kk * x.pitch() + r];
                        gi += int64_t(qx) * static_cast<const int8_t *>(g.data())[c * g.pitch() + kk];
                        ui += int64_t(qx) * static_cast<const int8_t *>(u.data())[c * u.pitch() + kk];
                    }
                    const float gg = float(gi) / 16384 * half_value(sg, c, 0) * half_value(tx, 0, r) + static_cast<const float *>(dg.value.contents)[r * f + c];
                    const float uu = float(ui) / 16384 * half_value(su, c, 0) * half_value(tx, 0, r) + static_cast<const float *>(du.value.contents)[r * f + c];
                    const float execution = gg / (1 + std::exp(-gg)) * uu;
                    if (!std::isfinite(hv) || std::abs(hv - execution) > .0003f + .04f * std::abs(execution))
                        throw std::runtime_error("corrected hidden execution oracle mismatch");
                    hidden_error += double(hv - hidden) * (hv - hidden); hidden_norm += double(hidden) * hidden;
                }
            }
            const double relative = std::sqrt(error / norm);
            const double hidden_relative = hidden_norm ? std::sqrt(hidden_error / hidden_norm) : 0;
            if (hidden_relative > .05) throw std::runtime_error("hidden source relative L2 failure: " + std::to_string(hidden_relative));
            if (relative > .05) {
                std::cerr << "DEBUG sg=" << half_value(sg, 0, 0) << " su=" << half_value(su, 0, 0) << " sd=" << half_value(sd, 0, 0)
                    << " tx=" << half_value(tx, 0, 2) << " hs=" << half_value(y, h, 2) << " norm=" << half_value(y, 0, 2)
                    << " out=" << std::bit_cast<float>(uint32_t(static_cast<const uint16_t *>(out.value.contents)[2 * h]) << 16) << "\n";
                throw std::runtime_error("W8A8 FFN relative L2 failure: " + std::to_string(relative));
            }
            std::cout << "PASS full private W8A8 SwiGLU, ANE H512/A8, GPU FP32 epilogue, adapter=" << adapter << " relative_l2=" << relative
                << " hidden_source_l2=" << hidden_relative << "\n";
        }
      } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
    }
}
