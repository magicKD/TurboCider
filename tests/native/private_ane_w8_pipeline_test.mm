#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
struct Buffer {
    id<MTLBuffer> native;
    std::shared_ptr<void> owner;
    Buffer(id<MTLDevice> d, size_t bytes) {
        native = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!native) throw std::runtime_error("test buffer allocation failed");
        owner = {(__bridge_retained void *)native, [](void *p) { CFRelease(p); }};
        std::memset(native.contents, 0, bytes);
    }
};
int ulp16(uint16_t a, uint16_t b) {
    const auto ordered = [](uint16_t v) { return v & 0x8000 ? 0x8000 - int(v & 0x7fff) : 0x8000 + int(v); };
    return std::abs(ordered(a) - ordered(b));
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    @autoreleasepool {
      try {
        auto gpu = MTLCreateSystemDefaultDevice(); Device device;
        constexpr int m = 33, k = 512, n = 16, physical = 1024;
        Buffer wa(gpu, (n + 2) * physical * 2), wb(gpu, (n + 2) * physical * 2), input(gpu, m * k * 2);
        for (int r = 0; r < n + 2; ++r) for (int c = 0; c < physical; ++c) {
            static_cast<uint16_t *>(wa.native.contents)[r * physical + c] = tc::gguf::float_to_bf16_rne(r ? float((r * 13 + c * 7) % 31 - 15) / 64 : 0);
            static_cast<uint16_t *>(wb.native.contents)[r * physical + c] = tc::gguf::float_to_bf16_rne(r ? float((r * 11 + c * 3) % 29 - 14) / 32 : 0);
        }
        for (int r = 0; r < m; ++r) for (int c = 0; c < k; ++c)
            static_cast<uint16_t *>(input.native.contents)[r * k + c] = tc::gguf::float_to_bf16_rne(r ? float((r * 3 + c * 11) % 23 - 11) / 32 : 0);
        auto weight = [&](Buffer &b) { return DeviceWeightView{(__bridge void *)b.native, b.native.length, 0, physical * 2,
            n + 2, physical, DeviceWeightEncoding::Dense, DType::BF16, 32, {}, {}, b.owner}; };
        DeviceWeightView xview{(__bridge void *)input.native, input.native.length, 0, k * 2, m, k,
            DeviceWeightEncoding::Dense, DType::BF16, 32, {}, {}, input.owner};
        // EXACTLY two reusable W8 banks, one A8 activation set. Neither graph
        // embeds a checkpoint, and preparing bank1 does not overwrite bank0.
        std::array<Surface, 2> w{Surface(device, n, k, Element::I8), Surface(device, n, k, Element::I8)};
        std::array<Surface, 2> sw{Surface(device, n, 1, Element::FP16), Surface(device, n, 1, Element::FP16)};
        Surface x(device, k, m, Element::I8), sx(device, 1, m, Element::FP16), y(device, n, m, Element::FP16);
        auto xs = device.stage_w8(xview, {0, m, 0, k, 128, 20260930, true}, x, sx);
        auto a = device.stage_w8(weight(wa), {0, n, 128, k, 128}, w[0], sw[0]);
        if (!xs.finish().ok || !a.finish().ok) throw std::runtime_error("initial W8/A8 GPU stage failed");
        Program program(device, w8_matmul_program({Kind::Matmul, m, k, n, k, n, false}), {}, argv[1]);
        Buffer output(gpu, m * n * 2);
        DeviceMatrixView out{(__bridge void *)output.native, output.native.length, 0, m, n, n * 2, DType::BF16, output.owner};
        uint64_t timeline = 0;
        int worst_ulp = 0;
        auto check = [&](int bank) {
            for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
                int64_t sum = 0;
                for (int kk = 0; kk < k; ++kk)
                    sum += int64_t(static_cast<const int8_t *>(w[bank].data())[col * w[bank].pitch() + kk]) *
                        int64_t(static_cast<const int8_t *>(x.data())[kk * x.pitch() + row]);
                const auto expected = tc::gguf::float_to_fp16_rne(float(sum) / 16384.f);
                const auto actual = reinterpret_cast<const uint16_t *>(static_cast<const char *>(y.data()) + col * y.pitch())[row];
                const int ulp = ulp16(actual, expected); worst_ulp = std::max(worst_ulp, ulp);
                if (ulp > 2) throw std::runtime_error("normalized W8A8 integer oracle exceeds2 ULP: " + std::to_string(ulp));
                const auto ws = *reinterpret_cast<const uint16_t *>(static_cast<const char *>(sw[bank].data()) + col * sw[bank].pitch());
                const auto as = static_cast<const uint16_t *>(sx.data())[row];
                // Epilogue oracle follows the measured normalized ANE value,
                // not the integer oracle, preserving the FP32 multiply order.
                const float recovered = (tc::gguf::fp16_to_float(actual) * tc::gguf::fp16_to_float(ws)) * tc::gguf::fp16_to_float(as);
                const auto expected_bf16 = tc::gguf::float_to_bf16_rne(recovered);
                if (static_cast<const uint16_t *>(output.native.contents)[row * n + col] != expected_bf16)
                    throw std::runtime_error("GPU W8 epilogue rounding/scale order mismatch");
            }
        };
        for (int trial = 0; trial < 3; ++trial) {
            const int bank = trial == 1 ? 1 : 0;
            const uint64_t ready = ++timeline, done = ++timeline;
            std::pair<std::string, Surface> inputs[]{{"w", w[bank]}, {"x", x}}, outputs[]{{"y", y}};
            auto io = device.prepare_transfer({}, {{y, out, 0, DType::BF16, 1.f, sw[bank], sx}}, ready, done);
            auto request = program.enqueue(inputs, outputs, ready, done, io.failure_callback()); io.submit();
            // Next bank uses its OWN ready event/queue while current ANE work
            // is in flight. Readiness must not raise the current done value.
            std::optional<QuantStage> next;
            if (!trial) next = device.stage_w8(weight(wb), {0, n, 128, k, 128}, w[1], sw[1]);
            if (!request.finish().ok || !io.finish().ok || io.validation_flags()) throw std::runtime_error("W8A8 evaluation/epilogue failed");
            check(bank);
            if (next && !next->finish().ok) throw std::runtime_error("second bank prefetch failed");
        }
        std::cout << "PASS private dynamic W8A8 GPU-stage/ANE/GPU-FP32 epilogue, H128, two banks A/B/A, integer normalized oracle worst_ulp=" << worst_ulp
            << "; arithmetic placement and full FFN/model qualification NOT established\n";
      } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
    }
}
