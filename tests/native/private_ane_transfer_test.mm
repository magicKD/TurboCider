#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/backends/private/ane_executor.hpp"
#include "../../native/backends/ane_runtime_convert.hpp"
#import <Metal/Metal.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
struct Storage {
    id<MTLBuffer> buffer;
    DeviceMatrixView view;
    Storage(id<MTLDevice> device, int rows, int cols, DType dtype) {
        const size_t item = dtype == DType::FP32 ? 4 : 2, pitch = size_t(cols) * item + 16, offset = 256;
        const size_t size = offset + size_t(rows) * pitch + 256;
        buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("test buffer allocation failed");
        auto retained = std::shared_ptr<void>((__bridge_retained void *)buffer, [](void *p) { CFRelease(p); });
        view = {(__bridge void *)buffer, size, offset, rows, cols, pitch, dtype, retained};
        std::memset(buffer.contents, 0x5a, size);
    }
    void *row(int r) { return static_cast<char *>(buffer.contents) + view.offset_bytes + size_t(r) * view.row_stride_bytes; }
    void guard() {
        auto *p = static_cast<const uint8_t *>(buffer.contents);
        for (size_t i = 0; i < view.offset_bytes; ++i) if (p[i] != 0x5a) throw std::runtime_error("prefix overwrite");
        const size_t item = view.dtype == DType::FP32 ? 4 : 2;
        for (int r = 0; r < view.rows; ++r) for (size_t c = size_t(view.cols) * item; c < view.row_stride_bytes; ++c)
            if (p[view.offset_bytes + size_t(r) * view.row_stride_bytes + c] != 0x5a) throw std::runtime_error("row padding overwrite");
    }
};
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    @autoreleasepool {
      try {
        id<MTLDevice> gpu = MTLCreateSystemDefaultDevice(); Device device;
        uint64_t timeline = 0;
        for (DType dtype : {DType::FP16, DType::BF16, DType::FP32}) {
            const int m = dtype == DType::FP16 ? 1024 : 33, n = dtype == DType::FP16 ? 64 : 65;
            Storage input(gpu, m, n, dtype), output(gpu, m, n, DType::BF16), half(gpu, m, n, DType::FP16);
            Surface surface(device, n, m, Element::FP16);
            for (int r = 0; r < m; ++r) for (int c = 0; c < n; ++c) {
                const unsigned i = unsigned(r * n + c);
                if (dtype == DType::FP16) {
                    uint16_t value = uint16_t(i); if ((value & 0x7c00) == 0x7c00) value = 0;
                    static_cast<uint16_t *>(input.row(r))[c] = value;
                } else {
                    const float values[]{0.f, -0.f, .000000059604645f, -.000000059604645f, 1.00048828125f,
                        1.00146484375f, 65504.f, -65504.f, .3333333f, -.01234567f};
                    const float value = dtype == DType::BF16 && i % 10 == 6 ? 65280.f :
                        dtype == DType::BF16 && i % 10 == 7 ? -65280.f : values[i % 10];
                    if (dtype == DType::FP32) static_cast<float *>(input.row(r))[c] = value;
                    else static_cast<uint16_t *>(input.row(r))[c] = round_bf16(value);
                }
            }
            // No ANE op here: same ready/done is a real GPU-only transpose
            // control. The subsequent test uses distinct ANE event values.
            const auto ready = ++timeline;
            auto job = device.prepare_transfer({{input.view, surface, 0, 1.f}},
                {{surface, output.view, 0, DType::BF16, 64.f}, {surface, half.view, 0, DType::FP16, 1.f}}, ready, ready);
            job.submit(); auto result = job.finish();
            if (!result.ok || job.validation_flags()) throw std::runtime_error("GPU roundtrip validation failed");
            std::vector<uint16_t> expected(n);
            for (int r = 0; r < m; ++r) {
                if (!convert_fp16_row(input.row(r), expected.data(), n, dtype, true)) throw std::runtime_error("CPU oracle overflow");
                for (int c = 0; c < n; ++c) {
                    const auto value = expected[c];
                    if (static_cast<uint16_t *>(half.row(r))[c] != value) throw std::runtime_error("GPU FP16 RNE/subnormal mismatch");
                    const uint16_t wanted = round_bf16(float(std::bit_cast<_Float16>(value)) * 64.f);
                    if (static_cast<uint16_t *>(output.row(r))[c] != wanted) throw std::runtime_error("GPU BF16 headroom RNE mismatch");
                }
            }
            input.guard(); output.guard(); half.guard();
        }
        GraphShape shape{Kind::Matmul, 33, 65, 65, 32, 32, false};
        Program program(device, fp16_program(shape), {}, argv[1]);
        Surface x(device, 65, 33, Element::FP16), w(device, 65, 65, Element::FP16), y(device, 65, 33, Element::FP16);
        std::memset(w.data(), 0, w.rows() * w.pitch());
        for (int c = 0; c < 65; ++c) reinterpret_cast<_Float16 *>(static_cast<char *>(w.data()) + size_t(c) * w.pitch())[c] = 1;
        Storage input(gpu, 66, 65, DType::BF16), output(gpu, 66, 65, DType::BF16);
        for (int r = 0; r < 66; ++r) for (int c = 0; c < 65; ++c)
            static_cast<uint16_t *>(input.row(r))[c] = round_bf16(((r * 3 + c * 7) % 17 - 8) / 8.f);
        std::pair<std::string, Surface> inputs[]{{"x", x}, {"w", w}}, outputs[]{{"y", y}};
        for (int begin : {0, 33}) {
            const uint64_t ready = ++timeline, done = ++timeline;
            auto job = device.prepare_transfer({{input.view, x, begin, 1}}, {{y, output.view, begin, DType::BF16, 1}}, ready, done);
            auto ticket = program.enqueue(inputs, outputs, ready, done, job.failure_callback()); job.submit();
            auto evaluated = ticket.finish(); auto copied = job.finish();
            if (!evaluated.ok || !copied.ok || job.validation_flags()) throw std::runtime_error("real GPU/ANE/GPU transfer failed");
        }
        for (int r = 0; r < 66; ++r) for (int c = 0; c < 65; ++c)
            if (static_cast<uint16_t *>(input.row(r))[c] != static_cast<uint16_t *>(output.row(r))[c]) throw std::runtime_error("real transfer numerical mismatch");
        input.guard(); output.guard();
        {
            setenv("TURBOCIDER_PRIVATE_ANE_GPU_IO", "1", 1);
            PrivateGraph graph({Kind::SwiGLU, 33, 64, 96, 32, 48, true}, 256u << 20,
                               std::filesystem::path(argv[1]) / "executor");
            std::string error;
            if (!graph.supports_device_io() || !graph.self_test(error)) throw std::runtime_error(error);
            std::vector<float> wg(96 * 64, .01f), wu(96 * 64, -.02f), wd(64 * 96, .03f);
            auto weight = [](const std::vector<float> &v, int rows, int cols) { return MatrixView{v.data(), v.size() * 4, rows, cols, 0, DType::FP32}; };
            std::vector<WeightView> weights{weight(wg, 96, 64), weight(wu, 96, 64), weight(wd, 64, 96)};
            Storage gx(gpu, 66, 64, DType::BF16), gy(gpu, 66, 64, DType::BF16),
                dg(gpu, 66, 96, DType::FP32), du(gpu, 66, 96, DType::FP32), gh(gpu, 66, 96, DType::BF16);
            for (int r = 0; r < 66; ++r) {
                std::fill_n(static_cast<uint16_t *>(gx.row(r)), 64, round_bf16(.1f));
                std::fill_n(static_cast<float *>(dg.row(r)), 96, .02f);
                std::fill_n(static_cast<float *>(du.row(r)), 96, -.03f);
            }
            auto run = [&](bool adapter) {
                graph.stage_weights(weights); if (!graph.wait_stage().ok) throw std::runtime_error("GPU executor stage failed");
                graph.launch_device(gx.view, gy.view, adapter ? std::optional<DeviceAdapterInput>({dg.view, du.view, gh.view}) : std::nullopt);
                auto result = graph.finish(); if (!result.ok) throw std::runtime_error(result.error);
                const float x = std::bit_cast<float>(uint32_t(*static_cast<uint16_t *>(gx.row(0))) << 16);
                const float g = x * 64 * wg[0] + (adapter ? .02f : 0), u = x * 64 * wu[0] + (adapter ? -.03f : 0);
                const float h = g / (1 + std::exp(-g)) * u, y = h * 96 * wd[0];
                for (int r = 0; r < 66; ++r) {
                    for (int c = 0; c < 64; ++c) {
                        const float actual = std::bit_cast<float>(uint32_t(static_cast<uint16_t *>(gy.row(r))[c]) << 16);
                        if (std::abs(actual - y) > .0003f + .03f * std::abs(y)) throw std::runtime_error("GPU executor FFN/LoRA oracle mismatch");
                    }
                    if (adapter) for (int c = 0; c < 96; ++c) {
                        const float actual = std::bit_cast<float>(uint32_t(static_cast<uint16_t *>(gh.row(r))[c]) << 16);
                        if (std::abs(actual - h) > .0003f + .03f * std::abs(h)) throw std::runtime_error("GPU executor hidden dtype/scale mismatch");
                    }
                }
                return result;
            };
            run(false); run(true); run(false);
            for (int r = 0; r < 66; ++r) std::fill_n(static_cast<uint16_t *>(gx.row(r)), 64, round_bf16(8));
            std::fill(wu.begin(), wu.end(), 32.f);
            const auto high = run(true);
            if (high.headroom_scale < 4 || !high.overflow_retries) throw std::runtime_error("GPU headroom retry not exercised");
            if (run(false).overflow_retries) throw std::runtime_error("GPU headroom not retained across layers");
            graph.launch_device(gx.view, gx.view);
            if (graph.finish().ok) throw std::runtime_error("GPU alias accepted");
            *static_cast<uint16_t *>(gx.row(0)) = 0x7f80;
            graph.launch_device(gx.view, gy.view);
            if (graph.finish().ok) throw std::runtime_error("GPU nonfinite input accepted");
            *static_cast<uint16_t *>(gx.row(0)) = round_bf16(8);
            run(false);
            gx.guard(); gy.guard(); gh.guard();
            std::cout << "PASS GPU Executor: multi-chunk SwiGLU/base-A-base, packed hidden, BF16 headroom, alias/nonfinite rejection and recovery\n";
        }
        auto invalid = input.view; invalid.buffer_bytes = invalid.offset_bytes + 1;
        try { device.prepare_transfer({{invalid, x, 0, 1}}, {{y, output.view, 0, DType::BF16, 1}}, timeline + 1, timeline + 2); return 1; }
        catch (const CapabilityError &) {}
        // A missing ANE producer MUST suppress the IOSurface read, leaving
        // caller-owned poison untouched even though the wait gets released.
        std::memset(output.buffer.contents, 0x5a, output.buffer.length);
        const uint64_t ready = timeline + 1, missing = timeline + 2, done = timeline + 3;
        auto job = device.prepare_transfer({{input.view, x, 0, 1}}, {{y, output.view, 0, DType::BF16, 1}}, ready, done);
        auto ticket = program.enqueue(inputs, outputs, missing, done, job.failure_callback()); job.submit();
        auto timed = ticket.finish(std::chrono::milliseconds(10)); auto copied = job.finish();
        if (!timed.timed_out || timed.ok || copied.ok) throw std::runtime_error("failed producer published GPU output");
        for (size_t i = 0; i < output.buffer.length; ++i) if (static_cast<uint8_t *>(output.buffer.contents)[i] != 0x5a)
            throw std::runtime_error("failed producer read/overwrote output");
        std::cout << "PASS GPU tiled I/O: all finite FP16 encodings, BF16/F32 RNE, padded strides/offsets/tails, real GPU/ANE/GPU, timeout suppression\n";
      } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
    }
}
