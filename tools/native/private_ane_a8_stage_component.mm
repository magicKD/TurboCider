// Short, explicitly scheduled GPU staging component. No model or ANE request.
// Reuses the production Device::stage_w8 API and source-independent surfaces.
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

#pragma clang fp contract(off)
using namespace tc::ane;
using namespace tc::ane::private_api;
namespace {
using Clock = std::chrono::steady_clock;
void check(bool value, const char *reason) { if (!value) throw std::runtime_error(reason); }
uint32_t random_word(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
struct Source {
    id<MTLBuffer> buffer;
    DeviceWeightView view;
    Source(id<MTLDevice> gpu, int rows, int cols, DType dtype) {
        const size_t item = dtype == DType::FP32 ? 4 : 2, pitch = size_t(cols) * item + 32;
        buffer = [gpu newBufferWithLength:256 + size_t(rows + 1) * pitch + 64 options:MTLResourceStorageModeShared];
        check(buffer != nil, "component source allocation failed");
        auto owner = std::shared_ptr<void>((__bridge_retained void *)buffer, [](void *p) { CFRelease(p); });
        view = {(__bridge void *)buffer, buffer.length, 256, pitch, rows + 1, cols, DeviceWeightEncoding::Dense, dtype, 32, {}, {}, owner, owner};
        std::memset(buffer.contents, 0x5a, buffer.length);
        for (int row = 0; row <= rows; ++row) for (int col = 0; col < cols; ++col) {
            float value = float(int(random_word(uint32_t(row * cols + col)) % 2048) - 1024) / 256;
            if (row == 1) value = 0;
            else if (row == 2) value = (col & 1) ? 0x1p-24f : -0x1p-24f;
            else if (row == 3) value = col == cols - 128 + 32 ? 123.f : col == 31 ? -.5f : 0;
            put(row, col, value);
        }
    }
    void put(int row, int col, float value) {
        auto *at = static_cast<uint8_t *>(buffer.contents) + view.offset_bytes + size_t(row) * view.row_stride_bytes;
        if (view.dense_dtype == DType::FP32) std::memcpy(at + size_t(col) * 4, &value, 4);
        else { const auto bits = tc::gguf::float_to_bf16_rne(value); std::memcpy(at + size_t(col) * 2, &bits, 2); }
    }
    float value(int row, int col) const {
        const auto *at = static_cast<const uint8_t *>(buffer.contents) + view.offset_bytes + size_t(row) * view.row_stride_bytes;
        if (view.dense_dtype == DType::FP32) { float out; std::memcpy(&out, at + size_t(col) * 4, 4); return out; }
        uint16_t bits; std::memcpy(&bits, at + size_t(col) * 2, 2); return std::bit_cast<float>(uint32_t(bits) << 16);
    }
};
struct Scale {
    id<MTLBuffer> buffer;
    DeviceMatrixView view;
    std::vector<float> values;
    Scale(id<MTLDevice> gpu, int cols) : values(cols) {
        buffer = [gpu newBufferWithLength:size_t(cols) * 4 options:MTLResourceStorageModeShared];
        check(buffer != nil, "component S1 allocation failed");
        auto owner = std::shared_ptr<void>((__bridge_retained void *)buffer, [](void *p) { CFRelease(p); });
        view = {(__bridge void *)buffer, buffer.length, 0, 1, cols, size_t(cols) * 4, DType::FP32, owner, owner};
        for (int col = 0; col < cols; ++col) values[col] = col % 11 == 0 ? .3f : col % 11 == 1 ? 1.7f : std::ldexp(1.f, col % 9 - 4);
        std::memcpy(buffer.contents, values.data(), buffer.length);
    }
};
struct Output {
    Surface codes, scales;
    Output(Device &device, int rows, int cols) : codes(device, cols, rows, Element::I8), scales(device, 1, rows, Element::FP16) {}
    void reset() { std::memset(codes.data(), 0x5a, codes.rows() * codes.pitch()); std::memset(scales.data(), 0x5a, scales.rows() * scales.pitch()); }
};
struct Budget { double total_ms = 0; unsigned commands = 0; } budget;
struct Staged { double ms; uint32_t flags; };
Staged stage(Device &device, const DeviceWeightView &source, const W8StageSpec &spec, Output &output, bool candidate, uint32_t expected_flags = 0) {
    check(budget.total_ms < 18000, "component staging budget exhausted; no further submissions");
    output.reset();
    const auto timeline = device.value();
    const auto start = Clock::now();
    auto job = device.stage_w8(source, spec, output.codes, output.scales); ++budget.commands;
    check(job.single_pass() == candidate, "actual selected pipeline differs from requested comparison");
    const auto finished = job.finish(std::chrono::seconds(2));
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count(); budget.total_ms += ms;
    if (finished.timed_out) throw std::runtime_error(finished.error);
    check(finished.ok == (expected_flags == 0), "unexpected staging success/failure");
    const auto flags = job.validation_flags(); check(flags == expected_flags, "staging flags mismatch");
    check(device.value() == timeline && job.ready_event() && ((__bridge id<MTLSharedEvent>)job.ready_event()).signaledValue == 1,
          "staging event advanced ANE timeline or did not signal");
    return {ms, flags};
}
void equivalent(const Output &base, const Output &candidate, int rows, int cols) {
    check(base.codes.pitch() == candidate.codes.pitch() && base.scales.pitch() == candidate.scales.pitch(), "surface pitch mismatch");
    check(!std::memcmp(base.codes.data(), candidate.codes.data(), base.codes.rows() * base.codes.pitch()), "single-pass codes/padding differ from two-pass");
    check(!std::memcmp(base.scales.data(), candidate.scales.data(), base.scales.rows() * base.scales.pitch()), "single-pass scale bits/padding differ from two-pass");
    for (int col = 0; col < cols; ++col) for (size_t at = size_t(rows); at < base.codes.pitch(); ++at)
        check(static_cast<const uint8_t *>(base.codes.data())[size_t(col) * base.codes.pitch() + at] == 0x5a, "transpose code padding overwritten");
    for (size_t at = size_t(rows) * 2; at < base.scales.pitch(); ++at)
        check(static_cast<const uint8_t *>(base.scales.data())[at] == 0x5a, "scale padding overwritten");
}
void oracle(const Source &source, const W8StageSpec &spec, const Output &output, std::span<const float> scale = {}) {
    // Bounded independent CPU replay, outside every measured span.
    for (int row : {0, 1, 2, spec.rows - 1}) {
        if (row >= spec.rows) continue;
        std::vector<float> rotated(spec.columns);
        for (int col = 0; col < spec.columns; ++col) rotated[col] = source.value(spec.row_begin + row, col) / (scale.empty() ? 1.f : scale[col]);
        for (int col = 0; col < spec.columns; col += 128) rotate_block({rotated.data() + col, 128}, spec.rotation_seed);
        float peak = 0; for (float value : rotated) peak = std::max(peak, std::abs(value));
        const auto expected = normalized_scale(peak);
        check(static_cast<const uint16_t *>(output.scales.data())[row] == expected, "GPU scale differs from independent CPU oracle");
        for (int col = 0; col < spec.columns; ++col)
            check(static_cast<const int8_t *>(output.codes.data())[size_t(col) * output.codes.pitch() + row] == quantize_rotated(rotated[col], expected),
                  "GPU signed code differs from independent CPU oracle");
    }
}
double median(std::vector<double> samples) { std::sort(samples.begin(), samples.end()); return samples[1]; }
NSArray *numbers(const std::vector<double> &values) { NSMutableArray *out = [NSMutableArray new]; for (double value : values) [out addObject:@(value)]; return out; }
NSDictionary *measure(id<MTLDevice> gpu, Device &base, Device &candidate, int cols) {
    constexpr int rows = 1056;
    Source source(gpu, rows, cols, DType::BF16);
    Output first(base, rows, cols), second(candidate, rows, cols);
    const W8StageSpec spec{1, rows, 0, cols, 128, 20260930, true};
    const auto before = candidate.a8_single_pass_stats();
    const auto warm_base = stage(base, source.view, spec, first, false), warm_candidate = stage(candidate, source.view, spec, second, true);
    equivalent(first, second, rows, cols); oracle(source, spec, second);
    std::vector<double> baseline, optimized;
    for (int sample = 0; sample < 3; ++sample) {
        if (sample % 2 == 0) { baseline.push_back(stage(base, source.view, spec, first, false).ms); optimized.push_back(stage(candidate, source.view, spec, second, true).ms); }
        else { optimized.push_back(stage(candidate, source.view, spec, second, true).ms); baseline.push_back(stage(base, source.view, spec, first, false).ms); }
        equivalent(first, second, rows, cols);
        std::cerr << "sample hidden=" << cols << " sample=" << sample << " two_pass_ms=" << baseline.back() << " single_pass_ms=" << optimized.back() << '\n';
    }
    const auto after = candidate.a8_single_pass_stats();
    check(after.requested && after.pipeline_compiled && after.eligible_submissions == before.eligible_submissions + 4 &&
          after.ineligible_submissions == before.ineligible_submissions, "candidate statistics do not confirm four real single-pass submissions");
    check(!base.a8_single_pass_stats().requested && !base.a8_single_pass_stats().pipeline_compiled, "baseline compiled candidate path");
    const double base_median = median(baseline), candidate_median = median(optimized);
    return @{ @"hidden":@(cols), @"rows":@(rows), @"dtype":@"BF16", @"warmups_per_path":@1, @"samples_per_path":@3,
        @"two_pass_first_use_ms":@(warm_base.ms), @"single_pass_first_use_ms":@(warm_candidate.ms),
        @"two_pass_samples_ms":numbers(baseline), @"single_pass_samples_ms":numbers(optimized),
        @"two_pass_median_ms":@(base_median), @"single_pass_median_ms":@(candidate_median),
        @"single_pass_relative_reduction":@(1 - candidate_median / base_median), @"code_mismatches":@0, @"scale_bit_mismatches":@0,
        @"max_scale_absolute_error":@0, @"status_flags":@0, @"padding_verified":@YES, @"cpu_oracle_rows":@4,
        @"candidate_submissions":@4, @"source_bytes":@(source.buffer.length), @"output_bytes_per_path":@(first.codes.bytes() + first.scales.bytes()) };
}
void small_contracts(id<MTLDevice> gpu, Device &base, Device &candidate, int cols) {
    constexpr int rows = 5;
    Source source(gpu, rows, cols, DType::FP32); Scale scale(gpu, cols);
    W8StageSpec spec{1, rows, 0, cols, 128, 20260930, true}; spec.column_scale = scale.view; spec.inverse_column_scale = true;
    Output first(base, rows, cols), second(candidate, rows, cols);
    stage(base, source.view, spec, first, false); stage(candidate, source.view, spec, second, true);
    equivalent(first, second, rows, cols); oracle(source, spec, second, scale.values);
    source.put(rows, cols - 1, std::numeric_limits<float>::quiet_NaN());
    stage(base, source.view, spec, first, false, 1); stage(candidate, source.view, spec, second, true, 1);
    equivalent(first, second, rows, cols);
    source.put(rows, cols - 1, .3f);
    stage(base, source.view, spec, first, false); stage(candidate, source.view, spec, second, true);
    equivalent(first, second, rows, cols); oracle(source, spec, second, scale.values);
    std::cerr << "small_contracts hidden=" << cols << " S1=PASS nonfinite_rejection=PASS fresh_status_recovery=PASS\n";
}
void lifecycle(id<MTLDevice> gpu, Device &candidate) {
    std::weak_ptr<void> input_lease, scale_lease;
    @autoreleasepool {
        constexpr int rows = 5, cols = 4096;
        Source source(gpu, rows, cols, DType::FP32); Scale scale(gpu, cols);
        W8StageSpec spec{1, rows, 0, cols, 128, 20260930, true}; spec.column_scale = scale.view; spec.inverse_column_scale = true;
        Output output(candidate, rows, cols); output.reset();
        input_lease = source.view.owner; scale_lease = scale.view.owner;
        check(budget.total_ms < 18000, "lifecycle test exceeds component staging budget");
        const auto start = Clock::now();
        auto ticket = candidate.stage_w8(source.view, spec, output.codes, output.scales); ++budget.commands;
        source.view.owner.reset(); source.buffer = nil; scale.view.owner.reset(); scale.buffer = nil; spec.column_scale.reset();
        check(!input_lease.expired() && !scale_lease.expired(), "ticket lost source/S1 lease before finish");
        const auto result = ticket.finish(std::chrono::seconds(2));
        budget.total_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        check(result.ok && ticket.single_pass() && !ticket.validation_flags(), "lease-held staging failed");
    }
    // Completion notification can precede callback return by a few CPU
    // instructions. Bound this check without submitting another GPU job.
    const auto deadline = Clock::now() + std::chrono::milliseconds(50);
    while ((!input_lease.expired() || !scale_lease.expired()) && Clock::now() < deadline) std::this_thread::yield();
    check(input_lease.expired() && scale_lease.expired(), "completed ticket retained source/S1 beyond callback teardown");
    std::cerr << "lifecycle source_and_S1_retained_before_finish=PASS released_after_teardown=PASS\n";
}
}
int main(int argc, char *argv[]) {
    if (argc != 2 || std::strcmp(argv[1], "--run-a8-component") != 0) { std::cerr << "requires explicit scheduled --run-a8-component\n"; return 2; }
    @autoreleasepool {
      try {
        auto gpu = MTLCreateSystemDefaultDevice(); check(gpu != nil, "Metal device unavailable");
        setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE", "0", 1); setenv("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE", "0", 1);
        setenv("TURBOCIDER_PRIVATE_ANE_A8_SINGLE_PASS", "0", 1); Device base;
        setenv("TURBOCIDER_PRIVATE_ANE_A8_SINGLE_PASS", "1", 1); Device candidate;
        NSMutableArray *geometries = [NSMutableArray new];
        for (int cols : {3840, 4096}) { @autoreleasepool { [geometries addObject:measure(gpu, base, candidate, cols)]; small_contracts(gpu, base, candidate, cols); } }
        lifecycle(gpu, candidate);
        check(budget.total_ms < 20000, "component staging spans exceeded 20 seconds");
        NSDictionary *report = @{ @"scope":@"GPU Dense A8 H128 staging, stage_w8-to-finish host readiness span; no ANE/model/end-to-end claim",
            @"device":gpu.name, @"passed":@YES, @"geometries":geometries, @"small_contracts":@"FP32 non-power-of-two S1, NaN status, fresh status recovery, CPU oracle: PASS",
            @"lifecycle":@"source/S1 held by ticket, released after completion teardown: PASS", @"gpu_stage_commands":@(budget.commands),
            @"total_stage_span_ms":@(budget.total_ms), @"models_loaded":@0, @"ane_requests":@0, @"stage_specialized":@NO, @"scale_cache_enabled":@NO };
        NSError *error = nil; NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingSortedKeys error:&error];
        check(json != nil, "component result JSON serialization failed");
        std::cout.write(static_cast<const char *>(json.bytes), json.length); std::cout << '\n';
      } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
