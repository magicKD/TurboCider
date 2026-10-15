// Standalone small Metal lab: no TurboCider/MLX/private framework linkage.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mach/mach_time.h>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Params { uint32_t m, n, k, count; float scale, bias; uint32_t flags, steps; };
static_assert(sizeof(Params) == 32);
void check(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void print_json(NSDictionary* value) {
    NSError* error = nil;
    NSData* data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];
    check(data != nil, "JSON encode failed");
    std::cout.write(static_cast<const char*>(data.bytes), data.length); std::cout << '\n';
}

struct Runner {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue, queue2;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;
    id<MTLCounterSampleBuffer> counter_buffer = nil;
    NSMutableArray<NSString*>* counter_labels = [NSMutableArray array];
    bool counter_at_stage = false;
    MTLTimestamp cpu_before = 0, gpu_before = 0;
    NSDictionary* reference_before = nil;
    NSMutableArray* completed_commands = [NSMutableArray array];
    NSDictionary* counter_capabilities;
    explicit Runner(const char* source, const std::string& diagnostic) {
        device = MTLCreateSystemDefaultDevice();
        check(device != nil, "Metal device unavailable in this execution context");
        queue = [device newCommandQueue]; queue2 = [device newCommandQueue];
        check(queue && queue2, "command queue unavailable");
        NSMutableArray* sets = [NSMutableArray array];
        id<MTLCounterSet> timestamp_set = nil;
        for (id<MTLCounterSet> set in device.counterSets) {
            NSMutableArray* counters = [NSMutableArray array];
            for (id<MTLCounter> counter in set.counters) [counters addObject:counter.name];
            [sets addObject:@{@"name":set.name, @"counters":counters}];
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamp_set = set;
        }
        const bool stage = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        const bool dispatch = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
        counter_capabilities = @{@"sets":sets, @"at_stage_boundary":@(stage), @"at_dispatch_boundary":@(dispatch)};
        if (diagnostic == "counter-stage" || diagnostic == "counter-dispatch") {
            counter_at_stage = diagnostic == "counter-stage";
            check(counter_at_stage ? stage : dispatch, "requested counter sampling point unsupported; no fallback");
            check(timestamp_set != nil, "timestamp counter set unavailable");
            auto descriptor = [MTLCounterSampleBufferDescriptor new];
            descriptor.counterSet = timestamp_set;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.sampleCount = 64;
            NSError* counter_error = nil;
            counter_buffer = [device newCounterSampleBufferWithDescriptor:descriptor error:&counter_error];
            check(counter_buffer != nil, counter_error ? counter_error.localizedDescription.UTF8String : "counter buffer unavailable");
        }
        NSError* error = nil;
        NSString* text = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:source] encoding:NSUTF8StringEncoding error:&error];
        check(text != nil, "MSL source unavailable");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.fastMathEnabled = NO;
        library = [device newLibraryWithSource:text options:options error:&error];
        check(library != nil, error ? error.localizedDescription.UTF8String : "MSL compilation failed");
        pipelines = [NSMutableDictionary dictionary];
    }
    id<MTLBuffer> allocate(size_t count) {
        check(count && count < (1ull << 26), "lab buffer cap exceeded");
        auto buffer = [device newBufferWithLength:(count + 16) * sizeof(float) options:MTLResourceStorageModeShared];
        check(buffer != nil, "buffer allocation failed");
        std::fill_n(static_cast<float*>(buffer.contents), count + 16, 0.f);
        return buffer;
    }
    id<MTLComputePipelineState> pipeline(NSString* name) {
        auto value = pipelines[name];
        if (value) return value;
        NSError* error = nil;
        value = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
        check(value != nil, error ? error.localizedDescription.UTF8String : "pipeline creation failed");
        pipelines[name] = value; return value;
    }
    void dispatch(id<MTLCommandBuffer> command, NSString* name, NSArray<id<MTLBuffer>>* buffers,
                  const Params& params, size_t count, size_t threads) {
        auto state = pipeline(name);
        check(threads && threads <= state.maxTotalThreadsPerThreadgroup, "illegal threadgroup size");
        const NSUInteger index = counter_labels.count * 2;
        id<MTLComputeCommandEncoder> encoder;
        if (counter_buffer && counter_at_stage) {
            check(index + 1 < counter_buffer.sampleCount, "counter sample cap exceeded");
            auto pass = [MTLComputePassDescriptor computePassDescriptor];
            pass.sampleBufferAttachments[0].sampleBuffer = counter_buffer;
            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = index;
            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = index + 1;
            encoder = [command computeCommandEncoderWithDescriptor:pass];
        } else {
            encoder = [command computeCommandEncoder];
        }
        check(encoder != nil, "compute encoder unavailable");
        encoder.label = name;
        if (counter_buffer && !counter_at_stage) {
            check(index + 1 < counter_buffer.sampleCount, "counter sample cap exceeded");
            [encoder sampleCountersInBuffer:counter_buffer atSampleIndex:index withBarrier:YES];
        }
        [encoder setComputePipelineState:state];
        for (NSUInteger i = 0; i < buffers.count; ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder setBytes:&params length:sizeof(params) atIndex:buffers.count];
        if ([name isEqualToString:@"gemm_sg8"]) {
            check(state.threadExecutionWidth == 32 && threads == 32, "teaching SG8 kernel needs 32-lane SIMD group");
            [encoder dispatchThreadgroups:MTLSizeMake((params.n + 7) / 8, (params.m + 7) / 8, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:MTLSizeMake((count + threads - 1) / threads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        if (counter_buffer && !counter_at_stage)
            [encoder sampleCountersInBuffer:counter_buffer atSampleIndex:index + 1 withBarrier:YES];
        [encoder endEncoding];
        if (counter_buffer) [counter_labels addObject:name];
    }
    NSDictionary* sample_clock_reference(MTLTimestamp& cpu, MTLTimestamp& gpu) {
        mach_timebase_info_data_t timebase{};
        check(mach_timebase_info(&timebase) == KERN_SUCCESS && timebase.denom > 0, "Mach timebase unavailable");
        const uint64_t mach_before = mach_absolute_time();
        const uint64_t ns_before = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        [device sampleTimestamps:&cpu gpuTimestamp:&gpu];
        const uint64_t ns_after = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        const uint64_t mach_after = mach_absolute_time();
        return @{@"cpu_raw":@(cpu), @"gpu_raw":@(gpu), @"mach_ticks_before":@(mach_before),
            @"mach_ticks_after":@(mach_after), @"uptime_ns_before":@(ns_before), @"uptime_ns_after":@(ns_after),
            @"mach_timebase_numer":@(timebase.numer), @"mach_timebase_denom":@(timebase.denom),
            @"source_kind":@"measured", @"reference_clock":@"mach_absolute_time and CLOCK_UPTIME_RAW; same process"};
    }
    void begin_diagnostic() {
        [counter_labels removeAllObjects];
        [completed_commands removeAllObjects];
        if (counter_buffer) reference_before = sample_clock_reference(cpu_before, gpu_before);
    }
    id resolve_diagnostic() {
        if (!counter_buffer) return [NSNull null];
        MTLTimestamp cpu_after = 0, gpu_after = 0;
        NSDictionary* reference_after = sample_clock_reference(cpu_after, gpu_after);
        NSData* data = [counter_buffer resolveCounterRange:NSMakeRange(0, counter_labels.count * 2)];
        check(data && data.length == counter_labels.count * 2 * sizeof(MTLCounterResultTimestamp), "missing/invalid resolved timestamp data");
        NSMutableArray* values = [NSMutableArray array];
        for (NSUInteger i = 0; i < counter_labels.count; ++i) {
            MTLCounterResultTimestamp start, end;
            std::memcpy(&start, static_cast<const uint8_t*>(data.bytes) + i * 2 * sizeof(start), sizeof(start));
            std::memcpy(&end, static_cast<const uint8_t*>(data.bytes) + (i * 2 + 1) * sizeof(end), sizeof(end));
            const bool valid = start.timestamp != MTLCounterErrorValue && end.timestamp != MTLCounterErrorValue &&
                start.timestamp > 0 && end.timestamp > start.timestamp;
            [values addObject:@{@"label":counter_labels[i], @"start_raw":@(start.timestamp), @"end_raw":@(end.timestamp),
                @"valid":@(valid), @"delta_raw":valid ? @(end.timestamp - start.timestamp) : (id)[NSNull null]}];
        }
        return @{@"sampling_point":counter_at_stage ? @"stage-boundary" : @"dispatch-boundary-barrier",
            @"source_kind":@"measured", @"clock_domain":@"device timestamp; raw unit not assumed",
            @"cpu_pair_before_raw":@(cpu_before), @"gpu_pair_before_raw":@(gpu_before),
            @"cpu_pair_after_raw":@(cpu_after), @"gpu_pair_after_raw":@(gpu_after),
            @"reference_before":reference_before, @"reference_after":reference_after,
            @"completed_command_windows":[completed_commands copy],
            @"duration_seconds": [NSNull null], @"samples":values};
    }
    double finish(id<MTLCommandBuffer> command) {
        [command waitUntilCompleted];
        check(command.status == MTLCommandBufferStatusCompleted,
              command.error ? command.error.localizedDescription.UTF8String : "device command failed");
        const double begin = command.GPUStartTime, end = command.GPUEndTime;
        const double duration = end - begin;
        check(std::isfinite(begin) && std::isfinite(end) && begin > 0 && duration > 0,
              "missing/nonpositive GPU timestamps; not a measured zero duration");
        if (counter_buffer) [completed_commands addObject:@{@"start_host_seconds":@(begin), @"end_host_seconds":@(end),
            @"duration_seconds":@(duration), @"source_kind":@"measured", @"scope":@"completed command buffer; not stage-only time"}];
        return duration;
    }
};

struct CaptureScope {
    bool active = false;
    void start(id<MTLDevice> device, const char* path) {
        auto manager = [MTLCaptureManager sharedCaptureManager];
        check([manager supportsDestination:MTLCaptureDestinationGPUTraceDocument], "GPU trace destination unavailable");
        auto descriptor = [MTLCaptureDescriptor new];
        descriptor.captureObject = device;
        descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        NSError* error = nil;
        active = [manager startCaptureWithDescriptor:descriptor error:&error];
        check(active, error ? error.localizedDescription.UTF8String : "capture failed");
    }
    void stop() { if (active) { [[MTLCaptureManager sharedCaptureManager] stopCapture]; active = false; } }
    ~CaptureScope() { stop(); }
};

int main(int argc, const char** argv) {
    @autoreleasepool {
        try {
            check(argc == 10 || argc == 12 || argc == 13, "usage: bench kernels.metal case repeats count m n k threads seed [diagnostic capture-path [steps]]");
            const std::string kind = argv[2];
            const std::string diagnostic = argc >= 12 ? argv[10] : "none";
            const int steps = argc == 13 ? std::stoi(argv[12]) : 16;
            check(steps >= 1 && steps <= 32, "steps must be 1..32 within counter capacity");
            check(diagnostic == "none" || diagnostic == "counter-stage" || diagnostic == "counter-dispatch" || diagnostic == "capture", "unknown diagnostic mode");
            const int repeats = std::stoi(argv[3]), count = std::stoi(argv[4]);
            const int m = std::stoi(argv[5]), n = std::stoi(argv[6]), k = std::stoi(argv[7]);
            const int threads = std::stoi(argv[8]), seed = std::stoi(argv[9]);
            check(repeats >= 4 && repeats <= 64 && repeats % 4 == 0 && count > 0 && count <= (1 << 22), "invalid repeat/vector cap");
            check(m > 0 && n > 0 && k > 0 && m <= 256 && n <= 256 && k <= 256 && threads > 0, "invalid geometry");
            check(kind == "vector" || kind == "fusion" || kind == "gemm" || kind == "norm-rope" || kind == "attention" || kind == "graph" || kind == "overlap", "unknown case");
            check(kind != "attention" || (n <= 128 && k <= 32), "attention oracle caps N128/D32");
            check(kind != "norm-rope" || k % 2 == 0, "RoPE needs even head dimension");
            auto setup = Clock::now(); Runner runtime(argv[1], diagnostic);
            size_t elements = (kind == "gemm") ? size_t(m) * n :
                              (kind == "norm-rope" || kind == "attention") ? size_t(m) * k : size_t(count);
            auto x = runtime.allocate(std::max(elements, size_t(m) * k));
            auto b = runtime.allocate(std::max(elements, size_t(std::max(n, k)) * std::max(n, k)));
            auto v = runtime.allocate(std::max(elements, size_t(n) * k));
            auto out = runtime.allocate(elements), temp = runtime.allocate(elements), temp2 = runtime.allocate(elements);
            auto a_out = runtime.allocate(elements), b_out = runtime.allocate(elements);
            auto event = [runtime.device newSharedEvent];
            if (kind == "overlap") check(event != nil, "shared events unavailable");
            Params params{uint32_t(m), uint32_t(n), uint32_t(k), uint32_t(elements), .99f, .01f, 1, uint32_t(steps)};
            // Materialize/compile all pipelines BEFORE measured samples.
            NSArray<NSString*>* names = kind == "vector" ? @[@"vector_scalar", @"vector4"] :
                kind == "fusion" ? @[@"scale_bias", @"silu_only", @"vector_scalar", @"fused_epilogue"] :
                kind == "gemm" ? @[@"gemm_scalar", @"gemm_sg8"] :
                kind == "norm-rope" ? @[@"rms_only", @"rope_only", @"norm_rope"] :
                kind == "attention" ? @[@"attention_dense", @"attention_online"] :
                kind == "graph" ? @[@"scale_bias"] : @[@"independent_branch", @"vector_scalar"];
            for (NSString* name in names) runtime.pipeline(name);
            const double setup_seconds = seconds(setup);
            NSMutableArray* samples = [NSMutableArray array];
            NSMutableArray* warmup_samples = [NSMutableArray array];
            CaptureScope capture;
            double maximum_error = 0;
            for (int trial = -4; trial < repeats; ++trial) {
                const bool candidate = ((trial + 4) % 4 == 1 || (trial + 4) % 4 == 2);
                auto* xp = static_cast<float*>(x.contents); auto* bp = static_cast<float*>(b.contents);
                auto* vp = static_cast<float*>(v.contents); auto* yp = static_cast<float*>(out.contents);
                const int input_generation = (trial + 4) / 2; // A/B matched pairs, fresh pair inputs
                for (size_t i = 0; i < x.length / 4; ++i) xp[i] = float(std::sin(double(i + seed + input_generation) * .017) * .5);
                for (size_t i = 0; i < b.length / 4; ++i) bp[i] = float(std::cos(double(i + 7) * .013) * .4);
                for (size_t i = 0; i < v.length / 4; ++i) vp[i] = float(std::sin(double(i + 13) * .019) * .3);
                std::fill_n(yp, elements, std::numeric_limits<float>::quiet_NaN());
                std::fill_n(yp + elements, 16, 1234567.f);
                double gpu_seconds = 0, intersection = 0;
                runtime.begin_diagnostic();
                if (diagnostic == "capture" && trial == 0) capture.start(runtime.device, argv[11]);
                const auto start = Clock::now();
                if (kind == "overlap") {
                    auto first = [runtime.queue commandBuffer];
                    auto second = [(candidate ? runtime.queue2 : runtime.queue) commandBuffer];
                    runtime.dispatch(first, @"independent_branch", @[x, b, a_out], params, elements, threads);
                    runtime.dispatch(second, @"independent_branch", @[x, b, b_out], params, elements, threads);
                    const uint64_t value = uint64_t(trial + 5);
                    [second encodeSignalEvent:event value:value];
                    auto join = [runtime.queue commandBuffer];
                    [join encodeWaitForEvent:event value:value];
                    runtime.dispatch(join, @"vector_scalar", @[a_out, b_out, out], params, elements, threads);
                    [first commit]; [second commit]; [join commit];
                    gpu_seconds = runtime.finish(first) + runtime.finish(second) + runtime.finish(join);
                    intersection = std::max(0., std::min(first.GPUEndTime, second.GPUEndTime) -
                                                 std::max(first.GPUStartTime, second.GPUStartTime));
                } else if (kind == "graph") {
                    id<MTLCommandBuffer> graph = candidate ? [runtime.queue commandBuffer] : nil;
                    id<MTLBuffer> input = x;
                    for (uint32_t step = 0; step < params.steps; ++step) {
                        auto output = step == params.steps - 1 ? out : (step % 2 ? temp2 : temp);
                        auto command = candidate ? graph : [runtime.queue commandBuffer];
                        runtime.dispatch(command, @"scale_bias", @[input, b, output], params, elements, threads);
                        if (!candidate) { [command commit]; gpu_seconds += runtime.finish(command); }
                        input = output;
                    }
                    if (candidate) { [graph commit]; gpu_seconds = runtime.finish(graph); }
                } else {
                    auto command = [runtime.queue commandBuffer];
                    if (kind == "vector") runtime.dispatch(command, candidate ? @"vector4" : @"vector_scalar", @[x, b, out], params, candidate ? (elements + 3) / 4 : elements, threads);
                    if (kind == "gemm") runtime.dispatch(command, candidate ? @"gemm_sg8" : @"gemm_scalar", @[x, b, out], params, elements, candidate ? 32 : threads);
                    if (kind == "fusion") {
                        if (candidate) runtime.dispatch(command, @"fused_epilogue", @[x, b, out], params, elements, threads);
                        else {
                            runtime.dispatch(command, @"scale_bias", @[x, b, temp], params, elements, threads);
                            runtime.dispatch(command, @"silu_only", @[temp, b, temp2], params, elements, threads);
                            runtime.dispatch(command, @"vector_scalar", @[temp2, b, out], params, elements, threads);
                        }
                    }
                    if (kind == "norm-rope") {
                        if (candidate) runtime.dispatch(command, @"norm_rope", @[x, b, out], params, m, threads);
                        else {
                            runtime.dispatch(command, @"rms_only", @[x, b, temp], params, m, threads);
                            runtime.dispatch(command, @"rope_only", @[temp, b, out], params, m, threads);
                        }
                    }
                    if (kind == "attention") runtime.dispatch(command, candidate ? @"attention_online" : @"attention_dense", @[x, b, v, out], params, m, threads);
                    [command commit]; gpu_seconds = runtime.finish(command);
                }
                const double wall = seconds(start);
                id counter_diagnostic = runtime.resolve_diagnostic();
                if (diagnostic == "capture" && trial == 0) capture.stop();
                // Validate EVERY output and guards, OUTSIDE the timed boundary.
                for (size_t index = 0; index < elements; ++index) {
                    double expected = 0;
                    if (kind == "vector") expected = double(xp[index]) + bp[index];
                    if (kind == "fusion") { const float t = xp[index] * params.scale + params.bias; expected = t / (1. + std::exp(-double(t))) + bp[index]; }
                    if (kind == "graph" || kind == "overlap") {
                        float value = xp[index];
                        for (uint32_t step = 0; step < params.steps; ++step) value = value * params.scale + params.bias;
                        expected = kind == "overlap" ? 2. * value : value;
                    }
                    if (kind == "gemm") {
                        const size_t row = index / n, col = index % n;
                        for (int inner = 0; inner < k; ++inner) expected += double(xp[row * k + inner]) * bp[size_t(inner) * n + col];
                    }
                    if (kind == "norm-rope") {
                        const size_t row = index / k, col = index % k, pair = col & ~size_t(1);
                        double energy = 0;
                        for (int d = 0; d < k; ++d) energy += double(xp[row * k + d]) * xp[row * k + d];
                        const double inverse = 1. / std::sqrt(energy / k + 1e-6), angle = row * std::pow(10000., -double(pair) / k);
                        const double a = xp[row * k + pair] * inverse, bb = xp[row * k + pair + 1] * inverse;
                        expected = col % 2 ? a * std::sin(angle) + bb * std::cos(angle) : a * std::cos(angle) - bb * std::sin(angle);
                    }
                    if (kind == "attention") {
                        const size_t row = index / k, dim = index % k;
                        std::vector<double> scores;
                        for (int col = 0; col < n && col <= int(row); ++col) {
                            double dot = 0;
                            for (int d = 0; d < k; ++d) dot += double(xp[row * k + d]) * bp[size_t(col) * k + d];
                            scores.push_back(dot / std::sqrt(double(k)));
                        }
                        const double peak = *std::max_element(scores.begin(), scores.end());
                        double denominator = 0;
                        for (size_t col = 0; col < scores.size(); ++col) { const double p = std::exp(scores[col] - peak); denominator += p; expected += p * vp[col * k + dim]; }
                        expected /= denominator;
                    }
                    const double error = std::abs(double(yp[index]) - expected);
                    check(std::isfinite(yp[index]) && error <= 1e-4 + 1e-4 * std::abs(expected), "nonfinite or numerical witness failure");
                    maximum_error = std::max(error, maximum_error);
                }
                for (size_t guard = 0; guard < 16; ++guard) check(yp[elements + guard] == 1234567.f, "output tail overwrite");
                NSDictionary* sample = @{@"trial":@(trial), @"warmup":@(trial < 0),
                    @"variant":candidate ? @"candidate" : @"baseline", @"pair":@((trial + 4) / 2 - 2),
                    @"input_generation":@(input_generation), @"wall_seconds":@(wall),
                    @"gpu_command_seconds_sum":@(gpu_seconds), @"counter_diagnostic":counter_diagnostic,
                    @"observed_gpu_branch_timestamp_intersection_seconds":@(intersection)};
                [(trial >= 0 ? samples : warmup_samples) addObject:sample];
            }
            capture.stop();
            print_json(@{@"schema":@"turbocider-metal-teaching-lab-v1", @"case":[NSString stringWithUTF8String:kind.c_str()],
                @"device":runtime.device.name, @"source_inputs_synthetic":@YES, @"actual_metal_dispatch":@YES,
                @"physical_ane_placement":@"not applicable: GPU-only experiment", @"production_qualified":@NO,
                @"diagnostic_mode":[NSString stringWithUTF8String:diagnostic.c_str()],
                @"formal_performance_score":@(diagnostic == "none"), @"counter_capabilities":runtime.counter_capabilities,
                @"captured_trial":diagnostic == "capture" ? @0 : (id)[NSNull null],
                @"steps":@(steps),
                @"setup_compile_pipeline_seconds":@(setup_seconds), @"excluded_warmup_samples":@4,
                @"shape":@[@(m),@(n),@(k)], @"vector_count":@(count), @"threads":@(threads), @"samples":samples, @"warmup_samples":warmup_samples,
                @"maximum_abs_error":@(maximum_error), @"all_outputs_finite":@YES, @"tail_guards_intact":@YES,
                @"scope":@"Small actual GPU operator boundary. Host wall includes encode/submit/wait; validation/setup excluded. GPU sums are not E2E or ANE overlap evidence."});
            return 0;
        } catch (const std::exception& error) {
            print_json(@{@"status":@"failed_or_unavailable", @"error":[NSString stringWithUTF8String:error.what()], @"production_qualified":@NO});
            return 3;
        }
    }
}
