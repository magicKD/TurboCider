// Standalone small Metal lab: no TurboCider/MLX/private framework linkage.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
    explicit Runner(const char* source) {
        device = MTLCreateSystemDefaultDevice();
        check(device != nil, "Metal device unavailable in this execution context");
        queue = [device newCommandQueue]; queue2 = [device newCommandQueue];
        check(queue && queue2, "command queue unavailable");
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
        auto encoder = [command computeCommandEncoder];
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
        [encoder endEncoding];
    }
    static double finish(id<MTLCommandBuffer> command) {
        [command waitUntilCompleted];
        check(command.status == MTLCommandBufferStatusCompleted,
              command.error ? command.error.localizedDescription.UTF8String : "device command failed");
        const double duration = command.GPUEndTime - command.GPUStartTime;
        check(std::isfinite(duration) && duration >= 0, "invalid GPU timestamps");
        return duration;
    }
};

int main(int argc, const char** argv) {
    @autoreleasepool {
        try {
            check(argc == 10, "usage: bench kernels.metal case repeats count m n k threads seed");
            const std::string kind = argv[2];
            const int repeats = std::stoi(argv[3]), count = std::stoi(argv[4]);
            const int m = std::stoi(argv[5]), n = std::stoi(argv[6]), k = std::stoi(argv[7]);
            const int threads = std::stoi(argv[8]), seed = std::stoi(argv[9]);
            check(repeats >= 4 && repeats <= 64 && repeats % 4 == 0 && count > 0 && count <= (1 << 22), "invalid repeat/vector cap");
            check(m > 0 && n > 0 && k > 0 && m <= 256 && n <= 256 && k <= 256 && threads > 0, "invalid geometry");
            check(kind == "vector" || kind == "fusion" || kind == "gemm" || kind == "norm-rope" || kind == "attention" || kind == "graph" || kind == "overlap", "unknown case");
            check(kind != "attention" || (n <= 128 && k <= 32), "attention oracle caps N128/D32");
            check(kind != "norm-rope" || k % 2 == 0, "RoPE needs even head dimension");
            auto setup = Clock::now(); Runner runtime(argv[1]);
            size_t elements = (kind == "gemm") ? size_t(m) * n :
                              (kind == "norm-rope" || kind == "attention") ? size_t(m) * k : size_t(count);
            auto x = runtime.allocate(std::max(elements, size_t(m) * k));
            auto b = runtime.allocate(std::max(elements, size_t(std::max(n, k)) * std::max(n, k)));
            auto v = runtime.allocate(std::max(elements, size_t(n) * k));
            auto out = runtime.allocate(elements), temp = runtime.allocate(elements), temp2 = runtime.allocate(elements);
            auto a_out = runtime.allocate(elements), b_out = runtime.allocate(elements);
            auto event = [runtime.device newSharedEvent];
            if (kind == "overlap") check(event != nil, "shared events unavailable");
            Params params{uint32_t(m), uint32_t(n), uint32_t(k), uint32_t(elements), .99f, .01f, 1, 16};
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
                    gpu_seconds = Runner::finish(first) + Runner::finish(second) + Runner::finish(join);
                    intersection = std::max(0., std::min(first.GPUEndTime, second.GPUEndTime) -
                                                 std::max(first.GPUStartTime, second.GPUStartTime));
                } else if (kind == "graph") {
                    id<MTLCommandBuffer> graph = candidate ? [runtime.queue commandBuffer] : nil;
                    id<MTLBuffer> input = x;
                    for (uint32_t step = 0; step < params.steps; ++step) {
                        auto output = step == params.steps - 1 ? out : (step % 2 ? temp2 : temp);
                        auto command = candidate ? graph : [runtime.queue commandBuffer];
                        runtime.dispatch(command, @"scale_bias", @[input, b, output], params, elements, threads);
                        if (!candidate) { [command commit]; gpu_seconds += Runner::finish(command); }
                        input = output;
                    }
                    if (candidate) { [graph commit]; gpu_seconds = Runner::finish(graph); }
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
                    [command commit]; gpu_seconds = Runner::finish(command);
                }
                const double wall = seconds(start);
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
                if (trial >= 0) [samples addObject:@{@"variant":candidate ? @"candidate" : @"baseline", @"pair":@(trial / 2),
                    @"input_generation":@(input_generation), @"wall_seconds":@(wall),
                    @"gpu_command_seconds_sum":@(gpu_seconds), @"observed_gpu_branch_timestamp_intersection_seconds":@(intersection)}];
            }
            print_json(@{@"schema":@"turbocider-metal-teaching-lab-v1", @"case":[NSString stringWithUTF8String:kind.c_str()],
                @"device":runtime.device.name, @"source_inputs_synthetic":@YES, @"actual_metal_dispatch":@YES,
                @"physical_ane_placement":@"not applicable: GPU-only experiment", @"production_qualified":@NO,
                @"setup_compile_pipeline_seconds":@(setup_seconds), @"excluded_warmup_samples":@4,
                @"shape":@[@(m),@(n),@(k)], @"vector_count":@(count), @"threads":@(threads), @"samples":samples,
                @"maximum_abs_error":@(maximum_error), @"all_outputs_finite":@YES, @"tail_guards_intact":@YES,
                @"scope":@"Small actual GPU operator boundary. Host wall includes encode/submit/wait; validation/setup excluded. GPU sums are not E2E or ANE overlap evidence."});
            return 0;
        } catch (const std::exception& error) {
            print_json(@{@"status":@"failed_or_unavailable", @"error":[NSString stringWithUTF8String:error.what()], @"production_qualified":@NO});
            return 3;
        }
    }
}
