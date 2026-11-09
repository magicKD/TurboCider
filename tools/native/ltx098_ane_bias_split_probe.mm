// Diagnostic only: real LTX 0.9.8 FFN+bias on CPU_AND_NE plus a GPU row
// complement. CPU_AND_NE can fall back; this does not prove ANE placement or
// whole-model speed/quality. Never used by the production model route.
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>

#include "../../native/backends/ane_runtime_convert.hpp"
#include "../../native/backends/mlx.hpp"
#include "../../native/models/ltx098/activation.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>

namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static void check(bool good, const std::string &message) {
    if (!good) throw std::runtime_error(message);
}
static std::string error_message(NSError *error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "Core ML operation failed";
}
static uint16_t half_bits(float value) { return std::bit_cast<uint16_t>(_Float16(value)); }

class Surface {
    CVPixelBufferRef pixel_ = nullptr;
    MLMultiArray *array_ = nil;
    int rows_, cols_;
  public:
    Surface(int rows, int cols) : rows_(rows), cols_(cols) {
        NSDictionary *attributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey: @{}};
        check(CVPixelBufferCreate(kCFAllocatorDefault, cols, rows, kCVPixelFormatType_OneComponent16Half,
              (__bridge CFDictionaryRef)attributes, &pixel_) == kCVReturnSuccess && pixel_,
              "cannot allocate diagnostic IOSurface");
        array_ = [[MLMultiArray alloc] initWithPixelBuffer:pixel_ shape:@[@(rows), @(cols)]];
        if (!array_ || !CVPixelBufferGetIOSurface(pixel_)) {
            CVPixelBufferRelease(pixel_); pixel_ = nullptr;
            throw std::runtime_error("cannot bind diagnostic IOSurface");
        }
    }
    ~Surface() { if (pixel_) CVPixelBufferRelease(pixel_); }
    Surface(const Surface &) = delete;
    MLMultiArray *array() const { return array_; }
    template<class F> void access(F function) {
        check(CVPixelBufferLockBaseAddress(pixel_, 0) == kCVReturnSuccess, "cannot lock diagnostic IOSurface");
        try { function(CVPixelBufferGetBaseAddress(pixel_), CVPixelBufferGetBytesPerRow(pixel_)); }
        catch (...) { CVPixelBufferUnlockBaseAddress(pixel_, 0); throw; }
        check(CVPixelBufferUnlockBaseAddress(pixel_, 0) == kCVReturnSuccess, "cannot unlock diagnostic IOSurface");
    }
    void fill_bf16(const void *source) {
        check(source != nullptr, "missing diagnostic BF16 source");
        access([&](void *base, size_t pitch) {
            std::atomic<bool> finite{true};
            auto convert = [&](size_t group) {
                for (int row = int(group * 16); row < std::min(rows_, int((group + 1) * 16)); ++row) {
                    auto *out = reinterpret_cast<uint16_t *>(static_cast<char *>(base) + size_t(row) * pitch);
                    const auto *in = static_cast<const uint16_t *>(source) + size_t(row) * cols_;
                    if (!tc::ane::convert_fp16_row(in, out, cols_, tc::ane::DType::BF16))
                        finite.store(false, std::memory_order_relaxed);
                }
            };
            auto *job = &convert;
            const size_t groups = (size_t(rows_) + 15) / 16;
            if (size_t(rows_) * cols_ < 65536) {
                for (size_t group = 0; group < groups; ++group) convert(group);
            } else {
                dispatch_apply(groups, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t group) { (*job)(group); });
            }
            check(finite.load(std::memory_order_relaxed), "BF16 input cannot be represented as finite FP16");
        });
    }
};

class BiasGraph {
    static constexpr int rows = 256, hidden = 2048, width = 8192;
    Surface input_{rows, hidden}, output_{rows, hidden}, up_{width, hidden}, down_{hidden, width};
    MLMultiArray *up_bias_ = nil, *down_bias_ = nil;
    MLModel *model_ = nil;
    MLDictionaryFeatureProvider *features_ = nil;
    MLPredictionOptions *options_ = nil;
    void check_feature(NSString *name, NSArray<NSNumber *> *shape, NSDictionary *descriptions) {
        MLFeatureDescription *feature = descriptions[name];
        check(feature && feature.type == MLFeatureTypeMultiArray &&
              feature.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat16 &&
              [feature.multiArrayConstraint.shape isEqual:shape], "diagnostic Core ML feature mismatch");
    }
    static void fill_bias(MLMultiArray *destination, const void *source, size_t count) {
        check(destination && destination.dataType == MLMultiArrayDataTypeFloat16 &&
              destination.count == NSInteger(count) && destination.dataPointer,
              "invalid diagnostic bias backing");
        check(tc::ane::convert_fp16_row(source, static_cast<uint16_t *>(destination.dataPointer), count,
                                       tc::ane::DType::BF16), "nonfinite FP16 bias");
    }
  public:
    double input_seconds = 0, prediction_seconds = 0, output_seconds = 0;
    uint64_t calls = 0;
    BiasGraph(const std::filesystem::path &compiled, bool cpu_only) {
        NSError *error = nil;
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = cpu_only ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
        model_ = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(compiled.c_str())]
                                  configuration:configuration error:&error];
        check(model_ != nil, error_message(error));
        up_bias_ = [[MLMultiArray alloc] initWithShape:@[@(width)] dataType:MLMultiArrayDataTypeFloat16 error:&error];
        check(up_bias_ != nil, error_message(error));
        down_bias_ = [[MLMultiArray alloc] initWithShape:@[@(hidden)] dataType:MLMultiArrayDataTypeFloat16 error:&error];
        check(down_bias_ != nil, error_message(error));
        NSDictionary *input_descriptions = model_.modelDescription.inputDescriptionsByName;
        NSDictionary *output_descriptions = model_.modelDescription.outputDescriptionsByName;
        check(input_descriptions.count == 5 && output_descriptions.count == 1,
              "unexpected diagnostic Core ML feature count");
        check_feature(@"x", @[@(rows), @(hidden)], input_descriptions);
        check_feature(@"wu", @[@(width), @(hidden)], input_descriptions);
        check_feature(@"wd", @[@(hidden), @(width)], input_descriptions);
        check_feature(@"bu", @[@(width)], input_descriptions);
        check_feature(@"bd", @[@(hidden)], input_descriptions);
        check_feature(@"y", @[@(rows), @(hidden)], output_descriptions);
        features_ = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"x":[MLFeatureValue featureValueWithMultiArray:input_.array()],
            @"wu":[MLFeatureValue featureValueWithMultiArray:up_.array()],
            @"wd":[MLFeatureValue featureValueWithMultiArray:down_.array()],
            @"bu":[MLFeatureValue featureValueWithMultiArray:up_bias_],
            @"bd":[MLFeatureValue featureValueWithMultiArray:down_bias_]} error:&error];
        check(features_ != nil, error_message(error));
        options_ = [MLPredictionOptions new];
        options_.outputBackings = @{@"y": output_.array()};
    }
    void stage(const mx::array &wu, const mx::array &wd, const mx::array &bu, const mx::array &bd) {
        up_.fill_bf16(wu.data<mx::bfloat16_t>());
        down_.fill_bf16(wd.data<mx::bfloat16_t>());
        fill_bias(up_bias_, bu.data<mx::bfloat16_t>(), width);
        fill_bias(down_bias_, bd.data<mx::bfloat16_t>(), hidden);
    }
    void fill_input(const void *data) {
        const auto start = Clock::now();
        input_.fill_bf16(data);
        input_seconds += seconds(start);
    }
    void predict(uint16_t *output, tc::ane::DType dtype) {
        NSError *error = nil;
        const auto started = Clock::now();
        id<MLFeatureProvider> result = [model_ predictionFromFeatures:features_ options:options_ error:&error];
        prediction_seconds += seconds(started);
        check(result != nil, error_message(error));
        ++calls;
        MLMultiArray *array = [result featureValueForName:@"y"].multiArrayValue;
        check(array && array.dataType == MLMultiArrayDataTypeFloat16 && [array.shape isEqual:@[@(rows), @(hidden)]],
              "diagnostic output geometry changed");
        const size_t row_stride = [array.strides[0] unsignedLongLongValue];
        const size_t col_stride = [array.strides[1] unsignedLongLongValue];
        __block bool good = true;
        const auto copied = Clock::now();
        [array getBytesWithHandler:^(const void *bytes, NSInteger size) {
            const size_t elements = size_t(rows - 1) * row_stride + size_t(hidden - 1) * col_stride + 1;
            if (!bytes || size < 0 || size_t(size) / 2 < elements) { good = false; return; }
            auto status = tc::ane::restore_fp16_matrix(static_cast<const uint16_t *>(bytes), output,
                rows, hidden, row_stride, col_stride, dtype, 1.f);
            good = status.source_finite && status.restored_finite;
        }];
        output_seconds += seconds(copied);
        check(good, "diagnostic Core ML output is nonfinite or invalid");
    }
    void self_test() {
        std::vector<uint16_t> output(size_t(rows) * hidden);
        for (float scale : {.125f, -.25f}) {
            up_.access([&](void *base, size_t pitch) {
                std::memset(base, 0, pitch * width);
                for (int r = 0; r < width; ++r)
                    reinterpret_cast<uint16_t *>(static_cast<char *>(base) + size_t(r) * pitch)[r % hidden] = half_bits(scale);
            });
            down_.access([&](void *base, size_t pitch) {
                std::memset(base, 0, pitch * hidden);
                for (int r = 0; r < hidden; ++r)
                    reinterpret_cast<uint16_t *>(static_cast<char *>(base) + size_t(r) * pitch)[r] = half_bits(scale);
            });
            auto *bu = static_cast<uint16_t *>(up_bias_.dataPointer);
            auto *bd = static_cast<uint16_t *>(down_bias_.dataPointer);
            std::fill_n(bu, width, half_bits(.03125f));
            std::fill_n(bd, hidden, half_bits(-.015625f));
            input_.access([&](void *base, size_t pitch) {
                for (int r = 0; r < rows; ++r) for (int c = 0; c < hidden; ++c)
                    reinterpret_cast<uint16_t *>(static_cast<char *>(base) + size_t(r) * pitch)[c] =
                        half_bits(float((r * 3 + c * 7) % 17 - 8) / 8.f);
            });
            predict(output.data(), tc::ane::DType::FP16);
            for (int r = 0; r < rows; ++r) for (int c = 0; c < hidden; ++c) {
                const float x = float((r * 3 + c * 7) % 17 - 8) / 8.f;
                const float z = x * scale + .03125f;
                const float expected = .5f * z * (1.f + std::tanh(.7978845608f * (z + .044715f * z * z * z))) * scale - .015625f;
                const float actual = float(std::bit_cast<_Float16>(output[size_t(r) * hidden + c]));
                check(std::isfinite(actual) && std::abs(actual - expected) <= .0002f + .02f * std::abs(expected),
                      "diagnostic Core ML biased FFN self-test failed");
            }
        }
        input_seconds = prediction_seconds = output_seconds = 0;
        calls = 0;
    }
};

#ifndef TURBOCIDER_LTX098_BIAS_GRAPH_ONLY
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 9, "usage: ltx098-ane-bias-split-probe MODEL.mlmodelc CHECKPOINT ACTIVATION NEW_OUTPUT CHUNKS REPEATS cpu|ne BLOCK");
            const std::filesystem::path output(argv[4]);
            const int chunks = std::stoi(argv[5]), repeats = std::stoi(argv[6]);
            const std::string units(argv[7]);
            const int block = std::stoi(argv[8]);
            check(!std::filesystem::exists(output) && chunks >= 1 && chunks <= 15 && repeats >= 2 && repeats <= 20 &&
                  (units == "cpu" || units == "ne") && block >= 0 && block < 28,
                  "invalid diagnostic output, chunks, repeats, compute units or block");
            auto checkpoint = mx::load_safetensors(argv[2]).first;
            auto capture = mx::load_safetensors(argv[3]).first.at("input");
            const std::string prefix = "model.diffusion_model.transformer_blocks." + std::to_string(block) + ".ff.net.";
            auto wu = mx::contiguous(checkpoint.at(prefix + "0.proj.weight"));
            auto bu = mx::contiguous(checkpoint.at(prefix + "0.proj.bias"));
            auto wd = mx::contiguous(checkpoint.at(prefix + "2.weight"));
            auto bd = mx::contiguous(checkpoint.at(prefix + "2.bias"));
            auto x = mx::contiguous(mx::reshape(capture, {4096, 2048}));
            check(wu.shape() == mx::Shape{8192, 2048} && wd.shape() == mx::Shape{2048, 8192} &&
                  bu.shape() == mx::Shape{8192} && bd.shape() == mx::Shape{2048} &&
                  x.dtype() == mx::bfloat16 && wu.dtype() == mx::bfloat16 && wd.dtype() == mx::bfloat16 &&
                  bu.dtype() == mx::bfloat16 && bd.dtype() == mx::bfloat16, "captured FFN geometry/dtype mismatch");
            mx::eval(x, wu, bu, wd, bd);
            const auto load_start = Clock::now();
            BiasGraph graph(argv[1], units == "cpu");
            const double load_seconds = seconds(load_start);
            graph.self_test();
            auto gpu = [&](const mx::array &input) {
                auto up = mx::matmul(input, mx::transpose(wu)) + bu;
                return mx::matmul(tc::ltx098::gelu_tanh(up), mx::transpose(wd)) + bd;
            };
            const int tail_rows = chunks * 256, head_rows = 4096 - tail_rows;
            const auto *tail_input = reinterpret_cast<const uint16_t *>(x.data<mx::bfloat16_t>()) + size_t(head_rows) * 2048;
            std::vector<uint16_t> tail_output(size_t(tail_rows) * 2048);
            std::vector<double> baseline_times, stage_times, split_times, total_times, predict_times, input_times, output_times, join_times;
            std::optional<mx::array> baseline, candidate;
            for (int iteration = 0; iteration < repeats + 2; ++iteration) {
                double baseline_time = 0, stage_time = 0, split_time = 0, total_time = 0, predict_time = 0, input_time = 0, output_time = 0, join_time = 0;
                auto full_gpu = [&] {
                    const auto start = Clock::now();
                    baseline = gpu(x); mx::eval(*baseline);
                    baseline_time = seconds(start);
                };
                auto split = [&] {
                    const auto total_start = Clock::now();
                    const auto stage_start = Clock::now();
                    graph.stage(wu, wd, bu, bd);
                    stage_time = seconds(stage_start);
                    graph.input_seconds = graph.prediction_seconds = graph.output_seconds = 0;
                    graph.calls = 0;
                    const auto start = Clock::now();
                    std::exception_ptr failure;
                    std::jthread worker([&] {
                        @autoreleasepool {
                            try {
                                for (int chunk = 0; chunk < chunks; ++chunk) {
                                    graph.fill_input(tail_input + size_t(chunk) * 256 * 2048);
                                    graph.predict(tail_output.data() + size_t(chunk) * 256 * 2048, tc::ane::DType::BF16);
                                }
                            } catch (...) { failure = std::current_exception(); }
                        }
                    });
                    auto head = gpu(mx::slice(x, {0, 0}, {head_rows, 2048}));
                    mx::eval(head);
                    const auto join_start = Clock::now();
                    worker.join();
                    join_time = seconds(join_start);
                    if (failure) std::rethrow_exception(failure);
                    check(graph.calls == uint64_t(chunks), "diagnostic prediction count mismatch");
                    auto tail = mx::array(reinterpret_cast<const mx::bfloat16_t *>(tail_output.data()),
                                          {tail_rows, 2048}, mx::bfloat16);
                    candidate = mx::concatenate({head, tail}, 0); mx::eval(*candidate);
                    split_time = seconds(start);
                    total_time = seconds(total_start);
                    predict_time = graph.prediction_seconds;
                    input_time = graph.input_seconds;
                    output_time = graph.output_seconds;
                };
                if (iteration % 2) { split(); full_gpu(); }
                else { full_gpu(); split(); }
                if (iteration >= 2) {
                    baseline_times.push_back(baseline_time); stage_times.push_back(stage_time);
                    split_times.push_back(split_time); total_times.push_back(total_time);
                    predict_times.push_back(predict_time); input_times.push_back(input_time);
                    output_times.push_back(output_time); join_times.push_back(join_time);
                }
            }
            mx::save_safetensors(output.string(), {{"gpu", *baseline}, {"split", *candidate}});
            auto write = [](const std::vector<double> &values) {
                std::cout << '[';
                for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
                std::cout << ']';
            };
            std::cout << std::setprecision(12) << "{\"scope\":\"real captured activations, weights and both biases; component only\","
                      << "\"physical_ane_placement\":\"unknown\",\"rows\":4096,\"ane_rows\":" << tail_rows
                      << ",\"gpu_rows\":" << head_rows << ",\"chunks\":" << chunks << ",\"block\":" << block
                      << ",\"self_test\":true,\"load_seconds\":" << load_seconds
                      << ",\"mlx_peak_bytes\":" << mx::get_peak_memory()
                      << ",\"gpu_seconds\":"; write(baseline_times);
            std::cout << ",\"stage_seconds\":"; write(stage_times);
            std::cout << ",\"split_seconds\":"; write(split_times);
            std::cout << ",\"split_with_stage_seconds\":"; write(total_times);
            std::cout << ",\"prediction_seconds\":"; write(predict_times);
            std::cout << ",\"input_seconds\":"; write(input_times);
            std::cout << ",\"output_seconds\":"; write(output_times);
            std::cout << ",\"join_seconds\":"; write(join_times);
            std::cout << "}\n";
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
#endif
