#include "ane_runtime.hpp"
#include "ane_runtime_convert.hpp"
#include "ane_runtime_quant.hpp"
#include "ane_runtime_packed.hpp"
#include "ane_memory.hpp"
#include "ane_artifact_lease.hpp"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <CommonCrypto/CommonDigest.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <cstdio>

namespace tc::ane {
namespace {
struct OutputOverflow : std::runtime_error { using std::runtime_error::runtime_error; };
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(bool good, const char *reason) {
    if (!good) throw std::runtime_error(reason);
}
std::string message(NSError *error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "Core ML operation failed";
}
size_t element_bytes(DType dtype) { return dtype == DType::FP32 ? 4 : 2; }
size_t stride(const MatrixView &view) {
    return view.row_stride_bytes ? view.row_stride_bytes : size_t(view.cols) * element_bytes(view.dtype);
}
void validate(const MatrixView &view) {
    check(view.dtype == DType::FP16 || view.dtype == DType::BF16 || view.dtype == DType::FP32,
          "unsupported runtime ANE matrix dtype");
    check(view.data && view.rows > 0 && view.cols > 0 && view.rows <= 1048576 &&
              view.cols <= 32768, "invalid runtime ANE matrix");
    const size_t row_bytes = size_t(view.cols) * element_bytes(view.dtype);
    check(stride(view) >= row_bytes && stride(view) <= SIZE_MAX / size_t(view.rows) &&
              view.bytes >= size_t(view.rows - 1) * stride(view) + row_bytes,
          "runtime ANE matrix storage is too small");
}

// A dispatch thread is never one of the conversion pool's workers. It has
// exactly one outstanding job, and teardown waits before releasing its slots.
class Worker {
    std::mutex mutex_;
    std::condition_variable condition_;
    std::function<void()> job_;
    bool busy_ = false, stop_ = false;
    std::exception_ptr error_;
    std::thread thread_;
  public:
    Worker() : thread_([this] {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [&] { return stop_ || job_; });
                if (stop_) return;
                // A moved-from std::function need not be empty (notably for
                // inline callables). Explicitly clear the queue or a completed
                // job can repeat while join() already reports idle.
                job = std::exchange(job_, {});
            }
            std::exception_ptr error;
            @autoreleasepool {
                try { job(); } catch (...) { error = std::current_exception(); }
            }
            {
                std::lock_guard lock(mutex_);
                error_ = error;
                busy_ = false;
            }
            condition_.notify_all();
        }
    }) {}
    ~Worker() {
        try { join(); } catch (...) {}
        { std::lock_guard lock(mutex_); stop_ = true; }
        condition_.notify_all();
        thread_.join();
    }
    void join() {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return !busy_; });
        auto error = std::exchange(error_, {});
        if (error) std::rethrow_exception(error);
    }
    void submit(std::function<void()> job) {
        join();
        { std::lock_guard lock(mutex_); job_ = std::move(job); busy_ = true; }
        condition_.notify_all();
    }
};

struct Surface {
    CVPixelBufferRef pixel = nullptr;
    MLMultiArray *array = nil;
    int rows, cols;
    Surface(int r, int c) : rows(r), cols(c) {
        NSDictionary *attributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey: @{}};
        check(CVPixelBufferCreate(kCFAllocatorDefault, c, r, kCVPixelFormatType_OneComponent16Half,
                  (__bridge CFDictionaryRef)attributes, &pixel) == kCVReturnSuccess && pixel,
              "cannot allocate runtime ANE IOSurface");
        // Own pixel immediately so a later interface failure cannot leak it.
        array = [[MLMultiArray alloc] initWithPixelBuffer:pixel shape:@[@(r), @(c)]];
        if (!array || !CVPixelBufferGetIOSurface(pixel)) {
            CVPixelBufferRelease(pixel);
            pixel = nullptr;
            throw std::runtime_error("cannot bind runtime ANE IOSurface");
        }
    }
    ~Surface() { if (pixel) CVPixelBufferRelease(pixel); }
    Surface(const Surface &) = delete;
    size_t bytes() const { return CVPixelBufferGetBytesPerRow(pixel) * size_t(rows); }
    template<class F> void access(F body) {
        check(CVPixelBufferLockBaseAddress(pixel, 0) == kCVReturnSuccess, "cannot lock IOSurface");
        try {
            body(CVPixelBufferGetBaseAddress(pixel), CVPixelBufferGetBytesPerRow(pixel));
        } catch (...) {
            CVPixelBufferUnlockBaseAddress(pixel, 0);
            throw;
        }
        check(CVPixelBufferUnlockBaseAddress(pixel, 0) == kCVReturnSuccess, "cannot unlock IOSurface");
    }
    template<class Row> void fill_rows(Row row, bool bounded_packed = false) {
        access([&](void *base, size_t pitch) {
            std::atomic<bool> bad{false};
            // Sixteen-row tasks amortize GCD dispatch for large weight matrices.
            auto convert = [&](size_t group) {
                for (int r = int(group * 16); r < std::min(rows, int((group + 1) * 16)); ++r) {
                    auto *output = reinterpret_cast<uint16_t *>(static_cast<char *>(base) + size_t(r) * pitch);
                    if (!row(r, output))
                        bad.store(true, std::memory_order_relaxed);
                }
            };
            const size_t groups = (size_t(rows) + 15) / 16;
            if (bounded_packed) {
                // At most eight active conversion callbacks; no one-task-per-
                // row-group fanout or layer-sized conversion scratch. The
                // dispatch owner is a separate std::thread, never this pool.
                const auto plan = plan_packed_conversion(rows, cols, std::thread::hardware_concurrency());
                check(bool(plan), "invalid runtime ANE packed conversion plan");
                std::atomic<size_t> next{0};
                auto worker = [&] {
                    for (;;) {
                        if (bad.load(std::memory_order_relaxed)) break;
                        const auto group = next.fetch_add(1, std::memory_order_relaxed);
                        if (group >= groups) break;
                        convert(group);
                    }
                };
                auto *fn = &worker;
                dispatch_apply(plan->workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                               ^(size_t) { (*fn)(); });
            } else if (size_t(rows) * cols < 65536) {
                for (size_t i = 0; i < groups; ++i) convert(i);
            } else {
                // Do not copy the atomic into an Objective-C block capture.
                auto *fn = &convert;
                dispatch_apply(groups, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                               ^(size_t i) { (*fn)(i); });
            }
            check(!bad.load(), "runtime ANE FP16 staging overflow/nonfinite input");
        });
    }
    void fill(MatrixView source, bool scalar_only, float scale = 1.f) {
        validate(source);
        check(source.rows == rows && source.cols == cols, "runtime ANE weight/input shape mismatch");
        fill_rows([&](int r, uint16_t *output) {
            const auto *input = static_cast<const char *>(source.data) + size_t(r) * stride(source);
            return convert_fp16_row(input, output, size_t(cols), source.dtype, scalar_only, scale);
        });
    }
    void fill(const AffineView &source, bool scalar_only, float scale = 1.f) {
        validate_affine_view(source);
        check(source.rows == rows && source.cols == cols, "runtime ANE affine weight shape mismatch");
        fill_rows([&](int r, uint16_t *output) {
            return affine_fp16_row(source, r, output, scalar_only, scale);
        });
    }
    void fill(const GgufView &source, bool scalar_only, float scale = 1.f) {
        validate_gguf_view(source);
        check(source.rows == rows && source.cols == cols, "runtime ANE raw GGUF shape mismatch");
        fill_rows([&](int row, uint16_t *output) {
            return gguf_fp16_row(source, size_t(row), output, scalar_only, scale);
        }, true);
    }
    void fill(const ConvrotView &source, bool scalar_only, float scale = 1.f) {
        validate_convrot_view(source);
        check(source.rows == rows && source.cols == cols, "runtime ANE ConvRot shape mismatch");
        fill_rows([&](int row, uint16_t *output) {
            return convrot_fp16_row(source, size_t(row), output, scalar_only, scale);
        }, true);
    }
    void fill(const ConvrotAffineView &source,bool scalar_only,float scale=1.f) {
        validate_convrot_affine_view(source);
        check(source.packed.rows==rows && source.packed.cols==cols,"runtime ANE packed ConvRot shape mismatch");
        fill_rows([&](int row,uint16_t *output) {
            return convrot_affine_fp16_row(source,size_t(row),output,scalar_only,scale);
        },true);
    }
    void fill_matmul_parts(const std::vector<MatrixView> &parts, bool scalar_only) {
        check(!parts.empty(), "runtime ANE MatMul needs weight parts");
        std::vector<int> ends;
        int total = 0;
        for (const auto &part : parts) {
            validate(part);
            check(part.cols == cols && part.rows <= rows - total,
                  "runtime ANE MatMul part geometry mismatch");
            total += part.rows;
            ends.push_back(total);
        }
        check(total == rows, "runtime ANE MatMul parts do not cover the weight slot");
        fill_rows([&](int r, uint16_t *output) {
            const auto i = size_t(std::upper_bound(ends.begin(), ends.end(), r) - ends.begin());
            const auto &part = parts[i];
            const int local_row = r - (i ? ends[i - 1] : 0);
            const auto *input = static_cast<const char *>(part.data) + size_t(local_row) * stride(part);
            return convert_fp16_row(input, output, size_t(cols), part.dtype, scalar_only);
        });
    }
};

int dimension(NSDictionary *manifest, NSString *key) {
    id value = manifest[key];
    check([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
              [value doubleValue] == [value intValue] && [value intValue] > 0 && [value intValue] <= 32768,
          "invalid runtime ANE graph dimension");
    return [value intValue];
}

// Core ML may keep referring to the compiled directory after model load.
// Keep this verified, private copy alive until the model and worker are gone;
// changes to the user's artifact must not change the bytes Core ML sees.
using ArtifactLease = detail::ArtifactLease;

bool matches_digest(NSData *bytes, NSString *digest) {
    if (!bytes || bytes.length > 64 * 1024 * 1024) return false;
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.bytes, CC_LONG(bytes.length), hash);
    char hex[65];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) std::snprintf(hex + 2 * i, 3, "%02x", hash[i]);
    return [digest isEqualToString:@(hex)];
}

NSDictionary *read_manifest_shape(const std::filesystem::path &path, GraphShape &shape) {
    check(!std::filesystem::is_symlink(path), "runtime ANE manifest must not be a symlink");
    check(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) <= (1u << 20),
          "runtime ANE manifest must be a bounded regular file");
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    NSError *error = nil;
    id decoded = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
    check([decoded isKindOfClass:NSDictionary.class], "cannot read runtime ANE manifest");
    NSDictionary *manifest = decoded;
    shape.lora_inputs = [manifest[@"graph_version"] isEqual:@2];
    check([manifest[@"schema_version"] isEqual:@1] &&
              ([manifest[@"graph_version"] isEqual:@1] || shape.lora_inputs) &&
              (shape.lora_inputs ? [manifest[@"lora_inputs"] isEqual:@YES] :
                  (!manifest[@"lora_inputs"] || [manifest[@"lora_inputs"] isEqual:@NO])) &&
              [manifest[@"backend"] isEqual:@"runtime_weight_fp16"] &&
              [manifest[@"layout"] isEqual:@"out_in"] && [manifest[@"biases"] isEqual:@NO] &&
              [manifest[@"compiled_model"] isEqual:@"graph.mlmodelc"], "unsupported runtime ANE graph ABI");
    if ([manifest[@"kind"] isEqual:@"matmul"]) shape.kind = Kind::Matmul;
    else if ([manifest[@"kind"] isEqual:@"swiglu"]) shape.kind = Kind::SwiGLU;
    else if ([manifest[@"kind"] isEqual:@"gelu"]) shape.kind = Kind::GELU;
    else throw std::runtime_error("unsupported runtime ANE graph kind");
    check(!shape.lora_inputs || shape.kind == Kind::SwiGLU,
          "runtime ANE activation corrections require SwiGLU");
    shape.rows = dimension(manifest, @"rows");
    shape.hidden = dimension(manifest, @"hidden");
    shape.width = dimension(manifest, @"width");
    shape.tile_k = dimension(manifest, @"tile_k");
    shape.tile_n = dimension(manifest, @"tile_n");
    return manifest;
}
std::filesystem::path verify_manifest(const std::filesystem::path &path, GraphShape &shape,
                                      ArtifactLease &lease) {
    NSDictionary *manifest = read_manifest_shape(path, shape);
    NSDictionary *files = manifest[@"files"];
    check([files isKindOfClass:NSDictionary.class] && files.count > 0, "missing runtime ANE artifact receipt");
    const auto root = path.parent_path();
    check(!std::filesystem::is_symlink(root / "graph.mlmodelc") &&
              std::filesystem::is_directory(root / "graph.mlmodelc"),
          "runtime ANE compiled artifact must be a real directory");
    const auto private_model = lease.root / "graph.mlmodelc";
    std::filesystem::create_directory(private_model);
    // Do not depend on the host application's umask. Nested directories and
    // copied files must remain private and eligible for guarded lease cleanup.
    std::filesystem::permissions(private_model, std::filesystem::perms::owner_all);
    size_t observed = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root / "graph.mlmodelc")) {
        check(!entry.is_symlink(), "runtime ANE compiled artifact contains a symlink");
        const auto relative = entry.path().lexically_relative(root).generic_string();
        const auto private_path = lease.root / relative;
        if (entry.is_directory()) {
            std::filesystem::create_directory(private_path);
            std::filesystem::permissions(private_path, std::filesystem::perms::owner_all);
            continue;
        }
        check(entry.is_regular_file(), "runtime ANE compiled artifact contains a nonregular entry");
        id digest = files[@(relative.c_str())];
        check([digest isKindOfClass:NSString.class] && [digest length] == 64,
              "runtime ANE artifact file absent from receipt");
        check(entry.file_size() <= 64 * 1024 * 1024, "runtime ANE graph unexpectedly contains large weights");
        NSData *bytes = [NSData dataWithContentsOfFile:@(entry.path().c_str())];
        check(bytes != nil && bytes.length <= 64 * 1024 * 1024,
              "cannot read bounded runtime ANE compiled artifact");
        check(matches_digest(bytes, digest), "runtime ANE artifact digest mismatch");
        check([bytes writeToFile:@(private_path.c_str()) atomically:NO],
              "cannot snapshot runtime ANE compiled artifact");
        std::filesystem::permissions(private_path, std::filesystem::perms::owner_read |
                                                  std::filesystem::perms::owner_write);
        check(matches_digest([NSData dataWithContentsOfFile:@(private_path.c_str())], digest),
              "runtime ANE private artifact digest mismatch");
        ++observed;
    }
    size_t expected = 0;
    for (id key in files) {
        check([key isKindOfClass:NSString.class], "runtime ANE receipt key must be a string");
        if ([(NSString *)key hasPrefix:@"graph.mlmodelc/"]) ++expected;
    }
    check(observed > 0 && observed == expected, "runtime ANE compiled artifact receipt is incomplete");
    return private_model;
}

size_t graph_estimate(const GraphShape &s) {
    const size_t matrices = s.kind == Kind::Matmul ? 1 : s.kind == Kind::GELU ? 2 : 3;
    const size_t weight_bytes = matrices * size_t(s.hidden) * s.width * 2;
    size_t estimate = 2 * weight_bytes + size_t(s.rows) * (6ull * s.width + 8ull * s.hidden) +
                      (64ull << 20);
    estimate += packed_conversion_max_workers * packed_conversion_scratch_per_worker;
    if (s.lora_inputs) estimate += size_t(s.rows) * s.width * 16;
    return estimate;
}

void admit_graph(size_t estimate, size_t budget, bool cpu_only) {
    if (estimate > budget) throw MemoryBudgetError("runtime ANE memory budget exceeded");
    if (!cpu_only) {
        const auto observed = observe_runtime_memory(0);
        const auto decision = admit_memory(observed, {uint64_t(4) << 30, budget}, 0, estimate);
        if (!decision.allowed())
            throw MemoryBudgetError("runtime ANE system memory admission denied (" +
                                    memory_denial_reason(decision.denial, observed) + ")");
    }
}

NSArray<NSString *> *weight_names(Kind kind) {
    return kind == Kind::Matmul ? @[@"w"] :
           kind == Kind::GELU ? @[@"wu", @"wd"] : @[@"wg", @"wu", @"wd"];
}

// Validate the loaded interface without allocating an IOSurface. An explicit
// invalid artifact must still fail preparation rather than become GPU fallback.
void check_model_interface(MLModel *model, const GraphShape &s) {
    NSDictionary *inputs = model.modelDescription.inputDescriptionsByName;
    NSDictionary *outputs = model.modelDescription.outputDescriptionsByName;
    NSArray<NSString *> *names = weight_names(s.kind);
    const NSUInteger input_count = 1 + names.count + (s.lora_inputs ? 2 : 0);
    check(inputs.count == input_count && outputs.count == (s.lora_inputs ? 2u : 1u),
          "runtime ANE feature count mismatch");
    auto check_feature = [&](NSString *name, int rows, int cols, NSDictionary *descriptions) {
        MLFeatureDescription *feature = descriptions[name];
        check(feature && feature.type == MLFeatureTypeMultiArray &&
                  feature.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat16 &&
                  [feature.multiArrayConstraint.shape isEqual:@[@(rows), @(cols)]],
              "runtime ANE compiled graph interface does not match its manifest");
    };
    check_feature(@"x", s.rows, s.hidden, inputs);
    for (NSString *name in names) {
        const bool down = [name isEqual:@"wd"];
        check_feature(name, down ? s.hidden : s.width, down ? s.width : s.hidden, inputs);
    }
    check_feature(@"y", s.rows, s.output_width(), outputs);
    if (s.lora_inputs) {
        check_feature(@"dg", s.rows, s.width, inputs);
        check_feature(@"du", s.rows, s.width, inputs);
        check_feature(@"h", s.rows, s.width, outputs);
    }
}
} // namespace

GraphShape runtime_template_shape(const std::filesystem::path &manifest) {
    @autoreleasepool {
        GraphShape shape;
        read_manifest_shape(manifest, shape);
        return shape;
    }
}

struct RuntimeGraph::Prepared::Impl {
    // The lease is destroyed last. Model load can keep lazy reads/file handles
    // into the snapshot even though this owner never predicts or creates slots.
    std::unique_ptr<ArtifactLease> lease;
    GraphShape shape;
    MLModel *model = nil;
    size_t estimate = 0;
    double artifact_time = 0, model_load_time = 0;
    bool cpu_only = false;
    ~Impl() { model = nil; }
};

RuntimeGraph::Prepared::Prepared(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RuntimeGraph::Prepared::~Prepared() = default;
RuntimeGraph::Prepared::Prepared(Prepared &&) noexcept = default;
RuntimeGraph::Prepared &RuntimeGraph::Prepared::operator=(Prepared &&) noexcept = default;
const GraphShape &RuntimeGraph::Prepared::shape() const {
    check(bool(impl_), "runtime ANE prepared graph was moved");
    return impl_->shape;
}
size_t RuntimeGraph::Prepared::estimated_bytes() const {
    check(bool(impl_), "runtime ANE prepared graph was moved");
    return impl_->estimate;
}
double RuntimeGraph::Prepared::artifact_seconds() const {
    check(bool(impl_), "runtime ANE prepared graph was moved");
    return impl_->artifact_time;
}
double RuntimeGraph::Prepared::model_load_seconds() const {
    check(bool(impl_), "runtime ANE prepared graph was moved");
    return impl_->model_load_time;
}

std::unique_ptr<RuntimeGraph::Prepared> RuntimeGraph::prepare(const std::filesystem::path &manifest,
        size_t budget, bool cpu_only, std::optional<GraphGeometry> expected) {
    @autoreleasepool {
        auto p = std::make_unique<Prepared::Impl>();
        auto artifact_start = Clock::now();
        p->lease = std::make_unique<ArtifactLease>();
        const auto path = verify_manifest(manifest, p->shape, *p->lease);
        const auto &s = p->shape;
        if (expected)
            check(s.kind == expected->kind && s.hidden == expected->hidden && s.width == expected->width &&
                      (!expected->require_lora_inputs || s.lora_inputs),
                  "runtime ANE graph does not match model FFN geometry");
        p->estimate = graph_estimate(s);
        p->cpu_only = cpu_only;
        // Charge the complete eventual graph before loading, not merely this
        // lightweight owner's currently visible payload.
        admit_graph(p->estimate, budget, cpu_only);
        p->artifact_time = seconds(artifact_start);
        auto model_start = Clock::now();
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = cpu_only ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
        NSError *error = nil;
        p->model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())]
                                   configuration:configuration error:&error];
        if (!p->model) throw CapabilityError("runtime ANE model load failed: " + message(error));
        check_model_interface(p->model, s);
        p->model_load_time = seconds(model_start);
        return std::unique_ptr<Prepared>(new Prepared(std::move(p)));
    }
}

struct RuntimeGraph::Impl {
    // First member is destroyed last, after Worker and MLModel have released
    // any lazy reads or file handles into the private compiled directory.
    std::unique_ptr<ArtifactLease> lease;
    GraphShape shape;
    MLModel *model = nil;
    MLDictionaryFeatureProvider *features = nil;
    MLPredictionOptions *options = nil;
    std::unique_ptr<Surface> input, output;
    std::unique_ptr<Surface> delta_gate, delta_up, hidden;
    std::vector<std::unique_ptr<Surface>> weights;
    size_t allocated = 0, estimate = 0;
    double artifact_time = 0, model_load_time = 0, bind_time = 0;
    bool staged = false, verified = false, scalar_staging = false;
    float headroom = 1.f;
    RunResult result;
    Worker worker; // destroyed first, before borrowed Core ML/IOSurface objects

    ~Impl() {
        // Explicitly release Core ML references before ArtifactLease removes
        // the compiled snapshot. Worker destruction alone is too late for the
        // body of this destructor, so drain it here as well.
        try { worker.join(); } catch (...) {}
        options = nil;
        features = nil;
        model = nil;
    }

    void copy_output(MLMultiArray *array, int width, uint16_t *destination, DType dtype) {
        check(array && array.dataType == MLMultiArrayDataTypeFloat16 &&
                  [array.shape isEqual:@[@(shape.rows), @(width)]],
              "runtime ANE output interface changed");
        const auto start = Clock::now();
        const size_t row_stride = [array.strides[0] unsignedLongLongValue];
        const size_t col_stride = [array.strides[1] unsignedLongLongValue];
        check(row_stride > 0 && col_stride > 0, "runtime ANE invalid output strides");
        __block bool finite = true;
        __block bool restored_finite = true;
        __block bool storage_valid = true;
        // dataPointer permanently locks an IOSurface-backed MLMultiArray,
        // which Core ML then refuses as output backing on the NEXT prediction.
        // Scoped access releases that lock before this arena is reused.
        const size_t elements = size_t(shape.rows - 1) * row_stride +
                                size_t(width - 1) * col_stride + 1;
        [array getBytesWithHandler:^(const void *bytes, NSInteger size) {
            if (!bytes || size < 0 || size_t(size) / 2 < elements) { storage_valid = false; return; }
            auto *src = static_cast<const uint16_t *>(bytes);
            const auto status = restore_fp16_matrix(src, destination, shape.rows, width,
                row_stride, col_stride, dtype, headroom, scalar_staging);
            finite = status.source_finite;
            restored_finite = status.restored_finite;
        }];
        check(storage_valid, "runtime ANE output storage is too small");
        if (destination) result.copied_output_bytes += size_t(shape.rows) * width * 2;
        result.output_seconds += seconds(start);
        if (!finite) throw OutputOverflow("runtime ANE returned NaN/Inf; GPU recomputation is required");
        check(restored_finite, "runtime ANE restored output exceeds destination dtype; GPU recomputation is required");
    }
    void predict(uint16_t *destination, DType dtype = DType::FP16, uint16_t *hidden_destination = nullptr) {
        NSError *error = nil;
        auto start = Clock::now();
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:features options:options error:&error];
        result.prediction_seconds += seconds(start);
        if (!prediction) throw std::runtime_error(message(error));
        ++result.calls;
        copy_output([prediction featureValueForName:@"y"].multiArrayValue,
                    shape.output_width(), destination, dtype);
        // Base requests can reuse the full LoRA graph without a hidden copy.
        // Still inspect its shape/storage and finite/restored range: a finite
        // y does not prove that the separately exposed hidden output is safe.
        if (shape.lora_inputs)
            copy_output([prediction featureValueForName:@"h"].multiArrayValue, shape.width,
                        hidden_destination, dtype);
    }
    void increase_headroom() {
        check(shape.kind != Kind::Matmul && headroom < 4096.f, "runtime ANE exhausted FP16 headroom");
        // Slot 1 is SwiGLU's up or GELU's down. Scaling the GELU INPUT
        // would change its nonlinear equation and is deliberately not used.
        auto &slot = *weights[1];
        slot.access([&](void *base, size_t pitch) {
            for (int r = 0; r < slot.rows; ++r) {
                auto *row = reinterpret_cast<uint16_t *>(static_cast<char *>(base) + size_t(r) * pitch);
                check(convert_fp16_row(row, row, slot.cols, DType::FP16, scalar_staging, .25f),
                      "runtime ANE nonfinite headroom weights");
            }
        });
        headroom *= 4.f;
        ++result.overflow_retries;
    }
};

RuntimeGraph::RuntimeGraph(const std::filesystem::path &manifest, size_t budget, bool cpu_only,
                           bool scalar_staging, std::optional<GraphGeometry> expected)
    : RuntimeGraph(prepare(manifest, budget, cpu_only, expected), budget, scalar_staging) {}

RuntimeGraph::RuntimeGraph(std::unique_ptr<Prepared> prepared, size_t budget, bool scalar_staging) {
    @autoreleasepool {
        const auto start = Clock::now();
        check(prepared && prepared->impl_, "runtime ANE binding requires a prepared graph");
        auto &source = *prepared->impl_;
        // A prepared model may have waited through encoder/weight loading.
        // Recheck the complete estimate against the NEW phase's live memory
        // before starting a worker or allocating the large slot bank.
        admit_graph(source.estimate, budget, source.cpu_only);
        impl_ = std::make_unique<Impl>();
        auto &p = *impl_;
        p.lease = std::move(source.lease);
        p.model = source.model;
        source.model = nil;
        p.shape = source.shape;
        p.estimate = source.estimate;
        p.artifact_time = source.artifact_time;
        p.model_load_time = source.model_load_time;
        p.scalar_staging = scalar_staging;
        const auto &s = p.shape;
        p.input = std::make_unique<Surface>(s.rows, s.hidden);
        p.output = std::make_unique<Surface>(s.rows, s.output_width());
        NSArray<NSString *> *names = weight_names(s.kind);
        NSMutableDictionary *values = [NSMutableDictionary dictionary];
        values[@"x"] = [MLFeatureValue featureValueWithMultiArray:p.input->array];
        if (s.lora_inputs) {
            p.delta_gate = std::make_unique<Surface>(s.rows, s.width);
            p.delta_up = std::make_unique<Surface>(s.rows, s.width);
            p.hidden = std::make_unique<Surface>(s.rows, s.width);
            values[@"dg"] = [MLFeatureValue featureValueWithMultiArray:p.delta_gate->array];
            values[@"du"] = [MLFeatureValue featureValueWithMultiArray:p.delta_up->array];
            p.allocated += p.delta_gate->bytes() + p.delta_up->bytes() + p.hidden->bytes();
        }
        for (NSString *name in names) {
            const bool down = [name isEqual:@"wd"];
            p.weights.push_back(std::make_unique<Surface>(down ? s.hidden : s.width, down ? s.width : s.hidden));
            values[name] = [MLFeatureValue featureValueWithMultiArray:p.weights.back()->array];
        }
        for (const auto &weight : p.weights) p.allocated += weight->bytes();
        p.allocated += p.input->bytes() + p.output->bytes();
        if (p.allocated > budget) throw MemoryBudgetError("runtime ANE padded IOSurfaces exceed memory budget");
        NSError *error = nil;
        // Also check the real backing shapes when binding. Preparation's ABI
        // validation cannot prove that newly allocated padded surfaces match.
        auto check_feature = [&](NSString *name, Surface &surface, NSDictionary *descriptions) {
            MLFeatureDescription *feature = descriptions[name];
            check(feature && feature.type == MLFeatureTypeMultiArray &&
                      feature.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat16 &&
                      [feature.multiArrayConstraint.shape isEqual:surface.array.shape],
                  "runtime ANE compiled graph interface does not match its manifest");
        };
        check(p.model.modelDescription.inputDescriptionsByName.count == values.count &&
                  p.model.modelDescription.outputDescriptionsByName.count == (s.lora_inputs ? 2u : 1u),
              "runtime ANE feature count mismatch");
        check_feature(@"x", *p.input, p.model.modelDescription.inputDescriptionsByName);
        for (NSUInteger i = 0; i < names.count; ++i)
            check_feature(names[i], *p.weights[i], p.model.modelDescription.inputDescriptionsByName);
        check_feature(@"y", *p.output, p.model.modelDescription.outputDescriptionsByName);
        if (s.lora_inputs) {
            check_feature(@"dg", *p.delta_gate, p.model.modelDescription.inputDescriptionsByName);
            check_feature(@"du", *p.delta_up, p.model.modelDescription.inputDescriptionsByName);
            check_feature(@"h", *p.hidden, p.model.modelDescription.outputDescriptionsByName);
        }
        p.features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&error];
        if (!p.features) throw std::runtime_error(message(error));
        p.options = [MLPredictionOptions new];
        p.options.outputBackings = s.lora_inputs ? @{@"y": p.output->array, @"h": p.hidden->array} :
                                                 @{@"y": p.output->array};
        p.bind_time = seconds(start);
    }
}
RuntimeGraph::~RuntimeGraph() = default;
const GraphShape &RuntimeGraph::shape() const { return impl_->shape; }
size_t RuntimeGraph::slot_bytes() const { return impl_->allocated; }
size_t RuntimeGraph::estimated_bytes() const { return impl_->estimate; }
double RuntimeGraph::artifact_seconds() const { return impl_->artifact_time; }
double RuntimeGraph::model_load_seconds() const { return impl_->model_load_time; }
double RuntimeGraph::bind_seconds() const { return impl_->bind_time; }
double RuntimeGraph::load_seconds() const {
    return artifact_seconds() + model_load_seconds() + bind_seconds();
}

bool RuntimeGraph::self_test(std::string &error) {
    auto &p = *impl_;
    p.worker.join();
    p.verified = p.staged = false;
    p.headroom = 1.f;
    p.worker.submit([&p] {
        const auto &s = p.shape;
        std::vector<uint16_t> result(size_t(s.rows) * s.output_width());
        // Only the capability check consumes hidden without an adapter.
        // Keep this storage local, not a permanent base-request scratch copy.
        std::vector<uint16_t> hidden_result(s.lora_inputs ? size_t(s.rows) * s.width : 0);
        p.result = {};
        // Sparse non-square, signed patterns check transpose, SiLU/GELU,
        // tail tiles and runtime input rebinding without allocating a second
        // full layer of weights or doing CPU GEMMs at the production shape.
        for (float scale : {0.125f, -0.25f}) {
            for (auto &weight : p.weights) weight->access([&](void *base, size_t pitch) {
                std::memset(base, 0, pitch * weight->rows);
                for (int r = 0; r < weight->rows; ++r)
                    reinterpret_cast<_Float16 *>(static_cast<char *>(base) + size_t(r) * pitch)[r % weight->cols] = _Float16(scale);
            });
            p.input->access([&](void *base, size_t pitch) {
                for (int r = 0; r < s.rows; ++r) for (int c = 0; c < s.hidden; ++c)
                    reinterpret_cast<_Float16 *>(static_cast<char *>(base) + size_t(r) * pitch)[c] =
                        _Float16(((r * 3 + c * 7) % 17 - 8) / 8.f);
            });
            if (s.lora_inputs) {
                // Nonzero signed corrections expose a graph that ignores the
                // adapter inputs or applies them after the nonlinearity.
                p.delta_gate->access([&](void *base, size_t pitch) {
                    for (int r = 0; r < s.rows; ++r) for (int c = 0; c < s.width; ++c)
                        reinterpret_cast<_Float16 *>(static_cast<char *>(base) + size_t(r) * pitch)[c] =
                            _Float16(scale * ((r + c) % 3 - 1));
                });
                p.delta_up->access([&](void *base, size_t pitch) {
                    for (int r = 0; r < s.rows; ++r) for (int c = 0; c < s.width; ++c)
                        reinterpret_cast<_Float16 *>(static_cast<char *>(base) + size_t(r) * pitch)[c] =
                            _Float16(scale * ((r * 2 + c) % 3 - 1));
                });
            }
            p.predict(result.data(), DType::FP16, hidden_result.data());
            for (int r = 0; r < s.rows; ++r) for (int c = 0; c < s.output_width(); ++c) {
                const int input_col = s.kind == Kind::Matmul ? c % s.hidden : (c % s.width) % s.hidden;
                const float x = ((r * 3 + input_col * 7) % 17 - 8) / 8.f;
                float expected = x * scale;
                if (s.kind == Kind::SwiGLU) {
                    const float gate = expected + (s.lora_inputs ? scale * ((r + c % s.width) % 3 - 1) : 0.f);
                    const float up = expected + (s.lora_inputs ? scale * ((r * 2 + c % s.width) % 3 - 1) : 0.f);
                    expected = gate / (1.f + std::exp(-gate)) * up * scale;
                }
                if (s.kind == Kind::GELU) expected = .5f * expected * (1.f + std::tanh(.7978845608f *
                    (expected + .044715f * expected * expected * expected))) * scale;
                const float actual = float(std::bit_cast<_Float16>(result[size_t(r) * s.output_width() + c]));
                if (!std::isfinite(actual) || std::abs(actual - expected) > 0.0002f + .02f * std::abs(expected))
                    throw std::runtime_error("runtime ANE numerical/weight-switch self-test failed: row=" +
                        std::to_string(r) + " col=" + std::to_string(c) + " scale=" + std::to_string(scale) +
                        " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
            }
            if (s.lora_inputs) for (int r = 0; r < s.rows; ++r) for (int c = 0; c < s.width; ++c) {
                const float x = ((r * 3 + (c % s.hidden) * 7) % 17 - 8) / 8.f;
                const float gate = x * scale + scale * ((r + c) % 3 - 1);
                const float up = x * scale + scale * ((r * 2 + c) % 3 - 1);
                const float expected = gate / (1.f + std::exp(-gate)) * up;
                const float actual = float(std::bit_cast<_Float16>(hidden_result[size_t(r) * s.width + c]));
                check(std::isfinite(actual) && std::abs(actual - expected) <= .0002f + .02f * std::abs(expected),
                      "runtime ANE hidden-output self-test failed");
            }
        }
        p.verified = true;
    });
    try { p.worker.join(); error.clear(); return true; }
    catch (const std::exception &e) { error = e.what(); return false; }
}

void RuntimeGraph::stage(std::vector<MatrixView> weights) {
    std::vector<WeightView> sources(weights.begin(), weights.end());
    stage_weights(std::move(sources));
}
void RuntimeGraph::stage_weights(std::vector<WeightView> weights) {
    auto &p = *impl_;
    p.worker.join();
    check(p.verified, "runtime ANE self-test must pass before staging");
    p.staged = false;
    p.result = {};
    p.worker.submit([&p, sources = std::move(weights)] {
        const auto start = Clock::now();
        try {
            check(sources.size() == p.weights.size(), "runtime ANE projection count mismatch");
            for (size_t i = 0; i < sources.size(); ++i)
                std::visit([&](const auto &source) {
                    p.weights[i]->fill(source, p.scalar_staging, i == 1 ? 1.f / p.headroom : 1.f);
                }, sources[i]);
            p.staged = p.result.ok = true;
        } catch (const std::exception &error) { p.result.error = error.what(); }
        p.result.stage_seconds = seconds(start);
    });
}
void RuntimeGraph::stage_matmul_parts(std::vector<MatrixView> parts) {
    auto &p = *impl_;
    p.worker.join();
    check(p.verified, "runtime ANE self-test must pass before staging");
    p.staged = false;
    p.result = {};
    p.worker.submit([&p, sources = std::move(parts)] {
        const auto start = Clock::now();
        try {
            check(p.shape.kind == Kind::Matmul && p.weights.size() == 1,
                  "runtime ANE parts require a MatMul weight slot");
            p.weights[0]->fill_matmul_parts(sources, p.scalar_staging);
            p.staged = p.result.ok = true;
        } catch (const std::exception &error) { p.result.error = error.what(); }
        p.result.stage_seconds = seconds(start);
    });
}
RunResult RuntimeGraph::wait_stage() { return finish(); }
void RuntimeGraph::launch(MatrixView input, uint16_t *output, size_t output_elements, DType output_dtype,
                          std::optional<AdapterInput> adapter) {
    auto &p = *impl_;
    p.worker.join();
    const double stage_time = p.result.stage_seconds;
    p.result = {};
    p.result.stage_seconds = stage_time;
    p.worker.submit([&p, input, output, output_elements, output_dtype, adapter] {
        const auto start = Clock::now();
        try {
            check(p.verified && p.staged, "runtime ANE has no verified staged weights");
            check(output_dtype == DType::FP16 || output_dtype == DType::BF16, "runtime ANE output dtype must be FP16/BF16");
            validate(input);
            const auto &s = p.shape;
            check(input.cols == s.hidden && input.rows % s.rows == 0 && output &&
                      output_elements >= size_t(input.rows) * s.output_width(), "runtime ANE chunk/output shape mismatch");
            check(!adapter || s.lora_inputs, "runtime ANE graph has no LoRA activation inputs");
            if (adapter) {
                validate(adapter->gate); validate(adapter->up);
                check(adapter->gate.rows == input.rows && adapter->up.rows == input.rows &&
                          adapter->gate.cols == s.width && adapter->up.cols == s.width &&
                          adapter->hidden && adapter->hidden_elements >= size_t(input.rows) * s.width,
                      "runtime ANE LoRA activation/hidden shape mismatch");
            } else if (s.lora_inputs) {
                for (auto *slot : {p.delta_gate.get(), p.delta_up.get()}) {
                    slot->access([&](void *base, size_t pitch) { std::memset(base, 0, pitch * slot->rows); });
                }
            }
            auto tile_view = [&](MatrixView source, int row) {
                source.data = static_cast<const char *>(source.data) + size_t(row) * stride(source);
                source.bytes -= size_t(row) * stride(source);
                source.rows = s.rows;
                return source;
            };
            for (int r = 0; r < input.rows; r += s.rows) {
                const auto convert_start = Clock::now();
                p.input->fill(tile_view(input, r), p.scalar_staging);
                if (adapter) p.delta_gate->fill(tile_view(adapter->gate, r), p.scalar_staging);
                p.result.input_seconds += seconds(convert_start);
                for (;;) {
                    try {
                        if (adapter) {
                            const auto delta_start = Clock::now();
                            // Refill on EVERY retry: base up and its activation
                            // delta must have the same changing headroom scale.
                            p.delta_up->fill(tile_view(adapter->up, r), p.scalar_staging, 1.f / p.headroom);
                            p.result.input_seconds += seconds(delta_start);
                        }
                        p.predict(output + size_t(r) * s.output_width(), output_dtype,
                                  adapter ? adapter->hidden + size_t(r) * s.width : nullptr);
                        break;
                    }
                    catch (const OutputOverflow &) {
                        if (s.kind == Kind::Matmul || p.headroom >= 4096.f) throw;
                        p.increase_headroom();
                    }
                }
            }
            p.result.ok = true;
        } catch (const std::exception &error) {
            p.result.error = error.what();
        }
        p.result.total_seconds = seconds(start);
        p.result.headroom_scale = p.headroom;
    });
}
RunResult RuntimeGraph::finish() {
    auto &p = *impl_;
    try { p.worker.join(); }
    catch (const std::exception &error) { p.result.ok = false; p.result.error = error.what(); }
    return p.result;
}
} // namespace tc::ane
