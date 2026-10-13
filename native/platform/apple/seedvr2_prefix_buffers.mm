#include "../../models/seedvr2/prefix_buffers.hpp"
#include "../../models/seedvr2/prefix_buffer_lease.hpp"
#include "../../models/seedvr2/fixed_feature_shape.hpp"
#include "../../backends/coreml_output_copy.hpp"
#import <CoreML/CoreML.h>
#include <algorithm>
#include <set>

namespace tc::seedvr2 {
namespace {
const void *allocation_pointer(const Tensor &tensor) {
    // Compare the allocation identity, not an offset data pointer. This does
    // not ask MLX to expose/copy GPU-private data into CPU-accessible storage.
    return tensor.buffer().ptr();
}
bool fixed_shape(MLMultiArrayConstraint *constraint, NSArray<NSNumber *> *expected) {
    auto shape = constraint.shapeConstraint;
    std::vector<size_t> dimensions;
    for (NSNumber *value in expected) dimensions.push_back(value.unsignedLongLongValue);
    std::vector<std::vector<size_t>> enumerated;
    for (NSArray<NSNumber *> *values in shape.enumeratedShapes) {
        std::vector<size_t> dimensions;
        for (NSNumber *value in values) dimensions.push_back(value.unsignedLongLongValue);
        enumerated.push_back(std::move(dimensions));
    }
    std::vector<std::pair<size_t,size_t>> ranges;
    for (NSValue *value in shape.sizeRangeForDimension) {
        auto range = value.rangeValue;
        ranges.emplace_back(range.location,range.length);
    }
    return [constraint.shape isEqual:expected] &&
        exact_fixed_feature_shape(int(shape.type),dimensions,enumerated,ranges);
}
}
struct PrefixBuffers::Impl {
    NSMutableArray<MLModel *> *models = [NSMutableArray array];
    NSMutableArray<MLMultiArray *> *backings = [NSMutableArray array];
    NSMutableArray<MLPredictionOptions *> *options = [NSMutableArray array];
    std::vector<Tensor> storage;
    PrefixBufferLease lease;
    int rows = 0, hidden = 0;
    uint64_t calls = 0, copied_bytes = 0;
    bool failed = false;
    ~Impl() {
        // Models/options retain caller backings. Drain those owners before
        // releasing the allocation whose pointer was passed to Core ML.
        @autoreleasepool {
            options = nil;
            backings = nil;
            models = nil;
        }
        storage.clear();
    }
};

PrefixBuffers::PrefixBuffers(const std::vector<std::filesystem::path> &paths,
                             int rows, int hidden, const Event &event,
                             std::atomic<bool> &cancelled, bool cpu_only_control)
    : impl_(std::make_unique<Impl>()) {
    require(rows > 0 && rows <= 1024 && hidden == 2560 &&
            !paths.empty() && paths.size() <= 32, "invalid SeedVR2 prefix buffer geometry");
    impl_->rows = rows;
    impl_->hidden = hidden;
    @autoreleasepool {
        auto configuration = [MLModelConfiguration new];
        // CPU-only is an explicit diagnostic control, never an environment
        // override. In either case this branch excludes GPU execution.
        configuration.computeUnits = cpu_only_control ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
        configuration.functionName = @"main";
        NSArray *shape = @[@1, @(hidden), @1, @(rows)];
        for (size_t index = 0; index < paths.size(); ++index) {
            checkpoint(cancelled);
            require(paths[index].is_absolute() && paths[index].extension() == ".mlmodelc" &&
                    std::filesystem::is_directory(paths[index]), "invalid SeedVR2 prefix artifact path");
            if (event) event("coreml_partition_load", int(index), int(paths.size()));
            NSError *error = nil;
            auto model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(paths[index].c_str())]
                                          configuration:configuration error:&error];
            require(model != nil, "SeedVR2 prefix model load failed: " +
                    std::string(error ? error.localizedDescription.UTF8String : "unknown"));
            const auto inputs = model.modelDescription.inputDescriptionsByName;
            const auto outputs = model.modelDescription.outputDescriptionsByName;
            const auto input = inputs[@"x"].multiArrayConstraint;
            const auto output = outputs[@"y"].multiArrayConstraint;
            require(inputs.count == 1 && outputs.count == 1 && input && output &&
                    input.dataType == MLMultiArrayDataTypeFloat16 &&
                    output.dataType == MLMultiArrayDataTypeFloat16 &&
                    fixed_shape(input,shape) && fixed_shape(output,shape),
                    "SeedVR2 prefix pool requires an exact fixed FP16 x/y feature ABI");
            [impl_->models addObject:model];
        }
        checkpoint(cancelled);
        if (event) event("coreml_partition_load", int(paths.size()), int(paths.size()));
    }
}

PrefixBuffers::~PrefixBuffers() {
    // Exception paths can leave a submitted GPU suffix or join outstanding.
    // Complete consumers before destroying externally writable output pools.
    if (impl_ && impl_->lease.active()) {
        try { mx::synchronize(); } catch (...) {}
    }
}

void PrefixBuffers::prepare(size_t chunks) {
    constexpr size_t maximum_bytes = 256ull << 20;
    const size_t per_slot = size_t(impl_->rows) * impl_->hidden * sizeof(mx::float16_t);
    require(!impl_->failed && !impl_->lease.active() && chunks > 0 && chunks <= maximum_chunks() &&
            chunks <= maximum_bytes / per_slot, "SeedVR2 prefix backing pool exceeds its 256-MiB bound or is still leased");
    // Allocation/materialization must happen before async suffix submission.
    // Shrinking a workload may reuse the larger, already bounded pool.
    try {
    const auto previous = impl_->storage.size();
    for (size_t slot = previous; slot < chunks; ++slot)
        impl_->storage.push_back(mx::contiguous(mx::zeros({1, impl_->rows, impl_->hidden}, mx::float16)));
    if (chunks > previous) {
        mx::eval(impl_->storage);
        @autoreleasepool {
            std::set<const void *> pointers;
            for (const auto &storage : impl_->storage) {
                require(storage.is_available() && storage.flags().row_contiguous &&
                        pointers.insert(allocation_pointer(storage)).second,
                        "SeedVR2 prefix output backings alias or are not materialized");
            }
            for (size_t slot = previous; slot < chunks; ++slot) {
                NSError *error = nil;
                auto backing = [[MLMultiArray alloc]
                    initWithDataPointer:impl_->storage[slot].data<mx::float16_t>()
                    shape:@[@1, @(impl_->hidden), @1, @(impl_->rows)]
                    dataType:MLMultiArrayDataTypeFloat16
                    strides:@[@(per_slot / 2), @1, @(per_slot / 2), @(impl_->hidden)]
                    deallocator:^(void *) {} error:&error];
                require(backing != nil, "SeedVR2 prefix output binding failed");
                auto options = [MLPredictionOptions new];
                options.outputBackings = @{@"y": backing};
                [impl_->backings addObject:backing];
                [impl_->options addObject:options];
            }
        }
    }
    impl_->lease.begin(chunks);
    } catch (...) {
        // A partial allocation/binding cannot be retried as an initialized
        // pool. The owner must destroy/recreate the failed partition session.
        impl_->failed = true;
        throw;
    }
}

Tensor PrefixBuffers::predict(int block, size_t chunk, const Tensor &input) {
    require(!impl_->failed && block >= 0 && size_t(block) < impl_->models.count &&
            input.shape() == mx::Shape{1, impl_->rows, impl_->hidden} &&
            input.dtype() == mx::float16 && input.flags().row_contiguous && input.is_available(),
            "SeedVR2 buffered prefix requires already-materialized contiguous FP16 input");
    impl_->lease.check(chunk);
    try {
    @autoreleasepool {
        // Deliberately no eval/input-packing work here: GPU suffix is in flight.
        NSError *error = nil;
        const size_t positions = size_t(impl_->rows) * impl_->hidden;
        auto x = [[MLMultiArray alloc] initWithDataPointer:(void *)input.data<mx::float16_t>()
            shape:@[@1, @(impl_->hidden), @1, @(impl_->rows)] dataType:MLMultiArrayDataTypeFloat16
            strides:@[@(positions), @1, @(positions), @(impl_->hidden)] deallocator:^(void *) {} error:&error];
        require(x != nil, "SeedVR2 prefix input binding failed");
        auto provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:
            @{@"x": [MLFeatureValue featureValueWithMultiArray:x]} error:&error];
        require(provider != nil, "SeedVR2 prefix feature provider failed");
        auto result = [impl_->models[block] predictionFromFeatures:provider
            options:impl_->options[chunk] error:&error];
        require(result != nil, "SeedVR2 prefix prediction failed: " +
            std::string(error ? error.localizedDescription.UTF8String : "unknown"));
        auto y = [result featureValueForName:@"y"].multiArrayValue;
        require(y && y.dataType == MLMultiArrayDataTypeFloat16 &&
                [y.shape isEqual:@[@1, @(impl_->hidden), @1, @(impl_->rows)]] && y.strides.count == 4,
                "SeedVR2 prefix output shape/dtype mismatch");
        // Pointer equality alone is insufficient: a framework-owned tensor can
        // have a different row/channel layout. Verify/copy by actual strides.
        const size_t row_stride = [y.strides[3] unsignedLongLongValue];
        const size_t channel_stride = [y.strides[1] unsignedLongLongValue];
        auto *destination = impl_->storage[chunk].data<mx::float16_t>();
        const bool shared = y.dataPointer == destination && row_stride == size_t(impl_->hidden) && channel_stride == 1;
        if (!shared) {
            require(y.dataPointer != destination, "SeedVR2 prefix backing returned aliased storage with incompatible strides");
            require(row_stride > 0 && channel_stride > 0,
                    "SeedVR2 prefix output has invalid strides");
            const auto address = reinterpret_cast<uintptr_t>(y.dataPointer);
            for (size_t slot = 0; slot < impl_->storage.size(); ++slot) {
                if (slot == chunk) continue;
                const auto begin = reinterpret_cast<uintptr_t>(impl_->storage[slot].data<mx::float16_t>());
                require(address < begin || address-begin >= impl_->storage[slot].nbytes(),
                        "SeedVR2 prefix prediction aliases another live output slot");
            }
            copy_coreml_fp16(reinterpret_cast<uint16_t *>(destination),
                static_cast<const uint16_t *>(y.dataPointer), impl_->rows, impl_->hidden,
                row_stride, channel_stride, true);
            impl_->copied_bytes += positions * 2;
        }
        ++impl_->calls;
        impl_->lease.mark_written(chunk);
        return impl_->storage[chunk];
    }
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}

void PrefixBuffers::release_after(const Tensor &owned_join) {
    require(owned_join.dtype() == mx::float16 && owned_join.ndim() == 2 &&
            owned_join.shape(1) == impl_->hidden && owned_join.is_available(),
            "SeedVR2 prefix pool release requires a completed FP16 joined result");
    for (const auto &storage : impl_->storage)
        // Comparing data pointers alone would miss an offset view into the
        // same backing. The entire joined allocation must be independent.
        require(allocation_pointer(owned_join) != allocation_pointer(storage),
                "SeedVR2 prefix joined result aliases externally mutable backing");
    impl_->lease.release();
}

uint64_t PrefixBuffers::calls() const { return impl_->calls; }
uint64_t PrefixBuffers::copied_bytes() const { return impl_->copied_bytes; }
size_t PrefixBuffers::backing_slots() const { return impl_->storage.size(); }
size_t PrefixBuffers::backing_bytes() const {
    return impl_->storage.size() * size_t(impl_->rows) * impl_->hidden * 2;
}
size_t PrefixBuffers::maximum_chunks() const {
    constexpr size_t budget = 256ull << 20;
    return std::min(size_t(64),budget / (size_t(impl_->rows)*impl_->hidden*2));
}
} // namespace tc::seedvr2
