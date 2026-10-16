// Original complete M1024/H4096/F12288 consumer; constant paired prefix S2048.
#pragma once
#include <mlx/array.h>
#include <mlx/allocator.h>
#include <mlx/compile.h>
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/version.h>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace tcq {
namespace mx = mlx::core;
struct Adoption { std::atomic<unsigned> releases{0}; std::atomic<bool> pointer_ok{true}; };
struct Span {
    int fd = -1; void *mapping = MAP_FAILED; size_t length = 0, delta = 0, bytes = 0;
    explicit Span(NSDictionary *node) {
        uint64_t offset = integer(node, @"offset", 1ull << 46); bytes = size_t(integer(node, @"bytes"));
        check(bytes > 0, "empty source span"); fd = open([node[@"file"] fileSystemRepresentation], O_RDONLY);
        check(fd >= 0, "source open failed"); struct stat status{};
        check(fstat(fd, &status) == 0 && offset <= uint64_t(status.st_size) && bytes <= uint64_t(status.st_size) - offset, "source span bounds changed");
        delta = offset % size_t(getpagesize()); length = bytes + delta;
        mapping = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, off_t(offset - delta));
        check(mapping != MAP_FAILED, "readonly source mapping failed");
        check([sha(data(), bytes) isEqual:node[@"sha256"]], "actual source span digest differs");
    }
    ~Span() { if (mapping != MAP_FAILED) munmap(mapping, length); if (fd >= 0) close(fd); }
    Span(const Span &) = delete;
    const char *data() const { return static_cast<const char *>(mapping) + delta; }
};
id<MTLBuffer> metal_buffer(const mx::array &value) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)value.buffer().ptr();
    check(buffer && buffer.storageMode == MTLStorageModeShared, "known shared Metal backend buffer required"); return buffer;
}
mx::array upload(NSDictionary *node, mx::Shape shape) {
    Span source(node); size_t elements = 1; for (auto dimension : shape) elements *= size_t(dimension);
    check(source.bytes == elements * 2 && [node[@"dtype"] isEqual:@"BF16"], "complete original BF16 upload required");
    auto buffer = mx::allocator::malloc(source.bytes); check(buffer.ptr(), "source GPU allocation failed");
    std::memcpy(buffer.raw_ptr(), source.data(), source.bytes);
    return mx::array(buffer, std::move(shape), mx::bfloat16);
}
mx::array formula(const mx::array &input, const mx::array &gu, const mx::array &down) {
    auto pair = mx::split(mx::matmul(input, mx::transpose(gu)), 2, -1);
    return mx::matmul(mx::multiply(mx::multiply(pair[0], mx::sigmoid(pair[0])), pair[1]), mx::transpose(down));
}
struct Layout { uint32_t row, total; };
Layout port(NSDictionary *value) {
    check(integer(value, @"Batches") == 1 && integer(value, @"Channels") == 1 && integer(value, @"Height") == M &&
          integer(value, @"Width") == H && [value[@"Type"] isEqual:@"Float16"], "only witnessed real 1x1xMxH FP16 port supported");
    if (value[@"Offset"]) check(integer(value, @"Offset") == 0, "nonzero feature offset unsupported");
    if (value[@"Depth"]) check(integer(value, @"Depth") == 1, "unknown feature depth");
    if (value[@"PlaneCount"]) check(integer(value, @"PlaneCount") == 1, "unknown feature planes");
    uint32_t row = uint32_t(integer(value, @"RowStride")), plane = uint32_t(integer(value, @"PlaneStride")), total = uint32_t(integer(value, @"BatchStride"));
    check(row >= H * 2 && row % 2 == 0 && plane >= M * row && total >= plane && total <= 64 * 1024 * 1024, "invalid real feature pitch");
    return {row, total};
}
size_t allocation_size(size_t payload) { const size_t page = size_t(getpagesize()); return (payload + 2 * page - 1) / page * page; }
IOSurfaceRef surface(size_t payload) {
    size_t bytes = allocation_size(payload);
    auto value = IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes)});
    check(value && IOSurfaceGetAllocSize(value) >= bytes && IOSurfaceGetAllocSize(value) % getpagesize() == 0 &&
          reinterpret_cast<uintptr_t>(IOSurfaceGetBaseAddress(value)) % getpagesize() == 0, "owned page-aligned mapping failed"); return value;
}
void lock(IOSurfaceRef value) { check(IOSurfaceLock(value, 0, nullptr) == kIOReturnSuccess, "surface CPU lock failed"); }
void unlock(IOSurfaceRef value) { check(IOSurfaceUnlock(value, 0, nullptr) == kIOReturnSuccess, "surface CPU unlock failed"); }
void poison(void *base, size_t payload, size_t allocated) {
    std::memset(base, 0xff, payload); std::memset(static_cast<char *>(base) + payload, 0x6b, allocated - payload);
}
bool tail(const void *base, size_t payload, size_t allocated) {
    const auto *bytes = static_cast<const unsigned char *>(base);
    return std::all_of(bytes + payload, bytes + allocated, [](unsigned char value) { return value == 0x6b; });
}
mx::array adopt(id<MTLBuffer> owner, const Layout &layout, const std::shared_ptr<Adoption> &state, bool &pending) {
    void *base = owner.contents; auto buffer = mx::allocator::make_buffer(base, owner.length);
    check(buffer.ptr(), "strict MLX adoption failed; no copy fallback");
    mx::array flat(buffer, {int32_t(owner.length / 2)}, mx::float16, [base, state](mx::allocator::Buffer value) {
        if (value.raw_ptr() != base) state->pointer_ok.store(false);
        mx::allocator::release(value); state->releases.fetch_add(1);
    });
    auto view = mx::as_strided(flat, {1, int32_t(M), int32_t(H)}, {int64_t(M) * layout.row / 2, int64_t(layout.row / 2), 1}, 0);
    pending = true; view.eval(); mx::synchronize(); pending = false;
    check(metal_buffer(view).contents == base && view.offset() == 0 && metal_buffer(view).length == owner.length, "actual MLX restore view did not alias owner");
    return view;
}
NSDictionary *identity(id request, IOSurfaceRef input, IOSurfaceRef output, id<MTLBuffer> ib, id<MTLBuffer> ob) {
    NSArray *ins = [request inputArray], *outs = [request outputArray]; check(ins.count == 1 && outs.count == 1, "request ports changed");
    IOSurfaceRef ai = [ins[0] ioSurface], ao = [outs[0] ioSurface]; check(ai && ao, "request surfaces unavailable");
    NSDictionary *result = @{@"request_input_matches": @(IOSurfaceGetID(ai) == IOSurfaceGetID(input)),
        @"request_output_matches": @(IOSurfaceGetID(ao) == IOSurfaceGetID(output)), @"metal_input_matches": @(ib.contents == IOSurfaceGetBaseAddress(ai)),
        @"metal_output_matches": @(ob.contents == IOSurfaceGetBaseAddress(ao)), @"input_output_distinct": @(IOSurfaceGetID(ai) != IOSurfaceGetID(ao))};
    for (NSNumber *value in result.allValues) check(value.boolValue, "actual request/Metal identity differs"); return result;
}
struct Event { NSString *name; uint64_t start, duration; };
struct Evaluation { bool ok; uint64_t start, duration; NSString *domain; NSInteger code; };
class Worker {
    std::mutex mutex; std::condition_variable ready; bool stop = false; std::function<void()> task;
    std::thread thread;
public:
    Worker() : thread([this] {
        for (;;) {
            std::function<void()> work;
            { std::unique_lock<std::mutex> lock(mutex); ready.wait(lock, [this] { return stop || bool(task); });
              if (stop && !task) return; work = std::move(task); task = nullptr; }
            @autoreleasepool { work(); }
        }
    }) {}
    ~Worker() { { std::lock_guard<std::mutex> lock(mutex); stop = true; } ready.notify_one(); thread.join(); }
    std::future<Evaluation> submit(std::function<Evaluation()> function) {
        auto work = std::make_shared<std::packaged_task<Evaluation()>>(std::move(function)); auto result = work->get_future();
        { std::lock_guard<std::mutex> lock(mutex); check(!task, "only one owned ANE task allowed"); task = [work] { (*work)(); }; }
        ready.notify_one(); return result;
    }
};
const char *pack_shader = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void pack_bf16(device const ushort *x [[buffer(0)]], device half *y [[buffer(1)]],
                      constant uint &row [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= 1024u * 4096u) return;
    y[(i / 4096u) * row + i % 4096u] = half(as_type<float>(uint(x[i]) << 16));
}
)METAL";
void pack(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, const mx::array &input,
          id<MTLBuffer> destination, uint32_t row, bool &pending) {
    check(input.dtype() == mx::bfloat16 && input.size() == size_t(M) * H && input.offset() == 0 && input.flags().row_contiguous, "complete ready row-contiguous BF16 source required");
    id<MTLCommandBuffer> command = [queue commandBuffer]; id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    check(command && encoder, "pack command unavailable");
    [encoder setComputePipelineState:pipeline]; [encoder setBuffer:metal_buffer(input) offset:0 atIndex:0];
    [encoder setBuffer:destination offset:0 atIndex:1]; [encoder setBytes:&row length:sizeof(row) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(M * H, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding]; pending = true; [command commit]; [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted, "owned pack did not complete"); pending = false;
}
NSDictionary *metric(const float *expected, const float *actual, size_t count) {
    double energy = 0, other = 0, error = 0, dot = 0, maximum = 0; bool finite = true;
    for (size_t i = 0; i < count; ++i) {
        double a = expected[i], b = actual[i], d = a - b;
        finite &= std::isfinite(a) && std::isfinite(b); energy += a*a; other += b*b; error += d*d; dot += a*b; maximum = std::max(maximum, std::abs(d));
    }
    if (!finite) return @{@"finite": @NO, @"relative_l2": NSNull.null, @"cosine": NSNull.null, @"max_abs": NSNull.null};
    return @{@"finite": @YES, @"relative_l2": @(energy ? std::sqrt(error / energy) : (error == 0 ? 0 : 1)),
             @"cosine": @(energy && other ? dot / std::sqrt(energy * other) : (error == 0 ? 1 : 0)), @"max_abs": @(maximum)};
}
bool passed(NSDictionary *value) {
    return [value[@"finite"] boolValue] && [value[@"relative_l2"] doubleValue] <= .03 && [value[@"cosine"] doubleValue] >= .999;
}
NSDictionary *sample_metric(NSDictionary *oracle, const float *actual) {
    NSArray *rows = oracle[@"rows"], *cols = oracle[@"cols"], *values = oracle[@"values"];
    check(rows.count == 4 && cols.count == 4 && values.count == 4, "independent full-K/F sample geometry required");
    // Preserve the independent double reference; do not round it to FP32.
    double energy = 0, other = 0, error = 0, dot = 0, maximum = 0; bool finite = true;
    for (size_t r = 0; r < 4; ++r) for (size_t c = 0; c < 4; ++c) {
        uint64_t row = [rows[r] unsignedLongLongValue], col = [cols[c] unsignedLongLongValue]; check(row < M && col < H, "sample coordinate cap");
        double a = [values[r][c] doubleValue], b = actual[row * H + col], delta = a - b;
        finite &= std::isfinite(a) && std::isfinite(b); energy += a*a; other += b*b; error += delta*delta; dot += a*b; maximum = std::max(maximum, std::abs(delta));
    }
    if (!finite) return @{@"finite": @NO, @"relative_l2": NSNull.null, @"cosine": NSNull.null, @"max_abs": NSNull.null};
    return @{@"finite": @YES, @"relative_l2": @(std::sqrt(error / energy)), @"cosine": @(dot / std::sqrt(energy * other)), @"max_abs": @(maximum)};
}
struct Point {
    NSString *name; mx::array input; std::vector<float> reference, prefix; NSDictionary *oracle;
    Point(NSDictionary *value) : name(value[@"id"]), input(upload(value[@"input"], {1, int32_t(M), int32_t(H)})), oracle(value[@"oracle"]) {
        Span source(value[@"reference"]); check(source.bytes == size_t(M) * H * 4, "full reference required");
        reference.resize(size_t(M) * H); std::memcpy(reference.data(), source.data(), source.bytes);
    }
};
void benchmark(Private &session, NSDictionary *manifest, NSMutableDictionary *report, const char *path,
               std::vector<IOSurfaceRef> &surfaces, NSMutableArray *owners, const std::shared_ptr<Adoption> &adoption,
               std::atomic<bool> &evaluation_unsafe, bool &gpu_pending, bool &mlx_pending) {
    Layout in = port(report[@"input_descriptor"]), out = port(report[@"output_descriptor"]);
    check([manifest[@"gates"] isEqual:@{@"relative_l2_max": @.03, @"cosine_min": @.999}], "fixed quality gates required");
    mx::set_default_device(mx::Device::gpu); check(mx::is_available(mx::Device::gpu) && std::string(mx::version()) == "0.32.0", "version-bound MLX GPU required");
    mx::enable_compile(); report[@"gpu_compile_enabled"] = @YES;
    report[@"mlx_version"] = [NSString stringWithUTF8String:mx::version()];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice(); check(device && device.hasUnifiedMemory, "unified-memory Metal required");
    report[@"gpu_name"] = device.name; report[@"shape_mhf"] = @[@(M), @(H), @(F)]; report[@"share"] = @(S);
    report[@"page_bytes"] = @(getpagesize());
    const auto &info = mx::device_info(); auto architecture = info.find("architecture");
    report[@"mlx_architecture"] = architecture == info.end() ? (id)NSNull.null : [NSString stringWithUTF8String:std::get<std::string>(architecture->second).c_str()];
    report[@"gates"] = manifest[@"gates"]; report[@"logical_gpu_weight_bytes"] = @(6ull * H * F * 2); report[@"logical_private_weight_bytes"] = @(6ull * H * S);
    report[@"boundary"] = @"ready original BF16 GPU input to complete BF16 GPU output; GPU pack, copy control, evaluate, suffix, FP32 restore/join and MLX drain included";
    report[@"pre_arm_validation_drain"] = @YES; report[@"formal_performance_score"] = @YES;
    auto gu = upload(manifest[@"weights"][0], {2 * int32_t(F), int32_t(H)}), down = upload(manifest[@"weights"][1], {int32_t(H), int32_t(F)});
    auto pg = mx::contiguous(mx::concatenate({mx::slice(gu, {0,0}, {int32_t(S),int32_t(H)}), mx::slice(gu, {int32_t(F),0}, {int32_t(F+S),int32_t(H)})}, 0));
    auto pd = mx::contiguous(mx::slice(down, {0,0}, {int32_t(H),int32_t(S)}));
    auto rg = mx::contiguous(mx::concatenate({mx::slice(gu, {int32_t(S),0}, {int32_t(F),int32_t(H)}), mx::slice(gu, {int32_t(F+S),0}, {2*int32_t(F),int32_t(H)})}, 0));
    auto rd = mx::contiguous(mx::slice(down, {0,int32_t(S)}, {int32_t(H),int32_t(F)}));
    mlx_pending = true; for (auto *value : {&gu,&down,&pg,&pd,&rg,&rd}) value->eval(); mx::synchronize(); mlx_pending = false;
    using Function = std::function<std::vector<mx::array>(const std::vector<mx::array> &)>;
    auto full = mx::compile(Function([gu,down](const auto &args) { return std::vector<mx::array>{formula(args[0],gu,down)}; }));
    auto suffix = mx::compile(Function([rg,rd](const auto &args) { return std::vector<mx::array>{formula(args[0],rg,rd)}; }));
    auto prefix = mx::compile(Function([pg,pd](const auto &args) { return std::vector<mx::array>{formula(args[0],pg,pd)}; }));
    auto join = mx::compile(Function([](const auto &args) {
        return std::vector<mx::array>{mx::astype(mx::add(mx::astype(args[0],mx::float32), mx::astype(args[1],mx::float32)),mx::bfloat16)};
    }));
    std::vector<Point> points;
    for (NSDictionary *value in manifest[@"points"]) {
        points.emplace_back(value); auto &point = points.back(); mlx_pending = true;
        auto reference = mx::astype(prefix({point.input})[0], mx::float32); reference.eval(); mx::synchronize();
        point.prefix.assign(reference.data<float>(), reference.data<float>() + M * H); mlx_pending = false;
    }
    check(points.size() >= 2 && points.size() <= 3, "alternating complete actual input points required");
    IOSurfaceRef input = surface(in.total); surfaces.push_back(input); IOSurfaceRef output = surface(out.total); surfaces.push_back(output);
    id<MTLBuffer> ib = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(input) length:IOSurfaceGetAllocSize(input) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ob = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(output) length:IOSurfaceGetAllocSize(output) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> si = [device newBufferWithLength:IOSurfaceGetAllocSize(input) options:MTLResourceStorageModeShared];
    id<MTLBuffer> so = [device newBufferWithLength:IOSurfaceGetAllocSize(output) options:MTLResourceStorageModeShared];
    check(ib && ob && si && so && ib.contents == IOSurfaceGetBaseAddress(input) && ob.contents == IOSurfaceGetBaseAddress(output), "actual surface/Metal mappings required");
    [owners addObjectsFromArray:@[ib,ob,si,so]];
    auto shared_view = adopt(ob, out, adoption, mlx_pending), copy_view = adopt(so, out, adoption, mlx_pending);
    report[@"restore_views_alias_actual_buffers"] = @(metal_buffer(shared_view).contents == ob.contents && metal_buffer(copy_view).contents == so.contents);
    report[@"surface_allocation_bytes"] = @[@(ib.length), @(ob.length)];
    id io = [session.io_class objectWithIOSurface:input], oo = [session.io_class objectWithIOSurface:output];
    id request = [session.request_class requestWithInputs:@[io] inputIndices:@[@0] outputs:@[oo] outputIndices:@[@0] perfStats:nil procedureIndex:@0];
    check(request && [request validate], "real private request failed validation"); report[@"initial_identity"] = identity(request,input,output,ib,ob);
    NSError *error = nil; id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:pack_shader] options:nil error:&error];
    check(queue && library, "original BF16 pack library unavailable");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pack_bf16"] error:&error];
    check(pipeline, "original BF16 pack pipeline unavailable"); Worker worker;
    NSArray *arms = @[@"gpu", @"copy-serial", @"copy-hybrid", @"shared-serial", @"shared-hybrid"];
    uint32_t repeats = uint32_t(integer(manifest, @"measured_groups")); check(repeats == 10 || repeats == 30, "bounded balanced groups required");
    report[@"measured_groups"] = @(repeats); NSMutableArray *point_names = [NSMutableArray array]; for (auto &point : points) [point_names addObject:point.name]; report[@"point_cycle"] = point_names;
    report[@"full_exact_runtime_join_witness"] = @YES;
    std::vector<uint16_t> previous_shared_output; NSString *previous_shared_point = nil;
    for (uint32_t group = 0; group < repeats + 3; ++group) {
        auto &point = points[group % points.size()]; NSString *phase = group == 0 ? @"first" : group < 3 ? @"warmup" : @"measured";
        uint32_t trial = group == 0 ? 0 : group < 3 ? group - 1 : group - 3;
        uint32_t order_row = group < 3 ? group : trial;
        // Ten rows: cyclic order and reversed cyclic order, every arm at every position twice.
        for (uint32_t position = 0; position < 5; ++position) {
            uint32_t index = (order_row % 5 + ((order_row / 5) % 2 ? 4 - position : position)) % 5;
            NSString *arm = arms[index]; bool gpu = [arm isEqual:@"gpu"], copied = [arm hasPrefix:@"copy"], hybrid = [arm hasSuffix:@"hybrid"];
            mlx_pending = true; mx::synchronize(); mlx_pending = false;
            if (!gpu) {
                lock(input); poison(ib.contents,in.total,ib.length); unlock(input); lock(output); poison(ob.contents,out.total,ob.length); unlock(output);
                poison(si.contents,in.total,si.length); poison(so.contents,out.total,so.length);
            }
            std::vector<Event> events; auto note = [&](NSString *name, uint64_t begin) { events.push_back({name,begin,now()-begin}); };
            const uint64_t started = now(); std::optional<mx::array> result, retained_remainder; NSDictionary *before = nil, *after = nil;
            if (gpu) { mlx_pending = true; result.emplace(full({point.input})[0]); result->eval(); mx::synchronize(); mlx_pending = false; }
            else {
                uint64_t t = now(); pack(queue,pipeline,point.input,copied ? si : ib,in.row/2,gpu_pending); note(@"gpu-bf16-to-fp16-pack-complete",t);
                if (copied) { t=now(); lock(input); std::memcpy(ib.contents,si.contents,in.total); unlock(input); note(@"cpu-input-feature-copy",t); }
                before = identity(request,input,output,ib,ob);
                auto evaluate = [&session,request,&evaluation_unsafe]() -> Evaluation {
                    @autoreleasepool {
                        NSError *error = nil; uint64_t t = now(); bool ok = false; NSString *domain = @"none"; NSInteger code = 0;
                        evaluation_unsafe.store(true);
                        @try { ok = [session.client evaluateWithModel:session.model options:@{@"kANEFPerformanceStatsMask": @0} request:request qos:25 error:&error];
                               if (!ok) { domain = error.domain ?: @"unknown"; code = error.code; } }
                        @catch (NSException *error) { domain = error.name; }
                        if (ok) evaluation_unsafe.store(false);
                        return {ok,t,now()-t,domain,code};
                    }
                };
                std::optional<Evaluation> evaluation; std::future<Evaluation> future;
                if (hybrid) future = worker.submit(evaluate); else evaluation.emplace(evaluate());
                uint64_t t2=now(); mlx_pending = true; retained_remainder.emplace(suffix({point.input})[0]); auto &remainder = *retained_remainder;
                remainder.eval(); mx::synchronize(); mlx_pending = false; note(@"gpu-suffix-output-ready",t2);
                if (hybrid) evaluation.emplace(future.get());
                events.push_back({@"private-evaluate-returned",evaluation->start,evaluation->duration});
                if (!evaluation->ok) report[@"private_error"] = @{@"domain":evaluation->domain,@"code":@(evaluation->code)};
                check(evaluation->ok, "synchronous real-prefix evaluate failed; mapping quarantined");
                after = identity(request,input,output,ib,ob);
                if (copied) { t2=now(); lock(output); std::memcpy(so.contents,ob.contents,out.total); unlock(output); note(@"cpu-output-feature-copy",t2); }
                t2=now(); mlx_pending = true; result.emplace(join({remainder,copied ? copy_view : shared_view})[0]); result->eval(); mx::synchronize(); mlx_pending = false;
                note(@"gpu-fp32-restore-join-bf16-drain",t2);
            }
            uint64_t wall=now()-started;
            // All validation conversions/readback are outside timing, drained before the next arm.
            mlx_pending = true; auto actual = mx::astype(*result,mx::float32); actual.eval(); mx::synchronize(); mlx_pending = false;
            const float *values = actual.data<float>();
            NSMutableDictionary *metrics = [@{@"full_original":metric(point.reference.data(),values,size_t(M)*H),
                @"independent_full_sample":sample_metric(point.oracle[@"full"],values)} mutableCopy];
            bool guards = true; NSDictionary *join_binding = nil, *stale_join = nil;
            if (!gpu) {
                mlx_pending=true; auto partial=mx::astype(copied ? copy_view : shared_view,mx::float32); partial.eval(); mx::synchronize(); mlx_pending=false;
                metrics[@"prefix_original"] = metric(point.prefix.data(),partial.data<float>(),size_t(M)*H);
                metrics[@"independent_prefix_sample"] = sample_metric(point.oracle[@"prefix"],partial.data<float>());
                mlx_pending=true; auto tail_values=mx::astype(*retained_remainder,mx::float32); tail_values.eval(); mx::synchronize(); mlx_pending=false;
                const float *tv=tail_values.data<float>(), *pv=partial.data<float>();
                check(result->dtype() == mx::bfloat16 && result->size() == size_t(M)*H, "complete original BF16 join output required");
                const auto *actual_bits = result->data<uint16_t>(); std::vector<uint16_t> expected_bits(size_t(M)*H);
                size_t mismatches=0;
                for (size_t i=0;i<expected_bits.size();++i) {
                    // Independent CPU FP32 addition then explicit BF16 round-to-nearest-even.
                    // This is outside timing and does not change the fixed model-accuracy budget.
                    float sum=tv[i]+pv[i]; uint32_t bits=0; std::memcpy(&bits,&sum,4);
                    expected_bits[i]=uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
                    mismatches += expected_bits[i] != actual_bits[i];
                }
                join_binding=@{@"elements":@(expected_bits.size()),@"exact_mismatches":@(mismatches),
                    @"expected_bf16_sha256":sha(expected_bits.data(),expected_bits.size()*2),@"actual_bf16_sha256":sha(actual_bits,expected_bits.size()*2),
                    @"reference_kind":@"independent CPU FP32 add and explicit BF16 RNE from actual tail/private outputs"};
                if ([arm isEqual:@"shared-hybrid"]) {
                    if (previous_shared_point && ![previous_shared_point isEqual:point.name]) {
                        size_t wrong=0; for (size_t i=0;i<expected_bits.size();++i) wrong += previous_shared_output[i] != expected_bits[i];
                        stale_join=@{@"previous_actual_point":previous_shared_point,@"expected_point":point.name,@"mismatched_elements":@(wrong),
                            @"used_for_speed_ratio":@NO,@"new_device_dispatch_performed":@NO};
                        check(wrong>0,"old actual compiled join output did not reject different actual point");
                    }
                    previous_shared_output.assign(actual_bits,actual_bits+expected_bits.size()); previous_shared_point=point.name;
                }
                guards &= mismatches == 0;
                lock(input); lock(output); guards = tail(ib.contents,in.total,ib.length) && tail(ob.contents,out.total,ob.length) &&
                    tail(si.contents,in.total,si.length) && tail(so.contents,out.total,so.length) && guards; unlock(output); unlock(input);
            }
            bool numeric = true; for (NSDictionary *value in metrics.allValues) numeric &= passed(value);
            if (gpu) numeric &= [metrics[@"full_original"][@"relative_l2"] doubleValue] == 0 && [metrics[@"full_original"][@"max_abs"] doubleValue] == 0;
            std::sort(events.begin(),events.end(),[](const Event &a,const Event &b){return a.start<b.start;}); NSMutableArray *host = [NSMutableArray array];
            for (const auto &event : events) [host addObject:@{@"name":event.name,@"start_ns":@(event.start-started),@"duration_ns":@(event.duration),@"clock_domain":@"same-process-uptime-raw-host"}];
            NSMutableDictionary *sample = [@{@"phase":phase,@"trial":@(trial),@"arm":arm,@"point":point.name,@"wall_ns":@(wall),
                @"output_dtype":@"bfloat16",@"metrics":metrics,@"numeric_pass":@(numeric),@"tail_guards_pass":@(guards),@"host_events":host,
                @"external_feature_copy_count":copied ? @2 : @0,@"external_feature_copy_bytes":copied ? @(in.total+out.total) : @0,
                @"physical_overlap_qualified":@NO,@"runtime_internal_copy_behavior":@"unknown"} mutableCopy];
            if (!gpu) { sample[@"identity_before"]=before; sample[@"identity_after"]=after; sample[@"exact_runtime_join"]=join_binding; }
            if (stale_join) sample[@"stale_actual_join_control"]=stale_join;
            [report[@"samples"] addObject:sample]; save(report,path);
            check(numeric && guards, "fixed complete-source/prefix/FP64 or tail gate rejected arm");
        }
    }
    report[@"final_identity"] = identity(request,input,output,ib,ob);
    report[@"compiled_join_uses_runtime_partial_argument"] = @YES;
    report[@"status"] = @"measured";
}
}
