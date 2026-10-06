#include "../../native/backends/ane_calibration_memory.hpp"
#include <cassert>
#include <array>
#include <iostream>

int main() {
    using namespace tc::ane;
    auto allocation = [](uint64_t r, uint64_t c, uint64_t item, uint64_t page) {
        const auto pitch = (c * item + 63) / 64 * 64;
        return (r * pitch + page - 1) / page * page;
    };
    for (uint64_t page : {4096u, 16384u}) {
        const std::array<CalibrationSurfaceShape, 3> snapshots{{{128, 33, 2}, {128, 1, 2}, {1, 33, 2}}};
        const auto base = plan_gpu_calibration_memory(33, 128, 1024, false, snapshots, page);
        const auto lora = plan_gpu_calibration_memory(33, 128, 1024, true, snapshots, page);
        assert(base && lora);
        const auto expected = 4 * allocation(1024, 128, 1, page) + 4 * allocation(1024, 1, 2, page) +
            2 * allocation(128, 1024, 1, page) + 2 * allocation(128, 1, 2, page) +
            2 * allocation(128, 33, 1, page) + 2 * allocation(1, 33, 2, page) +
            allocation(128, 33, 2, page) + allocation(128, 1, 2, page) + allocation(1, 33, 2, page);
        assert(base->surface_bytes == expected);
        assert(lora->surface_bytes == expected + 4 * allocation(1024, 33, 2, page));
        assert(base->estimated_bytes == base->surface_bytes + base->internal_allowance_bytes);
        assert(base->internal_allowance_bytes == (132u << 20));
        for (uint64_t rows : {1056u, 2112u, 4224u}) for (uint64_t hidden : {3840u, 4096u}) {
            const auto small = plan_gpu_calibration_memory(rows, hidden, 4096, false, {}, page);
            const auto large = plan_gpu_calibration_memory(rows, hidden, 8192, false, {}, page);
            assert(small && large && small->surface_bytes < large->surface_bytes);
            assert(large->estimated_bytes < (uint64_t(2) << 30));
        }
    }
    for (uint64_t page : {0u, 3u, 16383u}) assert(!plan_gpu_calibration_memory(33, 128, 1024, false, {}, page));
    assert(!plan_gpu_calibration_memory(4225, 128, 1024, false, {}, 16384));
    assert(!plan_gpu_calibration_memory(33, 127, 1024, false, {}, 16384));
    assert(!plan_gpu_calibration_memory(33, 128, 1025, false, {}, 16384));
    for (const auto invalid : {CalibrationSurfaceShape{0, 33, 2}, CalibrationSurfaceShape{128, 0, 2},
            CalibrationSurfaceShape{128, 33, 4}, CalibrationSurfaceShape{32769, 33, 2}}) {
        const std::array snapshots{invalid};
        assert(!plan_gpu_calibration_memory(33, 128, 1024, false, snapshots, 16384));
    }
    assert(!plan_gpu_calibration_memory(33, 128, 1024, false, {}, uint64_t(1) << 63));
    for (uint64_t page : {4096u, 16384u}) for (uint64_t rows : {1056u, 4128u}) {
        const uint64_t bucket = rows == 1056 ? 1056 : 4224;
        const auto batch = plan_native_channel_calibration_memory(rows, bucket, 3840, 10240, 4096, true, false, page);
        const auto streamed = plan_native_channel_calibration_memory(rows, bucket, 3840, 10240, 4096, true, true, page);
        assert(batch && streamed && batch->surface_bytes == streamed->surface_bytes &&
            batch->internal_allowance_bytes == streamed->internal_allowance_bytes && batch->input_bytes == streamed->input_bytes);
        const uint64_t retired = 6*bucket*(3840+4*4096+10240)+6*rows*3840;
        assert(batch->estimated_bytes-streamed->estimated_bytes == retired);
        assert(streamed->estimated_bytes == streamed->surface_bytes+streamed->internal_allowance_bytes+
            streamed->input_bytes+streamed->gpu_restore_bytes+streamed->gpu_scratch_upper_bytes);
        for(bool stream:{false,true}) {
            const auto original=plan_native_channel_calibration_memory(rows,bucket,3840,10240,4096,true,stream,page);
            const auto fp32=plan_native_channel_calibration_memory(rows,bucket,3840,10240,4096,true,stream,page,true);
            const uint64_t live=stream?1:4;
            assert(fp32 && original && fp32->surface_bytes==original->surface_bytes && fp32->input_bytes==original->input_bytes);
            assert(fp32->gpu_restore_bytes-original->gpu_restore_bytes==bucket*live*3840*2);
            assert(fp32->gpu_scratch_upper_bytes-original->gpu_scratch_upper_bytes==rows*live*3840*2);
            assert(fp32->estimated_bytes-original->estimated_bytes==(bucket+rows)*live*3840*2);
        }
    }
    const auto defaults = plan_channel_sampling(10240, [](int) { return true; });
    assert(defaults && defaults->channels == (std::array<int, 2>{4096, 8192}) && !defaults->memory_limited);
    for (const auto geometry : {std::array<uint64_t, 3>{3840,10240,4128}, {4096,12288,4096}}) {
        const auto [hidden, width, rows] = geometry;
        const auto sampling = plan_channel_sampling(int(width), [&](int candidate) {
            const auto plan = plan_native_channel_calibration_memory(rows,4224,hidden,width,candidate,true,true,16384);
            return plan && plan->estimated_bytes <= (2ull<<30);
        });
        assert(sampling && sampling->memory_limited && sampling->channels[0] < sampling->channels[1]);
        for (int candidate : sampling->channels) {
            assert(candidate%512 == 0 && candidate < int(width));
            const auto plan = plan_native_channel_calibration_memory(rows,4224,hidden,width,candidate,true,true,16384);
            assert(plan && plan->estimated_bytes <= (2ull<<30));
        }
    }
    assert(!plan_channel_sampling(10240, [](int c) { return c==512; }));
    const auto nonmonotonic = plan_channel_sampling(10240, [](int c) { return c==2560 || c==4096; });
    assert(nonmonotonic && nonmonotonic->channels == (std::array<int,2>{2560,4096}));
    for (int width : {0,512,1024,1025,32768}) assert(!plan_channel_sampling(width, [](int) { return true; }));
    assert(!plan_channel_sampling(10240, {}));
    assert(!plan_native_channel_calibration_memory(4225,4224,3840,10240,4096,true,true,16384));
    assert(!plan_native_channel_calibration_memory(1056,1056,3840,10240,4096,false,true,16384));
    std::cout << "PASS native calibration retention: same independent ANE arena, released GPU planes, complete payload accounting, "
                 "original/admission-limited aligned points without extrapolation\n";
    std::cout << "PASS calibration memory: exact page/pitch two W/A/correction banks, frozen snapshots, long model geometries and overflow rejection\n";
}
