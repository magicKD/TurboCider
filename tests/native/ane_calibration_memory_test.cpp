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
    std::cout << "PASS calibration memory: exact page/pitch two W/A/correction banks, frozen snapshots, long model geometries and overflow rejection\n";
}
