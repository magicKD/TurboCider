// CPU-only model-plan/workspace checks. Link with native_block and its native
// dependencies, but do not construct Backend or prepare kernels in this test.
#include "../../native/models/qwen21/native_block.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace qwen = tc::qwen21::native_graph;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid native Qwen block contract accepted");
}
size_t bytes(qwen::BlockGeometry geometry, uint32_t rows, uint32_t prefix,
             qwen::BlockPhase phase) {
    return qwen::NativeBlock::workspace_bytes(geometry, rows, prefix, phase);
}
} // namespace

int main() {
    using Phase = qwen::BlockPhase;
    const qwen::BlockGeometry production{};
    const qwen::BlockGeometry tiny{128, 1, 128, 256};
    check(production.hidden == 4096 && production.heads == 32 &&
          production.head_dim == 128 && production.ffn_width == 12288,
          "local Qwen base dimensions changed without contract update");
    qwen::validate_geometry(production);
    qwen::validate_geometry(tiny);
    // Different FFN width is an operator capability, not a different model's
    // topology qualification. Keep tiny plans available for bounded checks.
    qwen::validate_geometry({256, 2, 128, 128});

    for (auto invalid : {qwen::BlockGeometry{0,0,128,256},
                         {4096,32,64,12288}, {4096,31,128,12288},
                         {4224,33,128,12288}, {128,1,128,0},
                         {128,1,128,255}, {128,1,128,12416},
                         {UINT32_MAX,UINT32_MAX,128,UINT32_MAX}})
        rejects([&] { qwen::validate_geometry(invalid); });

    // These byte totals document the deliberately unaliased first schedule:
    // 13 activation-sized slots, one FFN hidden, plus joined K/V only when
    // decode consumes a nonempty prefix. Prefix cache/source weights/input/
    // output are external bindings and are not free or included in this arena.
    check(bytes(production,1024,0,Phase::SteadyDecode) == 134217728,
          "1024-row base workspace should be 128 MiB without joined prefix");
    check(bytes(production,1024,128,Phase::SteadyDecode) == 153092096,
          "decode prefix must budget joined K and V including target rows");
    check(bytes(production,1152,128,Phase::PrefixBuild) == 150994944,
          "prefix-build rows already contain prefix; do not add it again");
    check(bytes(production,1024,0,Phase::PrefixBuild) ==
          bytes(production,1024,0,Phase::SteadyDecode),
          "zero-prefix phases should have equal scratch footprints");
    check(bytes(tiny,1,0,Phase::SteadyDecode) == 3840, "tiny single-row workspace");
    check(bytes(tiny,3,2,Phase::SteadyDecode) == 14080, "tiny decode workspace");
    check(bytes(tiny,5,2,Phase::PrefixBuild) == 19200, "tiny prefix-build workspace");
    check(bytes(production,1024,129,Phase::SteadyDecode) >
          bytes(production,1024,128,Phase::SteadyDecode),
          "one more visible key must not decrease workspace accounting");

    for (Phase phase : {Phase::PrefixBuild, Phase::SteadyDecode}) {
        for (uint32_t rows : {0U,65537U,UINT32_MAX})
            rejects([&] { (void)bytes(production,rows,0,phase); });
        for (uint32_t prefix : {65537U,UINT32_MAX})
            rejects([&] { (void)bytes(production,1024,prefix,phase); });
        rejects([&] { (void)bytes(production,UINT32_MAX,UINT32_MAX,phase); });
    }
    rejects([&] { (void)bytes(tiny,5,5,Phase::PrefixBuild); });
    rejects([&] { (void)bytes(tiny,5,6,Phase::PrefixBuild); });
    rejects([&] { (void)bytes(tiny,65536,1,Phase::SteadyDecode); });
    rejects([&] { (void)bytes(tiny,65535,2,Phase::SteadyDecode); });
    rejects([&] { (void)bytes(tiny,3,0,static_cast<Phase>(255)); });

    // Maximum admitted row geometry is arithmetic only: no 8 GiB allocation.
    // Large uint inputs must reject before rows+prefix can wrap to small rows.
    check(bytes(production,65536,65535,Phase::PrefixBuild) == size_t(8589934592ULL),
          "maximum prefill byte count must use size_t arithmetic");
    check(bytes(production,1,65535,Phase::SteadyDecode) == size_t(1073872896ULL),
          "maximum visible decode K/V rows must preserve exact byte count");
    std::cout << "native_qwen_block_cpu_contract=PASS device_created=0 allocation_bytes=0\n";
}
