#include "../../native/backends/metal_graph/native_ane_stage_contract.hpp"

#include <iostream>

namespace contract = tc::metal_graph::native_ane_stage_contract;
using tc::metal_graph::ScalarType;
using contract::Layout;

namespace {
size_t checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class Exception = std::invalid_argument, class F>
void rejects(F&& operation, const char* message) {
    bool rejected = false;
    try { operation(); } catch (const Exception&) { rejected = true; }
    check(rejected, message);
}
void close(float value, double oracle, const char* message) {
    check(std::isfinite(value) && std::abs(double(value) - oracle) < 2e-8, message);
}

// Metadata only: storage numbers denote backing identities, not fake buffers.
// The sizes and offsets below are fixed layout oracles; nothing is allocated.
struct Fixture {
    contract::Geometry geometry;
    std::array<Layout,3> weights;
    std::array<Layout,6> bank;
    Layout input, codes, token_scales, y, output;
};
Fixture main_fixture() {
    Fixture f;
    f.geometry = {1024,4096,12288,6144};
    f.weights = {
        Layout{12288,4096,8192,0,100663296,ScalarType::BF16,1},
        Layout{12288,4096,8192,100663296,100663296,ScalarType::BF16,1},
        Layout{4096,12288,24576,0,100663296,ScalarType::BF16,2}};
    f.bank = {
        Layout{6144,4096,4096,0,25165824,ScalarType::Int8,3},
        Layout{6144,1,64,0,393216,ScalarType::FP16,4},
        Layout{6144,4096,4096,0,25165824,ScalarType::Int8,5},
        Layout{6144,1,64,0,393216,ScalarType::FP16,6},
        Layout{4096,6144,6144,0,25165824,ScalarType::Int8,7},
        Layout{4096,1,64,0,262144,ScalarType::FP16,8}};
    f.input = {1024,4096,8192,0,8388608,ScalarType::BF16,9};
    f.codes = {4096,1024,1024,0,4194304,ScalarType::Int8,10};
    f.token_scales = {1,1024,2048,0,2048,ScalarType::FP16,11};
    f.y = {4097,1024,2048,0,8390656,ScalarType::FP16,12};
    f.output = {1024,4096,16384,0,16777216,ScalarType::FP32,13};
    return f;
}
Fixture tiny_fixture() {
    Fixture f;
    f.geometry = {32,128,1024,512};
    f.weights = {
        Layout{1024,128,256,0,262144,ScalarType::BF16,1},
        Layout{1024,128,256,262144,262144,ScalarType::BF16,1},
        Layout{128,1024,2048,0,262144,ScalarType::BF16,2}};
    f.bank = {
        Layout{512,128,128,0,65536,ScalarType::Int8,3},
        Layout{512,1,64,0,32768,ScalarType::FP16,4},
        Layout{512,128,128,0,65536,ScalarType::Int8,5},
        Layout{512,1,64,0,32768,ScalarType::FP16,6},
        Layout{128,512,512,0,65536,ScalarType::Int8,7},
        Layout{128,1,64,0,8192,ScalarType::FP16,8}};
    f.input = {32,128,256,0,8192,ScalarType::BF16,9};
    f.codes = {128,32,64,0,8192,ScalarType::Int8,10};
    f.token_scales = {1,32,64,0,64,ScalarType::FP16,11};
    f.y = {129,32,64,0,8256,ScalarType::FP16,12};
    f.output = {32,128,512,0,16384,ScalarType::FP32,13};
    return f;
}

void test_parameters(const Fixture& f, bool main) {
    contract::validate_geometry(f.geometry);
    const auto weight = contract::weight_parameters(f.geometry, f.weights, f.bank);
    const uint32_t hidden = main ? 4096 : 128;
    const uint32_t channels = main ? 6144 : 512;
    for (size_t i = 0; i < 2; ++i) {
        const auto& p = weight[i];
        check(p.source_pitch == (main ? 8192U : 256U) && p.physical_cols == hidden &&
              p.row_begin == channels && p.rows == channels && p.column_begin == 0 &&
              p.columns == hidden && p.block == 128 && p.code_pitch == hidden &&
              p.scale_pitch == 64 && p.transpose == 0, "gate/up physical selection changed");
        close(p.norm, 0.08838834764831844, "H128 normalization changed");
    }
    const auto& down = weight[2];
    check(down.source_pitch == (main ? 24576U : 2048U) &&
          down.physical_cols == (main ? 12288U : 1024U) &&
          down.row_begin == 0 && down.rows == hidden && down.column_begin == channels &&
          down.columns == channels && down.block == 512 && down.code_pitch == channels &&
          down.scale_pitch == 64 && down.transpose == 0, "down source must retain its full pitch");
    close(down.norm, 0.04419417382415922, "H512 normalization changed");
    for (const auto& p : weight)
        check(p.encoding == 0 && p.dtype == 1 && p.group_size == 32 && p.source_aligned == 1 &&
              p.seed_low == 20260930 && p.seed_high == 0 && p.basis == 0 &&
              p.meta_pitch == 0 && p.meta_dtype == 0 && p.offset_pitch == 0 && p.offset_dtype == 0 &&
              p.has_offset == 0 && p.activation_group == 0 && p.has_column_scale == 0 &&
              p.inverse_column_scale == 0, "unexpected generic/S1/metadata fields");
    const auto gate_start = f.weights[0].offset_bytes + size_t(weight[0].row_begin) * weight[0].source_pitch;
    const auto up_start = f.weights[1].offset_bytes + size_t(weight[1].row_begin) * weight[1].source_pitch;
    check(f.weights[1].offset_bytes == (main ? 100663296U : 262144U) &&
          f.weights[0].storage == f.weights[1].storage &&
          gate_start == (main ? 50331648U : 131072U) &&
          up_start == (main ? 150994944U : 393216U), "fused gate/up offset added twice or lost");
    check(f.weights[2].offset_bytes + size_t(down.column_begin) * 2 == (main ? 12288U : 1024U),
          "down selected channel byte offset changed");

    const auto a8 = contract::activation_parameters(f.geometry, f.input, f.codes, f.token_scales);
    check(a8.rows == (main ? 1024U : 32U) && a8.columns == hidden &&
          a8.row_begin == 0 && a8.column_begin == 0 && a8.block == 128 &&
          a8.source_pitch == (main ? 8192U : 256U) && a8.physical_cols == hidden &&
          a8.code_pitch == (main ? 1024U : 64U) && a8.scale_pitch == 2 && a8.transpose == 1,
          "A8 token-major selection or logical scale pitch changed");
    check(f.token_scales.row_stride_bytes == (main ? 2048U : 64U) &&
          f.token_scales.view_size_bytes == (main ? 2048U : 64U), "tx physical surface pitch changed");
    close(a8.norm, 0.08838834764831844, "A8 normalization changed");

    const auto restore = contract::restore_parameters(f.geometry, f.y, f.bank[5], f.output);
    check(restore.rows == (main ? 1024U : 32U) && restore.cols == hidden &&
          restore.source_pitch == (main ? 2048U : 64U) &&
          restore.target_pitch == (main ? 16384U : 512U) &&
          restore.dtype == 2 && restore.validate_only == 0 && restore.scale == 1.f &&
          restore.row_scale_pitch == 64 && restore.scaled == 1 &&
          restore.second_scaled == 0 && restore.signed_row_scale == 0,
          "restore FP32/headroom/scale fields changed");
    check(contract::extent(f.y) == (main ? 8390656U : 8256U) &&
          contract::extent(f.output) == (main ? 16777216U : 16384U), "restore layout extent changed");
    contract::cross_stage_aliases(f.weights, f.bank, {f.input, f.codes, f.token_scales});
}

void test_layout_rejections() {
    const auto f = tiny_fixture();
    auto scale = f.bank[5];
    scale.view_size_bytes = 8130;
    check(contract::extent(scale) == 8130, "logical sd extent oracle changed");
    rejects([&] { contract::surface(scale,128,1,ScalarType::FP16); },
            "logical sd extent accepted instead of full 8192-byte surface");
    auto bad = f.codes;
    bad.offset_bytes = 64;
    rejects([&] { contract::surface(bad,128,32,ScalarType::Int8); }, "surface subview offset accepted");
    bad = f.codes; bad.row_stride_bytes = 128; bad.view_size_bytes = 16384;
    rejects([&] { contract::surface(bad,128,32,ScalarType::Int8); }, "alternate surface pitch accepted");
    bad = f.codes; --bad.view_size_bytes;
    rejects([&] { contract::surface(bad,128,32,ScalarType::Int8); }, "short full code surface accepted");
    bad = f.input; bad.storage = 0;
    rejects([&] { (void)contract::extent(bad); }, "missing backing accepted");
    bad = f.input; bad.rows = 0;
    rejects([&] { (void)contract::extent(bad); }, "empty matrix accepted");
    bad = f.input; bad.type = static_cast<ScalarType>(255);
    rejects([&] { (void)contract::extent(bad); }, "unknown scalar type accepted");
    bad = f.input; bad.row_stride_bytes = 255;
    rejects([&] { (void)contract::extent(bad); }, "unaligned/short BF16 pitch accepted");
    bad = f.input; bad.offset_bytes = 1;
    rejects([&] { (void)contract::extent(bad); }, "unaligned BF16 offset accepted");
    bad = f.input; --bad.view_size_bytes;
    rejects([&] { (void)contract::extent(bad); }, "one-byte-short matrix accepted");
    bad = {1,1,2,SIZE_MAX-1,2,ScalarType::BF16,1};
    rejects([&] { (void)contract::extent(bad); }, "offset plus view overflow accepted");
    bad = f.input; bad.row_stride_bytes = size_t(UINT32_MAX) + 1;
    rejects([&] { (void)contract::extent(bad); }, "source pitch narrowed to uint32");

    auto weights = f.weights; weights[2].columns = 512;
    rejects([&] { (void)contract::weight_parameters(f.geometry,weights,f.bank); },
            "down logical width compacted to the selected channel count");
    weights = f.weights; weights[0].type = ScalarType::FP16;
    rejects([&] { (void)contract::weight_parameters(f.geometry,weights,f.bank); }, "FP16 master source accepted");
    auto bank = f.bank; bank[0].columns = 127;
    rejects([&] { (void)contract::weight_parameters(f.geometry,f.weights,bank); }, "wrong bank geometry accepted");
    auto y = f.y; y.rows = 128;
    rejects([&] { (void)contract::restore_parameters(f.geometry,y,f.bank[5],f.output); },
            "restore surface omitted its headroom row");
    auto output = f.output; output.type = ScalarType::BF16;
    rejects([&] { (void)contract::restore_parameters(f.geometry,f.y,f.bank[5],output); },
            "restore accepted a prematurely narrowed partial");
}

void test_aliases() {
    const auto f = tiny_fixture();
    auto bank = f.bank; bank[2].storage = bank[0].storage;
    rejects([&] { (void)contract::weight_parameters(f.geometry,f.weights,bank); }, "aliased bank outputs accepted");
    bank = f.bank; bank[0].storage = f.weights[0].storage;
    rejects([&] { (void)contract::weight_parameters(f.geometry,f.weights,bank); }, "bank overwrote checkpoint backing");
    auto codes = f.codes; codes.storage = f.input.storage;
    rejects([&] { (void)contract::activation_parameters(f.geometry,f.input,codes,f.token_scales); },
            "A8 codes overwrote input backing");
    auto output = f.output; output.storage = f.y.storage;
    rejects([&] { (void)contract::restore_parameters(f.geometry,f.y,f.bank[5],output); },
            "restore output aliases ANE y");
    auto scale = f.bank[5]; scale.storage = f.y.storage;
    rejects([&] { (void)contract::restore_parameters(f.geometry,f.y,scale,f.output); }, "restore y aliases sd");

    std::array<Layout,3> activation{f.input,f.codes,f.token_scales};
    activation[1].storage = f.weights[0].storage;
    rejects([&] { contract::cross_stage_aliases(f.weights,f.bank,activation); }, "A8 x aliases checkpoint cross-stage");
    activation = {f.input,f.codes,f.token_scales}; activation[0].storage = f.bank[0].storage;
    rejects([&] { contract::cross_stage_aliases(f.weights,f.bank,activation); }, "A8 input aliases bank cross-stage");
    activation = {f.input,f.codes,f.token_scales}; activation[2].storage = f.bank[5].storage;
    rejects([&] { contract::cross_stage_aliases(f.weights,f.bank,activation); }, "tx aliases bank cross-stage");
}

void test_geometry_and_status() {
    for(uint32_t rows:{128U,1024U,1152U,1664U,2688U,4096U})for(uint32_t channels:{3072U,6144U,8192U}) {
        contract::validate_geometry({rows,4096,12288,channels});++checks;
    }
    for (auto g : {contract::Geometry{1053,4096,12288,6144}, contract::Geometry{1024,4096,12288,4096},
                   contract::Geometry{32,128,1024,0}, contract::Geometry{32,128,1024,1025},
                   contract::Geometry{32,256,1024,512}, contract::Geometry{0,0,0,0},contract::Geometry{4224,4096,12288,6144},contract::Geometry{64,4096,12288,6144}})
        rejects([&] { contract::validate_geometry(g); }, "unqualified stage geometry accepted");
    check(contract::weight_mask == 7 && contract::activation_mask == 8 &&
          contract::producer_mask == 15 && contract::restore_mask == 16, "writer mask ABI changed");
    contract::validate_status_mask(contract::StatusRole::Producer,15,0);
    contract::validate_status_mask(contract::StatusRole::Restore,16,0);
    for (uint32_t encoded : {0U,7U,8U,16U,31U})
        rejects([&] { contract::validate_status_mask(contract::StatusRole::Producer,encoded,0); },
                "incomplete/foreign producer writers accepted");
    for (uint32_t encoded : {0U,15U,17U})
        rejects([&] { contract::validate_status_mask(contract::StatusRole::Restore,encoded,0); },
                "incomplete/foreign restore writers accepted");
    for (uint32_t flags : {1U,8U,16U,UINT32_MAX}) {
        rejects<std::runtime_error>([&] { contract::validate_status_mask(contract::StatusRole::Producer,15,flags); },
                                   "nonzero producer validation flags accepted");
        rejects<std::runtime_error>([&] { contract::validate_status_mask(contract::StatusRole::Restore,16,flags); },
                                   "nonzero restore validation flags accepted");
    }
    rejects([&] { contract::validate_status_mask(static_cast<contract::StatusRole>(99),16,0); },
            "unknown status role accepted as Restore");
}
} // namespace

int main() {
    try {
        test_parameters(main_fixture(),true);
        test_parameters(tiny_fixture(),false);
        test_layout_rejections();
        test_aliases();
        test_geometry_and_status();
        std::cout << "native_ane_stage_contract_cpu=PASS checks=" << checks
                  << " device_created=0 model_loads=0\n";
    } catch (const std::exception& error) {
        std::cerr << "native_ane_stage_contract_cpu=FAIL " << error.what() << '\n';
        return 1;
    }
}
