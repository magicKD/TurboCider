// Source-format staging and transactional publication on public Core ML.
// This is not a placement/INT8/performance or full-model qualification.
#include "../../native/backends/ane_runtime_packed.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace tc::ane;
namespace {
void check(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
int sign(int i, int j) {
    constexpr int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1},
                            {1, -1, 1, 1}, {-1, 1, 1, 1}};
    int value = 1;
    for (int d = 0; d < 4; ++d, i /= 4, j /= 4) value *= h4[i % 4][j % 4];
    return value;
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 3 && (std::string(argv[2]) == "cpu" || std::string(argv[2]) == "ne"),
              "usage: ane-runtime-packed-probe MANIFEST cpu|ne");
        RuntimeGraph graph(argv[1], 2ull << 30, std::string(argv[2]) == "cpu");
        const auto &s = graph.shape();
        check(s.kind == Kind::Matmul && s.hidden % 256 == 0, "probe requires H256-aligned MatMul");
        std::string error;
        check(graph.self_test(error), error.c_str());
        std::vector<float> input(size_t(s.rows) * s.hidden);
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 13) - 6) / 16.f;
        MatrixView x{input.data(), input.size() * 4, s.rows, s.hidden, 0, DType::FP32};
        auto evaluate = [&](WeightView weight) {
            graph.stage_weights(std::vector<WeightView>{weight});
            auto stage = graph.wait_stage(); check(stage.ok, stage.error.c_str());
            std::vector<uint16_t> output(size_t(s.rows) * s.width);
            graph.launch(x, output.data(), output.size(), DType::FP16);
            auto result = graph.finish(); check(result.ok, result.error.c_str());
            return output;
        };
        // Actual source variants, byte-identical dense-slot controls, all
        // registered GGUF types, padded source rows and unaligned bases.
        for (const auto &type : tc::gguf::types) {
            const size_t row_bytes = size_t(s.hidden) / type.elements * type.bytes;
            const size_t pitch = row_bytes + 3;
            std::vector<std::byte> packed(1 + size_t(s.width) * pitch);
            for (int r = 0; r < s.width; ++r) for (int c = 0; c < s.hidden; c += type.elements) {
                auto *p = packed.data() + 1 + size_t(r) * pitch + size_t(c) / type.elements * type.bytes;
                for (unsigned i = 0; i < type.bytes; ++i) p[i] = std::byte((i * 17 + c + r) % 256);
                if (type.id == 0) { float f = float(r % 5 - 2) / 32.f; std::memcpy(p, &f, 4); }
                else if (type.id == 30) { auto bits = round_bf16(float(r % 5 - 2) / 32.f); std::memcpy(p, &bits, 2); }
                else {
                    auto bits = tc::gguf::float_to_fp16_rne(.00390625f);
                    std::memcpy(p + (type.id == 14 ? 208 : 0), &bits, 2);
                    if (type.id == 3 || type.id == 7 || type.id == 12 || type.id == 13) {
                        bits = tc::gguf::float_to_fp16_rne(-.001953125f); std::memcpy(p + 2, &bits, 2);
                    }
                }
            }
            GgufView source{packed.data() + 1, packed.size() - 1, s.width, s.hidden, pitch, type.id};
            std::vector<uint16_t> dense(size_t(s.width) * s.hidden);
            for (int r = 0; r < s.width; ++r)
                tc::gguf::decode_cpu_into({{packed.data() + 1 + size_t(r) * pitch, row_bytes},
                    type.id, 1, uint64_t(s.hidden)}, {0, 1, 0, uint64_t(s.hidden)},
                    {{reinterpret_cast<std::byte *>(dense.data() + size_t(r) * s.hidden), size_t(s.hidden) * 2},
                     tc::gguf::DecodeDType::f16, size_t(s.hidden) * 2, 2});
            const auto original = packed;
            auto reference = evaluate(MatrixView{dense.data(), dense.size() * 2, s.width, s.hidden, 0, DType::FP16});
            auto actual = evaluate(source);
            check(reference == actual && packed == original, "GGUF direct-slot output differs from FP16 oracle");
        }
        const size_t pitch = size_t(s.hidden) + 3;
        std::vector<int8_t> codes(1 + size_t(s.width) * pitch);
        std::vector<float> scales(s.width);
        std::vector<uint16_t> dense(size_t(s.width) * s.hidden);
        for (int r = 0; r < s.width; ++r) {
            scales[r] = float(r % 7 - 3) / 1024.f;
            for (int c = 0; c < s.hidden; ++c) codes[1 + size_t(r) * pitch + c] = int8_t((c * 37 + r * 19) % 256 - 128);
            for (int c = 0; c < s.hidden; ++c) {
                int64_t dot = 0;
                for (int k = 0; k < 256; ++k)
                    dot += int64_t(codes[1 + size_t(r) * pitch + c / 256 * 256 + k]) * sign(k, c % 256);
                dense[size_t(r) * s.hidden + c] = tc::gguf::float_to_fp16_rne((float(dot) / 16.f) * scales[r]);
            }
        }
        ConvrotView source{codes.data() + 1, codes.size() - 1, s.width, s.hidden, pitch,
            {scales.data(), scales.size() * 4, s.width, 1, 0, DType::FP32}};
        const auto reference = evaluate(MatrixView{dense.data(), dense.size() * 2, s.width, s.hidden, 0, DType::FP16});
        const auto original = codes;
        for (int cycle = 0; cycle < 10; ++cycle) {
            check(evaluate(source) == reference, "ConvRot A differs from independent oracle");
            codes[1] = int8_t(int(codes[1]) + 1);
            check(evaluate(source) != reference, "weight B did not change output");
            codes = original;
            check(evaluate(source) == reference, "weight A/B/A retained stale slot content");
        }
        // Partial/failed staging cannot publish old contents; recovery only
        // succeeds after a fresh complete fill, using the same graph/surfaces.
        const float saved = scales[0]; scales[0] = std::numeric_limits<float>::quiet_NaN();
        graph.stage_weights(std::vector<WeightView>{source});
        check(!graph.wait_stage().ok, "nonfinite source was accepted");
        std::vector<uint16_t> discarded(size_t(s.rows) * s.width, 0xabcd);
        graph.launch(x, discarded.data(), discarded.size(), DType::FP16);
        check(!graph.finish().ok, "failed staging left stale weights launchable");
        check(std::all_of(discarded.begin(), discarded.end(), [](auto v) { return v == 0xabcd; }),
              "failed staging published output");
        scales[0] = saved;
        check(evaluate(source) == reference, "fresh stage did not recover");
        // Same graph/surfaces, explicit legacy packed view. Retain actual F32
        // scales here; BF16 stored-scale rounding has an independent host gate.
        const int groups=s.hidden/32;
        std::vector<uint8_t> packed(size_t(s.width)*s.hidden);
        std::vector<float> group_scales(size_t(s.width)*groups),offsets(group_scales.size());
        for (int row=0;row<s.width;++row) {
            for (int c=0;c<s.hidden;++c) packed[size_t(row)*s.hidden+c]=uint8_t(int(codes[1+size_t(row)*pitch+c])+128);
            for (int g=0;g<groups;++g) {
                group_scales[size_t(row)*groups+g]=scales[row];offsets[size_t(row)*groups+g]=-128.f*scales[row];
            }
        }
        ConvrotAffineView legacy{{packed.data(),packed.size(),s.width,s.hidden,0,32,8,
            {group_scales.data(),group_scales.size()*4,s.width,groups,0,DType::FP32},
            MatrixView{offsets.data(),offsets.size()*4,s.width,groups,0,DType::FP32}}};
        for (int cycle=0;cycle<10;++cycle) {
            check(evaluate(legacy)==reference,"legacy packed inverse-H reference mismatch");
            packed[0]^=1;check(evaluate(legacy)!=reference,"legacy packed changed weight ignored");
            packed[0]^=1;check(evaluate(legacy)==reference,"legacy packed A/B/A stale source");
        }
        group_scales[1]*=2;
        graph.stage_weights(std::vector<WeightView>{legacy});
        check(!graph.wait_stage().ok,"nonuniform legacy ConvRot scale accepted");
        graph.launch(x,discarded.data(),discarded.size(),DType::FP16);
        check(!graph.finish().ok,"bad legacy stage left old content launchable");
        group_scales[1]=group_scales[0];check(evaluate(legacy)==reference,"legacy fresh-stage recovery failed");
        std::cout << "{\"status\":\"pass\",\"gguf_types\":11,\"weight_swap_cycles\":10,"
                     "\"legacy_convrot_swap_cycles\":10,"
                     "\"failed_stage_rejected\":true,\"same_graph\":true,"
                     "\"observed_placement\":\"unknown\",\"hardware_int8\":\"not_requested\"}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl; return 1;
    }
}
