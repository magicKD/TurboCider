// CPU-only source contract: preserve legacy arithmetic while moving POD
// parameters after the native command graph's retained buffer bindings.
#include "../../native/backends/metal_graph/kernels/native_ane_stages.hpp"
#include "../../native/backends/ane_transfer_kernels.hpp"
#include "../../native/backends/ane_w8_kernels.hpp"

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kernels = tc::metal_graph::native_ane_stage_kernels;
namespace {
size_t checks = 0;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}

struct Entry {
    std::string_view signature;
    std::string_view body;
};

// Independent extraction uses each named entry and balanced body braces; no
// source replacement or binding remapping is performed by this fixture.
Entry entry(std::string_view source, std::string_view name) {
    const std::string marker = "kernel void " + std::string(name) + "(";
    const size_t start = source.find(marker);
    check(start != std::string_view::npos, "kernel entry missing from source");
    check(source.find(marker, start + marker.size()) == std::string_view::npos,
          "duplicate kernel entry in source");
    const size_t open = source.find('{', start + marker.size());
    check(open != std::string_view::npos, "kernel entry has no body");
    size_t depth = 1;
    for (size_t cursor = open + 1; cursor < source.size(); ++cursor) {
        if (source[cursor] == '{') ++depth;
        if (source[cursor] == '}' && --depth == 0)
            return {source.substr(start, open - start), source.substr(open, cursor - open + 1)};
    }
    throw std::runtime_error("unbalanced kernel body braces");
}

void bindings(std::string_view signature, std::initializer_list<std::string_view> expected) {
    const std::vector<std::string_view> names(expected);
    const std::string text(signature);
    const std::regex attribute(R"(([A-Za-z_][A-Za-z_0-9]*)\s*\[\[buffer\(([0-9]+)\)\]\])");
    std::vector<bool> seen(names.size(), false);
    size_t count = 0;
    bool parameter_last = false;
    for (std::sregex_iterator it(text.begin(), text.end(), attribute), end; it != end; ++it) {
        const size_t index = std::stoul((*it)[2].str());
        check(index < names.size(), "native buffer index exceeds fixed binding oracle");
        check(!seen[index] && (*it)[1].str() == names[index],
              "native buffer binding name/index mismatch or duplicate index");
        if ((*it)[1].str() == "p") parameter_last = index + 1 == names.size();
        seen[index] = true;
        ++count;
    }
    check(count == names.size() && std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
          "native signature does not bind every index exactly once");
    check(parameter_last, "native POD parameter must occupy the final buffer index");
}

void source_dependencies(const std::string& source) {
    std::istringstream lines(source);
    const std::regex include(R"(^\s*#\s*include\s+(.+)$)");
    std::string line;
    size_t includes = 0;
    while (std::getline(lines, line)) {
        std::smatch match;
        if (!std::regex_match(line, match, include)) continue;
        check(match[1].str() == "<metal_stdlib>", "native source has a quoted or external include");
        ++includes;
    }
    check(includes == 1, "native source must retain its single Metal standard-library include");
}

void register_variant(const std::string& generic, uint32_t block) {
    const std::string source = kernels::register_source(block);
    const std::string_view legacy = tc::ane::gpu::w8_source;
    const auto scales = entry(source, "tc_ane_sylvester_register_scales");
    const auto codes = entry(source, "tc_ane_sylvester_register_codes");
    check(scales.body == entry(legacy, "tc_ane_sylvester_register_scales").body,
          "native register scale body differs from legacy arithmetic");
    check(codes.body == entry(legacy, "tc_ane_sylvester_register_codes").body,
          "native register code body differs from legacy arithmetic");
    bindings(scales.signature, {"src", "scales", "offsets", "dst", "status", "signs", "p"});
    bindings(codes.signature, {"src", "scales", "offsets", "row_scales", "dst", "status", "signs", "p"});
    check(scales.signature.find("uint lane [[thread_index_in_simdgroup]]") != std::string_view::npos &&
          codes.signature.find("uint lane [[thread_index_in_simdgroup]]") != std::string_view::npos,
          "register entry changed its SIMD lane contract");

    // Every other entry remains byte-for-byte identical to the generic bundle,
    // including the generic W8/A8 kernels and restore bindings/arithmetic.
    for (const char* name : {"tc_ane_upload", "tc_ane_restore", "tc_ane_w8_scales", "tc_ane_w8_codes",
                             "tc_ane_w8_group_scales", "tc_ane_comfy_register_scales", "tc_ane_comfy_register_codes",
                             "tc_ane_w8_scale_copy", "tc_ane_convrot_scales", "tc_ane_convrot_codes"}) {
        const auto actual = entry(source, name), expected = entry(generic, name);
        check(actual.signature == expected.signature && actual.body == expected.body,
              "register variant changed an unrelated kernel signature/body");
    }
    const char* block_declaration = block == 128 ? "constant uint tc_w8_block = 128;" :
                                                 "constant uint tc_w8_block = 512;";
    for (const char* declaration : {"constant bool tc_w8_specialize = true;",
                                     "constant uint tc_w8_encoding = 0;",
                                     "constant uint tc_w8_dtype = 1;", block_declaration}) {
        const size_t first = source.find(declaration);
        check(first != std::string::npos && source.find(declaration, first + 1) == std::string::npos,
              "register constant is missing, incorrect, or duplicated");
    }
    check(source.find("constant bool tc_w8_specialize = false;") == std::string::npos &&
          source.find(block == 128 ? "constant uint tc_w8_block = 512;" :
                                     "constant uint tc_w8_block = 128;") == std::string::npos &&
          source.find("function_constant") == std::string::npos,
          "register variant retains a conflicting or unbound constant");
    source_dependencies(source);
}
} // namespace

int main(int argc, char** argv) {
    try {
        // Emit exact source bytes for the before/after SHA256 contract check.
        if (argc == 2 && std::string_view(argv[1]) == "--print-generic-source") {
            std::cout << kernels::source();
            return 0;
        }
        check(argc == 1, "unknown source fixture argument");
        const std::string source = kernels::source();
        const std::string_view transfer = tc::ane::gpu::transfer_source;
        const std::string_view w8 = tc::ane::gpu::w8_source;
        const auto scales = entry(source, "tc_ane_w8_scales");
        const auto codes = entry(source, "tc_ane_w8_codes");
        const auto restore = entry(source, "tc_ane_restore");
        check(scales.body == entry(w8, "tc_ane_w8_scales").body,
              "native W8 scale body differs from legacy arithmetic");
        check(codes.body == entry(w8, "tc_ane_w8_codes").body,
              "native W8 code body differs from legacy arithmetic");
        check(restore.body == entry(transfer, "tc_ane_restore").body,
              "native restore body differs from legacy arithmetic");

        // Index-ordered independent oracle, including unchanged low bindings.
        bindings(scales.signature, {"src", "scales", "offsets", "dst", "status", "signs", "column_scale", "p"});
        bindings(codes.signature, {"src", "scales", "offsets", "row_scales", "dst", "status", "signs", "column_scale", "p"});
        bindings(restore.signature, {"src", "dst", "status", "failed", "row_scales", "token_scales", "second_scales", "p"});
        // Compact reuse binds the unchanged copy entry directly. Its uint3 is
        // a 16-byte ABI payload; the fourth host word is padding, not a stride.
        const auto copy = entry(source, "tc_ane_w8_scale_copy");
        bindings(copy.signature, {"src", "dst", "p"});
        check(copy.signature.find("constant uint3 &p") != std::string_view::npos &&
              copy.signature.find("uint row [[thread_position_in_grid]]") != std::string_view::npos,
              "compact scale copy parameter/grid ABI changed");
        check(copy.body == entry(w8, "tc_ane_w8_scale_copy").body &&
              copy.body.find("if(row<p.x)dst[ulong(row)*p.z]=src[ulong(row)*p.y];") != std::string_view::npos,
              "compact scale copy must preserve raw FP16 bits, bounds and element strides");

        const auto native_upload = entry(source, "tc_ane_upload"), legacy_upload = entry(transfer, "tc_ane_upload");
        check(native_upload.signature == legacy_upload.signature && native_upload.body == legacy_upload.body,
              "legacy upload signature/body changed");
        for (const char* name : {"tc_ane_sylvester_register_scales", "tc_ane_sylvester_register_codes",
                                 "tc_ane_w8_group_scales", "tc_ane_comfy_register_scales", "tc_ane_comfy_register_codes",
                                 "tc_ane_w8_scale_copy", "tc_ane_convrot_scales", "tc_ane_convrot_codes"}) {
            const auto native = entry(source, name), legacy = entry(w8, name);
            check(native.signature == legacy.signature && native.body == legacy.body,
                  "unused legacy kernel signature/body changed");
        }
        check(source.find("function_constant") == std::string::npos,
              "native generic source retains unbound function constants");
        for (const char* declaration : {"constant bool tc_w8_specialize = false;",
                                         "constant uint tc_w8_encoding = 0;",
                                         "constant uint tc_w8_dtype = 1;",
                                         "constant uint tc_w8_block = 128;"}) {
            const size_t first = source.find(declaration);
            check(first != std::string::npos && source.find(declaration, first + 1) == std::string::npos,
                  "native generic constant is missing, incorrect, or duplicated");
        }
        source_dependencies(source);

        register_variant(source, 128);
        register_variant(source, 512);
        for (const uint32_t block : {0u, 1u, 64u, 127u, 129u, 256u, 513u, 1024u, UINT32_MAX})
            rejects([&] { (void)kernels::register_source(block); }, "register source accepted an unsupported block");
        check(kernels::source() == source, "register assembly mutated the default generic source");

        std::string missing = "unchanged";
        rejects([&] { kernels::detail::replace_once(missing, "absent", "replacement"); },
                "source assembler accepted missing expected text");
        std::string duplicate = "expected expected";
        rejects([&] { kernels::detail::replace_once(duplicate, "expected", "replacement"); },
                "source assembler accepted duplicate expected text");
        std::cout << "native_ane_stage_source_cpu=PASS checks=" << checks
                  << " device_created=0 model_loads=0\n";
    } catch (const std::exception& error) {
        std::cerr << "native_ane_stage_source_cpu=FAIL " << error.what() << '\n';
        return 1;
    }
}
