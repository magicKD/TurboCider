#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace tc {
// Read only the GGUF directory before asking MLX to allocate model tensors.
// Type IDs are the on-disk GGML IDs, not filename-derived quantization labels.
inline void validate_native_gguf(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open GGUF: " + path.string());
    const auto size = std::filesystem::file_size(path);
    uint64_t cursor = 0;
    auto fail = [] { throw std::runtime_error("invalid or truncated GGUF directory"); };
    auto read = [&](void *target, uint64_t bytes) {
        if (bytes > size - cursor) fail();
        input.read(static_cast<char *>(target), bytes);
        if (!input) fail();
        cursor += bytes;
    };
    auto u32 = [&] { uint32_t value; read(&value, 4); return value; };
    auto u64 = [&] { uint64_t value; read(&value, 8); return value; };
    auto skip = [&](uint64_t bytes) {
        if (bytes > size - cursor) fail();
        input.seekg(bytes, std::ios::cur);
        if (!input) fail();
        cursor += bytes;
    };
    auto string = [&] { skip(u64()); };
    auto scalar_bytes = [&](uint32_t type) -> uint64_t {
        switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: fail(); return 0;
        }
    };
    if (u32() != 0x46554747) fail();
    const auto version = u32();
    if (version != 2 && version != 3) fail();
    const auto tensors = u64(), metadata = u64();
    if (!tensors || tensors > size / 24 || metadata > size / 12) fail();
    for (uint64_t i = 0; i < metadata; ++i) {
        string();
        auto type = u32();
        if (type == 8) { string(); continue; }
        if (type != 9) { skip(scalar_bytes(type)); continue; }
        type = u32();
        auto count = u64();
        if (type == 8) {
            if (count > (size - cursor) / 8) fail();
            for (uint64_t j = 0; j < count; ++j) string();
        } else {
            auto bytes = scalar_bytes(type);
            if (count > (size - cursor) / bytes) fail();
            skip(count * bytes);
        }
    }
    for (uint64_t i = 0; i < tensors; ++i) {
        string();
        auto rank = u32();
        if (!rank || rank > 4) fail();
        for (uint32_t d = 0; d < rank; ++d) if (!u64()) fail();
        const auto type = u32();
        if (type != 0 && type != 1 && type != 2 && type != 3 && type != 8 && type != 30)
            throw std::runtime_error("unsupported native MLX GGUF tensor type " +
                std::to_string(type) + "; supported: F32, F16, BF16, Q4_0, Q4_1, Q8_0; no fallback backend");
        if (u64() > size) fail();
    }
}
} // namespace tc
