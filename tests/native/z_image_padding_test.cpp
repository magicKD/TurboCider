#include "../../native/models/z_image/padding.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace tc;
    mx::set_default_device(mx::Device(mx::Device::cpu));
    for (const auto dtype : {mx::float16, mx::bfloat16, mx::float32}) {
        auto flat = mx::astype(mx::arange(64), dtype);
        auto row = mx::reshape(flat, {1, 64});
        for (const auto &input : {flat, row}) {
            auto normalized = z_image::padding_token_row(input, 64);
            assert(normalized.shape() == mx::Shape({1, 64}) && normalized.dtype() == dtype);
            assert(mx::all(normalized == row).item<bool>());
            for (int count : {1, 5, 31}) {
                auto padded = mx::concatenate({row, mx::repeat(normalized, count, 0)}, 0);
                assert(padded.shape() == mx::Shape({count + 1, 64}));
                assert(mx::all(padded == mx::repeat(row, count + 1, 0)).item<bool>());
            }
        }
        assert(z_image::padding_token_row(row, 64).id() == row.id());
    }
    for (const auto &input : {mx::zeros({64}, mx::uint32), mx::zeros({1, 64}, mx::int32),
                              mx::zeros({2, 64}), mx::zeros({64, 1}),
                              mx::zeros({1, 1, 64}), mx::zeros({63})}) {
        bool rejected = false;
        try { z_image::padding_token_row(input, 64); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    std::cout << "PASS Z-Image GGUF padding singleton geometry, dtype/value preservation and packed rejection\n";
}
