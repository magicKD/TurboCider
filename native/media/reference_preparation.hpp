#pragma once
#import <Foundation/Foundation.h>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tc {
// CPU-only ImageIO preparation. Empty PNG means reuse the source bytes, with
// oriented dimensions; original/automatic/fit512/portrait512/landscape512.
struct ReferencePreparation {
    int original_width = 0;
    int original_height = 0;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> png;
};
ReferencePreparation prepare_reference_image(const std::filesystem::path &, const std::string &preset);
NSDictionary *reference_preparation_metadata(const ReferencePreparation &);
// Strict JSON file API. A changed image requires a fresh .png destination in
// an existing directory; no-op returns source_path without touching output.
NSDictionary *prepare_image_file(NSDictionary *input);
}
