#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace tc::ane {
inline std::vector<int> normalized_gpu_layers(std::vector<int> layers, int limit = 128) {
    if (limit <= 0 || limit > 128 || layers.size() > size_t(limit))
        throw std::invalid_argument("invalid runtime GPU layer policy bound");
    std::sort(layers.begin(), layers.end());
    if (std::any_of(layers.begin(), layers.end(), [limit](int layer) { return layer<0 || layer>=limit; }) ||
            std::adjacent_find(layers.begin(), layers.end()) != layers.end())
        throw std::invalid_argument("invalid or repeated runtime GPU layer ordinal");
    return layers;
}
inline std::vector<int> parse_gpu_layers(const char *raw, int limit) {
    if(limit<=0 || limit>128)throw std::invalid_argument("invalid runtime GPU layer policy bound");
    if (!raw) return {};
    const std::string value(raw);
    if (value.empty() || value.back()==',') throw std::invalid_argument("empty/trailing runtime GPU layer list");
    std::vector<int> result;
    size_t begin = 0;
    while (begin < value.size()) {
        const auto comma=value.find(',',begin);
        const auto part=value.substr(begin,comma==std::string::npos?comma:comma-begin);
        if (part.empty() || part.size()>3 || part.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("runtime GPU layers require unsigned comma-separated ordinals");
        result.push_back(std::stoi(part));
        if (comma==std::string::npos) break;
        begin=comma+1;
    }
    return normalized_gpu_layers(std::move(result),limit);
}
} // namespace tc::ane
