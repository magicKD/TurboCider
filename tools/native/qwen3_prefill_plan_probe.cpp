#include "../../native/components/text/qwen3.hpp"

#include <cstdio>
#include <iostream>

namespace {

int integer(const char *value) {
    size_t end = 0;
    const int parsed = std::stoi(value, &end);
    tc::require(end == std::string(value).size() && parsed > 0 && parsed <= 8192,
                "Qwen3 prefill plan token counts must be in 1...8192");
    return parsed;
}

std::string json_string(const std::string &value) {
    std::string result = "\"";
    for (unsigned char byte : value) {
        switch (byte) {
        case '\"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (byte < 0x20) {
                char escaped[7];
                std::snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
                result += escaped;
            } else {
                result += char(byte);
            }
        }
    }
    return result + "\"";
}

} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 2,
                    "usage: qwen3-prefill-plan-probe MANIFEST [TOKENS ...]");
        const auto manifest = std::filesystem::absolute(argv[1]).lexically_normal();
        tc::require(std::filesystem::is_regular_file(manifest),
                    "Qwen3 prefill manifest does not exist");
        std::vector<int> tokens;
        if (argc == 2)
            tokens = {64, 128, 256, 512, 1024, 2048};
        else
            for (int index = 2; index < argc; ++index)
                tokens.push_back(integer(argv[index]));

        std::cout << "{\"format\":\"turbocider-qwen3-prefill-plan-v1\","
                  << "\"manifest\":" << json_string(manifest.string())
                  << ",\"cases\":[";
        for (size_t index = 0; index < tokens.size(); ++index) {
            const auto plan = tc::components::qwen3_prefill_plan(
                manifest, tokens[index]);
            if (index) std::cout << ',';
            std::cout << "{\"actual_tokens\":" << plan.actual_tokens
                      << ",\"selected_bucket\":" << plan.selected_bucket
                      << ",\"compute_tokens\":" << plan.compute_tokens
                      << ",\"padding_tokens\":" << plan.padding_tokens
                      << ",\"minimum_profitable_rows\":"
                      << plan.minimum_profitable_rows
                      << ",\"use_hybrid\":"
                      << (plan.use_hybrid ? "true" : "false")
                      << ",\"fixed_shape\":"
                      << (plan.fixed_shape ? "true" : "false")
                      << ",\"reason\":" << json_string(plan.reason) << "}";
        }
        std::cout << "]}" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
