#import <Foundation/Foundation.h>
#include "../../native/models/qwen21/pe_generation.hpp"
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            const auto start = tc::Clock::now();
            auto elapsed = [](tc::Clock::time_point begin) { return std::chrono::duration<double>(tc::Clock::now() - begin).count(); };
            tc::require(argc == 5 || argc == 6 || (argc == 7 && std::string(argv[6]) == "--chunked"), "usage: qwen35-pe-sample-probe PE_DIR SYSTEM_PROMPT RAW_PROMPT OUTPUT_JSON [max-new-tokens [--chunked]]");
            const int max_new_tokens = argc >= 6 ? std::stoi(argv[5]) : 16256;
            tc::require(max_new_tokens > 0 && max_new_tokens <= 16256, "invalid PE-T2I token budget");
            tc::configure_streams();
            tc::Weights weights;
            std::atomic<bool> cancelled{false};
            weights.load(argv[1], [](auto &, int, int) {}, cancelled);
            tc::Tokenizer tokenizer{std::filesystem::path(argv[1])};
            auto systemData = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];
            tc::require(systemData != nil, "cannot read PE system prompt");
            systemData = [systemData stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
            NSString *user = [@(argv[3]) stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
            tc::require(systemData.length > 0 && user.length > 0, "empty PE system/user prompt");
            std::string system = systemData.UTF8String;
            tc::qwen21::pe::SamplingProfile sampling;
            sampling.max_new_tokens = max_new_tokens;
            tc::qwen21::pe::LanguageConfig language;
            language.chunked_prefill = argc == 7;
            auto result = tc::qwen21::pe::generate_text(weights, tokenizer, system, user.UTF8String,
                sampling, 42, [](const std::string &stage, int current, int total) {
                    std::cerr << (stage == "pe_prefill" ? "prefill " : "decode ") << current << '/' << total << '\n';
                }, cancelled, language);
            const auto &text = result.raw;
            NSString *raw = [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
            tc::require(raw != nil, "PE output has incomplete/invalid UTF-8");
            const auto &parsed = result.rewrite;
            NSMutableDictionary *out = [@{ @"token_count": @(result.generated_tokens),
                @"stopped_eos": @(result.stopped_eos), @"complete": @(result.complete()),
                @"max_new_tokens": @(max_new_tokens), @"prompt_tokens": @(result.prompt_tokens),
                @"sampler": @"native_mt19937_64_u53", @"seed": @42,
                @"chunked_prefill": @(language.chunked_prefill),
                @"temperature": @(sampling.temperature), @"top_p": @(sampling.top_p),
                @"top_k": @(sampling.top_k), @"presence_penalty": @(sampling.presence_penalty),
                @"prefill_seconds": @(result.prefill_seconds), @"decode_seconds": @(result.decode_seconds),
                @"wall_seconds": @(elapsed(start)),
                @"raw": raw, @"parse_ok": @(parsed.parse_ok),
                @"positive_prompt": @(parsed.positive_prompt.c_str()),
                @"wh_ratio": @(parsed.wh_ratio.c_str()) } mutableCopy];
            NSData *json = [NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingPrettyPrinted error:nil];
            tc::require(json != nil && [json writeToFile:@(argv[4]) atomically:YES], "cannot write PE report");
            std::cout << text << '\n';
            return 0;
        } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
