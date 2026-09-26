#import <Foundation/Foundation.h>
#include "../../native/core/tokenizer.hpp"
#include "../../native/models/qwen21/prompt_rewrite.hpp"
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc == 3, "usage: qwen35-tokenizer-probe tokenizer-dir fixture.json");
            tc::Tokenizer tokenizer(argv[1]);
            NSData *data = [NSData dataWithContentsOfFile:@(argv[2])];
            NSArray *cases = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            tc::require([cases isKindOfClass:NSArray.class], "invalid tokenizer cases");
            int count = 0;
            for (NSDictionary *test in cases) {
                NSData *utf8 = [test[@"text"] dataUsingEncoding:NSUTF8StringEncoding];
                std::string text(static_cast<const char *>(utf8.bytes), utf8.length);
                if (test[@"system"]) {
                    auto chat = tc::qwen21::prompt_rewrite_chat([test[@"system"] UTF8String],
                        [test[@"prompt"] UTF8String], [test[@"references"] unsignedIntegerValue]);
                    tc::require(chat == text, "PE chat differs from official Jinja template");
                }
                auto tokens = tokenizer.raw_bounded(text, 32768);
                std::vector<int> expected;
                for (NSNumber *n in test[@"ids"]) expected.push_back(n.intValue);
                if (tokens.ids != expected) {
                    std::cerr << "tokenizer case " << count << " mismatch: native=";
                    for (int id : tokens.ids) std::cerr << id << ',';
                    std::cerr << " expected=";
                    for (int id : expected) std::cerr << id << ',';
                    std::cerr << '\n';
                    throw std::runtime_error("PE tokenizer encoding mismatch");
                }
                NSData *decoded = [test[@"decoded"] dataUsingEncoding:NSUTF8StringEncoding];
                tc::require(tokenizer.decode(tokens.ids) == std::string(static_cast<const char *>(decoded.bytes), decoded.length),
                            "PE token decode mismatch");
                std::string incremental;
                for (int id : tokens.ids) incremental += tokenizer.decode({id});
                tc::require(incremental == tokenizer.decode(tokens.ids), "incremental byte decode mismatch");
                bool legacy_rejected = false;
                try { (void)tokenizer.raw(text); } catch (const std::exception &) { legacy_rejected = true; }
                tc::require(legacy_rejected == (expected.size() > 512 || text.size() > 32768), "legacy tokenizer limit changed");
                ++count;
            }
            for (int id : {-1, 99999999}) {
                bool rejected = false;
                try { (void)tokenizer.decode({id}); } catch (const std::exception &) { rejected = true; }
                tc::require(rejected, "invalid token ID decoded");
            }
            tc::require(tokenizer.decode({}).empty(), "empty decode is not empty");
            std::cout << "PASS " << count << " tokenizer encoding/decoding/incremental/limit cases\n";
            return 0;
        } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
