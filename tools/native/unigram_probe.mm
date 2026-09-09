#import <Foundation/Foundation.h>
#include "../../native/core/unigram_tokenizer.hpp"
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc == 2, "usage: unigram-probe TOKENIZER_ROOT < prompts.json");
            tc::UnigramTokenizer tokenizer(std::filesystem::absolute(argv[1]));
            NSData *input = [[NSFileHandle fileHandleWithStandardInput] readDataToEndOfFile];
            id records = [NSJSONSerialization JSONObjectWithData:input options:0 error:nil];
            tc::require([records isKindOfClass:NSArray.class], "expected JSON prompt array");
            NSMutableArray *results = [NSMutableArray array];
            for (NSDictionary *record in records) {
                NSString *prompt = record[@"prompt"];
                NSData *bytes = [prompt dataUsingEncoding:NSUTF8StringEncoding];
                auto tokens = tokenizer.prompt(std::string((const char *)bytes.bytes, bytes.length),
                                                [record[@"limit"] intValue]);
                NSMutableArray *ids = [NSMutableArray array];
                for (int id : tokens.ids) [ids addObject:@(id)];
                [results addObject:@{@"ids":ids, @"valid":@(tokens.valid)}];
            }
            NSData *json = [NSJSONSerialization dataWithJSONObject:results options:0 error:nil];
            [[NSFileHandle fileHandleWithStandardOutput] writeData:json];
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
