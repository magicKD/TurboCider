#include "ltx_gemma_tokenizer.h"

#import <Foundation/Foundation.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

@interface LTXGemmaTokenizer : NSObject
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *vocab;
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *ranks;
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *special;
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *byteFallback;
@property(nonatomic) uint32_t padToken;
@property(nonatomic) uint32_t bosToken;
@end

@implementation LTXGemmaTokenizer
@end

static LTXGemmaTokenizer *ltx_tok(const ltx_gemma_tokenizer *opaque) {
    return (__bridge LTXGemmaTokenizer *)(void *)opaque;
}

static int ltx_fail(char *error, size_t error_size,
                    const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static NSString *ltx_pair_key(NSString *left, NSString *right) {
    return [NSString stringWithFormat:@"%@\uffff%@", left, right];
}

static NSArray<NSString *> *ltx_symbols(NSString *text) {
    NSMutableArray<NSString *> *result = [NSMutableArray array];
    for (NSUInteger index = 0; index < text.length;) {
        unichar first = [text characterAtIndex:index];
        NSUInteger units = 1u;
        if (first >= 0xd800u && first <= 0xdbffu && index + 1u < text.length) {
            unichar second = [text characterAtIndex:index + 1u];
            if (second >= 0xdc00u && second <= 0xdfffu) units = 2u;
        }
        [result addObject:[text substringWithRange:NSMakeRange(index, units)]];
        index += units;
    }
    return result;
}

static int ltx_append_piece(LTXGemmaTokenizer *tokenizer, NSString *piece,
                            NSMutableArray<NSNumber *> *output,
                            NSString **failure) {
    NSMutableArray<NSString *> *symbols = [ltx_symbols(piece) mutableCopy];
    while (symbols.count > 1u) {
        NSNumber *bestRank = nil;
        NSUInteger bestIndex = 0;
        for (NSUInteger index = 0; index + 1u < symbols.count; index++) {
            NSNumber *rank = tokenizer.ranks[ltx_pair_key(symbols[index],
                                                           symbols[index + 1u])];
            if (rank && (!bestRank || rank.unsignedIntegerValue <
                         bestRank.unsignedIntegerValue)) {
                bestRank = rank;
                bestIndex = index;
            }
        }
        if (!bestRank) break;
        NSString *merged = [symbols[bestIndex]
            stringByAppendingString:symbols[bestIndex + 1u]];
        [symbols replaceObjectAtIndex:bestIndex withObject:merged];
        [symbols removeObjectAtIndex:bestIndex + 1u];
    }
    for (NSString *symbol in symbols) {
        NSNumber *identifier = tokenizer.vocab[symbol];
        if (identifier) {
            [output addObject:identifier];
            continue;
        }
        NSData *data = [symbol dataUsingEncoding:NSUTF8StringEncoding];
        const unsigned char *bytes = data.bytes;
        for (NSUInteger index = 0; index < data.length; index++) {
            NSString *name = [NSString stringWithFormat:@"<0x%02X>",
                              (unsigned)bytes[index]];
            identifier = tokenizer.byteFallback[name];
            if (!identifier) {
                if (failure) *failure = [NSString stringWithFormat:
                    @"Gemma byte-fallback token is absent from vocabulary: %@",
                    name];
                return 0;
            }
            [output addObject:identifier];
        }
    }
    return 1;
}

static int ltx_encode_text(LTXGemmaTokenizer *tokenizer, NSString *input,
                           NSMutableArray<NSNumber *> *output,
                           NSString **failure) {
    /* Gemma's tokenizer.json contains a Replace normalizer, not NFC. */
    NSString *normalized = [input stringByReplacingOccurrencesOfString:@" "
                                                                withString:@"\u2581"];
    NSUInteger start = 0;
    while (start < normalized.length) {
        NSRange search = NSMakeRange(start, normalized.length - start);
        NSRange match = NSMakeRange(NSNotFound, 0);
        NSString *matched = nil;
        for (NSString *candidate in tokenizer.special) {
            NSRange found = [normalized rangeOfString:candidate options:0 range:search];
            if (found.location == NSNotFound) continue;
            if (match.location == NSNotFound || found.location < match.location ||
                (found.location == match.location && found.length > match.length)) {
                match = found;
                matched = candidate;
            }
        }
        NSUInteger stop = match.location == NSNotFound ? normalized.length : match.location;
        if (stop > start && !ltx_append_piece(tokenizer,
                                              [normalized substringWithRange:
                                                  NSMakeRange(start, stop - start)],
                                              output, failure)) return 0;
        if (matched) {
            [output addObject:tokenizer.special[matched]];
            start = NSMaxRange(match);
        } else {
            break;
        }
    }
    return 1;
}

ltx_gemma_tokenizer *ltx_gemma_tokenizer_load(
    const char *tokenizer_json, char *error, size_t error_size) {
    @autoreleasepool {
        if (error && error_size) error[0] = '\0';
        if (!tokenizer_json) {
            ltx_fail(error, error_size, "Gemma tokenizer path is required");
            return NULL;
        }
        NSString *path = [NSString stringWithUTF8String:tokenizer_json];
        NSData *data = [NSData dataWithContentsOfFile:path];
        if (!data) {
            ltx_fail(error, error_size, "cannot read Gemma tokenizer: %s",
                     tokenizer_json);
            return NULL;
        }
        NSError *jsonError = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                               options:0
                                                                 error:&jsonError];
        NSDictionary *model = [root isKindOfClass:NSDictionary.class] ?
            root[@"model"] : nil;
        NSDictionary *normalizer = [root isKindOfClass:NSDictionary.class] ?
            root[@"normalizer"] : nil;
        if (![model isKindOfClass:NSDictionary.class] ||
            ![normalizer isKindOfClass:NSDictionary.class] ||
            ![model[@"type"] isEqual:@"BPE"] ||
            ![model[@"vocab"] isKindOfClass:NSDictionary.class] ||
            ![model[@"merges"] isKindOfClass:NSArray.class] ||
            ![normalizer[@"type"] isEqual:@"Replace"] ||
            ![normalizer[@"pattern"] isKindOfClass:NSDictionary.class] ||
            ![normalizer[@"pattern"][@"String"] isEqual:@" "] ||
            ![normalizer[@"content"] isEqual:@"▁"]) {
            ltx_fail(error, error_size,
                     "unsupported Gemma tokenizer specification");
            return NULL;
        }

        LTXGemmaTokenizer *tokenizer = [[LTXGemmaTokenizer alloc] init];
        tokenizer.vocab = model[@"vocab"];
        NSMutableDictionary *special = [NSMutableDictionary dictionary];
        for (id raw in root[@"added_tokens"]) {
            if (![raw isKindOfClass:NSDictionary.class]) {
                ltx_fail(error, error_size, "invalid Gemma added token");
                return NULL;
            }
            NSDictionary *entry = raw;
            if ([entry[@"single_word"] boolValue] || [entry[@"lstrip"] boolValue] ||
                [entry[@"rstrip"] boolValue] || [entry[@"normalized"] boolValue]) {
                ltx_fail(error, error_size,
                         "unsupported Gemma added-token policy");
                return NULL;
            }
            NSString *content = entry[@"content"];
            NSNumber *identifier = entry[@"id"];
            if (![content isKindOfClass:NSString.class] ||
                ![identifier isKindOfClass:NSNumber.class]) {
                ltx_fail(error, error_size, "invalid Gemma added token");
                return NULL;
            }
            special[content] = identifier;
        }
        tokenizer.special = special;
        NSNumber *pad = tokenizer.vocab[@"<pad>"];
        if (!pad) pad = special[@"<pad>"];
        NSNumber *bos = tokenizer.vocab[@"<bos>"];
        if (!bos) bos = special[@"<bos>"];
        if (!pad || !bos) {
            ltx_fail(error, error_size,
                     "Gemma tokenizer requires <pad> and <bos> tokens");
            return NULL;
        }
        tokenizer.padToken = pad.unsignedIntValue;
        tokenizer.bosToken = bos.unsignedIntValue;

        NSMutableDictionary *ranks = [NSMutableDictionary dictionary];
        NSUInteger rank = 0;
        for (id raw in model[@"merges"]) {
            if (![raw isKindOfClass:NSArray.class] || [raw count] != 2u ||
                ![raw[0] isKindOfClass:NSString.class] ||
                ![raw[1] isKindOfClass:NSString.class]) {
                ltx_fail(error, error_size, "invalid Gemma BPE merge");
                return NULL;
            }
            ranks[ltx_pair_key(raw[0], raw[1])] = @(rank++);
        }
        tokenizer.ranks = ranks;

        NSMutableDictionary *fallback = [NSMutableDictionary dictionary];
        for (NSString *symbol in tokenizer.vocab) {
            if (symbol.length != 6u || ![symbol hasPrefix:@"<0x"] ||
                ![symbol hasSuffix:@">"]) continue;
            unsigned value = 0;
            NSScanner *scanner = [NSScanner scannerWithString:
                [symbol substringWithRange:NSMakeRange(3u, 2u)]];
            if ([scanner scanHexInt:&value] && scanner.isAtEnd)
                fallback[symbol] = tokenizer.vocab[symbol];
        }
        if (fallback.count != 256u) {
            ltx_fail(error, error_size,
                     "Gemma byte-fallback vocabulary is incomplete");
            return NULL;
        }
        tokenizer.byteFallback = fallback;
        return (__bridge_retained ltx_gemma_tokenizer *)tokenizer;
    }
}

void ltx_gemma_tokenizer_free(ltx_gemma_tokenizer *opaque) {
    if (opaque) CFBridgingRelease(opaque);
}

int ltx_gemma_tokenizer_encode(
    const ltx_gemma_tokenizer *opaque, const char *utf8,
    uint32_t max_length, uint32_t **ids, uint8_t **mask, size_t *count,
    char *error, size_t error_size) {
    @autoreleasepool {
        if (error && error_size) error[0] = '\0';
        if (!opaque || !utf8 || !ids || !mask || !count)
            return ltx_fail(error, error_size,
                            "invalid Gemma tokenizer arguments");
        *ids = NULL; *mask = NULL; *count = 0;
        NSString *input = [NSString stringWithUTF8String:utf8];
        if (!input) return ltx_fail(error, error_size,
                                    "prompt is not valid UTF-8");
        LTXGemmaTokenizer *tokenizer = ltx_tok(opaque);
        NSMutableArray<NSNumber *> *raw = [NSMutableArray array];
        /* ComfyUI's Gemma4SDTokenizer declares has_start_token=True with
         * start_token=2.  LTX conditioning therefore includes BOS before the
         * prompt BPE tokens; omitting it shifts every RoPE position. */
        [raw addObject:@(tokenizer.bosToken)];
        NSString *failure = nil;
        if (!ltx_encode_text(tokenizer, input, raw, &failure))
            return ltx_fail(error, error_size, "%s", failure.UTF8String);
        NSUInteger limit = max_length ? max_length : raw.count;
        NSUInteger first = raw.count > limit ? raw.count - limit : 0u;
        NSUInteger real = raw.count - first;
        uint32_t *outIDs = calloc(limit ? limit : 1u, sizeof(*outIDs));
        uint8_t *outMask = calloc(limit ? limit : 1u, sizeof(*outMask));
        if (!outIDs || !outMask) {
            free(outIDs); free(outMask);
            return ltx_fail(error, error_size,
                            "out of memory encoding Gemma prompt");
        }
        NSUInteger padding = limit - real;
        for (NSUInteger index = 0; index < padding; index++)
            outIDs[index] = tokenizer.padToken;
        for (NSUInteger index = 0; index < real; index++) {
            outIDs[padding + index] = raw[first + index].unsignedIntValue;
            outMask[padding + index] = 1u;
        }
        *ids = outIDs; *mask = outMask; *count = limit;
        return 1;
    }
}

void ltx_gemma_tokenizer_ids_free(uint32_t *ids, uint8_t *mask) {
    free(ids);
    free(mask);
}
