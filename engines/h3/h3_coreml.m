#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "h3_coreml.h"

#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

@interface H3CoreMLMLP : NSObject {
@public
    MLModel *model;
    MLMultiArray *input;
    MLMultiArray *output;
    MLDictionaryFeatureProvider *provider;
    MLPredictionOptions *options;
    NSString *outputName;
    dispatch_semaphore_t completion;
    dispatch_semaphore_t loadCompletion;
    NSError *asyncError;
    NSString *loadFailure;
    NSString *manifestSHA256;
    id<MLFeatureProvider> prediction;
    h3_gpu_tensor *inputTensor;
    h3_gpu_tensor *outputTensor;
    BOOL inflight;
    BOOL loadFinished;
    BOOL loaded;
    BOOL ownsTensors;
}
@end

@implementation H3CoreMLMLP
- (void)dealloc {
    if (!loadFinished && loadCompletion)
        dispatch_semaphore_wait(loadCompletion, DISPATCH_TIME_FOREVER);
    if (inflight) dispatch_semaphore_wait(completion, DISPATCH_TIME_FOREVER);
    if (ownsTensors) {
        h3_gpu_tensor_free(outputTensor);
        h3_gpu_tensor_free(inputTensor);
    }
}
@end

static H3CoreMLMLP *MLP(h3_coreml_mlp *opaque) {
    return (__bridge H3CoreMLMLP *)opaque;
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int parse_binary_option(const char *name, const char *value,
                               int *result,
                               char *error, size_t error_size) {
    if (!value || !*value || !strcmp(value, "0")) {
        *result = 0;
        return 1;
    }
    if (!strcmp(value, "1")) {
        *result = 1;
        return 1;
    }
    fail(error, error_size, "%s must be 0 or 1", name);
    return 0;
}

int h3_coreml_mlp_configure_fallback(
    const char *enabled_value, const char *forced_value,
    const char *latch_value, h3_coreml_mlp_fallback_options *options,
    char *error, size_t error_size) {
    if (!options) {
        fail(error, error_size, "Core ML fallback options are absent");
        return 0;
    }
    h3_coreml_mlp_fallback_options parsed = {0};
    if (!parse_binary_option("H3_COREML_GPU_FALLBACK", enabled_value,
                             &parsed.enabled, error, error_size) ||
        !parse_binary_option("H3_COREML_FORCE_GPU_FALLBACK", forced_value,
                             &parsed.forced, error, error_size) ||
        !parse_binary_option("H3_COREML_FALLBACK_LATCH", latch_value,
                             &parsed.latch, error, error_size)) return 0;
    if (!parsed.enabled && (parsed.forced || parsed.latch)) {
        fail(error, error_size,
             "H3_COREML_FORCE_GPU_FALLBACK and H3_COREML_FALLBACK_LATCH "
             "require H3_COREML_GPU_FALLBACK=1");
        return 0;
    }
    *options = parsed;
    return 1;
}

static int sha256_hex_valid(const char *value) {
    if (!value || strlen(value) != H3_COREML_SHA256_HEX_SIZE - 1) return 0;
    for (size_t index = 0; index < H3_COREML_SHA256_HEX_SIZE - 1; index++) {
        char octet = value[index];
        if (!((octet >= '0' && octet <= '9') ||
              (octet >= 'a' && octet <= 'f'))) return 0;
    }
    return 1;
}

static void sha256_bytes(const void *data, size_t bytes,
                         char output[H3_COREML_SHA256_HEX_SIZE]) {
    CC_SHA256_CTX context;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Init(&context);
    const unsigned char *cursor = data;
    while (bytes) {
        CC_LONG chunk = bytes > UINT32_MAX ? UINT32_MAX : (CC_LONG)bytes;
        CC_SHA256_Update(&context, cursor, chunk);
        cursor += chunk;
        bytes -= chunk;
    }
    CC_SHA256_Final(digest, &context);
    static const char hexadecimal[] = "0123456789abcdef";
    for (size_t index = 0; index < sizeof(digest); index++) {
        output[index * 2] = hexadecimal[digest[index] >> 4];
        output[index * 2 + 1] = hexadecimal[digest[index] & 15];
    }
    output[H3_COREML_SHA256_HEX_SIZE - 1] = '\0';
}

static int sha256_file(const char *path,
                       char output[H3_COREML_SHA256_HEX_SIZE],
                       char *error, size_t error_size) {
    if (!path || !*path) {
        fail(error, error_size, "SHA-256 file path is required");
        return 0;
    }
    @autoreleasepool {
        NSError *detail = nil;
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:path]
            options:NSDataReadingMappedIfSafe error:&detail];
        if (!data) {
            fail(error, error_size, "cannot read SHA-256 file %s: %s", path,
                 detail.localizedDescription.UTF8String);
            return 0;
        }
        sha256_bytes(data.bytes, data.length, output);
    }
    return 1;
}

static int manifest_string(NSDictionary *manifest, NSString *key,
                           NSString **value,
                           char *error, size_t error_size) {
    id candidate = manifest[key];
    if (![candidate isKindOfClass:[NSString class]] ||
        ![(NSString *)candidate length]) {
        fail(error, error_size, "Core ML manifest field %s must be a string",
             key.UTF8String);
        return 0;
    }
    *value = candidate;
    return 1;
}

static int manifest_uint(NSDictionary *manifest, NSString *key,
                         uint64_t expected,
                         char *error, size_t error_size) {
    id candidate = manifest[key];
    if (![candidate isKindOfClass:[NSNumber class]] ||
        [(NSNumber *)candidate unsignedLongLongValue] != expected) {
        fail(error, error_size,
             "Core ML manifest field %s does not match runtime (%llu)",
             key.UTF8String, (unsigned long long)expected);
        return 0;
    }
    return 1;
}

static int manifest_file_entry(NSDictionary *files, NSString *name,
                               const void *data, size_t bytes, int verify,
                               char *error, size_t error_size) {
    id candidate = files[name];
    if (![candidate isKindOfClass:[NSDictionary class]]) {
        fail(error, error_size, "Core ML manifest is missing %s",
             name.UTF8String);
        return 0;
    }
    NSDictionary *entry = candidate;
    id count = entry[@"bytes"];
    id digest = entry[@"sha256"];
    if (![count isKindOfClass:[NSNumber class]] ||
        ![digest isKindOfClass:[NSString class]] ||
        !sha256_hex_valid([(NSString *)digest UTF8String])) {
        fail(error, error_size,
             "Core ML manifest file entry %s is malformed",
             name.UTF8String);
        return 0;
    }
    if (bytes && [(NSNumber *)count unsignedLongLongValue] != bytes) {
        fail(error, error_size,
             "Core ML manifest byte count mismatch for %s",
             name.UTF8String);
        return 0;
    }
    if (!verify) return 1;
    char actual[H3_COREML_SHA256_HEX_SIZE];
    sha256_bytes(data, bytes, actual);
    if (strcmp(actual, [(NSString *)digest UTF8String])) {
        fail(error, error_size, "Core ML manifest SHA-256 mismatch for %s",
             name.UTF8String);
        return 0;
    }
    return 1;
}

static int validate_channel_selection(NSDictionary *selection,
                                      NSString *expected_method,
                                      uint32_t ane_intermediate,
                                      uint32_t full_intermediate,
                                      char *error, size_t error_size) {
    id method = selection[@"method"];
    id ane_value = selection[@"ane_indices"];
    id gpu_value = selection[@"gpu_indices"];
    if (![method isKindOfClass:[NSString class]] ||
        ![(NSString *)method isEqualToString:expected_method] ||
        ![ane_value isKindOfClass:[NSArray class]] ||
        ![gpu_value isKindOfClass:[NSArray class]]) {
        fail(error, error_size,
             "Core ML channel selection metadata is malformed");
        return 0;
    }
    NSArray *ane_indices = ane_value;
    NSArray *gpu_indices = gpu_value;
    if (ane_indices.count != ane_intermediate ||
        gpu_indices.count != full_intermediate - ane_intermediate) {
        fail(error, error_size,
             "Core ML channel selection count does not match split");
        return 0;
    }
    unsigned char *seen = calloc(full_intermediate, sizeof(*seen));
    if (!seen) {
        fail(error, error_size,
             "out of memory validating Core ML channel selection");
        return 0;
    }
    int prefix = [expected_method isEqualToString:@"prefix"];
    int ok = 1;
    for (unsigned group = 0; group < 2 && ok; group++) {
        NSArray *indices = group ? gpu_indices : ane_indices;
        uint32_t base = group ? ane_intermediate : 0;
        for (NSUInteger position = 0; position < indices.count; position++) {
            id candidate = indices[position];
            double numeric = [candidate isKindOfClass:[NSNumber class]] ?
                [(NSNumber *)candidate doubleValue] : NAN;
            if (!isfinite(numeric) || numeric < 0.0 ||
                numeric != floor(numeric) || numeric >= full_intermediate) {
                fail(error, error_size,
                     "Core ML channel selection index is invalid");
                ok = 0;
                break;
            }
            uint32_t index = (uint32_t)numeric;
            if (seen[index] || (prefix && index != base + position)) {
                fail(error, error_size,
                     "Core ML channel selection is not a valid partition");
                ok = 0;
                break;
            }
            seen[index] = 1;
        }
    }
    for (uint32_t index = 0; index < full_intermediate && ok; index++) {
        if (!seen[index]) {
            fail(error, error_size,
                 "Core ML channel selection does not cover all channels");
            ok = 0;
        }
    }
    free(seen);
    return ok;
}

int h3_coreml_mlp_validate_manifest(
    const char *manifest_path, const char *model_path,
    const char *checkpoint_index_path,
    const void *gpu_fc1, size_t gpu_fc1_bytes,
    const void *gpu_fc2, size_t gpu_fc2_bytes,
    const void *ane_fc1, size_t ane_fc1_bytes,
    const void *ane_fc2, size_t ane_fc2_bytes,
    const h3_coreml_mlp_manifest_expectation *expected,
    char manifest_sha256[H3_COREML_SHA256_HEX_SIZE],
    char *error, size_t error_size) {
    if (!manifest_path || !*manifest_path || !model_path || !*model_path ||
        !checkpoint_index_path || !*checkpoint_index_path ||
        !gpu_fc1_bytes || !gpu_fc2_bytes ||
        ((ane_fc1_bytes == 0) != (ane_fc2_bytes == 0)) ||
        ((gpu_fc1 == NULL) != (gpu_fc2 == NULL)) ||
        ((ane_fc1 == NULL) != (ane_fc2 == NULL)) ||
        (ane_fc1 != NULL && !ane_fc1_bytes) ||
        !expected || !manifest_sha256 || !expected->intermediate ||
        expected->intermediate >= expected->full_intermediate) {
        fail(error, error_size, "invalid Core ML manifest arguments");
        return 0;
    }
    @autoreleasepool {
        NSError *detail = nil;
        NSData *encoded = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:manifest_path]
            options:NSDataReadingMappedIfSafe error:&detail];
        if (!encoded) {
            fail(error, error_size, "cannot read Core ML manifest %s: %s",
                 manifest_path, detail.localizedDescription.UTF8String);
            return 0;
        }
        id root = [NSJSONSerialization JSONObjectWithData:encoded
                                                   options:0 error:&detail];
        if (![root isKindOfClass:[NSDictionary class]]) {
            fail(error, error_size, "cannot parse Core ML manifest %s: %s",
                 manifest_path, detail.localizedDescription.UTF8String);
            return 0;
        }
        NSDictionary *manifest = root;
        NSString *schema = nil;
        NSString *layout = nil;
        NSString *selection = nil;
        NSString *checkpoint = nil;
        NSString *checkpoint_index = nil;
        NSString *output_scale_bits = nil;
        NSString *model_filename = nil;
        NSString *source_shard = nil;
        NSString *fc1_name = nil;
        NSString *fc2_name = nil;
        if (!manifest_string(manifest, @"schema", &schema,
                             error, error_size) ||
            ![schema isEqualToString:@"h3-coreml-mlp-v1"] ||
            !manifest_string(manifest, @"layout", &layout,
                             error, error_size) ||
            ![layout isEqualToString:@"conv"] ||
            !manifest_string(manifest, @"channel_selection", &selection,
                             error, error_size) ||
            !([selection isEqualToString:@"prefix"] ||
              [selection isEqualToString:@"low_energy"]) ||
            !manifest_string(manifest, @"checkpoint_identity", &checkpoint,
                             error, error_size) ||
            !manifest_string(manifest, @"checkpoint_index_sha256",
                             &checkpoint_index, error, error_size) ||
            !manifest_string(manifest, @"output_scale_f32_bits",
                             &output_scale_bits, error, error_size) ||
            !manifest_string(manifest, @"model_filename", &model_filename,
                             error, error_size) ||
            !manifest_string(manifest, @"source_shard", &source_shard,
                             error, error_size) ||
            !manifest_string(manifest, @"fc1_name", &fc1_name,
                             error, error_size) ||
            !manifest_string(manifest, @"fc2_name", &fc2_name,
                             error, error_size)) {
            if (error && error_size && !error[0])
                fail(error, error_size, "Core ML manifest policy mismatch");
            return 0;
        }
        if (!sha256_hex_valid(checkpoint_index.UTF8String)) {
            fail(error, error_size,
                 "Core ML manifest checkpoint index SHA-256 is malformed");
            return 0;
        }
        char expected_checkpoint[17];
        snprintf(expected_checkpoint, sizeof(expected_checkpoint), "%016llx",
                 (unsigned long long)expected->checkpoint_identity);
        if (strcmp(checkpoint.UTF8String, expected_checkpoint)) {
            fail(error, error_size,
                 "Core ML manifest checkpoint identity mismatch: "
                 "artifact %s runtime %s",
                 checkpoint.UTF8String, expected_checkpoint);
            return 0;
        }
        char expected_fc1[96];
        char expected_fc2[96];
        snprintf(expected_fc1, sizeof(expected_fc1),
                 "blocks.%u.mlp.fc1.weight", expected->block_index);
        snprintf(expected_fc2, sizeof(expected_fc2),
                 "blocks.%u.mlp.fc2.weight", expected->block_index);
        NSString *actual_model = [[NSString stringWithUTF8String:model_path]
            lastPathComponent];
        if (![model_filename isEqualToString:actual_model] ||
            strcmp(fc1_name.UTF8String, expected_fc1) ||
            strcmp(fc2_name.UTF8String, expected_fc2)) {
            fail(error, error_size,
                 "Core ML manifest model/block tensor identity mismatch");
            return 0;
        }
        if (!manifest_uint(manifest, @"block_index", expected->block_index,
                           error, error_size) ||
            !manifest_uint(manifest, @"rows", expected->rows,
                           error, error_size) ||
            !manifest_uint(manifest, @"hidden", expected->hidden,
                           error, error_size) ||
            !manifest_uint(manifest, @"ane_intermediate",
                           expected->intermediate, error, error_size) ||
            !manifest_uint(manifest, @"full_intermediate",
                           expected->full_intermediate, error, error_size) ||
            !manifest_uint(manifest, @"gpu_intermediate",
                           expected->full_intermediate - expected->intermediate,
                           error, error_size)) return 0;
        id output_scale = manifest[@"output_scale"];
        double scale = [output_scale isKindOfClass:[NSNumber class]] ?
            [(NSNumber *)output_scale doubleValue] : NAN;
        double tolerance = fmax(1e-7, fabs(expected->output_scale) * 1e-6);
        if (!isfinite(scale) ||
            fabs(scale - (double)expected->output_scale) > tolerance) {
            fail(error, error_size,
                 "Core ML manifest output scale mismatch");
            return 0;
        }
        uint32_t expected_scale_bits = 0;
        memcpy(&expected_scale_bits, &expected->output_scale,
               sizeof(expected_scale_bits));
        char expected_scale_hex[9];
        snprintf(expected_scale_hex, sizeof(expected_scale_hex), "%08x",
                 expected_scale_bits);
        if (strcmp(output_scale_bits.UTF8String, expected_scale_hex)) {
            fail(error, error_size,
                 "Core ML manifest output scale bit pattern mismatch");
            return 0;
        }
        id bits = manifest[@"weight_bits"];
        unsigned long long weight_bits =
            [bits isKindOfClass:[NSNumber class]] ?
                [(NSNumber *)bits unsignedLongLongValue] : 0;
        if (weight_bits != 4 && weight_bits != 8 && weight_bits != 16) {
            fail(error, error_size,
                 "Core ML manifest weight precision is invalid");
            return 0;
        }
        id files_value = manifest[@"files"];
        if (![files_value isKindOfClass:[NSDictionary class]]) {
            fail(error, error_size, "Core ML manifest files are missing");
            return 0;
        }
        NSDictionary *files = files_value;
        if (!manifest_file_entry(files, @"gpu_fc1.bf16",
                                 gpu_fc1, gpu_fc1_bytes, gpu_fc1 != NULL,
                                 error, error_size) ||
            !manifest_file_entry(files, @"gpu_fc2.bf16",
                                 gpu_fc2, gpu_fc2_bytes, gpu_fc2 != NULL,
                                 error, error_size)) return 0;
        NSString *manifest_file =
            [NSString stringWithUTF8String:manifest_path];
        NSString *manifest_directory =
            [manifest_file stringByDeletingLastPathComponent];
        if (!manifest_directory.length) manifest_directory = @".";
        NSString *selection_path = [manifest_directory
            stringByAppendingPathComponent:@"channel_selection.json"];
        NSData *selection_data = [NSData dataWithContentsOfFile:selection_path
                                      options:NSDataReadingMappedIfSafe
                                        error:&detail];
        if (!selection_data) {
            fail(error, error_size,
                 "cannot read Core ML channel selection %s: %s",
                 selection_path.UTF8String,
                 detail.localizedDescription.UTF8String);
            return 0;
        }
        if (!manifest_file_entry(files, @"channel_selection.json",
                                 selection_data.bytes, selection_data.length,
                                 1, error, error_size)) return 0;
        id selection_root = [NSJSONSerialization
            JSONObjectWithData:selection_data options:0 error:&detail];
        if (![selection_root isKindOfClass:[NSDictionary class]]) {
            fail(error, error_size,
                 "cannot parse Core ML channel selection %s: %s",
                 selection_path.UTF8String,
                 detail.localizedDescription.UTF8String);
            return 0;
        }
        if (!validate_channel_selection(
                selection_root, selection, expected->intermediate,
                expected->full_intermediate, error, error_size)) return 0;
        if (ane_fc1_bytes &&
            (!manifest_file_entry(files, @"ane_fc1.bf16",
                                  ane_fc1, ane_fc1_bytes, ane_fc1 != NULL,
                                  error, error_size) ||
             !manifest_file_entry(files, @"ane_fc2.bf16",
                                  ane_fc2, ane_fc2_bytes, ane_fc2 != NULL,
                                  error, error_size))) return 0;
        char actual_index[H3_COREML_SHA256_HEX_SIZE];
        if (!sha256_file(checkpoint_index_path, actual_index,
                         error, error_size)) return 0;
        if (strcmp(actual_index, checkpoint_index.UTF8String)) {
            fail(error, error_size,
                 "Core ML manifest checkpoint index SHA-256 mismatch");
            return 0;
        }
        sha256_bytes(encoded.bytes, encoded.length, manifest_sha256);
        if (getenv("H3_PROFILE"))
            fprintf(stderr,
                    "h3: verified Core ML MLP manifest block=%u sha256=%s "
                    "source=%s\n",
                    expected->block_index, manifest_sha256,
                    source_shard.UTF8String);
    }
    return 1;
}

static int fallback_sibling_path_matches(NSString *manifest_path,
                                         const char *path,
                                         NSString *filename,
                                         char *error, size_t error_size) {
    if (!path || !*path) {
        fail(error, error_size, "Core ML fallback shard path is absent");
        return 0;
    }
    NSString *directory = [manifest_path stringByDeletingLastPathComponent];
    if (!directory.length) directory = @".";
    NSString *expected = [[directory stringByAppendingPathComponent:filename]
        stringByStandardizingPath];
    NSString *actual = [[NSString stringWithUTF8String:path]
        stringByStandardizingPath];
    if (!actual || ![actual isEqualToString:expected]) {
        fail(error, error_size,
             "Core ML fallback shard path mismatch for %s",
             filename.UTF8String);
        return 0;
    }
    return 1;
}

static int fallback_file_exact_size(const char *path, size_t expected_bytes,
                                    char *error, size_t error_size) {
    struct stat status;
    if (stat(path, &status) != 0) {
        fail(error, error_size, "cannot stat Core ML fallback shard %s: %s",
             path, strerror(errno));
        return 0;
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
        (uint64_t)status.st_size != (uint64_t)expected_bytes) {
        fail(error, error_size,
             "Core ML fallback shard size mismatch for %s", path);
        return 0;
    }
    return 1;
}

int h3_coreml_mlp_validate_fallback_shards(
    const char *manifest_path, const char *expected_manifest_sha256,
    const char *ane_fc1_path, const void *ane_fc1, size_t ane_fc1_bytes,
    const char *ane_fc2_path, const void *ane_fc2, size_t ane_fc2_bytes,
    char *error, size_t error_size) {
    if (!manifest_path || !*manifest_path ||
        !sha256_hex_valid(expected_manifest_sha256) ||
        !ane_fc1_bytes || !ane_fc2_bytes ||
        ((ane_fc1 == NULL) != (ane_fc2 == NULL))) {
        fail(error, error_size,
             "invalid Core ML fallback shard validation arguments");
        return 0;
    }
    @autoreleasepool {
        NSString *manifest_file =
            [NSString stringWithUTF8String:manifest_path];
        if (!manifest_file ||
            !fallback_sibling_path_matches(
                manifest_file, ane_fc1_path, @"ane_fc1.bf16",
                error, error_size) ||
            !fallback_sibling_path_matches(
                manifest_file, ane_fc2_path, @"ane_fc2.bf16",
                error, error_size)) return 0;
        NSError *detail = nil;
        NSData *encoded = [NSData dataWithContentsOfFile:manifest_file
                                  options:NSDataReadingMappedIfSafe
                                    error:&detail];
        if (!encoded) {
            fail(error, error_size, "cannot read Core ML manifest %s: %s",
                 manifest_path, detail.localizedDescription.UTF8String);
            return 0;
        }
        char actual_manifest[H3_COREML_SHA256_HEX_SIZE];
        sha256_bytes(encoded.bytes, encoded.length, actual_manifest);
        if (strcmp(actual_manifest, expected_manifest_sha256)) {
            fail(error, error_size,
                 "Core ML fallback manifest changed after load");
            return 0;
        }
        id root = [NSJSONSerialization JSONObjectWithData:encoded
                                                   options:0 error:&detail];
        if (![root isKindOfClass:[NSDictionary class]]) {
            fail(error, error_size, "cannot parse Core ML manifest %s: %s",
                 manifest_path, detail.localizedDescription.UTF8String);
            return 0;
        }
        id files_value = ((NSDictionary *)root)[@"files"];
        if (![files_value isKindOfClass:[NSDictionary class]]) {
            fail(error, error_size, "Core ML manifest files are missing");
            return 0;
        }
        NSDictionary *files = files_value;
        int verify = ane_fc1 != NULL;
        if (!fallback_file_exact_size(ane_fc1_path, ane_fc1_bytes,
                                      error, error_size) ||
            !fallback_file_exact_size(ane_fc2_path, ane_fc2_bytes,
                                      error, error_size) ||
            !manifest_file_entry(files, @"ane_fc1.bf16",
                                 ane_fc1, ane_fc1_bytes, verify,
                                 error, error_size) ||
            !manifest_file_entry(files, @"ane_fc2.bf16",
                                 ane_fc2, ane_fc2_bytes, verify,
                                 error, error_size)) return 0;
    }
    return 1;
}

size_t h3_coreml_f16_nonfinite_count(const uint16_t *values,
                                     size_t elements) {
    if (!values) return elements;
    size_t count = 0;
    size_t index = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const uint16x8_t exponent_mask = vdupq_n_u16(UINT16_C(0x7c00));
    uint64x2_t vector_count = vdupq_n_u64(0);
    for (; index + 8 <= elements; index += 8) {
        uint16x8_t bits = vld1q_u16(values + index);
        uint16x8_t flags = vshrq_n_u16(
            vceqq_u16(vandq_u16(bits, exponent_mask), exponent_mask), 15);
        vector_count = vaddq_u64(
            vector_count, vpaddlq_u32(vpaddlq_u16(flags)));
    }
    count = (size_t)vaddvq_u64(vector_count);
#endif
    for (; index < elements; index++)
        if ((values[index] & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) count++;
    return count;
}

int h3_coreml_mlp_fallback_required(const uint16_t *values, size_t elements,
                                    int forced,
                                    size_t *nonfinite_count) {
    size_t count = h3_coreml_f16_nonfinite_count(values, elements);
    if (nonfinite_count) *nonfinite_count = count;
    return forced || count != 0;
}

static NSArray<NSNumber *> *contiguous_strides(NSArray<NSNumber *> *shape) {
    NSMutableArray<NSNumber *> *strides =
        [NSMutableArray arrayWithCapacity:shape.count];
    size_t stride = 1;
    for (NSInteger index = (NSInteger)shape.count - 1; index >= 0; index--) {
        [strides insertObject:@(stride) atIndex:0];
        stride *= shape[(NSUInteger)index].unsignedLongLongValue;
    }
    return strides;
}

static int exact_shape(NSArray<NSNumber *> *shape,
                       uint32_t rows, uint32_t hidden) {
    return shape.count == 4 && shape[0].unsignedIntValue == 1 &&
        shape[1].unsignedIntValue == hidden &&
        shape[2].unsignedIntValue == 1 &&
        shape[3].unsignedIntValue == rows;
}

static NSURL *compiled_model_url(NSURL *source, BOOL *used_cache,
                                 NSError **error) {
    if (used_cache) *used_cache = NO;
    if ([source.pathExtension caseInsensitiveCompare:@"mlmodelc"] ==
        NSOrderedSame) {
        if (used_cache) *used_cache = YES;
        return source;
    }
    if (getenv("H3_COREML_USE_COMPILED")) {
        NSURL *cached = [[source URLByDeletingPathExtension]
            URLByAppendingPathExtension:@"mlmodelc"];
        BOOL is_directory = NO;
        if ([[NSFileManager defaultManager] fileExistsAtPath:cached.path
                                                 isDirectory:&is_directory] &&
            is_directory) {
            if (used_cache) *used_cache = YES;
            return cached;
        }
    }
    return [MLModel compileModelAtURL:source error:error];
}

static int load_jobs_valid = 1;

static dispatch_semaphore_t coreml_load_slots(void) {
    static dispatch_once_t once;
    static dispatch_semaphore_t slots;
    dispatch_once(&once, ^{
        long jobs = 4;
        const char *value = getenv("H3_COREML_LOAD_JOBS");
        if (value && *value) {
            char *tail = NULL;
            long parsed = strtol(value, &tail, 10);
            if (tail == value || *tail || parsed < 1 || parsed > 16)
                load_jobs_valid = 0;
            else
                jobs = parsed;
        }
        slots = dispatch_semaphore_create(jobs);
    });
    return slots;
}

static NSString *load_coreml_object(H3CoreMLMLP *object, NSURL *source,
                                    uint32_t rows, uint32_t input_width,
                                    uint32_t output_width) {
    NSError *detail = nil;
    BOOL used_compiled_cache = NO;
    BOOL profile = getenv("H3_PROFILE") != NULL;
    CFAbsoluteTime compile_started = profile ?
        CFAbsoluteTimeGetCurrent() : 0.0;
    NSURL *compiled = compiled_model_url(
        source, &used_compiled_cache, &detail);
    CFAbsoluteTime compile_seconds = profile ?
        CFAbsoluteTimeGetCurrent() - compile_started : 0.0;
    if (!compiled)
        return [NSString stringWithFormat:@"cannot compile Core ML MLP: %@",
                                          detail.localizedDescription];
    if (profile && used_compiled_cache)
        fprintf(stderr, "h3: Core ML compiled cache %s\n",
                compiled.fileSystemRepresentation);
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLOptimizationHints *hints = [MLOptimizationHints new];
    hints.reshapeFrequency = getenv("H3_COREML_RESHAPE_FREQUENT") ?
        MLReshapeFrequencyHintFrequent :
        MLReshapeFrequencyHintInfrequent;
    hints.specializationStrategy = getenv("H3_COREML_SPECIALIZE_FAST") ?
        MLSpecializationStrategyFastPrediction :
        MLSpecializationStrategyDefault;
    configuration.optimizationHints = hints;
    CFAbsoluteTime load_started = profile ?
        CFAbsoluteTimeGetCurrent() : 0.0;
    object->model = [MLModel modelWithContentsOfURL:compiled
                                      configuration:configuration
                                              error:&detail];
    CFAbsoluteTime load_seconds = profile ?
        CFAbsoluteTimeGetCurrent() - load_started : 0.0;
    if (!object->model)
        return [NSString stringWithFormat:@"cannot load Core ML MLP: %@",
                                          detail.localizedDescription];
    if (profile)
        fprintf(stderr,
                "h3: Core ML model load source=%s compiled-cache=%d "
                "compile=%.6fs load=%.6fs\n",
                source.fileSystemRepresentation, used_compiled_cache ? 1 : 0,
                (double)compile_seconds, (double)load_seconds);
    if (object->manifestSHA256) {
        id creator = object->model.modelDescription.metadata[
            MLModelCreatorDefinedKey];
        NSString *actual = [creator isKindOfClass:[NSDictionary class]] ?
            ((NSDictionary *)creator)[@"h3_manifest_sha256"] : nil;
        NSString *schema = [creator isKindOfClass:[NSDictionary class]] ?
            ((NSDictionary *)creator)[@"h3_manifest_schema"] : nil;
        if (![actual isKindOfClass:[NSString class]] ||
            ![schema isEqualToString:@"h3-coreml-mlp-v1"] ||
            ![actual isEqualToString:object->manifestSHA256])
            return [NSString stringWithFormat:
                @"Core ML package manifest identity mismatch: expected %@",
                object->manifestSHA256];
    }
    NSDictionary<NSString *, MLFeatureDescription *> *inputs =
        object->model.modelDescription.inputDescriptionsByName;
    NSDictionary<NSString *, MLFeatureDescription *> *outputs =
        object->model.modelDescription.outputDescriptionsByName;
    NSString *input_name = inputs.allKeys.firstObject;
    object->outputName = outputs.allKeys.firstObject;
    MLMultiArrayConstraint *input_constraint =
        inputs[input_name].multiArrayConstraint;
    MLMultiArrayConstraint *output_constraint =
        outputs[object->outputName].multiArrayConstraint;
    if (!input_name || !object->outputName || !input_constraint ||
        !output_constraint ||
        input_constraint.dataType != MLMultiArrayDataTypeFloat16 ||
        output_constraint.dataType != MLMultiArrayDataTypeFloat16 ||
        !exact_shape(input_constraint.shape, rows, input_width) ||
        !exact_shape(output_constraint.shape, rows, output_width))
        return [NSString stringWithFormat:
            @"Core ML program must have FP16 [1,%u,1,%u] input and "
             "[1,%u,1,%u] output",
            input_width, rows, output_width, rows];
    MLFeatureValue *input_value =
        [MLFeatureValue featureValueWithMultiArray:object->input];
    object->provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{input_name: input_value} error:&detail];
    object->options = [MLPredictionOptions new];
    object->options.outputBackings = @{
        object->outputName: object->output
    };
    if (!object->provider || !object->options)
        return [NSString stringWithFormat:
            @"cannot create Core ML MLP provider: %@",
            detail.localizedDescription];
    return nil;
}

static h3_coreml_mlp *h3_coreml_mlp_create_async_impl(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t input_width,
                                    uint32_t output_width,
                                    h3_gpu_tensor *shared_input,
                                    h3_gpu_tensor *shared_output,
                                    const char *manifest_sha256,
                                    char *error, size_t error_size) {
    if (!gpu || !model_path || !*model_path || !rows || !input_width ||
        !output_width) {
        fail(error, error_size, "invalid Core ML MLP arguments");
        return NULL;
    }
    if (manifest_sha256 && !sha256_hex_valid(manifest_sha256)) {
        fail(error, error_size,
             "Core ML manifest SHA-256 must be 64 lowercase hex digits");
        return NULL;
    }
    dispatch_semaphore_t slots = coreml_load_slots();
    if (!load_jobs_valid) {
        fail(error, error_size, "H3_COREML_LOAD_JOBS must be in [1, 16]");
        return NULL;
    }
    @autoreleasepool {
        H3CoreMLMLP *object = [H3CoreMLMLP new];
        if (manifest_sha256)
            object->manifestSHA256 = [NSString
                stringWithUTF8String:manifest_sha256];
        NSURL *source = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:model_path]];
        NSError *detail = nil;
        size_t input_elements = (size_t)rows * input_width;
        size_t output_elements = (size_t)rows * output_width;
        if ((shared_input == NULL) != (shared_output == NULL)) {
            fail(error, error_size,
                 "Core ML shared input/output must be supplied together");
            return NULL;
        }
        object->ownsTensors = shared_input == NULL;
        object->inputTensor = shared_input ? shared_input :
            h3_gpu_tensor_new_f16(gpu, input_elements);
        object->outputTensor = shared_output ? shared_output :
            h3_gpu_tensor_new_f16(gpu, output_elements);
        if (!object->inputTensor || !object->outputTensor) {
            fail(error, error_size, "cannot allocate Core ML shared tensors: %s",
                 h3_gpu_error(gpu));
            return NULL;
        }
        if (object->ownsTensors)
            memset(h3_gpu_tensor_host_pointer(object->inputTensor), 0,
                   input_elements * sizeof(uint16_t));
        NSArray<NSNumber *> *input_shape =
            @[@1, @(input_width), @1, @(rows)];
        NSArray<NSNumber *> *output_shape =
            @[@1, @(output_width), @1, @(rows)];
        NSArray<NSNumber *> *strides = contiguous_strides(input_shape);
        object->input = [[MLMultiArray alloc]
            initWithDataPointer:h3_gpu_tensor_host_pointer(object->inputTensor)
                          shape:input_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:strides deallocator:nil error:&detail];
        object->output = [[MLMultiArray alloc]
            initWithDataPointer:h3_gpu_tensor_host_pointer(object->outputTensor)
                          shape:output_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:contiguous_strides(output_shape)
                    deallocator:nil error:&detail];
        if (!object->input || !object->output) {
            fail(error, error_size, "cannot wrap Core ML shared tensors: %s",
                 detail.localizedDescription.UTF8String);
            return NULL;
        }
        object->completion = dispatch_semaphore_create(0);
        object->loadCompletion = dispatch_semaphore_create(0);
        if (!object->completion || !object->loadCompletion) {
            fail(error, error_size, "cannot create Core ML MLP semaphores");
            return NULL;
        }
        dispatch_async(dispatch_get_global_queue(
                           QOS_CLASS_USER_INITIATED, 0), ^{
            @autoreleasepool {
                dispatch_semaphore_wait(slots, DISPATCH_TIME_FOREVER);
                object->loadFailure = load_coreml_object(
                    object, source, rows, input_width, output_width);
                object->loaded = object->loadFailure == nil;
                dispatch_semaphore_signal(slots);
                dispatch_semaphore_signal(object->loadCompletion);
            }
        });
        return (__bridge_retained h3_coreml_mlp *)object;
    }
}

h3_coreml_mlp *h3_coreml_mlp_create_async(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    char *error, size_t error_size) {
    return h3_coreml_mlp_create_async_impl(
        gpu, model_path, rows, hidden, hidden, NULL, NULL, NULL,
        error, error_size);
}

h3_coreml_mlp *h3_coreml_mlp_create_async_shared(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    h3_gpu_tensor *shared_input,
                                    h3_gpu_tensor *shared_output,
                                    char *error, size_t error_size) {
    return h3_coreml_mlp_create_async_impl(
        gpu, model_path, rows, hidden, hidden,
        shared_input, shared_output, NULL,
        error, error_size);
}

h3_coreml_mlp *h3_coreml_mlp_create_async_shared_io(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t input_width,
                                    uint32_t output_width,
                                    h3_gpu_tensor *shared_input,
                                    h3_gpu_tensor *shared_output,
                                    char *error, size_t error_size) {
    return h3_coreml_mlp_create_async_impl(
        gpu, model_path, rows, input_width, output_width,
        shared_input, shared_output, NULL, error, error_size);
}

h3_coreml_mlp *h3_coreml_mlp_create_async_shared_verified(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    h3_gpu_tensor *shared_input,
                                    h3_gpu_tensor *shared_output,
                                    const char *manifest_sha256,
                                    char *error, size_t error_size) {
    return h3_coreml_mlp_create_async_impl(
        gpu, model_path, rows, hidden, hidden, shared_input, shared_output,
        manifest_sha256, error, error_size);
}

int h3_coreml_mlp_finish_loading(h3_coreml_mlp *opaque,
                                 char *error, size_t error_size) {
    if (!opaque) {
        fail(error, error_size, "Core ML MLP is absent");
        return 0;
    }
    H3CoreMLMLP *object = MLP(opaque);
    if (!object->loadFinished) {
        dispatch_semaphore_wait(object->loadCompletion,
                                DISPATCH_TIME_FOREVER);
        object->loadFinished = YES;
    }
    if (!object->loaded) {
        const char *detail = object->loadFailure.UTF8String;
        fail(error, error_size, "%s",
             detail ? detail : "Core ML MLP load failed");
        return 0;
    }
    return 1;
}

int h3_coreml_mlp_warmup(h3_coreml_mlp *opaque, unsigned iterations,
                         char *error, size_t error_size) {
    if (!opaque) {
        fail(error, error_size, "Core ML MLP is absent");
        return 0;
    }
    if (iterations > 100) {
        fail(error, error_size, "Core ML MLP warmups must be in [0, 100]");
        return 0;
    }
    if (!h3_coreml_mlp_finish_loading(opaque, error, error_size)) return 0;
    H3CoreMLMLP *object = MLP(opaque);
    @autoreleasepool {
        for (unsigned index = 0; index < iterations; index++) {
            NSError *detail = nil;
            id<MLFeatureProvider> prediction =
                [object->model predictionFromFeatures:object->provider
                                               options:object->options
                                                 error:&detail];
            MLMultiArray *actual = [prediction
                featureValueForName:object->outputName].multiArrayValue;
            if (!prediction || detail || actual != object->output) {
                fail(error, error_size, "Core ML MLP warmup failed: %s",
                     detail.localizedDescription.UTF8String);
                return 0;
            }
        }
    }
    return 1;
}

h3_coreml_mlp *h3_coreml_mlp_create(h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    char *error, size_t error_size) {
    h3_coreml_mlp *mlp = h3_coreml_mlp_create_async(
        gpu, model_path, rows, hidden, error, error_size);
    if (mlp && !h3_coreml_mlp_finish_loading(mlp, error, error_size)) {
        h3_coreml_mlp_free(mlp);
        return NULL;
    }
    return mlp;
}

void h3_coreml_mlp_free(h3_coreml_mlp *opaque) {
    if (!opaque) return;
    (void)h3_coreml_mlp_finish_loading(opaque, NULL, 0);
    if (MLP(opaque)->inflight)
        (void)h3_coreml_mlp_wait(opaque, NULL, 0);
    @autoreleasepool {
        H3CoreMLMLP *object = CFBridgingRelease(opaque);
        (void)object;
    }
}

h3_gpu_tensor *h3_coreml_mlp_input(h3_coreml_mlp *opaque) {
    return opaque ? MLP(opaque)->inputTensor : NULL;
}

const h3_gpu_tensor *h3_coreml_mlp_output(const h3_coreml_mlp *opaque) {
    return opaque ? MLP((h3_coreml_mlp *)(void *)opaque)->outputTensor : NULL;
}

int h3_coreml_mlp_prediction_inflight(const h3_coreml_mlp *opaque) {
    return opaque && MLP((h3_coreml_mlp *)(void *)opaque)->inflight;
}

int h3_coreml_mlp_start(h3_coreml_mlp *opaque,
                        char *error, size_t error_size) {
    if (!opaque) {
        fail(error, error_size, "Core ML MLP is absent");
        return 0;
    }
    if (!h3_coreml_mlp_finish_loading(opaque, error, error_size)) return 0;
    H3CoreMLMLP *object = MLP(opaque);
    if (object->inflight) {
        fail(error, error_size, "Core ML MLP prediction is already in flight");
        return 0;
    }
    object->asyncError = nil;
    object->prediction = nil;
    object->inflight = YES;
    H3CoreMLMLP *strong_object = object;
    [object->model predictionFromFeatures:object->provider
                                  options:object->options
                        completionHandler:^(id<MLFeatureProvider> value,
                                            NSError *detail) {
        strong_object->prediction = value;
        strong_object->asyncError = detail;
        dispatch_semaphore_signal(strong_object->completion);
    }];
    return 1;
}

int h3_coreml_mlp_wait(h3_coreml_mlp *opaque,
                       char *error, size_t error_size) {
    if (!opaque) {
        fail(error, error_size, "Core ML MLP is absent");
        return 0;
    }
    H3CoreMLMLP *object = MLP(opaque);
    if (!object->inflight) {
        fail(error, error_size, "Core ML MLP prediction was not started");
        return 0;
    }
    dispatch_semaphore_wait(object->completion, DISPATCH_TIME_FOREVER);
    object->inflight = NO;
    MLMultiArray *actual = [object->prediction
        featureValueForName:object->outputName].multiArrayValue;
    if (!object->prediction || object->asyncError || actual != object->output) {
        const char *detail = object->asyncError.localizedDescription.UTF8String;
        if (!detail) detail = actual != object->output ?
            "output backing mismatch" : "prediction returned no output";
        fail(error, error_size, "Core ML MLP prediction failed: %s", detail);
        return 0;
    }
    return 1;
}
