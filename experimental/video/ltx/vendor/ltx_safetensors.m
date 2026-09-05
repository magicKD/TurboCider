#include "ltx_safetensors.h"

#import <Foundation/Foundation.h>

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#define LTX_ST_MAX_HEADER (256u * 1024u * 1024u)

static int ltx_st_fail(char *error, size_t error_size,
                       const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static uint64_t ltx_u64_le(const unsigned char bytes[8]) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++)
        value |= (uint64_t)bytes[index] << (index * 8u);
    return value;
}

static int ltx_pread_exact(int descriptor, void *data, size_t bytes,
                           uint64_t offset, char *error, size_t error_size) {
    unsigned char *destination = data;
    size_t completed = 0;
    while (completed < bytes) {
        uint64_t current = offset + completed;
        if (current > (uint64_t)INT64_MAX)
            return ltx_st_fail(error, error_size, "file offset exceeds off_t");
        ssize_t count = pread(descriptor, destination + completed,
                              bytes - completed, (off_t)current);
        if (count < 0) {
            if (errno == EINTR) continue;
            return ltx_st_fail(error, error_size, "pread failed: %s",
                               strerror(errno));
        }
        if (!count)
            return ltx_st_fail(error, error_size, "unexpected end of file");
        completed += (size_t)count;
    }
    return 1;
}

static ltx_dtype ltx_parse_dtype(NSString *name) {
    if ([name isEqualToString:@"BOOL"]) return LTX_DTYPE_BOOL;
    if ([name isEqualToString:@"I8"]) return LTX_DTYPE_I8;
    if ([name isEqualToString:@"U8"]) return LTX_DTYPE_U8;
    if ([name isEqualToString:@"I16"]) return LTX_DTYPE_I16;
    if ([name isEqualToString:@"U16"]) return LTX_DTYPE_U16;
    if ([name isEqualToString:@"F16"]) return LTX_DTYPE_F16;
    if ([name isEqualToString:@"BF16"]) return LTX_DTYPE_BF16;
    if ([name isEqualToString:@"I32"]) return LTX_DTYPE_I32;
    if ([name isEqualToString:@"U32"]) return LTX_DTYPE_U32;
    if ([name isEqualToString:@"F32"]) return LTX_DTYPE_F32;
    if ([name isEqualToString:@"I64"]) return LTX_DTYPE_I64;
    if ([name isEqualToString:@"U64"]) return LTX_DTYPE_U64;
    if ([name isEqualToString:@"F64"]) return LTX_DTYPE_F64;
    return LTX_DTYPE_UNKNOWN;
}

size_t ltx_dtype_size(ltx_dtype dtype) {
    switch (dtype) {
        case LTX_DTYPE_BOOL:
        case LTX_DTYPE_I8:
        case LTX_DTYPE_U8: return 1;
        case LTX_DTYPE_I16:
        case LTX_DTYPE_U16:
        case LTX_DTYPE_F16:
        case LTX_DTYPE_BF16: return 2;
        case LTX_DTYPE_I32:
        case LTX_DTYPE_U32:
        case LTX_DTYPE_F32: return 4;
        case LTX_DTYPE_I64:
        case LTX_DTYPE_U64:
        case LTX_DTYPE_F64: return 8;
        case LTX_DTYPE_UNKNOWN: return 0;
    }
    return 0;
}

const char *ltx_dtype_name(ltx_dtype dtype) {
    switch (dtype) {
        case LTX_DTYPE_BOOL: return "BOOL";
        case LTX_DTYPE_I8: return "I8";
        case LTX_DTYPE_U8: return "U8";
        case LTX_DTYPE_I16: return "I16";
        case LTX_DTYPE_U16: return "U16";
        case LTX_DTYPE_F16: return "F16";
        case LTX_DTYPE_BF16: return "BF16";
        case LTX_DTYPE_I32: return "I32";
        case LTX_DTYPE_U32: return "U32";
        case LTX_DTYPE_F32: return "F32";
        case LTX_DTYPE_I64: return "I64";
        case LTX_DTYPE_U64: return "U64";
        case LTX_DTYPE_F64: return "F64";
        case LTX_DTYPE_UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

uint64_t ltx_st_tensor_elements(const ltx_st_tensor *tensor) {
    if (!tensor) return 0;
    uint64_t elements = 1;
    for (uint32_t index = 0; index < tensor->ndim; index++) {
        if (tensor->shape[index] &&
            elements > UINT64_MAX / tensor->shape[index]) return 0;
        elements *= tensor->shape[index];
    }
    return elements;
}

void ltx_st_free_header(ltx_st_header *header) {
    if (!header) return;
    for (size_t index = 0; index < header->tensor_count; index++)
        free(header->tensors[index].name);
    free(header->tensors);
    free(header->metadata_config);
    free(header->path);
    memset(header, 0, sizeof(*header));
}

int ltx_st_read_header(const char *path, ltx_st_header *header,
                       char *error, size_t error_size) {
    if (!path || !header)
        return ltx_st_fail(error, error_size, "missing safetensors path/header");
    memset(header, 0, sizeof(*header));

    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0)
        return ltx_st_fail(error, error_size, "open %s: %s", path,
                           strerror(errno));
    struct stat status;
    if (fstat(descriptor, &status) != 0 || status.st_size < 8) {
        int saved = errno;
        close(descriptor);
        return ltx_st_fail(error, error_size, "invalid safetensors file: %s",
                           saved ? strerror(saved) : "too short");
    }

    unsigned char prefix[8];
    if (!ltx_pread_exact(descriptor, prefix, sizeof(prefix), 0,
                         error, error_size)) {
        close(descriptor);
        return 0;
    }
    uint64_t header_size = ltx_u64_le(prefix);
    uint64_t file_size = (uint64_t)status.st_size;
    if (!header_size || header_size > LTX_ST_MAX_HEADER ||
        header_size > file_size - 8u) {
        close(descriptor);
        return ltx_st_fail(error, error_size,
                           "invalid safetensors header size: %llu",
                           (unsigned long long)header_size);
    }

    void *header_bytes = malloc((size_t)header_size);
    if (!header_bytes) {
        close(descriptor);
        return ltx_st_fail(error, error_size,
                           "out of memory reading safetensors header");
    }
    int read_ok = ltx_pread_exact(descriptor, header_bytes,
                                  (size_t)header_size, 8u,
                                  error, error_size);
    close(descriptor);
    if (!read_ok) {
        free(header_bytes);
        return 0;
    }

    int ok = 0;
    @autoreleasepool {
        NSData *data = [NSData dataWithBytesNoCopy:header_bytes
                                            length:(NSUInteger)header_size
                                      freeWhenDone:NO];
        NSError *json_error = nil;
        id value = [NSJSONSerialization JSONObjectWithData:data
                                                   options:0
                                                     error:&json_error];
        if (![value isKindOfClass:[NSDictionary class]]) {
            const char *description = json_error ?
                json_error.localizedDescription.UTF8String : NULL;
            ltx_st_fail(error, error_size, "invalid safetensors JSON: %s",
                description ? description : "root is not an object");
        } else {
            NSDictionary *root = (NSDictionary *)value;
            NSArray *keys = [[root allKeys]
                sortedArrayUsingSelector:@selector(compare:)];
            size_t tensor_count = 0;
            for (id key in keys)
                if ([key isKindOfClass:[NSString class]] &&
                    ![(NSString *)key isEqualToString:@"__metadata__"])
                    tensor_count++;

            ltx_st_tensor *tensors = calloc(tensor_count, sizeof(*tensors));
            if (!tensors && tensor_count) {
                ltx_st_fail(error, error_size,
                            "out of memory indexing safetensors tensors");
            } else {
                size_t output_index = 0;
                uint64_t data_start = 8u + header_size;
                ok = 1;
                for (id raw_key in keys) {
                    if (![raw_key isKindOfClass:[NSString class]]) continue;
                    NSString *key = (NSString *)raw_key;
                    if ([key isEqualToString:@"__metadata__"]) continue;
                    id raw_entry = root[key];
                    if (![raw_entry isKindOfClass:[NSDictionary class]]) {
                        ok = ltx_st_fail(error, error_size,
                                         "tensor %s is not an object",
                                         key.UTF8String);
                        break;
                    }
                    NSDictionary *entry = (NSDictionary *)raw_entry;
                    NSString *dtype_name = entry[@"dtype"];
                    NSArray *shape = entry[@"shape"];
                    NSArray *offsets = entry[@"data_offsets"];
                    if (![dtype_name isKindOfClass:[NSString class]] ||
                        ![shape isKindOfClass:[NSArray class]] ||
                        ![offsets isKindOfClass:[NSArray class]] ||
                        shape.count > 8u || offsets.count != 2u) {
                        ok = ltx_st_fail(error, error_size,
                                         "invalid tensor descriptor for %s",
                                         key.UTF8String);
                        break;
                    }

                    ltx_st_tensor *tensor = &tensors[output_index];
                    tensor->name = strdup(key.UTF8String);
                    tensor->dtype = ltx_parse_dtype(dtype_name);
                    tensor->ndim = (uint32_t)shape.count;
                    if (!tensor->name || tensor->dtype == LTX_DTYPE_UNKNOWN) {
                        ok = ltx_st_fail(error, error_size,
                                         "unsupported tensor name/dtype for %s",
                                         key.UTF8String);
                        break;
                    }
                    for (NSUInteger dimension = 0; dimension < shape.count;
                         dimension++) {
                        id number = shape[dimension];
                        if (![number isKindOfClass:[NSNumber class]]) {
                            ok = ltx_st_fail(error, error_size,
                                             "non-numeric shape for %s",
                                             key.UTF8String);
                            break;
                        }
                        tensor->shape[dimension] =
                            [(NSNumber *)number unsignedLongLongValue];
                    }
                    if (!ok) break;
                    tensor->data_begin =
                        [(NSNumber *)offsets[0] unsignedLongLongValue];
                    tensor->data_end =
                        [(NSNumber *)offsets[1] unsignedLongLongValue];
                    if (tensor->data_end < tensor->data_begin ||
                        tensor->data_end > file_size - data_start) {
                        ok = ltx_st_fail(error, error_size,
                                         "tensor offsets exceed file for %s",
                                         key.UTF8String);
                        break;
                    }
                    uint64_t elements = ltx_st_tensor_elements(tensor);
                    size_t item_size = ltx_dtype_size(tensor->dtype);
                    if (item_size && elements > UINT64_MAX / item_size) {
                        ok = ltx_st_fail(error, error_size,
                                         "tensor size overflow for %s",
                                         key.UTF8String);
                        break;
                    }
                    uint64_t expected = elements * item_size;
                    if (expected != tensor->data_end - tensor->data_begin) {
                        ok = ltx_st_fail(error, error_size,
                                         "shape/offset size mismatch for %s",
                                         key.UTF8String);
                        break;
                    }
                    tensor->file_offset = data_start + tensor->data_begin;
                    output_index++;
                }

                if (ok) {
                    NSDictionary *metadata = root[@"__metadata__"];
                    NSString *config = [metadata isKindOfClass:[NSDictionary class]] ?
                        metadata[@"config"] : nil;
                    header->path = strdup(path);
                    header->file_size = file_size;
                    header->header_size = header_size;
                    header->tensors = tensors;
                    header->tensor_count = tensor_count;
                    if ([config isKindOfClass:[NSString class]])
                        header->metadata_config = strdup(config.UTF8String);
                    if (!header->path ||
                        ([config isKindOfClass:[NSString class]] &&
                         !header->metadata_config)) {
                        ok = ltx_st_fail(error, error_size,
                                         "out of memory retaining metadata");
                    }
                }
                if (!ok) {
                    for (size_t index = 0; index < tensor_count; index++)
                        free(tensors[index].name);
                    free(tensors);
                }
            }
        }
    }
    free(header_bytes);
    if (!ok) ltx_st_free_header(header);
    return ok;
}

const ltx_st_tensor *ltx_st_find(const ltx_st_header *header,
                                 const char *name) {
    if (!header || !name) return NULL;
    for (size_t index = 0; index < header->tensor_count; index++)
        if (!strcmp(header->tensors[index].name, name))
            return &header->tensors[index];
    return NULL;
}

int ltx_st_read_data(const ltx_st_header *header,
                     const ltx_st_tensor *tensor, void *data, size_t bytes,
                     char *error, size_t error_size) {
    if (!header || !tensor || (!data && bytes))
        return ltx_st_fail(error, error_size, "missing tensor read argument");
    uint64_t expected = tensor->data_end - tensor->data_begin;
    if (expected != bytes)
        return ltx_st_fail(error, error_size,
                           "tensor read size mismatch: expected %llu, got %zu",
                           (unsigned long long)expected, bytes);
    int descriptor = open(header->path, O_RDONLY);
    if (descriptor < 0)
        return ltx_st_fail(error, error_size, "open %s: %s", header->path,
                           strerror(errno));
    int ok = ltx_pread_exact(descriptor, data, bytes, tensor->file_offset,
                             error, error_size);
    close(descriptor);
    return ok;
}

int ltx_st_map_open(const ltx_st_header *header, ltx_st_mapping *mapping,
                    char *error, size_t error_size) {
    if (!header || !header->path || !mapping)
        return ltx_st_fail(error, error_size,
                           "missing safetensors mapping argument");
    memset(mapping, 0, sizeof(*mapping));
    if (!header->file_size || header->file_size > SIZE_MAX)
        return ltx_st_fail(error, error_size,
                           "safetensors file is too large to map");
    int descriptor = open(header->path, O_RDONLY);
    if (descriptor < 0)
        return ltx_st_fail(error, error_size, "open %s: %s", header->path,
                           strerror(errno));
    size_t bytes = (size_t)header->file_size;
    void *address = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
    int saved = errno;
    close(descriptor);
    if (address == MAP_FAILED)
        return ltx_st_fail(error, error_size, "mmap %s: %s", header->path,
                           strerror(saved));
    mapping->address = address;
    mapping->bytes = bytes;
    return 1;
}

int ltx_st_map_discard(const ltx_st_mapping *mapping,
                       char *error, size_t error_size) {
    if (!mapping || !mapping->address || !mapping->bytes)
        return ltx_st_fail(error, error_size,
                           "invalid safetensors mapping discard");
    if (madvise(mapping->address, mapping->bytes, MADV_DONTNEED) != 0)
        return ltx_st_fail(error, error_size, "madvise safetensors: %s",
                           strerror(errno));
    return 1;
}

void ltx_st_map_close(ltx_st_mapping *mapping) {
    if (!mapping) return;
    if (mapping->address && mapping->bytes)
        munmap(mapping->address, mapping->bytes);
    memset(mapping, 0, sizeof(*mapping));
}

const void *ltx_st_map_tensor(const ltx_st_mapping *mapping,
                              const ltx_st_tensor *tensor,
                              size_t *bytes,
                              char *error, size_t error_size) {
    if (bytes) *bytes = 0;
    if (!mapping || !mapping->address || !tensor) {
        ltx_st_fail(error, error_size, "missing mapped tensor argument");
        return NULL;
    }
    uint64_t tensor_bytes = tensor->data_end - tensor->data_begin;
    if (tensor->file_offset > SIZE_MAX || tensor_bytes > SIZE_MAX ||
        (size_t)tensor->file_offset > mapping->bytes ||
        (size_t)tensor_bytes > mapping->bytes - (size_t)tensor->file_offset) {
        ltx_st_fail(error, error_size,
                    "mapped tensor range exceeds safetensors file");
        return NULL;
    }
    if (bytes) *bytes = (size_t)tensor_bytes;
    return (const unsigned char *)mapping->address +
        (size_t)tensor->file_offset;
}
