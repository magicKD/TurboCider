#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_attention.h"
#include "convrot.h"
#include "../../../models/relayvsr/flash.h"

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <torch/mps.h>
#include <array>
#include <cstdio>

namespace rv {
namespace {

constexpr int query_rows = 4, key_rows = 64, simdgroups = 8;
constexpr int threads = simdgroups * 32;

NSString* const source =
#include "flash_tensorops.inc"
    ;

struct Kernel {
  id<MTLComputePipelineState> pipeline = nil;

  Kernel() {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.4, *)) {
      @autoreleasepool {
        if (!tensorops_hardware_available()) return;
        auto device = at::mps::getCurrentMPSStream()->device();
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        options.preprocessorMacros = @{
          @"FLASH_QUERY_ROWS" : @(query_rows),
          @"FLASH_KEY_ROWS" : @(key_rows),
          @"FLASH_SIMDGROUPS" : @(simdgroups)
        };
        NSError* error = nil;
        auto library = [device newLibraryWithSource:source options:options error:&error];
        if (library) {
          auto function = [library newFunctionWithName:@"flash_tensorops_window"];
          if (function)
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        }
        if (!pipeline || pipeline.threadExecutionWidth != 32 ||
            pipeline.maxTotalThreadsPerThreadgroup < threads) {
          pipeline = nil;
          std::fprintf(stderr, "RelayVSR Flash TensorOps initialization failed: %s\n",
                       error.localizedDescription.UTF8String ?: "unsupported pipeline resources");
        }
      }
    }
#endif
  }
};

const Kernel& kernel() {
  static const Kernel value;
  return value;
}

struct Geometry {
  uint32_t h, w, planes, query_heads, kv_heads, condition;
};

void bind(id<MTLComputeCommandEncoder> enc, const at::Tensor& tensor, NSUInteger index) {
  auto storage = __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
  [enc setBuffer:storage offset:tensor.storage_offset() * tensor.element_size() atIndex:index];
}

Tensor aligned(const Tensor& value) {
  auto tensor = value.contiguous();
  return tensor.storage_offset() % 4 == 0 ? tensor : tensor.clone();
}

}  // namespace

bool flash_tensorops_available() { return torch::mps::is_available() && kernel().pipeline; }

Tensor tensorops_window_attention(const Tensor& query, std::span<const FlashKV> planes, int height,
                                  int width, bool condition) {
  TORCH_CHECK(height > 0 && width > 0 && query.is_mps() && query.scalar_type() == at::kBFloat16 &&
                  query.dim() == 4 && query.size(2) == int64_t(height) * width &&
                  query.size(3) == 64 && !planes.empty() && planes.size() <= 4 && kernel().pipeline,
              "Unsupported Flash TensorOps attention");
  const auto& key = planes.front().key;
  TORCH_CHECK(key.dim() == 4 && key.size(1) > 0 && query.size(1) % key.size(1) == 0 &&
                  key.size(0) == query.size(0) && key.size(2) == query.size(2) && key.size(3) == 64,
              "Invalid Flash key geometry");

  std::array<Tensor, 4> keys, values;
  for (size_t i = 0; i < planes.size(); ++i) {
    const auto& kv = planes[i];
    TORCH_CHECK(kv.key.sizes() == key.sizes() && kv.value.sizes() == key.sizes() &&
                    kv.key.device() == query.device() && kv.value.device() == query.device() &&
                    kv.key.scalar_type() == query.scalar_type() &&
                    kv.value.scalar_type() == query.scalar_type(),
                "Invalid Flash K/V planes");
    keys[i] = aligned(kv.key);
    values[i] = aligned(kv.value);
  }
  // Metal still needs buffers bound for the unused plane arguments.
  for (size_t i = planes.size(); i < keys.size(); ++i) {
    keys[i] = keys.front();
    values[i] = values.front();
  }

  const auto q = aligned(query);
  auto out = at::empty_like(q);
  const Geometry geometry{uint32_t(height),    uint32_t(width),       uint32_t(planes.size()),
                          uint32_t(q.size(1)), uint32_t(key.size(1)), uint32_t(condition)};
  auto* stream = at::mps::getCurrentMPSStream();
  const auto query_tiles = ((width + 7) / 8) * ((height + query_rows - 1) / query_rows);
  auto pipeline = kernel().pipeline;
  dispatch_sync(stream->queue(), ^{
    auto enc = stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    bind(enc, q, 0);
    bind(enc, out, 1);
    [enc setBytes:&geometry length:sizeof(geometry) atIndex:2];
    for (NSUInteger i = 0; i < keys.size(); ++i) {
      bind(enc, keys[i], 3 + i);
      bind(enc, values[i], 7 + i);
    }
    [enc dispatchThreadgroups:MTLSizeMake(query_tiles, q.size(1), q.size(0))
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  });
  return out;
}

}  // namespace rv
