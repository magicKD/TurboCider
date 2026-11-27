#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "convrot.h"

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <torch/mps.h>
#include <array>
#include <cstdio>

namespace rv {
namespace {

NSString* const rotation_source =
#include "convrot_input.inc"
    ;

NSString* const gemm_source =
#include "convrot_gemm.inc"
    ;

NSString* const projections = @R"metal(
#define PROJECTION(NAME, BM, BIAS, GELU) \
kernel void NAME(device const int8_t* a [[buffer(0)]], device const int8_t* w [[buffer(1)]], \
                 device bfloat* out [[buffer(2)]], constant uint& m [[buffer(3)]], \
                 constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]], \
                 device const float* a_scales [[buffer(6)]], device const float* w_scales [[buffer(7)]], \
                 constant uint& swizzle [[buffer(8)]], constant uint& tiles_m [[buffer(9)]], \
                 constant uint& tiles_n [[buffer(10)]], device const bfloat* bias [[buffer(11)]], \
                 uint2 tile [[threadgroup_position_in_grid]], uint group [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
  w8a8_gemm_impl<BM,128,512,32,BM/32,4,bfloat,BIAS,BIAS,GELU>(a,w,out,m,n,k,a_scales,w_scales, \
                                                             bias,swizzle,tiles_m,tiles_n,tile,group,lane); \
}
PROJECTION(relay_w8a8, 128, true, false)
PROJECTION(relay_w8a8_small, 32, true, false)
PROJECTION(relay_w8a8_gelu, 128, true, true)
PROJECTION(relay_w8a8_gelu_small, 32, true, true)
)metal";

struct Pipelines {
  id<MTLComputePipelineState> rotation = nil;
  std::array<id<MTLComputePipelineState>, 4> matmul{};
  NSUInteger scratch_limit = 0;

  Pipelines() {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.4, *)) {
      @autoreleasepool {
        if (!tensorops_hardware_available()) return;
        auto device = at::mps::getCurrentMPSStream()->device();
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        auto input = [device newLibraryWithSource:rotation_source options:options error:&error];
        if (input) {
          rotation = pipeline(device, input, @"relay_rotate_quantize", &error);
        }
        if (rotation && rotation.threadExecutionWidth == 32 &&
            rotation.maxTotalThreadsPerThreadgroup >= 256) {
          auto library =
              [device newLibraryWithSource:[gemm_source stringByAppendingString:projections]
                                   options:options
                                     error:&error];
          if (library) {
            NSArray<NSString*>* names = @[
              @"relay_w8a8", @"relay_w8a8_small", @"relay_w8a8_gelu", @"relay_w8a8_gelu_small"
            ];
            for (NSUInteger i = 0; i < names.count; ++i) {
              matmul[i] = pipeline(device, library, names[i], &error);
              const NSUInteger threads = i % 2 == 0 ? 512 : 128;
              if (!matmul[i] || matmul[i].threadExecutionWidth != 32 ||
                  matmul[i].maxTotalThreadsPerThreadgroup < threads)
                break;
              if (i + 1 == names.count &&
                  device.maxThreadgroupMemoryLength > rotation.staticThreadgroupMemoryLength) {
                scratch_limit =
                    device.maxThreadgroupMemoryLength - rotation.staticThreadgroupMemoryLength;
                return;
              }
            }
          }
        }
        std::fprintf(stderr, "RelayVSR TensorOps initialization failed: %s\n",
                     error.localizedDescription.UTF8String ?: "unsupported pipeline resources");
      }
    }
#endif
  }

  static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> lib,
                                              NSString* name, NSError** error) {
    auto function = [lib newFunctionWithName:name];
    return function ? [device newComputePipelineStateWithFunction:function error:error] : nil;
  }
};

const Pipelines& pipelines() {
  static const Pipelines value;
  return value;
}

void bind(id<MTLComputeCommandEncoder> enc, const at::Tensor& value, NSUInteger index) {
  auto storage = __builtin_bit_cast(id<MTLBuffer>, value.storage().data());
  [enc setBuffer:storage offset:value.storage_offset() * value.element_size() atIndex:index];
}

uint32_t tile_height(int64_t rows) { return rows <= 64 ? 32 : 128; }

}  // namespace

bool tensorops_hardware_available() {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (@available(macOS 26.4, *)) {
    if (torch::mps::is_available()) {
      auto device = at::mps::getCurrentMPSStream()->device();
      return [device supportsFamily:MTLGPUFamilyApple10];
    }
  }
#endif
  return false;
}

bool tensorops_int8_available() {
  return torch::mps::is_available() && pipelines().scratch_limit >= 14336 * sizeof(uint16_t);
}

bool convrot_int8_fits(int64_t rows, int64_t columns) {
  return rows > 0 && rows <= 65535 && columns > 0 && columns % 512 == 0 &&
         uint64_t(columns) * sizeof(uint16_t) <= pipelines().scratch_limit;
}

ConvRotRows prepare_convrot_int8(const at::Tensor& input) {
  TORCH_CHECK(input.is_mps() && input.scalar_type() == at::kBFloat16 && input.dim() == 2 &&
                  input.is_contiguous() && convrot_int8_fits(input.size(0), input.size(1)),
              "Unsupported RelayVSR ConvRot TensorOps activation");
  const uint32_t rows = input.size(0), columns = input.size(1);
  const auto height = tile_height(rows);
  const uint32_t padded = (rows + height - 1) / height * height;
  auto codes = at::empty({padded, columns}, input.options().dtype(at::kChar));
  auto scales = at::empty({padded}, input.options().dtype(at::kFloat));
  const auto& p = pipelines();
  auto* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    auto enc = stream->commandEncoder();
    [enc setComputePipelineState:p.rotation];
    bind(enc, input, 0);
    bind(enc, codes, 1);
    bind(enc, scales, 2);
    [enc setBytes:&rows length:sizeof(rows) atIndex:3];
    [enc setBytes:&columns length:sizeof(columns) atIndex:4];
    [enc setThreadgroupMemoryLength:columns * sizeof(uint16_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(padded, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  });
  return {codes, scales, rows};
}

at::Tensor project_int8(const ConvRotRows& input, const at::Tensor& weight,
                        const at::Tensor& scales, const at::Tensor& bias, bool gelu) {
  TORCH_CHECK(weight.dim() == 2 && weight.is_contiguous() && weight.scalar_type() == at::kChar &&
                  weight.size(0) % 128 == 0 && weight.size(1) == input.codes.size(1) &&
                  bias.defined() && bias.scalar_type() == at::kBFloat16 &&
                  scales.scalar_type() == at::kFloat && scales.numel() == weight.size(0),
              "Unsupported RelayVSR ConvRot TensorOps weight");
  const uint32_t m = input.rows, n = weight.size(0), k = weight.size(1);
  const auto height = tile_height(m);
  const uint32_t m_tiles = (m + height - 1) / height, n_tiles = n / 128;
  const uint32_t swizzle = m_tiles <= 3 ? 0 : m_tiles <= 6 ? 1 : 2;
  const uint32_t group = 1u << swizzle;
  auto out = at::empty({m, n}, bias.options());
  auto matmul = pipelines().matmul[(gelu ? 2 : 0) + (height == 32 ? 1 : 0)];
  auto* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    auto enc = stream->commandEncoder();
    [enc setComputePipelineState:matmul];
    bind(enc, input.codes, 0);
    bind(enc, weight, 1);
    bind(enc, out, 2);
    [enc setBytes:&m length:sizeof(m) atIndex:3];
    [enc setBytes:&n length:sizeof(n) atIndex:4];
    [enc setBytes:&k length:sizeof(k) atIndex:5];
    bind(enc, input.scales, 6);
    bind(enc, scales, 7);
    [enc setBytes:&swizzle length:sizeof(swizzle) atIndex:8];
    [enc setBytes:&m_tiles length:sizeof(m_tiles) atIndex:9];
    [enc setBytes:&n_tiles length:sizeof(n_tiles) atIndex:10];
    bind(enc, bias, 11);
    [enc dispatchThreadgroups:MTLSizeMake(n_tiles * group, (m_tiles + group - 1) / group, 1)
        threadsPerThreadgroup:MTLSizeMake(height * 4, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  });
  return out;
}

}  // namespace rv
