// W8A8 Linear V9 — Unified prefill (GEMM) + decode (FP MV) with bias
//
// M > 1: quantize activation → INT8 GEMM → y = acc * scale_a * scale_w + bias
// M == 1: FP MV → y = scale_w * sum(float(w_int8) * float(x)) + bias
//         Per-channel: single scale per row, no group switching overhead.

#include "w8a8_primitive.h"
#include "qwen21_cider_kernels_generated.hpp"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fstream>
#include <atomic>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace cider {

using namespace mlx::core;

struct TileLarge {
  static constexpr uint32_t BM = 128, BN = 128, THREADS = 512;
};
struct TileSmall {
  static constexpr uint32_t BM = 32, BN = 128, THREADS = 128;
};

// Local integration: embed the pinned kernels in the native library.
static std::string read_file(const std::string &path) {
  return tc::qwen21::embedded_cider_kernel(path.substr(path.find_last_of('/') + 1));
}

// ── Pipeline cache ──────────────────────────────────────────────
class PipelineCache {
public:
  static PipelineCache &instance() {
    static PipelineCache cache;
    return cache;
  }

  void ensure_init(const std::string &kernel_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && kernel_dir_ == kernel_dir) {
      return;
    }
    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *mtl_device = dev.mtl_device();
    matmul_lib_ = compile_source(mtl_device, kernel_dir + "/w8a8_matmul.metal");
    quantize_lib_ = compile_source(mtl_device, kernel_dir + "/w8a8_quantize.metal");
    mv_lib_ = compile_source(mtl_device, kernel_dir + "/w8a8_int8_mv.metal");
    convrot_quantize_lib_ = compile_source(mtl_device, kernel_dir + "/convrot_quantize.metal");
    kernel_dir_ = kernel_dir;
    pipelines_.clear();
    initialized_ = true;
  }

  MTL::ComputePipelineState *get(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) {
      return it->second;
    }
    auto &dev = metal::device(mlx::core::Device::gpu);
    auto *pso = make_pipeline(dev.mtl_device(), name);
    pipelines_[name] = pso;
    return pso;
  }

private:
  PipelineCache() = default;
  bool initialized_ = false;
  std::string kernel_dir_;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
  MTL::Library *matmul_lib_ = nullptr;
  MTL::Library *quantize_lib_ = nullptr;
  MTL::Library *mv_lib_ = nullptr;
  MTL::Library *convrot_quantize_lib_ = nullptr;
  std::mutex mutex_;

  MTL::Library *compile_source(MTL::Device *mtl_device,
                               const std::string &source_path) {
    std::string source = read_file(source_path);
    @autoreleasepool {
      NSString *src = [NSString stringWithUTF8String:source.c_str()];
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      // A rounded reciprocal can move exact half-way activation values to
      // the wrong RNE code. Preserve the FP32 absmax/divide contract here;
      // the historical Cider GEMM compilation policy remains unchanged.
      if (source_path.ends_with("/convrot_quantize.metal"))
        opts.mathMode = MTLMathModeSafe;
      // Keep Metal 4 at runtime while allowing builds with pre-26 SDK headers,
      // as in the native Metal graph backend.
      static_assert(MTLLanguageVersion3_1 == ((3U << 16) + 1));
      opts.languageVersion = static_cast<MTLLanguageVersion>(4U << 16);
      NSError *error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      id<MTLLibrary> lib_objc = [device_objc newLibraryWithSource:src
                                                          options:opts
                                                            error:&error];
      if (!lib_objc) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Metal compile failed (" + source_path +
                                 "): " + err);
      }
      return (__bridge MTL::Library *)(void *)CFBridgingRetain(lib_objc);
    }
  }

  MTL::ComputePipelineState *make_pipeline(MTL::Device *mtl_device,
                                           const std::string &name) {
    @autoreleasepool {
      NSString *fn_name = [NSString stringWithUTF8String:name.c_str()];
      id<MTLFunction> func = nil;
      for (auto *lib : {matmul_lib_, quantize_lib_, mv_lib_, convrot_quantize_lib_}) {
        if (!lib) continue;
        id<MTLLibrary> lib_objc = (__bridge id<MTLLibrary>)lib;
        func = [lib_objc newFunctionWithName:fn_name];
        if (func) break;
      }
      if (!func) {
        throw std::runtime_error("Kernel not found: " + name);
      }
      NSError *error = nil;
      id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
      // Keep the function name in Metal captures, including for pipelines
      // compiled and cached before a capture starts.
      MTLComputePipelineDescriptor *descriptor = [[MTLComputePipelineDescriptor alloc] init];
      descriptor.computeFunction = func;
      descriptor.label = fn_name;
      id<MTLComputePipelineState> pso = [device_objc newComputePipelineStateWithDescriptor:descriptor
                                                                                options:MTLPipelineOptionNone
                                                                             reflection:nil
                                                                                  error:&error];
      if (!pso) {
        std::string err =
            error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Pipeline failed for " + name + ": " + err);
      }
      return (__bridge MTL::ComputePipelineState *)(void *)CFBridgingRetain(
          pso);
    }
  }
};

namespace {
std::atomic<uint64_t> convrot_projections{0}, convrot_activation_rows{0};
}

bool convrot_w8a8_available() {
  // Integer TensorOps arrived in macOS 26.4 (floating NAX predates it).
  // MLX's internal capability helper is not exported by its release dylib.
  if (@available(macOS 26.4, *))
    return metal::device(Device::gpu).get_architecture_gen() >= 17;
  return false;
}
ConvrotW8A8Stats convrot_w8a8_stats() {
  return {convrot_projections.load(std::memory_order_relaxed),
          convrot_activation_rows.load(std::memory_order_relaxed)};
}

void ConvrotW8A8Linear::eval_gpu(const std::vector<array>& inputs,
                                std::vector<array>& outputs) {
  if (!convrot_w8a8_available())
    throw std::runtime_error("ConvRot W8A8 requires M5 Metal TensorOps and macOS 26.4 or later");
  const auto& x = inputs[0];
  const auto& w = inputs[1]; // physically padded N, unchanged signed codes
  const auto& scales = inputs[2];
  auto& output = outputs[0];
  const uint32_t m = output.shape(0), n = output.shape(1), k = x.shape(1);
  const bool small = m <= 64;
  const uint32_t bm = small ? 32 : 128, bn = 128;
  const uint32_t padded_m = ((m+bm-1)/bm)*bm;
  array activation({int(padded_m),int(k)},int8,nullptr,{});
  array token_scales({int(padded_m)},float32,nullptr,{});
  activation.set_data(allocator::malloc(activation.nbytes()));
  token_scales.set_data(allocator::malloc(token_scales.nbytes()));
  output.set_data(allocator::malloc(output.nbytes()));
  auto& cache = PipelineCache::instance();
  cache.ensure_init("builtin:cider-4d91fcee");
  auto& enc = metal::get_command_encoder(stream());
  const char* quantize = x.dtype()==bfloat16 ? "tc_convrot_quantize_bf16" :
                        x.dtype()==float16 ? "tc_convrot_quantize_f16" : "tc_convrot_quantize_f32";
  enc.set_compute_pipeline_state(cache.get(quantize));
  enc.set_input_array(x,0);
  enc.set_output_array(activation,1);
  enc.set_output_array(token_scales,2);
  enc.set_bytes(m,3);
  enc.set_bytes(k,4);
  enc.dispatch_threadgroups(MTL::Size(padded_m,1,1),MTL::Size(256,1,1));
  enc.barrier();
  enc.set_compute_pipeline_state(cache.get(small ? "tc_convrot_w8a8_f32_small" : "tc_convrot_w8a8_f32"));
  const uint32_t tiles_m = (m+bm-1)/bm, tiles_n = (n+bn-1)/bn;
  const uint32_t swizzle = tiles_m<=3 ? 0 : tiles_m<=6 ? 1 : 2;
  const uint32_t tile = 1u<<swizzle;
  enc.set_input_array(activation,0);
  enc.set_input_array(w,1);
  enc.set_output_array(output,2);
  enc.set_bytes(m,3); enc.set_bytes(n,4); enc.set_bytes(k,5);
  enc.set_input_array(token_scales,6); enc.set_input_array(scales,7);
  enc.set_bytes(swizzle,8); enc.set_bytes(tiles_m,9); enc.set_bytes(tiles_n,10);
  enc.dispatch_threadgroups(MTL::Size(tiles_n*tile,(tiles_m+tile-1)/tile,1),
                            MTL::Size(small ? 128 : 512,1,1));
  enc.add_temporary(activation); enc.add_temporary(token_scales);
  convrot_projections.fetch_add(1,std::memory_order_relaxed);
  convrot_activation_rows.fetch_add(m,std::memory_order_relaxed);
}

array convrot_linear(const array& rotated, const array& codes,
                     const array& row_scales, StreamOrDevice s) {
  if (!convrot_w8a8_available())
    throw std::invalid_argument("ConvRot W8A8 requires M5 Metal TensorOps and macOS 26.4 or later");
  if (rotated.ndim()!=2 || codes.ndim()!=2 || row_scales.ndim()!=2 ||
      (rotated.dtype()!=bfloat16 && rotated.dtype()!=float16 && rotated.dtype()!=float32) ||
      codes.dtype()!=int8 || row_scales.dtype()!=float32 ||
      rotated.shape(0)<=0 || codes.shape(0)<=0 || codes.shape(1)<=0 ||
      rotated.shape(1)!=codes.shape(1) || codes.shape(1)%256 ||
      codes.shape(1)>131072 || row_scales.shape()!=Shape{codes.shape(0),1} ||
      rotated.shape(0)>std::numeric_limits<int>::max()-128 ||
      codes.shape(0)>std::numeric_limits<int>::max()-128)
    throw std::invalid_argument("ConvRot W8A8 requires floating [M,K], original I8 [N,K], F32 [N,1], K aligned to256 and <=131072");
  const int n = codes.shape(0), padded_n = ((n+127)/128)*128;
  auto w = n==padded_n ? codes : pad(codes,std::vector<std::pair<int,int>>{{0,padded_n-n},{0,0}},array(0,int8),"constant",s);
  return array({rotated.shape(0),n},float32,
      std::make_shared<ConvrotW8A8Linear>(to_stream(s)),
      {contiguous(rotated,false,s),contiguous(w,false,s),contiguous(row_scales,false,s)});
}

void W8A8Linear::eval_gpu(const std::vector<array> &inputs,
                          std::vector<array> &outputs) {
  auto &x = inputs[0];       // [M, K] float16
  auto &w = inputs[1];       // [N, K] int8
  auto &scale_w = inputs[2]; // [N] float32
  auto &bias = inputs[3];    // [N] float16
  auto &out = outputs[0];    // [M, N] float16

  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t N = static_cast<uint32_t>(w.shape(0));
  uint32_t K = static_cast<uint32_t>(w.shape(1));

  out.set_data(allocator::malloc(out.nbytes()));

  auto &cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);

  if (M == 1) {
    // ── Decode path: FP MV (no activation quantization) ──
    // w8a8_int8_mv.metal buffer layout:
    //   x[0], W[1], y[2], scale_w[3], N[4], K[5], bias[6]
    auto *pso = cache.get("w8a8_int8_mv");
    enc.set_compute_pipeline_state(pso);
    enc.set_input_array(x, 0);       // [1, K] float16
    enc.set_input_array(w, 1);       // [N, K] int8
    enc.set_output_array(out, 2);    // [1, N] float16
    enc.set_input_array(scale_w, 3); // [N] float32
    enc.set_bytes(N, 4);
    enc.set_bytes(K, 5);
    enc.set_input_array(bias, 6);    // [N] float16

    constexpr uint32_t NUM_SIMDGROUPS = 2;
    constexpr uint32_t RESULTS_PER_SG = 4;
    constexpr uint32_t SIMD_SIZE = 32;
    uint32_t rows_per_tg = NUM_SIMDGROUPS * RESULTS_PER_SG;  // 16
    uint32_t grid_n = (N + rows_per_tg - 1) / rows_per_tg;

    enc.dispatch_threadgroups(MTL::Size::Make(grid_n, 1, 1),
                              MTL::Size::Make(SIMD_SIZE * NUM_SIMDGROUPS, 1, 1));
  } else {
    // ── Prefill path: quantize + INT8 GEMM ──
    size_t a_bytes = static_cast<size_t>(M) * K;
    size_t sa_bytes = static_cast<size_t>(M) * sizeof(float);

    array a_int8({static_cast<int>(M), static_cast<int>(K)}, int8, nullptr, {});
    a_int8.set_data(allocator::malloc(a_bytes));

    array sa({static_cast<int>(M)}, float32, nullptr, {});
    sa.set_data(allocator::malloc(sa_bytes));

    // DISPATCH 1: quantize_per_token
    {
      auto *pso = cache.get("quantize_per_token");
      enc.set_compute_pipeline_state(pso);
      enc.set_input_array(x, 0);
      enc.set_output_array(a_int8, 1);
      enc.set_output_array(sa, 2);
      enc.set_bytes(M, 3);
      enc.set_bytes(K, 4);
      uint32_t tg = std::min(256u, std::max(32u, ((K + 31) / 32) * 32));
      enc.dispatch_threadgroups(MTL::Size::Make(M, 1, 1),
                                MTL::Size::Make(tg, 1, 1));
    }

    enc.barrier();

    // DISPATCH 2: matmul_dequant
    {
      bool use_small = (M <= 64);
      const char *kname = use_small ? "w8a8_matmul_fused_dequant_small"
                                    : "w8a8_matmul_fused_dequant";
      uint32_t BM = use_small ? TileSmall::BM : TileLarge::BM;
      uint32_t BN = use_small ? TileSmall::BN : TileLarge::BN;
      uint32_t threads = use_small ? TileSmall::THREADS : TileLarge::THREADS;

      auto *pso = cache.get(kname);
      enc.set_compute_pipeline_state(pso);

      uint32_t tiles_n = (N + BN - 1) / BN;
      uint32_t tiles_m = (M + BM - 1) / BM;
      uint32_t swizzle_log;
      if (tiles_m <= 3) {
        swizzle_log = 0;
      } else if (tiles_m <= 6) {
        swizzle_log = 1;
      } else {
        swizzle_log = 2;
      }
      uint32_t tile = 1u << swizzle_log;
      uint32_t grid_x = tiles_n * tile;
      uint32_t grid_y = (tiles_m + tile - 1) / tile;

      enc.set_input_array(a_int8, 0);
      enc.set_input_array(w, 1);
      enc.set_output_array(out, 2);
      enc.set_bytes(M, 3);
      enc.set_bytes(N, 4);
      enc.set_bytes(K, 5);
      enc.set_input_array(sa, 6);
      enc.set_input_array(scale_w, 7);
      enc.set_bytes(swizzle_log, 8);
      enc.set_bytes(tiles_m, 9);
      enc.set_bytes(tiles_n, 10);
      enc.set_input_array(bias, 11);

      enc.dispatch_threadgroups(MTL::Size::Make(grid_x, grid_y, 1),
                                MTL::Size::Make(threads, 1, 1));
    }

    enc.add_temporary(a_int8);
    enc.add_temporary(sa);
  }
}

array perchannel_linear(const array &x, const array &w, const array &scale_w,
                  const array &bias, const std::string &kernel_dir,
                  StreamOrDevice s) {
  if (x.ndim() != 2) {
    throw std::invalid_argument("perchannel_linear: x must be 2D [M,K]");
  }
  if (w.ndim() != 2) {
    throw std::invalid_argument("perchannel_linear: w must be 2D [N,K]");
  }

  int M = x.shape(0);
  int N = w.shape(0);
  auto stream = to_stream(s);

  auto result =
      array({M, N}, float16, std::make_shared<W8A8Linear>(stream, kernel_dir),
            {astype(x, float16, stream), astype(w, int8, stream),
             astype(scale_w, float32, stream), astype(bias, float16, stream)});
  if (x.dtype() == bfloat16) {
    return astype(result, bfloat16, stream);
  }
  return result;
}

// ── Int8MatMulInt32 primitive (raw INT32 output, no dequant) ────

void Int8MatMulInt32::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &a = inputs[0]; // [M, K] int8
  auto &b = inputs[1]; // [N, K] int8

  uint32_t M = static_cast<uint32_t>(a.shape(0));
  uint32_t N = static_cast<uint32_t>(b.shape(0));
  uint32_t K = static_cast<uint32_t>(b.shape(1));

  auto &out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &cache = PipelineCache::instance();
  cache.ensure_init(kernel_dir_);

  bool use_small = (M <= 64);
  auto *pso = use_small ? cache.get("int8_matmul_int32_small")
                        : cache.get("int8_matmul_int32");

  uint32_t BM = use_small ? 32 : 128;
  uint32_t BN = 128;
  uint32_t threads = use_small ? 128 : 512;

  uint32_t tiles_n = (N + BN - 1) / BN;
  uint32_t tiles_m = (M + BM - 1) / BM;
  uint32_t swizzle_log;
  if (tiles_m <= 3) {
    swizzle_log = 0;
  } else if (tiles_m <= 6) {
    swizzle_log = 1;
  } else {
    swizzle_log = 2;
  }
  uint32_t tile = 1u << swizzle_log;
  uint32_t grid_x = tiles_n * tile;
  uint32_t grid_y = (tiles_m + tile - 1) / tile;

  auto &s = stream();
  auto &enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(pso);
  enc.set_input_array(a, 0);
  enc.set_input_array(b, 1);
  enc.set_output_array(out, 2);
  enc.set_bytes(&M, sizeof(M), 3);
  enc.set_bytes(&N, sizeof(N), 4);
  enc.set_bytes(&K, sizeof(K), 5);
  enc.set_bytes(&swizzle_log, sizeof(swizzle_log), 6);
  enc.set_bytes(&tiles_m, sizeof(tiles_m), 7);
  enc.set_bytes(&tiles_n, sizeof(tiles_n), 8);
  enc.dispatch_threadgroups(MTL::Size(grid_x, grid_y, 1),
                            MTL::Size(threads, 1, 1));
}

mx::array int8_matmul_int32(const mx::array &a, const mx::array &b,
                            const std::string &kernel_dir,
                            mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(a.shape(0));
  uint32_t N = static_cast<uint32_t>(b.shape(0));
  return mx::array(
      {static_cast<int>(M), static_cast<int>(N)}, mx::int32,
      std::make_shared<Int8MatMulInt32>(mx::to_stream(s), kernel_dir),
      {astype(a, int8, mx::to_stream(s)), astype(b, int8, mx::to_stream(s))});
}

} // namespace cider
