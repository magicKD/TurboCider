// Explicit opt-in compiler check only. No command queue/buffer, dispatch,
// IOSurface, ANE API or model is created by this executable.
#include "../../native/backends/private/ane_transfer_kernels.hpp"
#include "../../native/backends/private/ane_w8_kernels.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
std::string diagnostic(NSError *error) {
    return error ? std::string(error.localizedDescription.UTF8String ?: "unavailable diagnostic") : "none";
}
}
int main(int argc, const char *argv[]) {
    if (argc != 2 || std::strcmp(argv[1], "--compile-only") != 0) {
        std::cerr << "requires explicit --compile-only; no compute path exists\n"; return 2;
    }
    @autoreleasepool {
      @try {
        using namespace tc::ane::private_api;
        const std::string source = std::string(transfer_source) + a8_single_pass_source;
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(source.data(), CC_LONG(source.size()), digest);
        std::cout << "shader_input_sha256=";
        for (unsigned char byte : digest) std::cout << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
        std::cout << std::dec << "\nshader_input_bytes=" << source.size() << '\n';
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::cerr << "Metal device unavailable for compile-only validation\n"; return 77; }
        std::cout << "device=" << device.name.UTF8String << "\ndevice_max_threadgroup_bytes=" << device.maxThreadgroupMemoryLength << '\n';
        MTLCompileOptions *options = [MTLCompileOptions new]; options.fastMathEnabled = NO;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
        std::cout << "library_diagnostic=" << diagnostic(error) << '\n';
        if (!library) { std::cerr << "actual candidate library compile failed\n"; return 1; }
        std::cout << "library_compile=PASS\n";
        for (uint32_t columns : {3840u, 4096u}) {
            MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&columns type:MTLDataTypeUInt atIndex:0];
            error = nil;
            id<MTLFunction> function = [library newFunctionWithName:@"tc_ane_a8_single_pass" constantValues:constants error:&error];
            std::cout << "columns=" << columns << " function_diagnostic=" << diagnostic(error) << '\n';
            if (!function) { std::cerr << "candidate specialization failed\n"; return 1; }
            error = nil;
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            std::cout << "columns=" << columns << " pipeline_diagnostic=" << diagnostic(error) << '\n';
            if (!pipeline) { std::cerr << "candidate pipeline compile failed\n"; return 1; }
            std::cout << "columns=" << columns << " simd_width=" << pipeline.threadExecutionWidth
                << " max_threads=" << pipeline.maxTotalThreadsPerThreadgroup
                << " static_threadgroup_bytes=" << pipeline.staticThreadgroupMemoryLength << '\n';
            if (pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 128 ||
                pipeline.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength) {
                std::cerr << "candidate pipeline capacity unsupported\n"; return 1;
            }
        }
        std::cout << "compile_only=PASS command_buffers_created=0 dispatches=0 models_loaded=0\n";
        return 0;
      } @catch (NSException *error) {
        std::cerr << "Metal compile-only exception: " << (error.reason.UTF8String ?: "unknown") << '\n'; return 1;
      }
    }
}
