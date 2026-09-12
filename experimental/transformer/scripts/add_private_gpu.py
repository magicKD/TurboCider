from pathlib import Path
p=Path(__file__).resolve().parents[1];s=(p/'src/runner_private.mm').read_text()
s=s.replace('        if(!request_)throw std::runtime_error("private request failed");','''        if(!request_)throw std::runtime_error("private request failed");
        device_ = MTLCreateSystemDefaultDevice();queue_=[device_ newCommandQueue];
        NSError *me=nil;
        NSString *source=@R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void transpose16(device const half *x [[buffer(0)]],device half *y [[buffer(1)]],constant uint &rows [[buffer(2)]],constant uint &cols [[buffer(3)]],uint2 tid [[thread_position_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint ix=group.x*16+tid.x,iy=group.y*16+tid.y;
    if(ix<cols && iy<rows)tile[tid.y][tid.x]=x[iy*cols+ix];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ox=group.y*16+tid.x,oy=group.x*16+tid.y;
    if(ox<rows && oy<cols)y[oy*rows+ox]=tile[tid.x][tid.y];
})METAL";
        auto lib=[device_ newLibraryWithSource:source options:nil error:&me];
        pipeline_=[device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"transpose16"] error:&me];
        if(!pipeline_)throw std::runtime_error("private GPU transpose compile failed");
        inputBuffer_=[device_ newBufferWithBytesNoCopy:input_ length:size_t(m_)*k_*2 options:MTLResourceStorageModeShared deallocator:nil];
        outputBuffer_=[device_ newBufferWithBytesNoCopy:output_ length:size_t(m_)*n_*2 options:MTLResourceStorageModeShared deallocator:nil];
        inBuffer_=[device_ newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(in_) length:IOSurfaceGetAllocSize(in_) options:MTLResourceStorageModeShared deallocator:nil];
        outBuffer_=[device_ newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(out_) length:IOSurfaceGetAllocSize(out_) options:MTLResourceStorageModeShared deallocator:nil];
        if(!inputBuffer_||!outputBuffer_||!inBuffer_||!outBuffer_)throw std::runtime_error("private no-copy Metal binding failed");''',1)
a=s.index('        auto t=Clock::now();IOSurfaceLock(in_');b=s.index('NSError *e=nil;',a)
s=s[:a]+'''        auto t=Clock::now();transpose(inputBuffer_,inBuffer_,m_,k_);'''+s[b:]
a=s.index('        IOSurfaceLock(out_,kIOSurfaceLockReadOnly');b=s.index('return milliseconds(Clock::now()-t);',a)
s=s[:a]+'''        transpose(outBuffer_,outputBuffer_,n_,m_);'''+s[b:]
a=s.index('    static IOSurfaceRef surface(')
s=s[:a]+'''    void transpose(id<MTLBuffer> x,id<MTLBuffer> y,uint32_t rows,uint32_t cols){
        auto cb=[queue_ commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline_];
        [e setBuffer:x offset:0 atIndex:0];[e setBuffer:y offset:0 atIndex:1];[e setBytes:&rows length:4 atIndex:2];[e setBytes:&cols length:4 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((cols+15)/16,(rows+15)/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];checkCommandBuffer(cb);
    }
    id<MTLDevice> device_;id<MTLCommandQueue> queue_;id<MTLComputePipelineState> pipeline_;
    id<MTLBuffer> inputBuffer_,outputBuffer_,inBuffer_,outBuffer_;
'''+s[a:]
# Metal aliases must be released before IOSurface ownership is released.
s=s.replace('        if(in_)CFRelease(in_);','        inputBuffer_=nil;outputBuffer_=nil;inBuffer_=nil;outBuffer_=nil;\n        if(in_)CFRelease(in_);')
s=s.replace('        [cb commit];\n        gate.store(true, std::memory_order_release);', '\n        if(std::getenv("TC_SERIAL_BRANCHES")) {\n            gate.store(true,std::memory_order_release);\n            if(cpuWorker_)cpuWorker_->wait();\n            if(aneWorker_)aneWorker_->wait();\n        }\n        [cb commit];\n        gate.store(true, std::memory_order_release);',1)
(p/'src/runner_private_gpu.mm').write_text(s)
