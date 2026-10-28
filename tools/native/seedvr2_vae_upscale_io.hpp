#pragma once
// Public probe IO shared with the validated hot-convolution ownership recipe.
// Offline tooling only; backing identity is not a proof of internal zero copy.
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <mlx/mlx.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace tc::seedvr2_upscale_probe {
namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;
void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
uint64_t footprint(){task_vm_info_data_t value{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    check(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&value),&count)==KERN_SUCCESS,"footprint unavailable");
    return value.phys_footprint;}
id<MTLBuffer> buffer(const mx::array &value,id<MTLDevice> device) {
    check(value.is_available() && value.flags().row_contiguous && value.offset()==0,"unmaterialized/noncontiguous MLX operand");
    auto result=(__bridge id<MTLBuffer>)const_cast<void *>(value.buffer().ptr());
    check(result && result.device==device && result.length>=value.nbytes(),"MLX Metal extent/device mismatch");
    return result;
}
NSArray *shape(int channels,int rows,int cols){return @[@1,@(channels),@(rows),@(cols)];}
struct Surface {
    CVPixelBufferRef pixel=nullptr;
    MLMultiArray *array=nil;
    id<MTLBuffer> metal=nil;
    int channels,rows,cols;
    Surface(id<MTLDevice> device,int c,int h,int w):channels(c),rows(h),cols(w) {
        NSDictionary *attributes=@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{}};
        check(CVPixelBufferCreate(kCFAllocatorDefault,w,c*h,kCVPixelFormatType_OneComponent16Half,
            (__bridge CFDictionaryRef)attributes,&pixel)==kCVReturnSuccess && pixel,"surface allocation failed");
        try {
            array=[[MLMultiArray alloc] initWithPixelBuffer:pixel shape:shape(c,h,w)];
            auto surface=CVPixelBufferGetIOSurface(pixel);
            check(array && surface && array.dataType==MLMultiArrayDataTypeFloat16,"surface binding failed");
            auto address=IOSurfaceGetBaseAddress(surface);const size_t bytes=IOSurfaceGetAllocSize(surface),page=getpagesize();
            check(address && bytes && uintptr_t(address)%page==0 && bytes<=SIZE_MAX-(page-1),"surface VM address/size invalid");
            const size_t mapped_bytes=(bytes+page-1)/page*page;
            // CoreVideo can report logical bytes + metadata rather than a page
            // multiple. Metal requires whole pages. Verify the owning mapped
            // VM region before wrapping the tail; kernels never touch beyond
            // validated logical strides. Never guess a readable tail page.
            mach_vm_address_t region=reinterpret_cast<mach_vm_address_t>(address);mach_vm_size_t extent=0;
            vm_region_basic_info_data_64_t information{};mach_msg_type_number_t count=VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object=MACH_PORT_NULL;
            const auto status=mach_vm_region(mach_task_self(),&region,&extent,VM_REGION_BASIC_INFO_64,
                reinterpret_cast<vm_region_info_t>(&information),&count,&object);
            if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
            const auto begin=reinterpret_cast<mach_vm_address_t>(address);
            check(status==KERN_SUCCESS && region<=begin && begin-region<=extent && mapped_bytes<=extent-(begin-region) &&
                (information.protection&(VM_PROT_READ|VM_PROT_WRITE))==(VM_PROT_READ|VM_PROT_WRITE),
                "surface rounded VM mapping is not fully readable/writable");
            metal=[device newBufferWithBytesNoCopy:address length:mapped_bytes options:MTLResourceStorageModeShared deallocator:nil];
            check(metal && metal.contents==address,"public no-copy Metal surface binding failed");
            check(array.strides.count==4 && array.strides[3].unsignedLongLongValue==1 &&
                array.strides[2].unsignedLongLongValue==CVPixelBufferGetBytesPerRow(pixel)/2 &&
                array.strides[1].unsignedLongLongValue==size_t(h)*CVPixelBufferGetBytesPerRow(pixel)/2,
                "surface strides differ from public pixel pitch");
        }catch(...){array=nil;metal=nil;CVPixelBufferRelease(pixel);pixel=nullptr;throw;}
    }
    ~Surface(){metal=nil;array=nil;if(pixel)CVPixelBufferRelease(pixel);}
    Surface(const Surface &)=delete;
    size_t bytes()const{return metal.length;}
};
struct Slot {
    Surface input,output;
    MLDictionaryFeatureProvider *provider=nil;
    MLPredictionOptions *options=nil;
    int last_tile=-1;
    Slot(id<MTLDevice> device,int cin,int cout,int hi,int wi,int ho,int wo)
        :input(device,cin,hi,wi),output(device,cout,ho,wo) {
        NSError *error=nil;
        provider=[[MLDictionaryFeatureProvider alloc] initWithDictionary:
            @{@"x":[MLFeatureValue featureValueWithMultiArray:input.array]} error:&error];
        check(provider!=nil,"provider allocation failed");
        options=[MLPredictionOptions new];options.outputBackings=@{@"y":output.array};
    }
};
// A submission retains both MLX allocations and the MLMultiArrays that own the
// external IOSurfaces. Exceptional unwinding drains before releasing any owner.
struct Submission {
    id<MTLCommandBuffer> command=nil;
    std::vector<mx::array> owners;
    NSMutableArray *surfaces=[NSMutableArray array];
    bool joined=false;
    ~Submission(){if(command && !joined)[command waitUntilCompleted];}
    double join(){[command waitUntilCompleted];joined=true;
        check(command.status==MTLCommandBufferStatusCompleted,"Metal pack/consume failed");
        return std::max(0.,command.GPUEndTime-command.GPUStartTime);}
};
} // namespace tc::seedvr2_upscale_probe
