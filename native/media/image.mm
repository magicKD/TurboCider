#include "runtime.hpp"
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
namespace tc {
void save_png(const Tensor&pixels,const std::filesystem::path&output){
 require(pixels.ndim()==4&&pixels.shape(0)==1&&pixels.shape(3)==3,"expected NHWC RGB pixels");
 // Match the reference export: denormalize in the VAE dtype, then
 // convert to FP32 for byte quantization (BF16 rounding is observable).
 auto unit=mx::clip(pixels/Tensor(2.f,pixels.dtype())+Tensor(.5f,pixels.dtype()),Tensor(0.f,pixels.dtype()),Tensor(1.f,pixels.dtype()));
 auto bytes=mx::astype(mx::round(mx::astype(unit,mx::float32)*255.f),mx::uint8);mx::eval(bytes);
 auto parent=output.parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
 auto tmp=output.string()+"."+std::string(NSUUID.UUID.UUIDString.UTF8String)+".tmp";
 CGDataProviderRef provider=CGDataProviderCreateWithData(nullptr,bytes.data<uint8_t>(),bytes.size(),nullptr);
 CGColorSpaceRef color=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
 CGImageRef image=CGImageCreate(pixels.shape(2),pixels.shape(1),8,24,pixels.shape(2)*3,color,kCGBitmapByteOrderDefault,provider,nullptr,false,kCGRenderingIntentDefault);
 require(image!=nullptr,"could not create RGB image");
 CGImageDestinationRef dest=CGImageDestinationCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:@(tmp.c_str())],(__bridge CFStringRef)UTTypePNG.identifier,1,nullptr);
 bool ok=false;if(dest){CGImageDestinationAddImage(dest,image,nullptr);ok=CGImageDestinationFinalize(dest);CFRelease(dest);}
 CGImageRelease(image);CGColorSpaceRelease(color);CGDataProviderRelease(provider);
 if(!ok){std::filesystem::remove(tmp);throw std::runtime_error("PNG encoding failed");}
 try{std::filesystem::rename(tmp,output);}catch(...){std::filesystem::remove(tmp);throw;}
}
}
