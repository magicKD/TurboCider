#include "../platform/apple/bridge.hpp"
#include "image.hpp"
#import <ImageIO/ImageIO.h>
#include <cmath>
namespace tc {
struct RGBImage {int width,height;std::vector<uint8_t> bytes;};
static RGBImage read_rgb(const std::filesystem::path& path) {
    CGImageSourceRef source=CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:@(path.c_str())],nullptr);
    require(source!=nullptr,"cannot read image: "+path.string());
    NSDictionary *properties=CFBridgingRelease(CGImageSourceCopyPropertiesAtIndex(source,0,nullptr));
    int original_w=[properties[(__bridge NSString*)kCGImagePropertyPixelWidth] intValue];
    int original_h=[properties[(__bridge NSString*)kCGImagePropertyPixelHeight] intValue];
    if(original_w<=0||original_h<=0||uint64_t(original_w)*original_h>100000000){CFRelease(source);throw std::invalid_argument("image dimensions exceed import limit");}
    NSDictionary *options=@{(__bridge NSString*)kCGImageSourceCreateThumbnailFromImageAlways:@YES,(__bridge NSString*)kCGImageSourceCreateThumbnailWithTransform:@YES,(__bridge NSString*)kCGImageSourceThumbnailMaxPixelSize:@(std::max(original_w,original_h))};
    CGImageRef image=CGImageSourceCreateThumbnailAtIndex(source,0,(__bridge CFDictionaryRef)options);CFRelease(source);
    require(image!=nullptr,"image decode failed");
    int width=int(CGImageGetWidth(image)),height=int(CGImageGetHeight(image));
    std::vector<uint8_t> rgba(size_t(width)*height*4,255);
    CGColorSpaceRef color=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef context=CGBitmapContextCreate(rgba.data(),width,height,8,width*4,color,CGBitmapInfo(kCGImageAlphaPremultipliedLast)|kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color);
    if(!context){CGImageRelease(image);throw std::runtime_error("image bitmap allocation failed");}
    CGContextSetRGBFillColor(context,1,1,1,1);CGContextFillRect(context,CGRectMake(0,0,width,height));
    CGContextDrawImage(context,CGRectMake(0,0,width,height),image);CGContextRelease(context);CGImageRelease(image);
    RGBImage result{width,height,std::vector<uint8_t>(size_t(width)*height*3)};
    for(size_t i=0;i<size_t(width)*height;++i)std::copy_n(rgba.data()+i*4,3,result.bytes.data()+i*3);
    return result;
}
static double lanczos(double x) {
    x=std::abs(x);if(x==0)return 1;if(x>=3)return 0;
    double a=M_PI*x;return (std::sin(a)/a)*(std::sin(a/3)/(a/3));
}
static RGBImage resize_axis(const RGBImage& src,int size,bool horizontal) {
    int original=horizontal?src.width:src.height;
    if(size==original)return src;
    RGBImage dst{horizontal?size:src.width,horizontal?src.height:size,{}};
    dst.bytes.resize(size_t(dst.width)*dst.height*3);
    double scale=double(original)/size,filter_scale=std::max(1.,scale),support=3*filter_scale;
    for(int target=0;target<size;++target) {
        double center=(target+.5)*scale;
        int start=std::max(0,int(center-support+.5)),end=std::min(original,int(center+support+.5));
        std::vector<double> weights(end-start);double total=0;
        for(int i=start;i<end;++i){weights[i-start]=lanczos((i+.5-center)/filter_scale);total+=weights[i-start];}
        std::vector<int> fixed(weights.size());for(size_t i=0;i<weights.size();++i)fixed[i]=int(std::llround(weights[i]/total*(1<<22)));
        for(int other=0;other<(horizontal?src.height:src.width);++other)for(int c=0;c<3;++c) {
            int64_t sum=1<<21;
            for(int i=start;i<end;++i){int x=horizontal?i:other,y=horizontal?other:i;sum+=int64_t(src.bytes[(size_t(y)*src.width+x)*3+c])*fixed[i-start];}
            int x=horizontal?target:other,y=horizontal?other:target;
            dst.bytes[(size_t(y)*dst.width+x)*3+c]=uint8_t(std::clamp<int64_t>(sum>>22,0,255));
        }
    }
    return dst;
}
Tensor load_image_tensor(const std::filesystem::path& path,int width,int height,bool reference) {
    auto image=read_rgb(path);
    if(reference) {
        double scale=std::min(1.,std::sqrt(1048576./(double(image.width)*image.height)));
        int rw=int(std::nearbyint(image.width*scale)),rh=int(std::nearbyint(image.height*scale));
        image=resize_axis(resize_axis(image,rw,true),rh,false);
        width=rw/16*16;height=rh/16*16;
        require(width>=16&&height>=16,"reference image needs at least 16 pixels per side");
    } else {
        image=resize_axis(resize_axis(image,width,true),height,false);
    }
    int left=(image.width-width)/2,top=(image.height-height)/2;
    std::vector<uint8_t> cropped(size_t(width)*height*3);
    for(int y=0;y<height;++y)std::copy_n(image.bytes.data()+(size_t(top+y)*image.width+left)*3,width*3,cropped.data()+size_t(y)*width*3);
    auto pixels=mx::astype(Tensor(cropped.data(),{1,height,width,3},mx::uint8),mx::float32)/255.f;
    return pixels*2.f-1.f;
}
}
