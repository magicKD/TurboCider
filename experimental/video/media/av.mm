#include "runtime.hpp"
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <ImageIO/ImageIO.h>
extern "C" {
#include "../h3/vendor/h3_ffmpeg.h"
}
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
void failure(char *error,size_t size,const char *message) { if(error&&size)snprintf(error,size,"%s",message); }
AVURLAsset *asset_for(const char *path) { return [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(path)] options:nil]; }
std::vector<uint8_t> image_rgb(CGImageRef image,int width,int height,bool cover) {
    std::vector<uint8_t> rgba(size_t(width)*height*4,255),rgb(size_t(width)*height*3);
    CGColorSpaceRef color=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef context=CGBitmapContextCreate(rgba.data(),width,height,8,width*4,color,CGBitmapInfo(kCGImageAlphaPremultipliedLast)|kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color);tc::require(context!=nullptr,"cannot allocate media bitmap");
    CGContextSetRGBFillColor(context,1,1,1,1);CGContextFillRect(context,CGRectMake(0,0,width,height));
    CGRect rect=CGRectMake(0,0,width,height);
    if(cover){double scale=std::max(double(width)/CGImageGetWidth(image),double(height)/CGImageGetHeight(image));double w=CGImageGetWidth(image)*scale,h=CGImageGetHeight(image)*scale;rect=CGRectMake((width-w)/2,(height-h)/2,w,h);}
    CGContextSetInterpolationQuality(context,kCGInterpolationHigh);CGContextDrawImage(context,rect,image);CGContextRelease(context);
    for(size_t i=0;i<size_t(width)*height;++i)std::copy_n(rgba.data()+i*4,3,rgb.data()+i*3);
    return rgb;
}
CGImageRef visual_image(const char *path,CMTime time) {
    CGImageSourceRef source=CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:@(path)],nullptr);
    if(source){CGImageRef image=CGImageSourceCreateImageAtIndex(source,0,nullptr);CFRelease(source);if(image)return image;}
    auto generator=[[AVAssetImageGenerator alloc]initWithAsset:asset_for(path)];generator.appliesPreferredTrackTransform=YES;
    generator.requestedTimeToleranceBefore=kCMTimeZero;generator.requestedTimeToleranceAfter=kCMTimeZero;
    return [generator copyCGImageAtTime:time actualTime:nullptr error:nil];
}
int write_av(const char *path,const uint8_t *frames,int count,int width,int height,int fps,const float *pcm,int samples,int channels,int rate,char *error,size_t size) {
    @autoreleasepool {try {
        tc::require(path&&frames&&count>0&&width>0&&height>0&&fps>0,"invalid video output");
        std::filesystem::path output(path);if(!output.parent_path().empty())std::filesystem::create_directories(output.parent_path());
        NSString *temporary=[@(path) stringByAppendingFormat:@".%@.mp4",NSUUID.UUID.UUIDString];
        NSError *detail=nil;auto writer=[[AVAssetWriter alloc]initWithURL:[NSURL fileURLWithPath:temporary] fileType:AVFileTypeMPEG4 error:&detail];
        tc::require(writer!=nil,"cannot create video writer");
        auto video=[AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:@{AVVideoCodecKey:AVVideoCodecTypeH264,AVVideoWidthKey:@(width),AVVideoHeightKey:@(height),AVVideoColorPropertiesKey:@{AVVideoColorPrimariesKey:AVVideoColorPrimaries_ITU_R_709_2,AVVideoTransferFunctionKey:AVVideoTransferFunction_IEC_sRGB,AVVideoYCbCrMatrixKey:AVVideoYCbCrMatrix_ITU_R_709_2},AVVideoCompressionPropertiesKey:@{AVVideoAverageBitRateKey:@(std::max(1000000,width*height*fps/2)),AVVideoAllowFrameReorderingKey:@NO}}];
        video.expectsMediaDataInRealTime=NO;
        auto adaptor=[AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:video sourcePixelBufferAttributes:@{(NSString*)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA),(NSString*)kCVPixelBufferWidthKey:@(width),(NSString*)kCVPixelBufferHeightKey:@(height),(NSString*)kCVPixelBufferIOSurfacePropertiesKey:@{}}];
        tc::require([writer canAddInput:video],"unsupported video settings");[writer addInput:video];
        AVAssetWriterInput *audio=nil;
        if(pcm&&samples>0){
            tc::require(channels>0&&channels<=8&&rate>0,"invalid PCM format");
            audio=[AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeAudio outputSettings:@{AVFormatIDKey:@(kAudioFormatMPEG4AAC),AVSampleRateKey:@(rate),AVNumberOfChannelsKey:@(channels),AVEncoderBitRateKey:@192000}];
            tc::require([writer canAddInput:audio],"unsupported audio settings");[writer addInput:audio];
        }
        tc::require([writer startWriting],"video writer start failed");[writer startSessionAtSourceTime:kCMTimeZero];
        dispatch_queue_t video_queue=dispatch_queue_create("org.turbocider.video-export",DISPATCH_QUEUE_SERIAL);
        dispatch_queue_t audio_queue=dispatch_queue_create("org.turbocider.audio-export",DISPATCH_QUEUE_SERIAL);
        auto accepting=std::make_shared<std::atomic<bool>>(true);
        dispatch_group_t group=dispatch_group_create();dispatch_group_enter(group);
        __block int frame=0;__block bool video_done=false;
        [video requestMediaDataWhenReadyOnQueue:video_queue usingBlock:^{
            if(!accepting->load()||video_done)return;
            while(video.readyForMoreMediaData&&frame<count&&writer.status==AVAssetWriterStatusWriting){
                CVPixelBufferRef buffer=nullptr;CVReturn status=CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault,adaptor.pixelBufferPool,&buffer);
                if(status!=kCVReturnSuccess){[writer cancelWriting];break;}
                CVBufferSetAttachment(buffer,kCVImageBufferColorPrimariesKey,kCVImageBufferColorPrimaries_ITU_R_709_2,kCVAttachmentMode_ShouldPropagate);
                CVBufferSetAttachment(buffer,kCVImageBufferTransferFunctionKey,kCVImageBufferTransferFunction_sRGB,kCVAttachmentMode_ShouldPropagate);
                CVBufferSetAttachment(buffer,kCVImageBufferYCbCrMatrixKey,kCVImageBufferYCbCrMatrix_ITU_R_709_2,kCVAttachmentMode_ShouldPropagate);
                CVPixelBufferLockBaseAddress(buffer,0);auto dst=(uint8_t*)CVPixelBufferGetBaseAddress(buffer);size_t stride=CVPixelBufferGetBytesPerRow(buffer);
                for(int y=0;y<height;++y)for(int x=0;x<width;++x){const uint8_t *src=frames+((size_t(frame)*height+y)*width+x)*3;uint8_t *pixel=dst+size_t(y)*stride+x*4;pixel[0]=src[2];pixel[1]=src[1];pixel[2]=src[0];pixel[3]=255;}
                CVPixelBufferUnlockBaseAddress(buffer,0);BOOL ok=[adaptor appendPixelBuffer:buffer withPresentationTime:CMTimeMake(frame,fps)];CFRelease(buffer);
                if(!ok){[writer cancelWriting];break;}++frame;
            }
            if(frame==count||writer.status!=AVAssetWriterStatusWriting){video_done=true;[video markAsFinished];dispatch_group_leave(group);}
        }];
        if(audio){
            dispatch_group_enter(group);__block int offset=0;__block bool audio_done=false;
            [audio requestMediaDataWhenReadyOnQueue:audio_queue usingBlock:^{
                if(!accepting->load()||audio_done)return;
                while(audio.readyForMoreMediaData&&offset<samples&&writer.status==AVAssetWriterStatusWriting){
                    int chunk=std::min(1024,samples-offset);size_t bytes=size_t(chunk)*channels*sizeof(float);
                    CMBlockBufferRef block=nullptr;CMAudioFormatDescriptionRef format=nullptr;CMSampleBufferRef sample=nullptr;
                    OSStatus status=CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault,nullptr,bytes,kCFAllocatorDefault,nullptr,0,bytes,0,&block);
                    std::vector<float> interleaved(size_t(chunk)*channels);
                    for(int i=0;i<chunk;++i)for(int c=0;c<channels;++c)interleaved[size_t(i)*channels+c]=pcm[size_t(c)*samples+offset+i];
                    if(status==noErr)status=CMBlockBufferReplaceDataBytes(interleaved.data(),block,0,bytes);
                    AudioStreamBasicDescription asbd={double(rate),kAudioFormatLinearPCM,kAudioFormatFlagIsFloat|kAudioFormatFlagIsPacked,uint32_t(channels*4),1,uint32_t(channels*4),uint32_t(channels),32,0};
                    if(status==noErr)status=CMAudioFormatDescriptionCreate(kCFAllocatorDefault,&asbd,0,nullptr,0,nullptr,nullptr,&format);
                    CMSampleTimingInfo timing={CMTimeMake(1,rate),CMTimeMake(offset,rate),kCMTimeInvalid};size_t sample_size=channels*4;
                    if(status==noErr)status=CMSampleBufferCreateReady(kCFAllocatorDefault,block,format,chunk,1,&timing,1,&sample_size,&sample);
                    BOOL ok=status==noErr&&[audio appendSampleBuffer:sample];
                    if(sample)CFRelease(sample);if(format)CFRelease(format);if(block)CFRelease(block);
                    if(!ok){[writer cancelWriting];break;}offset+=chunk;
                }
                if(offset==samples||writer.status!=AVAssetWriterStatusWriting){audio_done=true;[audio markAsFinished];dispatch_group_leave(group);}
            }];
        }
        long timeout=dispatch_group_wait(group,dispatch_time(DISPATCH_TIME_NOW,120*NSEC_PER_SEC));
        // Stop and drain producer queues before releasing borrowed RGB/PCM memory.
        accepting->store(false);
        if(timeout)[writer cancelWriting];
        dispatch_sync(video_queue,^{});dispatch_sync(audio_queue,^{});
        if(timeout||writer.status!=AVAssetWriterStatusWriting){[writer cancelWriting];[[NSFileManager defaultManager]removeItemAtPath:temporary error:nil];throw std::runtime_error(writer.error?writer.error.localizedDescription.UTF8String:"media writer timeout");}
        dispatch_semaphore_t finished=dispatch_semaphore_create(0);
        [writer endSessionAtSourceTime:CMTimeMake(count,fps)];[writer finishWritingWithCompletionHandler:^{dispatch_semaphore_signal(finished);}];
        timeout=dispatch_semaphore_wait(finished,dispatch_time(DISPATCH_TIME_NOW,120*NSEC_PER_SEC));
        if(timeout||writer.status!=AVAssetWriterStatusCompleted){[writer cancelWriting];[[NSFileManager defaultManager]removeItemAtPath:temporary error:nil];throw std::runtime_error("media finalize failed");}
        std::filesystem::rename(temporary.UTF8String,output);return 1;
    }catch(const std::exception& e){failure(error,size,e.what());return 0;}}
}
}
extern "C" int h3_ffprobe_visual_size(const char *path,int *width,int *height,char *error,size_t size){@autoreleasepool{try{tc::require(path&&width&&height,"invalid media probe");CGImageRef image=visual_image(path,kCMTimeZero);tc::require(image!=nullptr,"cannot probe media");*width=int(CGImageGetWidth(image));*height=int(CGImageGetHeight(image));CGImageRelease(image);return 1;}catch(const std::exception&e){failure(error,size,e.what());return 0;}}}
extern "C" int h3_ffmpeg_read_image_f32(const char *path,int width,int height,h3_image_fit fit,float **pixels,char *error,size_t size){@autoreleasepool{try{
    tc::require(path&&pixels&&width>0&&height>0,"invalid image request");*pixels=nullptr;
    CGImageRef image=visual_image(path,kCMTimeZero);tc::require(image!=nullptr,"image decode failed");auto rgb=image_rgb(image,width,height,fit==H3_IMAGE_FIT_COVER);CGImageRelease(image);
    *pixels=(float*)malloc(size_t(width)*height*3*sizeof(float));tc::require(*pixels!=nullptr,"image allocation failed");
    for(size_t i=0;i<size_t(width)*height;++i)for(int c=0;c<3;++c)(*pixels)[size_t(c)*width*height+i]=rgb[i*3+c]/255.f;
    return 1;
}catch(const std::exception&e){failure(error,size,e.what());return 0;}}}
extern "C" int h3_ffmpeg_read_video_f32(const char *path,int width,int height,int max_frames,float **pixels,int *frames,char *error,size_t size){@autoreleasepool{try{
    tc::require(path&&pixels&&frames&&width>0&&height>0&&max_frames>=5,"invalid reference video");*pixels=nullptr;*frames=0;
    auto asset=asset_for(path);double duration=CMTimeGetSeconds(asset.duration);tc::require(std::isfinite(duration)&&duration>0,"invalid video duration");
    int count=std::min(max_frames,int(std::min(duration*24.,double(max_frames))));tc::require(count>=5,"reference video requires five frames");count=5+(count-5)/17*17;
    tc::require(uint64_t(count)*width*height<=500000000,"reference video exceeds import budget");
    std::vector<float> decoded(size_t(3)*count*width*height);
    auto generator=[[AVAssetImageGenerator alloc]initWithAsset:asset];generator.appliesPreferredTrackTransform=YES;
    AVAssetTrack *track=[asset tracksWithMediaType:AVMediaTypeVideo].firstObject;
    double source_fps=track.nominalFrameRate;
    if(!std::isfinite(source_fps)||source_fps<=0)source_fps=24;
    CMTime tolerance=CMTimeMakeWithSeconds(1.0/source_fps,60000);
    generator.requestedTimeToleranceBefore=tolerance;generator.requestedTimeToleranceAfter=tolerance;
    for(int t=0;t<count;++t){CGImageRef image=[generator copyCGImageAtTime:CMTimeMake(t,24) actualTime:nullptr error:nil];tc::require(image!=nullptr,"reference frame decode failed");auto rgb=image_rgb(image,width,height,true);CGImageRelease(image);for(size_t i=0;i<size_t(width)*height;++i)for(int c=0;c<3;++c)decoded[(size_t(c)*count+t)*width*height+i]=rgb[i*3+c]/255.f;}
    *pixels=(float*)malloc(decoded.size()*sizeof(float));tc::require(*pixels!=nullptr,"reference allocation failed");memcpy(*pixels,decoded.data(),decoded.size()*sizeof(float));*frames=count;return 1;
}catch(const std::exception&e){failure(error,size,e.what());return 0;}}}
extern "C" int h3_ffmpeg_read_audio_f32(const char *path,int max_samples,int truncate,float **pcm,int *samples,char *error,size_t size){@autoreleasepool{try{
    tc::require(path&&pcm&&samples&&max_samples>0,"invalid audio request");*pcm=nullptr;*samples=0;
    auto asset=asset_for(path);NSArray *tracks=[asset tracksWithMediaType:AVMediaTypeAudio];tc::require(tracks.count>0,"media has no audio track");
    auto reader=[[AVAssetReader alloc]initWithAsset:asset error:nil];
    auto output=[AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:tracks[0] outputSettings:@{AVFormatIDKey:@(kAudioFormatLinearPCM),AVSampleRateKey:@32000,AVNumberOfChannelsKey:@2,AVLinearPCMBitDepthKey:@32,AVLinearPCMIsFloatKey:@YES,AVLinearPCMIsNonInterleaved:@NO,AVLinearPCMIsBigEndianKey:@NO}];
    tc::require([reader canAddOutput:output],"unsupported audio conversion");[reader addOutput:output];tc::require([reader startReading],"audio reader start failed");
    std::vector<float> interleaved;
    while(CMSampleBufferRef sample=[output copyNextSampleBuffer]){
        CMBlockBufferRef buffer=CMSampleBufferGetDataBuffer(sample);size_t bytes=buffer?CMBlockBufferGetDataLength(buffer):0;
        size_t existing=interleaved.size(),wanted=bytes/sizeof(float),limit=size_t(max_samples)*2;
        if(existing+wanted>limit&&!truncate){CFRelease(sample);[reader cancelReading];throw std::invalid_argument("audio exceeds sample limit");}
        wanted=std::min(wanted,limit-existing);interleaved.resize(existing+wanted);
        OSStatus status=CMBlockBufferCopyDataBytes(buffer,0,wanted*sizeof(float),interleaved.data()+existing);CFRelease(sample);tc::require(status==noErr,"audio buffer read failed");
        if(interleaved.size()==limit){[reader cancelReading];break;}
    }
    tc::require(reader.status!=AVAssetReaderStatusFailed&&!interleaved.empty(),"audio decoding failed");
    int count=int(interleaved.size()/2);*pcm=(float*)malloc(size_t(count)*2*sizeof(float));tc::require(*pcm!=nullptr,"PCM allocation failed");
    for(int i=0;i<count;++i){(*pcm)[i]=interleaved[i*2];(*pcm)[count+i]=interleaved[i*2+1];}*samples=count;return 1;
}catch(const std::exception&e){failure(error,size,e.what());return 0;}}}
extern "C" int h3_ffmpeg_write_rgb24(const char *path,const uint8_t *frames,int count,int width,int height,int fps,char *error,size_t size){return write_av(path,frames,count,width,height,fps,nullptr,0,0,0,error,size);}
extern "C" int h3_ffmpeg_write_av_rgb24_f32(const char *path,const uint8_t *frames,int count,int width,int height,int fps,const float *pcm,int samples,int channels,int rate,char *error,size_t size){return write_av(path,frames,count,width,height,fps,pcm,samples,channels,rate,error,size);}
