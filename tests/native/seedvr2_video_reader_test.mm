#include "../../native/media/rgb_video_reader.hpp"
#include "../../native/media/video_input.hpp"
#include "../../native/media/video.hpp"
#include "../../native/models/seedvr2/media.hpp"
#import <AVFoundation/AVFoundation.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>

namespace mx=mlx::core;
namespace {
void check(bool value,const char *why){if(!value)throw std::runtime_error(why);}
template<class F>void rejects(F function){bool failed=false;try{function();}catch(const std::exception &){failed=true;}check(failed,"invalid reader input accepted");}
void transformed_copy(const std::filesystem::path &source,const std::filesystem::path &destination,CGAffineTransform transform) {
    @autoreleasepool {
        auto asset=[AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(source.c_str())] options:nil];
        dispatch_semaphore_t loaded=dispatch_semaphore_create(0);
        __block NSArray<AVAssetTrack *> *tracks=nil;
        [asset loadTracksWithMediaType:AVMediaTypeVideo completionHandler:^(NSArray<AVAssetTrack *> *value,NSError *){tracks=value;dispatch_semaphore_signal(loaded);}];
        check(dispatch_semaphore_wait(loaded,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))==0 && tracks.count,"fixture tracks unavailable");
        auto composition=[AVMutableComposition composition];
        auto track=[composition addMutableTrackWithMediaType:AVMediaTypeVideo preferredTrackID:kCMPersistentTrackID_Invalid];
        NSError *error=nil;
        check([track insertTimeRange:CMTimeRangeMake(kCMTimeZero,asset.duration) ofTrack:tracks.firstObject atTime:kCMTimeZero error:&error],"fixture track insert failed");
        track.preferredTransform=transform;
        auto exporter=[[AVAssetExportSession alloc] initWithAsset:composition presetName:AVAssetExportPresetPassthrough];
        check(exporter!=nil,"fixture exporter unavailable");
        exporter.outputURL=[NSURL fileURLWithPath:@(destination.c_str())];exporter.outputFileType=AVFileTypeMPEG4;
        dispatch_semaphore_t ready=dispatch_semaphore_create(0);
        [exporter exportAsynchronouslyWithCompletionHandler:^{dispatch_semaphore_signal(ready);}];
        check(dispatch_semaphore_wait(ready,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))==0 && exporter.status==AVAssetExportSessionStatusCompleted,
              "fixture transform export failed");
    }
}
void variable_rate_fixture(const std::filesystem::path &path) {
    @autoreleasepool {
        NSError *error=nil;
        auto writer=[[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:@(path.c_str())] fileType:AVFileTypeMPEG4 error:&error];
        check(writer!=nil,"VFR fixture writer unavailable");
        auto input=[AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:@{
            AVVideoCodecKey:AVVideoCodecTypeH264,AVVideoWidthKey:@64,AVVideoHeightKey:@64,
            AVVideoCompressionPropertiesKey:@{AVVideoAllowFrameReorderingKey:@NO,AVVideoExpectedSourceFrameRateKey:@24}}];
        input.mediaTimeScale=24;writer.movieTimeScale=24;input.expectsMediaDataInRealTime=NO;
        auto adaptor=[AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
            sourcePixelBufferAttributes:@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferWidthKey:@64,(id)kCVPixelBufferHeightKey:@64,(id)kCVPixelBufferIOSurfacePropertiesKey:@{}}];
        check([writer canAddInput:input],"VFR fixture settings unsupported");[writer addInput:input];
        check([writer startWriting],"VFR fixture writer did not start");[writer startSessionAtSourceTime:kCMTimeZero];
        for(int index=0;index<9;++index){
            const auto deadline=tc::Clock::now()+std::chrono::seconds(10);
            while(!input.readyForMoreMediaData){check(tc::Clock::now()<deadline,"VFR fixture producer timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
            CVPixelBufferRef pixel=nullptr;check(CVPixelBufferPoolCreatePixelBuffer(nullptr,adaptor.pixelBufferPool,&pixel)==kCVReturnSuccess,"VFR fixture buffer unavailable");
            check(CVPixelBufferLockBaseAddress(pixel,0)==kCVReturnSuccess,"VFR fixture buffer lock failed");
            std::memset(CVPixelBufferGetBaseAddress(pixel),index*20,CVPixelBufferGetBytesPerRow(pixel)*64);
            CVPixelBufferUnlockBaseAddress(pixel,0);
            const bool appended=[adaptor appendPixelBuffer:pixel withPresentationTime:CMTimeMake(index+(index>=5?1:0),24)];CFRelease(pixel);
            check(appended,"VFR fixture append failed");
        }
        [input markAsFinished];[writer endSessionAtSourceTime:CMTimeMake(10,24)];
        dispatch_semaphore_t done=dispatch_semaphore_create(0);[writer finishWritingWithCompletionHandler:^{dispatch_semaphore_signal(done);}];
        check(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))==0 && writer.status==AVAssetWriterStatusCompleted,
            "VFR fixture did not finish");
    }
}
}
int main(int argc,char **argv){
    try {
        check(argc==2,"fresh reader-test directory required");
        const std::filesystem::path root(argv[1]);check(!std::filesystem::exists(root),"fresh reader-test path required");
        std::filesystem::create_directories(root);mx::set_default_device(mx::Device::gpu);
        constexpr int width=96,height=64,frames=9,fps=24;
        std::vector<uint8_t> rgb(size_t(frames)*width*height*3);
        for(int f=0;f<frames;++f)for(int y=0;y<height;++y)for(int x=0;x<width;++x){
            auto *p=rgb.data()+((size_t(f)*height+y)*width+x)*3;
            p[0]=uint8_t((x*3+f*23)%256);p[1]=uint8_t((y*5+f*31)%256);p[2]=uint8_t((x+y*2+f*17)%256);}
        const auto base=root/"base.mp4";tc::write_video_rgb24(base,rgb.data(),frames,width,height,fps);
        std::atomic<bool> cancelled{false};int orientations=0,matched=0,normalized=0;
        const std::vector<CGAffineTransform> transforms{
            CGAffineTransformMake(1,0,0,1,0,0),CGAffineTransformMake(-1,0,0,1,width,0),
            CGAffineTransformMake(1,0,0,-1,0,height),CGAffineTransformMake(-1,0,0,-1,width,height),
            CGAffineTransformMake(0,1,-1,0,height,0),CGAffineTransformMake(0,-1,1,0,0,width),
            CGAffineTransformMake(0,1,1,0,0,0),CGAffineTransformMake(0,-1,-1,0,height,width)};
        for(size_t orientation=0;orientation<transforms.size();++orientation){
            const auto path=root/("orientation-"+std::to_string(orientation)+".mp4");
            transformed_copy(base,path,transforms[orientation]);
            auto legacy=tc::load_video_tensor(path,frames,fps,{},cancelled);
            auto expected=mx::astype(mx::round(legacy*255.f),mx::uint8);expected=mx::contiguous(expected);mx::eval(expected);
            auto resized=tc::seedvr2::resize_bicubic(legacy,128,128);
            auto old_normalized=tc::seedvr2::consume_resized_input(resized,true).pixels;
            auto normalized_bytes=mx::contiguous(mx::transpose(old_normalized,{0,2,3,4,1}));mx::eval(normalized_bytes);
            tc::RGB24VideoReader reader(path,frames,fps,{},cancelled);tc::RGBVideoFrame frame;
            const uint8_t *allocation=nullptr;
            for(int index=0;index<frames;++index){
                check(reader.next(frame),"streaming reader truncated a valid video");
                check(frame.index==index && frame.height==expected.shape(1) && frame.width==expected.shape(2),"reader display geometry/index changed");
                check(frame.pixels.size()==size_t(frame.height)*frame.width*3,"reader frame extent differs");
                if(index==0)allocation=frame.pixels.data();else check(allocation==frame.pixels.data(),"reader RGB storage was not reused");
                check(std::memcmp(frame.pixels.data(),expected.data<uint8_t>()+size_t(index)*frame.pixels.size(),frame.pixels.size())==0,
                      "streaming/legacy decoded display pixels are not bitwise equal");++matched;
                auto unit=mx::astype(tc::Tensor(frame.pixels.data(),{1,frame.height,frame.width,3},mx::uint8),mx::float32)/255.f;
                mx::eval(unit);unit=tc::seedvr2::resize_bicubic(unit,128,128);
                auto new_normalized=tc::seedvr2::consume_resized_input(unit,true).pixels;
                auto new_bytes=mx::contiguous(mx::transpose(new_normalized,{0,2,3,4,1}));mx::eval(new_bytes);
                check(std::memcmp(new_bytes.data<float>(),normalized_bytes.data<float>()+size_t(index)*128*128*3,
                    size_t(128)*128*3*4)==0,"per-frame FP32 resize/normalization differs bitwise from the buffered recipe");++normalized;
            }
            check(!reader.next(frame) && !reader.next(frame) && reader.frames_read()==frames,"reader EOF/count contract differs");++orientations;
        }
        for(int count:{8,10})rejects([&]{tc::RGB24VideoReader reader(base,count,fps,{},cancelled);tc::RGBVideoFrame f;while(reader.next(f)){};});
        rejects([&]{tc::RGB24VideoReader reader(base,frames,16,{},cancelled);});
        rejects([&]{tc::RGB24VideoReader reader(base,0,fps,{},cancelled);});
        const auto vfr=root/"variable-rate.mp4";variable_rate_fixture(vfr);
        rejects([&]{tc::RGB24VideoReader reader(vfr,9,24,{},cancelled);tc::RGBVideoFrame frame;while(reader.next(frame)){};});
        const auto changed=root/"source-changed.mp4";std::filesystem::copy_file(base,changed);
        rejects([&]{tc::RGB24VideoReader reader(changed,frames,fps,{},cancelled);tc::RGBVideoFrame frame;check(reader.next(frame),"mutation fixture unreadable");
            std::filesystem::last_write_time(changed,std::filesystem::last_write_time(changed)+std::chrono::seconds(1));while(reader.next(frame)){};});
        const auto bad_transform=root/"unsupported-transform.mp4";
        transformed_copy(base,bad_transform,CGAffineTransformMake(1.25,0,0,1,0,0));
        rejects([&]{tc::RGB24VideoReader reader(bad_transform,frames,fps,{},cancelled);});
        cancelled=true;bool typed=false;
        try{tc::RGB24VideoReader reader(base,frames,fps,{},cancelled);}catch(const tc::Cancelled &){typed=true;}
        check(typed,"pre-start cancellation lost its type");cancelled=false;
        {
            tc::RGB24VideoReader reader(base,frames,fps,[&](const std::string &,int step,int){if(step==1)cancelled=true;},cancelled);
            tc::RGBVideoFrame frame;typed=false;try{reader.next(frame);}catch(const tc::Cancelled &){typed=true;}
            check(typed && reader.frames_read()==1,"mid-source cancellation lost its type or read ahead");
        }
        cancelled=false;
        const auto long_path=root/"long1000.mp4";
        {
            tc::RGB24VideoWriter writer(long_path,1000,width,height,fps);
            for(int index=0;index<1000;++index)writer.append(index,rgb.data(),size_t(width)*height*3);
            writer.finish();
        }
        size_t capacity=0;int read=0;
        {
            tc::RGB24VideoReader reader(long_path,1000,fps,{},cancelled);tc::RGBVideoFrame frame;
            while(reader.next(frame)){++read;if(read==1)capacity=frame.pixels.capacity();check(frame.pixels.capacity()==capacity,"long reader RGB storage grew with frame count");}
            check(read==1000 && reader.frames_read()==1000,"long reader did not consume complete source");
        }
        // The reader's longer-clip contract does not change the model's cap.
        rejects([&]{tc::load_video_tensor(long_path,1000,fps,{},cancelled);});
        std::ofstream report(root/"report.json");check(bool(report),"cannot save reader report");
        report<<"{\"schema\":\"seedvr2-rgb-reader-screen-v1\",\"passed\":true,\"orientations\":"<<orientations
            <<",\"bitwise_matched_frames\":"<<matched<<",\"long_reader_frames\":"<<read<<",\"long_rgb_capacity_bytes\":"<<capacity
            <<",\"bitwise_normalized_frames\":"<<normalized
            <<",\"typed_cancellation_passed\":true,\"declared_count_fps_transform_guards_passed\":true,\"vfr_and_source_mutation_rejected\":true,\"scope\":\"reader only; no long-model or full-memory qualification\"}\n";
        std::cout<<"PASS reader orientations="<<orientations<<" bitwise_frames="<<matched<<" long_frames="<<read<<'\n';
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
