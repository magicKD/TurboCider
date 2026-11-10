/* Complete-frame RGB diagnostics using the production AVFoundation reader.
 * Pixel similarity/motion energy are descriptive, not semantic quality proof. */
#include "../../native/media/rgb_video_reader.hpp"
#include "../../native/media/video.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace {
struct Metrics {
    double correlation, cosine, mae, rmse, reference_std, candidate_std;
};
Metrics compare(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    tc::require(!a.empty() && a.size() == b.size(), "RGB extents differ");
    double sa = 0, sb = 0, aa = 0, bb = 0, ab = 0, absolute = 0, delta2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = a[i], y = b[i], d = x - y;
        sa += x; sb += y; aa += x*x; bb += y*y; ab += x*y;
        absolute += std::abs(d); delta2 += d*d;
    }
    const double count = a.size();
    double va = std::max(0., aa - sa*sa/count), vb = std::max(0., bb - sb*sb/count);
    return {(va && vb) ? (ab-sa*sb/count)/std::sqrt(va*vb) : double(a == b),
            (aa && bb) ? ab/std::sqrt(aa*bb) : double(a == b), absolute/count,
            std::sqrt(delta2/count), std::sqrt(va/count), std::sqrt(vb/count)};
}
double motion(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    return compare(a, b).mae;
}
void ppm(const std::filesystem::path& path, const tc::RGBVideoFrame& frame) {
    std::ofstream out(path, std::ios::binary);
    tc::require(out.good(), "cannot create sample frame");
    out << "P6\n" << frame.width << " " << frame.height << "\n255\n";
    out.write(reinterpret_cast<const char *>(frame.pixels.data()), frame.pixels.size());
    tc::require(out.good(), "cannot write sample frame");
}
void self_test(const std::filesystem::path& directory) {
    tc::require(!std::filesystem::exists(directory), "fresh self-test directory required");
    std::filesystem::create_directories(directory);
    auto equal = compare({0, 10, 20, 255}, {0, 10, 20, 255});
    auto shifted = compare({10, 20, 30, 40}, {12, 22, 32, 42});
    auto black = compare({0, 0, 0}, {255, 255, 255});
    tc::require(equal.mae == 0 && equal.rmse == 0 && equal.cosine == 1 && equal.correlation == 1,
                "identical RGB metric self-test failed");
    tc::require(shifted.mae == 2 && shifted.rmse == 2 && std::abs(shifted.correlation-1) < 1e-12,
                "offset RGB metric self-test failed");
    tc::require(black.mae == 255 && black.correlation == 0 && black.cosine == 0,
                "constant/different RGB metric self-test failed");
    constexpr int width = 64, height = 64, frames = 9, fps = 24;
    std::vector<uint8_t> rgb(width*height*frames*3);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t((i*7+i/(width*height*3)*23)%256);
    auto path = directory / "fixture.mp4";
    tc::write_video_rgb24(path, rgb.data(), frames, width, height, fps);
    std::atomic<bool> cancelled{false};
    tc::RGB24VideoReader a(path,frames,fps,{},cancelled), b(path,frames,fps,{},cancelled);
    tc::RGBVideoFrame x,y;
    int count = 0;
    while (a.next(x)) {
        tc::require(b.next(y) && x.pixels == y.pixels, "native decoded identity failed");
        ++count;
    }
    tc::require(!b.next(y) && count == frames, "native complete-frame decode failed");
    std::cout << "PASS native video metrics: RGB identity/offset/constant and all-frame CFR decode\n";
}
}

int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--self-test") { self_test(argv[2]); return 0; }
        tc::require(argc == 5 || argc == 6, "usage: compare-video-native REFERENCE CANDIDATE FRAMES FPS [FRESH_SAMPLE_DIR]");
        int frames = std::stoi(argv[3]), fps = std::stoi(argv[4]);
        std::filesystem::path sample_directory;
        if (argc == 6) {
            sample_directory = argv[5];
            tc::require(!std::filesystem::exists(sample_directory), "fresh sample directory required");
            std::filesystem::create_directories(sample_directory);
        }
        std::atomic<bool> cancelled{false};
        tc::RGB24VideoReader left(argv[1],frames,fps,{},cancelled), right(argv[2],frames,fps,{},cancelled);
        tc::RGBVideoFrame a,b;
        std::vector<uint8_t> previous_a, previous_b;
        @autoreleasepool {
            NSMutableArray *rows = [NSMutableArray new];
            double correlation_sum=0, cosine_sum=0, mae_sum=0, reference_motion_sum=0, candidate_motion_sum=0;
            double min_correlation=1, min_cosine=1, max_mae=0;
            int width=0,height=0;
            while (left.next(a)) {
                tc::require(right.next(b), "candidate video ended early");
                tc::require(a.width == b.width && a.height == b.height && a.index == b.index,
                            "video frame geometry/index differs");
                auto m = compare(a.pixels,b.pixels);
                width=a.width; height=a.height;
                correlation_sum+=m.correlation; cosine_sum+=m.cosine; mae_sum+=m.mae;
                min_correlation=std::min(min_correlation,m.correlation);
                min_cosine=std::min(min_cosine,m.cosine); max_mae=std::max(max_mae,m.mae);
                NSMutableDictionary *row = [@{@"frame":@(a.index), @"correlation":@(m.correlation),
                    @"cosine":@(m.cosine), @"mae_255":@(m.mae), @"rmse_255":@(m.rmse),
                    @"reference_rgb_std":@(m.reference_std), @"candidate_rgb_std":@(m.candidate_std)} mutableCopy];
                if (!previous_a.empty()) {
                    double ma=motion(previous_a,a.pixels), mb=motion(previous_b,b.pixels);
                    row[@"reference_motion_energy_255"]=@(ma); row[@"candidate_motion_energy_255"]=@(mb);
                    reference_motion_sum+=ma; candidate_motion_sum+=mb;
                }
                [rows addObject:row];
                if (!sample_directory.empty() && (a.index % 24 == 0 || a.index == frames-1)) {
                    ppm(sample_directory/("reference-"+std::to_string(a.index)+".ppm"),a);
                    ppm(sample_directory/("candidate-"+std::to_string(b.index)+".ppm"),b);
                }
                previous_a=a.pixels; previous_b=b.pixels;
            }
            tc::require(!right.next(b) && left.frames_read() == frames && right.frames_read() == frames,
                        "decoded frame count differs from requested workload");
            NSDictionary *report = @{@"schema":@"tc-native-rgb-video-comparison-v1",
                @"scope":@"all decoded RGB frames; descriptive pixel/motion metrics, not semantic qualification",
                @"reference":@(argv[1]), @"candidate":@(argv[2]), @"width":@(width), @"height":@(height),
                @"frames":@(frames), @"fps":@(fps), @"complete":@YES, @"frame_metrics":rows,
                @"mean_correlation":@(correlation_sum/frames), @"minimum_correlation":@(min_correlation),
                @"mean_cosine":@(cosine_sum/frames), @"minimum_cosine":@(min_cosine),
                @"mean_mae_255":@(mae_sum/frames), @"maximum_mae_255":@(max_mae),
                @"mean_motion_energy_reference":@(reference_motion_sum/std::max(1,frames-1)),
                @"mean_motion_energy_candidate":@(candidate_motion_sum/std::max(1,frames-1))};
            NSError *failure=nil;
            NSData *data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&failure];
            tc::require(data != nil, failure ? failure.localizedDescription.UTF8String : "cannot serialize RGB metrics");
            std::cout.write(static_cast<const char *>(data.bytes),data.length);std::cout<<'\n';
        }
        return 0;
    } catch (const std::exception& failure) { std::cerr<<failure.what()<<'\n';return 1; }
}
