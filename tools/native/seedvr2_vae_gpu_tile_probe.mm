// Native MLX completion-timed control for the offline Core ML tile screen.
#import <Foundation/Foundation.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mx=mlx::core;
namespace {
using Clock=std::chrono::steady_clock;
void check(bool value,const char *why){if(!value)throw std::runtime_error(why);}
}
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    @autoreleasepool {
        try {
            const std::string route=argv[2];
            check(route=="conv3d" || route=="conv2d" || route=="pack_conv2d" || route=="matmul","unknown GPU route");
            NSError *error=nil;
            NSData *data=[NSData dataWithContentsOfFile:@(argv[1])];
            check(data && data.length<65536,"bounded fixture missing");
            NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            check([spec[@"schema"] isEqual:@"seedvr2-vae-native-io-fixture-v1"],"fixture schema mismatch");
            const int cin=[spec[@"cin_per_tap"] intValue],taps=[spec[@"taps"] intValue],ks=[spec[@"ks"] intValue],
                cout=[spec[@"cout"] intValue],hi=[spec[@"input_rows"] intValue],wi=[spec[@"input_cols"] intValue],
                ho=[spec[@"rows"] intValue],wo=[spec[@"cols"] intValue],warm=[spec[@"iterations"] intValue];
            check(cin>0 && cin<=512 && cout>0 && cout<=4096 && (taps==1 || taps==3) && (ks==1 || ks==3) &&
                  ho>0 && ho<=256 && wo>0 && wo<=256 && hi==ho+ks-1 && wi==wo+ks-1 && warm>=3 && warm<=30,
                  "invalid fixed GPU geometry");
            check(route!="matmul" || (taps==1 && ks==1),"matmul requires a pointwise projection");
            NSData *input=[NSData dataWithContentsOfFile:spec[@"gpu_input"]],*weights=[NSData dataWithContentsOfFile:spec[@"gpu_weight"]],
                *reference=[NSData dataWithContentsOfFile:spec[@"reference"]];
            check(input.length==uint64_t(cin)*taps*hi*wi*2 && weights.length==uint64_t(cout)*cin*taps*ks*ks*2 &&
                  reference.length==uint64_t(cout)*ho*wo*4,"fixture extent differs");
            mx::set_default_device(mx::Device::gpu);
            mx::set_cache_limit(512ull<<20);
            mx::array x(static_cast<const mx::float16_t *>(input.bytes),{1,taps,hi,wi,cin},mx::float16),
                w(static_cast<const mx::float16_t *>(weights.bytes),{cout,taps,ks,ks,cin},mx::float16);
            auto pack=[&]() {return mx::reshape(mx::contiguous(mx::transpose(x,{0,2,3,1,4})),{1,hi,wi,taps*cin});};
            auto x2d=pack(),w2d=mx::reshape(mx::contiguous(mx::transpose(w,{0,2,3,1,4})),{cout,ks,ks,taps*cin});
            mx::eval({x,w,x2d,w2d});
            NSMutableArray *samples=[NSMutableArray array];
            std::vector<double> times;
            mx::array y(0);
            for(int iteration=0;iteration<warm+2;++iteration) {
                auto start=Clock::now();
                if(route=="conv3d")y=mx::conv3d(x,w);
                else if(route=="matmul")y=mx::reshape(mx::matmul(mx::reshape(x,{-1,cin}),
                    mx::transpose(mx::reshape(w,{cout,cin}))),{1,ho,wo,cout});
                else y=mx::conv2d(route=="conv2d"?x2d:pack(),w2d);
                mx::eval(y);
                const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
                if(iteration>=2){times.push_back(seconds);[samples addObject:@(seconds)];}
            }
            const auto *actual=y.data<mx::float16_t>();
            const auto *expected=static_cast<const float *>(reference.bytes);
            double nx=0,ny=0,dot=0,es=0,ma=0;
            for(int c=0;c<cout;++c)for(int r=0;r<ho;++r)for(int col=0;col<wo;++col) {
                const double a=float(actual[(size_t(r)*wo+col)*cout+c]),b=expected[(size_t(c)*ho+r)*wo+col];
                check(std::isfinite(a) && std::isfinite(b),"nonfinite GPU/reference value");
                nx+=b*b;ny+=a*a;dot+=a*b;es+=(a-b)*(a-b);ma=std::max(ma,std::abs(a-b));
            }
            const double l2=std::sqrt(es/std::max(nx,1e-20)),cosine=dot/std::max(std::sqrt(nx*ny),1e-20);
            check(l2<=.05 && cosine>=.999,"unchanged GPU component quality gate failed");
            std::sort(times.begin(),times.end());
            NSDictionary *report=@{@"schema":@"seedvr2-vae-native-gpu-tile-observation-v1",@"route":@(route.c_str()),
                @"warm_seconds":samples,@"warm_median_seconds":@((times[(times.size()-1)/2]+times[times.size()/2])/2),
                @"quality":@{@"passed":@YES,@"finite":@YES,@"relative_l2":@(l2),@"cosine":@(cosine),@"max_abs":@(ma)},
                @"active_peak_bytes":@(mx::get_peak_memory()),@"qualified":@NO,
                @"scope":@"native dispatch/completion, resident operands; pack_conv2d includes GPU packing; no Core ML IOSurface pack/sync/merge"};
            NSData *encoded=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingSortedKeys error:&error];
            check(encoded!=nil,"GPU report serialization failed");
            std::cout<<[[[NSString alloc] initWithData:encoded encoding:NSUTF8StringEncoding] UTF8String]<<'\n';
        }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
    }
}
