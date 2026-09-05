#import <Foundation/Foundation.h>
extern "C" {
#include "../h3/vendor/h3_ffmpeg.h"
}
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
int main(int argc,char **argv) {@autoreleasepool {
    if(argc!=2)return 2;
    constexpr int w=128,h=128,n=22,rate=32000;
    int samples=int(std::round(double(n)/24*rate));
    std::vector<uint8_t> rgb(size_t(w)*h*n*3);
    for(int t=0;t<n;++t)for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        size_t i=((size_t(t)*h+y)*w+x)*3;rgb[i]=uint8_t(x);rgb[i+1]=uint8_t(y);rgb[i+2]=uint8_t(t*10);
    }
    std::vector<float> pcm(samples*2);
    for(int c=0;c<2;++c)for(int i=0;i<samples;++i)pcm[c*samples+i]=.1f*std::sin(2*3.141592653589793*(440+c*220)*i/rate);
    char error[1024]={};
    if(!h3_ffmpeg_write_av_rgb24_f32(argv[1],rgb.data(),n,w,h,24,pcm.data(),samples,2,rate,error,sizeof(error))){fprintf(stderr,"write: %s\n",error);return 1;}
    int width=0,height=0;
    if(!h3_ffprobe_visual_size(argv[1],&width,&height,error,sizeof(error))||width!=w||height!=h){fprintf(stderr,"probe: %s\n",error);return 1;}
    float *decoded=nullptr;int frames=0;
    if(!h3_ffmpeg_read_video_f32(argv[1],w,h,n,&decoded,&frames,error,sizeof(error))||frames!=n){fprintf(stderr,"video: %s frames=%d\n",error,frames);free(decoded);return 1;}
    double square=0;
    for(int t=0;t<n;++t)for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c) {
        double difference=decoded[((size_t(c)*n+t)*h+y)*w+x]-rgb[((size_t(t)*h+y)*w+x)*3+c]/255.;square+=difference*difference;
    }
    free(decoded);double rmse=std::sqrt(square/rgb.size());if(rmse>.025){fprintf(stderr,"video RMSE %.5f\n",rmse);return 1;}
    int count=0;if(!h3_ffmpeg_read_audio_f32(argv[1],samples+2048,0,&decoded,&count,error,sizeof(error))){fprintf(stderr,"audio: %s\n",error);return 1;}
    double energy=0;for(int i=0;i<count*2;++i)energy+=decoded[i]*decoded[i];free(decoded);
    if(std::abs(count-samples)>2048||energy/count<.001){fprintf(stderr,"PCM invalid count=%d energy=%g\n",count,energy/count);return 1;}
    printf("{\"passed\":true,\"frames\":%d,\"audio_samples\":%d,\"video_rmse\":%.6f}\n",frames,count,rmse);return 0;
}}
