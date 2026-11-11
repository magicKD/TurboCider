#include "ltx_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static char error[2048];
static void check(bool ok) {
    if (!ok) throw std::runtime_error(error[0] ? error : "K/V pool assertion failed");
}
static uint16_t bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float f32(uint16_t x) {
    uint32_t u = uint32_t(x) << 16; float value; std::memcpy(&value, &u, 4); return value;
}
using Buffer = std::unique_ptr<ltx_gpu_buffer, decltype(&ltx_gpu_buffer_free)>;
static Buffer buffer(ltx_gpu *gpu, const std::vector<uint16_t> &data) {
    auto *b = ltx_gpu_buffer_new_copy(gpu, data.data(), data.size() * 2, error, sizeof(error));
    check(b != nullptr); return {b, ltx_gpu_buffer_free};
}

int main(int argc, char **argv) {
    try {
        check(argc == 2);
        std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)> gpu(
            ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        check(bool(gpu));
        if (!ltx_gpu_convrot_w8a8_info(gpu.get(), nullptr, nullptr)) {
            std::puts("SKIP spatial K/V pool requires M5 W8A8"); return 0;
        }
        constexpr unsigned frames=2, height=4, width=4, rows=32, heads=2, dim=128;
        constexpr size_t count = size_t(rows)*heads*dim;
        std::vector<uint16_t> q(count), k(count), v(count), gate(rows*heads),
            cosine(count/2, bf16(1)), sine(count/2), zero(count);
        for (unsigned row=0; row<rows; ++row) {
            unsigned frame=row/(height*width), y=row/width%height, x=row%width;
            bool high = y%2==0 && x%2==0;
            for (unsigned head=0; head<heads; ++head) {
                size_t base=(size_t(row)*heads+head)*dim;
                q[base]=bf16(8.f);
                k[base]=bf16((high ? 2.f : -2.f)+2.f*frame);
                for (unsigned d=0; d<dim; ++d)
                    v[base+d]=bf16((high ? 1.f : -1.f)*(frame+1.f));
                gate[row*heads+head]=bf16(.5f+.5f*(row%4));
            }
        }
        auto qb=buffer(gpu.get(),q), kb=buffer(gpu.get(),k), vb=buffer(gpu.get(),v);
        auto cb=buffer(gpu.get(),cosine), sb=buffer(gpu.get(),sine), gb=buffer(gpu.get(),gate);
        auto out=buffer(gpu.get(),zero);
        auto run = [&]() {
            check(ltx_gpu_self_attention_core_mps_bf16(gpu.get(),out.get(),qb.get(),kb.get(),vb.get(),
                cb.get(),sb.get(),gb.get(),rows,heads,dim,1.f/std::sqrt(float(dim)),error,sizeof(error)));
            std::vector<uint16_t> values(count);
            check(ltx_gpu_buffer_read(out.get(),values.data(),count*2,error,sizeof(error)));return values;
        };
        auto dense=run();
        check(!ltx_gpu_set_self_kv_pool_geometry(gpu.get(),2,3,4,error,sizeof(error)));
        error[0]=0;
        check(ltx_gpu_set_self_kv_pool_geometry(gpu.get(),frames,height,width,error,sizeof(error)));
        check(!ltx_gpu_set_self_kv_pool_geometry(gpu.get(),frames,height,width,error,sizeof(error)));
        error[0]=0;
        auto pooled=run();
        check(ltx_gpu_set_self_kv_pool_geometry(gpu.get(),0,0,0,error,sizeof(error)));
        auto restored=run();
        check(restored==dense); // Cache keys/scoped state must preserve the dense path.
        // Independent FP64 oracle: every spatial 2x2 K mean is -1 in frame 0,
        // +1 in frame 1; V means are -.5 and -1. Time is never pooled.
        double score=8./std::sqrt(double(dim));
        double probability=std::exp(score)/(std::exp(score)+std::exp(-score));
        double attention=-.5*(1-probability)-probability;
        double error2=0, norm2=0, difference2=0;
        for (size_t i=0;i<count;++i) {
            double expected=attention*f32(gate[i/dim]);
            double actual=f32(pooled[i]);check(std::isfinite(actual));
            error2+=(actual-expected)*(actual-expected);norm2+=expected*expected;
            double delta=actual-f32(dense[i]);difference2+=delta*delta;
        }
        double relative=std::sqrt(error2/norm2), difference=std::sqrt(difference2/count);
        check(relative<.012 && difference>.1);
        std::printf("PASS spatial K/V pool: FP64 oracle relL2=%.6g; dense difference RMS=%.6g; "
                    "all queries/gates preserved, temporal geometry and dense cache restored\n",relative,difference);
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
