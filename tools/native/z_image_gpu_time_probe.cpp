// Diagnostic only: compare synchronized wall time with the final Metal command
// buffer's execution timestamps. Not a substitute for whole-model timings.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <mlx/backend/metal/device.h>
#include "../../native/models/z_image/metal_kernels.hpp"
#include "z_image_gpu_benchmark_lock.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>

namespace mx = mlx::core;
struct Timestamp {
    MTL::CommandBuffer *buffer = nullptr;
    ~Timestamp() { if (buffer) buffer->release(); }
};

class CaptureBuffer final : public mx::Primitive {
    std::shared_ptr<Timestamp> timestamp_;
public:
    CaptureBuffer(mx::Stream stream, std::shared_ptr<Timestamp> timestamp)
        : Primitive(stream), timestamp_(std::move(timestamp)) {}
    const char *name() const override { return "ZCaptureBuffer"; }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("GPU-only timestamp marker");
    }
    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        outputs[0].copy_shared_buffer(inputs[0]);
        auto *buffer = mx::metal::get_command_encoder(stream()).get_command_buffer();
        if (!buffer) throw std::runtime_error("no current Metal command buffer");
        timestamp_->buffer = buffer->retain();
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());
    return (v[(v.size()-1)/2] + v[v.size()/2]) * .5;
}

int main(int argc, char **argv) {
    try {
        ZImageGpuBenchmarkLock benchmark_lock;
        // The default 50 MB buffer boundary can commit a GEMM before the
        // marker runs, leaving it an empty buffer. This diagnostic uses a
        // larger boundary and reports it; do not confuse it with defaults.
        if (!std::getenv("MLX_MAX_MB_PER_BUFFER"))
            setenv("MLX_MAX_MB_PER_BUFFER", "1024", 0);
        const int rows = argc > 1 ? std::stoi(argv[1]) : 1056;
        const int runs = argc > 2 ? std::stoi(argv[2]) : 10;
        if (rows < 1 || runs < 1) throw std::invalid_argument("positive rows/runs required");
        auto cpu = mx::new_thread_unsafe_stream(mx::Device(mx::Device::cpu));
        auto gpu = mx::new_thread_unsafe_stream(mx::Device(mx::Device::gpu));
        mx::set_default_stream(cpu); mx::set_default_stream(gpu);
        mx::set_default_device(mx::Device::gpu);
        auto [max_ops,max_mb] = mx::metal::device(mx::Device(mx::Device::gpu)).get_max_ops_mb_per_buffer();
        std::cerr << "command_buffer_limits ops=" << max_ops << " mb=" << max_mb << '\n';
        mx::random::seed(42);
        for (auto shape : {mx::Shape{11520,3840}, mx::Shape{3840,3840},
                           mx::Shape{10240,3840}, mx::Shape{3840,10240}}) {
            auto x = mx::random::normal({1,rows,shape[1]},mx::bfloat16);
            auto w = mx::random::normal(shape,mx::bfloat16);
            mx::eval(x,w);
            for (bool mpp : {false,true}) {
                auto fn = mx::compile([mpp](const std::vector<mx::array>& a) {
                    return std::vector<mx::array>{mpp ? tc::z_metal::projection(a[0],a[1])
                        : mx::matmul(a[0],mx::transpose(a[1]))};
                });
                std::vector<double> wall, device;
                for (int i = 0; i < runs + 3; ++i) {
                    auto stamp = std::make_shared<Timestamp>();
                    auto start = std::chrono::steady_clock::now();
                    auto y = fn({x,w})[0];
                    auto out = mx::array(y.shape(),y.dtype(),
                        std::make_shared<CaptureBuffer>(gpu,stamp),{y});
                    mx::eval(out);
                    stamp->buffer->waitUntilCompleted();
                    const double elapsed = std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count();
                    const double gpu_ms = 1000. * (stamp->buffer->GPUEndTime() -
                                                  stamp->buffer->GPUStartTime());
                    if (gpu_ms <= 0) {
                        std::cerr << "buffer_status=" << stamp->buffer->status()
                                  << " gpu_start=" << stamp->buffer->GPUStartTime()
                                  << " gpu_end=" << stamp->buffer->GPUEndTime() << '\n';
                        throw std::runtime_error("invalid GPU timestamps");
                    }
                    if (i >= 3) { wall.push_back(elapsed); device.push_back(gpu_ms); }
                }
                std::cout << "{\"rows\":" << rows << ",\"n\":" << shape[0]
                          << ",\"k\":" << shape[1] << ",\"mpp\":" << (mpp?"true":"false")
                          << ",\"wall_ms\":" << median(wall)
                          << ",\"last_command_buffer_gpu_ms\":" << median(device) << "}\n";
                std::cout.flush();
            }
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
