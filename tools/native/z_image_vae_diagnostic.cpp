// Diagnostic-only TU: reuse production anonymous-namespace VAE operators
// without exporting them or adding synchronization to the production path.
#include "../../native/models/z_image/z_image.cpp"
#include "z_image_gpu_benchmark_lock.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: probe vae.safetensors latent.safetensors");
        ZImageGpuBenchmarkLock lock;
        tc::configure_streams();
        tc::Weights w;
        w.load_file(argv[1]); w.materialize();
        auto latent = tc::mx::load_safetensors(argv[2]).first.at("tensor");
        latent = tc::mx::astype(latent, tc::mx::bfloat16);
        tc::mx::eval(latent);
        for (int run = 0; run < 3; ++run) {
            auto start = tc::Clock::now();
            auto reference = tc::z_vae_decode(latent,w);
            tc::mx::eval(reference);
            std::cout << "{\"run\":" << run << ",\"stage\":\"unmodified_decode\",\"ms\":"
                      << std::chrono::duration<double,std::milli>(tc::Clock::now()-start).count()
                      << "}" << std::endl;
            auto timed = [&](const std::string& name, auto fn) {
                auto t = tc::Clock::now(); auto y = fn(); tc::mx::eval(y);
                std::cout << "{\"run\":" << run << ",\"stage\":\"" << name << "\",\"ms\":"
                          << std::chrono::duration<double,std::milli>(tc::Clock::now()-t).count()
                          << "}" << std::endl;
                return y;
            };
            auto x = tc::mx::reshape(latent,{1,16,latent.shape(2),latent.shape(3)});
            x = x / tc::Tensor(tc::kVaeScale,x.dtype()) + tc::Tensor(tc::kVaeShift,x.dtype());
            tc::mx::eval(x);
            x = timed("conv_in",[&] {return tc::z_conv(x,w,"decoder.conv_in");});
            x = timed("mid_block_1",[&] {return tc::z_resnet(x,w,"decoder.mid.block_1");});
            x = timed("mid_attention",[&] {return tc::z_vae_attention(x,w,"decoder.mid.attn_1");});
            x = timed("mid_block_2",[&] {return tc::z_resnet(x,w,"decoder.mid.block_2");});
            for (int stage = 3; stage >= 0; --stage) {
                for (int block = 0; block < 3; ++block) {
                    auto p = "decoder.up."+std::to_string(stage)+".block."+std::to_string(block);
                    if (stage == 0 && block == 1) {
                        auto residual = x;
                        x = timed(p+".norm1",[&] {return tc::z_group_norm(x,w,p+".norm1",32);});
                        x = timed(p+".silu1",[&] {return tc::silu(x);});
                        x = timed(p+".conv1",[&] {return tc::z_conv(x,w,p+".conv1");});
                        x = timed(p+".norm2",[&] {return tc::z_group_norm(x,w,p+".norm2",32);});
                        x = timed(p+".silu2",[&] {return tc::silu(x);});
                        x = timed(p+".conv2",[&] {return tc::z_conv(x,w,p+".conv2");});
                        x = timed(p+".residual",[&] {return x+residual;});
                    } else {
                        x = timed(p,[&] {return tc::z_resnet(x,w,p);});
                    }
                }
                if (stage > 0) {
                    auto p = "decoder.up."+std::to_string(stage)+".upsample.conv";
                    x = timed(p,[&] {return tc::z_conv(tc::mx::repeat(tc::mx::repeat(x,2,2),2,3),w,p);});
                }
            }
            x = timed("output",[&] {return tc::z_conv(tc::silu(tc::z_group_norm(x,w,"decoder.norm_out",32)),w,"decoder.conv_out");});
            const auto error = tc::mx::max(tc::mx::abs(x-reference)).item<float>();
            std::cout << "{\"run\":" << run << ",\"max_abs\":" << error << "}" << std::endl;
            if (error != 0) return 1;
        }
    } catch (const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
