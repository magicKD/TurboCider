#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <iostream>
#include <fstream>
namespace mx = mlx::core;
int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: upscale-capture MODEL LATENT OUTPUT");
        const std::filesystem::path output(argv[3]);
        tc::require(!std::filesystem::exists(output), "fresh capture directory required");
        auto values = mx::load_safetensors(argv[2]).first;
        auto latent = values.at("latent"); values.clear();
        tc::require(latent.shape() == mx::Shape{1,16,7,128,128} &&
                    (latent.dtype() == mx::float16 || latent.dtype() == mx::float32),
                    "complete restored 1024/t25 latent required");
        std::filesystem::create_directories(output);
        mx::set_default_device(mx::Device::gpu);
        tc::seedvr2::AllocatorCacheScope allocator(1ull << 30);
        tc::seedvr2::VAE vae;
        vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config; config.history="frames"; vae.configure(config);
        int observed=0, captured=0, emitted=0;
        vae.observe_upscale_inputs([&](const std::string &key, const tc::Tensor &input, bool initial, bool temporal) {
            if (key != "decoder.up_blocks.2.upsamplers.0") return;
            tc::require(!temporal && initial == (observed == 0) &&
                input.shape() == mx::Shape{1,1,512,512,256} && input.dtype() == mx::float16,
                "captured spatial projection phase/shape changed");
            if (observed == 0 || observed == 24) {
                auto owned=mx::full(input.shape(),input,input.dtype()); mx::eval(owned);
                tc::require(mx::all(mx::isfinite(owned)).item<bool>(),"captured input is nonfinite");
                mx::save_safetensors(output/("input-frame-"+std::to_string(observed)+".safetensors"),
                    {{"input",owned}},{{"layer",key+".upscale_conv"},{"output_frame",std::to_string(observed)},
                    {"source","complete native restored t25 decoder"},{"initial",initial?"true":"false"}});
                ++captured;
            }
            ++observed;
        });
        std::atomic<bool> cancelled{false};
        vae.decode_stream(latent,25,[&](int index,const tc::Tensor &frame) {
            tc::require(index==emitted && frame.shape()==mx::Shape{1,3,1,1024,1024} &&
                mx::all(mx::isfinite(frame)).item<bool>(),"capture decode frame/finite check failed");
            ++emitted;
        },{},cancelled);
        tc::require(observed==25 && emitted==25 && captured==2,"capture did not preserve the complete video");
        std::ofstream report(output/"capture.json");
        report << "{\"schema\":\"seedvr2-upscale-real-capture-v1\",\"status\":\"complete\","
            "\"decoded_frames\":25,\"observed_projection_frames\":25,\"captured_frames\":[0,24],"
            "\"input_shape\":[1,1,512,512,256],\"scope\":\"whole t25 causal decode; selected real operands, not performance qualification\"}\n";
        tc::require(bool(report),"capture report write failed");
        std::cout << "PASS complete t25 decode, real projection inputs frame0/frame24\n";
    } catch (const std::exception &failure) { std::cerr << failure.what() << '\n'; return 1; }
}
