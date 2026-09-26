#include "../../native/media/image.hpp"
#include "../../native/models/qwen21/scheduler.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--import") {
            tc::configure_streams();
            tc::mx::save_safetensors(argv[3], {{"pixels", tc::load_rgba_image_tensor(argv[2])}});
            return 0;
        }
        tc::require(argc == 2, "output directory required");
        tc::configure_streams();
        std::filesystem::path output(argv[1]);
        const float rgba[] = {1,-1,-1,-1, -1,1,-1,0, -1,-1,1,1, 1,1,1,1};
        tc::Tensor pixels(rgba, {1,2,2,4}, tc::mx::float32);
        tc::save_rgba_png(pixels, output / "rgba.png");
        tc::save_png(tc::slice_axis(pixels, 3, 0, 3), output / "rgb.png");
        auto decoded = tc::load_rgba_image_tensor(output / "rgba.png");
        auto decoded_rgb = tc::load_rgba_image_tensor(output / "rgb.png");
        tc::mx::save_safetensors((output / "decoded.safetensors").string(), {{"rgba", decoded}, {"rgb", decoded_rgb}});
        auto one = tc::qwen21::sigmas(512, 512, 1);
        tc::mx::eval(one);
        tc::require(one.data<float>()[0] == 1.f && one.data<float>()[1] == 0.f, "one-step sigma invalid");
        std::unordered_map<std::string, tc::Tensor> schedules;
        for (int steps : {2, 4, 40}) {
            auto sigma = tc::qwen21::sigmas(512, 512, steps);
            tc::mx::eval(sigma);
            for (int i = 0; i < steps; ++i)
                tc::require(sigma.data<float>()[i] > sigma.data<float>()[i+1], "schedule is not strictly descending");
            schedules.emplace(std::to_string(steps), sigma);
        }
        tc::mx::save_safetensors((output / "schedules.safetensors").string(), schedules);
        bool rejected = false;
        try { tc::qwen21::sigmas(513,512,40); } catch (const std::exception &) { rejected = true; }
        tc::require(rejected, "unaligned dimensions accepted");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
