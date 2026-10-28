#include "../../native/media/image.hpp"
#include "../../native/models/qwen21/scheduler.hpp"
#include <iostream>
#include <cmath>

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
        const float turbo_nodes[]={1.f,.978453f,.95418f,.926626f,.89508f,.845148f,.704534f,.414568f,0.f};
        for(auto dims:{std::pair{512,512},std::pair{768,512},std::pair{1024,1024}}) {
            auto turbo=tc::qwen21::turbo_sigmas(dims.first,dims.second);tc::mx::eval(turbo);
            tc::require(turbo.shape()==tc::mx::Shape{9},"Turbo8 shape invalid");
            for(int i=0;i<9;++i)tc::require(turbo.data<float>()[i]==turbo_nodes[i],"Turbo nodes must be exact and independent of canvas shift");
            schedules.emplace("turbo_"+std::to_string(dims.first)+"x"+std::to_string(dims.second),turbo);
        }
        for(int invalid:{0,1,6,7,9,20,40}) {
            bool denied=false;try{(void)tc::qwen21::turbo_sigmas(512,512,invalid);}catch(const std::exception&){denied=true;}
            tc::require(denied,"Turbo accepted non-authored step count");
        }
        for (int steps : {2, 4, 40}) {
            auto sigma = tc::qwen21::sigmas(512, 512, steps);
            tc::mx::eval(sigma);
            for (int i = 0; i < steps; ++i)
                tc::require(sigma.data<float>()[i] > sigma.data<float>()[i+1], "schedule is not strictly descending");
            schedules.emplace(std::to_string(steps), sigma);
        }
        auto viggle = tc::qwen21::viggle_v021_sigmas(512, 512);
        tc::mx::eval(viggle);
        tc::require(viggle.shape() == tc::mx::Shape{7} && viggle.data<float>()[0] == 1.f &&
                    viggle.data<float>()[6] == 0.f, "Viggle six-step schedule shape/endpoints invalid");
        for (int i = 0; i < 6; ++i)
            tc::require(viggle.data<float>()[i] > viggle.data<float>()[i+1],
                        "Viggle six-step sigma nodes are not descending");
        schedules.emplace("viggle_512", viggle);
        const float raw[] = {1.f, .875f, .75f, .5f, .25f};
        for (const auto &dims : {std::pair{512,512}, std::pair{768,512}, std::pair{1024,1024}}) {
            auto five = tc::qwen21::viggle_v021_sigmas(dims.first, dims.second, 5);
            tc::mx::eval(five);
            tc::require(five.shape() == tc::mx::Shape{6} && five.data<float>()[5] == 0.f,
                        "Viggle five-step endpoints/shape invalid");
            const double slope = (.9 - .5) / (8192. - 256.);
            const double mu = slope * dims.first * dims.second / 256. + .5 - slope * 256.;
            for (int i = 0; i < 5; ++i) {
                const double expected = std::exp(mu) / (std::exp(mu) + 1. / raw[i] - 1.);
                tc::require(std::abs(five.data<float>()[i] - expected) < 1e-7,
                            "Viggle five-step authored node/shift differs");
                tc::require(five.data<float>()[i] > five.data<float>()[i+1],
                            "Viggle five-step nodes not descending");
            }
            schedules.emplace("viggle_five_" + std::to_string(dims.first) + "x" + std::to_string(dims.second), five);
        }
        auto restored = tc::qwen21::viggle_v021_sigmas(512, 512, 6);
        tc::mx::eval(restored);
        for (int i = 0; i < 7; ++i)
            tc::require(restored.data<float>()[i] == viggle.data<float>()[i], "six-step default changed");
        for (int invalid : {-1,0,1,4,7,40}) {
            bool rejected_steps = false;
            try { (void)tc::qwen21::viggle_v021_sigmas(512,512,invalid); }
            catch (const std::exception &) { rejected_steps = true; }
            tc::require(rejected_steps, "unsupported student step count accepted");
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
