#include "../../native/models/qwen21/encoder_calibration.hpp"
#include <iostream>

// Small existing profile, CPU arrays only. No checkpoint tensors are loaded.
int main(int argc, char** argv) {
    try {
        tc::require(argc == 3, "profile and checkpoint identity paths required");
        tc::mx::set_default_device(tc::mx::Device(tc::mx::Device::cpu));
        using tc::qwen21::EncoderScaleProfile;
        auto profile = EncoderScaleProfile::load(argv[1], argv[2], 3, 1024, 6144);
        const float first = profile.scales.front(), last = profile.scales.back();
        auto tensors = profile.tensors();
        profile.scales.clear(); // tensors must own copies, not borrow the host vector
        tc::mx::eval(tensors);
        tc::require(tensors.size() == 36 && tensors.front().shape() == tc::mx::Shape{1,4096} &&
                    tensors.front().data<float>()[0] == first &&
                    tensors.back().data<float>()[4095] == last, "profile factory ownership mismatch");
        for (const auto& domain : {std::array{2,1024,6144}, std::array{3,512,6144}, std::array{3,1024,3072}}) {
            bool rejected = false;
            try { (void)EncoderScaleProfile::load(argv[1], argv[2], domain[0], domain[1], domain[2]); }
            catch (const std::exception&) { rejected = true; }
            tc::require(rejected, "profile silently crossed reference/channel domain");
        }
        profile.scales.resize(4096);
        bool partial_rejected = false;
        try { (void)profile.tensors(); } catch (const std::exception&) { partial_rejected = true; }
        tc::require(partial_rejected, "partial layer profile accepted");
        std::cout << "encoder_profile=PASS layers=36 ownership=copy invalid_domains=3 partial=rejected models_loaded=0\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "encoder_profile=FAIL " << e.what() << '\n';
        return 1;
    }
}
