#include "../../native/models/seedvr2/media.hpp"
#include <mlx/random.h>
#include <cassert>
#include <iostream>

namespace mx = mlx::core;
int main() {
    mx::set_default_device(mx::Device::gpu);
    for (bool video : {false,true}) {
        for (int frames : (video ? std::vector<int>{1,5,25} : std::vector<int>{1})) {
            auto unit = mx::random::uniform(0.f,1.f,{frames,32,48,video ? 3 : 4},mx::float32,mx::random::key(42+frames));
            mx::eval(unit);
            const auto *source_allocation = unit.buffer().ptr();
            auto rgb = mx::slice(unit,{0,0,0,0},{frames,32,48,3});
            auto expected = mx::expand_dims(mx::transpose(rgb*2.f-1.f,{3,0,1,2}),0);
            tc::Tensor expected_alpha(0);
            if (!video) expected_alpha = mx::slice(unit,{0,0,0,3},{1,32,48,4})*2.f-1.f;
            mx::eval({expected,expected_alpha});
            // Ensure the test itself does not keep an RGB view alive.
            rgb = tc::Tensor(0);
            auto result = tc::seedvr2::consume_resized_input(unit,video,true);
            assert(unit.ndim() == 0 && result.pixels.is_available());
            assert(result.pixels.shape() == mx::Shape({1,3,frames,32,48}));
            assert(mx::array_equal(result.pixels,expected).item<bool>());
            assert(result.pixels.buffer().ptr() != source_allocation);
            if (video) assert(result.alpha.ndim() == 0);
            else {
                assert(result.alpha_is_owned);
                assert(result.alpha.is_available() && result.alpha.flags().row_contiguous);
                assert(result.alpha.nbytes() == 32*48*sizeof(float));
                assert(result.alpha.buffer().ptr() != source_allocation);
                assert(mx::array_equal(result.alpha*2.f-1.f,expected_alpha).item<bool>());
            }
        }
    }
    {
        auto unit = mx::random::uniform(0.f,1.f,{1,32,48,4},mx::float32,mx::random::key(17));
        mx::eval(unit);
        const auto *source_allocation = unit.buffer().ptr();
        auto expected = mx::slice(unit,{0,0,0,3},{1,32,48,4});
        auto result = tc::seedvr2::consume_resized_input(unit,false);
        mx::eval(result.alpha);
        assert(unit.ndim() == 0 && !result.alpha_is_owned);
        assert(result.alpha.buffer().ptr() == source_allocation);
        assert(mx::array_equal(result.alpha,expected).item<bool>());
    }
    for (auto shape : {mx::Shape{1,32,48,3},mx::Shape{2,32,48,4}}) {
        auto invalid = mx::zeros(shape,mx::float32);
        bool rejected = false;
        try { tc::seedvr2::consume_resized_input(invalid,false); }
        catch (const std::exception &) { rejected = true; }
        assert(rejected);
    }
    std::cout << "PASS bitwise normalization/alpha, owned/view policies, source release and shape guards\n";
}
