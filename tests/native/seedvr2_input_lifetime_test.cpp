#include "../../native/models/seedvr2/input_lifetime.hpp"
#include <cassert>
#include <iostream>

namespace mx = mlx::core;
using tc::Tensor;
template<class Action> void rejected(Action action) {
    bool caught = false;
    try { action(); } catch (const std::exception &) { caught = true; }
    assert(caught);
}
int main() {
    // CPU-only primitive ownership checks; no weights or Metal context.
    mx::set_default_device(mx::Device::cpu);
    for (int frames : {1,2,5,6,25,26}) {
        auto source = mx::ones({1,3,frames,32,48},mx::float32);
        mx::eval(source);
        const int padded_frames = tc::seedvr2::padded_frames(frames);
        auto padded = padded_frames == frames ? source : mx::ones({1,3,padded_frames,32,48},mx::float32);
        auto condition = mx::zeros({1,16,(padded_frames-1)/4+1,4,6},mx::float16);
        mx::eval(condition);
        auto retained_condition = condition;
        const auto bytes = source.nbytes();
        assert(tc::seedvr2::release_encoded_input(padded,source,condition) == bytes);
        assert(source.ndim() == 0 && padded.ndim() == 0);
        assert(mx::array_equal(condition,retained_condition).item<bool>());
    }
    auto source = mx::ones({1,3,5,32,48},mx::float32);
    mx::eval(source);
    auto padded = source;
    auto not_ready = mx::zeros({1,16,2,4,6},mx::float16);
    rejected([&] { tc::seedvr2::release_encoded_input(padded,source,not_ready); });
    assert(source.ndim() == 5 && padded.ndim() == 5);
    for (auto shape : {mx::Shape{1,16,1,4,6},mx::Shape{1,16,2,4,5}}) {
        auto wrong = mx::zeros(shape,mx::float16); mx::eval(wrong);
        rejected([&] { tc::seedvr2::release_encoded_input(padded,source,wrong); });
    }
    auto wrong_dtype = mx::zeros({1,16,2,4,6},mx::float32); mx::eval(wrong_dtype);
    rejected([&] { tc::seedvr2::release_encoded_input(padded,source,wrong_dtype); });
    auto bad_padding = mx::ones({1,3,9,32,48},mx::float32);
    auto ready = mx::zeros({1,16,2,4,6},mx::float16); mx::eval(ready);
    rejected([&] { tc::seedvr2::release_encoded_input(bad_padding,source,ready); });
    std::cout << "PASS CPU-only owned encoder input release, padding and readiness/precision guards\n";
}
