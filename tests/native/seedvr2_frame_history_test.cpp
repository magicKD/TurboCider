#include "../../native/models/seedvr2/frame_history.hpp"
#include "../../native/models/seedvr2/metal/causal_history.hpp"
#include "../../native/models/seedvr2/metal/conv_tiled.hpp"
#include <cstring>
#include <iostream>

namespace mx = mlx::core;
using tc::Tensor;
void same_bytes(const Tensor &a, const Tensor &b) {
    tc::require(a.shape() == b.shape() && a.dtype() == b.dtype(),"history shape/dtype mismatch");
    auto x = mx::contiguous(a), y = mx::contiguous(b);
    mx::eval({x,y});
    tc::require(std::memcmp(x.data<uint8_t>(),y.data<uint8_t>(),x.nbytes()) == 0,"history bytes differ");
}
int main() {
    try {
        mx::set_default_device(mx::Device::cpu);
        for (int count : {1,2}) {
            tc::seedvr2::FrameHistory frames;
            Tensor compact(0);
            int sequence = 0;
            for (int t : {1,4,1,1,2,1,4,1}) {
                std::vector<float> values(size_t(t)*24);
                for (size_t i = 0; i < values.size(); ++i) values[i] = float(sequence++)*.01f;
                const uint32_t negative_zero = 0x80000000u, payload_nan = 0x7fc00035u;
                std::memcpy(&values[0],&negative_zero,sizeof(float));
                std::memcpy(&values[1],&payload_nan,sizeof(float));
                Tensor input(values.data(),{1,t,2,3,4},mx::float32);
                const bool stored = !frames.empty();
                auto first = mx::slice(input,{0,0,0,0,0},{1,1,2,3,4});
                auto reference = tc::seedvr2::causal_history(input,stored ? compact : first,count,stored);
                auto next = tc::seedvr2::advance_frame_history(input,frames,count,stored,false);
                tc::require(int(next.size()) == count,"incorrect frame history capacity");
                same_bytes(mx::concatenate(next,1),reference);
                if (stored && t == 1 && count == 2)
                    tc::require(next.front().data<float>() == frames.back().data<float>(),"old tail frame was unnecessarily copied");
                tc::require(next.back().data<float>() != input.data<float>(),"untrusted input retained by view");
                frames = std::move(next); compact = reference; mx::eval(compact);
            }
        }
        Tensor owned(std::vector<float>(24,3.f).data(),{1,1,2,3,4},mx::float32);
        mx::eval(owned);
        auto shared = tc::seedvr2::advance_frame_history(owned,{},2,false,true);
        tc::require(shared[0].data<float>() == owned.data<float>() && shared[1].data<float>() == owned.data<float>(),
                    "owned initial singleton was copied or duplicated");
        auto source = mx::reshape(mx::arange(96,mx::float32),{1,4,2,3,4});
        auto view = mx::slice(source,{0,3,0,0,0},{1,4,2,3,4});
        auto compact_view = tc::seedvr2::advance_frame_history(view,{},2,false,false);
        mx::eval(view);
        tc::require(compact_view[0].data<float>() != view.data<float>(),"contiguous view pinned a larger source");
        bool rejected = false;
        try { tc::seedvr2::advance_frame_history(owned,{},2,true,false); } catch (const std::exception &) { rejected = true; }
        tc::require(rejected,"missing prior history was accepted");
        std::cout << "PASS: CPU frame history bytes, signed zero/NaN, phases, tail sharing and independent ownership\n";

        mx::set_default_device(mx::Device::gpu);
        for (int history_count : {1,2}) for (int channels : {3,8}) {
            const int stride = history_count == 1 ? 2 : 1;
            auto input = mx::astype(mx::random::normal({1,4,8,8,channels},mx::float32,mx::random::key(11)),mx::float16);
            auto h0 = mx::astype(mx::random::normal({1,1,8,8,channels},mx::float32,mx::random::key(12)),mx::float16);
            auto h1 = mx::astype(mx::random::normal({1,1,8,8,channels},mx::float32,mx::random::key(13)),mx::float16);
            auto weight = mx::astype(mx::random::normal({8,3,3,3,channels},mx::float32,mx::random::key(14))*.02f,mx::float16);
            auto history = history_count == 1 ? h0 : mx::concatenate({h0,h1},1);
            auto split = tc::seedvr2::metal::conv_tiled(input,h0,history_count,false,weight,stride,1,1,false,128,&h1);
            auto legacy = tc::seedvr2::metal::conv_tiled(input,history,history_count,false,weight,stride,1,1,false,128);
            same_bytes(split,legacy);
        }
        std::cout << "PASS: GPU split-history bounded convolution matches contiguous history for scalar/vector channels and both temporal strides\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
