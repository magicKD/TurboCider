#pragma once

#include "../../backends/mlx.hpp"

namespace tc::qwen21::runtime_ffn {
using Function = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
inline Function input_ranks(const Weights &weights,const std::string &prefix,int hidden=4096) {
    return mx::compile([&weights,prefix,hidden](const std::vector<Tensor> &a) {
        return weights.lora_input_ranks(a[0],prefix+"gate_up",0,hidden);
    });
}
inline std::vector<Tensor> rank_inputs(const std::vector<Tensor> &a) {
    require(a.size()>1,"shared Qwen LoRA rank inputs missing");
    return {a.begin()+1,a.end()};
}

// Request/calibration-local factories. Capture the SAME resident Weights;
// never retain these closures after an adapter rebind or use a global cache.
// Calibration and inference share the graph bodies and rounding boundaries,
// not just a mathematically similar eager implementation.
inline Function full(const Weights &weights, const std::string &prefix) {
    return mx::compile([&weights, prefix](const std::vector<Tensor> &a) {
        auto halves = mx::split(weights.project(a[0], prefix+"gate_up"), 2, -1);
        return std::vector<Tensor>{weights.project(silu(halves[0])*halves[1], prefix+"out")};
    });
}

inline Function corrections(const Weights &weights, const std::string &prefix,
                            int first, int count, int hidden = 4096, int width = 12288) {
    require(first >= 0 && count > 0 && first <= width-count && hidden > 0,
            "invalid Qwen FFN correction graph range");
    return mx::compile([&weights, prefix, first, count, hidden, width](const std::vector<Tensor> &a) {
        // Logical halves retain separate adapter intersections and the existing
        // FP32 accumulation/final BF16 delta rounding, including stacked LoRA.
        auto gate = weights.lora_delta_slice(a[0], prefix+"gate_up", first, first+count, 0, hidden);
        auto up = weights.lora_delta_slice(a[0], prefix+"gate_up", width+first, width+first+count, 0, hidden);
        return std::vector<Tensor>{mx::contiguous(gate), mx::contiguous(up)};
    });
}
inline Function corrections_shared_ranks(const Weights &weights,const std::string &prefix,
                                        int first,int count,int hidden=4096,int width=12288) {
    require(first>=0 && count>0 && first<=width-count,"invalid shared Qwen correction range");
    return mx::compile([&weights,prefix,first,count,hidden,width](const std::vector<Tensor> &a) {
        const auto ranks=rank_inputs(a);
        auto gate=weights.lora_delta_slice_with_ranks(a[0],prefix+"gate_up",ranks,first,first+count,0,hidden);
        auto up=weights.lora_delta_slice_with_ranks(a[0],prefix+"gate_up",ranks,width+first,width+first+count,0,hidden);
        return std::vector<Tensor>{mx::contiguous(gate),mx::contiguous(up)};
    });
}

inline Function channels(const Weights &weights, const std::string &prefix,
                         int first, int count, int hidden = 4096, int width = 12288,
                         bool fp32_partial = false,bool shared_ranks = false) {
    require(first >= 0 && count > 0 && first <= width-count && hidden > 0,
            "invalid Qwen FFN channel graph range");
    return mx::compile([&weights, prefix, first, count, hidden, width, fp32_partial,shared_ranks](const std::vector<Tensor> &a) {
        const auto ranks=shared_ranks ? rank_inputs(a) : std::vector<Tensor>{};
        auto gate = shared_ranks ? weights.project_slice_with_ranks(a[0],prefix+"gate_up",ranks,first,first+count,0,hidden,false) :
            weights.project_slice(a[0], prefix+"gate_up", first, first+count, 0, hidden, false);
        auto up = shared_ranks ? weights.project_slice_with_ranks(a[0],prefix+"gate_up",ranks,width+first,width+first+count,0,hidden,false) :
            weights.project_slice(a[0], prefix+"gate_up", width+first, width+first+count, 0, hidden, false);
        auto intermediate = silu(gate)*up;
        auto base = fp32_partial ? weights.project_base_slice_fp32(intermediate,prefix+"out",0,hidden,first,first+count) :
            weights.project_base_slice(intermediate, prefix+"out", 0, hidden, first, first+count, false);
        // Down-LoRA is NOT rounded once per shard. Apply it to joined hidden.
        return std::vector<Tensor>{base, intermediate};
    });
}

inline Function down_add(const Weights &weights, const std::string &prefix,
                         int hidden = 4096, int width = 12288) {
    require(hidden > 0 && width > 0, "invalid Qwen FFN down graph geometry");
    return mx::compile([&weights, prefix, hidden, width](const std::vector<Tensor> &a) {
        auto delta = weights.lora_delta_slice(a[0], prefix+"out", 0, hidden, 0, width);
        return std::vector<Tensor>{mx::astype(mx::astype(a[1], mx::float32)+mx::astype(delta, mx::float32), a[1].dtype())};
    });
}
inline Function down_gpu_ranks(const Weights &weights,const std::string &prefix,int gpu_channels,int width=12288) {
    require(gpu_channels>0 && gpu_channels<width,"invalid split down-rank GPU range");
    return mx::compile([&weights,prefix,gpu_channels](const std::vector<Tensor> &a) {
        auto ranks=weights.lora_input_ranks(a[0],prefix+"out",0,gpu_channels);
        for(const auto &rank:ranks)require(rank.dtype()==mx::float32,"split down ranks require FP32 rank arithmetic");
        return ranks;
    });
}
inline Function down_add_split_ranks(const Weights &weights,const std::string &prefix,int gpu_channels,
                                    int hidden=4096,int width=12288) {
    require(gpu_channels>0 && gpu_channels<width && hidden>0,"invalid split down-rank join geometry");
    return mx::compile([&weights,prefix,gpu_channels,hidden,width](const std::vector<Tensor> &a) {
        require(a.size()>2,"GPU down-rank partials missing");
        auto tail=weights.lora_input_ranks(a[0],prefix+"out",gpu_channels,width);
        require(tail.size()==a.size()-2,"split down adapter count mismatch");
        std::vector<Tensor> joined;
        for(size_t i=0;i<tail.size();++i) {
            require(tail[i].dtype()==mx::float32 && a[2+i].dtype()==mx::float32 && tail[i].shape()==a[2+i].shape(),
                    "split down rank partial shape/dtype mismatch");
            joined.push_back(a[2+i]+tail[i]);
        }
        auto logical_shape=a[0].shape();logical_shape.back()=width;
        auto delta=weights.lora_delta_from_ranks(logical_shape,a[0].dtype(),prefix+"out",joined,0,hidden,0,width);
        return std::vector<Tensor>{mx::astype(mx::astype(a[1],mx::float32)+mx::astype(delta,mx::float32),a[1].dtype())};
    });
}
} // namespace tc::qwen21::runtime_ffn
