#include "hybrid.hpp"
#include "prefix_batch_plan.hpp"

namespace tc::seedvr2 {
namespace {
Tensor rows(const Tensor &x, int first, int last) {
    return mx::slice(x,{first,0},{last,x.shape(1)});
}
HybridConfig validated_config(const DiT &dit, HybridConfig config) {
    require((config.rows == 128 || config.rows == 256 || config.rows == 512 || config.rows == 1024) &&
            config.channels > 0 && config.channels < 6912 && config.channels % 256 == 0 &&
            !config.blocks.empty() && config.blocks.size() == config.artifacts.size(),
            "invalid SeedVR2 hybrid configuration");
    require(config.source_operand_recipe == "original-fp16" && dit.stable_fp16_profile(),
            "SeedVR2 hybrid requires original-FP16 operands and the stable DiT profile");
    std::array<bool,32> seen{};
    for (int block : config.blocks) {
        require(block >= 0 && block < 32 && !seen[block],"invalid SeedVR2 hybrid block coverage");
        seen[block] = true;
    }
    return config;
}
}
HybridFFN::HybridFFN(const DiT &dit, HybridConfig config, const Event &event,
                     std::atomic<bool> &cancelled, bool cpu_only_control)
    : dit_(dit), config_(validated_config(dit,std::move(config))), cancelled_(cancelled),
      buffered_(config_.artifacts,config_.rows,2560,event,cancelled,cpu_only_control),
      cpu_only_control_(cpu_only_control) {
    slots_.fill(-1);
    for (size_t slot = 0; slot < config_.blocks.size(); ++slot) slots_[config_.blocks[slot]] = int(slot);
}
Tensor HybridFFN::predict(int block, const Tensor &input) {
    require(block >= 0 && block < 32 && input.ndim() == 2 && input.shape(0) > 0 &&
            input.shape(1) == 2560 && input.dtype() == mx::float16 && dit_.stable_fp16_profile(),
            "invalid SeedVR2 hybrid FFN input");
    checkpoint(cancelled_);
    if (slots_[block] < 0) {
        ++gpu_blocks_;
        return dit_.ffn_full(input,block);
    }
    const auto started = Clock::now();
    std::vector<Tensor> results;
    for (auto [first,last] : prefix_batch_ranges(input.shape(0),config_.rows,int(buffered_.maximum_chunks()))) {
        checkpoint(cancelled_);
        auto part = rows(input,first,last);
        std::vector<Tensor> packed;
        for (int begin = first; begin < last;) {
            const int count = std::min(config_.rows,last-begin);
            auto x = rows(input,begin,begin+count);
            if (count != config_.rows)
                x = mx::concatenate({x,mx::zeros({config_.rows-count,2560},mx::float16)},0);
            packed.push_back(mx::contiguous(mx::expand_dims(x,0)));
            begin += count;
        }
        // Finish every packing/allocation operation before suffix submission.
        // Prefix prediction must not introduce a default-stream input fence.
        mx::eval(packed);
        buffered_.prepare(packed.size());
        auto gpu = dit_.ffn_partial(part,block,config_.channels,6912);
        mx::async_eval(gpu);
        std::vector<Tensor> prefixes;
        for (size_t chunk = 0; chunk < packed.size(); ++chunk) {
            checkpoint(cancelled_);
            prefixes.push_back(buffered_.predict(slots_[block],chunk,packed[chunk]));
        }
        auto prefix = rows(mx::squeeze(mx::concatenate(prefixes,1),0),0,last-first);
        auto joined = mx::astype(mx::astype(gpu,mx::float32)+mx::astype(prefix,mx::float32),mx::float16);
        mx::eval(joined);
        buffered_.release_after(joined);
        results.push_back(joined);
        ++batches_;
    }
    auto result = results.size() == 1 ? results.front() : mx::concatenate(results,0);
    mx::eval(result);
    ++hybrid_blocks_;
    prediction_seconds_ += std::chrono::duration<double>(Clock::now()-started).count();
    return result;
}
} // namespace tc::seedvr2
