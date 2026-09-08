#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"
#include "../../runtime/session.hpp"

namespace tc {

class HybridSession;

struct LLaDAConditioning {
    Tensor features;
    int prompt_tokens = 0;
    int total_tokens = 0;
};

struct LLaDATransformerContext {
    Tensor features;
    Tensor frequencies;
    int caption_tokens = 0;
    int padded_tokens = 0;
};

LLaDAConditioning llada_encode_text(const std::filesystem::path &, const Tokens &,
                                    Weights &text, Weights &queryformer,
                                    Weights &projection, const Event &,
                                    std::atomic<bool> &,
                                    const std::string &dump_directory = {});
void normalize_llada_transformer(Weights &);
LLaDATransformerContext llada_prepare_transformer_context(
    const Tensor &, const Weights &, const Event &, std::atomic<bool> &,
    const std::string &dump_directory = {});
std::function<std::vector<Tensor>(const std::vector<Tensor> &)>
make_llada_hybrid_gpu_graph(int mlp_width, int gpu_mlp_start);
Tensor llada_transformer(const Tensor &, const LLaDATransformerContext &, float, int, int,
                         const Weights &, const Event &, std::atomic<bool> &,
                         HybridSession *hybrid,
                         const std::function<std::vector<Tensor>(
                             const std::vector<Tensor> &)> *gpu_graph,
                         bool compile_blocks, const std::string &dump_directory = {},
                         int dump_step = -1);
std::unique_ptr<ModelSession> create_llada_image_native(const std::filesystem::path &);

} // namespace tc
