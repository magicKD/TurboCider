#pragma once
#include "runtime.hpp"
namespace tc {
// Intermediate model-family components stay internal; callers exchange owned
// tensors, never scratch filenames or process environment variables.
std::pair<Tensor,Tensor> ltx_project_hidden(const std::vector<Tensor>&,const Tensor&,const Weights&);
std::vector<Tensor> ltx_gemma_hidden(NSDictionary *config,const Weights&,
    const Tensor& token_ids,const Tensor& attention_mask,const Event&,std::atomic<bool>&);
}
