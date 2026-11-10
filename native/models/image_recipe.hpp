#pragma once

#include "../core/contracts.hpp"

namespace tc::model_detail {

// Shared image pipeline; each model retains its own denoising step count.
static inline Recipe image_recipe(const char *model, int steps) {
    return Recipe{model,
                  {{"text_encode", {}},
                   {"denoise", {"text_encode"}, steps},
                   {"vae_decode", {"denoise"}},
                   {"export", {"vae_decode"}}},
                  true};
}

} // namespace tc::model_detail
