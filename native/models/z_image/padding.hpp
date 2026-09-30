#pragma once

#include "../../backends/mlx.hpp"

namespace tc::z_image {

// GGUF exporters may omit a padding token's leading singleton dimension.
// Restore only that storage convention, never reinterpret packed Q4/Q8 codes
// or change dtype/values. The regular safetensors [1,H] path is unchanged.
inline Tensor padding_token_row(const Tensor &token, int hidden) {
    const bool floating = token.dtype() == mx::float16 || token.dtype() == mx::bfloat16 ||
                          token.dtype() == mx::float32;
    require(hidden > 0 && floating &&
                (token.shape() == mx::Shape{hidden} || token.shape() == mx::Shape{1, hidden}),
            "Z-Image padding token must be an unquantized floating [H] or [1,H] tensor");
    return token.ndim() == 1 ? mx::reshape(token, {1, hidden}) : token;
}

} // namespace tc::z_image
