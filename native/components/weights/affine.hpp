#pragma once

#include "../../backends/mlx.hpp"

namespace tc::components {

// Geometry is supplied by a versioned checkpoint manifest, never guessed from
// a filename. This is independent of the narrower GGUF acceptance policy.
class AffineMatrix {
    Tensor packed_;
    Tensor scales_;
    std::optional<Tensor> offsets_;
    int group_size_;
    int bits_;
    int input_channels_;

  public:
    AffineMatrix(Tensor packed, Tensor scales, std::optional<Tensor> offsets,
                 int group_size, int bits);
    int input_channels() const { return input_channels_; }
    int output_channels() const { return packed_.shape(0); }
    // Preserve the source runtime's cast before adding the layer bias.
    Tensor project(const Tensor &) const;
    AffineMatrix slice(int row_begin, int row_end, int column_begin, int column_end) const;
};

} // namespace tc::components
