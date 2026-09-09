#include "affine.hpp"

#include <limits>

namespace tc::components {

AffineMatrix::AffineMatrix(Tensor packed, Tensor scales, std::optional<Tensor> offsets,
                           int group_size, int bits)
    : packed_(std::move(packed)), scales_(std::move(scales)), offsets_(std::move(offsets)),
      group_size_(group_size), bits_(bits), input_channels_(0) {
    require((bits == 4 || bits == 8) &&
                (group_size == 32 || group_size == 64 || group_size == 128),
            "unsupported affine quantization geometry");
    require(packed_.ndim() == 2 && packed_.dtype() == mx::uint32 &&
                packed_.shape(0) > 0 && packed_.shape(1) > 0 &&
                packed_.shape(1) <= std::numeric_limits<int>::max() / (32 / bits),
            "invalid affine packed matrix");
    input_channels_ = packed_.shape(1) * (32 / bits);
    require(input_channels_ % group_size == 0 && scales_.ndim() == 2 &&
                scales_.shape(0) == packed_.shape(0) &&
                scales_.shape(1) == input_channels_ / group_size &&
                (scales_.dtype() == mx::float16 || scales_.dtype() == mx::bfloat16 ||
                 scales_.dtype() == mx::float32),
            "invalid affine scale geometry or dtype");
    if (offsets_)
        require(offsets_->shape() == scales_.shape() && offsets_->dtype() == scales_.dtype(),
                "affine offsets must match scales");
}

Tensor AffineMatrix::project(const Tensor &x) const {
    require(x.ndim() >= 1 && x.shape(-1) == input_channels_ &&
                (x.dtype() == mx::float16 || x.dtype() == mx::bfloat16 || x.dtype() == mx::float32),
            "affine projection input geometry or dtype mismatch");
    return mx::astype(mx::quantized_matmul(x, packed_, scales_, offsets_, true,
                                         group_size_, bits_, "affine"), x.dtype());
}

AffineMatrix AffineMatrix::slice(int row_begin, int row_end, int column_begin, int column_end) const {
    require(row_begin >= 0 && row_end > row_begin && row_end <= output_channels() &&
                column_begin >= 0 && column_end > column_begin && column_end <= input_channels_ &&
                column_begin % group_size_ == 0 && column_end % group_size_ == 0,
            "affine slice must have valid rows and group-aligned columns");
    auto select = [&](const Tensor &x, int begin, int end) {
        return slice_axis(slice_axis(x, 0, row_begin, row_end), 1, begin, end);
    };
    auto packed = select(packed_, column_begin / (32 / bits_), column_end / (32 / bits_));
    auto scales = select(scales_, column_begin / group_size_, column_end / group_size_);
    std::optional<Tensor> offsets;
    if (offsets_) offsets = select(*offsets_, column_begin / group_size_, column_end / group_size_);
    return AffineMatrix(std::move(packed), std::move(scales), std::move(offsets), group_size_, bits_);
}

} // namespace tc::components
