#pragma once

namespace tc::h3_mlx {

struct Float32SinCos {
    float sine;
    float cosine;
};

// Match NumPy's contiguous float32 sin/cos ufunc on Apple Silicon without
// introducing a Python runtime dependency into the native conditioner.
Float32SinCos numpy_float32_sin_cos(float input);

} // namespace tc::h3_mlx
