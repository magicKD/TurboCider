#pragma once

#include "../../backends/mlx.hpp"

namespace tc::ltx098 {

// Comfy's first-frame inpainting mask. State/first/velocity are FP32 BFHWC;
// the first-frame condition is blended into both input and denoised output.
Tensor conditioned_input(const Tensor &state, const Tensor &first_frame, float strength);
Tensor euler_step(const Tensor &state, const Tensor &model_input, const Tensor &velocity,
                  const Tensor &first_frame, float strength, float sigma, float next_sigma);

// Lightricks' original multiscale image-conditioning semantics. The image is
// blended once at the start of each pass. Its first-frame timestep is capped
// at (1-strength), and it is frozen while the global sigma is above that cap.
Tensor original_condition(const Tensor &state, const Tensor &first_frame, float strength);
Tensor original_euler_step(const Tensor &state, const Tensor &velocity, float strength,
                           float sigma, float next_sigma);
float original_first_frame_strength(float strength, float sigma);

} // namespace tc::ltx098
