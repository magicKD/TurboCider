#include "scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace tc::ltx098 {
namespace {
void validate(const Tensor &state, const Tensor &first, float strength) {
    require(state.ndim() == 5 && state.shape(0) == 1 && state.shape(1) >= 2 && state.dtype() == mx::float32 &&
                first.shape() == mx::Shape{1, 1, state.shape(2), state.shape(3), state.shape(4)} &&
                first.dtype() == mx::float32 && std::isfinite(strength) && strength >= 0 && strength <= 1,
            "invalid LTX 0.9.8 first-frame scheduler geometry or strength");
}

Tensor first_frame(const Tensor &value) {
    return mx::slice(value, {0, 0, 0, 0, 0}, {1, 1, value.shape(2), value.shape(3), value.shape(4)});
}

Tensor rest(const Tensor &value) {
    return mx::slice(value, {0, 1, 0, 0, 0}, value.shape());
}
} // namespace

Tensor conditioned_input(const Tensor &state, const Tensor &first, float strength) {
    validate(state, first, strength);
    auto head = first_frame(state) * Tensor(1.f - strength, mx::float32) + first * Tensor(strength, mx::float32);
    return mx::concatenate({head, rest(state)}, 1);
}

Tensor euler_step(const Tensor &state, const Tensor &model_input, const Tensor &velocity,
                  const Tensor &first, float strength, float sigma, float next_sigma) {
    validate(state, first, strength);
    require(model_input.shape() == state.shape() && velocity.shape() == state.shape() &&
                model_input.dtype() == mx::float32 && velocity.dtype() == mx::float32 &&
                std::isfinite(sigma) && sigma > 0 && sigma <= 1 &&
                std::isfinite(next_sigma) && next_sigma >= 0 && next_sigma < sigma,
            "invalid LTX 0.9.8 Euler step");
    auto denoised = model_input - velocity * Tensor(sigma, mx::float32);
    auto head = first_frame(denoised) * Tensor(1.f - strength, mx::float32) + first * Tensor(strength, mx::float32);
    denoised = mx::concatenate({head, rest(denoised)}, 1);
    auto derivative = (state - denoised) / Tensor(sigma, mx::float32);
    return state + derivative * Tensor(next_sigma - sigma, mx::float32);
}

Tensor original_condition(const Tensor &state, const Tensor &first, float strength) {
    return conditioned_input(state, first, strength);
}

float original_first_frame_strength(float strength, float sigma) {
    require(std::isfinite(strength) && strength >= 0 && strength <= 1 &&
                std::isfinite(sigma) && sigma > 0 && sigma <= 1,
            "invalid LTX 0.9.8 original conditioning timestep");
    return 1.f - std::min(sigma, 1.f - strength) / sigma;
}

Tensor original_euler_step(const Tensor &state, const Tensor &velocity, float strength,
                           float sigma, float next_sigma) {
    require(state.ndim() == 5 && state.shape(0) == 1 && state.shape(1) >= 2 &&
                state.dtype() == mx::float32 && velocity.shape() == state.shape() &&
                velocity.dtype() == mx::float32 && std::isfinite(strength) && strength >= 0 && strength <= 1 &&
                std::isfinite(sigma) && sigma > 0 && sigma <= 1 &&
                std::isfinite(next_sigma) && next_sigma >= 0 && next_sigma < sigma,
            "invalid LTX 0.9.8 original Euler step");
    const float cap = 1.f - strength;
    const float first_dt = std::min(next_sigma, cap) - std::min(sigma, cap);
    auto head = first_frame(state) + first_frame(velocity) * Tensor(first_dt, mx::float32);
    auto tail = rest(state) + rest(velocity) * Tensor(next_sigma - sigma, mx::float32);
    return mx::concatenate({head, tail}, 1);
}

} // namespace tc::ltx098
