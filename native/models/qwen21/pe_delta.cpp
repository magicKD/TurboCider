#include "pe_delta.hpp"
#include <cmath>

namespace tc::qwen21::pe {
namespace {
Tensor solve_unit_lower(const Tensor &lower, const Tensor &rhs, int size,
                        std::atomic<bool> &cancelled) {
    std::vector<Tensor> solved;
    solved.reserve(size);
    for (int row = 0; row < size; ++row) {
        checkpoint(cancelled);
        auto current = slice_axis(rhs, -2, row, row + 1);
        if (row) {
            auto prior = mx::concatenate(solved, -2);
            auto coefficients = slice_axis(slice_axis(lower, -2, row, row + 1), -1, 0, row);
            current = current - mx::matmul(coefficients, prior);
        }
        solved.push_back(current);
    }
    return mx::concatenate(solved, -2);
}

void validate_delta(const Tensor &query, const Tensor &key,
    const Tensor &value, const Tensor &g, const Tensor &beta,
    const std::optional<Tensor> &initial_state) {
    require(query.ndim() == 4 && query.shape() == key.shape() &&
            value.ndim() == 4 && query.shape(0) > 0 && query.shape(1) > 0 &&
            query.shape(2) > 0 && query.shape(3) > 0 && value.shape(3) > 0 &&
            value.shape(0) == query.shape(0) && value.shape(1) == query.shape(1) &&
            value.shape(2) == query.shape(2), "Qwen35 delta Q/K/V geometry mismatch");
    const int batch = query.shape(0), time = query.shape(1), heads = query.shape(2);
    const int kd = query.shape(3), vd = value.shape(3);
    require(g.shape() == mx::Shape{batch, time, heads} && beta.shape() == g.shape(),
            "Qwen35 delta gate geometry mismatch");
    auto floating = [](mx::Dtype d) {
        return d == mx::float32 || d == mx::bfloat16 || d == mx::float16;
    };
    require(floating(query.dtype()) && key.dtype() == query.dtype() &&
            value.dtype() == query.dtype() && floating(g.dtype()) && floating(beta.dtype()),
            "Qwen35 delta requires matching floating Q/K/V tensors");
    const mx::Shape state_shape{batch, heads, kd, vd};
    require(!initial_state || (initial_state->shape() == state_shape && floating(initial_state->dtype())),
            "Qwen35 delta initial-state geometry/dtype mismatch");
}
}
DeltaResult recurrent_delta(const Tensor &query, const Tensor &key,
    const Tensor &value, const Tensor &g, const Tensor &beta,
    const std::optional<Tensor> &initial_state, bool normalize_qk,
    std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    validate_delta(query, key, value, g, beta, initial_state);
    const int batch = query.shape(0), time = query.shape(1), heads = query.shape(2);
    const int kd = query.shape(3), vd = value.shape(3);
    const mx::Shape state_shape{batch, heads, kd, vd};
    auto state = initial_state ? mx::astype(*initial_state, mx::float32) : mx::zeros(state_shape);
    auto q = mx::astype(query, mx::float32), k = mx::astype(key, mx::float32);
    auto v = mx::astype(value, mx::float32);
    if (normalize_qk) {
        q = q * mx::rsqrt(mx::sum(q * q, -1, true) + 1e-6f);
        k = k * mx::rsqrt(mx::sum(k * k, -1, true) + 1e-6f);
    }
    q = q / std::sqrt(float(kd));
    const auto decay = mx::exp(mx::astype(g, mx::float32));
    const auto update = mx::astype(beta, mx::float32);
    std::vector<Tensor> outputs;
    outputs.reserve(time);
    for (int t = 0; t < time; ++t) {
        checkpoint(cancelled);
        const auto qt = mx::reshape(slice_axis(q, 1, t, t + 1), {batch, heads, kd, 1});
        const auto kt = mx::reshape(slice_axis(k, 1, t, t + 1), {batch, heads, kd, 1});
        const auto vt = mx::reshape(slice_axis(v, 1, t, t + 1), {batch, heads, vd});
        state = state * mx::reshape(slice_axis(decay, 1, t, t + 1), {batch, heads, 1, 1});
        auto delta = (vt - mx::sum(state * kt, -2)) *
            mx::reshape(slice_axis(update, 1, t, t + 1), {batch, heads, 1});
        state = state + kt * mx::expand_dims(delta, -2);
        outputs.push_back(mx::sum(state * qt, -2));
        // Bound graph growth when used as a diagnostic prefill reference.
        if ((t + 1) % 32 == 0) {
            mx::eval(outputs);
            mx::eval(state);
        }
    }
    return {mx::astype(mx::stack(outputs, 1), query.dtype()), state};
}

DeltaResult chunked_delta(const Tensor &query, const Tensor &key,
    const Tensor &value, const Tensor &g, const Tensor &beta,
    const std::optional<Tensor> &initial_state, bool normalize_qk,
    std::atomic<bool> &cancelled, int chunk_size) {
    checkpoint(cancelled);
    validate_delta(query, key, value, g, beta, initial_state);
    require(chunk_size >= 1 && chunk_size <= 128, "Qwen35 delta chunk size must be 1...128");
    const int batch = query.shape(0), time = query.shape(1), heads = query.shape(2);
    const int kd = query.shape(3), vd = value.shape(3);
    const int padding = (chunk_size - time % chunk_size) % chunk_size;
    const int chunks = (time + padding) / chunk_size;
    auto q = mx::astype(mx::transpose(query, {0, 2, 1, 3}), mx::float32);
    auto k = mx::astype(mx::transpose(key, {0, 2, 1, 3}), mx::float32);
    auto v = mx::astype(mx::transpose(value, {0, 2, 1, 3}), mx::float32);
    if (normalize_qk) {
        q = q * mx::rsqrt(mx::sum(q * q, -1, true) + 1e-6f);
        k = k * mx::rsqrt(mx::sum(k * k, -1, true) + 1e-6f);
    }
    q = q / std::sqrt(float(kd));
    q = mx::pad(q, {{0, 0}, {0, 0}, {0, padding}, {0, 0}});
    k = mx::pad(k, {{0, 0}, {0, 0}, {0, padding}, {0, 0}});
    v = mx::pad(v, {{0, 0}, {0, 0}, {0, padding}, {0, 0}});
    auto gate = mx::pad(mx::astype(mx::transpose(g, {0, 2, 1}), mx::float32),
                        {{0, 0}, {0, 0}, {0, padding}});
    auto update = mx::expand_dims(mx::pad(mx::astype(mx::transpose(beta, {0, 2, 1}), mx::float32),
                                         {{0, 0}, {0, 0}, {0, padding}}), -1);
    auto kb = mx::reshape(k * update, {batch, heads, chunks, chunk_size, kd});
    auto vb = mx::reshape(v * update, {batch, heads, chunks, chunk_size, vd});
    q = mx::reshape(q, {batch, heads, chunks, chunk_size, kd});
    k = mx::reshape(k, q.shape());
    auto cumulative = mx::cumsum(mx::reshape(gate, {batch, heads, chunks, chunk_size}), -1);
    auto row = mx::reshape(mx::arange(chunk_size, mx::int32), {chunk_size, 1});
    auto column = mx::reshape(mx::arange(chunk_size, mx::int32), {1, chunk_size});
    // Mask BEFORE exp: upper-triangular positive decay differences can overflow.
    auto decay = mx::exp(mx::where(row >= column,
        mx::expand_dims(cumulative, -1) - mx::expand_dims(cumulative, -2), Tensor(-INFINITY)));
    auto kt = mx::swapaxes(k, -1, -2);
    auto system = mx::where(row > column, mx::matmul(kb, kt) * decay, Tensor(0.f));
    // Solve the same unit-lower-triangular system as the official reference.
    // Forward substitution is only 64 rows per chunk and each row stays on
    // Metal. This avoids the power-series inverse's observed instability on
    // the full PE system prompt; long autoregressive acceptance is separate.
    auto values = solve_unit_lower(system, vb, chunk_size, cancelled);
    auto decayed_keys = solve_unit_lower(
        system, kb * mx::expand_dims(mx::exp(cumulative), -1), chunk_size, cancelled);
    auto intra = mx::matmul(q, kt) * decay;
    q = q * mx::expand_dims(mx::exp(cumulative), -1);
    auto last = slice_axis(cumulative, -1, chunk_size - 1, chunk_size);
    k = k * mx::expand_dims(mx::exp(last - cumulative), -1);
    auto chunk_decay = mx::expand_dims(mx::exp(last), -1);
    auto state = initial_state ? mx::astype(*initial_state, mx::float32) :
        mx::zeros({batch, heads, kd, vd});
    auto at = [](const Tensor &tensor, int i) {
        return mx::squeeze(slice_axis(tensor, 2, i, i + 1), 2);
    };
    std::vector<Tensor> outputs;
    outputs.reserve(chunks);
    // Materialize intra-chunk transforms once before scanning chunk states.
    mx::eval({values, decayed_keys, intra, q, k, chunk_decay});
    for (int i = 0; i < chunks; ++i) {
        checkpoint(cancelled);
        auto delta = at(values, i) - mx::matmul(at(decayed_keys, i), state);
        outputs.push_back(mx::matmul(at(q, i), state) + mx::matmul(at(intra, i), delta));
        state = state * at(chunk_decay, i) + mx::matmul(mx::swapaxes(at(k, i), -1, -2), delta);
        mx::eval({outputs.back(), state});
    }
    auto output = mx::reshape(mx::stack(outputs, 2), {batch, heads, time + padding, vd});
    output = mx::transpose(slice_axis(output, 2, 0, time), {0, 2, 1, 3});
    checkpoint(cancelled);
    return {mx::astype(output, query.dtype()), state};
}
}
