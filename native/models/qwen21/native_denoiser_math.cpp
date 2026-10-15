#include "native_denoiser.hpp"

#include <cmath>

namespace tc::qwen21::native_graph {
std::array<uint16_t,512> timestep_input(float sigma) {
    if (!std::isfinite(sigma) || sigma<0.f || sigma>1.f)
        throw std::invalid_argument("native Qwen time must be finite in [0,1]");
    std::array<uint16_t,512> result{};
    for (int i=0;i<128;++i) {
        const float frequency=float(std::exp(-std::log(10000.0)*double(i)/128.0));
        const float scaled=sigma*1000.f;
        const float angle=scaled*frequency;
        result[size_t(i)]=metal_graph::step_ops::bf16_bits(std::cos(angle));
        result[size_t(i)+128]=metal_graph::step_ops::bf16_bits(std::sin(angle));
        result[size_t(i)+256]=metal_graph::step_ops::bf16_bits(1.f);
        result[size_t(i)+384]=0;
    }
    return result;
}
RopeTables make_rope_tables(const SequenceGeometry& sequence) {
    if (sequence.positions.empty() || sequence.positions.size()>65536 ||
        sequence.prefix_length<0 || size_t(sequence.prefix_length)>=sequence.positions.size())
        throw std::invalid_argument("native Qwen RoPE requires bounded sequence geometry");
    RopeTables result;
    result.cosine.reserve(sequence.positions.size()*64);
    result.sine.reserve(sequence.positions.size()*64);
    constexpr std::array<int,3> axes{16,56,56};
    for (const auto& position:sequence.positions) {
        for (size_t axis=0;axis<axes.size();++axis) {
            if (!std::isfinite(position[axis]))
                throw std::invalid_argument("native Qwen RoPE position must be finite");
            for (int i=0;i<axes[axis];i+=2) {
                const float omega=1.f/std::pow(10000.f,float(i)/float(axes[axis]));
                const float angle=position[axis]*omega;
                if (!std::isfinite(angle)) throw std::invalid_argument("native Qwen RoPE angle overflow");
                result.cosine.push_back(std::cos(angle));
                result.sine.push_back(std::sin(angle));
            }
        }
    }
    return result;
}
} // namespace tc::qwen21::native_graph
