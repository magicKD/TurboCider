#pragma once

#include "backend.hpp"
#include <mlx/array.h>
#include <cstdint>

namespace tc::metal_graph {

// Transitional stage/island boundary only. The native executor does not need
// MLX, and this importer never evaluates an array or copies tensor contents.
// The caller proves input readiness (including previous GPU queue completion)
// before submitting native commands. Native writes must physically finish
// before an imported output is published back to MLX.
struct ImportedMatrix {
    BufferView storage;
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t row_stride_bytes = 0;
};

ImportedMatrix import_mlx_matrix(Backend&, const mlx::core::array&);

// Allocated by MLX and imported with its original Data owner: one backing
// allocation, no native clone and no duplicate allocation-hook accounting.
mlx::core::array allocate_mlx_matrix(uint32_t rows, uint32_t columns,
                                    ScalarType type = ScalarType::BF16,
                                    bool batch_dimension = true);

} // namespace tc::metal_graph
