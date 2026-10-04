// CPU-only source -> caller-owned FP16 target; no framework/device execution.
#include "../../native/backends/ane_runtime_packed.hpp"

extern "C" int tc_convrot_stage(const int8_t *codes, size_t code_bytes,
        const float *scales, size_t scale_bytes, int rows, int cols,
        uint16_t *output, size_t output_elements, bool scalar_only, float headroom) {
    try {
        tc::ane::ConvrotView source{codes, code_bytes, rows, cols, 0,
            {scales, scale_bytes, rows, 1, 0, tc::ane::DType::FP32}};
        tc::ane::validate_convrot_view(source);
        if (!output || output_elements < size_t(rows) * cols) return 1;
        for (int row = 0; row < rows; ++row)
            if (!tc::ane::convrot_fp16_row(source, size_t(row), output + size_t(row) * cols,
                                          scalar_only, headroom)) return 2;
        return 0;
    } catch (const std::exception &) { return 3; }
}
