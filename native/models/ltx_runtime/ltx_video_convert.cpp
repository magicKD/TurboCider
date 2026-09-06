#include "ltx_video_convert.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

struct ConvertRows {
    uint8_t *output;
    const uint16_t *input;
    size_t plane_elements;
    size_t rows;
    size_t width;
    size_t jobs;
};

float bf16_to_f32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void convert_rows(void *opaque, size_t job) {
    const auto& context = *static_cast<const ConvertRows *>(opaque);
    const size_t first_row = context.rows * job / context.jobs;
    const size_t last_row = context.rows * (job + 1u) / context.jobs;
    for (size_t row = first_row; row < last_row; ++row) {
        const size_t source_row = row * context.width;
        const size_t destination_row = source_row * 3u;
        for (size_t x = 0; x < context.width; ++x) {
            const size_t source = source_row + x;
            const size_t destination = destination_row + x * 3u;
            for (size_t channel = 0; channel < 3u; ++channel) {
                float value = bf16_to_f32(
                    context.input[channel * context.plane_elements + source]);
                value = std::clamp(value * 0.5f + 0.5f, 0.0f, 1.0f);
                context.output[destination + channel] = static_cast<uint8_t>(
                    std::lround(value * 255.0f));
            }
        }
    }
}

int fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) std::snprintf(error, error_size, "%s", message);
    return 0;
}

}  // namespace

extern "C" int ltx_video_bf16_planar_to_rgb24(
        uint8_t *output, size_t output_bytes,
        const uint16_t *input, size_t input_elements,
        uint32_t frames, uint32_t height, uint32_t width,
        char *error, size_t error_size) {
    if (!output || !input || !frames || !height || !width)
        return fail(error, error_size, "invalid LTX RGB conversion arguments");
    if (static_cast<size_t>(frames) > SIZE_MAX / height ||
        static_cast<size_t>(frames) * height > SIZE_MAX / width)
        return fail(error, error_size, "LTX RGB conversion geometry overflows");
    const size_t plane_elements = static_cast<size_t>(frames) * height * width;
    if (plane_elements > SIZE_MAX / 3u ||
        input_elements != plane_elements * 3u ||
        output_bytes != plane_elements * 3u)
        return fail(error, error_size, "LTX RGB conversion shape mismatch");
    const size_t rows = static_cast<size_t>(frames) * height;
    const size_t jobs = std::min<size_t>(8u, rows);
    ConvertRows context{output, input, plane_elements, rows, width, jobs};
    dispatch_apply_f(jobs,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &context, convert_rows);
    return 1;
}
