#include "ane_mil.hpp"
#include <algorithm>
#include "../ane_w8a8_math.hpp"
#include <cstring>

namespace tc::ane::private_api {
namespace {
std::string shape(int rows, int cols) { return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(cols) + "]"; }
std::string tensor(int rows, int cols) { return "tensor<fp16, " + shape(rows, cols) + ">"; }
std::string buffer(int rows, int cols) {
    const int stride = (cols + 31) / 32 * 32;
    const auto plane = std::to_string(uint64_t(rows) * stride);
    return "tensor_buffer<fp16, shape=" + shape(rows, cols) + ", strides=[" + plane + ", " + plane + ", " +
        std::to_string(stride) + ", 1], interleave_factors=[1, 1, 1, 1]>";
}
}
std::string fp16_program(const GraphShape &s) {
    if (s.rows <= 0 || s.rows > 32768 || s.hidden <= 0 || s.hidden > 32768 || s.width <= 0 || s.width > 32768 ||
        s.tile_k <= 0 || s.tile_k > 32768 || s.tile_n <= 0 || s.tile_n > 32768 ||
        (s.kind != Kind::Matmul && s.kind != Kind::SwiGLU) || (s.lora_inputs && s.kind != Kind::SwiGLU))
        throw CapabilityError("private ANE FP16 micrograph geometry/kind unsupported");
    if (s.lora_inputs && s.hidden + s.width > 32768)
        throw CapabilityError("private ANE packed output exceeds surface height limit");
    // Prevent huge externally selected graph expansion before allocation.
    const auto tiles = [](int dimension, int tile) { return (uint64_t(dimension) + tile - 1) / tile; };
    const uint64_t up_nodes = tiles(s.hidden, s.tile_k) * tiles(s.width, s.tile_n);
    const uint64_t nodes = s.kind == Kind::Matmul ? up_nodes :
        2 * up_nodes + tiles(s.width, s.tile_k) * tiles(s.hidden, s.tile_n);
    if (nodes > 4096) throw CapabilityError("private ANE FP16 micrograph exceeds tile limit");
    std::string parameters, body;
    auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
    auto input = [&](const std::string &name, int rows, int cols) {
        if (!parameters.empty()) parameters += ", ";
        parameters += buffer(rows, cols) + " " + name;
        line(tensor(rows, cols) + " " + name + "_t = tensor_buffer_to_tensor<ios17>(input = " + name + ")");
    };
    auto value = [&](const std::string &name, int rows, const std::string &expr) {
        line(tensor(rows, s.rows) + " " + name + " = " + expr);
    };
    auto projection = [&](const std::string &name, const std::string &weight, const std::string &x, int n, int k) {
        std::vector<std::string> outputs;
        for (int nb = 0; nb < n; nb += s.tile_n) {
            const int count = std::min(s.tile_n, n - nb);
            std::string total;
            for (int kb = 0; kb < k; kb += s.tile_k) {
                const int width = std::min(s.tile_k, k - kb);
                const auto suffix = "_" + std::to_string(nb) + "_" + std::to_string(kb);
                const auto w = name + "w" + suffix, a = name + "x" + suffix, p = name + "p" + suffix;
                line(tensor(count, width) + " " + w + " = slice_by_size(x = " + weight +
                    ", begin = tensor<int32, [4]>([0, 0, " + std::to_string(nb) + ", " + std::to_string(kb) +
                    "]), size = tensor<int32, [4]>(" + shape(count, width) + "))");
                value(a, width, "slice_by_size(x = " + x + ", begin = tensor<int32, [4]>([0, 0, " +
                    std::to_string(kb) + ", 0]), size = tensor<int32, [4]>(" + shape(width, s.rows) + "))");
                value(p, count, "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + w + ", y = " + a + ")");
                if (total.empty()) total = p;
                else { const auto sum = name + "sum" + suffix; value(sum, count, "add(x = " + total + ", y = " + p + ")"); total = sum; }
            }
            outputs.push_back(total);
        }
        if (outputs.size() == 1) return outputs.front();
        std::string tuple = "(";
        for (const auto &out : outputs) { if (tuple.size() > 1) tuple += ", "; tuple += out; }
        tuple += ")";
        value(name, n, "concat(values = " + tuple + ", axis = int32(2), interleave = bool(false))");
        return name;
    };
    auto output = [&](const std::string &name, const std::string &x, int rows) {
        const int pitch = (s.rows + 31) / 32 * 32;
        const auto plane = std::to_string(uint64_t(rows) * pitch);
        line(buffer(rows, s.rows) + " " + name + " = tensor_to_tensor_buffer<ios17>(input = " + x +
            ", interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([" +
            plane + ", " + plane + ", " + std::to_string(pitch) + ", 1]))");
    };
    input("x", s.hidden, s.rows);
    if (s.kind == Kind::Matmul) {
        input("w", s.width, s.hidden);
        output("y", projection("m", "w_t", "x_t", s.width, s.hidden), s.width);
    } else {
        input("wg", s.width, s.hidden); input("wu", s.width, s.hidden); input("wd", s.hidden, s.width);
        auto g = projection("g", "wg_t", "x_t", s.width, s.hidden);
        auto u = projection("u", "wu_t", "x_t", s.width, s.hidden);
        if (s.lora_inputs) {
            input("dg", s.width, s.rows); input("du", s.width, s.rows);
            value("gc", s.width, "add(x = " + g + ", y = dg_t)");
            value("uc", s.width, "add(x = " + u + ", y = du_t)");
            g = "gc"; u = "uc";
        }
        // Match the public runtime's tested exp lowering. The private ANE
        // sigmoid LUT fails the signed sparse A/B/A oracle on M4 Max; do not
        // hide that compiler error by widening the numerical tolerance.
        value("neg", s.width, "mul(x = " + g + ", y = fp16(-1))");
        value("eg", s.width, "exp(x = neg)");
        value("denom", s.width, "add(x = eg, y = fp16(1))");
        value("silu", s.width, "real_div(x = " + g + ", y = denom)");
        value("hidden", s.width, "mul(x = silu, y = " + u + ")");
        const auto y = projection("d", "wd_t", "hidden", s.hidden, s.width);
        if (s.lora_inputs) {
            // One physical output symbol, so the ANE done event covers BOTH
            // down and corrected hidden. Do not race a second output using
            // a signal attached to only the first symbol.
            value("packed_yh", s.hidden + s.width, "concat(values = (" + y + ", hidden), axis = int32(2), interleave = bool(false))");
            output("y", "packed_yh", s.hidden + s.width);
        } else output("y", y, s.hidden);
    }
    return "program(1.3)\n{\n    func main_ane<ios18>(" + parameters + ") {\n" + body +
        "    } -> (y);\n}\n";
}
std::string w8_matmul_program(const GraphShape &s) {
    if (s.kind != Kind::Matmul || s.lora_inputs || s.rows <= 0 || s.rows > 2048 || s.hidden <= 0 || s.hidden > 4096 ||
        s.width <= 0 || s.width > 16384 || s.hidden % 128)
        throw CapabilityError("private W8A8 MatMul probe geometry unsupported");
    const auto typed = [](const char *type, int r, int c) { return std::string("tensor<") + type + ", " + shape(r, c) + ">"; };
    const auto input_buffer = [](int r, int c) {
        const int pitch = (c + 63) / 64 * 64;
        const auto plane = std::to_string(uint64_t(r) * pitch);
        return "tensor_buffer<int8, shape=" + shape(r, c) + ", strides=[" + plane + ", " + plane + ", " +
            std::to_string(pitch) + ", 1], interleave_factors=[1, 1, 1, 1]>";
    };
    std::string body;
    auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
    line(typed("int8", s.hidden, s.rows) + " xt = tensor_buffer_to_tensor<ios17>(input = x)");
    line(typed("int8", s.width, s.hidden) + " wt = tensor_buffer_to_tensor<ios17>(input = w)");
    line(typed("fp16", s.hidden, s.rows) + " xd = dequantize(input = xt, scale = fp16(0x1p-7))");
    line(typed("fp16", s.width, s.hidden) + " wd = dequantize(input = wt, scale = fp16(0x1p-7))");
    line(typed("fp16", s.width, s.rows) + " yn = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wd, y = xd)");
    const int pitch = (s.rows + 31) / 32 * 32;
    const auto plane = std::to_string(uint64_t(s.width) * pitch);
    line(buffer(s.width, s.rows) + " y = tensor_to_tensor_buffer<ios17>(input = yn, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
        "strides = tensor<int64, [4]>([" + plane + ", " + plane + ", " + std::to_string(pitch) + ", 1]))");
    return "program(1.3)\n{\n    func main_ane<ios18>(" + input_buffer(s.hidden, s.rows) + " x, " + input_buffer(s.width, s.hidden) +
        " w) {\n" + body + "    } -> (y);\n}\n";
}
W8FfnProgram w8_swiglu_program(const GraphShape &s, uint64_t seed, float headroom) {
    // An explicit 4224-row bucket can cover 1024px image tokens and up to
    // 128 caption tokens in one request. Keep the existing smaller buckets
    // available: fewer handoffs are not a model-speed/quality guarantee.
    // IOSurface extents, memory admission and full fallback remain checked.
    if (s.kind != Kind::SwiGLU || s.rows <= 0 || s.rows > 4224 || s.hidden <= 0 || s.hidden > 4096 || s.hidden % 128 ||
        s.width <= 0 || s.width > 16384 || s.width % 512 || !std::isfinite(headroom) || headroom < 1 || headroom > 4096 ||
        std::log2(headroom) != std::floor(std::log2(headroom)))
        throw CapabilityError("private W8A8 SwiGLU geometry/headroom unsupported");
    W8FfnProgram result;
    result.headroom = headroom; result.packed_rows = s.hidden + 1 + (s.lora_inputs ? s.width : 0);
    if (result.packed_rows > 32768) throw CapabilityError("private W8A8 packed output too tall");
    // A model-independent grouped H512 convolution blob, never checkpoint W.
    const uint64_t count = uint64_t(s.width) * 512, bytes = count * 2;
    result.constants.resize(128 + bytes);
    auto put = [&](size_t at, auto value) { std::memcpy(result.constants.data() + at, &value, sizeof(value)); };
    put(0, uint32_t(1)); put(4, uint32_t(2)); put(64, uint32_t(0xdeadbeef)); put(68, uint32_t(1));
    put(72, bytes); put(80, uint64_t(128));
    const float norm = 1.f / std::sqrt(512.f);
    for (int out = 0; out < s.width; ++out) for (int in = 0; in < 512; ++in) {
        const int lane = out % 512;
        const float h = (std::popcount(unsigned(lane & in)) & 1) ? -1.f : 1.f;
        put(128 + (size_t(out) * 512 + in) * 2, std::bit_cast<uint16_t>(_Float16(float(rotation_sign(seed, in)) * h * norm)));
    }
    const auto typed = [](const char *type, int r, int c) { return std::string("tensor<") + type + ", " + shape(r, c) + ">"; };
    const auto in_buffer = [](const char *type, int r, int c) {
        const int item = type[0] == 'i' ? 1 : 2, pitch = (c * item + 63) / 64 * 64 / item;
        const auto plane = std::to_string(uint64_t(r) * pitch);
        return std::string("tensor_buffer<") + type + ", shape=" + shape(r, c) + ", strides=[" + plane + ", " + plane + ", " +
            std::to_string(pitch) + ", 1], interleave_factors=[1, 1, 1, 1]>";
    };
    std::string parameters, body;
    auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
    auto input = [&](const char *type, const std::string &name, int r, int c) {
        if (!parameters.empty()) parameters += ", "; parameters += in_buffer(type, r, c) + " " + name;
        line(typed(type, r, c) + " " + name + "_t = tensor_buffer_to_tensor<ios17>(input = " + name + ")");
    };
    auto f = [&](const std::string &name, int r, const std::string &expression) { line(typed("fp16", r, s.rows) + " " + name + " = " + expression); };
    auto projection = [&](const std::string &name, const std::string &w, const std::string &x, int n, int k) {
        const int tile = s.tile_k > 0 ? std::min(s.tile_k, 2048) : 2048;
        if (tile < 128) throw CapabilityError("private W8A8 K tile too small");
        std::string total;
        for (int begin = 0; begin < k; begin += tile) {
            const int width = std::min(tile, k - begin); const auto suffix = std::to_string(begin);
            const auto wt = name + "w" + suffix, xt = name + "x" + suffix, p = name + "p" + suffix;
            // Slice INT8 first, then dequantize each MatMul operand directly.
            // This retains the Splash-style dequantize -> MatMul pattern;
            // slicing a full FP16 dequantized matrix can inhibit quantized
            // lowering. Actual hardware arithmetic still needs observation.
            line(typed("int8", n, width) + " " + wt + "q = slice_by_size(x = " + w + "_t, begin = tensor<int32, [4]>([0, 0, 0, " +
                suffix + "]), size = tensor<int32, [4]>(" + shape(n, width) + "))");
            line(typed("fp16", n, width) + " " + wt + " = dequantize(input = " + wt + "q, scale = fp16(0x1p-7))");
            line(typed("int8", width, s.rows) + " " + xt + "q = slice_by_size(x = " + x + ", begin = tensor<int32, [4]>([0, 0, " + suffix + ", 0]), size = tensor<int32, [4]>(" + shape(width, s.rows) + "))");
            f(xt, width, "dequantize(input = " + xt + "q, scale = fp16(0x1p-7))");
            f(p, n, "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + wt + ", y = " + xt + ")");
            if (total.empty()) total = p;
            else { const auto sum = name + "s" + suffix; f(sum, n, "add(x = " + total + ", y = " + p + ")"); total = sum; }
        }
        return total;
    };
    input("int8", "x", s.hidden, s.rows); input("fp16", "tx", 1, s.rows);
    input("int8", "wg", s.width, s.hidden); input("fp16", "sg", s.width, 1);
    input("int8", "wu", s.width, s.hidden); input("fp16", "su", s.width, 1);
    input("int8", "wd", s.hidden, s.width);
    const auto g = projection("g", "wg", "x_t", s.width, s.hidden), u = projection("u", "wu", "x_t", s.width, s.hidden);
    f("gs", s.width, "mul(x = " + g + ", y = sg_t)");
    f("gt", s.width, "mul(x = gs, y = tx_t)");
    line(typed("fp16", s.width, 1) + " su_safe = real_div(x = su_t, y = fp16(" + std::to_string(headroom) + "))");
    f("us_norm", s.width, "mul(x = " + u + ", y = su_safe)");
    f("us", s.width, "mul(x = us_norm, y = tx_t)");
    std::string gate = "gt", up = "us";
    if (s.lora_inputs) {
        input("fp16", "dg", s.width, s.rows); input("fp16", "du", s.width, s.rows);
        f("gc", s.width, "add(x = gt, y = dg_t)");
        f("du_safe", s.width, "real_div(x = du_t, y = fp16(" + std::to_string(headroom) + "))");
        f("uc", s.width, "add(x = us, y = du_safe)"); gate = "gc"; up = "uc";
    }
    f("neg", s.width, "mul(x = " + gate + ", y = fp16(-1))"); f("eg", s.width, "exp(x = neg)");
    f("denom", s.width, "add(x = eg, y = fp16(1))"); f("silu", s.width, "real_div(x = " + gate + ", y = denom)");
    f("hsafe", s.width, "mul(x = silu, y = " + up + ")");
    const auto c = std::to_string(s.width), m = std::to_string(s.rows);
    line("tensor<fp16, [1, " + c + ", 1, " + m + "]> h4 = reshape(x = hsafe, shape = tensor<int32, [4]>([1, " + c + ", 1, " + m + "]))");
    line("tensor<fp16, [" + c + ", 512, 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" + c + ", 512, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]");
    line("tensor<fp16, [1, " + c + ", 1, " + m + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
        std::to_string(s.width / 512) + "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)");
    f("hr", s.width, "reshape(x = hr4, shape = tensor<int32, [4]>(" + shape(s.width, s.rows) + "))");
    f("habs", s.width, "abs(x = hr)"); f("peak", 1, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
    f("floor", 1, "maximum(x = peak, y = fp16(0x1p-12))");
    f("ratio", s.width, "real_div(x = hr, y = floor)"); f("a8", s.width, "mul(x = ratio, y = fp16(127))");
    line(typed("int8", s.width, s.rows) + " hq = quantize(input = a8, scale = fp16(1), output_dtype = string(\"int8\"))");
    f("hscale", 1, "mul(x = floor, y = fp16(0x1.0204081020408p+0))");
    const auto y = projection("d", "wd", "hq", s.hidden, s.width);
    const std::string outputs = "(" + y + ", hscale" + (s.lora_inputs ? ", hsafe" : "") + ")";
    f("packed", result.packed_rows, "concat(values = " + outputs + ", axis = int32(2), interleave = bool(false))");
    const int pitch = (s.rows + 31) / 32 * 32; const auto plane = std::to_string(uint64_t(result.packed_rows) * pitch);
    line(buffer(result.packed_rows, s.rows) + " y = tensor_to_tensor_buffer<ios17>(input = packed, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([" + plane + ", " + plane + ", " + std::to_string(pitch) + ", 1]))");
    result.mil = "program(1.3)\n{\n    func main_ane<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n}\n";
    return result;
}
}
