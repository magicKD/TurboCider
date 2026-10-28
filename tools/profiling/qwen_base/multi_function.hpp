#pragma once
// Independent capability source, not a production FFN or speed claim. Both
// functions share max-row input/output pitch and the same runtime weight port.
#include "../../../native/backends/ane_runtime.hpp"
#include <string>

namespace tc::profiling {
inline std::string multi_function_matmul_mil() {
    auto dimensions = [](int rows,int columns) {return "[1, 1, "+std::to_string(rows)+", "+std::to_string(columns)+"]";};
    auto tensor = [&](int columns) {return "tensor<fp16, "+dimensions(128,columns)+">";};
    auto buffer = [&](int columns,int pitch) {
        const auto plane = std::to_string(128*pitch);
        return "tensor_buffer<fp16, shape="+dimensions(128,columns)+", strides=["+plane+", "+plane+", "+
            std::to_string(pitch)+", 1], interleave_factors=[1, 1, 1, 1]>";
    };
    std::string result="program(1.3)\n{\n";
    for (int rows : {32,64}) {
        result += "    func rows"+std::to_string(rows)+"<ios18>("+buffer(64,64)+" x, "+buffer(128,128)+" w) {\n";
        result += "        "+tensor(64)+" xt = tensor_buffer_to_tensor<ios17>(input = x);\n";
        result += "        "+tensor(128)+" wt = tensor_buffer_to_tensor<ios17>(input = w);\n";
        result += "        "+tensor(rows)+" selected = slice_by_size(x = xt, begin = tensor<int32, [4]>([0, 0, 0, 0]), size = tensor<int32, [4]>("+dimensions(128,rows)+"));\n";
        result += "        "+tensor(rows)+" product = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wt, y = selected);\n";
        result += "        "+buffer(rows,64)+" y = tensor_to_tensor_buffer<ios17>(input = product, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([8192, 8192, 64, 1]));\n";
        result += "    } -> (y);\n";
    }
    return result+"}\n";
}
} // namespace tc::profiling
