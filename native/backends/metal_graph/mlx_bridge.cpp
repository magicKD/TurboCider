#include "mlx_bridge.hpp"

#include <mlx/allocator.h>
#include <limits>
#include <memory>
#include <new>

namespace tc::metal_graph {
namespace {
namespace mx = mlx::core;
ScalarType scalar_type(mx::Dtype dtype) {
    if (dtype == mx::bfloat16) return ScalarType::BF16;
    if (dtype == mx::float16) return ScalarType::FP16;
    if (dtype == mx::float32) return ScalarType::FP32;
    throw std::invalid_argument("native Metal bridge requires BF16/FP16/FP32");
}
mx::Dtype mlx_dtype(ScalarType type) {
    if (type == ScalarType::BF16) return mx::bfloat16;
    if (type == ScalarType::FP16) return mx::float16;
    if (type == ScalarType::FP32) return mx::float32;
    throw std::invalid_argument("native Metal output requires floating dtype");
}
size_t checked_product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::invalid_argument("native Metal matrix extent overflows");
    return a * b;
}
}

ImportedMatrix import_mlx_matrix(Backend& backend, const mx::array& array) {
    if (!array.is_available() || !array.buffer().ptr() || array.offset() < 0)
        throw std::invalid_argument("native Metal import requires produced MLX storage");
    if (array.ndim() != 2 && !(array.ndim() == 3 && array.shape(0) == 1))
        throw std::invalid_argument("native Metal import requires matrix or batch=1");
    const auto type = scalar_type(array.dtype());
    const auto rows = array.shape(array.ndim() - 2);
    const auto columns = array.shape(array.ndim() - 1);
    if (rows <= 0 || columns <= 0 || array.strides().back() != 1)
        throw std::invalid_argument("native Metal import requires contiguous matrix columns");
    const auto stride = array.strides()[array.ndim() - 2];
    if (stride < columns)
        throw std::invalid_argument("native Metal import rejects overlapping matrix rows");
    const size_t pitch = checked_product(size_t(stride), array.itemsize());
    const size_t last = checked_product(size_t(columns), array.itemsize());
    const size_t prefix = checked_product(size_t(rows - 1), pitch);
    if (prefix > std::numeric_limits<size_t>::max() - last ||
        pitch > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("native Metal matrix range exceeds kernel ABI");
    const size_t extent = prefix + last;
    const size_t offset = size_t(array.offset());
    if (offset > array.buffer_size() || extent > array.buffer_size() - offset)
        throw std::invalid_argument("native Metal matrix escapes its backing buffer");
    // Keep the actual MLX Data allocation owner. Retaining only MTLBuffer would
    // allow the MLX allocator to donate/recycle the storage while Metal uses it.
    auto owner = std::make_shared<mx::array>(array);
    auto backing = backend.import_buffer(
        const_cast<void*>(array.buffer().ptr()), std::move(owner), {}, array.data_shared_ptr());
    return {backing.view(offset, extent, type),
            uint32_t(rows), uint32_t(columns), uint32_t(pitch)};
}

mx::array allocate_mlx_matrix(uint32_t rows, uint32_t columns,
                             ScalarType type, bool batch_dimension) {
    if (!rows || !columns || rows > uint32_t(std::numeric_limits<int>::max()) ||
        columns > uint32_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native Metal output geometry invalid");
    const auto dtype = mlx_dtype(type);
    const size_t bytes = checked_product(
        checked_product(rows, columns), scalar_bytes(type));
    auto storage = mx::allocator::malloc(bytes);
    if (!storage.ptr()) throw std::bad_alloc();
    const mx::Shape shape = batch_dimension
        ? mx::Shape{1, int(rows), int(columns)}
        : mx::Shape{int(rows), int(columns)};
    return mx::array(storage, shape, dtype);
}
} // namespace tc::metal_graph
