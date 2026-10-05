// Same-binary packed-QMM / bounded-predecode / dense-GEMM component screen.
// No model rewrite, persistent sidecar, default runtime route or E2E claim.
#include "runtime/streaming/affine_dense_window.hpp"
#include "backends/convrot_rotation.hpp"
#include "backends/mlx_fd_reader.hpp"
#include "core/gguf_affine.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

using namespace tc;
using streaming::AffineDenseWindow;
namespace {
using Clock = std::chrono::steady_clock;
using Packed = std::array<Tensor, 3>;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}
void samples(const std::vector<double> &values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
    std::cout << ']';
}
std::string sha256(const void *data, size_t bytes) {
    require(bytes <= UINT32_MAX, "component payload exceeds SHA256 one-shot limit");
    unsigned char out[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, CC_LONG(bytes), out);
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : out) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}
bool same_bits(const Tensor &a, const Tensor &b) {
    return a.dtype() == b.dtype() && a.shape() == b.shape() &&
        mx::all(mx::view(a, mx::uint8) == mx::view(b, mx::uint8)).item<bool>();
}
Packed fixture(int bits, int group, mx::Dtype dtype, int seed) {
    auto weight = mx::astype(mx::random::normal({65, 512}, mx::float32, mx::random::key(seed)) * .03f, dtype);
    auto arrays = mx::quantize(weight, group, bits);
    mx::eval(arrays);
    return {arrays[0], arrays[1], arrays[2]};
}
void self_test() {
    int cases = 0;
    for (int bits : {4, 8}) for (int group : {32, 64, 128}) for (auto dtype : {mx::float16, mx::bfloat16}) {
        auto a = fixture(bits, group, dtype, 41), b = fixture(bits, group, dtype, 42);
        const auto upper = streaming::gguf_storage::capacity_upper(65 * 512 * 2);
        MemoryLedger ledger(2 * upper);
        AffineDenseWindow window(ledger, 2 * upper);
        std::optional<Tensor> escaped;
        {
            auto decoded = window.prepare(a, bits, group, "A", 1);
            auto expected = mx::dequantize(a[0], a[1], a[2], group, bits, "affine", std::nullopt, dtype);
            require(same_bits(decoded, expected), "window changed original typed affine coefficients");
            auto hit = window.prepare(a, bits, group, "A", 1);
            require(decoded.buffer().ptr() == hit.buffer().ptr() && window.stats().hits == 1,
                    "same content ticket did not reuse dense backing");
            auto next = window.prepare(b, bits, group, "B", 1);
            require(!same_bits(decoded, next), "A/B fixture did not change source");
            escaped = decoded;
        }
        mx::synchronize();
        bool denied = false;
        try { (void)window.prepare(b, bits, group, "C", 2); }
        catch (const std::exception &) { denied = true; }
        require(denied && ledger.snapshot().storage_count == 2 && !ledger.snapshot().reserved_bytes,
                "escaped old reader escaped new decode admission/rollback");
        window.clear();
        mx::synchronize();
        require(ledger.snapshot().storage_count == 1, "cleared window lost escaped backing claim");
        {
            auto lazy = mx::sum(mx::astype(*escaped, mx::float32));
            escaped.reset();
            require(ledger.snapshot().storage_count == 1, "lazy consumer lost backing claim");
            mx::eval(lazy);
        }
        mx::synchronize();
        require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes, "window leaked backing/claim");
        {
            // Reusing the exact same source object under a new content
            // generation MUST decode again, not hit an address/shape cache.
            auto first = window.prepare(a, bits, group, "A", 3);
            auto changed = window.prepare(a, bits, group, "A", 4);
            require(same_bits(first, changed) && first.buffer().ptr() != changed.buffer().ptr(),
                    "new content generation reused previous decode");
        }
        window.clear();
        mx::synchronize();
        require(!ledger.snapshot().storage_bytes, "generation switch leaked backing");
        auto bad = a;
        bad[1] = mx::full(a[1].shape(), INFINITY, dtype);
        bool rejected = false;
        try { (void)window.prepare(bad, bits, group, "bad", 5); }
        catch (const std::exception &) { rejected = true; }
        require(rejected && !ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes &&
                !window.stats().retained_bytes, "nonfinite decode published or leaked");
        {
            auto recovered = window.prepare(a, bits, group, "A", 6);
            require(mx::all(mx::isfinite(recovered)).item<bool>(), "clean refill after failure failed");
        }
        window.clear();
        mx::synchronize();
        bool budget_rejected = false;
        try { AffineDenseWindow small(ledger, upper - 1); (void)small.prepare(a, bits, group, "A", 7); }
        catch (const std::exception &) { budget_rejected = true; }
        require(budget_rejected, "oversized matrix admitted by window");
        require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,
                "oversized matrix failure leaked live claims");
        ++cases;
    }
    std::cout << "PASS 12 typed affine dense window cases: Q4/Q8, FP16/BF16, g32/64/128, exact coefficients, reuse/generation, escaped/lazy claims, finite gate, budget/failure/refill\n";
    require(cases == 12, "unexpected window case count");
}
Packed load_gguf(const char *path, const char *name, int &bits, std::string &payload_hash) {
    struct File { int fd; ~File() { if (fd >= 0) ::close(fd); } } file{::open(path, O_RDONLY | O_CLOEXEC)};
    struct stat before{}, after{};
    require(file.fd >= 0 && ::fstat(file.fd, &before) == 0 && S_ISREG(before.st_mode), "cannot open GGUF component source");
    const auto directory = gguf::read_directory(file.fd, uint64_t(before.st_size));
    const auto &tensor = directory.tensor(name);
    require(tensor.dimensions.size() == 2 && (tensor.type == 2 || tensor.type == 3 || tensor.type == 8) &&
            tensor.rows() <= 16384 && tensor.columns() <= 16384 && tensor.bytes <= 256ull << 20,
            "component requires bounded two-dimensional native Q4_0/Q4_1/Q8_0 GGUF tensor");
    std::vector<std::byte> raw(tensor.bytes);
    uint64_t done = 0;
    while (done < tensor.bytes) {
        auto n = ::pread(file.fd, raw.data() + done, size_t(std::min<uint64_t>(tensor.bytes - done, 1ull << 20)),
                         off_t(tensor.file_offset + done));
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "GGUF component short read/source changed");
        done += uint64_t(n);
    }
    require(::fstat(file.fd, &after) == 0 && before.st_size == after.st_size &&
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec && before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
            "GGUF component source changed during read");
    bits = tensor.type == 8 ? 8 : 4;
    const size_t groups = tensor.rows() * tensor.columns() / 32;
    std::vector<std::byte> codes(groups * bits * 4), scales(groups * 2), biases(groups * 2);
    gguf::pack_native_affine_all({raw, tensor.type, tensor.rows(), tensor.columns()},
        {std::span(codes), std::span(scales), std::span(biases)});
    payload_hash = sha256(raw.data(), raw.size());
    const int r = int(tensor.rows()), meta_bytes = int(tensor.columns() / 32 * 2);
    auto byte_array = [&](const std::vector<std::byte> &data, int row_bytes) {
        return Tensor(reinterpret_cast<const uint8_t *>(data.data()), {r, row_bytes}, mx::uint8);
    };
    return {mx::view(byte_array(codes, int(tensor.columns() * bits / 8)), mx::uint32),
            mx::view(byte_array(scales, meta_bytes), mx::float16),
            mx::view(byte_array(biases, meta_bytes), mx::float16)};
}
Packed load_convrot(const char *path, const char *prefix, std::string &codes_hash) {
    // Keep one immutable read identity through lazy materialization. No
    // mutable pathname reopen, and no raw GPU pointer as a host hash span.
    MlxOwnedFd fd(::open(path, O_RDONLY | O_CLOEXEC));
    struct stat before{}, after{};
    require(fd && ::fstat(fd.get(), &before) == 0 && S_ISREG(before.st_mode), "cannot open ConvRot component source");
    const int observed_fd = fd.get();
    auto reader = std::make_shared<MlxLeaseFdReader>(std::move(fd), "ConvRot component");
    auto loaded = mx::load_safetensors(reader, mx::Device(mx::Device::cpu));
    const auto &codes = loaded.first.at(std::string(prefix) + ".weight");
    const auto &raw_scales = loaded.first.at(std::string(prefix) + ".weight_scale");
    require(codes.dtype() == mx::int8 && codes.ndim() == 2 && codes.shape(0) <= 16384 && codes.shape(1) <= 16384 &&
            codes.shape(1) % 256 == 0 && raw_scales.shape() == mx::Shape{codes.shape(0), 1}, "invalid raw ConvRot component");
    // Publish raw provenance after a completed CPU load, before packing.
    mx::eval(codes);
    mx::synchronize();
    require(codes.flags().row_contiguous && codes.offset() >= 0 && codes.buffer_size() >=
            size_t(codes.offset()) * codes.itemsize() + codes.nbytes(), "raw ConvRot hash span is not prepared/contiguous");
    codes_hash = sha256(codes.data<int8_t>(), codes.nbytes());
    // Same legacy packing/dtype recipe as Weights::pack_convrot_q8. No
    // inverse rotation and no claim that BF16 stored scales equal raw FP32.
    auto words = mx::view(mx::astype(mx::astype(codes, mx::int32) + Tensor(128, mx::int32), mx::uint8), mx::uint32);
    // Repeat may be a broadcast/stride-zero view. Materialize metadata for
    // byte provenance; nbytes() is logical size, not the readable raw span.
    auto scales = mx::contiguous(mx::repeat(mx::astype(raw_scales, mx::bfloat16), codes.shape(1) / 32, 1));
    auto biases = mx::contiguous(scales * Tensor(-128.f, mx::bfloat16));
    mx::eval({codes, words, scales, biases});
    require(::fstat(observed_fd, &after) == 0 && before.st_size == after.st_size &&
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec && before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
            "ConvRot component source changed during read/packing");
    return {words, scales, biases};
}
void benchmark(const Packed &packed, int bits, int rows, int iterations, bool convrot,
               const std::string &source_hash) {
    const int n = packed[0].shape(0), k = packed[1].shape(1) * 32;
    const auto dtype = packed[1].dtype();
    auto x = mx::astype(mx::reshape(mx::sin(mx::arange(rows * k, mx::float32) * .017f), {1, rows, k}), dtype);
    if (convrot) x = convrot_kernel::rotate(x, convrot_kernel::Rotation::Shared);
    mx::eval({x, packed[0], packed[1], packed[2]});
    const auto bytes = uint64_t(n) * k * 2, upper = streaming::gguf_storage::capacity_upper(bytes);
    MemoryLedger ledger(2 * upper);
    AffineDenseWindow window(ledger, 2 * upper);
    // Warm compiler separately; every decode sample then evicts/redecodes.
    { auto warm = window.prepare(packed, bits, 32, source_hash, 1); }
    window.clear();
    std::vector<double> decode_ms;
    uint64_t generation = 2;
    for (int i = 0; i < iterations; ++i) {
        window.clear();
        auto start = Clock::now();
        { auto decoded = window.prepare(packed, bits, 32, source_hash, generation++); }
        decode_ms.push_back(elapsed(start));
    }
    auto dense = window.prepare(packed, bits, 32, source_hash, generation - 1);
    auto qmm = [&] { return mx::quantized_matmul(x, packed[0], packed[1], packed[2], true, 32, bits, "affine"); };
    auto gemm = [&] { return mx::matmul(x, mx::transpose(dense)); };
    auto run = [&](bool predecoded) {
        auto start = Clock::now();
        auto y = predecoded ? gemm() : qmm();
        mx::eval(y);
        return elapsed(start);
    };
    for (int i = 0; i < 3; ++i) { run(false); run(true); }
    std::vector<double> packed_ms, dense_ms;
    for (int i = 0; i < iterations; ++i) {
        if (i & 1) { dense_ms.push_back(run(true)); packed_ms.push_back(run(false)); }
        else { packed_ms.push_back(run(false)); dense_ms.push_back(run(true)); }
    }
    auto a = qmm(), b = gemm();
    mx::eval({a, b});
    require(mx::all(mx::isfinite(a)).item<bool>() && mx::all(mx::isfinite(b)).item<bool>(), "nonfinite benchmark output");
    auto af = mx::astype(a, mx::float32), bf = mx::astype(b, mx::float32);
    const float rel = mx::sqrt(mx::sum(mx::square(af - bf)) / mx::sum(mx::square(af))).item<float>();
    const float abs = mx::max(mx::abs(af - bf)).item<float>();
    const double q = median(packed_ms), d = median(dense_ms), c = median(decode_ms);
    std::cout << std::setprecision(12) << "{\"kind\":\"real_packed_weight_consumer\",\"basis\":\""
        << (convrot ? "Comfy-H256-rotated-legacy-BF16-scale" : "GGUF-native-affine-FP16")
        << "\",\"" << (convrot ? "source_codes_sha256" : "source_payload_sha256")
        << "\":\"" << source_hash << "\",\"stored_scales_sha256\":\""
        << sha256(packed[1].data<void>(), packed[1].nbytes()) << "\",\"bits\":" << bits
        << ",\"M\":" << rows << ",\"N\":" << n << ",\"K\":" << k
        << ",\"dense_bytes\":" << bytes << ",\"two_dense_capacity_upper\":" << 2 * upper
        << ",\"actual_retained_dense_bytes\":" << window.stats().retained_bytes
        << ",\"warmups_per_compute_arm\":3,\"decode_samples_ms\":";
    samples(decode_ms); std::cout << ",\"packed_samples_ms\":"; samples(packed_ms);
    std::cout << ",\"dense_samples_ms\":"; samples(dense_ms);
    std::cout << ",\"decode_median_ms\":" << c << ",\"packed_median_ms\":" << q << ",\"dense_median_ms\":" << d
        << ",\"redecode_every_use_ms\":" << c + d << ",\"amortization_break_even_reuses\":";
    if (q > d) std::cout << std::max(1., std::floor(c / (q - d)) + 1.);
    else std::cout << "null";
    std::cout << ",\"output_bit_exact\":" << (same_bits(a, b) ? "true" : "false")
        << ",\"output_relative_l2\":" << rel << ",\"output_max_abs\":" << abs
        << ",\"excludes\":[\"source-read/packing\",\"input-rotation\",\"full-model/LoRA/quality\",\"whole-process-memory\",\"ANE/physical-overlap\"]}\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        configure_streams();
        mx::set_cache_limit(256ull << 20);
        if (argc == 1) { self_test(); return 0; }
        require(argc == 6 && (std::string(argv[1]) == "gguf" || std::string(argv[1]) == "convrot"),
                "usage: gpu-weight-consumer-probe [gguf|convrot checkpoint tensor/prefix M iterations]");
        const int rows = std::stoi(argv[4]), iterations = std::stoi(argv[5]);
        require(rows >= 1 && rows <= 4224 && iterations >= 9 && iterations <= 99 && iterations % 2,
                "component requires M=1..4224, odd iterations=9..99");
        int bits = 8;
        std::string hash;
        const bool convrot = std::string(argv[1]) == "convrot";
        auto source = convrot ? load_convrot(argv[2], argv[3], hash) : load_gguf(argv[2], argv[3], bits, hash);
        benchmark(source, bits, rows, iterations, convrot, hash);
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
