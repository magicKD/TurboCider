// Same-binary packed-QMM / bounded-predecode / dense-GEMM component screen.
// No model rewrite, persistent sidecar, default runtime route or E2E claim.
#include "runtime/streaming/affine_dense_window.hpp"
#include "backends/convrot_rotation.hpp"
#include "backends/affine_gpu_fp32.hpp"
#include "backends/affine_gpu_mpp.hpp"
#include "backends/affine_gpu_shared.hpp"
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
// Attribution-only mirror of the guarded synchronous prepare recipe. The
// added eval boundaries deliberately perturb scheduling; this is NOT a new
// faster window, an asynchronous decoder, or a production replacement.
struct PrepareParts { double cache, decode, finite, synchronize, publication, total; };
std::pair<Tensor,PrepareParts> prepare_parts(const Packed &packed,int bits,int group,
                                          MemoryLedger &ledger,uint64_t generation) {
    const auto total_start=Clock::now();
    const auto &[words,scales,biases]=packed;
    require(generation && (bits==4 || bits==8) && (group==32 || group==64 || group==128) &&
        words.ndim()==2 && words.dtype()==mx::uint32 && words.shape(0)>0 && words.shape(1)>0 &&
        (scales.dtype()==mx::float16 || scales.dtype()==mx::bfloat16) && biases.dtype()==scales.dtype() &&
        scales.ndim()==2 && biases.shape()==scales.shape() && scales.shape(0)==words.shape(0) &&
        uint64_t(words.shape(1))*(32/bits)==uint64_t(scales.shape(1))*group,
        "prepare attribution needs original typed affine geometry");
    const uint64_t bytes=gguf::checked_mul(gguf::checked_mul(uint64_t(words.shape(0)),
        uint64_t(words.shape(1))*(32/bits)),2);
    require(bytes<=uint64_t(256)<<20,"prepare attribution exceeds bounded component target");
    const uint64_t upper=streaming::gguf_storage::capacity_upper(bytes);
    auto reservation=ledger.try_reserve(MemoryClass::ConversionScratch,upper,"affine-prepare-attribution-v1");
    require(reservation.has_value(),"prepare attribution admission denied including escaped readers");
    PrepareParts parts{};
    auto tick=Clock::now();
    std::optional<streaming::gguf_storage::ExactCapacityCacheScope> exact;exact.emplace();
    parts.cache=elapsed(tick);tick=Clock::now();
    auto dense=mx::dequantize(words,scales,biases,group,bits,"affine",std::nullopt,scales.dtype());
    mx::eval(dense);parts.decode=elapsed(tick);tick=Clock::now();
    auto finite=mx::all(mx::isfinite(dense));mx::eval(finite);
    parts.finite=elapsed(tick);tick=Clock::now();
    mx::synchronize();parts.synchronize=elapsed(tick);tick=Clock::now();
    require(finite.item<bool>(),"prepare attribution rejects nonfinite decoded coefficients");
    dense.detach();dense.set_siblings({},0);
    auto data=dense.data_shared_ptr();const uint64_t actual=mx::allocator::allocator().size(data->buffer);
    require(actual>=bytes && actual<=upper,"prepare attribution allocation exceeds admitted upper");
    auto claim=std::make_shared<StorageLease>(reservation->commit({0x54435041525453ull,
        uint64_t(reinterpret_cast<uintptr_t>(data->buffer.ptr())),actual,generation}));
    auto prior=data->d;data->d=[prior,claim](mx::allocator::Buffer buffer){prior(buffer);};
    exact.reset();parts.publication=elapsed(tick);parts.total=elapsed(total_start);
    return {dense,parts};
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
        {
            auto measured=prepare_parts(a,bits,group,ledger,8);
            auto expected=mx::dequantize(a[0],a[1],a[2],group,bits,"affine",std::nullopt,dtype);
            require(same_bits(measured.first,expected),"attribution changed typed affine coefficients");
            auto reader=mx::sum(mx::astype(measured.first,mx::float32));
            measured.first=Tensor(0.f);
            require(ledger.snapshot().storage_count==1,"attribution lost lazy reader claim");
            mx::eval(reader);
        }
        mx::synchronize();
        require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"attribution leaked backing claim");
        bool bad_parts=false;
        try {(void)prepare_parts(bad,bits,group,ledger,9);}catch(const std::exception &){bad_parts=true;}
        require(bad_parts && !ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,
                "nonfinite attribution published/leaked backing");
        MemoryLedger denied_ledger(upper-1);bool denied_parts=false;
        try {(void)prepare_parts(a,bits,group,denied_ledger,10);}catch(const std::exception &){denied_parts=true;}
        require(denied_parts && !denied_ledger.snapshot().storage_bytes && !denied_ledger.snapshot().reserved_bytes,
                "attribution bypassed admission or leaked reservation");
        ++cases;
    }
    std::cout << "PASS 12 typed affine dense window cases: Q4/Q8, FP16/BF16, g32/64/128, exact coefficients, reuse/generation, escaped/lazy claims, finite gate, budget/failure/refill\n";
    require(cases == 12, "unexpected window case count");
    std::cout << "PASS 12 typed prepare attribution cases: exact coefficients, lazy claims, finite failure and admission rollback\n";
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
void benchmark_prepare(const Packed &packed,int bits,int iterations,bool convrot,const std::string &source_hash) {
    mx::eval({packed[0],packed[1],packed[2]});
    const int n=packed[0].shape(0),k=packed[1].shape(1)*32;
    const uint64_t bytes=uint64_t(n)*k*2,upper=streaming::gguf_storage::capacity_upper(bytes);
    require(bytes<=uint64_t(256)<<20,"prepare control exceeds component target bound");
    MemoryLedger ledger(2*upper);AffineDenseWindow window(ledger,upper,1);
    {
        auto control=window.prepare(packed,bits,32,source_hash,1);
        auto instrumented=prepare_parts(packed,bits,32,ledger,2);
        require(same_bits(control,instrumented.first),"real-source attribution changed typed coefficients");
    }
    window.clear();mx::synchronize();
    std::vector<double> controls,cache,decode,finite,synchronize,publication,totals;
    uint64_t generation=3;
    auto run_control=[&] {
        window.clear();mx::synchronize();require(!ledger.snapshot().storage_bytes,"prepare previous reader escaped");
        const auto start=Clock::now();
        {auto dense=window.prepare(packed,bits,32,source_hash,generation++);controls.push_back(elapsed(start));}
        window.clear();mx::synchronize();
    };
    auto run_parts=[&] {
        require(!ledger.snapshot().storage_bytes,"prepare attribution reader escaped");
        {
            auto result=prepare_parts(packed,bits,32,ledger,generation++);const auto &p=result.second;
            cache.push_back(p.cache);decode.push_back(p.decode);finite.push_back(p.finite);
            synchronize.push_back(p.synchronize);publication.push_back(p.publication);totals.push_back(p.total);
        }
        mx::synchronize();
    };
    for(int i=0;i<3;++i){run_control();run_parts();}
    controls.clear();cache.clear();decode.clear();finite.clear();synchronize.clear();publication.clear();totals.clear();
    for(int i=0;i<iterations;++i)if(i&1){run_parts();run_control();}else{run_control();run_parts();}
    require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"prepare attribution final claim leaked");
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-affine-prepare-attribution-v1\",\"basis\":\""
        <<(convrot?"Comfy-H256-legacy-BF16-scale":"GGUF-native-affine-FP16")<<"\",\"source_payload_or_codes_sha256\":\""<<source_hash
        <<"\",\"bits\":"<<bits<<",\"N\":"<<n<<",\"K\":"<<k<<",\"dense_bytes\":"<<bytes
        <<",\"two_dense_capacity_upper\":"<<2*upper<<",\"warmups_per_arm\":3,\"samples_per_arm\":"<<iterations
        <<",\"original_prepare_median_ms\":"<<median(controls)<<",\"attributed_prepare_median_ms\":"<<median(totals)
        <<",\"extra_eval_boundaries\":true,\"typed_weight_coefficients_exact\":true,\"parts\":[";
    const std::array<std::pair<const char *,const std::vector<double> *>,7> values{{
        {"original_prepare",&controls},{"attributed_total",&totals},{"cache_hint_and_clear",&cache},
        {"dequantize_and_eval",&decode},{"finite_graph_and_eval",&finite},{"synchronize",&synchronize},{"publication_and_restore_cache",&publication}}};
    for(size_t i=0;i<values.size();++i){const auto &[name,times]=values[i];
        std::cout<<(i?",":"")<<"{\"name\":\""<<name<<"\",\"median_ms\":"<<median(*times)<<",\"samples_ms\":";samples(*times);std::cout<<'}';}
    std::cout<<"],\"dense_claim_bytes_after_cleanup\":0,\"qualification_passed\":false,\"default_route_changed\":false,"
        "\"asynchronous_layer_ahead_decode_implemented\":false,\"scope\":\"guarded synchronous prepare attribution with perturbing eval boundaries; host spans, not device timestamps or a faster consumer\","
        "\"excludes\":[\"source-read/packing\",\"GEMM/complete-consumer-window\",\"full-model/LoRA/media\",\"whole-process-memory\",\"ANE/physical-overlap\"]}\n";
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

// Measure the whole decode + R actual consumers, rather than inferring it
// from separately measured medians. Every consumer gets a different prepared
// activation; one immutable weight generation is reused only within a trial.
// This is NOT a claim that a one/two-matrix LRU survives a full DiT traversal.
void benchmark_reuse(const Packed &packed, int bits, int rows, int iterations,
                     int reuses, bool convrot, const std::string &source_hash) {
    const int n = packed[0].shape(0), k = packed[1].shape(1)*32;
    const auto dtype = packed[1].dtype();
    require(uint64_t(rows)*k*2*reuses <= uint64_t(256)<<20,
            "prepared activation set exceeds 256MiB component limit");
    std::vector<Tensor> inputs;
    for (int i = 0; i < reuses; ++i) {
        auto x = mx::astype(mx::reshape(mx::sin(mx::arange(rows*k,mx::float32)*.017f +
            Tensor(float(i)*.123f)), {1,rows,k}), dtype);
        if (convrot) x = convrot_kernel::rotate(x,convrot_kernel::Rotation::Shared);
        inputs.push_back(x);
    }
    mx::eval(inputs);
    mx::eval({packed[0],packed[1],packed[2]});
    const uint64_t bytes = uint64_t(n)*k*2, upper = streaming::gguf_storage::capacity_upper(bytes);
    MemoryLedger ledger(upper);
    AffineDenseWindow window(ledger,upper,1);
    uint64_t generation = 1;
    bool exact = true;
    float maximum_relative_l2 = 0;
    for (const auto &input : inputs) {
        auto dense = window.prepare(packed,bits,32,source_hash,generation);
        auto a = mx::quantized_matmul(input,packed[0],packed[1],packed[2],true,32,bits,"affine");
        auto b = mx::matmul(input,mx::transpose(dense));
        mx::eval({a,b});
        require(mx::all(mx::isfinite(a)).item<bool>() && mx::all(mx::isfinite(b)).item<bool>(),
                "nonfinite complete reuse window output");
        exact = exact && same_bits(a,b);
        auto af = mx::astype(a,mx::float32), bf = mx::astype(b,mx::float32);
        const float error = mx::sqrt(mx::sum(mx::square(af-bf)) /
            mx::maximum(mx::sum(mx::square(af)),Tensor(1e-20f))).item<float>();
        maximum_relative_l2 = std::max(maximum_relative_l2,error);
    }
    window.clear();
    mx::synchronize();
    require(!ledger.snapshot().storage_bytes,"parity reader escaped before complete-window timing");

    struct Sample { double ms; uint64_t hits, misses, decoded_bytes; };
    auto run = [&](bool predecoded) {
        window.clear();
        mx::synchronize();
        require(!ledger.snapshot().storage_bytes,"previous reuse trial retained an escaped reader");
        ++generation;
        const auto before = window.stats();
        const auto start = Clock::now();
        for (const auto &input : inputs) {
            if (predecoded) {
                auto dense = window.prepare(packed,bits,32,source_hash,generation);
                auto y = mx::matmul(input,mx::transpose(dense));
                mx::eval(y);
            } else {
                auto y = mx::quantized_matmul(input,packed[0],packed[1],packed[2],true,32,bits,"affine");
                mx::eval(y);
            }
        }
        const double ms = elapsed(start);
        const auto &after = window.stats();
        Sample sample{ms,after.hits-before.hits,after.misses-before.misses,
                      after.decoded_bytes-before.decoded_bytes};
        require(sample.hits == (predecoded ? uint64_t(reuses-1) : 0) &&
                    sample.misses == uint64_t(predecoded) &&
                    sample.decoded_bytes == (predecoded ? bytes : 0),
                "reuse trial did not follow one-decode/R-consumer policy");
        return sample;
    };
    for (int i = 0; i < 3; ++i) { run(false); run(true); }
    std::vector<Sample> dense_samples;
    std::vector<double> packed_ms, dense_ms;
    for (int i = 0; i < iterations; ++i) {
        if (i & 1) {
            auto s = run(true); dense_samples.push_back(s); dense_ms.push_back(s.ms);
            packed_ms.push_back(run(false).ms);
        } else {
            packed_ms.push_back(run(false).ms);
            auto s = run(true); dense_samples.push_back(s); dense_ms.push_back(s.ms);
        }
    }
    window.clear();
    mx::synchronize();
    require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,
            "complete reuse screen leaked dense claim");
    std::cout << std::setprecision(12) << "{\"schema\":\"tc-real-weight-reuse-window-v1\","
        "\"scope\":\"serial complete operator host span; one decode plus actual distinct-input consumers; not model/prefetch/physical overlap qualification\","
        "\"basis\":\"" << (convrot ? "Comfy-H256-rotated-legacy-BF16-scale" : "GGUF-native-affine-FP16")
        << "\",\"source_payload_or_codes_sha256\":\"" << source_hash
        << "\",\"stored_scales_sha256\":\"" << sha256(packed[1].data<void>(),packed[1].nbytes())
        << "\",\"bits\":" << bits << ",\"M\":" << rows << ",\"N\":" << n << ",\"K\":" << k
        << ",\"actual_consumers_per_trial\":" << reuses << ",\"dense_bytes\":" << bytes
        << ",\"dense_capacity_upper_bytes\":" << upper << ",\"prepared_input_bytes\":" << uint64_t(rows)*k*2*reuses
        << ",\"warmups_per_arm\":3,\"cleared_every_trial\":true,\"packed_samples_ms\":";
    samples(packed_ms); std::cout << ",\"decode_plus_reuse_samples_ms\":"; samples(dense_ms);
    std::cout << ",\"dense_trial_counters\":[";
    for (size_t i = 0; i < dense_samples.size(); ++i) {
        const auto &s = dense_samples[i];
        std::cout << (i ? "," : "") << "{\"hits\":" << s.hits << ",\"misses\":" << s.misses
            << ",\"decoded_bytes\":" << s.decoded_bytes << '}';
    }
    std::cout << "],\"packed_median_ms\":" << median(packed_ms)
        << ",\"decode_plus_reuse_median_ms\":" << median(dense_ms)
        << ",\"all_inputs_output_bit_exact\":" << (exact ? "true" : "false")
        << ",\"maximum_output_relative_l2\":" << maximum_relative_l2
        << ",\"dense_claim_bytes_after_cleanup\":" << ledger.snapshot().storage_bytes
        << ",\"qualification_passed\":false,\"excludes\":[\"source-read/packing\",\"input-rotation\","
        "\"full-layer-traversal/LoRA/media\",\"whole-process-memory\",\"asynchronous-prefetch\",\"ANE\"]}\n";
}
void benchmark_mpp(const Packed &packed,int bits,int rows,int iterations,bool convrot,
                   const std::string &source_hash) {
    const int n=packed[0].shape(0),k=packed[1].shape(1)*32;
    const auto dtype=packed[1].dtype();
    const uint64_t bytes=uint64_t(n)*k*2;
    require(rows>=32 && bytes<=256ull<<20 && uint64_t(rows)*k*2<=256ull<<20,
            "MPP screen requires M>=32, dense/input extent each <=256MiB");
    auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*k,mx::float32)*.017f),{1,rows,k}),dtype);
    if(convrot)x=convrot_kernel::rotate(x,convrot_kernel::Rotation::Shared);
    mx::eval({x,packed[0],packed[1],packed[2]});
    const auto upper=streaming::gguf_storage::capacity_upper(bytes);
    MemoryLedger ledger(upper);AffineDenseWindow window(ledger,upper);
    // One bounded decode is a separate setup observation, NOT amortized away
    // when recommending predecode. Its bank is used only by the control arm.
    const auto start=Clock::now();
    auto dense=window.prepare(packed,bits,32,source_hash,1);
    const double decode_setup_ms=elapsed(start);
    auto expected=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(dense,mx::float32)));
    mx::eval(expected);
    struct Recipe {const char *name;int bk,sn,sm;};
    const std::vector<Recipe> recipes{{"packed_qmm",0,0,0},{"prepared_dense_gemm",0,0,0},
        {"original_f32_decode",0,0,0},{"mpp_register_32x32",32,32,16},
        {"mpp_register_32x64",32,64,16},{"mpp_register_64x32",64,32,16},
        {"mpp_register_m32_32x32",32,32,32},{"mpp_register_m64_32x32",32,32,64}};
    auto project=[&](size_t index) {
        if(index==0)return mx::quantized_matmul(x,packed[0],packed[1],packed[2],true,32,bits,"affine");
        if(index==1)return mx::matmul(x,mx::transpose(dense));
        auto value=index==2 ? affine_gpu::projection_fp32(x,packed[0],packed[1],packed[2],bits,0,n,0,k) :
            affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,0,n,0,k,recipes[index].bk,recipes[index].sn,recipes[index].sm);
        return mx::astype(value,dtype); // explicitly include original output cast
    };
    std::vector<double> f32_rel(recipes.size(),0),narrow_rel,maximum_abs;
    std::vector<bool> exact;
    auto reference=project(0);mx::eval(reference);
    for(size_t index=0;index<recipes.size();++index) {
        if(index>=2) {
            auto value=index==2 ? affine_gpu::projection_fp32(x,packed[0],packed[1],packed[2],bits,0,n,0,k) :
                affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,0,n,0,k,recipes[index].bk,recipes[index].sn,recipes[index].sm);
            const double error=mx::sqrt(mx::sum(mx::square(value-expected))/mx::sum(mx::square(expected))).item<float>();
            require(std::isfinite(error) && error<=3e-6,"real-weight typed F32 projection gate failed: "+
                std::string(recipes[index].name)+" error="+std::to_string(error));
            f32_rel[index]=error;
        }
        auto value=project(index);mx::eval(value);
        auto vf=mx::astype(value,mx::float32),rf=mx::astype(reference,mx::float32);
        const double error=mx::sqrt(mx::sum(mx::square(vf-rf))/mx::sum(mx::square(rf))).item<float>();
        require(mx::all(mx::isfinite(value)).item<bool>() && std::isfinite(error),"nonfinite real-weight MPP output");
        narrow_rel.push_back(error);maximum_abs.push_back(mx::max(mx::abs(vf-rf)).item<float>());
        exact.push_back(same_bits(value,reference));
        for(int warm=0;warm<3;++warm)mx::eval(project(index));
    }
    std::vector<std::vector<double>> times(recipes.size());
    for(int iteration=0;iteration<iterations;++iteration)for(size_t visit=0;visit<recipes.size();++visit) {
        const size_t index=(visit+iteration)%recipes.size();const auto started=Clock::now();
        auto value=project(index);mx::eval(value);times[index].push_back(elapsed(started));
    }
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-affine-register-mpp-screen-v1\","
        "\"scope\":\"serial full operator host span with original final dtype cast; real weights/synthetic prepared input; not full-model or GPU timestamps\","
        "\"basis\":\""<<(convrot?"Comfy-H256-rotated-legacy-BF16-scale":"GGUF-native-affine-FP16")
        <<"\",\"source_payload_or_codes_sha256\":\""<<source_hash
        <<"\",\"stored_scales_sha256\":\""<<sha256(packed[1].data<void>(),packed[1].nbytes())
        <<"\",\"bits\":"<<bits<<",\"M\":"<<rows<<",\"N\":"<<n<<",\"K\":"<<k
        <<",\"warmups_per_recipe\":3,\"samples_per_recipe\":"<<iterations
        <<",\"control_dense_setup_ms\":"<<decode_setup_ms<<",\"control_dense_bytes\":"<<bytes
        <<",\"mpp_global_dense_bank_bytes\":0,\"recipes\":[";
    for(size_t index=0;index<recipes.size();++index) {
        std::cout<<(index?",":"")<<"{\"name\":\""<<recipes[index].name<<"\",\"median_ms\":"<<median(times[index])
            <<",\"fp32_rel_l2_vs_independent_typed_coefficients\":";
        if(index<2)std::cout<<"null";else std::cout<<f32_rel[index];
        std::cout<<",\"narrow_rel_l2_vs_packed\":"<<narrow_rel[index]
            <<",\"max_abs_vs_packed\":"<<maximum_abs[index]<<",\"output_byte_exact_vs_packed\":"<<(exact[index]?"true":"false")
            <<",\"samples_ms\":";samples(times[index]);std::cout<<'}';
    }
    dense=Tensor(0.f);window.clear();mx::synchronize();
    require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"MPP control leaked dense bank claim");
    std::cout<<"],\"dense_control_claim_bytes_after_cleanup\":"<<ledger.snapshot().storage_bytes
        <<",\"qualification_passed\":false,\"default_route_changed\":false,"
        "\"excludes\":[\"source-read/packing\",\"input-rotation\",\"LoRA/full-model/media\",\"whole-process-memory\",\"ANE/physical-overlap\"]}\n";
}
void benchmark_shared(const Packed &packed,int bits,int rows,int iterations,bool convrot,
                      const std::string &source_hash,int requested_columns) {
    const int n=packed[0].shape(0),physical=packed[1].shape(1)*32,k=requested_columns?requested_columns:physical;
    require(rows>=32 && k>0 && k<=physical && k%32==0 && (!convrot || k%256==0),"shared screen requires bounded aligned original columns");
    const auto dtype=packed[1].dtype();
    auto full=mx::astype(mx::reshape(mx::sin(mx::arange(rows*physical,mx::float32)*.017f),{1,rows,physical}),dtype);
    if(convrot)full=convrot_kernel::rotate(full,convrot_kernel::Rotation::Shared);
    auto x=mx::contiguous(mx::slice(full,{0,0,0},{1,rows,k}));mx::eval({x,packed[0],packed[1],packed[2]});
    const uint64_t bytes=uint64_t(n)*physical*2;require(bytes<=uint64_t(256)<<20,"shared dense oracle exceeds component bound");
    MemoryLedger ledger(streaming::gguf_storage::capacity_upper(bytes));
    AffineDenseWindow window(ledger,streaming::gguf_storage::capacity_upper(bytes),1);
    const auto start=Clock::now();auto dense=window.prepare(packed,bits,32,source_hash,1);const double setup=elapsed(start);
    auto selected=mx::slice(dense,{0,0},{n,k});
    auto expected=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(selected,mx::float32)));mx::eval(expected);
    struct Recipe {std::string name;int bm,bn,bk;bool narrow;};
    std::vector<Recipe> recipes{{"packed_qmm",0,0,0,true},{"original_mpp_m64",0,0,0,false}};
    for(const auto &[bm,bn,bk]:std::vector<std::tuple<int,int,int>>{{32,64,32},{64,64,64},{128,64,64},{64,128,64},{64,64,128},{64,128,128}})
        for(bool narrow:{false,true})recipes.push_back({"shared_"+std::to_string(bm)+"x"+std::to_string(bn)+"k"+std::to_string(bk)+(narrow?"_narrow":"_f32_cast"),bm,bn,bk,narrow});
    auto raw=[&](size_t index) {
        if(index==0)return mx::quantized_matmul(x,mx::slice(packed[0],{0,0},{n,k/(32/bits)}),
            mx::slice(packed[1],{0,0},{n,k/32}),mx::slice(packed[2],{0,0},{n,k/32}),true,32,bits,"affine");
        if(index==1)return affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,0,n,0,k,32,32,64);
        const auto &r=recipes[index];return affine_gpu::projection_shared(x,packed[0],packed[1],packed[2],bits,0,n,0,k,
            r.bm,r.bn,r.bk,r.narrow?dtype:mx::float32);
    };
    auto project=[&](size_t index){return mx::astype(raw(index),dtype);};
    auto reference=project(0);mx::eval(reference);std::vector<double> rel,fp32_rel;std::vector<bool> exact;
    for(size_t i=0;i<recipes.size();++i) {
        auto value=raw(i);mx::eval(value);auto vf=mx::astype(value,mx::float32);
        const double f32=mx::sqrt(mx::sum(mx::square(vf-expected))/mx::maximum(mx::sum(mx::square(expected)),Tensor(1e-20f))).item<float>();
        require(mx::all(mx::isfinite(value)).item<bool>() && std::isfinite(f32) && f32<=(value.dtype()==mx::float32?3e-6:.003),"shared real-source numerical gate failed: "+recipes[i].name);
        auto output=mx::astype(value,dtype);auto of=mx::astype(output,mx::float32),rf=mx::astype(reference,mx::float32);
        rel.push_back(mx::sqrt(mx::sum(mx::square(of-rf))/mx::maximum(mx::sum(mx::square(rf)),Tensor(1e-20f))).item<float>());
        fp32_rel.push_back(f32);exact.push_back(same_bits(output,reference));
        for(int warm=0;warm<3;++warm)mx::eval(project(i));
    }
    std::vector<std::vector<double>> times(recipes.size());
    for(int iteration=0;iteration<iterations;++iteration)for(size_t visit=0;visit<recipes.size();++visit) {
        const size_t i=(visit+iteration)%recipes.size();const auto tick=Clock::now();mx::eval(project(i));times[i].push_back(elapsed(tick));
    }
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-affine-shared-word-screen-v1\",\"scope\":\"real original weight/synthetic prepared input, full operator host span including final output cast; not model/device trace\","
        "\"basis\":\""<<(convrot?"Comfy-H256-legacy-BF16-scale":"GGUF-native-affine-FP16")<<"\",\"source_payload_or_codes_sha256\":\""<<source_hash
        <<"\",\"stored_scales_sha256\":\""<<sha256(packed[1].data<void>(),packed[1].nbytes())<<"\",\"bits\":"<<bits
        <<",\"M\":"<<rows<<",\"N\":"<<n<<",\"K\":"<<k<<",\"physical_W_pitch\":"<<physical<<",\"warmups_per_recipe\":3,\"samples_per_recipe\":"<<iterations
        <<",\"control_dense_setup_ms\":"<<setup<<",\"control_dense_bytes\":"<<bytes<<",\"candidate_global_dense_bytes\":0,\"recipes\":[";
    for(size_t i=0;i<recipes.size();++i) {
        const auto &r=recipes[i];std::cout<<(i?",":"")<<"{\"name\":\""<<r.name<<"\",\"median_ms\":"<<median(times[i])
            <<",\"requested_BM\":"<<r.bm<<",\"requested_BN\":"<<r.bn<<",\"requested_BK\":"<<r.bk
            <<",\"effective_BK\":"<<(i>=2?affine_gpu::shared_effective_k_tile(k,r.bk):0)
            <<",\"explicit_threadgroup_weight_bytes\":"<<(i>=2?r.bn*affine_gpu::shared_effective_k_tile(k,r.bk)*2:0)
            <<",\"output_relative_l2_vs_typed_f32\":"<<fp32_rel[i]<<",\"narrow_relative_l2_vs_packed\":"<<rel[i]
            <<",\"output_byte_exact_vs_packed\":"<<(exact[i]?"true":"false")<<",\"samples_ms\":";samples(times[i]);std::cout<<'}';
    }
    expected=Tensor(0.f);selected=Tensor(0.f);dense=Tensor(0.f);window.clear();mx::synchronize();
    require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"shared oracle dense claim leaked");
    std::cout<<"],\"dense_control_claim_bytes_after_cleanup\":0,\"qualification_passed\":false,\"default_route_changed\":false,"
        "\"asynchronous_layer_ahead_decode_implemented\":false,\"excludes\":[\"source-read/packing\",\"input-rotation\",\"LoRA/full-model/media\",\"whole-process-memory\",\"ANE/physical-overlap\"]}\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        configure_streams();
        mx::set_cache_limit(256ull << 20);
        if (argc == 1) { self_test(); return 0; }
        require((argc == 6 || argc == 7) && (std::string(argv[1]) == "gguf" || std::string(argv[1]) == "convrot" ||
                std::string(argv[1]) == "gguf-mpp" || std::string(argv[1]) == "convrot-mpp" ||
                std::string(argv[1]) == "gguf-shared" || std::string(argv[1]) == "convrot-shared" ||
                std::string(argv[1]) == "gguf-prepare" || std::string(argv[1]) == "convrot-prepare"),
                "usage: gpu-weight-consumer-probe [gguf|convrot checkpoint tensor/prefix M iterations [actual-reuses]] or [gguf-mpp|convrot-mpp checkpoint tensor/prefix M iterations] or [gguf-shared|convrot-shared checkpoint tensor/prefix M iterations [selected-columns]] or [gguf-prepare|convrot-prepare checkpoint tensor/prefix 1 iterations]");
        const bool mpp=std::string(argv[1]).ends_with("-mpp");
        const bool shared=std::string(argv[1]).ends_with("-shared");
        const bool prepare=std::string(argv[1]).ends_with("-prepare");
        const int rows = std::stoi(argv[4]), iterations = std::stoi(argv[5]);
        const int reuses = argc == 7 ? std::stoi(argv[6]) : 0;
        require(rows >= 1 && rows <= 4224 && iterations >= 9 && iterations <= 99 && iterations % 2,
                "component requires M=1..4224, odd iterations=9..99");
        require(argc != 7 || shared || (reuses >= 1 && reuses <= 16),"actual-reuses must be 1..16");
        require(!mpp || (argc==6 && rows>=32),"MPP screen requires M>=32 and no actual-reuses option");
        require(!prepare || (argc==6 && rows==1),"prepare attribution requires M=1 and no reuse option");
        int bits = 8;
        std::string hash;
        const bool convrot = std::string(argv[1]).starts_with("convrot");
        auto source = convrot ? load_convrot(argv[2], argv[3], hash) : load_gguf(argv[2], argv[3], bits, hash);
        if(prepare)benchmark_prepare(source,bits,iterations,convrot,hash);
        else if(shared)benchmark_shared(source,bits,rows,iterations,convrot,hash,reuses);
        else if(mpp)benchmark_mpp(source,bits,rows,iterations,convrot,hash);
        else if (reuses) benchmark_reuse(source,bits,rows,iterations,reuses,convrot,hash);
        else benchmark(source, bits, rows, iterations, convrot, hash);
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
