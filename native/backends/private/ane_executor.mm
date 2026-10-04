#include "ane_executor.hpp"
#include "ane_program.hpp"
#include "ane_mil.hpp"
#include "../ane_memory.hpp"
#include "../ane_runtime_packed.hpp"
#import <Foundation/Foundation.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>

namespace tc::ane {
namespace {
using Clock = std::chrono::steady_clock;
using private_api::Surface;
using private_api::Element;
struct OutputOverflow : std::runtime_error { using std::runtime_error::runtime_error; };
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void check(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
size_t stride(const MatrixView &v) { return v.row_stride_bytes ? v.row_stride_bytes : size_t(v.cols) * (v.dtype == DType::FP32 ? 4 : 2); }
void validate(const MatrixView &v) {
    check(v.dtype == DType::FP16 || v.dtype == DType::BF16 || v.dtype == DType::FP32, "private ANE unsupported source dtype");
    validate_packed_storage(v.data, v.bytes, v.rows, v.cols, stride(v), size_t(v.cols) * (v.dtype == DType::FP32 ? 4 : 2));
}
uint16_t *row(Surface &s, size_t r) { return reinterpret_cast<uint16_t *>(static_cast<char *>(s.data()) + r * s.pitch()); }
class Worker {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> job_;
    std::exception_ptr error_;
    bool busy_ = false, stop_ = false;
    std::thread thread_;
  public:
    Worker() : thread_([this] {
        for (;;) {
            std::function<void()> job;
            { std::unique_lock lock(mutex_); cv_.wait(lock, [&] { return stop_ || job_; });
              if (stop_) return; job = std::exchange(job_, {}); }
            std::exception_ptr error;
            @autoreleasepool { try { job(); } catch (...) { error = std::current_exception(); } }
            { std::lock_guard lock(mutex_); error_ = error; busy_ = false; }
            cv_.notify_all();
        }
    }) {}
    ~Worker() {
        try { join(); } catch (...) {}
        { std::lock_guard lock(mutex_); stop_ = true; } cv_.notify_all(); thread_.join();
    }
    void join() {
        std::unique_lock lock(mutex_); cv_.wait(lock, [&] { return !busy_; });
        auto error = std::exchange(error_, {}); if (error) std::rethrow_exception(error);
    }
    void submit(std::function<void()> job) {
        join(); { std::lock_guard lock(mutex_); job_ = std::move(job); busy_ = true; } cv_.notify_all();
    }
};
void fill_weight(Surface &s, const WeightView &source, float scale = 1.f) {
    std::visit([&](const auto &v) {
        using V = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<V, MatrixView>) validate(v);
        else if constexpr (std::is_same_v<V, AffineView>) validate_affine_view(v);
        else if constexpr (std::is_same_v<V, GgufView>) validate_gguf_view(v);
        else if constexpr (std::is_same_v<V, ConvrotView>) validate_convrot_view(v);
        else validate_convrot_affine_view(v);
        const int rows = [&] { if constexpr (std::is_same_v<V, ConvrotAffineView>) return v.packed.rows; else return v.rows; }();
        const int cols = [&] { if constexpr (std::is_same_v<V, ConvrotAffineView>) return v.packed.cols; else return v.cols; }();
        check(rows == int(s.rows()) && cols == int(s.columns()), "private ANE weight geometry mismatch");
        for (int r = 0; r < rows; ++r) {
            bool ok;
            if constexpr (std::is_same_v<V, MatrixView>) ok = convert_fp16_row(static_cast<const char *>(v.data) + size_t(r) * stride(v), row(s, r), cols, v.dtype, false, scale);
            else if constexpr (std::is_same_v<V, AffineView>) ok = affine_fp16_row(v, r, row(s, r), false, scale);
            else if constexpr (std::is_same_v<V, GgufView>) ok = gguf_fp16_row(v, r, row(s, r), false, scale);
            else if constexpr (std::is_same_v<V, ConvrotView>) ok = convrot_fp16_row(v, r, row(s, r), false, scale);
            else ok = convrot_affine_fp16_row(v, r, row(s, r), false, scale);
            check(ok, "private ANE weight conversion overflow/nonfinite input");
        }
    }, source);
}
// One bounded row scratch, never a full dense activation/weight temporary.
void fill_transposed(Surface &s, MatrixView v, int begin, float scale = 1.f) {
    validate(v);
    check(v.cols == int(s.rows()) && begin >= 0 && v.rows - begin >= int(s.columns()), "private ANE input geometry mismatch");
    std::array<uint16_t, 32768> scratch;
    for (uint32_t r = 0; r < s.columns(); ++r) {
        const auto *input = static_cast<const char *>(v.data) + size_t(begin + r) * stride(v);
        check(convert_fp16_row(input, scratch.data(), v.cols, v.dtype, false, scale), "private ANE input conversion overflow/nonfinite");
        for (uint32_t c = 0; c < s.rows(); ++c) row(s, c)[r] = scratch[c];
    }
}
void restore(const Surface &s, uint16_t *output, DType dtype, float scale) {
    const auto status = restore_fp16_matrix(static_cast<const uint16_t *>(s.data()), output,
        s.columns(), s.rows(), 1, s.pitch() / 2, dtype, scale);
    if (!status.source_finite) throw OutputOverflow("private ANE output overflow/nonfinite; full GPU fallback required");
    check(status.restored_finite, "private ANE restored output exceeds destination dtype; full GPU fallback required");
}
} // namespace

struct PrivateGraph::Impl {
    GraphShape shape;
    size_t allocated = 0, estimate = 0;
    double load_time = 0;
    private_api::Device device;
    std::unique_ptr<private_api::Program> program;
    std::vector<std::pair<std::string, Surface>> inputs, outputs;
    bool verified = false, staged = false, disabled = false;
    bool device_io = false;
    uint64_t timeline = 0;
    float headroom = 1.f;
    RunResult result;
    Worker worker; // drain before model/event/surfaces are released
    Surface &input(const std::string &name) {
        for (auto &entry : inputs) if (entry.first == name) return entry.second;
        throw std::runtime_error("private ANE input slot absent");
    }
    Surface output(const std::string &name) {
        if (name == "h" && shape.lora_inputs) return outputs.front().second.slice_rows(shape.hidden, shape.width);
        if (name == "y") return outputs.front().second.slice_rows(0, shape.output_width());
        throw std::runtime_error("private ANE output slot absent");
    }
    void predict(uint16_t *y, DType dtype, uint16_t *hidden = nullptr) {
        check(!disabled && timeline <= UINT64_MAX - 2, "private ANE executor disabled/timeline exhausted");
        const auto start = Clock::now();
        const uint64_t ready = timeline + 1, done = timeline + 2; timeline = done;
        auto ticket = program->enqueue(inputs, outputs, ready, done);
        try { device.signal(ready); }
        catch (...) { ticket.finish(std::chrono::milliseconds(0)); disabled = true; throw; }
        const auto completion = ticket.finish();
        if (!completion.ok) { disabled = true; throw CapabilityError(completion.error); }
        if (!device.wait(done, std::chrono::seconds(30))) { disabled = true; throw CapabilityError("private ANE GPU event wait failed"); }
        result.prediction_seconds += elapsed(start); ++result.calls;
        const auto restore_start = Clock::now();
        restore(output("y"), y, dtype, headroom);
        result.copied_output_bytes += size_t(shape.rows) * shape.output_width() * 2;
        if (shape.lora_inputs) {
            restore(output("h"), hidden, dtype, headroom);
            if (hidden) result.copied_output_bytes += size_t(shape.rows) * shape.width * 2;
        }
        result.output_seconds += elapsed(restore_start);
    }
    void increase_headroom() {
        check(shape.kind == Kind::SwiGLU && headroom < 4096.f, "private ANE exhausted FP16 headroom");
        // Scale only up, never gate or the nonlinearity. Hidden and down output
        // are both restored by the same factor. No master/source mutation.
        auto &slot = input("wu");
        for (uint32_t r = 0; r < slot.rows(); ++r)
            check(convert_fp16_row(row(slot, r), row(slot, r), slot.columns(), DType::FP16, false, .25f),
                  "private ANE nonfinite headroom weights");
        headroom *= 4.f; ++result.overflow_retries;
    }
};
PrivateGraph::PrivateGraph(GraphShape shape, size_t budget, const std::filesystem::path &cache)
    : impl_([] {
        if (!private_api::Program::healthy()) throw CapabilityError("private ANE disabled after failed/missing completion");
        return std::make_unique<Impl>();
    }()) {
    @autoreleasepool {
        const auto start = Clock::now();
        auto &p = *impl_; p.shape = shape;
        if (const char *value = std::getenv("TURBOCIDER_PRIVATE_ANE_GPU_IO")) {
            check(std::string_view(value) == "0" || std::string_view(value) == "1", "TURBOCIDER_PRIVATE_ANE_GPU_IO requires 0 or 1");
            p.device_io = std::string_view(value) == "1";
        }
        const auto mil = private_api::fp16_program(shape); // validate BEFORE multiplication/allocation
        const auto &s = p.shape;
        const size_t weights = size_t(s.hidden) * s.width * 2 * (s.kind == Kind::Matmul ? 1 : 3);
        p.estimate = 2 * weights + size_t(s.rows) * (6ull * s.width + 8ull * s.hidden) + (64ull << 20) + 65536;
        if (s.lora_inputs) p.estimate += size_t(s.rows) * s.width * 16;
        if (p.estimate > budget) throw MemoryBudgetError("private ANE memory budget exceeded");
        const auto observed = observe_runtime_memory(0);
        const auto admission = admit_memory(observed, {uint64_t(4) << 30, budget}, 0, p.estimate);
        if (!admission.allowed()) throw MemoryBudgetError("private ANE system memory admission denied");
        auto add = [&](auto &list, const char *name, int rows, int cols) {
            list.emplace_back(name, Surface(p.device, rows, cols, Element::FP16));
            p.allocated += list.back().second.bytes();
        };
        add(p.inputs, "x", s.hidden, s.rows);
        if (s.kind == Kind::Matmul) add(p.inputs, "w", s.width, s.hidden);
        else {
            add(p.inputs, "wg", s.width, s.hidden); add(p.inputs, "wu", s.width, s.hidden); add(p.inputs, "wd", s.hidden, s.width);
        }
        if (s.lora_inputs) { add(p.inputs, "dg", s.width, s.rows); add(p.inputs, "du", s.width, s.rows); }
        add(p.outputs, "y", s.output_width() + (s.lora_inputs ? s.width : 0), s.rows);
        if (p.allocated > budget || p.allocated > p.estimate) throw MemoryBudgetError("private ANE actual padded allocations exceed admission");
        std::filesystem::path root = cache;
        if (root.empty()) {
            NSString *user_cache = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject;
            check(user_cache != nil, "private ANE cache unavailable");
            root = std::filesystem::path(user_cache.UTF8String) / "TurboCider/ane/private";
        }
        p.program = std::make_unique<private_api::Program>(p.device, mil, std::span<const uint8_t>{}, root);
        auto check_names = [&](const auto &slots, auto names) {
            std::vector<std::string> declared; for (const auto &entry : slots) declared.push_back(entry.first);
            std::sort(declared.begin(), declared.end()); std::sort(names.begin(), names.end());
            check(declared == names, "private ANE compiled interface differs from native micrograph");
        };
        check_names(p.inputs, p.program->inputs()); check_names(p.outputs, p.program->outputs());
        p.load_time = elapsed(start);
    }
}
PrivateGraph::~PrivateGraph() = default;
const GraphShape &PrivateGraph::shape() const { return impl_->shape; }
size_t PrivateGraph::slot_bytes() const { return impl_->allocated; }
size_t PrivateGraph::estimated_bytes() const { return impl_->estimate; }
double PrivateGraph::load_seconds() const { return impl_->load_time; }
bool PrivateGraph::self_test(std::string &error) {
    auto &p = *impl_; p.worker.join(); p.verified = p.staged = false;
    p.headroom = 1.f;
    p.worker.submit([&p] {
        const auto &s = p.shape;
        std::vector<uint16_t> result(size_t(s.rows) * s.output_width());
        std::vector<uint16_t> hidden(s.lora_inputs ? size_t(s.rows) * s.width : 0);
        p.result = {};
        for (float scale : {.125f, -.25f, .125f}) {
            for (auto &entry : p.inputs) {
                auto &surface = entry.second;
                std::memset(surface.data(), 0, surface.rows() * surface.pitch());
                if (entry.first.starts_with("w"))
                    for (uint32_t r = 0; r < surface.rows(); ++r) row(surface, r)[r % surface.columns()] = std::bit_cast<uint16_t>(_Float16(scale));
            }
            for (int c = 0; c < s.hidden; ++c) for (int r = 0; r < s.rows; ++r)
                row(p.input("x"), c)[r] = std::bit_cast<uint16_t>(_Float16(((r * 3 + c * 7) % 17 - 8) / 8.f));
            if (s.lora_inputs) for (int c = 0; c < s.width; ++c) for (int r = 0; r < s.rows; ++r) {
                row(p.input("dg"), c)[r] = std::bit_cast<uint16_t>(_Float16(scale * ((r + c) % 3 - 1)));
                row(p.input("du"), c)[r] = std::bit_cast<uint16_t>(_Float16(scale * ((r * 2 + c) % 3 - 1)));
            }
            p.predict(result.data(), DType::FP16, hidden.data());
            auto expected = [&](int r, int c, bool h) {
                const int f = h ? c : (s.kind == Kind::Matmul ? c : c % s.width);
                float value = ((r * 3 + (f % s.hidden) * 7) % 17 - 8) / 8.f * scale;
                if (s.kind == Kind::SwiGLU) {
                    const float g = value + (s.lora_inputs ? scale * ((r + f) % 3 - 1) : 0);
                    const float u = value + (s.lora_inputs ? scale * ((r * 2 + f) % 3 - 1) : 0);
                    value = g / (1.f + std::exp(-g)) * u * (h ? 1.f : scale);
                }
                return value;
            };
            for (const bool h : {false, true}) {
                if (h && !s.lora_inputs) continue;
                const int columns = h ? s.width : s.output_width();
                const auto &actual = h ? hidden : result;
                for (int r = 0; r < s.rows; ++r) for (int c = 0; c < columns; ++c) {
                    const float ref = expected(r, c, h), out = float(std::bit_cast<_Float16>(actual[size_t(r) * columns + c]));
                    if (!std::isfinite(out) || std::abs(out - ref) > .0002f + .02f * std::abs(ref))
                        throw CapabilityError("private ANE numerical/A-B-A/LoRA self-test failed: row=" + std::to_string(r) +
                            " col=" + std::to_string(c) + " scale=" + std::to_string(scale) + " hidden=" + std::to_string(h) +
                            " expected=" + std::to_string(ref) + " actual=" + std::to_string(out));
                }
            }
        }
        p.verified = true;
    });
    try { p.worker.join(); error.clear(); return true; }
    catch (const std::exception &failure) { error = failure.what(); return false; }
}
void PrivateGraph::stage_weights(std::vector<WeightView> weights) {
    auto &p = *impl_; p.worker.join(); check(p.verified && !p.disabled, "private ANE self-test required/executor disabled");
    p.staged = false; p.result = {};
    p.worker.submit([&p, weights = std::move(weights)] {
        const auto start = Clock::now();
        try {
            const std::vector<std::string> names = p.shape.kind == Kind::Matmul ? std::vector<std::string>{"w"} : std::vector<std::string>{"wg", "wu", "wd"};
            check(weights.size() == names.size(), "private ANE weight count mismatch");
            for (size_t i = 0; i < names.size(); ++i) fill_weight(p.input(names[i]), weights[i], names[i] == "wu" ? 1.f / p.headroom : 1.f);
            p.staged = p.result.ok = true;
        } catch (const std::exception &failure) { p.result.error = failure.what(); }
        p.result.stage_seconds = elapsed(start);
    });
}
RunResult PrivateGraph::wait_stage() { return finish(); }
void PrivateGraph::launch(MatrixView input, uint16_t *output, size_t elements, DType dtype, std::optional<AdapterInput> adapter) {
    auto &p = *impl_; p.worker.join(); const double stage_time = p.result.stage_seconds;
    p.result = {}; p.result.stage_seconds = stage_time;
    p.worker.submit([&p, input, output, elements, dtype, adapter] {
        const auto start = Clock::now();
        try {
            const auto &s = p.shape;
            check(p.verified && p.staged && !p.disabled, "private ANE no verified staged weights"); validate(input);
            check((dtype == DType::FP16 || dtype == DType::BF16) && output && input.cols == s.hidden && input.rows % s.rows == 0 &&
                elements >= size_t(input.rows) * s.output_width(), "private ANE output/chunk geometry mismatch");
            check(!adapter || s.lora_inputs, "private ANE graph lacks LoRA inputs");
            if (adapter) {
                validate(adapter->gate); validate(adapter->up);
                check(adapter->gate.rows == input.rows && adapter->up.rows == input.rows && adapter->gate.cols == s.width &&
                    adapter->up.cols == s.width && adapter->hidden && adapter->hidden_elements >= size_t(input.rows) * s.width,
                    "private ANE LoRA input/hidden geometry mismatch");
            } else if (s.lora_inputs) for (const auto &name : {"dg", "du"}) {
                auto &surface = p.input(name); std::memset(surface.data(), 0, surface.rows() * surface.pitch());
            }
            for (int r = 0; r < input.rows; r += s.rows) {
                const auto fill_start = Clock::now(); fill_transposed(p.input("x"), input, r);
                if (adapter) fill_transposed(p.input("dg"), adapter->gate, r);
                p.result.input_seconds += elapsed(fill_start);
                for (;;) {
                    if (adapter) {
                        const auto delta_start = Clock::now();
                        fill_transposed(p.input("du"), adapter->up, r, 1.f / p.headroom);
                        p.result.input_seconds += elapsed(delta_start);
                    }
                    try { p.predict(output + size_t(r) * s.output_width(), dtype, adapter ? adapter->hidden + size_t(r) * s.width : nullptr); break; }
                    catch (const OutputOverflow &) {
                        if (s.kind != Kind::SwiGLU || p.headroom >= 4096.f) throw;
                        p.increase_headroom();
                    }
                }
            }
            p.result.ok = true;
        } catch (const std::exception &failure) { p.result.error = failure.what(); }
        p.result.total_seconds = elapsed(start);
        p.result.headroom_scale = p.headroom;
    });
}
RunResult PrivateGraph::finish() {
    auto &p = *impl_;
    try { p.worker.join(); } catch (const std::exception &failure) { p.result.ok = false; p.result.error = failure.what(); }
    return p.result;
}
bool PrivateGraph::supports_device_io() const { return impl_->device_io; }
void PrivateGraph::launch_device(DeviceMatrixView input, DeviceMatrixView output, std::optional<DeviceAdapterInput> adapter) {
    auto &p = *impl_; p.worker.join();
    const double stage_time = p.result.stage_seconds; p.result = {}; p.result.stage_seconds = stage_time;
    p.worker.submit([&p, input = std::move(input), output = std::move(output), adapter = std::move(adapter)] {
        const auto start = Clock::now();
        try {
            const auto &s = p.shape;
            check(p.device_io && p.verified && p.staged && !p.disabled, "private ANE device I/O unavailable/unverified/unstaged");
            check(input.rows > 0 && input.rows % s.rows == 0 && input.cols == s.hidden && output.rows == input.rows &&
                output.cols == s.output_width() && (output.dtype == DType::FP16 || output.dtype == DType::BF16),
                "private ANE device I/O chunk/output geometry mismatch");
            check(!adapter || (s.lora_inputs && adapter->gate.rows == input.rows && adapter->up.rows == input.rows &&
                adapter->hidden.rows == input.rows && adapter->gate.cols == s.width && adapter->up.cols == s.width &&
                adapter->hidden.cols == s.width && adapter->hidden.dtype == output.dtype), "private ANE device LoRA geometry mismatch");
            const auto distinct = [](const DeviceMatrixView &a, const DeviceMatrixView &b) {
                if (a.buffer != b.buffer) return;
                const auto bytes = [](const DeviceMatrixView &v) {
                    const size_t item = v.dtype == DType::FP32 ? 4 : 2;
                    const size_t pitch = v.row_stride_bytes ? v.row_stride_bytes : size_t(v.cols) * item;
                    check(v.rows > 0 && pitch <= SIZE_MAX / size_t(v.rows) && v.offset_bytes <= SIZE_MAX - size_t(v.rows) * pitch,
                          "private ANE device alias extent overflow");
                    return size_t(v.rows) * pitch;
                };
                check(a.offset_bytes + bytes(a) <= b.offset_bytes || b.offset_bytes + bytes(b) <= a.offset_bytes,
                      "private ANE device source/output bindings overlap");
            };
            distinct(input, output);
            if (adapter) {
                distinct(input, adapter->hidden); distinct(output, adapter->hidden);
                for (const auto *delta : {&adapter->gate, &adapter->up}) { distinct(*delta, output); distinct(*delta, adapter->hidden); }
            } else if (s.lora_inputs) for (const auto &name : {"dg", "du"}) {
                auto &slot = p.input(name); std::memset(slot.data(), 0, slot.rows() * slot.pitch());
            }
            for (int r = 0; r < input.rows; r += s.rows) for (;;) {
                check(p.timeline <= UINT64_MAX - 2, "private ANE timeline exhausted");
                const uint64_t ready = p.timeline + 1, done = p.timeline + 2; p.timeline = done;
                std::vector<private_api::Upload> uploads{{input, p.input("x"), r, 1.f}};
                if (adapter) { uploads.push_back({adapter->gate, p.input("dg"), r, 1.f}); uploads.push_back({adapter->up, p.input("du"), r, 1.f / p.headroom}); }
                std::vector<private_api::Download> downloads{{p.output("y"), output, r, output.dtype, p.headroom}};
                if (s.lora_inputs) downloads.push_back({p.output("h"), adapter ? std::optional<DeviceMatrixView>(adapter->hidden) : std::nullopt,
                    r, output.dtype, p.headroom});
                const auto submit_start = Clock::now();
                auto transfer = p.device.prepare_transfer(std::move(uploads), std::move(downloads), ready, done);
                auto ticket = p.program->enqueue(p.inputs, p.outputs, ready, done, transfer.failure_callback());
                try { transfer.submit(); }
                catch (...) { ticket.finish(std::chrono::milliseconds(0)); p.disabled = true; throw; }
                p.result.input_seconds += elapsed(submit_start); // submission span, NOT GPU kernel duration
                const auto predict_start = Clock::now();
                auto evaluated = ticket.finish();
                p.result.prediction_seconds += elapsed(predict_start); ++p.result.calls;
                const auto restore_start = Clock::now(); auto copied = transfer.finish();
                p.result.output_seconds += elapsed(restore_start); // exposed host wait, NOT GPU kernel duration
                if (!evaluated.ok || !copied.ok) { p.disabled = true; throw CapabilityError(!evaluated.ok ? evaluated.error : copied.error); }
                const auto flags = transfer.validation_flags();
                check(!(flags & 1), "private ANE GPU input conversion overflow/nonfinite");
                if (flags & 2) {
                    if (s.kind != Kind::SwiGLU || p.headroom >= 4096.f) throw OutputOverflow("private ANE GPU output exhausted headroom");
                    p.increase_headroom(); continue;
                }
                check(!(flags & 4), "private ANE GPU restored output exceeds destination dtype");
                p.result.copied_output_bytes += size_t(s.rows) * (s.output_width() + (adapter ? s.width : 0)) * 2;
                break;
            }
            p.result.ok = true;
        } catch (const std::exception &failure) { p.result.error = failure.what(); }
        p.result.headroom_scale = p.headroom; p.result.total_seconds = elapsed(start);
    });
}
}
