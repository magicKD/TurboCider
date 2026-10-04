#include "ane_qkv.hpp"

#include <cstdlib>
#include <new>

namespace tc::ane {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
int configured_chunks() {
    const char *raw = std::getenv("TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS");
    if (!raw) return 1; // fixed and conservative until whole-request calibration
    std::string value(raw);
    if (value == "auto") return -1;
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
            "TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS requires auto or an integer 0...128");
    const auto count = std::stoul(value);
    require(count <= 128, "QKV runtime chunk count exceeds 128");
    return int(count);
}
MatrixView view(const Tensor &a) {
    require(a.ndim() == 2 && a.flags().row_contiguous,
            "QKV runtime needs contiguous rank-2 matrices");
    if (a.dtype() == mx::bfloat16)
        return {a.data<mx::bfloat16_t>(), a.nbytes(), a.shape(0), a.shape(1), 0, DType::BF16};
    if (a.dtype() == mx::float16)
        return {a.data<mx::float16_t>(), a.nbytes(), a.shape(0), a.shape(1), 0, DType::FP16};
    throw std::runtime_error("QKV runtime supports only BF16/FP16 source weights and input");
}
} // namespace

HybridQkv::HybridQkv(const std::filesystem::path &manifest, size_t budget,
                     std::atomic<bool> &cancelled) : budget_(budget) {
    checkpoint(cancelled);
    const int configured = configured_chunks();
    metrics_.chunks = configured < 0 ? 1 : configured;
    metrics_.auto_scheduling = configured < 0;
    if (configured < 0) scheduler_ = std::make_unique<QkvScheduler>();
    try {
        graph_ = std::make_unique<RuntimeGraph>(manifest, budget, false, false,
                                                 GraphGeometry{Kind::Matmul, 4096, 12288});
    } catch (const MemoryBudgetError &e) { fail(e.what(), -1); return; }
      catch (const CapabilityError &e) { fail(e.what(), -1); return; }
    metrics_.chunk_rows = graph_->shape().rows;
    metrics_.slot_bytes = graph_->slot_bytes();
    metrics_.estimated_bytes = graph_->estimated_bytes();
    metrics_.load_seconds = graph_->load_seconds();
    std::string error;
    if (!graph_->self_test(error)) fail("self_test_failed: " + error, -1);
    checkpoint(cancelled);
}
HybridQkv::~HybridQkv() { drain(); }
void HybridQkv::drain() {
    if (graph_ && pending_) { graph_->finish(); pending_ = false; }
    weights_.clear();
}
void HybridQkv::fail(std::string reason, int layer) {
    metrics_.failed = true;
    ++metrics_.failures;
    metrics_.failure_block = layer;
    metrics_.failure_reason = std::move(reason);
    // The failed route stays disabled. Drain by destroying the graph before
    // releasing tensors/scratch borrowed by its worker. This also releases
    // the private compiled snapshot during GPU fallback, not at session end.
    graph_.reset();
    pending_ = false;
    weights_.clear();
    scheduler_.reset();
    std::vector<uint16_t>().swap(output_);
    chunks_ = 0;
}
void HybridQkv::begin_request() {
    drain();
    layer_ = -1; rows_ = chunks_ = 0;
    block_plan_.reset();
    if (!available()) return;
    const auto resident = graph_->estimated_bytes() + output_.capacity() * sizeof(uint16_t);
    const auto observed = observe_runtime_memory(mx::get_active_memory());
    const auto decision = admit_memory(observed, {uint64_t(4) << 30, budget_}, resident, 0);
    if (!decision.allowed()) {
        fail("QKV resident admission denied (" + memory_denial_reason(decision.denial, observed) + ")", -1);
    }
}
QkvScheduler::Plan HybridQkv::plan_block(int layer, int rows) {
    require(!block_plan_ && layer >= 0 && rows > 0, "QKV runtime block already planned or invalid rows");
    const bool can_split = available() && metrics_.chunks > 0 && rows > graph_->shape().rows;
    auto plan = !can_split ? QkvScheduler::Plan{QkvScheduler::Mode::Gpu} :
                scheduler_ ? scheduler_->plan(layer, rows) :
                             QkvScheduler::Plan{QkvScheduler::Mode::HybridUntimed};
    if (plan.measured()) block_plan_ = plan;
    if (!plan.hybrid()) ++metrics_.gpu_blocks;
    return plan;
}
void HybridQkv::observe_block(int layer, int rows, double wall) {
    require(block_plan_ && layer >= 0 && rows > 0 && std::isfinite(wall) && wall > 0,
            "QKV runtime invalid completed block sample");
    if (scheduler_ && available()) scheduler_->observe(layer, rows, *block_plan_, wall);
    if (block_plan_->mode == QkvScheduler::Mode::GpuProbe) {
        ++metrics_.gpu_probe_blocks;
        metrics_.gpu_probe_seconds += wall;
    } else {
        ++metrics_.measured_hybrid_blocks;
        metrics_.measured_hybrid_seconds += wall;
    }
    block_plan_.reset();
}
bool HybridQkv::admit_output(int ane_rows) {
    const auto current = uint64_t(output_.capacity()) * sizeof(uint16_t);
    const auto required = uint64_t(ane_rows) * 12288 * sizeof(uint16_t);
    if (graph_->estimated_bytes() > budget_ || current > budget_ - graph_->estimated_bytes() ||
        required > budget_ - graph_->estimated_bytes()) {
        fail("QKV output exceeds optional memory budget", layer_);
        return false;
    }
    if (required > current) {
        const auto observed = observe_runtime_memory(mx::get_active_memory());
        const auto decision = admit_memory(observed, {uint64_t(4) << 30, budget_},
                                           graph_->estimated_bytes() + current, required);
        if (!decision.allowed()) {
            fail("QKV output admission denied (" + memory_denial_reason(decision.denial, observed) + ")", layer_);
            return false;
        }
        try {
            std::vector<uint16_t> replacement(required / sizeof(uint16_t));
            if (replacement.capacity() != required / sizeof(uint16_t))
                throw MemoryBudgetError("QKV output allocation exceeded admitted size");
            output_.swap(replacement);
        } catch (const std::bad_alloc &) {
            fail("QKV output allocation failed", layer_);
            return false;
        } catch (const MemoryBudgetError &e) {
            fail(e.what(), layer_);
            return false;
        }
    }
    output_.resize(required / sizeof(uint16_t));
    return true;
}
void HybridQkv::stage(int layer, int rows, std::vector<Tensor> parts) {
    drain();
    require(layer >= 0 && rows > 0 && parts.size() == 3,
            "QKV runtime stage requires a layer, rows and three matrices");
    layer_ = layer; rows_ = rows;
    chunks_ = available() ? std::min(metrics_.chunks, (rows - 1) / graph_->shape().rows) : 0;
    if (!chunks_) return;
    if (!admit_output(chunks_ * graph_->shape().rows)) { chunks_ = 0; return; }
    try {
        weights_ = std::move(parts);
        mx::eval(weights_);
        std::vector<MatrixView> sources;
        for (const auto &weight : weights_) {
            require(weight.shape() == mx::Shape{4096, 4096},
                    "QKV runtime requires three [4096,4096] weights");
            sources.push_back(view(weight));
        }
        graph_->stage_matmul_parts(std::move(sources));
        pending_ = true;
    } catch (const std::exception &e) {
        fail(e.what(), layer);
        chunks_ = 0;
    }
}
Tensor HybridQkv::run(int layer, const Tensor &input, const Gpu &gpu,
                      std::atomic<bool> &cancelled) {
    require(layer == layer_ && input.ndim() == 3 && input.shape() == mx::Shape{1, rows_, 4096} &&
                (input.dtype() == mx::bfloat16 || input.dtype() == mx::float16),
            "QKV runtime stage/run shape mismatch");
    auto full_gpu = [&] {
        auto result = gpu(layer, input);
        mx::eval(result);
        ++metrics_.gpu_blocks;
        return result;
    };
    const auto start = Clock::now();
    if (!chunks_ || !available()) return full_gpu();
    auto packed = mx::contiguous(mx::reshape(input, {rows_, 4096}));
    mx::eval(packed);
    checkpoint(cancelled);
    const auto wait_start = Clock::now();
    auto staged = graph_->wait_stage();
    pending_ = false;
    metrics_.stage_seconds += staged.stage_seconds;
    metrics_.stage_wait_seconds += seconds(wait_start);
    if (!staged.ok) {
        fail(staged.error, layer);
        ++metrics_.fallback_blocks;
        return full_gpu();
    }
    const int ane_rows = chunks_ * graph_->shape().rows;
    const int gpu_rows = rows_ - ane_rows;
    auto source = view(packed);
    source.data = static_cast<const char *>(source.data) + size_t(gpu_rows) * 4096 * packed.itemsize();
    source.bytes -= size_t(gpu_rows) * 4096 * packed.itemsize();
    source.rows = ane_rows;
    graph_->launch(source, output_.data(), output_.size(),
                   input.dtype() == mx::float16 ? DType::FP16 : DType::BF16);
    pending_ = true;
    std::optional<Tensor> head;
    try {
        // The GPU head is submitted while the Core ML worker consumes its
        // independent tail. Every exit drains both branches before reuse.
        head = gpu(layer, slice_axis(input, 1, 0, gpu_rows));
        mx::async_eval(*head);
        auto done = graph_->finish(); pending_ = false;
        metrics_.calls += done.calls;
        metrics_.prediction_seconds += done.prediction_seconds;
        metrics_.output_seconds += done.output_seconds;
        checkpoint(cancelled);
        Tensor tail = [&] {
            if (!done.ok) {
                fail(done.error, layer);
                ++metrics_.fallback_blocks;
                // Never use partly written chunk results after a failure.
                return gpu(layer, slice_axis(input, 1, gpu_rows, rows_));
            }
            const mx::Shape shape{1, ane_rows, 12288};
            return input.dtype() == mx::float16
                ? Tensor(reinterpret_cast<const mx::float16_t *>(output_.data()), shape, mx::float16)
                : Tensor(reinterpret_cast<const mx::bfloat16_t *>(output_.data()), shape, mx::bfloat16);
        }();
        auto result = mx::concatenate({*head, tail}, 1);
        mx::eval(result); // owns completed output before the next prediction
        ++metrics_.hybrid_blocks;
        metrics_.ane_rows += ane_rows;
        metrics_.wall_seconds += seconds(start);
        return result;
    } catch (...) {
        if (graph_ && pending_) { graph_->finish(); pending_ = false; }
        if (head) { try { mx::eval(*head); } catch (...) {} }
        throw;
    }
}

} // namespace tc::ane
