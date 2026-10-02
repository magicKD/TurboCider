#include "ane_ffn.hpp"
#include "ane_runtime_quant.hpp"
#include "ane_runtime_packed.hpp"

#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <new>

namespace tc::ane {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
MatrixView view(const Tensor &a) {
    require(a.ndim() == 2 && a.flags().row_contiguous, "runtime ANE needs contiguous matrices");
    if (a.dtype() == mx::bfloat16)
        return {a.data<mx::bfloat16_t>(), a.nbytes(), a.shape(0), a.shape(1), 0, DType::BF16};
    if (a.dtype() == mx::float16)
        return {a.data<mx::float16_t>(), a.nbytes(), a.shape(0), a.shape(1), 0, DType::FP16};
    if (a.dtype() == mx::float32)
        return {a.data<float>(), a.nbytes(), a.shape(0), a.shape(1), 0, DType::FP32};
    throw std::runtime_error("runtime ANE supports only BF16/FP16/FP32 staging");
}
WeightView weight_view(const FfnWeight &w) {
    require(w.transform==FfnWeight::Transform::None || w.transform==FfnWeight::Transform::ComfyH256Inverse,
            "runtime ANE unknown FFN transform");
    if (!w.scales) {
        require(w.transform==FfnWeight::Transform::None,"runtime ANE rotation requires explicit packed source");
        require(!w.offsets, "runtime ANE dense weights cannot have affine offsets");
        return view(w.values);
    }
    const auto &a = w.values;
    require(a.ndim() == 2 && a.flags().row_contiguous && a.dtype() == mx::uint32 &&
                (w.bits == 4 || w.bits == 8) && a.shape(1) > 0 && a.shape(1) <= 8192,
            "runtime ANE affine weights require contiguous Q4/Q8 uint32 codes");
    AffineView source{a.data<uint32_t>(), a.nbytes(), a.shape(0), a.shape(1) * (32 / w.bits),
                      0, w.group_size, w.bits, view(*w.scales), std::nullopt};
    if (w.offsets) source.offsets = view(*w.offsets);
    validate_affine_view(source);
    if (w.transform==FfnWeight::Transform::ComfyH256Inverse) {
        ConvrotAffineView rotated{source};validate_convrot_affine_view(rotated);return rotated;
    }
    return source;
}
int configured_chunks() {
    const char *raw = std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS");
    if (!raw || std::string(raw) == "auto") return -1;
    const std::string value(raw);
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
            "TURBOCIDER_RUNTIME_ANE_CHUNKS requires auto or an integer 0...128");
    const auto chunks = std::stoul(value);
    require(chunks <= 128, "runtime ANE chunk count exceeds 128");
    return int(chunks);
}
} // namespace

HybridFfn::HybridFfn(const std::filesystem::path &manifest, int hidden, int width,
                     size_t budget, std::atomic<bool> &cancelled, bool require_lora_inputs)
    : memory_budget_(budget) {
    checkpoint(cancelled);
    const int chunks = configured_chunks();
    const char *profile = std::getenv("TURBOCIDER_RUNTIME_ANE_PROFILE");
    require(!profile || std::string(profile) == "0" || std::string(profile) == "1",
            "TURBOCIDER_RUNTIME_ANE_PROFILE accepts 0 or 1");
    profile_ = profile && std::string(profile) == "1";
    metrics_.weight_variant = "runtime_fp16";
    metrics_.mlp_output_kind = "runtime_weight_swiglu";
    metrics_.hidden = metrics_.output_channels = hidden;
    metrics_.mlp_width = width;
    metrics_.block_count = 32;
    // Explicit invalid artifacts must fail, not masquerade as a successful
    // acceleration request. A valid graph failing its capability self-test
    // is instead a reported GPU fallback on this machine.
    try {
        graph_ = std::make_unique<RuntimeGraph>(manifest, budget, false, false,
                                               GraphGeometry{Kind::SwiGLU, hidden, width, require_lora_inputs});
    }
    catch (const MemoryBudgetError &error) { degrade(error.what(), -1); return; }
    catch (const CapabilityError &error) { degrade(error.what(), -1); return; }
    metrics_.bucket = graph_->shape().rows;
    if (graph_->shape().lora_inputs) metrics_.mlp_output_kind = "runtime_weight_swiglu_lora_inputs";
    metrics_.load_seconds = graph_->load_seconds();
    metrics_.runtime_weight_slot_bytes = graph_->slot_bytes();
    metrics_.runtime_weight_estimated_bytes = graph_->estimated_bytes();
    std::string error;
    const auto start = Clock::now();
    if (!graph_->self_test(error)) degrade("self_test_failed: " + error, -1);
    metrics_.zero_input_warmup_seconds = elapsed(start); // sparse nonzero self-test, not measured FFN
    scheduler_ = std::make_unique<RowScheduler>(graph_->shape().rows, chunks);
    checkpoint(cancelled);
}
HybridFfn::~HybridFfn() { drain(); }
void HybridFfn::drain() {
    if (graph_ && pending_) { graph_->finish(); pending_ = false; }
    weights_.clear();
}
void HybridFfn::release_for_memory(const std::string &reason) {
    // Owner-thread safe point: finish the worker while its borrowed weights
    // and scratch still exist, then release the optional graph and buffers.
    drain();
    graph_.reset();
    scheduler_.reset();
    std::vector<uint16_t>().swap(output_);
    std::vector<uint16_t>().swap(hidden_);
    degrade(reason, -1);
    chunks_ = 0;
}
bool HybridFfn::admit_scratch(int ane_rows, bool adapter) {
    if (!graph_ || ane_rows <= 0) return false;
    const auto existing_output = uint64_t(output_.capacity()) * sizeof(uint16_t);
    const auto existing_hidden = uint64_t(hidden_.capacity()) * sizeof(uint16_t);
    const auto scratch = plan_host_scratch(ane_rows, metrics_.hidden, metrics_.mlp_width,
                                           adapter, existing_output, existing_hidden);
    if (!scratch || graph_->estimated_bytes() > memory_budget_ ||
        existing_output > memory_budget_ - graph_->estimated_bytes() ||
        existing_hidden > memory_budget_ - graph_->estimated_bytes() - existing_output) {
        release_for_memory("runtime ANE host scratch exceeds memory budget");
        return false;
    }
    // The old vector remains live while a larger replacement is allocated.
    // Recheck immediately before allocating; skip the Mach calls when no new
    // host payload is needed (resident pressure is checked per request).
    if (scratch->new_payload_bytes) {
        const auto observed = observe_runtime_memory(mx::get_active_memory());
        const auto decision = admit_memory(observed,
            {uint64_t(4) << 30, memory_budget_},
            graph_->estimated_bytes() + existing_output + existing_hidden,
            scratch->new_payload_bytes);
        if (!decision.allowed()) {
            release_for_memory("runtime ANE host scratch admission denied (" +
                               memory_denial_reason(decision.denial, observed) + ")");
            return false;
        }
    }
    return true;
}
void HybridFfn::begin_request(const std::string &adapter_identity,
                              std::optional<MemoryObservation> observation) {
    drain();
    if (graph_) {
        const auto retained = uint64_t(output_.capacity() + hidden_.capacity()) * sizeof(uint16_t);
        const auto observed = observation ? *observation : observe_runtime_memory(mx::get_active_memory());
        const auto decision = admit_memory(observed,
            {uint64_t(4) << 30, memory_budget_}, graph_->estimated_bytes() + retained, 0);
        if (!decision.allowed())
            release_for_memory("runtime ANE resident memory admission denied (" +
                               memory_denial_reason(decision.denial, observed) + ")");
    }
    if (adapter_identity != adapter_identity_) {
        // A new adapter changes GPU correction costs and may change the best
        // split. Keep the executable, but not another adapter's timing model.
        if (graph_) scheduler_ = std::make_unique<RowScheduler>(graph_->shape().rows, configured_chunks());
        adapter_identity_ = adapter_identity;
    }
    // Retain warm scheduling samples and capability state across resident
    // requests, but all timing/counters below are explicitly session totals.
    layer_ = -1; rows_ = chunks_ = 0;
    planned_ = false;
    block_plan_.reset();
}
RowScheduler::Plan HybridFfn::plan_block(int layer, int rows) {
    require(!planned_ && !block_plan_ && rows > 0, "runtime ANE block already planned or invalid rows");
    drain();
    const auto plan = available() ? scheduler_->plan(layer, rows)
                                 : RowScheduler::Plan{RowScheduler::Mode::Gpu, 0};
    layer_ = layer; rows_ = rows; chunks_ = plan.chunks;
    planned_ = plan.split();
    block_sample_valid_ = plan.mode == RowScheduler::Mode::GpuProbe;
    block_gpu_seconds_ = block_ane_seconds_ = 0;
    if (plan.measured() || plan.split()) block_plan_ = plan;
    else {
        ++metrics_.runtime_weight_gpu_blocks;
        ++metrics_.runtime_weight_unsplit_gpu_blocks;
    }
    return plan;
}
void HybridFfn::observe_block(int layer, int rows, double wall) {
    require(block_plan_ && block_plan_->measured() && !planned_ && !pending_ && layer == layer_ && rows == rows_ &&
                std::isfinite(wall) && wall > 0, "runtime ANE invalid completed block sample");
    if (block_plan_->mode == RowScheduler::Mode::GpuProbe) {
        ++metrics_.runtime_weight_gpu_blocks;
        ++metrics_.runtime_weight_unsplit_gpu_blocks;
        ++metrics_.runtime_weight_full_gpu_probe_blocks;
        metrics_.runtime_weight_full_gpu_probe_seconds += wall;
    }
    // Whole-block samples are supplied by the family AFTER the residual, so
    // full GPU probes and hybrid blocks cover the same arithmetic window.
    if (block_sample_valid_ && scheduler_)
        scheduler_->observe(layer, rows, block_plan_->chunks, wall,
                            block_gpu_seconds_, block_ane_seconds_);
    if (profile_) std::cerr << "{\"runtime_ane_block\":" << layer << ",\"rows\":" << rows
        << ",\"chunks\":" << block_plan_->chunks << ",\"full_gpu_probe\":"
        << (block_plan_->mode == RowScheduler::Mode::GpuProbe ? "true" : "false")
        << ",\"seconds\":" << wall << ",\"sample_valid\":"
        << (block_sample_valid_ ? "true" : "false") << "}\n";
    block_plan_.reset();
}
void HybridFfn::degrade(const std::string &error, int layer) {
    failed_ = metrics_.runtime_failed = true;
    ++metrics_.runtime_failures;
    metrics_.runtime_failure_block = layer;
    reason_ = error;
}
void HybridFfn::stage(int layer, int rows, std::vector<Tensor> weights) {
    std::vector<FfnWeight> sources;
    sources.reserve(weights.size());
    for (auto &weight : weights) sources.push_back({std::move(weight), std::nullopt, std::nullopt});
    stage_weights(layer, rows, std::move(sources));
}
void HybridFfn::stage_weights(int layer, int rows, std::vector<FfnWeight> weights) {
    require(!block_plan_ || (planned_ && block_plan_->split()),
            "runtime ANE cannot stage a full GPU probe or restage a measured block");
    require(!planned_ || (layer == layer_ && rows == rows_), "runtime ANE plan/stage mismatch");
    pre_start_ = Clock::now();
    drain();
    layer_ = layer; rows_ = rows;
    if (!planned_) chunks_ = available() ? scheduler_->select(layer, rows) : 0;
    planned_ = false;
    if (!chunks_) return;
    if (!admit_scratch(chunks_ * graph_->shape().rows, false)) return;
    try {
        uint64_t metadata_growth=0;
        for (auto &weight : weights) if (weight.transform==FfnWeight::Transform::ComfyH256Inverse) {
            for (auto *part : {&weight.scales,&weight.offsets}) if (*part && !(**part).flags().row_contiguous) {
                const uint64_t bytes=(**part).nbytes();
                require(bytes<=UINT64_MAX-metadata_growth,"runtime ConvRot metadata extent overflow");
                metadata_growth+=bytes;
            }
        }
        if (metadata_growth) {
            const auto observed=observe_runtime_memory(mx::get_active_memory());
            const uint64_t retained=uint64_t(output_.capacity()+hidden_.capacity())*sizeof(uint16_t);
            const auto decision=admit_memory(observed,{uint64_t(4)<<30,memory_budget_},
                graph_->estimated_bytes()+retained,metadata_growth);
            if (!decision.allowed()) {
                release_for_memory("runtime ConvRot metadata scratch admission denied");return;
            }
            // Only repeated scale/offset metadata, NOT codes or a dense W
            // matrix. Broadcast MLX views cannot be advertised as contiguous.
            for (auto &weight : weights) if (weight.transform==FfnWeight::Transform::ComfyH256Inverse)
                for (auto *part : {&weight.scales,&weight.offsets})
                    if (*part && !(**part).flags().row_contiguous) *part=mx::contiguous(**part);
        }
        weights_.reserve(weights.size() * 3);
        for (const auto &weight : weights) {
            weights_.push_back(weight.values);
            if (weight.scales) weights_.push_back(*weight.scales);
            if (weight.offsets) weights_.push_back(*weight.offsets);
        }
        // The models materialize their checkpoint before denoising. eval only
        // resolves shallow gate/up slices here, never creates a second model.
        mx::eval(weights_);
        std::vector<WeightView> sources;
        for (const auto &w : weights) sources.push_back(weight_view(w));
        const bool rotated=std::any_of(weights.begin(),weights.end(),[](const auto &w) {
            return w.transform==FfnWeight::Transform::ComfyH256Inverse;
        });
        if (rotated) {
            require(std::all_of(weights.begin(),weights.end(),[](const auto &w) {
                return w.transform==FfnWeight::Transform::ComfyH256Inverse;
            }),"runtime ANE mixed ConvRot FFN recipes unsupported");
            metrics_.runtime_weight_source_recipe="convrot-legacy-packed-scale-inverse-h256-f16-v1";
            ++metrics_.runtime_weight_convrot_stage_submissions;
        }
        graph_->stage_weights(std::move(sources));
        pending_ = true;
    } catch (const std::exception &error) {
        degrade(error.what(), layer); chunks_ = 0;
    }
}
Tensor HybridFfn::run(int layer, const Tensor &input, const Gpu &gpu,
                      std::atomic<bool> &cancelled, const Adapter *adapter) {
    require(input.ndim() == 3 && input.shape(0) == 1 && layer == layer_ &&
                input.shape(1) == rows_ && input.shape(2) == metrics_.hidden,
            "runtime ANE FFN stage/run mismatch");
    require(!adapter || (adapter->gate_up && adapter->down_and_add &&
                (!graph_ || graph_->shape().lora_inputs)),
            "runtime ANE LoRA requires activation-input graph and complete callbacks");
    // An untimed early plan still owns stage/run sequencing, but cannot feed
    // an in-FFN sample into a controller calibrated on whole-block timing.
    // Release it on every return/failure, including GPU error recomputation.
    struct UntimedPlanGuard {
        std::optional<RowScheduler::Plan> &plan;
        ~UntimedPlanGuard() { if (plan && !plan->measured()) plan.reset(); }
    } untimed_plan_guard{block_plan_};
    // Stage while the GPU computes attention and, for a real adapter, its
    // activation corrections. One readiness fence suffices before borrowing
    // any host pointers; do not first drain attention and then submit LoRA.
    auto packed = mx::contiguous(mx::reshape(input, {rows_, input.shape(2)}));
    std::optional<Tensor> delta_gate, delta_up;
    double lora_input_ready_seconds = 0;
    if (chunks_ && available() && !admit_scratch(chunks_ * graph_->shape().rows, adapter != nullptr))
        chunks_ = 0;
    if (adapter && chunks_ && available()) {
        checkpoint(cancelled);
        const auto ready_start = Clock::now();
        const int ane_rows = chunks_ * graph_->shape().rows;
        auto deltas = adapter->gate_up(slice_axis(input, 1, rows_ - ane_rows, rows_));
        const mx::Shape expected{1, ane_rows, metrics_.mlp_width};
        require(deltas.first.shape() == expected && deltas.second.shape() == expected,
                "runtime ANE LoRA callback correction shape mismatch");
        delta_gate = mx::contiguous(mx::reshape(deltas.first, {ane_rows, metrics_.mlp_width}));
        delta_up = mx::contiguous(mx::reshape(deltas.second, {ane_rows, metrics_.mlp_width}));
        mx::eval({packed, *delta_gate, *delta_up});
        lora_input_ready_seconds = elapsed(ready_start);
    } else mx::eval(packed);
    checkpoint(cancelled);
    const double pre_seconds = elapsed(pre_start_);
    metrics_.runtime_weight_pre_seconds += pre_seconds;
    // This window includes upstream attention, not just LoRA kernels. Keep
    // it separate from the legacy post-input gate/up timer and inside pre.
    metrics_.runtime_weight_lora_input_ready_seconds += lora_input_ready_seconds;
    const auto start = Clock::now();
    if (!chunks_ || !available()) {
        auto output = gpu(input);
        mx::eval(output);
        if (block_plan_) block_sample_valid_ = available();
        else if (scheduler_) scheduler_->observe(layer, rows_, 0, pre_seconds + elapsed(start));
        ++metrics_.runtime_weight_gpu_blocks;
        return output;
    }
    const auto wait_start = Clock::now();
    auto staged = graph_->wait_stage();
    pending_ = false;
    const double exposed_stage = elapsed(wait_start);
    metrics_.runtime_weight_stage_seconds += staged.stage_seconds;
    metrics_.runtime_weight_stage_wait_seconds += exposed_stage;
    if (!staged.ok) {
        degrade(staged.error, layer);
        ++metrics_.runtime_weight_fallback_blocks;
        auto output = gpu(input); mx::eval(output);
        ++metrics_.runtime_weight_gpu_blocks;
        return output;
    }
    const int ane_rows = chunks_ * graph_->shape().rows, gpu_rows = rows_ - ane_rows;
    auto tail_input = view(packed);
    tail_input.data = static_cast<const char *>(tail_input.data) + size_t(gpu_rows) * packed.shape(1) * packed.itemsize();
    tail_input.bytes -= size_t(gpu_rows) * packed.shape(1) * packed.itemsize();
    tail_input.rows = ane_rows;
    // GGUF floating metadata can promote the baseline residual to FP32.
    // Recover scaled outputs through BF16 in that case, preserving exponent
    // range before casting back, rather than overflowing a FP16 tail buffer.
    const auto dtype = input.dtype() == mx::float16 ? DType::FP16 : DType::BF16;
    // vector::resize may grow geometrically: allocate an exact-sized
    // replacement so the preflight covers coexistence with the old payload.
    // An OS allocation failure is an optional-tier failure, not a reason to
    // return an incomplete block or terminate the generation.
    auto resize_scratch = [](std::vector<uint16_t> &buffer, size_t elements) {
        if (elements <= buffer.capacity()) {
            buffer.resize(elements);
        } else {
            std::vector<uint16_t> replacement(elements);
            if (replacement.capacity() != elements)
                throw MemoryBudgetError("runtime ANE host allocation exceeded admitted size");
            buffer.swap(replacement);
        }
    };
    std::optional<AdapterInput> adapter_input;
    try {
        resize_scratch(output_, size_t(ane_rows) * input.shape(2));
        if (adapter) {
            resize_scratch(hidden_, size_t(ane_rows) * metrics_.mlp_width);
            adapter_input = AdapterInput{view(*delta_gate), view(*delta_up),
                                         hidden_.data(), hidden_.size()};
        }
    } catch (const std::bad_alloc &) {
        release_for_memory("runtime ANE host scratch allocation failed");
    } catch (const MemoryBudgetError &error) {
        release_for_memory(error.what());
    }
    if (!available()) {
        ++metrics_.runtime_weight_fallback_blocks;
        block_sample_valid_ = false;
        auto result = gpu(input); mx::eval(result);
        ++metrics_.runtime_weight_gpu_blocks;
        return result;
    }
    graph_->launch(tail_input, output_.data(), output_.size(), dtype, adapter_input);
    pending_ = true;
    // Steady auto plans do not feed branch timings to the scheduler. Submit
    // their GPU head without a host wait, so ANE output ownership/corrections
    // can be prepared while that head is still running. Sampling and profile
    // paths retain the synchronous head boundary and its timing semantics.
    const bool async_head = !profile_ && block_plan_ &&
        block_plan_->mode == RowScheduler::Mode::HybridUntimed;
    std::optional<Tensor> head;
    // Always join before exceptions/cancellation can release borrowed input.
    try {
        const auto gpu_start = Clock::now();
        head = gpu(slice_axis(input, 1, 0, gpu_rows));
        if (async_head) mx::async_eval(*head);
        else mx::eval(*head);
        const double gpu_seconds = async_head ? 0 : elapsed(gpu_start);
        const auto join_start = Clock::now();
        auto result = graph_->finish();
        pending_ = false;
        const double join = elapsed(join_start);
        checkpoint(cancelled);
        metrics_.calls += result.calls; metrics_.runtime_calls += result.calls;
        metrics_.prediction_seconds += result.prediction_seconds;
        metrics_.model_prediction_seconds += result.prediction_seconds;
        metrics_.output_handling_seconds += result.output_seconds;
        metrics_.copied_bytes += result.copied_output_bytes;
        // An async join overlaps GPU work: it is NOT exposed ANE time and
        // must never enter the old post-GPU join counter or rate estimator.
        if (!async_head) metrics_.runtime_weight_join_seconds += join;
        metrics_.runtime_weight_gpu_seconds += gpu_seconds;
        metrics_.runtime_weight_overflow_retries += result.overflow_retries;
        metrics_.runtime_weight_headroom = result.headroom_scale;
        const auto post_join_start = Clock::now();
        Tensor tail = [&] {
            if (!result.ok) {
                degrade(result.error, layer);
                ++metrics_.runtime_weight_fallback_blocks;
                // Recompute ALL requested tail rows, even when earlier chunks
                // succeeded. Never consume partially written scratch buffers.
                return gpu(slice_axis(input, 1, gpu_rows, rows_));
            }
            // The typed-pointer constructors copy, unlike a borrowed-buffer
            // view. Retained callbacks/lazy consumers must survive the next
            // prediction and destruction of this runtime's scratch storage.
            auto copy_output = [&](const std::vector<uint16_t> &data, int width) {
                const mx::Shape shape{1, ane_rows, width};
                auto owned = dtype == DType::FP16
                    ? Tensor(reinterpret_cast<const mx::float16_t *>(data.data()), shape, mx::float16)
                    : Tensor(reinterpret_cast<const mx::bfloat16_t *>(data.data()), shape, mx::bfloat16);
                return mx::astype(owned, input.dtype());
            };
            auto base_down = copy_output(output_, input.shape(2));
            if (!adapter) return base_down;
            auto corrected = adapter->down_and_add(copy_output(hidden_, metrics_.mlp_width), base_down);
            require(corrected.shape() == base_down.shape() && corrected.dtype() == base_down.dtype(),
                    "runtime ANE LoRA down/add callback shape or dtype mismatch");
            return corrected;
        }();
        auto output = mx::concatenate({*head, tail}, 1);
        mx::eval(output); // consumer owns storage before the next graph launch
        // Reuse the required output-ownership fence: measuring these exposed
        // spans must not add another eval or serialize the parallel branches.
        const double post_join_seconds = elapsed(post_join_start);
        metrics_.runtime_weight_post_join_seconds += post_join_seconds;
        const double wall = elapsed(start);
        // Include attention/staging contention in the on/off decision, not
        // only the FFN window. The chunk balancer still uses FFN branch costs.
        // Headroom discovery may re-run the same chunk several times. It is
        // a one-off correctness recovery, not steady ANE throughput.
        if (block_plan_) {
            block_sample_valid_ = result.ok && result.overflow_retries == 0;
            block_gpu_seconds_ = gpu_seconds;
            block_ane_seconds_ = result.total_seconds;
        } else if (result.ok && result.overflow_retries == 0)
            scheduler_->observe(layer, rows_, chunks_, pre_seconds + wall, gpu_seconds, result.total_seconds);
        ++metrics_.runtime_weight_hybrid_blocks;
        if (async_head) {
            ++metrics_.runtime_weight_async_hybrid_blocks;
            metrics_.runtime_weight_async_ane_wait_seconds += join;
        }
        if (block_plan_ && block_plan_->mode == RowScheduler::Mode::HybridUntimed)
            ++metrics_.runtime_weight_untimed_hybrid_blocks;
        metrics_.runtime_weight_ane_rows += ane_rows;
        metrics_.runtime_weight_wall_seconds += wall;
        if (profile_) std::cerr << "{\"runtime_ane_layer\":" << layer << ",\"rows\":" << rows_
            << ",\"ane_rows\":" << ane_rows << ",\"stage_seconds\":" << staged.stage_seconds
            << ",\"stage_wait_seconds\":" << exposed_stage << ",\"gpu_seconds\":" << gpu_seconds
            << ",\"pre_ffn_seconds\":" << pre_seconds
            << ",\"ane_seconds\":" << result.total_seconds << ",\"join_seconds\":" << join
            << ",\"lora_gate_up_seconds\":0"
            << ",\"lora_input_ready_seconds\":" << lora_input_ready_seconds
            << ",\"post_join_seconds\":" << post_join_seconds
            << ",\"ffn_seconds\":" << wall << ",\"ok\":" << (result.ok ? "true" : "false") << "}\n";
        return output;
    } catch (...) {
        graph_->finish(); pending_ = false;
        // Cancellation/down-callback errors may precede the final ownership
        // fence. Drain the submitted GPU head as well, preserving the original
        // exception if the GPU reports an error during cleanup.
        if (async_head && head) {
            try { mx::eval(*head); } catch (...) {}
        }
        throw;
    }
}
HybridMetrics HybridFfn::metrics() const {
    auto metrics = metrics_;
    metrics.prefill_plan_reason = reason_;
    return metrics;
}

} // namespace tc::ane
