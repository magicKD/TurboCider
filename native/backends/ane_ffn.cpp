#include "ane_ffn.hpp"
#include "ane_backend.hpp"
#include "ane_runtime_quant.hpp"
#include "ane_runtime_packed.hpp"
#include "../platform/apple/platform.hpp"

#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <new>
#include <sys/stat.h>
#include <mlx/allocator.h>

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
DeviceMatrixView device_view(const Tensor &a, int rows, int cols, int begin_row = 0) {
    require(a.flags().row_contiguous && rows > 0 && cols > 0 && begin_row >= 0 &&
        size_t(rows + begin_row) * cols <= a.size(), "runtime ANE device matrix geometry mismatch");
    const auto dtype = a.dtype() == mx::float16 ? DType::FP16 : a.dtype() == mx::bfloat16 ? DType::BF16 : DType::FP32;
    require(a.dtype() == mx::float16 || a.dtype() == mx::bfloat16 || a.dtype() == mx::float32,
            "runtime ANE device dtype unsupported");
    require(a.buffer().ptr() != nullptr && a.offset() >= 0, "runtime ANE device buffer is not allocated/ready");
    const size_t pitch = size_t(cols) * a.itemsize();
    return {const_cast<void *>(a.buffer().ptr()), a.buffer_size(), size_t(a.offset()) + size_t(begin_row) * pitch,
            rows, cols, pitch, dtype, std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
Tensor device_output(int rows, int cols, DType dtype) {
    const size_t bytes = size_t(rows) * cols * 2;
    auto buffer = mx::allocator::malloc(bytes);
    if (!buffer.ptr()) throw std::bad_alloc();
    return Tensor(buffer, {1, rows, cols}, dtype == DType::FP16 ? mx::float16 : mx::bfloat16);
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
DeviceWeightView device_weight_view(const FfnWeight &w) {
    require(w.transform == FfnWeight::Transform::None, "W8 source transform requires separate qualification");
    const auto &a=w.values;
    require(a.ndim()==2 && a.flags().row_contiguous && a.offset()>=0 && a.buffer().ptr(), "W8 weight requires produced contiguous buffer");
    DeviceWeightView source;
    source.buffer=const_cast<void*>(a.buffer().ptr());source.buffer_bytes=a.buffer_size();source.offset_bytes=size_t(a.offset());
    source.rows=a.shape(0);source.owner=std::make_shared<Tensor>(a);source.group_size=w.group_size;
    source.allocation_identity=a.data_shared_ptr();source.immutable_generation=true;
    if (!w.scales) {
        const auto view=device_view(a,a.shape(0),a.shape(1));source.cols=view.cols;source.row_stride_bytes=view.row_stride_bytes;source.dense_dtype=view.dtype;
        require(!w.offsets,"W8 dense source has affine offsets");
    } else {
        require(a.dtype()==mx::uint32 && (w.bits==4||w.bits==8),"W8 affine source requires Q4/Q8 words");
        source.cols=a.shape(1)*(32/w.bits);source.row_stride_bytes=size_t(a.shape(1))*4;
        source.encoding=w.bits==4?DeviceWeightEncoding::AffineQ4:DeviceWeightEncoding::AffineQ8;
        source.scales=device_view(*w.scales,w.scales->shape(0),w.scales->shape(1));
        if (w.offsets) source.offsets=device_view(*w.offsets,w.offsets->shape(0),w.offsets->shape(1));
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
RowSchedulerCache &scheduler_cache() { static RowSchedulerCache cache; return cache; }
std::string identity_part(const std::string &value) { return ":" + std::to_string(value.size()) + ":" + value; }
} // namespace

HybridFfn::HybridFfn(const std::filesystem::path &manifest, int hidden, int width,
                     size_t budget, std::atomic<bool> &cancelled, bool require_lora_inputs,
                     std::optional<PreparationResult<RuntimeGraph::Prepared>> prepared,
                     std::string scheduler_identity)
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
    bool verified = false;
    double verified_seconds = 0;
    try {
        if (prepared) {
            require(public_runtime_preparation_allowed(configured_backend()) &&
                        private_channel_count(width) == 0,
                    "prepared Core ML graph requires the public backend without channel splitting");
            metrics_.runtime_weight_prepared_early = true;
            metrics_.runtime_weight_prepare_seconds = prepared->seconds;
            metrics_.runtime_weight_prepare_wait_seconds = prepared->wait_seconds;
            metrics_.runtime_weight_prepare_before_join_seconds = prepared->before_join_seconds;
            // Do not retry a failed load or turn an invalid graph into fallback.
            if (prepared->error) std::rethrow_exception(prepared->error);
            require(bool(prepared->value), "runtime ANE preparation returned no graph");
            const auto &shape = prepared->value->shape();
            require(shape.kind == Kind::SwiGLU && shape.hidden == hidden && shape.width == width &&
                        (!require_lora_inputs || shape.lora_inputs),
                    "runtime ANE prepared graph does not match model FFN geometry");
            graph_ = std::make_unique<RuntimeGraph>(std::move(prepared->value), budget);
        } else {
            auto built = build_runtime_executor(manifest, budget,
                GraphGeometry{Kind::SwiGLU, hidden, width, require_lora_inputs}, configured_backend());
            graph_ = std::move(built.executor); verified = built.self_test_passed;
            verified_seconds = built.self_test_seconds;
            metrics_.runtime_weight_backend_fallback_reason = std::move(built.fallback_reason);
            if (!graph_) { failed_ = true; reason_ = metrics_.runtime_weight_backend_fallback_reason; return; }
        }
    }
    catch (const MemoryBudgetError &error) { degrade(error.what(), -1); return; }
    catch (const CapabilityError &error) { degrade(error.what(), -1); return; }
    metrics_.runtime_weight_backend = graph_->backend() == BackendKind::PrivateANE ? "private_ane" : "public_coreml";
    metrics_.runtime_weight_io_path = graph_->supports_device_io() ? "gpu_iosurface" : "host";
    metrics_.runtime_weight_data_path = graph_->data_path();
    metrics_.runtime_weight_source_recipe = graph_->weight_recipe();
    metrics_.runtime_weight_launch_fence_enabled=graph_->device_submission_fence_enabled();
    metrics_.runtime_weight_a8_lookahead_enabled=graph_->activation_lookahead_enabled();
    if (graph_->shape().width != width) {
        require(graph_->shape().width > 0 && graph_->shape().width < width && graph_->supports_device_weight_regions() &&
                graph_->supports_device_io(), "runtime channel split requires a smaller device-source graph");
        axis_ = PartitionAxis::IntermediateChannels;
        metrics_.runtime_weight_partition_axis = "intermediate_channels";
        metrics_.runtime_weight_ane_channels = graph_->shape().width;
        metrics_.runtime_weight_gpu_channels = width - graph_->shape().width;
    }
    if (metrics_.runtime_weight_data_path == "w8a8_hadamard") {
        metrics_.weight_variant = "runtime_w8a8";
    }
    metrics_.bucket = graph_->shape().rows;
    if (graph_->shape().lora_inputs) metrics_.mlp_output_kind = "runtime_weight_swiglu_lora_inputs";
    metrics_.load_seconds = graph_->load_seconds();
    metrics_.manifest_validation_seconds = graph_->artifact_seconds();
    metrics_.model_load_seconds = graph_->model_load_seconds();
    metrics_.output_backing_setup_seconds = graph_->bind_seconds();
    metrics_.runtime_weight_slot_bytes = graph_->slot_bytes();
    metrics_.runtime_weight_estimated_bytes = graph_->estimated_bytes();
    std::string error;
    const auto start = Clock::now();
    const bool self_test_passed = verified || graph_->self_test(error);
    metrics_.zero_input_warmup_seconds = verified ? verified_seconds : elapsed(start);
    if (!self_test_passed) {
        degrade("self_test_failed: " + error, -1);
        checkpoint(cancelled);
        return;
    }
    scheduler_ = std::make_unique<RowScheduler>(graph_->shape().rows, chunks, axis_, profile_);
    if (!scheduler_identity.empty() && !profile_ && chunks == -1) {
        const auto &shape = graph_->shape();
        scheduler_identity_ = "runtime-scheduler-v2" + identity_part(scheduler_identity) +
            identity_part(std::filesystem::canonical(manifest).string()) + identity_part(sha256_file(manifest)) +
            identity_part(executor_configuration_identity()) + identity_part(metrics_.runtime_weight_backend) +
            identity_part(metrics_.runtime_weight_data_path) + identity_part(metrics_.runtime_weight_partition_axis) +
            ":" + std::to_string(shape.rows) + ":" + std::to_string(hidden) + ":" + std::to_string(width) +
            ":" + std::to_string(shape.width) + ":" + std::to_string(shape.lora_inputs);
    }
    const char *prefetch = std::getenv("TURBOCIDER_PRIVATE_ANE_PREFETCH");
    require(!prefetch || std::string(prefetch)=="0" || std::string(prefetch)=="1", "private ANE prefetch requires 0 or 1");
    // Matched M4 Max channel pilots show real future-bank hits but a net
    // bandwidth tax. Keep implemented prefetch explicit until calibration
    // measures a profitable policy; never silently lose the row/channel win.
    prefetch_ = graph_->supports_weight_prefetch() && prefetch && std::string(prefetch)=="1";
    metrics_.runtime_weight_prefetch_enabled = prefetch_;
    checkpoint(cancelled);
}
HybridFfn::~HybridFfn() { drain(); save_scheduler(); }
void HybridFfn::save_scheduler() {
    if (available() && scheduler_ && !scheduler_cache_key_.empty()) {
        try { scheduler_cache().save(scheduler_cache_key_, *scheduler_); }
        catch (const std::bad_alloc &) { /* Optional metadata must not prevent resource retirement. */ }
    }
}
std::string HybridFfn::scheduler_source_identity(const std::filesystem::path &source) {
    const auto path = std::filesystem::canonical(source);
    struct stat info{};
    require(::stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode), "runtime scheduler source is not a regular file");
    std::string value = identity_part(path.string()) + ":" + std::to_string(info.st_dev) + ":" +
        std::to_string(info.st_ino) + ":" + std::to_string(info.st_size);
#if defined(__APPLE__)
    value += ":" + std::to_string(info.st_mtimespec.tv_sec) + ":" + std::to_string(info.st_mtimespec.tv_nsec) +
             ":" + std::to_string(info.st_ctimespec.tv_sec) + ":" + std::to_string(info.st_ctimespec.tv_nsec);
#else
    value += ":" + std::to_string(info.st_mtim.tv_sec) + ":" + std::to_string(info.st_mtim.tv_nsec) +
             ":" + std::to_string(info.st_ctim.tv_sec) + ":" + std::to_string(info.st_ctim.tv_nsec);
#endif
    return value;
}
std::string HybridFfn::executor_configuration_identity() {
    std::string identity;
    for (const char *key : {"TURBOCIDER_ANE_BACKEND","TURBOCIDER_ALLOW_PRIVATE_ANE","TURBOCIDER_PRIVATE_ANE_DATA_PATH",
                           "TURBOCIDER_PRIVATE_ANE_GPU_IO","TURBOCIDER_PRIVATE_ANE_CHANNELS","TURBOCIDER_PRIVATE_ANE_PREFETCH",
                           "TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE",
                           "TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD","TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE",
                           "TURBOCIDER_PRIVATE_ANE_A8_SINGLE_PASS",
                           "TURBOCIDER_RUNTIME_ANE_PROFILE","TURBOCIDER_PRIVATE_ANE_CACHE_DIR",
                           "TURBOCIDER_PRIVATE_ANE_S1_PROFILE",
                           "TURBOCIDER_Z_RUNTIME_LORA_MPP","TURBOCIDER_Z_MPP_PROJECTIONS",
                           "TURBOCIDER_Z_DISABLE_MPP_PROJECTIONS","TURBOCIDER_Z_MPP_SWIGLU",
                           "TURBOCIDER_Z_DISABLE_MPP_SWIGLU","TURBOCIDER_Z_MPP_SWIGLU_DUAL",
                           "TURBOCIDER_QWEN21_RUNTIME_LORA_FP16"}) {
        const char *raw = std::getenv(key);
        const std::string value = raw ? raw : "<unset>";
        identity += ":" + std::to_string(value.size()) + ":" + value;
    }
    return identity;
}
uint64_t HybridFfn::retained_bytes() const {
    return uint64_t(output_.capacity() + hidden_.capacity()) * sizeof(uint16_t) +
        metrics_.runtime_weight_s1_bytes + smoothquant_bank_margin();
}
uint64_t HybridFfn::smoothquant_bank_margin() const {
    // Even after replacing/clearing the full profile, the two weight banks
    // can retain up to two previous layer vectors until overwritten.
    return graph_ && graph_->data_path() == "w8a8_hadamard"
        ? uint64_t(2) * metrics_.hidden * sizeof(float) : 0;
}
void HybridFfn::set_smoothquant(const std::string &digest,
                              const std::function<std::vector<Tensor>()> &make_scales) {
    require(bool(make_scales) == !digest.empty(), "runtime S1 producer and digest must be supplied together");
    if (!digest.empty()) {
        require(digest.size() == 64 && digest.find_first_not_of("0123456789abcdef") == std::string::npos &&
                bool(make_scales), "runtime S1 requires a validated profile digest");
        require(!graph_ || (graph_->backend() == BackendKind::PrivateANE &&
                    graph_->data_path() == "w8a8_hadamard" && graph_->supports_device_io() &&
                    graph_->supports_device_weight_regions() && channel_split()),
                "runtime S1 requires the private W8A8 device channel executor");
    }
    if (digest == metrics_.runtime_weight_s1_digest &&
            (digest.empty() || !smoothquant_.empty())) return;
    // Current and prefetched banks borrow these vectors until completion.
    drain();
    smoothquant_.clear();
    metrics_.runtime_weight_s1_requested = !digest.empty();
    metrics_.runtime_weight_s1_digest = digest;
    metrics_.runtime_weight_s1_bytes = 0;
    metrics_.runtime_weight_s1_stage_submissions = 0;
    metrics_.runtime_weight_s1_hybrid_blocks = 0;
    if (digest.empty() || !graph_) return;
    const uint64_t growth = uint64_t(32) * metrics_.hidden * sizeof(float);
    const auto observed = observe_runtime_memory(mx::get_active_memory());
    const auto decision = admit_memory(observed, {uint64_t(4) << 30, memory_budget_},
        graph_->estimated_bytes() + retained_bytes(), growth);
    if (!decision.allowed()) {
        release_for_memory("runtime S1 vector admission denied (" +
            memory_denial_reason(decision.denial, observed) + ")");
        return;
    }
    // MLX's host-vector constructor allocates immediately. The factory must
    // run AFTER admission, and a same-digest hit must never call it.
    auto scales = make_scales();
    require(scales.size() == 32, "runtime S1 requires a complete 32-layer profile");
    for (const auto &scale : scales)
        require(scale.ndim() == 2 && scale.shape(0) == 1 && scale.shape(1) == metrics_.hidden &&
                scale.dtype() == mx::float32 && scale.flags().row_contiguous,
                "runtime S1 requires immutable FP32 [1,H] vectors");
    mx::eval(scales);
    smoothquant_ = std::move(scales);
    metrics_.runtime_weight_s1_bytes = growth;
}
void HybridFfn::drain(bool discard_future) {
    if (graph_ && pending_) { graph_->finish(); pending_ = false; }
    if (graph_ && discard_future) {
        graph_->discard_prefetched_weights();
        if(prefetched_layer_>=0)++metrics_.runtime_weight_prefetch_discards;
        prefetched_layer_=-1;
    }
    weights_.clear();
}
void HybridFfn::release_for_memory(const std::string &reason) {
    degrade(reason, -1);
}
bool HybridFfn::admit_scratch(int ane_rows, bool adapter) {
    if (!graph_ || ane_rows <= 0) return false;
    const auto existing_output = uint64_t(output_.capacity()) * sizeof(uint16_t);
    const auto existing_hidden = uint64_t(hidden_.capacity()) * sizeof(uint16_t);
    const auto scratch = plan_host_scratch(ane_rows, metrics_.hidden, metrics_.mlp_width,
                                           adapter, existing_output, existing_hidden);
    const auto resident = graph_->estimated_bytes() + metrics_.runtime_weight_s1_bytes + smoothquant_bank_margin();
    if (!scratch || resident > memory_budget_ ||
        existing_output > memory_budget_ - resident ||
        existing_hidden > memory_budget_ - resident - existing_output) {
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
            resident + existing_output + existing_hidden,
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
    save_scheduler();
    if (graph_) {
        const auto retained = retained_bytes();
        const auto observed = observation ? *observation : observe_runtime_memory(mx::get_active_memory());
        const auto decision = admit_memory(observed,
            {uint64_t(4) << 30, memory_budget_}, graph_->estimated_bytes() + retained, 0);
        if (!decision.allowed())
            release_for_memory("runtime ANE resident memory admission denied (" +
                               memory_denial_reason(decision.denial, observed) + ")");
    }
    const auto cache_key = scheduler_identity_.empty() ? std::string{} :
        scheduler_identity_ + identity_part(adapter_identity);
    if (adapter_identity != adapter_identity_ || cache_key != scheduler_cache_key_) {
        // A new adapter changes GPU correction costs and may change the best
        // split. Keep the executable, but not another adapter's timing model.
        if (graph_) scheduler_ = std::make_unique<RowScheduler>(graph_->shape().rows, configured_chunks(), axis_, profile_);
        adapter_identity_ = adapter_identity;
        scheduler_cache_key_ = cache_key;
        if (scheduler_ && !cache_key.empty())
            metrics_.runtime_weight_scheduler_cache_hit = scheduler_cache().restore(cache_key, *scheduler_);
    }
    metrics_.runtime_weight_scheduler_cache_entries = scheduler_cache().size();
    // Retain warm scheduling samples and capability state across resident
    // requests, but all timing/counters below are explicitly session totals.
    layer_ = -1; rows_ = chunks_ = 0;
    planned_ = false;
    block_plan_.reset();
}
RowScheduler::Plan HybridFfn::plan_block(int layer, int rows) {
    require(!planned_ && !block_plan_ && rows > 0, "runtime ANE block already planned or invalid rows");
    drain(false);
    const auto plan = available() ? scheduler_->plan(layer, rows)
                                 : RowScheduler::Plan{RowScheduler::Mode::Gpu, 0};
    if(!plan.split() && graph_ && prefetched_layer_>=0) {
        graph_->discard_prefetched_weights();prefetched_layer_=-1;++metrics_.runtime_weight_prefetch_discards;
    }
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
    if (!scheduler_cache_key_.empty()) scheduler_cache().erase(scheduler_cache_key_);
    failed_ = metrics_.runtime_failed = true;
    ++metrics_.runtime_failures;
    metrics_.runtime_failure_block = layer;
    reason_ = error;
    // A failed optional route is never retried by this instance. Retire its
    // Core ML worker/model/lease now, rather than retaining them throughout
    // the GPU fallback session. RuntimeGraph destruction drains the worker;
    // keep every borrowed tensor and scratch buffer alive until it returns.
    fill_activation_stage_metrics(metrics_);
    graph_.reset();
    pending_ = false;
    weights_.clear();
    smoothquant_.clear();
    metrics_.runtime_weight_s1_bytes = 0;
    scheduler_.reset();
    std::vector<uint16_t>().swap(output_);
    std::vector<uint16_t>().swap(hidden_);
    chunks_ = 0;
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
    drain(false);
    layer_ = layer; rows_ = rows;
    if (!planned_) chunks_ = available() ? scheduler_->select(layer, rows) : 0;
    planned_ = false;
    if (!chunks_) { if(graph_)graph_->discard_prefetched_weights();prefetched_layer_=-1;return; }
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
            const uint64_t retained=retained_bytes();
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
        if (!graph_->supports_device_weights()) for (const auto &w : weights) sources.push_back(weight_view(w));
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
        if (graph_->supports_device_weights()) {
            auto regions=device_regions(weights,layer);
            const auto prefetch_wait=Clock::now();
            auto hit=prefetched_layer_==layer ? graph_->activate_prefetched_weights(regions) : std::nullopt;
            if(hit) {
                ++metrics_.runtime_weight_prefetch_hits;
                if(!hit->ok)++metrics_.runtime_weight_prefetch_failures;
                metrics_.runtime_weight_prefetch_wait_seconds+=elapsed(prefetch_wait);
                prefetched_layer_=-1;
            } else {
                graph_->discard_prefetched_weights();
                if(prefetched_layer_>=0)++metrics_.runtime_weight_prefetch_discards;
                prefetched_layer_=-1;
                graph_->stage_device_weight_regions(std::move(regions));
            }
        } else graph_->stage_weights(std::move(sources));
        pending_ = true;
        if (!smoothquant_.empty()) ++metrics_.runtime_weight_s1_stage_submissions;
    } catch (const std::exception &error) {
        if(graph_)graph_->discard_prefetched_weights();prefetched_layer_=-1;
        degrade(error.what(), layer); chunks_ = 0;
    }
}
std::vector<DeviceWeightRegion> HybridFfn::device_regions(const std::vector<FfnWeight> &weights,int layer) const {
    require(graph_ && weights.size()==3,"device SwiGLU requires three immutable sources");
    std::vector<DeviceWeightView> source;for(const auto&w:weights)source.push_back(device_weight_view(w));
    require(source[0].rows==metrics_.mlp_width&&source[1].rows==metrics_.mlp_width&&source[2].cols==metrics_.mlp_width,
            "device FFN requires complete physical sources");
    const int first=channel_split()?gpu_channels():0,width=graph_->shape().width,h=metrics_.hidden;
    std::vector<DeviceWeightRegion> regions = {{std::move(source[0]),{first,width,0,h,128}}, {std::move(source[1]),{first,width,0,h,128}},
            {std::move(source[2]),{0,h,first,width,512}}};
    if (!smoothquant_.empty()) {
        require(layer >= 0 && size_t(layer) < smoothquant_.size(), "runtime S1 layer outside profile");
        for (const auto &weight : weights)
            require(!weight.scales && !weight.offsets && weight.transform == FfnWeight::Transform::None,
                    "runtime S1 requires dense local base weights");
        const auto scale = device_view(smoothquant_[size_t(layer)],1,h);
        regions[0].selection.column_scale = regions[1].selection.column_scale = scale;
    }
    return regions;
}
void HybridFfn::maybe_prefetch(int next,int rows,const NextWeights &provider) {
    if(!prefetch_ || !available() || !provider || prefetched_layer_>=0)return;
    const auto plan=scheduler_->peek_plan(next,rows);
    if(!plan.split() || !plan.chunks)return;
    try {
        auto sources=provider(next);if(sources.empty())return;
        // Existing resident sources/slices only. The provider must not decode
        // or create a second checkpoint; all planes are already materialized.
        std::vector<Tensor> ready;
        for(const auto&w:sources){ready.push_back(w.values);if(w.scales)ready.push_back(*w.scales);if(w.offsets)ready.push_back(*w.offsets);}
        mx::eval(ready);
        graph_->prefetch_device_weight_regions(device_regions(sources,next));prefetched_layer_=next;
        ++metrics_.runtime_weight_prefetch_submissions;
    }catch(const std::exception&) {
        graph_->discard_prefetched_weights();prefetched_layer_=-1;++metrics_.runtime_weight_prefetch_failures;
        // Optional prefetch failure never turns a successful CURRENT output
        // into success for the future layer. Its ordinary stage still runs.
    }
}
Tensor HybridFfn::run(int layer, const Tensor &input, const Gpu &gpu,
                      std::atomic<bool> &cancelled, const Adapter *adapter, const ChannelGpu &channel_gpu, const NextWeights &next_weights) {
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
    if (channel_split()) return run_channels(layer, input, gpu, channel_gpu, cancelled, adapter,next_weights);
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
    const bool device_io = graph_->supports_device_io();
    MatrixView tail_input;
    if (!device_io) {
        tail_input = view(packed);
        tail_input.data = static_cast<const char *>(tail_input.data) + size_t(gpu_rows) * packed.shape(1) * packed.itemsize();
        tail_input.bytes -= size_t(gpu_rows) * packed.shape(1) * packed.itemsize();
        tail_input.rows = ane_rows;
    }
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
    std::optional<Tensor> device_tail, device_hidden;
    std::optional<DeviceAdapterInput> device_adapter;
    try {
        if (device_io) {
            device_tail = device_output(ane_rows, input.shape(2), dtype);
            if (adapter) {
                device_hidden = device_output(ane_rows, metrics_.mlp_width, dtype);
                device_adapter = DeviceAdapterInput{device_view(*delta_gate, ane_rows, metrics_.mlp_width),
                    device_view(*delta_up, ane_rows, metrics_.mlp_width), device_view(*device_hidden, ane_rows, metrics_.mlp_width)};
            }
        } else {
            resize_scratch(output_, size_t(ane_rows) * input.shape(2));
            if (adapter) {
                resize_scratch(hidden_, size_t(ane_rows) * metrics_.mlp_width);
                adapter_input = AdapterInput{view(*delta_gate), view(*delta_up), hidden_.data(), hidden_.size()};
            }
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
    if (device_io) graph_->launch_device(device_view(packed, ane_rows, input.shape(2), gpu_rows),
        device_view(*device_tail, ane_rows, input.shape(2)), device_adapter);
    else graph_->launch(tail_input, output_.data(), output_.size(), dtype, adapter_input);
    pending_ = true;
    maybe_prefetch(layer+1,rows_,next_weights);
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
        if (device_io) metrics_.runtime_weight_device_io_calls += result.calls;
        metrics_.prediction_seconds += result.prediction_seconds;
        metrics_.model_prediction_seconds += result.prediction_seconds;
        metrics_.output_handling_seconds += result.output_seconds;
        metrics_.copied_bytes += result.copied_output_bytes;
        // An async join overlaps GPU work: it is NOT exposed ANE time and
        // must never enter the old post-GPU join counter or rate estimator.
        if (!async_head) metrics_.runtime_weight_join_seconds += join;
        metrics_.runtime_weight_gpu_seconds += gpu_seconds;
        metrics_.runtime_weight_overflow_retries += result.overflow_retries;
        metrics_.runtime_weight_a8_prefetches += result.activation_prefetches;
        metrics_.runtime_weight_a8_wait_seconds += result.activation_wait_seconds;
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
            auto base_down = device_io ? mx::astype(*device_tail, input.dtype()) : copy_output(output_, input.shape(2));
            if (!adapter) return base_down;
            auto corrected_hidden = device_io ? mx::astype(*device_hidden, input.dtype()) : copy_output(hidden_, metrics_.mlp_width);
            auto corrected = adapter->down_and_add(corrected_hidden, base_down);
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
        if (graph_ && pending_) { graph_->finish(); pending_ = false; }
        // Cancellation/down-callback errors may precede the final ownership
        // fence. Drain the submitted GPU head as well, preserving the original
        // exception if the GPU reports an error during cleanup.
        if (async_head && head) {
            try { mx::eval(*head); } catch (...) {}
        }
        throw;
    }
}
Tensor HybridFfn::run_channels(int layer, const Tensor &input, const Gpu &gpu, const ChannelGpu &channel_gpu,
                              std::atomic<bool> &cancelled, const Adapter *adapter,const NextWeights &next_weights) {
    require(bool(channel_gpu), "channel split requires a complete GPU range/hidden implementation");
    auto fallback = [&] {
        block_sample_valid_ = false;
        ++metrics_.runtime_weight_gpu_blocks;
        auto output = gpu(input); mx::eval(output);
        return output;
    };
    if (!chunks_ || !available()) return fallback();
    const int chunk = graph_->shape().rows, h = metrics_.hidden, fa = ane_channels(), fg = gpu_channels();
    const uint64_t padded64 = (uint64_t(rows_) + chunk - 1) / chunk * chunk;
    require(padded64 <= INT_MAX && padded64 / chunk <= 128, "channel split row bucket extent too large");
    const int padded = int(padded64);
    // Optional-tier allocations, including temporary padding/corrections and
    // the complete hidden needed for ONE down-LoRA. No checkpoint W copies.
    const uint64_t scratch = padded64 * (uint64_t(h) * 2 + (adapter ? uint64_t(fa) * 10 : 0)) +
        (padded != rows_ ? padded64 * h * input.itemsize() : 0) +
        (adapter ? uint64_t(rows_) * metrics_.mlp_width * input.itemsize() : 0) + uint64_t(rows_) * h * 4;
    const uint64_t resident = graph_->estimated_bytes() + retained_bytes();
    if (resident > memory_budget_ || scratch > memory_budget_ - resident ||
        !admit_memory(observe_runtime_memory(mx::get_active_memory()), {uint64_t(4)<<30,memory_budget_},
                      resident, scratch).allowed()) {
        release_for_memory("runtime channel scratch memory admission denied");
        ++metrics_.runtime_weight_fallback_blocks;
        return fallback();
    }
    auto pad = [&](const Tensor &value, int columns) {
        require(value.shape() == mx::Shape({1,rows_,columns}), "channel correction/input geometry mismatch");
        return mx::contiguous(padded == rows_ ? value : mx::concatenate(
            {value,mx::zeros({1,padded-rows_,columns},value.dtype())},1));
    };
    std::optional<Tensor> packed, gate, up, output, hidden;
    const DType dtype = input.dtype() == mx::float16 ? DType::FP16 : DType::BF16;
    double lora_ready = 0;
    try {
        packed = pad(input,h);
        if (adapter) {
            const auto ready_start = Clock::now();
            auto deltas = adapter->channel_gate_up ? adapter->channel_gate_up(input,fg,fa) :
                                                   adapter->gate_up(input);
            const int correction_width = adapter->channel_gate_up ? fa : metrics_.mlp_width;
            require(deltas.first.shape() == mx::Shape({1,rows_,correction_width}) &&
                    deltas.second.shape() == deltas.first.shape(), "channel LoRA correction geometry mismatch");
            gate = pad(adapter->channel_gate_up ? deltas.first : slice_axis(deltas.first,-1,fg,fg+fa),fa);
            up = pad(adapter->channel_gate_up ? deltas.second : slice_axis(deltas.second,-1,fg,fg+fa),fa);
            mx::eval({*packed,*gate,*up}); lora_ready = elapsed(ready_start);
        } else mx::eval(*packed);
        output = device_output(padded,h,dtype);
        if (adapter) hidden = device_output(padded,fa,dtype);
    } catch (const std::bad_alloc &) {
        release_for_memory("runtime channel scratch allocation failed");
        ++metrics_.runtime_weight_fallback_blocks; return fallback();
    }
    checkpoint(cancelled);
    const double pre_seconds = elapsed(pre_start_);
    metrics_.runtime_weight_pre_seconds += pre_seconds;
    metrics_.runtime_weight_lora_input_ready_seconds += lora_ready;
    const auto wait_start = Clock::now(); const auto staged = graph_->wait_stage(); pending_ = false;
    metrics_.runtime_weight_stage_seconds += staged.stage_seconds;
    metrics_.runtime_weight_stage_wait_seconds += elapsed(wait_start);
    if (!staged.ok) {
        degrade(staged.error,layer); ++metrics_.runtime_weight_fallback_blocks;
        return fallback();
    }
    std::optional<DeviceAdapterInput> correction;
    if (adapter) correction = DeviceAdapterInput{device_view(*gate,padded,fa),device_view(*up,padded,fa),device_view(*hidden,padded,fa)};
    const auto start = Clock::now();
    graph_->launch_device(device_view(*packed,padded,h),device_view(*output,padded,h),correction); pending_ = true;
    maybe_prefetch(layer+1,rows_,next_weights);
    const bool async_head = !profile_ && block_plan_ && block_plan_->mode == RowScheduler::Mode::HybridUntimed;
    std::optional<std::pair<Tensor,Tensor>> head;
    try {
        const auto gpu_start = Clock::now(); head = channel_gpu(input,0,fg);
        require(head->first.shape() == input.shape() && head->second.shape() == mx::Shape({1,rows_,fg}) &&
                head->first.dtype() == input.dtype() && head->second.dtype() == input.dtype(),
                "channel GPU base-down/hidden contract mismatch");
        if (async_head) mx::async_eval({head->first,head->second}); else mx::eval({head->first,head->second});
        const double gpu_seconds = async_head ? 0 : elapsed(gpu_start);
        const auto join_start = Clock::now(); const auto result = graph_->finish(); pending_ = false;
        const double join = elapsed(join_start); checkpoint(cancelled);
        metrics_.calls += result.calls; metrics_.runtime_calls += result.calls;
        metrics_.runtime_weight_device_io_calls += result.calls;
        metrics_.prediction_seconds += result.prediction_seconds;
        metrics_.model_prediction_seconds += result.prediction_seconds;
        metrics_.output_handling_seconds += result.output_seconds;
        metrics_.copied_bytes += result.copied_output_bytes;
        metrics_.runtime_weight_overflow_retries += result.overflow_retries;
        metrics_.runtime_weight_a8_prefetches += result.activation_prefetches;
        metrics_.runtime_weight_a8_wait_seconds += result.activation_wait_seconds;
        metrics_.runtime_weight_headroom = result.headroom_scale;
        metrics_.runtime_weight_gpu_seconds += gpu_seconds;
        if (!async_head) metrics_.runtime_weight_join_seconds += join;
        if (!result.ok) {
            degrade(result.error,layer); ++metrics_.runtime_weight_fallback_blocks;
            if (async_head) mx::eval({head->first,head->second});
            // A late chunk may have written scratch. Recompute the COMPLETE
            // FFN (including full hidden/down-LoRA), never publish that scratch.
            return fallback();
        }
        const auto post_start = Clock::now();
        auto tail = mx::astype(slice_axis(*output,1,0,rows_),input.dtype());
        auto merged = mx::astype(mx::astype(head->first,mx::float32)+mx::astype(tail,mx::float32),input.dtype());
        if (adapter) {
            auto ane_hidden = mx::astype(slice_axis(*hidden,1,0,rows_),input.dtype());
            auto full_hidden = mx::concatenate({head->second,ane_hidden},-1);
            merged = adapter->down_and_add(full_hidden,merged);
            require(merged.shape() == input.shape() && merged.dtype() == input.dtype(), "channel down-LoRA output contract mismatch");
        }
        mx::eval(merged);
        metrics_.runtime_weight_post_join_seconds += elapsed(post_start);
        const double wall = elapsed(start);
        if (block_plan_) {
            block_sample_valid_ = result.overflow_retries == 0;
            block_gpu_seconds_ = gpu_seconds; block_ane_seconds_ = result.total_seconds;
        } else if (!result.overflow_retries) scheduler_->observe(layer,rows_,1,pre_seconds+wall);
        ++metrics_.runtime_weight_hybrid_blocks; ++metrics_.runtime_weight_channel_blocks;
        if (!smoothquant_.empty()) ++metrics_.runtime_weight_s1_hybrid_blocks;
        metrics_.runtime_weight_ane_rows += rows_;
        metrics_.runtime_weight_wall_seconds += wall;
        if (block_plan_ && block_plan_->mode == RowScheduler::Mode::HybridUntimed) ++metrics_.runtime_weight_untimed_hybrid_blocks;
        if (async_head) { ++metrics_.runtime_weight_async_hybrid_blocks; metrics_.runtime_weight_async_ane_wait_seconds += join; }
        if (profile_) std::cerr << "{\"runtime_ane_layer\":" << layer << ",\"axis\":\"channels\",\"rows\":" << rows_
            << ",\"padded_rows\":" << padded << ",\"gpu_channels\":" << fg << ",\"ane_channels\":" << fa
            << ",\"stage_seconds\":" << staged.stage_seconds << ",\"gpu_seconds\":" << gpu_seconds
            << ",\"ane_seconds\":" << result.total_seconds << ",\"ffn_seconds\":" << wall << "}\n";
        return merged;
    } catch (...) {
        if (graph_ && pending_) { graph_->finish(); pending_ = false; }
        if (head) { try { mx::eval({head->first,head->second}); } catch (...) {} }
        throw;
    }
}
void HybridFfn::fill_activation_stage_metrics(HybridMetrics &metrics) const {
    if (!graph_) return;
    const auto activation=graph_->activation_stage_stats();
    metrics.runtime_weight_a8_single_pass_requested=activation.requested;
    metrics.runtime_weight_a8_single_pass_pipeline_compiled=activation.pipeline_compiled;
    metrics.runtime_weight_a8_single_pass_submissions=activation.eligible_submissions;
    metrics.runtime_weight_a8_single_pass_ineligible_submissions=activation.ineligible_submissions;
}
HybridMetrics HybridFfn::metrics() const {
    auto metrics = metrics_;
    metrics.runtime_weight_s1_bank_margin_bytes = smoothquant_bank_margin();
    fill_activation_stage_metrics(metrics);
    if(graph_) {
        const auto pipelines=graph_->stage_pipeline_stats();
        metrics.runtime_weight_stage_specialized=pipelines.specialized;
        metrics.runtime_weight_stage_pipeline_variants=pipelines.variants;
        const auto cache=graph_->weight_cache_stats();
        metrics.runtime_weight_scale_cache_enabled=cache.enabled;
        metrics.runtime_weight_scale_cache_hits=cache.hits;metrics.runtime_weight_scale_cache_misses=cache.misses;
        metrics.runtime_weight_scale_cache_entries=cache.entries;metrics.runtime_weight_scale_cache_bytes=cache.bytes;
        metrics.runtime_weight_scale_cache_evictions=cache.evictions;
    }
    metrics.prefill_plan_reason = reason_;
    if (metrics.runtime_weight_s1_hybrid_blocks)
        metrics.runtime_weight_source_recipe += ":smoothquant-s1-fp32-v1:" + metrics.runtime_weight_s1_digest;
    return metrics;
}

} // namespace tc::ane
