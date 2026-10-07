#include "ane_ffn.hpp"
#include "ane_gpu_layer_policy.hpp"
#include "ane_backend.hpp"
#include "ane_runtime_quant.hpp"
#include "ane_runtime_packed.hpp"

#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <new>
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
    const size_t bytes = size_t(rows) * cols * (dtype==DType::FP32?4:2);
    auto buffer = mx::allocator::malloc(bytes);
    if (!buffer.ptr()) throw std::bad_alloc();
    return Tensor(buffer, {1, rows, cols}, dtype == DType::FP32 ? mx::float32 : dtype == DType::FP16 ? mx::float16 : mx::bfloat16);
}
WeightView weight_view(const FfnWeight &w) {
    if (w.raw_gguf) {
        const auto &a=w.values;
        require(!w.scales && !w.offsets && w.transform==FfnWeight::Transform::None &&
                a.dtype()==mx::uint8 && a.ndim()==2 && a.flags().row_contiguous,
                "runtime ANE raw GGUF cannot carry affine metadata/rotation");
        GgufView source{a.data<uint8_t>(),a.nbytes(),a.shape(0),w.raw_gguf->columns,
                        size_t(a.shape(1)),w.raw_gguf->type};
        require(gguf_row_bytes(source)==size_t(a.shape(1)),"runtime ANE raw GGUF physical row mismatch");
        validate_gguf_view(source);
        return source;
    }
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
    const bool comfy=w.transform==FfnWeight::Transform::ComfyH256Inverse;
    require(w.transform == FfnWeight::Transform::None || comfy, "W8 source transform unsupported");
    const auto &a=w.values;
    require(a.ndim()==2 && a.flags().row_contiguous && a.offset()>=0 && a.buffer().ptr(), "W8 weight requires produced contiguous buffer");
    DeviceWeightView source;
    source.buffer=const_cast<void*>(a.buffer().ptr());source.buffer_bytes=a.buffer_size();source.offset_bytes=size_t(a.offset());
    source.rows=a.shape(0);source.owner=std::make_shared<Tensor>(a);source.group_size=w.group_size;
    source.allocation_identity=a.data_shared_ptr();source.immutable_generation=true;
    if (w.raw_gguf) {
        require(!comfy && !w.scales && !w.offsets && a.dtype()==mx::uint8,
                "W8 raw GGUF cannot carry affine metadata/rotation");
        switch(w.raw_gguf->type) {
            case 2:source.encoding=DeviceWeightEncoding::GgufQ4_0;break;
            case 8:source.encoding=DeviceWeightEncoding::GgufQ8_0;break;
            case 12:source.encoding=DeviceWeightEncoding::GgufQ4_K;break;
            case 14:source.encoding=DeviceWeightEncoding::GgufQ6_K;break;
            default:throw std::invalid_argument("W8 raw GGUF encoding unsupported");
        }
        source.cols=w.raw_gguf->columns;source.row_stride_bytes=size_t(a.shape(1));
        source.logical_content_identity=w.raw_gguf->logical_content_identity;
        const auto row_bytes=gguf_row_bytes({nullptr,0,source.rows,source.cols,0,w.raw_gguf->type});
        require(row_bytes==source.row_stride_bytes && source.offset_bytes<=source.buffer_bytes &&
                size_t(source.rows)<= (source.buffer_bytes-source.offset_bytes)/row_bytes,
                "W8 raw GGUF physical storage mismatch");
        return source;
    }
    if (!w.scales) {
        require(!comfy,"W8 ConvRot requires explicit row scales");
        const auto view=device_view(a,a.shape(0),a.shape(1));source.cols=view.cols;source.row_stride_bytes=view.row_stride_bytes;source.dense_dtype=view.dtype;
        require(!w.offsets,"W8 dense source has affine offsets");
    } else {
        if(comfy && a.dtype()==mx::int8) {
            require(!w.offsets,"W8 raw ConvRot cannot have affine offsets");
            source.cols=a.shape(1);source.row_stride_bytes=size_t(source.cols);
            source.encoding=DeviceWeightEncoding::ConvrotQ8Signed;
            source.scales=device_view(*w.scales,a.shape(0),1);
            require(source.scales->dtype==DType::FP32,"W8 raw ConvRot requires original FP32 scales");
            return source;
        }
        require(a.dtype()==mx::uint32 && (w.bits==4||w.bits==8),"W8 affine source requires Q4/Q8 words");
        require(!comfy || (w.bits==8 && w.offsets),"W8 packed ConvRot requires Q8 signed offsets");
        source.cols=a.shape(1)*(32/w.bits);source.row_stride_bytes=size_t(a.shape(1))*4;
        source.encoding=comfy?DeviceWeightEncoding::ConvrotQ8Packed:w.bits==4?DeviceWeightEncoding::AffineQ4:DeviceWeightEncoding::AffineQ8;
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
} // namespace

HybridFfn::HybridFfn(const std::filesystem::path &manifest, int hidden, int width,
                     size_t budget, std::atomic<bool> &cancelled, bool require_lora_inputs,
                     const CalibrationWorkload *calibration, std::optional<int> calibrated_channels)
    : memory_budget_(budget) {
    checkpoint(cancelled);
    const char *prefetch_after=std::getenv("TURBOCIDER_RUNTIME_ANE_PREFETCH_AFTER_GPU");
    require(!prefetch_after || std::string(prefetch_after)=="0" || std::string(prefetch_after)=="1",
            "runtime ANE prefetch placement requires 0 or 1");
    prefetch_after_gpu_=!prefetch_after || std::string(prefetch_after)=="1";
    metrics_.runtime_weight_prefetch_after_gpu=prefetch_after_gpu_;
    if (private_channel_count(width) < 0 && !calibrated_channels) {
        require(calibration != nullptr, "automatic ANE channels require a model-supplied calibration workload");
        const auto selection = calibrate_channels(manifest, hidden, width, budget, cancelled, require_lora_inputs, *calibration);
        calibrated_channels = selection.channels;
        metrics_.runtime_weight_calibration = selection.report;
        calibration_reason_ = selection.reason + (selection.cache_hit ? "; cache hit" : "; measured/not cached");
        calibration_declined_ = selection.channels == 0;
    }
    const int selected_channels = resolved_private_channel_count(width, calibrated_channels);
    const int chunks = configured_chunks();
    const char *lora_range = std::getenv("TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE");
    require(!lora_range || std::string(lora_range)=="0" || std::string(lora_range)=="1",
            "runtime ANE LoRA channel range requires 0 or 1");
    lora_channel_range_ = !lora_range || std::string(lora_range)=="1";
    const char *profile = std::getenv("TURBOCIDER_RUNTIME_ANE_PROFILE");
    require(!profile || std::string(profile) == "0" || std::string(profile) == "1",
            "TURBOCIDER_RUNTIME_ANE_PROFILE accepts 0 or 1");
    profile_ = profile && std::string(profile) == "1";
    const char *fixed_async = std::getenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
    require(!fixed_async || std::string(fixed_async)=="0" || std::string(fixed_async)=="1",
            "runtime ANE fixed async requires 0 or 1");
    fixed_async_ = fixed_async && std::string(fixed_async)=="1";
    require(!fixed_async_ || (chunks>0 && !profile_),
            "runtime ANE fixed async requires positive fixed chunks and profiling disabled");
    const char *defer = std::getenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN");
    require(!defer || std::string(defer)=="0" || std::string(defer)=="1",
            "runtime ANE deferred channel join requires 0 or 1");
    const bool requested_defer = defer && std::string(defer)=="1";
    const bool requested_fp32=configured_fp32_channel_join();
    require(!requested_defer || (fixed_async_ && (selected_channels>0 || calibration_declined_) &&
            configured_backend().allow_private &&
            (configured_backend().preferred==BackendPreference::Private ||
             configured_backend().preferred==BackendPreference::Auto)),
            "runtime ANE deferred join requires authorized private channels and fixed async");
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
        auto built = build_runtime_executor(manifest, budget,
            GraphGeometry{Kind::SwiGLU, hidden, width, require_lora_inputs}, configured_backend(), calibrated_channels);
        graph_ = std::move(built.executor); verified = built.self_test_passed;
        verified_seconds = built.self_test_seconds;
        metrics_.runtime_weight_backend_fallback_reason = std::move(built.fallback_reason);
        if (!graph_) { failed_ = !calibration_declined_; reason_ = calibration_declined_ ? calibration_reason_ :
            metrics_.runtime_weight_backend_fallback_reason; return; }
    }
    catch (const MemoryBudgetError &error) { degrade(error.what(), -1); return; }
    catch (const CapabilityError &error) { degrade(error.what(), -1); return; }
    metrics_.runtime_weight_backend = graph_->backend() == BackendKind::PrivateANE ? "private_ane" : "public_coreml";
    metrics_.runtime_weight_io_path = graph_->supports_device_io() ? "gpu_iosurface" : "host";
    metrics_.runtime_weight_data_path = graph_->data_path();
    metrics_.runtime_weight_source_recipe = graph_->weight_recipe();
    metrics_.runtime_weight_launch_fence_enabled=graph_->device_submission_fence_enabled();
    metrics_.runtime_weight_a8_lookahead_enabled=graph_->activation_lookahead_enabled();
    metrics_.runtime_weight_a8_group_size=graph_->activation_group_size();
    metrics_.runtime_weight_hidden_a8_group_size=graph_->hidden_activation_group_size();
    if (graph_->shape().width != width) {
        require(graph_->shape().width > 0 && graph_->shape().width < width && graph_->supports_device_weight_regions() &&
                graph_->supports_device_io(), "runtime channel split requires a smaller device-source graph");
        axis_ = PartitionAxis::IntermediateChannels;
        metrics_.runtime_weight_partition_axis = "intermediate_channels";
        metrics_.runtime_weight_ane_channels = graph_->shape().width;
        metrics_.runtime_weight_gpu_channels = width - graph_->shape().width;
    }
    if (metrics_.runtime_weight_data_path == "w8a8_hadamard" || metrics_.runtime_weight_data_path == "w8a8_convrot") {
        metrics_.weight_variant = "runtime_w8a8";
    }
    require(!requested_fp32 || (channel_split() && graph_->supports_fp32_device_output()),
            "F32 channel join requires a supported Private W8 channel executor");
    fp32_channel_join_=requested_fp32;
    metrics_.runtime_weight_fp32_channel_join_enabled=fp32_channel_join_;
    if(fp32_channel_join_)metrics_.runtime_weight_source_recipe+="+fp32-partial-join-v1";
    defer_channel_join_ = requested_defer && channel_split();
    metrics_.runtime_weight_deferred_join_enabled = defer_channel_join_;
    metrics_.bucket = graph_->shape().rows;
    if (graph_->shape().lora_inputs) metrics_.mlp_output_kind = "runtime_weight_swiglu_lora_inputs";
    metrics_.load_seconds = graph_->load_seconds();
    metrics_.runtime_weight_slot_bytes = graph_->slot_bytes();
    metrics_.runtime_weight_estimated_bytes = graph_->estimated_bytes();
    std::string error;
    const auto start = Clock::now();
    if (!verified && !graph_->self_test(error)) degrade("self_test_failed: " + error, -1);
    metrics_.zero_input_warmup_seconds = verified ? verified_seconds : elapsed(start); // sparse nonzero self-test, not measured FFN
    scheduler_ = std::make_unique<RowScheduler>(graph_->shape().rows, chunks, axis_);
    const char *prefetch = std::getenv("TURBOCIDER_PRIVATE_ANE_PREFETCH");
    require(!prefetch || std::string(prefetch)=="0" || std::string(prefetch)=="1", "private ANE prefetch requires 0 or 1");
    // Matched M4 Max channel pilots show real future-bank hits but a net
    // bandwidth tax. Keep implemented prefetch explicit until calibration
    // measures a profitable policy; never silently lose the row/channel win.
    prefetch_ = graph_->supports_weight_prefetch() && prefetch && std::string(prefetch)=="1";
    metrics_.runtime_weight_prefetch_enabled = prefetch_;
    checkpoint(cancelled);
}
DeviceWeightView HybridFfn::calibration_source(const FfnWeight &weight) { return device_weight_view(weight); }
HybridFfn::~HybridFfn() { drain(); }
std::string HybridFfn::executor_configuration_identity() {
    std::string identity;
    for (const char *key : {"TURBOCIDER_ANE_BACKEND","TURBOCIDER_ALLOW_PRIVATE_ANE","TURBOCIDER_PRIVATE_ANE_DATA_PATH",
                           "TURBOCIDER_PRIVATE_ANE_GPU_IO","TURBOCIDER_PRIVATE_ANE_CHANNELS","TURBOCIDER_PRIVATE_ANE_PREFETCH",
                           "TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE",
                           "TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD","TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE",
                           "TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE","TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC",
                           "TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN","TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN",
                           "TURBOCIDER_RUNTIME_ANE_PREFETCH_AFTER_GPU",
                           "TURBOCIDER_RUNTIME_ANE_CHUNKS","TURBOCIDER_RUNTIME_ANE_PROFILE",
                           "TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE","TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE",
                           "TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES",
                           "TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES"}) {
        const char *raw = std::getenv(key);
        const std::string value = raw ? raw : "<unset>";
        identity += ":" + std::to_string(value.size()) + ":" + value;
    }
    return identity;
}
void HybridFfn::drain(bool discard_future) {
    // Staging can be in flight BEFORE launch sets pending_. In particular a
    // retained encoder must finish those borrowed-source reads before a
    // failed attention scope destroys its request-local checkpoint arrays.
    if (graph_) { graph_->finish(); pending_ = false; }
    if (graph_ && discard_future) {
        graph_->discard_prefetched_weights();
        if(prefetched_layer_>=0)++metrics_.runtime_weight_prefetch_discards;
        prefetched_layer_=-1;
    }
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
        if (graph_) scheduler_ = std::make_unique<RowScheduler>(graph_->shape().rows, configured_chunks(), axis_);
        adapter_identity_ = adapter_identity;
    }
    // Retain warm scheduling samples and capability state across resident
    // requests, but all timing/counters below are explicitly session totals.
    layer_ = -1; rows_ = chunks_ = 0;
    planned_ = false;
    block_plan_.reset();
    row_policy_={};
}
void HybridFfn::set_gpu_layers(std::vector<int> layers) {
    layers=normalized_gpu_layers(std::move(layers));
    require(!planned_ && !pending_ && !block_plan_,"cannot change runtime GPU layer policy during an operation");
    if(layers==metrics_.runtime_weight_gpu_layers)return;
    drain();
    if(graph_)scheduler_=std::make_unique<RowScheduler>(graph_->shape().rows,configured_chunks(),axis_);
    metrics_.runtime_weight_gpu_layers=std::move(layers);
}
RowScheduler::Plan HybridFfn::plan_block(int layer, int rows,RowPolicy row_policy) {
    require(!planned_ && !block_plan_ && rows > 0, "runtime ANE block already planned or invalid rows");
    const int eligible=eligible_ane_rows(rows,row_policy);
    require(!channel_split() || (row_policy.placement==RowPlacement::Suffix && !row_policy.protected_suffix_rows),
            "channel split cannot use a row placement policy");
    if(row_policy.placement!=RowPlacement::Suffix)
        require(metrics_.runtime_weight_row_placement=="suffix" ||
                metrics_.runtime_weight_row_placement==row_placement_name(row_policy.placement),
                "row placement cannot change inside a runtime session");
    drain(false);
    const bool forced=std::binary_search(metrics_.runtime_weight_gpu_layers.begin(),metrics_.runtime_weight_gpu_layers.end(),layer);
    auto plan = available() && !forced ? scheduler_->plan(layer, rows)
                                 : RowScheduler::Plan{RowScheduler::Mode::Gpu, 0};
    if(plan.split() && !channel_split()) {
        const int requested_chunks=plan.chunks;
        plan.chunks=std::min(plan.chunks,eligible/graph_->shape().rows);
        if(requested_chunks>0 && !plan.chunks)plan={RowScheduler::Mode::Gpu,0};
    }
    // Fixed partitions do not learn from timing samples. Let an explicit
    // ablation use the already-owned untimed plan instead of adding a GPU
    // head completion fence and a whole-block measurement on every visit.
    // Zero-chunk probes and adaptive calibration keep their original plan.
    if (fixed_async_ && plan.mode==RowScheduler::Mode::Hybrid)
        plan.mode=RowScheduler::Mode::HybridUntimed;
    if(!plan.split() && graph_ && prefetched_layer_>=0) {
        graph_->discard_prefetched_weights();prefetched_layer_=-1;++metrics_.runtime_weight_prefetch_discards;
    }
    layer_ = layer; rows_ = rows; chunks_ = plan.chunks;
    row_policy_=row_policy;
    planned_ = plan.split();
    block_sample_valid_ = plan.mode == RowScheduler::Mode::GpuProbe;
    block_gpu_seconds_ = block_ane_seconds_ = 0;
    if (plan.measured() || plan.split()) block_plan_ = plan;
    else {
        ++metrics_.runtime_weight_gpu_blocks;
        ++metrics_.runtime_weight_unsplit_gpu_blocks;
        if(forced)++metrics_.runtime_weight_forced_gpu_blocks;
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
    require(!std::binary_search(metrics_.runtime_weight_gpu_layers.begin(),metrics_.runtime_weight_gpu_layers.end(),layer),
            "explicit GPU layer must not stage the FFN bridge");
    require(!block_plan_ || (planned_ && block_plan_->split()),
            "runtime ANE cannot stage a full GPU probe or restage a measured block");
    require(!planned_ || (layer == layer_ && rows == rows_), "runtime ANE plan/stage mismatch");
    pre_start_ = Clock::now();
    drain(false);
    layer_ = layer; rows_ = rows;
    if (!planned_) {
        row_policy_={};
        chunks_ = available() ? scheduler_->select(layer, rows) : 0;
    }
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
        if (!graph_->supports_device_weights()) for (const auto &w : weights) sources.push_back(weight_view(w));
        const bool rotated=std::any_of(weights.begin(),weights.end(),[](const auto &w) {
            return w.transform==FfnWeight::Transform::ComfyH256Inverse;
        });
        if (rotated) {
            require(std::all_of(weights.begin(),weights.end(),[](const auto &w) {
                return w.transform==FfnWeight::Transform::ComfyH256Inverse;
            }),"runtime ANE mixed ConvRot FFN recipes unsupported");
            metrics_.runtime_weight_source_recipe=graph_->data_path()=="w8a8_convrot"?graph_->weight_recipe():
                "convrot-legacy-packed-scale-inverse-h256-f16-v1";
            if(fp32_channel_join_)metrics_.runtime_weight_source_recipe+="+fp32-partial-join-v1";
            ++metrics_.runtime_weight_convrot_stage_submissions;
        }
        if (graph_->supports_device_weights()) {
            auto regions=device_regions(weights);
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
    } catch (const std::exception &error) {
        if(graph_)graph_->discard_prefetched_weights();prefetched_layer_=-1;
        degrade(error.what(), layer); chunks_ = 0;
    }
}
std::vector<DeviceWeightRegion> HybridFfn::device_regions(const std::vector<FfnWeight> &weights) const {
    require(graph_ && weights.size()==3,"device SwiGLU requires three immutable sources");
    std::vector<DeviceWeightView> source;for(const auto&w:weights)source.push_back(device_weight_view(w));
    require(source[0].rows==metrics_.mlp_width&&source[1].rows==metrics_.mlp_width&&source[2].cols==metrics_.mlp_width,
            "device FFN requires complete physical sources");
    const int first=channel_split()?gpu_channels():0,width=graph_->shape().width,h=metrics_.hidden;
    const bool comfy=graph_->data_path()=="w8a8_convrot";
    const int up=comfy?256:128,down=comfy?256:512;const uint64_t seed=comfy?0:20260930;
    const auto basis=comfy?W8Basis::ComfyH256:W8Basis::SylvesterDH;
    return {{std::move(source[0]),{first,width,0,h,up,seed,false,basis}},
            {std::move(source[1]),{first,width,0,h,up,seed,false,basis}},
            {std::move(source[2]),{0,h,first,width,down,seed,false,basis}}};
}
void HybridFfn::fail_staging(int layer,int rows,const std::string &reason) {
    require(planned_ && block_plan_ && layer==layer_ && rows==rows_ && !reason.empty(),
            "runtime ANE source failure requires matching planned block/reason");
    drain();planned_=false;block_sample_valid_=false;
    if(!block_plan_->measured())block_plan_.reset();
    degrade(reason,layer);chunks_=0;++metrics_.runtime_weight_fallback_blocks;
}
void HybridFfn::maybe_prefetch(int next,int rows,const NextWeights &provider) {
    if(!prefetch_ || !available() || !provider || prefetched_layer_>=0)return;
    if(std::binary_search(metrics_.runtime_weight_gpu_layers.begin(),metrics_.runtime_weight_gpu_layers.end(),next))return;
    const auto plan=scheduler_->peek_plan(next,rows);
    if(!plan.split() || !plan.chunks)return;
    try {
        auto sources=provider(next);if(sources.empty())return;
        // Existing resident sources/slices only. The provider must not decode
        // or create a second checkpoint; all planes are already materialized.
        std::vector<Tensor> ready;
        for(const auto&w:sources){ready.push_back(w.values);if(w.scales)ready.push_back(*w.scales);if(w.offsets)ready.push_back(*w.offsets);}
        mx::eval(ready);
        graph_->prefetch_device_weight_regions(device_regions(sources));prefetched_layer_=next;
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
        RowPolicy &rows;
        ~UntimedPlanGuard() { if (plan && !plan->measured()) plan.reset();rows={}; }
    } untimed_plan_guard{block_plan_,row_policy_};
    if (channel_split()) return run_channels(layer, input, gpu, channel_gpu, cancelled, adapter,next_weights);
    // Stage while the GPU computes attention and, for a real adapter, its
    // activation corrections. One readiness fence suffices before borrowing
    // any host pointers; do not first drain attention and then submit LoRA.
    auto packed = mx::contiguous(mx::reshape(input, {rows_, input.shape(2)}));
    std::optional<Tensor> delta_gate, delta_up;
    double lora_input_ready_seconds = 0;
    if (chunks_ && available() && !admit_scratch(chunks_ * graph_->shape().rows, adapter != nullptr))
        chunks_ = 0;
    const auto window=chunks_ && available()?
        plan_row_window(rows_,chunks_*graph_->shape().rows,row_policy_):RowWindow{};
    const uint64_t row_pack_bytes=window.gpu_before()>0 && window.gpu_after()>0?
        uint64_t(rows_-window.count)*uint64_t(input.shape(2))*input.itemsize():0;
    if(row_pack_bytes) {
        const uint64_t retained=uint64_t(output_.capacity()+hidden_.capacity())*sizeof(uint16_t);
        const auto observed=observe_runtime_memory(mx::get_active_memory());
        const auto decision=admit_memory(observed,{uint64_t(4)<<30,memory_budget_},
            graph_->estimated_bytes()+retained,row_pack_bytes);
        if(!decision.allowed()) {
            release_for_memory("runtime row complement pack memory admission denied");chunks_=0;
            ++metrics_.runtime_weight_fallback_blocks;
        }
    }
    if (adapter && chunks_ && available()) {
        checkpoint(cancelled);
        const auto ready_start = Clock::now();
        const int ane_rows = chunks_ * graph_->shape().rows;
        auto deltas = adapter->gate_up(slice_axis(input, 1, window.first,window.end()));
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
        tail_input.data = static_cast<const char *>(tail_input.data) + size_t(window.first) * packed.shape(1) * packed.itemsize();
        tail_input.bytes -= size_t(window.first) * packed.shape(1) * packed.itemsize();
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
    if (device_io) graph_->launch_device(device_view(packed, ane_rows, input.shape(2),window.first),
        device_view(*device_tail, ane_rows, input.shape(2)), device_adapter);
    else graph_->launch(tail_input, output_.data(), output_.size(), dtype, adapter_input);
    pending_ = true;
    if(!prefetch_after_gpu_)maybe_prefetch(layer+1,rows_,next_weights);
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
        auto gpu_input=window.gpu_before()==0?slice_axis(input,1,window.end(),rows_):
            window.gpu_after()==0?slice_axis(input,1,0,window.first):
            mx::concatenate({slice_axis(input,1,0,window.first),slice_axis(input,1,window.end(),rows_)},1);
        head = gpu(gpu_input);
        if (async_head) mx::async_eval(*head);
        else mx::eval(*head);
        const double gpu_seconds = async_head ? 0 : elapsed(gpu_start);
        // Raw providers may synchronously read source bytes. Submit the GPU
        // complement first so this optional future work does not postpone its
        // producer. This is host scheduling, not physical-overlap evidence.
        if(prefetch_after_gpu_)maybe_prefetch(layer+1,rows_,next_weights);
        const auto join_start = Clock::now();
        auto result = graph_->finish();
        pending_ = false;
        const double join = elapsed(join_start);
        checkpoint(cancelled);
        metrics_.runtime_weight_overflow_events.record({layer,rows_,metrics_.runtime_calls,result.calls,
            result.overflow_retries,result.headroom_start_scale,result.headroom_scale,result.ok});
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
                return gpu(slice_axis(input, 1, window.first,window.end()));
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
        auto output=window.gpu_before()==0?mx::concatenate({tail,*head},1):
            window.gpu_after()==0?mx::concatenate({*head,tail},1):
            mx::concatenate({slice_axis(*head,1,0,window.gpu_before()),tail,
                slice_axis(*head,1,window.gpu_before(),gpu_rows)},1);
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
        if(result.ok) {
            metrics_.runtime_weight_row_pack_peak_bytes=std::max(metrics_.runtime_weight_row_pack_peak_bytes,row_pack_bytes);
            if(row_policy_.placement==RowPlacement::Suffix)++metrics_.runtime_weight_row_suffix_blocks;
            else {
                metrics_.runtime_weight_row_placement=row_placement_name(row_policy_.placement);
                if(row_policy_.placement==RowPlacement::ImagePrefix)++metrics_.runtime_weight_row_prefix_blocks;
                else ++metrics_.runtime_weight_row_image_tail_blocks;
                metrics_.runtime_weight_row_protected_rows+=row_policy_.protected_suffix_rows;
            }
        }
        metrics_.runtime_weight_wall_seconds += wall;
        if (profile_) std::cerr << "{\"runtime_ane_layer\":" << layer << ",\"rows\":" << rows_
            << ",\"ane_rows\":" << ane_rows << ",\"stage_seconds\":" << staged.stage_seconds
            << ",\"ane_row_first\":"<<window.first<<",\"protected_suffix_rows\":"<<row_policy_.protected_suffix_rows
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
Tensor HybridFfn::run_channels(int layer, const Tensor &input, const Gpu &gpu, const ChannelGpu &channel_gpu,
                              std::atomic<bool> &cancelled, const Adapter *adapter,const NextWeights &next_weights) {
    require(bool(channel_gpu), "channel split requires a complete GPU range/hidden implementation");
    require(!fp32_channel_join_ || input.dtype()==mx::bfloat16,
            "experimental F32 channel join requires original BF16 activations");
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
    const uint64_t scratch = padded64 * (uint64_t(h) * (fp32_channel_join_?4:2) + (adapter ? uint64_t(fa) * 10 : 0)) +
        (padded != rows_ ? padded64 * h * input.itemsize() : 0) +
        (adapter ? uint64_t(rows_) * metrics_.mlp_width * input.itemsize() : 0) + uint64_t(rows_) * h * 4 +
        (fp32_channel_join_?uint64_t(rows_)*h*2:0); // extra GPU head partial bytes, not just restored tail
    if (graph_->estimated_bytes() > memory_budget_ || scratch > memory_budget_ - graph_->estimated_bytes() ||
        !admit_memory(observe_runtime_memory(mx::get_active_memory()), {uint64_t(4)<<30,memory_budget_},
                      graph_->estimated_bytes(), scratch).allowed()) {
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
    const DType output_dtype=fp32_channel_join_?DType::FP32:dtype;
    double lora_ready = 0;
    try {
        packed = pad(input,h);
        if (adapter) {
            const auto ready_start = Clock::now();
            const bool narrow = lora_channel_range_ && bool(adapter->gate_up_channels);
            auto deltas = narrow ? adapter->gate_up_channels(input,fg,fa) : adapter->gate_up(input);
            require(deltas.first.shape() == mx::Shape({1,rows_,narrow ? fa : metrics_.mlp_width}) &&
                    deltas.second.shape() == deltas.first.shape(), "channel LoRA correction geometry mismatch");
            gate = pad(narrow ? deltas.first : slice_axis(deltas.first,-1,fg,fg+fa),fa);
            up = pad(narrow ? deltas.second : slice_axis(deltas.second,-1,fg,fg+fa),fa);
            mx::eval({*packed,*gate,*up}); lora_ready = elapsed(ready_start);
            if (narrow) ++metrics_.runtime_weight_lora_channel_range_calls;
            else ++metrics_.runtime_weight_lora_channel_full_calls;
        } else mx::eval(*packed);
        output = device_output(padded,h,output_dtype);
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
    if(!prefetch_after_gpu_)maybe_prefetch(layer+1,rows_,next_weights);
    const bool async_head = !profile_ && block_plan_ && block_plan_->mode == RowScheduler::Mode::HybridUntimed;
    std::optional<std::pair<Tensor,Tensor>> head;
    try {
        const auto gpu_start = Clock::now(); head = channel_gpu(input,0,fg);
        require(head->first.shape() == input.shape() && head->second.shape() == mx::Shape({1,rows_,fg}) &&
                head->first.dtype() == (fp32_channel_join_?mx::float32:input.dtype()) && head->second.dtype() == input.dtype(),
                "channel GPU base-down/hidden contract mismatch");
        if (async_head) mx::async_eval({head->first,head->second}); else mx::eval({head->first,head->second});
        const double gpu_seconds = async_head ? 0 : elapsed(gpu_start);
        if(prefetch_after_gpu_)maybe_prefetch(layer+1,rows_,next_weights);
        const auto join_start = Clock::now(); const auto result = graph_->finish(); pending_ = false;
        const double join = elapsed(join_start); checkpoint(cancelled);
        metrics_.runtime_weight_overflow_events.record({layer,rows_,metrics_.runtime_calls,result.calls,
            result.overflow_retries,result.headroom_start_scale,result.headroom_scale,result.ok});
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
        auto tail = mx::astype(slice_axis(*output,1,0,rows_),fp32_channel_join_?mx::float32:input.dtype());
        auto merged = mx::astype(mx::astype(head->first,mx::float32)+mx::astype(tail,mx::float32),input.dtype());
        if (adapter) {
            auto ane_hidden = mx::astype(slice_axis(*hidden,1,0,rows_),input.dtype());
            auto full_hidden = mx::concatenate({head->second,ane_hidden},-1);
            merged = adapter->down_and_add(full_hidden,merged);
            require(merged.shape() == input.shape() && merged.dtype() == input.dtype(), "channel down-LoRA output contract mismatch");
        }
        // ANE and its GPU restoration are already complete. Each output and
        // hidden backing is independently MLX-owned, not the next job's y
        // surface. Only the explicit fixed/untimed channel experiment may
        // leave the join/down-LoRA lazy for the family's normal consumer.
        // Measured plans, fallback and exception cleanup keep their fences.
        const bool deferred = defer_channel_join_ && async_head;
        if (!deferred) mx::eval(merged);
        if (deferred) ++metrics_.runtime_weight_deferred_join_blocks;
        metrics_.runtime_weight_post_join_seconds += elapsed(post_start);
        const double wall = elapsed(start);
        if (block_plan_) {
            block_sample_valid_ = result.overflow_retries == 0;
            block_gpu_seconds_ = gpu_seconds; block_ane_seconds_ = result.total_seconds;
        } else if (!result.overflow_retries) scheduler_->observe(layer,rows_,1,pre_seconds+wall);
        ++metrics_.runtime_weight_hybrid_blocks; ++metrics_.runtime_weight_channel_blocks;
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
        graph_->finish(); pending_ = false;
        if (head) { try { mx::eval({head->first,head->second}); } catch (...) {} }
        throw;
    }
}
HybridMetrics HybridFfn::metrics() const {
    auto metrics = metrics_;
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
    return metrics;
}

} // namespace tc::ane
