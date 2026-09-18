#include "../../runtime/acceleration.hpp"
#include "flux.hpp"
#include "flux_streaming.hpp"
#include "../../components/text/qwen3.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/residency.hpp"
#include "../../runtime/streaming/audit.hpp"
#include "../../runtime/streaming/canonical_encoding.hpp"
#include "../../runtime/streaming/resolved_request.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include "../../runtime/streaming/actual_receipt.hpp"
#include "../../media/image.hpp"
#include "../../backends/coreml.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
namespace tc {
namespace {

bool flux_exact_streaming_requested(const Request &request) {
    if (!request.streaming.active()) return false;
    const auto stage = request.streaming.stages.find("denoiser");
    return stage != request.streaming.stages.end() &&
        stage->second.residency &&
        *stage->second.residency == "streamed";
}

constexpr const char *kFluxExactKernelRevision =
    "flux2-klein-9b-mlx-eager-block-v1";
constexpr const char *kFluxPublicComponentPolicy =
    "flux2-klein-9b-components-v1";

void append_public_artifacts(
        const std::filesystem::path &root, std::string_view prefix,
        std::vector<streaming::SourceFileIdentity> &files) {
    std::error_code error;
    if (!std::filesystem::is_directory(root, error)) return;
    std::vector<std::filesystem::path> paths;
    std::filesystem::directory_iterator iterator(
        root, std::filesystem::directory_options::skip_permission_denied,
        error);
    require(!error, "streaming_source_identity: cannot enumerate FLUX artifacts");
    const std::filesystem::directory_iterator end;
    for (; iterator != end; iterator.increment(error)) {
        require(!error,
                "streaming_source_identity: cannot enumerate FLUX artifacts");
        std::error_code status_error;
        const auto &entry = *iterator;
        const bool symlink = entry.is_symlink(status_error);
        require(!status_error,
                "streaming_source_identity: cannot inspect FLUX artifact");
        const bool regular = entry.is_regular_file(status_error);
        require(!status_error,
                "streaming_source_identity: cannot inspect FLUX artifact");
        if (!symlink && regular &&
            entry.path().extension() == ".safetensors")
            paths.push_back(entry.path());
    }
    require(!error, "streaming_source_identity: cannot enumerate FLUX artifacts");
    std::sort(paths.begin(), paths.end());
    for (const auto &path : paths) {
        streaming::SourceFileIdentity file;
        const auto relative = std::filesystem::relative(path, root, error)
                                  .generic_string();
        file.logical_id = prefix.empty() ? relative :
            std::string(prefix) + "/" + relative;
        require(!error && !file.logical_id.empty(),
                "streaming_source_identity: invalid FLUX artifact path");
        file.path = path;
        files.push_back(std::move(file));
    }
}

streaming::PresetSourceIdentity flux_public_source_identity(
        const streaming::SourceLease &lease) {
    streaming::CanonicalEncoder encoder("flux2-public-source-v1");
    encoder.string_field("lease_digest", lease.digest());
    encoder.unsigned_field("artifact_count", lease.file_count());
    return {"flux2-klein-9b-bf16", "diffusers-bf16-sharded", encoder.sha256(),
            std::string(lease.digest())};
}

streaming::PresetRuntimeIdentity flux_public_runtime_identity() {
    return {"turbocider-streaming-2026-09-18", "public-streaming-runtime-v2",
            "flux2-klein-9b-public-adapter-v1",
            "flux2-sharded-pread-bf16-lease-v1", kFluxExactKernelRevision,
            "mlx-request-cache-policy-v1"};
}

} // namespace

Flux::Flux(const std::filesystem::path &root, std::string model_id)
    : root_(root), model_id_(std::move(model_id)), tokenizer_(root / "tokenizer") {
    auto configuration = flux_configuration(root, model_id_);
    heads_ = configuration.heads; hidden_ = configuration.hidden;
    dual_layers_ = configuration.dual_layers; single_layers_ = configuration.single_layers;
}
Flux::~Flux() = default;

std::shared_ptr<const streaming::ModelStreamingProbe>
Flux::probe_public_streaming(
        const streaming::PublicResolveInput &input) const {
    const auto &request = input.request;
    require(model_id_ == "flux2-klein-9b" && request.model == model_id_,
            "streaming_engine_model_mismatch");
    require(request.operation == "image.generate" && request.inputs.empty() &&
                request.frames == 1 && !request.audio &&
                request.execution == "gpu" && request.ane_manifest.empty() &&
                request.encoder_ane_manifest.empty() &&
                !request.allow_approximation && !request.compile_gpu &&
                request.loras.empty(),
            "streaming_route_unsupported: FLUX public card requires BF16 eager GPU text-to-image without LoRA/ANE/compiled graph");
    require(request.width >= 16 && request.height >= 16 &&
                request.width % 16 == 0 && request.height % 16 == 0,
            "streaming_workload_invalid: FLUX dimensions must be multiples of 16");
    std::vector<streaming::SourceFileIdentity> files;
    auto add = [&](std::string logical, std::filesystem::path path) {
        streaming::SourceFileIdentity file;
        file.logical_id = std::move(logical);
        file.path = std::move(path);
        files.push_back(std::move(file));
    };
    const auto transformer_root = root_ / "transformer";
    add("config.json", transformer_root / "config.json");
    add("diffusion_pytorch_model.safetensors.index.json",
        transformer_root / "diffusion_pytorch_model.safetensors.index.json");
    std::error_code error;
    const auto index = files.back().path;
    require(std::filesystem::is_regular_file(index, error) && !error,
            "streaming_route_unsupported: FLUX index is missing");
    require(std::filesystem::is_directory(transformer_root, error) && !error,
            "streaming_route_unsupported: FLUX transformer directory is missing");
    // Capture every transformer shard before constructing metadata.  The
    // lease-backed metadata parser then treats the index as authoritative and
    // rejects missing, duplicate, or unexpected shard identities.
    append_public_artifacts(transformer_root, "", files);
    require(files.size() >= 4,
            "streaming_route_unsupported: FLUX indexed shards are missing");
    add("text_encoder/config.json", root_ / "text_encoder/config.json");
    append_public_artifacts(root_ / "text_encoder", "text_encoder", files);
    add("vae/config.json", root_ / "vae/config.json");
    append_public_artifacts(root_ / "vae", "vae", files);
    add("tokenizer/tokenizer.json", root_ / "tokenizer/tokenizer.json");
    auto lease = streaming::SourceLease::capture(std::move(files));
    flux2::StreamingMetadata metadata(lease, model_id_);
    metadata.check_unchanged();

    const auto tokens = tokenizer_.prompt(request.prompt, request.dynamic_text);
    streaming::PresetWorkload workload;
    workload.model = model_id_;
    workload.operation = request.operation;
    workload.execution = request.execution;
    workload.device_class = input.device.device_class;
    workload.execution_container = input.execution_container;
    workload.width = static_cast<uint32_t>(request.width);
    workload.height = static_cast<uint32_t>(request.height);
    workload.frames = 1;
    workload.steps = static_cast<uint32_t>(request.steps);
    workload.batch = 1;
    workload.audio = false;
    workload.dynamic_text = request.dynamic_text;
    workload.approximation = false;
    workload.conditioning_revision = "qwen3-flux2-klein-v1";
    workload.vae_policy_revision = "flux2-vae-v1";
    workload.feature_digest = std::string("reference_tokens:0");
    workload.token_shapes.push_back({
        "qwen3", "qwen3-flux2-v1", "flux2-template-v1",
        static_cast<uint32_t>(tokens.valid),
        static_cast<uint32_t>(tokens.ids.size()),
        static_cast<uint32_t>(tokens.ids.size())});
    return std::make_shared<streaming::ValueModelStreamingProbe>(
        streaming::ValueModelStreamingProbe::Values{
            model_id_, flux_public_source_identity(*lease),
            std::move(workload), flux_public_runtime_identity(),
            kFluxPublicComponentPolicy, std::move(lease)});
}

std::shared_ptr<const streaming::ModelStreamingSnapshot>
Flux::compile_public_streaming(
        std::shared_ptr<const streaming::ModelStreamingProbe> probe,
        const streaming::StreamingPresetRecord &record) const {
    auto value_probe = std::dynamic_pointer_cast<
        const streaming::ValueModelStreamingProbe>(probe);
    require(value_probe != nullptr, "streaming_public_probe_type_mismatch");
    require(value_probe->model_id() == model_id_ &&
                value_probe->component_policy_revision() ==
                    record.plan.component_policy_revision &&
                record.source == value_probe->source_identity() &&
                record.workload == value_probe->workload_identity() &&
                record.runtime == value_probe->runtime_identity(),
            "streaming_record_identity_mismatch");
    const auto &workload = value_probe->workload_identity();
    require(workload.token_shapes.size() == 1,
            "streaming_workload_invalid: FLUX token shape count");
    flux2::StreamingWorkload descriptor_workload{
        workload.width, workload.height,
        workload.token_shapes.front().padded_rows, 0, workload.steps};
    auto plan = std::make_shared<flux2::StreamingPlanView>(
        value_probe->lease_ptr(), model_id_, record.plan.canonical_config,
        descriptor_workload);
    require(plan->layout().digest == record.plan.layout_digest,
            "streaming_layout_digest_mismatch");
    return std::make_shared<streaming::ValueModelStreamingSnapshot>(
        streaming::ValueModelStreamingSnapshot::Values{
            model_id_, value_probe->source_identity(),
            value_probe->runtime_identity(), plan->descriptor(),
            plan->layout(), std::string(value_probe->component_policy_revision()),
            value_probe->lease_ptr()});
}

RunResult Flux::generate_resolved(
        std::shared_ptr<const streaming::ResolvedRequestExecution> execution,
        const Event &event, std::atomic<bool> &cancelled) {
    require(execution && execution->probe && execution->model_snapshot,
            "streaming_authority_mismatch");
    require(execution->request.streaming.active(),
            "streaming_actual_plan_mismatch");
    auto value_probe = std::dynamic_pointer_cast<
        const streaming::ValueModelStreamingProbe>(execution->probe);
    require(value_probe != nullptr, "streaming_public_probe_type_mismatch");
    auto lease = value_probe->lease_ptr();
    require(lease && execution->probe->source_lease() == lease.get() &&
                execution->model_snapshot->source_lease() == lease.get(),
            "streaming_source_lease_mismatch");
    require(execution->model_snapshot->model_id() == model_id_ &&
                execution->selection.record.source ==
                    execution->model_snapshot->source_identity() &&
                execution->selection.record.runtime ==
                    execution->model_snapshot->runtime_identity() &&
                execution->selection.record.plan.layout_digest ==
                    execution->model_snapshot->layout().digest &&
                execution->selection.record.plan.component_policy_revision ==
                    execution->model_snapshot->component_policy_revision(),
            "streaming_authority_mismatch");
    const auto target = execution->selection.exact_selector
                            .target_request_memory_bytes;
    require(target && streaming::supported_streaming_target(*target),
            "streaming_target_unsupported");
    require(!public_stream_lease_, "streaming_public_request_reentrant");
    public_stream_lease_ = std::move(lease);
    const auto previous_target = public_stream_target_bytes_;
    public_stream_target_bytes_ = *target;
    try {
        auto result = run(execution->request, event, cancelled, false);
        public_stream_target_bytes_ = previous_target;
        public_stream_lease_.reset();
        return result;
    } catch (...) {
        public_stream_target_bytes_ = previous_target;
        public_stream_lease_.reset();
        throw;
    }
}
void Flux::select_loras(const Request &request) {
    std::string identity;
    std::vector<LoRAAsset> normalized;
    normalized.reserve(request.loras.size());
    for (const auto &adapter : request.loras) {
        std::error_code error;
        auto canonical = std::filesystem::canonical(adapter.path, error);
        require(!error && std::filesystem::is_regular_file(canonical),
                "LoRA file missing: " + adapter.path);
        auto bytes = std::filesystem::file_size(canonical, error);
        require(!error, "cannot inspect LoRA size: " + canonical.string());
        auto mtime = std::filesystem::last_write_time(canonical, error);
        require(!error, "cannot inspect LoRA timestamp: " + canonical.string());
        auto key = canonical.string();
        auto found = lora_hash_cache_.find(key);
        if (found == lora_hash_cache_.end() || found->second.bytes != bytes ||
            found->second.mtime != mtime) {
            auto digest = sha256_file(canonical);
            auto final_bytes = std::filesystem::file_size(canonical, error);
            require(!error && final_bytes == bytes,
                    "LoRA changed while it was being hashed: " + canonical.string());
            auto final_mtime = std::filesystem::last_write_time(canonical, error);
            require(!error && final_mtime == mtime,
                    "LoRA changed while it was being hashed: " + canonical.string());
            lora_hash_cache_[key] = {bytes, mtime, std::move(digest)};
            found = lora_hash_cache_.find(key);
        }
        auto selected = adapter;
        selected.path = key;
        normalized.push_back(std::move(selected));
        identity += adapter.role + ":" + key + ":" + std::to_string(bytes) + ":" +
                    std::to_string(static_cast<long long>(mtime.time_since_epoch().count())) + ":" +
                    found->second.sha256 + ":" +
                    std::to_string(std::bit_cast<uint32_t>(adapter.strength)) + ";";
    }
    if (identity == cached_lora_identity_) return;
    cached_lora_identity_ = std::move(identity);
    active_loras_ = std::move(normalized);
    exact_stream_.reset();
    hybrid_.reset(); encoder_hybrid_.reset(); cached_conditioning_.reset(); cached_prompt_.clear();
    cached_encoder_manifest_.clear();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
    transformer_.clear(); vae_.clear(); mx::clear_cache();
}
LoadResult Flux::load(const Event &event, std::atomic<bool> &cancelled) {
    // Load image weights only: Qwen is intentionally staged during prompt encoding.
    require(device_info().physical_memory >= (16ull << 30),
            "insufficient memory for BF16 image weights");
    bool transformer_cold = transformer_.bytes() == 0;
    transformer_.load(root_ / "transformer", event, cancelled);
    if (transformer_cold && !active_loras_.empty())
        transformer_.apply_loras(active_loras_, "transformer", event, cancelled);
    vae_.load(root_ / "vae", event, cancelled);
    event("prepare_image_weights", 0, 2);
    transformer_.materialize();
    event("prepare_image_weights", 1, 2);
    vae_.materialize();
    event("prepare_image_weights", 2, 2);
    checkpoint(cancelled);
    return {transformer_.bytes() + vae_.bytes(), mx::get_active_memory()};
}
void Flux::unload() {
    exact_stream_.reset();
    hybrid_.reset();
    encoder_hybrid_.reset();
    hybrid_gpu_graph_ = {};
    hybrid_gpu_mlp_start_ = -1;
    cached_conditioning_.reset();
    cached_prompt_.clear();
    cached_encoder_manifest_.clear();
    transformer_.clear();
    vae_.clear();
}
bool Flux::conditioning(const Request &r, const Tokens &tokens, const Event &event,
                        std::atomic<bool> &cancelled) {
    bool hit = cached_conditioning_.has_value() && cached_prompt_ == r.prompt &&
               cached_dynamic_ == r.dynamic_text &&
               cached_encoder_manifest_ == r.encoder_ane_manifest;
    if (hit) {
        event("text_cache_hit", 1, 1);
        return true;
    }
    // Keep preloaded image weights on machines with enough conservative headroom.
    // Smaller devices and explicit low budgets retain the staged text policy.
    bool retain =
        ResidencyPolicy::for_request(r, device_info().physical_memory).retain_images_during_text;
    if (!retain) {
        hybrid_.reset();
        transformer_.clear();
        vae_.clear();
    }
    cached_conditioning_.reset();
    mx::clear_cache();
    if (!r.encoder_ane_manifest.empty()) {
        const auto prefill = components::qwen3_prefill_plan(
            r.encoder_ane_manifest, int(tokens.ids.size()));
        for (const auto &lora : active_loras_)
            require(lora.role != "text_encoder",
                    "Qwen3 encoder hybrid does not yet support text-encoder LoRA");
        if (prefill.use_hybrid) {
            if (!encoder_hybrid_ || encoder_hybrid_->manifest != r.encoder_ane_manifest)
                encoder_hybrid_ = std::make_unique<HybridSession>(
                    r.encoder_ane_manifest, root_ / "text_encoder",
                    int(tokens.ids.size()), event, cancelled, r.warmup_iterations,
                    components::qwen3_checkpoint_path(root_ / "text_encoder"),
                    std::vector<LoRAAsset>{}, 0, 27, true);
            encoder_hybrid_->set_tokens(int(tokens.ids.size()));
            require(encoder_hybrid_->rows == prefill.compute_tokens,
                    "Qwen3 encoder manifest bucket changed during prefill setup");
        } else {
            encoder_hybrid_.reset();
            event("qwen3_encoder_gpu_" + prefill.reason, 1, 1);
        }
    } else {
        encoder_hybrid_.reset();
    }
    auto encoded = encode(tokens, event, cancelled, encoder_hybrid_.get());
    checkpoint(cancelled);
    cached_conditioning_ = encoded;
    cached_prompt_ = r.prompt;
    cached_dynamic_ = r.dynamic_text;
    cached_encoder_manifest_ = r.encoder_ane_manifest;
    mx::clear_cache();
    return false;
}
std::string Flux::select_acceleration(Request &r, int count, const Event &event,
                                      std::atomic<bool> &cancelled) {
    const bool automatic = r.execution == "auto";
    const AccelerationCase *matched = nullptr;
    if (automatic) {
        r.execution = "gpu";
        if (!r.allow_approximation || r.ane_manifest.empty()) {
            hybrid_.reset();
            return "gpu: no opted-in compatible local partition";
        }
        auto system = device_info();
        // Automatic selection is limited to the exact device profile on which
        // this partition policy was measured. M4 Max uses the 6,144-channel
        // prefix artifact; the older M4 Pro profile remains valid separately.
        const bool m4_pro_profile =
            system.gpu == "Apple M4 Pro" && system.physical_memory == (48ull << 30);
        const bool m4_max_profile =
            system.gpu == "Apple M4 Max" && system.physical_memory == (64ull << 30);
        if (!m4_pro_profile && !m4_max_profile) {
            hybrid_.reset();
            return "gpu: automatic hybrid policy not validated on this hardware";
        }
        matched = hybrid_case(r, count, system.gpu, system.physical_memory);
        if (!matched) {
            hybrid_.reset();
            return "gpu: no measured hybrid case for operation, dimensions, steps and token bucket";
        }
        uint64_t estimate = (16ull << 30) + uint64_t(r.width) * r.height * 8192;
        if (device_info().physical_memory < estimate + (4ull << 30) ||
            (r.memory_budget_bytes && r.memory_budget_bytes < estimate)) {
            hybrid_.reset();
            return "gpu: hybrid memory budget unavailable";
        }
        r.execution = "gpu_ane";
    }
    if (r.execution != "gpu_ane") {
        hybrid_.reset();
        return "gpu: explicitly selected";
    }
    try {
        checkpoint(cancelled);
        if (automatic && hybrid_ && hybrid_->rows != matched->bucket)
            hybrid_.reset();
        if (!hybrid_ || hybrid_->manifest != r.ane_manifest)
            hybrid_ = std::make_unique<HybridSession>(r.ane_manifest, root_, count, event,
                                                      cancelled, r.warmup_iterations,
                                                      std::filesystem::path{},
                                                      active_loras_,
                                                      matched ? matched->bucket : 0);
        require(count <= hybrid_->rows, "Core ML token bucket cannot serve this request");
        if (automatic) {
            auto system = device_info();
            if (system.gpu == "Apple M4 Max")
                require(hybrid_->mlp_width == 9216 && hybrid_->ane_mlp_start == 0 &&
                            hybrid_->ane_mlp_end == 6144,
                        "M4 Max automatic profile requires the validated 6144-channel ANE prefix");
            else if (system.gpu == "Apple M4 Pro")
                require(hybrid_->mlp_width == 9216 && hybrid_->ane_mlp_start == 0 &&
                            hybrid_->ane_mlp_end == 9216,
                        "M4 Pro automatic profile requires the validated full ANE MLP partition");
        }
        return automatic ? std::string("gpu_ane: measured case ") + matched->id
                         : "gpu_ane: explicitly selected";
    } catch (const Cancelled &) {
        throw;
    } catch (const std::exception &error) {
        if (!automatic)
            throw;
        hybrid_.reset();
        mx::clear_cache();
        r.execution = "gpu";
        event("acceleration_gpu_fallback", 1, 1);
        return std::string("gpu: ") + error.what();
    }
}
RunResult Flux::prepare(const Request &requested, bool warmup, const Event &event,
                        std::atomic<bool> &cancelled) {
    if (warmup)
        return run(requested, event, cancelled, true);
    require(!flux_exact_streaming_requested(requested),
            "streaming_route_unsupported: FLUX exact streaming does not "
            "support prepare-only requests");
    auto r = requested;
    auto begin = Clock::now();
    auto plan = make_plan(r);
    require(r.model == model_id_ && !r.prompt.empty(), "FLUX preparation requires a prompt");
    ResidencyPolicy::validate_budget(plan, device_info().physical_memory);
    mx::set_cache_limit(r.allocator_cache_bytes);
    select_loras(r);
    auto tokens = tokenizer_.prompt(r.prompt, r.dynamic_text);
    bool hit = conditioning(r, tokens, event, cancelled);
    load(event, cancelled);
    int count = int(tokens.ids.size()) + (r.width / 16) * (r.height / 16);
    for (auto &input : r.inputs)
        if (r.operation == "image.edit") {
            auto image = load_image_tensor(input.path, r.width, r.height, true);
            count += (image.shape(1) / 16) * (image.shape(2) / 16);
        }
    auto selection = select_acceleration(r, count, event, cancelled);
    if (r.execution == "gpu" && model_id_ == "flux2-klein-4b" &&
        !std::getenv("TURBOCIDER_FLUX_EAGER_BLOCKS"))
        r.compile_gpu = true;
    plan = make_plan(r);
    event(r.execution == "gpu_ane" ? "route_gpu_ane" : "route_gpu", 1, 1);
    RunResult result;
    result.prepared = true;
    result.selection = selection;
    result.prompt_cache_hit = hit;
    result.request = r;
    // `select_acceleration` may resolve `auto` to a concrete GPU/ANE route
    // and the second `make_plan` above captures the resulting compile and
    // execution flags.  Preserve that resolved plan in the preparation
    // result so load-only telemetry describes the same route as generation.
    result.plan = std::move(plan);
    result.text_tokens = int(tokens.ids.size());
    result.total_tokens = count;
    result.timings.wall = std::chrono::duration<double>(Clock::now() - begin).count();
    result.active_bytes = mx::get_active_memory();
    if (hybrid_)
        result.hybrid = hybrid_->metrics();
    if (encoder_hybrid_)
        result.encoder_hybrid = encoder_hybrid_->metrics();
    return result;
}
RunResult Flux::generate(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
    return run(r, event, cancelled, false);
}
RunResult Flux::run(const Request &requested, const Event &event, std::atomic<bool> &cancelled,
                    bool warmup) {
    auto r = requested;
    const bool exact_streaming = flux_exact_streaming_requested(r);
    const bool public_streaming = public_stream_lease_ != nullptr;
    require(!public_streaming ||
                (exact_streaming && public_stream_target_bytes_ != 0),
            "streaming_public_binding_mismatch");
    auto begin = Clock::now();
    auto plan = make_plan(r);
    require(r.model == model_id_, "model executor unavailable; see static acceptance plan");
    require(!r.prompt.empty() && (warmup || !r.output.empty()), "prompt and output are required");
    require(warmup || std::filesystem::path(r.output).extension() == ".png",
            "native image output must be .png");
    auto physical = device_info().physical_memory;
    if (!exact_streaming)
        ResidencyPolicy::validate_budget(plan, physical);
    auto residency = ResidencyPolicy::for_request(r, physical);
    mx::reset_peak_memory();
    mx::set_cache_limit(r.allocator_cache_bytes);
    select_loras(r);
    if (exact_streaming) {
        require(model_id_ == "flux2-klein-9b" && active_loras_.empty(),
                "streaming_route_unsupported: FLUX exact streaming supports "
                "Klein 9B BF16 without LoRA");
        // Exact retention is request-scoped. Never let a preceding resident
        // request or failed exact executor coexist with the compiled slots.
        mx::synchronize();
        exact_stream_.reset();
        // Request-boundary release before constructing slot backing. These
        // calls are intentionally outside the block/class/pass steady path.
        streaming::audit_increment(
            streaming::AuditCounter::CacheClearOrUnloadCalls, 2);
        transformer_.clear();
        hybrid_.reset();
        hybrid_gpu_graph_ = {};
        hybrid_gpu_mlp_start_ = -1;
        if (public_streaming) {
            // A public request cannot reuse conditioning or VAE arrays created
            // from an earlier path-based source generation. Rebuild every
            // source-dependent component through the request-scoped lease.
            encoder_hybrid_.reset();
            cached_conditioning_.reset();
            cached_prompt_.clear();
            cached_encoder_manifest_.clear();
            vae_.clear();
        }
        mx::clear_cache();
    }
    auto tokens = tokenizer_.prompt(r.prompt, r.dynamic_text);
    if (r.execution != "gpu_ane" && r.execution != "auto")
        hybrid_.reset();
    auto dump = [&](const std::string &name, const Tensor &a) {
        if (!r.dump.empty()) {
            std::filesystem::create_directories(r.dump);
            mx::save_safetensors((std::filesystem::path(r.dump) / (name + ".safetensors")).string(),
                                 {{"tensor", a}});
        }
    };
    auto text_start = Clock::now();
    bool prompt_hit = conditioning(r, tokens, event, cancelled);
    double text_s = std::chrono::duration<double>(Clock::now() - text_start).count();
    std::optional<Tensor> reference_latents, clean_latents;
    std::vector<float> reference_ids;
    auto image_start = Clock::now();
    if (!r.inputs.empty()) {
        vae_.load(root_ / "vae", event, cancelled);
        for (size_t index = 0; index < r.inputs.size(); ++index) {
            checkpoint(cancelled);
            auto image = load_image_tensor(r.inputs[index].path, r.width, r.height,
                                           r.operation == "image.edit");
            dump("input_image_" + std::to_string(index), image);
            auto encoded = encode_image(image, event, cancelled);
            dump("image_latent_" + std::to_string(index), encoded);
            if (r.operation == "image.transform")
                clean_latents = encoded;
            else {
                reference_latents =
                    reference_latents ? mx::concatenate({*reference_latents, encoded}, 1) : encoded;
                int h = image.shape(1) / 16, w = image.shape(2) / 16;
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        reference_ids.push_back(float(10 + 10 * index));
                        reference_ids.push_back(float(y));
                        reference_ids.push_back(float(x));
                        reference_ids.push_back(0);
                    }
            }
        }
    }
    if (residency.release_before_denoise) {
        vae_.clear();
        mx::clear_cache();
    }
    double image_s = std::chrono::duration<double>(Clock::now() - image_start).count();
    int actual_tokens = int(tokens.ids.size()) + (r.width / 16) * (r.height / 16) +
                        (reference_latents ? reference_latents->shape(1) : 0);
    require(actual_tokens <= 20000, "request exceeds native token workspace budget");
    auto hybrid_start = Clock::now();
    auto selection = select_acceleration(r, actual_tokens, event, cancelled);
    if (r.execution == "gpu" && model_id_ == "flux2-klein-4b" &&
        !std::getenv("TURBOCIDER_FLUX_EAGER_BLOCKS"))
        r.compile_gpu = true;
    event(r.execution == "gpu_ane" ? "route_gpu_ane" : "route_gpu", 1, 1);
    plan = make_plan(r);
    double hybrid_s = std::chrono::duration<double>(Clock::now() - hybrid_start).count();
    auto text = *cached_conditioning_;
    dump("conditioning", text);
    checkpoint(cancelled);
    if (!exact_streaming) {
        bool transformer_cold = transformer_.bytes() == 0;
        transformer_.load(root_ / "transformer", event, cancelled);
        if (transformer_cold && !active_loras_.empty())
            transformer_.apply_loras(active_loras_, "transformer", event,
                                     cancelled);
    }
    auto z = mx::astype(mx::random::normal({1, 128, r.height / 16, r.width / 16}, mx::float32, 0, 1,
                                           mx::random::key(r.seed)),
                        mx::bfloat16);
    z = mx::transpose(mx::reshape(z, {1, 128, (r.height / 16) * (r.width / 16)}), {0, 2, 1});
    mx::eval(z);
    dump("initial_latent", z);
    auto sigmas = flux_gpu_sigmas(z.shape(1), r.steps);
    int start_step = 0;
    if (clean_latents && r.inputs[0].strength > 0) {
        start_step = std::max(1, int(r.steps * r.inputs[0].strength));
        auto sigma = Tensor(sigmas[start_step]);
        z = (Tensor(1.f) - sigma) * (*clean_latents) + sigma * z;
        mx::eval(z);
        dump("conditioned_initial_latent", z);
    }
    if (exact_streaming) {
        require(r.execution == "gpu" && !hybrid_ && !r.compile_gpu &&
                    start_step == 0,
                "streaming_route_unsupported: FLUX exact streaming requires "
                "the eager GPU route and a full denoise schedule");
        require(exact_stream_generation_ != UINT64_MAX,
                "FLUX exact request generation overflow");
        ++exact_stream_generation_;
        const flux2::StreamingWorkload workload{
            uint32_t(r.width), uint32_t(r.height),
            uint32_t(text.shape(1)),
            uint32_t(reference_latents ? reference_latents->shape(1) : 0),
            uint32_t(r.steps - start_step)};
        if (public_stream_lease_) {
            exact_stream_ = std::make_unique<FluxExactStream>(
                public_stream_lease_, model_id_, r.streaming, workload,
                transformer_, event, cancelled, exact_stream_generation_);
            exact_stream_->enable_receipt({
                exact_stream_->plan().layout().digest,
                exact_stream_->implementation(),
                public_stream_lease_->generation()});
        } else {
            exact_stream_ = std::make_unique<FluxExactStream>(
                root_ / "transformer", model_id_, r.streaming, workload,
                transformer_, event, cancelled, exact_stream_generation_);
        }
    }
    auto dit_start = Clock::now();
    for (int i = start_step; i < r.steps; ++i) {
        checkpoint(cancelled);
        event("denoise", i, r.steps);
        auto model_input = reference_latents ? mx::concatenate({z, *reference_latents}, 1) : z;
        auto noise = denoise(model_input, text, sigmas[i], r.height, r.width, event, cancelled,
                             reference_ids, r.compile_gpu,
                             exact_stream_.get(), uint32_t(i - start_step));
        if (reference_latents)
            noise = slice_axis(noise, 1, 0, z.shape(1));
        mx::eval(noise);
        dump("noise_" + std::to_string(i), noise);
        z = euler_step(z, noise, sigmas[i + 1] - sigmas[i]);
        mx::eval(z);
        dump("latent_" + std::to_string(i), z);
        require(mx::all(mx::isfinite(z)).item<bool>(), "nonfinite latent");
        event("denoise", i + 1, r.steps);
    }
    double dit_s = std::chrono::duration<double>(Clock::now() - dit_start).count();
    std::optional<BlockResidencyMetrics> exact_residency;
    std::optional<StreamingRuntimeMetrics> exact_runtime;
    std::shared_ptr<const streaming::ActualExecutionReceipt> exact_receipt;
    if (exact_streaming) {
        exact_stream_->finish();
        const auto counters = exact_stream_->counters();
        const auto pager = exact_stream_->pager_metrics();
        const auto &compiled = exact_stream_->plan().layout();
        const auto &stage = compiled.stages.front();
        require(counters.pool_creates == 2 && counters.slot_bundles == 4 &&
                    counters.fills == uint64_t(stage.groups.size()) *
                        uint64_t(r.steps - start_step) &&
                    counters.groups_submitted == counters.fills &&
                    pager.slot_fills == counters.fills,
                "FLUX exact counters differ from the retained two-class layout");

        BlockResidencyMetrics residency_metrics;
        residency_metrics.enabled = true;
        residency_metrics.active_blocks = uint32_t(stage.groups.size());
        residency_metrics.streamed_blocks = uint32_t(stage.groups.size());
        residency_metrics.refill_slots = uint32_t(counters.slot_bundles);
        residency_metrics.memory_budget_bytes = public_streaming ?
            public_stream_target_bytes_ : r.memory_budget_bytes;
        residency_metrics.block_bytes = std::max(
            exact_stream_->plan().metadata().dual_block_bytes(),
            exact_stream_->plan().metadata().single_block_bytes());
        residency_metrics.estimated_working_set_bytes =
            stage.resident_bytes + stage.peak_pool_bytes;
        residency_metrics.request_bytes_loaded =
            pager.resident_bytes_loaded + pager.streamed_bytes_loaded;
        residency_metrics.request_slot_allocations = counters.slot_bundles;
        residency_metrics.request_slot_refills = counters.fills;
        residency_metrics.request_slot_fills = counters.fills;
        residency_metrics.request_load_seconds =
            pager.resident_load_seconds + pager.streamed_load_seconds;
        residency_metrics.request_wait_seconds = counters.wait_seconds;
        residency_metrics.request_refill_load_seconds =
            pager.streamed_load_seconds;
        residency_metrics.request_max_refill_seconds =
            pager.maximum_fill_seconds;
        residency_metrics.request_max_refill_block =
            int(pager.maximum_fill_group);
        exact_residency = std::move(residency_metrics);

        StreamingRuntimeMetrics runtime;
        runtime.implementation = exact_stream_->implementation();
        runtime.layout_digest = compiled.digest;
        runtime.resident_prefix_blocks = stage.prefix;
        runtime.block_group_size = stage.group_size;
        runtime.slot_count = stage.slot_count;
        runtime.prefetch_distance = stage.distance;
        runtime.io_workers = stage.workers;
        runtime.group_count = uint32_t(stage.groups.size());
        runtime.pass_count = stage.pass_count;
        runtime.startup_policy = "prefetch_window_before_prefix";
        runtime.pass_transition = "reload";
        runtime.retention = public_streaming ?
            "request" : "request;multi_pool=retain_all";
        runtime.reader_revision = 1;
        runtime.weight_format = "diffusers-bf16-sharded";
        runtime.kernel_revision = kFluxExactKernelRevision;
        runtime.conditioning_recipe = "qwen3-flux2-klein-v1";
        runtime.upsample_boundary = "no-upsample;release-before-vae";
        runtime.component_policy_revision = public_stream_lease_ ?
            kFluxPublicComponentPolicy : "flux2-private-components-v1";
        runtime.multi_pool_policy = "retain_all";
        runtime.pool_count = static_cast<uint32_t>(stage.pools.size());
        runtime.slot_bundle_count = counters.slot_bundles;
        runtime.refill_worker_count = stage.workers;
        runtime.drained = true;
        if (public_stream_lease_) {
            const auto stage_receipt = exact_stream_->receipt();
            require(stage_receipt != nullptr,
                    "streaming_actual_receipt_missing");
            exact_receipt = std::make_shared<
                const streaming::ActualExecutionReceipt>(
                    streaming::make_actual_execution_receipt(
                        exact_stream_->implementation(), compiled.digest,
                        kFluxPublicComponentPolicy,
                        std::vector<streaming::ActualStageReceipt>{
                            *stage_receipt}));
        }
        exact_runtime = std::move(runtime);

        // The executor has drained and destroyed both slot pools. Resident
        // fixed tensors are request-scoped as well and must be released before
        // loading the VAE.
        streaming::audit_increment(
            streaming::AuditCounter::CacheClearOrUnloadCalls, 2);
        transformer_.clear();
        exact_stream_.reset();
        mx::clear_cache();
    }
    std::optional<HybridMetrics> hybrid_metrics;
    if (hybrid_)
        hybrid_metrics = hybrid_->metrics();
    checkpoint(cancelled);
    if (!exact_streaming && residency.release_after_denoise) {
        transformer_.clear();
        hybrid_.reset();
        mx::clear_cache();
    }
    if (public_stream_lease_) {
        std::vector<std::string> artifacts;
        for (const auto &file : public_stream_lease_->descriptor().files)
            if (file.logical_id.starts_with("vae/") &&
                file.logical_id.ends_with(".safetensors"))
                artifacts.push_back(file.logical_id);
        std::sort(artifacts.begin(), artifacts.end());
        vae_.load_lease(public_stream_lease_, artifacts, event, cancelled);
    } else {
        vae_.load(root_ / "vae", event, cancelled);
    }
    auto decode_start = Clock::now();
    auto pixels = decode(z, r.height, r.width, event, cancelled, r.dump);
    dump("pixels_nhwc", pixels);
    require(mx::all(mx::isfinite(pixels)).item<bool>(), "nonfinite decoded pixels");
    double decode_s = std::chrono::duration<double>(Clock::now() - decode_start).count();
    checkpoint(cancelled);
    if (!warmup) {
        event("export", 0, 1);
        checkpoint(cancelled);
        save_png(pixels, r.output);
        event("export", 1, 1);
    } else
        event("warmup_complete", 1, 1);

    if (residency.release_after_decode) {
        vae_.clear();
        mx::clear_cache();
    }
    double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    RunResult result;
    result.selection = selection;
    result.warmup = warmup;
    result.request = r;
    result.plan = std::move(plan);
    result.reference_tokens = reference_latents ? reference_latents->shape(1) : 0;
    result.actual_steps = r.steps - start_step;
    result.text_tokens = int(tokens.ids.size());
    result.valid_text_tokens = tokens.valid;
    result.prompt_cache_hit = prompt_hit;
    result.timings = {seconds, text_s, image_s, hybrid_s, dit_s, decode_s};
    result.peak_bytes = mx::get_peak_memory();
    result.active_bytes = mx::get_active_memory();
    result.hybrid = hybrid_metrics;
    result.block_residency = std::move(exact_residency);
    result.streaming_runtime = std::move(exact_runtime);
    result.streaming_receipt = std::move(exact_receipt);
    if (encoder_hybrid_)
        result.encoder_hybrid = encoder_hybrid_->metrics();
    return result;
}
} // namespace tc
