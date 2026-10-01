#include "qwen3_gguf.hpp"
#include "../../runtime/streaming/context.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>

namespace tc::components {
namespace {
uint64_t aligned(uint64_t bytes) {
    constexpr uint64_t a = streaming::GgufWeightPager::buffer_alignment;
    return gguf::checked_add(bytes, a - 1) & ~(a - 1);
}
using Shape = std::vector<uint64_t>;
struct TensorSpec { Shape shape; std::string target; };
std::map<std::string, TensorSpec> layer_specs(const Qwen3GgufConfig &c) {
    return {{"attn_norm.weight", {{c.hidden}, "input_layernorm.weight"}},
        {"attn_q.weight", {{c.heads * c.head_dim, c.hidden}, "self_attn.q_proj.weight"}},
        {"attn_k.weight", {{c.kv_heads * c.head_dim, c.hidden}, "self_attn.k_proj.weight"}},
        {"attn_v.weight", {{c.kv_heads * c.head_dim, c.hidden}, "self_attn.v_proj.weight"}},
        {"attn_q_norm.weight", {{c.head_dim}, "self_attn.q_norm.weight"}},
        {"attn_k_norm.weight", {{c.head_dim}, "self_attn.k_norm.weight"}},
        {"attn_output.weight", {{c.hidden, c.heads * c.head_dim}, "self_attn.o_proj.weight"}},
        {"ffn_norm.weight", {{c.hidden}, "post_attention_layernorm.weight"}},
        {"ffn_gate.weight", {{c.intermediate, c.hidden}, "mlp.gate_proj.weight"}},
        {"ffn_up.weight", {{c.intermediate, c.hidden}, "mlp.up_proj.weight"}},
        {"ffn_down.weight", {{c.hidden, c.intermediate}, "mlp.down_proj.weight"}}};
}
void check_metadata(const gguf::Directory &d, const Qwen3GgufConfig &c) {
    const auto *architecture = d.meta("general.architecture");
    require(architecture && architecture->type == 8 && architecture->text == "qwen3",
            "qe_adapter_mismatch: GGUF is not standard Qwen3");
    for (const auto &[key, expected] : std::map<std::string, uint32_t>{
        {"qwen3.block_count", c.layers}, {"qwen3.context_length", c.context_length},
        {"qwen3.embedding_length", c.hidden}, {"qwen3.feed_forward_length", c.intermediate},
        {"qwen3.attention.head_count", c.heads}, {"qwen3.attention.head_count_kv", c.kv_heads},
        {"qwen3.attention.key_length", c.head_dim}, {"qwen3.attention.value_length", c.head_dim},
        {"tokenizer.ggml.bos_token_id", 151643}, {"tokenizer.ggml.eos_token_id", 151645}}) {
        const auto *value = d.meta(key);
        require(value && (value->type == 0 || value->type == 2 || value->type == 4 || value->type == 10) &&
                value->unsigned_value == expected, "qe_adapter_mismatch: GGUF/config differs at " + key);
    }
    for (const auto &[key, expected] : std::map<std::string, float>{
        {"qwen3.rope.freq_base", c.rope_theta}, {"qwen3.attention.layer_norm_rms_epsilon", c.epsilon}}) {
        const auto *value = d.meta(key);
        require(value && (value->type == 6 || value->type == 12) && float(value->float_value) == expected,
                "qe_adapter_mismatch: GGUF/config differs at " + key);
    }
}
void validate_eval_policy() {
    require(!std::getenv("TURBOCIDER_QWEN3_DEFER_LAYER_EVAL"),
            "qe_config_conflict: bounded Qwen3 cannot defer layer eval");
    const char *interval = std::getenv("TURBOCIDER_QWEN3_EVAL_INTERVAL");
    require(!interval || std::string(interval) == "1",
            "qe_config_conflict: bounded Qwen3 requires each-layer eval");
}
}

Qwen3GgufPlan describe_qwen3_gguf(std::shared_ptr<const streaming::SourceLease> lease,
        const Qwen3GgufConfig &c, const Tokens &tokens, uint32_t prefetch, gguf::DecodeOptions options, const std::string &residency) {
    require(lease && lease->has_verified_content() && prefetch <= 2 &&
            c.hidden == 2560 && c.heads == 32 && c.kv_heads == 8 && c.head_dim == 128 &&
            c.layers >= 35 && c.layers <= 128 && c.intermediate && c.intermediate <= 32768 && c.vocabulary,
            "qe_adapter_mismatch: unsupported/unverified Qwen3 consumer geometry");
    require(residency == "packed_resident" || residency == "packed_streamed", "qe_config_conflict: unknown Qwen3 source residency");
    require(!tokens.ids.empty() && tokens.ids.size() <= 1024 && tokens.valid > 0 &&
            tokens.valid <= int(tokens.ids.size()), "qe_adapter_mismatch: invalid Qwen3 conditioning tokens");
    for (int id : tokens.ids) require(id >= 0 && uint32_t(id) < c.vocabulary, "qe_decode_invalid: token ID outside vocabulary");
    const auto &file = lease->file("text_encoder"); auto fd = lease->duplicate_fd("text_encoder");
    const auto directory = gguf::read_directory(fd.get(), file.bytes);
    check_metadata(directory, c);
    const auto specs = layer_specs(c);
    std::map<std::string, TensorSpec> expected;
    for (uint32_t i = 0; i < c.layers; ++i) for (const auto &[name, spec] : specs)
        expected.emplace("blk." + std::to_string(i) + "." + name, spec);
    expected.emplace("token_embd.weight", TensorSpec{{c.vocabulary, c.hidden}, "model.embed_tokens.weight"});
    expected.emplace("output_norm.weight", TensorSpec{{c.hidden}, "model.norm.weight"});
    if (std::any_of(directory.tensors.begin(), directory.tensors.end(), [](const auto &t) { return t.name == "output.weight"; }))
        expected.emplace("output.weight", TensorSpec{{c.vocabulary, c.hidden}, "lm_head.weight"});
    require(directory.tensors.size() == expected.size(), "qe_adapter_mismatch: Qwen3 tensor set incomplete/unexplained");
    Qwen3GgufPlan result;
    auto &d = result.descriptor;
    d.model = "qwen3-z-conditioning-gguf"; d.checkpoint_identity = std::string(lease->artifact_digest());
    d.backend_revision = "qwen3-z-gguf-cpu-bf16-rne-v1";
    d.artifacts.push_back({"text_encoder", file.content_digest, file.bytes, streaming::SourceIdentityKind::content_sha256});
    d.workload = {{"precision", "qwen3-z-source-mixed-v1"}, {"qk_layout", "hf-half-split-v1"},
        {"decode_backend", options.use_simd ? "cpu_simd" : "cpu_scalar"},
        {"source_residency", residency},
        {"hidden_tap", "block34-post-residual-no-final-norm"}, {"submit_policy", "each-layer-eval-v1"},
        {"token_rows", std::to_string(tokens.ids.size())}, {"valid_rows", std::to_string(tokens.valid)},
        {"tokenizer_sha256", lease->file("tokenizer").content_digest},
        {"config_sha256", lease->file("config").content_digest}};
    streaming::StageDescriptor stage;
    stage.id = "text_encoder"; stage.adapter_revision = d.backend_revision;
    stage.min_slots = 1; stage.max_slots = 3; stage.max_group_size = 1;
    stage.passes.push_back({0, "conditioning-prefill", {tokens.ids.size(), uint64_t(tokens.valid)}});
    for (uint32_t i = 0; i < 35; ++i) {
        streaming::BlockSpec block; block.id = i; block.layout_class = "qwen3-z-gguf-layer-v1";
        stage.blocks.push_back(std::move(block));
    }
    for (const auto &tensor : directory.tensors) {
        const auto wanted = expected.find(tensor.name);
        require(wanted != expected.end() && wanted->second.shape == tensor.logical_shape(),
                "qe_adapter_mismatch: Qwen3 GGUF name/shape differs: " + tensor.name);
        if (tensor.name == "output_norm.weight" || tensor.name == "output.weight" || tensor.name.starts_with("blk.35.")) continue;
        const bool embedding = tensor.name == "token_embd.weight";
        uint32_t layer = 0;
        if (!embedding) {
            const auto end = tensor.name.find('.', 4); const auto number = tensor.name.substr(4, end - 4);
            layer = uint32_t(std::stoul(number));
            require(number == std::to_string(layer), "qe_adapter_mismatch: noncanonical layer index");
            if (layer >= 35) continue;
        }
        const char *format = embedding ? "U8" : tensor.type == 0 ? "F32" : tensor.type == 1 ? "F16" : "BF16";
        streaming::Materialization m;
        m.format = format; m.storage_mode = "mlx-metal-shared";
        m.conversion = embedding ? "gguf-packed-gather-source-v1" : "gguf-cpu-rne-v1";
        m.shape = embedding ? Shape{tensor.bytes} : tensor.logical_shape();
        m.reads.push_back({0, tensor.file_offset, tensor.bytes, tensor.name, gguf::type_info(tensor.type).name, tensor.logical_shape()});
        streaming::FieldSpec field;
        field.name = embedding ? "embedding.packed" : wanted->second.target;
        field.storage_id = "qwen3-gguf:" + tensor.name;
        field.bytes = embedding ? tensor.bytes : gguf::checked_mul(tensor.elements, std::string_view(format) == "F32" ? 4 : 2);
        field.alignment = streaming::GgufWeightPager::buffer_alignment; field.materialization = std::move(m);
        if (residency == "packed_resident") result.packed_capacity = gguf::checked_add(result.packed_capacity, aligned(tensor.bytes));
        if (embedding) {
            stage.resident_fields.push_back(std::move(field));
            result.gather_capacity = aligned(gguf::checked_mul(gguf::checked_mul(tokens.ids.size(), c.hidden), tensor.type == 0 ? 4 : 2));
        } else {
            result.names.emplace(tensor.name, "model.layers." + std::to_string(layer) + "." + wanted->second.target);
            stage.blocks[layer].fields.push_back(std::move(field));
        }
    }
    for (auto &block : stage.blocks)
        std::sort(block.fields.begin(), block.fields.end(), [](const auto &a, const auto &b) { return a.name < b.name; });
    d.stages.push_back(std::move(stage));
    auto &config = result.config;
    config.enabled = true; config.schema_version = 1; config.selection = "manual"; config.retention = "request";
    config.stages["text_encoder"] = {"streamed", 1, 1 + prefetch, 0, prefetch, 1};
    result.layout = streaming::compile_layout(config, d);
    if (residency == "packed_streamed") result.read_capacity = streaming::GgufWeightPager::default_read_buffer_bytes;
    lease->revalidate_open_files(); lease->revalidate_paths();
    return result;
}

struct Qwen3GgufEncoder::Impl {
    std::shared_ptr<const streaming::SourceLease> lease;
    Qwen3GgufConfig config;
    Qwen3GgufPlan plan;
    MemoryLedger ledger;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<streaming::GgufWeightPager> source;
    std::unique_ptr<Qwen3ConditioningState> math;
    std::atomic<bool> &cancel;
    // Executor/fill workers never borrow the engine's cancellation storage;
    // a quarantined bundle owns the token even if the engine is destroyed.
    std::atomic<bool> stage_cancel{false};
    Event event;
    uint32_t prefetch;
    gguf::DecodeOptions decode_options;
    std::string residency;
    bool started = false, completed = false;
    struct Adapter final : streaming::ModelSlotAdapter {
        struct Job { Adapter *owner = nullptr; const streaming::Group *group = nullptr; std::array<char, 512> error{}; };
        Impl &state;
        std::vector<Job> jobs;
        Weights current;
        uint32_t pool = 0; uint64_t sequence = 0;
        Adapter(Impl &s) : state(s), jobs(1 + s.prefetch) { for (auto &j : jobs) j.owner = this; }
        void create_pool(const streaming::PoolLayout &p) override { pool = p.id; state.source->create_pool(p); }
        streaming::FillJob make_fill_job(const streaming::Group &g, const tc_stream_slot_ticket_v1 &ticket) override {
            auto &job = jobs.at(ticket.slot); job.group = &g; job.error[0] = 0;
            return {ticket, &job, [](void *raw, const tc_stream_slot_ticket_v1 *t, const std::atomic<bool> *cancel, uint64_t *bytes) {
                auto &j = *static_cast<Job *>(raw);
                try { *bytes = j.owner->state.source->fill(*j.group, *t, cancel); return 0; }
                catch (const std::exception &e) { std::snprintf(j.error.data(), j.error.size(), "%s", e.what()); return -1; }
                catch (...) { std::snprintf(j.error.data(), j.error.size(), "unknown Qwen3 GGUF fill error"); return -1; }
            }};
        }
        void encode_prefix(uint32_t pass) override { require(pass == 0 && state.math, "Qwen3 GGUF pass mismatch"); state.source->check_unchanged(); }
        void prepare_group(const streaming::Group &g, const tc_stream_slot_ticket_v1 &t) override {
            require(g.blocks.size() == 1 && g.blocks.front() == uint32_t(state.math->next_layer()), "Qwen3 GGUF layer out of order");
            current = state.source->bind(g, t);
            current.remap_keys([&](const std::string &key) { return state.plan.names.at(key); });
        }
        bool overlap_next_fill_after_claim() const noexcept override { return jobs.size() > 1; }
        streaming::ReaderSet encode_group(const streaming::Group &, const tc_stream_slot_ticket_v1 &, streaming::CompletionMailbox &) override {
            checkpoint(state.cancel);
            state.math->advance(current, state.event, state.cancel);
            state.math->materialize(); // Last weight reader REALLY complete before slot reuse.
            checkpoint(state.cancel); current.clear();
            require(sequence != UINT64_MAX, "Qwen3 GGUF reader sequence overflow");
            streaming::ReaderSet readers; readers.count = 1; readers.fences[0] = {1, ++sequence}; readers.already_complete = true;
            return readers;
        }
        bool drain() noexcept override { try { mx::synchronize(); return true; } catch (...) { return false; } }
        void destroy_pool() noexcept override { current.clear(); state.source->destroy_pool(pool); }
    };
    std::shared_ptr<Adapter> adapter;
    std::unique_ptr<streaming::StageExecutor> executor;
    Impl(const std::filesystem::path &path, const std::filesystem::path &cfg, const std::filesystem::path &tok,
         uint32_t p, uint64_t budget, Event e, std::atomic<bool> &c, gguf::DecodeOptions options, std::string mode)
        : ledger(budget), cancel(c), event(std::move(e)), prefetch(p), decode_options(options), residency(std::move(mode)) {
        validate_eval_policy(); require(prefetch <= 2, "qe_config_conflict: Qwen3 GGUF supports p=0/1/2");
        require(residency == "packed_resident" || residency == "packed_streamed", "qe_config_conflict: unknown Qwen3 source residency");
        if (!event) event = [](const std::string &, int, int) {};
        std::vector<streaming::SourceFileIdentity> files;
        for (const auto &[id, file] : std::vector<std::pair<std::string, std::filesystem::path>>{
            {"text_encoder", path}, {"config", cfg}, {"tokenizer", tok}}) {
            streaming::SourceFileIdentity f; f.logical_id = id; f.path = file; files.push_back(std::move(f));
        }
        lease = streaming::SourceLease::capture_verified(std::move(files), &cancel);
        auto config_fd = lease->duplicate_fd("config"); config = read_qwen3_gguf_config(config_fd.get(), lease->file("config").bytes);
        auto source_fd = lease->duplicate_fd("text_encoder"); const auto directory = gguf::read_directory(source_fd.get(), lease->file("text_encoder").bytes);
        check_metadata(directory, config);
        auto token_fd = lease->duplicate_fd("tokenizer");
        verify_qwen3_gguf_tokenizer(token_fd.get(), lease->file("tokenizer").bytes, source_fd.get(), directory, config);
        tokenizer = std::make_unique<Tokenizer>(token_fd.get(), lease->file("tokenizer").bytes);
        lease->revalidate_open_files(); lease->revalidate_paths();
    }
};
Qwen3GgufEncoder::Qwen3GgufEncoder(const std::filesystem::path &p, const std::filesystem::path &c,
        const std::filesystem::path &t, uint32_t prefetch, uint64_t budget, const Event &e, std::atomic<bool> &cancel,
        gguf::DecodeOptions options, const std::string &residency)
    : impl_(std::make_unique<Impl>(p, c, t, prefetch, budget, e, cancel, options, residency)) {}
Qwen3GgufEncoder::~Qwen3GgufEncoder() { if (!drain_safely()) (void)impl_.release(); }
Tokens Qwen3GgufEncoder::tokenize(const std::string &prompt, bool dynamic) {
    impl_->lease->revalidate_open_files(); impl_->lease->revalidate_paths();
    return impl_->tokenizer->z_image_prompt(prompt, dynamic);
}
std::string Qwen3GgufEncoder::identity() const {
    return std::string(impl_->lease->artifact_digest()) + ":qwen3-z-source-mixed-v1:p=" +
        std::to_string(impl_->prefetch) + ":managed=" + std::to_string(impl_->ledger.snapshot().budget_bytes) +
        ":decode=" + (impl_->decode_options.use_simd ? "cpu_simd" : "cpu_scalar") + ":source=" + impl_->residency;
}
bool Qwen3GgufEncoder::drain_safely() noexcept {
    if (!impl_) return true;
    return !impl_->executor || impl_->executor->retry_drain();
}
Tensor Qwen3GgufEncoder::encode(const Tokens &tokens) {
    auto &s = *impl_; require(!s.started, "Qwen3 GGUF encoder is request-scoped"); s.started = true;
    s.plan = describe_qwen3_gguf(s.lease, s.config, tokens, s.prefetch, s.decode_options, s.residency);
    const auto floor = gguf::checked_add(gguf::checked_add(gguf::checked_add(s.plan.packed_capacity, s.plan.read_capacity),
        s.plan.gather_capacity), s.plan.layout.stages.front().peak_pool_bytes);
    require(floor <= s.ledger.snapshot().budget_bytes, "qe_budget_floor: Qwen3 GGUF source/slots/gather exceed managed weight ceiling");
    s.source = std::make_unique<streaming::GgufWeightPager>(s.lease, s.plan.descriptor,
        s.plan.descriptor.stages.front(), s.plan.layout.stages.front(), s.ledger, s.decode_options);
    s.event("load_qwen3_gguf_packed", 0, 1); s.source->load_packed(&s.cancel); s.event("load_qwen3_gguf_packed", 1, 1);
    std::vector<uint64_t> rows(tokens.ids.begin(), tokens.ids.end());
    auto embedding = s.source->gather_rows("token_embd.weight", rows, &s.cancel);
    s.math = std::make_unique<Qwen3ConditioningState>(tokens, mx::expand_dims(embedding, 0), Qwen3Conditioning::z_image());
    s.adapter = std::make_shared<Impl::Adapter>(s);
    s.executor = std::make_unique<streaming::StageExecutor>(0, s.lease->generation(), s.adapter);
    s.executor->begin(s.plan.layout.stages.front());
    try {
        checkpoint(s.cancel);
        s.executor->run_pass(0, 0, s.stage_cancel); s.executor->finish();
        s.source->check_unchanged(); auto result = s.math->finish(s.event); s.completed = true; return result;
    } catch (const std::exception &error) {
        s.stage_cancel.store(true, std::memory_order_release);
        (void)drain_safely(); checkpoint(s.cancel);
        for (const auto &job : s.adapter->jobs) if (job.error[0])
            throw std::runtime_error(std::string(error.what()) + "; " + job.error.data());
        throw;
    }
}
QuantizedExecutionMetrics Qwen3GgufEncoder::metrics() const {
    const auto &s = *impl_; require(s.completed, "Qwen3 GGUF metrics require successful execution");
    const auto m = s.source->metrics(); const auto e = s.executor->counters();
    QuantizedExecutionMetrics out; out.source_sha256 = s.lease->file("text_encoder").content_digest;
    out.layout_digest = s.plan.layout.digest; out.packed_bytes = m.packed_source_bytes; out.packed_capacity_bytes = m.packed_capacity_bytes;
    out.source_float_bytes = m.source_float_bytes; out.dense_capacity_bytes = m.maximum_dense_pool_capacity_bytes;
    out.managed_peak_bytes = s.ledger.snapshot().peak_committed_bytes; out.fills = m.fill_count; out.decoded_bytes = m.decoded_bytes;
    out.source_load_seconds = m.packed_read_seconds; out.decode_seconds = m.decode_seconds; out.exposed_wait_seconds = e.wait_seconds;
    out.slots = s.plan.layout.stages.front().slot_count; out.prefetch = s.prefetch;
    out.decode_backend = s.decode_options.use_simd ? "cpu_simd" : "cpu_scalar";
    out.source_residency = s.residency; out.source_logical_bytes = m.source_logical_bytes;
    out.read_buffer_bytes = m.read_buffer_capacity_bytes; out.source_read_bytes = m.source_read_bytes;
    out.streamed_read_seconds = m.streamed_read_seconds;
    require(out.fills == 35 && e.groups_submitted == 35, "Qwen3 GGUF fill/compute count mismatch"); return out;
}
} // namespace tc::components
