#include "flux.hpp"
#include "flux_streaming.hpp"
#include "../../backends/coreml.hpp"
#include "../../runtime/streaming/audit.hpp"
#include <cmath>
#include <cstring>
#include <cstdlib>
namespace tc {
namespace {
std::vector<Tensor> flux_rope_pair(const Tensor &q, const Tensor &k,
                                   const Tensor &cos, const Tensor &sin) {
    if (std::getenv("TURBOCIDER_FLUX_EAGER_ROPE"))
        return {rope_pairs(q, cos, sin), rope_pairs(k, cos, sin)};
    return rope_pairs_pair(q, k, cos, sin);
}

bool flux_force_fused_sdpa() {
    // MLX already selects its fused Metal kernel for the validated FLUX
    // geometry.  The forced-path A/B differed by only about 0.2% on M4 Max,
    // so keep the upstream heuristic as the default and retain this opt-in
    // for profiling unusual token shapes and memory pressure.
    return std::getenv("TURBOCIDER_FLUX_FORCE_FUSED_SDPA") != nullptr;
}

std::function<std::vector<Tensor>(const std::vector<Tensor> &)>
make_hybrid_gpu_graph(int hidden, int head_count, int mlp_width, int gpu_mlp_start) {
    require(hidden == 3072 && mlp_width == 9216 && head_count == 24,
            "unsupported FLUX hybrid graph geometry");
    require(gpu_mlp_start > 0 && gpu_mlp_start <= mlp_width,
            "invalid FLUX GPU MLP complement");
    auto graph = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>(
        [hidden, head_count, mlp_width, gpu_mlp_start](const std::vector<Tensor> &args) {
            const int projection_offset = hidden * 3;
            auto qkv = mx::matmul(
                args[0], mx::transpose(slice_axis(args[3], 0, 0, projection_offset)));
            auto parts = mx::split(qkv, 3, -1);
            auto q = mx::astype(
                mx::fast::rms_norm(
                    mx::astype(heads(parts[0], head_count, 128), mx::float32), args[4],
                    1e-5f),
                mx::bfloat16);
            auto k = mx::astype(
                mx::fast::rms_norm(
                    mx::astype(heads(parts[1], head_count, 128), mx::float32), args[5],
                    1e-5f),
                mx::bfloat16);
            auto rotated = flux_rope_pair(q, k, args[1], args[2]);
            auto attention = attend(rotated[0], rotated[1],
                                    heads(parts[2], head_count, 128), false, {},
                                    flux_force_fused_sdpa());
            auto result = mx::matmul(
                attention, mx::transpose(slice_axis(args[6], 1, 0, hidden)));
            if (gpu_mlp_start < mlp_width) {
                auto gate = mx::matmul(
                    args[0],
                    mx::transpose(slice_axis(args[3], 0,
                                             projection_offset + gpu_mlp_start,
                                             projection_offset + mlp_width)));
                auto up = mx::matmul(
                    args[0],
                    mx::transpose(slice_axis(args[3], 0,
                                             projection_offset + mlp_width + gpu_mlp_start,
                                             projection_offset + 2 * mlp_width)));
                auto activated = silu(gate) * up;
                auto complement = mx::matmul(
                    activated,
                    mx::transpose(slice_axis(args[6], 1, hidden + gpu_mlp_start,
                                             hidden + mlp_width)));
                result = result + complement;
            }
            return std::vector<Tensor>{result};
        });
    return mx::compile(std::move(graph));
}

void flux_exact_dual_block(
        Tensor &image, Tensor &context, const Weights &weights,
        uint32_t block, const std::vector<Tensor> &image_modulation,
        const std::vector<Tensor> &text_modulation, const Tensor &cos,
        const Tensor &sin, int text_tokens, int total_tokens, int heads_count) {
    const auto prefix = "transformer_blocks." + std::to_string(block);
    auto feed_forward = [&](const Tensor &value, const std::string &name) {
        auto parts = mx::split(
            linear(value, weights, name + ".linear_in"), 2, -1);
        return linear(silu(parts[0]) * parts[1], weights,
                      name + ".linear_out");
    };
    auto normalize = [&](const Tensor &value, const Tensor &weight) {
        return mx::astype(
            mx::fast::rms_norm(mx::astype(value, mx::float32), weight, 1e-5f),
            value.dtype());
    };
    auto qkv = [&](const Tensor &value, const std::string &name,
                   bool text) {
        auto q = heads(linear(value, weights,
                              name + (text ? ".add_q_proj" : ".to_q")),
                       heads_count, 128);
        auto k = heads(linear(value, weights,
                              name + (text ? ".add_k_proj" : ".to_k")),
                       heads_count, 128);
        auto v = heads(linear(value, weights,
                              name + (text ? ".add_v_proj" : ".to_v")),
                       heads_count, 128);
        return std::vector<Tensor>{
            normalize(q, weights.at(name +
                (text ? ".norm_added_q.weight" : ".norm_q.weight"))),
            normalize(k, weights.at(name +
                (text ? ".norm_added_k.weight" : ".norm_k.weight"))),
            v};
    };
    auto image_qkv = qkv(
        norm(image) * (1 + image_modulation[1]) + image_modulation[0],
        prefix + ".attn", false);
    auto text_qkv = qkv(
        norm(context) * (1 + text_modulation[1]) + text_modulation[0],
        prefix + ".attn", true);
    auto rotated = flux_rope_pair(
        mx::concatenate({text_qkv[0], image_qkv[0]}, 2),
        mx::concatenate({text_qkv[1], image_qkv[1]}, 2), cos, sin);
    auto values = mx::concatenate({text_qkv[2], image_qkv[2]}, 2);
    auto attention = attend(rotated[0], rotated[1], values, false, {},
                            flux_force_fused_sdpa());
    image = image + image_modulation[2] * linear(
        slice_axis(attention, 1, text_tokens, total_tokens), weights,
        prefix + ".attn.to_out.0");
    context = context + text_modulation[2] * linear(
        slice_axis(attention, 1, 0, text_tokens), weights,
        prefix + ".attn.to_add_out");
    image = image + image_modulation[5] * feed_forward(
        norm(image) * (1 + image_modulation[4]) + image_modulation[3],
        prefix + ".ff");
    context = context + text_modulation[5] * feed_forward(
        norm(context) * (1 + text_modulation[4]) + text_modulation[3],
        prefix + ".ff_context");
}

void flux_exact_single_block(
        Tensor &value, const Weights &weights, uint32_t block,
        const std::vector<Tensor> &modulation, const Tensor &cos,
        const Tensor &sin, int hidden, int heads_count) {
    const auto prefix = "single_transformer_blocks." +
        std::to_string(block) + ".attn";
    auto input = norm(value) * (1 + modulation[1]) + modulation[0];
    auto projection = linear(input, weights, prefix + ".to_qkv_mlp_proj");
    auto parts = mx::split(
        projection,
        mx::Shape{hidden, hidden * 2, hidden * 3, hidden * 6}, -1);
    require(parts.size() == 5 && parts[0].shape(-1) == hidden &&
                parts[1].shape(-1) == hidden &&
                parts[2].shape(-1) == hidden &&
                parts[3].shape(-1) == hidden * 3 &&
                parts[4].shape(-1) == hidden * 3,
            "FLUX exact single-block packed projection layout mismatch");
    auto q = mx::astype(
        mx::fast::rms_norm(
            mx::astype(heads(parts[0], heads_count, 128), mx::float32),
            weights.at(prefix + ".norm_q.weight"), 1e-5f),
        mx::bfloat16);
    auto k = mx::astype(
        mx::fast::rms_norm(
            mx::astype(heads(parts[1], heads_count, 128), mx::float32),
            weights.at(prefix + ".norm_k.weight"), 1e-5f),
        mx::bfloat16);
    auto rotated = flux_rope_pair(q, k, cos, sin);
    auto attention = attend(
        rotated[0], rotated[1], heads(parts[2], heads_count, 128), false, {},
        flux_force_fused_sdpa());
    auto mlp = silu(parts[3]) * parts[4];
    value = value + modulation[2] * linear(
        mx::concatenate({attention, mlp}, -1), weights,
        prefix + ".to_out");
}

class FluxExactAdapter final : public streaming::ModelSlotAdapter {
    struct Job {
        FluxExactAdapter *owner = nullptr;
        const streaming::Group *group = nullptr;
        std::array<char, 512> error{};
    };

    flux2::StreamingPlanView &plan_;
    streaming::MlxWeightPager &source_;
    const Event &event_;
    std::atomic<bool> &cancelled_;
    uint32_t dual_blocks_ = 0;
    uint32_t single_blocks_ = 0;
    uint32_t active_pool_ = std::numeric_limits<uint32_t>::max();
    std::array<Job, 4> jobs_{};
    Weights current_;
    Tensor *image_ = nullptr;
    Tensor *context_ = nullptr;
    const std::vector<Tensor> *image_modulation_ = nullptr;
    const std::vector<Tensor> *text_modulation_ = nullptr;
    const std::vector<Tensor> *single_modulation_ = nullptr;
    const Tensor *cos_ = nullptr;
    const Tensor *sin_ = nullptr;
    int text_tokens_ = 0;
    int total_tokens_ = 0;
    uint32_t pass_ = 0;
    uint32_t step_ = 0;
    uint64_t reader_sequence_ = 0;
    bool single_started_ = false;

    void require_context() const {
        require(image_ && context_ && image_modulation_ && text_modulation_ &&
                    single_modulation_ && cos_ && sin_,
                "FLUX exact adapter has no active pass context");
    }

    Job &job(const tc_stream_slot_ticket_v1 &ticket) {
        require(ticket.pool < 2 && ticket.slot < 2,
                "FLUX exact job index is outside the two-class K2 layout");
        return jobs_[ticket.pool * 2 + ticket.slot];
    }

  public:
    FluxExactAdapter(flux2::StreamingPlanView &plan,
                     streaming::MlxWeightPager &source,
                     const Event &event, std::atomic<bool> &cancelled)
        : plan_(plan), source_(source), event_(event), cancelled_(cancelled),
          dual_blocks_(plan.metadata().dual_block_count()),
          single_blocks_(plan.metadata().single_block_count()) {
        for (auto &value : jobs_)
            value.owner = this;
    }

    void bind_pass(
            uint32_t pass, uint32_t step, Tensor &image, Tensor &context,
            const std::vector<Tensor> &image_modulation,
            const std::vector<Tensor> &text_modulation,
            const std::vector<Tensor> &single_modulation,
            const Tensor &cos, const Tensor &sin, int text_tokens,
            int total_tokens) {
        require(!image_, "FLUX exact pass context is already bound");
        pass_ = pass;
        step_ = step;
        image_ = &image;
        context_ = &context;
        image_modulation_ = &image_modulation;
        text_modulation_ = &text_modulation;
        single_modulation_ = &single_modulation;
        cos_ = &cos;
        sin_ = &sin;
        text_tokens_ = text_tokens;
        total_tokens_ = total_tokens;
        single_started_ = false;
    }

    void unbind_pass() noexcept {
        current_.clear();
        image_ = nullptr;
        context_ = nullptr;
        image_modulation_ = nullptr;
        text_modulation_ = nullptr;
        single_modulation_ = nullptr;
        cos_ = nullptr;
        sin_ = nullptr;
    }

    std::string fill_error() const {
        for (const auto &value : jobs_)
            if (value.error[0]) return value.error.data();
        return {};
    }

    bool supports_multi_pool_policy(
            streaming::MultiPoolPolicy policy) const noexcept override {
        return policy == streaming::MultiPoolPolicy::retain_all;
    }

    void create_pool(const streaming::PoolLayout &pool) override {
        require(pool.id < 2 && pool.slots.size() == 2,
                "FLUX exact adapter requires two retained K2 pools");
        source_.create_pool(pool);
        active_pool_ = pool.id;
    }

    void select_pool(const streaming::PoolLayout &pool) override {
        require(pool.id < 2, "FLUX exact pool selection is invalid");
        active_pool_ = pool.id;
    }

    streaming::FillJob make_fill_job(
            const streaming::Group &group,
            const tc_stream_slot_ticket_v1 &ticket) override {
        require(group.blocks.size() == 1 && ticket.pool == group.pool &&
                    ticket.slot == group.slot,
                "FLUX exact fill identity differs from group");
        auto &value = job(ticket);
        value.group = &group;
        value.error[0] = 0;
        return {ticket, &value,
                [](void *raw, const tc_stream_slot_ticket_v1 *ticket,
                   const std::atomic<bool> *worker_cancel,
                   uint64_t *bytes) -> int {
                    auto &value = *static_cast<Job *>(raw);
                    try {
                        *bytes = value.owner->source_.fill(
                            *value.group, *ticket, worker_cancel);
                        return 0;
                    } catch (const std::exception &error) {
                        std::snprintf(value.error.data(), value.error.size(),
                                      "%s", error.what());
                        return -1;
                    } catch (...) {
                        std::snprintf(value.error.data(), value.error.size(),
                                      "%s", "unknown FLUX fill failure");
                        return -1;
                    }
                }};
    }

    void encode_prefix(uint32_t pass) override {
        require_context();
        require(pass == pass_, "FLUX exact prefix pass mismatch");
        plan_.metadata().check_unchanged();
        source_.check_open_files();
    }

    void prepare_group(
            const streaming::Group &group,
            const tc_stream_slot_ticket_v1 &ticket) override {
        require_context();
        require(group.blocks.size() == 1 && ticket.item.pass == pass_ &&
                    ticket.item.step == step_ &&
                    ticket.item.group == group.id,
                "FLUX exact prepare ticket mismatch");
        const uint32_t block = group.blocks.front();
        if (block == dual_blocks_) {
            require(!single_started_,
                    "FLUX exact single-stream boundary repeated");
            *image_ = mx::concatenate({*context_, *image_}, 1);
            single_started_ = true;
        }
        current_ = source_.bind(group, ticket);
    }

    bool overlap_next_fill_after_claim() const noexcept override {
        return true;
    }

    streaming::ReaderSet encode_group(
            const streaming::Group &group,
            const tc_stream_slot_ticket_v1 &ticket,
            streaming::CompletionMailbox &) override {
        require_context();
        require(group.blocks.size() == 1 && ticket.item.pass == pass_ &&
                    ticket.item.step == step_,
                "FLUX exact encode ticket mismatch");
        checkpoint(cancelled_);
        const uint32_t block = group.blocks.front();
        event_("transformer_block", int(block),
               int(dual_blocks_ + single_blocks_));
        if (block < dual_blocks_) {
            flux_exact_dual_block(
                *image_, *context_, current_, block, *image_modulation_,
                *text_modulation_, *cos_, *sin_, text_tokens_, total_tokens_,
                32);
            mx::eval({*image_, *context_});
        } else {
            require(single_started_,
                    "FLUX exact single block ran before class boundary");
            flux_exact_single_block(
                *image_, current_, block - dual_blocks_,
                *single_modulation_, *cos_, *sin_, 4096, 32);
            mx::eval(*image_);
        }
        checkpoint(cancelled_);
        require(reader_sequence_ != std::numeric_limits<uint64_t>::max(),
                "FLUX exact reader sequence overflow");
        streaming::ReaderSet readers;
        readers.count = 1;
        readers.fences[0] = {1, ++reader_sequence_};
        readers.already_complete = true;
        current_.clear();
        return readers;
    }

    bool drain() noexcept override {
        try {
            mx::synchronize();
            return true;
        } catch (...) {
            return false;
        }
    }

    void destroy_pool() noexcept override {
        current_.clear();
        if (active_pool_ < 2)
            source_.destroy_pool(active_pool_);
        active_pool_ = std::numeric_limits<uint32_t>::max();
    }

    void destroy_pool(uint32_t pool) noexcept override {
        current_.clear();
        source_.destroy_pool(pool);
        if (active_pool_ == pool)
            active_pool_ = std::numeric_limits<uint32_t>::max();
    }
};

// Benchmark-only narrow reference scheduler for the frozen Flux 9B tuple.
// It intentionally reuses the same pager, adapter, two persistent I/O workers,
// K2 backing, D1 dispatch window and synchronous reader completion as the
// generic StageExecutor.  The only removed layer is StageExecutor's generic
// owner orchestration, making direct/generic timing a meaningful framework
// overhead comparison.  It is selected only by an internal environment flag
// on the already-private candidate route and is never an automatic fallback.
class FluxDirectExecutor final {
    struct PoolState {
        std::unique_ptr<streaming::SlotSafetyTracker> safety;
        std::vector<tc_stream_slot_ticket_v1> tickets;
        bool live = false;
    };

    const streaming::StageLayout &layout_;
    uint64_t request_ = 0;
    std::shared_ptr<FluxExactAdapter> adapter_;
    std::vector<PoolState> pools_;
    std::unique_ptr<streaming::CompletionMailbox> mailbox_;
    std::unique_ptr<streaming::IoExecutor> io_;
    streaming::ExecutionCounters counters_{};
    uint32_t passes_ = 0;
    bool failed_ = false, finished_ = false, pools_live_ = false;

    void check_cancel(const std::atomic<bool> &cancel) const {
        if (cancel.load(std::memory_order_acquire))
            throw std::runtime_error("streaming_cancelled");
        if (mailbox_->overflowed())
            throw std::runtime_error("streaming_mailbox_overflow");
    }

    bool consume(uint32_t pool_index) {
        require(pool_index < pools_.size() && pools_[pool_index].safety,
                "FLUX direct completion pool is unavailable");
        bool progress = false;
        tc_stream_completion_v1 record{};
        while (mailbox_->pop(record)) {
            if (record.struct_size != sizeof(record) ||
                record.version != TC_STREAM_SLOT_ABI_V1 || record.status ||
                record.kind != TC_STREAM_FILL_COMPLETE ||
                record.ticket.pool != layout_.pools[pool_index].id)
                throw std::runtime_error("streaming_completion_failed");
            pools_[pool_index].safety->accept_ready(
                record.ticket, record.bytes);
            require(record.bytes <= UINT64_MAX - counters_.bytes_loaded,
                    "FLUX direct byte counter overflow");
            counters_.bytes_loaded += record.bytes;
            ++counters_.fills;
            progress = true;
        }
        return progress;
    }

    bool cleanup() noexcept {
        if (io_) {
            io_->shutdown_and_join();
            io_.reset();
        }
        if (!pools_live_) return true;
        if (!adapter_->drain()) return false;
        for (size_t index = pools_.size(); index-- > 0;) {
            if (!pools_[index].live) continue;
            adapter_->destroy_pool(layout_.pools[index].id);
            pools_[index].safety.reset();
            pools_[index].tickets.clear();
            pools_[index].live = false;
        }
        pools_live_ = false;
        return true;
    }

  public:
    FluxDirectExecutor(const streaming::StageLayout &layout,
                       uint64_t request,
                       std::shared_ptr<FluxExactAdapter> adapter)
        : layout_(layout), request_(request), adapter_(std::move(adapter)) {
        require(request_ && adapter_ && !layout_.resident &&
                    layout_.prefix == 0 && layout_.group_size == 1 &&
                    layout_.slot_count == 2 && layout_.distance == 1 &&
                    layout_.workers == 2 && layout_.pass_count > 0 &&
                    layout_.pass_transition ==
                        streaming::PassTransition::reload &&
                    layout_.multi_pool_policy ==
                        streaming::MultiPoolPolicy::retain_all &&
                    layout_.pools.size() == 2 &&
                    layout_.groups.size() == 32,
                "FLUX direct baseline requires P0/G1/K2/D1/Q2 retained layout");
        pools_.resize(layout_.pools.size());
        mailbox_ = std::make_unique<streaming::CompletionMailbox>(
            layout_.slot_count * (1 + TC_STREAM_MAX_READER_QUEUES));
        try {
            for (size_t index = 0; index < layout_.pools.size(); ++index) {
                const auto &pool = layout_.pools[index];
                require(pool.slots.size() == layout_.slot_count,
                        "FLUX direct pool slot count differs");
                std::vector<uint64_t> capacities;
                capacities.reserve(pool.slots.size());
                for (const auto &slot : pool.slots)
                    capacities.push_back(slot.capacity_bytes);
                auto &runtime = pools_[index];
                runtime.safety =
                    std::make_unique<streaming::SlotSafetyTracker>(
                        pool.id, request_, capacities);
                runtime.tickets.assign(pool.slots.size(), {});
                runtime.live = true;
                pools_live_ = true;
                adapter_->create_pool(pool);
                streaming::audit_increment(
                    streaming::AuditCounter::PoolAllocations);
                ++counters_.pool_creates;
                counters_.slot_bundles += pool.slots.size();
            }
            adapter_->select_pool(layout_.pools.front());
            io_ = std::make_unique<streaming::IoExecutor>(
                layout_.workers, layout_.slot_count, *mailbox_);
        } catch (...) {
            failed_ = true;
            if (!cleanup()) std::terminate();
            throw;
        }
    }

    ~FluxDirectExecutor() {
        if (pools_live_ && !cleanup()) std::terminate();
    }

    void run_pass(uint32_t pass, uint32_t step,
                  std::atomic<bool> &cancel,
                  std::chrono::milliseconds timeout =
                      std::chrono::seconds(60)) {
        try {
            require(!failed_ && !finished_ && pools_live_ &&
                        pass == passes_ && pass < layout_.pass_count &&
                        timeout.count() > 0,
                    "FLUX direct executor lifecycle mismatch");
            bool prefix = false;
            size_t segment_begin = 0;
            uint32_t pool_index = 0;
            while (segment_begin < layout_.groups.size()) {
                require(pool_index < pools_.size(),
                        "FLUX direct pool index overflow");
                const uint32_t pool_id =
                    layout_.groups[segment_begin].pool;
                size_t segment_end = segment_begin + 1;
                while (segment_end < layout_.groups.size() &&
                       layout_.groups[segment_end].pool == pool_id)
                    ++segment_end;
                require(pool_id == layout_.pools[pool_index].id,
                        "FLUX direct group pool order mismatch");
                adapter_->select_pool(layout_.pools[pool_index]);
                auto &runtime = pools_[pool_index];
                auto &safety = *runtime.safety;
                size_t next = segment_begin, dispatch = segment_begin;
                auto last_progress = std::chrono::steady_clock::now();
                while (next < segment_end || !safety.quiescent()) {
                    check_cancel(cancel);
                    bool progress = consume(pool_index);
                    while (dispatch < segment_end &&
                           dispatch <= next + layout_.distance) {
                        const auto &group = layout_.groups[dispatch];
                        if (safety.state(group.slot) !=
                            streaming::ContentState::Vacant)
                            break;
                        auto ticket = safety.begin_fill(
                            group.slot, {0, pass, step, group.id},
                            group.bytes);
                        auto job = adapter_->make_fill_job(group, ticket);
                        job.ticket = ticket;
                        if (!io_->enqueue(job))
                            throw std::logic_error(
                                "FLUX direct slot and I/O credits diverged");
                        runtime.tickets[group.slot] = ticket;
                        ++dispatch;
                        progress = true;
                    }
                    if (!prefix) {
                        adapter_->encode_prefix(pass);
                        prefix = true;
                        progress = true;
                    }
                    if (next < segment_end) {
                        const auto &group = layout_.groups[next];
                        if (safety.state(group.slot) ==
                            streaming::ContentState::Ready) {
                            const auto &ticket =
                                runtime.tickets[group.slot];
                            adapter_->prepare_group(group, ticket);
                            safety.begin_use(ticket);
                            const auto readers = adapter_->encode_group(
                                group, ticket, *mailbox_);
                            require(readers.already_complete &&
                                        readers.count > 0 &&
                                        readers.count <= readers.fences.size(),
                                    "FLUX direct baseline requires synchronous readers");
                            safety.seal_readers(
                                ticket,
                                {readers.fences.data(), readers.count});
                            for (uint32_t reader = 0;
                                 reader < readers.count; ++reader)
                                safety.complete_reader(
                                    ticket, readers.fences[reader]);
                            ++next;
                            ++counters_.groups_submitted;
                            progress = true;
                        }
                    }
                    if (progress) {
                        last_progress = std::chrono::steady_clock::now();
                    } else {
                        if (std::chrono::steady_clock::now() - last_progress >
                            timeout)
                            throw std::runtime_error(
                                "streaming_stall_timeout");
                        const auto wait_start =
                            std::chrono::steady_clock::now();
                        mailbox_->wait_for(std::chrono::milliseconds(5));
                        counters_.wait_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - wait_start)
                                .count();
                    }
                }
                if (!adapter_->drain())
                    throw std::runtime_error(
                        "streaming_pass_drain_failed");
                consume(pool_index);
                require(safety.quiescent(),
                        "FLUX direct drain left active slot content");
                segment_begin = segment_end;
                ++pool_index;
            }
            check_cancel(cancel);
            ++passes_;
        } catch (...) {
            failed_ = true;
            if (!cleanup()) std::terminate();
            throw;
        }
    }

    streaming::ExecutionCounters finish() {
        require(!failed_ && !finished_ &&
                    passes_ == layout_.pass_count,
                "FLUX direct execution is incomplete");
        io_->shutdown_and_join();
        consume(static_cast<uint32_t>(pools_.size() - 1));
        require(!mailbox_->overflowed(),
                "streaming_mailbox_overflow");
        if (!cleanup())
            throw std::runtime_error("streaming_drain_failed");
        finished_ = true;
        return counters_;
    }

    streaming::ExecutionCounters counters() const noexcept {
        return counters_;
    }
};
} // namespace

struct FluxExactStream::Impl {
    flux2::StreamingPlanView plan;
    streaming::MlxWeightPager source;
    std::shared_ptr<FluxExactAdapter> adapter;
    std::unique_ptr<streaming::StageExecutor> executor;
    std::unique_ptr<FluxDirectExecutor> direct_executor;
    std::atomic<bool> &cancelled;
    streaming::ExecutionCounters final_counters{};
    bool direct = false;
    bool finished = false;

    Impl(const std::filesystem::path &transformer_directory,
         const std::string &model_id, const StreamingConfig &config,
         const flux2::StreamingWorkload &workload, Weights &resident,
         const Event &event, std::atomic<bool> &cancelled,
         uint64_t request_generation)
        : plan(transformer_directory.string(), model_id, config, workload),
          source(transformer_directory, plan.descriptor(),
                 plan.descriptor().stages.front(),
                 plan.layout().stages.front()),
          adapter(std::make_shared<FluxExactAdapter>(
              plan, source, event, cancelled)),
          cancelled(cancelled) {
        const char *direct_value =
            std::getenv("TURBOCIDER_FLUX_DIRECT_STREAMING_BASELINE");
        require(!direct_value || std::strcmp(direct_value, "1") == 0,
                "TURBOCIDER_FLUX_DIRECT_STREAMING_BASELINE must be 1");
        direct = direct_value != nullptr;
        plan.metadata().check_unchanged();
        source.load_resident(resident, &cancelled);
        if (direct) {
            direct_executor = std::make_unique<FluxDirectExecutor>(
                plan.layout().stages.front(), request_generation, adapter);
        } else {
            executor = std::make_unique<streaming::StageExecutor>(
                0, request_generation, adapter);
            executor->begin(plan.layout().stages.front());
        }
    }
};

FluxExactStream::FluxExactStream(
        const std::filesystem::path &transformer_directory,
        const std::string &model_id, const StreamingConfig &config,
        const flux2::StreamingWorkload &workload, Weights &resident,
        const Event &event, std::atomic<bool> &cancelled,
        uint64_t request_generation)
    : impl_(std::make_unique<Impl>(
          transformer_directory, model_id, config, workload, resident,
          event, cancelled, request_generation)) {}

FluxExactStream::~FluxExactStream() = default;

void FluxExactStream::run_pass(
        uint32_t pass, uint32_t step, Tensor &image, Tensor &context,
        const std::vector<Tensor> &image_modulation,
        const std::vector<Tensor> &text_modulation,
        const std::vector<Tensor> &single_modulation,
        const Tensor &cos, const Tensor &sin, int text_tokens,
        int total_tokens) {
    require(impl_ && !impl_->finished,
            "FLUX exact executor is unavailable");
    impl_->adapter->bind_pass(
        pass, step, image, context, image_modulation, text_modulation,
        single_modulation, cos, sin, text_tokens, total_tokens);
    try {
        if (impl_->direct)
            impl_->direct_executor->run_pass(
                pass, step, impl_->cancelled);
        else
            impl_->executor->run_pass(pass, step, impl_->cancelled);
    } catch (...) {
        impl_->adapter->unbind_pass();
        const auto detail = impl_->adapter->fill_error();
        if (!detail.empty())
            throw std::runtime_error("FLUX exact fill failed: " + detail);
        throw;
    }
    impl_->adapter->unbind_pass();
}

void FluxExactStream::finish() {
    require(impl_ && !impl_->finished,
            "FLUX exact executor is already finished");
    impl_->final_counters = impl_->direct ?
        impl_->direct_executor->finish() : impl_->executor->finish();
    impl_->finished = true;
}

const flux2::StreamingPlanView &FluxExactStream::plan() const {
    require(impl_ != nullptr, "FLUX exact plan is unavailable");
    return impl_->plan;
}

const streaming::MlxWeightPagerMetrics &FluxExactStream::pager_metrics() const {
    require(impl_ != nullptr, "FLUX exact pager metrics are unavailable");
    return impl_->source.metrics();
}

streaming::ExecutionCounters FluxExactStream::counters() const {
    require(impl_ != nullptr, "FLUX exact counters are unavailable");
    if (impl_->finished) return impl_->final_counters;
    return impl_->direct ? impl_->direct_executor->counters() :
                           impl_->executor->counters();
}

const char *FluxExactStream::implementation() const noexcept {
    return impl_ && impl_->direct ? "flux_direct_same_layout_v1" :
                                   "generic_stage_executor_v1";
}

Tensor Flux::denoise(const Tensor &latent, const Tensor &text, float sigma, int height, int width,
                     const Event &event, std::atomic<bool> &cancelled,
                     const std::vector<float> &reference_ids, bool compile_blocks,
                     FluxExactStream *exact_stream, uint32_t stream_pass) {
    auto &w = transformer_;
    int nt = text.shape(1), ni = latent.shape(1), n = nt + ni;
    auto t = Tensor(sigma * 1000.f, latent.dtype());
    auto freq = mx::exp(mx::arange(128, mx::float32) * (-std::log(10000.f) / 128.f));
    auto angle = mx::reshape(mx::astype(t, mx::float32) * freq, {1, 128});
    auto emb = mx::concatenate({mx::cos(angle), mx::sin(angle)}, -1);
    auto temb =
        mx::astype(linear(silu(linear(emb, w, "time_guidance_embed.timestep_embedder.linear_1")), w,
                          "time_guidance_embed.timestep_embedder.linear_2"),
                   mx::bfloat16);
    auto mod = [&](const std::string &p, int count) {
        return mx::split(mx::expand_dims(linear(silu(temb), w, p + ".linear"), 1), count, -1);
    };
    auto mi = mod("double_stream_modulation_img", 6), mt = mod("double_stream_modulation_txt", 6),
         ms = mod("single_stream_modulation", 3);
    std::vector<float> ids(n * 4, 0);
    for (int i = 0; i < nt; ++i)
        ids[i * 4 + 3] = i;
    for (int i = 0; i < ni; ++i) {
        ids[(nt + i) * 4 + 1] = i / (width / 16);
        ids[(nt + i) * 4 + 2] = i % (width / 16);
    }
    if (!reference_ids.empty()) {
        require(reference_ids.size() <= size_t(ni) * 4, "reference ID overflow");
        std::copy(reference_ids.begin(), reference_ids.end(), ids.end() - reference_ids.size());
    }
    auto coord = Tensor(ids.data(), {n, 4}, mx::float32);
    std::vector<Tensor> cosines, sines;
    auto rf = 1.f / mx::power(Tensor(2000.f), mx::arange(0, 32, 2, mx::float32) / 32.f);
    for (int axis = 0; axis < 4; ++axis) {
        auto angles = slice_axis(coord, 1, axis, axis + 1) * rf;
        cosines.push_back(mx::cos(angles));
        sines.push_back(mx::sin(angles));
    }
    auto cos = mx::concatenate(cosines, -1), sin = mx::concatenate(sines, -1);
    auto x = linear(latent, w, "x_embedder"), c = linear(text, w, "context_embedder");
    auto ff = [&](const Tensor &a, const std::string &p) {
        auto parts = mx::split(linear(a, w, p + ".linear_in"), 2, -1);
        return linear(silu(parts[0]) * parts[1], w, p + ".linear_out");
    };
    auto qnorm = [&](const Tensor &a, const Tensor &weight, float eps) {
        return mx::astype(mx::fast::rms_norm(mx::astype(a, mx::float32), weight, eps), a.dtype());
    };
    auto qkv = [&](const Tensor &a, const std::string &p, bool ctx) {
        auto q = heads(linear(a, w, p + (ctx ? ".add_q_proj" : ".to_q")), heads_, 128),
             k = heads(linear(a, w, p + (ctx ? ".add_k_proj" : ".to_k")), heads_, 128),
             v = heads(linear(a, w, p + (ctx ? ".add_v_proj" : ".to_v")), heads_, 128);
        return std::vector<Tensor>{
            qnorm(q, w.at(p + (ctx ? ".norm_added_q.weight" : ".norm_q.weight")), 1e-5f),
            qnorm(k, w.at(p + (ctx ? ".norm_added_k.weight" : ".norm_k.weight")), 1e-5f), v};
    };
    const int total_blocks = dual_layers_ + single_layers_;
    if (exact_stream) {
        require(model_id_ == "flux2-klein-9b" && !hybrid_ &&
                    !compile_blocks && dual_layers_ == 8 &&
                    single_layers_ == 24 && heads_ == 32 && hidden_ == 4096,
                "FLUX exact streaming requires the eager Klein 9B GPU path");
        exact_stream->run_pass(
            stream_pass, stream_pass, x, c, mi, mt, ms, cos, sin, nt, n);
    } else {
      for (int i = 0; i < dual_layers_; ++i) {
        checkpoint(cancelled);
        event("transformer_block", i, total_blocks);
        auto p = "transformer_blocks." + std::to_string(i);
        auto a = qkv(norm(x) * (1 + mi[1]) + mi[0], p + ".attn", false),
             b = qkv(norm(c) * (1 + mt[1]) + mt[0], p + ".attn", true);
        auto qk = flux_rope_pair(mx::concatenate({b[0], a[0]}, 2),
                                 mx::concatenate({b[1], a[1]}, 2), cos, sin);
        auto v = mx::concatenate({b[2], a[2]}, 2);
        auto att = attend(qk[0], qk[1], v, false, {}, flux_force_fused_sdpa());
        x = x + mi[2] * linear(slice_axis(att, 1, nt, n), w, p + ".attn.to_out.0");
        c = c + mt[2] * linear(slice_axis(att, 1, 0, nt), w, p + ".attn.to_add_out");
        x = x + mi[5] * ff(norm(x) * (1 + mi[4]) + mi[3], p + ".ff");
        c = c + mt[5] * ff(norm(c) * (1 + mt[4]) + mt[3], p + ".ff_context");
        mx::eval({x, c});
      }
      x = mx::concatenate({c, x}, 1);
      auto sqnorm = [&](const Tensor &a, const Tensor &weight, float eps) {
        return mx::astype(mx::fast::rms_norm(mx::astype(a, mx::float32), weight, eps),
                          mx::bfloat16);
      };
      for (int i = 0; i < single_layers_; ++i) {
        checkpoint(cancelled);
        event("transformer_block", dual_layers_ + i, total_blocks);
        auto p = "single_transformer_blocks." + std::to_string(i) + ".attn";
        auto a = norm(x) * (1 + ms[1]) + ms[0];
        if (hybrid_) {
            auto packed = mx::astype(a, mx::float16);
            if (n < hybrid_->rows)
                packed = mx::concatenate(
                    {packed, mx::zeros({1, hybrid_->rows - n, 3072}, mx::float16)}, 1);
            mx::eval({a, packed});
            // The ANE model may own only a prefix of the 9,216-channel MLP.
            // Compile attention plus the complementary GPU suffix as one graph,
            // then submit it before the blocking Core ML prediction so both devices
            // execute concurrently. Weights remain explicit inputs and cannot be
            // retained across a model/LoRA identity change.
            if (!hybrid_gpu_graph_ || hybrid_gpu_mlp_start_ != hybrid_->ane_mlp_end) {
                hybrid_gpu_graph_ = make_hybrid_gpu_graph(
                    hidden_, heads_, hybrid_->mlp_width, hybrid_->ane_mlp_end);
                hybrid_gpu_mlp_start_ = hybrid_->ane_mlp_end;
            }
            auto gpu = hybrid_gpu_graph_({a, cos, sin, w.at(p + ".to_qkv_mlp_proj.weight"),
                                          w.at(p + ".norm_q.weight"), w.at(p + ".norm_k.weight"),
                                          w.at(p + ".to_out.weight")})[0];
            mx::async_eval({gpu});
            auto ane = slice_axis(hybrid_->predict(i, packed), 1, 0, n);
            x = x + ms[2] * (gpu + mx::astype(ane, gpu.dtype()));
        } else if (compile_blocks && model_id_ == "flux2-klein-4b") {
            static auto single_graph = mx::compile([](const std::vector<Tensor> &args) {
                auto input = norm(args[0]) * (1 + args[2]) + args[1];
                auto parts = mx::split(mx::matmul(input, mx::transpose(args[6])),
                                       mx::Shape{3072, 6144, 9216, 18432}, -1);
                auto q =
                    mx::astype(mx::fast::rms_norm(mx::astype(heads(parts[0], 24, 128), mx::float32),
                                                  args[7], 1e-5f),
                               mx::bfloat16);
                auto k =
                    mx::astype(mx::fast::rms_norm(mx::astype(heads(parts[1], 24, 128), mx::float32),
                                                  args[8], 1e-5f),
                               mx::bfloat16);
                auto rotated = flux_rope_pair(q, k, args[4], args[5]);
                auto att = attend(rotated[0], rotated[1], heads(parts[2], 24, 128),
                                  false, {}, flux_force_fused_sdpa());
                auto mlp = silu(parts[3]) * parts[4];
                return std::vector<Tensor>{
                    args[0] +
                    args[3] * mx::matmul(mx::concatenate({att, mlp}, -1), mx::transpose(args[9]))};
            });
            x = single_graph({x, ms[0], ms[1], ms[2], cos, sin, w.at(p + ".to_qkv_mlp_proj.weight"),
                              w.at(p + ".norm_q.weight"), w.at(p + ".norm_k.weight"),
                              w.at(p + ".to_out.weight")})[0];
        } else {
            auto proj = linear(a, w, p + ".to_qkv_mlp_proj");
            auto parts = mx::split(proj, mx::Shape{hidden_, hidden_ * 2, hidden_ * 3,
                                                   hidden_ * 6}, -1);
            require(parts.size() == 5 && parts[0].shape(-1) == hidden_ &&
                        parts[1].shape(-1) == hidden_ && parts[2].shape(-1) == hidden_ &&
                        parts[3].shape(-1) == hidden_ * 3 && parts[4].shape(-1) == hidden_ * 3,
                    "FLUX single-block packed projection layout mismatch");
            auto q = sqnorm(heads(parts[0], heads_, 128),
                            w.at(p + ".norm_q.weight"), 1e-5f);
            auto k = sqnorm(heads(parts[1], heads_, 128),
                            w.at(p + ".norm_k.weight"), 1e-5f);
            auto rotated = flux_rope_pair(q, k, cos, sin);
            auto att = attend(rotated[0], rotated[1], heads(parts[2], heads_, 128),
                              false, {}, flux_force_fused_sdpa());
            auto mlp = silu(parts[3]) * parts[4];
            x = x + ms[2] * linear(mx::concatenate({att, mlp}, -1), w, p + ".to_out");
        }
        // A compiled pure-GPU block retains its allocator dependency chain;
        // the caller synchronizes the final denoiser output once per step.
        // Avoiding twenty host waits per FLUX.2 Klein 4B step is safe because
        // hybrid Core ML boundaries and eager/9B paths still synchronize.
        if (hybrid_ || !compile_blocks || model_id_ != "flux2-klein-4b" ||
            std::getenv("TURBOCIDER_FLUX_SYNC_BLOCKS"))
            mx::eval(x);
      }
    }
    auto outmod = mx::split(mx::expand_dims(linear(silu(temb), w, "norm_out.linear"), 1), 2, -1);
    return linear(norm(slice_axis(x, 1, nt, n)) * (1 + outmod[0]) + outmod[1], w, "proj_out");
}
} // namespace tc
