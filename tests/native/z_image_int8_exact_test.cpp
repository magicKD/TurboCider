#include "models/z_image/weight_stream.hpp"
#include "runtime/streaming/context.hpp"
#include <fstream>
#include <iostream>
#include <set>

using namespace tc;

class VerifyAdapter final : public streaming::ModelSlotAdapter {
    struct Job { VerifyAdapter *owner; uint32_t slot, block; };
    ZImageWeightStream &source_;
    std::function<void(const Weights &)> verify_;
    std::vector<Job> jobs_;
    std::set<const void *> buffers_;
    uint32_t prefix_;
    uint64_t sequence_ = 0;
  public:
    VerifyAdapter(ZImageWeightStream &source, uint32_t slots, uint32_t prefix,
                  std::function<void(const Weights &)> verify)
        : source_(source), verify_(std::move(verify)), jobs_(slots), prefix_(prefix) {
        for (uint32_t i = 0; i < slots; ++i) jobs_[i] = {this, i, 0};
    }
    void create_pool(const streaming::PoolLayout &pool) override {
        source_.create_exact_pool(uint32_t(pool.slots.size()), pool.slots[0].capacity_bytes);
    }
    streaming::FillJob make_fill_job(const streaming::Group &group,
                                    const tc_stream_slot_ticket_v1 &ticket) override {
        auto &job = jobs_.at(ticket.slot);
        job.block = group.blocks.at(0);
        return {ticket, &job, [](void *raw, const tc_stream_slot_ticket_v1 *,
                                const std::atomic<bool> *cancel, uint64_t *bytes) -> int {
            auto &j = *static_cast<Job *>(raw);
            try { *bytes = j.owner->source_.fill_exact(j.slot, j.block, cancel); return 0; }
            catch (...) { return -1; }
        }};
    }
    void encode_prefix(uint32_t) override {
        source_.check_unchanged();
        for (uint32_t i = 0; i < prefix_; ++i) verify_(source_.prefix_weights(i));
    }
    void prepare_group(const streaming::Group &, const tc_stream_slot_ticket_v1 &) override {}
    streaming::ReaderSet encode_group(const streaming::Group &group,
        const tc_stream_slot_ticket_v1 &ticket, streaming::CompletionMailbox &) override {
        const auto weights = source_.bind_exact(ticket.slot, group.blocks[0]);
        verify_(weights);
        for (const auto &name : weights.sorted_keys()) buffers_.insert(weights.at(name).data<char>());
        streaming::ReaderSet result;
        result.count = 1;
        result.fences[0] = {1, ++sequence_};
        result.already_complete = true;
        return result;
    }
    bool drain() noexcept override { try { mx::synchronize(); return true; } catch (...) { return false; } }
    void destroy_pool() noexcept override { source_.destroy_exact_pool(); }
    size_t buffers() const { return buffers_.size(); }
};

int main(int argc, char **argv) try {
    require(argc == 4 || argc == 5, "expected checkpoint slots suffix [workers]");
    configure_streams();
    const uint32_t slots = unsigned(std::stoi(argv[2]));
    const bool suffix = std::stoi(argv[3]) != 0;
    const bool fp32 = std::getenv("TURBOCIDER_Z_CONVROT_FP32_SCALES") != nullptr;
    std::atomic<bool> cancel{false};
    auto event = [](const std::string &, int, int) {};
    StreamingConfig config;
    config.enabled = true; config.schema_version = 1;
    config.selection = "manual"; config.retention = "request";
    const uint32_t workers = argc == 5 ? unsigned(std::stoi(argv[4])) : 1;
    config.stages["denoiser"] = {"streamed", 1, slots, 3, slots - 1, workers};
    const z_image::StreamingWorkload workload{256, 256, 32, 3, suffix ? 256u : 0u, fp32};
    z_image::StreamingPlanView plan(argv[1], config, workload);
    const auto &metadata = plan.metadata();
    const auto &layout = plan.layout().stages.front();
    require(metadata.convrot() && metadata.tensors_per_block() == 31, "incorrect INT8 descriptor");
    const auto slot_capacity = layout.pools.front().slots.front().capacity_bytes;
    require(layout.peak_pool_bytes == slots * slot_capacity && slot_capacity >= metadata.block_bytes(),
            "incorrect materialized pool size");
    require(layout.source_read_bytes_per_pass != layout.suffix_content_bytes_per_pass,
            "source and materialized byte counts were conflated");
    require(plan.descriptor().artifacts.size() == (suffix ? 2u : 1u), "missing derived artifact");
    auto scale_workload = workload;
    scale_workload.fp32_scales = !fp32;
    z_image::StreamingPlanView scale_plan(argv[1], config, scale_workload);
    require(scale_plan.layout().digest != plan.layout().digest, "scale precision missing from identity");
    if (suffix) {
        Weights interrupted_fixed;
        bool caught = false;
        try {
            ZImageWeightStream interrupted(metadata, 3, slots, 0, 0, interrupted_fixed,
                [&](const std::string &phase, int done, int) {
                    if (phase == "pack_z_image_suffix" && done == 1) cancel = true;
                }, cancel);
        } catch (const Cancelled &) { caught = true; }
        require(caught && !interrupted_fixed.bytes(), "cancelled pack leaked fixed weights");
        cancel = false;
    }
    Weights reference, fixed;
    reference.load_file(argv[1]);
    reference.cast_unquantized_float32(mx::bfloat16);
    reference.pack_convrot_q8();
    reference.materialize();
    auto verify = [&](const Weights &weights) {
        for (const auto &name : weights.sorted_keys()) {
            const auto &actual = weights.at(name);
            auto expected = reference.at(name);
            const bool compact = suffix && !name.starts_with("context_refiner.") &&
                name.find(".feed_forward.") != std::string::npos && !name.ends_with(".comfy_quant");
            if (compact && (name.find(".w1.") != std::string::npos || name.find(".w3.") != std::string::npos))
                expected = slice_axis(expected, 0, 256, 512);
            else if (compact) {
                const int first = name.ends_with(".weight") ? 64 : 8;
                expected = slice_axis(expected, 1, first, first * 2);
            }
            mx::eval(expected);
            require(actual.shape() == expected.shape() && actual.dtype() == expected.dtype(),
                    "materialized shape/dtype differs: " + name);
            require(mx::array_equal(actual, expected).item<bool>(), "INT8 conversion differs: " + name);
        }
    };
    uint64_t fixed_capacity = 0;
    for (const auto &r : metadata.fixed_records()) fixed_capacity += (r.bytes + 255) / 256 * 256;
    require(layout.resident_bytes == fixed_capacity, "fixed materializations missing from descriptor");
    const auto budget = fixed_capacity + (3 + slots) * slot_capacity +
                        (slots + 1) * metadata.scratch_bytes_per_slot();
    {
        Weights rejected_fixed;
        bool rejected = false;
        try {
            ZImageWeightStream too_small(metadata, 3, slots, budget - 1, 0,
                                         rejected_fixed, event, cancel);
        } catch (const std::exception &error) {
            rejected = std::string(error.what()).find("exceeds") != std::string::npos;
        }
        require(rejected && !rejected_fixed.bytes(), "exact layout was silently resized to fit budget");
    }
    ZImageWeightStream source(metadata, 3, slots, budget, 0, fixed, event, cancel);
    require(source.metrics().estimated_working_set_bytes == budget, "scratch missing from budget");
    require(source.metrics().request_bytes_loaded ==
                *layout.resident_source_read_bytes + *layout.prefix_source_read_bytes,
            "fixed/prefix source reads differ from descriptor");
    verify(fixed);
    auto adapter = std::make_shared<VerifyAdapter>(source, slots, 3, verify);
    streaming::StageExecutor executor(0, 1, adapter);
    executor.begin(layout);
    source.reset_metrics();
    for (uint32_t pass = 0; pass < 3; ++pass) executor.run_pass(pass, pass, cancel);
    const auto counters = executor.finish();
    require(counters.fills == 81 && counters.groups_submitted == 81, "incorrect generic execution count");
    require(counters.bytes_loaded == 81 * metadata.block_bytes(), "completion confused I/O with content");
    require(source.metrics().request_bytes_loaded == 3 * *layout.source_read_bytes_per_pass,
            "actual logical reads differ from descriptor");
    require(adapter->buffers() == slots * 31, "slot buffers grew across passes");
    source.create_exact_pool(slots, slot_capacity);
    std::atomic<bool> worker_cancel{true};
    bool worker_stopped = false;
    try { source.fill_exact(0, 3, &worker_cancel); }
    catch (const std::exception &) { worker_stopped = true; }
    require(worker_stopped, "worker cancellation ignored");
    source.destroy_exact_pool();
    std::ofstream(argv[1], std::ios::app).put('x');
    bool changed = false;
    try { source.check_unchanged(); } catch (const std::exception &) { changed = true; }
    require(changed, "source mutation was hidden by packed suffix");
    std::cout << "PASS generic INT8: exact resident values, source/read accounting, K=" << slots
              << ", suffix=" << suffix << ", fp32_scales=" << fp32
              << ", bounded buffers, cancellation and source mutation\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
