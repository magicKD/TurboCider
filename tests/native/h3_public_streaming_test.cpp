#include "models/h3_runtime/h3_streaming_descriptor.hpp"
#include "runtime/session.hpp"
#include "runtime/streaming/resolved_request.hpp"

#include <atomic>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc {
std::unique_ptr<ModelSession> create_h3(const std::filesystem::path &);
}

namespace {

tc::StreamingConfig config() {
    tc::StreamingConfig value;
    value.enabled = true;
    value.schema_version = 1;
    value.selection = "manual";
    value.retention = "request";
    value.stages["denoiser"] = {
        "streamed", 1, 2, 1, 1, 1};
    return value;
}

tc::Request request() {
    tc::Request value;
    value.model = "minimax-h3-turbo";
    value.operation = "video.generate";
    value.prompt = "g";
    value.output = "/tmp/not-generated-h3-public.mp4";
    value.execution = "gpu";
    // PublicStreamingCoordinator::resolve_normalized() only calls the model
    // probe after make_plan has normalized the request.  A square H3 request
    // is therefore 768x768 here; 512x512 would be an unnormalised API input
    // and must not be accepted by the descriptor.
    value.width = 768;
    value.height = 768;
    value.frames = 22;
    value.fps = 24;
    value.steps = 4;
    value.audio = false;
    value.dynamic_text = true;
    return value;
}

tc::streaming::StreamingDeviceIdentity device() {
    return {"Apple Test GPU", "Apple Test GPU/16GiB", "test-os",
            16ull << 30};
}

template <class Function>
void rejects(Function &&function, const char *part) {
    try {
        function();
    } catch (const std::exception &error) {
        if (std::string(error.what()).find(part) != std::string::npos)
            return;
        std::cerr << error.what() << " expected " << part << '\n';
        std::abort();
    }
    std::cerr << "expected rejection containing " << part << '\n';
    std::abort();
}

std::vector<std::string> transformer_ids(
        const tc::streaming::SourceLease &lease) {
    std::vector<std::string> ids;
    for (const auto &file : lease.descriptor().files)
        if (file.logical_id.starts_with("FL2VA/transformer/") &&
            file.logical_id.ends_with(".safetensors"))
            ids.push_back(file.logical_id);
    return ids;
}

} // namespace

int main(int argc, char **argv) {
    assert(argc == 2);
    try {
        auto session = tc::create_h3(argv[1]);
        const auto base_request = request();
        const auto probe = session->probe_public_streaming(
            {base_request, device(), "embedded_app"});
        assert(probe && probe->model_id() == "minimax-h3-turbo");
        assert(probe->source_lease() != nullptr);
        assert(probe->source_lease()->file_count() >= 20);
        assert(probe->workload_identity().width == 768);
        assert(probe->workload_identity().height == 768);
        assert(probe->workload_identity().frames == 22);
        assert(probe->workload_identity().steps == 4);
        assert(probe->workload_identity().token_shapes.size() == 1);

        const auto value_probe = std::dynamic_pointer_cast<
            const tc::streaming::ValueModelStreamingProbe>(probe);
        assert(value_probe && value_probe->lease_ptr().get() ==
                                  probe->source_lease());
        const auto ids = transformer_ids(value_probe->lease());
        assert(ids.size() == 13);
        const auto &workload = probe->workload_identity();
        tc::h3::StreamingWorkload descriptor_workload;
        descriptor_workload.width = workload.width;
        descriptor_workload.height = workload.height;
        descriptor_workload.frames = workload.frames;
        descriptor_workload.fps = workload.fps;
        descriptor_workload.text_rows =
            workload.token_shapes.front().padded_rows;
        descriptor_workload.steps = workload.steps;
        descriptor_workload.active_blocks = H3_DIT_BLOCKS;
        descriptor_workload.audio = false;
        const tc::h3::StreamingPlanView expected(
            value_probe->lease_ptr(), ids, config(),
            descriptor_workload, 1);

        tc::streaming::StreamingPresetRecord record;
        record.id = "h3-public-host-test";
        record.revision = 1;
        record.catalog_revision = "host-test-r1";
        record.source = probe->source_identity();
        record.workload = probe->workload_identity();
        record.runtime = probe->runtime_identity();
        record.plan.canonical_config = config();
        record.plan.layout_digest = expected.layout().digest;
        record.plan.component_policy_revision =
            std::string(probe->component_policy_revision());
        record.plan.pass_transition = "carry_first_group";
        record.plan.multi_pool_policy = "serial";

        const auto snapshot = session->compile_public_streaming(probe, record);
        assert(snapshot && snapshot->source_lease() == probe->source_lease());
        assert(snapshot->layout().digest == record.plan.layout_digest);
        assert(snapshot->layout().materializations_complete);
        snapshot->revalidate_source();

        auto wrong = record;
        wrong.source.model_variant += "-wrong";
        rejects([&] { session->compile_public_streaming(probe, wrong); },
                "streaming_record_identity_mismatch");
        wrong = record;
        wrong.plan.layout_digest = std::string(64, '0');
        rejects([&] { session->compile_public_streaming(probe, wrong); },
                "streaming_layout_digest_mismatch");

        auto unsupported = base_request;
        unsupported.audio = true;
        rejects([&] {
            session->probe_public_streaming(
                {unsupported, device(), "embedded_app"});
        }, "streaming_route_unsupported");
        unsupported = base_request;
        unsupported.ane_manifest = "/tmp/not-opened-h3-ane.json";
        rejects([&] {
            session->probe_public_streaming(
                {unsupported, device(), "embedded_app"});
        }, "streaming_route_unsupported");

        tc::streaming::ResolvedStreamingSelection selection;
        selection.record = record;
        selection.device = device();
        selection.exact_selector.enabled = true;
        selection.exact_selector.selection = "preset";
        selection.exact_selector.target_request_memory_bytes = 1ull << 30;
        auto resolved_request = base_request;
        resolved_request.streaming = config();
        auto execution = std::make_shared<
            const tc::streaming::ResolvedRequestExecution>(
                tc::streaming::ResolvedRequestExecution{
                    resolved_request, selection, probe, snapshot,
                    "host-test-request"});
        std::atomic<bool> cancelled{false};
        const tc::Event event = [](const std::string &, int, int) {};
        rejects([&] {
            session->generate_resolved(execution, event, cancelled);
        }, "streaming_target_unsupported");
        rejects([&] {
            session->generate_resolved(execution, event, cancelled);
        }, "streaming_target_unsupported");

        auto bound_selection = selection;
        bound_selection.exact_selector.target_request_memory_bytes =
            8ull << 30;
        auto invalid_output = resolved_request;
        invalid_output.output = "/tmp/not-generated-h3-public.png";
        auto bound_execution = std::make_shared<
            const tc::streaming::ResolvedRequestExecution>(
                tc::streaming::ResolvedRequestExecution{
                    invalid_output, bound_selection, probe, snapshot,
                    "host-test-bound-request"});
        rejects([&] {
            session->generate_resolved(bound_execution, event, cancelled);
        }, "H3 output must be .mp4");
        rejects([&] {
            session->generate_resolved(bound_execution, event, cancelled);
        }, "H3 output must be .mp4");

        const auto shard = std::filesystem::path(argv[1]) /
            "FL2VA/transformer/model-00013-of-00013.safetensors";
        const auto moved = shard.string() + ".moved";
        std::filesystem::rename(shard, moved);
        {
            std::ofstream replacement(shard, std::ios::binary);
            replacement << "replacement";
        }
        rejects([&] { snapshot->revalidate_source(); }, "source");

        std::cout << "PASS H3 public adapter: complete source lease, "
                     "fd tokenizer/descriptor, exact identity/layout, route "
                     "rejection, failure cleanup and source replacement detection\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
