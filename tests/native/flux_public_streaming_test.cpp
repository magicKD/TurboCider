#include "models/flux2/flux.hpp"
#include "models/flux2/streaming_descriptor.hpp"
#include "runtime/streaming/resolved_request.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

tc::StreamingConfig config() {
    tc::StreamingConfig value;
    value.enabled = true;
    value.schema_version = 1;
    value.selection = "manual";
    value.retention = "request";
    value.stages["denoiser"] = {
        "streamed", 1, 2, 0, 1, 2};
    return value;
}

tc::Request request() {
    tc::Request value;
    value.model = "flux2-klein-9b";
    value.operation = "image.generate";
    value.prompt = "g";
    value.output = "/tmp/not-generated-flux-public.png";
    value.execution = "gpu";
    value.width = 256;
    value.height = 256;
    value.steps = 3;
    value.frames = 1;
    value.audio = false;
    value.dynamic_text = true;
    return value;
}

tc::streaming::StreamingDeviceIdentity device() {
    return {"Apple Test GPU", "Apple Test GPU/16GiB", "test-os", 16ull << 30};
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

} // namespace

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 3);
    try {
        const std::string model = argc == 3 ? argv[2] : "flux2-klein-9b";
        const bool klein4 = model == "flux2-klein-4b";
        tc::Flux session(argv[1], model);
        auto base_request = request();
        base_request.model = model;
        const auto probe = session.probe_public_streaming(
            {base_request, device(), "embedded_app"});
        assert(probe && probe->model_id() == model);
        assert(probe->source_lease() != nullptr);
        assert(probe->source_lease()->file_count() == (klein4 ? 7 : 9));
        std::vector<std::string> logical_ids;
        for (const auto &file : probe->source_lease()->descriptor().files)
            logical_ids.push_back(file.logical_id);
        if (klein4) {
            assert(logical_ids == std::vector<std::string>({
                "config.json", "diffusion_pytorch_model.safetensors",
                "text_encoder/config.json", "text_encoder/model.safetensors",
                "tokenizer/tokenizer.json", "vae/ae.safetensors", "vae/config.json"}));
        } else assert(logical_ids == std::vector<std::string>({
            "config.json",
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
            "diffusion_pytorch_model.safetensors.index.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
            "vae/ae.safetensors",
            "vae/config.json",
        }));
        assert(probe->workload_identity().width == 256);
        assert(probe->workload_identity().height == 256);
        assert(probe->workload_identity().steps == 3);
        assert(probe->workload_identity().token_shapes.size() == 1);
        assert(probe->workload_identity().feature_digest.size() == 64);
        assert(probe->workload_identity().feature_digest.find_first_not_of(
                   "0123456789abcdef") == std::string::npos);

        const auto value_probe = std::dynamic_pointer_cast<
            const tc::streaming::ValueModelStreamingProbe>(probe);
        assert(value_probe && value_probe->lease_ptr().get() ==
                                  probe->source_lease());
        const auto &token = probe->workload_identity().token_shapes.front();
        const tc::flux2::StreamingWorkload workload{
            256, 256, token.padded_rows, 0, 3};
        const tc::flux2::StreamingPlanView expected(
            value_probe->lease_ptr(), model, config(), workload);

        tc::streaming::StreamingPresetRecord record;
        record.id = "flux-public-host-test";
        record.revision = 1;
        record.catalog_revision = "host-test-r1";
        record.source = probe->source_identity();
        record.workload = probe->workload_identity();
        record.runtime = probe->runtime_identity();
        record.plan.canonical_config = config();
        record.plan.layout_digest = expected.layout().digest;
        record.plan.component_policy_revision =
            std::string(probe->component_policy_revision());
        record.plan.pass_transition = "reload";
        record.plan.multi_pool_policy = "retain_all";

        const auto snapshot = session.compile_public_streaming(probe, record);
        assert(snapshot && snapshot->source_lease() == probe->source_lease());
        assert(snapshot->layout().digest == record.plan.layout_digest);
        assert(snapshot->layout().materializations_complete);
        snapshot->revalidate_source();

        auto wrong = record;
        wrong.source.model_variant += "-wrong";
        rejects([&] { session.compile_public_streaming(probe, wrong); },
                "streaming_record_identity_mismatch");
        wrong = record;
        wrong.plan.layout_digest = std::string(64, '0');
        rejects([&] { session.compile_public_streaming(probe, wrong); },
                "streaming_layout_digest_mismatch");
        wrong = record;
        wrong.plan.component_policy_revision += "-wrong";
        rejects([&] { session.compile_public_streaming(probe, wrong); },
                "streaming_record_identity_mismatch");

        auto unsupported = base_request;
        unsupported.compile_gpu = true;
        rejects([&] {
            session.probe_public_streaming(
                {unsupported, device(), "embedded_app"});
        }, "streaming_route_unsupported");
        unsupported = base_request;
        unsupported.ane_manifest = "/tmp/not-opened-flux-ane.json";
        rejects([&] {
            session.probe_public_streaming(
                {unsupported, device(), "embedded_app"});
        }, "streaming_route_unsupported");

        tc::streaming::ResolvedStreamingSelection selection;
        selection.record = record;
        selection.device = device();
        selection.exact_selector.enabled = true;
        selection.exact_selector.selection = "preset";
        selection.exact_selector.target_request_memory_bytes = 1ull << 30;
        tc::Request resolved_request = base_request;
        resolved_request.streaming = config();
        auto execution = std::make_shared<
            const tc::streaming::ResolvedRequestExecution>(
                tc::streaming::ResolvedRequestExecution{
                    resolved_request, selection, probe, snapshot,
                    "host-test-request"});
        std::atomic<bool> cancelled{false};
        const tc::Event event = [](const std::string &, int, int) {};

        // Bind a supported public target, then fail inside run() before any
        // model load. Repeating the call proves the session-scoped migration
        // binding is restored on exceptions and does not poison the session.
        auto bound_selection = selection;
        bound_selection.exact_selector.target_request_memory_bytes =
            8ull << 30;
        auto invalid_output_request = resolved_request;
        invalid_output_request.output = "/tmp/not-generated-flux-public.jpg";
        auto bound_execution = std::make_shared<
            const tc::streaming::ResolvedRequestExecution>(
                tc::streaming::ResolvedRequestExecution{
                    invalid_output_request, bound_selection, probe, snapshot,
                    "host-test-bound-request"});
        rejects([&] {
            session.generate_resolved(bound_execution, event, cancelled);
        }, "native image output");
        rejects([&] {
            session.generate_resolved(bound_execution, event, cancelled);
        }, "native image output");

        rejects([&] {
            session.generate_resolved(execution, event, cancelled);
        }, "streaming_target_unsupported");
        rejects([&] {
            session.generate_resolved(execution, event, cancelled);
        }, "streaming_target_unsupported");

        const auto shard = std::filesystem::path(argv[1]) /
            (klein4 ? "transformer/diffusion_pytorch_model.safetensors" :
                      "transformer/diffusion_pytorch_model-00002-of-00002.safetensors");
        const auto moved = shard.string() + ".moved";
        std::filesystem::rename(shard, moved);
        {
            std::ofstream replacement(shard, std::ios::binary);
            replacement << "replacement";
        }
        rejects([&] { snapshot->revalidate_source(); }, "source path");

        std::cout << "PASS FLUX public adapter: shared source closure, "
                     "lease-backed descriptor, exact identity/layout, "
                     "route rejection, target failure cleanup and source "
                     "replacement detection\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
