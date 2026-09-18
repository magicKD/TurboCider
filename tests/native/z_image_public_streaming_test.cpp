#include "models/z_image/z_image.hpp"
#include "models/z_image/streaming_descriptor.hpp"
#include "runtime/streaming/resolved_request.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

tc::StreamingConfig config() {
    tc::StreamingConfig value;
    value.enabled = true;
    value.schema_version = 1;
    value.selection = "manual";
    value.retention = "request";
    value.stages["denoiser"] = {
        "streamed", 1, 2, 3, 0, 1};
    return value;
}

tc::Request request() {
    tc::Request value;
    value.model = "z-image-turbo";
    value.operation = "image.generate";
    value.prompt = "g";
    value.output = "/tmp/not-generated-z-image-public.png";
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
    assert(argc == 2);
    try {
        tc::ZImage session(argv[1]);
        const auto base_request = request();
        tc::streaming::PublicResolveInput input{
            base_request, device(), "embedded_app"};
        const auto probe = session.probe_public_streaming(input);
        assert(probe && probe->model_id() == "z-image-turbo");
        assert(probe->source_lease() != nullptr);
        assert(probe->source_lease()->file_count() == 3);
        assert(probe->workload_identity().width == 256);
        assert(probe->workload_identity().height == 256);
        assert(probe->workload_identity().steps == 3);
        assert(probe->workload_identity().token_shapes.size() == 1);

        const auto value_probe = std::dynamic_pointer_cast<
            const tc::streaming::ValueModelStreamingProbe>(probe);
        assert(value_probe && value_probe->lease_ptr().get() ==
                                  probe->source_lease());
        const auto &token = probe->workload_identity().token_shapes.front();
        const tc::z_image::StreamingWorkload workload{
            256, 256, token.padded_rows, 3};
        const tc::z_image::StreamingPlanView expected(
            value_probe->lease_ptr(), config(), workload);

        tc::streaming::StreamingPresetRecord record;
        record.id = "z-image-public-host-test";
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
        record.plan.multi_pool_policy = "serial";

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

        auto unsupported = base_request;
        unsupported.compile_gpu = true;
        rejects([&] {
            session.probe_public_streaming(
                {unsupported, device(), "embedded_app"});
        }, "streaming_route_unsupported");
        unsupported = base_request;
        unsupported.ane_manifest = "/tmp/not-opened-z-image-ane.json";
        rejects([&] {
            session.probe_public_streaming(
                {unsupported, device(), "embedded_app"});
        }, "streaming_route_unsupported");

        tc::streaming::ResolvedStreamingSelection selection;
        selection.record = record;
        selection.device = device();
        tc::Request resolved_request = base_request;
        resolved_request.streaming = config();
        auto execution = std::make_shared<
            const tc::streaming::ResolvedRequestExecution>(
                tc::streaming::ResolvedRequestExecution{
                    resolved_request, selection, probe, snapshot,
                    "host-test-request"});
        std::atomic<bool> cancelled{false};
        const tc::Event event = [](const std::string &, int, int) {};
        // A rejected target must not leave the request-scoped lease bound on
        // the session. The second call must fail for the same reason, not as a
        // false reentrant request.
        rejects([&] {
            session.generate_resolved(execution, event, cancelled);
        }, "streaming_target_unsupported");
        rejects([&] {
            session.generate_resolved(execution, event, cancelled);
        }, "streaming_target_unsupported");

        const auto transformer = std::filesystem::path(argv[1]) /
            "split_files/diffusion_models/z_image_turbo_bf16.safetensors";
        const auto moved = transformer.string() + ".moved";
        std::filesystem::rename(transformer, moved);
        {
            std::ofstream replacement(transformer, std::ios::binary);
            replacement << "replacement";
        }
        rejects([&] { snapshot->revalidate_source(); },
                "source path");

        std::cout << "PASS Z-Image public adapter: shared three-artifact lease, "
                     "exact identity/layout snapshot, route rejection, target "
                     "failure cleanup and source replacement detection\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
